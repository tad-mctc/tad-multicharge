# This file is part of tad-multicharge.
#
# SPDX-Identifier: Apache-2.0
# Copyright (C) 2024 Grimme Group
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Compare dense EEQ solvers on synthetic nonperiodic molecular clusters.

Run from the repository with ``PYTHONPATH=src python examples/benchmark-eeq.py``.
Each row runs in a fresh process so peak RSS is independent of preceding rows.
Compilation and warmup are excluded from the median timings. Peak RSS includes
the interpreter, PyTorch, compilation and all three workloads; CUDA peak allocated
memory covers the warmed-up workloads and reference validation. This is a modest reproducible
comparison, not a chemically representative crossover study.

Examples
--------
CPU eager and vectorized execution at several sizes and hardness shifts::

    PYTHONPATH=src python examples/benchmark-eeq.py --sizes 32 128 512

Compilation (start small: the upstream CG loop unrolls twice the atom count)::

    PYTHONPATH=src python examples/benchmark-eeq.py --sizes 8 --executions compile

First-order compiled derivatives require PyTorch >= 2.11. Compiled higher
derivatives are unsupported and are not measured here.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
from tad_mctc import storch
from tad_mctc.io.structure import Structure
from tad_mctc.ncoord import cn_eeq
from tad_mctc.typing import Tensor
from torch.func import jacrev, vmap
from torch.utils._pytree import tree_leaves

from tad_multicharge.model import SolveMode, eeq


def _cluster(size: int, device: str, dtype: str) -> Structure:
    """Create a reproducible cluster with mixed atomic numbers in bohr."""
    dd = {"device": device, "dtype": getattr(torch, dtype)}
    index = torch.arange(size, device=device)
    width = int(size ** (1 / 3)) + 1
    positions = torch.stack(
        (index % width, (index // width) % width, index // width**2), dim=-1
    ).to(**dd)
    positions = 3.0 * positions + 0.1 * torch.sin(positions + 0.3)
    elements = torch.tensor([1, 6, 7, 8], device=device)
    return Structure(
        numbers=elements[index % 4],
        positions=positions,
        charge=positions.new_tensor(1.0),
    )


def _max_error(actual: Any, reference: Any) -> float:
    """Largest absolute difference over a tensor or parameter tree."""
    errors = [
        (a - b).abs().max().item()
        for a, b in zip(tree_leaves(actual), tree_leaves(reference))
    ]
    if not all(math.isfinite(error) for error in errors):
        raise RuntimeError(
            "Non-finite output: a transformed CG solve failed its acceptance check."
        )
    return max(errors)


def _measure(case: dict[str, Any]) -> dict[str, Any]:
    """Measure one row, including independently computed reference errors."""
    torch.set_num_threads(case["threads"])
    structure = _cluster(case["atoms"], case["device"], case["dtype"])
    model = eeq.EEQModel.param2019(
        device=structure.positions.device, dtype=structure.positions.dtype
    )
    # Increasing diagonal hardness changes conditioning without making A indefinite.
    model = model.replace(eta=model.eta + case["hardness_shift"])
    positions = structure.positions
    execution = case["execution"]
    if execution == "vmap":
        positions = torch.stack((positions, positions * 1.01))

    def functions(mode: SolveMode) -> list[Callable]:
        def charges(m: eeq.EEQModel, pos: Tensor) -> Tensor:
            return m(structure.replace(positions=pos), solve_mode=mode)

        def energy(m: eeq.EEQModel, pos: Tensor) -> Tensor:
            return m(
                structure.replace(positions=pos),
                return_energy=True,
                solve_mode=mode,
            )[1].sum()

        return [charges, jacrev(energy, argnums=1), jacrev(energy, argnums=0)]

    def synchronize() -> None:
        if case["device"].startswith("cuda"):
            torch.cuda.synchronize()

    def prepare(func: Callable) -> Callable:
        if execution == "vmap":
            return vmap(func, in_dims=(None, 0))
        if execution == "compile":
            return torch.compile(func, fullgraph=True)
        return func

    row = dict(case)
    a = eeq._coulomb_matrix_0d(
        structure.numbers,
        structure.positions,
        model.rad[structure.numbers],
        model.eta[structure.numbers],
    )
    row["condition"] = torch.linalg.cond(a).item()
    if case["solve_mode"] == "cg":
        b = (
            -model.chi[structure.numbers]
            + storch.safe_sqrt(cn_eeq(structure)) * model.kcn[structure.numbers]
        )
        result = storch.linsolve.implicit_solve(
            eeq._coulomb_matvec,
            torch.stack((b, torch.ones_like(b))),
            a,
            method="cg",
            preconditioner=eeq._jacobi,
            rtol=eeq.CG_RTOL,
            atol=eeq.CG_ATOL,
        )
        row["aux_iterations"] = "/".join(
            str(i) for i in result.iterations.tolist()
        )
        row["iteration_budget"] = 2 * case["atoms"]
    else:
        row["aux_iterations"] = ""
        row["iteration_budget"] = ""
    del a

    refs = [
        prepare(f) if execution == "vmap" else f for f in functions("schur")
    ]
    workloads = [prepare(f) for f in functions(case["solve_mode"])]
    labels = ["forward", "forces", "parameters"]
    try:
        # Compile/warm up every workload before resetting CUDA's peak counter.
        for func, ref in zip(workloads, refs):
            _max_error(func(model, positions), ref(model, positions))
        synchronize()
        if case["device"].startswith("cuda"):
            torch.cuda.reset_peak_memory_stats()

        for label, func, ref in zip(labels, workloads, refs):
            row[f"{label}_max_error"] = _max_error(
                func(model, positions), ref(model, positions)
            )
            times = []
            for _ in range(case["repeats"]):
                synchronize()
                start = time.perf_counter()
                func(model, positions)
                synchronize()
                times.append(1000 * (time.perf_counter() - start))
            row[f"{label}_ms"] = statistics.median(times)
    except Exception as error:
        row["error"] = str(error)

    try:
        import resource

        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        row["peak_rss_mib"] = rss / (
            1024**2 if sys.platform == "darwin" else 1024
        )
    except ImportError:
        row["peak_rss_mib"] = "unavailable"
    row["peak_cuda_mib"] = (
        torch.cuda.max_memory_allocated() / 1024**2
        if case["device"].startswith("cuda")
        else ""
    )
    row["torch"] = torch.__version__
    row["platform"] = platform.platform()
    return row


def main() -> None:
    """Write CSV benchmark results to stdout, keeping failures visible."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", nargs="+", type=int, default=[32, 128, 512])
    parser.add_argument(
        "--hardness-shifts", nargs="+", type=float, default=[0.0, 5.0]
    )
    parser.add_argument(
        "--executions",
        nargs="+",
        choices=["eager", "vmap", "compile"],
        default=["eager", "vmap"],
    )
    parser.add_argument(
        "--dtype", choices=["float32", "float64"], default="float64"
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument(
        "--timeout",
        type=float,
        default=300,
        help="Maximum seconds per row, including compilation.",
    )
    parser.add_argument("--worker", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        case = json.loads(args.worker)
        try:
            print(json.dumps(_measure(case)))
        # Report failed workloads alongside successful rows.
        except Exception as error:
            print(json.dumps({**case, "error": str(error)}))
        return

    if min(args.sizes + [args.threads, args.repeats]) < 1:
        parser.error("Sizes, threads and repeats must be positive.")
    fields = [
        "atoms",
        "hardness_shift",
        "execution",
        "solve_mode",
        "dtype",
        "device",
        "threads",
        "repeats",
        "condition",
        "aux_iterations",
        "iteration_budget",
        "forward_ms",
        "forces_ms",
        "parameters_ms",
        "forward_max_error",
        "forces_max_error",
        "parameters_max_error",
        "peak_rss_mib",
        "peak_cuda_mib",
        "torch",
        "platform",
        "error",
    ]
    writer = csv.DictWriter(sys.stdout, fieldnames=fields)
    writer.writeheader()
    sys.stdout.flush()
    failed = False
    for size, shift, execution, mode in itertools.product(
        args.sizes, args.hardness_shifts, args.executions, ("schur", "cg")
    ):
        case = {
            "atoms": size,
            "hardness_shift": shift,
            "execution": execution,
            "solve_mode": mode,
            "dtype": args.dtype,
            "device": args.device,
            "threads": args.threads,
            "repeats": args.repeats,
        }
        print(f"Measuring {case}", file=sys.stderr, flush=True)
        env = dict(
            os.environ,
            OMP_NUM_THREADS=str(args.threads),
            MKL_NUM_THREADS=str(args.threads),
        )
        try:
            result = subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--worker",
                    json.dumps(case),
                ],
                env=env,
                capture_output=True,
                text=True,
                check=True,
                timeout=args.timeout,
            )
            row = json.loads(result.stdout)
        except subprocess.TimeoutExpired:
            row = {
                **case,
                "error": f"Timed out after {args.timeout} seconds, including compilation.",
            }
        if "error" in row:
            print(row["error"], file=sys.stderr)
            failed = True
        writer.writerow(row)
        sys.stdout.flush()
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
