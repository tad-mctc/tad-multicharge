# SPDX-Identifier: CC0-1.0
"""
Opt-in EEQ transform scalability validation on water clusters.

Run from the repository root with its source on PYTHONPATH, for example::

    PYTHONPATH=src python examples/validation/matrix_free_transforms.py \
        --device cuda --sizes 99 300 999 --timeout 120 \
        --output /tmp/eeq-transforms.jsonl

Each case runs in a fresh subprocess with a deadline. CUDA initialization,
reference evaluation and the first (possibly compiling) evaluation are
separated from steady-state timing. All derivatives are of a scalar
energy/charge objective; forward derivatives use a single direction.
The mixed derivative is restricted to --mixed-max-atoms (300 by default).
No solver tolerance, iteration budget or refinement setting is changed.

Instrumentation delegates to the original solver and records diagnostics
outside iteration loops. It uses private functorch unwrapping only to copy
those diagnostics, never to modify differentiated inputs or computations.
This harness is not imported or executed by the test suite.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import signal
import statistics
import subprocess
import sys
import time
from collections import Counter
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import torch
from tad_mctc import __version__ as mctc_version
from tad_mctc.io.structure import Structure
from tad_mctc.neighbor import build_neighborlist
from tad_mctc.storch import pairwise
from torch.func import jacfwd, jacrev, jvp, vmap

from tad_multicharge import eeq
from tad_multicharge.model import _matrix_free as mf

MODES = (
    "eager",
    "native",
    "native-jacobi",
    "jacrev",
    "jvp",
    "jacfwd",
    "vmap",
    "mixed",
)


def emit(value: dict[str, Any]) -> None:
    print(json.dumps(value, allow_nan=False), flush=True)


def snapshot(tensor: torch.Tensor | None) -> Any:
    """Copy numerical diagnostics without leaking transformed tensors."""
    if tensor is None:
        return None
    functorch = torch._C._functorch
    while functorch.is_functorch_wrapped_tensor(tensor):
        tensor = functorch.get_unwrapped(tensor)

    with torch._C._DisableFuncTorch():
        return tensor.detach().cpu().tolist()


@contextmanager
def instrument(records: dict[str, Any]) -> Iterator[None]:
    """Observe selection, operator calls and solves without changing them."""
    preconditioner = mf._preconditioner
    matvec = mf._matvec
    constrained = mf.linsolve.constrained_solve
    projected = mf.linsolve.constrained.projected_cg
    fused_product = pairwise._fused_product
    tile_product = pairwise._tile_product

    def record_preconditioner(
        *args: Any, **kwargs: Any
    ) -> Callable[..., torch.Tensor]:
        result = preconditioner(*args, **kwargs)
        selected = "jacobi" if result is mf._jacobi else "balancing"
        records["preconditioners"].append(selected)
        emit({"event": "preconditioner", "selected": selected})
        return result

    def record_matvec(*args: Any, **kwargs: Any) -> torch.Tensor:
        precision = "fast" if kwargs.get("dtype") == torch.float32 else "exact"
        records["operator_calls"][precision] += 1
        return matvec(*args, **kwargs)

    def record_fused(*args: Any, **kwargs: Any) -> torch.Tensor:
        records["pair_paths"]["fused_product"] += 1
        return fused_product(*args, **kwargs)

    def record_tile(*args: Any, **kwargs: Any) -> torch.Tensor:
        records["pair_paths"]["eager_tile"] += 1
        return tile_product(*args, **kwargs)

    def record_projected(
        *args: Any, **kwargs: Any
    ) -> mf.linsolve.LinearSolveResult:
        result = projected(*args, **kwargs)
        records["inner_solves"].append(
            {
                "status": snapshot(result.status),
                "iterations": snapshot(result.iterations),
            }
        )
        return result

    def record_constrained(
        *args: Any, **kwargs: Any
    ) -> mf.linsolve.ConstrainedSolveResult:
        result = constrained(*args, **kwargs)
        diagnostic = {
            name: snapshot(getattr(result, name))
            for name in (
                "status",
                "iterations",
                "residual",
                "target",
                "constraint_residual",
                "refinement_status",
                "fallback_iterations",
            )
        }
        records["public_solves"].append(diagnostic)
        emit({"event": "primal", "diagnostic": diagnostic})
        return result

    pairwise._fused_product = record_fused
    pairwise._tile_product = record_tile
    mf._preconditioner = record_preconditioner
    mf._matvec = record_matvec
    mf.linsolve.constrained_solve = record_constrained
    mf.linsolve.constrained.projected_cg = record_projected
    try:
        yield
    finally:
        pairwise._fused_product = fused_product
        pairwise._tile_product = tile_product
        mf._preconditioner = preconditioner
        mf._matvec = matvec
        mf.linsolve.constrained_solve = constrained
        mf.linsolve.constrained.projected_cg = projected


def new_records() -> dict[str, Any]:
    return {
        "preconditioners": [],
        "operator_calls": Counter(),
        "pair_paths": Counter(),
        "inner_solves": [],
        "public_solves": [],
    }


def water_cluster(natoms: int, device: torch.device) -> Structure:
    """Deterministic water molecules at roughly liquid density, in bohr."""
    molecules = max(1, natoms // 3)
    side = math.ceil(molecules ** (1 / 3))
    axes = torch.arange(side, dtype=torch.double, device=device)
    centers = (
        5.86
        * torch.stack(
            torch.meshgrid(axes, axes, axes, indexing="ij"), -1
        ).reshape(-1, 3)[:molecules]
    )
    water = torch.tensor(
        [[0.0, 0.0, 0.0], [1.43, 1.11, 0.0], [-1.43, 1.11, 0.0]],
        dtype=torch.double,
        device=device,
    )
    positions = (centers[:, None, :] + water).reshape(-1, 3)
    numbers = torch.tensor([8, 1, 1], device=device).repeat(molecules)
    return Structure(
        numbers=numbers, positions=positions, charge=positions.new_tensor(-1.0)
    )


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def evaluate_case(args: argparse.Namespace) -> dict[str, Any]:
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.ones(1, device=device).sum().item()
    structure = water_cluster(args.size, device)
    natoms = structure.numbers.shape[-1]
    model = eeq.EEQModel.param2019(device=device, dtype=torch.double)
    if args.mode == "native-jacobi":
        model = model.replace(
            matrix_free=eeq.MatrixFreeOptions(coarse_memory=0)
        )
    neighbors = build_neighborlist(structure, model.cn.cutoff, skin=1.0)
    positions = structure.positions
    direction = torch.sin(
        torch.arange(positions.numel(), device=device, dtype=torch.double)
    ).reshape_as(positions)
    direction = direction / torch.linalg.vector_norm(direction)
    eta_direction = torch.zeros_like(model.eta)
    eta_direction[1] = 1.0
    zero = positions.new_zeros(())
    one = positions.new_ones(())

    def objective(
        current_positions: torch.Tensor,
        mode: str,
        eta_offset: torch.Tensor = zero,
    ) -> torch.Tensor:
        current_model = model.replace(
            eta=model.eta + eta_offset * eta_direction
        )
        current = structure.replace(positions=current_positions)
        charges, energies = current_model(
            current, neighbors, solve_mode=mode, return_energy=True
        )
        return energies.sum() + 0.01 * charges.square().sum()

    def matrix_free_objective(current_positions: torch.Tensor) -> torch.Tensor:
        return objective(current_positions, "matrix-free")

    def dense_objective(current_positions: torch.Tensor) -> torch.Tensor:
        return objective(current_positions, "schur")

    def displacement(value: torch.Tensor) -> torch.Tensor:
        return matrix_free_objective(positions + value * direction)

    def force_projection(eta_offset: torch.Tensor, mode: str) -> torch.Tensor:
        def energy(current_positions: torch.Tensor) -> torch.Tensor:
            return objective(current_positions, mode, eta_offset)

        position_gradient = jacrev(energy)
        return (position_gradient(positions) * direction).sum()

    def matrix_free_projection(value: torch.Tensor) -> torch.Tensor:
        return force_projection(value, "matrix-free")

    def dense_projection(value: torch.Tensor) -> torch.Tensor:
        return force_projection(value, "schur")

    reverse = jacrev(matrix_free_objective)
    forward = jacfwd(displacement)
    vectorized = vmap(matrix_free_objective)
    batched_positions = torch.stack((positions, positions + 0.01 * direction))

    def run() -> torch.Tensor:
        if args.mode == "eager":
            return matrix_free_objective(positions)
        if args.mode in ("native", "native-jacobi"):
            differentiable = positions.clone().requires_grad_(True)
            value = matrix_free_objective(differentiable)
            value.backward()
            assert differentiable.grad is not None
            return differentiable.grad
        if args.mode == "jacrev":
            return reverse(positions)
        if args.mode == "jvp":
            return jvp(matrix_free_objective, (positions,), (direction,))[1]
        if args.mode == "jacfwd":
            return forward(zero)
        if args.mode == "vmap":
            return vectorized(batched_positions)
        return jvp(matrix_free_projection, (zero,), (one,))[1]

    records = new_records()
    synchronize(device)
    start = time.perf_counter()
    with instrument(records):
        cold_output = run()
    synchronize(device)
    cold_seconds = time.perf_counter() - start
    emit({"event": "warmup_complete", "seconds": cold_seconds})

    # Dense references and initialization stay outside the measured region.
    if args.mode == "eager":
        reference = dense_objective(positions).detach()
    elif args.mode == "vmap":
        reference = torch.stack(
            [dense_objective(value).detach() for value in batched_positions]
        )
    elif args.mode == "mixed":
        reference = jvp(dense_projection, (zero,), (one,))[1].detach()
    else:
        dense_gradient = jacrev(dense_objective)(positions).detach()
        reference = (
            (dense_gradient * direction).sum()
            if args.mode in ("jvp", "jacfwd")
            else dense_gradient
        )
    del cold_output

    times = []
    peaks = []
    final_records = {}
    for _ in range(args.repeats):
        records = new_records()
        synchronize(device)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
            baseline = torch.cuda.memory_allocated(device)
        start = time.perf_counter()
        with instrument(records):
            output = run()
        synchronize(device)
        times.append(time.perf_counter() - start)
        if device.type == "cuda":
            peaks.append(torch.cuda.max_memory_allocated(device) - baseline)
        final_records = records

    error = (output.detach() - reference).abs().max().item()
    finite = bool(torch.isfinite(output).all())
    return {
        "atoms": natoms,
        "mode": args.mode,
        "device": str(device),
        "device_name": (
            torch.cuda.get_device_name(device)
            if device.type == "cuda"
            else "cpu"
        ),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "tad_mctc": mctc_version,
        "threads": args.threads,
        "dtype": "float64",
        "solver_mode": "matrix-free",
        "mixed_precision": True,
        "fallback": True,
        "coarse_memory": (
            0 if args.mode == "native-jacobi" else 64 * 1024 * 1024
        ),
        "cold_seconds": cold_seconds,
        "steady_seconds": statistics.median(times),
        "repeat_seconds": times,
        "peak_incremental_bytes": max(peaks) if peaks else None,
        "max_absolute_error": error if math.isfinite(error) else None,
        "status": "ok" if finite and error < 1e-8 else "accuracy_failure",
        "source_sha256": hashlib.sha256(
            Path(mf.__file__).read_bytes()
        ).hexdigest(),
        **final_records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--sizes", type=int, nargs="+", default=[99, 300, 999])
    parser.add_argument(
        "--modes", choices=MODES, nargs="+", default=list(MODES)
    )
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--mixed-max-atoms", type=int, default=300)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument(
        "--output", type=Path, default=Path("/tmp/eeq-transforms.jsonl")
    )
    parser.add_argument("--case", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--size", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--mode", choices=MODES, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.repeats < 1 or args.threads < 1 or args.timeout <= 0:
        parser.error("repeats, threads and timeout must be positive")
    if any(size < 3 for size in args.sizes) or (args.case and args.size < 3):
        parser.error("water clusters require at least three atoms")
    if args.case:
        emit({"event": "result", "result": evaluate_case(args)})
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as destination:
        for size in args.sizes:
            for mode in args.modes:
                if mode == "mixed" and size > args.mixed_max_atoms:
                    row = {
                        "atoms": size,
                        "mode": mode,
                        "status": "not_run",
                        "reason": "mixed derivative size limit",
                    }
                else:
                    command = [
                        sys.executable,
                        __file__,
                        "--case",
                        "--size",
                        str(size),
                        "--mode",
                        mode,
                        "--device",
                        args.device,
                        "--repeats",
                        str(args.repeats),
                        "--threads",
                        str(args.threads),
                    ]
                    start = time.perf_counter()
                    process = subprocess.Popen(
                        command,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                        start_new_session=os.name == "posix",
                    )
                    timed_out = False
                    try:
                        stdout, stderr = process.communicate(
                            timeout=args.timeout
                        )
                    except subprocess.TimeoutExpired:
                        timed_out = True
                        if os.name == "posix":
                            os.killpg(process.pid, signal.SIGKILL)
                        else:
                            process.kill()
                        stdout, stderr = process.communicate()
                    events = []
                    for line in stdout.splitlines():
                        try:
                            events.append(json.loads(line))
                        except json.JSONDecodeError:
                            pass
                    results = [
                        event["result"]
                        for event in events
                        if event.get("event") == "result"
                    ]
                    row = (
                        results[-1]
                        if results
                        else {
                            "atoms": size,
                            "mode": mode,
                            "device": args.device,
                            "status": "timeout" if timed_out else "error",
                            "wall_seconds": time.perf_counter() - start,
                            "progress": events,
                            "stderr_tail": stderr[-2000:],
                        }
                    )
                destination.write(json.dumps(row, allow_nan=False) + "\n")
                destination.flush()
                emit(row)


if __name__ == "__main__":
    main()
