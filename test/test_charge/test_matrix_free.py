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
"""Physical and memory acceptance of molecular matrix-free EEQ."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from typing import Any

import pytest
import torch
from tad_mctc.autograd import no_vmap_fallback
from tad_mctc.exceptions import ConvergenceError
from tad_mctc.io.structure import Structure
from tad_mctc.storch import linsolve
from tad_mctc.tools.testing import requires_compile
from tad_mctc.tree import stack
from tad_mctc.typing import DD, Tensor
from torch.func import vmap

from tad_multicharge.model import _matrix_free as mf
from tad_multicharge.model import eeq

from ..conftest import DEVICE
from ..test_periodic.samples import load as load_periodic
from ..utils import load_batch, load_structure

DD_DOUBLE: DD = {"dtype": torch.double, "device": DEVICE}


def _model(**options: Any) -> eeq.EEQModel:
    return eeq.EEQModel.param2019(**DD_DOUBLE).replace(
        matrix_free=eeq.MatrixFreeOptions(fused=False, **options)
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize(
    "name,charge",
    [("LiH", 0.0), ("SiH4", 1.0), ("ZnOOH-", -1.0), ("AmF3", 0.0)],
)
@pytest.mark.parametrize("mixed", [False, True])
def test_charges_and_energies(
    name: str, charge: float, dtype: torch.dtype, mixed: bool
) -> None:
    """Tiled projected charges, energy, and total charge match both dense modes."""
    dd: DD = {"dtype": dtype, "device": DEVICE}
    structure = load_structure(name, dd, charge)
    model = eeq.EEQModel.param2019(**dd).replace(
        matrix_free=eeq.MatrixFreeOptions(
            tile=2, fused=False, mixed_precision=mixed
        )
    )
    q, e = model(structure, solve_mode="matrix-free", return_energy=True)
    tolerance = 2e-5 if dtype == torch.float32 else 1e-9
    for mode in ("schur", "cg"):
        qref, eref = model(structure, solve_mode=mode, return_energy=True)
        torch.testing.assert_close(q, qref, atol=tolerance, rtol=tolerance)
        torch.testing.assert_close(e, eref, atol=tolerance, rtol=tolerance)
    torch.testing.assert_close(
        q.sum(), q.new_tensor(charge), atol=tolerance, rtol=0
    )


def test_no_dense_coulomb(monkeypatch: pytest.MonkeyPatch) -> None:
    """Both charges and energies bypass every dense Coulomb constructor."""
    structure = load_structure("ZnOOH-", DD_DOUBLE, -1.0)
    model = _model()
    reference = model(structure, return_energy=True)

    def forbidden(*args: Any, **kwargs: Any) -> Tensor:
        raise AssertionError("A dense Coulomb matrix was requested")

    monkeypatch.setattr(eeq, "_coulomb_matrix_0d", forbidden)
    monkeypatch.setattr(eeq, "coulomb_matrix_3d", forbidden)
    result = model(structure, return_energy=True, solve_mode="matrix-free")
    for actual, expected in zip(result, reference):
        torch.testing.assert_close(actual, expected, atol=1e-9, rtol=1e-9)
    torch.testing.assert_close(
        model(structure, solve_mode="matrix-free"), reference[0]
    )


def test_padding_and_ghosts() -> None:
    """Interior ghosts and packed padding stay zero and do not interact."""
    structure = load_structure("LiH", DD_DOUBLE, 1.0)
    ghost = Structure(
        numbers=torch.cat(
            (
                structure.numbers[:1],
                structure.numbers.new_zeros(1),
                structure.numbers[1:],
            )
        ),
        positions=torch.cat(
            (
                structure.positions[:1],
                structure.positions.new_zeros(1, 3),
                structure.positions[1:],
            )
        ),
        charge=structure.charge,
    )
    model = _model()
    q, e = model(ghost, solve_mode="matrix-free", return_energy=True)
    qref, eref = model(structure, return_energy=True)
    torch.testing.assert_close(q[[0, 2]], qref)
    torch.testing.assert_close(e[[0, 2]], eref)
    assert q[1] == 0 and e[1] == 0
    batch = load_batch(["LiH", "ZnOOH-"], DD_DOUBLE, [0.0, -1.0])
    q, e = model(batch, solve_mode="matrix-free", return_energy=True)
    assert torch.equal(q[0, 2:], torch.zeros_like(q[0, 2:]))
    assert torch.equal(e[0, 2:], torch.zeros_like(e[0, 2:]))


@pytest.mark.parametrize("budget", [0, 4096, 65536])
def test_coarse_memory_budget(
    budget: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Balancing setup bounds its coarse dimension before allocating AZ."""
    n = 32
    positions = torch.zeros(n, 3, **DD_DOUBLE)
    positions[:, 0] = 3 * torch.arange(n, device=DEVICE)
    numbers = torch.ones(n, dtype=torch.long, device=DEVICE)
    structure = Structure(numbers=numbers, positions=positions)
    model = _model(coarse_memory=budget, cell_size=1.0)
    sums = mf.pair_cell_sums
    sizes = []

    def bounded_sums(*args: Any, **kwargs: Any) -> Tensor:
        k = args[3]
        sizes.append(k)
        # The implementation reserves eight float64 NxK buffers for
        # assembly and coarse workspaces; the caller's budget bounds K.
        assert k <= n
        assert 8 * 8 * n * k <= budget
        return sums(*args, **kwargs)

    monkeypatch.setattr(mf, "pair_cell_sums", bounded_sums)
    q = model(structure, solve_mode="matrix-free")
    torch.testing.assert_close(q, torch.zeros_like(q), atol=1e-10, rtol=0)
    assert bool(sizes) == (budget > 0)


def test_preconditioner_rebuilt() -> None:
    """Moved geometries rebuild the preconditioner without changing stored state."""
    structure = load_structure("ZnOOH-", DD_DOUBLE, -1.0)
    model = _model(cell_size=1.0)
    first = model(structure, solve_mode="matrix-free")
    moved = structure.replace(positions=structure.positions * 1.2)
    torch.testing.assert_close(
        model(moved, solve_mode="matrix-free"),
        model(moved),
        atol=1e-9,
        rtol=1e-9,
    )
    torch.testing.assert_close(
        model(structure, solve_mode="matrix-free"), first
    )


def test_vmap_structures_models_and_charge() -> None:
    """Structure, model-tree and charge batching match independent dense solves."""
    model = _model()
    batch = load_batch(["LiH", "ZnOOH-"], DD_DOUBLE, [0.0, -1.0])
    with no_vmap_fallback():
        q = vmap(lambda s: model(s, solve_mode="matrix-free"))(batch)
    torch.testing.assert_close(q, model(batch), atol=1e-9, rtol=1e-9)
    structure = load_structure("SiH4", DD_DOUBLE, 0.0)
    models = [model, model.replace(chi=model.chi * 1.1)]
    with no_vmap_fallback():
        q = vmap(lambda m: m(structure, solve_mode="matrix-free"))(
            stack(models)
        )
        totals = torch.tensor([-1.0, 0.0, 1.0], **DD_DOUBLE)
        charged = vmap(
            lambda total: model(
                structure.replace(charge=total), solve_mode="matrix-free"
            )
        )(totals)
    torch.testing.assert_close(
        q, torch.stack([m(structure) for m in models]), atol=1e-9, rtol=1e-9
    )
    torch.testing.assert_close(charged.sum(-1), totals, atol=1e-12, rtol=0)


def test_iteration_limit() -> None:
    """A deliberately incomplete solve raises with its original diagnostics."""
    structure = load_structure("ZnOOH-", DD_DOUBLE, -1.0)
    model = _model(coarse_memory=0, mixed_precision=False, maxiter=1)
    with pytest.raises(ConvergenceError, match="matrix-free"):
        model(structure, solve_mode="matrix-free")


def test_periodic_rejected() -> None:
    """The molecular backend explicitly rejects periodic Ewald inputs."""
    structure = load_periodic("CO2+", DD_DOUBLE)
    with pytest.raises(NotImplementedError, match="nonperiodic"):
        _model().solve(
            structure,
            torch.zeros_like(structure.numbers, dtype=torch.double),
            solve_mode="matrix-free",
        )


@pytest.mark.parametrize(
    "options",
    [
        {"tile": 0},
        {"coarse_memory": -1},
        {"cell_size": 0},
        {"cell_size": float("nan")},
        {"maxiter": 0},
    ],
)
def test_invalid_options(options: dict[str, Any]) -> None:
    """Invalid memory and execution bounds fail at configuration time."""
    with pytest.raises(ValueError):
        eeq.MatrixFreeOptions(**options)


def test_options_immutable() -> None:
    """Solver options remain hashable static context in the model pytree."""
    options = eeq.MatrixFreeOptions()
    assert hash(options) == hash(eeq.MatrixFreeOptions())
    with pytest.raises(FrozenInstanceError):
        options.tile = 2  # type: ignore[misc]


def test_convenience_functions() -> None:
    """All charge/energy convenience APIs forward the matrix-free mode."""
    structure = load_structure("LiH", DD_DOUBLE, 0.0)
    q, e = eeq.get_eeq(structure, solve_mode="matrix-free", return_energy=True)
    torch.testing.assert_close(
        eeq.get_charges(structure, solve_mode="matrix-free"), q
    )
    torch.testing.assert_close(
        eeq.get_energy(structure, solve_mode="matrix-free"), e
    )


@pytest.mark.parametrize("vectorized", [False, True])
def test_independent_charge_acceptance(
    vectorized: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A passing solver status cannot hide a failed physical charge check."""
    structure = load_structure("LiH", DD_DOUBLE, 0.0)
    model = _model()
    constrained_solve = linsolve.constrained_solve

    def corrupted_constraint(
        *args: Any, **kwargs: Any
    ) -> linsolve.ConstrainedSolveResult:
        result = constrained_solve(*args, **kwargs)
        q = result.solution + 1e-6
        residual = (q.sum(-1) - args[3].squeeze(-1)).abs().detach()
        return result._replace(solution=q, constraint_residual=residual)

    monkeypatch.setattr(linsolve, "constrained_solve", corrupted_constraint)
    if vectorized:
        charges = vmap(lambda s: model(s, solve_mode="matrix-free"))(
            load_batch(["LiH", "SiH4"], DD_DOUBLE)
        )
        assert torch.isnan(charges).all()
    else:
        with pytest.raises(ConvergenceError, match="total charge"):
            model(structure, solve_mode="matrix-free")


@pytest.mark.parametrize("charge", [0.0, 1.0])
def test_empty_padded_system(charge: float) -> None:
    """An all-padding system is zero when neutral and rejects nonzero charge."""
    structure = Structure(
        numbers=torch.zeros(4, dtype=torch.long, device=DEVICE),
        positions=torch.zeros(4, 3, **DD_DOUBLE),
        charge=torch.tensor(charge, **DD_DOUBLE),
    )
    model = _model()
    if charge != 0:
        with pytest.raises(ConvergenceError):
            model.solve(
                structure, torch.zeros(4, **DD_DOUBLE), solve_mode="matrix-free"
            )
    else:
        q = model.solve(
            structure, torch.zeros(4, **DD_DOUBLE), solve_mode="matrix-free"
        )
        assert torch.equal(q, torch.zeros_like(q))


def test_coarse_factorization_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """Jacobi retains the physical result when a coarse factorization fails."""
    structure = load_structure("ZnOOH-", DD_DOUBLE, -1.0)

    def failed_factorization(*args: Any, **kwargs: Any) -> Any:
        raise torch.linalg.LinAlgError("coarse factorization failed")

    monkeypatch.setattr(linsolve, "balancing", failed_factorization)
    model = _model()
    torch.testing.assert_close(
        model(structure, solve_mode="matrix-free"),
        model(structure),
        atol=1e-9,
        rtol=1e-9,
    )


@requires_compile
def test_compilation_rejected() -> None:
    """Compiling the whole model rejects the unsupported pair-product Function."""
    structure = load_structure("LiH", DD_DOUBLE, 0.0)
    model = _model()

    def evaluate(positions: Tensor) -> Tensor:
        return model.solve(
            structure.replace(positions=positions),
            torch.zeros_like(structure.numbers, dtype=positions.dtype),
            solve_mode="matrix-free",
        )

    compiled = torch.compile(evaluate, backend="eager")
    with pytest.raises(RuntimeError, match="matrix-free.*torch.compile"):
        compiled(structure.positions)
