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
"""Implicit derivatives and failed-system isolation of matrix-free EEQ."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest
import torch
from tad_mctc.autograd import dgradcheck, dgradgradcheck, no_vmap_fallback
from tad_mctc.exceptions import ConvergenceError
from tad_mctc.io.structure import Structure
from tad_mctc.typing import DD, Tensor
from torch.func import jacfwd, jacrev, vmap

from tad_multicharge.model import _matrix_free as mf
from tad_multicharge.model import eeq

from ..conftest import DEVICE
from ..utils import load_batch, load_structure

DD_DOUBLE: DD = {"dtype": torch.double, "device": DEVICE}
DERIVATIVES = {
    "reverse": jacrev,
    "forward": jacfwd,
    "reverse-reverse": lambda f: jacrev(jacrev(f)),
    "forward-reverse": lambda f: jacfwd(jacrev(f)),
    "reverse-forward": lambda f: jacrev(jacfwd(f)),
    "forward-forward": lambda f: jacfwd(jacfwd(f)),
    "third": lambda f: jacfwd(jacrev(jacrev(f))),
}


def _model(mixed: bool = True) -> eeq.EEQModel:
    return eeq.EEQModel.param2019(**DD_DOUBLE).replace(
        matrix_free=eeq.MatrixFreeOptions(
            tile=2, fused=False, mixed_precision=mixed
        )
    )


@pytest.mark.parametrize("derivative", DERIVATIVES)
@pytest.mark.parametrize("mixed", [False, True])
def test_position_derivatives(derivative: str, mixed: bool) -> None:
    """Charges and energies agree with dense first through third derivatives."""
    structure = load_structure("LiH", DD_DOUBLE, 1.0)
    model = _model(mixed)

    def observable(mode: str) -> Callable:
        def evaluate(positions: Tensor) -> tuple[Tensor, Tensor]:
            q, e = model(
                structure.replace(positions=positions),
                solve_mode=mode,
                return_energy=True,
            )
            return q, e.sum()

        return evaluate

    # Tuple-valued nested jacrev has nested output trees. Test charge and
    # scalar energy separately to keep the reference comparison explicit.
    for index in (0, 1):
        f = lambda p: observable("matrix-free")(p)[index]
        reference = lambda p: observable("schur")(p)[index]
        actual = DERIVATIVES[derivative](f)(structure.positions)
        expected = DERIVATIVES[derivative](reference)(structure.positions)
        torch.testing.assert_close(actual, expected, atol=1e-8, rtol=1e-8)


@pytest.mark.parametrize("param", ["chi", "kcn", "eta", "rad"])
@pytest.mark.parametrize("differentiate", [jacrev, jacfwd])
def test_parameter_and_force_derivatives(
    param: str, differentiate: Callable
) -> None:
    """Every physical parameter contributes correctly to charges, energies and forces."""
    structure = load_structure("SiH4", DD_DOUBLE, 1.0)
    model = _model()
    parameter = getattr(model, param)[1:]

    def observables(mode: str) -> Callable:
        def evaluate(value: Tensor, positions: Tensor) -> tuple[Tensor, Tensor]:
            full = torch.cat((getattr(model, param)[:1], value))
            q, e = model.replace(**{param: full})(
                structure.replace(positions=positions),
                solve_mode=mode,
                return_energy=True,
            )
            return q, e.sum()

        return evaluate

    args = (parameter, structure.positions)
    for actual, expected in zip(
        differentiate(observables("matrix-free"), argnums=0)(*args),
        differentiate(observables("schur"), argnums=0)(*args),
    ):
        torch.testing.assert_close(actual, expected, atol=1e-8, rtol=1e-8)
    energy = lambda value, positions: observables("matrix-free")(
        value, positions
    )[1]
    reference_energy = lambda value, positions: observables("schur")(
        value, positions
    )[1]
    mixed = differentiate(jacrev(energy, argnums=1), argnums=0)(*args)
    expected = differentiate(jacrev(reference_energy, argnums=1), argnums=0)(
        *args
    )
    torch.testing.assert_close(mixed, expected, atol=1e-8, rtol=1e-8)


@pytest.mark.parametrize("derivative", DERIVATIVES)
def test_total_charge_derivatives(derivative: str) -> None:
    """The constraint retains its first, second and third implicit derivatives."""
    structure = load_structure("LiH", DD_DOUBLE, 1.0)
    model = _model()
    assert structure.charge is not None
    for index in (0, 1):

        def observable(charge: Tensor, mode: str) -> Tensor:
            q, e = model(
                structure.replace(charge=charge),
                solve_mode=mode,
                return_energy=True,
            )
            return q if index == 0 else e.sum()

        actual = DERIVATIVES[derivative](
            lambda q: observable(q, "matrix-free")
        )(structure.charge)
        expected = DERIVATIVES[derivative](lambda q: observable(q, "schur"))(
            structure.charge
        )
        torch.testing.assert_close(actual, expected, atol=1e-8, rtol=1e-8)


@pytest.mark.parametrize("param", ["chi", "kcn", "eta", "rad"])
@pytest.mark.parametrize("differentiate", [jacrev, jacfwd])
@pytest.mark.parametrize(
    "shared", [False, True], ids=["vmap-derivative", "derivative-vmap"]
)
@pytest.mark.parametrize("nonfinite", [False, True], ids=["raw", "nonfinite"])
def test_failed_system_derivatives(
    param: str,
    differentiate: Callable,
    shared: bool,
    nonfinite: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An indefinite SiH4 system cannot poison the shared-parameter LiH derivatives."""
    if nonfinite:
        projected_cg = mf.linsolve.constrained.projected_cg

        def failed_iterates(
            *args: Any, **kwargs: Any
        ) -> mf.linsolve.LinearSolveResult:
            result = projected_cg(*args, **kwargs)
            converged = result.status == mf.linsolve.STATUS_CONVERGED
            return result._replace(
                solution=torch.where(
                    converged.unsqueeze(-1), result.solution, torch.nan
                )
            )

        # Inject before implicit AD attaches, including its saved solution.
        monkeypatch.setattr(
            mf.linsolve.constrained, "projected_cg", failed_iterates
        )
    batch = load_batch(["LiH", "SiH4"], DD_DOUBLE, [0.0, 0.0])
    lih = load_structure("LiH", DD_DOUBLE, 0.0)
    model = _model()
    eta = model.eta.clone()
    eta[14] = -10.0
    model = model.replace(eta=eta)
    parameter = getattr(model, param)[1:]

    def evaluate(
        value: Tensor, structure: Structure, mode: str = "matrix-free"
    ) -> tuple[Tensor, Tensor]:
        full = torch.cat((getattr(model, param)[:1], value))
        q, e = model.replace(**{param: full})(
            structure, solve_mode=mode, return_energy=True
        )
        return q, e.sum(-1)

    with pytest.raises(ConvergenceError):
        evaluate(parameter, batch)
    primal = vmap(lambda s: evaluate(parameter, s))(batch)
    assert torch.isfinite(primal[0][0]).all()
    assert torch.isfinite(primal[1][0])
    assert torch.isnan(primal[0][1]).all()
    assert torch.isnan(primal[1][1])

    with no_vmap_fallback():
        if shared:
            result = differentiate(
                lambda value: vmap(lambda s: evaluate(value, s))(batch)
            )(parameter)
        else:
            result = vmap(
                differentiate(evaluate, argnums=0), in_dims=(None, 0)
            )(parameter, batch)
    expected = differentiate(lambda value: evaluate(value, lih, "schur"))(
        parameter
    )
    torch.testing.assert_close(
        result[0][0, :2], expected[0], atol=1e-9, rtol=1e-9
    )
    torch.testing.assert_close(result[1][0], expected[1], atol=1e-9, rtol=1e-9)
    assert torch.equal(result[0][0, 2:], torch.zeros_like(result[0][0, 2:]))
    # H and Si are entries 0 and 13 after excluding the padding entry.
    assert torch.isnan(result[0][1, :, [0, 13]]).all()
    assert torch.isnan(result[1][1, [0, 13]]).all()


@pytest.mark.grad
@pytest.mark.parametrize("check", [dgradcheck, dgradgradcheck])
def test_finite_differences(check: Callable) -> None:
    """Implicit force and charge responses satisfy numerical gradient checks."""
    structure = load_structure("LiH", DD_DOUBLE, 0.0)
    model = _model()
    positions = structure.positions.clone().requires_grad_(True)

    def charges(p: Tensor) -> Tensor:
        return model(structure.replace(positions=p), solve_mode="matrix-free")

    assert check(charges, (positions,))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_native_backward(dtype: torch.dtype) -> None:
    """Native autograd, including fused CUDA products, retains all parameter gradients."""
    dd: DD = {"dtype": dtype, "device": DEVICE}
    structure = load_structure("ZnOOH-", dd, -1.0)
    positions = structure.positions.clone().requires_grad_(True)
    structure = structure.replace(positions=positions)
    model = eeq.EEQModel.param2019(**dd)
    names = ("chi", "kcn", "eta", "rad")
    parameters = tuple(
        getattr(model, name).clone().requires_grad_(True) for name in names
    )
    model = model.replace(**dict(zip(names, parameters)))

    def objective(mode: str) -> Tensor:
        q, e = model(structure, return_energy=True, solve_mode=mode)
        # The charge term ensures a nonzero adjoint, independently of the
        # variational energy's cancellation of the charge response.
        return e.sum() + 0.3 * q.square().sum()

    actual = torch.autograd.grad(
        objective("matrix-free"), (positions, *parameters)
    )
    expected = torch.autograd.grad(objective("schur"), (positions, *parameters))
    tolerance = 2e-4 if dtype == torch.float32 else 1e-8
    for gradient, reference in zip(actual, expected):
        torch.testing.assert_close(
            gradient, reference, atol=tolerance, rtol=tolerance
        )


@pytest.mark.parametrize("param", ["chi", "kcn", "eta", "rad"])
def test_parameter_hessian(param: str) -> None:
    """Second parameter derivatives include the implicit matrix response."""
    structure = load_structure("SiH4", DD_DOUBLE, 1.0)
    model = _model()
    indices = torch.tensor([1, 14], device=DEVICE)
    parameter = getattr(model, param)

    def energy(value: Tensor, mode: str) -> Tensor:
        full = parameter.index_copy(0, indices, value)
        return model.replace(**{param: full})(
            structure, solve_mode=mode, return_energy=True
        )[1].sum()

    selected = parameter[indices]
    actual = jacfwd(jacrev(lambda value: energy(value, "matrix-free")))(
        selected
    )
    expected = jacrev(jacrev(lambda value: energy(value, "schur")))(selected)
    torch.testing.assert_close(actual, expected, atol=1e-8, rtol=1e-8)


@pytest.mark.parametrize(
    "derivative",
    [
        "reverse-reverse",
        "forward-reverse",
        "reverse-forward",
        "forward-forward",
        "third",
    ],
)
def test_failed_system_higher_derivatives(derivative: str) -> None:
    """Per-system higher derivatives remain invalid after a curvature failure."""
    batch = load_batch(["LiH", "SiH4"], DD_DOUBLE, [0.0, 0.0])
    model = _model()
    eta = model.eta.clone()
    eta[14] = -10.0
    model = model.replace(eta=eta)
    direction = torch.zeros_like(model.chi)
    direction[1] = 1.0

    def energy(
        offset: Tensor, structure: Structure, mode: str = "matrix-free"
    ) -> Tensor:
        return model.replace(chi=model.chi + offset * direction)(
            structure, solve_mode=mode, return_energy=True
        )[1].sum()

    offset = torch.zeros((), **DD_DOUBLE)
    with no_vmap_fallback():
        actual = vmap(DERIVATIVES[derivative](energy), in_dims=(None, 0))(
            offset, batch
        )
    lih = load_structure("LiH", DD_DOUBLE, 0.0)
    expected = DERIVATIVES[derivative](
        lambda value: energy(value, lih, "schur")
    )(offset)
    torch.testing.assert_close(actual[0], expected, atol=1e-8, rtol=1e-8)
    assert torch.isnan(actual[1]).all()


def test_saved_graph_is_tiled() -> None:
    """The first-derivative graph retains vectors and tiles, not an NxN matrix."""
    n = 64
    positions = torch.zeros(n, 3, **DD_DOUBLE)
    positions[:, 0] = 3 * torch.arange(n, device=DEVICE)
    positions.requires_grad_(True)
    structure = Structure(
        numbers=torch.ones(n, dtype=torch.long, device=DEVICE),
        positions=positions,
    )
    cn = torch.linspace(0.0, 2.0, n, **DD_DOUBLE)
    model = _model(False).replace(
        matrix_free=eeq.MatrixFreeOptions(
            tile=4, coarse_memory=0, mixed_precision=False, fused=False
        )
    )
    saved_shapes = []

    def pack(tensor: Tensor) -> Tensor:
        saved_shapes.append(tuple(tensor.shape))
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
        _, energy = model.solve(
            structure, cn, solve_mode="matrix-free", return_energy=True
        )
    assert saved_shapes
    assert all(shape != (n, n) for shape in saved_shapes)
    assert (
        max(torch.tensor(shape).prod().item() for shape in saved_shapes) < n * n
    )
    gradient = torch.autograd.grad(energy.sum(), positions)[0]
    reference_energy = model.solve(structure, cn, return_energy=True)[1]
    reference = torch.autograd.grad(reference_energy.sum(), positions)[0]
    torch.testing.assert_close(gradient, reference, atol=1e-8, rtol=1e-8)
