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


def _reverse_reverse(function: Callable) -> Callable:
    first_derivative = jacrev(function)
    return jacrev(first_derivative)


def _forward_reverse(function: Callable) -> Callable:
    first_derivative = jacrev(function)
    return jacfwd(first_derivative)


def _reverse_forward(function: Callable) -> Callable:
    first_derivative = jacfwd(function)
    return jacrev(first_derivative)


def _forward_forward(function: Callable) -> Callable:
    first_derivative = jacfwd(function)
    return jacfwd(first_derivative)


def _third_derivative(function: Callable) -> Callable:
    first_derivative = jacrev(function)
    second_derivative = jacrev(first_derivative)
    return jacfwd(second_derivative)


DERIVATIVES = {
    "reverse": jacrev,
    "forward": jacfwd,
    "reverse-reverse": _reverse_reverse,
    "forward-reverse": _forward_reverse,
    "reverse-forward": _reverse_forward,
    "forward-forward": _forward_forward,
    "third": _third_derivative,
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

    def evaluate(positions: Tensor, mode: str) -> tuple[Tensor, Tensor]:
        displaced_structure = structure.replace(positions=positions)
        charges, energies = model(
            displaced_structure, solve_mode=mode, return_energy=True
        )
        return charges, energies.sum()

    def matrix_free_charges(positions: Tensor) -> Tensor:
        charges, _ = evaluate(positions, "matrix-free")
        return charges

    def dense_charges(positions: Tensor) -> Tensor:
        charges, _ = evaluate(positions, "schur")
        return charges

    def matrix_free_energy(positions: Tensor) -> Tensor:
        _, total_energy = evaluate(positions, "matrix-free")
        return total_energy

    def dense_energy(positions: Tensor) -> Tensor:
        _, total_energy = evaluate(positions, "schur")
        return total_energy

    # Tuple-valued nested jacrev has nested output trees. Test charge and
    # scalar energy separately to keep the reference comparison explicit.
    observable_pairs = (
        (matrix_free_charges, dense_charges),
        (matrix_free_energy, dense_energy),
    )
    derivative_transform = DERIVATIVES[derivative]

    for observable, reference_observable in observable_pairs:
        evaluate_derivative = derivative_transform(observable)
        evaluate_reference_derivative = derivative_transform(
            reference_observable
        )

        actual = evaluate_derivative(structure.positions)
        expected = evaluate_reference_derivative(structure.positions)

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

    def evaluate(
        value: Tensor, positions: Tensor, mode: str
    ) -> tuple[Tensor, Tensor]:
        full_parameter = torch.cat((getattr(model, param)[:1], value))
        parameter_model = model.replace(**{param: full_parameter})
        displaced_structure = structure.replace(positions=positions)
        charges, energies = parameter_model(
            displaced_structure, solve_mode=mode, return_energy=True
        )
        return charges, energies.sum()

    def matrix_free_observables(
        value: Tensor, positions: Tensor
    ) -> tuple[Tensor, Tensor]:
        return evaluate(value, positions, "matrix-free")

    def dense_observables(
        value: Tensor, positions: Tensor
    ) -> tuple[Tensor, Tensor]:
        return evaluate(value, positions, "schur")

    def matrix_free_energy(value: Tensor, positions: Tensor) -> Tensor:
        _, total_energy = matrix_free_observables(value, positions)
        return total_energy

    def dense_energy(value: Tensor, positions: Tensor) -> Tensor:
        _, total_energy = dense_observables(value, positions)
        return total_energy

    arguments = (parameter, structure.positions)
    evaluate_parameter_derivative = differentiate(
        matrix_free_observables, argnums=0
    )
    evaluate_reference_derivative = differentiate(dense_observables, argnums=0)

    parameter_derivatives = evaluate_parameter_derivative(*arguments)
    reference_derivatives = evaluate_reference_derivative(*arguments)

    for actual, expected in zip(parameter_derivatives, reference_derivatives):
        torch.testing.assert_close(actual, expected, atol=1e-8, rtol=1e-8)

    # Differentiate the position gradient with respect to parameter argument 0.
    position_gradient = jacrev(matrix_free_energy, argnums=1)
    mixed_derivative = differentiate(position_gradient, argnums=0)
    reference_position_gradient = jacrev(dense_energy, argnums=1)
    reference_mixed_derivative = differentiate(
        reference_position_gradient, argnums=0
    )

    mixed = mixed_derivative(*arguments)
    expected = reference_mixed_derivative(*arguments)

    torch.testing.assert_close(mixed, expected, atol=1e-8, rtol=1e-8)


@pytest.mark.parametrize("derivative", DERIVATIVES)
def test_total_charge_derivatives(derivative: str) -> None:
    """The constraint retains its first, second and third implicit derivatives."""
    structure = load_structure("LiH", DD_DOUBLE, 1.0)
    model = _model()

    assert structure.charge is not None

    def evaluate(total_charge: Tensor, mode: str) -> tuple[Tensor, Tensor]:
        charged_structure = structure.replace(charge=total_charge)
        return model(charged_structure, solve_mode=mode, return_energy=True)

    def matrix_free_charges(total_charge: Tensor) -> Tensor:
        charges, _ = evaluate(total_charge, "matrix-free")
        return charges

    def dense_charges(total_charge: Tensor) -> Tensor:
        charges, _ = evaluate(total_charge, "schur")
        return charges

    def matrix_free_energy(total_charge: Tensor) -> Tensor:
        _, energies = evaluate(total_charge, "matrix-free")
        return energies.sum()

    def dense_energy(total_charge: Tensor) -> Tensor:
        _, energies = evaluate(total_charge, "schur")
        return energies.sum()

    observable_pairs = (
        (matrix_free_charges, dense_charges),
        (matrix_free_energy, dense_energy),
    )
    derivative_transform = DERIVATIVES[derivative]

    for observable, reference_observable in observable_pairs:
        evaluate_derivative = derivative_transform(observable)
        evaluate_reference_derivative = derivative_transform(
            reference_observable
        )

        actual = evaluate_derivative(structure.charge)
        expected = evaluate_reference_derivative(structure.charge)

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
        value: Tensor, current_structure: Structure, mode: str = "matrix-free"
    ) -> tuple[Tensor, Tensor]:
        full_parameter = torch.cat((getattr(model, param)[:1], value))
        parameter_model = model.replace(**{param: full_parameter})
        charges, energies = parameter_model(
            current_structure, solve_mode=mode, return_energy=True
        )
        return charges, energies.sum(-1)

    def evaluate_structure(
        current_structure: Structure,
    ) -> tuple[Tensor, Tensor]:
        return evaluate(parameter, current_structure)

    def evaluate_batch(value: Tensor) -> tuple[Tensor, Tensor]:
        def evaluate_with_parameter(
            current_structure: Structure,
        ) -> tuple[Tensor, Tensor]:
            return evaluate(value, current_structure)

        evaluate_structures = vmap(evaluate_with_parameter)
        return evaluate_structures(batch)

    def dense_reference(value: Tensor) -> tuple[Tensor, Tensor]:
        return evaluate(value, lih, "schur")

    with pytest.raises(ConvergenceError):
        evaluate(parameter, batch)

    evaluate_structures = vmap(evaluate_structure)
    primal = evaluate_structures(batch)

    assert torch.isfinite(primal[0][0]).all()
    assert torch.isfinite(primal[1][0])
    assert torch.isnan(primal[0][1]).all()
    assert torch.isnan(primal[1][1])

    with no_vmap_fallback():
        if shared:
            evaluate_batch_derivative = differentiate(evaluate_batch)
            result = evaluate_batch_derivative(parameter)
        else:
            system_derivative = differentiate(evaluate, argnums=0)
            evaluate_derivatives = vmap(system_derivative, in_dims=(None, 0))
            result = evaluate_derivatives(parameter, batch)

    evaluate_reference_derivative = differentiate(dense_reference)
    expected = evaluate_reference_derivative(parameter)

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

    def charges(positions: Tensor) -> Tensor:
        displaced_structure = structure.replace(positions=positions)
        return model(displaced_structure, solve_mode="matrix-free")

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
        charges, energies = model(
            structure, return_energy=True, solve_mode=mode
        )

        # The charge term ensures a nonzero adjoint, independently of the
        # variational energy's cancellation of the charge response.
        return energies.sum() + 0.3 * charges.square().sum()

    differentiable_inputs = (positions, *parameters)
    matrix_free_objective = objective("matrix-free")
    actual = torch.autograd.grad(matrix_free_objective, differentiable_inputs)

    dense_objective = objective("schur")
    expected = torch.autograd.grad(dense_objective, differentiable_inputs)

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
        full_parameter = parameter.index_copy(0, indices, value)
        parameter_model = model.replace(**{param: full_parameter})
        _, energies = parameter_model(
            structure, solve_mode=mode, return_energy=True
        )
        return energies.sum()

    def matrix_free_energy(value: Tensor) -> Tensor:
        return energy(value, "matrix-free")

    def dense_energy(value: Tensor) -> Tensor:
        return energy(value, "schur")

    selected_parameters = parameter[indices]
    matrix_free_gradient = jacrev(matrix_free_energy)
    matrix_free_hessian = jacfwd(matrix_free_gradient)
    dense_gradient = jacrev(dense_energy)
    dense_hessian = jacrev(dense_gradient)

    actual = matrix_free_hessian(selected_parameters)
    expected = dense_hessian(selected_parameters)

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
    lih = load_structure("LiH", DD_DOUBLE, 0.0)
    model = _model()
    eta = model.eta.clone()
    eta[14] = -10.0
    model = model.replace(eta=eta)
    direction = torch.zeros_like(model.chi)
    direction[1] = 1.0

    def energy(
        offset: Tensor, current_structure: Structure, mode: str = "matrix-free"
    ) -> Tensor:
        parameter_model = model.replace(chi=model.chi + offset * direction)
        _, energies = parameter_model(
            current_structure, solve_mode=mode, return_energy=True
        )
        return energies.sum()

    def dense_energy(value: Tensor) -> Tensor:
        return energy(value, lih, "schur")

    offset = torch.zeros((), **DD_DOUBLE)

    with no_vmap_fallback():
        energy_derivative = DERIVATIVES[derivative](energy)
        evaluate_derivatives = vmap(energy_derivative, in_dims=(None, 0))
        actual = evaluate_derivatives(offset, batch)

    evaluate_reference_derivative = DERIVATIVES[derivative](dense_energy)
    expected = evaluate_reference_derivative(offset)

    torch.testing.assert_close(actual[0], expected, atol=1e-8, rtol=1e-8)
    assert torch.isnan(actual[1]).all()


def test_saved_graph_is_tiled() -> None:
    """The first-derivative graph retains vectors and tiles, not an NxN matrix."""
    natoms = 64
    positions = torch.zeros(natoms, 3, **DD_DOUBLE)
    positions[:, 0] = 3 * torch.arange(natoms, device=DEVICE)
    positions.requires_grad_(True)
    structure = Structure(
        numbers=torch.ones(natoms, dtype=torch.long, device=DEVICE),
        positions=positions,
    )
    coordination_numbers = torch.linspace(0.0, 2.0, natoms, **DD_DOUBLE)
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
            structure,
            coordination_numbers,
            solve_mode="matrix-free",
            return_energy=True,
        )

    assert saved_shapes
    assert all(shape != (natoms, natoms) for shape in saved_shapes)
    assert (
        max(torch.tensor(shape).prod().item() for shape in saved_shapes)
        < natoms * natoms
    )

    gradient = torch.autograd.grad(energy.sum(), positions)[0]
    reference_energy = model.solve(
        structure, coordination_numbers, return_energy=True
    )[1]
    reference = torch.autograd.grad(reference_energy.sum(), positions)[0]

    torch.testing.assert_close(gradient, reference, atol=1e-8, rtol=1e-8)
