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

            injected = result._replace(
                solution=torch.where(
                    converged.unsqueeze(-1), result.solution, torch.nan
                )
            )

            # Only the numerical iterate changes; diagnostics must describe
            # the original solve, including any refinement or exact fallback.
            for name in (
                "status",
                "residual",
                "target",
                "iterations",
                "refinement_status",
                "fallback_iterations",
            ):
                assert getattr(injected, name) is getattr(result, name)

            return injected

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


@pytest.mark.parametrize("differentiate", [jacrev, jacfwd])
@pytest.mark.parametrize("solve_mode", ["schur", "cg", "matrix-free"])
def test_empty_batch_derivatives(
    differentiate: Callable, solve_mode: str
) -> None:
    """Empty position and physical-parameter derivatives match dense semantics."""
    structure = Structure(
        numbers=torch.empty(0, 2, dtype=torch.long, device=DEVICE),
        positions=torch.empty(0, 2, 3, **DD_DOUBLE),
        charge=torch.empty(0, **DD_DOUBLE),
    )
    model = _model()

    def total_energy(
        positions: Tensor, chi: Tensor, total_charge: Tensor
    ) -> Tensor:
        current = structure.replace(positions=positions, charge=total_charge)
        _, energies = model.replace(chi=chi)(
            current,
            solve_mode=solve_mode,
            return_energy=True,
        )
        return energies.sum()

    evaluate_derivative = differentiate(total_energy, argnums=(0, 1, 2))
    gradients = evaluate_derivative(
        structure.positions, model.chi, structure.charge
    )

    assert gradients[0].shape == (0, 2, 3)
    assert gradients[1].shape == model.chi.shape
    assert gradients[2].shape == (0,)
    for gradient in gradients:
        assert torch.equal(gradient, torch.zeros_like(gradient))


@pytest.mark.parametrize("differentiate", [jacrev, jacfwd])
@pytest.mark.parametrize("configured", [False, True])
@pytest.mark.parametrize(
    "tensor_cn", [False, True], ids=["static-cn", "tensor-cn"]
)
@pytest.mark.parametrize("solve_mode", ["schur", "cg", "matrix-free"])
def test_complete_model_pytree(
    differentiate: Callable,
    configured: bool,
    tensor_cn: bool,
    solve_mode: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Whole-model AD preserves static context and differentiates the CN subtree."""
    from tad_mctc.ncoord.common import CNModel
    from tad_mctc.tree import leaf_paths
    from torch.utils._pytree import tree_flatten

    structure = load_structure("SiH4", DD_DOUBLE, 1.0)
    model = eeq.EEQModel.param2019(**DD_DOUBLE)
    if tensor_cn:
        cn_model = model.cn.replace(
            cn_max=torch.tensor(model.cn.cn_max, **DD_DOUBLE),
            rcov=model.cn.rcov(**DD_DOUBLE),
            en=model.cn.en(**DD_DOUBLE),
        )
        model = model.replace(cn=cn_model)

    original_paths = leaf_paths(model)
    original_leaves, _ = tree_flatten(model)
    if configured:
        model = model.replace(
            matrix_free=eeq.MatrixFreeOptions(
                tile=2,
                coarse_memory=0,
                mixed_precision=False,
                fused=False,
            )
        )

    assert leaf_paths(model) == original_paths
    for original, current in zip(original_leaves, tree_flatten(model)[0]):
        assert original is current
    assert all("matrix_free" not in path for path in leaf_paths(model))

    def forbidden_cells(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError(
            "Whole-model differentiation built data-dependent cells"
        )

    monkeypatch.setattr(mf.linsolve, "cell_index", forbidden_cells)

    def model_energy(current_model: eeq.EEQModel) -> Tensor:
        _, energies = current_model(
            structure, solve_mode=solve_mode, return_energy=True
        )
        return energies.sum()

    def dense_energy(current_model: eeq.EEQModel) -> Tensor:
        _, energies = current_model(
            structure, solve_mode="schur", return_energy=True
        )
        return energies.sum()

    with no_vmap_fallback():
        model_gradient = differentiate(model_energy)(model)
        reference_gradient = jacrev(dense_energy)(model)

    assert isinstance(model_gradient, eeq.EEQModel)
    assert isinstance(model_gradient.cn, CNModel)
    assert model_gradient.matrix_free == model.matrix_free
    assert leaf_paths(model_gradient) == leaf_paths(model)
    actual_leaves, actual_spec = tree_flatten(model_gradient)
    reference_leaves, _ = tree_flatten(reference_gradient)
    assert actual_spec == tree_flatten(model)[1]
    assert len(actual_leaves) == (7 if tensor_cn else 4)

    for actual, expected, primal in zip(
        actual_leaves, reference_leaves, tree_flatten(model)[0]
    ):
        assert actual.shape == primal.shape
        torch.testing.assert_close(actual, expected, atol=1e-9, rtol=1e-9)

    for name in ("chi", "kcn", "eta", "rad"):
        assert getattr(model_gradient, name).shape == getattr(model, name).shape
        assert getattr(model_gradient, name).abs().max() > 0

    assert model_gradient.cn.count is model.cn.count
    assert model_gradient.cn.cutoff == model.cn.cutoff
    if tensor_cn:
        assert model_gradient.cn.cn_max.shape == ()
        assert model_gradient.cn.rcov.shape == model.cn.rcov.shape
        assert model_gradient.cn.rcov.abs().max() > 0
    else:
        assert leaf_paths(model_gradient.cn) == []
        assert model_gradient.cn.cn_max == model.cn.cn_max
        assert model_gradient.cn.rcov is model.cn.rcov


@pytest.mark.parametrize("differentiate", [jacrev, jacfwd])
@pytest.mark.parametrize("failure", ["indefinite", "iteration-limit"])
def test_direct_failed_differentiation(
    differentiate: Callable, failure: str
) -> None:
    """Direct AD raises rather than silently masking a failed molecular solve."""
    model = _model()
    name = "SiH4" if failure == "indefinite" else "ZnOOH-"
    structure = load_structure(name, DD_DOUBLE, 0.0)
    if failure == "indefinite":
        eta = model.eta.clone()
        eta[14] = -10.0
        model = model.replace(eta=eta)
    else:
        model = model.replace(
            matrix_free=eeq.MatrixFreeOptions(
                coarse_memory=0,
                mixed_precision=False,
                maxiter=1,
                fused=False,
            )
        )

    def energy(chi: Tensor) -> Tensor:
        _, energies = model.replace(chi=chi)(
            structure, solve_mode="matrix-free", return_energy=True
        )
        return energies.sum()

    evaluate_derivative = differentiate(energy)
    with pytest.raises(ConvergenceError, match="EEQ matrix-free solve"):
        evaluate_derivative(model.chi)


@pytest.mark.parametrize("solve_mode", ["schur", "cg", "matrix-free"])
@pytest.mark.parametrize("param", ["chi", "kcn", "eta", "rad"])
@pytest.mark.parametrize(
    "composition",
    [
        "vmap-reverse",
        "vmap-forward",
        "reverse-vmap",
        "forward-vmap",
        "vmap-reverse-reverse",
        "vmap-forward-reverse",
        "reverse-forward",
        "forward-reverse",
    ],
)
def test_shared_parameter_compositions(
    solve_mode: str,
    param: str,
    composition: str,
) -> None:
    """Mapped AD retains output axes followed by the ordered parameter axes."""
    batch = load_batch(["LiH", "SiH4"], DD_DOUBLE, [0.0, 1.0])
    singles = [
        Structure(
            numbers=batch.numbers[index],
            positions=batch.positions[index],
            charge=batch.charge[index],
        )
        for index in range(2)
    ]
    model = _model()
    indices = torch.tensor([1, 3, 14], device=DEVICE)
    parameter = getattr(model, param)
    selected = parameter[indices]

    def evaluate(
        value: Tensor, structure: Structure, mode: str = solve_mode
    ) -> Tensor:
        full = parameter.index_copy(0, indices, value)
        current_model = model.replace(**{param: full})
        charges, energies = current_model(
            structure, solve_mode=mode, return_energy=True
        )
        return torch.stack((charges, energies))

    def evaluate_batch(value: Tensor) -> Tensor:
        def evaluate_structure(structure: Structure) -> Tensor:
            return evaluate(value, structure)

        evaluate_structures = vmap(evaluate_structure)
        return evaluate_structures(batch)

    def dense_reference(value: Tensor) -> Tensor:
        return torch.stack(
            [evaluate(value, structure, "schur") for structure in singles]
        )

    if composition.startswith("vmap-"):
        derivative_name = composition.removeprefix("vmap-")
        order = 1 if derivative_name in ("reverse", "forward") else 2
        per_system_derivative = DERIVATIVES[derivative_name](evaluate)
        evaluate_derivative = vmap(per_system_derivative, in_dims=(None, 0))
        arguments = (selected, batch)
    else:
        derivative_name = composition.removesuffix("-vmap")
        order = 1 if derivative_name in ("reverse", "forward") else 2
        evaluate_derivative = DERIVATIVES[derivative_name](evaluate_batch)
        arguments = (selected,)

    reference_name = "reverse" if order == 1 else "reverse-reverse"
    evaluate_reference_derivative = DERIVATIVES[reference_name](dense_reference)

    with no_vmap_fallback():
        actual = evaluate_derivative(*arguments)
        expected = evaluate_reference_derivative(selected)

    assert actual.shape == (2, 2, 5) + (3,) * order
    torch.testing.assert_close(actual, expected, atol=1e-9, rtol=1e-9)
    charge_response = actual[:, 0].sum(dim=1)
    torch.testing.assert_close(
        charge_response, torch.zeros_like(charge_response), atol=1e-10, rtol=0
    )


@pytest.mark.parametrize("differentiate", [jacrev, jacfwd])
@pytest.mark.parametrize("shared", ["chi", "charge"])
@pytest.mark.parametrize("solve_mode", ["schur", "cg", "matrix-free"])
def test_nested_vmap_shared_derivatives(
    differentiate: Callable,
    shared: str,
    solve_mode: str,
) -> None:
    """Two mapped molecular axes preserve shared-input charge and energy derivatives."""
    from tad_mctc.tree import stack

    batch = load_batch(["LiH", "SiH4"], DD_DOUBLE, [0.0, 1.0])
    second_batch = batch.replace(positions=batch.positions * 1.01)
    nested = stack([batch, second_batch])
    model = _model()
    direction = torch.zeros_like(model.chi)
    direction[1] = 1.0
    offset = torch.zeros((), **DD_DOUBLE)

    def evaluate(value: Tensor, structure: Structure, mode: str) -> Tensor:
        current_model = (
            model.replace(chi=model.chi + value * direction)
            if shared == "chi"
            else model
        )
        if shared == "charge":
            structure = structure.replace(charge=structure.charge + value)
        charges, energies = current_model(
            structure, solve_mode=mode, return_energy=True
        )
        return torch.stack((charges, energies))

    def evaluate_batch(value: Tensor) -> Tensor:
        def evaluate_structure(structure: Structure) -> Tensor:
            return evaluate(value, structure, solve_mode)

        evaluate_inner = vmap(evaluate_structure)
        evaluate_outer = vmap(evaluate_inner)
        return evaluate_outer(nested)

    def dense_reference(value: Tensor) -> Tensor:
        outer_results = []
        for current_batch in (batch, second_batch):
            inner_results = []
            for index in range(2):
                single = Structure(
                    numbers=current_batch.numbers[index],
                    positions=current_batch.positions[index],
                    charge=current_batch.charge[index],
                )
                inner_results.append(evaluate(value, single, "schur"))
            outer_results.append(torch.stack(inner_results))
        return torch.stack(outer_results)

    evaluate_derivative = differentiate(evaluate_batch)
    evaluate_reference_derivative = jacrev(dense_reference)

    with no_vmap_fallback():
        actual = evaluate_derivative(offset)
        expected = evaluate_reference_derivative(offset)

    assert actual.shape == (2, 2, 2, 5)
    torch.testing.assert_close(actual, expected, atol=1e-9, rtol=1e-9)
    charge_response = actual[:, :, 0].sum(-1)
    expected_response = (
        torch.ones_like(charge_response)
        if shared == "charge"
        else torch.zeros_like(charge_response)
    )
    torch.testing.assert_close(
        charge_response, expected_response, atol=1e-12, rtol=0
    )


@pytest.mark.parametrize(
    "derivative,quantity",
    [
        ("reverse-reverse", "charges"),
        ("forward-reverse", "charges"),
        ("reverse-forward", "charges"),
        ("forward-forward", "charges"),
        ("third", "charges"),
        ("third", "energy"),
    ],
)
def test_analytically_zero_failed_derivatives(
    derivative: str, quantity: str
) -> None:
    """Linear charges and quadratic energy have zero higher charge derivatives."""
    model = _model()
    eta = model.eta.clone()
    eta[14] = -10.0
    model = model.replace(eta=eta)
    batch = load_batch(["LiH", "SiH4"], DD_DOUBLE, [0.0, 0.0])
    total_charge = torch.zeros((), **DD_DOUBLE)

    def observable(charge: Tensor, structure: Structure) -> Tensor:
        current = structure.replace(charge=charge)
        charges, energies = model(
            current, solve_mode="matrix-free", return_energy=True
        )
        return charges if quantity == "charges" else energies.sum()

    derivative_function = DERIVATIVES[derivative](observable)
    evaluate_derivatives = vmap(derivative_function, in_dims=(None, 0))

    with no_vmap_fallback():
        result = evaluate_derivatives(total_charge, batch)

    torch.testing.assert_close(
        result[0], torch.zeros_like(result[0]), atol=1e-12, rtol=0
    )
    assert torch.isnan(result[1]).all()


@pytest.mark.cuda
@pytest.mark.parametrize("solve_mode", ["schur", "cg", "matrix-free"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_cuda_native_backward_values(
    solve_mode: str,
    dtype: torch.dtype,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Default fused CUDA products preserve values and native backward gradients."""
    from tad_mctc.storch import pairwise

    dd: DD = {"device": torch.device("cuda"), "dtype": dtype}
    original = load_structure("SiH4", dd, 1.0)
    names = ("chi", "kcn", "eta", "rad")
    fused_calls = []
    fused_product = pairwise._fused_product

    def record_fused(*args: Any, **kwargs: Any) -> Tensor:
        fused_calls.append(True)
        return fused_product(*args, **kwargs)

    monkeypatch.setattr(pairwise, "_fused_product", record_fused)

    def evaluate(mode: str) -> tuple[Tensor, Tensor, tuple[Tensor, ...]]:
        model = eeq.EEQModel.param2019(**dd)
        parameters = tuple(
            getattr(model, name).clone().requires_grad_(True) for name in names
        )
        model = model.replace(**dict(zip(names, parameters)))
        positions = original.positions.clone().requires_grad_(True)
        structure = original.replace(positions=positions)

        charges, energies = model(
            structure, solve_mode=mode, return_energy=True
        )
        objective = energies.sum() + 0.3 * charges.square().sum()
        objective.backward()
        gradients = tuple(tensor.grad for tensor in (positions, *parameters))
        return charges.detach(), energies.detach(), gradients

    actual_charges, actual_energies, actual_gradients = evaluate(solve_mode)
    reference_charges, reference_energies, reference_gradients = evaluate(
        "schur"
    )

    tolerance = 2e-4 if dtype == torch.float32 else 1e-8
    torch.testing.assert_close(
        actual_charges, reference_charges, atol=tolerance, rtol=tolerance
    )
    torch.testing.assert_close(
        actual_energies, reference_energies, atol=tolerance, rtol=tolerance
    )
    for actual, expected in zip(actual_gradients, reference_gradients):
        assert actual is not None and torch.isfinite(actual).all()
        torch.testing.assert_close(
            actual, expected, atol=tolerance, rtol=tolerance
        )

    if solve_mode == "matrix-free":
        assert fused_calls


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("transform", ["jacrev", "jacfwd", "vmap"])
def test_cuda_fused_option_under_transforms(
    dtype: torch.dtype,
    transform: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """fused=True remains correct when torch.func deliberately uses eager tiles."""
    from tad_mctc.storch import pairwise

    dd: DD = {"device": torch.device("cuda"), "dtype": dtype}
    structure = load_structure("SiH4", dd, 1.0)
    model = eeq.EEQModel.param2019(**dd)
    requested_fusion = []
    fused_calls = []
    eager_tile_calls = []
    pair_product = mf.pair_product
    fused_product = pairwise._fused_product
    tile_product = pairwise._tile_product

    def record_request(*args: Any, **kwargs: Any) -> Tensor:
        requested_fusion.append(kwargs["fused"])
        return pair_product(*args, **kwargs)

    def record_fused(*args: Any, **kwargs: Any) -> Tensor:
        fused_calls.append(True)
        return fused_product(*args, **kwargs)

    def record_eager_tile(*args: Any, **kwargs: Any) -> Tensor:
        eager_tile_calls.append(True)
        return tile_product(*args, **kwargs)

    monkeypatch.setattr(mf, "pair_product", record_request)
    monkeypatch.setattr(pairwise, "_fused_product", record_fused)
    monkeypatch.setattr(pairwise, "_tile_product", record_eager_tile)

    def matrix_free_observable(positions: Tensor) -> Tensor:
        current = structure.replace(positions=positions)
        charges, energies = model(
            current, solve_mode="matrix-free", return_energy=True
        )
        return torch.stack((charges, energies))

    def dense_observable(positions: Tensor) -> Tensor:
        current = structure.replace(positions=positions)
        charges, energies = model(
            current, solve_mode="schur", return_energy=True
        )
        return torch.stack((charges, energies))

    with no_vmap_fallback():
        if transform == "vmap":
            positions = torch.stack(
                [structure.positions + shift for shift in (0.0, 0.01, -0.01)]
            )
            evaluate = vmap(matrix_free_observable)
            actual = evaluate(positions)
            expected = torch.stack(
                [dense_observable(position) for position in positions]
            )
            assert actual.shape == (3, 2, 5)
        else:
            differentiate = jacrev if transform == "jacrev" else jacfwd
            evaluate = differentiate(matrix_free_observable)
            evaluate_reference = jacrev(dense_observable)
            actual = evaluate(structure.positions)
            expected = evaluate_reference(structure.positions)
            assert actual.shape == (2, 5, 5, 3)

    tolerance = 2e-5 if dtype == torch.float32 else 1e-9
    torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)
    assert requested_fusion and all(requested_fusion)

    # A custom Function may unwrap mapped inputs and fuse its primal.
    # Derivative products must also exercise the eager tile implementation.
    if transform in ("jacrev", "jacfwd"):
        assert eager_tile_calls
    else:
        assert fused_calls or eager_tile_calls
