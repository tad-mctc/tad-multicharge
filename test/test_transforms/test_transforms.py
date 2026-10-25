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
`vmap`, `jacrev` and `jacfwd` of the EEQ model. `torch.compile` is in
`test_compile.py`.

Each check compares against something that needs no reference: a plain
Python loop or the eager batched call for `vmap`, finite differences for
`jacrev`, and `jacrev` for `jacfwd`. The `vmap` checks run with the vmap
fallback disabled, so an operation without a batching rule fails instead
of silently looping.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

import pytest
import torch
from tad_mctc.autograd import (
    jacfwd_matches_jacrev,
    jacrev_matches_finite_diff,
    no_vmap_fallback,
    vmap_matches_loop,
)
from tad_mctc.io.structure import Structure
from tad_mctc.tree import stack
from tad_mctc.typing import DD, Tensor
from torch.func import jacfwd, jacrev, vmap

from tad_multicharge.model import SolveMode, eeq

from ..conftest import DEVICE
from ..utils import load_batch, load_structure

DD_DOUBLE: DD = {"device": DEVICE, "dtype": torch.double}
SOLVE_MODES = ["schur", "linear", "cg", "matrix-free"]


def _single() -> Structure:
    return load_structure("SiH4", DD_DOUBLE, 0.0)


def _batch() -> Structure:
    return load_batch(["LiH", "ZnOOH-"], DD_DOUBLE, [0.0, -1.0])


STRUCTURES = pytest.mark.parametrize(
    "load", [_single, _batch], ids=["single", "batch"]
)
OUTPUTS = pytest.mark.parametrize("return_energy", [False, True])


def _model() -> eeq.EEQModel:
    return eeq.EEQModel.param2019(**DD_DOUBLE)


def _observable(
    model: eeq.EEQModel,
    structure: Structure,
    solve_mode: SolveMode,
    return_energy: bool,
) -> Tensor:
    """Represent both public return forms as tensors for the shared checks."""
    result = model(
        structure, solve_mode=solve_mode, return_energy=return_energy
    )
    if return_energy:
        return torch.stack(result)
    return result


@STRUCTURES
@OUTPUTS
@pytest.mark.parametrize("solve_mode", SOLVE_MODES)
def test_vmap_over_positions(
    load: Callable[[], Structure], solve_mode: SolveMode, return_energy: bool
) -> None:
    """Position batching matches independent calls for both return forms."""
    structure, model = load(), _model()

    def evaluate(positions: Tensor) -> Tensor:
        displaced_structure = structure.replace(positions=positions)
        return _observable(
            model, displaced_structure, solve_mode, return_energy
        )

    positions = torch.stack(
        [
            structure.positions,
            structure.positions + 0.01,
            structure.positions - 0.01,
        ]
    )

    with no_vmap_fallback():
        assert vmap_matches_loop(evaluate, positions, atol=1e-10)


@STRUCTURES
@OUTPUTS
@pytest.mark.parametrize("solve_mode", SOLVE_MODES)
def test_vmap_over_charge(
    load: Callable[[], Structure], solve_mode: SolveMode, return_energy: bool
) -> None:
    """Total-charge batching matches independent calls for every backend."""
    structure, model = load(), _model()
    assert structure.charge is not None

    def evaluate(charge: Tensor) -> Tensor:
        charged_structure = structure.replace(charge=charge)
        return _observable(model, charged_structure, solve_mode, return_energy)

    charges = torch.stack(
        [structure.charge + delta for delta in (-1.0, 0.0, 1.0)]
    )

    with no_vmap_fallback():
        assert vmap_matches_loop(evaluate, charges, atol=1e-10)


@OUTPUTS
@pytest.mark.parametrize("solve_mode", SOLVE_MODES)
def test_vmap_over_structures(
    solve_mode: SolveMode, return_energy: bool
) -> None:
    """Packed-structure vmap agrees with batching and a dense reference."""
    structure, model = _batch(), _model()

    def evaluate(current_structure: Structure) -> Tensor:
        return _observable(model, current_structure, solve_mode, return_energy)

    with no_vmap_fallback():
        evaluate_structures = vmap(evaluate)
        actual = evaluate_structures(structure)

    batched = _observable(model, structure, solve_mode, return_energy)
    reference = _observable(model, structure, "schur", return_energy)

    # The return-energy observable adds its leading output dimension before
    # the ordinary batch; vmap puts its molecular dimension first instead.
    if return_energy:
        batched = batched.movedim(0, 1)
        reference = reference.movedim(0, 1)

    assert torch.allclose(actual, batched, atol=1e-10)
    assert torch.allclose(actual, reference, atol=1e-10, rtol=0)


@OUTPUTS
@pytest.mark.parametrize("solve_mode", SOLVE_MODES)
@pytest.mark.parametrize(
    "configured", [False, True], ids=["default", "options"]
)
def test_vmap_over_models(
    solve_mode: SolveMode, return_energy: bool, configured: bool
) -> None:
    """All four parameter families batch with static matrix-free options."""
    structure, model = _single(), _model()
    if configured:
        model = model.replace(
            matrix_free=eeq.MatrixFreeOptions(
                tile=2,
                fused=False,
                coarse_memory=0,
                mixed_precision=False,
            )
        )

    # chi/kcn change the RHS; eta/rad change the Coulomb operator.
    models = [
        model,
        model.replace(chi=model.chi * 1.1, kcn=model.kcn * 0.9),
        model.replace(eta=model.eta * 0.9, rad=model.rad * 1.05),
    ]
    batched_models = stack(models)

    def evaluate(current_model: eeq.EEQModel) -> Tensor:
        return _observable(current_model, structure, solve_mode, return_energy)

    with no_vmap_fallback():
        evaluate_models = vmap(evaluate)
        actual = evaluate_models(batched_models)

    looped = torch.stack([evaluate(current_model) for current_model in models])
    reference = torch.stack(
        [
            _observable(current_model, structure, "linear", return_energy)
            for current_model in models
        ]
    )

    assert torch.allclose(actual, looped, atol=1e-10)
    assert torch.allclose(actual, reference, atol=1e-10, rtol=0)


@STRUCTURES
@OUTPUTS
@pytest.mark.parametrize("solve_mode", SOLVE_MODES)
def test_jacrev_wrt_positions(
    load: Callable[[], Structure], solve_mode: SolveMode, return_energy: bool
) -> None:
    """Reverse position derivatives match finite differences and Schur."""
    structure, model = load(), _model()

    def evaluate(positions: Tensor) -> Tensor:
        displaced_structure = structure.replace(positions=positions)
        return _observable(
            model, displaced_structure, solve_mode, return_energy
        )

    def dense_reference(positions: Tensor) -> Tensor:
        displaced_structure = structure.replace(positions=positions)
        return _observable(model, displaced_structure, "schur", return_energy)

    with no_vmap_fallback():
        assert jacrev_matches_finite_diff(
            evaluate, structure.positions, atol=1e-7
        )
        actual = jacrev(evaluate)(structure.positions)
        reference = jacrev(dense_reference)(structure.positions)

    torch.testing.assert_close(actual, reference, atol=1e-10, rtol=1e-8)


@STRUCTURES
@OUTPUTS
@pytest.mark.parametrize("solve_mode", SOLVE_MODES)
@pytest.mark.parametrize("differentiate", [jacrev, jacfwd])
def test_derivatives_wrt_charge(
    load: Callable[[], Structure],
    solve_mode: SolveMode,
    return_energy: bool,
    differentiate: Callable,
) -> None:
    """Both charge derivative rules match finite differences and dense Schur."""
    structure, model = load(), _model()
    assert structure.charge is not None

    def evaluate(charge: Tensor) -> Tensor:
        charged_structure = structure.replace(charge=charge)
        return _observable(model, charged_structure, solve_mode, return_energy)

    def dense_reference(charge: Tensor) -> Tensor:
        charged_structure = structure.replace(charge=charge)
        return _observable(model, charged_structure, "schur", return_energy)

    with no_vmap_fallback():
        assert jacrev_matches_finite_diff(evaluate, structure.charge, atol=1e-7)
        actual = differentiate(evaluate)(structure.charge)
        reference = jacrev(dense_reference)(structure.charge)

    torch.testing.assert_close(actual, reference, atol=1e-10, rtol=1e-8)


@STRUCTURES
@OUTPUTS
@pytest.mark.parametrize("solve_mode", SOLVE_MODES)
def test_jacfwd_matches_jacrev(
    load: Callable[[], Structure], solve_mode: SolveMode, return_energy: bool
) -> None:
    """Forward position derivatives match reverse mode and dense Schur."""
    structure, model = load(), _model()

    def evaluate(positions: Tensor) -> Tensor:
        displaced_structure = structure.replace(positions=positions)
        return _observable(
            model, displaced_structure, solve_mode, return_energy
        )

    def dense_reference(positions: Tensor) -> Tensor:
        displaced_structure = structure.replace(positions=positions)
        return _observable(model, displaced_structure, "schur", return_energy)

    with no_vmap_fallback():
        assert jacfwd_matches_jacrev(evaluate, structure.positions)
        actual = jacfwd(evaluate)(structure.positions)
        reference = jacrev(dense_reference)(structure.positions)

    torch.testing.assert_close(actual, reference, atol=1e-10, rtol=1e-8)


@pytest.mark.parametrize("solve_mode", SOLVE_MODES)
def test_jacrev_wrt_model(solve_mode: SolveMode) -> None:
    """A model-valued gradient agrees with independent parameter derivatives."""
    structure, model = _single(), _model()

    def energy(current_model: eeq.EEQModel) -> Tensor:
        _, atomic_energies = current_model(
            structure,
            return_energy=True,
            solve_mode=solve_mode,
        )
        return atomic_energies.sum()

    gradient = cast(eeq.EEQModel, jacrev(energy)(model))
    assert isinstance(gradient, eeq.EEQModel)

    for name in ("chi", "kcn", "eta", "rad"):

        def parameter_energy(value: Tensor) -> Tensor:
            return energy(model.replace(**{name: value}))

        reference = jacrev(parameter_energy)(getattr(model, name))
        assert torch.allclose(getattr(gradient, name), reference, atol=1e-12)

    def chi_energy(chi: Tensor) -> Tensor:
        return energy(model.replace(chi=chi))

    assert jacrev_matches_finite_diff(chi_energy, model.chi, atol=1e-7)


@OUTPUTS
@pytest.mark.parametrize("solve_mode", ["schur", "cg", "matrix-free"])
def test_nested_vmap(solve_mode: SolveMode, return_energy: bool) -> None:
    """Two mapped axes preserve padding, output order and molecular charge."""
    names = ["LiH", "SiH4", "ZnOOH-"]
    charges = [0.0, 1.0, -1.0]
    batch = load_batch(names, DD_DOUBLE, charges)
    shifted_batch = batch.replace(positions=batch.positions + 0.02)
    nested_structures = stack([batch, shifted_batch])
    model = _model()

    def evaluate(current_structure: Structure) -> Tensor:
        return _observable(model, current_structure, solve_mode, return_energy)

    with no_vmap_fallback():
        evaluate_inner_batch = vmap(evaluate)
        evaluate_outer_batch = vmap(evaluate_inner_batch)
        actual = evaluate_outer_batch(nested_structures)

    # Evaluate original, unpadded molecules independently, including the
    # outer position shift; no vectorized reference can hide an axis error.
    references = []
    looped = []
    for shift in (0.0, 0.02):
        reference_row = []
        looped_row = []
        for name, total_charge in zip(names, charges):
            single = load_structure(name, DD_DOUBLE, total_charge)
            single = single.replace(positions=single.positions + shift)
            padding = batch.numbers.shape[-1] - single.numbers.shape[-1]
            reference = _observable(model, single, "schur", return_energy)
            independent = evaluate(single)
            reference_row.append(
                torch.nn.functional.pad(reference, (0, padding))
            )
            looped_row.append(
                torch.nn.functional.pad(independent, (0, padding))
            )
        references.append(torch.stack(reference_row))
        looped.append(torch.stack(looped_row))

    expected = torch.stack(references)
    torch.testing.assert_close(actual, expected, atol=1e-10, rtol=0)
    torch.testing.assert_close(actual, torch.stack(looped), atol=1e-10, rtol=0)
    assert actual.shape == ((2, 3, 2, 5) if return_energy else (2, 3, 5))
    atomic_charges = actual[:, :, 0] if return_energy else actual
    requested = torch.tensor(charges, **DD_DOUBLE).expand(2, 3)
    torch.testing.assert_close(
        atomic_charges.sum(-1), requested, atol=1e-12, rtol=0
    )
    padding_mask = (batch.numbers == 0).expand(2, 3, 5)
    if return_energy:
        padding_mask = padding_mask.unsqueeze(2).expand_as(actual)
    assert torch.equal(
        actual.masked_select(padding_mask),
        torch.zeros_like(actual.masked_select(padding_mask)),
    )
