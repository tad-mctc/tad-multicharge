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
from torch.func import jacrev, vmap

from tad_multicharge.model import SolveMode, eeq

from ..conftest import DEVICE
from ..utils import load_batch, load_structure

DD_DOUBLE: DD = {"device": DEVICE, "dtype": torch.double}

SOLVE_MODES = ["schur", "linear"]


def _single() -> Structure:
    return load_structure("SiH4", DD_DOUBLE, 0.0)


def _batch() -> Structure:
    return load_batch(["LiH", "ZnOOH-"], DD_DOUBLE, [0.0, -1.0])


STRUCTURES = pytest.mark.parametrize(
    "load", [_single, _batch], ids=["single", "batch"]
)


def _model() -> eeq.EEQModel:
    return eeq.EEQModel.param2019(**DD_DOUBLE)


########################################################################
# vmap


@STRUCTURES
@pytest.mark.parametrize("solve_mode", SOLVE_MODES)
def test_vmap_over_positions(load, solve_mode: SolveMode) -> None:  # type: ignore[no-untyped-def]
    structure, model = load(), _model()

    def f(positions: Tensor) -> Tensor:
        s = structure.replace(positions=positions)
        return model(s, solve_mode=solve_mode)

    batch = torch.stack(
        [
            structure.positions,
            structure.positions + 0.01,
            structure.positions - 0.01,
        ]
    )
    with no_vmap_fallback():
        assert vmap_matches_loop(f, batch, atol=1e-10)


@STRUCTURES
def test_vmap_over_charge(load) -> None:  # type: ignore[no-untyped-def]
    structure, model = load(), _model()
    assert structure.charge is not None

    def f(charge: Tensor) -> Tensor:
        return model(structure.replace(charge=charge))

    batch = torch.stack([structure.charge + d for d in (-1.0, 0.0, 1.0)])
    with no_vmap_fallback():
        assert vmap_matches_loop(f, batch, atol=1e-10)


@pytest.mark.parametrize("solve_mode", SOLVE_MODES)
def test_vmap_over_structures(solve_mode: SolveMode) -> None:
    """`vmap` over a packed batch of structures (a `Node`) matches the
    eager batched call."""
    structure, model = _batch(), _model()

    with no_vmap_fallback():
        q, e = vmap(
            lambda s: model(s, return_energy=True, solve_mode=solve_mode)
        )(structure)

    qref, eref = model(structure, return_energy=True, solve_mode=solve_mode)
    assert torch.allclose(q, qref, atol=1e-10)
    assert torch.allclose(e, eref, atol=1e-10)


def test_vmap_over_models() -> None:
    """`vmap` over a stack of models (a `Node`) matches a loop."""
    structure, model = _single(), _model()
    models = [
        model,
        model.replace(chi=model.chi * 1.1),
        model.replace(eta=model.eta * 0.9, rad=model.rad * 1.05),
    ]

    with no_vmap_fallback():
        batched = vmap(lambda m: m(structure))(stack(models))

    looped = torch.stack([m(structure) for m in models])
    assert torch.allclose(batched, looped, atol=1e-10)


########################################################################
# jacrev / jacfwd


@STRUCTURES
@pytest.mark.parametrize("solve_mode", SOLVE_MODES)
def test_jacrev_wrt_positions(load, solve_mode: SolveMode) -> None:  # type: ignore[no-untyped-def]
    structure, model = load(), _model()

    def f(positions: Tensor) -> Tensor:
        s = structure.replace(positions=positions)
        return model(s, return_energy=True, solve_mode=solve_mode)[1]

    assert jacrev_matches_finite_diff(f, structure.positions, atol=1e-7)


@STRUCTURES
def test_jacrev_wrt_charge(load) -> None:  # type: ignore[no-untyped-def]
    structure, model = load(), _model()
    assert structure.charge is not None

    def f(charge: Tensor) -> Tensor:
        return model(structure.replace(charge=charge))

    assert jacrev_matches_finite_diff(f, structure.charge, atol=1e-7)


@STRUCTURES
@pytest.mark.parametrize("solve_mode", SOLVE_MODES)
def test_jacfwd_matches_jacrev(load, solve_mode: SolveMode) -> None:  # type: ignore[no-untyped-def]
    structure, model = load(), _model()

    def f(positions: Tensor) -> Tensor:
        s = structure.replace(positions=positions)
        return model(s, solve_mode=solve_mode)

    assert jacfwd_matches_jacrev(f, structure.positions)


def test_jacrev_wrt_model() -> None:
    """`jacrev` over the model itself (a `Node`) gives the derivative with
    respect to every parameter at once, as a model of Jacobians."""
    structure, model = _single(), _model()

    def energy(m: eeq.EEQModel) -> Tensor:
        return m(structure, return_energy=True)[1].sum()

    grad = cast(eeq.EEQModel, jacrev(energy)(model))
    assert isinstance(grad, eeq.EEQModel)

    for name in ("chi", "kcn", "eta", "rad"):
        ref = jacrev(lambda x: energy(model.replace(**{name: x})))(
            getattr(model, name)
        )
        assert torch.allclose(getattr(grad, name), ref, atol=1e-12)

    def f(chi: Tensor) -> Tensor:
        return energy(model.replace(chi=chi))

    assert jacrev_matches_finite_diff(f, model.chi, atol=1e-7)
