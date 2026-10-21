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
`vmap`, `jacrev`, `jacfwd` and `torch.compile` of the EEQ model for 3D
periodic structures.

The coordination number needs periodic shifts built ahead of time under
these transforms (see :meth:`tad_mctc.ncoord.common.CNModel.__call__`),
which are passed through the model call. The Ewald sum itself has fixed
shapes.
"""

from __future__ import annotations

import pytest
import torch
from tad_mctc.autograd import (
    jacfwd_matches_jacrev,
    jacrev_matches_finite_diff,
    no_vmap_fallback,
    vmap_matches_loop,
)
from tad_mctc.io.structure import Structure, pack_structures
from tad_mctc.ncoord import cn_eeq
from tad_mctc.neighbor import PeriodicShifts, build_periodic_shifts
from tad_mctc.tools.compile import compile_fullgraph
from tad_mctc.tools.testing import requires_compile
from tad_mctc.typing import DD, Tensor
from torch.func import jacrev, vmap

from tad_multicharge.model import SolveMode, eeq

from ..conftest import DEVICE
from .samples import SOLVE_MODES, load

DD_DOUBLE: DD = {"device": DEVICE, "dtype": torch.double}


def _single() -> Structure:
    return load("CO2+", DD_DOUBLE)


def _batch() -> Structure:
    return pack_structures([load("CO2+", DD_DOUBLE), load("urea", DD_DOUBLE)])


STRUCTURES = pytest.mark.parametrize(
    "load_structure", [_single, _batch], ids=["single", "batch"]
)


def _model() -> eeq.EEQModel:
    return eeq.EEQModel.param2019(**DD_DOUBLE)


def _shifts(structure: Structure, scale: float = 1.0) -> PeriodicShifts:
    """Shifts for the coordination number, also for a cell compressed
    down to ``1 / scale``."""
    assert structure.lattice is not None and structure.periodic is not None
    return build_periodic_shifts(
        structure.lattice / scale, structure.periodic, cn_eeq.cutoff
    )


########################################################################
# vmap


@STRUCTURES
@pytest.mark.parametrize("solve_mode", SOLVE_MODES)
def test_vmap_over_positions(load_structure, solve_mode: SolveMode) -> None:  # type: ignore[no-untyped-def]
    structure, model = load_structure(), _model()
    shifts = _shifts(structure)

    def f(positions: Tensor) -> Tensor:
        s = structure.replace(positions=positions)
        return model(s, shifts, solve_mode=solve_mode)

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
def test_vmap_over_lattice(load_structure) -> None:  # type: ignore[no-untyped-def]
    """Strained cells, each with its own Ewald parameter."""
    structure, model = load_structure(), _model()
    shifts = _shifts(structure, scale=1.02)

    def f(lattice: Tensor) -> Tensor:
        return model(structure.replace(lattice=lattice), shifts)

    assert structure.lattice is not None
    batch = torch.stack([structure.lattice * s for s in (1.0, 1.02, 0.98)])
    with no_vmap_fallback():
        assert vmap_matches_loop(f, batch, atol=1e-10)


@pytest.mark.parametrize("solve_mode", SOLVE_MODES)
def test_vmap_over_structures(solve_mode: SolveMode) -> None:
    """`vmap` over a packed batch of cells (a `Node`) matches the eager
    batched call."""
    structure, model = _batch(), _model()
    shifts = _shifts(structure)

    with no_vmap_fallback():
        q, e = vmap(
            lambda s: model(
                s, shifts, return_energy=True, solve_mode=solve_mode
            )
        )(structure)

    qref, eref = model(structure, return_energy=True, solve_mode=solve_mode)
    assert torch.allclose(q, qref, atol=1e-10)
    assert torch.allclose(e, eref, atol=1e-10)


########################################################################
# jacrev / jacfwd


@STRUCTURES
@pytest.mark.parametrize("wrt", ["positions", "lattice"])
def test_jacrev(load_structure, wrt: str) -> None:  # type: ignore[no-untyped-def]
    structure, model = load_structure(), _model()
    shifts = _shifts(structure, scale=1.02)

    def f(x: Tensor) -> Tensor:
        s = structure.replace(**{wrt: x})
        return model(s, shifts, return_energy=True)[1]

    assert jacrev_matches_finite_diff(f, getattr(structure, wrt), atol=1e-7)


@pytest.mark.parametrize("wrt", ["positions", "lattice"])
def test_jacfwd(wrt: str) -> None:
    structure, model = _single(), _model()
    shifts = _shifts(structure, scale=1.02)

    def f(x: Tensor) -> Tensor:
        s = structure.replace(**{wrt: x})
        return model(s, shifts, return_energy=True)[1].sum()

    assert jacfwd_matches_jacrev(f, getattr(structure, wrt), atol=1e-10)


def test_jacrev_wrt_charge() -> None:
    """The neutralising background of a charged cell."""
    structure, model = _single(), _model()
    assert structure.charge is not None

    def f(charge: Tensor) -> Tensor:
        return model(structure.replace(charge=charge), _shifts(structure))

    assert jacrev_matches_finite_diff(f, structure.charge, atol=1e-7)


########################################################################
# torch.compile


@requires_compile
@pytest.mark.usefixtures("reset_dynamo")
def test_compile() -> None:
    """
    Compiling the periodic model takes about 20 s, mostly for the unrolled
    bisection of the Ewald parameter, so only the single cell is compiled.
    """
    structure, model = _single(), _model()
    shifts = _shifts(structure)

    def f(positions: Tensor, lattice: Tensor) -> tuple[Tensor, Tensor]:
        s = structure.replace(positions=positions, lattice=lattice)
        return model(s, shifts, return_energy=True)

    assert structure.lattice is not None
    args = (structure.positions, structure.lattice)
    q, e = compile_fullgraph(f)(*args)
    qref, eref = f(*args)

    assert torch.allclose(q, qref, atol=1e-10, rtol=0)
    assert torch.allclose(e, eref, atol=1e-10, rtol=0)


@requires_compile
@pytest.mark.usefixtures("reset_dynamo")
def test_compile_jacrev() -> None:
    """``torch.compile(jacrev(f))``: compiled forces and lattice
    derivatives equal eager ones."""
    structure, model = _single(), _model()
    shifts = _shifts(structure)

    def energy(positions: Tensor, lattice: Tensor) -> Tensor:
        s = structure.replace(positions=positions, lattice=lattice)
        return model(s, shifts, return_energy=True)[1].sum()

    assert structure.lattice is not None
    args = (structure.positions, structure.lattice)
    grads = compile_fullgraph(jacrev(energy, argnums=(0, 1)))(*args)
    refs = jacrev(energy, argnums=(0, 1))(*args)

    for grad, ref in zip(grads, refs):
        assert torch.allclose(grad, ref, atol=1e-10, rtol=0)
