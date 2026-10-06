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
Testing full hessian (functorch, vmap).

The analytic Hessian (``jacrev`` over ``jacrev``) is checked against central
finite differences of the analytic gradient, which `test_dedr.py` checks
against finite differences of the energy.
"""

from __future__ import annotations

import pytest
import torch
from tad_mctc.autograd import jacrev_matches_finite_diff, no_vmap_fallback
from tad_mctc.io.structure import Structure
from tad_mctc.typing import DD, Tensor

from tad_multicharge.model import eeq

from ..conftest import DEVICE
from ..utils import load_batch, load_structure
from .samples_dedr import samples

SAMPLE_LIST = ["LiH", "SiH4", "AmF3", "Ag2Cl22-", "ZnOOH-"]
SAMPLE_LIST_LARGE = ["PbH4-BiH3", "MB16_43_01"]


def _energy_fn(structure: Structure):  # type: ignore[no-untyped-def]
    model = eeq.EEQModel.param2019(
        device=structure.positions.device, dtype=structure.positions.dtype
    )

    def energy(pos: Tensor) -> Tensor:
        s = structure.replace(positions=pos)
        return model(s, return_energy=True)[1].sum()

    return energy


def hessian(structure: Structure) -> Tensor:
    """
    Hessian of the total energy with respect to the positions,
    reverse-over-reverse, shape ``(nat, 3, nat, 3)``.
    """
    energy = _energy_fn(structure)
    return torch.func.jacrev(torch.func.jacrev(energy))(structure.positions)


def single(dtype: torch.dtype, name: str) -> None:
    dd: DD = {"device": DEVICE, "dtype": dtype}
    structure = load_structure(name, dd, samples[name]["charge"])
    nat = structure.numbers.shape[-1]

    assert hessian(structure).shape == (nat, 3, nat, 3)

    gradient = torch.func.grad(_energy_fn(structure))
    assert jacrev_matches_finite_diff(gradient, structure.positions, atol=1e-7)


@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name", SAMPLE_LIST)
def test_single(dtype: torch.dtype, name: str) -> None:
    single(dtype, name)


@pytest.mark.large
@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name", SAMPLE_LIST_LARGE)
def test_single_large(dtype: torch.dtype, name: str) -> None:
    single(dtype, name)


@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name1", ["LiH"])
@pytest.mark.parametrize("name2", SAMPLE_LIST)
def test_batch(dtype: torch.dtype, name1: str, name2: str) -> None:
    """`vmap` of the Hessian over a stacked (padded) batch of structures
    equals the Hessian of each padded system on its own."""
    dd: DD = {"device": DEVICE, "dtype": dtype}
    charges = [samples[name1]["charge"], samples[name2]["charge"]]
    batch = load_batch([name1, name2], dd, charges)
    nat = batch.numbers.shape[-1]

    with no_vmap_fallback():
        hess = torch.func.vmap(hessian)(batch)
    assert hess.shape == (2, nat, 3, nat, 3)

    ref = torch.stack(
        [
            hessian(
                Structure(
                    numbers=batch.numbers[i],
                    positions=batch.positions[i],
                    charge=charges[i].to(**dd),
                )
            )
            for i in range(2)
        ]
    )
    assert torch.allclose(hess, ref, atol=1e-10)
