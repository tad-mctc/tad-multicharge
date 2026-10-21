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
Derivatives of the energy of 3D periodic structures: the nuclear gradient
against multicharge, the virial against finite differences of a strain
(as multicharge's ``test_numsigma``), and `gradcheck` with respect to the
positions and the lattice.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest
import torch
from tad_mctc.autograd import (
    dgradcheck,
    dgradgradcheck,
    jacrev_matches_finite_diff,
)
from tad_mctc.io.structure import Structure
from tad_mctc.typing import DD, Tensor

from tad_multicharge.model import eeq

from ..conftest import DEVICE, FAST_MODE
from ..utils import CHECK_IDS
from .samples import load, samples

dd: DD = {"device": DEVICE, "dtype": torch.double}


def energy(structure: Structure) -> Tensor:
    model = eeq.EEQModel.param2019(**dd)
    return model(structure, return_energy=True)[1].sum()


@pytest.mark.parametrize("name", list(samples))
def test_gradient(name: str) -> None:
    structure = load(name, dd)
    positions = structure.positions.clone().requires_grad_(True)

    ref = samples[name]["grad"]
    if ref is None:
        # no reference, see `samples.py`
        def f(x: Tensor) -> Tensor:
            return energy(structure.replace(positions=x))

        assert jacrev_matches_finite_diff(f, structure.positions, atol=1e-8)
        return

    e = energy(structure.replace(positions=positions))
    (grad,) = torch.autograd.grad(e, positions)
    assert pytest.approx(ref.cpu(), abs=1e-12) == grad.cpu()


def _virial(structure: Structure) -> Tensor:
    """Derivative of the energy with respect to a strain of the cell, from
    the derivatives with respect to the positions and the lattice."""
    assert structure.lattice is not None
    positions = structure.positions.clone().requires_grad_(True)
    lattice = structure.lattice.clone().requires_grad_(True)

    e = energy(structure.replace(positions=positions, lattice=lattice))
    dedr, dedl = torch.autograd.grad(e, (positions, lattice))
    return positions.mT.detach() @ dedr + lattice.mT.detach() @ dedl


@pytest.mark.parametrize("name", ["CO2+", "ice-vi"])
def test_virial(name: str) -> None:
    structure = load(name, dd)
    assert structure.lattice is not None
    sigma = _virial(structure)

    # Smaller steps lose more digits to the cancellation of the energies
    # than they gain in the truncation of the central difference.
    step = 1.0e-4
    numerical = torch.zeros(3, 3, **dd)
    for a in range(3):
        for b in range(3):
            strained = []
            for sign in (1.0, -1.0):
                eps = torch.eye(3, **dd)
                eps[a, b] += sign * step
                strained.append(
                    energy(
                        structure.replace(
                            positions=structure.positions @ eps,
                            lattice=structure.lattice @ eps,
                        )
                    )
                )
            numerical[a, b] = (strained[0] - strained[1]) / (2 * step)

    assert pytest.approx(numerical.cpu(), rel=1e-7, abs=1e-9) == sigma.cpu()


def _gradchecker(
    wrt: str,
) -> tuple[Callable[[Tensor], Tensor], Tensor]:
    structure = load("CO2+", dd)
    model = eeq.EEQModel.param2019(**dd)

    def func(x: Tensor) -> Tensor:
        return model(structure.replace(**{wrt: x}), return_energy=True)[1]

    value = getattr(structure, wrt)
    return func, value.clone().requires_grad_(True)


@pytest.mark.grad
@pytest.mark.parametrize("check", [dgradcheck, dgradgradcheck], ids=CHECK_IDS)
@pytest.mark.parametrize("wrt", ["positions", "lattice"])
def test_gradcheck(check: Callable[..., bool], wrt: str) -> None:
    """
    Check the analytical first (`gradcheck`) and second (`gradgradcheck`)
    derivatives w.r.t. the positions and the lattice against numerical
    ones.
    """
    func, diffvars = _gradchecker(wrt)
    assert check(func, diffvars, atol=1e-7, fast_mode=FAST_MODE)
