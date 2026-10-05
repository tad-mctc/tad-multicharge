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
Testing the charges module
==========================

This module tests the EEQ charge model including:
 - single molecule
 - batched
 - ghost atoms
 - autograd via `gradcheck`

The coordination number is computed once and held fixed, so only the
gradient through the linear solve is checked here.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest
import torch
from tad_mctc.autograd import dgradcheck, dgradgradcheck
from tad_mctc.ncoord import cn_eeq
from tad_mctc.typing import DD, Tensor

from tad_multicharge.model import eeq

from ..conftest import DEVICE, FAST_MODE
from ..utils import load_batch, load_structure

sample_list = ["NH3", "NH3-dimer", "PbH4-BiH3", "C6H5I-CH3SH"]

tol = 1e-7


def gradchecker(
    dtype: torch.dtype, name: str
) -> tuple[Callable[[Tensor, Tensor], Tensor], tuple[Tensor, Tensor]]:
    """Prepare gradient check from `torch.autograd`."""
    dd: DD = {"device": DEVICE, "dtype": dtype}

    structure = load_structure(name, dd)
    total_charge = torch.tensor(0.0, **dd)

    eeq_model = eeq.EEQModel.param2019(**dd)
    cn = cn_eeq(structure)

    # variables to be differentiated
    positions = structure.positions.clone().requires_grad_(True)
    total_charge.requires_grad_(True)

    def func(pos: Tensor, tchrg: Tensor) -> Tensor:
        s = structure.replace(positions=pos, charge=tchrg)
        return eeq_model.solve(s, cn)

    return func, (positions, total_charge)


@pytest.mark.grad
@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name", sample_list)
def test_grad(dtype: torch.dtype, name: str) -> None:
    """
    Check a single analytical gradient of positions against numerical
    gradient from `torch.autograd.gradcheck`.
    """
    func, diffvars = gradchecker(dtype, name)
    assert dgradcheck(func, diffvars, atol=tol, fast_mode=FAST_MODE)


@pytest.mark.grad
@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name", sample_list)
def test_gradgrad(dtype: torch.dtype, name: str) -> None:
    """
    Check a single analytical gradient of positions against numerical
    gradient from `torch.autograd.gradgradcheck`.
    """
    func, diffvars = gradchecker(dtype, name)
    assert dgradgradcheck(func, diffvars, atol=tol, fast_mode=FAST_MODE)


def gradchecker_batch(
    dtype: torch.dtype, name1: str, name2: str
) -> tuple[Callable[[Tensor, Tensor], Tensor], tuple[Tensor, Tensor]]:
    """Prepare gradient check from `torch.autograd`."""
    dd: DD = {"device": DEVICE, "dtype": dtype}

    structure = load_batch([name1, name2], dd)
    total_charge = torch.tensor([0.0, 0.0], **dd)

    eeq_model = eeq.EEQModel.param2019(**dd)
    cn = cn_eeq(structure)

    # variables to be differentiated
    positions = structure.positions.clone().requires_grad_(True)
    total_charge.requires_grad_(True)

    def func(pos: Tensor, tchrg: Tensor) -> Tensor:
        s = structure.replace(positions=pos, charge=tchrg)
        return eeq_model.solve(s, cn)

    return func, (positions, total_charge)


@pytest.mark.grad
@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name1", ["NH3"])
@pytest.mark.parametrize("name2", sample_list)
def test_grad_batch(dtype: torch.dtype, name1: str, name2: str) -> None:
    """
    Check a single analytical gradient of positions against numerical
    gradient from `torch.autograd.gradcheck`.
    """
    func, diffvars = gradchecker_batch(dtype, name1, name2)
    assert dgradcheck(func, diffvars, atol=tol, fast_mode=FAST_MODE)


@pytest.mark.grad
@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name1", ["NH3"])
@pytest.mark.parametrize("name2", sample_list)
def test_gradgrad_batch(dtype: torch.dtype, name1: str, name2: str) -> None:
    """
    Check a single analytical gradient of positions against numerical
    gradient from `torch.autograd.gradgradcheck`.
    """
    func, diffvars = gradchecker_batch(dtype, name1, name2)
    assert dgradgradcheck(func, diffvars, atol=tol, fast_mode=FAST_MODE)
