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

Note that `torch.linalg.solve` gives slightly different results (around 1e-5
to 1e-6) across different PyTorch versions (1.11.0 vs 1.13.0) for single
precision. For double precision, however the results are identical.
"""
from __future__ import annotations

import pytest
import torch
from tad_mctc.autograd import dgradcheck, dgradgradcheck
from tad_mctc.batch import pack
from tad_mctc.data.molecules import mols as samples
from tad_mctc.ncoord import cn_eeq

from tad_multicharge.model import eeq
from tad_multicharge.typing import DD, Callable, Tensor

from ..conftest import DEVICE, FAST_MODE

sample_list = ["NH3", "NH3-dimer", "PbH4-BiH3", "C6H5I-CH3SH"]

device = None

tol = 1e-7


def gradchecker(
    dtype: torch.dtype, name: str
) -> tuple[Callable[[Tensor, Tensor], Tensor], tuple[Tensor, Tensor]]:
    """Prepare gradient check from `torch.autograd`."""
    dd: DD = {"device": DEVICE, "dtype": dtype}

    sample = samples[name]
    numbers = sample["numbers"].to(DEVICE)
    positions = sample["positions"].to(**dd)
    total_charge = torch.tensor(0.0, **dd)

    eeq_model = eeq.EEQModel.param2019(**dd)

    # variables to be differentiated
    positions.requires_grad_(True)
    total_charge.requires_grad_(True)

    def func(pos: Tensor, tchrg: Tensor) -> Tensor:
        cn = cn_eeq(numbers, positions)
        return eeq_model.solve(numbers, pos, tchrg, cn)[0]

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

    sample1, sample2 = samples[name1], samples[name2]
    numbers = pack(
        [
            sample1["numbers"].to(DEVICE),
            sample2["numbers"].to(DEVICE),
        ]
    )
    positions = pack(
        [
            sample1["positions"].to(**dd),
            sample2["positions"].to(**dd),
        ]
    )
    total_charge = torch.tensor([0.0, 0.0], **dd)

    eeq_model = eeq.EEQModel.param2019(**dd)

    # variables to be differentiated
    positions.requires_grad_(True)
    total_charge.requires_grad_(True)

    def func(pos: Tensor, tchrg: Tensor) -> Tensor:
        cn = cn_eeq(numbers, positions)
        return eeq_model.solve(numbers, pos, tchrg, cn)[0]

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
