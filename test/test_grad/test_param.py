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
Testing the gradient with respect to the model parameters (autodiff).
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

sample_list = ["LiH", "AmF3", "SiH4", "MB16_43_01"]

tol = 1e-8


def gradchecker(
    dtype: torch.dtype, name: str
) -> tuple[Callable[[Tensor], Tensor], Tensor]:
    dd: DD = {"device": DEVICE, "dtype": dtype}

    structure = load_structure(name, dd, 0.0)

    cn = cn_eeq(structure)
    model = eeq.EEQModel.param2019(**dd)

    # variable to be differentiated
    chi = model.chi.clone().requires_grad_(True)

    def func(_chi: Tensor) -> Tensor:
        return model.replace(chi=_chi).solve(structure, cn, return_energy=True)[
            1
        ]

    return func, chi


@pytest.mark.grad
@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name", sample_list)
def test_gradcheck(dtype: torch.dtype, name: str) -> None:
    """
    Check a single analytical gradient of parameters against numerical
    gradient from `torch.autograd.gradcheck`.
    """
    func, diffvars = gradchecker(dtype, name)
    assert dgradcheck(func, diffvars, atol=tol, fast_mode=FAST_MODE)


@pytest.mark.grad
@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name", sample_list)
def test_gradgradcheck(dtype: torch.dtype, name: str) -> None:
    """
    Check a single analytical gradient of parameters against numerical
    gradient from `torch.autograd.gradgradcheck`.
    """
    func, diffvars = gradchecker(dtype, name)
    assert dgradgradcheck(func, diffvars, atol=tol, fast_mode=FAST_MODE)


def gradchecker_batch(
    dtype: torch.dtype, name1: str, name2: str
) -> tuple[Callable[[Tensor], Tensor], Tensor]:
    dd: DD = {"device": DEVICE, "dtype": dtype}

    structure = load_batch([name1, name2], dd)

    cn = cn_eeq(structure)
    model = eeq.EEQModel.param2019(**dd)

    # variable to be differentiated
    chi = model.chi.clone().requires_grad_(True)

    def func(_chi: Tensor) -> Tensor:
        return model.replace(chi=_chi).solve(structure, cn, return_energy=True)[
            1
        ]

    return func, chi


@pytest.mark.grad
@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name1", ["LiH"])
@pytest.mark.parametrize("name2", sample_list)
def test_gradcheck_batch(dtype: torch.dtype, name1: str, name2: str) -> None:
    """
    Check a single analytical gradient of parameters against numerical
    gradient from `torch.autograd.gradcheck`.
    """
    func, diffvars = gradchecker_batch(dtype, name1, name2)
    assert dgradcheck(func, diffvars, atol=tol, fast_mode=FAST_MODE)


@pytest.mark.grad
@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name1", ["LiH"])
@pytest.mark.parametrize("name2", sample_list)
def test_gradgradcheck_batch(
    dtype: torch.dtype, name1: str, name2: str
) -> None:
    """
    Check a single analytical gradient of parameters against numerical
    gradient from `torch.autograd.gradgradcheck`.
    """
    func, diffvars = gradchecker_batch(dtype, name1, name2)
    assert dgradgradcheck(func, diffvars, atol=tol, fast_mode=FAST_MODE)
