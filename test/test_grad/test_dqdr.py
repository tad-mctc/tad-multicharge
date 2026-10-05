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
Testing charge gradient (autodiff).
"""

from __future__ import annotations

from collections.abc import Callable

import pytest
import torch
from tad_mctc.autograd import dgradcheck, dgradgradcheck, numgrad
from tad_mctc.convert import reshape_fortran
from tad_mctc.typing import DD, Tensor

from tad_multicharge.model import eeq

from ..conftest import DEVICE, FAST_MODE
from ..utils import load_batch, load_structure
from .samples_dqdr import samples

sample_list = [
    "LiH",
    "SiH4",
    "AmF3",
    "PbH4-BiH3",
    "MB16_43_01",
    "MB16_43_02",
    "Ag2Cl22-",
    "ZnOOH-",
    "vancoh2",
]

tol = 1e-8


def gradchecker(dtype: torch.dtype, name: str) -> tuple[
    Callable[[Tensor], Tensor],  # autograd function
    Tensor,  # differentiable variables
]:
    dd: DD = {"device": DEVICE, "dtype": dtype}

    structure = load_structure(name, dd, 0.0)
    positions = structure.positions.clone().requires_grad_(True)

    def func(pos: Tensor) -> Tensor:
        return eeq.get_charges(structure.replace(positions=pos))

    return func, positions


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


def gradchecker_batch(dtype: torch.dtype, name1: str, name2: str) -> tuple[
    Callable[[Tensor], Tensor],  # autograd function
    Tensor,  # differentiable variables
]:
    dd: DD = {"device": DEVICE, "dtype": dtype}

    structure = load_batch([name1, name2], dd, [0.0, 0.0])

    # variable to be differentiated
    positions = structure.positions.clone().requires_grad_(True)

    def func(pos: Tensor) -> Tensor:
        return eeq.get_charges(structure.replace(positions=pos))

    return func, positions


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


@pytest.mark.grad
@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name", sample_list[:-1])
def test_jacobian(dtype: torch.dtype, name: str) -> None:
    """Compare with reference values from tblite."""
    dd: DD = {"device": DEVICE, "dtype": dtype}
    charge = {"ZnOOH-": -1.0, "Ag2Cl22-": -2.0}.get(name, 0.0)
    structure = load_structure(name, dd, charge)
    nat = structure.numbers.shape[-1]

    # (3*nat*nat) -> (3, nat, nat) -> (nat, nat, 3)
    ref = samples[name]["grad"].to(**dd)
    ref = reshape_fortran(ref, torch.Size((3, nat, nat)))
    ref = torch.einsum("xij->jix", ref)

    num = numgrad(eeq.get_charges, structure)

    def f(pos: Tensor) -> Tensor:
        return eeq.get_charges(structure.replace(positions=pos))

    jacobian = torch.func.jacrev(f)(structure.positions)

    # 1 / 768 element in MB16_43_01 is slightly off
    assert pytest.approx(ref.cpu(), abs=tol * 10.5) == jacobian.cpu()

    assert pytest.approx(ref.cpu(), abs=tol * 10) == num.cpu()
    assert pytest.approx(num.cpu(), abs=tol * 10) == jacobian.cpu()
