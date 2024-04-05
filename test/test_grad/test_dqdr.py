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
Testing dispersion gradient (autodiff).
"""
from __future__ import annotations

import pytest
import torch
from tad_mctc.autograd import dgradcheck, dgradgradcheck, jacrev
from tad_mctc.batch import pack
from tad_mctc.convert import reshape_fortran, tensor_to_numpy

from tad_multicharge.model import eeq
from tad_multicharge.typing import DD, Callable, Tensor

from ..conftest import DEVICE, FAST_MODE
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

    sample = samples[name]
    numbers = sample["numbers"].to(DEVICE)
    positions = sample["positions"].to(**dd)
    charge = torch.tensor(0.0, **dd)

    positions.requires_grad_(True)

    def func(pos: Tensor) -> Tensor:
        return eeq.get_charges(numbers, pos, charge)

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
    charge = torch.tensor([0.0, 0.0], **dd)

    # variable to be differentiated
    positions.requires_grad_(True)

    def func(pos: Tensor) -> Tensor:
        return eeq.get_charges(numbers, pos, charge)

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
def test_gradgradcheck_batch(dtype: torch.dtype, name1: str, name2: str) -> None:
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

    sample = samples[name]
    numbers = sample["numbers"].to(DEVICE)
    positions = sample["positions"].to(**dd)

    if name == "ZnOOH-":
        charge = torch.tensor(-1.0, **dd)
    elif name == "Ag2Cl22-":
        charge = torch.tensor(-2.0, **dd)
    else:
        charge = torch.tensor(0.0, **dd)

    # (3*nat*nat) -> (3, nat, nat) -> (nat, nat, 3)
    ref = sample["grad"].to(**dd)
    ref = reshape_fortran(ref, torch.Size((3, *2 * (numbers.shape[-1],))))
    ref = torch.einsum("xij->jix", ref)

    numgrad = calc_numgrad(numbers, positions, charge)

    # variable to be differentiated
    positions.requires_grad_(True)

    fjac = jacrev(eeq.get_charges, argnums=1)
    jacobian: Tensor = fjac(numbers, positions, charge)  # type: ignore

    positions.detach_()
    jac_np = tensor_to_numpy(jacobian)

    # 1 / 768 element in MB16_43_01 is slightly off
    assert pytest.approx(ref.cpu(), abs=tol * 10.5) == jac_np

    assert pytest.approx(ref.cpu(), abs=tol * 10) == numgrad.cpu()
    assert pytest.approx(numgrad.cpu(), abs=tol * 10) == jac_np


def calc_numgrad(numbers: Tensor, positions: Tensor, charge: Tensor) -> Tensor:
    gradient = torch.zeros(torch.Size((*2 * (numbers.shape[-1],), 3)))
    step = 1.0e-6

    for i in range(numbers.shape[-1]):
        for j in range(3):
            positions[i, j] += step
            qr = eeq.get_charges(numbers, positions, charge)

            positions[i, j] -= 2 * step
            ql = eeq.get_charges(numbers, positions, charge)

            positions[i, j] += step
            gradient[:, i, j] = 0.5 * (qr - ql) / step

    return gradient
