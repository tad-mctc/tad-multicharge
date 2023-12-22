# This file is part of tad-multicharge.
#
# SPDX-Identifier: LGPL-3.0
# Copyright (C) 2023 Marvin Friede
#
# tad-multicharge is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# tad-multicharge is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with tad-multicharge. If not, see <https://www.gnu.org/licenses/>.
"""
Testing dispersion gradient (autodiff).
"""
from __future__ import annotations

import pytest
import torch
from tad_mctc.autograd import dgradcheck, dgradgradcheck, jac
from tad_mctc.batch import pack
from tad_mctc.convert import tensor_to_numpy

from tad_multicharge import eeq
from tad_multicharge.typing import DD, Callable, Tensor

from ..conftest import DEVICE, FAST_MODE
from .samples_dedr import samples

sample_list = [
    "LiH",
    "SiH4",
    "AmF3",
    "PbH4-BiH3",
    "MB16_43_01",
    "MB16_43_02",
    "Ag2Cl22-",
    "ZnOOH-",
]
sample_list_large = ["vancoh2"]

tol = 1e-8


def gradchecker(
    dtype: torch.dtype, name: str
) -> tuple[
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
        return eeq.get_energy(numbers, pos, charge)

    return func, positions


@pytest.mark.grad
@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name", sample_list + sample_list_large)
def test_gradcheck(dtype: torch.dtype, name: str) -> None:
    """
    Check a single analytical gradient of parameters against numerical
    gradient from `torch.autograd.gradcheck`.
    """
    func, diffvars = gradchecker(dtype, name)
    assert dgradcheck(func, diffvars, atol=tol, fast_mode=FAST_MODE)


@pytest.mark.grad
@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name", sample_list + sample_list_large)
def test_gradgradcheck(dtype: torch.dtype, name: str) -> None:
    """
    Check a single analytical gradient of parameters against numerical
    gradient from `torch.autograd.gradgradcheck`.
    """
    func, diffvars = gradchecker(dtype, name)
    assert dgradgradcheck(func, diffvars, atol=tol, fast_mode=FAST_MODE)


def gradchecker_batch(
    dtype: torch.dtype, name1: str, name2: str
) -> tuple[
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
        return eeq.get_energy(numbers, pos, charge)

    return func, positions


@pytest.mark.grad
@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name1", ["LiH"])
@pytest.mark.parametrize("name2", sample_list + sample_list_large)
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
@pytest.mark.parametrize("name2", sample_list + sample_list_large)
def test_gradgradcheck_batch(dtype: torch.dtype, name1: str, name2: str) -> None:
    """
    Check a single analytical gradient of parameters against numerical
    gradient from `torch.autograd.gradgradcheck`.
    """
    func, diffvars = gradchecker_batch(dtype, name1, name2)
    assert dgradgradcheck(func, diffvars, atol=tol, fast_mode=FAST_MODE)


def run_autograd(dtype: torch.dtype, name: str) -> None:
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

    numgrad = calc_numgrad_jacobian(numbers, positions, charge).sum(0)

    # variable to be differentiated
    positions.requires_grad_(True)

    # automatic gradient
    energy = eeq.get_energy(numbers, positions, charge)
    (grad,) = torch.autograd.grad(energy.sum(), positions)

    positions.detach_()
    grad.detach_()

    assert pytest.approx(numgrad.cpu(), abs=tol * 10) == grad.cpu()

    # ref = sample["grad"].to(**dd)
    # assert pytest.approx(ref.cpu(), abs=tol * 10) == grad.cpu()


@pytest.mark.grad
@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name", sample_list)
def test_autograd(dtype: torch.dtype, name: str) -> None:
    run_autograd(dtype, name)


@pytest.mark.grad
@pytest.mark.large
@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name", sample_list_large)
def test_autograd_large(dtype: torch.dtype, name: str) -> None:
    run_autograd(dtype, name)


def run_backward(dtype: torch.dtype, name: str) -> None:
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

    numgrad = calc_numgrad_jacobian(numbers, positions, charge).sum(0)

    # variable to be differentiated
    positions.requires_grad_(True)

    # automatic gradient
    energy = eeq.get_energy(numbers, positions, charge).sum()
    energy.backward()

    assert positions.grad is not None
    grad_backward = positions.grad.clone()

    # also zero out gradients when using `.backward()`
    positions.detach_()
    positions.grad.data.zero_()

    assert pytest.approx(numgrad.cpu(), abs=tol * 10) == grad_backward.cpu()

    # ref = sample["grad"].to(**dd)
    # assert pytest.approx(ref.cpu(), abs=tol * 10) == grad_backward.cpu()


@pytest.mark.grad
@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name", sample_list)
def test_backward(dtype: torch.dtype, name: str) -> None:
    run_backward(dtype, name)


@pytest.mark.grad
@pytest.mark.large
@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name", sample_list_large)
def test_backward_large(dtype: torch.dtype, name: str) -> None:
    run_backward(dtype, name)


def run_jacobian(dtype: torch.dtype, name: str, atol: float) -> None:
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

    numgrad = calc_numgrad_jacobian(numbers, positions, charge)

    # variable to be differentiated
    positions.requires_grad_(True)

    fjac = jac(eeq.get_energy, argnums=1)
    jacobian: Tensor = fjac(numbers, positions, charge)

    positions.detach_()
    jac_np = tensor_to_numpy(jacobian)

    assert pytest.approx(numgrad.cpu(), abs=atol) == jac_np


@pytest.mark.grad
@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name", sample_list)
def test_jacobian(dtype: torch.dtype, name: str) -> None:
    run_jacobian(dtype, name, tol)


@pytest.mark.grad
@pytest.mark.large
@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name", sample_list_large)
def test_jacobian_large(dtype: torch.dtype, name: str) -> None:
    run_jacobian(dtype, name, 1e-6)


def calc_numgrad_jacobian(numbers: Tensor, positions: Tensor, charge: Tensor) -> Tensor:
    gradient = torch.zeros(torch.Size((*2 * (numbers.shape[-1],), 3)))
    step = 1.0e-6

    for i in range(numbers.shape[-1]):
        for j in range(3):
            positions[i, j] += step
            er = eeq.get_energy(numbers, positions, charge)

            positions[i, j] -= 2 * step
            el = eeq.get_energy(numbers, positions, charge)

            positions[i, j] += step
            gradient[:, i, j] = 0.5 * (er - el) / step

    return gradient
