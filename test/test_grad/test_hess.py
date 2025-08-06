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
"""
from __future__ import annotations

import pytest
import torch
from tad_mctc._version import __tversion__
from tad_mctc.autograd import hess_fn_rev
from tad_mctc.batch import pack
from tad_mctc.typing import DD, Tensor

from tad_multicharge.model import eeq

from ..conftest import DEVICE
from .samples_dedr import samples

sample_list = ["LiH", "SiH4", "AmF3", "Ag2Cl22-", "ZnOOH-"]
sample_list_large = ["PbH4-BiH3", "MB16_43_01"]

tol = 1e-5


def single(dtype: torch.dtype, name: str) -> None:
    dd: DD = {"device": DEVICE, "dtype": dtype}

    sample = samples[name]
    numbers = sample["numbers"].to(DEVICE)
    positions = sample["positions"].to(**dd)
    charge = sample["charge"].to(**dd)

    def _energy(num: Tensor, pos: Tensor, chrg: Tensor) -> Tensor:
        """
        Returns energy as scalar, which is required for Hessian computation
        to obtain the correct shape of ``(..., nat, 3, nat, 3)``.
        """
        return eeq.get_energy(num, pos, chrg).sum()

    # variable to be differentiated
    pos = positions.clone().requires_grad_(True)

    hess = hess_fn_rev(_energy, argnums=1)(numbers, pos, charge)
    hess = hess.detach().cpu()
    assert isinstance(hess, Tensor)
    assert hess.shape == (numbers.shape[-1], 3, numbers.shape[-1], 3)

    # Numerical Hessian for comparison
    num_hess = calc_num_hessian(numbers, positions, charge)
    assert num_hess.shape == hess.shape
    assert pytest.approx(num_hess.cpu(), rel=tol, abs=tol * 0.1) == hess

    pos.detach_()


@pytest.mark.skipif(__tversion__ < (2, 0, 0), reason="Requires PyTorch>=2.0.0")
@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name", sample_list)
def test_single(dtype: torch.dtype, name: str) -> None:
    single(dtype, name)


@pytest.mark.skipif(__tversion__ < (2, 0, 0), reason="Requires PyTorch>=2.0.0")
@pytest.mark.large
@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name", sample_list_large)
def test_single_large(dtype: torch.dtype, name: str) -> None:
    single(dtype, name)


@pytest.mark.skipif(__tversion__ < (2, 0, 0), reason="Requires PyTorch>=2.0.0")
@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name1", ["LiH"])
@pytest.mark.parametrize("name2", sample_list)
def test_batch(dtype: torch.dtype, name1: str, name2: str) -> None:
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
    charge = pack(
        [
            sample1["charge"].to(**dd),
            sample2["charge"].to(**dd),
        ]
    )

    def _energy(num: Tensor, pos: Tensor, chrg: Tensor) -> Tensor:
        """
        Returns energy as scalar, which is required for Hessian computation
        to obtain the correct shape of ``(..., nat, 3, nat, 3)``.
        """
        return eeq.get_energy(num, pos, chrg).sum()

    # variable to be differentiated
    pos = positions.clone().requires_grad_(True)

    hess_fn = hess_fn_rev(_energy, argnums=1)
    hess_fn_batch = torch.func.vmap(hess_fn, in_dims=(0, 0, 0))

    hess = hess_fn_batch(numbers, pos, charge)
    hess = hess.detach().cpu()
    assert isinstance(hess, Tensor)
    assert hess.shape == (2, numbers.shape[-1], 3, numbers.shape[-1], 3)

    # Numerical Hessian for comparison
    num_hess = calc_num_hessian_batch(numbers, positions, charge)
    assert num_hess.shape == hess.shape

    assert pytest.approx(num_hess.cpu(), rel=tol, abs=tol * 0.1) == hess

    pos.detach_()


def calc_num_hessian(
    numbers: Tensor,
    positions: Tensor,
    charge: Tensor,
    step: float = 5.0e-4,  # sensitive!
) -> Tensor:
    """
    Numerically approximate the full Hessian of the energy with respect to
    atomic positions via 4-point central-difference.

    Parameters
    ----------
    numbers : Tensor
        Atomic numbers for all atoms in the system of shape ``(nat, )``.
    positions : Tensor
        Cartesian coordinates of all atoms (shape: ``(nat, 3)``).
    charge : Tensor
        Total charge of the system (shape: ``(, )``).
    step : float, optional
        Step size for numerical differentiation, by default 5.0e-4.

    Returns
    -------
    Tensor
        Tensor of second derivatives (shape: ``(nat, 3, nat, 3)``)
    """
    # assume single-sample (no leading batch dims) for simplicity
    nat = positions.shape[-2]

    H_pqrs = []
    for p in range(nat):
        H_qrs = []
        for q in range(3):
            H_rs = []
            # prebuild the (p,q) kick
            e_pq = torch.zeros_like(positions)
            e_pq[p, q] = step

            for r in range(nat):
                row = []
                for s in range(3):
                    e_rs = torch.zeros_like(positions)
                    e_rs[r, s] = step

                    E_pp = eeq.get_energy(
                        numbers, positions + e_pq + e_rs, charge
                    ).sum()
                    E_pm = eeq.get_energy(
                        numbers, positions + e_pq - e_rs, charge
                    ).sum()
                    E_mp = eeq.get_energy(
                        numbers, positions - e_pq + e_rs, charge
                    ).sum()
                    E_mm = eeq.get_energy(
                        numbers, positions - e_pq - e_rs, charge
                    ).sum()

                    val = (E_pp - E_pm - E_mp + E_mm) / (4 * step * step)
                    row.append(val)
                # now one row of shape (3,)
                H_rs.append(torch.stack(row))
            # stack over r -> shape (nat, 3)
            H_qrs.append(torch.stack(H_rs, dim=0))
        # stack over q -> shape (3, nat, 3)
        H_pqrs.append(torch.stack(H_qrs, dim=0))
    # stack over p -> shape (nat, 3, nat, 3)
    return torch.stack(H_pqrs, dim=0)


def calc_num_hessian_batch(
    numbers: Tensor,
    positions: Tensor,
    charge: Tensor,
    step: float = 5.0e-4,  # sensitive!
) -> Tensor:
    """
    Compute a batch of Hessians.

    Parameters
    ----------
    numbers : Tensor
        Atomic numbers for all atoms in the system of shape ``(..., nat)``.
    positions : Tensor
        Cartesian coordinates of all atoms (shape: ``(..., nat, 3)``).
    charge : Tensor
        Total charge of the system (shape: ``(..., )``).
    step : float, optional
        Step size for numerical differentiation, by default 5.0e-4.

    Returns
    -------
    Tensor
        Numerical Hessian for each system in the batch of shape
        ``(..., nat, 3, nat, 3)``.
    """

    def _calc_num_hessian(nums: Tensor, pos: Tensor, ch: Tensor) -> Tensor:
        """Calculate the numerical Hessian for a single system."""
        return calc_num_hessian(nums, pos, ch, step)

    # vmap over axis=0 of each input
    hess_fn = torch.func.vmap(_calc_num_hessian, in_dims=(0, 0, 0), out_dims=0)
    return hess_fn(numbers, positions, charge)
