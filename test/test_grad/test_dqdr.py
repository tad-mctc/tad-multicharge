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

from collections.abc import Callable, Sequence

import pytest
import torch
from tad_mctc.autograd import (
    dgradcheck,
    dgradgradcheck,
    numgrad,
    positions_gradchecker,
)
from tad_mctc.convert import reshape_fortran
from tad_mctc.typing import DD, Tensor

from tad_multicharge.model import eeq

from ..conftest import DEVICE, FAST_MODE
from ..utils import CHECK_IDS, load_samples, load_structure, single_and_paired
from .samples_dqdr import samples

SAMPLE_LIST = [
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

TOL = 1e-8


def gradchecker(
    dtype: torch.dtype, names: Sequence[str]
) -> tuple[Callable[[Tensor], Tensor], Tensor]:
    """Prepare a gradient check of `eeq.get_charges` w.r.t. positions."""
    dd: DD = {"device": DEVICE, "dtype": dtype}
    return positions_gradchecker(eeq.get_charges, load_samples(names, dd, 0.0))


@pytest.mark.grad
@pytest.mark.parametrize("check", [dgradcheck, dgradgradcheck], ids=CHECK_IDS)
@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize(
    "names", single_and_paired(SAMPLE_LIST, "LiH"), ids="+".join
)
def test_gradcheck(
    check: Callable[..., bool], dtype: torch.dtype, names: list[str]
) -> None:
    """
    Check the analytical first (`gradcheck`) and second (`gradgradcheck`)
    derivatives w.r.t. positions against numerical ones.
    """
    func, diffvars = gradchecker(dtype, names)
    assert check(func, diffvars, atol=TOL, fast_mode=FAST_MODE)


@pytest.mark.grad
@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name", SAMPLE_LIST[:-1])
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
    assert pytest.approx(ref.cpu(), abs=TOL * 10.5) == jacobian.cpu()

    assert pytest.approx(ref.cpu(), abs=TOL * 10) == num.cpu()
    assert pytest.approx(num.cpu(), abs=TOL * 10) == jacobian.cpu()
