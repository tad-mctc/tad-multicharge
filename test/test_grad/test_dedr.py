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
Testing energy gradient (autodiff).
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
from tad_mctc.io.structure import Structure
from tad_mctc.typing import DD, Tensor

from tad_multicharge.model import eeq

from ..conftest import DEVICE, FAST_MODE
from ..utils import CHECK_IDS, load_samples, load_structure, single_and_paired
from .samples_dedr import samples

SAMPLE_LIST = [
    "LiH",
    "SiH4",
    "AmF3",
    "PbH4-BiH3",
    "MB16_43_01",
    "MB16_43_02",
    "Ag2Cl22-",
    "ZnOOH-",
]
SAMPLE_LIST_LARGE = ["vancoh2"]

TOL = 1e-8


def gradchecker(
    dtype: torch.dtype, names: Sequence[str]
) -> tuple[Callable[[Tensor], Tensor], Tensor]:
    """Prepare a gradient check of `eeq.get_energy` w.r.t. positions."""
    dd: DD = {"device": DEVICE, "dtype": dtype}
    return positions_gradchecker(eeq.get_energy, load_samples(names, dd, 0.0))


@pytest.mark.grad
@pytest.mark.parametrize("check", [dgradcheck, dgradgradcheck], ids=CHECK_IDS)
@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize(
    "names",
    single_and_paired(SAMPLE_LIST + SAMPLE_LIST_LARGE, "LiH"),
    ids="+".join,
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


def _structure(name: str, dd: DD) -> Structure:
    return load_structure(name, dd, samples[name]["charge"])


def run_jacobian(dtype: torch.dtype, name: str, atol: float) -> None:
    dd: DD = {"device": DEVICE, "dtype": dtype}
    structure = _structure(name, dd)

    num = numgrad(eeq.get_energy, structure)

    def f(pos: Tensor) -> Tensor:
        return eeq.get_energy(structure.replace(positions=pos))

    jacobian = torch.func.jacrev(f)(structure.positions)

    assert pytest.approx(num.cpu(), abs=atol) == jacobian.cpu()


@pytest.mark.grad
@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name", SAMPLE_LIST)
def test_jacobian(dtype: torch.dtype, name: str) -> None:
    run_jacobian(dtype, name, 1e-7)


@pytest.mark.grad
@pytest.mark.large
@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name", SAMPLE_LIST_LARGE)
def test_jacobian_large(dtype: torch.dtype, name: str) -> None:
    run_jacobian(dtype, name, 1e-6)
