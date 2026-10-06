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

from collections.abc import Callable, Sequence

import pytest
import torch
from tad_mctc.autograd import dgradcheck, dgradgradcheck
from tad_mctc.ncoord import cn_eeq
from tad_mctc.typing import DD, Tensor

from tad_multicharge.model import eeq

from ..conftest import DEVICE, FAST_MODE
from ..utils import CHECK_IDS, load_samples, single_and_paired

SAMPLE_LIST = ["NH3", "NH3-dimer", "PbH4-BiH3", "C6H5I-CH3SH"]

TOL = 1e-7


def gradchecker(
    dtype: torch.dtype, names: Sequence[str]
) -> tuple[Callable[[Tensor, Tensor], Tensor], tuple[Tensor, Tensor]]:
    """Prepare gradient check from `torch.autograd`."""
    dd: DD = {"device": DEVICE, "dtype": dtype}

    structure = load_samples(names, dd, 0.0)
    total_charge = torch.zeros(structure.positions.shape[:-2], **dd)

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
@pytest.mark.parametrize("check", [dgradcheck, dgradgradcheck], ids=CHECK_IDS)
@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize(
    "names", single_and_paired(SAMPLE_LIST, "NH3"), ids="+".join
)
def test_gradcheck(
    check: Callable[..., bool], dtype: torch.dtype, names: list[str]
) -> None:
    """
    Check the analytical first (`gradcheck`) and second (`gradgradcheck`)
    derivatives w.r.t. positions and total charge against numerical ones.
    """
    func, diffvars = gradchecker(dtype, names)
    assert check(func, diffvars, atol=TOL, fast_mode=FAST_MODE)
