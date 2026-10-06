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

from collections.abc import Callable, Sequence

import pytest
import torch
from tad_mctc.autograd import dgradcheck, dgradgradcheck
from tad_mctc.ncoord import cn_eeq
from tad_mctc.typing import DD, Tensor

from tad_multicharge.model import eeq

from ..conftest import DEVICE, FAST_MODE
from ..utils import CHECK_IDS, load_samples, single_and_paired

SAMPLE_LIST = ["LiH", "AmF3", "SiH4", "MB16_43_01"]

TOL = 1e-8


def gradchecker(
    dtype: torch.dtype, names: Sequence[str]
) -> tuple[Callable[[Tensor], Tensor], Tensor]:
    dd: DD = {"device": DEVICE, "dtype": dtype}

    structure = load_samples(names, dd, 0.0)

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
    derivatives w.r.t. the electronegativities against numerical ones.
    """
    func, diffvars = gradchecker(dtype, names)
    assert check(func, diffvars, atol=TOL, fast_mode=FAST_MODE)
