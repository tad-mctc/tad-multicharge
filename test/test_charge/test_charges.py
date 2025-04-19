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

The tests surrounding the EEQ charge model include:
 - single molecule
 - batched
 - ghost atoms
 - autograd via `gradcheck`

Note that `torch.linalg.solve` gives slightly different results (around 1e-5
to 1e-6) across different PyTorch versions (1.11.0 vs 1.13.0) for single
precision. For double precision, however the results are identical.

In PR #22, problems with double precision also appeared and tests failed for
a few matrix elements on Linux and Windows (notably not on macOS). Locally,
all tests were passing.

Due to the above inconsistencies, loose tolerances are adapted in the EEQ tests.
"""
from __future__ import annotations

import pytest
import torch
from tad_mctc.batch import pack
from tad_mctc.ncoord import cn_eeq

from tad_multicharge.model import eeq
from tad_multicharge.typing import DD

from ..conftest import DEVICE
from .samples import samples


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
def test_single(dtype: torch.dtype) -> None:
    dd: DD = {"device": DEVICE, "dtype": dtype}
    tol = 1e-4 if dtype == torch.float32 else 1e-6

    sample = samples["NH3-dimer"]
    numbers = sample["numbers"].to(DEVICE)
    positions = sample["positions"].to(**dd)
    total_charge = sample["total_charge"].to(**dd)

    qref = sample["q"].to(**dd)
    eref = sample["energy"].to(**dd)

    cn = torch.tensor([3.0, 3.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], **dd)
    eeq_model = eeq.EEQModel.param2019(**dd)
    qat, energy = eeq_model.solve(
        numbers, positions, total_charge, cn, return_energy=True
    )
    tot = torch.sum(qat, -1)

    assert qat.dtype == energy.dtype == dtype
    assert pytest.approx(total_charge.cpu(), abs=1e-6) == tot.cpu()
    assert pytest.approx(qref.cpu(), abs=tol) == qat.cpu()
    assert pytest.approx(eref.cpu(), abs=tol) == energy.cpu()


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name", ["AmF3"])
def test_single_with_cn(dtype: torch.dtype, name: str) -> None:
    dd: DD = {"device": DEVICE, "dtype": dtype}
    tol = 1e-4 if dtype == torch.float32 else 1e-6

    sample = samples[name]
    numbers = sample["numbers"].to(DEVICE)
    positions = sample["positions"].to(**dd)
    total_charge = sample["total_charge"].to(**dd)

    qref = sample["q"].to(**dd)
    eref = sample["energy"].to(**dd)

    cn = cn_eeq(numbers, positions)
    eeq_model = eeq.EEQModel.param2019(**dd)
    qat, energy = eeq_model.solve(
        numbers, positions, total_charge, cn, return_energy=True
    )
    tot = torch.sum(qat, -1)

    assert qat.dtype == energy.dtype == dtype
    assert pytest.approx(total_charge.cpu(), abs=1e-6) == tot.cpu()
    assert pytest.approx(qref.cpu(), abs=tol) == qat.cpu()
    assert pytest.approx(eref.cpu(), abs=tol) == energy.cpu()


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
def test_ghost(dtype: torch.dtype) -> None:
    dd: DD = {"device": DEVICE, "dtype": dtype}
    tol = 1e-4 if dtype == torch.float32 else 1e-6

    sample = samples["NH3-dimer"]
    numbers = sample["numbers"].clone().to(DEVICE)
    numbers[[1, 5, 6, 7]] = 0
    positions = sample["positions"].to(**dd)
    total_charge = sample["total_charge"].to(**dd)
    cn = torch.tensor([3.0, 3.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], **dd)

    qref = torch.tensor(
        [
            -0.8189238943,
            +0.0000000000,
            +0.2730378155,
            +0.2728482633,
            +0.2730378155,
            +0.0000000000,
            +0.0000000000,
            +0.0000000000,
        ],
        **dd,
    )
    eref = torch.tensor(
        [
            -1.0891341309,
            +0.0000000000,
            +0.3345037494,
            +0.3342715255,
            +0.3345037494,
            +0.0000000000,
            +0.0000000000,
            +0.0000000000,
        ],
        **dd,
    )

    eeq_model = eeq.EEQModel.param2019(**dd)
    qat, energy = eeq_model.solve(
        numbers, positions, total_charge, cn, return_energy=True
    )
    tot = torch.sum(qat, -1)

    assert qat.dtype == energy.dtype == dtype
    assert pytest.approx(total_charge.cpu(), abs=1e-6) == tot.cpu()
    assert pytest.approx(qref.cpu(), abs=tol) == qat.cpu()
    assert pytest.approx(eref.cpu(), abs=tol) == energy.cpu()


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
def test_batch(dtype: torch.dtype) -> None:
    dd: DD = {"device": DEVICE, "dtype": dtype}
    tol = 1e-4 if dtype == torch.float32 else 1e-6

    sample1, sample2 = (
        samples["PbH4-BiH3"],
        samples["C6H5I-CH3SH"],
    )
    numbers = pack(
        (
            sample1["numbers"].to(DEVICE),
            sample2["numbers"].to(DEVICE),
        )
    )
    positions = pack(
        (
            sample1["positions"].to(**dd),
            sample2["positions"].to(**dd),
        )
    )
    total_charge = torch.tensor([0.0, 0.0], **dd).view(-1, 1)
    eref = pack(
        (
            sample1["energy"].to(**dd),
            sample2["energy"].to(**dd),
        )
    )
    qref = pack(
        (
            sample1["q"].to(**dd),
            sample2["q"].to(**dd),
        )
    )

    cn = torch.tensor(
        [
            [
                3.9195758978,
                0.9835975866,
                0.9835977083,
                0.9835977083,
                0.9832391350,
                2.9579090955,
                0.9874520816,
                0.9874522118,
                0.9874520816,
                0.0000000000,
                0.0000000000,
                0.0000000000,
                0.0000000000,
                0.0000000000,
                0.0000000000,
                0.0000000000,
                0.0000000000,
                0.0000000000,
            ],
            [
                3.0173754479,
                3.0134898523,
                3.0173773978,
                3.1580192128,
                3.0178688039,
                3.1573804880,
                1.3525004230,
                0.9943449208,
                0.9943846525,
                0.9942776053,
                0.9943862103,
                0.9942779112,
                2.0535643452,
                0.9956985559,
                3.9585744304,
                0.9940553724,
                0.9939077317,
                0.9939362885,
            ],
        ],
        **dd,
    )
    eeq_model = eeq.EEQModel.param2019(**dd)
    qat, energy = eeq_model.solve(
        numbers, positions, total_charge, cn, return_energy=True
    )
    tot = torch.sum(qat, -1).view(-1, 1)

    assert qat.dtype == energy.dtype == dtype
    assert pytest.approx(total_charge.cpu(), abs=1e-6) == tot.cpu()
    assert pytest.approx(qref.cpu(), abs=tol) == qat.cpu()
    assert pytest.approx(eref.cpu(), abs=tol) == energy.cpu()
