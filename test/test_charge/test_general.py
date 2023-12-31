# This file is part of tad_multicharge.
#
# SPDX-Identifier: LGPL-3.0
# Copyright (C) 2022 Marvin Friede
#
# tad_multicharge is free software: you can redistribute it and/or modify it
# under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# tad_multicharge is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with tad_multicharge. If not, see <https://www.gnu.org/licenses/>.
"""
Testing the charges module
==========================

This module tests the EEQ charge model including:
 - single molecule
 - batched
 - ghost atoms
 - autograd via `gradcheck`

Note that `torch.linalg.solve` gives slightly different results (around 1e-5
to 1e-6) across different PyTorch versions (1.11.0 vs 1.13.0) for single
precision. For double precision, however the results are identical.
"""
from __future__ import annotations

import pytest
import torch
from tad_mctc.convert import str_to_device

from tad_multicharge import eeq


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.float64])
def test_change_type(dtype: torch.dtype) -> None:
    model = eeq.EEQModel.param2019().type(dtype)
    assert model.dtype == dtype


def test_change_type_fail() -> None:
    model = eeq.EEQModel.param2019()

    # trying to use setter
    with pytest.raises(AttributeError):
        model.dtype = torch.float64

    # passing disallowed dtype
    with pytest.raises(ValueError):
        model.type(torch.bool)


@pytest.mark.cuda
@pytest.mark.parametrize("device_str", ["cpu", "cuda"])
def test_change_device(device_str: str) -> None:
    device = str_to_device(device_str)
    model = eeq.EEQModel.param2019().to(device)
    assert model.device == device


def test_change_device_fail() -> None:
    model = eeq.EEQModel.param2019()

    # trying to use setter
    with pytest.raises(AttributeError):
        model.device = torch.device("cpu")


def test_init_dtype_fail() -> None:
    t = torch.rand(5)

    # all tensor must have the same type
    with pytest.raises(RuntimeError):
        eeq.EEQModel(t.type(torch.double), t, t, t)


@pytest.mark.cuda
def test_init_device_fail() -> None:
    t = torch.rand(5)
    if "cuda" in str(t.device):
        t = t.cpu()
    elif "cpu" in str(t.device):
        t = t.cuda()

    # all tensor must be on the same device
    with pytest.raises(RuntimeError):
        eeq.EEQModel(t, t, t, t)


def test_solve_dtype_fail() -> None:
    t = torch.rand(5, dtype=torch.float64)
    model = eeq.EEQModel.param2019()

    # all tensor must have the same type
    with pytest.raises(RuntimeError):
        eeq.solve(t, t.type(torch.float16), t, model, t)


@pytest.mark.cuda
def test_solve_device_fail() -> None:
    t = torch.rand(5)
    t2 = t.clone()
    model = eeq.EEQModel.param2019()

    if "cuda" in str(t.device):
        t2 = t2.cpu()
    elif "cpu" in str(t.device):
        t2 = t2.cuda()

    # all tensor must be on the same device
    with pytest.raises(RuntimeError):
        eeq.solve(t, t2, t, model, t)
