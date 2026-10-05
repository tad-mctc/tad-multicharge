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
General tests of the charge model as a frozen `Node`: construction checks,
conversion, immutability and the checks of `solve`.
"""

from __future__ import annotations

import dataclasses

import pytest
import torch
from tad_mctc.convert import str_to_device
from tad_mctc.io.structure import Structure
from tad_mctc.ncoord import cn_eeq
from tad_mctc.typing import DD

from tad_multicharge.model import ChargeModel, eeq

from ..conftest import DEVICE
from ..utils import load_structure


def _structure(dtype: torch.dtype = torch.double, **kwargs) -> Structure:  # type: ignore[no-untyped-def]
    structure = load_structure("NH3", {"device": DEVICE, "dtype": dtype})
    return structure.replace(**kwargs)


def test_abstract() -> None:
    t = torch.rand(5)
    with pytest.raises(TypeError):
        ChargeModel(chi=t, kcn=t, eta=t, rad=t, cn=cn_eeq)  # type: ignore[abstract]


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.float64])
def test_change_type(dtype: torch.dtype) -> None:
    model = eeq.EEQModel.param2019()
    converted = model.type(dtype)
    assert converted.dtype == dtype
    assert converted.chi.dtype == converted.rad.dtype == dtype

    # conversion returns a new model; the CN model is carried over
    assert model.dtype == torch.get_default_dtype()
    assert converted.cn is model.cn


@pytest.mark.cuda
@pytest.mark.parametrize("device_str", ["cpu", "cuda"])
def test_change_device(device_str: str) -> None:
    device = str_to_device(device_str)
    model = eeq.EEQModel.param2019().to(device)
    assert model.device == device


def test_frozen() -> None:
    model = eeq.EEQModel.param2019()

    with pytest.raises(AttributeError):
        model.dtype = torch.float64  # type: ignore[misc]

    with pytest.raises(AttributeError):
        model.device = torch.device("cpu")  # type: ignore[misc]

    with pytest.raises(dataclasses.FrozenInstanceError):
        model.chi = torch.zeros(5)  # type: ignore[misc]


def test_replace() -> None:
    model = eeq.EEQModel.param2019(dtype=torch.double)
    chi = model.chi * 2
    cn = cn_eeq.replace(cutoff=10.0)

    new = model.replace(chi=chi, cn=cn)
    assert new.chi is chi and new.cn is cn
    assert new.eta is model.eta
    assert model.cn is cn_eeq


def test_init_dtype_fail() -> None:
    t = torch.rand(5)

    # all floating-point tensors must have the same dtype
    with pytest.raises(TypeError):
        eeq.EEQModel(chi=t.double(), kcn=t, eta=t, rad=t)


def test_init_not_floating_fail() -> None:
    t = torch.rand(5)
    with pytest.raises(TypeError, match="floating-point"):
        eeq.EEQModel(chi=torch.ones(5, dtype=torch.long), kcn=t, eta=t, rad=t)


def test_init_shape_fail() -> None:
    t = torch.rand(5)
    with pytest.raises(ValueError, match="one-dimensional"):
        eeq.EEQModel(chi=torch.rand(5, 1), kcn=t, eta=t, rad=t)


@pytest.mark.cuda
def test_init_device_fail() -> None:
    cpu_tensor = torch.rand(5, device=torch.device("cpu"))
    cuda_tensor = cpu_tensor.to("cuda")

    # tensors on different devices must fail
    with pytest.raises(RuntimeError):
        eeq.EEQModel(
            chi=cpu_tensor,
            kcn=cuda_tensor,
            eta=cuda_tensor,
            rad=cuda_tensor,
        )


def test_solve_dtype_fail() -> None:
    model = eeq.EEQModel.param2019(device=DEVICE, dtype=torch.float32)
    structure = _structure(torch.double)

    with pytest.raises(RuntimeError, match="same dtype"):
        model.solve(structure, torch.ones(4, dtype=torch.double))


@pytest.mark.cuda
def test_solve_device_fail() -> None:
    model = eeq.EEQModel.param2019(device=torch.device("cpu"))
    structure = _structure(torch.get_default_dtype()).to(device="cuda")

    with pytest.raises(RuntimeError, match="same device"):
        model.solve(structure, torch.ones(4, device="cuda"))


def test_solve_periodic_fail() -> None:
    model = eeq.EEQModel.param2019(device=DEVICE, dtype=torch.double)
    structure = _structure(
        lattice=torch.eye(3, dtype=torch.double, device=DEVICE) * 10
    )

    with pytest.raises(NotImplementedError, match="periodic"):
        model(structure)


def test_solve_unknown_mode_fail() -> None:
    model = eeq.EEQModel.param2019(device=DEVICE, dtype=torch.double)
    structure = _structure()
    cn = torch.tensor([3.0, 1.0, 1.0, 1.0], dtype=torch.double, device=DEVICE)

    with pytest.raises(ValueError, match="Unknown EEQ solve mode"):
        model.solve(structure, cn, solve_mode="invalid")  # type: ignore[call-overload]


@pytest.mark.parametrize("charge", [0.0, [0.0], 1.0, [1.0]])
def test_total_charge_shapes(charge: float | list[float]) -> None:
    """A 0-d and a ``(1,)`` charge of an unbatched structure both work."""
    dd: DD = {"device": DEVICE, "dtype": torch.double}
    total = torch.tensor(charge, **dd)

    q = eeq.EEQModel.param2019(**dd)(_structure(charge=total))
    assert q.shape == (4,)
    assert torch.allclose(q.sum(), total.sum(), atol=1e-12)


def test_absent_charge_is_neutral() -> None:
    dd: DD = {"device": DEVICE, "dtype": torch.double}
    model = eeq.EEQModel.param2019(**dd)
    structure = _structure()

    q = model(structure)
    q0 = model(structure.replace(charge=torch.tensor(0.0, **dd)))
    assert torch.allclose(q, q0, atol=1e-12)
