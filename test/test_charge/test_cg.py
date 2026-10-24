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
"""Acceptance of the reconstructed dense-CG EEQ equations."""

from __future__ import annotations

import math

import pytest
import torch
from tad_mctc import storch
from tad_mctc.autograd import no_vmap_fallback
from tad_mctc.exceptions import ConvergenceError
from tad_mctc.tools.testing import requires_compile
from tad_mctc.typing import DD, Tensor
from torch.func import vmap

from tad_multicharge.model import eeq

from ..conftest import DEVICE
from ..utils import load_structure

DD_DOUBLE: DD = {"device": DEVICE, "dtype": torch.double}


def test_amplified_multiplier(monkeypatch: pytest.MonkeyPatch) -> None:
    """Converged auxiliary solves can still violate KKT stationarity."""
    model = eeq.EEQModel.param2019(**DD_DOUBLE)
    a = 1e6 * torch.eye(4, **DD_DOUBLE)
    b = torch.tensor([1.0, -1.0, 0.0, 0.0], **DD_DOUBLE)
    c = torch.ones_like(b)
    total = b.new_ones(1)
    solve = storch.linsolve.implicit_solve

    def approximate(*args, **kwargs):  # type: ignore[no-untyped-def]
        result = solve(*args, **kwargs)
        x = result.solution.clone()
        x[..., 1, :] += 2e-19 * b
        residual = torch.linalg.vector_norm(
            eeq._coulomb_matvec(x, a) - torch.stack((b, c)), dim=-1
        )
        assert (residual <= result.target).all()
        return result._replace(solution=x, residual=residual)

    monkeypatch.setattr(storch.linsolve, "implicit_solve", approximate)

    # A direct solve of the full KKT system remains accurate.
    qref = model._solve_linear(b, c, a, total, False)
    mu = ((b - a @ qref) * c).sum() / c.square().sum()
    assert (a @ qref + c * mu - b).norm() < 1e-10
    assert torch.allclose(c @ qref, total.squeeze(), atol=1e-14, rtol=0)

    with pytest.raises(ConvergenceError, match="stationarity"):
        model._solve_cg(b, c, a, total, False)


@pytest.mark.parametrize(
    "execution",
    ["eager", "batch", "vmap", pytest.param("compile", marks=requires_compile)],
)
def test_charge_cancellation(execution: str) -> None:
    """Loss of total charge raises or masks only the failing systems."""
    model = eeq.EEQModel.param2019(**DD_DOUBLE)
    a = torch.eye(4, **DD_DOUBLE)
    b = torch.full((4,), 1e16, **DD_DOUBLE)
    c = torch.ones_like(b)

    def charges(total: Tensor) -> Tensor:
        return model._solve_cg(b, c, a, total, False)

    totals = b.new_tensor([[0.0], [1.0]])
    if execution in ("eager", "batch"):
        with pytest.raises(ConvergenceError, match="charge"):
            charges(totals[1] if execution == "eager" else totals)
        return

    if execution == "vmap":
        with no_vmap_fallback():
            q = vmap(charges)(totals)
    else:
        q = torch.compile(charges, backend="eager", fullgraph=True)(totals)

    assert torch.equal(q[0], torch.zeros_like(b))
    assert torch.isnan(q[1]).all()


@pytest.mark.parametrize("size", [4, 32])
@pytest.mark.parametrize("scale", [1e-4, 1e4])
@pytest.mark.parametrize("condition", [1.0, 1e2])
def test_scaled_spd(size: int, scale: float, condition: float) -> None:
    """Scaled SPD solves satisfy both physical equations and a KKT reference."""
    model = eeq.EEQModel.param2019(**DD_DOUBLE)
    # Four distinct eigenvalues, including two separated by `condition`.
    # A fixed orthogonal basis avoids random ill-conditioning of the basis.
    index = torch.arange(size, **DD_DOUBLE)
    basis = torch.cos(torch.pi * (index[:, None] + 0.5) * index / size)
    basis[:, 0] /= 2**0.5
    basis *= (2 / size) ** 0.5
    eigenvalues = torch.logspace(
        0, math.log10(condition), 4, **DD_DOUBLE
    ).repeat(size // 4)
    a = scale * (basis * eigenvalues) @ basis.mT
    b = torch.sin(index + 0.5)
    c = torch.ones_like(b)
    total = b.new_tensor([2.0])

    q = model._solve_cg(b, c, a, total, False)
    qref = model._solve_linear(b, c, a, total, False)
    mu = ((b - a @ q) * c).sum() / c.square().sum()
    residual = (a @ q + c * mu - b).norm()
    scale_kkt = torch.cat((b, total)).norm()
    roundoff = 10 * torch.finfo(b.dtype).eps
    rounding_scale = (a.abs() @ q.abs()).norm() + (c * mu).norm()
    assert (
        residual
        <= eeq.CG_ATOL + eeq.CG_RTOL * scale_kkt + roundoff * rounding_scale
    )
    assert (c @ q - total).abs().max() <= eeq.CG_ATOL + roundoff * (
        (c * q).abs().sum() + total.abs().sum()
    )
    assert torch.allclose(q, qref, atol=1e-7, rtol=1e-8)


@pytest.mark.parametrize("size", [4, 64])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_ill_conditioned_jacobi(size: int, dtype: torch.dtype) -> None:
    """Jacobi handles a diagonal SPD matrix with condition number 1e6."""
    dd: DD = {"device": DEVICE, "dtype": dtype}
    model = eeq.EEQModel.param2019(**dd)
    a = torch.diag(torch.logspace(0, 6, size, **dd))
    b = torch.linspace(-1, 1, size, **dd)
    c = torch.ones_like(b)
    total = b.new_tensor([3.0])
    q = model._solve_cg(b, c, a, total, False)
    # Evaluate the independent dense KKT reference in double precision.
    ref = model._solve_linear(
        b.double(), c.double(), a.double(), total.double(), False
    )
    tol = 1e-5 if dtype == torch.float32 else 1e-12
    assert torch.allclose(q.double(), ref, atol=tol, rtol=tol)
    assert torch.allclose(c @ q, total.squeeze(), atol=tol, rtol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize(
    "chi_scale,hardness_shift,total", [(0.2, 0.0, -5.0), (20.0, 2.0, 3.0)]
)
def test_parameter_ranges(
    dtype: torch.dtype, chi_scale: float, hardness_shift: float, total: float
) -> None:
    """Charged molecules with altered parameters agree with dense KKT solves."""
    dd: DD = {"device": DEVICE, "dtype": dtype}
    structure = load_structure("ZnOOH-", dd, total)
    model = eeq.EEQModel.param2019(**dd)
    model = model.replace(
        chi=model.chi * chi_scale,
        eta=model.eta + hardness_shift,
        rad=model.rad * 1.25,
    )
    q = model(structure, solve_mode="cg")
    qref = model(structure, solve_mode="linear")
    a = eeq._coulomb_matrix_0d(
        structure.numbers,
        structure.positions,
        model.rad[structure.numbers],
        model.eta[structure.numbers],
    )
    b = (
        -model.chi[structure.numbers]
        + storch.safe_sqrt(model.cn(structure)) * model.kcn[structure.numbers]
    )
    mu = (b - a @ q).mean()
    tol = 1e-4 if dtype == torch.float32 else 1e-9
    assert torch.allclose(q, qref, atol=tol, rtol=tol)
    assert torch.allclose(q.sum(), q.new_tensor(total), atol=tol, rtol=0)
    assert (a @ q + mu - b).norm() <= tol * (1 + b.norm())
