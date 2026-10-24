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
from typing import Any

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
    # The large hardness gives mu = -2.5e5 for Q = 1. This amplifies a
    # tiny error in y = A^-1 c when reconstructing q = z - mu y.
    coulomb = 1e6 * torch.eye(4, **DD_DOUBLE)
    rhs = torch.tensor([1.0, -1.0, 0.0, 0.0], **DD_DOUBLE)
    constraint = torch.ones_like(rhs)
    total_charge = rhs.new_ones(1)
    auxiliary_rhs = torch.stack((rhs, constraint))
    implicit_solve = storch.linsolve.implicit_solve

    def solve_with_constraint_error(
        *args: Any, **kwargs: Any
    ) -> storch.linsolve.LinearSolveResult:
        """Perturb y within its auxiliary residual tolerance."""
        result = implicit_solve(*args, **kwargs)
        solutions = result.solution.clone()  # RHS order: [z, y]
        # A zero-sum perturbation leaves c^T y unchanged. The failure is
        # therefore stationarity, independently of charge conservation.
        solutions[..., 1, :] += 2e-19 * rhs
        residual = torch.linalg.vector_norm(
            eeq._coulomb_matvec(solutions, coulomb) - auxiliary_rhs, dim=-1
        )
        # Both auxiliary solves still satisfy their true-residual targets.
        assert (residual <= result.target).all()
        return result._replace(solution=solutions, residual=residual)

    monkeypatch.setattr(
        storch.linsolve, "implicit_solve", solve_with_constraint_error
    )

    # The full bordered solve avoids the approximate auxiliary solutions
    # and verifies that the physical problem itself is solvable accurately.
    reference_charges = model._solve_linear(
        rhs, constraint, coulomb, total_charge, False
    )
    reference_multiplier = (
        (rhs - coulomb @ reference_charges) * constraint
    ).sum() / constraint.square().sum()
    assert (
        coulomb @ reference_charges + constraint * reference_multiplier - rhs
    ).norm() < 1e-10
    assert torch.allclose(
        constraint @ reference_charges,
        total_charge.squeeze(),
        atol=1e-14,
        rtol=0,
    )

    with pytest.raises(ConvergenceError, match="stationarity"):
        model._solve_cg(rhs, constraint, coulomb, total_charge, False)


@pytest.mark.parametrize(
    "execution",
    ["eager", "batch", "vmap", pytest.param("compile", marks=requires_compile)],
)
def test_charge_cancellation(execution: str) -> None:
    """Loss of total charge raises or masks only the failing systems."""
    model = eeq.EEQModel.param2019(**DD_DOUBLE)
    coulomb = torch.eye(4, **DD_DOUBLE)
    rhs = torch.full((4,), 1e16, **DD_DOUBLE)
    constraint = torch.ones_like(rhs)

    def solve_charges(total_charge: Tensor) -> Tensor:
        return model._solve_cg(rhs, constraint, coulomb, total_charge, False)

    # With A = I, z = 1e16 and y = 1. Float64 cannot retain the Q = 1
    # correction in mu = (sum(z) - Q) / sum(y), so q incorrectly becomes
    # zero. The neutral system is an exact, valid control in the same batch.
    total_charges = rhs.new_tensor([[0.0], [1.0]])
    if execution in ("eager", "batch"):
        requested_charge = (
            total_charges[1] if execution == "eager" else total_charges
        )
        with pytest.raises(ConvergenceError, match="charge"):
            solve_charges(requested_charge)
        return

    # Transforms must preserve the valid system and mask only the failure.
    # The compile parameter carries the shared requires_compile marker.
    if execution == "vmap":
        with no_vmap_fallback():
            charges = vmap(solve_charges)(total_charges)
    else:
        charges = torch.compile(solve_charges, backend="eager", fullgraph=True)(
            total_charges
        )

    assert torch.equal(charges[0], torch.zeros_like(rhs))
    assert torch.isnan(charges[1]).all()


@pytest.mark.parametrize("size", [4, 32])
@pytest.mark.parametrize("scale", [1e-4, 1e4])
@pytest.mark.parametrize("condition", [1.0, 1e2])
def test_scaled_spd(size: int, scale: float, condition: float) -> None:
    """Scaled SPD solves satisfy both physical equations and a KKT reference."""
    model = eeq.EEQModel.param2019(**DD_DOUBLE)
    # A deterministic orthonormal cosine basis (DCT-II) makes Q^T Q = I.
    # In A = scale * Q diag(eigenvalues) Q^T, the spectrum then controls
    # conditioning directly, without relying on a random matrix.
    index = torch.arange(size, **DD_DOUBLE)
    basis = torch.cos(torch.pi * (index[:, None] + 0.5) * index / size)
    basis[:, 0] /= 2**0.5
    basis *= (2 / size) ** 0.5
    eigenvalues = torch.logspace(
        0, math.log10(condition), 4, **DD_DOUBLE
    ).repeat(size // 4)
    # `scale` changes magnitude while `condition` changes eigenvalue ratios.
    coulomb = scale * (basis * eigenvalues) @ basis.mT
    rhs = torch.sin(index + 0.5)
    constraint = torch.ones_like(rhs)
    total_charge = rhs.new_tensor([2.0])

    charges = model._solve_cg(rhs, constraint, coulomb, total_charge, False)
    reference_charges = model._solve_linear(
        rhs, constraint, coulomb, total_charge, False
    )

    # Recover mu by projecting b - A q onto c, then check the two physical
    # equations separately from agreement with the dense KKT reference.
    multiplier = (
        (rhs - coulomb @ charges) * constraint
    ).sum() / constraint.square().sum()
    stationarity_residual = (
        coulomb @ charges + constraint * multiplier - rhs
    ).norm()
    rhs_norm = torch.cat((rhs, total_charge)).norm()
    roundoff_factor = 10 * torch.finfo(rhs.dtype).eps
    stationarity_rounding_scale = (coulomb.abs() @ charges.abs()).norm() + (
        constraint * multiplier
    ).norm()
    stationarity_target = (
        eeq.CG_ATOL
        + eeq.CG_RTOL * rhs_norm
        + roundoff_factor * stationarity_rounding_scale
    )
    charge_residual = (constraint @ charges - total_charge).abs().max()
    charge_target = eeq.CG_ATOL + roundoff_factor * (
        (constraint * charges).abs().sum() + total_charge.abs().sum()
    )
    assert stationarity_residual <= stationarity_target
    assert charge_residual <= charge_target
    assert torch.allclose(charges, reference_charges, atol=1e-7, rtol=1e-8)


@pytest.mark.parametrize("size", [4, 64])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_ill_conditioned_jacobi(size: int, dtype: torch.dtype) -> None:
    """Jacobi handles a diagonal SPD matrix with condition number 1e6."""
    dd: DD = {"device": DEVICE, "dtype": dtype}
    model = eeq.EEQModel.param2019(**dd)
    # The unpreconditioned matrix spans six orders of magnitude. Jacobi
    # exactly rescales a diagonal matrix, so this remains a tractable test
    # of large condition numbers in both floating-point types.
    coulomb = torch.diag(torch.logspace(0, 6, size, **dd))
    rhs = torch.linspace(-1, 1, size, **dd)
    constraint = torch.ones_like(rhs)
    total_charge = rhs.new_tensor([3.0])
    charges = model._solve_cg(rhs, constraint, coulomb, total_charge, False)

    # Use double precision for the independent bordered reference so the
    # float32 comparison measures CG accuracy, not reference rounding.
    reference_charges = model._solve_linear(
        rhs.double(),
        constraint.double(),
        coulomb.double(),
        total_charge.double(),
        False,
    )
    tolerance = 1e-5 if dtype == torch.float32 else 1e-12
    assert torch.allclose(
        charges.double(), reference_charges, atol=tolerance, rtol=tolerance
    )
    assert torch.allclose(
        constraint @ charges, total_charge.squeeze(), atol=tolerance, rtol=0
    )


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
    # Alter the RHS through electronegativities and the Coulomb operator
    # through hardnesses and radii, with nonzero total molecular charges.
    model = model.replace(
        chi=model.chi * chi_scale,
        eta=model.eta + hardness_shift,
        rad=model.rad * 1.25,
    )
    charges = model(structure, solve_mode="cg")
    reference_charges = model(structure, solve_mode="linear")
    numbers = structure.numbers
    coulomb = eeq._coulomb_matrix_0d(
        numbers,
        structure.positions,
        model.rad[numbers],
        model.eta[numbers],
    )
    rhs = (
        -model.chi[numbers]
        + storch.safe_sqrt(model.cn(structure)) * model.kcn[numbers]
    )
    # All atoms are real, so c is a vector of ones and mu = mean(b - A q).
    multiplier = (rhs - coulomb @ charges).mean()
    tolerance = 1e-4 if dtype == torch.float32 else 1e-9
    assert torch.allclose(
        charges, reference_charges, atol=tolerance, rtol=tolerance
    )
    assert torch.allclose(
        charges.sum(), charges.new_tensor(total), atol=tolerance, rtol=0
    )
    assert (coulomb @ charges + multiplier - rhs).norm() <= tolerance * (
        1 + rhs.norm()
    )
