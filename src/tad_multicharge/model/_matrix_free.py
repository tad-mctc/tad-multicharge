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
"""Tiled Gaussian Coulomb products and the constrained molecular EEQ solve."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial

import torch
from tad_mctc.autograd import is_functorch_tensor, is_vmapped
from tad_mctc.exceptions import ConvergenceError
from tad_mctc.storch import linsolve
from tad_mctc.storch.pairwise import pair_cell_sums, pair_product
from tad_mctc.typing import Tensor


@dataclass(frozen=True)
class MatrixFreeOptions:
    """
    Memory and execution options for molecular matrix-free EEQ.

    Parameters
    ----------
    tile : int | None, optional
        Pair-product rows per tile. ``None`` uses tad-mctc's device-specific
        defaults. Derivative tiles stay within its eager memory bound.
    coarse_memory : int, optional
        Budget in bytes for balancing-preconditioner storage and setup
        workspaces, shared across a batch. Defaults to 64 MiB. Zero uses
        Jacobi only. Pair tiles and autograd graphs are separate allocations.
    cell_size : float, optional
        Initial coarse-cell edge in bohr. Cells enlarge to fit the budget.
    mixed_precision : bool, optional
        Refine float32 products against the exact float64 operator. Float32
        inputs use their own precision. Defaults to ``True``.
    fused : bool, optional
        Fuse CUDA pair products when supported. Defaults to ``True``;
        transforms and CPU execution use eager tiles.
    maxiter : int | None, optional
        Iteration limit per projected solve. ``None`` uses tad-mctc's limit.
    fallback : bool, optional
        Retry failed mixed-precision refinement with the exact operator.
        Defaults to ``True``. An unconverged fallback still fails normally.

    Notes
    -----
    Geometry-dependent preconditioners are rebuilt on every call. Under
    torch.func transforms the solver uses Jacobi, avoiding data-dependent
    cell construction. This changes convergence speed, not the equations.
    """

    tile: int | None = None
    coarse_memory: int = 64 * 1024 * 1024
    cell_size: float = 12.0
    mixed_precision: bool = True
    fused: bool = True
    maxiter: int | None = None
    fallback: bool = True

    def __post_init__(self) -> None:
        if self.tile is not None and self.tile < 1:
            raise ValueError("The pair tile size must be positive.")
        if self.coarse_memory < 0:
            raise ValueError("The coarse memory budget cannot be negative.")
        if not math.isfinite(self.cell_size) or self.cell_size <= 0:
            raise ValueError(
                "The coarse cell size must be finite and positive."
            )
        if self.maxiter is not None and self.maxiter < 1:
            raise ValueError("The iteration limit must be positive.")


def _kernel(
    distance: Tensor, pi: tuple[Tensor, ...], pj: tuple[Tensor, ...]
) -> Tensor:
    """Off-diagonal Gaussian interaction; padding is excluded by the caller."""
    gamma = torch.rsqrt(pi[0].square() + pj[0].square())
    return torch.erf(distance * gamma) / distance


def _diagonal(rad: Tensor, eta: Tensor, real: Tensor) -> Tensor:
    """Unit padding diagonal and finite dummy radii, including in backward."""
    radius = torch.where(real, rad, 1.0)
    return torch.where(real, eta + math.sqrt(2.0 / math.pi) / radius, 1.0)


def _matvec(
    q: Tensor,
    positions: Tensor,
    rad: Tensor,
    eta: Tensor,
    real: Tensor,
    *,
    options: MatrixFreeOptions,
    dtype: torch.dtype | None = None,
) -> Tensor:
    # Failed iterates can be nonfinite, including the primal saved by
    # implicit AD. Keep zero adjoints of those systems from introducing
    # 0 * NaN into shared parameter pullbacks. The caller still rejects
    # the original nonfinite solution; every finite product is unchanged.
    q = torch.nan_to_num(q, nan=0.0, posinf=0.0, neginf=0.0)
    pairs = pair_product(
        _kernel,
        positions,
        q,
        torch.where(real, rad, 1.0),
        mask=real,
        tile=options.tile,
        dtype=dtype,
        fused=options.fused and positions.is_cuda,
    )
    return _diagonal(rad, eta, real) * q + pairs


def _jacobi(
    r: Tensor, positions: Tensor, rad: Tensor, eta: Tensor, real: Tensor
) -> Tensor:
    """Evaluate from explicit solve arguments, including under nested AD."""
    diagonal = _diagonal(rad, eta, real).detach()
    # Positive preconditioning lets projected CG report indefinite curvature.
    return r / diagonal.abs().clamp_min(torch.finfo(r.dtype).eps)


def _preconditioner(
    positions: Tensor,
    rad: Tensor,
    eta: Tensor,
    real: Tensor,
    options: MatrixFreeOptions,
) -> Callable[..., Tensor]:
    diagonal = _diagonal(rad, eta, real).detach()

    n = positions.shape[-2]
    batch = math.prod(positions.shape[:-2])
    dtype = torch.float32 if options.mixed_precision else positions.dtype
    # AZ may be float32 while its coarse Cholesky is float64. Reserve in
    # the larger dtype so the budget covers both assembly and factorization.
    itemsize = max(
        torch.empty((), dtype=dtype).element_size(), positions.element_size()
    )
    # K <= N bounds K^2 by N*K. Eight N*K buffers conservatively account
    # for AZ assembly, its copies, and the coarse matrices/factorization.
    max_cells = min(
        n, options.coarse_memory // (8 * itemsize * batch * max(n, 1))
    )
    if max_cells < 1 or any(
        is_functorch_tensor(t) for t in (positions, rad, eta, real)
    ):
        return _jacobi

    with torch.no_grad():
        cells, size = linsolve.cell_index(
            positions, options.cell_size, mask=real, max_cells=max_cells
        )
        if size == 0:
            return _jacobi
        products = pair_cell_sums(
            _kernel,
            positions,
            cells,
            size,
            torch.where(real, rad, 1.0),
            mask=real,
            tile=options.tile,
            dtype=dtype,
        )
        # Padding is outside all cells; its unit diagonal must not enter AZ.
        products = products.scatter_add(
            -1,
            cells.clamp_min(0).unsqueeze(-1),
            torch.where(real, diagonal, 0.0).to(dtype).unsqueeze(-1),
        )
        # A failed coarse factorization should not obscure projected-CG
        # failure handling for an unphysical model. Jacobi is always valid.
        try:
            return linsolve.balancing(diagonal, cells, products)
        except torch.linalg.LinAlgError:
            return _jacobi


def solve(
    cc: Tensor,
    constraint: Tensor,
    positions: Tensor,
    rad: Tensor,
    eta: Tensor,
    total_charge: Tensor,
    return_energy: bool,
    options: MatrixFreeOptions | None,
    *,
    rtol: float,
    atol: float,
) -> Tensor | tuple[Tensor, Tensor]:
    """Solve the molecular KKT equations without constructing a dense matrix."""
    if options is None:
        options = MatrixFreeOptions()
    real = constraint.bool()
    exact = partial(_matvec, options=options)
    fast = (
        partial(_matvec, options=options, dtype=torch.float32)
        if options.mixed_precision and positions.dtype == torch.float64
        else None
    )
    result = linsolve.constrained_solve(
        exact,
        cc,
        constraint.unsqueeze(-2),
        total_charge,
        positions,
        rad,
        eta,
        real,
        preconditioner=_preconditioner(positions, rad, eta, real, options),
        fast_matvec=fast,
        fallback=options.fallback,
        maxiter=options.maxiter,
        rtol=rtol,
        atol=atol,
    )
    # Do not feed convergence-masked NaNs into the quadratic energy.
    # Status remains separate until both final physical outputs are ready.
    if not is_vmapped(result.status):
        linsolve.require_convergence(result, name="EEQ matrix-free solve")
    finite = torch.isfinite(result.solution).all(dim=-1)
    q = torch.nan_to_num(result.solution, nan=0.0, posinf=0.0, neginf=0.0)
    rounding = 10 * torch.finfo(q.dtype).eps
    with torch.no_grad():
        charge_scale = (
            (constraint * q).abs().sum(-1) + total_charge.squeeze(-1).abs()
        ).clamp_min(1.0)
        charge_ok = (
            finite
            & torch.isfinite(result.constraint_residual)
            & (result.constraint_residual <= atol + rounding * charge_scale)
        )
    if not is_vmapped(charge_ok) and not bool(charge_ok.all()):
        raise ConvergenceError("EEQ matrix-free solve violates total charge.")
    accepted = (
        (result.status == linsolve.STATUS_CONVERGED) & charge_ok
    ).unsqueeze(-1)
    if not return_energy:
        return linsolve.mask_converged(q, accepted)
    e = q * (0.5 * exact(q, positions, rad, eta, real) - cc)
    return (
        linsolve.mask_converged(q, accepted),
        linsolve.mask_converged(e, accepted),
    )
