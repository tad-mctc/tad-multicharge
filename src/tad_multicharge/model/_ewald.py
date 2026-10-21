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
Model: Ewald summation
======================

Coulomb matrix of Gaussian charges in a 3D periodic structure, port of
``get_amat_3d`` of multicharge (``src/multicharge/model/eeq.f90``).

The interaction of two Gaussian charges, :math:`\\mathrm{erf}(\\gamma r)/r`,
is summed over all images by Ewald summation. Like the Fortran code, the
real-space and reciprocal-space sums are truncated at a fixed two cells
along every axis, centred on the nearest images of each pair (Wigner-Seitz
cell, :func:`~tad_mctc.neighbor.wignerseitz_cell`). The Ewald splitting
parameter adapts to the lattice. The :math:`\\mathbf{G} = 0` term is left
out, i.e., a charged cell has a neutralising background.

The images are weighted smoothly as in mctc-lib's Wigner-Seitz cell,
which multicharge 0.5.0 does not use yet: it averages the nearest images
with equal weights, and centres the diagonal on the wrong image (see
``get_pairs`` in its ``wignerseitz.f90``). The sums are truncated, so this
shows in the charges, by up to about ``5e-7`` (formamide of X23).

The splitting parameter is a discrete choice, without gradient, as in the
Fortran derivatives. Everything else, including the weights of the
images, is differentiable with respect to the positions and the lattice.
"""

from __future__ import annotations

import math

import torch
from tad_mctc.batch import real_atoms, real_pairs
from tad_mctc.ewald import get_alpha
from tad_mctc.kpoints import reciprocal_lattice
from tad_mctc.neighbor import (
    lattice_shifts_mctclib,
    wignerseitz_cell,
    wrap_to_central_cell,
)
from tad_mctc.typing import Tensor

__all__ = ["coulomb_matrix_3d"]


_REP = (2, 2, 2)
"""Cells summed over along each axis in each direction, in real and
reciprocal space (``get_dir_trans`` and ``get_rec_trans``)."""

_IMAGE_REP = (1, 1, 1)
"""Cells searched for the nearest images along each axis in each direction,
see :func:`~tad_mctc.neighbor.wignerseitz_cell`."""

_EPS = math.sqrt(torch.finfo(torch.float64).eps)
"""Real-space distance below which a term is the atom itself, and left
out (``eps`` of ``eeq.f90``)."""


def coulomb_matrix_3d(
    numbers: Tensor,
    positions: Tensor,
    lattice: Tensor,
    periodic: Tensor,
    rad: Tensor,
    eta: Tensor,
) -> Tensor:
    """
    Coulomb matrix of the EEQ model for a 3D periodic structure.

    Parameters
    ----------
    numbers : Tensor
        Atomic numbers, shape ``(..., nat)``. ``0`` marks padding.
    positions : Tensor
        Cartesian coordinates, shape ``(..., nat, 3)``.
    lattice : Tensor
        Lattice vectors as rows, shape ``(..., 3, 3)``.
    periodic : Tensor
        Periodic axes, shape ``(3,)`` or ``(..., 3)``. All must be
        periodic.
    rad : Tensor
        Charge widths of the atoms, shape ``(..., nat)``.
    eta : Tensor
        Chemical hardnesses of the atoms, shape ``(..., nat)``.

    Returns
    -------
    Tensor
        Coulomb matrix, shape ``(..., nat, nat)``. Padding atoms have a
        unit diagonal and no interactions.
    """
    real = real_atoms(numbers)
    mask = real_pairs(numbers, mask_diagonal=False)

    # The nearest images are searched in the neighbouring cells only.
    positions, _ = wrap_to_central_cell(
        positions, lattice, periodic.unsqueeze(-2)
    )

    alpha = get_alpha(lattice)[..., None, None]  # (..., 1, 1)
    wsc = wignerseitz_cell(positions, lattice, periodic)

    # entry `(i, j)` points from atom `i` to atom `j`
    vectors = positions[..., None, :, :] - positions[..., :, None, :]

    # Padding atoms have zero width. A unit width keeps their (masked)
    # terms and derivatives finite.
    rad = torch.where(real, rad, 1.0)

    # the self-interaction of a Gaussian charge is the diagonal at `i = j`
    gamma = 1.0 / torch.sqrt(rad[..., :, None] ** 2 + rad[..., None, :] ** 2)

    amat = _real_space(vectors, wsc.weights, wsc.shifts, lattice, gamma, alpha)
    amat = amat + _reciprocal_space(vectors, lattice, alpha)
    amat = torch.where(mask, amat, 0.0)

    diagonal = (
        eta
        + math.sqrt(2.0 / math.pi) / rad
        - 2.0 * alpha[..., 0] / math.sqrt(math.pi)
    )
    diagonal = torch.where(real, diagonal, 1.0)
    return amat + torch.diag_embed(diagonal)


def _real_space(
    vectors: Tensor,
    weights: Tensor,
    images: Tensor,
    lattice: Tensor,
    gamma: Tensor,
    alpha: Tensor,
) -> Tensor:
    """
    Real-space part of the Ewald sum, :math:`(\\mathrm{erf}(\\gamma r) -
    \\mathrm{erf}(\\alpha r)) / r`, over the cells around the nearest images
    (``get_amat_dir_3d``).

    The sum over the nearest images ``k`` and the cells ``t`` around them
    is one sum over the shifts ``s = k + t``, each with the summed weight
    of the images it is reached from. This keeps the shapes fixed
    (``vmap``, ``torch.compile``), whatever the number of nearest images.

    Parameters
    ----------
    vectors : Tensor
        Pair vectors ``x_j - x_i``, shape ``(..., nat, nat, 3)``.
    weights : Tensor
        Weights of the nearest images, shape ``(..., nat, nat, n_img)``.
    images : Tensor
        Integer shifts of the nearest images, shape ``(n_img, 3)``.
    lattice : Tensor
        Lattice vectors as rows, shape ``(..., 3, 3)``.
    gamma : Tensor
        Gaussian exponents of the pairs, shape ``(..., nat, nat)``.
    alpha : Tensor
        Ewald splitting parameter, shape ``(..., 1, 1)``.

    Returns
    -------
    Tensor
        Real-space part, shape ``(..., nat, nat)``.
    """
    shifts, reached = _combined_shifts(images)
    coefficients = weights @ reached.to(weights.dtype)  # (..., nat, nat, ns)

    translations = shifts.to(vectors.dtype) @ lattice  # (..., ns, 3)
    diff = vectors[..., None, :] + translations[..., None, None, :, :]
    distance_squared = (diff * diff).sum(-1)
    del diff

    valid = (coefficients > 0) & (distance_squared >= _EPS * _EPS)

    # mask before the square root, whose derivative is infinite at zero
    distance = torch.where(valid, distance_squared, 1.0).sqrt()
    terms = (
        torch.erf(gamma[..., None] * distance)
        - torch.erf(alpha[..., None] * distance)
    ) / distance

    return torch.where(valid, coefficients * terms, 0.0).sum(-1)


def _combined_shifts(images: Tensor) -> tuple[Tensor, Tensor]:
    """
    Every sum ``s = k + t`` of an image shift ``k`` and a cell shift ``t``
    (from ``-2`` to ``2`` along each axis, :data:`_REP`), and which image
    reaches which sum.

    Parameters
    ----------
    images : Tensor
        Integer image shifts, shape ``(n_img, 3)``, from ``-1`` to ``1``
        along each axis (:data:`_IMAGE_REP`).

    Returns
    -------
    tuple[Tensor, Tensor]
        The box of sums in row-major order, shape ``(ns, 3)``, and the
        boolean reach, shape ``(n_img, ns)``. The cell shifts fill the box
        around every image, so a sum is reached at most once.
    """
    # Python integers, not read from the tensors, which would break the
    # graph under `torch.compile`
    reach = [a + b for a, b in zip(_IMAGE_REP, _REP)]
    axes = [torch.arange(-r, r + 1, device=images.device) for r in reach]
    shifts = torch.cartesian_prod(*axes)  # (ns, 3)

    rep = torch.tensor(_REP, device=images.device)
    reached = ((shifts[None, :, :] - images[:, None, :]).abs() <= rep).all(-1)
    return shifts, reached


def _reciprocal_space(
    vectors: Tensor, lattice: Tensor, alpha: Tensor
) -> Tensor:
    """
    Reciprocal-space part of the Ewald sum (``get_amat_rec_3d``), without
    the :math:`\\mathbf{G} = 0` term.

    The Fortran code evaluates it at the nearest images, but it is periodic,
    so the pair vectors themselves give the same.

    Parameters
    ----------
    vectors : Tensor
        Pair vectors, shape ``(..., nat, nat, 3)``.
    lattice : Tensor
        Lattice vectors as rows, shape ``(..., 3, 3)``.
    alpha : Tensor
        Ewald splitting parameter, shape ``(..., 1, 1)``.

    Returns
    -------
    Tensor
        Reciprocal-space part, shape ``(..., nat, nat)``.
    """
    shifts = lattice_shifts_mctclib(_REP, origin=False, device=vectors.device)
    g = shifts.to(vectors.dtype) @ reciprocal_lattice(lattice)  # (..., ng, 3)
    g2 = (g * g).sum(-1)  # (..., ng)

    volume = torch.linalg.det(lattice).abs()[..., None, None]  # (..., 1, 1)
    damping = torch.exp(-0.25 * g2[..., None, None, :] / alpha[..., None] ** 2)
    damping = damping / g2[..., None, None, :]  # (..., 1, 1, ng)

    phase = torch.einsum("...ijx,...gx->...ijg", vectors, g)
    return 4.0 * math.pi / volume * (torch.cos(phase) * damping).sum(-1)
