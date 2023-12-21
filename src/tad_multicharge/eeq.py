# This file is part of tad-multicharge.
#
# SPDX-Identifier: LGPL-3.0
# Copyright (C) 2023 Marvin Friede
#
# tad-multicharge is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# tad-multicharge is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with tad-multicharge. If not, see <https://www.gnu.org/licenses/>.
"""
Electronegativity equilibration charge model
============================================

Implementation of the electronegativity equlibration model for obtaining
atomic partial charges as well as atom-resolved electrostatic energies.

Example
-------
>>> import torch
>>> from tad_multicharge import eeq
>>> numbers = torch.tensor([7, 7, 1, 1, 1, 1, 1, 1])
>>> positions = torch.tensor([
...     [-2.98334550857544, -0.08808205276728, +0.00000000000000],
...     [+2.98334550857544, +0.08808205276728, +0.00000000000000],
...     [-4.07920360565186, +0.25775116682053, +1.52985656261444],
...     [-1.60526800155640, +1.24380481243134, +0.00000000000000],
...     [-4.07920360565186, +0.25775116682053, -1.52985656261444],
...     [+4.07920360565186, -0.25775116682053, -1.52985656261444],
...     [+1.60526800155640, -1.24380481243134, +0.00000000000000],
...     [+4.07920360565186, -0.25775116682053, +1.52985656261444],
... ])
>>> total_charge = torch.tensor(0.0)
>>> cn = torch.tensor([3.0, 3.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
>>> eeq_model = eeq.EEQModel.param2019()
>>> energy, qat = eeq.solve(numbers, positions, total_charge, eeq_model, cn)
>>> print(torch.sum(energy, -1))
tensor(-0.1750)
>>> print(qat)
tensor([-0.8347, -0.8347,  0.2731,  0.2886,  0.2731,  0.2731,  0.2886,  0.2731])
"""
from __future__ import annotations

import math

import torch
from tad_mctc import storch
from tad_mctc.batch import real_atoms, real_pairs
from tad_mctc.ncoord import cn_eeq, erf_count

from . import defaults
from .model import ChargeModel
from .param import eeq2019
from .typing import DD, Any, CountingFunction, Tensor, get_default_dtype

__all__ = ["EEQModel", "solve", "get_charges"]


class EEQModel(ChargeModel):
    """
    Electronegativity equilibration charge model published in

    - E. Caldeweyher, S. Ehlert, A. Hansen, H. Neugebauer, S. Spicher,
      C. Bannwarth and S. Grimme, *J. Chem. Phys.*, **2019**, 150, 154122.
      DOI: `10.1063/1.5090222 <https://dx.doi.org/10.1063/1.5090222>`__
    """

    @classmethod
    def param2019(
        cls,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> EEQModel:
        """
        Create the EEQ model from the standard (2019) parametrization.

        Parameters
        ----------
        device : torch.device | None, optional
            PyTorch device for the tensors. Defaults to `None`.
        dtype : torch.dtype | None, optional
            PyTorch floating point type for the tensors. Defaults to `None`.

        Returns
        -------
        EEQModel
            Instance of the EEQ charge model class.
        """
        dd: DD = {
            "device": device,
            "dtype": dtype if dtype is not None else get_default_dtype(),
        }

        return cls(
            eeq2019.chi.to(**dd),
            eeq2019.kcn.to(**dd),
            eeq2019.eta.to(**dd),
            eeq2019.rad.to(**dd),
            **dd,
        )


def solve(
    numbers: Tensor,
    positions: Tensor,
    total_charge: Tensor,
    model: ChargeModel,
    cn: Tensor,
) -> tuple[Tensor, Tensor]:
    """
    Solve the electronegativity equilibration for the partial charges minimizing
    the electrostatic energy.

    Parameters
    ----------
    numbers : Tensor
        Atomic numbers of all atoms in the system.
    positions : Tensor
        Cartesian coordinates of the atoms in the system (batch, natoms, 3).
    total_charge : Tensor
        Total charge of the system.
    model : ChargeModel
        Charge model to use.
    cn : Tensor
        Coordination numbers for all atoms in the system.

    Returns
    -------
    (Tensor, Tensor)
        Tuple of electrostatic energies and partial charges.

    Example
    -------
    >>> import torch
    >>> from tad_multicharge import eeq
    >>> numbers = torch.tensor([7, 1, 1, 1])
    >>> positions = torch.tensor([
    ...     [+0.00000000000000, +0.00000000000000, -0.54524837997150],
    ...     [-0.88451840382282, +1.53203081565085, +0.18174945999050],
    ...     [-0.88451840382282, -1.53203081565085, +0.18174945999050],
    ...     [+1.76903680764564, +0.00000000000000, +0.18174945999050],
    ... ], requires_grad=True)
    >>> total_charge = torch.tensor(0.0, requires_grad=True)
    >>> cn = torch.tensor([3.0, 1.0, 1.0, 1.0])
    >>> eeq_model = eeq.EEQModel.param2019()
    >>> energy = torch.sum(eeq.solve(numbers, positions, total_charge, eeq_model, cn)[0], -1)
    >>> energy.backward()
    >>> print(positions.grad)
    tensor([[-9.3132e-09,  7.4506e-09, -4.8064e-02],
            [-1.2595e-02,  2.1816e-02,  1.6021e-02],
            [-1.2595e-02, -2.1816e-02,  1.6021e-02],
            [ 2.5191e-02, -6.9849e-10,  1.6021e-02]])
    >>> print(total_charge.grad)
    tensor(0.6312)
    """
    dd: DD = {"device": positions.device, "dtype": positions.dtype}

    if model.device != positions.device:
        name = model.__class__.__name__
        raise RuntimeError(
            f"All tensors of '{name}' must be on the same device!\n"
            f"Use `{name}.param2019(device=device)` to correctly set the it."
        )

    if model.dtype != positions.dtype:
        name = model.__class__.__name__
        raise RuntimeError(
            f"All tensors of '{name}' must have the same dtype!\n"
            f"Use `{name}.param2019(dtype=dtype)` to correctly set it."
        )

    eps = torch.tensor(torch.finfo(positions.dtype).eps, **dd)
    zero = torch.tensor(0.0, **dd)
    stop = torch.sqrt(torch.tensor(2.0 / math.pi, **dd))  # sqrt(2/pi)

    real = real_atoms(numbers)
    mask = real_pairs(numbers, mask_diagonal=True)

    distances = torch.where(mask, storch.cdist(positions, positions, p=2), eps)
    diagonal = mask.new_zeros(mask.shape)
    diagonal.diagonal(dim1=-2, dim2=-1).fill_(True)

    cn_sqrt = torch.sqrt(torch.clamp(cn, min=eps))
    cc = torch.where(
        real,
        -model.chi[numbers] + cn_sqrt * model.kcn[numbers],
        zero,
    )
    rhs = torch.concat((cc, total_charge.unsqueeze(-1)), dim=-1)

    # radii
    rad = model.rad[numbers]
    rads = torch.clamp(rad.unsqueeze(-1) ** 2 + rad.unsqueeze(-2) ** 2, min=eps)
    gamma = torch.where(mask, 1.0 / torch.sqrt(rads), zero)

    # hardness
    eta = torch.where(
        real,
        model.eta[numbers] + stop / rad,
        torch.tensor(1.0, **dd),
    )

    coulomb = torch.where(
        diagonal,
        eta.unsqueeze(-1),
        torch.where(
            mask,
            torch.erf(distances * gamma) / distances,
            zero,
        ),
    )

    constraint = torch.where(
        real,
        torch.ones(numbers.shape, **dd),
        torch.zeros(numbers.shape, **dd),
    )
    zeros = torch.zeros(numbers.shape[:-1], **dd)

    matrix = torch.concat(
        (
            torch.concat((coulomb, constraint.unsqueeze(-1)), dim=-1),
            torch.concat((constraint, zeros.unsqueeze(-1)), dim=-1).unsqueeze(-2),
        ),
        dim=-2,
    )

    x = torch.linalg.solve(matrix, rhs)
    e = x * (0.5 * torch.einsum("...ij,...j->...i", matrix, x) - rhs)
    return e[..., :-1], x[..., :-1]


def get_eeq(
    numbers: Tensor,
    positions: Tensor,
    chrg: Tensor,
    *,
    counting_function: CountingFunction = erf_count,
    rcov: Tensor | None = None,
    cutoff: Tensor | float | int | None = defaults.EEQ_CN_CUTOFF,
    cn_max: Tensor | float | int | None = defaults.EEQ_CN_MAX,
    kcn: Tensor | float | int = defaults.EEQ_KCN,
    **kwargs: Any,
) -> tuple[Tensor, Tensor]:
    """
    Calculate atomic EEQ charges and energies.

    Parameters
    ----------
    numbers : Tensor
        Atomic numbers of all atoms in the system.
    positions : Tensor
        Cartesian coordinates of the atoms in the system (batch, natoms, 3).
    chrg : Tensor
        Total charge of system.
    counting_function : CountingFunction
        Calculate weight for pairs. Defaults to `erf_count`.
    rcov : Tensor | None, optional
        Covalent radii for each species. Defaults to `None`.
    cutoff : Tensor | float | int | None, optional
        Real-space cutoff. Defaults to `defaults.CUTOFF_EEQ`.
    cn_max : Tensor | float | int | None, optional
        Maximum coordination number. Defaults to `defaults.CUTOFF_EEQ_MAX`.
    kcn : Tensor | float | int, optional
        Steepness of the counting function.

    Returns
    -------
    (Tensor, Tensor)
        Tuple of electrostatic energies and partial charges.
    """
    eeq = EEQModel.param2019(device=positions.device, dtype=positions.dtype)
    cn = cn_eeq(
        numbers,
        positions,
        counting_function=counting_function,
        rcov=rcov,
        cutoff=cutoff,
        cn_max=cn_max,
        kcn=kcn,
        **kwargs,
    )
    return solve(numbers, positions, chrg, eeq, cn)


def get_charges(
    numbers: Tensor,
    positions: Tensor,
    chrg: Tensor,
    cutoff: Tensor | None = None,
) -> Tensor:
    """
    Calculate atomic EEQ charges.

    Parameters
    ----------
    numbers : Tensor
        Atomic numbers of all atoms in the system.
    positions : Tensor
        Cartesian coordinates of the atoms in the system (batch, natoms, 3).
    chrg : Tensor
        Total charge of system.
    cutoff : Tensor | None, optional
        Real-space cutoff. Defaults to `None`.

    Returns
    -------
    Tensor
        Atomic charges.
    """
    return get_eeq(numbers, positions, chrg, cutoff=cutoff)[1]


def get_energy(
    numbers: Tensor,
    positions: Tensor,
    chrg: Tensor,
    cutoff: Tensor | None = None,
) -> Tensor:
    """
    Calculate atomic EEQ energies.

    Parameters
    ----------
    numbers : Tensor
        Atomic numbers of all atoms in the system.
    positions : Tensor
        Cartesian coordinates of the atoms in the system (batch, natoms, 3).
    chrg : Tensor
        Total charge of system.
    cutoff : Tensor | None, optional
        Real-space cutoff. Defaults to `None`.

    Returns
    -------
    Tensor
        Atomic energies.
    """
    return get_eeq(numbers, positions, chrg, cutoff=cutoff)[0]
