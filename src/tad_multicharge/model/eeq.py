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
>>> qat, energy = eeq_model.solve(
...     numbers, positions, total_charge, cn, return_energy=True
... )
>>> print(torch.sum(energy, -1))
tensor(-0.1750)
>>> print(qat)
tensor([-0.8347, -0.8347,  0.2731,  0.2886,  0.2731,  0.2731,  0.2886,  0.2731])
"""

from __future__ import annotations

import math
from typing import Literal, overload

import torch
from tad_mctc import storch
from tad_mctc.batch import real_atoms, real_pairs
from tad_mctc.ncoord import coordination_number, erf_count

from ..param import defaults, eeq2019
from ..typing import DD, Any, CountingFunction, Tensor, get_default_dtype
from .base import ChargeModel

__all__ = ["EEQModel", "get_charges"]


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

    @overload
    def solve(
        self,
        numbers: Tensor,
        positions: Tensor,
        total_charge: Tensor,
        cn: Tensor,
        return_energy: Literal[False] = False,
        solve_mode: Literal["schur", "linear"] = "schur",
    ) -> Tensor: ...

    @overload
    def solve(
        self,
        numbers: Tensor,
        positions: Tensor,
        total_charge: Tensor,
        cn: Tensor,
        return_energy: Literal[True],
        solve_mode: Literal["schur", "linear"] = "schur",
    ) -> tuple[Tensor, Tensor]: ...

    def solve(
        self,
        numbers: Tensor,
        positions: Tensor,
        total_charge: Tensor,
        cn: Tensor,
        return_energy: bool = False,
        solve_mode: Literal["schur", "linear"] = "schur",
    ) -> Tensor | tuple[Tensor, Tensor]:
        """
        Solve the electronegativity equilibration for the partial charges
        minimizing the electrostatic energy.

        Parameters
        ----------
        numbers : Tensor
            Atomic numbers of all atoms in the system.
            (shape: ``(..., nat)``).
        positions : Tensor
            Cartesian coordinates of the atoms in system
            (shape: ``(..., nat, 3)``).
        total_charge : Tensor
            Total charge of the system.
        cn : Tensor
            Coordination numbers for all atoms in the system.
        return_energy : bool, optional
            Return the EEQ energy as well. Defaults to `False`.
        solve_mode : Literal["schur", "linear"], optional
            Choose the solution method for the linear system.
            - ``"schur"``: Use Schur-complement based method with Cholesky
              factorization (default, recommended).
            - ``"linear"``: Solve the full bordered linear system directly.
              Less stable and slower for large systems.
            Defaults to ``"schur"``.

        Returns
        -------
        Tensor | (Tensor, Tensor)
            Tensor of electrostatic charges or tuple of partial charges and
            electrostatic energies if ``return_energy=True``.

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
        >>> e = eeq_model.solve(numbers, positions, total_charge, cn)[0]
        >>> energy = torch.sum(e, -1)
        >>> energy.backward()
        >>> print(positions.grad)
        tensor([[-9.3132e-09,  7.4506e-09, -4.8064e-02],
                [-1.2595e-02,  2.1816e-02,  1.6021e-02],
                [-1.2595e-02, -2.1816e-02,  1.6021e-02],
                [ 2.5191e-02, -6.9849e-10,  1.6021e-02]])
        >>> print(total_charge.grad)
        tensor(0.6312)
        """
        if self.device != positions.device:
            name = self.__class__.__name__
            raise RuntimeError(
                f"All tensors of '{name}' must be on the same device!\n"
                f"Use `{name}.param2019(device=device)` to correctly set it."
            )

        if self.dtype != positions.dtype:
            name = self.__class__.__name__
            raise RuntimeError(
                f"All tensors of '{name}' must have the same dtype!\n"
                f"Use `{name}.param2019(dtype=dtype)` to correctly set it."
            )

        total_charge = torch.atleast_1d(total_charge)

        # Attempt reshaping to proper batch shape: (n,) -> (n, 1)
        if total_charge.ndim == 1:
            if len(total_charge) != 1:
                total_charge = total_charge.view(-1, 1)

        if total_charge.ndim != numbers.ndim:
            raise ValueError(
                f"Total charge must have the same number of dimensions as "
                f"the atomic numbers tensor. Got\n"
                f"- atomic numbers: {numbers.shape}\n"
                f"- total charge:   {total_charge.shape}"
            )

        eps = torch.tensor(torch.finfo(positions.dtype).eps, **self.dd)
        zero = torch.tensor(0.0, **self.dd)
        stop = torch.sqrt(torch.tensor(2.0 / math.pi, **self.dd))  # sqrt(2/pi)

        real = real_atoms(numbers)
        mask = real_pairs(numbers, mask_diagonal=True)

        distances = torch.where(
            mask,
            storch.cdist(positions, positions, p=2),
            eps,
        )
        diagonal = mask.new_zeros(mask.shape)
        diagonal.diagonal(dim1=-2, dim2=-1).fill_(True)

        #############
        # Build RHS #
        #############

        cc = torch.where(
            real,
            -self.chi[numbers] + storch.sqrt(cn) * self.kcn[numbers],
            zero,
        )

        ##################
        # Build A matrix #
        ##################

        # radii
        rad = self.rad[numbers]
        rads = rad.unsqueeze(-1) ** 2 + rad.unsqueeze(-2) ** 2
        gamma = torch.where(mask, 1.0 / storch.sqrt(rads), zero)

        # hardness
        eta = torch.where(
            real,
            self.eta[numbers] + stop / rad,
            torch.tensor(1.0, **self.dd),
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

        ##############
        # Constraint #
        ##############

        # Build 'ones' vector for the constraint
        constraint = torch.where(
            real,
            torch.ones(numbers.shape, **self.dd),
            torch.zeros(numbers.shape, **self.dd),
        )

        #######################
        # Solve linear system #
        #######################

        if solve_mode == "schur":
            return self._solve_schur(
                cc, constraint, coulomb, total_charge, return_energy
            )

        if solve_mode == "linear":
            return self._solve_linear(
                cc, constraint, coulomb, total_charge, return_energy
            )

        raise ValueError(f"Unknown EEQ solve mode '{solve_mode}'!")

    def _solve_linear(
        self,
        cc: Tensor,
        constraint: Tensor,
        coulomb: Tensor,
        total_charge: Tensor,
        return_energy: bool,
    ) -> Tensor | tuple[Tensor, Tensor]:
        """
        Solve the EEQ linear system via standard linear solver.

        Parameters
        ----------
        cc : Tensor
            Right-hand side vector.
        constraint : Tensor
            Constraint vector (ones for real atoms, zeros else).
        coulomb : Tensor
            Coulomb interaction matrix.
        total_charge : Tensor
            Total charge of the system.
        return_energy : bool
            Whether to return the electrostatic energy as well.

        Returns
        -------
        Tensor | (Tensor, Tensor)
            Partial charges or tuple of partial charges and energies.
        """
        zeros = torch.zeros(cc.shape[:-1], **self.dd)

        rhs = torch.concat((cc, total_charge), dim=-1)

        # | Coulomb    Constraint |
        # | Constraint     0      |
        matrix = torch.concat(
            (
                torch.concat((coulomb, constraint.unsqueeze(-1)), dim=-1),
                torch.concat(
                    (constraint, zeros.unsqueeze(-1)), dim=-1
                ).unsqueeze(-2),
            ),
            dim=-2,
        )

        x = torch.linalg.solve(matrix, rhs)

        # do not compute energy unless specifically requested
        if return_energy is False:
            return x[..., :-1]

        # remove constraint for energy calculation
        _x = x[..., :-1]
        _m = matrix[..., :-1, :-1]
        _rhs = rhs[..., :-1]

        # E_scalar = 0.5 * x^T @ A @ x - b @ x^T
        # E_vector =  x * (0.5 * A @ x - b)
        _e = _x * (0.5 * torch.einsum("...ij,...j->...i", _m, _x) - _rhs)

        return _x, _e

    def _solve_schur(
        self,
        cc: Tensor,
        constraint: Tensor,
        coulomb: Tensor,
        total_charge: Tensor,
        return_energy: bool,
    ) -> Tensor | tuple[Tensor, Tensor]:
        """
        Solve the EEQ linear system via Schur-complement method.

        [ A    C ][ q ] = [ b ]
        [ C^T  0 ][ m ]   [ Q ]

        q = A^{-1}(b - C m)
        m = (C^T A^{-1} b - Q) / (C^T A^{-1} C)

        Parameters
        ----------
        cc : Tensor
            Right-hand side vector.
        constraint : Tensor
            Constraint vector (ones for real atoms, zeros else).
        coulomb : Tensor
            Coulomb interaction matrix.
        total_charge : Tensor
            Total charge of the system.
        return_energy : bool
            Whether to return the electrostatic energy as well.

        Returns
        -------
        Tensor | (Tensor, Tensor)
            Partial charges or tuple of partial charges and energies.
        """
        # Solve A X = B for two RHS at once: B = [b, 1].
        # Stack along last dimension giving `(..., nat, 2)`.
        B = torch.stack((cc, constraint), dim=-1)

        # Factor once via Cholesky: A = L L^T
        # (fast & stable since A is SPD; bordered systems is indefinite)
        L = torch.linalg.cholesky(coulomb)  # (..., nat, nat)

        # Solve A X = B for both RHS at once using the Cholesky factor
        # X[..., :, 0] = A^{-1} b ;  X[..., :, 1] = A^{-1} C
        X = torch.cholesky_solve(B, L)  # (..., nat, 2)
        z = X[..., :, 0]  # A^{-1} b, (..., nat)
        y = X[..., :, 1]  # A^{-1} C, (..., nat)

        # m = (C^T z - Q) / (C^T y) ; shape (..., 1)
        num = (constraint * z).sum(dim=-1, keepdim=True) - total_charge
        den = (constraint * y).sum(dim=-1, keepdim=True)
        m = num / den

        # q = z - y * m (broadcast m over the `nat` dimension)
        q = z - y * m  # (..., nat)

        # Do not compute energy unless specifically requested
        if return_energy is False:
            return q

        # E_scalar = 0.5 * x^T @ A @ x - b @ x^T
        # E_vector =  x * (0.5 * A @ x - b)
        e = q * (0.5 * torch.einsum("...ij,...j->...i", coulomb, q) - cc)

        return q, e


@overload
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
    return_energy: Literal[False],
    **kwargs: Any,
) -> Tensor: ...


@overload
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
    return_energy: Literal[True],
    **kwargs: Any,
) -> tuple[Tensor, Tensor]: ...


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
    return_energy: bool = False,
    solve_mode: Literal["schur", "linear"] = "schur",
    **kwargs: Any,
) -> Tensor | tuple[Tensor, Tensor]:
    """
    Calculate atomic EEQ charges and energies.

    Parameters
    ----------
    numbers : Tensor
        Atomic numbers for all atoms in the system of shape ``(..., nat)``.
    positions : Tensor
        Cartesian coordinates of all atoms (shape: ``(..., nat, 3)``).
    chrg : Tensor
        Total charge of system.
    counting_function : CountingFunction
        Calculate weight for pairs.
        Defaults to :func:`tad_mctc.ncoord.erf_count`.
    rcov : Tensor | None, optional
        Covalent radii for each species. Defaults to ``None``.
    cutoff : Tensor | float | int | None, optional
        Real-space cutoff.
        Defaults to :data:`tad_multicharge.defaults.CUTOFF_EEQ`.
    cn_max : Tensor | float | int | None, optional
        Maximum coordination number.
        Defaults to :data:`tad_multicharge.defaults.CUTOFF_EEQ_MAX`.
    kcn : Tensor | float | int, optional
        Steepness of the counting function.
    return_energy : bool, optional
        Return the EEQ energy as well. Defaults to ``False``.
    solve_mode : Literal["schur", "linear"], optional
        Choose the solution method for the linear system.
        - ``"schur"``: Use Schur-complement based method with Cholesky
          factorization (default, recommended).
        - ``"linear"``: Solve the full bordered linear system directly.
          Less stable and slower for large systems.
        Defaults to ``"schur"``.
    **kwargs : Any
        Additional keyword arguments for EEQ CN calculation.

    Returns
    -------
    (Tensor, Tensor)
        Tuple of electrostatic energies and partial charges.
    """
    eeq = EEQModel.param2019(device=positions.device, dtype=positions.dtype)
    cn = coordination_number(
        numbers,
        positions,
        counting_function=counting_function,
        rcov=rcov,
        cutoff=cutoff,
        cn_max=cn_max,
        kcn=kcn,
        **kwargs,
    )
    return eeq.solve(
        numbers,
        positions,
        chrg,
        cn,
        return_energy=return_energy,
        solve_mode=solve_mode,
    )


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
        Atomic numbers for all atoms in the system of shape ``(..., nat)``.
    positions : Tensor
        Cartesian coordinates of all atoms (shape: ``(..., nat, 3)``).
    chrg : Tensor
        Total charge of system.
    cutoff : Tensor | None, optional
        Real-space cutoff. Defaults to ``None``.

    Returns
    -------
    Tensor
        Atomic charges.
    """
    return get_eeq(numbers, positions, chrg, cutoff=cutoff, return_energy=False)


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
        Atomic numbers for all atoms in the system of shape ``(..., nat)``.
    positions : Tensor
        Cartesian coordinates of all atoms (shape: ``(..., nat, 3)``).
    chrg : Tensor
        Total charge of system.
    cutoff : Tensor | None, optional
        Real-space cutoff. Defaults to ``None``.

    Returns
    -------
    Tensor
        Atomic energies.
    """
    return get_eeq(
        numbers,
        positions,
        chrg,
        cutoff=cutoff,
        return_energy=True,
    )[1]
