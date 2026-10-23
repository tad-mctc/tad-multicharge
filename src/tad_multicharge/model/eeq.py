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
>>> from tad_mctc.io.structure import Structure
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
>>> structure = Structure(numbers=numbers, positions=positions)
>>> cn = torch.tensor([3.0, 3.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
>>> eeq_model = eeq.EEQModel.param2019()
>>> qat, energy = eeq_model.solve(structure, cn, return_energy=True)
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
from tad_mctc.autograd import is_functorch_tensor
from tad_mctc.batch import real_atoms, real_pairs
from tad_mctc.io.structure import Structure
from tad_mctc.ncoord import cn_eeq
from tad_mctc.ncoord.common import CNModel
from tad_mctc.tools import is_compiling
from tad_mctc.tree import child
from tad_mctc.typing import DD, Tensor, get_default_dtype

from ..param import eeq2019
from ._ewald import coulomb_matrix_3d
from .base import ChargeModel, SolveMode

__all__ = ["EEQModel", "get_charges", "get_eeq", "get_energy"]


class EEQModel(ChargeModel):
    """
    Electronegativity equilibration charge model published in

    - E. Caldeweyher, S. Ehlert, A. Hansen, H. Neugebauer, S. Spicher,
      C. Bannwarth and S. Grimme, *J. Chem. Phys.*, **2019**, 150, 154122.
      DOI: `10.1063/1.5090222 <https://dx.doi.org/10.1063/1.5090222>`__

    The coordination number defaults to :data:`tad_mctc.ncoord.cn_eeq`. A
    different one is set with ``model.replace(cn=cn_eeq.replace(cutoff=...))``.

    A structure with a lattice, periodic along all three lattice vectors,
    is evaluated with an Ewald summation, ported from multicharge: the
    lattice sums are truncated at two cells around the nearest images of
    each pair (weighted as in mctc-lib's Wigner-Seitz cell), with an Ewald
    parameter adapted to the lattice. The positions are wrapped into the
    cell first. A charged cell has a neutralising background. multicharge
    0.5.0 centres some sums on the wrong image, which changes its charges
    by up to about ``5e-7``.

    Example
    -------
    >>> import torch
    >>> from tad_mctc.data.structures import get_structure
    >>> from tad_multicharge import eeq
    >>> structure = get_structure("x23", "CO2", dtype=torch.double)
    >>> eeq_model = eeq.EEQModel.param2019(dtype=torch.double)
    >>> qat, energy = eeq_model(structure, return_energy=True)
    >>> print(qat[[0, 4]])
    tensor([ 0.4563, -0.2281], dtype=torch.float64)
    >>> print(energy.sum())
    tensor(-0.1245, dtype=torch.float64)
    """

    # A factory, not `default=cn_eeq`: `torch.compile` cannot trace a `Node`
    # as a plain field default when the model is built in compiled code.
    cn: CNModel = child(default_factory=lambda: cn_eeq)

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
            chi=eeq2019.chi.to(**dd),
            kcn=eeq2019.kcn.to(**dd),
            eta=eeq2019.eta.to(**dd),
            rad=eeq2019.rad.to(**dd),
        )

    @overload
    def solve(
        self,
        structure: Structure,
        cn: Tensor,
        *,
        return_energy: Literal[False] = False,
        solve_mode: SolveMode = "schur",
    ) -> Tensor: ...

    @overload
    def solve(
        self,
        structure: Structure,
        cn: Tensor,
        *,
        return_energy: Literal[True],
        solve_mode: SolveMode = "schur",
    ) -> tuple[Tensor, Tensor]: ...

    @overload
    def solve(
        self,
        structure: Structure,
        cn: Tensor,
        *,
        return_energy: bool = False,
        solve_mode: SolveMode = "schur",
    ) -> Tensor | tuple[Tensor, Tensor]: ...

    def solve(
        self,
        structure: Structure,
        cn: Tensor,
        *,
        return_energy: bool = False,
        solve_mode: SolveMode = "schur",
    ) -> Tensor | tuple[Tensor, Tensor]:
        """
        Solve the electronegativity equilibration for the partial charges
        minimizing the electrostatic energy.

        Parameters
        ----------
        structure : Structure
            The molecule(s) to evaluate. ``structure.charge`` is the total
            charge; absent means neutral.
        cn : Tensor
            Coordination numbers for all atoms in the system, shape
            ``(..., nat)``.
        return_energy : bool, optional
            Return the atom-resolved energy as well. Defaults to ``False``.
        solve_mode : SolveMode, optional
            Choose the solution method for the linear system.

            - ``"schur"``: Use Schur-complement based method with Cholesky
              factorization (default, recommended).
            - ``"linear"``: Solve the full bordered linear system directly.
              Less stable and slower for large systems.
            - ``"cg"``: Schur-complement method with the two SPD solves done
              by Jacobi-preconditioned conjugate gradient, with implicit
              derivatives (see :meth:`_solve_cg`). Can be faster for large
              systems; slower for small ones.

            Defaults to ``"schur"``.

        Returns
        -------
        Tensor | (Tensor, Tensor)
            Partial charges, or partial charges and atom-resolved energies
            if ``return_energy=True``.

        Raises
        ------
        NotImplementedError
            The structure is periodic along one or two lattice vectors
            only (1D or 2D).
        RuntimeError
            The structure is on a different device or has a different dtype
            than the model.
        ValueError
            ``solve_mode`` is unknown.
        ConvergenceError
            ``solve_mode="cg"`` did not converge (outside of ``vmap`` and
            ``torch.compile``, where the charges are NaN instead).

        Example
        -------
        >>> import torch
        >>> from tad_mctc.io.structure import Structure
        >>> from tad_multicharge import eeq
        >>> numbers = torch.tensor([7, 1, 1, 1])
        >>> positions = torch.tensor([
        ...     [+0.00000000000000, +0.00000000000000, -0.54524837997150],
        ...     [-0.88451840382282, +1.53203081565085, +0.18174945999050],
        ...     [-0.88451840382282, -1.53203081565085, +0.18174945999050],
        ...     [+1.76903680764564, +0.00000000000000, +0.18174945999050],
        ... ], requires_grad=True)
        >>> total_charge = torch.tensor(0.0, requires_grad=True)
        >>> structure = Structure(
        ...     numbers=numbers, positions=positions, charge=total_charge
        ... )
        >>> cn = torch.tensor([3.0, 1.0, 1.0, 1.0])
        >>> eeq_model = eeq.EEQModel.param2019()
        >>> _, e = eeq_model.solve(structure, cn, return_energy=True)
        >>> energy = torch.sum(e, -1)
        >>> energy.backward()
        >>> print(positions.grad[:, 2])
        tensor([-0.0481,  0.0160,  0.0160,  0.0160])
        >>> print(total_charge.grad)
        tensor(1.2625)
        """
        if solve_mode not in ("schur", "linear", "cg"):
            raise ValueError(f"Unknown EEQ solve mode '{solve_mode}'!")

        self._check_structure(structure)

        numbers = structure.numbers
        positions = structure.positions
        total_charge = self._total_charge(structure)

        real = real_atoms(numbers)

        #############
        # Build RHS #
        #############

        cc = torch.where(
            real,
            -self.chi[numbers] + storch.safe_sqrt(cn) * self.kcn[numbers],
            0.0,
        )

        ##################
        # Build A matrix #
        ##################

        rad, eta = self.rad[numbers], self.eta[numbers]

        if structure.lattice is None:
            coulomb = _coulomb_matrix_0d(numbers, positions, rad, eta)
        else:
            # `Structure` fills in a mask whenever it has a lattice.
            periodic = structure.periodic
            assert periodic is not None

            # Reads the values of the mask, so it is skipped under `vmap`,
            # `jacrev` and `torch.compile`.
            if (
                not is_functorch_tensor(periodic)
                and not is_compiling()
                and not bool(periodic.all())
            ):
                raise NotImplementedError(
                    f"'{type(self).__name__}' supports molecules and "
                    "structures periodic along all three lattice vectors "
                    "(3D), but not 1D or 2D periodic structures."
                )

            coulomb = coulomb_matrix_3d(
                numbers, positions, structure.lattice, periodic, rad, eta
            )

        ##############
        # Constraint #
        ##############

        # 'ones' vector for the constraint (zero for padding atoms)
        constraint = real.to(positions.dtype)

        #######################
        # Solve linear system #
        #######################

        if solve_mode == "schur":
            return self._solve_schur(
                cc, constraint, coulomb, total_charge, return_energy
            )

        if solve_mode == "cg":
            return self._solve_cg(
                cc, constraint, coulomb, total_charge, return_energy
            )

        return self._solve_linear(
            cc, constraint, coulomb, total_charge, return_energy
        )

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
            Total charge of the system, shape ``(..., 1)``.
        return_energy : bool
            Whether to return the electrostatic energy as well.

        Returns
        -------
        Tensor | (Tensor, Tensor)
            Partial charges or tuple of partial charges and energies.
        """
        zeros = cc.new_zeros(cc.shape[:-1])

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
            Total charge of the system, shape ``(..., 1)``.
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

        return _schur_charges(
            z, y, cc, constraint, coulomb, total_charge, return_energy
        )

    def _solve_cg(
        self,
        cc: Tensor,
        constraint: Tensor,
        coulomb: Tensor,
        total_charge: Tensor,
        return_energy: bool,
    ) -> Tensor | tuple[Tensor, Tensor]:
        """
        Solve the EEQ linear system via the Schur-complement method, with
        the two solves of the SPD Coulomb matrix done by Jacobi-preconditioned
        conjugate gradient (see :meth:`_solve_schur`).

        The derivatives of the charges are the implicit derivatives of the
        linear system (:func:`tad_mctc.storch.linsolve.implicit_solve`), so
        they are independent of the CG iterations. The matrix is still
        built densely: one CG iteration costs :math:`O(N^2)`.

        Parameters
        ----------
        cc : Tensor
            Right-hand side vector.
        constraint : Tensor
            Constraint vector (ones for real atoms, zeros else).
        coulomb : Tensor
            Coulomb interaction matrix.
        total_charge : Tensor
            Total charge of the system, shape ``(..., 1)``.
        return_energy : bool
            Whether to return the electrostatic energy as well.

        Returns
        -------
        Tensor | (Tensor, Tensor)
            Partial charges or tuple of partial charges and energies.

        Raises
        ------
        ConvergenceError
            CG did not converge, e.g. because the Coulomb matrix is not
            positive definite (not under ``vmap`` or ``torch.compile``,
            where the charges of that system are NaN instead).
        """
        # Both right-hand sides as a batch dimension: (..., 2, nat)
        B = torch.stack((cc, constraint), dim=-2)

        result = storch.linsolve.implicit_solve(
            _coulomb_matvec,
            B,
            coulomb,
            method="cg",
            preconditioner=_jacobi,
            rtol=CG_RTOL,
            atol=CG_ATOL,
        )
        X = storch.linsolve.require_convergence(result, name="EEQ CG solve")
        z = X[..., 0, :]  # A^{-1} b, (..., nat)
        y = X[..., 1, :]  # A^{-1} C, (..., nat)

        return _schur_charges(
            z, y, cc, constraint, coulomb, total_charge, return_energy
        )


CG_RTOL = 1e-12
"""Relative tolerance of the CG solves of ``solve_mode="cg"`` (raised by
tad-mctc to 10 machine epsilons for single precision)."""

CG_ATOL = 1e-14
"""Absolute tolerance of the CG solves of ``solve_mode="cg"``."""


def _coulomb_matvec(x: Tensor, coulomb: Tensor) -> Tensor:
    """Coulomb matrix times the right-hand sides, ``(..., k, nat)``."""
    return torch.einsum("...ij,...kj->...ki", coulomb, x)


def _jacobi(r: Tensor, coulomb: Tensor) -> Tensor:
    """Jacobi (diagonal) preconditioner of the Coulomb matrix."""
    return r / coulomb.diagonal(dim1=-2, dim2=-1).unsqueeze(-2)


def _schur_charges(
    z: Tensor,
    y: Tensor,
    cc: Tensor,
    constraint: Tensor,
    coulomb: Tensor,
    total_charge: Tensor,
    return_energy: bool,
) -> Tensor | tuple[Tensor, Tensor]:
    """
    Charges (and energies) from the solutions ``z = A^{-1} b`` and
    ``y = A^{-1} C`` of the Schur-complement method.
    """
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


def _coulomb_matrix_0d(
    numbers: Tensor, positions: Tensor, rad: Tensor, eta: Tensor
) -> Tensor:
    """
    Coulomb matrix of the EEQ model for a molecule.

    Parameters
    ----------
    numbers : Tensor
        Atomic numbers, shape ``(..., nat)``. ``0`` marks padding.
    positions : Tensor
        Cartesian coordinates, shape ``(..., nat, 3)``.
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
    eps = torch.finfo(positions.dtype).eps
    stop = math.sqrt(2.0 / math.pi)

    real = real_atoms(numbers)
    mask = real_pairs(numbers, mask_diagonal=True)
    diagonal = torch.eye(
        numbers.shape[-1], dtype=torch.bool, device=numbers.device
    )

    distances = torch.where(
        mask,
        storch.cdist(positions, positions, p=2),
        eps,
    )

    # radii
    rads = rad.unsqueeze(-1) ** 2 + rad.unsqueeze(-2) ** 2
    gamma = torch.where(mask, 1.0 / storch.safe_sqrt(rads), 0.0)

    # hardness (unity for padding atoms to keep the matrix regular)
    eta = torch.where(real, eta + stop / rad, 1.0)

    return torch.where(
        diagonal,
        eta.unsqueeze(-1),
        torch.where(
            mask,
            torch.erf(distances * gamma) / distances,
            0.0,
        ),
    )


@overload
def get_eeq(
    structure: Structure,
    *,
    cn: CNModel = cn_eeq,
    return_energy: Literal[False] = False,
    solve_mode: SolveMode = "schur",
) -> Tensor: ...


@overload
def get_eeq(
    structure: Structure,
    *,
    cn: CNModel = cn_eeq,
    return_energy: Literal[True],
    solve_mode: SolveMode = "schur",
) -> tuple[Tensor, Tensor]: ...


def get_eeq(
    structure: Structure,
    *,
    cn: CNModel = cn_eeq,
    return_energy: bool = False,
    solve_mode: SolveMode = "schur",
) -> Tensor | tuple[Tensor, Tensor]:
    """
    Calculate atomic EEQ charges and energies with the standard (2019)
    parametrization.

    Parameters
    ----------
    structure : Structure
        The molecule(s) to evaluate. ``structure.charge`` is the total
        charge; absent means neutral.
    cn : CNModel, optional
        Coordination number. Defaults to :data:`tad_mctc.ncoord.cn_eeq`;
        another cutoff, for example, is ``cn_eeq.replace(cutoff=...)``.
    return_energy : bool, optional
        Return the EEQ energy as well. Defaults to ``False``.
    solve_mode : SolveMode, optional
        Solution method for the linear system, see
        :meth:`EEQModel.solve`. Defaults to ``"schur"``.

    Returns
    -------
    Tensor | (Tensor, Tensor)
        Partial charges, or partial charges and atom-resolved energies if
        ``return_energy=True``.
    """
    model = EEQModel.param2019(
        device=structure.positions.device, dtype=structure.positions.dtype
    ).replace(cn=cn)
    return model(structure, return_energy=return_energy, solve_mode=solve_mode)


def get_charges(structure: Structure, *, cn: CNModel = cn_eeq) -> Tensor:
    """
    Calculate atomic EEQ charges.

    Parameters
    ----------
    structure : Structure
        The molecule(s) to evaluate. ``structure.charge`` is the total
        charge; absent means neutral.
    cn : CNModel, optional
        Coordination number. Defaults to :data:`tad_mctc.ncoord.cn_eeq`.

    Returns
    -------
    Tensor
        Atomic charges, shape ``(..., nat)``.
    """
    return get_eeq(structure, cn=cn, return_energy=False)


def get_energy(structure: Structure, *, cn: CNModel = cn_eeq) -> Tensor:
    """
    Calculate atomic EEQ energies.

    Parameters
    ----------
    structure : Structure
        The molecule(s) to evaluate. ``structure.charge`` is the total
        charge; absent means neutral.
    cn : CNModel, optional
        Coordination number. Defaults to :data:`tad_mctc.ncoord.cn_eeq`.

    Returns
    -------
    Tensor
        Atom-resolved energies, shape ``(..., nat)``.
    """
    return get_eeq(structure, cn=cn, return_energy=True)[1]
