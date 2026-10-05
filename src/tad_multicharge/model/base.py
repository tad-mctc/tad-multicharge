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
Model: Base Charge Model
========================

Base class of the charge models.

A charge model is a frozen :class:`~tad_mctc.tree.Node`: its parameters
are pytree leaves, so a model can be passed through ``torch.func.vmap``,
``jacrev``, ``jacfwd`` and ``torch.compile`` like a tensor, and the
derivative with respect to all parameters is one ``jacrev`` over the model.
A different parametrization is obtained with :meth:`~tad_mctc.tree.Node.replace`
(e.g. ``model.replace(chi=chi)``), never by assignment.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Literal, overload

from tad_mctc.io.structure import Structure
from tad_mctc.ncoord.common import CNModel
from tad_mctc.tree import Node, child
from tad_mctc.typing import Tensor

__all__ = ["ChargeModel", "SolveMode"]


SolveMode = Literal["schur", "linear"]
"""Solution method of the linear system, see :meth:`ChargeModel.solve`."""

_PARAMETERS = ("chi", "kcn", "eta", "rad")


class ChargeModel(Node, ABC):
    """
    Model for electronegativity equilibration.

    Parameters
    ----------
    chi : Tensor
        Electronegativity for each element, shape ``(nelem,)``.
    kcn : Tensor
        Coordination number dependency of the electronegativity, shape
        ``(nelem,)``.
    eta : Tensor
        Chemical hardness for each element, shape ``(nelem,)``.
    rad : Tensor
        Atomic radii for each element, shape ``(nelem,)``.
    cn : CNModel
        Coordination number used by :meth:`__call__`.

    Raises
    ------
    TypeError
        A parameter is not a floating-point tensor, or the parameters have
        different dtypes.
    ValueError
        A parameter is not one-dimensional.
    RuntimeError
        The parameters are on different devices.
    """

    chi: Tensor = child()
    kcn: Tensor = child()
    eta: Tensor = child()
    rad: Tensor = child()
    cn: CNModel = child()

    def _validate(self) -> None:
        for name in _PARAMETERS:
            value = getattr(self, name)
            if not isinstance(value, Tensor) or not value.is_floating_point():
                raise TypeError(
                    f"{type(self).__name__}.{name} must be a floating-point "
                    "tensor."
                )
            if value.ndim != 1:
                raise ValueError(
                    f"{type(self).__name__}.{name} must be one-dimensional "
                    f"(one entry per element), got shape {tuple(value.shape)}."
                )

    def _check_structure(self, structure: Structure) -> None:
        """
        Check that the model can be applied to the structure. Reads only
        metadata, so it is safe under ``vmap`` and ``torch.compile``.
        """
        name = type(self).__name__
        if structure.lattice is not None:
            raise NotImplementedError(
                f"'{name}' does not support periodic structures."
            )

        if self.device != structure.positions.device:
            raise RuntimeError(
                f"All tensors of '{name}' must be on the same device!\n"
                f"Use `{name}.param2019(device=device)` or `.to(device)` to "
                "correctly set it."
            )

        if self.dtype != structure.positions.dtype:
            raise RuntimeError(
                f"All tensors of '{name}' must have the same dtype!\n"
                f"Use `{name}.param2019(dtype=dtype)` or `.type(dtype)` to "
                "correctly set it."
            )

    @staticmethod
    def _total_charge(structure: Structure) -> Tensor:
        """
        Total charge of each structure, shape ``(..., 1)``. An absent
        charge means neutral.
        """
        batch = structure.numbers.shape[:-1]
        if structure.charge is None:
            return structure.positions.new_zeros((*batch, 1))
        return structure.charge.reshape(*batch, 1)

    @overload
    def __call__(
        self,
        structure: Structure,
        *,
        return_energy: Literal[False] = ...,
        solve_mode: SolveMode = ...,
    ) -> Tensor: ...

    @overload
    def __call__(
        self,
        structure: Structure,
        *,
        return_energy: Literal[True],
        solve_mode: SolveMode = ...,
    ) -> tuple[Tensor, Tensor]: ...

    @overload
    def __call__(
        self,
        structure: Structure,
        *,
        return_energy: bool,
        solve_mode: SolveMode = ...,
    ) -> Tensor | tuple[Tensor, Tensor]: ...

    def __call__(
        self,
        structure: Structure,
        *,
        return_energy: bool = False,
        solve_mode: SolveMode = "schur",
    ) -> Tensor | tuple[Tensor, Tensor]:
        """
        Compute the coordination number with :attr:`cn` and solve for the
        partial charges (see :meth:`solve`).

        Parameters
        ----------
        structure : Structure
            The molecule(s) to evaluate.
        return_energy : bool, optional
            Return the atom-resolved energy as well. Defaults to ``False``.
        solve_mode : SolveMode, optional
            Solution method, see :meth:`solve`. Defaults to ``"schur"``.

        Returns
        -------
        Tensor | (Tensor, Tensor)
            Partial charges, or partial charges and atom-resolved energies
            if ``return_energy=True``.
        """
        return self.solve(
            structure,
            self.cn(structure),
            return_energy=return_energy,
            solve_mode=solve_mode,
        )

    @abstractmethod
    def solve(
        self,
        structure: Structure,
        cn: Tensor,
        *,
        return_energy: bool = False,
        solve_mode: SolveMode = "schur",
    ) -> Tensor | tuple[Tensor, Tensor]:
        """
        Solve for the partial charges (and atom-resolved energies if
        ``return_energy=True``) of the structure with the coordination
        numbers ``cn``. See :meth:`.EEQModel.solve`.
        """
