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

Implementation of a base class for charge models.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Literal, overload

import torch

from ..typing import ModuleLike, Tensor

__all__ = ["ChargeModel"]


class ChargeModel(ModuleLike):
    """
    Model for electronegativity equilibration.
    """

    chi: Tensor
    """Electronegativity for each element"""

    kcn: Tensor
    """Coordination number dependency of the electronegativity"""

    eta: Tensor
    """Chemical hardness for each element"""

    rad: Tensor
    """Atomic radii for each element"""

    def __init__(
        self,
        chi: Tensor,
        kcn: Tensor,
        eta: Tensor,
        rad: Tensor,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        tensors = (chi, kcn, eta, rad)
        inferred_device = tensors[0].device
        inferred_dtype = tensors[0].dtype

        target_device = device if device is not None else inferred_device
        target_dtype = dtype if dtype is not None else inferred_dtype

        self._validate_requested_dtype(target_dtype)

        if device is None and dtype is None:
            self._validate_tensor_devices(tensors, target_device)
            self._validate_tensor_dtypes(tensors, target_dtype)
        else:
            tensors = tuple(
                tensor.to(device=target_device, dtype=target_dtype)
                for tensor in tensors
            )

        for name, tensor in zip(
            ("chi", "kcn", "eta", "rad"),
            tensors,
            strict=True,
        ):
            self.register_buffer(name, tensor)

    @overload
    def solve(
        self,
        numbers: Tensor,
        positions: Tensor,
        total_charge: Tensor,
        cn: Tensor,
        return_energy: Literal[False] = ...,
        solve_mode: Literal["schur", "linear"] = ...,
    ) -> Tensor: ...

    @overload
    def solve(
        self,
        numbers: Tensor,
        positions: Tensor,
        total_charge: Tensor,
        cn: Tensor,
        return_energy: Literal[True],
        solve_mode: Literal["schur", "linear"] = ...,
    ) -> tuple[Tensor, Tensor]: ...

    @abstractmethod
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
        model : ChargeModel
            Charge model to use.
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
        """
