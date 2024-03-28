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

import torch

from ..typing import Tensor, TensorLike

__all__ = ["ChargeModel"]


class ChargeModel(TensorLike):
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

    __slots__ = ["chi", "kcn", "eta", "rad"]

    def __init__(
        self,
        chi: Tensor,
        kcn: Tensor,
        eta: Tensor,
        rad: Tensor,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__(device, dtype)
        self.chi = chi
        self.kcn = kcn
        self.eta = eta
        self.rad = rad

        if any(
            tensor.device != self.device
            for tensor in (self.chi, self.kcn, self.eta, self.rad)
        ):
            raise RuntimeError("All tensors must be on the same device!")

        if any(
            tensor.dtype != self.dtype
            for tensor in (self.chi, self.kcn, self.eta, self.rad)
        ):
            raise RuntimeError("All tensors must have the same dtype!")

    @abstractmethod
    def solve(
        self,
        numbers: Tensor,
        positions: Tensor,
        total_charge: Tensor,
        cn: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """
        Solve the electronegativity equilibration for the partial charges
        minimizing the electrostatic energy.

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
        """
