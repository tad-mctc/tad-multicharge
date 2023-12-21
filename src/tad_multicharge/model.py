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
Charge Model
============

Implementation of the electronegativity equlibration model for obtaining
atomic partial charges as well as atom-resolved electrostatic energies.
"""
from __future__ import annotations

import torch

from .typing import Tensor, TensorLike

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
