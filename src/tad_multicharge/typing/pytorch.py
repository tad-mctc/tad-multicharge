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
Type annotations: PyTorch
=========================

PyTorch-related type annotations for this project.
"""
from tad_mctc.typing import (
    DD,
    CountingFunction,
    Molecule,
    Tensor,
    TensorLike,
    get_default_device,
    get_default_dtype,
)

__all__ = [
    "DD",
    "CountingFunction",
    "Molecule",
    "Tensor",
    "TensorLike",
    "get_default_device",
    "get_default_dtype",
]
