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
Utility functions for testing.
"""

from __future__ import annotations

from collections.abc import Sequence

from tad_mctc.data.structures import get_structure
from tad_mctc.io.structure import Structure, pack_structures
from tad_mctc.typing import DD, Tensor

__all__ = [
    "CHECK_IDS",
    "SOURCES",
    "load_batch",
    "load_samples",
    "load_structure",
    "single_and_paired",
]


SOURCES: dict[str, tuple[str, str]] = {
    "AmF3": ("other", "AmF3"),
    "Ag2Cl22-": ("other", "Ag2Cl22-"),
    "C6H5I-CH3SH": ("other", "C6H5I-CH3SH"),
    "LiH": ("mb16_43", "LiH"),
    "MB16_43_01": ("mb16_43", "01"),
    "MB16_43_02": ("mb16_43", "02"),
    "NH3": ("heavy28", "nh3"),
    "NH3-dimer": ("other", "NH3-dimer"),
    "PbH4-BiH3": ("heavy28", "pbh4_bih3"),
    "SiH4": ("mb16_43", "SiH4"),
    "ZnOOH-": ("other", "ZnOOH-"),
    "vancoh2": ("other", "vancoh2"),
}
"""Historical sample name -> ``(collection, record)`` of
:func:`tad_mctc.data.structures.get_structure`."""


def load_structure(
    name: str, dd: DD, charge: Tensor | float | None = None
) -> Structure:
    """
    Load a sample by its historical name, moved to ``dd``.

    Parameters
    ----------
    name : str
        Key of :data:`SOURCES`.
    dd : DD
        Device and dtype.
    charge : Tensor | float | None, optional
        Total charge. ``None`` (default) keeps the record's charge; no
        record in :data:`SOURCES` stores one, so this means neutral.

    Returns
    -------
    Structure
        The unbatched structure.
    """
    return get_structure(*SOURCES[name], charge=charge, **dd)


def load_batch(
    names: Sequence[str],
    dd: DD,
    charges: Sequence[Tensor | float | None] | None = None,
) -> Structure:
    """
    Load samples by their historical names and pack them into one batched
    structure (see :func:`load_structure`).
    """
    if charges is None:
        charges = [None] * len(names)
    return pack_structures(
        [load_structure(n, dd, c) for n, c in zip(names, charges)]
    )


def load_samples(
    names: Sequence[str], dd: DD, charge: Tensor | float | None = None
) -> Structure:
    """
    Load one sample unbatched (:func:`load_structure`) or several as one
    batch (:func:`load_batch`), all with the same total ``charge``.
    """
    if len(names) == 1:
        return load_structure(names[0], dd, charge)
    return load_batch(names, dd, [charge] * len(names))


CHECK_IDS = ["grad", "gradgrad"]
"""Test ids for parametrizing over ``[dgradcheck, dgradgradcheck]``."""


def single_and_paired(names: Sequence[str], first: str) -> list[list[str]]:
    """
    Sample lists for :func:`load_samples`: each name alone, then each name
    batched after ``first``.
    """
    return [[name] for name in names] + [[first, name] for name in names]
