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

import shutil
import sys
from collections.abc import Callable, Sequence
from typing import Any

import torch
from tad_mctc.data.structures import get_structure
from tad_mctc.io.structure import Structure, pack_structures
from tad_mctc.tools.compile import is_compile_supported
from tad_mctc.typing import DD, Tensor

__all__ = [
    "COMPILE_BACKEND",
    "DYNAMO_SUPPORTED",
    "DYNAMO_UNSUPPORTED_REASON",
    "SOURCES",
    "compile_fullgraph",
    "load_batch",
    "load_structure",
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
        Total charge. ``None`` (default) means neutral.

    Returns
    -------
    Structure
        The unbatched structure.
    """
    structure = get_structure(*SOURCES[name], **dd)
    if charge is None:
        return structure
    return structure.replace(charge=torch.as_tensor(charge, **dd))


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


DYNAMO_SUPPORTED = is_compile_supported()
"""Whether ``torch.compile`` is supported on this Python/PyTorch
combination."""

DYNAMO_UNSUPPORTED_REASON = (
    "torch.compile/Dynamo is not supported on this Python/PyTorch combination"
)


def _has_cxx_compiler() -> bool:
    """Whether the C++ compiler that TorchInductor calls is on ``PATH``."""
    names = ["cl"] if sys.platform == "win32" else ["c++", "g++", "clang++"]
    return any(shutil.which(name) is not None for name in names)


COMPILE_BACKEND = "inductor" if _has_cxx_compiler() else "aot_eager"
"""The ``torch.compile`` backend for tests. ``fullgraph=True`` is decided by
Dynamo before any backend runs, so ``"aot_eager"`` still checks that a
function traces as one graph, just without generating C++ code."""


def compile_fullgraph(fn: Callable[..., Any]) -> Callable[..., Any]:
    """``torch.compile(fn)`` as one graph with static shapes on
    :data:`COMPILE_BACKEND`."""
    return torch.compile(
        fn, fullgraph=True, dynamic=False, backend=COMPILE_BACKEND
    )
