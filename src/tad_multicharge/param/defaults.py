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
Parameters: Defaults
====================

Default global parameters of the charge models.

For the EEQ model, the following defaults are set:
- real-space cutoffs for the coordination number

- maximum coordination number

- steepness of CN counting function
"""

EEQ_CN_CUTOFF = 25.0
"""Coordination number cutoff within EEQ (25.0)."""

EEQ_CN_MAX = 8.0
"""Maximum coordination number (8.0)."""

EEQ_KCN = 7.5
"""Steepness of counting function in EEQ model (7.5)."""
