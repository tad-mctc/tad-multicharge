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
