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
Molecules for testing the charges module.
"""
from __future__ import annotations

import torch
from tad_mctc.data.molecules import merge_nested_dicts, mols

from tad_multicharge.typing import Molecule, Tensor, TypedDict


class Refs(TypedDict):
    """Format of reference records containing GFN1-xTB and GFN2-xTB reference values."""

    total_charge: Tensor
    """Reference values total charge of molecule."""

    q: Tensor
    """Atomic charges."""

    energy: Tensor
    """Atom-resolved electrostatic energy."""


class Record(Molecule, Refs):
    """Store for molecular information and reference values"""


refs: dict[str, Refs] = {
    "NH3-dimer": Refs(
        {
            "total_charge": torch.tensor(0.0),
            "q": torch.tensor(
                [
                    -0.8347351804,
                    -0.8347351804,
                    +0.2730523336,
                    +0.2886305132,
                    +0.2730523336,
                    +0.2730523336,
                    +0.2886305132,
                    +0.2730523336,
                ],
            ),
            "energy": torch.tensor(
                [
                    -1.1076985038,
                    -1.1076985038,
                    +0.3337155415,
                    +0.3527546763,
                    +0.3337155415,
                    +0.3337155415,
                    +0.3527546763,
                    +0.3337155415,
                ],
            ),
        }
    ),
    "NH3": Refs(
        {
            "total_charge": torch.tensor(0.0),
            "q": torch.tensor([0.0]),
            "energy": torch.tensor([0.0]),
        }
    ),
    "PbH4-BiH3": Refs(
        {
            "total_charge": torch.tensor(0.0),
            "q": torch.tensor(
                [
                    +0.1830965969,
                    -0.0434600885,
                    -0.0434600949,
                    -0.0434600949,
                    -0.0452680726,
                    +0.0727632554,
                    -0.0267371663,
                    -0.0267371688,
                    -0.0267371663,
                ]
            ),
            "energy": torch.tensor(
                [
                    +0.2105113837,
                    -0.0512108813,
                    -0.0512108896,
                    -0.0512108896,
                    -0.0533415115,
                    +0.0847424122,
                    -0.0315042729,
                    -0.0315042752,
                    -0.0315042729,
                ],
            ),
        }
    ),
    "C6H5I-CH3SH": Refs(
        {
            "total_charge": torch.tensor(0.0),
            "q": torch.tensor(
                [
                    -0.1029278713,
                    -0.1001905841,
                    -0.1028043772,
                    -0.0774975738,
                    -0.0007325498,
                    -0.0780660341,
                    -0.1962493355,
                    +0.1120891066,
                    +0.1205055899,
                    +0.1123282728,
                    +0.1197578368,
                    +0.1121635250,
                    -0.1711138357,
                    +0.1212508178,
                    -0.2031014175,
                    +0.1153482095,
                    +0.1143692362,
                    +0.1048709842,
                ]
            ),
            "energy": torch.tensor(
                [
                    -0.1314732976,
                    -0.1279802409,
                    -0.1313155409,
                    -0.0988971754,
                    -0.0009357164,
                    -0.0996230320,
                    -0.2509842439,
                    +0.1371201812,
                    +0.1474161209,
                    +0.1374128412,
                    +0.1465013873,
                    +0.1372113095,
                    -0.2193897263,
                    +0.1483258209,
                    -0.2578882516,
                    +0.1411075101,
                    +0.1399101025,
                    +0.1282906877,
                ],
            ),
        }
    ),
    "AmF3": Refs(
        {
            "total_charge": torch.tensor(0.0),
            "q": torch.tensor(
                [
                    +7.7369226896274756e-01,
                    -2.5789822950551927e-01,
                    -2.5789930164742653e-01,
                    -2.5789473780980177e-01,
                ]
            ),
            "energy": torch.tensor(
                [
                    +0.8993736021,
                    -0.3547707945,
                    -0.3547722861,
                    -0.3547659963,
                ],
            ),
        }
    ),
}


samples: dict[str, Record] = merge_nested_dicts(mols, refs)
