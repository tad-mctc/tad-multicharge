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
Torch Autodiff Multicharge
==========================

Implementation of charge models in PyTorch.
This module allows to process a single structure or a batch of structures for
the calculation of atom-resolved dispersion energies.

.. note::

   This project is still in early development and the API is subject to change.
   Contributions are welcome, please checkout our
   `contributing guidelines <https://github.com/tad-mctc/tad-multicharge/blob/main/CONTRIBUTING.md>`_.

Example
-------
>>> import torch
>>> import tad_multicharge as mc
>>> import tad_mctc as mctc
>>>
>>> # S22 system 4: formamide dimer
>>> numbers = mctc.batch.pack((
...     mctc.data.pse.to_number("C C N N H H H H H H O O".split()),
...     mctc.data.pse.to_number("C O N H H H".split()),
... ))
>>>
>>> # coordinates in Bohr
>>> positions = mctc.batch.pack((
...     torch.tensor([
...         [-3.81469488143921, +0.09993441402912, 0.00000000000000],
...         [+3.81469488143921, -0.09993441402912, 0.00000000000000],
...         [-2.66030049324036, -2.15898251533508, 0.00000000000000],
...         [+2.66030049324036, +2.15898251533508, 0.00000000000000],
...         [-0.73178529739380, -2.28237795829773, 0.00000000000000],
...         [-5.89039325714111, -0.02589114569128, 0.00000000000000],
...         [-3.71254944801331, -3.73605775833130, 0.00000000000000],
...         [+3.71254944801331, +3.73605775833130, 0.00000000000000],
...         [+0.73178529739380, +2.28237795829773, 0.00000000000000],
...         [+5.89039325714111, +0.02589114569128, 0.00000000000000],
...         [-2.74426102638245, +2.16115570068359, 0.00000000000000],
...         [+2.74426102638245, -2.16115570068359, 0.00000000000000],
...     ]),
...     torch.tensor([
...         [-0.55569743203406, +1.09030425468557, 0.00000000000000],
...         [+0.51473634678469, +3.15152550263611, 0.00000000000000],
...         [+0.59869690244446, -1.16861263789477, 0.00000000000000],
...         [-0.45355203669134, -2.74568780438064, 0.00000000000000],
...         [+2.52721209544999, -1.29200800956867, 0.00000000000000],
...         [-2.63139587595376, +0.96447869452240, 0.00000000000000],
...     ]),
... ))
>>>
>>> # total charge of both systems
>>> charge = torch.tensor([0.0, 0.0])
>>>
>>> # calculate electrostatic energy in Hartree
>>> energy = torch.sum(eeq.get_energy(numbers, positions, charge), -1)
>>>
>>> torch.set_printoptions(precision=10)
>>> print(energy)
>>> # tensor([-0.2086755037, -0.0972094536])
>>> print(energy[0] - 2 * energy[1])
>>> # tensor(-0.0142565966)
"""
import torch

from . import model, param, typing
from .__version__ import __version__
from .model import eeq
from .model.eeq import get_charges as get_eeq_charges

__all__ = [
    "__version__",
    "get_eeq_charges",
    "model",
    "param",
    "typing",
    "eeq",
]
