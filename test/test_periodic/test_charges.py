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
Charges and energies of 3D periodic structures (Ewald summation) against
multicharge, single and batched.
"""

from __future__ import annotations

import pytest
import torch
from tad_mctc.data.structures import get_structure
from tad_mctc.io.structure import pack_structures
from tad_mctc.typing import DD

from tad_multicharge.model import SolveMode, eeq

from ..conftest import DEVICE
from .samples import SOLVE_MODES, load, samples

SAMPLES = list(samples)


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("solve_mode", SOLVE_MODES)
@pytest.mark.parametrize("name", SAMPLES)
def test_single(dtype: torch.dtype, solve_mode: SolveMode, name: str) -> None:
    dd: DD = {"device": DEVICE, "dtype": dtype}
    tol = 1e-4 if dtype == torch.float else 1e-12

    structure = load(name, dd)
    ref = samples[name]

    model = eeq.EEQModel.param2019(**dd)
    q, e = model(structure, return_energy=True, solve_mode=solve_mode)

    assert q.dtype == e.dtype == dtype
    assert pytest.approx(ref["total_charge"].item(), abs=tol) == q.sum().item()
    assert pytest.approx(ref["q"].cpu(), abs=tol) == q.cpu()
    assert pytest.approx(ref["energy"].item(), abs=tol) == e.sum().item()


@pytest.mark.parametrize("solve_mode", SOLVE_MODES)
def test_batch(solve_mode: SolveMode) -> None:
    """Cells of different size and lattice (and charge) in one batch."""
    dd: DD = {"device": DEVICE, "dtype": torch.double}
    names = ["CO2+", "ice-vi", "urea"]

    structure = pack_structures([load(name, dd) for name in names])
    assert structure.lattice is not None and structure.lattice.ndim == 3

    model = eeq.EEQModel.param2019(**dd)
    q, e = model(structure, return_energy=True, solve_mode=solve_mode)

    for i, name in enumerate(names):
        ref = samples[name]
        nat = ref["q"].shape[-1]
        assert pytest.approx(ref["q"].cpu(), abs=1e-12) == q[i, :nat].cpu()
        assert (q[i, nat:] == 0).all()
        assert (
            pytest.approx(ref["energy"].item(), abs=1e-12) == e[i].sum().item()
        )


def test_multicharge_unit_tests() -> None:
    """
    References of multicharge 0.5.0's own tests (``test/unit/test_pbc.f90``),
    computed from the positions as stored in mstore. Cyanamide agrees.
    Formamide differs by about ``7e-7`` in the energies: multicharge 0.5.0
    centres the diagonal of the truncated real-space sum on the wrong
    image (see ``get_pairs`` in its ``wignerseitz.f90``), and formamide has
    atoms outside the cell, which the EEQ model wraps into it (as dftd4
    does before calling multicharge).
    """
    dd: DD = {"device": DEVICE, "dtype": torch.double}
    model = eeq.EEQModel.param2019(**dd)

    qref = torch.tensor(
        [
            +3.47274007973765e-1, +3.47273017259661e-1, +3.47276806231462e-1,
            +3.47267235121650e-1, +3.47288751410308e-1, +3.47296840650318e-1,
            +3.47299868243747e-1, +3.47299359754750e-1, +3.45885710186856e-1,
            +3.45874246032015e-1, +3.45888242047875e-1, +3.45877451600398e-1,
            +3.45902365123333e-1, +3.45902162041418e-1, +3.45900336974539e-1,
            +3.45903770121699e-1, +3.58996736955583e-1, +3.58987991603151e-1,
            +3.59000859720791e-1, +3.58990960169615e-1, +3.58990422147650e-1,
            +3.58997842028106e-1, +3.58997124263551e-1, +3.59001728635141e-1,
            -5.86637659146611e-1, -5.86601835343945e-1, -5.86625078062709e-1,
            -5.86579999312624e-1, -5.86654394920376e-1, -5.86669360667441e-1,
            -5.86667481028248e-1, -5.86674873278027e-1, -4.65529092171338e-1,
            -4.65527713392387e-1, -4.65546450764424e-1, -4.65540905286785e-1,
            -4.65529860758785e-1, -4.65533757413217e-1, -4.65527683275370e-1,
            -4.65527691475100e-1,
        ],
        **dd,
    )  # fmt: skip
    q = model(get_structure("x23", "cyanamide", **dd))
    assert pytest.approx(qref.cpu(), abs=1e-12) == q.cpu()

    eref = torch.tensor(
        [
            +4.01878116759118e-1, +4.01884779867146e-1, +4.01836430415694e-1,
            +4.01859412981015e-1, +4.30132679242368e-1, +4.30105708698129e-1,
            +4.30085668404029e-1, +4.30108945050621e-1, +1.90400193163051e-1,
            +1.90401822781550e-1, +1.90404130981992e-1, +1.90407447306916e-1,
            +3.33868335196572e-1, +3.33848682655264e-1, +3.33875977331594e-1,
            +3.33893030511429e-1, -8.35635584733599e-1, -8.35614926719694e-1,
            -8.35542684637248e-1, -8.35589488474056e-1, -6.31079122091240e-1,
            -6.31059677948463e-1, -6.31085206912995e-1, -6.31081747027041e-1,
        ],
        **dd,
    )  # fmt: skip
    _, e = model(get_structure("x23", "formamide", **dd), return_energy=True)
    assert pytest.approx(eref.cpu(), abs=1e-6) == e.cpu()


def test_wrap_invariant() -> None:
    """Moving atoms by lattice vectors does not change anything."""
    dd: DD = {"device": DEVICE, "dtype": torch.double}
    structure = load("urea", dd)
    assert structure.lattice is not None

    shift = torch.zeros_like(structure.positions)
    shift[0] = 3.0 * structure.lattice[0] - structure.lattice[2]
    shift[5] = -2.0 * structure.lattice[1]
    moved = structure.replace(positions=structure.positions + shift)

    model = eeq.EEQModel.param2019(**dd)
    q, e = model(structure, return_energy=True)
    q2, e2 = model(moved, return_energy=True)

    assert pytest.approx(q.cpu(), abs=1e-14) == q2.cpu()
    assert pytest.approx(e.cpu(), abs=1e-14) == e2.cpu()


@pytest.mark.parametrize("name", ["CO2", "ice-vi"])
def test_supercell(name: str) -> None:
    """
    A supercell has the charges of its cell. multicharge (and the EEQ
    model) truncate the lattice sums at two cells, of the supercell or of
    the cell, which shows at about ``1e-6`` (and up to ``2e-5`` for
    urea).
    """
    dd: DD = {"device": DEVICE, "dtype": torch.double}
    structure = load(name, dd)
    lattice = structure.lattice
    assert lattice is not None

    supercell = structure.replace(
        numbers=structure.numbers.repeat(2),
        positions=torch.cat(
            [structure.positions, structure.positions + lattice[0]]
        ),
        lattice=lattice * torch.tensor([[2.0], [1.0], [1.0]], **dd),
    )

    model = eeq.EEQModel.param2019(**dd)
    q, e = model(structure, return_energy=True)
    q2, e2 = model(supercell, return_energy=True)

    assert pytest.approx(q.repeat(2).cpu(), abs=1e-5) == q2.cpu()
    assert pytest.approx(2 * e.sum().item(), abs=1e-5) == e2.sum().item()


def test_ghost() -> None:
    """Padding atoms in a cell do not change the result."""
    dd: DD = {"device": DEVICE, "dtype": torch.double}
    structure = load("CO2", dd)
    nat = structure.numbers.shape[-1]

    padded = structure.replace(
        numbers=torch.cat([structure.numbers, structure.numbers.new_zeros(3)]),
        positions=torch.cat(
            [structure.positions, structure.positions.new_zeros(3, 3)]
        ),
    )

    model = eeq.EEQModel.param2019(**dd)
    q, e = model(padded, return_energy=True)

    ref = samples["CO2"]
    assert pytest.approx(ref["q"].cpu(), abs=1e-12) == q[:nat].cpu()
    assert (q[nat:] == 0).all() and (e[nat:] == 0).all()
    assert pytest.approx(ref["energy"].item(), abs=1e-12) == e.sum().item()
