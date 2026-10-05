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
`torch.compile(fullgraph=True)` of the EEQ model: building the `Structure`
and the model inside the compiled function, passing both in as `Node`
arguments, and compiling `jacrev` of the energy.

Compiling costs a few seconds per case, so only a handful are compiled.
`vmap`, `jacrev` and `jacfwd` are in `test_transforms.py`.
"""

from __future__ import annotations

import pytest
import torch
from tad_mctc.io.structure import Structure
from tad_mctc.typing import DD, Tensor
from torch.func import jacrev

from tad_multicharge.model import SolveMode, eeq

from ..conftest import DEVICE
from ..utils import (
    DYNAMO_SUPPORTED,
    DYNAMO_UNSUPPORTED_REASON,
    compile_fullgraph,
    load_batch,
    load_structure,
)

DD_DOUBLE: DD = {"device": DEVICE, "dtype": torch.double}

pytestmark = pytest.mark.skipif(
    not DYNAMO_SUPPORTED, reason=DYNAMO_UNSUPPORTED_REASON
)


@pytest.fixture(name="structure")
def fixture_structure(request: pytest.FixtureRequest) -> Structure:
    names = request.param
    if len(names) == 1:
        return load_structure(names[0], DD_DOUBLE, 0.0)
    return load_batch(names, DD_DOUBLE, [0.0] * len(names))


@pytest.mark.parametrize(
    "structure,solve_mode",
    [
        pytest.param(("SiH4",), "schur", id="single-schur"),
        pytest.param(("LiH", "ZnOOH-"), "schur", id="batch-schur"),
        pytest.param(("SiH4",), "linear", id="single-linear"),
    ],
    indirect=["structure"],
)
def test_construction_inside_compile(
    structure: Structure, solve_mode: SolveMode
) -> None:
    """Build the `Structure` and the model inside the compiled function."""

    def f(positions: Tensor, charge: Tensor) -> tuple[Tensor, Tensor]:
        s = Structure(
            numbers=structure.numbers, positions=positions, charge=charge
        )
        return eeq.get_eeq(s, return_energy=True, solve_mode=solve_mode)

    assert structure.charge is not None
    args = (structure.positions, structure.charge)

    torch._dynamo.reset()  # pylint: disable=protected-access
    q, e = compile_fullgraph(f)(*args)
    qref, eref = f(*args)

    assert torch.allclose(q, qref, atol=1e-10, rtol=0)
    assert torch.allclose(e, eref, atol=1e-10, rtol=0)


def test_compile_jacrev() -> None:
    """``torch.compile(jacrev(f))``: compiled forces equal eager ones."""
    structure = load_structure("SiH4", DD_DOUBLE, 0.0)
    model = eeq.EEQModel.param2019(**DD_DOUBLE)

    def energy(positions: Tensor) -> Tensor:
        s = structure.replace(positions=positions)
        return model(s, return_energy=True)[1].sum()

    torch._dynamo.reset()  # pylint: disable=protected-access
    forces = compile_fullgraph(jacrev(energy))(structure.positions)
    ref = jacrev(energy)(structure.positions)

    assert torch.allclose(forces, ref, atol=1e-10, rtol=0)


def test_nodes_as_arguments_no_recompile() -> None:
    """
    The model and the structure are pytrees: other values of the same
    shapes reuse the compiled graph.

    The model is called through ``solve``: calling it as ``model(...)``
    gives the same values, but Dynamo in torch 2.6 and 2.7 guards a called
    object that is a graph input on its identity, so every new model
    object recompiles there (not in 2.8 and later).
    """
    from torch._dynamo.testing import CompileCounter

    def f(model: eeq.EEQModel, structure: Structure) -> Tensor:
        return model.solve(structure, model.cn(structure))

    def g(model: eeq.EEQModel, structure: Structure) -> Tensor:
        return model(structure)

    model = eeq.EEQModel.param2019(**DD_DOUBLE)
    s1 = load_structure("SiH4", DD_DOUBLE, 0.0)
    assert s1.charge is not None
    s2 = s1.replace(positions=s1.positions * 1.01, charge=s1.charge + 1.0)
    m2 = model.replace(chi=model.chi * 1.1)
    cases = ((model, s1), (m2, s1), (model, s2))

    torch._dynamo.reset()  # pylint: disable=protected-access
    counter = CompileCounter()
    compiled = torch.compile(f, backend=counter, fullgraph=True)
    for m, s in cases:
        assert torch.allclose(compiled(m, s), g(m, s), atol=1e-10, rtol=0)
    assert counter.frame_count == 1

    torch._dynamo.reset()  # pylint: disable=protected-access
    compiled = torch.compile(g, backend="eager", fullgraph=True)
    for m, s in cases:
        assert torch.allclose(compiled(m, s), g(m, s), atol=1e-10, rtol=0)
