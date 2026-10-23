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
Derivatives of the CG solve mode (`solve_mode="cg"`).

The Cholesky-Schur default is the reference: CG solves the same equations,
and its derivatives are the implicit derivatives of the same linear system,
so all derivatives must agree to the solver tolerance, for every nesting of
reverse and forward mode.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest
import torch
from tad_mctc.autograd import dgradcheck, dgradgradcheck, no_vmap_fallback
from tad_mctc.exceptions import ConvergenceError
from tad_mctc.io.structure import Structure
from tad_mctc.ncoord import cn_eeq
from tad_mctc.neighbor import build_periodic_shifts
from tad_mctc.typing import DD, Tensor
from torch.func import jacfwd, jacrev, vmap

from tad_multicharge.model import SolveMode, eeq

from ..conftest import DEVICE
from ..test_periodic.samples import load as load_periodic
from ..utils import load_batch, load_samples, load_structure

DD_DOUBLE: DD = {"device": DEVICE, "dtype": torch.double}

SAMPLES = ["LiH", "SiH4", "AmF3", "Ag2Cl22-", "ZnOOH-"]


def _model() -> eeq.EEQModel:
    return eeq.EEQModel.param2019(**DD_DOUBLE)


def _energy(structure: Structure, mode: SolveMode) -> Callable:
    model = _model()

    def energy(positions: Tensor) -> Tensor:
        s = structure.replace(positions=positions)
        return model(s, return_energy=True, solve_mode=mode)[1].sum()

    return energy


def assert_close(x: Tensor, y: Tensor, tol: float = 1e-9) -> None:
    assert x.shape == y.shape
    assert pytest.approx(y.cpu(), abs=tol, rel=tol) == x.cpu()


@pytest.mark.parametrize("name", SAMPLES)
def test_charges_and_energies(name: str) -> None:
    structure = load_structure(name, DD_DOUBLE, 0.0)
    model = _model()

    q, e = model(structure, return_energy=True, solve_mode="cg")
    qref, eref = model(structure, return_energy=True, solve_mode="schur")

    assert_close(q, qref, 1e-10)
    assert_close(e, eref, 1e-10)


@pytest.mark.parametrize("name", SAMPLES)
def test_forces(name: str) -> None:
    structure = load_structure(name, DD_DOUBLE, 0.0)

    forces = jacrev(_energy(structure, "cg"))(structure.positions)
    ref = jacrev(_energy(structure, "schur"))(structure.positions)
    assert_close(forces, ref)


HESSIANS = {
    "rev-rev": lambda f: jacrev(jacrev(f)),
    "fwd-rev": lambda f: jacfwd(jacrev(f)),
    "fwd-fwd": lambda f: jacfwd(jacfwd(f)),
}


@pytest.mark.parametrize("name", ["LiH", "SiH4", "ZnOOH-"])
@pytest.mark.parametrize("nested", HESSIANS)
def test_hessian(name: str, nested: str) -> None:
    structure = load_structure(name, DD_DOUBLE, 0.0)

    hess = HESSIANS[nested](_energy(structure, "cg"))(structure.positions)
    ref = jacrev(jacrev(_energy(structure, "schur")))(structure.positions)
    assert_close(hess, ref)


def test_third_derivative() -> None:
    structure = load_structure("LiH", DD_DOUBLE, 0.0)

    def third(mode: SolveMode) -> Tensor:
        return jacfwd(jacrev(jacrev(_energy(structure, mode))))(
            structure.positions
        )

    assert_close(third("cg"), third("schur"), 1e-8)


@pytest.mark.parametrize("param", ["chi", "kcn", "eta", "rad"])
def test_parameters(param: str) -> None:
    """Gradient and Hessian with respect to every model parameter."""
    structure = load_structure("ZnOOH-", DD_DOUBLE, -1.0)
    cn = cn_eeq(structure)
    model = _model()

    def energy(mode: SolveMode) -> Callable:
        def f(value: Tensor) -> Tensor:
            m = model.replace(**{param: value})
            return m.solve(structure, cn, return_energy=True, solve_mode=mode)[
                1
            ].sum()

        return f

    value = getattr(model, param)
    assert_close(jacrev(energy("cg"))(value), jacrev(energy("schur"))(value))
    assert_close(
        jacfwd(jacrev(energy("cg")))(value),
        jacrev(jacrev(energy("schur")))(value),
    )


def test_mixed_position_parameter() -> None:
    """Derivative of the forces with respect to the electronegativities."""
    structure = load_structure("SiH4", DD_DOUBLE, 0.0)
    model = _model()

    def energy(mode: SolveMode) -> Callable:
        def f(positions: Tensor, chi: Tensor) -> Tensor:
            s = structure.replace(positions=positions)
            m = model.replace(chi=chi)
            return m(s, return_energy=True, solve_mode=mode)[1].sum()

        return f

    args = (structure.positions, model.chi)
    out = jacrev(jacrev(energy("cg"), argnums=0), argnums=1)(*args)
    ref = jacrev(jacrev(energy("schur"), argnums=0), argnums=1)(*args)
    assert_close(out, ref)


def test_total_charge() -> None:
    """First and second derivatives with respect to the total charge."""
    structure = load_structure("ZnOOH-", DD_DOUBLE, -1.0)
    model = _model()

    def energy(mode: SolveMode) -> Callable:
        def f(charge: Tensor) -> Tensor:
            s = structure.replace(charge=charge)
            return model(s, return_energy=True, solve_mode=mode)[1].sum()

        return f

    charge = torch.tensor(-1.0, **DD_DOUBLE)
    for t in (jacrev, lambda f: jacfwd(jacrev(f))):
        assert_close(t(energy("cg"))(charge), t(energy("schur"))(charge))


@pytest.mark.parametrize("name", ["CO2+", "urea"])
def test_periodic(name: str) -> None:
    """Forces, lattice derivatives and their mixed Hessian block of a 3D
    periodic cell (Ewald Coulomb matrix)."""
    structure = load_periodic(name, DD_DOUBLE)
    assert structure.lattice is not None and structure.periodic is not None
    shifts = build_periodic_shifts(
        structure.lattice, structure.periodic, cn_eeq.cutoff
    )
    model = _model()

    def energy(mode: SolveMode) -> Callable:
        def f(positions: Tensor, lattice: Tensor) -> Tensor:
            s = structure.replace(positions=positions, lattice=lattice)
            return model(s, shifts, return_energy=True, solve_mode=mode)[
                1
            ].sum()

        return f

    args = (structure.positions, structure.lattice)
    grads = jacrev(energy("cg"), argnums=(0, 1))(*args)
    refs = jacrev(energy("schur"), argnums=(0, 1))(*args)
    for grad, ref in zip(grads, refs):
        assert_close(grad, ref)

    mixed = jacfwd(jacrev(energy("cg"), argnums=0), argnums=1)(*args)
    ref = jacrev(jacrev(energy("schur"), argnums=0), argnums=1)(*args)
    assert_close(mixed, ref)


@pytest.mark.grad
@pytest.mark.parametrize("check", [dgradcheck, dgradgradcheck])
@pytest.mark.parametrize("names", [["LiH"], ["LiH", "SiH4"]], ids="+".join)
def test_gradcheck(check: Callable[..., bool], names: list[str]) -> None:
    """Charges with respect to positions against finite differences."""
    structure = load_samples(names, DD_DOUBLE, 0.0)
    model = _model()
    positions = structure.positions.clone().requires_grad_(True)

    def charges(pos: Tensor) -> Tensor:
        return model(structure.replace(positions=pos), solve_mode="cg")

    assert check(charges, positions, atol=1e-7)


def test_vmap_hessian() -> None:
    """`vmap` of the Hessian over a padded batch of structures."""
    batch = load_batch(["LiH", "ZnOOH-"], DD_DOUBLE, [0.0, -1.0])

    def hessian(mode: SolveMode) -> Callable:
        def f(structure: Structure) -> Tensor:
            return jacrev(jacrev(_energy(structure, mode)))(structure.positions)

        return f

    with no_vmap_fallback():
        hess = vmap(hessian("cg"))(batch)
    ref = vmap(hessian("schur"))(batch)
    assert_close(hess, ref)


def test_not_positive_definite() -> None:
    """
    An unphysical parametrization with a negative hardness makes the Coulomb
    matrix indefinite: CG raises in eager mode, is NaN only for the affected
    system under `vmap`, and never silently returns a wrong solution.
    """
    batch = load_batch(["LiH", "SiH4"], DD_DOUBLE, [0.0, 0.0])
    model = _model()
    eta = model.eta.clone()
    eta[14] = -10.0  # silicon (indexed by atomic number)
    bad = model.replace(eta=eta)

    with pytest.raises(ConvergenceError):
        bad(batch, solve_mode="cg")

    with pytest.raises(torch.linalg.LinAlgError):
        bad(batch, solve_mode="schur")

    q = vmap(lambda s: bad(s, solve_mode="cg"))(batch)
    qref = model(batch, solve_mode="cg")
    assert_close(q[0], qref[0], 1e-10)
    assert torch.isnan(q[1]).all()
