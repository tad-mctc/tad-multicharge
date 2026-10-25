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
from typing import Any

import pytest
import torch
from tad_mctc import storch
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
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_forces(name: str, dtype: torch.dtype) -> None:
    """Position derivatives agree in single and double precision."""
    dd: DD = {"device": DEVICE, "dtype": dtype}
    structure = load_structure(name, dd, 0.0)
    model = eeq.EEQModel.param2019(**dd)

    def energy(positions: Tensor, mode: SolveMode) -> Tensor:
        _, atomic_energies = model(
            structure.replace(positions=positions),
            return_energy=True,
            solve_mode=mode,
        )
        return atomic_energies.sum()

    # Comparing energy gradients also compares forces: their common minus
    # sign does not affect agreement between the two solvers.
    gradient = jacrev(lambda positions: energy(positions, "cg"))(
        structure.positions
    )
    reference_gradient = jacrev(lambda positions: energy(positions, "schur"))(
        structure.positions
    )
    tolerance = 1e-5 if dtype == torch.float32 else 1e-9
    assert_close(gradient, reference_gradient, tolerance)


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
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_parameters(param: str, dtype: torch.dtype) -> None:
    """Gradient and Hessian with respect to every model parameter."""
    dd: DD = {"device": DEVICE, "dtype": dtype}
    structure = load_structure("ZnOOH-", dd, -1.0)
    cn = cn_eeq(structure)
    model = eeq.EEQModel.param2019(**dd)
    tolerance = 1e-4 if dtype == torch.float32 else 1e-9

    def energy_for_mode(mode: SolveMode) -> Callable[[Tensor], Tensor]:
        def energy(value: Tensor) -> Tensor:
            parameter_model = model.replace(**{param: value})
            _, atomic_energies = parameter_model.solve(
                structure, cn, return_energy=True, solve_mode=mode
            )
            return atomic_energies.sum()

        return energy

    parameter = getattr(model, param)
    cg_energy = energy_for_mode("cg")
    reference_energy = energy_for_mode("schur")
    gradient = jacrev(cg_energy)(parameter)
    reference_gradient = jacrev(reference_energy)(parameter)
    assert_close(gradient, reference_gradient, tolerance)

    # Forward-over-reverse exercises both implicit AD rules; the dense
    # reverse-over-reverse result supplies the reference Hessian.
    hessian = jacfwd(jacrev(cg_energy))(parameter)
    reference_hessian = jacrev(jacrev(reference_energy))(parameter)
    assert_close(hessian, reference_hessian, tolerance)


@pytest.mark.parametrize("param", ["chi", "kcn", "eta", "rad"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_mixed_position_parameter(param: str, dtype: torch.dtype) -> None:
    """Force derivatives agree for all four model parameter families."""
    dd: DD = {"device": DEVICE, "dtype": dtype}
    structure = load_structure("SiH4", dd, 0.0)
    model = eeq.EEQModel.param2019(**dd)

    def energy_for_mode(mode: SolveMode) -> Callable[[Tensor, Tensor], Tensor]:
        def energy(positions: Tensor, value: Tensor) -> Tensor:
            displaced_structure = structure.replace(positions=positions)
            parameter_model = model.replace(**{param: value})
            _, atomic_energies = parameter_model(
                displaced_structure, return_energy=True, solve_mode=mode
            )
            return atomic_energies.sum()

        return energy

    args = (structure.positions, getattr(model, param))
    # Differentiate dE/dR (argument 0) with respect to the selected model
    # parameter (argument 1). For eta/rad this includes the dependence of
    # the Coulomb matrix on that parameter, as well as the charge response.
    mixed_derivative = jacrev(
        jacrev(energy_for_mode("cg"), argnums=0), argnums=1
    )(*args)
    reference_mixed_derivative = jacrev(
        jacrev(energy_for_mode("schur"), argnums=0), argnums=1
    )(*args)
    tolerance = 1e-4 if dtype == torch.float32 else 1e-9
    assert_close(mixed_derivative, reference_mixed_derivative, tolerance)


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


@pytest.mark.parametrize("differentiate", [jacrev, jacfwd])
@pytest.mark.parametrize("composition", ["vmap-derivative", "derivative-vmap"])
@pytest.mark.parametrize("param", ["chi", "kcn", "eta", "rad"])
@pytest.mark.parametrize("nonfinite", [False, True], ids=["raw", "nonfinite"])
def test_not_positive_definite_shared_derivatives(
    differentiate: Callable,
    composition: str,
    param: str,
    nonfinite: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Failed SiH4 solves preserve LiH derivatives through shared parameters."""
    batch = load_batch(["LiH", "SiH4"], DD_DOUBLE, [0.0, 0.0])
    lih = load_structure("LiH", DD_DOUBLE, 0.0)
    model = _model()
    eta = model.eta.clone()
    eta[14] = -10.0  # Only the silicon-containing system is indefinite.
    model = model.replace(eta=eta)

    if nonfinite:
        cg = storch.linsolve.cg

        def nonfinite_failed_iterates(
            *args: Any, **kwargs: Any
        ) -> storch.linsolve.LinearSolveResult:
            # A solver breakdown can leave nonfinite raw iterates. Inject
            # these before implicit_solve attaches its derivative rules;
            # successful solves and all original diagnostics are unchanged.
            result = cg(*args, **kwargs)
            converged = result.status == storch.linsolve.STATUS_CONVERGED
            return result._replace(
                solution=torch.where(
                    converged.unsqueeze(-1), result.solution, torch.nan
                )
            )

        monkeypatch.setattr(storch.linsolve, "cg", nonfinite_failed_iterates)

    def parameter_model(value: Tensor) -> eeq.EEQModel:
        # Entry zero is reserved for padding, not a physical element.
        # Differentiate the physical parameters while keeping it constant.
        full = torch.cat((getattr(model, param)[:1], value))
        return model.replace(**{param: full})

    def observable(
        value: Tensor, structure: Structure
    ) -> tuple[Tensor, Tensor]:
        charges, energies = parameter_model(value)(
            structure, solve_mode="cg", return_energy=True
        )
        # Sum only after EEQ has evaluated and masked its atomic energies.
        return charges, energies.sum(-1)

    def batched(value: Tensor) -> tuple[Tensor, Tensor]:
        return vmap(lambda structure: observable(value, structure))(batch)

    parameter = getattr(model, param)[1:]
    charges, energies = batched(parameter)
    assert torch.isfinite(charges[0]).all()
    assert torch.isfinite(energies[0])
    assert torch.isnan(charges[1]).all()
    assert torch.isnan(energies[1])

    with no_vmap_fallback():
        if composition == "vmap-derivative":
            # Each system differentiates the same parameter tensor.
            charge_jacobian, energy_gradient = vmap(
                differentiate(observable, argnums=0), in_dims=(None, 0)
            )(parameter, batch)
        else:
            # Reverse mode must also handle inactive failed-output
            # cotangents when differentiating the accepted system jointly.
            charge_jacobian, energy_gradient = differentiate(batched)(parameter)

    # This independent, unpadded LiH calculation uses the dense solver.
    def reference(value: Tensor) -> tuple[Tensor, Tensor]:
        q, e = parameter_model(value)(
            lih, solve_mode="schur", return_energy=True
        )
        return q, e.sum()

    reference_jacobian, reference_gradient = differentiate(reference)(parameter)
    nat = lih.numbers.shape[-1]
    assert_close(charge_jacobian[0, :nat], reference_jacobian)
    assert torch.equal(
        charge_jacobian[0, nat:], torch.zeros_like(charge_jacobian[0, nat:])
    )
    assert_close(energy_gradient[0], reference_gradient)
    # Only derivatives with respect to elements actually used in SiH4 are
    # active; unused parameter entries may correctly have structural zeros.
    assert torch.isnan(charge_jacobian[1, :, [0, 13]]).all()
    assert torch.isnan(energy_gradient[1, [0, 13]]).all()
