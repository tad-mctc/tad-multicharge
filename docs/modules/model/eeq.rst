EEQ solve modes
===============

``solve_mode="schur"`` remains the recommended default, using a dense
Cholesky factorization. ``"linear"`` solves the full bordered system.
``"cg"`` uses Jacobi-preconditioned conjugate gradient for the two auxiliary
Schur systems, with implicit derivatives. All three modes construct the
same dense Coulomb matrix and require quadratic memory. The CG mode does
not implement the matrix-free, projected-CG backend.

The keyword is available through ``model(...)``, ``model.solve(...)``,
``get_eeq(...)``, ``get_charges(...)`` and ``get_energy(...)``.

Numerical acceptance
--------------------

Auxiliary convergence alone does not certify the reconstructed solution:
for auxiliary residuals :math:`r_z=Az-b` and :math:`r_y=Ay-c`, stationarity
has residual :math:`Aq+c\mu-b=r_z-\mu r_y`. CG therefore checks the final
stationarity and charge residuals independently, including when only
charges are requested.

With :math:`\tau=\max(10^{-12},10\epsilon)` and :math:`a=10^{-14}`,
stationarity must satisfy

.. math::

   \|Aq+c\mu-b\|_2 \le a + \tau\|(b,Q)\|_2
       + 10\epsilon\left(\|\,|A|\,|q|\,\|_2 + \|c\mu\|_2\right).

The final term allows for rounding in evaluating the equation; the solver
tolerance itself does not increase with :math:`|\mu|`. Charge conservation
uses the separate rounding-scale bound (charges in electrons)

.. math::

   |c^Tq-Q| \le a + 10\epsilon\max\left(1,\sum_i|c_iq_i|+|Q|\right).

Failure of either auxiliary solve or either final check raises
``ConvergenceError`` in eager execution. Under ``vmap`` and compilation,
charges and energies of the affected systems become NaN. A converged
auxiliary solve can consequently still be rejected. Residual checks are
diagnostics and do not add derivatives to the implicit solve.

The public ``tad_mctc.storch.linsolve.mask_converged`` utility also invalidates
failed-system derivatives. This requires ``tad-mctc>=0.11.2``. Accepted systems
retain their usual charge derivatives under per-system ``vmap``/``jacrev``
and eager forward/reverse compositions.

Select accepted systems before nonlinear downstream operations on masked
charges: even first derivatives can be contaminated by ``0 * NaN`` if
nonlinear operations run on failed entries before selection. EEQ's own
auxiliary and KKT failure masks are deferred until after Schur
reconstruction and quadratic energy evaluation. Failed auxiliary iterates
are replaced with finite placeholders before implicit derivatives are
attached; their original
convergence status is retained. Both auxiliary solves and both KKT checks
must pass to accept a system. Charges and energies are masked separately
at the output boundary. This preserves valid-system charge and energy
derivatives with shared model parameters, including when another system's
auxiliary solve fails.

For higher derivatives involving shared parameters, select accepted outputs
before differentiating, or differentiate per system with ``vmap``.
Joint higher derivatives that
include failed outputs can propagate NaNs through ordinary ``0 * NaN``
arithmetic outside the mask.

Transforms and performance
--------------------------

Eager derivatives, including Hessians and third derivatives, are supported.
Compiled higher-order differentiation is unsupported. Compiled first-order
``jacrev`` requires PyTorch 2.11 or newer.
Use ordinary batched tensors for compiled first derivatives; compiling a
``vmap`` around the convergence mask is unsupported because Dynamo can
discard its custom derivative rules.

Under ``vmap`` and ``torch.compile``, the upstream CG loop runs its full
iteration budget, currently twice the atom count, even after individual
systems converge. Eager, vectorized and compiled performance should be
measured separately.

.. automodule:: tad_multicharge.model.eeq
   :members:
