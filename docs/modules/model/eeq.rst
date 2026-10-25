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

Matrix-free molecular backend
-----------------------------

``solve_mode="matrix-free"`` applies the same Gaussian Coulomb operator in
row tiles, including the diagonal hardness, and solves the constrained
KKT equations with projected CG. It bypasses the dense Coulomb matrix.
The total molecular charge is an explicit constraint, and positions,
radii and hardnesses are explicit differentiable operator arguments.

.. code-block:: python

   from tad_multicharge import eeq

   model = eeq.EEQModel.param2019(dtype=structure.positions.dtype,
                                device=structure.positions.device)
   model = model.replace(matrix_free=eeq.MatrixFreeOptions(
       coarse_memory=64 * 1024 * 1024,
       mixed_precision=True,
   ))
   charges, atomic_energies = model(
       structure, solve_mode="matrix-free", return_energy=True,
   )

``MatrixFreeOptions`` is immutable model context, so the physical parameter
tree retains its usual ``vmap``, ``jacrev`` and ``jacfwd`` behavior. Solver
choice also works through the convenience functions. Molecular and padded
batches are supported, including empty leading batch dimensions. Empty
batches return empty charges and atomic energies with the usual shapes
and zero derivatives for shared parameters; they skip coarse-cell setup.
A nonempty all-padding system remains distinct and must satisfy the total
charge constraint. Periodic inputs are rejected explicitly: their
Ewald operator needs a separate matrix-free implementation. The dense
periodic modes and the default ``"schur"`` are unchanged.

Memory and precision
~~~~~~~~~~~~~~~~~~~~

The default coarse-space budget is 64 MiB for the whole batch. Before
constructing the balancing preconditioner, the number of cells is bounded
by this budget, the atom count and the larger of the product and factorization
dtypes. The bound reserves
eight ``N*K`` buffers for cell-product assembly and coarse workspaces,
with ``K <= N``. Cells enlarge until they fit. This budget excludes pair
tiles, inputs, outputs and autograd graphs; it is not a process-wide memory
limit. Zero disables balancing and uses Jacobi. Geometry-dependent setup
is rebuilt per call, avoiding stale preconditioners after geometry or
parameter changes. Under ``torch.func`` transforms, Jacobi avoids the
data-dependent cell construction. A failed coarse Cholesky factorization
also falls back to positive Jacobi; the physical convergence checks still
apply.

With float64 inputs, mixed precision uses float32 pair products for the
inner iterations and refines against the float64 operator at the same
``1e-12`` relative and ``1e-14`` absolute tolerances as dense CG. This also
applies to the implicit derivative solves. An unsuccessful refinement
retries with the exact operator by default; ``fallback=False`` disables
that retry. Float32 inputs keep their original precision and tad-mctc's
roundoff tolerance floor. ``maxiter`` bounds each projected solve.

The solver checks the true residual of the bordered system and independently
checks charge conservation. Eager failure raises ``ConvergenceError``;
under transforms, rejected charges, energies and their active derivatives
are NaN. Failure masking happens after energy evaluation, and nonfinite
failed iterates cannot contaminate accepted systems through shared
parameter pullbacks. The nonlinear downstream-operation and joint
higher-derivative limitations described above still apply.

First derivatives recompute pair tiles, retaining bounded pair memory.
Higher derivatives, including force-parameter derivatives and Hessians,
may retain all tile graphs and require quadratic memory. Supply a neighbor
list through ``model(structure, pairs=...)`` when coordination-number
construction must also use sparse pair enumeration.

CUDA products use fused tiles by default where supported. CPU execution
and transformed derivative products use eager tiles. A custom autograd
Function can unwrap transformed inputs and still fuse its CUDA primal;
``fused=True`` does not require every product to compile. The full matrix-free
model does not support ``torch.compile``; internal fusion is independent
of compiling the model. Setting ``fused=False`` disables internal fusion.

Nested eager ``vmap`` and forward/reverse derivatives also support shared
model parameters. Differentiating the complete model preserves the static
``matrix_free`` configuration and the coordination-number pytree. These
checks use the existing ``tad-mctc`` Node implementation.

Jacobi avoids data-dependent coarse setup under transforms, but is not the
only source of transform overhead. Upstream batched Krylov and refinement
loops can perform their full masked iteration budgets after convergence,
including inactive exact fallbacks. In water-cluster measurements through
999 atoms this preserved accuracy but made ``jacrev``, directional
``jacfwd`` and ``vmap`` substantially slower than native backward. A single
``torch.func.jvp`` retained early exit and was much closer to native runtime.
The opt-in harness ``examples/validation/matrix_free_transforms.py`` and its
accompanying report record timings, operator counts, memory, environments
and convergence diagnostics. No iteration budget is relaxed for transforms.
