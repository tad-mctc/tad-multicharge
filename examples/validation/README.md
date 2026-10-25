# Matrix-free EEQ transform validation

Validated on October 25, 2026, against the merged PR #68 baseline
`1f382c1`, with `tad-mctc 0.11.2`. This work preserves the solver equations,
tolerances, refinement, memory budgets and failure masks.

## Implementation and regression coverage

The only executable production change handles an empty leading batch in
`_preconditioner`: it returns Jacobi before the coarse-memory division.
`Structure`, `schur` and `cg` already accept these batches. Matrix-free now
returns the same empty charge/energy shapes and zero shared-parameter
derivatives, without coarse-cell setup. Nonempty all-padding systems retain
their existing charge-constraint behavior.

The expanded tests cover:

- Shared position, total-charge, structure and model-pytree transforms for
  both charge-only and charge/atomic-energy return forms. Independent Python
  loops, finite differences and dense solves supply the references.
- Nested `vmap` over differently padded molecules, including output-axis
  ordering, zero padding, total charge and shared-parameter/charge derivatives.
- Whole-model `jacrev` and `jacfwd`: all four physical parameter leaves,
  static `MatrixFreeOptions`, the default CN subtree and tensor-valued CN
  parameters. Data-dependent cell construction is forbidden during these tests.
- Eight mixed AD/vectorization compositions over each physical parameter
  family, with explicit derivative-axis shapes and independent dense references.
- Direct differentiation of indefinite and iteration-limited systems, mixed
  accepted/rejected systems with shared parameters, injected nonfinite iterates
  with unchanged diagnostics, and analytically zero higher derivatives.
- Finite reconstruction/energy intermediates followed by separate final masks.
  Arbitrary nonlinear operations on already masked NaNs remain outside this
  guarantee: select accepted systems before applying them.
- CUDA float32/float64 native `.backward()` for positions and all four model
  parameters, plus `jacrev`, `jacfwd` and `vmap` with the default `fused=True`.
  Instrumentation verifies fused native products and eager derivative tiles.
- Dense fullgraph compilation with nondefault matrix-free context, compiled
  first-order reverse derivatives and graph reuse for equal-shaped parameter
  replacements. The explicit matrix-free compilation rejection remains tested.

No existing numerical expectations were weakened. Float32 CUDA checks use
scoped tolerances: the existing native-backward envelope is `2e-4`, and the
new transformed charge/energy checks use `2e-5`. Float64 comparisons use
`1e-8` or tighter. The shared finite-difference checks retain their original
tolerances; there is no global matrix-free exception.

## Verified compatibility

**V** means verified by the suite in the environments below. **U** means
known unsupported behavior, with a retained rejection or compatibility guard.
**E** means unavailable environment. There are no unresolved test failures.
These entries describe eager transforms unless compilation is named explicitly.

| Capability | `schur` | `cg` | `matrix-free` |
| --- | --- | --- | --- |
| Eager `vmap` | V | V | V |
| Nested `vmap` | V | V | V |
| Eager `jacrev` | V | V | V |
| Eager `jacfwd` | V | V | V |
| Hessians | V | V | V |
| Third derivatives | V | V | V |
| `vmap(jacrev)` | V | V | V |
| `jacrev(vmap)` | V | V | V |
| `vmap(jacfwd)` | V | V | V |
| `jacfwd(vmap)` | V | V | V |
| Full-model pytree differentiation | V | V | V |
| Native CUDA backward | V on 2.14.1 | V on 2.14.1 | V on 2.14.1, fused |
| `torch.compile` forward, fullgraph | V | V | U: explicit rejection |
| Compiled first-order `jacrev` | V | V on 2.14.1; U before 2.11 | U |

Eager CPU coverage includes PyTorch 2.7.1 and 2.14.1. CUDA coverage uses
2.14.1; CUDA with 2.7.1 is **E** because that installed environment is CPU-only.
Python 3.14/PyTorch 2.9.1 is **E**; its existing upstream compile guard is
preserved. No claim is made for every intervening PyTorch release.

Compiled `vmap` differentiation through the shared failure mask and compiled
higher derivatives remain unsupported. On PyTorch 2.7, calling a model passed
as a graph argument can guard its identity and recompile; the existing test
checks a single graph via `model.solve` on all versions, and via `model(...)`
only from 2.8 onward. Replacing physical parameter values at fixed shapes
still reuses the dense `solve` graph with nondefault matrix-free options.

## Reproducing the scalability investigation

The opt-in harness is independent of pytest and adds no dependencies:

```sh
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src python \
  examples/validation/matrix_free_transforms.py \
  --device cuda --sizes 99 300 999 --repeats 1 --timeout 150 \
  --output /tmp/eeq-transform-cuda.jsonl

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src python \
  examples/validation/matrix_free_transforms.py \
  --device cuda --sizes 3000 --modes eager native native-jacobi jvp \
  --repeats 1 --timeout 150 --output /tmp/eeq-transform-cuda-large.jsonl

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src python \
  examples/validation/matrix_free_transforms.py \
  --device cpu --sizes 99 300 --repeats 1 --timeout 120 \
  --output /tmp/eeq-transform-cpu.jsonl
```

The clusters contain deterministic water geometries at 5.86-bohr molecular
spacing and total charge -1. A prebuilt neighbor list also bounds CN work.
All cases use float64 and the scalar objective
`sum(atomic_energies) + 0.01 * sum(charges**2)`. The charge term requires a
nonzero implicit adjoint independently of variational energy cancellation.
`native` measures a position gradient with `.backward()`; `jacrev` measures
the same gradient. `jvp` and `jacfwd` measure one normalized position direction,
with `jacfwd` differentiating a scalar displacement, not a full position
Jacobian. `vmap` evaluates two nearby geometries. `mixed` differentiates a
position-gradient projection in the hydrogen-hardness direction, through
300 atoms only. No large full Jacobian or Hessian is constructed.

Each case has its own subprocess and deadline. CUDA initialization and
neighbor construction precede timing. The first evaluation, including any
kernel compilation, is recorded separately as `cold_seconds`. Dense Schur
references are outside the timed region. Reported times below are **one warm
sample per case**, with diagnostic instrumentation; they establish the scale
of overhead, not a statistical performance guarantee. The harness defaults
to two warm repeats for subsequent investigations.

Instrumentation delegates to the original solver, recording selected
preconditioners, fast/exact operator calls, pair-product paths, per-solve
status/iterations and final residuals. Diagnostic copies can synchronize the
GPU once per inner solve. CUDA memory is incremental PyTorch allocator peak
above already-live inputs, neighbor lists and reference tensors; it excludes
driver/compiler memory and is not process-wide peak memory. CPU allocator
peak is not measured (`null` in the raw data).

Hardware: AMD Ryzen 7 5700X, one CPU thread; NVIDIA GeForce RTX 4070 (12 GiB),
driver 580.178.04. Benchmark runtime: Python 3.12.10, PyTorch 2.14.1+cu130,
CUDA 13.0, `tad-mctc 0.11.2`. Mixed precision and exact fallback are enabled
throughout. All settings are default except the explicitly labeled
`native-jacobi` control (`coarse_memory=0`).

Raw CUDA snapshots were taken before the final fusion-docstring clarification;
CPU snapshots include that documentation edit. Each row records its source
SHA256. The executable solver is identical between those snapshots.
The complete records are in [results/cuda.jsonl](results/cuda.jsonl) and
[results/cpu.jsonl](results/cpu.jsonl).

## CUDA findings

Warm times in seconds:

| Atoms | Eager | Native backward | Native Jacobi control | `jacrev` | `jvp` | Directional `jacfwd` | `vmap` (two systems) | Mixed AD |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 99 | 0.127 | 0.233 | 0.180 | 3.183 | 0.256 | 3.261 | 3.131 | 4.224 |
| 300 | 0.140 | 0.271 | 0.228 | 9.554 | 0.341 | 9.596 | 9.307 | 12.207 |
| 999 | 0.166 | 0.324 | 0.333 | 31.079 | 0.438 | 32.665 | 32.214 | Not run |
| 3,000 | 0.207 | 0.465 | 0.538 | Not run | 0.670 | Not run | Not run | Not run |

All 27 measured CUDA cases converged with finite observables/derivatives and
agreed with independent dense references. Maximum absolute discrepancy was
`4.98e-14` for the scalar observable and `2.13e-15` for derivatives. Public
solver status and every recorded inner status were converged. No exact
fallback iteration was needed. Larger mixed derivatives and batched-basis
transforms at 3,000 atoms were deliberately not run; there is no claim about
their runtime or peak memory.

The Jacobi fallback increases actual Krylov iteration counts, but does not
by itself explain the large transform slowdown:

| Atoms | Primal iterations, balancing | Primal iterations, Jacobi control | All operator calls, native balancing | All operator calls, native Jacobi | All operator calls, `jacrev` |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 99 | 44 | 47 | 104 | 110 | 1,478 |
| 300 | 51 | 61 | 119 | 134 | 4,306 |
| 999 | 58 | 87 | 130 | 183 | 14,118 |
| 3,000 | 59 | 109 | 134 | 226 | Not run |

The counts include fast and exact products for both primal and derivative
solves; primal iterations sum the mixed-precision correction solves. At
3,000 atoms Jacobi nearly doubles primal iterations, but native runtime rises
only from 0.465 to 0.538 s. At 999 atoms `jacrev` instead takes about 96 times
native backward despite converging its individual inner solves within 33
iterations. It makes 12,105 fast and 2,013 exact operator calls.

Inspection of upstream [`_exit_early`](https://github.com/tad-mctc/tad-mctc/blob/v0.11.2/src/tad_mctc/storch/linsolve/krylov.py#L156)
and [`_refine`](https://github.com/tad-mctc/tad-mctc/blob/v0.11.2/src/tad_mctc/storch/linsolve/implicit.py#L300)
explains this difference: a mapped active mask disables
data-dependent early exit. Batched derivative/refinement work can run six
full correction budgets plus an inactive exact fallback. Here the Krylov
budget is twice the bordered vector size, `2*(N+1)`. Diagnostic iteration
counts stop tracking converged systems, but operator calls continue. Thus
reported iteration counts alone substantially understate transformed work.
A plain directional `jvp` keeps early exit in these cases and remains much
closer to native runtime. No tolerance or iteration limit was changed.

At 999 atoms, incremental CUDA peaks were 28.18 MiB eager, 121.54 MiB native
backward, 136.12 MiB `jacrev`, 138.89 MiB directional `jacfwd`, 153.47 MiB
`jvp` and 23.01 MiB for two-system `vmap`. At 3,000 atoms they were 44.91 MiB
eager, 185.16 MiB native and 700.25 MiB `jvp`. These numbers include the
complete evaluated observable and its CN/AD work, not just the linear solve.
Higher derivative graphs may retain quadratic memory; the existing tiled
first-derivative saved-tensor regression remains in place.

`fused=True` is compatible with the tested transforms. It does not mean
every transformed product uses eager tiles: custom Functions may unwrap
their primal inputs or expose an ordinary batch. The 999-atom `jacrev`
measurement recorded 14,119 fused products and two eager derivative tiles.
Kernel-path selection cannot explain its large runtime by itself.

## CPU findings

All 16 CPU measurements converged, with maximum absolute discrepancy
`3.11e-15` against the dense reference. The same fixed-budget overhead is
visible without CUDA fusion:

| Atoms | Eager | Native backward | Native Jacobi control | `jacrev` | `jvp` | Directional `jacfwd` | `vmap` (two systems) | Mixed AD |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 99 | 0.050 | 0.101 | 0.085 | 1.937 | 0.132 | 1.971 | 2.053 | 2.681 |
| 300 | 0.130 | 0.254 | 0.263 | 10.541 | 0.356 | 12.007 | 20.093 | 13.565 |

Times are seconds with one CPU thread; CPU peak memory is unmeasured.
These ordered water clusters establish a practical execution-cost limitation,
not a general convergence guarantee for every molecular geometry or model.

## Validation commands and results

All pytest commands use `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1` from the
repository root. No new skip or xfail hides a failing eager composition.

| Environment / command | Result |
| --- | --- |
| 2.14.1 CPU: focused matrix-free charge/grad, shared transforms and compile | 417 passed, 12 CUDA-only skipped; 137.38 s |
| 2.14.1 CPU: full `pytest -q --tb=line --cov=src/tad_multicharge --cov-branch` | 814 passed, 16 CUDA-only skipped; 288.42 s; 99.54% coverage |
| 2.7.1 CPU / Python 3.10.17: full `pytest -q --tb=line` | 811 passed, 19 skipped; 136.58 s |
| 2.14.1 CUDA: full `pytest --cuda --slow -q --tb=line` | 830 passed, 0 skipped; 479.11 s |
| Black, isort, mypy, pyupgrade, `git diff --check` | Passed; mypy checks all 9 source files |

The 2.7 skips comprise 16 CUDA-only tests and three guarded compiled-CG
reverse tests (two option configurations and the existing failure-mask
regression). There are no expected failures in these local runs. PyTorch
2.7 compiled Schur derivatives and dense forward compilation were executed.
The matrix-free module has 100% statement and branch coverage (93 statements,
22 branches) in the current CPU run.

## Separate upstream recommendations

1. Prioritize transform-compatible termination or suppression of inactive
   Krylov/refinement/fallback products in `tad-mctc`. Preserve independent
   lane convergence and custom AD. Benchmark operator counts as well as
   recorded iterations; lowering downstream iteration budgets would hide
   the mechanism and can reduce robustness.
2. Investigate reusable detached balancing preconditioners for a fixed
   geometry and operator, with explicit invalidation when either changes.
   This could reduce the increasing Jacobi iteration cost, but is a separate
   architecture change and would not alone remove fixed-budget overhead.
3. Keep whole-model matrix-free compilation as a coordinated upstream
   pairwise/implicit-solver/masking work package. Compiled `vmap` failure-mask
   differentiation and compiled higher derivatives remain separate limitations.

No upstream source change, new dependency or new release is required for
this compatibility-hardening package.
