# Dense EEQ solver benchmark

Measured on 2026-10-24 against PR #67 head `6694a71` plus the review fixes in this worktree.
The default Cholesky–Schur solver remains recommended: these measurements do not establish a general dense-CG speedup or a large-system crossover.
CG was slower for all successfully measured workloads in this run.

## Reproduction and method

From the repository, using its Python environment:

```bash
PYTHONPATH=src python examples/benchmark-eeq.py --sizes 32 128 512 --repeats 3 > eager-vmap.csv
PYTHONPATH=src python examples/benchmark-eeq.py --sizes 8 --executions compile --repeats 3 > compile.csv
```

The first command exits with status 1 because three cases fail numerical acceptance; it still writes all rows, including the errors.
The compiled command completed successfully.
[Raw data](benchmark-eeq.csv) includes accuracy, conditioning, iteration counts, memory and error messages; [the script](benchmark-eeq.py) supports other sizes, dtypes, devices, thread counts and a per-row timeout.

Environment: AMD Ryzen 7 5700X, Linux x86-64, Python 3.12.10, PyTorch 2.14.1+cu130, clean `tad-mctc` 0.11.1 (`78f20d7`), CPU float64, one PyTorch/BLAS thread.
CUDA was unavailable.

The inputs are deterministic synthetic clusters of H/C/N/O on a perturbed grid with 3-bohr spacing and total charge +1.
The two conditioning regimes use the 2019 parameters and a +5 shift to chemical hardness.
These are numerical workloads, not optimized or representative molecular geometries.
Every evaluation includes coordination numbers, construction of the dense Coulomb matrix, and the requested solve or derivative.
The force workload computes the energy gradient with respect to positions; its timing also applies to forces, which differ by a sign.
Parameter gradients include `chi`, `kcn`, `eta` and `rad` in one model-pytree `jacrev`.

Times are medians of three warmed evaluations in milliseconds, excluding compilation and initial warmup.
`vmap` processes two structures (the base cluster and one scaled by 1.01); its times are per two-structure call.
Compiled timings use `torch.compile(fullgraph=True)` with the default Inductor backend.
Only first derivatives are compiled; higher compiled derivatives remain unsupported.
Eight atoms keeps the unrolled CG graph small; this is no evidence for large compiled systems.

Each row runs in a separate process.
Peak RSS includes Python, PyTorch, diagnostics, warmup, compilation when applicable, reference validation and the measured workloads; it is not isolated tensor memory or an incremental memory saving.
The large runtime baseline and CPU allocator reuse prevent inferring small memory differences between backends.
Rejected rows stop during warmup.
All solvers still require quadratic Coulomb storage.

CG updates are the upstream-reported `z/y` update counts for the unscaled cluster, measured separately in eager mode.
The fixed loop budget under `vmap` and compilation includes true-residual checks and is `2*N`: 64, 256 and 1024 trips for the molecular sizes here, and 16 trips for the compiled case.
Thus, for example, 16/15 auxiliary updates at 512 atoms still means a budget of 1024 loop trips under `vmap`, with additional implicit solves for derivatives.

## Eager

| Atoms | Hardness shift | cond(A) | Mode | Charges ms | Forces ms | Parameters ms | CG updates z/y | Peak RSS MiB |
| ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 32 | 0.0 | 37.14 | schur | 0.457 | 1.807 | 1.889 | — | 673.4 |
| 32 | 0.0 | 37.14 | cg | 3.487 | 23.940 | 24.347 | 17/17 | 673.9 |
| 32 | 5.0 | 2.17 | schur | 0.455 | 1.788 | 1.790 | — | 673.6 |
| 32 | 5.0 | 2.17 | cg | 2.268 | 21.483 | 21.487 | 9/9 | 673.8 |
| 128 | 0.0 | 89.75 | schur | 1.549 | 3.925 | 3.534 | — | 678.2 |
| 128 | 0.0 | 89.75 | cg | rejected | rejected | rejected | 41/40 | 666.7 |
| 128 | 5.0 | 4.05 | schur | 1.622 | 4.048 | 3.572 | — | 677.7 |
| 128 | 5.0 | 4.05 | cg | 4.079 | 78.835 | 76.371 | 13/12 | 678.3 |
| 512 | 0.0 | 301.86 | schur | 21.404 | 70.248 | 49.698 | — | 752.1 |
| 512 | 0.0 | 301.86 | cg | 32.357 | 422.509 | 378.364 | 60/60 | 748.3 |
| 512 | 5.0 | 8.96 | schur | 20.037 | 75.707 | 46.802 | — | 750.1 |
| 512 | 5.0 | 8.96 | cg | 21.470 | 395.013 | 371.988 | 16/15 | 750.2 |

## Vectorized, two structures

| Atoms | Hardness shift | cond(A) | Mode | Charges ms | Forces ms | Parameters ms | CG updates z/y | Peak RSS MiB |
| ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 32 | 0.0 | 37.14 | schur | 0.806 | 2.358 | 2.346 | — | 673.6 |
| 32 | 0.0 | 37.14 | cg | 17.849 | 45.869 | 45.862 | 17/17 | 673.7 |
| 32 | 5.0 | 2.17 | schur | 0.830 | 2.414 | 2.448 | — | 673.4 |
| 32 | 5.0 | 2.17 | cg | 17.591 | 45.810 | 45.468 | 9/9 | 673.9 |
| 128 | 0.0 | 89.75 | schur | 2.910 | 6.711 | 5.843 | — | 681.7 |
| 128 | 0.0 | 89.75 | cg | rejected | rejected | rejected | 41/40 | 669.6 |
| 128 | 5.0 | 4.05 | schur | 2.935 | 6.571 | 6.001 | — | 681.3 |
| 128 | 5.0 | 4.05 | cg | 72.690 | 181.605 | 180.901 | 13/12 | 681.4 |
| 512 | 0.0 | 301.86 | schur | 41.884 | 208.005 | 101.138 | — | 797.5 |
| 512 | 0.0 | 301.86 | cg | rejected | rejected | rejected | 60/60 | 699.7 |
| 512 | 5.0 | 8.96 | schur | 42.983 | 198.571 | 98.401 | — | 799.9 |
| 512 | 5.0 | 8.96 | cg | 515.044 | 1111.648 | 1038.138 | 16/15 | 798.8 |

## Compiled, single structure

| Atoms | Hardness shift | cond(A) | Mode | Charges ms | Forces ms | Parameters ms | CG updates z/y | Peak RSS MiB |
| ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 8 | 0.0 | 9.14 | schur | 0.114 | 0.198 | 0.203 | — | 824.4 |
| 8 | 0.0 | 9.14 | cg | 0.198 | 0.369 | 0.333 | 8/8 | 951.8 |
| 8 | 5.0 | 1.41 | schur | 0.120 | 0.225 | 0.222 | — | 823.5 |
| 8 | 5.0 | 1.41 | cg | 0.179 | 0.319 | 0.312 | 7/7 | 882.7 |

## Accuracy and rejection cases

Across accepted CG rows, maximum absolute differences from Cholesky–Schur were 6.93e-12 for charges, 1.48e-16 for forces, 2.66e-14 for parameter gradients.

Three cases were rejected rather than timed as successful solves:

- At 128 atoms with no hardness shift, eager CG's auxiliary systems converged in 41/40 updates, but the reconstructed stationarity residual was `1.572e-11`, above its `1.499e-11` target. The public path raises `ConvergenceError`.
- The corresponding 128-atom `vmap` case returns NaNs for an affected system.
- At 512 atoms without the hardness shift, the scaled member of the `vmap` call has stationarity residual `3.430e-11`, above its `2.960e-11` target, and becomes NaN. The unscaled member passes (`2.656e-11` against `2.954e-11`).

A separate diagnostic evaluation confirmed that both auxiliary RHS statuses are converged for both members of the 128- and 512-atom `vmap` calls. Their charge residuals are below `5e-15`; these failures are specifically the reconstructed stationarity checks.

The benchmark treats any non-finite charge or derivative as a failure and emits the failed row explicitly.
These rejections demonstrate why the final physical residual must be checked independently of auxiliary convergence; they are not evidence that dense CG always reaches the requested KKT accuracy.
No tolerance was relaxed to make these benchmark rows pass.

For these sizes and conditioning regimes, Cholesky–Schur remains the practical baseline.
Improved iteration-budget control may reduce transform overhead in a follow-up, but no shorter global budget is inferred from these few synthetic clusters.
The matrix-free nonperiodic operator, projected CG, balancing preconditioner and refinement remain a separate integration task; periodic Ewald operators require their own treatment.
