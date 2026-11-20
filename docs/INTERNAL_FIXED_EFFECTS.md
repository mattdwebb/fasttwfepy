# Internal additive fixed effects: FEABSORB-1

The numerical core computes weighted projection residuals for one or any number of additive categorical fixed effects, for a vector or multiple right-hand sides. One-way FE is group demeaning. Multiway FE uses cyclic alternating weighted group demeaning. Compiling caches integer codes, counts, weight sums, stable sorted boundaries and singleton counts; Python additionally records two-way connected components and exact rank. No observation-by-FE dummy matrix is constructed in the main algorithm.

Weights are strictly positive, finite numerical least-squares weights, normalized by a common factor. Zero/negative/nonfinite weights and zero group totals are rejected. This is the projection metric, not an inferential analytic/probability/frequency-weight convention. Estimator APIs do not acquire new user-facing weight semantics from the core.

After each sweep, the maximum absolute weighted group mean divided by each original input column's maximum absolute value must be <=1e-12 (zero columns use scale one). The default cap is 10,000 sweeps. Diagnostics report iterations, convergence, final metric, tolerance and cap. Nonconvergence raises an explicit error by default. Singleton observations are retained and their groups reported; inferential pruning belongs to the estimator. Disconnected components have well-defined projections without arbitrary FE coefficient restrictions. BJS prediction cannot join disconnected training components.

Python/R user layers accept integer, nonconsecutive and string labels; Mata takes compiled numeric labels, and the Stata wrapper maps strings with built-in commands. Reordering observations/labels/FE dimensions is checked to convergence tolerance. Optional FE coefficient recovery exists only where BJS predictions need it. The experimental standard-library backend also contains a pivoted small dense solver; it is not a replacement for the full NumPy-based estimators.

Fourteen FEFIXTURE-1 designs test one/two/three dimensions, weighting, multiple columns, singletons, disconnected graphs, near-collinearity and difficult graphs against explicit dummy least squares, orthogonality, idempotence and invariance. Thirteen converge; the weak bridge deliberately fails at cap 100. The chain needs 2,252 cyclic iterations. Symmetric sweeps passed the same projection checks but did more projections without useful acceleration, so no accelerator is publicly dispatched.

Unsupported: heterogeneous FE slopes, group-level-outcome individual FE, IV/GMM, Poisson, arbitrary multiway cluster inference, detailed FE coefficient reporting, full reghdfe degrees-of-freedom heuristics and every Stata factor-variable corner case. The MNW estimator supports one or two FE dimensions with exact nuisance rank; arbitrary K is supported by the projection core only.

## Package integration

State FE are demeaned internally once at setup. Cached state-period cross-products and a small time/covariate Schur system preserve fast repeated RI/placebo assignments. Rank-sensitive fallback uses the internal full FE projection. Direct explicit-dummy OLS remains a small-sample validation oracle. The setup primitive supports the established state/time model, ordinary covariates and no covariates; all existing inference procedures are unchanged.
