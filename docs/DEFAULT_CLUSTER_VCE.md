# Native clustered inference

`Design.native_cluster(null=0, engine="fast")` exposes One-way state-clustered OLS CR1 sandwich. Multiplier G/(G-1)*(N-1)/(N-K), with K including identified FE and treatment/covariate parameters; Student t(G-1) p-values.

Returns native_cluster_method, native_cluster_se, native_cluster_stat, native_cluster_df, native_cluster_p, native_cluster_distribution and cluster_scores. Jackknife remains a separate method and its existing definition is unchanged. The transparent `engine="reference"` forms explicit dummy scores on small samples. This addition retains the existing Design sample and weighting scope; it does not expand weighted support.

Ten matched deterministic fixtures compare point estimates, SEs, statistics and p-values against Stata built-in regress, vce(cluster state). See native_cluster_external_results.csv.
