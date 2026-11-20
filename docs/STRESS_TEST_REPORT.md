# Stress-test report

{
  "package": "fasttwfepy",
  "candidate": false,
  "counts": {
    "pass": 23,
    "expected_failure": 5,
    "not_applicable": 2
  },
  "maxima": {
    "point_gap": 7.508194066474516e-11,
    "component_gap": 0.0,
    "weight_gap": 0.0,
    "delete_gap": 1.8834600545858393e-10,
    "jk_gap": 2.093789142065816e-09,
    "ri_gap": 1.111788439089878e-10,
    "placebo_gap": 5.793332480408253e-10
  }
}

See stress_test_results.csv for each design and stress_test_draws.json for fixed-assignment statistics and inner p-values. Missing cells are an explicit inference-adapter restriction; unsupported weights are N/A. Singleton-cohort deletion failures retain G and all deletions. Heterogeneous effects are diagnostic, not a validity proof. Reference point checks gate inference.
