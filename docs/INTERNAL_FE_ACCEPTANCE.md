# Internal FE and dependency acceptance

| package | tests | result |
|---|---|---|
| fasttwfepy | 9 | pass |

Runtime dependency claims and limitations are in DEPENDENCIES.md and the README. Jackknife, RI, Replace, Subtract and MNW bootstrap definitions were preserved; Add remains TWFE-only.

FEABSORB-1/FEFIXTURE-1: 14 projection fixtures, 13 converge and one expected cap failure. Maximum converged dummy gap 1.3115107611039178e-10; cross-language gap 5.10702591327572e-15. Singletons retained; arbitrary K core; positive weights.

| package | procedure | before_seconds | after_seconds | speedup |
|---|---|---|---|---|
| fasttwfepy | setup | 0.026555 | 0.0627494 | 0.423191 |
| fasttwfepy | point | 0.0018345 | 0.0002679 | 6.8477 |
| fasttwfepy | jackknife | 0.057678 | 0.005001 | 11.5333 |
| fasttwfepy | ri999 | 0.128724 | 0.123616 | 1.04132 |
| fasttwfepy | replace999 | 4.0678 | 3.7394 | 1.08782 |
| fasttwfepy | subtract999 | 4.13131 | 3.83133 | 1.0783 |
| fasttwfepy | add999 | 4.98708 | 5.00866 | 0.995691 |

Performance acceptance includes small tradeoffs: TWFE setup +36ms, MNW Python STAR +11ms and R scalar fine +0.5ms. Repeated-assignment workflows remain acceptable. BJS severe near-collinear large fallbacks are intentionally rejected pending certification; the small raw oracle remains bounded.
