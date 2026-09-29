**Blind Craftsman** (threshold: 3.5 reward, 90% of the way from the initial to the best final reward)

| Method | Final reward | Env. steps to threshold | Runs reaching it | Fewer steps than No Transfer |
|---|:---:|:---:|:---:|:---:|
| No Transfer | -20.1 ± 1.4 | 1547k ± 6k | 0/20 |  |
| Policy Distillation | -2.6 ± 9.4 | 1246k ± 73k | 17/20 | +19% (p=0.00038) |
| Kickstarting | 15.5 ± 0.4 | 822k ± 39k | 20/20 | +47% (p=3.4e-08) |
| Automaton Distillation | 17.5 ± 0.1 | 489k ± 19k | 20/20 | +68% (p=3.4e-08) |
| CADENT | 16.8 ± 0.1 | 429k ± 9k | 20/20 | +72% (p=3.4e-08) |
