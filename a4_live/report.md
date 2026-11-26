# A4 live monitoring report (measured on local GPU)

| Condition | n | p95 s | mean s | incomplete % | blocked % |
|---|---|---|---|---|---|
| baseline | 31 | 53.1 | 47.7 | 77.4 | 80.6 |
| squeeze | 31 | 18.1 | 17.4 | 93.5 | 93.5 |
| contention | 31 | 170.0 | 152.1 | 77.4 | 80.6 |

| Condition | Alert | Fired |
|---|---|---|
| squeeze | abs_p95_>120s | no |
| squeeze | abs_incomplete_>10% | YES |
| squeeze | rel_p95_>1.5x_baseline | no |
| squeeze | rel_incomplete_+10pts | YES |
| contention | abs_p95_>120s | YES |
| contention | abs_incomplete_>10% | YES |
| contention | rel_p95_>1.5x_baseline | YES |
| contention | rel_incomplete_+10pts | no |