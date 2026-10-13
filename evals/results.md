# Eval results

Labels: 30 jobs (16 apply, 3 maybe, 11 skip). Bands: apply >= 50, maybe >= 50, else skip.

| Model | Jobs | Agreement | MAE (label steps) | Spearman | Recommended cutoff (apply) | P / R / F1 |
|---|---|---|---|---|---|---|
| claude-haiku-5-5 | 30 | 77% | 0.37 | 0.69 | 43 | 78% / 88% / 0.82 |
| claude-sonnet-5-5 | 30 | 73% | 0.43 | 0.68 | 43 | 87% / 81% / 0.84 |
| pipeline | 30 | 73% | 0.43 | 0.67 | 43 | 87% / 81% / 0.84 |

### Confusion: claude-haiku-5-5 (rows = your label, columns = model band)

| | skip | maybe | apply |
|---|---|---|---|
| **apply** | 3 | 0 | 13 |
| **maybe** | 1 | 0 | 2 |
| **skip** | 10 | 0 | 1 |

Apply vs skip: 13 of your apply jobs reach apply, 3 land in skip; 1 of your skip jobs reach apply.

### Confusion: claude-sonnet-5-5 (rows = your label, columns = model band)

| | skip | maybe | apply |
|---|---|---|---|
| **apply** | 4 | 0 | 12 |
| **maybe** | 2 | 0 | 1 |
| **skip** | 10 | 0 | 1 |

Apply vs skip: 12 of your apply jobs reach apply, 4 land in skip; 1 of your skip jobs reach apply.

### Confusion: pipeline (rows = your label, columns = model band)

| | skip | maybe | apply |
|---|---|---|---|
| **apply** | 4 | 0 | 12 |
| **maybe** | 2 | 0 | 1 |
| **skip** | 10 | 0 | 1 |

Apply vs skip: 12 of your apply jobs reach apply, 4 land in skip; 1 of your skip jobs reach apply.

## Tailoring cutoff

Current: tailor at fit >= 50. Based on your labels, the pipeline's scores separate your "apply" jobs best at **>= 43** (precision 87%, recall 81%, 15 of 30 jobs tailored).
Counting "maybe" as worth tailoring too, the best cutoff is >= 30 (precision 73%, recall 100%).
Small sample: treat this as a direction, and re-check after the next batch of labels.
