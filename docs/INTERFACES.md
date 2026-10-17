# Interfaces and data policy

Changes to this file need approval from both A and B.

## Data policy
Only measured signals (HR, accelerometer) may enter training or evaluation. Synthetic
signals live under `synthetic_demo/`, are for demos and UI tests only, and must be labelled
synthetic wherever shown. Splits are by participant, never by epoch.

## Raw layout
`data/raw/<subject>/<night>/{hr.csv, motion.csv, labels.mat}` (gitignored).

## Time and epoch rules
- `labels.mat`: `expert_label` (stage per epoch), `recStart` (text, local time).
  `recStart` is interpreted as America/New_York and converted to unix seconds.
- Epoch index is zero-based: `floor((t - rec_start) / 30)`. Readings outside
  `[0, n_epochs)` are dropped from the epoch table.
- Loader sanity check: first HR timestamp must be within a few minutes of `rec_start`.

## Label coding
BIDSleep: 0=Wake, 1=N1, 2=N2, 3=N3, 4=REM.
Internal binary: 0=wake, 1=sleep. Internal 3-class: 0=wake, 1=NREM, 2=REM.

## Contracts
| Contract | Owner | Consumers | Shape |
|---|---|---|---|
| `load_night(dir) -> NightData` | A | all | see `noctiband/types.py` |
| Epoch table | A | all | one row per epoch: `subject, night, epoch, label, <measured features>` |
| `predict_night(dir) -> NightPrediction` | A | B (app) | `epoch_labels, probs, rec_start, coverage` |
| `make_report(NightPrediction) -> dict` | B | A (app glue) | TST, sleep efficiency, WASO, latency, hypnogram figure |
| `configs/split.json` | A | all | train/val/test subject lists, fixed seed |

## Ownership
A: `io`, `features`, `models`, `evaluation`, `inference`. B: `reporting`, `app/`, `tests/`, `docs/`.