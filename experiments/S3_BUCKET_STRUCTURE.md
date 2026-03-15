# S3 Bucket: `modelsfinetuned`

All experiment data, model checkpoints, and evaluation results are stored in a
single S3 bucket called **`modelsfinetuned`**.

**Current totals:** 3,290 objects, ~10.25 TB

## Top-Level Prefixes

| Prefix | Objects | Size | Status | Used by |
|--------|---------|------|--------|---------|
| `experiments/` | 1,305 | 1,245 GB | **Active (Winogender)** | `experiments/winogender/s3_utils.py`, `experiments/s3_utils.py` |
| `gpt2-xl-finetuned/` | 658 | 4,337 GB | **Active (StereoSet checkpoints)** | `experiments/stereoset/stereoset_finetuning.py`, `experiments/fine_tuned_test.py` |
| `gpt2-xl-finetuned-winogender/` | 705 | 4,654 GB | **OLD / pre-migration duplicate** | `experiments/winogender/migrate_s3.py` (source) |
| `stereoset_experiments/` | 622 | 14 GB | **Active (StereoSet)** | `experiments/stereoset/s3_utils.py` |

---

## 1. `experiments/` (Winogender)

Accessed via `experiments/winogender/s3_utils.py` with `S3_PREFIX = "experiments"`.
All paths in code omit the prefix (e.g. `s3_utils.read_csv("outputs/gpt2-xl/winogender/baseline/test/pronoun_probs.csv")`
maps to the S3 key `experiments/outputs/gpt2-xl/winogender/baseline/test/pronoun_probs.csv`).

```
experiments/
├── data/winogender/
│   ├── winogender_dataset.json
│   ├── winogender_metadata.json
│   ├── winogender_paired_dataset.json
│   ├── winogender_paired_metadata.json
│   ├── winogender_test_dataset.json
│   ├── winogender_test_metadata.json
│   ├── fine-tune-dpo/winogender_dpo.jsonl
│   └── fine-tune-sft/winogender_sft.jsonl
│
├── outputs/gpt2-xl/winogender/
│   │
│   │   # ---- OLD flat baseline files (pre-migration, 4 files) ----
│   ├── pronoun_probs.csv                      # OLD: should be in baseline/train/
│   ├── suffix_probs.csv                       # OLD: should be in baseline/train/
│   ├── accumulated_impact_winogender.csv      # OLD: should be baseline/train/accumulated_impact.csv
│   ├── out_DLA_winogender.csv                 # OLD: should be baseline/train/
│   │
│   │   # ---- Clean structure (post-migration) ----
│   ├── baseline/
│   │   ├── train/
│   │   │   ├── pronoun_probs.csv
│   │   │   ├── suffix_probs.csv
│   │   │   ├── accumulated_impact.csv
│   │   │   └── out_DLA_winogender.csv
│   │   └── test/
│   │       ├── pronoun_probs.csv
│   │       ├── suffix_probs.csv
│   │       └── accumulated_impact.csv
│   │
│   ├── fine_tuned/
│   │   ├── logs/
│   │   │   ├── all_experiment_results.json
│   │   │   └── {run_id}.json                  # one per run (152 runs)
│   │   ├── checkpoints/
│   │   │   └── best_model_{run_id}_epoch_{N}.pt  # ~6.6 GB each, 188 files
│   │   ├── train/{run_id}/                    # (not populated yet)
│   │   └── test/{run_id}/                     # 152 run dirs, 3 files each
│   │       ├── pronoun_probs.csv
│   │       ├── suffix_probs.csv
│   │       └── accumulated_impact.csv
│   │
│   │   # ---- OLD finetuned/ eval results (pre-migration, 456 files) ----
│   ├── finetuned/
│   │   └── {run_id}[_test]/
│   │       ├── pronoun_probs.csv
│   │       ├── suffix_probs.csv
│   │       └── accumulated_impact_winogender.csv  # OLD name
│   │
│   └── plots/                                 # 34 PDF plots from analysis notebooks
│       └── *.pdf
```

## 2. `stereoset_experiments/` (StereoSet)

Accessed via `experiments/stereoset/s3_utils.py` with `S3_PREFIX = "stereoset_experiments"`.

```
stereoset_experiments/
├── data/
│   ├── bias_attribute_words.json
│   ├── crows/crows_pairs_anonymized.csv
│   ├── seat/*.jsonl                           # SEAT benchmark data
│   └── stereoset/
│       ├── splits/
│       │   ├── gender_train.json
│       │   └── gender_test.json
│       ├── test.json
│       └── dev.json
│
└── outputs/gpt2-xl/
    ├── dev_tests/
    │   ├── accumulated_impact_gender_train.csv
    │   └── out_DLA_gender_train.csv
    └── fine_tuned/
        ├── logs/
        │   ├── all_experiment_results.json
        │   └── {run_id}.json
        └── results/{run_id}/
            └── out_DLA_gender_test.csv
```

## 3. `gpt2-xl-finetuned/` (StereoSet Checkpoints)

Accessed directly by `experiments/stereoset/stereoset_finetuning.py` and
`experiments/fine_tuned_test.py` with `s3_prefix = "gpt2-xl-finetuned"`.

```
gpt2-xl-finetuned/
└── best_model_{run_id}_epoch_{N}.pt           # 658 files, ~6.6 GB each
```

Run IDs follow the pattern: `{loss}_{component}_{percentile}_{hyperparam}_{lr}`,
e.g. `best_model_dpo_attn_0.5_beta0.3_lr1e-05_epoch_2.pt`.

## 4. `gpt2-xl-finetuned-winogender/` (OLD Winogender Checkpoints)

**This is the pre-migration location.** The `migrate_s3.py` script was supposed
to copy these into `experiments/outputs/gpt2-xl/winogender/fine_tuned/checkpoints/`
and then optionally delete the originals.

```
gpt2-xl-finetuned-winogender/
└── best_model_{run_id}_epoch_{N}.pt           # 705 files, ~6.6 GB each
```

---

## Migration Status

The migration (`experiments/winogender/migrate_s3.py`) partially completed.
Files were **copied** to the new layout but **old files were not deleted**:

| Old location | New location | Status |
|-------------|-------------|--------|
| `experiments/outputs/gpt2-xl/winogender/*.csv` (4 flat files) | `experiments/.../baseline/train/` and `baseline/test/` | Copied to new, **old still present** |
| `experiments/.../finetuned/{run_id}[_test]/` (456 files) | `experiments/.../fine_tuned/train\|test/{run_id}/` | Copied to new, **old still present** |
| `gpt2-xl-finetuned-winogender/` (705 checkpoints) | `experiments/.../fine_tuned/checkpoints/` (188 files) | **Partially copied** (188 of 705) |

### Cleanup actions

To complete the migration and reclaim ~4.6 TB:

1. **Finish copying remaining checkpoints** — re-run `python -m experiments.winogender.migrate_s3`
   to copy any checkpoints not yet in the new location.
2. **Delete old duplicates** — re-run with `--delete-old` to remove the pre-migration copies:
   ```bash
   python -m experiments.winogender.migrate_s3 --delete-old
   ```
3. **Verify** nothing references the old paths before deleting. The codebase already
   uses the new paths everywhere except `winogender_analysis.ipynb` which still
   references `finetuned/` (old layout) for historical train-split analysis.

---

## Code Path Reference

| s3_utils module | S3_PREFIX | Used for |
|----------------|-----------|----------|
| `experiments/winogender/s3_utils.py` | `experiments` | Winogender data, outputs, plots |
| `experiments/s3_utils.py` | `experiments` | Same prefix (shared utilities) |
| `experiments/stereoset/s3_utils.py` | `stereoset_experiments` | StereoSet data and outputs |
| Direct boto3 calls | `gpt2-xl-finetuned` | StereoSet checkpoint download |
| Direct boto3 calls | `gpt2-xl-finetuned-winogender` | OLD Winogender checkpoint download |
