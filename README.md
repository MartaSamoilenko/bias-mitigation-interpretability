# Diploma: bias mitigation & evaluation for language models

This repository contains experiments for **stereotype / gender bias** (StereoSet) and **coreference gender bias** (Winogender) using [Transformer Lens](https://github.com/TransformerLensOrg/TransformerLens) (`HookedTransformer`), **DLA-based layer selection**, **DPO / SFT fine-tuning**, and downstream checks with **[lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)** (vendored under `lm-evaluation-harness/`). There is a separate **Spectrum** tool (`spectrum/`) that scores layers by **SNR** to choose which parameters to treat as "important" for training configs.

**Important:** most pipelines assume **GPU**, **Hugging Face** access, **AWS S3**, and credentials in the environment. Without S3 and the same key layout, scripts that read datasets or upload checkpoints will not run as-is.

## Requirements

- **Python** 3.10+ (matches `lm-evaluation-harness` `pyproject.toml`).
- **CUDA** recommended for any training or long evaluations.
- Accounts / keys (as used in code):
  - `HF_TOKEN` — Hugging Face (e.g. Llama, gated models).
  - `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` — S3.
  - `OPENAI_API_KEY` — only for OpenAI-based data scripts (paraphrase / test set generation).

Optional: `.env` in the project root; code uses `python-dotenv` `load_dotenv()` in several places.

## Setup

```bash
cd /path/to/diploma
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -U pip
pip install -r requirements.txt
pip install -e ./lm-evaluation-harness
```

- Root `requirements.txt` pins **PyTorch 2.8** and **Transformers 4.56.2**; use a **CUDA** build of `torch` that matches your system if you install from scratch.

`spectrum/` has its own small `spectrum/requirements.txt` if you run that tool in isolation.

## Project layout (high level)

| Path | Role |
|------|------|
| `experiments/s3_utils.py` | S3 read/write helpers (`experiments/…` key prefix, bucket `modelsfinetuned`). |
| `experiments/winogender/` | Winogender: data prep, DLA / bias search, fine-tuning, test eval, perplexity, `run_lm_harness_tests.py`. |
| `experiments/stereoset/` | StereoSet: paraphrase / dataset prep, DLA, fine-tuning, batch test, `run_lm_harness_tests.py`. |
| `spectrum/spectrum.py` | SNR over layer weights → JSON / YAML lists of "unfrozen" layer patterns. |
| `lm-evaluation-harness/` | Local install of `lm_eval` (see `pyproject.toml`). |

Many scripts use `import s3_utils` and expect to be run with **`experiments/` on `PYTHONPATH`** (e.g. from `experiments/stereoset` or `experiments/winogender`) **or** from the repo root with the root on `PYTHONPATH`. Others use `from experiments import s3_utils` and are intended as **`python -m …`** from the **repo root** (see docstrings in e.g. `experiments/winogender/fine_tuned_test.py`).

## Winogender pipeline (overview)

1. **Prepare paired data** (writes to S3 via `s3_utils`):

   ```bash
   cd experiments/winogender
   python prepare_data.py
   ```

2. **DLA / bias analysis** — `winogender_bias_search.py` (argparse; see file for options).

3. **Fine-tuning** — `winogender_finetuning.py`:

   - `--mode dla` | `random` | `all`
   - DLA sweep: DPO/SFT over DLA-selected layers; `random` runs random-layer ablations on fixed "top" configs.

   ```bash
   cd experiments/winogender
   python winogender_finetuning.py --mode dla
   ```

4. **Batch evaluation on the test set** (loads checkpoints from S3, runs tracing / impact; supports filters and comparison mode — see top-of-file usage in the script):

   ```bash
   cd /path/to/diploma
   python -m experiments.winogender.fine_tuned_test --help
   ```

5. **Perplexity** — `compute_perplexity.py` (requires `--model` from a fixed set of keys; can filter DLA/SNR, comparison mode for Llama).

6. **LM harness benchmarks** — `run_lm_harness_tests.py` builds a table for baseline + top-5 fine-tuned runs and writes `all_models_benchmark_results.csv` in the **current working directory** when run to completion. It uses **Arc, HellaSwag, PIQA, WinoGrande, Social IQA, LAMBADA**, etc., as defined in the file.

7. **Optional LLM data generation** — `generate_test_set.py` (OpenAI). **Requires** `OPENAI_API_KEY`.

## StereoSet pipeline (overview)

1. **LLM paraphrase & JSONL SFT/DPO data** — `stereoset_paraphrase.py` (calls `rephrase_stereoset` / `generate_sft_v2_dataset` / `generate_dpo_triplet_dataset` when used as `__main__`). **Requires** `OPENAI_API_KEY`.

2. **Fine-tuning** — `stereoset_finetuning.py`:

   - `--mode dla` | `random` | `all`
   - `--experiments all` | `top5` (subset of hyperparameter grid vs. top-5 DPO configs only)

   ```bash
   cd experiments/stereoset
   python stereoset_finetuning.py --mode dla --experiments all
   ```

3. **Batch test / comparison** — `fine_tuned_test.py` (defaults to Llama checkpoints under `stereoset_experiments/…` unless `--comparison` for GPT-2 style paths — see `if __name__` block).

4. **LM harness** — `experiments/stereoset/run_lm_harness_tests.py` (same overall idea as Winogender; StereoSet S3 path layout).

5. **Bias search** — `stereoset_bias_search.py` (see file for `__main__` behavior).

## Spectrum (SNR → YAML for "unfrozen" layers)

From `spectrum/` (after dependencies are available):

```bash
cd spectrum
python spectrum.py --model-name <hf_model_id> --select-all [--top-percent 15] [--batch-size 1]
```

- With **no** existing `model_snr_results/snr_results_<slug>.json`, the script can compute SNR (interactive weight-type selection unless `--select-all`).
- If that JSON **already** exists, it can generate a YAML of top-SNR layers (see `ModelModifier.generate_unfrozen_params_yaml` in `spectrum/spectrum.py`).

## lm-evaluation-harness CLI (general)

After `pip install -e ./lm-evaluation-harness`, the `lm_eval` entrypoint is available. This repository's experiment scripts also call `lm_eval` **programmatically**; see the upstream docs for `lm_eval` CLI options if you run evaluations outside these wrappers.

## License / attribution

- `lm-evaluation-harness/` is vendored (EleutherAI, MIT in their `pyproject.toml`).
- Winogender schema tooling under `experiments/winogender/winogender-schemas/` has its own README and upstream attribution.
