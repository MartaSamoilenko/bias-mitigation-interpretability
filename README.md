# Mechanistic Localization and Mitigation of Gender Bias in LLMs

The pipeline implemented here:

1. **Locate** attention heads and MLP blocks that causally drive stereotypical generations using a layer-normalization-aware Direct Logit Attribution with multi-token "accumulated impact" weighting.
2. **Freeze** every parameter except the top-ranked components.
3. **Fine-tune** only the unfrozen subset with DPO or an unlikelihood-augmented SFT objective.

We evaluate on **GPT-2 XL**, **Gemma-2B**, and **Llama-3.2-1B** against the **StereoSet** (intrasentence, gender) and **Winogender** benchmarks, with comparisons against full fine-tuning, random-component selection, and a Spectrum/SNR baseline. Refer to the accompanying paper for full results, derivations, and a documented negative result on Llama-3.2-1B at high MLP sparsity.


## Setup

Tested on Python 3.10 with CUDA 12.x. A single A100 (40 GB) is sufficient for all reported experiments; larger MLP-from-attention sweeps on Llama-3.2-1B benefit from 48 GB.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
pip install -e ./lm-evaluation-harness        # editable install of the vendored harness
pip install -r spectrum/requirements.txt      # only if you intend to re-run Spectrum
```

Key pinned dependencies (see `requirements.txt` for the full list): `torch==2.8.0`, `transformers==4.56.2`, `transformer_lens`, `accelerate==1.10.1`, `openai==2.28.0`, `boto3==1.41.2`, `huggingface-hub`, `datasets`.

### Environment variables

The scripts read credentials from the environment or from a `.env` file at the repo root (`load_dotenv()` is called in each entry point). Required variables, depending on which scripts you run:

| Variable                  | Used for                                                  |
|---------------------------|-----------------------------------------------------------|
| `HF_TOKEN`                | Loading gated models (Gemma-2B, Llama-3.2-1B)             |
| `AWS_ACCESS_KEY_ID`       | S3 reads/writes for checkpoints, datasets, logs           |
| `AWS_SECRET_ACCESS_KEY`   | S3 reads/writes for checkpoints, datasets, logs           |
| `OPENAI_API_KEY`          | StereoSet rephrasing and Winogender held-out construction |


### S3 layout

`experiments/s3_utils.py` defines the bucket and prefix used throughout (`modelsfinetuned/experiments/...`). The S3 layout is hard-coded in the entry points (see `S3_PREFIX`/`LOGS_DIR`/`RESULTS_BASE` constants near the top of each `*_finetuning.py` and `fine_tuned_test.py`). Reproducing runs against your own infrastructure requires either adjusting these constants or stubbing `s3_utils` to read/write locally.


## Reproducing the paper

All commands assume the repo root as the working directory. Use `python -m` so the `experiments` package resolves correctly.

### 1. (One-time) build the datasets

```bash
# StereoSet: rephrase so BLANK is the final token
python -m experiments.stereoset.stereoset_paraphrase

# Winogender: instantiate templates + generate the 120-pair held-out test set
python -m experiments.winogender.prepare_data
python -m experiments.winogender.generate_ft_data
python -m experiments.winogender.generate_test_set      # uses OPENAI_API_KEY
```

The paraphrasing and test-set construction call `gpt-4o`. Generated artefacts are uploaded to S3 under `data/stereoset/...` and `data/winogender/...`.

### 2. Compute Direct Logit Attribution

```bash
# StereoSet
python -m experiments.stereoset.stereoset_bias_search

# Winogender
python -m experiments.winogender.winogender_bias_search
```

Implements Equation (2) of the paper (the prefix-probability-weighted accumulated impact) using the scale-corrected unembedding direction from Equation (1). The model name is configured at the top of each script. Per-component impact and per-token probability tables are saved to S3 under `outputs/<model>/.../accumulated_impact_*.csv` and `outputs/<model>/.../out_DLA_*.csv`.

### 3. Run the targeted fine-tuning sweep

```bash
# StereoSet — full DLA sweep (all sparsities) and random-control ablation
python -m experiments.stereoset.stereoset_finetuning --mode dla  --experiments all
python -m experiments.stereoset.stereoset_finetuning --mode random --experiments top5

# Winogender — same
python -m experiments.winogender.winogender_finetuning --mode dla
python -m experiments.winogender.winogender_finetuning --mode random
```

Flags:

* `--mode {dla,random,all}` — DLA-targeted, random-component control, or both.
* `--experiments {all,top5}` (StereoSet only) — full sweep across percentiles and learning rates, or the top-5 paper configurations.

The loss type (`dpo` or `sft_improved`), DPO `beta`, unlikelihood `ul_weight`, learning rate, and percentile sweep are configured in the `ExperimentConfig` dataclass at the top of each `*_finetuning.py`. Checkpoints and per-epoch logs are written under `outputs/<model>/<dataset>/fine_tuned_*/`.

### 4. SNR/Spectrum baseline

```bash
# (a) one-time SNR scan per model — Marchenko-Pastur thresholded
python -m spectrum.spectrum --model-name gpt2-xl    --top-percent 5
python -m spectrum.spectrum --model-name google/gemma-2b --top-percent 15
python -m spectrum.spectrum --model-name meta-llama/Llama-3.2-1B --top-percent 50

# (b) fine-tune using the SNR-selected layers
python -m experiments.stereoset.comparison_finetuning
python -m experiments.winogender.comparison_finetuning
```

The cached SNR rankings (used for the paper) are committed under `spectrum/snr_results_*.json` and `spectrum/snr_results_*_unfrozenparameters_*.yaml`, so step (a) can be skipped.

### 5. Evaluate

```bash
# bias metrics (SS / ICAT / ΔP for StereoSet; P_gb / P_sp / PPL for Winogender)
python -m experiments.stereoset.fine_tuned_test --filter all --comparison
python -m experiments.winogender.fine_tuned_test --filter all --comparison --skip_existing

# downstream zero-shot capability (ARC-e/c, HellaSwag, PIQA, WinoGrande, LAMBADA, Social IQA)
python -m experiments.stereoset.run_lm_harness_tests
python -m experiments.winogender.run_lm_harness_tests
```

Flags for `fine_tuned_test.py`:

* `--filter {all,dla,random,snr}` — restrict to one family of runs.
* `--run_id <id>` — single run rather than a batch.
* `--dataset_path <path>` — override the default test set (Winogender only).
* `--skip_existing` — skip runs whose result file already exists.
* `--comparison` — emit side-by-side baseline-vs-finetuned summary.

Results are uploaded to S3 under `outputs/<model>/<dataset>/fine_tuned*/results/`.

---

## End-to-end example

The single best Gemma-2B configuration on StereoSet (paper Table 1, row "DPO | DLA MLP-from-Attn (9.96)"):

```bash
# 1. dataset (one-time)
python -m experiments.stereoset.stereoset_paraphrase

# 2. DLA — make sure the model name in the script is set to google/gemma-2b
python -m experiments.stereoset.stereoset_bias_search

# 3. fine-tune on the top-5 paper configurations
python -m experiments.stereoset.stereoset_finetuning --mode dla --experiments top5

# 4. evaluate
python -m experiments.stereoset.fine_tuned_test --filter dla --comparison
```

Expected result for the `mlp_from_attn @ percentile=10, beta=0.3, lr=5e-6` run (single seed): SS ≈ 49.80, ICAT ≈ 51.56, ΔP ≈ 7.2 × 10⁻²¹.

---

## Datasets

* **StereoSet** — Nadeem et al. (2020), `bias_type=gender`, intrasentence. `stereoset_paraphrase.py` rephrases each example so `BLANK` is the final token (full prompt and validation rules in paper Appendix C.1).
* **Winogender** — Rudinger et al. (2018). Original templates and BLS occupation statistics live under `experiments/winogender/winogender-schemas/data/`. `prepare_data.py` instantiates them and `generate_test_set.py` builds the 120-pair held-out test set with GPT-4o, with manual BLS verification (full pipeline in paper Appendix C.2).
* **Filtering for fine-tuning.** Only examples on which the unmodified baseline already prefers the stereotype completion (StereoSet) or the BLS-stereotypical pronoun (Winogender) are used for gradient updates.

---

## Reproducibility notes

* Random-selection baselines run under five seeds; both mean and best are reported in the paper tables.
* Targeted DPO precomputes and caches the reference-model log-probabilities before the optimization loop, removing the reference model from VRAM during training.
* Spectrum/SNR uses the Marchenko–Pastur bulk-edge threshold; see paper Appendix G for the exact formula and the top-5 sweep.

---

## Citation

A blind citation will be added on acceptance. For the time being, please reference the OpenReview submission ID once it is assigned.

---

## Licensing

Code in this repository is released under the MIT License unless noted otherwise. Vendored components retain their upstream licenses:

* `lm-evaluation-harness/` — see `lm-evaluation-harness/LICENSE.md` (EleutherAI).
* `spectrum/` — see `spectrum/LICENSE`.
* `experiments/winogender/winogender-schemas/` — see `experiments/winogender/winogender-schemas/LICENSE`.

Model weights are governed by the respective Hugging Face model cards (GPT-2: MIT; Gemma: Gemma Terms of Use; Llama-3.2: Llama 3.2 Community License). Dataset components retain their upstream licenses (StereoSet: CC BY-SA 4.0; Winogender: MIT).
