# Bias mitigation and interpretability for language models

Research code for locating and mitigating gender bias in language models
using mechanistic interpretability. The repo has two layers of work built on
[TransformerLens](https://github.com/TransformerLensOrg/TransformerLens)
(`HookedTransformer`):

- **`experiments/comparison/`** — the current, primary project: a systematic
  comparison of four attribution methods (DLA, AtP, EAP-IG, DE) against
  activation-patching ground truth for gender-bias localization on a StereoSet
  variant, across GPT-2 XL, Llama-3.2-1B and Gemma-2-2B, followed by a
  mitigation sweep that ablates each method's top-k components and measures
  bias reduction against capability cost.
- **`experiments/stereoset/`** and **`experiments/winogender/`** — earlier
  work: direct linear attribution to pick layers, then DPO/SFT fine-tuning on
  StereoSet- and Winogender-style tasks, evaluated with a local copy of
  EleutherAI's `lm-evaluation-harness`.

[Models and dataset collection on Hugging Face](https://hf.co/collections/Retrogradi/bias-mitigation-via-mechanistic-interpretability)

## Setup

Needs a GPU for training and most evaluation, Python 3.10+ (to match the
harness), and a CUDA build of PyTorch for your machine.

```bash
cd /path/to/bias-mitigation-interpretability
python -m venv .venv && source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
pip install -e ./lm-evaluation-harness
```

`requirements.txt` pins `transformer-lens`, `transformers`, `torch` and
related libraries as version-sensitive; everything else is a floor. For
Jacobian Lens, `pip install -e ./jacobian-lens`. For Spectrum, see
[`spectrum/requirements.txt`](spectrum/requirements.txt).

Credentials — set as environment variables or in a `.env` at the repo root
(`load_dotenv()` picks it up):

| Variable | Used for |
| --- | --- |
| `HF_TOKEN` | Downloading gated models (Llama, Gemma) |
| `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` | S3 reads/writes via `s3_utils.py` (pass `--no-s3` to every script to use local disk instead) |
| `OPENAI_API_KEY` | A few data-generation scripts (StereoSet paraphrasing, Winogender test-set generation) |

`HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1` forces the local Hugging Face cache
when a host has no outbound network but the models are already cached.

**Note on `outputs/` and `s3_downloads/`:** both are gitignored, along with a
few of the newest `experiments/comparison/` scripts and notebooks. A fresh
clone will not have run artifacts.

## Repository map

```
experiments/
  comparison/        current project: attribution comparison + mitigation sweep
  stereoset/         StereoSet: bias search, DPO/SFT fine-tuning, benchmarks
  winogender/        Winogender: same idea, Winogender schemas
  s3_utils.py        shared S3 read/write helpers (bucket/key layout fixed in code)
spectrum/            vendored SNR-based layer-selection scanner
lm-evaluation-harness/  vendored EleutherAI harness (editable install)
jacobian-lens/       vendored reference implementation of the Jacobian lens
j_lens_experiments/  exploratory notebook applying jacobian-lens to this project's data
datasets/            StereoSet-derived JSON/JSONL used by experiments/comparison and stereoset
data/                generic text corpus (WikiText-103) for perplexity checks
outputs/             all run artifacts — CSVs, figures, LaTeX tables (gitignored)
checkpoints/         fine-tuned model checkpoints (empty by default; gitignored contents)
s3_downloads/        one-off scripts for pulling/deleting S3 experiment data
.mechinterp_learning/  personal interview-prep notes, unrelated to the research
```

## `experiments/comparison/` — attribution comparison and mitigation

Four attribution methods, benchmarked against activation patching (the causal
ground truth), on a gender-focused StereoSet variant, across three
architectures.

**Methods.** DLA (Direct Logit Attribution, one forward pass), AtP
(Attribution Patching, one backward pass), EAP-IG (Edge Attribution Patching
with Integrated Gradients, `m` backward passes along an interpolation path),
DE (Direct Effect, the frozen-norm component of DLA's error) — all compared
against AP (Activation Patching: one forward pass per component, the
expensive ground truth).

**Pipeline** (four stages, two GPU-expensive, two CPU-cheap):

```
Stage 1 (GPU)                    Stage 2 (CPU, ~15 min)
  dla_error_analysis.py    ─┐
  eap_ig_error_analysis.py ─┼─> paper_figures.py         Table 1, error-budget
                            │   subsample_stability.py    table, all intrinsic
                            │                             figures, figure_data.json
                            └─> Stage 3 (GPU)             Stage 4 (CPU, seconds)
                                mitigation_sweep.py   ──>  mitigation_figures.py
```

- **`dla_error_analysis.py`** — loads a model, runs DLA/DE/AP over sampled
  StereoSet examples under zero- and mean-ablation, decomposes DLA's error
  into a frozen-norm term (Source A) and an indirect-effect term (Source B),
  writes per-component CSV records.
- **`eap_ig_error_analysis.py`** — same records file schema, adds EAP-IG and
  AtP scores (AtP is EAP-IG with one integration step), merges against the AP
  records for NMAE/rank-agreement summaries.
- **`mitigation_sweep.py`** — for each method, jointly ablates its top-k
  components (mean- or zero-ablation) and measures bias reduction (mean
  stereotype − anti-stereotype logit difference, plus StereoSet SS/LMS/ICAT)
  against capability cost (perplexity on held-out generic text) — all on
  held-out StereoSet items disjoint from the selection set.

Supporting scripts: `prepare_generic_text.py` (builds the WikiText-103
perplexity corpus), `validate_mean_baseline.py` (checks whether the
mean-ablation baseline's position scope is driving a capability-cost number),
`verify_atp.py` (correctness checks on the AtP implementation),
`abstract_slots.py` (fills the paper abstract's numeric slots from the current
artifacts, so the abstract can't silently drift from the data),
`analysis.ipynb` (scratch analysis). `dla_error_analysis_plots.ipynb` and
`eap_ig_comparison_plots.ipynb` are superseded by `paper_figures.py` and kept
only for reference.

Minimal run:

```bash
python experiments/comparison/dla_error_analysis.py --tag dev --no-s3
python experiments/comparison/eap_ig_error_analysis.py --tag dev --no-s3
python experiments/comparison/paper_figures.py --tag dev --out-dir outputs/paper
python experiments/comparison/subsample_stability.py --tag dev --out-dir outputs/paper

python experiments/comparison/mitigation_sweep.py --tag dev \
    --generic-text data/wikitext103_valid.txt --no-s3
python experiments/comparison/mitigation_figures.py \
    --sweep outputs/mitigation/sweep_dev.csv --out-dir outputs/paper
```

`training_job.sh` is the SLURM entry point that runs the headline mitigation
sweep plus the heads-only and zero-ablation robustness arms.

## `experiments/stereoset/` and `experiments/winogender/`

Earlier, parallel work: use DLA to pick candidate layers, then fine-tune with
DPO or SFT and evaluate. Both trees follow the same shape; each script's
module docstring documents `python -m` usage from the repo root.

- **`stereoset_paraphrase.py`** — builds the StereoSet-derived JSONL,
  rephrasing contexts so the blank is the sentence's last word (needed for
  single-token next-token scoring); `validate_rephrased_contexts.py` checks
  the result.
- **`stereoset_bias_search.py`** / **`winogender_bias_search.py`** — DLA-based
  search for the layers/components most responsible for the biased behavior.
- **`stereoset_finetuning.py`** / **`winogender_finetuning.py`** — DPO and SFT
  fine-tuning, selectable with `--mode`.
- **`comparison_finetuning.py`** — compares fine-tuning configurations.
- **`fine_tuned_test.py`** — batch evaluation and before/after comparison.
- **`run_lm_harness_tests.py`** — gathers `lm-evaluation-harness` benchmark
  results for baseline vs. fine-tuned checkpoints.
- Winogender-only: **`prepare_data.py`** (builds the paired dataset from
  `winogender-schemas/`), **`generate_test_set.py`** / **`generate_ft_data.py`**
  (LLM-assisted test-set and fine-tuning-data generation),
  **`compute_perplexity.py`**.

The Winogender schema templates live in
[`experiments/winogender/winogender-schemas/`](experiments/winogender/winogender-schemas/);
see its own README for that subproject.

## Vendored and supporting tools

- **`spectrum/`** — a vendored Spectrum-style SNR scanner for picking which
  parameters to unfreeze during fine-tuning; outputs YAML unfreeze lists.
  Usage, options, and the paper reference are in
  [`spectrum/README.md`](spectrum/README.md).
- **`lm-evaluation-harness/`** — EleutherAI's evaluation framework, installed
  in editable mode and called from Python in several experiment scripts (see
  `run_lm_harness_tests.py` in both `stereoset/` and `winogender/`). CLI
  details in [`lm-evaluation-harness/README.md`](lm-evaluation-harness/README.md);
  remains under its own upstream license.
- **`jacobian-lens/`** — a vendored reference implementation (not authored in
  this repo) of the Jacobian lens from *Verbalizable Representations Form a
  Global Workspace in Language Models*: reads out what a residual-stream
  activation is disposed to make the model say, by linearly transporting it
  into the final-layer basis via the averaged input–output Jacobian. See
  [`jacobian-lens/README.md`](jacobian-lens/README.md).
- **`j_lens_experiments/`** — exploratory notebook (`j_lens_iteration.ipynb`)
  applying the Jacobian lens to compare group-averaged Jacobians across
  synonym/control conditions on GPT-2. Exploratory, not part of the main
  comparison study.

## Data

- **`datasets/`** — StereoSet-derived files consumed by `experiments/stereoset`
  and `experiments/comparison`: `gender_test_rephrased_v2.json` (the
  767-example development/selection split — components are selected on this),
  `gender_dev_rephrased.json` (the 254-example held-out test split used for
  every reported mitigation number — the StereoSet dump names the splits the
  wrong way round), `gender_test_rephrased_fix_log.json` (paraphrase-fix audit
  log), `dpo_pairs_triplet_v2.jsonl` / `sft_bias_mitigation_v2.jsonl`
  (fine-tuning data derived from the same StereoSet items: DPO triplets with
  `debias`/`lms` pair types, and SFT completions paired with their stereotype
  counterpart).
- **`data/wikitext103_valid.txt`** — generic-text corpus for the mitigation
  sweep's perplexity metric, built by `prepare_generic_text.py` from
  WikiText-103 validation (natural text, disjoint from the bias data, and
  identical across every ablation configuration so only the *relative*
  perplexity change is meaningful — GPT-2's absolute perplexity is not
  comparable to Llama's or Gemma's since it was trained on WebText, not
  Wikipedia).

## Outputs

`outputs/` (gitignored) holds every run artifact:

- `outputs/dla_error_analysis/` — per-model, per-mode attribution record CSVs
  (the Stage-1 output that everything downstream reads).
- `outputs/mitigation/` — sweep CSVs, per-item bootstrap arrays (`*_items.npz`),
  and baseline-validation CSVs.
- `outputs/paper/` — the current, canonical set of figures, LaTeX tables, and
  `figure_data.json` for the comparison paper.
- `outputs/paper_dev/`, `outputs/paper_backup_untagged_*` — provenance-check
  and backup snapshots kept during development.
- `outputs/archive/` — older attribution record CSVs kept for reference.
- `outputs/figures/` — figures from the superseded plotting notebooks.

## Licensing

Code authored in this repo has no separate license file at the root.
Vendored trees keep their own licenses: `lm-evaluation-harness/` (upstream
license in that directory) and `jacobian-lens/` (Apache 2.0, see
`jacobian-lens/LICENSE`). Other vendored or third-party trees may ship their
own `LICENSE`/`README` in place — check before reusing.
