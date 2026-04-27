# Diploma: bias mitigation and evaluation for language models

This repository holds research code for studying and mitigating gender-related bias in language models. The main work uses the Transformer Lens (HookedTransformer) on StereoSet and Winogender-style tasks, with direct linear attribution to pick layers, DPO, and supervised fine-tuning, and a follow-up evaluation using a local copy of EleutherAI's lm-evaluation-harness. Shared helpers for AWS S3 live under `experiments/s3_utils.py` (the bucket and key layout are fixed in code, so replicating runs off the default infrastructure requires edits or stubs).

You need a GPU for training and most evaluations, Python 3.10 or newer to match the harness, and a CUDA-compiled PyTorch build that fits your machine. The root `requirements.txt` pins torch, transformers, and related libraries. Install the harness in editable mode after the base requirements are met. For Spectrum, see `spectrum/requirements.txt` and the documentation in that folder.

Set credentials the scripts expect, or place them in a `.env` file at the repo root where `load_dotenv()` applies. Hugging Face (`HF_TOKEN`) is used to access the model. AWS keys (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`) back S3 reads and writes. A few data scripts call the OpenAI API and need `OPENAI_API_KEY`.

"`bash
cd /path/to/diploma
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
pip install -e ./lm-evaluation-harness
```

The `experiments/winogender` tree holds Winogender-related scripts: data preparation (`prepare_data.py`), DLA and bias search (`winogender_bias_search.py`), fine-tuning with a `--mode` flag (`winogender_finetuning.py`), batch testing and comparison modes (`fine_tuned_test.py`, see its module docstring for `python -m' usage from the repo root), perplexity (`compute_perplexity.py`), optional test-set generation with an LLM (`generate_test_set.py`), and benchmark gathering (`run_lm_harness_tests.py`). The Winogender schema templates and their description are in `experiments/winogender/winogender-schemas/`; read `experiments/winogender/winogender-schemas/README.md` for that subproject.

The `experiments/stereoset` tree mirrors the same idea for StereoSet: paraphrase and JSONL building (`stereoset_paraphrase.py`), fine-tuning with `--mode` and `--experiments` on `stereoset_finetuning.py`, batch evaluation (`fine_tuned_test.py`), bias search, and `run_lm_harness_tests.py`.

`spectrum/` contains a vendored Spectrum-style SNR scanner for layer selection and YAML output for which parameters to unfreeze. Usage, options, and paper reference are in `spectrum/README.md`.

Install and CLI details for the evaluation framework are in `lm-evaluation-harness/README.md`. This repo calls `lm_eval` from Python in several experiment scripts; for standalone `lm_eval` use, follow that README.

Licensing: the harness remains under its upstream license in `lm-evaluation-harness/`. Other vendored or third-party trees may ship their own `LICENSE` or `README` files in place.

[Models and dataset collection](https://hf.co/collections/Retrogradi/bias-mitigation-via-mechanistic-interpretability) 