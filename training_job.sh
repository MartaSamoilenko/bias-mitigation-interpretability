#!/bin/bash

python experiments/comparison/dla_error_analysis.py \
    --models gpt2-xl meta-llama/Llama-3.2-1B google/gemma-2-2b \
    --metric logit_diff --ablation-modes zero mean \
    --tag dev --no-s3

python experiments/comparison/eap_ig_error_analysis.py \
    --models gpt2-xl meta-llama/Llama-3.2-1B google/gemma-2-2b \
    --metric logit_diff --ablation-modes zero mean \
    --n-ig-steps 10 --ig-rule midpoint --ig-path embed_mean \
    --tag dev --no-s3

set -euo pipefail

python experiments/comparison/mitigation_sweep.py \
    --models meta-llama/Llama-3.2-1B gpt2-xl google/gemma-2-2b \
    --records-dir outputs/dla_error_analysis --tag dev \
    --dev-dataset datasets/gender_test_rephrased_v2.json \
    --test-dataset datasets/gender_dev_rephrased.json \
    --generic-text data/wikitext103_valid.txt \
    --ks 1 2 3 4 6 8 12 16 24 32 \
    --methods dla atp eap_ig ap random \
    --random-seeds 5 \
    --metric logit_diff --ablation-mode mean --topk-sign signed \
    --out outputs/mitigation/sweep_dev.csv --no-s3

# Robustness arms. --no-stereoset drops 762 of the 1,216 forwards per
# configuration, so these cost about a third of the run above.
#
# heads-only: MLPs dominate both the bias reduction and the capability cost, so
# this separates "which method" from "did an MLP happen to be picked".
python experiments/comparison/mitigation_sweep.py \
    --models meta-llama/Llama-3.2-1B gpt2-xl google/gemma-2-2b \
    --records-dir outputs/dla_error_analysis --tag dev \
    --dev-dataset datasets/gender_test_rephrased_v2.json \
    --test-dataset datasets/gender_dev_rephrased.json \
    --generic-text data/wikitext103_valid.txt \
    --ks 1 2 3 4 6 8 12 16 24 32 \
    --methods dla atp eap_ig ap random --random-seeds 3 \
    --component-types heads --no-stereoset \
    --metric logit_diff --ablation-mode mean --topk-sign signed \
    --out outputs/mitigation/sweep_heads.csv --no-s3

# zero-ablation baseline instead of mean
python experiments/comparison/mitigation_sweep.py \
    --models meta-llama/Llama-3.2-1B gpt2-xl google/gemma-2-2b \
    --records-dir outputs/dla_error_analysis --tag dev \
    --dev-dataset datasets/gender_test_rephrased_v2.json \
    --test-dataset datasets/gender_dev_rephrased.json \
    --generic-text data/wikitext103_valid.txt \
    --ks 1 2 3 4 6 8 12 16 24 32 \
    --methods dla atp eap_ig ap random --random-seeds 3 \
    --ablation-mode zero --no-stereoset \
    --metric logit_diff --topk-sign signed \
    --out outputs/mitigation/sweep_zero.csv --no-s3

# Figures and tables (CPU, seconds). The capability gate runs first.
python experiments/comparison/mitigation_figures.py \
    --sweep outputs/mitigation/sweep_dev.csv --out-dir outputs/paper
