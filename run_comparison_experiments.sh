#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="/workspace/bias-mitigation-interpretability"
cd "$PROJECT_DIR"

# ============================================================================
# STEP 1: Generate SNR results (skip if sorted JSON already exists)
# ============================================================================

echo "=========================================="
echo "STEP 1: SNR computation"
echo "=========================================="

# GPT-2 XL — already computed, skip
if [ -f "spectrum/snr_results_gpt2-xl_sorted.json" ]; then
    echo "GPT-2 XL SNR results already exist. Skipping."
else
    echo "Computing SNR for GPT-2 XL ..."
    cd spectrum
    python3 spectrum.py --model-name gpt2-xl --batch-size 1 --select-all
    cd "$PROJECT_DIR"
fi

# Gemma-2B
if [ -f "spectrum/snr_results_google-gemma-2b_sorted.json" ]; then
    echo "Gemma-2B SNR results already exist. Skipping."
else
    echo "Computing SNR for Gemma-2B ..."
    cd spectrum
    python3 spectrum.py --model-name google/gemma-2b --batch-size 1 --select-all
    cd "$PROJECT_DIR"
fi

# Llama-3.2-1B
if [ -f "spectrum/snr_results_meta-llama-Llama-3.2-1B_sorted.json" ]; then
    echo "Llama-3.2-1B SNR results already exist. Skipping."
else
    echo "Computing SNR for Llama-3.2-1B ..."
    cd spectrum
    python3 spectrum.py --model-name meta-llama/Llama-3.2-1B --batch-size 1 --select-all
    cd "$PROJECT_DIR"
fi

# ============================================================================
# STEP 2: StereoSet comparison experiments (GPT-2 XL)
# ============================================================================

echo ""
echo "=========================================="
echo "STEP 2: StereoSet DLA vs SNR (Llama-3.2-1B)"
echo "=========================================="

SNR_JSON_GPT2_XL="spectrum/snr_results_gpt2-xl_sorted.json"
SNR_JSON_GEMMA_2B="spectrum/snr_results_google-gemma-2b_sorted.json"
SNR_JSON_LLAMA="spectrum/snr_results_meta-llama-Llama-3.2-1B_sorted.json"

# DPO experiments
# echo "--- StereoSet DPO: all methods, all modes ---"
# python3 -m experiments.stereoset.comparison_finetuning \
#     --snr-json "$SNR_JSON_GPT2_XL" \
#     --method snr \
#     --mode all \
#     --layer-counts 1 2 4 8 \
#     --loss-type dpo \
#     --dpo-beta 0.3 \
#     --learning-rate 5e-6

# SFT experiments
# echo "--- StereoSet SFT: all methods, all modes ---"
# python3 -m experiments.stereoset.comparison_finetuning \
#     --snr-json "$SNR_JSON_GPT2_XL" \
#     --method snr \
#     --mode all \
#     --layer-counts 1 2 4 8 \
#     --loss-type sft_improved \
#     --ul-weight 1.0 \
#     --learning-rate 5e-6

# python3 -m experiments.stereoset.comparison_finetuning \
#     --snr-json "$SNR_JSON_GEMMA_2B" \
#     --method snr \
#     --mode all \
#     --layer-counts 1 2 4 8 \
#     --loss-type dpo \
#     --dpo-beta 0.3 \
#     --learning-rate 5e-6

# # SFT experiments
# echo "--- StereoSet SFT: all methods, all modes ---"
# python3 -m experiments.stereoset.comparison_finetuning \
#     --snr-json "$SNR_JSON_GEMMA_2B" \
#     --method snr \
#     --mode all \
#     --layer-counts 1 2 4 8 \
#     --loss-type sft_improved \
#     --ul-weight 1.0 \
#     --learning-rate 5e-6

python3 -m experiments.stereoset.comparison_finetuning \
    --snr-json "$SNR_JSON_LLAMA" \
    --method snr \
    --mode all \
    --layer-counts 1 2 4 8 \
    --loss-type dpo \
    --dpo-beta 0.3 \
    --learning-rate 5e-6

# SFT experiments
echo "--- StereoSet SFT: all methods, all modes ---"
python3 -m experiments.stereoset.comparison_finetuning \
    --snr-json "$SNR_JSON_LLAMA" \
    --method snr \
    --mode all \
    --layer-counts 1 2 4 8 \
    --loss-type sft_improved \
    --ul-weight 1.0 \
    --learning-rate 5e-6

# ============================================================================
# STEP 3: Winogender comparison experiments (Llama-3.2-1B)
# ============================================================================
echo ""
echo "=========================================="
echo "STEP 3: Winogender DLA vs SNR"
echo "=========================================="

DPO experiments
echo "--- Winogender DPO: all methods, all modes ---"
python3 -m experiments.winogender.comparison_finetuning \
    --snr-json "$SNR_JSON_GPT2_XL" \
    --method snr \
    --mode all \
    --layer-counts 1 2 4 8 \
    --loss-type dpo \
    --dpo-beta 0.3 \
    --learning-rate 5e-6

# SFT experiments
echo "--- Winogender SFT: all methods, all modes ---"
python3 -m experiments.winogender.comparison_finetuning \
    --snr-json "$SNR_JSON_GPT2_XL" \
    --method snr \
    --mode all \
    --layer-counts 1 2 4 8 \
    --loss-type sft_improved \
    --ul-weight 1.0 \
    --learning-rate 5e-6


# DPO experiments
# echo "--- Winogender DPO: all methods, all modes ---"
# python3 -m experiments.winogender.comparison_finetuning \
#     --snr-json "$SNR_JSON_GEMMA_2B" \
#     --method snr \
#     --mode all \
#     --layer-counts 1 2 4 8 \
#     --loss-type dpo \
#     --dpo-beta 0.3 \
#     --learning-rate 5e-6

# # SFT experiments
# echo "--- Winogender SFT: all methods, all modes ---"
# python3 -m experiments.winogender.comparison_finetuning \
#     --snr-json "$SNR_JSON_GEMMA_2B" \
#     --method snr \
#     --mode all \
#     --layer-counts 1 2 4 8 \
#     --loss-type sft_improved \
#     --ul-weight 1.0 \
#     --learning-rate 5e-6


# DPO experiments
# echo "--- Winogender DPO: all methods, all modes ---"
# python3 -m experiments.winogender.comparison_finetuning \
#     --snr-json "$SNR_JSON_LLAMA" \
#     --method snr \
#     --mode all \
#     --layer-counts 1 2 4 8 \
#     --loss-type dpo \
#     --dpo-beta 0.3 \
#     --learning-rate 5e-6

# SFT experiments
# echo "--- Winogender SFT: all methods, all modes ---"
# python3 -m experiments.winogender.comparison_finetuning \
#     --snr-json "$SNR_JSON_LLAMA" \
#     --method snr \
#     --mode all \
#     --layer-counts 1 2 4 8 \
#     --loss-type sft_improved \
#     --ul-weight 1.0 \
#     --learning-rate 5e-6

echo ""
echo "=========================================="
echo "All experiments completed."
echo "=========================================="
