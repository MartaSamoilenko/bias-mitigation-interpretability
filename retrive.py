import os
import pandas as pd

PERCENTAGES = [0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]
CONDITIONS = ['attn', 'mlp', 'attn_mlp']
RESULTS_DIR = "results_spectrum"

import transformers
from huggingface_hub import login

from transformer_lens import HookedTransformer

login(token="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")
model_name = 'gpt2-xl'
tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
tokenizer.add_bos_token = False

hf_model = transformers.AutoModelForCausalLM.from_pretrained(model_name).to('cpu')
model = HookedTransformer.from_pretrained(
    model_name,
    hf_model=hf_model,
    #n_devices=2,
    device='cpu',
    tokenizer=tokenizer
)

model.cfg.use_attn_in = True
model.cfg.use_split_qkv_input = True
model.cfg.use_attn_result = True
model.cfg.use_hook_mlp_in = True

import json
with open("spectrum/snr_results_gpt2-xl_sorted.json") as f:
    layers = json.load(f)

os.makedirs(RESULTS_DIR, exist_ok=True)

attn_finetune = {}
mlp_finetune = {}
attn_mlp_finetune = {}

attn_finetune_params = {}
mlp_finetune_params = {}
attn_mlp_finetune_params = {}

total_params = 0

for percent in PERCENTAGES:
    attn_finetune[percent] = []
    mlp_finetune[percent] = []
    attn_mlp_finetune[percent] = []

    attn_finetune_params[percent] = 0
    mlp_finetune_params[percent] = 0
    attn_mlp_finetune_params[percent] = 0

    # Calculate layer subsets
    c_attn_finetune = list(layers['attn.c_attn'])[:round(len(layers['attn.c_attn']) * percent)]
    c_proj_finetune = list(layers['attn.c_proj'])[:round(len(layers['attn.c_proj']) * percent)]
    mlp_c_fc_finetune = list(layers['mlp.c_fc'])[:round(len(layers['mlp.c_fc']) * percent)]
    mlp_c_proj_finetune = list(layers['mlp.c_proj'])[:round(len(layers['mlp.c_proj']) * percent)]

    # Iterate over parameters to build the lists
    for name, param in model.named_parameters():
        if 'mlp' in name:
            number = name.split('.')[1]
            is_mlp_param = False
            for c_fc in mlp_c_fc_finetune:
                if number == c_fc.split('.')[2]:
                    is_mlp_param = True
                    break
            if not is_mlp_param:
                for c_proj in mlp_c_proj_finetune:
                    if number == c_proj.split('.')[2]:
                        is_mlp_param = True
                        break

            if is_mlp_param:
                mlp_finetune[percent].append(name)
                attn_mlp_finetune[percent].append(name)
                mlp_finetune_params[percent] += param.numel()
                attn_mlp_finetune_params[percent] += param.numel()

        if '_O' in name.split('.')[-1]:  # Attn proj
            number = name.split('.')[1]
            for c_proj in c_proj_finetune:
                if number == c_proj.split('.')[2]:
                    attn_finetune[percent].append(name)
                    attn_mlp_finetune[percent].append(name)
                    attn_finetune_params[percent] += param.numel()
                    attn_mlp_finetune_params[percent] += param.numel()
                    break
        elif 'attn' in name:  # Attn query, key, value
            number = name.split('.')[1]
            for c_attn in c_attn_finetune:
                if number == c_attn.split('.')[2]:
                    attn_finetune[percent].append(name)
                    attn_mlp_finetune[percent].append(name)
                    attn_finetune_params[percent] += param.numel()
                    attn_mlp_finetune_params[percent] += param.numel()
                    break

print("Pre-computation complete.")

param_count_map = {
        'attn': attn_finetune_params,
        'mlp': mlp_finetune_params,
        'attn_mlp': attn_mlp_finetune_params
    }

for condition in CONDITIONS:
    for percent in PERCENTAGES:
        if f"new_full_results_{condition}_{percent}.csv" in os.listdir(RESULTS_DIR):
            print(f"Skipping experiment: Condition = {condition}, Percent = {percent}")
            trainable_params_count = param_count_map[condition][percent]

            full_result = pd.read_csv(f"results_spectrum/new_full_results_{condition}_{percent}.csv")
            biased_results = full_result[abs(full_result["female_yes_ratio"] - full_result["male_yes_ratio"]) > 0.15]

            biased_results.to_csv(f"results_spectrum/new_bias_results_{condition}_{percent}.csv")