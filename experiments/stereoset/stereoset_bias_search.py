import pandas as pd
import json
import torch
from transformer_lens import HookedTransformer
from huggingface_hub import login
import os
import random
import sys

import s3_utils

s3_utils.set_use_s3("--no-s3" not in sys.argv)

SENTENCEPIECE_MODELS = {"gemma", "llama", "mistral", "t5"}
BPE_MODELS = {"gpt2", "gpt-j", "opt", "llama-3"}

def get_model_family(model_name: str) -> str:
    name_lower = model_name.lower()
    for family in BPE_MODELS:
        if family in name_lower:
            return "bpe"
    for family in SENTENCEPIECE_MODELS:
        if family in name_lower:
            return "sentencepiece"
    return "unknown"

def tokenize_candidate(model, word: str, model_name: str) -> list[int] | None:
    family = get_model_family(model_name)

    if family == "sentencepiece":
        # SentencePiece
        DUMMY = "The"
        dummy_ids = model.tokenizer.encode(DUMMY, add_special_tokens=False)
        combined_ids = model.tokenizer.encode(
            f"{DUMMY} {word}", add_special_tokens=False
        )
        candidate_ids = combined_ids[len(dummy_ids):]
    else:
        # BPE (GPT-2, Llama-3 Tiktoken)
        try:
            candidate_ids = model.tokenizer.encode(
                word, add_special_tokens=False, add_prefix_space=True
            )
        except TypeError:
            candidate_ids = model.tokenizer.encode(
                ' ' + word, add_special_tokens=False
            )
    
    if len(candidate_ids) == 0:
        print(f"[WARNING] Word '{word}' produced 0 tokens. Skipping.")
        return None

    return candidate_ids


def validate_model_compatibility(model):
    cfg = model.cfg
    ln = model.ln_final
    ln_type = type(ln).__name__

    # --- Norm type report ---
    ln_weight = get_ln_final_weight(model)
    if ln_weight is not None:
        print(f"[OK] ln_final type : {ln_type} — has learnable gamma "
              f"(shape: {ln_weight.shape})")
    else:
        print(f"[OK] ln_final type : {ln_type} — weightless (RMSNormPre). "
              f"DLA will use raw_unembed_dir / ln_scale.")

    assert hasattr(ln, 'hook_scale'), (
        f"ln_final ({ln_type}) has no hook_scale HookPoint. "
        f"Update your TransformerLens version (>= 0.13 required)."
    )
    print(f"[OK] ln_final.hook_scale is present.")

    n_heads = cfg.n_heads
    n_kv_heads = getattr(cfg, 'n_key_value_heads', n_heads)
    if n_kv_heads < n_heads:
        print(f"[OK] GQA detected: n_heads={n_heads}, "
              f"n_key_value_heads={n_kv_heads}. "
              f"TransformerLens expands hook_z to n_heads — einsum is valid.")
    else:
        print(f"[OK] MHA: n_heads={n_heads} (no GQA).")

    act_fn = cfg.act_fn
    gated = act_fn in ("silu", "geglu", "swiglu", "gelu_new")
    print(f"[OK] act_fn={act_fn} → {'GatedMLP' if gated else 'StandardMLP'}. "
          f"hook_mlp_out is valid for DLA in both cases.")

    print(f"\n[READY] '{cfg.model_name}' | "
          f"layers={cfg.n_layers} | d_model={cfg.d_model} | "
          f"d_head={cfg.d_head}")

def get_ln_final_weight(model) -> torch.Tensor | None:
    ln = model.ln_final

    if hasattr(ln, 'w') and ln.w is not None:
        return ln.w  # [d_model]

    return None

login(token=os.environ["HF_TOKEN"])

model = HookedTransformer.from_pretrained("meta-llama/Llama-3.2-1B")
validate_model_compatibility(model)

device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Device : {device}")

model.eval()

try:
    rephrased_stereoset = s3_utils.read_json('data/stereoset/gender_test_rephrased.json')
    raw_data = s3_utils.read_json('data/stereoset/test.json')
except Exception as e:
    print(f"Error loading data from S3: {e}")
    rephrased_stereoset = []
    raw_data = {}

test_dict = {}
for sub_dict in rephrased_stereoset:
    test_dict[sub_dict['id']] = sub_dict['rephrased_context']

full_intrasentence_list = raw_data.get('data', {}).get('intrasentence', [])
id_to_biastype = {item['id']: item['bias_type'] for item in full_intrasentence_list if item['bias_type'] == 'gender'}
print(f"Loaded {len(id_to_biastype)} bias type mappings.")

def get_logit_attribution_old(model, cache, target_token_id, layer):

    raw_unembed_dir = model.W_U[:, target_token_id]
    target_unembed_dir = raw_unembed_dir

    attn_result = cache[f"blocks.{layer}.attn.hook_z"][0, -1]

    W_O = model.W_O[layer]

    head_contributions = torch.einsum("hd, hdm, m -> h",
                                      attn_result,
                                      W_O,
                                      target_unembed_dir)

    mlp_out = cache[f"blocks.{layer}.hook_mlp_out"][0, -1]
    mlp_contribution = torch.dot(mlp_out, target_unembed_dir)

    return head_contributions, mlp_contribution

def get_logit_attribution(model, cache, target_token_id, layer):

    ln_scale = cache["ln_final.hook_scale"][0, -1, 0]
    ln_weight = get_ln_final_weight(model)

    raw_unembed_dir = model.W_U[:, target_token_id]
    target_unembed_dir = raw_unembed_dir

    if ln_weight is not None:
        # LayerNorm / RMSNorm: gamma rescales each dimension individually.
        effective_unembed_dir = (ln_weight / ln_scale) * target_unembed_dir
    else:
        # RMSNormPre (Gemma-2B, Llama-3): no gamma, just RMS scaling.
        effective_unembed_dir = raw_unembed_dir / ln_scale

    attn_result = cache[f"blocks.{layer}.attn.hook_z"][0, -1]
    W_O = model.W_O[layer]

    head_contributions = torch.einsum("hd, hdm, m -> h",
                                      attn_result,
                                      W_O,
                                      effective_unembed_dir) #target_unembed_dir

    mlp_out = cache[f"blocks.{layer}.hook_mlp_out"][0, -1]
    mlp_contribution = torch.dot(mlp_out, effective_unembed_dir) #target_unembed_dir

    return head_contributions, mlp_contribution


def accumulative_layer_impact(filename):
    print("Loading CSV from S3...")

    df = s3_utils.read_csv(filename)

    df = df.sort_values(by=['ID', 'Candidate', 'Type', 'Layer', 'Token_Position'])

    print("Calculating Prefix Probabilities...")

    group_cols = ['ID', 'Candidate', 'Type', 'Layer']
    df['Prefix_Prob'] = df.groupby(group_cols)['Layer_Accumulated_Prob'].shift(1)
    df['Prefix_Prob'] = df['Prefix_Prob'].fillna(1.0)

    print("Calculating Weighted Impacts...")

    df['Weighted_MLP'] = df['MLP_Logit_Impact'] * df['Prefix_Prob']

    head_cols = [c for c in df.columns if c.startswith('Head_')]
    weighted_head_cols = []

    for col in head_cols:
        w_col_name = f'Weighted_{col}'
        df[w_col_name] = df[col] * df['Prefix_Prob']
        weighted_head_cols.append(w_col_name)

    print("Aggregating results...")

    agg_dict = { 'Weighted_MLP': 'sum' }
    for col in weighted_head_cols:
        agg_dict[col] = 'sum'

    final_df = df.groupby(group_cols).agg(agg_dict).reset_index()

    # ID | Candidate | Type | Layer | Component | Accumulated_Impact
    print("Formatting final table...")

    mlp_data = final_df.melt(
        id_vars=group_cols,
        value_vars=['Weighted_MLP'],
        var_name='Component',
        value_name='Accumulated_Impact'
    )
    mlp_data['Component'] = 'MLP'

    head_data = final_df.melt(
        id_vars=group_cols,
        value_vars=weighted_head_cols,
        var_name='Component',
        value_name='Accumulated_Impact'
    )
    head_data['Component'] = head_data['Component'].str.replace('Weighted_', '')

    result_df = pd.concat([mlp_data, head_data], ignore_index=True)

    return result_df


def layer_tracing(dataset,
                  output_filename):

    output_path = f"outputs/llama3.2_1b/dev_tests/{output_filename}"
    try:
        print(f"Attempting to resume from S3 ({output_path})...")
        df_existing = s3_utils.read_csv(output_path)
        all_data = df_existing.to_dict('records')
        completed_ids = set(df_existing['ID'].unique())
    except Exception:
        print("No existing file found on S3. Starting fresh.")
        all_data = []
        completed_ids = set()

    print(f"Already processed {len(completed_ids)} IDs.")

    for idx, sub_dict in enumerate(dataset):
        current_id = sub_dict['id']

        if current_id in completed_ids:
            continue

        if current_id not in id_to_biastype or id_to_biastype[current_id] != 'gender':
            continue

        print(f"Processing item {idx} (ID: {current_id})...")

        if len(all_data) > 0 and len(all_data) % 100 == 0:
            print("Saving intermediate results to S3...")
            s3_utils.write_csv(pd.DataFrame(all_data), output_path)

        original_prompt = test_dict[current_id].split('BLANK')[0].strip()
        candidates = sub_dict['targets']

        for stereotype_key, word in candidates.items():
            # word_with_space = ' ' + word
            # target_tokens = model.tokenizer.encode(word_with_space, add_special_tokens=False)

            target_tokens = tokenize_candidate(model, word, model.cfg.model_name)

            current_prompt = original_prompt
            layer_accumulated_probs = torch.ones(model.cfg.n_layers, device=device)

            # print(current_prompt)
            for token_pos, token_id in enumerate(target_tokens):
                with torch.no_grad():
                    _, cache = model.run_with_cache(current_prompt, 
                                                    return_type=None)

                try:
                    for layer in range(model.cfg.n_layers):
                        hidden_state = cache[f"blocks.{layer}.hook_resid_post"][0:1, -1:]
                        normalized_resid = model.ln_final(hidden_state)
                        layer_logits_3d = model.unembed(normalized_resid)

                        layer_logits = layer_logits_3d[0, 0]
                        layer_probs = torch.softmax(layer_logits, dim=-1)
                        p_token = layer_probs[token_id].item()

                        layer_accumulated_probs[layer] *= p_token

                        head_contribs, mlp_contrib = get_logit_attribution(
                            model, cache, token_id, layer
                        )

                        row = {
                            "ID": current_id,
                            "Prompt": current_prompt,
                            "Candidate": word,
                            "Token_Str": model.to_string(token_id),
                            "Token_Position": token_pos,
                            "Is_First_Token": (token_pos == 0),
                            "Type": stereotype_key,
                            "Layer": layer,
                            "Layer_Accumulated_Prob": layer_accumulated_probs[layer].item(),
                            "Token_Instant_Prob": p_token,
                            "MLP_Logit_Impact": mlp_contrib.item(),
                        }

                        for head_idx, score in enumerate(head_contribs):
                            row[f"Head_{head_idx}"] = score.item()

                        all_data.append(row)
                        pass
                finally:
                    del cache
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                current_prompt += model.to_string(token_id)

    df = pd.DataFrame(all_data)
    s3_utils.write_csv(df, output_path)
    return all_data

SPLITTING = False
TRACING = True
ACC_ANALYSIS = True

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="StereoSet DLA bias search")
    parser.add_argument(
        "--no-s3", action="store_true",
        help="Read/write datasets and results on local disk instead of S3 "
             "(use when AWS credentials/S3 access are unavailable). "
             "Note: this flag is already honored at import time; see the "
             "module-level s3_utils.set_use_s3() call near the top of this file.")
    args = parser.parse_args()

    if SPLITTING:
        print("Starting 80/20 Split for Fine-tuning sets...")

        gender_data = []
        for item in rephrased_stereoset:
            if item['id'] in id_to_biastype and id_to_biastype[item['id']] == 'gender':
                gender_data.append(item)

        print(f"Total gender examples found: {len(gender_data)}")

        if len(gender_data) == 0:
            print("No gender data found to split. Check input files.")
        else:
            random.seed(42)
            random.shuffle(gender_data)

            split_idx = int(len(gender_data) * 0.8)
            train_set = gender_data[:split_idx]
            test_set = gender_data[split_idx:]

            train_path = "data/stereoset/splits/gender_train.json"
            test_path = "data/stereoset/splits/gender_test.json"

            s3_utils.write_json(train_set, train_path)
            s3_utils.write_json(test_set, test_path)

            print(f"Successfully created splits on S3:")
            print(f" - Train: {len(train_set)} examples ({train_path})")
            print(f" - Test:  {len(test_set)} examples ({test_path})")

        exit(0)

    if TRACING:
        test_file_path = "data/stereoset/gender_test_rephrased.json"
        print(f"Loading testing data from S3 ({test_file_path})...")
        test_data = s3_utils.read_json(test_file_path)
        print(f"Loaded {len(test_data)} testing examples.")

        print("Starting Tracing on Testing Data...")
        all_data = layer_tracing(test_data, "out_DLA_gender_baseline_test_v2_norm.csv")
        print("Tracing Complete.")

    if ACC_ANALYSIS:
        print("Starting Accumulation Analysis...")
        filename = "outputs/llama3.2_1b/dev_tests/out_DLA_gender_baseline_test_v2_norm.csv"
        output_filename = "outputs/llama3.2_1b/dev_tests/accumulated_impact_gender_baseline_test_v2_norm.csv"

        try:
            result_df = accumulative_layer_impact(filename)
            s3_utils.write_csv(result_df, output_filename)
            print(f"Done! Saved accumulated results to S3 ({output_filename})")
        except Exception as e:
            print(f"File {filename} not found on S3: {e}. Run TRACING first.")