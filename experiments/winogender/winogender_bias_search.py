"""
DLA tracing and accumulated impact analysis for the Winogender dataset.
Reuses the core functions from fine_tuned_test.py.
"""
import argparse
import os
import sys

import boto3
import pandas as pd
import torch
from dotenv import load_dotenv
from transformer_lens import HookedTransformer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import s3_utils

load_dotenv()

TRACING = True
ACC_ANALYSIS = True



def two_stage_tracing(model, dataset, pronoun_probs_path, suffix_probs_path):
    """Two-stage Winogender bias measurement.

    Stage 1 — Pronoun prediction: forward pass on prefix only. At the last
    prefix position, record P(pronoun) and DLA for each candidate pronoun.
    Output is compatible with accumulative_layer_impact().

    Stage 2 — Coreference resolution: for each pronoun variant, forward pass
    on the full sentence. Compute log P(suffix | prefix + pronoun) to
    determine which pronoun best fits the continuation.
    """
    done_ids_s1 = set()
    try:
        df = s3_utils.read_csv(pronoun_probs_path)
        done_ids_s1 = set(df['ID'].unique().tolist())
    except Exception:
        pass

    stage1_data = []
    stage2_data = []

    for idx, sub_dict in enumerate(dataset):
        if sub_dict['id'] in done_ids_s1:
            continue

        print(f"Processing item {idx} ({sub_dict['id']})...")
        if len(stage1_data) > 0 and idx % 10 == 0:
            print("Saving intermediate results to S3...")
            s3_utils.write_csv(pd.DataFrame(stage1_data), pronoun_probs_path)
            s3_utils.write_csv(pd.DataFrame(stage2_data), suffix_probs_path)

        ID = sub_dict['id']
        prefix = sub_dict['prefix']
        full_sentences = sub_dict['full_sentences']
        candidates = sub_dict['targets']

        # ── Stage 1: pronoun prediction from prefix ──────────────────────
        prefix_tokens = model.tokenizer.encode(prefix)
        prefix_len = len(prefix_tokens)
        cache_pos = prefix_len - 1  # last prefix token predicts the pronoun

        with torch.no_grad():
            _, prefix_cache = model.run_with_cache(prefix)

        for stereotype_key, pronoun_word in candidates.items():
            pronoun_with_space = ' ' + pronoun_word
            pronoun_token_ids = model.tokenizer.encode(pronoun_with_space)

            for tok_offset, target_token_id in enumerate(pronoun_token_ids):
                layer_accumulated_probs = torch.ones(model.cfg.n_layers, device='cpu')

                for layer in range(model.cfg.n_layers):
                    hidden = prefix_cache[f"blocks.{layer}.hook_resid_post"][0, cache_pos]
                    norm_hidden = model.ln_final(hidden)
                    logits = model.unembed(norm_hidden)
                    probs = torch.softmax(logits, dim=-1)

                    p_token = probs[target_token_id].item()
                    layer_accumulated_probs[layer] *= p_token

                    unembed_dir = model.W_U[:, target_token_id]

                    attn_result = prefix_cache[f"blocks.{layer}.attn.hook_z"][0, cache_pos]
                    W_O = model.W_O[layer]
                    head_contribs = torch.einsum("hd, hdm, m -> h",
                                                 attn_result, W_O, unembed_dir)

                    mlp_out = prefix_cache[f"blocks.{layer}.hook_mlp_out"][0, cache_pos]
                    mlp_contrib = torch.dot(mlp_out, unembed_dir)

                    row = {
                        "ID": ID,
                        "Prompt": prefix,
                        "Candidate": pronoun_word,
                        "Token_Str": model.to_string(target_token_id),
                        "Token_Position": tok_offset,
                        "Is_First_Token": (tok_offset == 0),
                        "Type": stereotype_key,
                        "Layer": layer,
                        "Layer_Accumulated_Prob": layer_accumulated_probs[layer].item(),
                        "Token_Instant_Prob": p_token,
                        "MLP_Logit_Impact": mlp_contrib.item(),
                    }
                    for head_idx, score in enumerate(head_contribs):
                        row[f"Head_{head_idx}"] = score.item()

                    stage1_data.append(row)

        del prefix_cache

        # ── Stage 2: suffix probability conditioned on each pronoun ──────
        for stereotype_key, pronoun_word in candidates.items():
            full_sent = full_sentences[stereotype_key]
            full_tokens = model.tokenizer.encode(full_sent)

            pronoun_with_space = ' ' + pronoun_word
            pronoun_token_ids = model.tokenizer.encode(pronoun_with_space)
            suffix_start = prefix_len + len(pronoun_token_ids)

            with torch.no_grad():
                _, full_cache = model.run_with_cache(full_sent)

            n_layers = model.cfg.n_layers
            suffix_log_prob = 0.0
            for t_pos in range(suffix_start, len(full_tokens)):
                hidden = full_cache[f"blocks.{n_layers - 1}.hook_resid_post"][0, t_pos - 1]
                logits_t = model.unembed(model.ln_final(hidden))
                probs_t = torch.softmax(logits_t, dim=-1)
                suffix_log_prob += torch.log(probs_t[full_tokens[t_pos]] + 1e-30).item()

            stage2_data.append({
                "ID": ID,
                "Type": stereotype_key,
                "Candidate": pronoun_word,
                "Suffix_Log_Prob": suffix_log_prob,
            })

            del full_cache

    if stage1_data:
        s3_utils.write_csv(pd.DataFrame(stage1_data), pronoun_probs_path)
    if stage2_data:
        s3_utils.write_csv(pd.DataFrame(stage2_data), suffix_probs_path)

    return stage1_data, stage2_data


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
    agg_dict = {'Weighted_MLP': 'sum'}
    for col in weighted_head_cols:
        agg_dict[col] = 'sum'

    final_df = df.groupby(group_cols).agg(agg_dict).reset_index()

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


PRONOUN_PROBS_PATH = "outputs/gpt2-xl/winogender/pronoun_probs.csv"
SUFFIX_PROBS_PATH = "outputs/gpt2-xl/winogender/suffix_probs.csv"
ACC_PATH = "outputs/gpt2-xl/winogender/accumulated_impact_winogender.csv"


def run_baseline():
    print("Loading GPT-2 XL baseline...")
    model = HookedTransformer.from_pretrained("gpt2-xl")
    model.eval()

    print("Loading Winogender dataset from S3...")
    dataset = s3_utils.read_json("data/winogender/winogender_dataset.json")
    print(f"Loaded {len(dataset)} examples.")

    if TRACING:
        print("Starting two-stage tracing...")
        two_stage_tracing(model, dataset, PRONOUN_PROBS_PATH, SUFFIX_PROBS_PATH)
        print("Tracing complete.")

    if ACC_ANALYSIS:
        print("Computing accumulated impact from Stage 1 pronoun DLA...")
        result_df = accumulative_layer_impact(PRONOUN_PROBS_PATH)
        s3_utils.write_csv(result_df, ACC_PATH)
        print(f"Saved accumulated impact to S3: {ACC_PATH}")


def run_finetuned(run_id):
    s3_bucket = "modelsfinetuned"
    s3_prefix = "gpt2-xl-finetuned"

    s3_client = boto3.client(
        's3',
        aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
    )

    print(f"Loading GPT-2 XL and applying checkpoint for {run_id}...")
    model = HookedTransformer.from_pretrained("gpt2-xl")
    model.eval()

    log = s3_utils.read_json(f"outputs/gpt2-xl/fine_tuned/logs/{run_id}.json")
    best_epoch = log["best_epoch"] - 1

    checkpoint_key = f"{s3_prefix}/best_model_{run_id}_epoch_{best_epoch}.pt"
    local_tmp = f"checkpoints/{run_id}.pt"
    os.makedirs("checkpoints", exist_ok=True)

    print(f"Downloading checkpoint s3://{s3_bucket}/{checkpoint_key} ...")
    s3_client.download_file(s3_bucket, checkpoint_key, local_tmp)
    model.load_state_dict(torch.load(local_tmp, weights_only=True))
    os.remove(local_tmp)
    print("Checkpoint loaded.")

    dataset = s3_utils.read_json("data/winogender/winogender_dataset.json")
    print(f"Loaded {len(dataset)} examples.")

    ft_base = f"outputs/gpt2-xl/winogender/finetuned/{run_id}"
    ft_pronoun_path = f"{ft_base}/pronoun_probs.csv"
    ft_suffix_path = f"{ft_base}/suffix_probs.csv"
    ft_acc_path = f"{ft_base}/accumulated_impact_winogender.csv"

    if TRACING:
        print("Starting two-stage tracing on fine-tuned model...")
        two_stage_tracing(model, dataset, ft_pronoun_path, ft_suffix_path)
        print("Tracing complete.")

    if ACC_ANALYSIS:
        print("Computing accumulated impact for fine-tuned model...")
        result_df = accumulative_layer_impact(ft_pronoun_path)
        s3_utils.write_csv(result_df, ft_acc_path)
        print(f"Saved accumulated impact to S3: {ft_acc_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Winogender DLA tracing")
    parser.add_argument("--run_id", type=str, default=None,
                        help="Fine-tuned run ID. Omit for baseline evaluation.")
    args = parser.parse_args()

    if args.run_id:
        run_finetuned(args.run_id)
    else:
        run_baseline()
