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



def full_sentence_tracing(model, dataset, output_path):
    """Compute full-sentence probability and DLA at the pronoun position.

    For each example and each pronoun variant, tokenizes the full sentence,
    runs a single forward pass, and:
      - computes per-layer probability of the pronoun token (for DLA analysis)
      - computes full-sentence log-probability (sum of log P(t_i | t_<i))
      - records DLA (head/MLP contributions) at the pronoun token position
    """
    df_ids = []
    try:
        df = s3_utils.read_csv(output_path)
        df_ids = df['ID'].tolist()
    except Exception:
        pass

    all_data = []

    for idx, sub_dict in enumerate(dataset):
        if sub_dict['id'] in df_ids:
            all_data.append(df_ids.index(sub_dict['id']))
            continue

        print(f"Processing item {idx} ({sub_dict['id']})...")
        if len(all_data) % 10 == 0 and idx != 0:
            print("Saving intermediate results to S3...")
            df = pd.DataFrame(all_data)
            s3_utils.write_csv(df, output_path)

        ID = sub_dict['id']
        prefix = sub_dict['prefix']
        full_sentences = sub_dict['full_sentences']
        candidates = sub_dict['targets']

        prefix_tokens = model.tokenizer.encode(prefix)
        prefix_len = len(prefix_tokens)

        for stereotype_key, pronoun_word in candidates.items():
            full_sent = full_sentences[stereotype_key]
            full_tokens = model.tokenizer.encode(full_sent)

            pronoun_with_space = ' ' + pronoun_word
            pronoun_tokens = model.tokenizer.encode(pronoun_with_space)
            pronoun_len = len(pronoun_tokens)

            with torch.no_grad():
                _, cache = model.run_with_cache(full_sent)

            # Full-sentence log-probability (teacher-forced, from last layer)
            sentence_log_prob = 0.0
            for t_pos in range(1, len(full_tokens)):
                hidden_last_layer = cache[f"blocks.{model.cfg.n_layers - 1}.hook_resid_post"][0, t_pos - 1]
                logits_t = model.unembed(model.ln_final(hidden_last_layer))
                probs_t = torch.softmax(logits_t, dim=-1)
                sentence_log_prob += torch.log(probs_t[full_tokens[t_pos]] + 1e-30).item()

            # DLA at each layer, focused on pronoun token positions
            for tok_offset in range(pronoun_len):
                token_pos_in_sent = prefix_len + tok_offset
                if token_pos_in_sent >= len(full_tokens):
                    break
                target_token_id = full_tokens[token_pos_in_sent]

                # The position we read from the cache is token_pos_in_sent - 1
                # (the hidden state after processing the *previous* token predicts *this* token)
                cache_pos = token_pos_in_sent - 1 if token_pos_in_sent > 0 else 0

                layer_accumulated_probs = torch.ones(model.cfg.n_layers, device='cpu')

                for layer in range(model.cfg.n_layers):
                    hidden_state = cache[f"blocks.{layer}.hook_resid_post"][0, cache_pos]
                    normalized_resid = model.ln_final(hidden_state)
                    layer_logits = model.unembed(normalized_resid)
                    layer_probs = torch.softmax(layer_logits, dim=-1)

                    p_token = layer_probs[target_token_id].item()
                    layer_accumulated_probs[layer] *= p_token

                    # DLA: project head outputs and MLP output onto the target token direction
                    raw_unembed_dir = model.W_U[:, target_token_id]

                    attn_result = cache[f"blocks.{layer}.attn.hook_z"][0, cache_pos]
                    W_O = model.W_O[layer]
                    head_contributions = torch.einsum("hd, hdm, m -> h",
                                                      attn_result, W_O, raw_unembed_dir)

                    mlp_out = cache[f"blocks.{layer}.hook_mlp_out"][0, cache_pos]
                    mlp_contribution = torch.dot(mlp_out, raw_unembed_dir)

                    row = {
                        "ID": ID,
                        "Prompt": full_sent,
                        "Candidate": pronoun_word,
                        "Token_Str": model.to_string(target_token_id),
                        "Token_Position": tok_offset,
                        "Is_First_Token": (tok_offset == 0),
                        "Type": stereotype_key,
                        "Layer": layer,
                        "Layer_Accumulated_Prob": layer_accumulated_probs[layer].item(),
                        "Token_Instant_Prob": p_token,
                        "Sentence_Log_Prob": sentence_log_prob,
                        "MLP_Logit_Impact": mlp_contribution.item(),
                    }

                    for head_idx, score in enumerate(head_contributions):
                        row[f"Head_{head_idx}"] = score.item()

                    all_data.append(row)

    df = pd.DataFrame(all_data)
    s3_utils.write_csv(df, output_path)
    return all_data


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


DLA_PATH = "outputs/gpt2-xl/winogender/out_DLA_winogender.csv"
ACC_PATH = "outputs/gpt2-xl/winogender/accumulated_impact_winogender.csv"


def run_baseline():
    print("Loading GPT-2 XL baseline...")
    model = HookedTransformer.from_pretrained("gpt2-xl")
    model.eval()

    print("Loading Winogender dataset from S3...")
    dataset = s3_utils.read_json("data/winogender/winogender_dataset.json")
    print(f"Loaded {len(dataset)} examples.")

    if TRACING:
        print("Starting full-sentence DLA tracing...")
        full_sentence_tracing(model, dataset, DLA_PATH)
        print("Tracing complete.")

    if ACC_ANALYSIS:
        print("Computing accumulated impact...")
        result_df = accumulative_layer_impact(DLA_PATH)
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

    ft_dla_path = f"outputs/gpt2-xl/winogender/finetuned/{run_id}/out_DLA_winogender.csv"
    ft_acc_path = f"outputs/gpt2-xl/winogender/finetuned/{run_id}/accumulated_impact_winogender.csv"

    if TRACING:
        print("Starting full-sentence DLA tracing on fine-tuned model...")
        full_sentence_tracing(model, dataset, ft_dla_path)
        print("Tracing complete.")

    if ACC_ANALYSIS:
        print("Computing accumulated impact for fine-tuned model...")
        result_df = accumulative_layer_impact(ft_dla_path)
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
