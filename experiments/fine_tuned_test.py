import pandas as pd
import json
import torch
from transformer_lens import HookedTransformer
from huggingface_hub import login
import boto3
import os
from dotenv import load_dotenv

import s3_utils

load_dotenv()

login(token=os.environ["HF_TOKEN"])

TRACING = True
ACC_ANALYSIS = True

def get_logit_attribution(model, cache, target_token_id, layer):

    raw_unembed_dir = model.W_U[:, target_token_id]
    target_unembed_dir = raw_unembed_dir

    attn_result = cache[f"blocks.{layer}.attn.hook_z"][0, -1]

    W_O = model.W_O[layer]

    # Head Contributions
    # h = heads, d = d_head, m = d_model
    # attn_result (h, d) * W_O (h, d, m) -> projects to residual stream (h, m)
    # ... * target_unembed_dir (m) -> projects to logit (h)
    head_contributions = torch.einsum("hd, hdm, m -> h",
                                      attn_result,
                                      W_O,
                                      target_unembed_dir)

    # MLP Contribution
    # Shape: [d_model]
    mlp_out = cache[f"blocks.{layer}.hook_mlp_out"][0, -1]
    mlp_contribution = torch.dot(mlp_out, target_unembed_dir)

    return head_contributions, mlp_contribution

def accumulative_layer_impact(filename):
    print("Loading CSV from S3...")
    df = s3_utils.read_csv(filename)

    df = df.sort_values(by=['ID', 'Candidate', 'Type', 'Layer', 'Token_Position'])

    print("Calculating Prefix Probabilities...")
    group_cols = ['ID', 'Candidate', 'Type', 'Layer']
    df['Prefix_Prob'] = df.groupby(group_cols)['Layer_Accumulated_Prob'].shift(1)

    # The shift operation creates NaNs for the first token in every group.
    # For k=1, the prefix probability is 1.0.
    df['Prefix_Prob'] = df['Prefix_Prob'].fillna(1.0)

    # Multiply the raw DLA (Instant impact) by the Prefix_Prob
    print("Calculating Weighted Impacts...")

    # For MLP
    df['Weighted_MLP'] = df['MLP_Logit_Impact'] * df['Prefix_Prob']

    # For Heads (dynamically find all Head columns)
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

    # Group and Sum
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

def layer_tracing(model, dataset, output_path):
    df_ids = []
    try:
        df = s3_utils.read_csv(output_path)
        df_ids = df['ID'].tolist()
    except:
        pass

    all_data = []

    for idx, sub_dict in enumerate(dataset):
        if sub_dict['id'] in df_ids:
            all_data.append(df_ids.index(sub_dict['id']))
            continue

        print(f"Processing item {idx}...")
        if len(all_data) % 10 == 0 and idx != 0:
            print("Saving intermediate results to S3...")
            df = pd.DataFrame(all_data)
            s3_utils.write_csv(df, output_path)

        ID = sub_dict['id']

        original_prompt = sub_dict['rephrased_context']
        candidates = sub_dict['targets']

        for stereotype_key, word in candidates.items():
            word_with_space = ' ' + word
            target_tokens = model.tokenizer.encode(word_with_space)

            current_prompt = original_prompt
            layer_accumulated_probs = torch.ones(model.cfg.n_layers, device='cpu')

            for token_pos, token_id in enumerate(target_tokens):

                with torch.no_grad():
                    _, cache = model.run_with_cache(current_prompt)

                for layer in range(model.cfg.n_layers):

                    hidden_state = cache[f"blocks.{layer}.hook_resid_post"][0, -1]

                    normalized_resid = model.ln_final(hidden_state)

                    layer_logits = model.unembed(normalized_resid)
                    layer_probs = torch.softmax(layer_logits, dim=-1)

                    p_token = layer_probs[token_id].item()

                    layer_accumulated_probs[layer] *= p_token

                    head_contribs, mlp_contrib = get_logit_attribution(
                        model, cache, token_id, layer
                    )

                    row = {
                        "ID": ID,
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

                current_prompt += model.to_string(token_id)
    df = pd.DataFrame(all_data)
    s3_utils.write_csv(df, output_path)

    return all_data

def run_experiments_finetuned(run_ids,
                              s3_bucket: str = "modelsfinetuned",
                              s3_prefix: str = "gpt2-xl-finetuned"):
    s3_client = boto3.client('s3',
                             aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
                             aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"])

    test_model = HookedTransformer.from_pretrained("gpt2-xl")
    test_model.eval()

    for run_id in run_ids:
        print(f"\n{'='*60}\nEvaluating run: {run_id}\n{'='*60}")

        log = s3_utils.read_json(f"outputs/gpt2-xl/fine_tuned/logs/{run_id}.json")
        best_epoch = log["best_epoch"] - 1

        checkpoint_key = f"{s3_prefix}/best_model_{run_id}_epoch_{best_epoch}.pt"
        local_tmp = f"checkpoints/{run_id}.pt"
        os.makedirs("checkpoints", exist_ok=True)

        print(f"Downloading checkpoint s3://{s3_bucket}/{checkpoint_key} ...")
        s3_client.download_file(s3_bucket, checkpoint_key, local_tmp)
        test_model.load_state_dict(torch.load(local_tmp, weights_only=True))
        os.remove(local_tmp)
        print("Checkpoint loaded.")

        results_base = f"outputs/gpt2-xl/fine_tuned/results/{run_id}"

        if TRACING:
            test_file_path = "data/stereoset/splits/gender_test.json"
            print(f"Loading testing data from S3 ({test_file_path})...")
            test_data = s3_utils.read_json(test_file_path)
            print(f"Loaded {len(test_data)} testing examples.")

            dla_path = f"{results_base}/out_DLA_gender_test.csv"
            print("Starting Tracing on Testing Data...")
            layer_tracing(test_model, test_data, dla_path)
            print("Tracing Complete.")

        if ACC_ANALYSIS:
            print("Starting Accumulation Analysis...")
            dla_path = f"{results_base}/out_DLA_gender_test.csv"
            acc_path = f"{results_base}/accumulated_impact_gender_test.csv"

            try:
                result_df = accumulative_layer_impact(dla_path)
                s3_utils.write_csv(result_df, acc_path)
                print(f"Done! Saved accumulated results to S3 ({acc_path})")
            except Exception as e:
                print(f"DLA file not found on S3: {e}. Run with TRACING=True first.")

        print(f"Done analysis for {run_id}!")


if __name__ == "__main__":
    log_keys = s3_utils.list_keys("outputs/gpt2-xl/fine_tuned/logs/")
    prefix = s3_utils.s3_key("outputs/gpt2-xl/fine_tuned/logs/")
    run_ids = [
        k[len(prefix):].replace(".json", "")
        for k in log_keys
        if k.endswith(".json") and "all_experiment" not in k
    ]
    print(f"Discovered {len(run_ids)} run(s): {run_ids}")
    run_experiments_finetuned(run_ids)