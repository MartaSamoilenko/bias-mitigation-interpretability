import pandas as pd
import json
import torch
from transformer_lens import HookedTransformer
from huggingface_hub import login
import boto3
import os


login(token=os.environ["HF_TOKEN"])

CONDITION = "full"
TRACING = True
ACC_ANALYSIS = True
PERCENTILE = [100]

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
    print("Loading CSV...")
    df = pd.read_csv(filename)

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

def layer_tracing(model,
                  percentile,
                  dataset):
    df_ids = []
    try:
        df = pd.read_csv(f"outputs/gpt2-xl/fine_tuned/out_DLA_gender_test_fine_tuned_{CONDITION}_{percentile}.csv")
        df_ids = df['ID'].tolist()
    except:
        pass

    all_data = []

    for idx, sub_dict in enumerate(dataset):
        if sub_dict['id'] in df_ids:
            all_data.append(df_ids.index(sub_dict['id']))
            continue

        print(f"Processing item {idx}...")
        if len(all_data) % 10 == 0 and id != 0:
            print("Saving intermediate results...")
            df = pd.DataFrame(all_data)
            df.to_csv(f"outputs/gpt2-xl/fine_tuned/out_DLA_gender_test_fine_tuned_{CONDITION}_{percentile}.csv", index=False)

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

                    # calculation of accumulated probability
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
                        "Layer_Accumulated_Prob": layer_accumulated_probs[layer].item(), # Prob of seq up to this token
                        "Token_Instant_Prob": p_token, # Prob of just this token
                        "MLP_Logit_Impact": mlp_contrib.item(),
                    }

                    for head_idx, score in enumerate(head_contribs):
                        row[f"Head_{head_idx}"] = score.item()

                    all_data.append(row)

                current_prompt += model.to_string(token_id)
    df = pd.DataFrame(all_data)
    df.to_csv(f"outputs/gpt2-xl/fine_tuned/out_DLA_gender_test_fine_tuned_{CONDITION}_{percentile}.csv", index=False)

    return all_data

def run_experiments_finetuned(percentile_list,
                              s3_bucket: str = "modelsfinetuned",
                              s3_prefix: str = "gpt2-xl-finetuned"
                            ):
    s3_client = boto3.client('s3',
                             aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
                             aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"])

    test_model = HookedTransformer.from_pretrained("gpt2-xl")
    test_model.eval()
    for percentile in percentile_list:

        fine_tuned_model = f"best_model_{CONDITION}_{percentile}"

        epoch = 10
        while True:
            try:
                s3_client.download_file(s3_bucket, f"{s3_prefix}/{fine_tuned_model}_epoch_{epoch}.pt", f"checkpoints/{fine_tuned_model}.pt")
                test_model.load_state_dict(torch.load(f"checkpoints/{fine_tuned_model}.pt"))
                os.remove(f"checkpoints/{fine_tuned_model}.pt")
                break
            except:
                epoch -= 1
                if epoch == -1:
                    raise Exception("Model not found in S3.")

        if TRACING:
            test_file_path = "data/stereoset/splits/gender_test.json"

            if not os.path.exists(test_file_path):
                print(f"Test file not found at {test_file_path}.")
                print("Please set SPLITTING = True to generate it first.")
            else:
                print(f"Loading testing data from {test_file_path}...")
                test_data = json.load(open(test_file_path))
                print(f"Loaded {len(test_data)} testing examples.")

                print("Starting Tracing on Testing Data...")
                layer_tracing(test_model, percentile, test_data)
                print("Tracing Complete.")

        if ACC_ANALYSIS:
            print("Starting Accumulation Analysis...")
            output_filename = f"outputs/gpt2-xl/fine_tuned/accumulated_impact_gender_test_fine_tuned_{CONDITION}_{percentile}.csv"
            filename = f"outputs/gpt2-xl/fine_tuned/out_DLA_gender_test_fine_tuned_{CONDITION}_{percentile}.csv"

            if os.path.exists(filename):
                result_df = accumulative_layer_impact(filename)
                result_df.to_csv(output_filename, index=False)
                print(f"Done! Saved accumulated results to {output_filename}")
            else:
                print(f"File {filename} not found. Run TRACING first.")

        print(f"Done analysis for percentile {percentile}! Saved accumulated results to {output_filename}.")

if __name__ == "__main__":
    run_experiments_finetuned(PERCENTILE)