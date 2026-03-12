import pandas as pd
import json
import torch
from transformer_lens import HookedTransformer
from huggingface_hub import login
import os
import random

from experiments import s3_utils

login(token=os.environ["HF_TOKEN"])

model = HookedTransformer.from_pretrained("gpt2-xl")

model.eval()

# rephrased_stereoset = json.load(open('data/stereoset/rephrased_stereoset.json'))
# print(f"Loaded {len(rephrased_stereoset)} examples.")
#
# rephrased_stereoset_synonyms = json.load(open('data/stereoset/rephrased_stereoset_synonyms.json'))
# print(f"Loaded {len(rephrased_stereoset)} examples.")

try:
    rephrased_stereoset = s3_utils.read_json('data/stereoset/gender_new_test_stereoset.json')
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

def get_logit_attribution(model, cache, target_token_id, layer):

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

    output_path = f"outputs/gpt2-xl/dev_tests/{output_filename}"

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

                current_prompt += model.to_string(token_id)

    df = pd.DataFrame(all_data)
    s3_utils.write_csv(df, output_path)
    return all_data

SPLITTING = False
TRACING = True
ACC_ANALYSIS = True

if __name__ == "__main__":
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
        test_file_path = "data/stereoset/splits/gender_test.json"
        print(f"Loading testing data from S3 ({test_file_path})...")
        test_data = s3_utils.read_json(test_file_path)
        print(f"Loaded {len(test_data)} testing examples.")

        print("Starting Tracing on Testing Data...")
        all_data = layer_tracing(test_data, "out_DLA_gender_test.csv")
        print("Tracing Complete.")

    if ACC_ANALYSIS:
        print("Starting Accumulation Analysis...")
        filename = "outputs/gpt2-xl/dev_tests/out_DLA_gender_test.csv"
        output_filename = "outputs/gpt2-xl/dev_tests/accumulated_impact_gender_test.csv"

        try:
            result_df = accumulative_layer_impact(filename)
            s3_utils.write_csv(result_df, output_filename)
            print(f"Done! Saved accumulated results to S3 ({output_filename})")
        except Exception as e:
            print(f"File {filename} not found on S3: {e}. Run TRACING first.")