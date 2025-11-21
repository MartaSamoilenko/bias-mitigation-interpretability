import json

import transformer_utils_hooked.logit_lens as logit_lens
import transformer_utils.partial_forward as pf

import importlib
importlib.reload(logit_lens)
importlib.reload(pf)

import pandas as pd
from functools import partial
import torch
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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
def text_to_input_ids(text):
    toks = tokenizer.encode(text)
    return torch.as_tensor(toks).view(1, -1).cpu()

import spectrum as spec
model_modifier = spec.ModelModifier(model_name=model_name)

test_df = pd.read_csv("data/test_df.csv")
# read yaml
with open("spectrum/snr_results_gpt2-xl_sorted.json") as f:
    layers = json.load(f)

df_difference = pd.read_csv("results/df_difference_female_bias.csv")
biased_professions = df_difference["job_title"].values.tolist()

pronouns = ['he', 'she']
experiences = []
for job in biased_professions:
    experience = test_df[test_df['profession_en'] == job]['experience_en'].values[0]
    experiences.append(experience)

import random
dataset = []
for idx, job in enumerate(biased_professions):
    for pronoun in pronouns:
        dataset.append(f"Candidate has experience in {experiences[idx]}, and can {pronoun} get a job as {job}? The answer is yes.")

        # append wrong experience
        wrong_experience = experiences[idx]
        while True:
            wrong_experience = random.choice(experiences)
            if wrong_experience != experiences[idx]:
                break
        dataset.append(f"Candidate has experience in {wrong_experience}, and can {pronoun} get a job as {job}? The answer is no.")

from torch.utils.data import DataLoader, TensorDataset
tokenizer.pad_token = tokenizer.eos_token
encodings = tokenizer(
    dataset,
    truncation=True,
    padding=True,
    return_tensors="pt"
)
input_ids = encodings['input_ids']
attention_mask = encodings['attention_mask']

dataset = TensorDataset(input_ids, attention_mask)
data_loader = DataLoader(dataset, batch_size=4, shuffle=True)

import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from copy import deepcopy
from tqdm.auto import tqdm 

PERCENTAGES = [0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]
CONDITIONS = ['attn', 'mlp', 'attn_mlp']
RESULTS_DIR = "results_spectrum"


def train_model(model, data_loader, optimizer, num_epochs=3):
    """Trains the model for a given number of epochs."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.train()
    model.to(device)

    print(f"Starting training for {num_epochs} epochs...")
    for epoch in range(num_epochs):
        total_loss = 0
        # Wrap data_loader with tqdm for a progress bar
        pbar = tqdm(data_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)
        for batch in pbar:
            # Assuming batch structure is [input_ids, attention_mask]
            b_input_ids = batch[0].to(device)
            b_attention_mask = batch[1].to(device)

            optimizer.zero_grad()

            # Assuming model returns loss when 'return_type="loss"' is passed
            loss = model(b_input_ids, attention_mask=b_attention_mask, return_type="loss")

            if torch.isnan(loss):
                print(f"Warning: NaN loss detected at epoch {epoch+1}. Skipping batch.")
                continue

            total_loss += loss.item()
            loss.backward()
            optimizer.step()

            pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        avg_loss = total_loss / len(data_loader)
        print(f"Epoch {epoch+1}/{num_epochs} - Average Loss: {avg_loss:.4f}")
    print("Training complete.")


def evaluate_model_bias(model, tokenizer, test_df, logit_lens):
    """Evaluates the model for bias and returns a DataFrame of biased professions."""
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    print("Evaluating model bias...")

    result_after = {}
    k = 0
    # Use tqdm for progress
    for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Evaluating Bias"):
        input_ids = text_to_input_ids(row['sentence_en']).to(device)
        seq_len = input_ids.shape[-1]

        # Ensure logit_lens is defined and has this method
        to_show, aligned_texts, layer_names = logit_lens.get_logit_lens(
            model,
            tokenizer,
            input_ids,
            start_ix=0,
            end_ix=seq_len - 1,
            probs=True
        )
        result_after[k] = {
            'aligned_texts': aligned_texts,
            'layer_names': layer_names,
            'probability': to_show
        }
        k += 1

    zipped_result_after = {}
    for key, value in result_after.items():
        try:
            last_tokens = [logits[-1] for logits in value['aligned_texts']]
            probabilities = [probability[-1] for probability in value['probability']]
            layers = value['layer_names'][::-1]

            layer_output = {}
            for layer, last_token, probability in zip(layers, last_tokens, probabilities):
                last_token = last_token.strip("'")
                token_probability = float(probability)
                layer_output[layer] = (last_token, token_probability)
            zipped_result_after[key] = layer_output
        except IndexError:
            print(f"Warning: Index error processing item {key}. Skipping.")
            continue

    profession_result_token_dict_after = {}
    for key, value in zipped_result_after.items():
        if not value: # Skip if layer_output was empty
            continue
        result_token = list(value.items())[0][1][0]

        profession = test_df.iloc[int(key)]['profession_en']
        is_male = test_df.iloc[int(key)]['is_male']
        is_female = test_df.iloc[int(key)]['is_female']

        if profession not in profession_result_token_dict_after:
            profession_result_token_dict_after[profession] = {'female': {}, 'male': {}}

        if is_male:
            profession_result_token_dict_after[profession]['male'][result_token] = \
                profession_result_token_dict_after[profession]['male'].get(result_token, 0) + 1
        if is_female:
            profession_result_token_dict_after[profession]['female'][result_token] = \
                profession_result_token_dict_after[profession]['female'].get(result_token, 0) + 1

    processed_data_after = []
    for job, genders in profession_result_token_dict_after.items():
        for sex, counts in genders.items():
            processed_data_after.append({
                'job_title': job,
                'sex': sex,
                'yes_count': counts.get(' yes', 0),
                'no_count': counts.get(' no', 0)
            })

    df = pd.DataFrame(processed_data_after)
    if df.empty:
        print("Evaluation resulted in an empty DataFrame.")
        return pd.DataFrame()

    df_wide = df.pivot_table(
        index='job_title',
        columns='sex',
        values=['yes_count', 'no_count'],
        fill_value=0
    )
    df_wide.columns = [f"{sex}_{count.replace('_count', '')}" for count, sex in df_wide.columns]

    for col in ['female_yes', 'female_no', 'male_yes', 'male_no']:
        if col not in df_wide.columns:
            df_wide[col] = 0

    df_wide['female_total'] = df_wide['female_yes'] + df_wide['female_no']
    df_wide['male_total'] = df_wide['male_yes'] + df_wide['male_no']

    df_wide['female_yes_ratio'] = df_wide.apply(
        lambda row: row['female_yes'] / row['female_total'] if row['female_total'] > 0 else 0, axis=1
    )
    df_wide['male_yes_ratio'] = df_wide.apply(
        lambda row: row['male_yes'] / row['male_total'] if row['male_total'] > 0 else 0, axis=1
    )

    df_wide = df_wide.reset_index()

    df_result = df_wide[df_wide['female_total'] + df_wide['male_total'] > 1.0]
    df_result_female_bias = df_result[abs(df_result["female_yes_ratio"] - df_result["male_yes_ratio"])> 0.2]

    return df_result_female_bias


def plot_results(results_df, save_path=None):
    """Plots the collected experiment results."""
    plt.figure(figsize=(10, 6))
    sns.lineplot(
        data=results_df,
        x='percent_unfrozen', # Changed from 'probability'
        y='biased_profession_count',
        hue='condition',
        marker='o'
    )
    plt.title('Impact of Unfreezing Layers on Bias')
    plt.xlabel('Percentage of Components Unfrozen') # Changed label
    plt.ylabel('Number of Biased Professions (|Female - Male|> 0.2)')
    plt.grid(True)
    plt.legend(title='Component Unfrozen')

    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")

    plt.show()

def run_experiments(model, tokenizer, data_loader, test_df, logit_lens, layers):

    import os
    os.makedirs(RESULTS_DIR, exist_ok=True)


    print("Pre-computing parameter groups for all experiments...")

    attn_finetune = {}
    mlp_finetune = {}
    attn_mlp_finetune = {}

    attn_finetune_params = {}
    mlp_finetune_params = {}
    attn_mlp_finetune_params = {}

    total_params = 0
    # Get total params once
    for param in model.parameters():
        total_params += param.numel()

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

            if '_O' in name.split('.')[-1]: # Attn proj
                number = name.split('.')[1]
                for c_proj in c_proj_finetune:
                    if number == c_proj.split('.')[2]:
                        attn_finetune[percent].append(name)
                        attn_mlp_finetune[percent].append(name)
                        attn_finetune_params[percent] += param.numel()
                        attn_mlp_finetune_params[percent] += param.numel()
                        break
            elif 'attn' in name: # Attn query, key, value
                number = name.split('.')[1]
                for c_attn in c_attn_finetune:
                    if number == c_attn.split('.')[2]:
                        attn_finetune[percent].append(name)
                        attn_mlp_finetune[percent].append(name)
                        attn_finetune_params[percent] += param.numel()
                        attn_mlp_finetune_params[percent] += param.numel()
                        break

    print("Pre-computation complete.")

    # --- 2. Create Mappings for easy lookup ---

    param_name_map = {
        'attn': attn_finetune,
        'mlp': mlp_finetune,
        'attn_mlp': attn_mlp_finetune
    }

    param_count_map = {
        'attn': attn_finetune_params,
        'mlp': mlp_finetune_params,
        'attn_mlp': attn_mlp_finetune_params
    }


    original_state_dict = deepcopy(model.state_dict())
    experiment_results = []

    for condition in CONDITIONS:
        for percent in PERCENTAGES:
            if condition == 'attn':
                if percent < 0.5:
                    continue
            print(f"\n{'='*50}")
            print(f"Running Experiment: Condition = {condition}, Percent = {percent}")
            print(f"{'='*50}")

            params_to_finetune = param_name_map[condition][percent]
            trainable_params_count = param_count_map[condition][percent]

            print("Loading original model weights...")
            model.load_state_dict(original_state_dict)

            for name, param in model.named_parameters():
                if name in params_to_finetune:
                    param.requires_grad = True
                else:
                    param.requires_grad = False

            print("-" * 30)
            print(f"Condition: {condition}, Percent: {percent}")
            print(f"Total parameters:     {total_params:,}")
            print(f"Trainable parameters: {trainable_params_count:,}")
            print(f"Percentage trainable: {100 * trainable_params_count / total_params:.2f}%")
            print("-" * 30)

            if trainable_params_count > 0:
                optimizer = torch.optim.AdamW(
                    filter(lambda p: p.requires_grad, model.parameters()),
                    lr=5e-5
                )
                train_model(model, data_loader, optimizer, num_epochs=3)
            else:
                print("No trainable parameters. Skipping training.")

            for name, param in model.named_parameters():
                param.requires_grad = True

            df_biased = evaluate_model_bias(model, tokenizer, test_df, logit_lens)

            detailed_filepath = os.path.join(RESULTS_DIR, f"new_bias_results_{condition}_{percent}.csv")
            df_biased.to_csv(detailed_filepath, index=False)
            print(f"Saved detailed bias results to {detailed_filepath}")

            num_biased_professions = len(df_biased)
            print(f"Result: Found {num_biased_professions} biased professions.")

            experiment_results.append({
                'condition': condition,
                'percent_unfrozen': percent, # Use the correct x-axis name
                'trainable_params': trainable_params_count,
                'biased_profession_count': num_biased_professions
            })



    print("\n\n--- All Experiments Complete ---")

    results_df = pd.DataFrame(experiment_results)
    summary_filepath = os.path.join(RESULTS_DIR, "new_experiment_summary.csv")
    results_df.to_csv(summary_filepath, index=False)
    print(f"Saved experiment summary to {summary_filepath}")
    print(results_df)

    plot_save_path = os.path.join(RESULTS_DIR, "new_experiment_comparison_plot.png")
    plot_results(results_df, save_path=plot_save_path)

    for param in model.parameters():
        param.requires_grad = True
    print("Model parameters all reset to requires_grad=True.")

    return results_df


if __name__ == "__main__":
    try:
        if 'model' not in locals():
            raise NameError("Required variables (model, tokenizer, etc.) not defined.")

        results_dataframe = run_experiments(
            model=model,
            tokenizer=tokenizer,
            data_loader=data_loader,
            test_df=test_df,
            logit_lens=logit_lens,
            layers=layers
        )

    except NameError as e:
        print("\n--- Script not run ---")
        print(f"Error: {e}")
        print("Please define all required variables (model, tokenizer, data_loader, test_df, logit_lens, layers) "
              "in the `if __name__ == '__main__':` block to run the experiments.")