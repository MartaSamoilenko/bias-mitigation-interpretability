import json

import transformer_utils_hooked.logit_lens as logit_lens
import transformer_utils.partial_forward as pf

import importlib

import pandas as pd
from functools import partial

import torch
import transformers
from huggingface_hub import login
# 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
login(token="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")

from transformer_lens import HookedTransformer

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

test_df = pd.read_csv(
    "data/test_df.csv")

with open("results/profession_result_token_dict.json") as outfile:
    profession_result_token_dict = json.load(outfile)
df_difference = pd.read_csv("results/df_difference_female_bias.csv")
biased_professions = df_difference["job_title"].values.tolist()

pronouns = ['he', 'she']
experiences = []
test_df = pd.read_csv("data/test_df.csv")
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

PROBABILITES = [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95]
conditions = ['attn', 'mlp', 'attn_mlp']

import torch
from copy import deepcopy
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
import re

try:
    import lm_eval
    from lm_eval.models.huggingface import HFLM

    importlib.reload(lm_eval)
except ImportError:
    print("Warning: 'lm-eval' library not found. ")
    print("Please install it: pip install lm-eval")
    lm_eval = None

def get_layer_index_from_name(param_name):
    """
    Extracts layer index from parameter names like:
    - transformer.h.11.attn.c_attn.weight (GPT-2)
    - model.layers.11.self_attn.q_proj.weight (Llama)
    """
    # regex to find the integer surrounded by dots or at the start
    match = re.search(r'\.(\d+)\.', param_name)
    if match:
        return int(match.group(1))
    return None

def identify_biased_layers(zipped_result_json, test_df, biased_professions, probability):
    biased_layers = set()
    for idx, layer_output_prob in zipped_result_json.items():
        if test_df.iloc[int(idx)]['profession_en'] not in biased_professions:
            continue

        result_token = list(layer_output_prob.items())[0][1][0]

        for layer_name, output_prob in layer_output_prob.items():
            if output_prob[1] > probability and output_prob[0] == result_token:
                biased_layers.add(layer_name)
    return biased_layers


def setup_model_for_training(model, biased_layers, condition):
    for param in model.parameters():
        param.requires_grad = False

    total_params = 0
    trainable_params = 0

    print(f"Unfreezing parameters for condition: '{condition}'")
    for name, param in model.named_parameters():
        total_params += param.numel()

        if 'attn' in name and condition == 'mlp':
            continue
        elif 'mlp' in name and condition == 'attn':
            continue

        try:
            layer_num = name.split(".")[1]
            formatted_name = f"m{layer_num}"
        except IndexError:
            continue

        if any(layer_name in formatted_name for layer_name in biased_layers):
            param.requires_grad = True
            trainable_params += param.numel()

    print("-" * 30)
    print(f"Total parameters:     {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Percentage trainable: {100 * trainable_params / total_params:.2f}%")

    return total_params, trainable_params, 100 * trainable_params / total_params


def train_model(model, data_loader, optimizer, num_epochs=3):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.train()
    model.to(device)

    for epoch in range(num_epochs):
        total_loss = 0
        for batch in data_loader:
            b_input_ids = batch[0].to(device)
            b_attention_mask = batch[1].to(device)

            optimizer.zero_grad()

            loss = model(b_input_ids, attention_mask=b_attention_mask, return_type="loss")

            total_loss += loss.item()
            loss.backward()
            optimizer.step()

        avg_loss = total_loss / len(data_loader)
        print(f"Epoch {epoch+1}/{num_epochs} - Average Loss: {avg_loss:.4f}")

def evaluate_model_performance(model, tokenizer, tasks, batch_size=8):
    if lm_eval is None:
        print("lm-eval library not imported. Skipping performance evaluation.")
        return {}

    print(f"\n--- Running lm-evaluation-harness on tasks: {tasks} ---")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Manually moving model to device: {device}")
    model.to(device)

    try:
        results = lm_eval.simple_evaluate(
            model="hf-auto",  # Keep using hf-auto
            model_args={
                "pretrained": model,    # Pass the HookedTransformer object
                "tokenizer": tokenizer,
            },
            tasks=tasks,
            num_fewshot=0,
            batch_size=batch_size,
            device=None
        )

    except Exception as e:
        print(f"Error during lm-evaluation-harness run: {e}")
        print("Please ensure 'lm-eval' and 'transformers' libraries are installed and compatible.")
        return {"error": str(e)}

    print("lm-evaluation-harness results (raw):")
    print(json.dumps(results['results'], indent=2))

    flat_results = {}
    for task_name, metrics in results['results'].items():
        for metric_name, value in metrics.items():
            clean_metric = metric_name.split(",")[0]
            flat_results[f"harness_{task_name}_{clean_metric}"] = value

    print("lm-evaluation-harness results (flattened):")
    print(flat_results)

    return flat_results

def evaluate_model_bias(model, tokenizer, test_df, logit_lens):
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    result_after = {}
    k = 0
    for _, row in test_df.iterrows():
        input_ids = text_to_input_ids(row['sentence_en']).to(device)
        seq_len = input_ids.shape[-1]

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
        last_tokens = [logits[-1] for logits in value['aligned_texts']]
        probabilities = [probability[-1] for probability in value['probability']]
        layers = value['layer_names'][::-1]

        layer_output = {}
        for layer, last_token, probability in zip(layers, last_tokens, probabilities):
            last_token = last_token.strip("'")
            token_probability = float(probability)
            layer_output[layer] = (last_token, token_probability)
        zipped_result_after[key] = layer_output

    profession_result_token_dict_after = {}
    for key, value in zipped_result_after.items():
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
    df_result_bias = df_result[abs(df_result["female_yes_ratio"] - df_result["male_yes_ratio"]) > 0.1]

    return df_result_bias


def plot_results(results_df, save_path=None):
    plt.figure(figsize=(10, 6))
    sns.lineplot(
        data=results_df,
        x='probability',
        y='biased_profession_count',
        hue='condition',
        marker='o'
    )
    plt.title('Impact of Unfreezing Layers on Bias')
    plt.xlabel('Probability Threshold for Unfreezing')
    plt.ylabel('Number of Biased Professions (Female < Male)')
    plt.grid(True)
    plt.legend(title='Component Unfrozen')

    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")

    plt.show()



def run_experiments(model, tokenizer, data_loader, test_df, biased_professions,
                    zipped_result_json, conditions, probabilities, logit_lens,
                    text_to_input_ids, optimizer_class, optimizer_lr=5e-5,
                    harness_tasks=None, harness_batch_size=8):

    results_dir = "results"
    print(f"Results will be saved in '{results_dir}/' directory.")
    os.makedirs(results_dir, exist_ok=True)


    original_state_dict = deepcopy(model.state_dict())

    experiment_results = []

    for condition in conditions:
        for probability in probabilities:
            print(f"\n--- Running Experiment: Condition={condition}, Probability={probability} ---")

            print("Loading original model weights...")
            model.load_state_dict(original_state_dict)

            biased_layers = identify_biased_layers(
                zipped_result_json,
                test_df,
                biased_professions,
                probability
            )
            print(f"Found {len(biased_layers)} biased layers to target.")

            total_params, trainable_params_count, trainable_pct = setup_model_for_training(
                model,
                biased_layers,
                condition
            )

            if trainable_params_count > 0:
                optimizer = optimizer_class(
                    filter(lambda p: p.requires_grad, model.parameters()),
                    lr=optimizer_lr
                )
                train_model(model, data_loader, optimizer, num_epochs=3)
            else:
                print("No trainable parameters. Skipping training.")

            harness_results = {}
            if harness_tasks:
                harness_results = evaluate_model_performance(
                    model,
                    tokenizer,
                    harness_tasks,
                    harness_batch_size
                )
            else:
                print("Skipping lm-evaluation-harness: No tasks provided.")

            print("Evaluating model bias post-training...")
            for name, param in model.named_parameters():
                param.requires_grad = True
            df_biased = evaluate_model_bias(model, tokenizer, test_df, logit_lens)

            detailed_filepath = f"results/bias_results_{condition}_{probability}.csv"
            df_biased.to_csv(detailed_filepath, index=False)
            print(f"Saved detailed bias results to {detailed_filepath}")

            num_biased_professions = len(df_biased)
            print(f"Result: Found {num_biased_professions} biased professions.")

            experiment_data = {
                'condition': condition,
                'probability': probability,
                'trainable_params': trainable_params_count,
                'trainable_pct': trainable_pct,
                'biased_profession_count': num_biased_professions
            }

            experiment_data.update(harness_results)
            experiment_results.append(experiment_data)

    print("\n--- Experiments Complete. Resetting all model parameters to trainable. ---")
    for param in model.parameters():
        param.requires_grad = True

    results_df = pd.DataFrame(experiment_results)
    summary_filepath = "results/experiment_summary.csv"
    results_df.to_csv(summary_filepath, index=False)
    print(f"Saved experiment summary to {summary_filepath}")

    print("\n--- Experiment Summary ---")
    print(results_df)

    plot_save_path = "results/experiment_comparison_plot.png"
    plot_results(results_df, save_path=plot_save_path)

    return results_df

with open("results/gbem_ua_translated_test.json") as outfile:
    zipped_result_json = json.load(outfile)

try:
    # ['winogrande', 'hellaswag', 'arc_easy']
    EVAL_HARNESS_TASKS = None
    EVAL_HARNESS_BATCH_SIZE = 8

    results_dataframe = run_experiments(
        model=model,
        tokenizer=tokenizer,
        data_loader=data_loader,
        test_df=test_df,
        biased_professions=biased_professions,
        zipped_result_json=zipped_result_json,
        conditions=conditions,
        probabilities=PROBABILITES,
        logit_lens=logit_lens,
        text_to_input_ids=text_to_input_ids,
        optimizer_class=torch.optim.AdamW,
        optimizer_lr=5e-5,
        harness_tasks=EVAL_HARNESS_TASKS,
        harness_batch_size=EVAL_HARNESS_BATCH_SIZE
    )
except NameError as e:
    print(f"\nScript not run: A required variable is missing. Error: {e}")
    print("Please ensure all variables (model, tokenizer, data_loader, etc.) are defined to run the experiment.")
except Exception as e:
    print(f"\nAn error occurred during the experiment: {e}")