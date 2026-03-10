import os
import json
import boto3
import torch
import tempfile
import numpy as np
import pandas as pd
from copy import deepcopy
from typing import List, Tuple, Dict, Optional, Any, Set
from dataclasses import dataclass, field
from torch.utils.data import Dataset, DataLoader, random_split
from torch import nn
from dotenv import load_dotenv

import s3_utils

# try:
from transformer_lens import HookedTransformer
# except ImportError:
#     HookedTransformer = Any

load_dotenv()

@dataclass
class ExperimentConfig:
    """Central configuration for the fine-tuning experiments."""
    # Dataset paths
    fine_tune_dataset: str = "data/stereoset/fine-tune-sft/sft_bias_mitigation_v2.jsonl"
    dpo_dataset: str = "data/stereoset/fine-tune-dpo/dpo_pairs_triplet.jsonl"
    train_file_path: str = "data/stereoset/splits/gender_train.json"

    # Infrastructure
    s3_bucket: str = "modelsfinetuned"
    s3_prefix: str = "gpt2-xl-finetuned"
    checkpoint_dir: str = "../checkpoints"
    results_dir: str = "outputs/gpt2-xl/fine_tuned/logs"

    # Training hyperparameters
    batch_size: int = 4
    learning_rate: float = 1e-5
    num_epochs: int = 5
    patience: int = 5
    max_token_length: int = 48

    # Loss configuration: "dpo" or "sft_improved"
    loss_type: str = "dpo"
    dpo_beta: float = 0.1
    ul_weight: float = 1.0

    # Experiment sweep
    percentiles: List[int] = field(default_factory=lambda: [100])
    bias_type: str = 'gender'
    experiment_type: str = "full"

    def __post_init__(self):
        os.makedirs(self.checkpoint_dir, exist_ok=True)

def save_checkpoint(model, s3_client=None, s3_bucket=None, s3_key=None):
    """Saves locally and optionally uploads to S3."""

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pt") as tmp:
        torch.save(model.state_dict(), tmp.name)

        try:
            s3_client.upload_file(tmp.name, s3_bucket, s3_key)
            print(f"--> Uploaded to s3://{s3_bucket}/{s3_key}")
        except Exception as e:
            print(f"!! Failed to upload to s3://{s3_bucket}/{s3_key}: {e}")
        finally:
            os.remove(tmp.name)

def identify_top_impact_heads(
    df_impact: pd.DataFrame,
    df_probs: pd.DataFrame,
    df_impact_analysis: pd.DataFrame,
    percentile: float
) -> Tuple[pd.Series, List[str]]:

    head_df = df_impact_analysis[
        (df_impact_analysis['Model_Preference'] == 'stereotype') &
        (df_impact_analysis['Component'].str.startswith('Head'))
    ].copy()

    head_df['Head_ID'] = head_df['Layer'].astype(str) + "_" + head_df['Component']

    mean_impact = head_df.groupby('Head_ID')['Accumulated_Impact'].mean()

    threshold = mean_impact.quantile(1 - (percentile / 100))
    top_heads = mean_impact[mean_impact >= threshold].sort_values(ascending=False)

    target_ids = head_df['ID'].unique().tolist()

    return top_heads, target_ids

def identify_top_mlp_impact(
    df_impact: pd.DataFrame,
    df_probs: pd.DataFrame,
    df_impact_analysis: pd.DataFrame,
    percentile: float,
):

    mlp_df = df_impact_analysis[
        (df_impact_analysis['Model_Preference'] == 'stereotype') &
        (df_impact_analysis['Component'].str.startswith('MLP'))
    ].copy()

    mlp_df['MLP_ID'] = mlp_df['Layer'].astype(str) + "_" + mlp_df['Component']

    mean_impact = mlp_df.groupby('MLP_ID')['Accumulated_Impact'].mean()

    threshold = mean_impact.quantile(1 - (percentile / 100))
    top_mlps = mean_impact[mean_impact >= threshold].sort_values(ascending=False)

    target_ids = mlp_df['ID'].unique().tolist()

    return top_mlps, target_ids

def identify_top_mlp_probability(
    df_impact: pd.DataFrame,
    df_probs: pd.DataFrame,
    df_impact_analysis : pd.DataFrame,
    percentile: float,
) -> Tuple[pd.Series, List[str]]:

    mlp_df = df_impact_analysis[
        (df_impact_analysis['Model_Preference'] == 'stereotype') &
        (df_impact_analysis['Component'].str.startswith('MLP'))
    ].copy()

    mlp_df['MLP_ID'] = mlp_df['Layer'].astype(str) + "_" + mlp_df['Component']

    target_ids = mlp_df['ID'].unique().tolist()

    if not target_ids:
        return pd.Series(dtype=float), []


    idx_layer_last_token = df_probs.groupby(['ID', 'Type', 'Layer'])['Token_Position'].transform('max') == df_probs['Token_Position']

    stereo_probs = df_probs[
        (df_probs['ID'].isin(target_ids)) &
        (df_probs['Type'] == 'stereotype') &
        (idx_layer_last_token)
    ].copy()

    stereo_probs = stereo_probs.drop_duplicates(subset=['ID', 'Layer'])

    mean_layer_probs = stereo_probs.groupby('Layer')['Layer_Accumulated_Prob'].mean()

    mean_layer_probs.index = mean_layer_probs.index.astype(str) + "_MLP"

    threshold = mean_layer_probs.quantile(1 - (percentile / 100))
    top_mlps = mean_layer_probs[mean_layer_probs >= threshold].sort_values(ascending=False)

    return top_mlps, target_ids



def identify_mlp_from_attn(
    df_impact: pd.DataFrame,
    df_probs: pd.DataFrame,
    df_impact_analysis: pd.DataFrame,
    percentile: float,
) -> Tuple[pd.Series, List[str]]:
    """Selects MLP layers at the same layers where top-impact attention heads reside.
    Only MLP component IDs are returned -- attention heads stay frozen.
    """
    head_df = df_impact_analysis[
        (df_impact_analysis['Model_Preference'] == 'stereotype') &
        (df_impact_analysis['Component'].str.startswith('Head'))
    ].copy()
    head_df['Head_ID'] = head_df['Layer'].astype(str) + "_" + head_df['Component']
    mean_head_impact = head_df.groupby('Head_ID')['Accumulated_Impact'].mean()
    threshold = mean_head_impact.quantile(1 - (percentile / 100))
    top_heads = mean_head_impact[mean_head_impact >= threshold]
    top_layers = set(int(h.split('_')[0]) for h in top_heads.index)

    mlp_df = df_impact_analysis[
        (df_impact_analysis['Model_Preference'] == 'stereotype') &
        (df_impact_analysis['Component'] == 'MLP') &
        (df_impact_analysis['Layer'].isin(top_layers))
    ].copy()
    mlp_df['MLP_ID'] = mlp_df['Layer'].astype(str) + "_MLP"
    mlp_series = mlp_df.groupby('MLP_ID')['Accumulated_Impact'].mean().sort_values(ascending=False)

    target_ids = df_impact_analysis[
        df_impact_analysis['Model_Preference'] == 'stereotype'
    ]['ID'].unique().tolist()

    return mlp_series, target_ids


def df_impact_analysis_selection(
    df_impact: pd.DataFrame,
    df_probs: pd.DataFrame
):
    df_impact_analysis = df_impact.copy()

    max_layer = df_impact_analysis['Layer'].max()
    idx_last_token = df_probs.groupby(['ID', 'Type'])['Token_Position'].transform('max') == df_probs['Token_Position']

    final_probs = df_probs[
        (df_probs['Layer'] == max_layer) &
        (idx_last_token)
    ].copy()
    final_probs = final_probs.drop_duplicates(subset=['ID', 'Type'])

    prob_pivot = final_probs.pivot(index='ID', columns='Type', values='Layer_Accumulated_Prob')

    conditions = [
        (prob_pivot['stereotype'] > prob_pivot['anti-stereotype']) & (prob_pivot['stereotype'] > prob_pivot['unrelated']),
        (prob_pivot['anti-stereotype'] > prob_pivot['stereotype']) & (prob_pivot['anti-stereotype'] > prob_pivot['unrelated']),
        (prob_pivot['unrelated'] > prob_pivot['stereotype']) & (prob_pivot['unrelated'] > prob_pivot['anti-stereotype'])
    ]
    prob_pivot['Winner_Type'] = np.select(conditions, ['stereotype', 'anti-stereotype', 'unrelated'], default='tie')
    df_impact_analysis['Model_Preference'] = df_impact_analysis['ID'].map(prob_pivot['Winner_Type'].to_dict())

    return df_impact_analysis

import os
import json
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from typing import Optional, List


def _load_jsonl(json_path: str) -> list:
    """Loads a JSON or JSONL file from S3."""
    try:
        return s3_utils.read_jsonl(json_path)
    except Exception:
        return s3_utils.read_json(json_path)


def _setup_tokenizer(tokenizer):
    if tokenizer.padding_side != 'right':
        tokenizer.padding_side = 'right'
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _tokenize_pair(tokenizer, prompt: str, completion: str, max_length: int):
    """Tokenizes prompt+completion and returns ids, mask, and prompt length."""
    full_text = prompt + completion
    encoded_full = tokenizer(
        full_text, truncation=True, max_length=max_length,
        padding="max_length", return_tensors="pt"
    )
    encoded_prompt = tokenizer(
        prompt, truncation=True, max_length=max_length,
        add_special_tokens=False, return_tensors="pt"
    )
    return (
        encoded_full["input_ids"].squeeze(0),
        encoded_full["attention_mask"].squeeze(0),
        encoded_prompt["input_ids"].shape[1]
    )


class DPODataset(Dataset):
    """Dataset for DPO training with (prompt, chosen, rejected) triples."""

    def __init__(self, json_path: str, tokenizer,
                 target_ids: Optional[List[str]] = None,
                 max_length: int = 128):
        self.tokenizer = _setup_tokenizer(tokenizer)
        self.max_length = max_length
        self.data = []

        raw_data = _load_jsonl(json_path)
        target_ids_set = set(str(i) for i in target_ids) if target_ids else None

        print(f"Loading DPO data from {json_path}...")
        for item in raw_data:
            if target_ids_set is not None and 'id' in item:
                if str(item['id']) not in target_ids_set:
                    continue
            prompt = item.get('prompt', '')
            chosen = item.get('chosen', '')
            rejected = item.get('rejected', '')
            if prompt and chosen and rejected:
                self.data.append((prompt, chosen, rejected))

        print(f"Loaded {len(self.data)} DPO preference pairs.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        prompt, chosen, rejected = self.data[idx]

        chosen_ids, chosen_mask, prompt_len = _tokenize_pair(
            self.tokenizer, prompt, chosen, self.max_length)
        rejected_ids, rejected_mask, _ = _tokenize_pair(
            self.tokenizer, prompt, rejected, self.max_length)

        return {
            "chosen_ids": chosen_ids,
            "chosen_mask": chosen_mask,
            "rejected_ids": rejected_ids,
            "rejected_mask": rejected_mask,
            "prompt_length": prompt_len,
        }



class ImprovedSFTDataset(Dataset):
    """SFT dataset that also provides the stereotype completion for unlikelihood loss."""

    def __init__(self, json_path: str, tokenizer,
                 target_ids: Optional[List[str]] = None,
                 max_length: int = 128):
        self.tokenizer = _setup_tokenizer(tokenizer)
        self.max_length = max_length
        self.data = []

        raw_data = _load_jsonl(json_path)
        target_ids_set = set(str(i) for i in target_ids) if target_ids else None

        print(f"Loading improved SFT data from {json_path}...")
        for item in raw_data:
            if target_ids_set is not None and 'id' in item:
                if str(item['id']) not in target_ids_set:
                    continue
            prompt = item.get('prompt', '')
            completion = item.get('completion', '')
            stereo = item.get('stereotype_completion', '')
            if prompt and completion and stereo:
                self.data.append((prompt, completion, stereo))

        print(f"Loaded {len(self.data)} improved SFT examples.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        prompt, completion, stereo_completion = self.data[idx]

        input_ids, attention_mask, prompt_length = _tokenize_pair(
            self.tokenizer, prompt, completion, self.max_length)

        labels = input_ids.clone()
        labels[:prompt_length] = -100
        labels[attention_mask == 0] = -100

        stereo_ids, stereo_mask, _ = _tokenize_pair(
            self.tokenizer, prompt, stereo_completion, self.max_length)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "stereo_ids": stereo_ids,
            "stereo_mask": stereo_mask,
            "prompt_length": prompt_length,
        }



class DebiasingDataset(Dataset):
    def __init__(self, json_path: str, tokenizer,
                 target_ids: Optional[List[str]] = None,
                 max_length: int = 128):
        self.tokenizer = _setup_tokenizer(tokenizer)
        self.data = []
        self.max_length = max_length

        raw_data = _load_jsonl(json_path)
        target_ids_set = set(target_ids) if target_ids else None
        print(f"Loading data from {json_path}...")

        for item in raw_data:
            if target_ids_set is not None and 'id' in item:
                if str(item['id']) not in target_ids_set:
                    continue
            if 'prompt' in item and 'completion' in item:
                prompt_text = item['prompt']
                completion_text = item['completion']
                if prompt_text and completion_text:
                    self.data.append((prompt_text, completion_text))

        print(f"Loaded {len(self.data)} examples for fine-tuning.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        prompt, completion = self.data[idx]
        input_ids, attention_mask, prompt_length = _tokenize_pair(
            self.tokenizer, prompt, completion, self.max_length)
        labels = input_ids.clone()
        labels[:prompt_length] = -100
        labels[attention_mask == 0] = -100
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

def get_gradient_mask_hook(mask: torch.Tensor):
    """Creates a hook that applies a binary mask to gradients."""
    def hook(grad):
        return grad * mask
    return hook

def configure_trainable_parameters(
    model: HookedTransformer,
    target_components: List[str],
    condition: str = 'attn'
) -> Tuple[HookedTransformer, int, List[Any]]:
    """
    Freezes the model and selectively unfreezes specific attention heads or MLPs.
    Returns the model, the count of active parameters, and a list of hook handles.
    """
    if condition != 'full':
        for param in model.parameters():
            param.requires_grad = False

    attn_head_targets_by_layer = {}
    mlp_targets = set()

    if condition == 'attn':
        for item in target_components:
            parts = item.split('_')
            if len(parts) >= 3:
                layer_idx, head_idx = int(parts[0]), int(parts[2])
                attn_head_targets_by_layer.setdefault(layer_idx, []).append(head_idx)

    elif condition in ['mlp_impact_only', 'mlp_probability_only', 'mlp_from_attn']:
        for item in target_components:
            parts = item.split('_')
            mlp_targets.add(int(parts[0]))

    active_params_count = 0
    total_params = 0
    hook_handles = []
    n_heads = model.cfg.n_heads

    for name, param in model.named_parameters():
        total_params += param.numel()
        parts = name.split(".")
        if len(parts) < 3 or parts[0] != "blocks":
            continue
        try:
            layer_idx = int(parts[1])
        except ValueError:
            continue

        if condition == 'attn' and layer_idx in attn_head_targets_by_layer and "attn" in name:
            active_heads = attn_head_targets_by_layer[layer_idx]
            if param.shape[0] == n_heads:
                param.requires_grad = True
                mask = torch.zeros_like(param)
                mask[active_heads, ...] = 1.0
                handle = param.register_hook(get_gradient_mask_hook(mask))
                hook_handles.append(handle)
                params_per_head = param.numel() // n_heads
                active_params_count += params_per_head * len(active_heads)

        elif condition in ['mlp_impact_only', 'mlp_probability_only', 'mlp_from_attn'] and layer_idx in mlp_targets and "mlp" in name:
            param.requires_grad = True
            active_params_count += param.numel()

    print(f"\n--- Unfreezing Summary ({condition}) ---")
    if condition == 'attn':
        print(f"Targeted Layers (Attn): {list(attn_head_targets_by_layer)}")
    elif condition in ['mlp_impact_only', 'mlp_probability_only', 'mlp_from_attn']:
        print(f"Targeted Layers (MLP): {list(mlp_targets)}")
    if condition == 'full':
        active_params_count = total_params
    print(f"Active parameters: {active_params_count:,} / {total_params:,}\n")

    return model, active_params_count, hook_handles


def _get_s3_client(config):
    if config.s3_bucket:
        return boto3.client(
            's3',
            aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
            aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"]
        )
    return None


def _sequence_log_probs(logits, token_ids, mask):
    """Compute per-sequence sum of log-probs over masked completion tokens.

    Args:
        logits: [batch, seq_len, vocab] model output logits
        token_ids: [batch, seq_len] input token IDs
        mask: [batch, seq_len] binary mask, 1 for completion tokens to score

    Returns:
        [batch] sum of log-probs per sequence
    """
    log_probs = F.log_softmax(logits, dim=-1)
    # logits[:, t, :] predicts token_ids[:, t+1]
    gathered = torch.gather(log_probs[:, :-1, :], 2, token_ids[:, 1:].unsqueeze(-1)).squeeze(-1)
    completion_mask = mask[:, 1:].float()
    return (gathered * completion_mask).sum(dim=-1)


def compute_preference_accuracy(model, dataloader, ref_model=None, device="cpu"):
    """Computes the fraction of examples where model prefers chosen over rejected.

    Works with DPO dataloaders. If ref_model is provided, uses DPO-style
    log-ratio comparison; otherwise uses raw log-prob comparison.
    """
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in dataloader:
            chosen_ids = batch["chosen_ids"].to(device)
            chosen_mask = batch["chosen_mask"].to(device)
            rejected_ids = batch["rejected_ids"].to(device)
            rejected_mask = batch["rejected_mask"].to(device)
            prompt_length = batch["prompt_length"]

            bsz, seq_len = chosen_ids.shape
            pos = torch.arange(seq_len, device=device).unsqueeze(0).expand(bsz, -1)
            chosen_comp_mask = (pos >= prompt_length.to(device).unsqueeze(1)) & (chosen_mask.bool())
            rejected_comp_mask = (pos >= prompt_length.to(device).unsqueeze(1)) & (rejected_mask.bool())

            chosen_logits = model(chosen_ids)
            rejected_logits = model(rejected_ids)

            chosen_lp = _sequence_log_probs(chosen_logits, chosen_ids, chosen_comp_mask)
            rejected_lp = _sequence_log_probs(rejected_logits, rejected_ids, rejected_comp_mask)

            if ref_model is not None:
                ref_chosen_logits = ref_model(chosen_ids)
                ref_rejected_logits = ref_model(rejected_ids)
                ref_chosen_lp = _sequence_log_probs(ref_chosen_logits, chosen_ids, chosen_comp_mask)
                ref_rejected_lp = _sequence_log_probs(ref_rejected_logits, rejected_ids, rejected_comp_mask)
                margin = (chosen_lp - rejected_lp) - (ref_chosen_lp - ref_rejected_lp)
            else:
                margin = chosen_lp - rejected_lp

            correct += (margin > 0).sum().item()
            total += bsz

    return correct / total if total > 0 else 0.0



# DPO Training
def dpo_loss(policy_chosen_logps, policy_rejected_logps,
             ref_chosen_logps, ref_rejected_logps, beta):
    """Standard DPO loss from Rafailov et al. 2023."""
    log_ratio_policy = policy_chosen_logps - policy_rejected_logps
    log_ratio_ref = ref_chosen_logps - ref_rejected_logps
    losses = -F.logsigmoid(beta * (log_ratio_policy - log_ratio_ref))
    return losses.mean()


def run_training_dpo(
    model: HookedTransformer,
    ref_model: HookedTransformer,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    config: ExperimentConfig,
    run_id: str,
    num_params: int = 0
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    ref_model.to(device)
    ref_model.eval()

    best_val_loss = float('inf')
    best_epoch = 0
    patience_counter = 0
    s3_client = _get_s3_client(config)
    epoch_logs = []

    total_params = sum(p.numel() for p in model.parameters())

    print(f"Starting DPO training run: {run_id} (beta={config.dpo_beta})")

    for epoch in range(config.num_epochs):
        model.train()
        total_train_loss = 0.0

        for batch in train_loader:
            chosen_ids = batch["chosen_ids"].to(device)
            chosen_mask = batch["chosen_mask"].to(device)
            rejected_ids = batch["rejected_ids"].to(device)
            rejected_mask = batch["rejected_mask"].to(device)
            prompt_length = batch["prompt_length"]

            bsz, seq_len = chosen_ids.shape
            pos = torch.arange(seq_len, device=device).unsqueeze(0).expand(bsz, -1)
            chosen_comp_mask = (pos >= prompt_length.to(device).unsqueeze(1)) & (chosen_mask.bool())
            rejected_comp_mask = (pos >= prompt_length.to(device).unsqueeze(1)) & (rejected_mask.bool())

            optimizer.zero_grad()

            policy_chosen_logits = model(chosen_ids)
            policy_rejected_logits = model(rejected_ids)

            with torch.no_grad():
                ref_chosen_logits = ref_model(chosen_ids)
                ref_rejected_logits = ref_model(rejected_ids)

            policy_chosen_lp = _sequence_log_probs(policy_chosen_logits, chosen_ids, chosen_comp_mask)
            policy_rejected_lp = _sequence_log_probs(policy_rejected_logits, rejected_ids, rejected_comp_mask)
            ref_chosen_lp = _sequence_log_probs(ref_chosen_logits, chosen_ids, chosen_comp_mask)
            ref_rejected_lp = _sequence_log_probs(ref_rejected_logits, rejected_ids, rejected_comp_mask)

            loss = dpo_loss(
                policy_chosen_lp, policy_rejected_lp,
                ref_chosen_lp, ref_rejected_lp,
                config.dpo_beta
            )

            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)

        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                chosen_ids = batch["chosen_ids"].to(device)
                chosen_mask = batch["chosen_mask"].to(device)
                rejected_ids = batch["rejected_ids"].to(device)
                rejected_mask = batch["rejected_mask"].to(device)
                prompt_length = batch["prompt_length"]

                bsz, seq_len = chosen_ids.shape
                pos = torch.arange(seq_len, device=device).unsqueeze(0).expand(bsz, -1)
                chosen_comp_mask = (pos >= prompt_length.to(device).unsqueeze(1)) & (chosen_mask.bool())
                rejected_comp_mask = (pos >= prompt_length.to(device).unsqueeze(1)) & (rejected_mask.bool())

                pc_logits = model(chosen_ids)
                pr_logits = model(rejected_ids)
                rc_logits = ref_model(chosen_ids)
                rr_logits = ref_model(rejected_ids)

                val_loss = dpo_loss(
                    _sequence_log_probs(pc_logits, chosen_ids, chosen_comp_mask),
                    _sequence_log_probs(pr_logits, rejected_ids, rejected_comp_mask),
                    _sequence_log_probs(rc_logits, chosen_ids, chosen_comp_mask),
                    _sequence_log_probs(rr_logits, rejected_ids, rejected_comp_mask),
                    config.dpo_beta
                )
                total_val_loss += val_loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        pref_acc = compute_preference_accuracy(model, val_loader, ref_model, device)

        epoch_logs.append({
            "epoch": epoch + 1,
            "train_loss": round(avg_train_loss, 6),
            "val_loss": round(avg_val_loss, 6),
            "pref_acc": round(pref_acc, 6),
        })

        print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} | Pref Acc: {pref_acc:.2%}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch + 1
            patience_counter = 0
            s3_key = f"{config.s3_prefix}/best_model_{run_id}_epoch_{epoch}.pt"
            print(f"--> Improvement detected. Saving...")
            save_checkpoint(model, s3_client, config.s3_bucket, s3_key)
        else:
            patience_counter += 1
            if patience_counter >= config.patience:
                print("--> Early stopping triggered.")
                break

    result_dict = {
        "run_id": run_id,
        "loss_type": "dpo",
        "experiment_type": config.experiment_type,
        "percentile": config.percentiles[0] if len(config.percentiles) == 1 else config.percentiles,
        "active_parameters": num_params,
        "total_parameters": total_params,
        "active_pct": round((num_params / total_params) * 100, 4) if total_params > 0 else 0.0,
        "dpo_beta": config.dpo_beta,
        "learning_rate": config.learning_rate,
        "batch_size": config.batch_size,
        "best_epoch": best_epoch,
        "total_epochs": len(epoch_logs),
        "best_val_loss": round(best_val_loss, 6),
        "epochs": epoch_logs,
    }

    log_path = f"{config.results_dir}/{run_id}.json"
    s3_utils.write_json(result_dict, log_path)
    print(f"Saved training log to s3 ({log_path})")

    return result_dict



# Improved SFT Training (CE + Unlikelihood)

def run_training_sft_improved(
    model: HookedTransformer,
    ref_model: HookedTransformer,
    train_loader: DataLoader,
    val_loader: DataLoader,
    val_dpo_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    config: ExperimentConfig,
    run_id: str,
    num_params: int = 0
):
    """Improved SFT: cross-entropy on anti-stereotype + unlikelihood on stereotype.

    val_dpo_loader is a DPODataset loader used for preference accuracy tracking.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    ref_model.to(device)
    ref_model.eval()

    best_val_loss = float('inf')
    best_epoch = 0
    patience_counter = 0
    s3_client = _get_s3_client(config)
    loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
    epoch_logs = []

    total_params = sum(p.numel() for p in model.parameters())

    print(f"Starting improved SFT training run: {run_id} (ul_weight={config.ul_weight})")

    for epoch in range(config.num_epochs):
        model.train()
        total_train_loss = 0.0
        total_ce_loss = 0.0
        total_ul_loss = 0.0

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            stereo_ids = batch["stereo_ids"].to(device)
            stereo_mask = batch["stereo_mask"].to(device)
            prompt_length = batch["prompt_length"]

            optimizer.zero_grad()

            logits = model(input_ids)

            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            ce_loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )

            stereo_logits = model(stereo_ids)
            bsz, seq_len, vocab = stereo_logits.shape
            pos = torch.arange(seq_len, device=device).unsqueeze(0).expand(bsz, -1)
            comp_mask = (pos >= prompt_length.to(device).unsqueeze(1)) & (stereo_mask.bool())

            stereo_probs = F.softmax(stereo_logits[:, :-1, :], dim=-1)
            stereo_target_probs = torch.gather(
                stereo_probs, 2, stereo_ids[:, 1:].unsqueeze(-1)
            ).squeeze(-1)

            ul_mask = comp_mask[:, 1:].float()
            num_ul_tokens = ul_mask.sum().clamp(min=1.0)
            ul_loss = -(torch.log(1.0 - stereo_target_probs.clamp(max=0.999)) * ul_mask).sum() / num_ul_tokens

            loss = ce_loss + config.ul_weight * ul_loss

            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            total_ce_loss += ce_loss.item()
            total_ul_loss += ul_loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        avg_ce_loss = total_ce_loss / len(train_loader)
        avg_ul_loss = total_ul_loss / len(train_loader)

        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                labels = batch["labels"].to(device)
                logits = model(input_ids)
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                val_loss = loss_fct(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1)
                )
                total_val_loss += val_loss.item()

        avg_val_loss = total_val_loss / len(val_loader)

        pref_acc = 0.0
        if val_dpo_loader is not None:
            pref_acc = compute_preference_accuracy(model, val_dpo_loader, ref_model, device)

        epoch_logs.append({
            "epoch": epoch + 1,
            "train_loss": round(avg_train_loss, 6),
            "ce_loss": round(avg_ce_loss, 6),
            "ul_loss": round(avg_ul_loss, 6),
            "val_loss": round(avg_val_loss, 6),
            "pref_acc": round(pref_acc, 6),
        })

        print(f"Epoch {epoch+1} | Train: {avg_train_loss:.4f} (CE: {avg_ce_loss:.4f}, "
              f"UL: {avg_ul_loss:.4f}) | Val: {avg_val_loss:.4f} | Pref Acc: {pref_acc:.2%}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch + 1
            patience_counter = 0
            s3_key = f"{config.s3_prefix}/best_model_{run_id}_epoch_{epoch}.pt"
            print(f"--> Improvement detected. Saving...")
            save_checkpoint(model, s3_client, config.s3_bucket, s3_key)
        else:
            patience_counter += 1
            if patience_counter >= config.patience:
                print("--> Early stopping triggered.")
                break

    result_dict = {
        "run_id": run_id,
        "loss_type": "sft_improved",
        "experiment_type": config.experiment_type,
        "percentile": config.percentiles[0] if len(config.percentiles) == 1 else config.percentiles,
        "active_parameters": num_params,
        "total_parameters": total_params,
        "active_pct": round((num_params / total_params) * 100, 4) if total_params > 0 else 0.0,
        "ul_weight": config.ul_weight,
        "learning_rate": config.learning_rate,
        "batch_size": config.batch_size,
        "best_epoch": best_epoch,
        "total_epochs": len(epoch_logs),
        "best_val_loss": round(best_val_loss, 6),
        "epochs": epoch_logs,
    }

    log_path = f"{config.results_dir}/{run_id}.json"
    s3_utils.write_json(result_dict, log_path)
    print(f"Saved training log to s3 ({log_path})")

    return result_dict



def run_experiments(
    model: HookedTransformer,
    tokenizer,
    df_impact: pd.DataFrame,
    df_probability_info: pd.DataFrame,
    config: ExperimentConfig
):
    original_state_dict = deepcopy(model.state_dict())
    experiment_results = {}

    df_impact_analysis = df_impact_analysis_selection(df_impact, df_probability_info)

    for percentile in config.percentiles:
        print(f"\n{'='*40}\nRunning Experiment: Top {percentile}% Targets | Loss: {config.loss_type}\n{'='*40}")

        target_ids = []
        top_heads = pd.Series()
        top_mlps = pd.Series()

        if config.experiment_type == 'attn':
            top_heads, target_ids = identify_top_impact_heads(
                df_impact, df_probability_info, df_impact_analysis, percentile)
        if config.experiment_type == 'mlp_from_attn':
            top_mlps, target_ids = identify_mlp_from_attn(
                df_impact, df_probability_info, df_impact_analysis, percentile)
        if config.experiment_type == 'mlp_impact_only':
            top_mlps, target_ids = identify_top_mlp_impact(
                df_impact, df_probability_info, df_impact_analysis, percentile)
        if config.experiment_type == 'mlp_probability_only':
            top_mlps, target_ids = identify_top_mlp_probability(
                df_impact, df_probability_info, df_impact_analysis, percentile)
        if config.experiment_type == 'full':
            target_ids = df_impact_analysis[
                df_impact_analysis['Model_Preference'] == 'stereotype'
            ]['ID'].unique().tolist()

        if len(target_ids) == 0:
            print("No target examples found for this percentile. Skipping.")
            continue


        target_components = []
        if config.experiment_type == 'attn':
            target_components = top_heads.index.tolist()
        elif config.experiment_type in ['mlp_impact_only', 'mlp_probability_only', 'mlp_from_attn']:
            target_components = top_mlps.index.tolist()

        model, num_params, hook_handles = configure_trainable_parameters(
            model, target_components=target_components, condition=config.experiment_type)

        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=config.learning_rate, weight_decay=0.0)

        ref_model = HookedTransformer.from_pretrained("gpt2-xl")
        for param in ref_model.parameters():
            param.requires_grad = False


        if config.loss_type == "dpo":
            run_id = f"dpo_{config.experiment_type}_{percentile}_beta{config.dpo_beta}_lr{config.learning_rate}"
        else:
            run_id = f"sft_{config.experiment_type}_{percentile}_ul{config.ul_weight}_lr{config.learning_rate}"

        if config.loss_type == "dpo":
            dataset = DPODataset(
                config.dpo_dataset, tokenizer,
                target_ids=[str(i) for i in target_ids],
                max_length=config.max_token_length)

            if len(dataset) == 0:
                print("DPO dataset is empty after filtering. Skipping.")
                continue

            train_size = int(0.8 * len(dataset))
            val_size = len(dataset) - train_size
            train_set, val_set = random_split(
                dataset, [train_size, val_size],
                generator=torch.Generator().manual_seed(42))

            train_loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True)
            val_loader = DataLoader(val_set, batch_size=config.batch_size, shuffle=False)

            result = run_training_dpo(
                model, ref_model, train_loader, val_loader, optimizer,
                config, run_id=run_id, num_params=num_params)

        elif config.loss_type == "sft_improved":
            sft_dataset = ImprovedSFTDataset(
                config.fine_tune_dataset, tokenizer,
                target_ids=[str(i) for i in target_ids],
                max_length=config.max_token_length)

            if len(sft_dataset) == 0:
                print("SFT dataset is empty after filtering. Skipping.")
                continue

            train_size = int(0.8 * len(sft_dataset))
            val_size = len(sft_dataset) - train_size
            train_set, val_set = random_split(
                sft_dataset, [train_size, val_size],
                generator=torch.Generator().manual_seed(42))

            train_loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True)
            val_loader = DataLoader(val_set, batch_size=config.batch_size, shuffle=False)

            dpo_val_dataset = DPODataset(
                config.dpo_dataset, tokenizer,
                target_ids=[str(i) for i in target_ids],
                max_length=config.max_token_length)
            val_dpo_loader = None
            if len(dpo_val_dataset) > 0:
                _, dpo_val_set = random_split(
                    dpo_val_dataset,
                    [int(0.8 * len(dpo_val_dataset)),
                     len(dpo_val_dataset) - int(0.8 * len(dpo_val_dataset))],
                    generator=torch.Generator().manual_seed(42))
                val_dpo_loader = DataLoader(
                    dpo_val_set, batch_size=config.batch_size, shuffle=False)

            result = run_training_sft_improved(
                model, ref_model, train_loader, val_loader, val_dpo_loader,
                optimizer, config, run_id=run_id, num_params=num_params)

        else:
            raise ValueError(f"Unknown loss_type: {config.loss_type}")

        experiment_results[(config.experiment_type, percentile)] = result

        print("Cleaning up hooks and resetting weights...")
        for handle in hook_handles:
            handle.remove()
        hook_handles.clear()

        model.load_state_dict(original_state_dict)
        for param in model.parameters():
            param.requires_grad = True

    return experiment_results


ALL_EXPERIMENT_TYPES = ['attn', 'mlp_from_attn', 'mlp_impact_only', 'full']
DEFAULT_PERCENTILES = [0.5, 0.8, 1.0, 5.0, 10.0]


def run_all_experiments(
    model: HookedTransformer,
    tokenizer,
    df_impact: pd.DataFrame,
    df_probability_info: pd.DataFrame,
    config: ExperimentConfig,
    experiment_types: List[str] = None,
    percentiles: List[float] = None,
):
    """Runs all experiment types and percentiles, skipping duplicate active-param configs.

    For 'full' experiment type, only a single run is performed (ignores percentiles).
    A summary JSON is saved at the end with all results.
    """
    if experiment_types is None:
        experiment_types = ALL_EXPERIMENT_TYPES
    if percentiles is None:
        percentiles = DEFAULT_PERCENTILES

    original_state_dict = deepcopy(model.state_dict())
    df_impact_analysis = df_impact_analysis_selection(df_impact, df_probability_info)

    all_results = {}
    seen_active_params: Set[int] = set()

    for exp_type in experiment_types:
        pcts = [100] if exp_type == 'full' else percentiles

        for percentile in pcts:
            if exp_type == 'mlp_from_attn' and percentile == 0.5:
                continue
            print(f"\n{'='*60}")
            print(f"Experiment: {exp_type} | Percentile: {percentile}% | Loss: {config.loss_type}")
            print(f"{'='*60}")

            target_ids = []
            top_heads = pd.Series()
            top_mlps = pd.Series()

            if exp_type == 'attn':
                top_heads, target_ids = identify_top_impact_heads(
                    df_impact, df_probability_info, df_impact_analysis, percentile)
            elif exp_type == 'mlp_from_attn':
                top_mlps, target_ids = identify_mlp_from_attn(
                    df_impact, df_probability_info, df_impact_analysis, percentile)
            elif exp_type == 'mlp_impact_only':
                top_mlps, target_ids = identify_top_mlp_impact(
                    df_impact, df_probability_info, df_impact_analysis, percentile)
            elif exp_type == 'full':
                target_ids = df_impact_analysis[
                    df_impact_analysis['Model_Preference'] == 'stereotype'
                ]['ID'].unique().tolist()

            if len(target_ids) == 0:
                print("No target examples found. Skipping.")
                continue

            target_components = []
            if exp_type == 'attn':
                target_components = top_heads.index.tolist()
            elif exp_type in ['mlp_impact_only', 'mlp_from_attn']:
                target_components = top_mlps.index.tolist()

            model, num_params, hook_handles = configure_trainable_parameters(
                model, target_components=target_components, condition=exp_type)

            if num_params in seen_active_params:
                print(f"SKIP: {num_params:,} active parameters already tested. Skipping {exp_type} @ {percentile}%.")
                for handle in hook_handles:
                    handle.remove()
                hook_handles.clear()
                model.load_state_dict(original_state_dict)
                for param in model.parameters():
                    param.requires_grad = True
                continue

            seen_active_params.add(num_params)

            run_config = ExperimentConfig(
                loss_type=config.loss_type,
                dpo_beta=config.dpo_beta,
                ul_weight=config.ul_weight,
                learning_rate=config.learning_rate,
                batch_size=config.batch_size,
                num_epochs=config.num_epochs,
                patience=config.patience,
                max_token_length=config.max_token_length,
                fine_tune_dataset=config.fine_tune_dataset,
                dpo_dataset=config.dpo_dataset,
                s3_bucket=config.s3_bucket,
                s3_prefix=config.s3_prefix,
                checkpoint_dir=config.checkpoint_dir,
                results_dir=config.results_dir,
                percentiles=[percentile],
                experiment_type=exp_type,
                bias_type=config.bias_type,
            )

            optimizer = torch.optim.AdamW(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=config.learning_rate, weight_decay=0.0)

            ref_model = HookedTransformer.from_pretrained("gpt2-xl")
            for param in ref_model.parameters():
                param.requires_grad = False

            if config.loss_type == "dpo":
                run_id = f"dpo_{exp_type}_{percentile}_beta{config.dpo_beta}_lr{config.learning_rate}"
            else:
                run_id = f"sft_{exp_type}_{percentile}_ul{config.ul_weight}_lr{config.learning_rate}"

            if config.loss_type == "dpo":
                dataset = DPODataset(
                    config.dpo_dataset, tokenizer,
                    target_ids=[str(i) for i in target_ids],
                    max_length=config.max_token_length)

                if len(dataset) == 0:
                    print("DPO dataset is empty after filtering. Skipping.")
                    for handle in hook_handles:
                        handle.remove()
                    hook_handles.clear()
                    model.load_state_dict(original_state_dict)
                    for param in model.parameters():
                        param.requires_grad = True
                    continue

                train_size = int(0.8 * len(dataset))
                val_size = len(dataset) - train_size
                train_set, val_set = random_split(
                    dataset, [train_size, val_size],
                    generator=torch.Generator().manual_seed(42))

                train_loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True)
                val_loader = DataLoader(val_set, batch_size=config.batch_size, shuffle=False)

                result = run_training_dpo(
                    model, ref_model, train_loader, val_loader, optimizer,
                    run_config, run_id=run_id, num_params=num_params)

            elif config.loss_type == "sft_improved":
                sft_dataset = ImprovedSFTDataset(
                    config.fine_tune_dataset, tokenizer,
                    target_ids=[str(i) for i in target_ids],
                    max_length=config.max_token_length)

                if len(sft_dataset) == 0:
                    print("SFT dataset is empty after filtering. Skipping.")
                    for handle in hook_handles:
                        handle.remove()
                    hook_handles.clear()
                    model.load_state_dict(original_state_dict)
                    for param in model.parameters():
                        param.requires_grad = True
                    continue

                train_size = int(0.8 * len(sft_dataset))
                val_size = len(sft_dataset) - train_size
                train_set, val_set = random_split(
                    sft_dataset, [train_size, val_size],
                    generator=torch.Generator().manual_seed(42))

                train_loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True)
                val_loader = DataLoader(val_set, batch_size=config.batch_size, shuffle=False)

                dpo_val_dataset = DPODataset(
                    config.dpo_dataset, tokenizer,
                    target_ids=[str(i) for i in target_ids],
                    max_length=config.max_token_length)
                val_dpo_loader = None
                if len(dpo_val_dataset) > 0:
                    _, dpo_val_set = random_split(
                        dpo_val_dataset,
                        [int(0.8 * len(dpo_val_dataset)),
                         len(dpo_val_dataset) - int(0.8 * len(dpo_val_dataset))],
                        generator=torch.Generator().manual_seed(42))
                    val_dpo_loader = DataLoader(
                        dpo_val_set, batch_size=config.batch_size, shuffle=False)

                result = run_training_sft_improved(
                    model, ref_model, train_loader, val_loader, val_dpo_loader,
                    optimizer, run_config, run_id=run_id, num_params=num_params)

            else:
                raise ValueError(f"Unknown loss_type: {config.loss_type}")

            all_results[(exp_type, percentile)] = result

            print("Cleaning up hooks and resetting weights...")
            for handle in hook_handles:
                handle.remove()
            hook_handles.clear()

            model.load_state_dict(original_state_dict)
            for param in model.parameters():
                param.requires_grad = True

    summary_path = f"{config.results_dir}/all_experiment_results.json"
    serializable = {f"{k[0]}_{k[1]}": v for k, v in all_results.items()}
    s3_utils.write_json(serializable, summary_path)
    print(f"\nSaved summary of all experiments to s3 ({summary_path})")

    return all_results

if __name__ == "__main__":
    model = HookedTransformer.from_pretrained("gpt2-xl")
    tokenizer = model.tokenizer

    df_impact = s3_utils.read_csv("outputs/gpt2-xl/dev_tests/accumulated_impact_gender_train.csv")
    df_probs = s3_utils.read_csv("outputs/gpt2-xl/dev_tests/out_DLA_gender_train.csv")

    ALL_LRS = [1e-5, 5e-6, 1e-6]
    FULL_LRS = [5e-6, 1e-6]

    for beta in [0.3, 0.5]:
        for lr in ALL_LRS:
            exp_types = [t for t in ALL_EXPERIMENT_TYPES if t != 'full']
            if lr in FULL_LRS:
                exp_types = ALL_EXPERIMENT_TYPES
            print(f"\n{'#'*60}\n# DPO: beta={beta}, lr={lr}\n{'#'*60}")
            config_dpo = ExperimentConfig(loss_type="dpo", dpo_beta=beta, learning_rate=lr)
            run_all_experiments(model, tokenizer, df_impact, df_probs, config_dpo,
                                experiment_types=exp_types)

    for ul_w in [0.5, 1.0]:
        for lr in ALL_LRS:
            exp_types = [t for t in ALL_EXPERIMENT_TYPES if t != 'full']
            if lr in FULL_LRS:
                exp_types = ALL_EXPERIMENT_TYPES
            print(f"\n{'#'*60}\n# SFT: ul_weight={ul_w}, lr={lr}\n{'#'*60}")
            config_sft = ExperimentConfig(loss_type="sft_improved", ul_weight=ul_w, learning_rate=lr)
            run_all_experiments(model, tokenizer, df_impact, df_probs, config_sft,
                                experiment_types=exp_types)