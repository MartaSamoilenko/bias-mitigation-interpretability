import gc
import os
import json
import time
import boto3
import torch
import psutil
import tempfile
import numpy as np
import pandas as pd
import torch.nn.functional as F
from copy import deepcopy
from typing import Dict, FrozenSet, List, Optional, Set, Tuple, Any
from dataclasses import dataclass, field
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
from dotenv import load_dotenv
from huggingface_hub import login
from transformer_lens import HookedTransformer

from experiments.winogender import s3_utils

load_dotenv()
login(token=os.environ["HF_TOKEN"])

PRONOUN_PROBS_PATH = "outputs/llama3.2_1b/winogender/baseline/train/pronoun_probs.csv"
ACC_IMPACT_PATH = "outputs/llama3.2_1b/winogender/baseline/train/accumulated_impact.csv"
METADATA_PATH = "data/winogender/winogender_paired_metadata.json"

DPO_DATASET = "data/winogender/fine-tune-dpo/winogender_dpo.jsonl"
SFT_DATASET = "data/winogender/fine-tune-sft/winogender_sft.jsonl"
RESULTS_DIR = "outputs/llama3.2_1b/winogender/fine_tuned/logs"
S3_PREFIX = "experiments/outputs/llama3.2_1b/winogender/fine_tuned/checkpoints"

DLA_EXPERIMENT_TYPES = ["attn", "mlp_from_attn", "mlp_impact_only", "full"]
RANDOM_EXPERIMENT_TYPES = ["random_attn", "random_mlp"]
ALL_EXPERIMENT_TYPES = DLA_EXPERIMENT_TYPES + RANDOM_EXPERIMENT_TYPES
RANDOM_SEEDS = [42]
DEFAULT_PERCENTILES = [0.5, 0.8, 1.0, 5.0, 10.0]


@dataclass
class ExperimentConfig:
    fine_tune_dataset: str = "data/stereoset/fine-tune-sft/sft_bias_mitigation_v2.jsonl"
    dpo_dataset: str = "data/stereoset/fine-tune-dpo/dpo_pairs_triplet.jsonl"
    train_file_path: str = "data/stereoset/gender_test_rephrased.json"

    use_s3: bool = True
    s3_bucket: str = "modelsfinetuned"
    s3_prefix: str = "stereoset_experiments/outputs/gpt2-xl/fine_tuned_v2/checkpoints"
    checkpoint_dir: str = "../checkpoints"
    results_dir: str = "stereoset_experiments/outputs/gpt2-xl/fine_tuned_v2/logs"

    batch_size: int = 4
    learning_rate: float = 1e-5
    num_epochs: int = 5
    patience: int = 5
    max_token_length: int = 48

    loss_type: str = "dpo"
    dpo_beta: float = 0.1
    ul_weight: float = 1.0

    percentiles: List[int] = field(default_factory=lambda: [100])
    bias_type: str = 'gender'
    experiment_type: str = "full"

    def __post_init__(self):
        os.makedirs(self.checkpoint_dir, exist_ok=True)

def save_checkpoint(model, s3_client=None, s3_bucket=None, s3_key=None,
                     local_dir=None, local_name=None):
    if local_dir is not None:
        os.makedirs(local_dir, exist_ok=True)
        final_path = os.path.join(local_dir, local_name)
        tmp_path = final_path + ".tmp"
        torch.save(model.state_dict(), tmp_path)
        os.replace(tmp_path, final_path)
        print(f"--> Saved local checkpoint (best so far): {final_path}")
        return

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

def identify_mlp_from_attn(
    df_impact: pd.DataFrame,
    df_probs: pd.DataFrame,
    df_impact_analysis: pd.DataFrame,
    percentile: float,
) -> Tuple[pd.Series, List[str]]:
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


def generate_random_heads(n_heads: int, n_layers: int, heads_per_layer: int, seed: int = 42) -> List[str]:
    rng = np.random.default_rng(seed)
    all_heads = [f"{layer}_Head_{head}" for layer in range(n_layers) for head in range(heads_per_layer)]
    chosen = rng.choice(all_heads, size=min(n_heads, len(all_heads)), replace=False)
    return sorted(chosen.tolist())


def generate_random_mlps(n_mlps: int, n_layers: int, seed: int = 42) -> List[str]:
    rng = np.random.default_rng(seed)
    all_mlps = [f"{layer}_MLP" for layer in range(n_layers)]
    chosen = rng.choice(all_mlps, size=min(n_mlps, len(all_mlps)), replace=False)
    return sorted(chosen.tolist())


def _load_jsonl(json_path: str) -> list:
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
    full_text = prompt + completion
    encoded_full = tokenizer(
        full_text, truncation=True, max_length=max_length,
        padding="max_length", return_tensors="pt"
    )
    encoded_prompt = tokenizer(
        prompt, truncation=True, max_length=max_length,
        return_tensors="pt", add_special_tokens=True
    )
    return (
        encoded_full["input_ids"].squeeze(0),
        encoded_full["attention_mask"].squeeze(0),
        encoded_prompt["input_ids"].shape[1]
    )


class DPODataset(Dataset):
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
            "sample_idx": idx,
        }


class ImprovedSFTDataset(Dataset):
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


def get_gradient_mask_hook(mask: torch.Tensor):
    def hook(grad):
        return grad * mask
    return hook

def configure_trainable_parameters(
    model: HookedTransformer,
    target_components: List[str],
    condition: str = 'attn'
) -> Tuple[HookedTransformer, int, List[Any]]:
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
    n_kv_heads = getattr(model.cfg, 'n_key_value_heads', None) or n_heads
    is_gqa = n_kv_heads != n_heads
    shared_kv_params = 0

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
            elif is_gqa and param.shape[0] == n_kv_heads:
                param.requires_grad = True
                active_params_count += param.numel()
                shared_kv_params += param.numel()

        elif condition in ['mlp_impact_only', 'mlp_probability_only', 'mlp_from_attn'] and layer_idx in mlp_targets and "mlp" in name:
            param.requires_grad = True
            active_params_count += param.numel()

    print(f"\n--- Unfreezing Summary ({condition}) ---")
    if condition == 'attn':
        print(f"Targeted Layers (Attn): {list(attn_head_targets_by_layer)}")
        if is_gqa:
            print(f"GQA detected (n_kv_heads={n_kv_heads}): shared K/V params unfrozen: {shared_kv_params:,}")
    elif condition in ['mlp_impact_only', 'mlp_probability_only', 'mlp_from_attn']:
        print(f"Targeted Layers (MLP): {list(mlp_targets)}")
    if condition == 'full':
        active_params_count = total_params
    print(f"Active parameters: {active_params_count:,} / {total_params:,}\n")

    return model, active_params_count, hook_handles


def _get_s3_client(config):
    if config.use_s3 and config.s3_bucket:
        return boto3.client(
            's3',
            aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
            aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"]
        )
    return None


def _sequence_log_probs(logits, token_ids, mask):
    log_probs = F.log_softmax(logits, dim=-1)
    gathered = torch.gather(log_probs[:, :-1, :], 2, token_ids[:, 1:].unsqueeze(-1)).squeeze(-1)
    completion_mask = mask[:, 1:].float()
    return (gathered * completion_mask).sum(dim=-1)


def compute_preference_accuracy(model, dataloader, ref_model=None, device="cpu",
                                 ref_chosen_lps=None, ref_rejected_lps=None):
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

            if ref_chosen_lps is not None and "sample_idx" in batch:
                ref_chosen_lp = ref_chosen_lps[batch["sample_idx"]].to(device)
                ref_rejected_lp = ref_rejected_lps[batch["sample_idx"]].to(device)
                margin = (chosen_lp - rejected_lp) - (ref_chosen_lp - ref_rejected_lp)
            elif ref_model is not None:
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


def dpo_loss(policy_chosen_logps, policy_rejected_logps,
             ref_chosen_logps, ref_rejected_logps, beta):
    log_ratio_policy = policy_chosen_logps - policy_rejected_logps
    log_ratio_ref = ref_chosen_logps - ref_rejected_logps
    losses = -F.logsigmoid(beta * (log_ratio_policy - log_ratio_ref))
    return losses.mean()


def _collect_memory_stats():
    if torch.cuda.is_available():
        vram_allocated = round(torch.cuda.max_memory_allocated() / (1024 ** 2), 2)
        vram_reserved = round(torch.cuda.max_memory_reserved() / (1024 ** 2), 2)
    else:
        vram_allocated = None
        vram_reserved = None
    ram_rss = round(psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2), 2)
    return vram_allocated, vram_reserved, ram_rss


def _precompute_ref_log_probs(ref_model, dataset, batch_size, device):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    n = len(dataset)
    ref_chosen_lps = torch.zeros(n)
    ref_rejected_lps = torch.zeros(n)

    ref_model.eval()
    with torch.no_grad():
        for batch in loader:
            idxs = batch["sample_idx"]
            chosen_ids = batch["chosen_ids"].to(device)
            chosen_mask = batch["chosen_mask"].to(device)
            rejected_ids = batch["rejected_ids"].to(device)
            rejected_mask = batch["rejected_mask"].to(device)
            prompt_length = batch["prompt_length"]

            bsz, seq_len = chosen_ids.shape
            pos = torch.arange(seq_len, device=device).unsqueeze(0).expand(bsz, -1)
            chosen_comp_mask = (pos >= prompt_length.to(device).unsqueeze(1)) & (chosen_mask.bool())
            rejected_comp_mask = (pos >= prompt_length.to(device).unsqueeze(1)) & (rejected_mask.bool())

            ref_chosen_logits = ref_model(chosen_ids)
            ref_rejected_logits = ref_model(rejected_ids)
            ref_chosen_lps[idxs] = _sequence_log_probs(ref_chosen_logits, chosen_ids, chosen_comp_mask).cpu()
            ref_rejected_lps[idxs] = _sequence_log_probs(ref_rejected_logits, rejected_ids, rejected_comp_mask).cpu()

    return ref_chosen_lps, ref_rejected_lps


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
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    model.to(device)
    ref_model.to(device)
    ref_model.eval()

    base_dataset = getattr(train_loader.dataset, "dataset", train_loader.dataset)
    print("Pre-computing reference log-probs...")
    cached_ref_chosen, cached_ref_rejected = _precompute_ref_log_probs(
        ref_model, base_dataset, config.batch_size, device)
    ref_model.cpu()
    del ref_model
    gc.collect()
    torch.cuda.empty_cache()
    print("Reference model deleted, GPU memory freed.")

    best_val_loss = float('inf')
    best_epoch = 0
    patience_counter = 0
    s3_client = _get_s3_client(config)
    epoch_logs = []

    total_params = sum(p.numel() for p in model.parameters())

    print(f"Starting DPO training run: {run_id} (beta={config.dpo_beta})")
    run_start = time.perf_counter()

    for epoch in range(config.num_epochs):
        model.train()
        epoch_start = time.perf_counter()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
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

            policy_chosen_lp = _sequence_log_probs(policy_chosen_logits, chosen_ids, chosen_comp_mask)
            policy_rejected_lp = _sequence_log_probs(policy_rejected_logits, rejected_ids, rejected_comp_mask)
            ref_chosen_lp = cached_ref_chosen[batch["sample_idx"]].to(device)
            ref_rejected_lp = cached_ref_rejected[batch["sample_idx"]].to(device)

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

                val_loss = dpo_loss(
                    _sequence_log_probs(pc_logits, chosen_ids, chosen_comp_mask),
                    _sequence_log_probs(pr_logits, rejected_ids, rejected_comp_mask),
                    cached_ref_chosen[batch["sample_idx"]].to(device),
                    cached_ref_rejected[batch["sample_idx"]].to(device),
                    config.dpo_beta
                )
                total_val_loss += val_loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        pref_acc = compute_preference_accuracy(
            model, val_loader, ref_model=None, device=device,
            ref_chosen_lps=cached_ref_chosen, ref_rejected_lps=cached_ref_rejected)

        epoch_time = round(time.perf_counter() - epoch_start, 2)
        vram_alloc, vram_resv, ram_rss = _collect_memory_stats()

        epoch_logs.append({
            "epoch": epoch + 1,
            "train_loss": round(avg_train_loss, 6),
            "val_loss": round(avg_val_loss, 6),
            "pref_acc": round(pref_acc, 6),
            "epoch_time_sec": epoch_time,
            "peak_vram_allocated_mb": vram_alloc,
            "peak_vram_reserved_mb": vram_resv,
            "ram_rss_mb": ram_rss,
        })

        print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} | Pref Acc: {pref_acc:.2%}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch + 1
            patience_counter = 0
            print(f"--> Improvement detected. Saving...")
            if config.use_s3:
                s3_key = f"{config.s3_prefix}/best_model_{run_id}_epoch_{epoch}.pt"
                save_checkpoint(model, s3_client, config.s3_bucket, s3_key)
            else:
                save_checkpoint(model, local_dir=config.checkpoint_dir,
                                 local_name=f"best_model_{run_id}.pt")
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
        "total_training_time_sec": round(time.perf_counter() - run_start, 2),
        "peak_vram_allocated_mb": max((e["peak_vram_allocated_mb"] for e in epoch_logs if e["peak_vram_allocated_mb"] is not None), default=None),
        "peak_vram_reserved_mb": max((e["peak_vram_reserved_mb"] for e in epoch_logs if e["peak_vram_reserved_mb"] is not None), default=None),
        "epochs": epoch_logs,
    }

    log_path = f"{config.results_dir}/{run_id}.json"
    s3_utils.write_json(result_dict, log_path)
    print(f"Saved training log to s3 ({log_path})")

    return result_dict


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
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    model.to(device)
    ref_model.to(device)
    ref_model.eval()

    cached_ref_chosen = None
    cached_ref_rejected = None
    if val_dpo_loader is not None:
        dpo_base = getattr(val_dpo_loader.dataset, "dataset", val_dpo_loader.dataset)
        print("Pre-computing reference log-probs for DPO validation...")
        cached_ref_chosen, cached_ref_rejected = _precompute_ref_log_probs(
            ref_model, dpo_base, config.batch_size, device)
    ref_model.cpu()
    del ref_model
    gc.collect()
    torch.cuda.empty_cache()
    print("Reference model deleted, GPU memory freed.")

    best_val_loss = float('inf')
    best_epoch = 0
    patience_counter = 0
    s3_client = _get_s3_client(config)
    loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
    epoch_logs = []

    total_params = sum(p.numel() for p in model.parameters())

    print(f"Starting improved SFT training run: {run_id} (ul_weight={config.ul_weight})")
    run_start = time.perf_counter()

    for epoch in range(config.num_epochs):
        model.train()
        epoch_start = time.perf_counter()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
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
            pref_acc = compute_preference_accuracy(
                model, val_dpo_loader, ref_model=None, device=device,
                ref_chosen_lps=cached_ref_chosen, ref_rejected_lps=cached_ref_rejected)

        epoch_time = round(time.perf_counter() - epoch_start, 2)
        vram_alloc, vram_resv, ram_rss = _collect_memory_stats()

        epoch_logs.append({
            "epoch": epoch + 1,
            "train_loss": round(avg_train_loss, 6),
            "ce_loss": round(avg_ce_loss, 6),
            "ul_loss": round(avg_ul_loss, 6),
            "val_loss": round(avg_val_loss, 6),
            "pref_acc": round(pref_acc, 6),
            "epoch_time_sec": epoch_time,
            "peak_vram_allocated_mb": vram_alloc,
            "peak_vram_reserved_mb": vram_resv,
            "ram_rss_mb": ram_rss,
        })

        print(f"Epoch {epoch+1} | Train: {avg_train_loss:.4f} (CE: {avg_ce_loss:.4f}, "
              f"UL: {avg_ul_loss:.4f}) | Val: {avg_val_loss:.4f} | Pref Acc: {pref_acc:.2%}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch + 1
            patience_counter = 0
            print(f"--> Improvement detected. Saving...")
            if config.use_s3:
                s3_key = f"{config.s3_prefix}/best_model_{run_id}_epoch_{epoch}.pt"
                save_checkpoint(model, s3_client, config.s3_bucket, s3_key)
            else:
                save_checkpoint(model, local_dir=config.checkpoint_dir,
                                 local_name=f"best_model_{run_id}.pt")
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
        "total_training_time_sec": round(time.perf_counter() - run_start, 2),
        "peak_vram_allocated_mb": max((e["peak_vram_allocated_mb"] for e in epoch_logs if e["peak_vram_allocated_mb"] is not None), default=None),
        "peak_vram_reserved_mb": max((e["peak_vram_reserved_mb"] for e in epoch_logs if e["peak_vram_reserved_mb"] is not None), default=None),
        "epochs": epoch_logs,
    }

    log_path = f"{config.results_dir}/{run_id}.json"
    s3_utils.write_json(result_dict, log_path)
    print(f"Saved training log to s3 ({log_path})")

    return result_dict


def winogender_impact_analysis_selection(
    df_impact: pd.DataFrame,
    df_probs: pd.DataFrame,
    metadata: list,
    last_layer: int,
) -> pd.DataFrame:
    bls_map = {m["id"]: m["bls_pct_female"] for m in metadata}

    occ_last = df_probs[
        (df_probs["Sentence_Role"] == "occupation")
        & (df_probs["Layer"] == last_layer)
        & (df_probs["Is_First_Token"] == True)  # noqa: E712
    ].copy()

    pivot = occ_last.pivot_table(
        index="ID", columns="Gender", values="Token_Instant_Prob", aggfunc="first"
    )

    pref_map = {}
    for pair_id, row in pivot.iterrows():
        p_male = row.get("male", 0.0)
        p_female = row.get("female", 0.0)
        p_neutral = row.get("neutral", 0.0)

        winner = max(
            [("male", p_male), ("female", p_female), ("neutral", p_neutral)],
            key=lambda x: x[1],
        )[0]

        if winner == "neutral":
            pref_map[pair_id] = "neutral"
            continue

        bls = bls_map.get(pair_id, 50.0)
        male_dominated = bls < 50.0
        is_stereo = (male_dominated and winner == "male") or (
            not male_dominated and winner == "female"
        )
        pref_map[pair_id] = "stereotype" if is_stereo else "anti-stereotype"

    df_out = df_impact.copy()
    df_out["Model_Preference"] = df_out["ID"].map(pref_map)
    return df_out


def _safe_split(dataset, train_frac=0.8, seed=42):
    n = len(dataset)
    train_size = max(1, int(train_frac * n))
    val_size = n - train_size
    if val_size == 0:
        train_size -= 1
        val_size = 1
    return random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(seed),
    )


def _cleanup(model, hook_handles, original_state_dict):
    for handle in hook_handles:
        handle.remove()
    hook_handles.clear()
    model.load_state_dict(original_state_dict)
    for param in model.parameters():
        param.requires_grad = True


def run_all_experiments_winogender(
    model: HookedTransformer,
    tokenizer,
    df_impact: pd.DataFrame,
    df_probs: pd.DataFrame,
    metadata: list,
    config: ExperimentConfig,
    experiment_types: List[str] = None,
    percentiles: List[float] = None,
):
    if experiment_types is None:
        experiment_types = ALL_EXPERIMENT_TYPES
    if percentiles is None:
        percentiles = DEFAULT_PERCENTILES

    original_state_dict = deepcopy(model.state_dict())
    last_layer = model.cfg.n_layers - 1
    df_impact_analysis = winogender_impact_analysis_selection(
        df_impact, df_probs, metadata, last_layer
    )

    all_results = {}
    seen_configs: Set[Tuple[str, FrozenSet[str]]] = set()

    for exp_type in experiment_types:
        pcts = [100] if exp_type == "full" else percentiles

        for percentile in pcts:
            if exp_type == "mlp_from_attn" and percentile == 0.5:
                continue

            print(f"\n{'=' * 60}")
            print(f"Experiment: {exp_type} | Percentile: {percentile}% | "
                  f"Loss: {config.loss_type}")
            print(f"{'=' * 60}")

            top_heads = pd.Series()
            top_mlps = pd.Series()
            target_ids = []

            if exp_type == "attn":
                top_heads, target_ids = identify_top_impact_heads(
                    df_impact, df_probs, df_impact_analysis, percentile)
            elif exp_type == "mlp_from_attn":
                top_mlps, target_ids = identify_mlp_from_attn(
                    df_impact, df_probs, df_impact_analysis, percentile)
            elif exp_type == "mlp_impact_only":
                top_mlps, target_ids = identify_top_mlp_impact(
                    df_impact, df_probs, df_impact_analysis, percentile)
            elif exp_type == "random_attn":
                top_heads, target_ids = identify_top_impact_heads(
                    df_impact, df_probs, df_impact_analysis, percentile)
            elif exp_type == "random_mlp":
                top_mlps, target_ids = identify_top_mlp_impact(
                    df_impact, df_probs, df_impact_analysis, percentile)
            elif exp_type == "full":
                target_ids = df_impact_analysis[
                    df_impact_analysis["Model_Preference"] == "stereotype"
                ]["ID"].unique().tolist()

            if len(target_ids) == 0:
                print("No target examples found. Skipping.")
                continue

            seeds = RANDOM_SEEDS if exp_type in RANDOM_EXPERIMENT_TYPES else [None]

            for rand_seed in seeds:
                target_components = []
                if exp_type == "attn":
                    target_components = top_heads.index.tolist()
                elif exp_type in ("mlp_impact_only", "mlp_from_attn"):
                    target_components = top_mlps.index.tolist()
                elif exp_type == "random_attn":
                    target_components = generate_random_heads(
                        len(top_heads), n_layers=model.cfg.n_layers,
                        heads_per_layer=model.cfg.n_heads, seed=rand_seed)
                elif exp_type == "random_mlp":
                    target_components = generate_random_mlps(
                        len(top_mlps), n_layers=model.cfg.n_layers, seed=rand_seed)

                condition = ("attn" if exp_type == "random_attn"
                             else "mlp_impact_only" if exp_type == "random_mlp"
                             else exp_type)

                model, num_params, hook_handles = configure_trainable_parameters(
                    model, target_components=target_components, condition=condition)

                config_key = (exp_type, frozenset(target_components))
                if config_key in seen_configs:
                    print(f"SKIP: identical component config already tested. "
                          f"Skipping {exp_type} @ {percentile}%.")
                    _cleanup(model, hook_handles, original_state_dict)
                    continue

                seen_configs.add(config_key)

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
                    use_s3=config.use_s3,
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
                    lr=config.learning_rate, weight_decay=0.0,
                )

                ref_model = HookedTransformer.from_pretrained("meta-llama/Llama-3.2-1B")
                for param in ref_model.parameters():
                    param.requires_grad = False

                seed_suffix = f"_seed{rand_seed}" if rand_seed is not None else ""
                if config.loss_type == "dpo":
                    run_id = (f"wino_dpo_{exp_type}_{percentile}"
                              f"_beta{config.dpo_beta}_lr{config.learning_rate}"
                              f"{seed_suffix}")
                else:
                    run_id = (f"wino_sft_{exp_type}_{percentile}"
                              f"_ul{config.ul_weight}_lr{config.learning_rate}"
                              f"{seed_suffix}")

                if config.loss_type == "dpo":
                    dataset = DPODataset(
                        config.dpo_dataset, tokenizer,
                        target_ids=None,
                        max_length=config.max_token_length,
                    )

                    if len(dataset) == 0:
                        print("DPO dataset is empty. Skipping.")
                        _cleanup(model, hook_handles, original_state_dict)
                        continue

                    train_set, val_set = _safe_split(dataset)
                    train_loader = DataLoader(
                        train_set, batch_size=config.batch_size, shuffle=True)
                    val_loader = DataLoader(
                        val_set, batch_size=config.batch_size, shuffle=False)

                    result = run_training_dpo(
                        model, ref_model, train_loader, val_loader, optimizer,
                        run_config, run_id=run_id, num_params=num_params,
                    )

                elif config.loss_type == "sft_improved":
                    sft_dataset = ImprovedSFTDataset(
                        config.fine_tune_dataset, tokenizer,
                        target_ids=None,
                        max_length=config.max_token_length,
                    )

                    if len(sft_dataset) == 0:
                        print("SFT dataset is empty. Skipping.")
                        _cleanup(model, hook_handles, original_state_dict)
                        continue

                    train_set, val_set = _safe_split(sft_dataset)
                    train_loader = DataLoader(
                        train_set, batch_size=config.batch_size, shuffle=True)
                    val_loader = DataLoader(
                        val_set, batch_size=config.batch_size, shuffle=False)

                    dpo_val_dataset = DPODataset(
                        config.dpo_dataset, tokenizer,
                        target_ids=None,
                        max_length=config.max_token_length,
                    )
                    val_dpo_loader = None
                    if len(dpo_val_dataset) > 0:
                        _, dpo_val_set = _safe_split(dpo_val_dataset)
                        val_dpo_loader = DataLoader(
                            dpo_val_set, batch_size=config.batch_size,
                            shuffle=False)

                    result = run_training_sft_improved(
                        model, ref_model, train_loader, val_loader,
                        val_dpo_loader, optimizer, run_config,
                        run_id=run_id, num_params=num_params,
                    )

                else:
                    raise ValueError(f"Unknown loss_type: {config.loss_type}")

                all_results[(exp_type, percentile, rand_seed)] = result

                print("Cleaning up hooks and resetting weights ...")
                _cleanup(model, hook_handles, original_state_dict)

    summary_path = f"{config.results_dir}/all_experiment_results.json"
    serializable = {f"{k[0]}_{k[1]}_s{k[2]}": v for k, v in all_results.items()}
    s3_utils.write_json(serializable, summary_path)
    print(f"\nSaved summary to S3 ({summary_path})")

    return all_results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Winogender fine-tuning: DLA sweep, random ablation, or both.")
    parser.add_argument(
        "--mode", choices=["dla", "random", "all"], default="dla",
        help="dla = DLA hyper-param sweep (default); "
             "random = random-layer ablation for top configs; "
             "all = DLA then random.")
    parser.add_argument(
        "--no-s3", action="store_true",
        help="Save/load checkpoints and results on local disk instead of S3 "
             "(use when AWS credentials/S3 access are unavailable).")
    args = parser.parse_args()

    s3_utils.set_use_s3(not args.no_s3)

    TOP_N_CONFIGS = [
        {"percentile": 1.0, "loss_type": "dpo", "dpo_beta": 0.3, "learning_rate": 1e-6, "experiment_type": "attn"},
        {"percentile": 0.5, "loss_type": "sft_improved", "ul_weight": 1.0, "learning_rate": 1e-5, "experiment_type": "mlp_impact_only"},
        {"percentile": 0.8, "loss_type": "dpo", "dpo_beta": 0.3, "learning_rate": 1e-6, "experiment_type": "attn"},
        {"percentile": 10.0, "loss_type": "sft_improved", "ul_weight": 0.5, "learning_rate": 1e-6, "experiment_type": "attn"},
        {"percentile": 0.5, "loss_type": "sft_improved", "ul_weight": 1.0, "learning_rate": 5e-6, "experiment_type": "mlp_impact_only"},
    ]

    print("Loading model ...")
    model = HookedTransformer.from_pretrained("meta-llama/Llama-3.2-1B")
    tokenizer = model.tokenizer

    print("Loading Winogender DLA data from S3 ...")
    df_impact = s3_utils.read_csv(ACC_IMPACT_PATH)
    df_probs = s3_utils.read_csv(PRONOUN_PROBS_PATH)
    metadata = s3_utils.read_json(METADATA_PATH)

    ALL_LRS = [1e-5, 5e-6, 1e-6]
    FULL_LRS = [5e-6, 1e-6]

    def _make_config(**kwargs):
        return ExperimentConfig(
            fine_tune_dataset=SFT_DATASET,
            dpo_dataset=DPO_DATASET,
            results_dir=RESULTS_DIR,
            s3_prefix=S3_PREFIX,
            max_token_length=64,
            batch_size=2,
            use_s3=not args.no_s3,
            **kwargs,
        )

    def run_dla_sweep():
        for beta in [0.3, 0.5]:
            for lr in ALL_LRS:
                exp_types = [t for t in DLA_EXPERIMENT_TYPES if t != "full"]
                if lr in FULL_LRS:
                    exp_types = DLA_EXPERIMENT_TYPES
                print(f"\n{'#' * 60}\n# DPO: beta={beta}, lr={lr}\n{'#' * 60}")
                cfg = _make_config(loss_type="dpo", dpo_beta=beta, learning_rate=lr)
                run_all_experiments_winogender(
                    model, tokenizer, df_impact, df_probs, metadata, cfg,
                    experiment_types=exp_types,
                )

        for ul_w in [0.5, 1.0]:
            for lr in ALL_LRS:
                exp_types = [t for t in DLA_EXPERIMENT_TYPES if t != "full"]
                if lr in FULL_LRS:
                    exp_types = DLA_EXPERIMENT_TYPES
                print(f"\n{'#' * 60}\n# SFT: ul_weight={ul_w}, lr={lr}\n{'#' * 60}")
                cfg = _make_config(
                    loss_type="sft_improved", ul_weight=ul_w, learning_rate=lr)
                run_all_experiments_winogender(
                    model, tokenizer, df_impact, df_probs, metadata, cfg,
                    experiment_types=exp_types,
                )

    def run_random_ablation():
        RANDOM_MAP = {
            "attn": "random_attn",
            "mlp_impact_only": "random_mlp",
            "mlp_from_attn": "random_mlp",
        }
        for i, cfg in enumerate(TOP_N_CONFIGS, 1):
            exp_type = cfg["experiment_type"]
            random_type = RANDOM_MAP[exp_type]
            hp_label = (f"beta={cfg['dpo_beta']}" if cfg["loss_type"] == "dpo"
                        else f"ul_weight={cfg['ul_weight']}")
            print(f"\n{'#'*60}")
            print(f"# Random ablation {i}/{len(TOP_N_CONFIGS)}: "
                  f"{random_type} (control for {exp_type}), "
                  f"percentile={cfg['percentile']}, "
                  f"{hp_label}, lr={cfg['learning_rate']}")
            print(f"{'#'*60}")
            config_kwargs = {
                "loss_type": cfg["loss_type"],
                "learning_rate": cfg["learning_rate"],
            }
            if cfg["loss_type"] == "dpo":
                config_kwargs["dpo_beta"] = cfg["dpo_beta"]
            else:
                config_kwargs["ul_weight"] = cfg["ul_weight"]
            config = _make_config(**config_kwargs)
            run_all_experiments_winogender(
                model, tokenizer, df_impact, df_probs, metadata, config,
                experiment_types=[random_type],
                percentiles=[cfg["percentile"]],
            )

    if args.mode in ("dla", "all"):
        run_dla_sweep()
    if args.mode in ("random", "all"):
        run_random_ablation()
