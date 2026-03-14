"""Fine-tuning pipeline for Winogender bias mitigation.

Imports reusable training infrastructure from the StereoSet pipeline and
provides Winogender-specific component selection based on pronoun probability
gender direction + BLS occupation statistics.
"""
import os
from copy import deepcopy
from typing import List, Set

import pandas as pd
import torch
from dotenv import load_dotenv
from torch.utils.data import DataLoader, random_split
from transformer_lens import HookedTransformer

from experiments import s3_utils
from experiments.stereoset.stereoset_finetuning import (
    ExperimentConfig,
    DPODataset,
    ImprovedSFTDataset,
    configure_trainable_parameters,
    identify_top_impact_heads,
    identify_top_mlp_impact,
    identify_mlp_from_attn,
    run_training_dpo,
    run_training_sft_improved,
)

load_dotenv()

PRONOUN_PROBS_PATH = "outputs/gpt2-xl/winogender/pronoun_probs.csv"
ACC_IMPACT_PATH = "outputs/gpt2-xl/winogender/accumulated_impact_winogender.csv"
METADATA_PATH = "data/winogender/winogender_paired_metadata.json"

DPO_DATASET = "data/winogender/fine-tune-dpo/winogender_dpo.jsonl"
SFT_DATASET = "data/winogender/fine-tune-sft/winogender_sft.jsonl"
RESULTS_DIR = "outputs/gpt2-xl/winogender/fine_tuned/logs"
S3_PREFIX = "gpt2-xl-finetuned-winogender"

ALL_EXPERIMENT_TYPES = ["attn", "mlp_from_attn", "mlp_impact_only", "full"]
DEFAULT_PERCENTILES = [0.5, 0.8, 1.0, 5.0, 10.0]

N_LAYERS = 48
LAST_LAYER = N_LAYERS - 1


def winogender_impact_analysis_selection(
    df_impact: pd.DataFrame,
    df_probs: pd.DataFrame,
    metadata: list,
) -> pd.DataFrame:
    """Annotate accumulated-impact DF with a Model_Preference column.

    Uses baseline pronoun probabilities (occupation sentence, last layer) and
    BLS statistics to determine whether the model's preferred pronoun is
    stereotypical for each occupation.
    """
    bls_map = {m["id"]: m["bls_pct_female"] for m in metadata}

    occ_last = df_probs[
        (df_probs["Sentence_Role"] == "occupation")
        & (df_probs["Layer"] == LAST_LAYER)
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
    """Split dataset ensuring at least 1 sample in each partition."""
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
    """Run all experiment types x percentiles for the Winogender dataset."""
    if experiment_types is None:
        experiment_types = ALL_EXPERIMENT_TYPES
    if percentiles is None:
        percentiles = DEFAULT_PERCENTILES

    original_state_dict = deepcopy(model.state_dict())
    df_impact_analysis = winogender_impact_analysis_selection(
        df_impact, df_probs, metadata
    )

    all_results = {}
    seen_active_params: Set[int] = set()

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
            elif exp_type == "full":
                target_ids = df_impact_analysis[
                    df_impact_analysis["Model_Preference"] == "stereotype"
                ]["ID"].unique().tolist()

            if len(target_ids) == 0:
                print("No target examples found. Skipping.")
                continue

            target_components = []
            if exp_type == "attn":
                target_components = top_heads.index.tolist()
            elif exp_type in ("mlp_impact_only", "mlp_from_attn"):
                target_components = top_mlps.index.tolist()

            model, num_params, hook_handles = configure_trainable_parameters(
                model, target_components=target_components, condition=exp_type)

            if num_params in seen_active_params:
                print(f"SKIP: {num_params:,} active params already tested. "
                      f"Skipping {exp_type} @ {percentile}%.")
                _cleanup(model, hook_handles, original_state_dict)
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
                lr=config.learning_rate, weight_decay=0.0,
            )

            ref_model = HookedTransformer.from_pretrained("gpt2-xl")
            for param in ref_model.parameters():
                param.requires_grad = False

            if config.loss_type == "dpo":
                run_id = (f"wino_dpo_{exp_type}_{percentile}"
                          f"_beta{config.dpo_beta}_lr{config.learning_rate}")
            else:
                run_id = (f"wino_sft_{exp_type}_{percentile}"
                          f"_ul{config.ul_weight}_lr{config.learning_rate}")

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
                        dpo_val_set, batch_size=config.batch_size, shuffle=False)

                result = run_training_sft_improved(
                    model, ref_model, train_loader, val_loader,
                    val_dpo_loader, optimizer, run_config,
                    run_id=run_id, num_params=num_params,
                )

            else:
                raise ValueError(f"Unknown loss_type: {config.loss_type}")

            all_results[(exp_type, percentile)] = result

            print("Cleaning up hooks and resetting weights ...")
            _cleanup(model, hook_handles, original_state_dict)

    summary_path = f"{config.results_dir}/all_experiment_results.json"
    serializable = {f"{k[0]}_{k[1]}": v for k, v in all_results.items()}
    s3_utils.write_json(serializable, summary_path)
    print(f"\nSaved summary to S3 ({summary_path})")

    return all_results


if __name__ == "__main__":
    print("Loading GPT-2 XL ...")
    model = HookedTransformer.from_pretrained("gpt2-xl")
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
            **kwargs,
        )

    for beta in [0.3, 0.5]:
        for lr in ALL_LRS:
            exp_types = [t for t in ALL_EXPERIMENT_TYPES if t != "full"]
            if lr in FULL_LRS:
                exp_types = ALL_EXPERIMENT_TYPES
            print(f"\n{'#' * 60}\n# DPO: beta={beta}, lr={lr}\n{'#' * 60}")
            cfg = _make_config(loss_type="dpo", dpo_beta=beta, learning_rate=lr)
            run_all_experiments_winogender(
                model, tokenizer, df_impact, df_probs, metadata, cfg,
                experiment_types=exp_types,
            )

    for ul_w in [0.5, 1.0]:
        for lr in ALL_LRS:
            exp_types = [t for t in ALL_EXPERIMENT_TYPES if t != "full"]
            if lr in FULL_LRS:
                exp_types = ALL_EXPERIMENT_TYPES
            print(f"\n{'#' * 60}\n# SFT: ul_weight={ul_w}, lr={lr}\n{'#' * 60}")
            cfg = _make_config(
                loss_type="sft_improved", ul_weight=ul_w, learning_rate=lr)
            run_all_experiments_winogender(
                model, tokenizer, df_impact, df_probs, metadata, cfg,
                experiment_types=exp_types,
            )
