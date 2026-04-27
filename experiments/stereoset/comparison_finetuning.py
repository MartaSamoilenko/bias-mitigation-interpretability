import gc
import json
import os

import numpy as np
import pandas as pd
import torch
from copy import deepcopy
from typing import List, Tuple
from torch.utils.data import DataLoader, random_split
from dotenv import load_dotenv

from transformer_lens import HookedTransformer

from experiments.stereoset import s3_utils
from experiments.stereoset.stereoset_finetuning import (
    ExperimentConfig,
    df_impact_analysis_selection,
    DPODataset,
    ImprovedSFTDataset,
    run_training_dpo,
    run_training_sft_improved,
)

load_dotenv()


def load_snr_layer_ranking(
    snr_json_path: str, component_type: str,
) -> List[Tuple[int, float]]:
    with open(snr_json_path) as f:
        snr_data = json.load(f)

    if component_type == "attn":
        relevant = {
            k: v for k, v in snr_data.items()
            if ("attn" in k or "self_attn" in k)
            and "ln" not in k
            and "norm" not in k
        }
    elif component_type == "mlp":
        relevant = {k: v for k, v in snr_data.items() if "mlp" in k}
    else:
        raise ValueError(f"Unknown component_type: {component_type}")

    layer_snr: dict[int, list[float]] = {}
    for wtype_dict in relevant.values():
        for param_name, snr_val in wtype_dict.items():
            parts = param_name.split(".")
            layer_idx = int(next(p for p in parts if p.isdigit()))
            layer_snr.setdefault(layer_idx, []).append(snr_val)

    avg = [(layer, float(np.mean(vals))) for layer, vals in layer_snr.items()]
    avg.sort(key=lambda x: x[1], reverse=True)
    return avg


def select_top_snr_layers(
    snr_json_path: str, component_type: str, n_layers: int,
) -> List[int]:
    if component_type in ("attn", "mlp"):
        ranking = load_snr_layer_ranking(snr_json_path, component_type)
        return [layer for layer, _ in ranking[:n_layers]]

    attn_ranking = dict(load_snr_layer_ranking(snr_json_path, "attn"))
    mlp_ranking = dict(load_snr_layer_ranking(snr_json_path, "mlp"))
    all_layers = set(attn_ranking) | set(mlp_ranking)

    combined: list[tuple[int, float]] = []
    for layer in all_layers:
        scores = []
        if layer in attn_ranking:
            scores.append(attn_ranking[layer])
        if layer in mlp_ranking:
            scores.append(mlp_ranking[layer])
        combined.append((layer, float(np.mean(scores))))

    combined.sort(key=lambda x: x[1], reverse=True)
    return [layer for layer, _ in combined[:n_layers]]


def select_top_dla_attn_layers(
    df_impact_analysis: pd.DataFrame, n_layers: int,
) -> List[int]:
    head_df = df_impact_analysis[
        (df_impact_analysis["Model_Preference"] == "stereotype")
        & (df_impact_analysis["Component"].str.startswith("Head"))
    ]
    layer_impact = (
        head_df.groupby("Layer")["Accumulated_Impact"]
        .mean()
        .sort_values(ascending=False)
    )
    return layer_impact.head(n_layers).index.tolist()


def select_top_dla_mlp_layers(
    df_impact_analysis: pd.DataFrame, n_layers: int,
) -> List[int]:
    mlp_df = df_impact_analysis[
        (df_impact_analysis["Model_Preference"] == "stereotype")
        & (df_impact_analysis["Component"].str.startswith("MLP"))
    ]
    layer_impact = (
        mlp_df.groupby("Layer")["Accumulated_Impact"]
        .mean()
        .sort_values(ascending=False)
    )
    return layer_impact.head(n_layers).index.tolist()


def select_top_dla_both_layers(
    df_impact_analysis: pd.DataFrame, n_layers: int,
) -> List[int]:
    stereo = df_impact_analysis[
        df_impact_analysis["Model_Preference"] == "stereotype"
    ]
    layer_impact = (
        stereo.groupby("Layer")["Accumulated_Impact"]
        .mean()
        .sort_values(ascending=False)
    )
    return layer_impact.head(n_layers).index.tolist()



def configure_trainable_layers(
    model: HookedTransformer,
    layer_indices: List[int],
    mode: str,
) -> Tuple[HookedTransformer, int]:
    for param in model.parameters():
        param.requires_grad = False

    layer_set = set(layer_indices)
    active_params = 0
    total_params = 0

    for name, param in model.named_parameters():
        total_params += param.numel()
        parts = name.split(".")
        if parts[0] != "blocks" or not parts[1].isdigit():
            continue
        layer_idx = int(parts[1])
        if layer_idx not in layer_set:
            continue

        is_attn = "attn" in name
        is_mlp = "mlp" in name

        unfreeze = (
            (mode == "attn" and is_attn)
            or (mode == "mlp" and is_mlp)
            or (mode == "both" and (is_attn or is_mlp))
        )
        if unfreeze:
            param.requires_grad = True
            active_params += param.numel()

    print(f"\n--- Unfreezing Summary (layer-level, mode={mode}) ---")
    print(f"Selected layers: {sorted(layer_set)}")
    print(f"Active parameters: {active_params:,} / {total_params:,}\n")

    return model, active_params


def run_comparison_experiments(
    model: HookedTransformer,
    tokenizer,
    df_impact: pd.DataFrame,
    df_probs: pd.DataFrame,
    snr_json_path: str,
    config: ExperimentConfig,
    methods: List[str],
    modes: List[str],
    layer_counts: List[int],
):
    original_state_dict = deepcopy(model.state_dict())
    df_analysis = df_impact_analysis_selection(df_impact, df_probs)

    target_ids = df_analysis[
        df_analysis["Model_Preference"] == "stereotype"
    ]["ID"].unique().tolist()

    if len(target_ids) == 0:
        print("No stereotype target examples found. Aborting.")
        return {}

    all_results: dict = {}

    for method in methods:
        for mode in modes:
            for n_layers in layer_counts:
                print(f"\n{'=' * 60}")
                print(
                    f"Comparison: {method.upper()} | mode={mode} | "
                    f"layers={n_layers} | loss={config.loss_type}"
                )
                print(f"{'=' * 60}")

                if method == "dla":
                    if mode == "attn":
                        layers = select_top_dla_attn_layers(df_analysis, n_layers)
                    elif mode == "mlp":
                        layers = select_top_dla_mlp_layers(df_analysis, n_layers)
                    else:
                        layers = select_top_dla_both_layers(df_analysis, n_layers)
                else:
                    layers = select_top_snr_layers(snr_json_path, mode, n_layers)

                model, num_params = configure_trainable_layers(model, layers, mode)

                if num_params == 0:
                    print("WARNING: 0 trainable parameters. Skipping.")
                    model.load_state_dict(original_state_dict)
                    for p in model.parameters():
                        p.requires_grad = True
                    continue

                if config.loss_type == "dpo":
                    run_id = (
                        f"cmp_dpo_{method}_{mode}_{n_layers}layers"
                        f"_beta{config.dpo_beta}_lr{config.learning_rate}"
                    )
                else:
                    run_id = (
                        f"cmp_sft_{method}_{mode}_{n_layers}layers"
                        f"_ul{config.ul_weight}_lr{config.learning_rate}"
                    )

                log_path = f"{config.results_dir}/{run_id}.json"
                if s3_utils.exists(log_path):
                    print(f"Experiment {run_id} already done (log on S3). Skipping.")
                    model.load_state_dict(original_state_dict)
                    for p in model.parameters():
                        p.requires_grad = True
                    continue

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
                    experiment_type=f"{method}_{mode}_{n_layers}layers",
                    bias_type=config.bias_type,
                )

                optimizer = torch.optim.AdamW(
                    filter(lambda p: p.requires_grad, model.parameters()),
                    lr=config.learning_rate,
                    weight_decay=0.0,
                )

                ref_model = HookedTransformer.from_pretrained("gpt2-xl")
                for param in ref_model.parameters():
                    param.requires_grad = False

                result = None
                try:
                    if config.loss_type == "dpo":
                        dataset = DPODataset(
                            config.dpo_dataset,
                            tokenizer,
                            target_ids=[str(i) for i in target_ids],
                            max_length=config.max_token_length,
                        )
                        if len(dataset) == 0:
                            print("DPO dataset is empty after filtering. Skipping.")
                            continue

                        train_size = int(0.8 * len(dataset))
                        val_size = len(dataset) - train_size
                        train_set, val_set = random_split(
                            dataset,
                            [train_size, val_size],
                            generator=torch.Generator().manual_seed(42),
                        )
                        train_loader = DataLoader(
                            train_set, batch_size=config.batch_size, shuffle=True)
                        val_loader = DataLoader(
                            val_set, batch_size=config.batch_size, shuffle=False)

                        result = run_training_dpo(
                            model, ref_model, train_loader, val_loader,
                            optimizer, run_config,
                            run_id=run_id, num_params=num_params,
                        )

                    elif config.loss_type == "sft_improved":
                        sft_dataset = ImprovedSFTDataset(
                            config.fine_tune_dataset,
                            tokenizer,
                            target_ids=[str(i) for i in target_ids],
                            max_length=config.max_token_length,
                        )
                        if len(sft_dataset) == 0:
                            print("SFT dataset is empty after filtering. Skipping.")
                            continue

                        train_size = int(0.8 * len(sft_dataset))
                        val_size = len(sft_dataset) - train_size
                        train_set, val_set = random_split(
                            sft_dataset,
                            [train_size, val_size],
                            generator=torch.Generator().manual_seed(42),
                        )
                        train_loader = DataLoader(
                            train_set, batch_size=config.batch_size, shuffle=True)
                        val_loader = DataLoader(
                            val_set, batch_size=config.batch_size, shuffle=False)

                        dpo_val_dataset = DPODataset(
                            config.dpo_dataset,
                            tokenizer,
                            target_ids=[str(i) for i in target_ids],
                            max_length=config.max_token_length,
                        )
                        val_dpo_loader = None
                        if len(dpo_val_dataset) > 0:
                            _, dpo_val_set = random_split(
                                dpo_val_dataset,
                                [
                                    int(0.8 * len(dpo_val_dataset)),
                                    len(dpo_val_dataset)
                                    - int(0.8 * len(dpo_val_dataset)),
                                ],
                                generator=torch.Generator().manual_seed(42),
                            )
                            val_dpo_loader = DataLoader(
                                dpo_val_set,
                                batch_size=config.batch_size,
                                shuffle=False,
                            )

                        result = run_training_sft_improved(
                            model, ref_model, train_loader, val_loader,
                            val_dpo_loader, optimizer, run_config,
                            run_id=run_id, num_params=num_params,
                        )

                    else:
                        raise ValueError(f"Unknown loss_type: {config.loss_type}")

                finally:
                    model.load_state_dict(original_state_dict)
                    for p in model.parameters():
                        p.requires_grad = True

                    del ref_model, optimizer
                    gc.collect()
                    torch.cuda.empty_cache()

                if result is not None:
                    key = f"{method}_{mode}_{n_layers}layers"
                    all_results[key] = result

                    summary_path = (
                        f"{config.results_dir}/comparison_results.json"
                    )
                    s3_utils.write_json(all_results, summary_path)
                    print(f"Saved incremental summary to S3 ({summary_path})")

    summary_path = f"{config.results_dir}/comparison_results.json"
    s3_utils.write_json(all_results, summary_path)
    print(f"\nSaved final summary to S3 ({summary_path})")

    return all_results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="StereoSet: DLA vs SNR layer-level comparison experiments.",
    )
    parser.add_argument(
        "--snr-json",
        type=str,
        default="spectrum/snr_results_gpt2-xl_sorted.json",
        help="Path to sorted SNR JSON from spectrum.py.",
    )
    parser.add_argument(
        "--method",
        choices=["dla", "snr", "all"],
        default="all",
        help="Selection method to run.",
    )
    parser.add_argument(
        "--mode",
        choices=["attn", "mlp", "both", "all"],
        default="all",
        help="Unfreezing mode.",
    )
    parser.add_argument(
        "--layer-counts",
        type=int,
        nargs="+",
        default=[1, 2, 4, 8],
        help="Number of layers to unfreeze per experiment.",
    )
    parser.add_argument(
        "--loss-type",
        choices=["dpo", "sft_improved"],
        default="dpo",
    )
    parser.add_argument("--dpo-beta", type=float, default=0.3)
    parser.add_argument("--ul-weight", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    args = parser.parse_args()

    methods = ["dla", "snr"] if args.method == "all" else [args.method]
    modes = ["attn", "mlp", "both"] if args.mode == "all" else [args.mode]

    print("Loading gpt2-xl ...")
    model = HookedTransformer.from_pretrained("gpt2-xl")
    tokenizer = model.tokenizer

    print("Loading StereoSet DLA data from S3 ...")
    df_impact = s3_utils.read_csv(
        "outputs/gpt2-xl/dev_tests/accumulated_impact_gender_baseline_test_v2.csv"
    )
    df_probs = s3_utils.read_csv(
        "outputs/gpt2-xl/dev_tests/out_DLA_gender_baseline_test_v2.csv"
    )

    config = ExperimentConfig(
        loss_type=args.loss_type,
        dpo_beta=args.dpo_beta,
        ul_weight=args.ul_weight,
        learning_rate=args.learning_rate,
        results_dir="stereoset_experiments/outputs/gpt2-xl/fine_tuned_v2/comparison_logs",
        s3_prefix="stereoset_experiments/outputs/gpt2-xl/fine_tuned_v2/comparison_checkpoints",
    )

    run_comparison_experiments(
        model,
        tokenizer,
        df_impact,
        df_probs,
        snr_json_path=args.snr_json,
        config=config,
        methods=methods,
        modes=modes,
        layer_counts=args.layer_counts,
    )
