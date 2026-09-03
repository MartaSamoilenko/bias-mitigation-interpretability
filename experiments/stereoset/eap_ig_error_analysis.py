# EAP-IG (Edge Attribution Patching with Integrated Gradients) error analysis.

import argparse
import json
import math
import os
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from huggingface_hub import login
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import s3_utils
from dla_error_analysis import (
    NMAE_PLAUSIBLE_HI,
    VALIDATION_N,
    build_component_catalog,
    compute_activation_means,
    load_model,
    nmae_against,
    bootstrap_nmae_ci,
    bootstrap_spearman_delta_ci,
    tokenize_candidate,
    validate_hook_z_shape,
    validate_model_compatibility,
)

DEFAULT_MODELS = [
    "gpt2-xl",
    "meta-llama/Llama-3.2-1B",
    "google/gemma-2-2b",
]

MERGE_KEYS = [
    "model", "example_id", "candidate", "candidate_type",
    "target_token_id", "layer", "component_type", "head_idx",
]


def _extract_clean_activations(cache, n_layers):
    """Read clean activations at position -1 from a TransformerLens cache."""
    head_z = {}
    mlp_out = {}
    for layer in range(n_layers):
        head_z[layer] = cache[f"blocks.{layer}.attn.hook_z"][0, -1].detach().clone()
        mlp_out[layer] = cache[f"blocks.{layer}.hook_mlp_out"][0, -1].detach().clone()
    return head_z, mlp_out


def _build_baseline_activations(n_layers, n_heads, d_head, d_model,
                                device, means=None):
    """Build per-layer baseline activations (zero or mean)."""
    head_means, mlp_means = means if means is not None else (None, None)
    base_z = {}
    base_mlp = {}
    for layer in range(n_layers):
        if head_means is not None:
            base_z[layer] = torch.stack(
                [head_means[(layer, h)] for h in range(n_heads)]
            ).to(device)
            base_mlp[layer] = mlp_means[layer].to(device)
        else:
            base_z[layer] = torch.zeros(n_heads, d_head, device=device)
            base_mlp[layer] = torch.zeros(d_model, device=device)
    return base_z, base_mlp


def _make_head_z_hook(leaf_z):
    """Hook replacing position -1 of hook_z with a gradient-tracked leaf."""
    def hook(act, hook):
        prefix = act[:, :-1].detach()
        suffix = leaf_z.unsqueeze(0).unsqueeze(0)
        return torch.cat([prefix, suffix], dim=1)
    return hook


def _make_mlp_out_hook(leaf_mlp):
    """Hook replacing position -1 of hook_mlp_out with a gradient-tracked leaf."""
    def hook(act, hook):
        prefix = act[:, :-1].detach()
        suffix = leaf_mlp.unsqueeze(0).unsqueeze(0)
        return torch.cat([prefix, suffix], dim=1)
    return hook


def compute_eap_ig(model, tokens, target_token_id, cache,
                   means=None, n_steps=5):
    cfg = model.cfg
    n_layers = cfg.n_layers
    n_heads = cfg.n_heads
    d_head = cfg.d_head
    d_model = cfg.d_model
    device = next(model.parameters()).device

    clean_z, clean_mlp = _extract_clean_activations(cache, n_layers)
    base_z, base_mlp = _build_baseline_activations(
        n_layers, n_heads, d_head, d_model, device, means,
    )

    grad_acc_z = {L: torch.zeros(n_heads, d_head, device=device)
                  for L in range(n_layers)}
    grad_acc_mlp = {L: torch.zeros(d_model, device=device)
                    for L in range(n_layers)}

    for step in range(1, n_steps + 1):
        alpha = step / n_steps

        leaves_z = {}
        leaves_mlp = {}
        fwd_hooks = []

        for layer in range(n_layers):
            lz = (base_z[layer]
                  + alpha * (clean_z[layer] - base_z[layer])
                  ).detach().requires_grad_(True)
            leaves_z[layer] = lz
            fwd_hooks.append(
                (f"blocks.{layer}.attn.hook_z", _make_head_z_hook(lz)))

            lm = (base_mlp[layer]
                  + alpha * (clean_mlp[layer] - base_mlp[layer])
                  ).detach().requires_grad_(True)
            leaves_mlp[layer] = lm
            fwd_hooks.append(
                (f"blocks.{layer}.hook_mlp_out", _make_mlp_out_hook(lm)))

        captured = {}

        def _capture(act, hook, _c=captured):
            _c["resid"] = act
            return act

        fwd_hooks.append(
            (f"blocks.{n_layers - 1}.hook_resid_post", _capture))

        model.run_with_hooks(
            tokens, fwd_hooks=fwd_hooks, return_type=None)

        resid = captured["resid"][0, -1]
        normed = model.ln_final(resid.unsqueeze(0).unsqueeze(0))[0, 0]
        metric = normed @ model.W_U[:, target_token_id]
        metric.backward()

        for layer in range(n_layers):
            g = leaves_z[layer].grad
            if g is not None:
                grad_acc_z[layer] = grad_acc_z[layer] + g
            g = leaves_mlp[layer].grad
            if g is not None:
                grad_acc_mlp[layer] = grad_acc_mlp[layer] + g

        del captured, leaves_z, leaves_mlp
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    attributions = []
    for layer in range(n_layers):
        mean_grad_z = grad_acc_z[layer] / n_steps
        delta_z = clean_z[layer] - base_z[layer]
        for h in range(n_heads):
            attr = torch.dot(delta_z[h], mean_grad_z[h]).item()
            attributions.append(attr)

        mean_grad_mlp = grad_acc_mlp[layer] / n_steps
        delta_mlp = clean_mlp[layer] - base_mlp[layer]
        attributions.append(torch.dot(delta_mlp, mean_grad_mlp).item())

    return np.array(attributions)


def analyze_model_eap_ig(model_name, examples, device,
                         ablation_modes=("zero", "mean"),
                         n_ig_steps=5):
    print(f"\n{'=' * 70}")
    print(f"  EAP-IG analysis: {model_name}  "
          f"(modes: {', '.join(ablation_modes)}, "
          f"IG steps: {n_ig_steps})")
    print(f"{'=' * 70}")

    model = load_model(model_name, device)
    validate_model_compatibility(model)
    model.eval()

    for p in model.parameters():
        p.requires_grad_(False)

    validate_hook_z_shape(model, examples[0])

    catalog = build_component_catalog(model)
    n_comp = len(catalog)
    print(f"Components: {n_comp}  "
          f"({model.cfg.n_layers}L x {model.cfg.n_heads}H + "
          f"{model.cfg.n_layers} MLPs)")

    means = None
    if "mean" in ablation_modes:
        means = compute_activation_means(model, examples)
        print(f"[MEAN-ABL] Precomputed activation means over "
              f"{len(examples)} examples")

    records_by_mode = {m: [] for m in ablation_modes}
    n_candidates = 0
    n_multi_token = 0

    for ex_idx, example in enumerate(examples):
        context = example["rephrased_context"].split("BLANK")[0].strip()
        ex_id = example["id"]
        print(f"  [{ex_idx + 1}/{len(examples)}] {ex_id[:12]}...  "
              f"'{context[:50]}...'")

        for cand_type, word in example["targets"].items():
            toks = tokenize_candidate(model, word, model.cfg.model_name)
            if toks is None:
                continue
            n_candidates += 1
            if len(toks) > 1:
                n_multi_token += 1
                print(f"    [MULTI-TOKEN] '{word}' -> "
                      f"{len(toks)} sub-tokens; scoring first only")
            target_id = toks[0]
            tokens = model.to_tokens(context)

            with torch.no_grad():
                _, cache = model.run_with_cache(
                    tokens, return_type=None)

            for mode in ablation_modes:
                mode_means = means if mode == "mean" else None
                eap_scores = compute_eap_ig(
                    model, tokens, target_id, cache,
                    means=mode_means, n_steps=n_ig_steps,
                )

                for i, (layer, ctype, hidx) in enumerate(catalog):
                    records_by_mode[mode].append({
                        "model": model_name,
                        "example_id": ex_id,
                        "candidate": word,
                        "candidate_type": cand_type,
                        "target_token_id": target_id,
                        "layer": layer,
                        "component_type": ctype,
                        "head_idx": hidx if hidx is not None else -1,
                        "eap_ig": float(eap_scores[i]),
                    })

            del cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    pct = 100.0 * n_multi_token / max(1, n_candidates)
    print(f"[TOKENIZATION] {n_multi_token}/{n_candidates} "
          f"multi-token ({pct:.1f}%)")

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return records_by_mode, (n_multi_token, n_candidates)


def per_example_selection_metrics_generic(records_df, col_a, col_b,
                                          ks=(5, 10, 25)):
    rhos, sign_agree = [], []
    jaccard = {k: [] for k in ks}
    for _, sub in records_df.groupby("example_id"):
        a = sub[col_a].values
        b = sub[col_b].values
        rho, _ = spearmanr(a, b)
        rhos.append(rho)
        sign_agree.append(float(np.mean(np.sign(a) == np.sign(b))))
        for k in ks:
            top_a = set(np.argsort(-np.abs(a))[:k])
            top_b = set(np.argsort(-np.abs(b))[:k])
            jaccard[k].append(
                len(top_a & top_b) / len(top_a | top_b))
    return {
        "spearman_median": float(np.median(rhos)),
        "spearman_iqr": [float(np.percentile(rhos, 25)),
                         float(np.percentile(rhos, 75))],
        "spearman_per_example": np.array(rhos),
        "sign_agreement": float(np.mean(sign_agree)),
        **{f"top{k}_jaccard_median": float(np.median(jaccard[k]))
           for k in ks},
    }


def _load_ap_records(model_name, mode, output_dir):
    safe = model_name.replace("/", "_")
    candidates = [
        f"{output_dir}/{safe}_{mode}_ablation_records.csv",
        f"{output_dir}/{safe}_records.csv",
    ]
    for path in candidates:
        try:
            df = s3_utils.read_csv(path)
            if "ap_precap" in df.columns:
                print(f"  [AP] Loaded {len(df)} AP records from {path}")
                return df
        except Exception:
            continue
    return None


def compute_summary_eap_ig(eap_df, model_name, mode, ap_df=None):
    summary = {
        "model": model_name,
        "ablation_type": mode,
        "n_records": len(eap_df),
        "n_examples": int(eap_df["example_id"].nunique()),
    }

    if ap_df is None:
        return summary

    merged = eap_df.merge(
        ap_df[MERGE_KEYS + ["ap_precap", "dla"]],
        on=MERGE_KEYS, how="inner",
    )
    if len(merged) == 0:
        print("  [WARN] No matching AP records after merge — skipping "
              "comparison metrics.")
        return summary

    eap = merged["eap_ig"].values
    ap = merged["ap_precap"].values
    dla = merged["dla"].values
    eids = merged["example_id"].values

    eap_vs_ap = nmae_against(eap, ap, ap)
    ci = bootstrap_nmae_ci(eap, ap, eids, denom_source=ap)
    summary["eap_ig_vs_ap_nmae"] = round(eap_vs_ap, 2)
    summary["eap_ig_vs_ap_ci_lo"] = round(ci[0], 2)
    summary["eap_ig_vs_ap_ci_hi"] = round(ci[1], 2)

    dla_vs_ap = nmae_against(dla, ap, ap)
    summary["dla_vs_ap_nmae"] = round(dla_vs_ap, 2)

    sel_eap = per_example_selection_metrics_generic(
        merged, "eap_ig", "ap_precap")
    for k, v in sel_eap.items():
        if k != "spearman_per_example":
            summary[f"eap_ig_{k}"] = v

    sel_dla = per_example_selection_metrics_generic(
        merged, "dla", "ap_precap")
    for k, v in sel_dla.items():
        if k != "spearman_per_example":
            summary[f"dla_{k}"] = v

    rhos_eap = sel_eap["spearman_per_example"]
    rhos_dla = sel_dla["spearman_per_example"]
    delta_pt, delta_lo, delta_hi, delta_p = bootstrap_spearman_delta_ci(
        rhos_dla, rhos_eap)
    summary["delta_spearman_median"] = round(delta_pt, 4)
    summary["delta_spearman_ci_lo"] = round(delta_lo, 4)
    summary["delta_spearman_ci_hi"] = round(delta_hi, 4)
    summary["delta_spearman_pvalue"] = round(delta_p, 4)

    return summary


def print_eap_ig_report(summary):
    print(f"\n{'─' * 60}")
    abl = summary.get("ablation_type", "")
    label = f" ({abl} ablation)" if abl else ""
    print(f"Model : {summary['model']}{label}")
    print(f"Records: {summary['n_records']}  |  "
          f"Examples: {summary['n_examples']}")
    print(f"{'─' * 60}")

    if "eap_ig_vs_ap_nmae" in summary:
        print(f"  EAP-IG vs AP  NMAE = {summary['eap_ig_vs_ap_nmae']:.2f}%  "
              f"[{summary['eap_ig_vs_ap_ci_lo']:.2f}%, "
              f"{summary['eap_ig_vs_ap_ci_hi']:.2f}%]")
        if "dla_vs_ap_nmae" in summary:
            print(f"  DLA    vs AP  NMAE = {summary['dla_vs_ap_nmae']:.2f}%  "
                  f"(reference)")
        print(f"  EAP-IG selection:  "
              f"Spearman={summary['eap_ig_spearman_median']:.3f}  "
              f"sign-agree="
              f"{summary['eap_ig_sign_agreement'] * 100:.1f}%  "
              f"top5-J={summary['eap_ig_top5_jaccard_median']:.2f}")
        if "dla_spearman_median" in summary:
            print(f"  DLA    selection:  "
                  f"Spearman={summary['dla_spearman_median']:.3f}  "
                  f"sign-agree="
                  f"{summary['dla_sign_agreement'] * 100:.1f}%  "
                  f"top5-J={summary['dla_top5_jaccard_median']:.2f}")
        if "delta_spearman_median" in summary:
            sig = "*" if summary["delta_spearman_pvalue"] < 0.05 else "n.s."
            print(f"  Δρ (EAP-IG − DLA): {summary['delta_spearman_median']:+.4f}  "
                  f"95% CI [{summary['delta_spearman_ci_lo']:+.4f}, "
                  f"{summary['delta_spearman_ci_hi']:+.4f}]  "
                  f"p={summary['delta_spearman_pvalue']:.4f} ({sig})")
    else:
        print("  (no AP records available for comparison)")
    print(f"{'─' * 60}")


def run_validation_checkpoint_eap_ig(model_name, examples, device,
                                     n_ig_steps, output_dir):
    print(f"\n{'─' * 60}")
    print(f"  Validating EAP-IG: {model_name} "
          f"({VALIDATION_N} examples, {n_ig_steps} IG steps)")
    print(f"{'─' * 60}")

    val_examples = examples[:VALIDATION_N]
    records_by_mode, _ = analyze_model_eap_ig(
        model_name, val_examples, device,
        ablation_modes=("zero",), n_ig_steps=n_ig_steps,
    )

    records = records_by_mode["zero"]
    df = pd.DataFrame(records)
    scores = df["eap_ig"].values

    n_nonfinite = int(np.sum(~np.isfinite(scores)))
    if n_nonfinite > 0:
        raise ValueError(
            f"[VALIDATION FAIL] {model_name}: "
            f"{n_nonfinite} non-finite EAP-IG scores")

    if np.allclose(scores, 0.0):
        raise ValueError(
            f"[VALIDATION FAIL] {model_name}: "
            f"all EAP-IG scores are zero")

    ap_df = _load_ap_records(model_name, "zero", output_dir)
    if ap_df is not None:
        merged = df.merge(
            ap_df[MERGE_KEYS + ["ap_precap"]],
            on=MERGE_KEYS, how="inner",
        )
        if len(merged) > 0:
            eap = merged["eap_ig"].values
            ap = merged["ap_precap"].values
            val_nmae = nmae_against(eap, ap, ap)
            if not math.isfinite(val_nmae):
                raise ValueError(
                    f"[VALIDATION FAIL] {model_name}: "
                    f"NMAE(EAP-IG, AP) is non-finite ({val_nmae})")
            if val_nmae > NMAE_PLAUSIBLE_HI:
                raise ValueError(
                    f"[VALIDATION FAIL] {model_name}: "
                    f"NMAE(EAP-IG, AP) = {val_nmae:.2f}% exceeds "
                    f"{NMAE_PLAUSIBLE_HI}%")
            print(f"[VALIDATION OK] {model_name}: "
                  f"NMAE(EAP-IG, AP) = {val_nmae:.1f}%")
        else:
            print(f"[VALIDATION WARN] {model_name}: "
                  f"no matching AP rows found for NMAE check")
    else:
        print(f"[VALIDATION WARN] {model_name}: "
              f"no AP records on disk — skipping NMAE gate")

    print(f"[VALIDATION OK] {model_name}: "
          f"finite={len(scores) - n_nonfinite}/{len(scores)}, "
          f"max|score|={np.max(np.abs(scores)):.4f}")


def main():
    parser = argparse.ArgumentParser(
        description="EAP-IG error analysis: Integrated-gradient "
                    "attribution vs Activation Patching",
    )
    parser.add_argument(
        "--models", nargs="+", default=DEFAULT_MODELS,
        help="HuggingFace / TransformerLens model names",
    )
    parser.add_argument("--n-examples", type=int, default=100,
                        help="Number of StereoSet examples to sample")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-ig-steps", type=int, default=5,
                        help="Number of IG interpolation steps")
    parser.add_argument(
        "--ablation-modes", nargs="+", default=["zero", "mean"],
        choices=["zero", "mean"],
        help="Baseline ablation modes",
    )
    parser.add_argument("--no-s3", action="store_true",
                        help="Use local disk instead of S3")
    parser.add_argument("--skip-validation", action="store_true",
                        help="Skip the 5-example validation checkpoint")
    parser.add_argument("--summary-only", action="store_true",
                        help="Recompute summaries from existing on-disk "
                             "CSVs without re-running attribution")
    args = parser.parse_args()

    s3_utils.set_use_s3(not args.no_s3)

    output_dir = "outputs/dla_error_analysis"

    if args.summary_only:
        print("\n" + "=" * 70)
        print("  SUMMARY-ONLY MODE: recomputing from on-disk CSVs")
        print("=" * 70)

        all_summaries = []
        for model_name in args.models:
            safe = model_name.replace("/", "_")
            for mode in args.ablation_modes:
                eap_path = (f"{output_dir}/"
                            f"{safe}_eap_ig_{mode}_ablation_records.csv")
                try:
                    eap_df = s3_utils.read_csv(eap_path)
                except Exception:
                    print(f"  [SKIP] Missing EAP-IG records: {eap_path}")
                    continue
                print(f"  Loaded {len(eap_df)} EAP-IG records from "
                      f"{eap_path}")

                ap_df = _load_ap_records(model_name, mode, output_dir)
                summary = compute_summary_eap_ig(
                    eap_df, model_name, mode, ap_df=ap_df)
                summary["n_ig_steps"] = args.n_ig_steps
                all_summaries.append(summary)
                print_eap_ig_report(summary)

        summary_path = f"{output_dir}/summary_all_models_eap_ig.json"
        s3_utils.write_json(all_summaries, summary_path)
        print(f"\nUnified EAP-IG summary -> {summary_path}")
        print("Done.")
        return

    hf_token = os.environ.get("HF_TOKEN", "")
    if hf_token:
        login(token=hf_token)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    dataset = s3_utils.read_json("datasets/gender_test_rephrased_v2.json")
    print(f"Loaded {len(dataset)} examples from "
          f"gender_test_rephrased_v2.json")

    random.seed(args.seed)
    if len(dataset) > args.n_examples:
        examples = random.sample(dataset, args.n_examples)
    else:
        examples = list(dataset)
    print(f"Sampled {len(examples)} examples (seed={args.seed})\n")

    if not args.skip_validation:
        print("\n" + "=" * 70)
        print(f"  PHASE 1: EAP-IG validation checkpoint "
              f"({VALIDATION_N} examples per model)")
        print("=" * 70)
        for model_name in args.models:
            run_validation_checkpoint_eap_ig(
                model_name, examples, device,
                args.n_ig_steps, output_dir,
            )
        print("\n[ALL VALIDATION PASSED] Proceeding to full analysis.\n")
    else:
        print("[SKIP] Validation checkpoint skipped.\n")

    print("\n" + "=" * 70)
    print(f"  PHASE 2: Full EAP-IG analysis "
          f"({len(examples)} examples per model, "
          f"{args.n_ig_steps} IG steps)")
    print("=" * 70)

    all_summaries = []

    for model_name in args.models:
        records_by_mode, token_info = analyze_model_eap_ig(
            model_name, examples, device,
            ablation_modes=tuple(args.ablation_modes),
            n_ig_steps=args.n_ig_steps,
        )
        n_multi, n_cands = token_info
        mt_pct = 100.0 * n_multi / max(1, n_cands)

        safe = model_name.replace("/", "_")
        for mode, records in records_by_mode.items():
            df = pd.DataFrame(records)
            csv_path = (f"{output_dir}/"
                        f"{safe}_eap_ig_{mode}_ablation_records.csv")
            s3_utils.write_csv(df, csv_path)
            print(f"  -> saved {len(df)} records to {csv_path}")

            ap_df = _load_ap_records(model_name, mode, output_dir)
            summary = compute_summary_eap_ig(
                df, model_name, mode, ap_df=ap_df)
            summary["multi_token_pct"] = round(mt_pct, 1)
            summary["n_ig_steps"] = args.n_ig_steps
            all_summaries.append(summary)
            print_eap_ig_report(summary)

    summary_path = f"{output_dir}/summary_all_models_eap_ig.json"
    s3_utils.write_json(all_summaries, summary_path)
    print(f"\nUnified EAP-IG summary -> {summary_path}")
    print("Done.")


if __name__ == "__main__":
    main()
