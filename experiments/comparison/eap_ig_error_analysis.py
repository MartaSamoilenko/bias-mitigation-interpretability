"""EAP-IG (attribution patching with integrated gradients) error analysis.
"""

import argparse
import math
import os
import random
import sys

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from huggingface_hub import login

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import s3_utils
from dla_error_analysis import (
    NMAE_PLAUSIBLE_HI,
    VALIDATION_N,
    bootstrap_nmae_ci,
    bootstrap_spearman_delta_ci,
    build_component_catalog,
    build_metric_specs,
    compute_activation_means,
    load_model,
    nmae_against,
    nmae_per_unit,
    per_unit_selection_metrics,
    scale_diagnostics,
    validate_completeness,
    validate_hook_z_shape,
    validate_model_compatibility,
    validate_residual_decomposition,
)

DEFAULT_MODELS = [
    "gpt2-xl",
    "meta-llama/Llama-3.2-1B",
    "google/gemma-2-2b",
]

MERGE_KEYS = [
    "model", "example_id", "unit_id", "layer", "component_type", "head_idx",
]

DE_COLLINEARITY_MAX = 0.995

EAP_NMAE_WARN = 200.0
EAP_NMAE_HARD = 2000.0


@torch.no_grad()
def compute_embed_mean(model, examples):
    """Mean of blocks.0.hook_resid_pre at the final position (IG path start)."""
    acc = None
    n = 0
    for example in examples:
        context = example["rephrased_context"].split("BLANK")[0].strip()
        tokens = model.to_tokens(context)
        _, cache = model.run_with_cache(
            tokens, return_type=None,
            names_filter=lambda nm: nm == "blocks.0.hook_resid_pre")
        v = cache["blocks.0.hook_resid_pre"][0, -1].float()
        acc = v if acc is None else acc + v
        n += 1
        del cache
    return (acc / n).to(model.W_O.dtype)


@torch.no_grad()
def extract_clean_activations(cache, n_layers):
    head_z, mlp_out = {}, {}
    for layer in range(n_layers):
        head_z[layer] = cache[
            f"blocks.{layer}.attn.hook_z"][0, -1].detach().clone()
        mlp_out[layer] = cache[
            f"blocks.{layer}.hook_mlp_out"][0, -1].detach().clone()
    return head_z, mlp_out


def build_baseline_activations(model, means=None):
    """Per-node ablation baseline, matched to what compute_ap() ablates to."""
    cfg = model.cfg
    device = next(model.parameters()).device
    dtype = model.W_O.dtype
    head_means, mlp_means = means if means is not None else (None, None)

    base_z, base_mlp = {}, {}
    for layer in range(cfg.n_layers):
        if head_means is not None:
            base_z[layer] = torch.stack(
                [head_means[(layer, h)] for h in range(cfg.n_heads)]
            ).to(device=device, dtype=dtype)
            base_mlp[layer] = mlp_means[layer].to(device=device, dtype=dtype)
        else:
            base_z[layer] = torch.zeros(cfg.n_heads, cfg.d_head,
                                        device=device, dtype=dtype)
            base_mlp[layer] = torch.zeros(cfg.d_model, device=device,
                                          dtype=dtype)
    return base_z, base_mlp


def compute_eap_ig(model, tokens, metric_spec, cache, means=None,
                   n_steps=5, ig_rule="midpoint", embed_baseline=None):
    """Integrated-gradient node attribution with an INTACT forward graph."""
    cfg = model.cfg
    n_layers = cfg.n_layers

    clean_z, clean_mlp = extract_clean_activations(cache, n_layers)
    base_z, base_mlp = build_baseline_activations(model, means)

    clean_embed = cache["blocks.0.hook_resid_pre"][0, -1].detach().clone()
    if embed_baseline is None:
        embed_baseline = torch.zeros_like(clean_embed)

    grad_z = {L: torch.zeros_like(clean_z[L], dtype=torch.float32)
              for L in range(n_layers)}
    grad_mlp = {L: torch.zeros_like(clean_mlp[L], dtype=torch.float32)
                for L in range(n_layers)}

    for step in range(1, n_steps + 1):
        alpha = ((step - 0.5) / n_steps if ig_rule == "midpoint"
                 else step / n_steps)

        interp = (embed_baseline
                  + alpha * (clean_embed - embed_baseline)
                  ).detach().requires_grad_(True)

        def _embed_hook(act, hook, _v=interp):
            out = act.clone()
            out[0, -1] = _v
            return out

        captured: dict = {}

        def _capture_resid(act, hook, _c=captured):
            _c["resid"] = act
            return act

        model.reset_hooks()
        model.add_hook("blocks.0.hook_resid_pre", _embed_hook, dir="fwd")
        model.add_hook(f"blocks.{n_layers - 1}.hook_resid_post",
                       _capture_resid, dir="fwd")

        store: dict = {}

        def _make_bwd(name):
            def _bwd(grad, hook, _n=name, _s=store):
                _s[_n] = grad[0, -1].detach().float()
                return None
            return _bwd

        for L in range(n_layers):
            zname = f"blocks.{L}.attn.hook_z"
            mname = f"blocks.{L}.hook_mlp_out"
            model.add_hook(zname, _make_bwd(zname), dir="bwd")
            model.add_hook(mname, _make_bwd(mname), dir="bwd")

        model(tokens, return_type=None)

        resid = captured["resid"][0, -1]
        normed = model.ln_final(resid.unsqueeze(0).unsqueeze(0))[0, 0]
        logits = normed @ metric_spec.columns(model)
        metric = metric_spec.apply_cap(logits, None)
        metric.backward()

        for L in range(n_layers):
            zname = f"blocks.{L}.attn.hook_z"
            mname = f"blocks.{L}.hook_mlp_out"
            if zname in store:
                grad_z[L] += store[zname]
            if mname in store:
                grad_mlp[L] += store[mname]

        model.reset_hooks()
        model.zero_grad(set_to_none=True)
        del captured, store, interp, resid, normed, logits, metric
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    attributions = []
    for L in range(n_layers):
        gz = grad_z[L] / n_steps
        dz = (clean_z[L] - base_z[L]).float()
        for h in range(cfg.n_heads):
            attributions.append(float(torch.dot(dz[h], gz[h]).item()))
        gm = grad_mlp[L] / n_steps
        dm = (clean_mlp[L] - base_mlp[L]).float()
        attributions.append(float(torch.dot(dm, gm).item()))

    return np.array(attributions)


def analyze_model_eap_ig(model_name, examples, device,
                         ablation_modes=("zero", "mean"), n_ig_steps=5,
                         ig_rule="midpoint", ig_path="embed_mean",
                         metric_mode="logit_diff", multi_token_policy="skip",
                         run_gates=True, enable_grad_flags=False,
                         also_atp=True):
    print(f"\n{'=' * 70}")
    print(f"  EAP-IG: {model_name}  (modes: {', '.join(ablation_modes)}, "
          f"m={n_ig_steps}, rule={ig_rule}, path={ig_path}, "
          f"metric={metric_mode})")
    print(f"{'=' * 70}")

    model = load_model(model_name, device, enable_grad_flags)
    validate_model_compatibility(model)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    validate_hook_z_shape(model, examples[0])

    catalog = build_component_catalog(model)
    print(f"Components: {len(catalog)}")

    stats = {"n_candidates": 0, "n_multi_token": 0, "n_skipped": 0}

    if run_gates:
        validate_residual_decomposition(model, examples[0])
        probe = build_metric_specs(model, examples[0], metric_mode,
                                   multi_token_policy, dict(stats))
        if probe:
            validate_completeness(model, examples[0], probe[0])

    means = None
    if "mean" in ablation_modes:
        means = compute_activation_means(model, examples)
        print(f"[MEAN-ABL] means over {len(examples)} examples")

    embed_baseline = None
    if ig_path == "embed_mean":
        embed_baseline = compute_embed_mean(model, examples)
        print(f"[IG-PATH] embed_mean baseline, "
              f"|mean|={embed_baseline.norm().item():.3f}")
    else:
        print("[IG-PATH] zero-embedding baseline")

    records_by_mode: dict = {m: [] for m in ablation_modes}

    for ex_idx, example in enumerate(examples):
        context = example["rephrased_context"].split("BLANK")[0].strip()
        ex_id = example["id"]
        print(f"  [{ex_idx + 1}/{len(examples)}] {ex_id[:12]}...")

        specs = build_metric_specs(model, example, metric_mode,
                                   multi_token_policy, stats)
        if not specs:
            continue
        tokens = model.to_tokens(context)

        for spec in specs:
            with torch.no_grad():
                _, cache = model.run_with_cache(tokens, return_type=None)

            for mode in ablation_modes:
                mode_means = means if mode == "mean" else None
                eap = compute_eap_ig(
                    model, tokens, spec, cache, means=mode_means,
                    n_steps=n_ig_steps, ig_rule=ig_rule,
                    embed_baseline=embed_baseline)

                atp = None
                if also_atp:
                    atp = compute_eap_ig(
                        model, tokens, spec, cache, means=mode_means,
                        n_steps=1, ig_rule="right",
                        embed_baseline=embed_baseline)

                for i, (layer, ctype, hidx) in enumerate(catalog):
                    rec = {
                        "model": model_name,
                        "example_id": ex_id,
                        "unit_id": spec.unit_id,
                        "candidate": "|".join(spec.words),
                        "candidate_type": spec.label,
                        "metric": metric_mode,
                        "layer": layer,
                        "component_type": ctype,
                        "head_idx": hidx if hidx is not None else -1,
                        "eap_ig": float(eap[i]),
                    }
                    if atp is not None:
                        rec["atp"] = float(atp[i])
                    records_by_mode[mode].append(rec)

            del cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    nc = max(1, stats["n_candidates"])
    print(f"[TOKENIZATION] {stats['n_multi_token']}/{nc} multi-token, "
          f"{stats['n_skipped']} skipped")

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return records_by_mode, stats


def _load_ap_records(model_name, mode, output_dir, suffix=""):
    safe = model_name.replace("/", "_")
    path = f"{output_dir}/{safe}_{mode}_ablation_records{suffix}.csv"
    try:
        df = s3_utils.read_csv(path)
    except Exception:
        print(f"  [AP] not found: {path}")
        return None
    if "ap_precap" not in df.columns:
        print(f"  [AP] {path} has no ap_precap column")
        return None
    print(f"  [AP] loaded {len(df)} records from {path}")
    return df


def compute_summary_eap_ig(eap_df, model_name, mode, ap_df=None,
                           topk_mode="abs"):
    summary = {
        "model": model_name,
        "ablation_type": mode,
        "n_records": len(eap_df),
        "n_examples": int(eap_df["example_id"].nunique()),
    }
    if ap_df is None:
        return summary

    cols = MERGE_KEYS + ["ap_precap", "dla", "de_precap", "clean_precap"]
    cols = [c for c in cols if c in ap_df.columns]
    merged = eap_df.merge(ap_df[cols], on=MERGE_KEYS, how="inner")

    if len(merged) != len(eap_df):
        print(f"  [WARN] merge kept {len(merged)}/{len(eap_df)} EAP rows — "
              f"the AP run and this run disagree on examples/units. "
              f"Check --n-examples/--seed/--metric match.")
    if len(merged) == 0:
        print("  [WARN] no matching AP records; skipping comparison metrics")
        return summary

    eap = merged["eap_ig"].values
    ap = merged["ap_precap"].values
    eids = merged["example_id"].values

    if "de_precap" in merged.columns:
        r_de = float(np.corrcoef(eap, merged["de_precap"].values)[0, 1])
        summary["eap_ig_corr_with_de"] = round(r_de, 5)
        if abs(r_de) > DE_COLLINEARITY_MAX:
            print(f"  [!!] corr(EAP-IG, DE) = {r_de:.5f} > "
                  f"{DE_COLLINEARITY_MAX}: the estimator is still "
                  f"direct-effect-only. Do not report these numbers.")

    summary["eap_ig_vs_ap_nmae"] = round(nmae_against(eap, ap, ap), 2)
    ci = bootstrap_nmae_ci(eap, ap, eids, denom_source=ap)
    summary["eap_ig_vs_ap_ci_lo"] = round(ci[0], 2)
    summary["eap_ig_vs_ap_ci_hi"] = round(ci[1], 2)
    med, q25, q75 = nmae_per_unit(merged, "eap_ig", "ap_precap")
    summary["eap_ig_vs_ap_nmae_per_unit_median"] = round(med, 2)
    summary["eap_ig_vs_ap_nmae_per_unit_iqr"] = [round(q25, 2), round(q75, 2)]

    methods = {"eap_ig": "eap_ig"}
    if "dla" in merged.columns:
        methods["dla"] = "dla"
    if "atp" in merged.columns:
        methods["atp"] = "atp"

    sels = {}
    for name, col in methods.items():
        summary[f"{name}_vs_ap_nmae"] = round(
            nmae_against(merged[col].values, ap, ap), 2)

        ratio, s_opt, cal = scale_diagnostics(merged[col].values, ap)
        summary[f"{name}_magnitude_ratio"] = round(ratio, 3)
        summary[f"{name}_l1_scale"] = round(s_opt, 4)
        summary[f"{name}_vs_ap_nmae_calibrated"] = round(cal, 2)
        sel = per_unit_selection_metrics(merged, col, "ap_precap",
                                         topk_mode=topk_mode)
        sels[name] = sel
        for k, v in sel.items():
            if k != "spearman_per_unit":
                summary[f"{name}_{k}"] = v

    if "dla" in sels:
        d, lo, hi, p = bootstrap_spearman_delta_ci(
            sels["dla"]["spearman_per_unit"],
            sels["eap_ig"]["spearman_per_unit"])
        summary["delta_spearman_median"] = round(d, 4)
        summary["delta_spearman_ci_lo"] = round(lo, 4)
        summary["delta_spearman_ci_hi"] = round(hi, 4)
        summary["delta_spearman_pvalue"] = round(p, 4)

    return summary


def print_eap_ig_report(summary):
    print(f"\n{'─' * 66}")
    abl = summary.get("ablation_type", "")
    print(f"Model : {summary['model']} ({abl} ablation)")
    print(f"Records: {summary['n_records']}  |  "
          f"Examples: {summary['n_examples']}")
    print(f"{'─' * 66}")
    if "eap_ig_vs_ap_nmae" not in summary:
        print("  (no AP records available for comparison)")
        print(f"{'─' * 66}")
        return

    if "eap_ig_corr_with_de" in summary:
        print(f"  corr(EAP-IG, DE) = {summary['eap_ig_corr_with_de']:.4f}   "
              f"(must be < {DE_COLLINEARITY_MAX})")
    for name, label in [("dla", "DLA   "), ("atp", "AtP   "),
                        ("eap_ig", "EAP-IG")]:
        if f"{name}_vs_ap_nmae" not in summary:
            continue
        print(f"  {label} NMAE={summary[f'{name}_vs_ap_nmae']:8.2f}%  "
              f"(calib {summary[f'{name}_vs_ap_nmae_calibrated']:6.2f}%, "
              f"mag x{summary[f'{name}_magnitude_ratio']:.2f})  "
              f"rho={summary[f'{name}_spearman_median']:+.3f}  "
              f"tau={summary[f'{name}_kendall_tau_median']:+.3f}  "
              f"RBO={summary[f'{name}_rbo_median']:.3f}  "
              f"J5={summary[f'{name}_top5_jaccard_median']:.2f}")
    if "delta_spearman_median" in summary:
        sig = "*" if summary["delta_spearman_pvalue"] < 0.05 else "n.s."
        print(f"  d-rho (EAP-IG - DLA): "
              f"{summary['delta_spearman_median']:+.4f}  "
              f"95% CI [{summary['delta_spearman_ci_lo']:+.4f}, "
              f"{summary['delta_spearman_ci_hi']:+.4f}]  "
              f"p={summary['delta_spearman_pvalue']:.4f} ({sig})")
    print(f"{'─' * 66}")


def run_ig_convergence_check(model_name, examples, device, metric_mode,
                             multi_token_policy, ig_path, enable_grad_flags,
                             steps=(1, 5, 10, 20)):
    from scipy.stats import spearmanr

    print(f"\n[IG-CONVERGENCE] {model_name} over m in {steps}")
    model = load_model(model_name, device, enable_grad_flags)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    stats = {"n_candidates": 0, "n_multi_token": 0, "n_skipped": 0}
    specs = build_metric_specs(model, examples[0], metric_mode,
                               multi_token_policy, stats)
    if not specs:
        print("  [SKIP] no usable candidate in example 0")
        del model
        return

    context = examples[0]["rephrased_context"].split("BLANK")[0].strip()
    tokens = model.to_tokens(context)
    with torch.no_grad():
        _, cache = model.run_with_cache(tokens, return_type=None)

    embed_baseline = (compute_embed_mean(model, examples[:5])
                      if ig_path == "embed_mean" else None)

    prev, ref = None, None
    for m in steps:
        scores = compute_eap_ig(model, tokens, specs[0], cache, means=None,
                                n_steps=m, ig_rule="midpoint",
                                embed_baseline=embed_baseline)
        if ref is None:
            ref = scores
        rho_prev = (spearmanr(prev, scores)[0] if prev is not None
                    else float("nan"))
        print(f"  m={m:3d}  max|attr|={np.max(np.abs(scores)):.4f}  "
              f"rho(vs m_prev)={rho_prev:.4f}")
        prev = scores

    del cache, model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_validation_checkpoint_eap_ig(model_name, examples, device, n_ig_steps,
                                     output_dir, metric_mode,
                                     multi_token_policy, ig_rule, ig_path,
                                     enable_grad_flags, suffix=""):
    print(f"\n{'─' * 60}")
    print(f"  Validating EAP-IG: {model_name} "
          f"({VALIDATION_N} examples, m={n_ig_steps})")
    print(f"{'─' * 60}")

    records_by_mode, _ = analyze_model_eap_ig(
        model_name, examples[:VALIDATION_N], device,
        ablation_modes=("zero",), n_ig_steps=n_ig_steps, ig_rule=ig_rule,
        ig_path=ig_path, metric_mode=metric_mode,
        multi_token_policy=multi_token_policy, run_gates=True,
        enable_grad_flags=enable_grad_flags)

    df = pd.DataFrame(records_by_mode["zero"])
    scores = df["eap_ig"].values

    n_bad = int(np.sum(~np.isfinite(scores)))
    if n_bad:
        raise ValueError(f"[VALIDATION FAIL] {model_name}: {n_bad} non-finite "
                         f"EAP-IG scores")
    if np.allclose(scores, 0.0):
        raise ValueError(f"[VALIDATION FAIL] {model_name}: all scores zero")

    ap_df = _load_ap_records(model_name, "zero", output_dir, suffix)
    if ap_df is None:
        print(f"[VALIDATION WARN] {model_name}: no AP records — "
              f"cannot run the collinearity or NMAE gates")
    else:
        cols = [c for c in MERGE_KEYS + ["ap_precap", "de_precap"]
                if c in ap_df.columns]
        merged = df.merge(ap_df[cols], on=MERGE_KEYS, how="inner")
        if len(merged) == 0:
            print(f"[VALIDATION WARN] {model_name}: no matching AP rows")
        else:
            if "de_precap" in merged.columns:
                r_de = float(np.corrcoef(
                    merged["eap_ig"].values,
                    merged["de_precap"].values)[0, 1])
                print(f"[GATE] corr(EAP-IG, DE) = {r_de:.5f}")
                if abs(r_de) > DE_COLLINEARITY_MAX:
                    raise ValueError(
                        f"[VALIDATION FAIL] {model_name}: corr(EAP-IG, DE) = "
                        f"{r_de:.5f} > {DE_COLLINEARITY_MAX}. The gradient is "
                        f"still direct-effect-only — indirect paths are being "
                        f"severed somewhere in the hook setup.")
            apv = merged["ap_precap"].values
            for col, label in [("eap_ig", "EAP-IG"), ("atp", "AtP   ")]:
                if col not in merged.columns:
                    continue
                v = merged[col].values
                val = nmae_against(v, apv, apv)
                ratio, s_opt, cal = scale_diagnostics(v, apv)
                rho = spearmanr(v, apv)[0]
                print(f"  {label} vs AP: NMAE={val:8.1f}%  "
                      f"calibrated={cal:7.1f}%  magnitude x{ratio:.2f}  "
                      f"(L1 scale {s_opt:.3f})  pooled rho={rho:+.3f}")
                if not math.isfinite(val):
                    raise ValueError(f"[VALIDATION FAIL] {model_name}: "
                                     f"NMAE({label.strip()}, AP) non-finite")
                if val > EAP_NMAE_HARD:
                    raise ValueError(
                        f"[VALIDATION FAIL] {model_name}: "
                        f"NMAE({label.strip()}, AP) = {val:.1f}% exceeds the "
                        f"hard limit {EAP_NMAE_HARD}% — this is not a large "
                        f"approximation error, it is broken output.")
                if val > EAP_NMAE_WARN:
                    print(f"  [WARN] NMAE({label.strip()}, AP) = {val:.1f}% "
                          f"is above {EAP_NMAE_WARN}%. If 'calibrated' is far "
                          f"lower, the error is magnitude, not ranking — that "
                          f"is a reportable finding, not a bug. The "
                          f"correctness gate is corr(EAP-IG, DE) above.")

    print(f"[VALIDATION OK] {model_name}: finite={len(scores)}, "
          f"max|score|={np.max(np.abs(scores)):.4f}")


def main():
    parser = argparse.ArgumentParser(
        description="EAP-IG error analysis vs Activation Patching")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--n-examples", type=int, default=0,
                        help="0 = whole file. Must match the AP run.")
    parser.add_argument(
        "--dataset", default="datasets/gender_test_rephrased_v2.json",
        help="Must match the AP run's --dataset, or the merge drops rows.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-ig-steps", type=int, default=10)
    parser.add_argument("--ig-rule", choices=["midpoint", "right"],
                        default="midpoint")
    parser.add_argument("--ig-path", choices=["embed_mean", "zero"],
                        default="embed_mean")
    parser.add_argument("--metric", choices=["logit", "logit_diff"],
                        default="logit_diff")
    parser.add_argument("--multi-token-policy", choices=["skip", "first"],
                        default="skip")
    parser.add_argument("--topk-mode", choices=["abs", "signed"],
                        default="abs")
    parser.add_argument("--ablation-modes", nargs="+",
                        default=["zero", "mean"], choices=["zero", "mean"])
    parser.add_argument("--no-atp", action="store_true",
                        help="Skip the m=1 attribution-patching baseline")
    parser.add_argument("--no-s3", action="store_true")
    parser.add_argument("--skip-validation", action="store_true")
    parser.add_argument("--convergence-check", action="store_true",
                        help="Report EAP-IG rank stability over m and exit")
    parser.add_argument("--summary-only", action="store_true")
    parser.add_argument("--enable-tl-grad-flags", action="store_true")
    parser.add_argument("--tag", default="")
    args = parser.parse_args()

    s3_utils.set_use_s3(not args.no_s3)
    output_dir = "outputs/dla_error_analysis"
    suffix = f"_{args.tag}" if args.tag else ""

    if args.summary_only:
        print("\n  SUMMARY-ONLY MODE\n")
        all_summaries = []
        for model_name in args.models:
            safe = model_name.replace("/", "_")
            for mode in args.ablation_modes:
                path = (f"{output_dir}/"
                        f"{safe}_eap_ig_{mode}_ablation_records{suffix}.csv")
                try:
                    eap_df = s3_utils.read_csv(path)
                except Exception:
                    print(f"  [SKIP] missing {path}")
                    continue
                ap_df = _load_ap_records(model_name, mode, output_dir, suffix)
                summary = compute_summary_eap_ig(
                    eap_df, model_name, mode, ap_df=ap_df,
                    topk_mode=args.topk_mode)
                summary["n_ig_steps"] = args.n_ig_steps
                all_summaries.append(summary)
                print_eap_ig_report(summary)
        s3_utils.write_json(
            all_summaries,
            f"{output_dir}/summary_all_models_eap_ig{suffix}.json")
        print("Done.")
        return

    hf_token = os.environ.get("HF_TOKEN", "")
    if hf_token:
        login(token=hf_token)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    dataset = s3_utils.read_json(args.dataset)
    random.seed(args.seed)
    if args.n_examples and len(dataset) > args.n_examples:
        examples = random.sample(dataset, args.n_examples)
        print(f"Sampled {len(examples)} of {len(dataset)} "
              f"(seed={args.seed})\n")
    else:
        examples = list(dataset)
        print(f"Using all {len(examples)} examples\n")

    if args.convergence_check:
        for model_name in args.models:
            run_ig_convergence_check(
                model_name, examples, device, args.metric,
                args.multi_token_policy, args.ig_path,
                args.enable_tl_grad_flags)
        return

    if not args.skip_validation:
        print("\n" + "=" * 70)
        print(f"  PHASE 1: EAP-IG validation ({VALIDATION_N} examples)")
        print("=" * 70)
        for model_name in args.models:
            run_validation_checkpoint_eap_ig(
                model_name, examples, device, args.n_ig_steps, output_dir,
                args.metric, args.multi_token_policy, args.ig_rule,
                args.ig_path, args.enable_tl_grad_flags, suffix)
        print("\n[ALL VALIDATION PASSED]\n")

    print("\n" + "=" * 70)
    print(f"  PHASE 2: Full EAP-IG analysis ({len(examples)} examples, "
          f"m={args.n_ig_steps})")
    print("=" * 70)

    all_summaries = []
    for model_name in args.models:
        records_by_mode, stats = analyze_model_eap_ig(
            model_name, examples, device,
            ablation_modes=tuple(args.ablation_modes),
            n_ig_steps=args.n_ig_steps, ig_rule=args.ig_rule,
            ig_path=args.ig_path, metric_mode=args.metric,
            multi_token_policy=args.multi_token_policy,
            run_gates=args.skip_validation,
            enable_grad_flags=args.enable_tl_grad_flags,
            also_atp=not args.no_atp)

        nc = max(1, stats["n_candidates"])
        safe = model_name.replace("/", "_")
        for mode, records in records_by_mode.items():
            df = pd.DataFrame(records)
            csv_path = (f"{output_dir}/"
                        f"{safe}_eap_ig_{mode}_ablation_records{suffix}.csv")
            s3_utils.write_csv(df, csv_path)
            print(f"  -> saved {len(df)} records to {csv_path}")

            ap_df = _load_ap_records(model_name, mode, output_dir, suffix)
            summary = compute_summary_eap_ig(
                df, model_name, mode, ap_df=ap_df, topk_mode=args.topk_mode)
            summary["multi_token_pct"] = round(
                100.0 * stats["n_multi_token"] / nc, 1)
            summary["n_ig_steps"] = args.n_ig_steps
            summary["ig_rule"] = args.ig_rule
            summary["ig_path"] = args.ig_path
            summary["metric"] = args.metric
            all_summaries.append(summary)
            print_eap_ig_report(summary)

    s3_utils.write_json(
        all_summaries,
        f"{output_dir}/summary_all_models_eap_ig{suffix}.json")
    print("\nDone.")


if __name__ == "__main__":
    main()