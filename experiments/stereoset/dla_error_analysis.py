import argparse
import math
import os
import random
import sys

import numpy as np
import pandas as pd
import torch
from huggingface_hub import login
from scipy.stats import spearmanr
from transformer_lens import HookedTransformer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import s3_utils

s3_utils.set_use_s3("--no-s3" not in sys.argv)


SENTENCEPIECE_MODELS = {"gemma", "llama", "mistral", "t5"}
BPE_MODELS = {"gpt2", "gpt-j", "opt", "llama-3"}

GEMMA_2_SOFT_CAP = 30.0
VALIDATION_N = 5
NMAE_PLAUSIBLE_HI = 200.0


def get_model_family(model_name: str) -> str:
    name_lower = model_name.lower()
    for family in BPE_MODELS:
        if family in name_lower:
            return "bpe"
    for family in SENTENCEPIECE_MODELS:
        if family in name_lower:
            return "sentencepiece"
    return "unknown"


def tokenize_candidate(model, word: str, model_name: str):
    family = get_model_family(model_name)
    if family == "sentencepiece":
        DUMMY = "The"
        dummy_ids = model.tokenizer.encode(DUMMY, add_special_tokens=False)
        combined_ids = model.tokenizer.encode(
            f"{DUMMY} {word}", add_special_tokens=False
        )
        candidate_ids = combined_ids[len(dummy_ids):]
    else:
        candidate_ids = model.tokenizer.encode(
            " " + word, add_special_tokens=False
        )

    if len(candidate_ids) == 0:
        print(f"[WARNING] Word '{word}' produced 0 tokens. Skipping.")
        return None

    first_decoded = model.tokenizer.decode([candidate_ids[0]])
    assert first_decoded.startswith((" ", "\u0120", "\u2581")), (
        f"Expected space-prefixed token for '{word}', got {first_decoded!r} "
        f"(id={candidate_ids[0]})")
    return candidate_ids


def get_ln_final_weight(model):
    ln = model.ln_final
    if hasattr(ln, "w") and ln.w is not None:
        return ln.w
    return None


def validate_model_compatibility(model):
    cfg = model.cfg
    ln = model.ln_final
    ln_type = type(ln).__name__
    ln_weight = get_ln_final_weight(model)
    if ln_weight is not None:
        print(f"[OK] ln_final type : {ln_type} — learnable gamma "
              f"(shape: {ln_weight.shape})")
    else:
        print(f"[OK] ln_final type : {ln_type} — weightless (RMSNormPre)")
    assert hasattr(ln, "hook_scale"), (
        f"ln_final ({ln_type}) has no hook_scale — "
        f"update TransformerLens (>= 0.13)"
    )
    print("[OK] ln_final.hook_scale is present.")
    n_heads = cfg.n_heads
    n_kv = getattr(cfg, "n_key_value_heads", None) or n_heads
    tag = f"GQA n_heads={n_heads} n_kv={n_kv}" if n_kv < n_heads else f"MHA n_heads={n_heads}"
    print(f"[OK] {tag}")
    act_fn = cfg.act_fn
    gated = getattr(cfg, "gated_mlp", False) or act_fn in (
        "silu", "geglu", "swiglu", "gelu_new")
    print(f"[OK] act_fn={act_fn} → {'GatedMLP' if gated else 'StandardMLP'}")
    print(f"[READY] '{cfg.model_name}' | layers={cfg.n_layers} | "
          f"d_model={cfg.d_model} | d_head={cfg.d_head}\n")


def load_model(model_name, device):
    model = HookedTransformer.from_pretrained(model_name, device=device)
    name_lower = model_name.lower()
    if "llama-3" in name_lower:
        model.set_use_split_qkv_input(True)
        model.set_use_attn_result(True)
        model.set_use_hook_mlp_in(True)
        print(f"[LOAD] {model_name}: enabling GQA flags "
              f"(use_split_qkv_input, use_attn_result, use_hook_mlp_in)")
    return model


def validate_hook_z_shape(model, example):
    context = example["rephrased_context"].split("BLANK")[0].strip()
    tokens = model.to_tokens(context)
    with torch.no_grad():
        _, cache = model.run_with_cache(tokens, return_type=None)
    z = cache["blocks.0.attn.hook_z"]
    expected = (1, tokens.shape[1], model.cfg.n_heads, model.cfg.d_head)
    assert z.shape == expected, (
        f"hook_z shape mismatch: got {z.shape}, expected {expected}")
    print(f"[OK] hook_z shape validated: {z.shape}")
    del cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def resolve_softcap(model, model_name):
    softcap = getattr(model.cfg, "output_logits_soft_cap", None)
    if softcap is not None and softcap > 0:
        if "gemma-2" in model_name.lower() and not math.isclose(
                float(softcap), GEMMA_2_SOFT_CAP):
            raise ValueError(
                f"{model_name} exposes output_logits_soft_cap={softcap}; "
                f"expected {GEMMA_2_SOFT_CAP}")
        print(f"[SOFTCAP] output_logits_soft_cap = {softcap}")
        return float(softcap)
    if "gemma-2" in model_name.lower():
        print(f"[SOFTCAP] Field absent/non-positive; "
              f"falling back to {GEMMA_2_SOFT_CAP} for {model_name}")
        return GEMMA_2_SOFT_CAP
    print("[SOFTCAP] None")
    return None


def build_component_catalog(model):
    components = []
    for layer in range(model.cfg.n_layers):
        for h in range(model.cfg.n_heads):
            components.append((layer, "head", h))
        components.append((layer, "mlp", None))
    return components


@torch.no_grad()
def compute_dla_and_vectors(model, cache, target_token_id):
    ln_scale = cache["ln_final.hook_scale"][0, -1, 0]
    ln_weight = get_ln_final_weight(model)
    raw_unembed_dir = model.W_U[:, target_token_id]

    if ln_weight is not None:
        effective_unembed = (ln_weight / ln_scale) * raw_unembed_dir
    else:
        effective_unembed = raw_unembed_dir / ln_scale

    components = []
    dla_scores = []
    comp_vectors = []

    for layer in range(model.cfg.n_layers):
        attn_z = cache[f"blocks.{layer}.attn.hook_z"][0, -1]  # [n_heads, d_head]
        W_O = model.W_O[layer]  # [n_heads, d_head, d_model]
        for h in range(model.cfg.n_heads):
            head_vec = attn_z[h] @ W_O[h]  # [d_model]
            dla = torch.dot(head_vec, effective_unembed).item()
            components.append((layer, "head", h))
            dla_scores.append(dla)
            comp_vectors.append(head_vec)

        mlp_vec = cache[f"blocks.{layer}.hook_mlp_out"][0, -1]  # [d_model]
        dla = torch.dot(mlp_vec, effective_unembed).item()
        components.append((layer, "mlp", None))
        dla_scores.append(dla)
        comp_vectors.append(mlp_vec)

    return dla_scores, comp_vectors, components

@torch.no_grad()
def build_mean_baseline_vectors(model, means):
    """Return list of c_mean vectors aligned with build_component_catalog(model)."""
    head_means, mlp_means = means
    c_means = []
    for layer in range(model.cfg.n_layers):
        W_O = model.W_O[layer]
        for h in range(model.cfg.n_heads):
            c_means.append(head_means[(layer, h)] @ W_O[h])   # [d_model]
        c_means.append(mlp_means[layer])                       # [d_model]
    return c_means

@torch.no_grad()
def compute_de(model, cache, target_token_id, comp_vectors,
               clean_logit_precap, softcap=None, c_mean_vectors=None):
    """DE(c) = logit(LN(r)) - logit(LN(r_ablated)) for every component c.

    r_ablated is (r - c) for zero-mode, (r - c + c_mean) for mean-mode.
    Also returns rho = scale(r) / scale(r-c), the frozen-norm ratio
    (always computed on the zero-baseline r-c regardless of mode).
    """
    n_layers = model.cfg.n_layers
    r = cache[f"blocks.{n_layers - 1}.hook_resid_post"][0, -1]  # [d_model]

    c_stack = torch.stack(comp_vectors)        # [N, d_model]
    r_minus_c = r.unsqueeze(0) - c_stack       # [N, d_model]

    if c_mean_vectors is not None:
        c_mean_stack = torch.stack(c_mean_vectors)
        r_ablated = r_minus_c + c_mean_stack
    else:
        r_ablated = r_minus_c

    # rho: frozen-norm ratio
    sigma_r = cache["ln_final.hook_scale"][0, -1, 0]  # scalar
    uses_rms = "RMS" in type(model.ln_final).__name__
    if uses_rms:
        sigma_rmc = torch.sqrt(
            (r_minus_c ** 2).mean(dim=-1) + model.cfg.eps)
    else:
        centered = r_minus_c - r_minus_c.mean(dim=-1, keepdim=True)
        sigma_rmc = torch.sqrt(
            (centered ** 2).mean(dim=-1) + model.cfg.eps)
    rho = (sigma_r / sigma_rmc).cpu().numpy()  # [N]

    # DE from r_ablated
    normed = model.ln_final(r_ablated.unsqueeze(1))[:, 0, :]  # [N, d_model]
    ablated_logits = normed @ model.W_U[:, target_token_id]    # [N]

    de_precap = (clean_logit_precap - ablated_logits).cpu().numpy()

    de_postcap = None
    if softcap is not None:
        clean_post = softcap * math.tanh(clean_logit_precap / softcap)
        ablated_post = softcap * torch.tanh(ablated_logits / softcap)
        de_postcap = (clean_post - ablated_post).cpu().numpy()

    return de_precap, de_postcap, rho


@torch.no_grad()
def compute_activation_means(model, examples):
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    device = next(model.parameters()).device

    head_accum: dict[tuple[int, int], list[torch.Tensor]] = {}
    mlp_accum: dict[int, list[torch.Tensor]] = {}

    for example in examples:
        context = example["rephrased_context"].split("BLANK")[0].strip()
        tokens = model.to_tokens(context)
        _, cache = model.run_with_cache(tokens, return_type=None)

        for layer in range(n_layers):
            z = cache[f"blocks.{layer}.attn.hook_z"][0, -1]   # [n_heads, d_head]
            for h in range(n_heads):
                head_accum.setdefault((layer, h), []).append(z[h].cpu())

            mlp_accum.setdefault(layer, []).append(
                cache[f"blocks.{layer}.hook_mlp_out"][0, -1].cpu()
            )

        del cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    head_means = {
        k: torch.stack(v).mean(0).to(device) for k, v in head_accum.items()
    }
    mlp_means = {
        k: torch.stack(v).mean(0).to(device) for k, v in mlp_accum.items()
    }
    return head_means, mlp_means


def _build_ablation_hooks(chunk_components, means=None):
    head_means, mlp_means = means if means is not None else (None, None)

    hook_specs: dict[tuple, dict] = {}
    for row_idx, (layer, comp_type, head_idx) in enumerate(chunk_components):
        key = (layer, comp_type)
        hook_specs.setdefault(key, {})[row_idx] = head_idx

    hooks = []
    for (layer, comp_type), row_map in hook_specs.items():
        if comp_type == "head":
            name = f"blocks.{layer}.attn.hook_z"

            def _make_head_hook(rm=dict(row_map), _layer=layer,
                                _hm=head_means):
                def _hook(act, hook):
                    for ri, hi in rm.items():
                        if _hm is not None:
                            act[ri, -1, hi, :] = _hm[(_layer, hi)]
                        else:
                            act[ri, -1, hi, :] = 0.0
                    return act
                return _hook


            hooks.append((name, _make_head_hook()))
        else:
            name = f"blocks.{layer}.hook_mlp_out"

            def _make_mlp_hook(rm=dict(row_map), _layer=layer,
                               _mm=mlp_means):
                def _hook(act, hook):
                    for ri in rm:
                        if _mm is not None:
                            act[ri, -1, :] = _mm[_layer]
                        else:
                            act[ri, -1, :] = 0.0
                    return act
                return _hook

            hooks.append((name, _make_mlp_hook()))
    return hooks


@torch.no_grad()
def compute_ap(model, tokens, target_token_id, components,
               patch_batch_size, clean_logit_precap, softcap=None,
               means=None):

    n_layers = model.cfg.n_layers
    n_comp = len(components)
    ap_precap = np.zeros(n_comp)
    ap_postcap = np.zeros(n_comp) if softcap is not None else None

    for chunk_start in range(0, n_comp, patch_batch_size):
        chunk_end = min(chunk_start + patch_batch_size, n_comp)
        chunk = components[chunk_start:chunk_end]
        chunk_size = len(chunk)

        batch_tokens = tokens.repeat(chunk_size, 1)

        captured: dict = {}

        def _capture(act, hook, _c=captured):
            _c["resid"] = act.detach()
            return act

        abl_hooks = _build_ablation_hooks(chunk, means=means)
        final_hook = (f"blocks.{n_layers - 1}.hook_resid_post", _capture)

        with torch.no_grad():
            model.run_with_hooks(
                batch_tokens,
                fwd_hooks=abl_hooks + [final_hook],
                return_type=None,
            )

        resid = captured["resid"][:, -1, :]                    # [B, d_model]
        normed = model.ln_final(resid.unsqueeze(1))[:, 0, :]   # [B, d_model]
        abl_logits = normed @ model.W_U[:, target_token_id]    # [B]

        chunk_ap = clean_logit_precap - abl_logits.cpu().numpy()
        ap_precap[chunk_start:chunk_end] = chunk_ap

        if softcap is not None:
            clean_post = softcap * math.tanh(clean_logit_precap / softcap)
            abl_post = softcap * np.tanh(abl_logits.cpu().numpy() / softcap)
            ap_postcap[chunk_start:chunk_end] = clean_post - abl_post

        del captured, resid, normed, abl_logits, batch_tokens
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return ap_precap, ap_postcap



# def nmae(a: np.ndarray, b: np.ndarray) -> float:
#     """NMAE = Σ|a−b| / Σ|b| × 100 %."""
#     denom = np.sum(np.abs(b))
#     if denom == 0:
#         return float("inf")
#     return float(np.sum(np.abs(a - b)) / denom * 100.0)


# take all fast-guess numbers, 
# take all true numbers, 
# see how far apart, 
# add up size of gaps, 
# divide by size of true numbers, times 100

def nmae_against(a, b, denom_source):
    """Σ|a−b| / Σ|denom_source| × 100 %."""
    denom = float(np.sum(np.abs(denom_source)))
    if denom == 0:
        return float("inf")
    return float(np.sum(np.abs(a - b)) / denom * 100.0)


def bootstrap_nmae_ci(a, b, example_ids, n_resamples=1000,
                       ci=0.95, seed=42, denom_source=None):
    """Percentile bootstrap CI for NMAE, resampling over examples."""
    rng = np.random.RandomState(seed)
    unique_ids = np.unique(example_ids)
    n_ex = len(unique_ids)

    id_to_idx: dict[str, list[int]] = {}
    for i, eid in enumerate(example_ids):
        id_to_idx.setdefault(eid, []).append(i)

    samples = np.empty(n_resamples)
    for s in range(n_resamples):
        drawn = rng.choice(unique_ids, size=n_ex, replace=True)
        idx = np.concatenate([id_to_idx[eid] for eid in drawn])
        if denom_source is not None:
            samples[s] = nmae_against(a[idx], b[idx], denom_source[idx])
        else:
            samples[s] = nmae(a[idx], b[idx])


    alpha = (1 - ci) / 2
    return float(np.percentile(samples, alpha * 100)), \
           float(np.percentile(samples, (1 - alpha) * 100))


def bootstrap_spearman_delta_ci(rhos_a, rhos_b, n_resamples=1000,
                                ci=0.95, seed=42):
    """Paired percentile bootstrap CI for Δρ = median(rhos_b) − median(rhos_a).

    ``rhos_a`` and ``rhos_b`` are aligned per-example Spearman correlation
    arrays (one scalar per example, same ordering).  Resampling draws
    example indices with replacement so the within-example pairing between
    the two methods is preserved.

    Returns ``(delta_point, ci_lo, ci_hi, p_two_sided)`` where *p_two_sided*
    is 2 · min(P(Δρ_b ≤ 0), P(Δρ_b ≥ 0)) from the bootstrap distribution.
    """
    rhos_a = np.asarray(rhos_a, dtype=float)
    rhos_b = np.asarray(rhos_b, dtype=float)
    assert len(rhos_a) == len(rhos_b), (
        "per-example rho arrays must have equal length")

    n_ex = len(rhos_a)
    delta_point = float(np.median(rhos_b) - np.median(rhos_a))

    rng = np.random.RandomState(seed)
    deltas = np.empty(n_resamples)
    for s in range(n_resamples):
        idx = rng.choice(n_ex, size=n_ex, replace=True)
        deltas[s] = np.median(rhos_b[idx]) - np.median(rhos_a[idx])

    alpha_half = (1 - ci) / 2
    ci_lo = float(np.percentile(deltas, alpha_half * 100))
    ci_hi = float(np.percentile(deltas, (1 - alpha_half) * 100))

    frac_le_zero = np.mean(deltas <= 0)
    frac_ge_zero = np.mean(deltas >= 0)
    p_two_sided = float(min(2 * min(frac_le_zero, frac_ge_zero), 1.0))

    return delta_point, ci_lo, ci_hi, p_two_sided


def per_example_selection_metrics(records_df, ks=(5, 10, 25)):
    rhos, sign_agree = [], []
    jaccard = {k: [] for k in ks}
    for eid, sub in records_df.groupby("example_id"):
        dla = sub["dla"].values
        ap  = sub["ap_precap"].values
        rho, _ = spearmanr(dla, ap)
        rhos.append(rho)
        sign_agree.append(float(np.mean(np.sign(dla) == np.sign(ap))))
        for k in ks:
            top_dla = set(np.argsort(-np.abs(dla))[:k])
            top_ap  = set(np.argsort(-np.abs(ap))[:k])
            jaccard[k].append(len(top_dla & top_ap) / len(top_dla | top_ap))
    return {
        "spearman_median":  float(np.median(rhos)),
        "spearman_iqr":     [float(np.percentile(rhos, 25)),
                             float(np.percentile(rhos, 75))],
        "sign_agreement":   float(np.mean(sign_agree)),
        **{f"top{k}_jaccard_median": float(np.median(jaccard[k])) for k in ks},
    }


def compute_summary(records_df, model_name, has_softcap):
    dla = records_df["dla"].values
    de = records_df["de_precap"].values
    ap = records_df["ap_precap"].values
    eids = records_df["example_id"].values

    sa = nmae_against(dla, de, ap)
    sb = nmae_against(de,  ap, ap)
    st = nmae_against(dla, ap, ap)

    ci_a = bootstrap_nmae_ci(dla, de, eids, denom_source=ap)
    ci_b = bootstrap_nmae_ci(de, ap, eids, denom_source=ap)
    ci_t = bootstrap_nmae_ci(dla, ap, eids, denom_source=ap)

    summary: dict = {
        "model": model_name,
        "n_records": len(records_df),
        "n_examples": int(records_df["example_id"].nunique()),
        "source_a_frozen_norm_nmae": round(sa, 2),
        "source_a_ci_lo": round(ci_a[0], 2),
        "source_a_ci_hi": round(ci_a[1], 2),
        "source_b_indirect_effect_nmae": round(sb, 2),
        "source_b_ci_lo": round(ci_b[0], 2),
        "source_b_ci_hi": round(ci_b[1], 2),
        "total_dla_error_nmae": round(st, 2),
        "total_ci_lo": round(ci_t[0], 2),
        "total_ci_hi": round(ci_t[1], 2),
    }

    sel = per_example_selection_metrics(records_df)
    summary.update(sel)

    if "rho" in records_df.columns and "clean_logit_precap" in records_df.columns:
        rho_arr = records_df["rho"].values
        clean_arr = records_df["clean_logit_precap"].values
        de_pred = clean_arr * (1.0 - rho_arr) + rho_arr * dla
        summary["frozen_norm_identity_max_residual"] = float(
            np.max(np.abs(de - de_pred))
        )

    if has_softcap:
        de_pre = records_df["de_precap"].values
        de_post = records_df["de_postcap"].values
        ap_pre = records_df["ap_precap"].values
        ap_post = records_df["ap_postcap"].values

        de_dist = np.abs(de_post - de_pre)
        ap_dist = np.abs(ap_post - ap_pre)
        de_denom = np.sum(np.abs(de_pre))
        ap_denom = np.sum(np.abs(ap_pre))

        summary["cap_distortion_de_mean_abs"] = round(float(np.mean(de_dist)), 4)
        summary["cap_distortion_de_pct_of_precap"] = (
            round(float(np.sum(de_dist) / de_denom * 100), 2)
            if de_denom > 0 else float("inf")
        )
        summary["cap_distortion_ap_mean_abs"] = round(float(np.mean(ap_dist)), 4)
        summary["cap_distortion_ap_pct_of_precap"] = (
            round(float(np.sum(ap_dist) / ap_denom * 100), 2)
            if ap_denom > 0 else float("inf")
        )

    return summary


def print_report(summary):
    print(f"\n{'─' * 60}")
    abl = summary.get("ablation_type", "")
    label = f" ({abl} ablation)" if abl else ""
    print(f"Model : {summary['model']}{label}")
    print(f"Records: {summary['n_records']}  |  "
          f"Examples: {summary['n_examples']}")
    print(f"{'─' * 60}")
    print(f"  Source A  NMAE(DLA,DE) = {summary['source_a_frozen_norm_nmae']:.2f}%  "
          f"[{summary['source_a_ci_lo']:.2f}%, {summary['source_a_ci_hi']:.2f}%]")
    print(f"  Source B  NMAE(DE,AP)  = {summary['source_b_indirect_effect_nmae']:.2f}%  "
          f"[{summary['source_b_ci_lo']:.2f}%, {summary['source_b_ci_hi']:.2f}%]")
    print(f"  Total     NMAE(DLA,AP) = {summary['total_dla_error_nmae']:.2f}%  "
          f"[{summary['total_ci_lo']:.2f}%, {summary['total_ci_hi']:.2f}%]")
    if "spearman_median" in summary:
        print(f"  Selection:  Spearman median={summary['spearman_median']:.3f}  "
              f"IQR=[{summary['spearman_iqr'][0]:.3f}, "
              f"{summary['spearman_iqr'][1]:.3f}]  "
              f"sign-agree={summary['sign_agreement']*100:.1f}%")
        print(f"              top-k Jaccard  "
              f"5={summary['top5_jaccard_median']:.2f}  "
              f"10={summary['top10_jaccard_median']:.2f}  "
              f"25={summary['top25_jaccard_median']:.2f}")
    if "frozen_norm_identity_max_residual" in summary:
        print(f"  Identity:   max |DE - predicted| = "
              f"{summary['frozen_norm_identity_max_residual']:.4e}")
    if "cap_distortion_de_mean_abs" in summary:
        print(f"\n  [SOFTCAP DISTORTION]")
        print(f"    DE  mean |post-pre| = "
              f"{summary['cap_distortion_de_mean_abs']:.4f}  "
              f"({summary['cap_distortion_de_pct_of_precap']:.2f}% of |pre-cap|)")
        print(f"    AP  mean |post-pre| = "
              f"{summary['cap_distortion_ap_mean_abs']:.4f}  "
              f"({summary['cap_distortion_ap_pct_of_precap']:.2f}% of |pre-cap|)")
    if "multi_token_pct" in summary:
        print(f"  Tokenization: {summary['multi_token_pct']:.1f}% multi-token")
    print(f"{'─' * 60}")


def analyze_model(model_name, examples, patch_batch_size, device,
                  ablation_modes=("zero", "mean")):
    """DLA / DE / AP analysis for model with dual ablation modes.

    Returns (records_by_mode, has_softcap, (n_multi_token, n_candidates)).
    DLA is computed once; DE is computed per ablation mode (inside the
    cache-alive block); AP runs once per mode.
    """
    print(f"\n{'=' * 70}")
    print(f"  Analyzing: {model_name}  "
          f"(ablation modes: {', '.join(ablation_modes)})")
    print(f"{'=' * 70}")

    model = load_model(model_name, device)
    validate_model_compatibility(model)
    model.eval()

    softcap = resolve_softcap(model, model_name)
    validate_hook_z_shape(model, examples[0])

    n_layers = model.cfg.n_layers
    catalog = build_component_catalog(model)
    n_comp = len(catalog)
    print(f"Components: {n_comp}  "
          f"({model.cfg.n_layers}L x {model.cfg.n_heads}H + "
          f"{model.cfg.n_layers} MLPs)")

    means = None
    c_mean_vecs = None
    if "mean" in ablation_modes:
        means = compute_activation_means(model, examples)
        c_mean_vecs = build_mean_baseline_vectors(model, means)
        print(f"[MEAN-ABL] Precomputed activation means over "
              f"{len(examples)} examples")

    records_by_mode: dict[str, list[dict]] = {m: [] for m in ablation_modes}
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
                _, cache = model.run_with_cache(tokens, return_type=None)

                r = cache[f"blocks.{n_layers - 1}.hook_resid_post"][0, -1]
                r_normed = model.ln_final(
                    r.unsqueeze(0).unsqueeze(0)
                )[0, 0]
                clean_logit = (
                    r_normed @ model.W_U[:, target_id]
                ).item()

                dla_scores, comp_vecs, components = (
                    compute_dla_and_vectors(model, cache, target_id)
                )

                de_by_mode: dict[str, tuple] = {}
                for mode in ablation_modes:
                    c_mean_for_de = c_mean_vecs if mode == "mean" else None
                    de_by_mode[mode] = compute_de(
                        model, cache, target_id, comp_vecs,
                        clean_logit, softcap=softcap,
                        c_mean_vectors=c_mean_for_de,
                    )

            del cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            for mode in ablation_modes:
                ap_means = means if mode == "mean" else None
                ap_pre, ap_post = compute_ap(
                    model, tokens, target_id, components,
                    patch_batch_size, clean_logit, softcap=softcap,
                    means=ap_means,
                )

                de_pre_m, de_post_m, rho_m = de_by_mode[mode]

                if ex_idx == 0 and mode == "zero":
                    last_mlp = next(
                        i for i, (l, ct, _) in enumerate(components)
                        if l == n_layers - 1 and ct == "mlp"
                    )
                    diff = abs(float(de_pre_m[last_mlp])
                               - float(ap_pre[last_mlp]))
                    print(f"[SANITY] last-MLP |DE - AP| = {diff:.4e}")
                    assert diff < 1e-3, (
                        f"Position-matched ablation broken: |delta|={diff}")

                for i, (layer, ctype, hidx) in enumerate(components):
                    rec = {
                        "model": model_name,
                        "example_id": ex_id,
                        "candidate": word,
                        "candidate_type": cand_type,
                        "target_token_id": target_id,
                        "clean_logit_precap": clean_logit,
                        "layer": layer,
                        "component_type": ctype,
                        "head_idx": hidx if hidx is not None else -1,
                        "dla": dla_scores[i],
                        "de_precap": float(de_pre_m[i]),
                        "ap_precap": float(ap_pre[i]),
                        "rho": float(rho_m[i]),
                    }
                    if softcap is not None:
                        rec["de_postcap"] = float(de_post_m[i])
                        rec["ap_postcap"] = float(ap_post[i])
                    records_by_mode[mode].append(rec)

    pct = 100.0 * n_multi_token / max(1, n_candidates)
    print(f"[TOKENIZATION] {n_multi_token}/{n_candidates} "
          f"multi-token ({pct:.1f}%)")

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return records_by_mode, softcap is not None, (n_multi_token, n_candidates)



def run_validation_checkpoint(model_name, examples, patch_batch_size,
                              device):
    """Run VALIDATION_N-example gate: check NMAE ranges and soft-cap."""
    print(f"\n{'─' * 60}")
    print(f"  Validating: {model_name} ({VALIDATION_N} examples)")
    print(f"{'─' * 60}")

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    val_examples = examples[:VALIDATION_N]
    records_by_mode, has_softcap, token_info = analyze_model(
        model_name, val_examples, patch_batch_size, device)

    if torch.cuda.is_available():
        peak_alloc_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
        peak_reserved_gb = torch.cuda.max_memory_reserved() / (1024 ** 3)
        print(f"[CUDA] Peak allocated: {peak_alloc_gb:.2f} GB  |  "
              f"Peak reserved: {peak_reserved_gb:.2f} GB")

    for mode, records in records_by_mode.items():
        df = pd.DataFrame(records)
        dla = df["dla"].values
        de = df["de_precap"].values
        ap = df["ap_precap"].values

        sa = nmae_against(dla, de, ap)
        sb = nmae_against(de,  ap, ap)
        st = nmae_against(dla, ap, ap)

        for label, val in [("Source A", sa), ("Source B", sb),
                           ("Total", st)]:
            if not math.isfinite(val):
                raise ValueError(
                    f"[VALIDATION FAIL] {model_name} ({mode}): "
                    f"{label} NMAE is non-finite ({val})")
            if val > NMAE_PLAUSIBLE_HI:
                raise ValueError(
                    f"[VALIDATION FAIL] {model_name} ({mode}): "
                    f"{label} NMAE = {val:.2f}% exceeds "
                    f"{NMAE_PLAUSIBLE_HI}%")

        print(f"[VALIDATION OK] {model_name} ({mode}): "
              f"Source A={sa:.1f}%  Source B={sb:.1f}%  "
              f"Total={st:.1f}%")

    if has_softcap:
        for mode, records in records_by_mode.items():
            df = pd.DataFrame(records)
            de_diff = (df["de_postcap"] - df["de_precap"]).abs().sum()
            ap_diff = (df["ap_postcap"] - df["ap_precap"]).abs().sum()
            if de_diff == 0:
                raise ValueError(
                    f"[VALIDATION FAIL] {model_name} ({mode}): "
                    f"de_postcap == de_precap -- soft-cap not applied")
            print(f"[SOFTCAP OK] {model_name} ({mode}): "
                  f"sum|de_post - de_pre| = {de_diff:.4f}, "
                  f"sum|ap_post - ap_pre| = {ap_diff:.4f}")


DEFAULT_MODELS = [
    "gpt2-xl",
    "meta-llama/Llama-3.2-1B",
    "google/gemma-2-2b",
]


def main():
    parser = argparse.ArgumentParser(
        description="DLA error budget: DLA vs Direct Effect vs "
                    "Activation Patching"
    )
    parser.add_argument(
        "--models", nargs="+", default=DEFAULT_MODELS,
        help="HuggingFace / TransformerLens model names",
    )
    parser.add_argument("--n-examples", type=int, default=100,
                        help="Number of StereoSet examples to sample")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patch-batch-size", type=int, default=16,
                        help="Batch size for AP forward passes")
    parser.add_argument("--no-s3", action="store_true",
                        help="Use local disk instead of S3")
    parser.add_argument("--skip-validation", action="store_true",
                        help="Skip the 5-example validation checkpoint")
    args = parser.parse_args()

    s3_utils.set_use_s3(not args.no_s3)

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

    output_dir = "outputs/dla_error_analysis"

    # ── Phase 1: Validation checkpoint ─────────────────────────────
    if not args.skip_validation:
        print("\n" + "=" * 70)
        print(f"  PHASE 1: Validation checkpoint "
              f"({VALIDATION_N} examples per model)")
        print("=" * 70)
        for model_name in args.models:
            run_validation_checkpoint(
                model_name, examples, args.patch_batch_size, device)
        print("\n[ALL VALIDATION PASSED] Proceeding to full analysis.\n")
    else:
        print("[SKIP] Validation checkpoint skipped.\n")

    # ── Phase 2: Full analysis ─────────────────────────────────────
    print("\n" + "=" * 70)
    print(f"  PHASE 2: Full analysis "
          f"({len(examples)} examples per model)")
    print("=" * 70)

    all_summaries: list[dict] = []

    for model_name in args.models:
        records_by_mode, has_softcap, token_info = analyze_model(
            model_name, examples,
            args.patch_batch_size, device,
        )
        n_multi, n_cands = token_info
        mt_pct = 100.0 * n_multi / max(1, n_cands)

        safe = model_name.replace("/", "_")
        for mode, records in records_by_mode.items():
            df = pd.DataFrame(records)
            csv_path = (f"{output_dir}/"
                        f"{safe}_{mode}_ablation_records.csv")
            s3_utils.write_csv(df, csv_path)
            print(f"  -> saved {len(df)} records to {csv_path}")

            summary = compute_summary(df, model_name, has_softcap)
            summary["ablation_type"] = mode
            summary["multi_token_pct"] = round(mt_pct, 1)
            all_summaries.append(summary)
            print_report(summary)

    summary_path = f"{output_dir}/summary_all_models.json"
    s3_utils.write_json(all_summaries, summary_path)
    print(f"\nUnified summary → {summary_path}")
    print("Done.")


if __name__ == "__main__":
    main()
