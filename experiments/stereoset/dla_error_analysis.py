"""DLA / DE / AP error-budget analysis.
"""

import argparse
import math
import os
import random
import sys

import numpy as np
import pandas as pd
import torch
from huggingface_hub import login
from scipy.stats import kendalltau, spearmanr
from transformer_lens import HookedTransformer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import s3_utils

s3_utils.set_use_s3("--no-s3" not in sys.argv)


SENTENCEPIECE_MODELS = {"gemma", "llama", "mistral", "t5"}
BPE_MODELS = {"gpt2", "gpt-j", "opt", "llama-3"}

GEMMA_2_SOFT_CAP = 30.0
VALIDATION_N = 5
NMAE_PLAUSIBLE_HI = 200.0

DECOMP_TOL = 2e-3
IDENTITY_TOL = 1e-3

STEREO_KEYS = ("stereotype", "stereo", "bias", "biased")
ANTI_KEYS = ("anti-stereotype", "anti_stereotype", "antistereotype",
             "anti", "unbiased")


class MetricSpec:

    def __init__(self, token_ids, signs, unit_id, label, words):
        assert len(token_ids) == len(signs)
        self.token_ids = list(token_ids)
        self.signs = list(signs)
        self.unit_id = unit_id
        self.label = label
        self.words = words

    def columns(self, model):
        """[d_model, k] slice of W_U for this metric's tokens."""
        return model.W_U[:, self.token_ids]

    def sign_vec(self, device, dtype):
        return torch.tensor(self.signs, device=device, dtype=dtype)

    def direction(self, model):
        """sum_i sign_i * W_U[:, t_i]   ->  [d_model]."""
        cols = self.columns(model)
        signs = self.sign_vec(cols.device, cols.dtype)
        return cols @ signs

    def apply_cap(self, logits, softcap):
        """logits: [..., k] -> scalar metric per row, post soft-cap."""
        signs = self.sign_vec(logits.device, logits.dtype)
        if softcap is None:
            return logits @ signs
        return (softcap * torch.tanh(logits / softcap)) @ signs


def _find_key(targets, candidates):
    lowered = {k.lower().replace(" ", "").replace("_", "-"): k
               for k in targets}
    for cand in candidates:
        norm = cand.lower().replace(" ", "").replace("_", "-")
        if norm in lowered:
            return lowered[norm]
    return None


def build_metric_specs(model, example, metric_mode, multi_token_policy,
                       stats):
    """Return a list of MetricSpec for one example (may be empty).

    ``stats`` is a dict accumulating {"n_candidates", "n_multi_token",
    "n_skipped"}.
    """
    targets = example["targets"]
    model_name = model.cfg.model_name

    def _tok(word):
        stats["n_candidates"] += 1
        toks = tokenize_candidate(model, word, model_name)
        if toks is None:
            stats["n_skipped"] += 1
            return None
        if len(toks) > 1:
            stats["n_multi_token"] += 1
            if multi_token_policy == "skip":
                stats["n_skipped"] += 1
                return None
        return toks[0]

    if metric_mode == "logit":
        specs = []
        for cand_type, word in targets.items():
            tid = _tok(word)
            if tid is None:
                continue
            specs.append(MetricSpec([tid], [1.0], unit_id=word,
                                    label=cand_type, words=[word]))
        return specs

    # logit_diff  [M1]
    s_key = _find_key(targets, STEREO_KEYS)
    a_key = _find_key(targets, ANTI_KEYS)
    if s_key is None or a_key is None:
        raise KeyError(
            f"--metric logit_diff needs stereotype and anti-stereotype keys "
            f"in example['targets']; got {sorted(targets)}. "
            f"Extend STEREO_KEYS / ANTI_KEYS.")
    s_id = _tok(targets[s_key])
    a_id = _tok(targets[a_key])
    if s_id is None or a_id is None:
        return []
    if s_id == a_id:
        stats["n_skipped"] += 2
        return []
    unit = f"{targets[s_key]}|{targets[a_key]}"
    return [MetricSpec([s_id, a_id], [1.0, -1.0], unit_id=unit,
                       label="stereo_minus_anti",
                       words=[targets[s_key], targets[a_key]])]


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
    assert first_decoded.startswith((" ", "Ġ", "▁")), (
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
        print(f"[OK] ln_final type : {ln_type} — weightless (folded/RMSNormPre)")
    assert hasattr(ln, "hook_scale"), (
        f"ln_final ({ln_type}) has no hook_scale — "
        f"update TransformerLens (>= 0.13)"
    )
    print("[OK] ln_final.hook_scale is present.")
    n_heads = cfg.n_heads
    n_kv = getattr(cfg, "n_key_value_heads", None) or n_heads
    tag = (f"GQA n_heads={n_heads} n_kv={n_kv}" if n_kv < n_heads
           else f"MHA n_heads={n_heads}")
    print(f"[OK] {tag}")
    # [S11] gelu_new is NOT a gated activation; report the cfg flag only.
    gated = bool(getattr(cfg, "gated_mlp", False))
    print(f"[OK] act_fn={cfg.act_fn}  gated_mlp={gated}")
    print(f"[READY] '{cfg.model_name}' | layers={cfg.n_layers} | "
          f"d_model={cfg.d_model} | d_head={cfg.d_head}\n")


def load_model(model_name, device, enable_grad_flags=False):
    model = HookedTransformer.from_pretrained(model_name, device=device)
    if enable_grad_flags:
        model.set_use_split_qkv_input(True)
        model.set_use_attn_result(True)
        model.set_use_hook_mlp_in(True)
        print(f"[LOAD] {model_name}: TL grad flags enabled (opt-in)")
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
def _attn_out_from_z(model, cache, layer):
    """Reconstruct the attention block's residual-stream write from hook_z."""
    z = cache[f"blocks.{layer}.attn.hook_z"]          # [b, p, n_heads, d_head]
    W_O = model.W_O[layer]                             # [n_heads, d_head, d_model]
    per_head = torch.einsum("bphd,hdm->bpm", z, W_O)
    return per_head + model.b_O[layer]


@torch.no_grad()
def compute_attn_gains(model, cache, tol=DECOMP_TOL):
    """Per-layer diagonal gain applied to the attention write, or None.

    Returns (gains, corrected) where gains[layer] is a [d_model] tensor or
    None, and ``corrected`` says whether any layer needed one.
    """
    gains = {}
    corrected = False
    for layer in range(model.cfg.n_layers):
        pre = cache[f"blocks.{layer}.hook_resid_pre"][0, -1]
        mid_key = f"blocks.{layer}.hook_resid_mid"
        if mid_key not in cache:
            gains[layer] = None
            continue
        target = cache[mid_key][0, -1] - pre
        prenorm = _attn_out_from_z(model, cache, layer)[0, -1]

        if (target - prenorm).abs().max().item() <= tol:
            gains[layer] = None
            continue

        ln = getattr(model.blocks[layer], "ln1_post", None)
        scale_key = f"blocks.{layer}.ln1_post.hook_scale"
        gain = None
        if ln is not None and scale_key in cache:
            S = cache[scale_key][0, -1, 0]
            w = getattr(ln, "w", None)
            gain = ((w / S) if w is not None
                    else torch.ones_like(prenorm) / S)
        else:
            # fallback: recover the diagonal empirically where it is defined
            denom = prenorm.clone()
            floor = 1e-6 * denom.abs().max()
            safe = denom.abs() > floor
            gain = torch.ones_like(denom)
            gain[safe] = target[safe] / denom[safe]
            if safe.sum() > 0:
                gain[~safe] = gain[safe].median()

        recon_err = (prenorm * gain - target).abs().max().item()
        if recon_err > tol:
            raise ValueError(
                f"[POST-NORM FAIL] layer {layer}: could not recover the "
                f"post-attention normalisation as a diagonal gain "
                f"(reconstruction error {recon_err:.3e} > {tol}). "
                f"ln1_post may not be an RMSNorm on this model, or "
                f"TransformerLens has renamed the hook. Inspect "
                f"model.blocks[{layer}] and the keys of the cache.")
        gains[layer] = gain
        corrected = True

    return gains, corrected


@torch.no_grad()
def head_writes(model, cache, layer, gains):
    """Per-head residual-stream writes at position -1:  [n_heads, d_model]."""
    z = cache[f"blocks.{layer}.attn.hook_z"][0, -1]     # [n_heads, d_head]
    per_head = torch.einsum("hd,hdm->hm", z, model.W_O[layer])
    gain = gains.get(layer) if gains else None
    if gain is not None:
        per_head = per_head * gain
    return per_head


@torch.no_grad()
def validate_residual_decomposition(model, example, tol=DECOMP_TOL):
    context = example["rephrased_context"].split("BLANK")[0].strip()
    tokens = model.to_tokens(context)
    _, cache = model.run_with_cache(tokens, return_type=None)

    has_mid = "blocks.0.hook_resid_mid" in cache
    gains, corrected = compute_attn_gains(model, cache, tol=tol)
    if corrected:
        print(f"[POST-NORM] {model.cfg.model_name}: post-attention "
              f"normalisation detected; per-head writes use the recovered "
              f"diagonal gain (see the note in the source).")

    worst_attn, worst_mlp, worst_layer = 0.0, 0.0, -1

    for layer in range(model.cfg.n_layers):
        pre = cache[f"blocks.{layer}.hook_resid_pre"][0, -1]
        post = cache[f"blocks.{layer}.hook_resid_post"][0, -1]
        mlp_out = cache[f"blocks.{layer}.hook_mlp_out"][0, -1]

        # per-head writes summed + the (gain-corrected) b_O term
        heads = head_writes(model, cache, layer, gains).sum(0)
        gain = gains.get(layer)
        bias = model.b_O[layer]
        attn_out = heads + (bias * gain if gain is not None else bias)

        if has_mid:
            mid = cache[f"blocks.{layer}.hook_resid_mid"][0, -1]
            d_attn = (mid - pre - attn_out).abs().max().item()
            d_mlp = (post - mid - mlp_out).abs().max().item()
        else:  # parallel attn+mlp block
            d_attn = 0.0
            d_mlp = (post - pre - attn_out - mlp_out).abs().max().item()

        if max(d_attn, d_mlp) > max(worst_attn, worst_mlp):
            worst_layer = layer
        worst_attn = max(worst_attn, d_attn)
        worst_mlp = max(worst_mlp, d_mlp)

    scale = post.abs().max().item()
    print(f"[DECOMP] {model.cfg.model_name}: max|resid_mid - resid_pre - "
          f"sum(head writes) - b_O| = {worst_attn:.3e}   "
          f"max|resid_post - resid_mid - mlp_out| = {worst_mlp:.3e}   "
          f"(|resid| ~ {scale:.2f}, worst layer {worst_layer})")

    del cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if max(worst_attn, worst_mlp) > tol:
        raise ValueError(
            f"[DECOMP FAIL] {model.cfg.model_name}: the residual stream is not "
            f"the sum of the per-head writes, b_O and hook_mlp_out "
            f"(max residual {max(worst_attn, worst_mlp):.3e} > {tol}). "
            f"DLA and DE are invalid for this model until the hook points "
            f"are changed.")
    print(f"[DECOMP OK] {model.cfg.model_name}"
          f"{'  (post-norm gain applied)' if corrected else ''}")
    return corrected


@torch.no_grad()
def validate_completeness(model, example, metric_spec, tol=DECOMP_TOL):
    """Catalog + residual bucket must reconstruct the clean metric exactly."""
    context = example["rephrased_context"].split("BLANK")[0].strip()
    tokens = model.to_tokens(context)
    _, cache = model.run_with_cache(tokens, return_type=None)

    n_layers = model.cfg.n_layers
    r = cache[f"blocks.{n_layers - 1}.hook_resid_post"][0, -1]
    dla, comp_vecs, _, u_eff = compute_dla_and_vectors(
        model, cache, metric_spec)

    c_sum = torch.stack(comp_vecs).sum(0)
    bucket = r - c_sum                       # embeddings + b_O + everything else
    total = float(np.sum(dla)) + float(torch.dot(bucket, u_eff).item())

    r_normed = model.ln_final(r.unsqueeze(0).unsqueeze(0))[0, 0]
    clean = float((r_normed @ metric_spec.direction(model)).item())

    resid = abs(total - clean)
    frac_catalog = (float(np.sum(dla)) / clean) if clean != 0 else float("nan")
    print(f"[COMPLETE] clean={clean:.5f}  catalog+bucket={total:.5f}  "
          f"|diff|={resid:.3e}  catalog explains {frac_catalog * 100:.1f}%")

    del cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if resid > tol * max(1.0, abs(clean)):
        raise ValueError(
            f"[COMPLETENESS FAIL] |catalog+bucket - clean| = {resid:.3e}")
    return True


@torch.no_grad()
def validate_chunk_invariance(model, example, metric_spec, components,
                              softcap, tol=1e-3):
    """AP must not depend on patch_batch_size (guards hook/row leakage)."""
    context = example["rephrased_context"].split("BLANK")[0].strip()
    tokens = model.to_tokens(context)
    _, cache = model.run_with_cache(tokens, return_type=None)
    r = cache[f"blocks.{model.cfg.n_layers - 1}.hook_resid_post"][0, -1]
    r_normed = model.ln_final(r.unsqueeze(0).unsqueeze(0))[0, 0]
    clean_logits = r_normed @ metric_spec.columns(model)
    clean = float(metric_spec.apply_cap(clean_logits, None).item())
    del cache

    sub = components[: min(24, len(components))]
    ap1, _ = compute_ap(model, tokens, metric_spec, sub, 1, clean,
                        softcap=None)
    ap16, _ = compute_ap(model, tokens, metric_spec, sub, 16, clean,
                         softcap=None)
    diff = float(np.max(np.abs(ap1 - ap16)))
    print(f"[CHUNK] max|AP(bs=1) - AP(bs=16)| = {diff:.3e} over {len(sub)} "
          f"components")
    if diff > tol:
        raise ValueError(
            f"[CHUNK FAIL] AP depends on patch_batch_size (max diff {diff:.3e})"
            f" — ablation hooks are leaking across batch rows.")
    return True


#  DLA / DE / AP
@torch.no_grad()
def compute_dla_and_vectors(model, cache, metric_spec, gains=None):
    ln_scale = cache["ln_final.hook_scale"][0, -1, 0]
    ln_weight = get_ln_final_weight(model)
    raw_dir = metric_spec.direction(model)

    if ln_weight is not None:
        effective_unembed = (ln_weight / ln_scale) * raw_dir
    else:
        effective_unembed = raw_dir / ln_scale

    if gains is None:
        gains, _ = compute_attn_gains(model, cache)

    components = []
    comp_vectors = []

    for layer in range(model.cfg.n_layers):
        per_head = head_writes(model, cache, layer, gains)
        for h in range(model.cfg.n_heads):
            comp_vectors.append(per_head[h])
            components.append((layer, "head", h))
        comp_vectors.append(cache[f"blocks.{layer}.hook_mlp_out"][0, -1])
        components.append((layer, "mlp", None))

    dla_zero = (torch.stack(comp_vectors) @ effective_unembed).cpu().numpy()
    return dla_zero, comp_vectors, components, effective_unembed


@torch.no_grad()
def dla_for_baseline(comp_vectors, effective_unembed, c_mean_vectors=None):
    c_stack = torch.stack(comp_vectors)
    if c_mean_vectors is not None:
        c_stack = c_stack - torch.stack(c_mean_vectors)
    return (c_stack @ effective_unembed).cpu().numpy()


@torch.no_grad()
def build_mean_baseline_vectors(model, means, gains=None):
    head_means, mlp_means = means
    c_means = []
    for layer in range(model.cfg.n_layers):
        W_O = model.W_O[layer]
        gain = gains.get(layer) if gains else None
        for h in range(model.cfg.n_heads):
            v = head_means[(layer, h)] @ W_O[h]
            c_means.append(v * gain if gain is not None else v)
        c_means.append(mlp_means[layer])
    return c_means


@torch.no_grad()
def compute_de(model, cache, metric_spec, comp_vectors, clean_precap,
               softcap=None, c_mean_vectors=None):
    """DE(c) = metric(LN(r)) - metric(LN(r_ablated)) for every component.
    """
    n_layers = model.cfg.n_layers
    r = cache[f"blocks.{n_layers - 1}.hook_resid_post"][0, -1]

    c_stack = torch.stack(comp_vectors)
    r_ablated = r.unsqueeze(0) - c_stack
    if c_mean_vectors is not None:
        r_ablated = r_ablated + torch.stack(c_mean_vectors)

    sigma_r = cache["ln_final.hook_scale"][0, -1, 0]
    uses_rms = "RMS" in type(model.ln_final).__name__
    if uses_rms:
        sigma_abl = torch.sqrt((r_ablated ** 2).mean(dim=-1) + model.cfg.eps)
    else:
        centered = r_ablated - r_ablated.mean(dim=-1, keepdim=True)
        sigma_abl = torch.sqrt((centered ** 2).mean(dim=-1) + model.cfg.eps)
    rho = (sigma_r / sigma_abl).cpu().numpy()

    normed = model.ln_final(r_ablated.unsqueeze(1))[:, 0, :]
    abl_logits = normed @ metric_spec.columns(model)          # [N, k]

    abl_precap = metric_spec.apply_cap(abl_logits, None)
    de_precap = (clean_precap - abl_precap).cpu().numpy()

    de_postcap = None
    if softcap is not None:
        clean_post = _clean_postcap(model, cache, metric_spec, softcap)
        abl_post = metric_spec.apply_cap(abl_logits, softcap)
        de_postcap = (clean_post - abl_post).cpu().numpy()

    return de_precap, de_postcap, rho


@torch.no_grad()
def _clean_postcap(model, cache, metric_spec, softcap):
    n_layers = model.cfg.n_layers
    r = cache[f"blocks.{n_layers - 1}.hook_resid_post"][0, -1]
    normed = model.ln_final(r.unsqueeze(0).unsqueeze(0))[0, 0]
    logits = normed @ metric_spec.columns(model)
    return float(metric_spec.apply_cap(logits, softcap).item())


@torch.no_grad()
def compute_activation_means(model, examples):
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    device = next(model.parameters()).device

    head_sum: dict = {}
    mlp_sum: dict = {}
    n_seen = 0

    for example in examples:
        context = example["rephrased_context"].split("BLANK")[0].strip()
        tokens = model.to_tokens(context)
        _, cache = model.run_with_cache(tokens, return_type=None)

        for layer in range(n_layers):
            z = cache[f"blocks.{layer}.attn.hook_z"][0, -1].float()
            for h in range(n_heads):
                key = (layer, h)
                head_sum[key] = head_sum.get(key, 0) + z[h]
            m = cache[f"blocks.{layer}.hook_mlp_out"][0, -1].float()
            mlp_sum[layer] = mlp_sum.get(layer, 0) + m
        n_seen += 1

        del cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    dtype = model.W_O.dtype
    head_means = {k: (v / n_seen).to(device=device, dtype=dtype)
                  for k, v in head_sum.items()}
    mlp_means = {k: (v / n_seen).to(device=device, dtype=dtype)
                 for k, v in mlp_sum.items()}
    return head_means, mlp_means


def _build_ablation_hooks(chunk_components, means=None):
    head_means, mlp_means = means if means is not None else (None, None)

    hook_specs: dict = {}
    for row_idx, (layer, comp_type, head_idx) in enumerate(chunk_components):
        hook_specs.setdefault((layer, comp_type), {})[row_idx] = head_idx

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
def compute_ap(model, tokens, metric_spec, components, patch_batch_size,
               clean_precap, softcap=None, means=None, clean_postcap=None):
    if softcap is not None and clean_postcap is None:
        raise ValueError("compute_ap: softcap given but clean_postcap is None")

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

        model.run_with_hooks(
            batch_tokens,
            fwd_hooks=abl_hooks + [final_hook],
            return_type=None,
        )

        resid = captured["resid"][:, -1, :]
        normed = model.ln_final(resid.unsqueeze(1))[:, 0, :]
        abl_logits = normed @ metric_spec.columns(model)       # [B, k]

        abl_pre = metric_spec.apply_cap(abl_logits, None).cpu().numpy()
        ap_precap[chunk_start:chunk_end] = clean_precap - abl_pre

        if softcap is not None:
            abl_post = metric_spec.apply_cap(
                abl_logits, softcap).cpu().numpy()
            ap_postcap[chunk_start:chunk_end] = clean_postcap - abl_post

        del captured, resid, normed, abl_logits, batch_tokens
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return ap_precap, ap_postcap


# ──────────────────────────────────────────────────────────────────────
#  Metrics
# ──────────────────────────────────────────────────────────────────────
def nmae_against(a, b, denom_source):
    """Sum|a-b| / Sum|denom_source| * 100%."""
    denom = float(np.sum(np.abs(denom_source)))
    if denom == 0:
        return float("inf")
    return float(np.sum(np.abs(a - b)) / denom * 100.0)


def nmae_per_unit(records_df, col_a, col_b, denom_col="ap_precap",
                  group_cols=("example_id", "unit_id")):
    """[S7] Normalise INSIDE each evaluation unit, then take the median.

    The pooled NMAE uses one global denominator, so it is dominated by the few
    units with the largest |AP| (in GPT-2 XL, layer-0 MLP knockouts with
    |AP| ~ 10 against a typical |AP| ~ 0.1). This variant is the robust
    companion number and should be reported next to it.
    """
    vals = []
    for _, sub in records_df.groupby(list(group_cols)):
        vals.append(nmae_against(sub[col_a].values, sub[col_b].values,
                                 sub[denom_col].values))
    vals = np.array([v for v in vals if np.isfinite(v)])
    if len(vals) == 0:
        return float("nan"), float("nan"), float("nan")
    return (float(np.median(vals)),
            float(np.percentile(vals, 25)),
            float(np.percentile(vals, 75)))


def l1_scale(a, b):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    mask = np.abs(a) > 0
    if not mask.any():
        return float("nan")
    r = b[mask] / a[mask]
    w = np.abs(a[mask])
    order = np.argsort(r)
    r, w = r[order], w[order]
    cw = np.cumsum(w)
    idx = int(np.searchsorted(cw, cw[-1] / 2.0))
    idx = min(idx, len(r) - 1)
    return float(r[idx])


def scale_diagnostics(a, b):
    """Magnitude ratio and NMAE after the L1-optimal rescaling of a onto b."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    denom = float(np.sum(np.abs(b)))
    ratio = (float(np.sum(np.abs(a))) / denom) if denom else float("nan")
    s = l1_scale(a, b)
    cal = nmae_against(s * a, b, b) if np.isfinite(s) else float("nan")
    return ratio, s, cal


def bootstrap_nmae_ci(a, b, example_ids, n_resamples=1000,
                      ci=0.95, seed=42, denom_source=None):
    if denom_source is None:
        raise ValueError("bootstrap_nmae_ci requires denom_source")
    rng = np.random.RandomState(seed)
    unique_ids = np.unique(example_ids)
    n_ex = len(unique_ids)

    id_to_idx: dict = {}
    for i, eid in enumerate(example_ids):
        id_to_idx.setdefault(eid, []).append(i)

    samples = np.empty(n_resamples)
    for s in range(n_resamples):
        drawn = rng.choice(unique_ids, size=n_ex, replace=True)
        idx = np.concatenate([id_to_idx[eid] for eid in drawn])
        samples[s] = nmae_against(a[idx], b[idx], denom_source[idx])

    alpha = (1 - ci) / 2
    return (float(np.percentile(samples, alpha * 100)),
            float(np.percentile(samples, (1 - alpha) * 100)))


def bootstrap_spearman_delta_ci(rhos_a, rhos_b, n_resamples=1000,
                                ci=0.95, seed=42):
    """Paired percentile bootstrap CI for  median(rhos_b) - median(rhos_a)."""
    rhos_a = np.asarray(rhos_a, dtype=float)
    rhos_b = np.asarray(rhos_b, dtype=float)
    assert len(rhos_a) == len(rhos_b), (
        "per-unit rho arrays must have equal length")

    n = len(rhos_a)
    delta_point = float(np.median(rhos_b) - np.median(rhos_a))

    rng = np.random.RandomState(seed)
    deltas = np.empty(n_resamples)
    for s in range(n_resamples):
        idx = rng.choice(n, size=n, replace=True)
        deltas[s] = np.median(rhos_b[idx]) - np.median(rhos_a[idx])

    alpha_half = (1 - ci) / 2
    ci_lo = float(np.percentile(deltas, alpha_half * 100))
    ci_hi = float(np.percentile(deltas, (1 - alpha_half) * 100))

    frac_le = np.mean(deltas <= 0)
    frac_ge = np.mean(deltas >= 0)
    p_two = float(min(2 * min(frac_le, frac_ge), 1.0))
    return delta_point, ci_lo, ci_hi, p_two


def rank_biased_overlap(rank_a, rank_b, p=0.9, depth=None):
    if depth is None:
        depth = min(len(rank_a), len(rank_b))
    seen_a, seen_b = set(), set()
    overlap, rbo = 0, 0.0
    for d in range(1, depth + 1):
        seen_a.add(rank_a[d - 1])
        seen_b.add(rank_b[d - 1])
        overlap = len(seen_a & seen_b)
        rbo += (p ** (d - 1)) * (overlap / d)
    return float((1 - p) * rbo)


def _rank_indices(values, topk_mode):
    """Descending order of indices under the chosen convention.  [S8]"""
    v = np.abs(values) if topk_mode == "abs" else values
    return np.argsort(-v, kind="stable")


def _mcc(sign_a, sign_b):
    """Matthews correlation on the sign(+/-) agreement.  [S9]"""
    a = sign_a > 0
    b = sign_b > 0
    tp = np.sum(a & b)
    tn = np.sum(~a & ~b)
    fp = np.sum(~a & b)
    fn = np.sum(a & ~b)
    denom = math.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    if denom == 0:
        return 0.0
    return float((tp * tn - fp * fn) / denom)


def per_unit_selection_metrics(records_df, col_a, col_b, ks=(5, 10, 25),
                               topk_mode="abs", rbo_p=0.9,
                               group_cols=("example_id", "unit_id")):
    rhos, taus, rbos = [], [], []
    sign_agree, sign_base, mccs = [], [], []
    jaccard = {k: [] for k in ks}

    n_degenerate = 0
    for _, sub in records_df.groupby(list(group_cols)):
        a = sub[col_a].values
        b = sub[col_b].values
        if len(a) < 3:
            continue

        if np.ptp(a) == 0 or np.ptp(b) == 0:
            n_degenerate += 1
        else:
            rho, _ = spearmanr(a, b)
            tau, _ = kendalltau(a, b, variant="b")
            if np.isfinite(rho) and np.isfinite(tau):
                rhos.append(rho)
                taus.append(tau)
            else:
                n_degenerate += 1

        ra = _rank_indices(a, topk_mode)
        rb = _rank_indices(b, topk_mode)
        rbos.append(rank_biased_overlap(list(ra), list(rb), p=rbo_p))

        agree = float(np.mean(np.sign(a) == np.sign(b)))
        pos_b = float(np.mean(b > 0))
        sign_agree.append(agree)
        sign_base.append(max(pos_b, 1.0 - pos_b))
        mccs.append(_mcc(a, b))

        for k in ks:
            top_a = set(ra[:k].tolist())
            top_b = set(rb[:k].tolist())
            jaccard[k].append(len(top_a & top_b) / len(top_a | top_b))

    if not sign_agree:
        return {"spearman_median": float("nan"), "n_units": 0,
                "n_degenerate_units": n_degenerate,
                "spearman_per_unit": np.array([])}

    _med = lambda v: float(np.median(v)) if len(v) else float("nan")
    out = {
        "n_degenerate_units": n_degenerate,
        "spearman_median": _med(rhos),
        "spearman_iqr": [float(np.percentile(rhos, 25)),
                         float(np.percentile(rhos, 75))]
        if rhos else [float("nan"), float("nan")],
        "spearman_per_unit": np.array(rhos),
        "kendall_tau_median": _med(taus),
        "rbo_median": _med(rbos),
        "sign_agreement": float(np.mean(sign_agree)),
        "sign_agreement_baseline": float(np.mean(sign_base)),
        "sign_agreement_over_baseline": float(
            np.mean(sign_agree) - np.mean(sign_base)),
        "sign_mcc": float(np.mean(mccs)),
        "n_units": len(sign_agree),
        "n_rank_units": len(rhos),
    }
    out.update({f"top{k}_jaccard_median": float(np.median(jaccard[k]))
                for k in ks})
    return out


def compute_summary(records_df, model_name, has_softcap, topk_mode="abs"):
    dla = records_df["dla"].values
    de = records_df["de_precap"].values
    ap = records_df["ap_precap"].values
    eids = records_df["example_id"].values

    sa = nmae_against(dla, de, ap)
    sb = nmae_against(de, ap, ap)
    st = nmae_against(dla, ap, ap)

    ci_a = bootstrap_nmae_ci(dla, de, eids, denom_source=ap)
    ci_b = bootstrap_nmae_ci(de, ap, eids, denom_source=ap)
    ci_t = bootstrap_nmae_ci(dla, ap, eids, denom_source=ap)

    summary: dict = {
        "model": model_name,
        "n_records": len(records_df),
        "n_examples": int(records_df["example_id"].nunique()),
        "n_units": int(records_df.groupby(
            ["example_id", "unit_id"]).ngroups),
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

    # [S7] robust companions to the pooled NMAE
    for label, ca, cb in [("source_a", "dla", "de_precap"),
                          ("source_b", "de_precap", "ap_precap"),
                          ("total", "dla", "ap_precap")]:
        med, q25, q75 = nmae_per_unit(records_df, ca, cb)
        summary[f"{label}_nmae_per_unit_median"] = round(med, 2)
        summary[f"{label}_nmae_per_unit_iqr"] = [round(q25, 2), round(q75, 2)]

    no_l0 = records_df[records_df["layer"] > 0]
    if len(no_l0):
        summary["total_dla_error_nmae_excl_layer0"] = round(
            nmae_against(no_l0["dla"].values, no_l0["ap_precap"].values,
                         no_l0["ap_precap"].values), 2)

    sel = per_unit_selection_metrics(records_df, "dla", "ap_precap",
                                     topk_mode=topk_mode)
    sel.pop("spearman_per_unit", None)
    summary.update(sel)

    if "rho" in records_df.columns and "clean_precap" in records_df.columns:
        rho_arr = records_df["rho"].values
        clean_arr = records_df["clean_precap"].values
        de_pred = clean_arr * (1.0 - rho_arr) + rho_arr * dla
        summary["frozen_norm_identity_max_residual"] = float(
            np.max(np.abs(de - de_pred)))

    if has_softcap:
        de_pre = records_df["de_precap"].values
        de_post = records_df["de_postcap"].values
        ap_pre = records_df["ap_precap"].values
        ap_post = records_df["ap_postcap"].values
        de_dist = np.abs(de_post - de_pre)
        ap_dist = np.abs(ap_post - ap_pre)
        de_denom = np.sum(np.abs(de_pre))
        ap_denom = np.sum(np.abs(ap_pre))
        summary["cap_distortion_de_mean_abs"] = round(
            float(np.mean(de_dist)), 4)
        summary["cap_distortion_de_pct_of_precap"] = (
            round(float(np.sum(de_dist) / de_denom * 100), 2)
            if de_denom > 0 else float("inf"))
        summary["cap_distortion_ap_mean_abs"] = round(
            float(np.mean(ap_dist)), 4)
        summary["cap_distortion_ap_pct_of_precap"] = (
            round(float(np.sum(ap_dist) / ap_denom * 100), 2)
            if ap_denom > 0 else float("inf"))

    return summary


def print_report(summary):
    print(f"\n{'─' * 66}")
    abl = summary.get("ablation_type", "")
    label = f" ({abl} ablation)" if abl else ""
    print(f"Model : {summary['model']}{label}  "
          f"metric={summary.get('metric', '?')}")
    print(f"Records: {summary['n_records']}  |  "
          f"Examples: {summary['n_examples']}  |  "
          f"Units: {summary.get('n_units', '?')}")
    print(f"{'─' * 66}")
    print(f"  Source A  NMAE(DLA,DE) = "
          f"{summary['source_a_frozen_norm_nmae']:.2f}%  "
          f"[{summary['source_a_ci_lo']:.2f}, {summary['source_a_ci_hi']:.2f}]"
          f"   per-unit median "
          f"{summary['source_a_nmae_per_unit_median']:.2f}%")
    print(f"  Source B  NMAE(DE,AP)  = "
          f"{summary['source_b_indirect_effect_nmae']:.2f}%  "
          f"[{summary['source_b_ci_lo']:.2f}, {summary['source_b_ci_hi']:.2f}]"
          f"   per-unit median "
          f"{summary['source_b_nmae_per_unit_median']:.2f}%")
    print(f"  Total     NMAE(DLA,AP) = "
          f"{summary['total_dla_error_nmae']:.2f}%  "
          f"[{summary['total_ci_lo']:.2f}, {summary['total_ci_hi']:.2f}]"
          f"   per-unit median "
          f"{summary['total_nmae_per_unit_median']:.2f}%")
    if "total_dla_error_nmae_excl_layer0" in summary:
        print(f"            excl. layer 0 = "
              f"{summary['total_dla_error_nmae_excl_layer0']:.2f}%")
    if "spearman_median" in summary:
        print(f"  Selection: Spearman={summary['spearman_median']:.3f}  "
              f"tau-b={summary['kendall_tau_median']:.3f}  "
              f"RBO={summary['rbo_median']:.3f}")
        print(f"             sign={summary['sign_agreement'] * 100:.1f}% "
              f"(base rate {summary['sign_agreement_baseline'] * 100:.1f}%, "
              f"MCC={summary['sign_mcc']:.3f})")
        print(f"             Jaccard  5={summary['top5_jaccard_median']:.2f}  "
              f"10={summary['top10_jaccard_median']:.2f}  "
              f"25={summary['top25_jaccard_median']:.2f}")
    if "frozen_norm_identity_max_residual" in summary:
        print(f"  Identity:  max |DE - predicted| = "
              f"{summary['frozen_norm_identity_max_residual']:.4e}")
    if "cap_distortion_de_mean_abs" in summary:
        print(f"  [SOFTCAP] DE {summary['cap_distortion_de_pct_of_precap']:.2f}"
              f"% of |pre-cap|, AP "
              f"{summary['cap_distortion_ap_pct_of_precap']:.2f}%")
    if "multi_token_pct" in summary:
        print(f"  Tokenization: {summary['multi_token_pct']:.1f}% multi-token, "
              f"{summary.get('skipped_pct', 0):.1f}% skipped")
    print(f"{'─' * 66}")


def analyze_model(model_name, examples, patch_batch_size, device,
                  ablation_modes=("zero", "mean"), metric_mode="logit",
                  multi_token_policy="skip", topk_mode="abs",
                  run_gates=True, enable_grad_flags=False,
                  min_abs_clean=0.0):
    print(f"\n{'=' * 70}")
    print(f"  Analyzing: {model_name}  "
          f"(modes: {', '.join(ablation_modes)}, metric: {metric_mode})")
    print(f"{'=' * 70}")

    model = load_model(model_name, device, enable_grad_flags)
    validate_model_compatibility(model)
    model.eval()

    softcap = resolve_softcap(model, model_name)
    validate_hook_z_shape(model, examples[0])

    n_layers = model.cfg.n_layers
    catalog = build_component_catalog(model)
    print(f"Components: {len(catalog)}  "
          f"({n_layers}L x {model.cfg.n_heads}H + {n_layers} MLPs)")

    stats = {"n_candidates": 0, "n_multi_token": 0, "n_skipped": 0}

    if run_gates:
        # [S4] structural gates — cheap, and they block everything downstream
        validate_residual_decomposition(model, examples[0])
        probe = build_metric_specs(model, examples[0], metric_mode,
                                   multi_token_policy, dict(stats))
        if probe:
            validate_completeness(model, examples[0], probe[0])
            validate_chunk_invariance(model, examples[0], probe[0],
                                      catalog[:24], softcap)

    means = None
    if "mean" in ablation_modes:
        means = compute_activation_means(model, examples)
        print(f"[MEAN-ABL] Precomputed activation means over "
              f"{len(examples)} examples")

    records_by_mode: dict = {m: [] for m in ablation_modes}
    clean_values: list = []
    catalog_fracs: list = []
    postnorm_corrected = False
    gates_fired: set = set()

    for ex_idx, example in enumerate(examples):
        context = example["rephrased_context"].split("BLANK")[0].strip()
        ex_id = example["id"]
        print(f"  [{ex_idx + 1}/{len(examples)}] {ex_id[:12]}...  "
              f"'{context[:50]}...'")

        specs = build_metric_specs(model, example, metric_mode,
                                   multi_token_policy, stats)
        if not specs:
            continue

        tokens = model.to_tokens(context)

        for spec in specs:
            with torch.no_grad():
                _, cache = model.run_with_cache(tokens, return_type=None)

                r = cache[f"blocks.{n_layers - 1}.hook_resid_post"][0, -1]
                r_normed = model.ln_final(
                    r.unsqueeze(0).unsqueeze(0))[0, 0]
                clean_logits = r_normed @ spec.columns(model)
                clean_precap = float(
                    spec.apply_cap(clean_logits, None).item())
                clean_postcap = (
                    float(spec.apply_cap(clean_logits, softcap).item())
                    if softcap is not None else None)

            if abs(clean_precap) < min_abs_clean:
                stats["n_low_signal"] = stats.get("n_low_signal", 0) + 1
                del cache
                continue

            with torch.no_grad():

                gains, corrected = compute_attn_gains(model, cache)
                postnorm_corrected = postnorm_corrected or corrected

                dla_zero, comp_vecs, components, u_eff = (
                    compute_dla_and_vectors(model, cache, spec, gains=gains))

                c_mean_vecs = (build_mean_baseline_vectors(model, means, gains)
                               if means is not None else None)

                per_mode: dict = {}
                for mode in ablation_modes:
                    cm = c_mean_vecs if mode == "mean" else None
                    # [S2] baseline-matched DLA per mode
                    dla_mode = dla_zero if cm is None else dla_for_baseline(
                        comp_vecs, u_eff, cm)
                    de_pre, de_post, rho = compute_de(
                        model, cache, spec, comp_vecs, clean_precap,
                        softcap=softcap, c_mean_vectors=cm)
                    per_mode[mode] = (dla_mode, de_pre, de_post, rho)

            clean_values.append(clean_precap)
            if clean_precap != 0:
                catalog_fracs.append(
                    float(np.sum(dla_zero)) / clean_precap)

            del cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            for mode in ablation_modes:
                dla_m, de_pre_m, de_post_m, rho_m = per_mode[mode]
                ap_means = means if mode == "mean" else None
                ap_pre, ap_post = compute_ap(
                    model, tokens, spec, components, patch_batch_size,
                    clean_precap, softcap=softcap, means=ap_means,
                    clean_postcap=clean_postcap)

                de_pred = clean_precap * (1.0 - rho_m) + rho_m * dla_m
                id_res = float(np.max(np.abs(de_pre_m - de_pred)))
                if mode not in gates_fired:
                    gates_fired.add(mode)
                    print(f"[IDENTITY] {mode}: max|DE - (clean(1-rho)+rho*DLA)|"
                          f" = {id_res:.3e}")
                    if id_res > IDENTITY_TOL * max(1.0, abs(clean_precap)):
                        raise ValueError(
                            f"[IDENTITY FAIL] {mode}: residual {id_res:.3e}. "
                            f"DLA and DE are not baseline-matched.")

                    last_mlp = next(
                        i for i, (l, ct, _) in enumerate(components)
                        if l == n_layers - 1 and ct == "mlp")
                    diff = abs(float(de_pre_m[last_mlp])
                               - float(ap_pre[last_mlp]))
                    print(f"[SANITY] {mode} last-MLP |DE - AP| = {diff:.4e}")
                    assert diff < 1e-3, (
                        f"Position-matched ablation broken: |delta|={diff}")

                for i, (layer, ctype, hidx) in enumerate(components):
                    rec = {
                        "model": model_name,
                        "example_id": ex_id,
                        "unit_id": spec.unit_id,
                        "candidate": "|".join(spec.words),
                        "candidate_type": spec.label,
                        "target_token_id": spec.token_ids[0],
                        "metric": metric_mode,
                        "clean_precap": clean_precap,
                        "layer": layer,
                        "component_type": ctype,
                        "head_idx": hidx if hidx is not None else -1,
                        "dla": float(dla_m[i]),
                        "de_precap": float(de_pre_m[i]),
                        "ap_precap": float(ap_pre[i]),
                        "rho": float(rho_m[i]),
                    }
                    if softcap is not None:
                        rec["de_postcap"] = float(de_post_m[i])
                        rec["ap_postcap"] = float(ap_post[i])
                    records_by_mode[mode].append(rec)

    nc = max(1, stats["n_candidates"])
    print(f"[TOKENIZATION] {stats['n_multi_token']}/{nc} multi-token "
          f"({100.0 * stats['n_multi_token'] / nc:.1f}%), "
          f"{stats['n_skipped']} skipped "
          f"({100.0 * stats['n_skipped'] / nc:.1f}%)")

    if clean_values:
        cv = np.array(clean_values)
        print(f"[CLEAN-METRIC] n={len(cv)}  mean={cv.mean():+.3f}  "
              f"median={np.median(cv):+.3f}  "
              f"|median|={np.median(np.abs(cv)):.3f}  "
              f"frac>0={np.mean(cv > 0) * 100:.1f}%  "
              f"frac |x|<0.1={np.mean(np.abs(cv) < 0.1) * 100:.1f}%")
    if catalog_fracs:
        cf = np.array(catalog_fracs)
        print(f"[CATALOG-SHARE] median={np.median(cf) * 100:.1f}%  "
              f"IQR=[{np.percentile(cf, 25) * 100:.1f}%, "
              f"{np.percentile(cf, 75) * 100:.1f}%]  "
              f"— share of the clean metric carried by heads+MLPs; the rest "
              f"is embeddings/b_O and is not selectable by any method")
        stats["catalog_share_median"] = float(np.median(cf))
    stats["clean_metric_values"] = clean_values
    stats["postnorm_corrected"] = postnorm_corrected

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return records_by_mode, softcap is not None, stats


def run_validation_checkpoint(model_name, examples, patch_batch_size, device,
                              metric_mode, multi_token_policy, topk_mode,
                              enable_grad_flags):
    print(f"\n{'─' * 60}")
    print(f"  Validating: {model_name} ({VALIDATION_N} examples)")
    print(f"{'─' * 60}")

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    records_by_mode, has_softcap, _ = analyze_model(
        model_name, examples[:VALIDATION_N], patch_batch_size, device,
        metric_mode=metric_mode, multi_token_policy=multi_token_policy,
        topk_mode=topk_mode, run_gates=True,
        enable_grad_flags=enable_grad_flags)

    if torch.cuda.is_available():
        print(f"[CUDA] Peak allocated: "
              f"{torch.cuda.max_memory_allocated() / 1024 ** 3:.2f} GB  |  "
              f"Peak reserved: "
              f"{torch.cuda.max_memory_reserved() / 1024 ** 3:.2f} GB")

    for mode, records in records_by_mode.items():
        df = pd.DataFrame(records)
        dla, de, ap = (df["dla"].values, df["de_precap"].values,
                       df["ap_precap"].values)
        sa, sb, st = (nmae_against(dla, de, ap), nmae_against(de, ap, ap),
                      nmae_against(dla, ap, ap))
        for label, val in [("Source A", sa), ("Source B", sb), ("Total", st)]:
            if not math.isfinite(val):
                raise ValueError(f"[VALIDATION FAIL] {model_name} ({mode}): "
                                 f"{label} NMAE non-finite ({val})")
            if val > NMAE_PLAUSIBLE_HI:
                raise ValueError(f"[VALIDATION FAIL] {model_name} ({mode}): "
                                 f"{label} NMAE = {val:.2f}% exceeds "
                                 f"{NMAE_PLAUSIBLE_HI}%")
        print(f"[VALIDATION OK] {model_name} ({mode}): "
              f"A={sa:.1f}%  B={sb:.1f}%  Total={st:.1f}%")

    if has_softcap:
        for mode, records in records_by_mode.items():
            df = pd.DataFrame(records)
            de_diff = (df["de_postcap"] - df["de_precap"]).abs().sum()
            ap_diff = (df["ap_postcap"] - df["ap_precap"]).abs().sum()
            if de_diff == 0:
                raise ValueError(f"[VALIDATION FAIL] {model_name} ({mode}): "
                                 f"soft-cap not applied")
            print(f"[SOFTCAP OK] {model_name} ({mode}): "
                  f"sum|de_post-de_pre|={de_diff:.4f}, "
                  f"sum|ap_post-ap_pre|={ap_diff:.4f}")


DEFAULT_MODELS = [
    "gpt2-xl",
    "meta-llama/Llama-3.2-1B",
    "google/gemma-2-2b",
]


def main():
    parser = argparse.ArgumentParser(
        description="DLA error budget: DLA vs Direct Effect vs "
                    "Activation Patching")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--n-examples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patch-batch-size", type=int, default=16)
    parser.add_argument("--no-s3", action="store_true")
    parser.add_argument("--skip-validation", action="store_true")
    parser.add_argument("--validate-only", action="store_true",
                        help="Run the Phase-A structural gates and exit")
    parser.add_argument("--metric", choices=["logit", "logit_diff"],
                        default="logit_diff",
                        help="logit_diff = stereotype - anti-stereotype [M1]")
    parser.add_argument("--multi-token-policy", choices=["skip", "first"],
                        default="skip",
                        help="'first' scores only the first sub-token, which "
                             "is ' a' for candidates like 'a band' [S5]")
    parser.add_argument("--topk-mode", choices=["abs", "signed"],
                        default="abs", help="top-k convention [S8]")
    parser.add_argument("--min-abs-clean", type=float, default=0.0,
                        help="Skip units whose |clean metric| is below this. "
                             "With --metric logit_diff a near-zero value means "
                             "the model shows no measurable preference on that "
                             "item, so there is no bias to localise and the "
                             "NMAE denominator is near zero.")
    parser.add_argument("--ablation-modes", nargs="+",
                        default=["zero", "mean"], choices=["zero", "mean"])
    parser.add_argument("--enable-tl-grad-flags", action="store_true",
                        help="Restore the old Llama-only TL flags [S11]")
    parser.add_argument("--tag", default="",
                        help="Suffix for output filenames")
    args = parser.parse_args()

    s3_utils.set_use_s3(not args.no_s3)

    hf_token = os.environ.get("HF_TOKEN", "")
    if hf_token:
        login(token=hf_token)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}   metric: {args.metric}   "
          f"multi-token: {args.multi_token_policy}")

    dataset = s3_utils.read_json("datasets/gender_test_rephrased_v2.json")
    print(f"Loaded {len(dataset)} examples")

    random.seed(args.seed)
    examples = (random.sample(dataset, args.n_examples)
                if len(dataset) > args.n_examples else list(dataset))
    print(f"Sampled {len(examples)} examples (seed={args.seed})\n")

    output_dir = "outputs/dla_error_analysis"
    suffix = f"_{args.tag}" if args.tag else ""

    if args.validate_only:
        print("\n" + "=" * 70)
        print("  PHASE-A GATES ONLY")
        print("=" * 70)
        for model_name in args.models:
            model = load_model(model_name, device, args.enable_tl_grad_flags)
            model.eval()
            validate_model_compatibility(model)
            validate_residual_decomposition(model, examples[0])
            stats = {"n_candidates": 0, "n_multi_token": 0, "n_skipped": 0}
            specs = build_metric_specs(model, examples[0], args.metric,
                                       args.multi_token_policy, stats)
            if specs:
                validate_completeness(model, examples[0], specs[0])
                validate_chunk_invariance(
                    model, examples[0], specs[0],
                    build_component_catalog(model)[:24],
                    resolve_softcap(model, model_name))
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        print("\n[ALL GATES PASSED]")
        return

    if not args.skip_validation:
        print("\n" + "=" * 70)
        print(f"  PHASE 1: Validation ({VALIDATION_N} examples per model)")
        print("=" * 70)
        for model_name in args.models:
            run_validation_checkpoint(
                model_name, examples, args.patch_batch_size, device,
                args.metric, args.multi_token_policy, args.topk_mode,
                args.enable_tl_grad_flags)
        print("\n[ALL VALIDATION PASSED]\n")

    print("\n" + "=" * 70)
    print(f"  PHASE 2: Full analysis ({len(examples)} examples per model)")
    print("=" * 70)

    all_summaries: list = []

    for model_name in args.models:
        records_by_mode, has_softcap, stats = analyze_model(
            model_name, examples, args.patch_batch_size, device,
            ablation_modes=tuple(args.ablation_modes),
            metric_mode=args.metric,
            multi_token_policy=args.multi_token_policy,
            topk_mode=args.topk_mode,
            run_gates=args.skip_validation,
            enable_grad_flags=args.enable_tl_grad_flags,
            min_abs_clean=args.min_abs_clean)

        nc = max(1, stats["n_candidates"])
        safe = model_name.replace("/", "_")
        for mode, records in records_by_mode.items():
            df = pd.DataFrame(records)
            csv_path = (f"{output_dir}/"
                        f"{safe}_{mode}_ablation_records{suffix}.csv")
            s3_utils.write_csv(df, csv_path)
            print(f"  -> saved {len(df)} records to {csv_path}")

            summary = compute_summary(df, model_name, has_softcap,
                                      topk_mode=args.topk_mode)
            summary["ablation_type"] = mode
            summary["metric"] = args.metric
            summary["multi_token_pct"] = round(
                100.0 * stats["n_multi_token"] / nc, 1)
            summary["skipped_pct"] = round(
                100.0 * stats["n_skipped"] / nc, 1)
            summary["postnorm_corrected"] = bool(
                stats.get("postnorm_corrected", False))
            summary["n_low_signal_skipped"] = int(
                stats.get("n_low_signal", 0))
            summary["min_abs_clean"] = args.min_abs_clean
            if "catalog_share_median" in stats:
                summary["catalog_share_median"] = round(
                    stats["catalog_share_median"], 4)
            all_summaries.append(summary)
            print_report(summary)

    summary_path = f"{output_dir}/summary_all_models{suffix}.json"
    s3_utils.write_json(all_summaries, summary_path)
    print(f"\nUnified summary → {summary_path}")
    print("Done.")


if __name__ == "__main__":
    main()