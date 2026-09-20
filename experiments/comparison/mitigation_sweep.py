import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import s3_utils
from dla_error_analysis import (
    ANTI_KEYS,
    STEREO_KEYS,
    _find_key,
    build_component_catalog,
    build_metric_specs,
    compute_activation_means,
    load_model,
    resolve_softcap,
    validate_model_compatibility,
)

UNRELATED_KEYS = ("unrelated", "meaningless", "irrelevant", "random")

METHOD_COLS = {
    "dla": "dla",
    "de": "de_precap",
    "atp": "atp",
    "eap_ig": "eap_ig",
    "ap": "ap_precap",
}
GROUP = ["example_id", "unit_id"]
MERGE_KEYS = ["model", "example_id", "unit_id", "layer",
              "component_type", "head_idx"]


def population_ranking(df, col, signed=True):
    acc, n, comp = None, 0, None
    seq = None
    for _, sub in df.groupby(GROUP):
        v = sub[col].values
        this_seq = tuple(map(tuple,
                             sub[["layer", "component_type",
                                  "head_idx"]].values))
        if seq is None:
            seq = this_seq
        elif this_seq != seq:
            raise ValueError(
                "population_ranking: evaluation units do not list components "
                "in the same order, so positional rank accumulation would mix "
                "different components. Sort the records by "
                "(layer, component_type, head_idx) within each unit first.")
        key = v if signed else np.abs(v)
        order = np.argsort(-key, kind="stable")
        ranks = np.empty(len(v))
        ranks[order] = np.arange(len(v))
        acc = ranks if acc is None else acc + ranks
        n += 1
        comp = sub[["layer", "component_type", "head_idx"]]
    mean_rank = acc / max(1, n)
    idx = np.argsort(mean_rank, kind="stable")
    tuples = [tuple(r) for r in comp.values]
    return [tuples[i] for i in idx]


def random_ranking(catalog, seed):
    rng = np.random.RandomState(seed)
    items = [(L, ct, -1 if h is None else h) for (L, ct, h) in catalog]
    rng.shuffle(items)
    return items


def build_joint_ablation_hooks(components, means=None):
    head_means, mlp_means = means if means is not None else (None, None)
    by_layer_heads, mlp_layers = {}, set()
    for (layer, ctype, head_idx) in components:
        if ctype == "head":
            by_layer_heads.setdefault(int(layer), []).append(int(head_idx))
        else:
            mlp_layers.add(int(layer))

    hooks = []
    for layer, heads in by_layer_heads.items():
        def _mk(_heads=tuple(heads), _layer=layer, _hm=head_means):
            def _hook(act, hook):                 # [b, pos, n_heads, d_head]
                for h in _heads:
                    if _hm is not None:
                        act[:, :, h, :] = _hm[(_layer, h)]
                    else:
                        act[:, :, h, :] = 0.0
                return act
            return _hook
        hooks.append((f"blocks.{layer}.attn.hook_z", _mk()))

    for layer in mlp_layers:
        def _mk(_layer=layer, _mm=mlp_means):
            def _hook(act, hook):                 # [b, pos, d_model]
                if _mm is not None:
                    act[:, :, :] = _mm[_layer]
                else:
                    act[:, :, :] = 0.0
                return act
            return _hook
        hooks.append((f"blocks.{layer}.hook_mlp_out", _mk()))

    return hooks


@torch.no_grad()
def bias_metric(model, examples, metric_mode, multi_token_policy, hooks=None,
                softcap=None):
    pre, post = [], []
    for ex in examples:
        stats = {"n_candidates": 0, "n_multi_token": 0, "n_skipped": 0}
        specs = build_metric_specs(model, ex, metric_mode,
                                   multi_token_policy, stats)
        if not specs:
            continue
        context = ex["rephrased_context"].split("BLANK")[0].strip()
        tokens = model.to_tokens(context)
        for spec in specs:
            captured = {}

            def _cap(act, hook, _c=captured):
                _c["r"] = act.detach()
                return act

            fwd = list(hooks or []) + [
                (f"blocks.{model.cfg.n_layers - 1}.hook_resid_post", _cap)]
            model.run_with_hooks(tokens, fwd_hooks=fwd, return_type=None)
            r = captured["r"][0, -1]
            normed = model.ln_final(r.unsqueeze(0).unsqueeze(0))[0, 0]
            logits = normed @ spec.columns(model)
            pre.append(float(spec.apply_cap(logits, None).item()))
            if softcap is not None:
                post.append(float(spec.apply_cap(logits, softcap).item()))
    return (float(np.mean(pre)) if pre else float("nan"),
            np.asarray(pre, dtype=float),
            float(np.mean(post)) if post else float("nan"),
            np.asarray(post, dtype=float))


@torch.no_grad()
def capability_metric(model, texts, hooks=None, max_len=256):
    nll, ntok = 0.0, 0
    doc_nll, doc_ntok = [], []
    for t in texts:
        tokens = model.to_tokens(t)[:, :max_len]
        if tokens.shape[1] < 2:
            continue
        logits = model.run_with_hooks(tokens, fwd_hooks=list(hooks or []),
                                      return_type="logits")
        lp = torch.log_softmax(logits[0, :-1].float(), dim=-1)
        tgt = tokens[0, 1:]
        d = -lp[torch.arange(len(tgt)), tgt].sum().item()
        nll += d
        ntok += len(tgt)
        doc_nll.append(d)
        doc_ntok.append(int(len(tgt)))
    return (float(np.exp(nll / ntok)) if ntok else float("nan"),
            np.asarray(doc_nll, dtype=float),
            np.asarray(doc_ntok, dtype=float))



@torch.no_grad()
def _candidate_logprob(model, context, word, hooks, length_norm=True):
    ctx = model.to_tokens(context)
    cont = model.to_tokens(" " + word, prepend_bos=False)
    if cont.shape[1] == 0:
        return None
    toks = torch.cat([ctx, cont], dim=1)
    logits = model.run_with_hooks(toks, fwd_hooks=list(hooks or []),
                                  return_type="logits")
    lp = torch.log_softmax(logits[0].float(), dim=-1)
    n_ctx = ctx.shape[1]
    # logits[t] predicts token t+1, so the continuation starts at n_ctx-1
    idx = torch.arange(n_ctx - 1, toks.shape[1] - 1, device=toks.device)
    total = float(lp[idx, toks[0, n_ctx:]].sum().item())
    return total / cont.shape[1] if length_norm else total


@torch.no_grad()
def stereoset_metrics(model, examples, hooks=None, length_norm=True):
    n = n_stereo = n_meaningful = n_lms_usable = 0
    ss_items, lms_items = [], []
    for ex in examples:
        t = ex["targets"]
        s_key, a_key = _find_key(t, STEREO_KEYS), _find_key(t, ANTI_KEYS)
        if s_key is None or a_key is None:
            continue
        ctx = ex["rephrased_context"].split("BLANK")[0].strip()
        lp_s = _candidate_logprob(model, ctx, t[s_key], hooks, length_norm)
        lp_a = _candidate_logprob(model, ctx, t[a_key], hooks, length_norm)
        if lp_s is None or lp_a is None:
            continue
        n += 1
        n_stereo += int(lp_s > lp_a)
        ss_items.append(int(lp_s > lp_a))

        u_key = _find_key(t, UNRELATED_KEYS)
        if u_key is not None:
            lp_u = _candidate_logprob(model, ctx, t[u_key], hooks, length_norm)
            if lp_u is not None:
                n_lms_usable += 1
                n_meaningful += int(max(lp_s, lp_a) > lp_u)
                lms_items.append(int(max(lp_s, lp_a) > lp_u))

    if n == 0:
        return {}
    ss = 100.0 * n_stereo / n
    out = {"ss": ss, "n_ss": n, "ss_items": np.asarray(ss_items, dtype=float)}
    if n_lms_usable:
        lms = 100.0 * n_meaningful / n_lms_usable
        out["lms"] = lms
        out["icat"] = lms * min(ss, 100.0 - ss) / 50.0
        out["n_lms"] = n_lms_usable
        out["lms_items"] = np.asarray(lms_items, dtype=float)
    return out


def _pct(v, ci=0.95):
    a = (1 - ci) / 2
    return (float(np.percentile(v, a * 100)),
            float(np.percentile(v, (1 - a) * 100)))


def boot_ratio_ci(ablated, clean, n_boot=1000, seed=0, ci=0.95):
    if len(ablated) == 0 or len(clean) == 0 or len(ablated) != len(clean):
        return float("nan"), float("nan")
    rng = np.random.RandomState(seed)
    n = len(ablated)
    out = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.randint(0, n, n)
        cm = clean[idx].mean()
        out[b] = 1.0 - ablated[idx].mean() / cm if cm else np.nan
    out = out[np.isfinite(out)]
    return _pct(out, ci) if len(out) else (float("nan"), float("nan"))


def boot_rate_ci(items, n_boot=1000, seed=0, ci=0.95, scale=100.0):
    items = np.asarray(items, dtype=float)
    if len(items) == 0:
        return float("nan"), float("nan")
    rng = np.random.RandomState(seed)
    n = len(items)
    out = np.array([items[rng.randint(0, n, n)].mean() * scale
                    for _ in range(n_boot)])
    return _pct(out, ci)


def boot_ppl_ratio_ci(nll, ntok, base_nll, base_ntok, n_boot=1000, seed=0,
                      ci=0.95):
    if len(nll) == 0 or len(nll) != len(base_nll):
        return float("nan"), float("nan")
    rng = np.random.RandomState(seed)
    n = len(nll)
    out = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.randint(0, n, n)
        t, bt = ntok[idx].sum(), base_ntok[idx].sum()
        if t <= 0 or bt <= 0:
            out[b] = np.nan
            continue
        out[b] = np.exp(nll[idx].sum() / t - base_nll[idx].sum() / bt) - 1.0
    out = out[np.isfinite(out)]
    return _pct(out, ci) if len(out) else (float("nan"), float("nan"))


def load_eval_set(dev_dataset, test_dataset, n_eval, eval_seed,
                  allow_overlap=False):
    dev_ids = {e["id"] for e in dev_dataset}
    test_ids = {e["id"] for e in test_dataset}
    overlap = dev_ids & test_ids
    if overlap:
        msg = (f"[LEAKAGE] {len(overlap)} of {len(test_ids)} test ids also "
               f"appear in the development file "
               f"({len(overlap) / len(test_ids) * 100:.1f}%). "
               f"Components were selected on these items, so any result "
               f"measured on them is contaminated. Examples: "
               f"{sorted(overlap)[:3]}")
        if not allow_overlap:
            raise SystemExit(
                msg + "\n  Pass --allow-overlap only if you intend to report "
                      "contaminated numbers, and say so in the paper.")
        print(f"  {msg}\n  [--allow-overlap] continuing anyway")

    evalset = list(test_dataset)
    rng = random.Random(eval_seed)
    if n_eval and len(evalset) > n_eval:
        evalset = rng.sample(evalset, n_eval)
    return evalset, len(overlap)


def load_generic_texts(path, n):
    """Generic corpus for perplexity. Any held-out natural text works;
    WikiText-103 validation is the conventional choice."""
    if path and Path(path).exists():
        lines = [l.strip() for l in Path(path).read_text().splitlines()
                 if len(l.strip()) > 200]
        return lines[:n]
    raise SystemExit(
        "Pass --generic-text pointing at a plain-text file (e.g. WikiText-103 "
        "validation). Perplexity needs text the model did not see in selection.")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+",
                   default=["meta-llama/Llama-3.2-1B"])
    p.add_argument("--records-dir", default="outputs/dla_error_analysis")
    p.add_argument("--out", default=None,
                   help="default: outputs/mitigation/sweep[_TAG].csv, so two "
                        "tagged runs cannot silently overwrite each other")
    p.add_argument("--tag", default="")
    p.add_argument("--ks", type=int, nargs="+",
                   default=[1, 2, 3, 4, 6, 8, 12, 16, 24, 32])
    p.add_argument("--methods", nargs="+",
                   choices=sorted(METHOD_COLS) + ["random"],
                   default=["dla", "atp", "eap_ig", "ap", "random"])
    p.add_argument("--component-types", choices=["both", "heads", "mlps"],
                   default="both",
                   help="restrict the selectable pool. MLPs dominate both the "
                        "bias reduction and the capability cost, so a "
                        "heads-only arm separates 'which method' from 'did an "
                        "MLP happen to be picked'")
    p.add_argument("--random-seeds", type=int, default=5,
                   help="number of permutations for the Random control. It is "
                        "the reference line for AOR, so a single draw gives it "
                        "no error bar and makes small-k points pure noise")
    p.add_argument("--n-boot", type=int, default=1000,
                   help="bootstrap resamples for the per-item CIs")
    # the six decisions
    p.add_argument("--ablation-mode", choices=["mean", "zero"], default="mean")
    p.add_argument("--topk-sign", choices=["signed", "absolute"],
                   default="signed",
                   help="signed = components pushing TOWARD the stereotype "
                        "(what a debiasing edit should remove); absolute = "
                        "components that matter at all, matching the "
                        "intrinsic half's convention")
    p.add_argument("--metric", choices=["logit_diff", "logit"],
                   default="logit_diff")
    p.add_argument("--multi-token-policy", choices=["skip", "first"],
                   default="skip")
    p.add_argument(
        "--dev-dataset", default="datasets/gender_test_rephrased_v2.json",
        help="Development set — must match the --dataset the attribution run "
             "used. Only its ids are read here, to check for leakage.")
    p.add_argument(
        "--test-dataset", default="datasets/gender_dev_rephrased.json",
        help="Held-out TEST set. Every reported number comes from here. "
             "(The filenames are the wrong way round in the StereoSet dump: "
             "the larger *_test_* file is the development split.)")
    p.add_argument("--allow-overlap", action="store_true",
                   help="proceed even if dev and test share ids — "
                        "contaminated, say so in the paper")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-eval", type=int, default=0,
                   help="0 = the whole test file")
    p.add_argument("--n-mean-examples", type=int, default=100,
                   help="dev examples used for the mean-ablation baselines; "
                        "0 = all. Means converge fast, so a cap here is "
                        "cheap and harmless.")
    p.add_argument("--eval-seed", type=int, default=7)
    p.add_argument("--generic-text", default=None)
    p.add_argument("--n-generic", type=int, default=200)
    p.add_argument("--ss-sum", action="store_true",
                   help="score SS/LMS candidates by SUMMED log-prob instead "
                        "of the length-normalised mean. Candidates differ in "
                        "length ('nervous' vs 'a band'), so the mean is the "
                        "safer default.")
    p.add_argument("--no-stereoset", action="store_true",
                   help="skip SS/LMS/ICAT (roughly triples eval cost)")
    p.add_argument("--no-s3", action="store_true")
    args = p.parse_args()

    s3_utils.set_use_s3(not args.no_s3)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    suffix = f"_{args.tag}" if args.tag else ""
    out_path = Path(args.out) if args.out else Path(
        f"outputs/mitigation/sweep{suffix}.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    items_path = out_path.with_name(out_path.stem + "_items.npz")

    dev = s3_utils.read_json(args.dev_dataset)
    test = s3_utils.read_json(args.test_dataset)
    heldout, n_overlap = load_eval_set(dev, test, args.n_eval, args.eval_seed,
                                       allow_overlap=args.allow_overlap)
    print(f"dev (selection) {len(dev)} from {args.dev_dataset}")
    print(f"test (eval)     {len(heldout)} of {len(test)} from "
          f"{args.test_dataset}   overlap={n_overlap}")
    generic = load_generic_texts(args.generic_text, args.n_generic)

    rows = []
    item_store: dict = {}
    for model_name in args.models:
        safe = model_name.replace("/", "_")
        rec = Path(args.records_dir) / (
            f"{safe}_{args.ablation_mode}_ablation_records{suffix}.csv")
        df = pd.read_csv(rec)
        eap = Path(args.records_dir) / (
            f"{safe}_eap_ig_{args.ablation_mode}_ablation_records{suffix}.csv")
        if eap.exists():
            e = pd.read_csv(eap)
            keep = [c for c in ("eap_ig", "atp") if c in e.columns]
            df = df.merge(e[MERGE_KEYS + keep], on=MERGE_KEYS, how="inner")
        print(f"{model_name}: {len(df):,} rows, "
              f"{df.groupby(GROUP).ngroups} units")

        if "metric" in df.columns:
            rec_metrics = set(df["metric"].dropna().unique())
            if rec_metrics != {args.metric}:
                raise SystemExit(
                    f"[PROVENANCE] {rec} was produced with metric="
                    f"{sorted(rec_metrics)} but this sweep evaluates "
                    f"--metric {args.metric}. Components would be selected for "
                    f"one objective and scored on another. Re-run the "
                    f"attribution with --metric {args.metric}, or run this "
                    f"sweep with the records' metric.")
        else:
            print(f"  [warn] {rec.name} has no 'metric' column — cannot verify "
                  f"that selection and evaluation share an objective")

        model = load_model(model_name, device)
        model.eval()
        validate_model_compatibility(model)
        softcap = resolve_softcap(model, model_name)
        catalog = build_component_catalog(model)

        if args.component_types != "both":
            want = "head" if args.component_types == "heads" else "mlp"
            catalog = [c for c in catalog if c[1] == want]
            df = df[df["component_type"] == want].copy()
            print(f"  [{args.component_types}-only] pool restricted to "
                  f"{len(catalog)} components")

        means = None
        if args.ablation_mode == "mean":
            random.seed(args.seed)
            sel = (random.sample(dev, args.n_mean_examples)
                   if args.n_mean_examples and
                   len(dev) > args.n_mean_examples else list(dev))
            print(f"  mean-ablation baselines from {len(sel)} dev examples "
                  f"+ {len(generic)} generic texts, all non-BOS positions")
            means = compute_activation_means(
                model, sel, positions="all_nonbos", extra_texts=generic)

        # baselines with nothing ablated
        base_bias, base_bias_items, base_bias_post, base_bias_post_items = \
            bias_metric(model, heldout, args.metric, args.multi_token_policy,
                        softcap=softcap)
        base_ppl, base_nll, base_ntok = capability_metric(model, generic)
        base_ss = ({} if args.no_stereoset else
                   stereoset_metrics(model, heldout,
                                     length_norm=not args.ss_sum))
        print(f"  clean: bias={base_bias:+.4f} (n={len(base_bias_items)})  "
              f"ppl={base_ppl:.3f}  "
              f"SS={base_ss.get('ss', float('nan')):.1f} "
              f"LMS={base_ss.get('lms', float('nan')):.1f} "
              f"ICAT={base_ss.get('icat', float('nan')):.1f}")
        if not args.no_stereoset and "lms" not in base_ss:
            print("  [warn] no 'unrelated' candidate found in targets — LMS "
                  "and ICAT unavailable. Extend UNRELATED_KEYS if your data "
                  "has one under another name.")
        item_store[f"{model_name}|clean|0|bias"] = base_bias_items
        item_store[f"{model_name}|clean|0|nll"] = base_nll
        item_store[f"{model_name}|clean|0|ntok"] = base_ntok
        if "ss_items" in base_ss:
            item_store[f"{model_name}|clean|0|ss"] = base_ss["ss_items"]

        rankings = {}
        for m in args.methods:
            if m == "random":
                for s in range(args.random_seeds):
                    rankings[f"random_s{s}" if args.random_seeds > 1
                             else "random"] = random_ranking(
                                 catalog, args.seed + s)
            elif METHOD_COLS[m] in df.columns:
                rankings[m] = population_ranking(df, METHOD_COLS[m],
                                                 signed=(args.topk_sign == "signed"))
            else:
                print(f"  [skip] {m}: column missing")

        for m, ranked in rankings.items():
            for k in args.ks:
                if k > len(ranked):
                    continue
                comps = ranked[:k]
                hooks = build_joint_ablation_hooks(
                    comps, means=means if args.ablation_mode == "mean" else None)
                bias, bias_items, bias_post, bias_post_items = bias_metric(
                    model, heldout, args.metric, args.multi_token_policy,
                    hooks=hooks, softcap=softcap)
                ppl, nll, ntok = capability_metric(model, generic, hooks=hooks)
                ss = ({} if args.no_stereoset else
                      stereoset_metrics(model, heldout, hooks=hooks,
                                        length_norm=not args.ss_sum))

                red_lo, red_hi = boot_ratio_ci(
                    bias_items, base_bias_items, args.n_boot, args.seed)
                cost_lo, cost_hi = boot_ppl_ratio_ci(
                    nll, ntok, base_nll, base_ntok, args.n_boot, args.seed)
                ss_lo, ss_hi = (boot_rate_ci(ss["ss_items"], args.n_boot,
                                             args.seed)
                                if "ss_items" in ss
                                else (float("nan"), float("nan")))

                key = f"{model_name}|{m}|{k}"
                item_store[f"{key}|bias"] = bias_items
                item_store[f"{key}|nll"] = nll
                item_store[f"{key}|ntok"] = ntok
                if "ss_items" in ss:
                    item_store[f"{key}|ss"] = ss["ss_items"]

                rows.append({
                    "model": model_name, "method": m, "k": k,
                    "ablation_mode": args.ablation_mode,
                    "topk_sign": args.topk_sign,
                    "component_types": args.component_types,
                    "bias": bias, "bias_clean": base_bias,
                    "bias_postcap": bias_post,
                    "bias_postcap_clean": base_bias_post,
                    "bias_reduction": 1.0 - bias / base_bias
                    if base_bias else float("nan"),
                    "bias_reduction_ci_lo": red_lo,
                    "bias_reduction_ci_hi": red_hi,
                    "n_bias_items": int(len(bias_items)),
                    "ppl": ppl, "ppl_clean": base_ppl,
                    "capability_cost": ppl / base_ppl - 1.0
                    if base_ppl else float("nan"),
                    "capability_cost_ci_lo": cost_lo,
                    "capability_cost_ci_hi": cost_hi,
                    "n_heads_ablated": sum(1 for c in comps if c[1] == "head"),
                    "n_mlps_ablated": sum(1 for c in comps if c[1] == "mlp"),
                    "components": json.dumps([list(map(str, c))
                                              for c in comps]),
                    "ss": ss.get("ss"), "lms": ss.get("lms"),
                    "icat": ss.get("icat"),
                    "ss_ci_lo": ss_lo, "ss_ci_hi": ss_hi,
                    "n_ss": ss.get("n_ss"), "n_lms": ss.get("n_lms"),
                    "ss_clean": base_ss.get("ss"),
                    "lms_clean": base_ss.get("lms"),
                    "icat_clean": base_ss.get("icat"),
                    "ss_bias_reduction": (
                        1.0 - abs(ss["ss"] - 50.0)
                        / max(abs(base_ss["ss"] - 50.0), 1e-9)
                        if "ss" in ss and "ss" in base_ss else None),
                })
                r = rows[-1]
                print(f"  {m:10s} k={k:4d}  bias={bias:+.4f} "
                      f"({r['bias_reduction']*100:+6.1f}% "
                      f"[{red_lo*100:+.1f},{red_hi*100:+.1f}])  "
                      f"ppl={ppl:9.3f} ({r['capability_cost']*100:+8.2f}%)  "
                      f"SS={ss.get('ss', float('nan')):5.1f} "
                      f"ICAT={ss.get('icat', float('nan')):5.1f}")
                pd.DataFrame(rows).to_csv(out_path, index=False)
                np.savez_compressed(items_path, **item_store)

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    pd.DataFrame(rows).to_csv(out_path, index=False)
    np.savez_compressed(items_path, **item_store)
    print(f"\n-> {out_path}")
    print(f"-> {items_path}  (per-item values for curve-level bootstrap)")


if __name__ == "__main__":
    main()