"""Publication figures
"""

import argparse
import json
import os
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr

# ── house style ───────────────────────────────────────────────────────
mpl.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "figure.facecolor": "#fcfcfb",
    "axes.facecolor": "#fcfcfb",
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 9,
    "axes.titlesize": 9.5,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "legend.frameon": False,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": "#8a8a85",
    "axes.linewidth": 0.6,
    "grid.color": "#e3e3df",
    "grid.linewidth": 0.5,
    "xtick.color": "#8a8a85",
    "ytick.color": "#8a8a85",
})

INK = "#1c1c1a"
INK_MUTED = "#6b6b66"

# fixed categorical order — colour follows the METHOD, never its rank
METHOD_COLOR = {
    "dla": "#2F6FB5",
    "atp": "#D2691E",
    "eap_ig": "#00897B",
    "de_precap": "#8E44AD",
    "random": "#9a9a95",
}
METHOD_LABEL = {
    "dla": "DLA",
    "atp": "AtP",
    "eap_ig": "EAP-IG",
    "de_precap": "DE",
    "random": "Random",
}
METHOD_ORDER = ["dla", "de_precap", "atp", "eap_ig", "random"]

MODELS = ["gpt2-xl", "meta-llama/Llama-3.2-1B", "google/gemma-2-2b"]
SHORT = {"gpt2-xl": "GPT-2 XL",
         "meta-llama/Llama-3.2-1B": "Llama-3.2-1B",
         "google/gemma-2-2b": "Gemma-2-2B"}
MODES = ["zero", "mean"]
GROUP = ["example_id", "unit_id"]
MERGE_KEYS = ["model", "example_id", "unit_id", "layer",
              "component_type", "head_idx"]


# ── metrics (kept in sync with dla_error_analysis.py) ─────────────────
def nmae_against(a, b, denom_source):
    denom = float(np.sum(np.abs(denom_source)))
    return (float(np.sum(np.abs(a - b)) / denom * 100.0)
            if denom else float("nan"))


def l1_scale(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = np.abs(a) > 0
    if not m.any():
        return float("nan")
    r, w = b[m] / a[m], np.abs(a[m])
    o = np.argsort(r)
    r, w = r[o], w[o]
    cw = np.cumsum(w)
    return float(r[min(int(np.searchsorted(cw, cw[-1] / 2.0)), len(r) - 1)])


def rank_biased_overlap(ra, rb, p=0.9, depth=None):
    depth = depth or min(len(ra), len(rb))
    sa, sb, acc = set(), set(), 0.0
    for d in range(1, depth + 1):
        sa.add(ra[d - 1])
        sb.add(rb[d - 1])
        acc += (p ** (d - 1)) * (len(sa & sb) / d)
    return float((1 - p) * acc)


def rank_idx(v, mode="abs"):
    return np.argsort(-(np.abs(v) if mode == "abs" else np.asarray(v)),
                      kind="stable")


def mcc(a, b):
    A, B = a > 0, b > 0
    tp, tn = np.sum(A & B), np.sum(~A & ~B)
    fp, fn = np.sum(~A & B), np.sum(A & ~B)
    d = np.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    return float((tp * tn - fp * fn) / d) if d else 0.0


def unit_metrics(df, col, ref="ap_precap", topk_mode="abs", ks=(5, 10, 25)):
    """Per-unit agreement of `col` with `ref`, aggregated across units."""
    rho, tau, rbo, sgn, base, mccs = [], [], [], [], [], []
    jac = {k: [] for k in ks}
    for _, sub in df.groupby(GROUP):
        a, b = sub[col].values, sub[ref].values
        if len(a) < 3:
            continue
        if np.ptp(a) and np.ptp(b):
            r, _ = spearmanr(a, b)
            t, _ = kendalltau(a, b, variant="b")
            if np.isfinite(r):
                rho.append(r)
                tau.append(t)
        ra, rb = rank_idx(a, topk_mode), rank_idx(b, topk_mode)
        rbo.append(rank_biased_overlap(list(ra), list(rb)))
        sgn.append(float(np.mean(np.sign(a) == np.sign(b))))
        pb = float(np.mean(b > 0))
        base.append(max(pb, 1 - pb))
        mccs.append(mcc(a, b))
        for k in ks:
            sa, sb = set(ra[:k].tolist()), set(rb[:k].tolist())
            jac[k].append(len(sa & sb) / len(sa | sb))
    out = {
        "rho": float(np.median(rho)) if rho else np.nan,
        "rho_per_unit": np.array(rho),
        "tau": float(np.median(tau)) if tau else np.nan,
        "rbo": float(np.median(rbo)),
        "rbo_per_unit": np.array(rbo),
        "sign": float(np.mean(sgn)),
        "sign_base": float(np.mean(base)),
        "mcc": float(np.mean(mccs)),
        "n_units": len(rbo),
    }
    for k in ks:
        out[f"j{k}"] = float(np.median(jac[k]))
        out[f"j{k}_per_unit"] = np.array(jac[k])
    return out


def boot_ci(vals, n=2000, seed=0, ci=0.95):
    v = np.asarray(vals, float)
    v = v[np.isfinite(v)]
    if len(v) < 2:
        return (np.nan, np.nan)
    rng = np.random.RandomState(seed)
    med = np.array([np.median(v[rng.randint(0, len(v), len(v))])
                    for _ in range(n)])
    a = (1 - ci) / 2
    return float(np.percentile(med, a * 100)), \
        float(np.percentile(med, (1 - a) * 100))


def load_all(records_dir, tag=""):
    rec_dir = Path(records_dir)
    suffix = f"_{tag}" if tag else ""
    data, missing = {}, []
    for model in MODELS:
        safe = model.replace("/", "_")
        for mode in MODES:
            ap_p = rec_dir / f"{safe}_{mode}_ablation_records{suffix}.csv"
            # print(ap_p)
            if not ap_p.exists():
                missing.append(str(ap_p))
                continue
            df = pd.read_csv(ap_p)
            eap_p = (rec_dir /
                     f"{safe}_eap_ig_{mode}_ablation_records{suffix}.csv")
            if eap_p.exists():
                e = pd.read_csv(eap_p)
                keep = [c for c in ["eap_ig", "atp"] if c in e.columns]
                before = len(df)
                df = df.merge(e[MERGE_KEYS + keep], on=MERGE_KEYS,
                              how="inner")
                if len(df) != before:
                    print(f"  [warn] {SHORT[model]} {mode}: merge kept "
                          f"{len(df)}/{before} rows")
            else:
                missing.append(str(eap_p))
            df["rel_depth"] = df["layer"] / df["layer"].max()
            # Random ranker: the floor every method must clear. Without it a
            # reported rho of 0.2 has no reference point.
            import zlib
            rs = np.random.RandomState(
                zlib.crc32(f"{model}|{mode}".encode()) % (2 ** 31))
            df["random"] = rs.normal(size=len(df))
            data[(model, mode)] = df
            print(f"  {SHORT[model]:14s} {mode:5s}  {len(df):>8,} rows  "
                  f"{df.groupby(GROUP).ngroups:>4} units  "
                  f"methods: {[m for m in METHOD_ORDER if m in df.columns]}")
    if missing:
        print("\n  [missing]")
        for m in missing:
            print(f"    {m}")
    return data


def methods_in(df, include_random=True):
    return [m for m in METHOD_ORDER
            if m in df.columns and (include_random or m != "random")]


# ── Figure 1 — error budget ───────────────────────────────────────────
def fig_error_budget(data, out_dir, payload):
    keys = [k for k in data if k[1] in MODES]
    keys.sort(key=lambda k: (MODELS.index(k[0]), MODES.index(k[1])))
    if not keys:
        return
    fig, ax = plt.subplots(figsize=(6.6, 2.9))
    labels, w = [], 0.26
    x = np.arange(len(keys))
    series = [("Source A (frozen-norm)", "dla", "de_precap", "#2F6FB5"),
              ("Source B (indirect effect)", "de_precap", "ap_precap",
               "#D2691E"),
              ("Total (DLA vs AP)", "dla", "ap_precap", "#8E44AD")]
    rows = {}
    for i, (lab, ca, cb, colr) in enumerate(series):
        vals, los, his = [], [], []
        for k in keys:
            df = data[k]
            per = [nmae_against(s[ca].values, s[cb].values,
                                s["ap_precap"].values)
                   for _, s in df.groupby(GROUP)]
            v = nmae_against(df[ca].values, df[cb].values,
                             df["ap_precap"].values)
            lo, hi = boot_ci(per)
            vals.append(v)
            los.append(max(0, v - lo))
            his.append(max(0, hi - v))
        ax.bar(x + (i - 1) * w, vals, w * 0.92,
               yerr=[los, his], capsize=2, color=colr, label=lab,
               linewidth=0)
        rows[lab] = vals
    for k in keys:
        labels.append(f"{SHORT[k[0]]}\n{k[1]}")
    ax.axhline(100, color=INK_MUTED, ls=(0, (4, 3)), lw=0.8)
    # label the reference line outside the plotting area so it can never
    # collide with a bar
    ax.text(1.008, 100, "100%\nconstant-zero\npredictor",
            transform=ax.get_yaxis_transform(), ha="left", va="center",
            fontsize=7, color=INK_MUTED, linespacing=1.3)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, color=INK)
    ax.set_ylabel("NMAE vs AP (%)", color=INK)
    ax.grid(axis="y", alpha=0.8)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", bbox_to_anchor=(0, 1.22), ncol=3)
    fig.savefig(Path(out_dir) / "fig_error_budget.pdf")
    plt.close(fig)
    payload["error_budget"] = {"configs": labels, **rows}


# ── Figure 2 — method fingerprint (the "different components" claim) ──
def fig_fingerprint(data, out_dir, payload, k=10, topk_mode="abs"):
    keys = [(m, "zero") for m in MODELS if (m, "zero") in data]
    if not keys:
        return
    nb = 5
    edges = np.linspace(0, 1.0001, nb + 1)
    centres = (edges[:-1] + edges[1:]) / 2
    fig, axes = plt.subplots(1, len(keys), figsize=(6.6, 2.4), sharey=True)
    axes = np.atleast_1d(axes)
    store = {}
    for ax, key in zip(axes, keys):
        df = data[key]
        # A depth bin holds a different NUMBER of components in each model
        # (equal-width bins over discrete layers), so a raw share of top-k is
        # confounded by bin size.  Divide by each bin's share of the whole
        # catalog: 1.0 means "no depth preference", >1 means over-selection.
        one = df.groupby(GROUP).get_group(list(df.groupby(GROUP).groups)[0])
        avail, _ = np.histogram(one["rel_depth"].values, bins=edges)
        avail_share = avail / max(1, avail.sum())
        meths = methods_in(df) + ["ap_precap"]
        for m in meths:
            shares = np.zeros(nb)
            n_units = 0
            for _, sub in df.groupby(GROUP):
                idx = rank_idx(sub[m].values, topk_mode)[:k]
                h, _ = np.histogram(sub["rel_depth"].values[idx], bins=edges)
                shares += h / max(1, h.sum())
                n_units += 1
            shares = shares / max(1, n_units)
            with np.errstate(divide="ignore", invalid="ignore"):
                enrich = np.where(avail_share > 0, shares / avail_share,
                                  np.nan)
            colr = (INK if m == "ap_precap"
                    else METHOD_COLOR.get(m, INK_MUTED))
            lab = "AP (ground truth)" if m == "ap_precap" else METHOD_LABEL[m]
            ax.plot(centres, enrich, marker="o", markersize=3.5, lw=1.6,
                    color=colr, label=lab,
                    ls="-" if m != "ap_precap" else (0, (3, 2)))
            store[f"{SHORT[key[0]]}|{lab}"] = list(np.round(enrich, 3))
        ax.axhline(1.0, color=INK_MUTED, lw=0.6, ls=(0, (2, 2)))
        ax.set_title(SHORT[key[0]], color=INK)
        ax.set_xlabel("relative depth", color=INK)
        ax.grid(axis="y", alpha=0.8)
        ax.set_axisbelow(True)
    axes[0].set_ylabel(f"top-{k} enrichment", color=INK)
    axes[-1].legend(loc="upper left", bbox_to_anchor=(1.02, 1.0))
    fig.savefig(Path(out_dir) / "fig_fingerprint.pdf")
    plt.close(fig)
    payload["fingerprint"] = store


# ── Figure 3 — rank agreement with CIs ────────────────────────────────
def fig_rank_agreement(data, out_dir, payload, topk_mode="abs"):
    keys = sorted([k for k in data],
                  key=lambda k: (MODELS.index(k[0]), MODES.index(k[1])))
    if not keys:
        return
    all_m = sorted({m for k in keys for m in methods_in(data[k])},
                   key=METHOD_ORDER.index)
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.7), sharex=True)
    x = np.arange(len(keys))
    w = 0.8 / max(1, len(all_m))
    store = {}
    for ax, stat, name in [(axes[0], "rho", r"Spearman $\rho$ vs AP"),
                           (axes[1], "rbo", "RBO vs AP")]:
        for j, m in enumerate(all_m):
            vals, los, his = [], [], []
            for k in keys:
                df = data[k]
                if m not in df.columns:
                    vals.append(np.nan)
                    los.append(0)
                    his.append(0)
                    continue
                um = unit_metrics(df, m, topk_mode=topk_mode)
                v = um[stat]
                lo, hi = boot_ci(um[f"{stat}_per_unit"])
                vals.append(v)
                los.append(max(0, v - lo))
                his.append(max(0, hi - v))
                store[f"{SHORT[k[0]]}|{k[1]}|{METHOD_LABEL[m]}|{stat}"] = \
                    [round(v, 4), round(lo, 4), round(hi, 4)]
            ax.bar(x + (j - (len(all_m) - 1) / 2) * w, vals, w * 0.9,
                   yerr=[los, his], capsize=2, linewidth=0,
                   color=METHOD_COLOR[m], label=METHOD_LABEL[m])
        ax.axhline(0, color=INK_MUTED, lw=0.6)
        ax.set_title(name, color=INK)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{SHORT[k[0]]}\n{k[1]}" for k in keys],
                           color=INK)
        ax.grid(axis="y", alpha=0.8)
        ax.set_axisbelow(True)
    axes[0].legend(loc="upper left", bbox_to_anchor=(0, 1.26),
                   ncol=len(all_m))
    fig.savefig(Path(out_dir) / "fig_rank_agreement.pdf")
    plt.close(fig)
    payload["rank_agreement"] = store


# ── Figure 4 — scale vs shape ─────────────────────────────────────────
def fig_scale_vs_shape(data, out_dir, payload):
    keys = sorted([k for k in data],
                  key=lambda k: (MODELS.index(k[0]), MODES.index(k[1])))
    if not keys:
        return
    fig, ax = plt.subplots(figsize=(6.6, 2.7))
    all_m = sorted({m for k in keys for m in methods_in(data[k])},
                   key=METHOD_ORDER.index)
    x = np.arange(len(keys))
    w = 0.8 / max(1, len(all_m))
    store = {}
    for j, m in enumerate(all_m):
        raw, cal = [], []
        for k in keys:
            df = data[k]
            if m not in df.columns:
                raw.append(np.nan)
                cal.append(np.nan)
                continue
            a, b = df[m].values, df["ap_precap"].values
            r = nmae_against(a, b, b)
            s = l1_scale(a, b)
            c = nmae_against(s * a, b, b) if np.isfinite(s) else np.nan
            raw.append(r)
            cal.append(c)
            store[f"{SHORT[k[0]]}|{k[1]}|{METHOD_LABEL[m]}"] = {
                "raw": round(r, 1), "calibrated": round(c, 1),
                "l1_scale": round(s, 4)}
        pos = x + (j - (len(all_m) - 1) / 2) * w
        ax.bar(pos, raw, w * 0.9, color=METHOD_COLOR[m], linewidth=0,
               alpha=0.30)
        # surface-coloured ring separates the calibrated bar from the raw
        # bar it sits inside
        ax.bar(pos, cal, w * 0.9, color=METHOD_COLOR[m],
               edgecolor="#fcfcfb", linewidth=0.7, label=METHOD_LABEL[m])
    ax.axhline(100, color=INK_MUTED, ls=(0, (4, 3)), lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{SHORT[k[0]]}\n{k[1]}" for k in keys], color=INK)
    ax.set_ylabel("NMAE vs AP (%)", color=INK)
    ax.set_title("Solid = after the L1-optimal rescaling; pale = raw. "
                 "A large gap means the error is magnitude, not ranking.",
                 fontsize=8, color=INK_MUTED, loc="left")
    ax.grid(axis="y", alpha=0.8)
    ax.set_axisbelow(True)
    handles, labs = ax.get_legend_handles_labels()
    handles.append(mpl.patches.Patch(facecolor=INK_MUTED, alpha=0.30))
    labs.append("raw (unscaled)")
    ax.legend(handles, labs, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    fig.savefig(Path(out_dir) / "fig_scale_vs_shape.pdf")
    plt.close(fig)
    payload["scale_vs_shape"] = store


# ── Figure 5 — stability: within-method vs between-method ─────────────
def population_topk(df, col, k, topk_mode="abs", units=None):
    """Components a method selects for the task overall (mean |score| rank)."""
    g = df.groupby(GROUP)
    keys = list(g.groups) if units is None else units
    acc = None
    n = 0
    for key in keys:
        sub = g.get_group(key)
        v = sub[col].values
        order = rank_idx(v, topk_mode)
        ranks = np.empty(len(v))
        ranks[order] = np.arange(len(v))
        acc = ranks if acc is None else acc + ranks
        n += 1
    comp = sub[["layer", "component_type", "head_idx"]].apply(tuple, axis=1)
    mean_rank = acc / max(1, n)
    idx = np.argsort(mean_rank)[:k]
    return set(comp.values[idx].tolist())


def fig_stability(data, out_dir, payload, k=10, n_boot=40, seed=0):
    """If a method disagrees with ITSELF across data as much as it disagrees
    with AP, the localisation exercise is noise-limited."""
    keys = [(m, "zero") for m in MODELS if (m, "zero") in data]
    if not keys:
        return
    rng = np.random.RandomState(seed)
    fig, ax = plt.subplots(figsize=(6.6, 2.7))
    all_m = sorted({m for kk in keys for m in methods_in(data[kk])},
                   key=METHOD_ORDER.index) + ["ap_precap"]
    x = np.arange(len(keys))
    w = 0.8 / max(1, len(all_m))
    store = {}
    ceiling_drawn = False
    for j, m in enumerate(all_m):
        within, between = [], []
        for key in keys:
            df = data[key]
            if m != "ap_precap" and m not in df.columns:
                within.append(np.nan)
                between.append(np.nan)
                continue
            units = list(df.groupby(GROUP).groups)
            js = []
            for _ in range(n_boot):
                ua = [units[i] for i in rng.randint(0, len(units), len(units))]
                ub = [units[i] for i in rng.randint(0, len(units), len(units))]
                sa = population_topk(df, m, k, units=ua)
                sb = population_topk(df, m, k, units=ub)
                js.append(len(sa & sb) / max(1, len(sa | sb)))
            within.append(float(np.median(js)))
            sm = population_topk(df, m, k)
            sap = population_topk(df, "ap_precap", k)
            between.append(len(sm & sap) / max(1, len(sm | sap)))
            store[f"{SHORT[key[0]]}|{METHOD_LABEL.get(m, 'AP')}"] = {
                "within": round(within[-1], 3),
                "between_vs_ap": round(between[-1], 3)}
        pos = x + (j - (len(all_m) - 1) / 2) * w
        if m == "ap_precap":
            # AP's agreement with itself is the CEILING: no method can match
            # AP more closely than AP's own estimate is reproducible.
            for xi, v in zip(x, within):
                ax.plot([xi - 0.45, xi + 0.45], [v, v], color=INK, lw=1.4,
                        ls=(0, (4, 2)),
                        label="AP self-agreement (ceiling)"
                        if not ceiling_drawn else None)
                ceiling_drawn = True
            continue
        colr = METHOD_COLOR[m]
        ax.bar(pos, within, w * 0.9, color=colr, alpha=0.30, linewidth=0)
        ax.plot(pos, between, "o", ms=5, color=colr,
                markeredgecolor="#fcfcfb", markeredgewidth=0.8,
                label=METHOD_LABEL[m])
    ax.set_xticks(x)
    ax.set_xticklabels([SHORT[kk[0]] for kk in keys], color=INK)
    ax.set_ylabel(f"top-{k} Jaccard", color=INK)
    ax.set_title("Pale bar = a method's agreement with itself across data "
                 "resamples; dot = its agreement with AP.",
                 fontsize=8, color=INK_MUTED, loc="left")
    ax.grid(axis="y", alpha=0.8)
    ax.set_axisbelow(True)
    h, lb = ax.get_legend_handles_labels()
    h.append(mpl.patches.Patch(facecolor=INK_MUTED, alpha=0.30))
    lb.append("self-agreement (bar)")
    ax.legend(h, lb, loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=7)
    fig.savefig(Path(out_dir) / "fig_stability.pdf")
    plt.close(fig)
    payload["stability"] = store


# ── Appendix figures ──────────────────────────────────────────────────
def fig_per_layer(data, out_dir):
    keys = [(m, "zero") for m in MODELS if (m, "zero") in data]
    if not keys:
        return
    fig, axes = plt.subplots(1, len(keys), figsize=(6.6, 2.2), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, key in zip(axes, keys):
        df = data[key]
        layers = sorted(df["layer"].unique())
        for m in methods_in(df):
            y = [nmae_against(df.loc[df.layer == L, m].values,
                              df.loc[df.layer == L, "ap_precap"].values,
                              df.loc[df.layer == L, "ap_precap"].values)
                 for L in layers]
            ax.plot(layers, y, lw=1.2, color=METHOD_COLOR[m],
                    label=METHOD_LABEL[m])
        ax.set_title(SHORT[key[0]], color=INK)
        ax.set_xlabel("layer", color=INK)
        ax.grid(alpha=0.8)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("NMAE vs AP (%)", color=INK)
    axes[-1].legend(loc="upper left", bbox_to_anchor=(1.02, 1.0))
    fig.savefig(Path(out_dir) / "fig_per_layer.pdf")
    plt.close(fig)


def fig_heads_vs_mlps(data, out_dir):
    keys = sorted([k for k in data],
                  key=lambda k: (MODELS.index(k[0]), MODES.index(k[1])))
    if not keys:
        return
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.5), sharey=True)
    for ax, ct in zip(axes, ["head", "mlp"]):
        all_m = sorted({m for k in keys for m in methods_in(data[k])},
                       key=METHOD_ORDER.index)
        x = np.arange(len(keys))
        w = 0.8 / max(1, len(all_m))
        for j, m in enumerate(all_m):
            vals = []
            for k in keys:
                d = data[k]
                d = d[d.component_type == ct]
                vals.append(nmae_against(d[m].values, d["ap_precap"].values,
                                         d["ap_precap"].values)
                            if m in d.columns and len(d) else np.nan)
            ax.bar(x + (j - (len(all_m) - 1) / 2) * w, vals, w * 0.9,
                   color=METHOD_COLOR[m], linewidth=0, label=METHOD_LABEL[m])
        ax.set_title("Attention heads" if ct == "head" else "MLPs", color=INK)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{SHORT[k[0]]}\n{k[1]}" for k in keys],
                           fontsize=7, color=INK)
        ax.grid(axis="y", alpha=0.8)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("NMAE vs AP (%)", color=INK)
    axes[0].legend(loc="upper left", bbox_to_anchor=(0, 1.26), ncol=4)
    fig.savefig(Path(out_dir) / "fig_heads_vs_mlps.pdf")
    plt.close(fig)


# ── LaTeX tables ──────────────────────────────────────────────────────
def _fmt(v, n=2):
    return "--" if (v is None or not np.isfinite(v)) else f"{v:.{n}f}"


def tab_main(data, out_dir, topk_mode="abs"):
    keys = sorted([k for k in data],
                  key=lambda k: (MODELS.index(k[0]), MODES.index(k[1])))
    lines = [
        r"\begin{table}[t]", r"\centering",
        r"\caption{Cheap attribution methods against activation-patching "
        r"ground truth. NMAE is normalised by $\sum|\mathrm{AP}|$; "
        r"\emph{cal.} is NMAE after the $L_1$-optimal rescaling, so a large "
        r"raw--cal.\ gap means the error is magnitude rather than ranking. "
        r"$\rho$, $\tau_b$, RBO and J10 are medians over evaluation units "
        r"(one per example--candidate pair). Sign agreement is reported "
        r"against its empirical base rate.}",
        r"\label{tab:main}", r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{llrrrrrrrr}", r"\toprule",
        r"Model & Mode & Method & NMAE & cal. & $\rho$ & $\tau_b$ & RBO & "
        r"J10 & MCC \\", r"\midrule",
    ]
    for key in keys:
        df = data[key]
        ms = methods_in(df)
        for i, m in enumerate(ms):
            um = unit_metrics(df, m, topk_mode=topk_mode)
            a, b = df[m].values, df["ap_precap"].values
            raw = nmae_against(a, b, b)
            s = l1_scale(a, b)
            cal = nmae_against(s * a, b, b) if np.isfinite(s) else np.nan
            lhs = (f"\\multirow{{{len(ms)}}}{{*}}{{{SHORT[key[0]]}}} & "
                   f"\\multirow{{{len(ms)}}}{{*}}{{{key[1]}}}"
                   if i == 0 else " & ")
            lines.append(
                f"{lhs} & {METHOD_LABEL[m]} & {_fmt(raw,1)} & {_fmt(cal,1)} & "
                f"{_fmt(um['rho'],3)} & {_fmt(um['tau'],3)} & "
                f"{_fmt(um['rbo'],3)} & {_fmt(um['j10'],3)} & "
                f"{_fmt(um['mcc'],3)} \\\\")
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines += [r"\end{tabular}", r"\end{table}"]
    (Path(out_dir) / "tab_main.tex").write_text("\n".join(lines))


def tab_budget(data, out_dir):
    keys = sorted([k for k in data],
                  key=lambda k: (MODELS.index(k[0]), MODES.index(k[1])))
    lines = [
        r"\begin{table}[t]", r"\centering",
        r"\caption{Error budget. Source A is the frozen-norm term "
        r"$\mathrm{NMAE}(\mathrm{DLA},\mathrm{DE})$ and Source B the "
        r"indirect-effect term $\mathrm{NMAE}(\mathrm{DE},\mathrm{AP})$, both "
        r"normalised by $\sum|\mathrm{AP}|$. \emph{pooled} uses one global "
        r"denominator; \emph{per-unit} is the median of the per-unit value "
        r"and is the robust companion.}",
        r"\label{tab:budget}", r"\small",
        r"\begin{tabular}{llrrrrrr}", r"\toprule",
        r" & & \multicolumn{2}{c}{Source A} & \multicolumn{2}{c}{Source B} & "
        r"\multicolumn{2}{c}{Total} \\",
        r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}",
        r"Model & Mode & pooled & per-unit & pooled & per-unit & pooled & "
        r"per-unit \\", r"\midrule",
    ]
    for key in keys:
        df = data[key]
        cells = []
        for ca, cb in [("dla", "de_precap"), ("de_precap", "ap_precap"),
                       ("dla", "ap_precap")]:
            pooled = nmae_against(df[ca].values, df[cb].values,
                                  df["ap_precap"].values)
            per = [nmae_against(s[ca].values, s[cb].values,
                                s["ap_precap"].values)
                   for _, s in df.groupby(GROUP)]
            per = [p for p in per if np.isfinite(p)]
            cells += [_fmt(pooled, 1), _fmt(float(np.median(per)), 1)]
        lines.append(f"{SHORT[key[0]]} & {key[1]} & " + " & ".join(cells) +
                     r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (Path(out_dir) / "tab_budget.tex").write_text("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records-dir", default="outputs/dla_error_analysis")
    ap.add_argument("--out-dir", default="outputs/paper")
    ap.add_argument("--tag", default="")
    ap.add_argument("--topk-mode", choices=["abs", "signed"], default="abs")
    ap.add_argument("--topk", type=int, default=10)
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("Loading records...")
    data = load_all(args.records_dir, args.tag)
    if not data:
        print("No records found — check --records-dir and --tag.")
        return

    payload: dict = {"topk_mode": args.topk_mode, "topk": args.topk}
    print("\nFigures...")
    for fn, name in [
        (lambda: fig_error_budget(data, out, payload), "error budget"),
        (lambda: fig_fingerprint(data, out, payload, k=args.topk,
                                 topk_mode=args.topk_mode), "fingerprint"),
        (lambda: fig_rank_agreement(data, out, payload,
                                    topk_mode=args.topk_mode),
         "rank agreement"),
        (lambda: fig_scale_vs_shape(data, out, payload), "scale vs shape"),
        (lambda: fig_stability(data, out, payload, k=args.topk), "stability"),
        (lambda: fig_per_layer(data, out), "per-layer"),
        (lambda: fig_heads_vs_mlps(data, out), "heads vs MLPs"),
    ]:
        try:
            fn()
            print(f"  ok   {name}")
        except Exception as exc:  # noqa: BLE001
            print(f"  FAIL {name}: {exc!r}")

    print("\nTables...")
    for fn, name in [(lambda: tab_main(data, out, args.topk_mode), "main"),
                     (lambda: tab_budget(data, out), "budget")]:
        try:
            fn()
            print(f"  ok   {name}")
        except Exception as exc:  # noqa: BLE001
            print(f"  FAIL {name}: {exc!r}")

    (out / "figure_data.json").write_text(json.dumps(payload, indent=2,
                                                     default=float))
    print(f"\nWrote to {out}/ — figures, tables, figure_data.json")


if __name__ == "__main__":
    main()