import argparse
from pathlib import Path

import matplotlib as mpl
import numpy as np
import pandas as pd

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

mpl.rcParams.update({
    "figure.dpi": 120, "savefig.dpi": 300, "savefig.bbox": "tight",
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
    "font.family": "serif", "font.size": 9, "legend.frameon": False,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#8a8a85", "axes.linewidth": 0.6,
    "grid.color": "#e3e3df", "grid.linewidth": 0.5,
    "xtick.color": "#8a8a85", "ytick.color": "#8a8a85",
    "xtick.labelsize": 8, "ytick.labelsize": 8,
})
INK, INK_MUTED = "#1c1c1a", "#6b6b66"
METHOD_COLOR = {"dla": "#2F6FB5", "atp": "#D2691E", "eap_ig": "#00897B",
                "de_precap": "#8E44AD", "random": "#9a9a95"}
METHOD_LABEL = {"dla": "DLA", "atp": "AtP", "eap_ig": "EAP-IG",
                "de_precap": "DE", "random": "Random"}
ORDER = ["dla", "atp", "eap_ig", "de_precap"]

MODELS = ["gpt2-xl", "meta-llama/Llama-3.2-1B", "google/gemma-2-2b"]
SHORT = {"gpt2-xl": "GPT-2 XL",
         "meta-llama/Llama-3.2-1B": "Llama-3.2-1B",
         "google/gemma-2-2b": "Gemma-2-2B"}
GROUP = ["example_id", "unit_id"]
MERGE_KEYS = ["model", "example_id", "unit_id", "layer",
              "component_type", "head_idx"]


def load(records_dir, model, mode, tag=""):
    suffix = f"_{tag}" if tag else ""
    safe = model.replace("/", "_")
    ap = Path(records_dir) / f"{safe}_{mode}_ablation_records{suffix}.csv"
    if not ap.exists():
        return None
    df = pd.read_csv(ap)
    eap = Path(records_dir) / f"{safe}_eap_ig_{mode}_ablation_records{suffix}.csv"
    if eap.exists():
        e = pd.read_csv(eap)
        for d in (df, e):
            for c in ("layer", "head_idx"):
                d[c] = d[c].astype("int64")
            for c in ("model", "example_id", "unit_id", "component_type"):
                d[c] = d[c].astype(str)
        keep = [c for c in ("eap_ig", "atp") if c in e.columns]
        df = df.merge(e[MERGE_KEYS + keep], on=MERGE_KEYS, how="inner")
    return df


def unit_spearman(df, col):
    """One Spearman per unit, so sub-sampling units is just a re-slice."""
    from scipy.stats import spearmanr
    out = {}
    for key, sub in df.groupby(GROUP):
        a, b = sub[col].values, sub["ap_precap"].values
        if np.ptp(a) == 0 or np.ptp(b) == 0:
            continue
        r, _ = spearmanr(a, b)
        if np.isfinite(r):
            out[key] = r
    return out


def curve(per_unit, sizes, n_draw, seed=0):
    """Median of the per-unit statistic at each sub-sample size."""
    rng = np.random.RandomState(seed)
    vals = np.array(list(per_unit.values()))
    rows = []
    for n in sizes:
        if n > len(vals):
            continue
        draws = [np.median(vals[rng.choice(len(vals), n, replace=False)])
                 for _ in range(n_draw)]
        rows.append((n, float(np.median(draws)),
                     float(np.percentile(draws, 2.5)),
                     float(np.percentile(draws, 97.5))))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records-dir", default="outputs/dla_error_analysis")
    ap.add_argument("--out-dir", default="outputs/paper")
    ap.add_argument("--tag", default="")
    ap.add_argument("--mode", default="zero", choices=["zero", "mean"])
    ap.add_argument("--n-draw", type=int, default=200)
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    frames = {}
    for m in MODELS:
        df = load(args.records_dir, m, args.mode, args.tag)
        if df is not None:
            frames[m] = df
    if not frames:
        raise SystemExit("no records found")

    fig, axes = plt.subplots(1, len(frames), figsize=(3.2 * len(frames), 2.9),
                             sharey=True)
    axes = np.atleast_1d(axes)

    for ax, (model, df) in zip(axes, frames.items()):
        n_units = df.groupby(GROUP).ngroups
        sizes = [s for s in (10, 20, 40, 80, 160, 320, 640, 1280)
                 if s <= n_units] + [n_units]
        sizes = sorted(set(sizes))
        print(f"\n{SHORT[model]}  ({n_units} units)")
        print(f"  {'method':8s} {'n':>6}  {'median rho':>11}  "
              f"{'95% band':>18}  {'band width':>10}")
        for col in [c for c in ORDER if c in df.columns]:
            per_unit = unit_spearman(df, col)
            rows = curve(per_unit, sizes, args.n_draw)
            if not rows:
                continue
            n, med, lo, hi = zip(*rows)
            ax.plot(n, med, marker="o", ms=3.5, lw=1.5,
                    color=METHOD_COLOR[col], label=METHOD_LABEL[col])
            ax.fill_between(n, lo, hi, color=METHOD_COLOR[col], alpha=0.15,
                            linewidth=0)
            for r in rows:
                print(f"  {METHOD_LABEL[col]:8s} {r[0]:6d}  {r[1]:11.3f}  "
                      f"[{r[2]:+.3f}, {r[3]:+.3f}]  {r[3] - r[2]:10.3f}")
        ax.set_xscale("log", base=2)
        ax.set_title(SHORT[model], color=INK)
        ax.set_xlabel("evaluation units sampled", color=INK)
        ax.grid(alpha=0.8)
        ax.set_axisbelow(True)
    axes[0].set_ylabel(r"Spearman $\rho$ vs AP (median)", color=INK)
    axes[-1].legend(loc="upper left", bbox_to_anchor=(1.02, 1.0))
    fig.suptitle("Shaded band = 95% over sub-samples. Converged once the band "
                 "is narrower than the\ngap between the methods you are "
                 "separating.", fontsize=8, color=INK_MUTED, y=1.10)
    p = out / "fig_subsample_stability.pdf"
    fig.savefig(p)
    plt.close(fig)
    print(f"\n-> {p}")


if __name__ == "__main__":
    main()
