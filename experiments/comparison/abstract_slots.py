import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import paper_figures as P


def jaccard_to_shared(j, k=10):
    """Top-k Jaccard -> how many of the k are shared. |A|=|B|=k."""
    return 2 * k * j / (1 + j)


def spread(vals, fmt="{:.0f}", unit=""):
    v = [x for x in vals if np.isfinite(x)]
    if not v:
        return "--"
    lo, hi = min(v), max(v)
    if fmt.format(lo) == fmt.format(hi):
        return fmt.format(lo) + unit
    return f"{fmt.format(lo)}–{fmt.format(hi)}{unit}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records-dir", default="outputs/dla_error_analysis")
    ap.add_argument("--tag", default="test")
    ap.add_argument("--sweep", default=None)
    ap.add_argument("--budget", type=float, default=0.05,
                    help="capability-cost budget the mitigation slots read at")
    args = ap.parse_args()

    data = P.load_all(args.records_dir, args.tag)
    if not data:
        raise SystemExit("no records found")

    slots, notes = {}, []

    # scope
    models = list(dict.fromkeys(m for m, _ in data))
    slots["N_MODELS"] = {1: "one", 2: "two", 3: "three"}.get(
        len(models), str(len(models)))
    slots["MODELS"] = ", ".join(P.SHORT[m] for m in models)
    slots["N_UNITS"] = spread(
        [data[k].groupby(P.GROUP).ngroups for k in data], "{:.0f}")
    slots["N_MODES"] = {1: "one", 2: "two"}.get(
        len({md for _, md in data}), "several")
    # methods actually present, excluding the random floor and the AP skyline
    meth = set()
    for df in data.values():
        meth |= {m for m in P.methods_in(df, include_random=False)}
    slots["N_METHODS"] = {1: "one", 2: "two", 3: "three", 4: "four",
                          5: "five"}.get(len(meth), str(len(meth)))
    slots["METHOD_NAMES"] = ", ".join(
        P.METHOD_LABEL[m] for m in P.METHOD_ORDER if m in meth)

    # error budget
    a_share, b_pool, tot = [], [], []
    for key, df in data.items():
        A = P.nmae_against(df["dla"], df["de_precap"], df["ap_precap"])
        B = P.nmae_against(df["de_precap"], df["ap_precap"], df["ap_precap"])
        T = P.nmae_against(df["dla"], df["ap_precap"], df["ap_precap"])
        a_share.append(100 * A / T if T else np.nan)
        b_pool.append(B)
        tot.append(T)
    slots["A_SHARE"] = spread(a_share, "{:.0f}", "%")
    slots["B_SHARE"] = spread(b_pool, "{:.0f}", "%")
    slots["TOTAL"] = spread(tot, "{:.0f}", "%")
    hi_a = max(x for x in a_share if np.isfinite(x))
    slots["A_SHARE_WORDS"] = (
        "under a tenth" if hi_a < 10 else
        "under a sixth" if hi_a < 16.7 else
        "under a fifth" if hi_a < 20 else
        "under a quarter" if hi_a < 25 else f"up to {hi_a:.0f}%")
    if min(tot) < 100 <= max(tot):
        slots["TOTAL_WORDS"] = "comparable to or worse than"
    elif max(tot) < 100:
        slots["TOTAL_WORDS"] = "approaching"
        notes.append(
            "Total error is BELOW 100% everywhere, so 'worse than a "
            "constant-zero predictor' is NOT supported. Template uses "
            "'approaching'.")
    else:
        slots["TOTAL_WORDS"] = "worse than"

    # per-method recovery of AP's top 10
    per_method = {}
    for key, df in data.items():
        for m in P.methods_in(df, include_random=False):
            if m == "de_precap":
                continue
            um = P.unit_metrics(df, m)
            per_method.setdefault(m, []).append(jaccard_to_shared(um["j10"]))
    for m, vals in per_method.items():
        tag = P.METHOD_LABEL[m].upper().replace("-", "")
        slots[f"{tag}_TOP10"] = spread(vals, "{:.0f}")
    if "dla" in per_method:
        hi = max(per_method["dla"])
        slots["DLA_TOP10_WORDS"] = (
            "fewer than half" if hi < 5 else
            "about half" if hi <= 6 else f"{hi:.0f} of ten")

    ncomp = [df.groupby(["layer", "component_type", "head_idx"]).ngroups
             for df in data.values()]
    slots["AP_COST"] = spread(ncomp, "{:.0f}")
    slots["CHEAP_COST"] = "three"
    slots["SPEEDUP"] = spread([n / 3 for n in ncomp], "{:.0f}", "x")

    # mitigation
    if args.sweep and Path(args.sweep).exists():
        import mitigation_figures as M
        sw = pd.read_csv(args.sweep)
        rows = []
        for model in dict.fromkeys(sw["model"]):
            dm = sw[sw.model == model]
            for meth in M._sorted_methods(dm):
                sub = dm[dm.method == meth]
                rows.append((model, meth,
                             M.reduction_at_cost(sub, args.budget),
                             M.area_over_random(dm, meth)))
        ok = [r for r in rows if np.isfinite(r[2])]
        if ok:
            best = {}
            for _, meth, red, _a in ok:
                best.setdefault(meth, []).append(red)
            order = sorted(best, key=lambda m: -np.median(best[m]))
            slots["MITIG_ORDER"] = " > ".join(
                M.METHOD_LABEL[m] for m in order)
            slots["MITIG_BUDGET"] = f"{args.budget * 100:.0f}%"
            for m in best:
                slots[f"MITIG_{M.METHOD_LABEL[m].upper().replace('-','').replace(' ','').replace('(SKYLINE)','')}"] = \
                    spread([100 * x for x in best[m]], "{:.0f}", "%")
            med = {m: np.median(best[m]) for m in best}
            gap_atp_dla = med.get("atp", np.nan) - med.get("dla", np.nan)
            gap_dla_rand = med.get("dla", np.nan) - med.get("random", np.nan)
            if np.isfinite(gap_atp_dla) and gap_atp_dla > 0.10:
                slots["MITIG_BRANCH"] = "A (faithfulness transfers)"
            elif np.isfinite(gap_dla_rand) and gap_dla_rand < 0.05:
                slots["MITIG_BRANCH"] = "C (nothing beats random)"
            else:
                slots["MITIG_BRANCH"] = "B (methods tie downstream)"
        if "icat" in sw.columns and sw["icat"].notna().any():
            slots["HAS_ICAT"] = "yes"
    else:
        slots["MITIG_BRANCH"] = "D (no mitigation section)"
        notes.append("No sweep CSV — use template branch D, which omits the "
                     "mitigation sentence entirely. Do not promise it.")

    print(f"\n{'slot':22s} value")
    print("-" * 60)
    for k, v in slots.items():
        print(f"{{{{{k}}}}}".ljust(22) + f" {v}")
    if notes:
        print("\nnotes:")
        for n in notes:
            print(f"  ! {n}")
    print("\nPaste into ABSTRACT_TEMPLATE.md. Every number in the abstract "
          "should appear above;\nif you are about to type one that does not, "
          "it is not backed by the artifacts.")


if __name__ == "__main__":
    main()
