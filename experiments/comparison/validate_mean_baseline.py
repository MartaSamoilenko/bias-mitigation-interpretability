import argparse
import os
import random
import sys
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dla_error_analysis import (
    build_component_catalog,
    compute_activation_means,
    load_model,
)
from mitigation_sweep import (
    GROUP,
    MERGE_KEYS,
    METHOD_COLS,
    build_joint_ablation_hooks,
    capability_metric,
    load_generic_texts,
    population_ranking,
    random_ranking,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="meta-llama/Llama-3.2-1B")
    p.add_argument("--records-dir", default="outputs/dla_error_analysis")
    p.add_argument("--tag", default="dev")
    p.add_argument("--ablation-mode", default="mean", choices=["mean"])
    p.add_argument("--generic-text", default="data/wikitext103_valid.txt")
    p.add_argument("--n-generic", type=int, default=30)
    p.add_argument("--n-mean-examples", type=int, default=30)
    p.add_argument("--dev-dataset",
                   default="datasets/gender_test_rephrased_v2.json")
    p.add_argument("--ks", type=int, nargs="+", default=[1, 2, 4, 8])
    p.add_argument("--methods", nargs="+", default=["random", "ap"])
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    import json
    dev = json.loads(Path(args.dev_dataset).read_text())
    generic = load_generic_texts(args.generic_text, args.n_generic)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}  generic={len(generic)} texts")

    suffix = f"_{args.tag}" if args.tag else ""
    safe = args.model.replace("/", "_")
    rec = Path(args.records_dir) / (
        f"{safe}_{args.ablation_mode}_ablation_records{suffix}.csv")
    df = pd.read_csv(rec)
    eap = Path(args.records_dir) / (
        f"{safe}_eap_ig_{args.ablation_mode}_ablation_records{suffix}.csv")
    if eap.exists():
        e = pd.read_csv(eap)
        keep = [c for c in ("eap_ig", "atp") if c in e.columns]
        df = df.merge(e[MERGE_KEYS + keep], on=MERGE_KEYS, how="inner")
    print(f"{args.model}: {len(df):,} rows, {df.groupby(GROUP).ngroups} units")

    model = load_model(args.model, device)
    model.eval()
    catalog = build_component_catalog(model)

    random.seed(args.seed)
    sel = (random.sample(dev, args.n_mean_examples)
           if len(dev) > args.n_mean_examples else list(dev))

    print("\ncomputing both baselines ...")
    means = {
        "last": compute_activation_means(model, sel, positions="last"),
        "all_nonbos": compute_activation_means(
            model, sel, positions="all_nonbos", extra_texts=generic),
    }

    base_ppl, _, _ = capability_metric(model, generic)
    print(f"clean ppl = {base_ppl:.3f}\n")

    rankings = {}
    for m in args.methods:
        if m == "random":
            rankings[m] = random_ranking(catalog, args.seed)
        elif METHOD_COLS.get(m) in df.columns:
            rankings[m] = population_ranking(df, METHOD_COLS[m], signed=True)

    print(f"{'method':8s} {'k':>3s} {'MLPs':>5s} "
          f"{'cost(last)':>12s} {'cost(all_nonbos)':>18s} {'ratio':>9s}")
    print("-" * 62)
    rows = []
    for m, ranked in rankings.items():
        for k in args.ks:
            if k > len(ranked):
                continue
            comps = ranked[:k]
            n_mlp = sum(1 for c in comps if c[1] == "mlp")
            costs = {}
            for name, mu in means.items():
                hooks = build_joint_ablation_hooks(comps, means=mu)
                ppl, _, _ = capability_metric(model, generic, hooks=hooks)
                costs[name] = ppl / base_ppl - 1.0
            ratio = ((1 + costs["last"]) / (1 + costs["all_nonbos"])
                     if costs["all_nonbos"] > -1 else float("nan"))
            print(f"{m:8s} {k:3d} {n_mlp:5d} "
                  f"{costs['last']*100:11.2f}% {costs['all_nonbos']*100:17.2f}% "
                  f"{ratio:9.2f}x")
            rows.append({"model": args.model, "method": m, "k": k,
                         "n_mlps": n_mlp,
                         "cost_last": costs["last"],
                         "cost_all_nonbos": costs["all_nonbos"],
                         "ppl_ratio_last_over_nonbos": ratio})

    out = Path("outputs/mitigation") / f"baseline_validation_{safe}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
