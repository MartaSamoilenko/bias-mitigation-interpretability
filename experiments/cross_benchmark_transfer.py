"""Cross-benchmark transfer evaluation.

Evaluates top-performing models from one benchmark (StereoSet or Winogender)
on the other benchmark to measure how well debiasing transfers.

Usage:
    python cross_benchmark_transfer.py \
        --source_benchmark stereoset \
        --run_id dpo_attn_1.0_beta0.3_lr5e-06 \
        --best_epoch 2 \
        --target_benchmark winogender
"""

import argparse
import gc
import json
import os
import sys
import tempfile

import boto3
import numpy as np
import pandas as pd
import torch
from dotenv import load_dotenv
from scipy import stats
from transformer_lens import HookedTransformer

load_dotenv()

S3_BUCKET = "modelsfinetuned"
STEREOSET_PREFIX = "stereoset_experiments"
WINOGENDER_PREFIX = "experiments"
CKPT_PREFIX = "gpt2-xl-finetuned"


def _s3_client():
    return boto3.client(
        "s3",
        aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
    )


def load_checkpoint(run_id: str, best_epoch: int) -> HookedTransformer:
    """Download checkpoint from S3 and load into a fresh HookedTransformer."""
    ckpt_key = f"{CKPT_PREFIX}/best_model_{run_id}_epoch_{best_epoch}.pt"
    model = HookedTransformer.from_pretrained("gpt2-xl")
    client = _s3_client()
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=True) as tmp:
        print(f"Downloading s3://{S3_BUCKET}/{ckpt_key} ...")
        client.download_file(S3_BUCKET, ckpt_key, tmp.name)
        model.load_state_dict(torch.load(tmp.name, weights_only=True))
    model.eval()
    return model


# ---------------------------------------------------------------------------
# StereoSet evaluation functions
# ---------------------------------------------------------------------------
def _read_s3_csv(prefix: str, path: str) -> pd.DataFrame:
    import io
    client = _s3_client()
    key = f"{prefix}/{path}"
    obj = client.get_object(Bucket=S3_BUCKET, Key=key)
    return pd.read_csv(io.BytesIO(obj["Body"].read()))


def _read_s3_json(prefix: str, path: str):
    client = _s3_client()
    key = f"{prefix}/{path}"
    obj = client.get_object(Bucket=S3_BUCKET, Key=key)
    return json.loads(obj["Body"].read().decode("utf-8"))


def _write_s3_json(prefix: str, path: str, data):
    client = _s3_client()
    key = f"{prefix}/{path}"
    body = json.dumps(data, indent=2).encode("utf-8")
    client.put_object(Bucket=S3_BUCKET, Key=key, Body=body)


def compute_stereoset_metrics(prob_df: pd.DataFrame):
    """Compute SS, LMS, ICAT for StereoSet."""
    max_layer = prob_df["Layer"].max()
    last_tok_idx = prob_df.groupby(["ID", "Type", "Layer"])["Token_Position"].idxmax()
    final = prob_df.loc[last_tok_idx]
    final = final[final["Layer"] == max_layer]
    pivot = final.pivot(index="ID", columns="Type", values="Layer_Accumulated_Prob").fillna(0)

    n_total = len(pivot)
    related = 0
    n_stereo = n_anti = 0

    has_all_three = {"stereotype", "anti-stereotype", "unrelated"} <= set(pivot.columns)
    has_stereo_anti = {"stereotype", "anti-stereotype"} <= set(pivot.columns)

    if has_all_three:
        related += int((pivot["stereotype"] > pivot["unrelated"]).sum())
        related += int((pivot["anti-stereotype"] > pivot["unrelated"]).sum())
    if has_stereo_anti:
        n_stereo = int((pivot["stereotype"] > pivot["anti-stereotype"]).sum())
        n_anti = int((pivot["anti-stereotype"] > pivot["stereotype"]).sum())

    lms = (related / (2 * n_total) * 100) if (n_total > 0 and has_all_three) else 0.0
    denom = n_stereo + n_anti
    ss = (n_stereo / denom * 100) if denom > 0 else 50.0
    icat = lms * (min(ss, 100.0 - ss) / 50.0)

    return {"SS": ss, "LMS": lms, "ICAT": icat}


def compute_winogender_metrics(prob_df: pd.DataFrame, meta_df: pd.DataFrame):
    """Compute SS, BLS_score, ICAT for Winogender."""
    max_layer = prob_df["Layer"].max()
    last_tok = prob_df.groupby(["ID", "Gender", "Layer"])["Token_Position"].idxmax()
    final = prob_df.loc[last_tok]
    final = final[final["Layer"] == max_layer]

    occ_df = prob_df[prob_df["Sentence_Role"] == "occupation"] if "Sentence_Role" in prob_df.columns else prob_df

    max_layer = occ_df["Layer"].max()
    last_tok = occ_df.groupby(["ID", "Gender", "Layer"])["Token_Position"].idxmax()
    final = occ_df.loc[last_tok]
    final = final[final["Layer"] == max_layer]
    pivot = final.pivot_table(
        index="ID", columns="Gender",
        values="Layer_Accumulated_Prob", aggfunc="first",
    ).fillna(0)

    result = pd.DataFrame({"ID": pivot.index})
    result["P_male"] = pivot.get("male", 0).values
    result["P_female"] = pivot.get("female", 0).values
    result = result.reset_index(drop=True)

    merged = result.merge(meta_df[["ID", "bls_pct_female"]], on="ID", how="left")
    merged["stereo_gender"] = merged["bls_pct_female"].apply(
        lambda x: "female" if x > 50 else "male"
    )
    merged["P_stereo"] = np.where(
        merged["stereo_gender"] == "male", merged["P_male"], merged["P_female"]
    )
    merged["P_anti"] = np.where(
        merged["stereo_gender"] != "male", merged["P_male"], merged["P_female"]
    )

    n_stereo = (merged["P_stereo"] > merged["P_anti"]).sum()
    n_anti = (merged["P_anti"] > merged["P_stereo"]).sum()
    denom = n_stereo + n_anti
    ss = (n_stereo / denom * 100) if denom > 0 else 50.0

    if len(merged) > 2:
        bls_r, _ = stats.pearsonr(merged["bls_pct_female"], merged["P_stereo"])
    else:
        bls_r = 0.0
    bls_score = (1 - abs(bls_r)) * 100
    icat = bls_score * (min(ss, 100 - ss) / 50.0)

    return {"SS": ss, "BLS_score": bls_score, "ICAT": icat}


# ---------------------------------------------------------------------------
# DLA tracing (lightweight — reuse bias_search functions)
# ---------------------------------------------------------------------------
def run_stereoset_dla(model: HookedTransformer):
    """Run the StereoSet DLA tracing pipeline and return (prob_df, impact_df)."""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "stereoset"))
    from stereoset.stereoset_bias_search import get_logit_attribution, accumulative_layer_impact

    test_data = _read_s3_json(STEREOSET_PREFIX, "data/stereoset/splits/gender_test.json")

    all_prob_rows = []
    all_impact_rows = []
    tokenizer = model.tokenizer

    for entry in test_data:
        sid = entry["id"]
        context = entry["context"]
        for sentence_info in entry["sentences"]:
            sentence = sentence_info["sentence"]
            stype = sentence_info["gold_label"]
            tokens = tokenizer.encode(sentence)
            if len(tokens) == 0:
                continue
            try:
                logit_attr, probs = get_logit_attribution(model, tokens)
            except Exception:
                continue

            for layer in range(logit_attr.shape[0]):
                for tok_pos in range(logit_attr.shape[2]):
                    all_prob_rows.append({
                        "ID": sid, "Type": stype, "Layer": layer,
                        "Token_Position": tok_pos,
                        "Layer_Accumulated_Prob": float(probs[layer, tok_pos]) if tok_pos < probs.shape[1] else 0.0,
                    })

    prob_df = pd.DataFrame(all_prob_rows)
    return prob_df


def run_winogender_dla(model: HookedTransformer):
    """Run Winogender DLA and return prob_df."""
    meta = _read_s3_json(WINOGENDER_PREFIX, "data/winogender/winogender_test_metadata.json")
    meta_df = pd.DataFrame(meta).rename(columns={"id": "ID"})

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "winogender"))
    from winogender.winogender_bias_search import get_logit_attribution

    all_rows = []
    for _, mrow in meta_df.iterrows():
        for gender in ["male", "female", "neutral"]:
            key_col = f"pronoun_{gender}"
            if key_col not in mrow:
                continue
            sentence = mrow.get("occupation_sentence", "")
            if not sentence:
                continue
            tokens = model.tokenizer.encode(sentence)
            if len(tokens) == 0:
                continue
            try:
                logit_attr, probs = get_logit_attribution(model, tokens)
            except Exception:
                continue
            for layer in range(probs.shape[0]):
                for tok_pos in range(probs.shape[1]):
                    all_rows.append({
                        "ID": mrow["ID"], "Gender": gender,
                        "Sentence_Role": "occupation",
                        "Layer": layer, "Token_Position": tok_pos,
                        "Layer_Accumulated_Prob": float(probs[layer, tok_pos]),
                    })

    return pd.DataFrame(all_rows)


def evaluate_transfer(run_id: str, best_epoch: int, source_benchmark: str, target_benchmark: str):
    """Load a model fine-tuned on source_benchmark and evaluate on target_benchmark."""
    print(f"\nLoading model: {run_id} (epoch {best_epoch}) trained on {source_benchmark}")
    model = load_checkpoint(run_id, best_epoch)

    if target_benchmark == "stereoset":
        print("Running StereoSet evaluation...")
        prob_df = run_stereoset_dla(model)
        metrics = compute_stereoset_metrics(prob_df)
    elif target_benchmark == "winogender":
        print("Running Winogender evaluation...")
        meta = _read_s3_json(WINOGENDER_PREFIX, "data/winogender/winogender_test_metadata.json")
        meta_df = pd.DataFrame(meta).rename(columns={"id": "ID"})
        prob_df = run_winogender_dla(model)
        metrics = compute_winogender_metrics(prob_df, meta_df)
    else:
        raise ValueError(f"Unknown target_benchmark: {target_benchmark}")

    del model
    gc.collect()
    torch.cuda.empty_cache()

    result = {
        "run_id": run_id,
        "source_benchmark": source_benchmark,
        "target_benchmark": target_benchmark,
        **metrics,
    }

    out_path = f"outputs/gpt2-xl/cross_benchmark/{source_benchmark}_to_{target_benchmark}_{run_id}.json"
    prefix = STEREOSET_PREFIX if source_benchmark == "stereoset" else WINOGENDER_PREFIX
    _write_s3_json(prefix, out_path, result)
    print(f"Saved: {out_path}")
    print(f"Results: {metrics}")

    return result


def main():
    parser = argparse.ArgumentParser(description="Cross-benchmark transfer evaluation")
    parser.add_argument("--source_benchmark", required=True, choices=["stereoset", "winogender"])
    parser.add_argument("--run_id", required=True)
    parser.add_argument("--best_epoch", required=True, type=int)
    parser.add_argument("--target_benchmark", required=True, choices=["stereoset", "winogender"])
    args = parser.parse_args()

    result = evaluate_transfer(
        args.run_id, args.best_epoch,
        args.source_benchmark, args.target_benchmark,
    )
    print(f"\nFinal result: {json.dumps(result, indent=2)}")


if __name__ == "__main__":
    main()
