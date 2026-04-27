import argparse
import os
from copy import deepcopy

import boto3
import numpy as np
import torch
import torch.nn.functional as F
from dotenv import load_dotenv
from huggingface_hub import login
from transformer_lens import HookedTransformer

from experiments import s3_utils

_env_dir = os.path.dirname(os.path.abspath(__file__))
for _d in [_env_dir, os.path.dirname(_env_dir), os.path.dirname(os.path.dirname(_env_dir))]:
    _candidate = os.path.join(_d, ".env")
    if os.path.isfile(_candidate):
        load_dotenv(_candidate)
        break
else:
    load_dotenv()

login(token=os.environ["HF_TOKEN"])

MODEL_CONFIGS = {
    "gpt2-xl": {
        "hf_name": "gpt2-xl",
        "log_dir": "outputs/gpt2-xl/winogender/fine_tuned/logs",
        "checkpoint_prefix": "experiments/outputs/gpt2-xl/winogender/fine_tuned/checkpoints",
        "results_base": "outputs/gpt2-xl/winogender/fine_tuned/test",
        "baseline_results": "outputs/gpt2-xl/winogender/baseline/test",
    },
    "gemma-2b": {
        "hf_name": "gemma-2b",
        "log_dir": "outputs/gemma-2b/winogender/fine_tuned/logs",
        "checkpoint_prefix": "experiments/outputs/gemma-2b/winogender/fine_tuned/checkpoints",
        "results_base": "outputs/gemma-2b/winogender/fine_tuned/test",
        "baseline_results": "outputs/gemma-2b/winogender/baseline/test",
    },
    "llama3.2_1b": {
        "hf_name": "meta-llama/Llama-3.2-1B",
        "log_dir": "outputs/llama3.2_1b/winogender/fine_tuned/logs",
        "checkpoint_prefix": "experiments/outputs/llama3.2_1b/winogender/fine_tuned/checkpoints",
        "results_base": "outputs/llama3.2_1b/winogender/fine_tuned/test",
        "baseline_results": "outputs/llama3.2_1b/winogender/baseline/test",
    },
}

S3_BUCKET = "modelsfinetuned"
TEST_DATASET_PATH = "data/winogender/winogender_test_dataset.json"


def compute_sentence_perplexity(model, text):
    tokens = model.to_tokens(text)
    if tokens.shape[1] < 2:
        return float("nan")
    with torch.no_grad():
        logits = model(tokens, return_type="logits")
    shift_logits = logits[:, :-1, :]
    shift_labels = tokens[:, 1:]
    loss = F.cross_entropy(
        shift_logits.reshape(-1, shift_logits.size(-1)),
        shift_labels.reshape(-1),
        reduction="mean",
    )
    return torch.exp(loss).item()


def build_sentences(dataset):
    sentences = []
    for pair in dataset:
        pair_id = pair["id"]
        pronouns = pair["pronouns"]
        for role, sent_key in [("occupation", "sentence_occ"),
                               ("participant", "sentence_part")]:
            sent = pair[sent_key]
            prefix = sent["prefix"]
            suffix = sent.get("suffix", "")
            for gender, pronoun in pronouns.items():
                if suffix:
                    text = f"{prefix} {pronoun} {suffix}"
                else:
                    text = f"{prefix} {pronoun}"
                sentences.append({
                    "id": pair_id,
                    "role": role,
                    "gender": gender,
                    "text": text,
                })
    return sentences


def compute_perplexity_for_model(model, sentences):
    per_sentence = []
    for sent in sentences:
        ppl = compute_sentence_perplexity(model, sent["text"])
        per_sentence.append({
            "id": sent["id"],
            "role": sent["role"],
            "gender": sent["gender"],
            "ppl": ppl,
        })

    valid_ppls = [s["ppl"] for s in per_sentence if np.isfinite(s["ppl"])]
    return {
        "mean_ppl": float(np.mean(valid_ppls)) if valid_ppls else None,
        "median_ppl": float(np.median(valid_ppls)) if valid_ppls else None,
        "n_sentences": len(per_sentence),
        "n_valid": len(valid_ppls),
        "per_sentence": per_sentence,
    }


def _ppl_exists(path):
    try:
        existing = s3_utils.list_keys(path)
        return len(existing) > 0
    except Exception:
        return False


def discover_run_ids(log_dir):
    log_keys = s3_utils.list_keys(log_dir + "/")
    prefix = s3_utils.s3_key(log_dir + "/")
    return [
        k[len(prefix):].replace(".json", "")
        for k in log_keys
        if k.endswith(".json") and "all_experiment" not in k
    ]


def run_perplexity(
    model_key,
    run_ids=None,
    baseline_only=False,
    skip_existing=False,
):
    cfg = MODEL_CONFIGS[model_key]

    s3_client = boto3.client(
        "s3",
        aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
    )

    print(f"Loading model: {cfg['hf_name']} ...")
    model = HookedTransformer.from_pretrained(cfg["hf_name"])
    model.eval()
    original_state_dict = deepcopy(model.state_dict())

    print(f"Loading Winogender test dataset ({TEST_DATASET_PATH}) ...")
    dataset = s3_utils.read_json(TEST_DATASET_PATH)
    sentences = build_sentences(dataset)
    print(f"Built {len(sentences)} sentences from {len(dataset)} pairs.")

    baseline_ppl_path = f"{cfg['baseline_results']}/perplexity.json"
    if skip_existing and _ppl_exists(baseline_ppl_path):
        print(f"Baseline perplexity already exists. Skipping.")
    else:
        print("Computing baseline perplexity ...")
        result = compute_perplexity_for_model(model, sentences)
        s3_utils.write_json(result, baseline_ppl_path)
        print(f"  Saved: {baseline_ppl_path}")
        print(f"  Mean PPL: {result['mean_ppl']:.2f}, "
              f"Median PPL: {result['median_ppl']:.2f}")

    if baseline_only:
        print("Baseline-only mode. Done.")
        return

    if run_ids is None:
        run_ids = discover_run_ids(cfg["log_dir"])
        print(f"Discovered {len(run_ids)} run(s).")

    os.makedirs("checkpoints", exist_ok=True)

    for idx, run_id in enumerate(run_ids):
        print(f"\n{'=' * 60}")
        print(f"[{idx + 1}/{len(run_ids)}] {run_id}")
        print(f"{'=' * 60}")

        ppl_path = f"{cfg['results_base']}/{run_id}/perplexity.json"
        if skip_existing and _ppl_exists(ppl_path):
            print("  Perplexity already exists. Skipping.")
            continue

        try:
            log = s3_utils.read_json(f"{cfg['log_dir']}/{run_id}.json")
        except Exception as e:
            print(f"  Could not read log: {e}. Skipping.")
            continue

        best_epoch = log["best_epoch"] - 1
        checkpoint_key = (
            f"{cfg['checkpoint_prefix']}/best_model_{run_id}_epoch_{best_epoch}.pt"
        )
        local_tmp = f"checkpoints/{run_id}.pt"

        try:
            print(f"  Downloading checkpoint ...")
            s3_client.download_file(S3_BUCKET, checkpoint_key, local_tmp)
        except Exception as e:
            print(f"  Checkpoint download failed: {e}. Skipping.")
            continue

        model.load_state_dict(torch.load(local_tmp, weights_only=True))
        os.remove(local_tmp)

        print("  Computing perplexity ...")
        result = compute_perplexity_for_model(model, sentences)
        s3_utils.write_json(result, ppl_path)
        print(f"  Saved: {ppl_path}")
        print(f"  Mean PPL: {result['mean_ppl']:.2f}, "
              f"Median PPL: {result['median_ppl']:.2f}")

        model.load_state_dict(original_state_dict)

    print(f"\nAll {len(run_ids)} run(s) processed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute Winogender sentence perplexity for baseline and fine-tuned models")
    parser.add_argument(
        "--model", type=str, required=True,
        choices=list(MODEL_CONFIGS.keys()),
        help="Model key to evaluate.")
    parser.add_argument(
        "--run_id", type=str, default=None,
        help="Evaluate a single run ID instead of all discovered runs.")
    parser.add_argument(
        "--skip_existing", action="store_true",
        help="Skip runs that already have perplexity.json on S3.")
    parser.add_argument(
        "--baseline_only", action="store_true",
        help="Only compute baseline perplexity (no fine-tuned runs).")
    parser.add_argument(
        "--comparison", action="store_true",
        help="Evaluate comparison experiment checkpoints (different S3 paths, Llama only).")
    parser.add_argument(
        "--filter", choices=["all", "dla", "snr"], default="all",
        dest="filter_mode",
        help="Filter discovered runs: all = all runs; dla = DLA comparison only; snr = SNR only.")
    args = parser.parse_args()

    if args.comparison:
        if args.model != "llama3.2_1b":
            parser.error("--comparison is only supported for llama3.2_1b")
        MODEL_CONFIGS["llama3.2_1b"]["log_dir"] = (
            "outputs/llama3.2_1b/winogender/comparison/logs"
        )
        MODEL_CONFIGS["llama3.2_1b"]["checkpoint_prefix"] = (
            "experiments/outputs/llama3.2_1b/winogender/comparison/checkpoints"
        )
        MODEL_CONFIGS["llama3.2_1b"]["results_base"] = (
            "outputs/llama3.2_1b/winogender/comparison/test"
        )

    if args.run_id:
        run_ids = [args.run_id]
    elif not args.baseline_only and args.filter_mode != "all":
        cfg = MODEL_CONFIGS[args.model]
        all_ids = discover_run_ids(cfg["log_dir"])
        if args.filter_mode == "snr":
            run_ids = [r for r in all_ids if "snr" in r]
        elif args.filter_mode == "dla":
            run_ids = [r for r in all_ids if "snr" not in r]
        print(f"Filtered to {len(run_ids)} run(s) ({args.filter_mode})")
    else:
        run_ids = None

    run_perplexity(
        model_key=args.model,
        run_ids=run_ids,
        baseline_only=args.baseline_only,
        skip_existing=args.skip_existing,
    )
