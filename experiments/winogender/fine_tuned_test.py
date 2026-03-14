"""
Usage:
    python -m experiments.winogender.fine_tuned_test
    python -m experiments.winogender.fine_tuned_test --run_id wino_dpo_attn_0.5_beta0.3_lr1e-05
    python -m experiments.winogender.fine_tuned_test --skip_existing
    python -m experiments.winogender.fine_tuned_test --dataset_path data/winogender/winogender_paired_dataset.json
"""
import argparse
import os
from copy import deepcopy

import boto3
import torch
from dotenv import load_dotenv
from transformer_lens import HookedTransformer

from experiments import s3_utils
from experiments.winogender.winogender_bias_search import (
    paired_tracing,
    accumulative_layer_impact,
)

load_dotenv()

LOGS_DIR = "outputs/gpt2-xl/winogender/fine_tuned/logs"
S3_BUCKET = "modelsfinetuned"
S3_PREFIX = "experiments/outputs/gpt2-xl/winogender/fine_tuned/checkpoints"
TEST_DATASET_PATH = "data/winogender/winogender_test_dataset.json"
RESULTS_BASE = "outputs/gpt2-xl/winogender/fine_tuned/test"


def discover_run_ids():
    """List all Winogender fine-tuned run IDs from S3 log files."""
    log_keys = s3_utils.list_keys(LOGS_DIR + "/")
    prefix = s3_utils.s3_key(LOGS_DIR + "/")
    run_ids = [
        k[len(prefix):].replace(".json", "")
        for k in log_keys
        if k.endswith(".json") and "all_experiment" not in k
    ]
    return run_ids


def _results_exist(run_id):
    """Check whether pronoun_probs.csv already exists for this run."""
    check_path = f"{RESULTS_BASE}/{run_id}/pronoun_probs.csv"
    try:
        existing_keys = s3_utils.list_keys(check_path)
        return len(existing_keys) > 0
    except Exception:
        return False


def run_experiments_finetuned_winogender(
    run_ids,
    dataset_path=None,
    skip_existing=False,
):
    """Evaluate multiple fine-tuned Winogender models on the test dataset.

    Loads GPT-2 XL once and swaps state dicts between runs to avoid
    reloading the full model for each checkpoint.
    """
    if dataset_path is None:
        dataset_path = TEST_DATASET_PATH

    s3_client = boto3.client(
        "s3",
        aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
    )

    print("Loading GPT-2 XL ...")
    model = HookedTransformer.from_pretrained("gpt2-xl")
    model.eval()
    original_state_dict = deepcopy(model.state_dict())

    print(f"Loading Winogender test dataset from S3 ({dataset_path}) ...")
    dataset = s3_utils.read_json(dataset_path)
    print(f"Loaded {len(dataset)} pairs.")

    os.makedirs("checkpoints", exist_ok=True)

    for idx, run_id in enumerate(run_ids):
        print(f"\n{'=' * 60}")
        print(f"[{idx + 1}/{len(run_ids)}] Evaluating: {run_id}")
        print(f"{'=' * 60}")

        if skip_existing and _results_exist(run_id):
            print(f"  Results already exist. Skipping.")
            continue

        try:
            log = s3_utils.read_json(f"{LOGS_DIR}/{run_id}.json")
        except Exception as e:
            print(f"  Could not read log file: {e}. Skipping.")
            continue

        best_epoch = log["best_epoch"] - 1
        checkpoint_key = f"{S3_PREFIX}/best_model_{run_id}_epoch_{best_epoch}.pt"
        local_tmp = f"checkpoints/{run_id}.pt"

        try:
            print(f"  Downloading s3://{S3_BUCKET}/{checkpoint_key} ...")
            s3_client.download_file(S3_BUCKET, checkpoint_key, local_tmp)
        except Exception as e:
            print(f"  Checkpoint download failed: {e}. Skipping.")
            continue

        model.load_state_dict(torch.load(local_tmp, weights_only=True))
        os.remove(local_tmp)
        print("  Checkpoint loaded.")

        ft_base = f"{RESULTS_BASE}/{run_id}"
        ft_pronoun_path = f"{ft_base}/pronoun_probs.csv"
        ft_suffix_path = f"{ft_base}/suffix_probs.csv"
        ft_acc_path = f"{ft_base}/accumulated_impact.csv"

        print("  Running paired tracing ...")
        paired_tracing(model, dataset, ft_pronoun_path, ft_suffix_path)
        print("  Tracing complete.")

        print("  Computing accumulated impact ...")
        try:
            result_df = accumulative_layer_impact(ft_pronoun_path)
            s3_utils.write_csv(result_df, ft_acc_path)
            print(f"  Saved: {ft_acc_path}")
        except Exception as e:
            print(f"  Accumulated impact failed: {e}")

        print("  Resetting model to baseline weights ...")
        model.load_state_dict(original_state_dict)

        print(f"  Done with {run_id}!")

    print(f"\nAll {len(run_ids)} run(s) processed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Batch evaluation of Winogender fine-tuned models")
    parser.add_argument(
        "--run_id", type=str, default=None,
        help="Evaluate a single run ID instead of all discovered runs.")
    parser.add_argument(
        "--dataset_path", type=str, default=None,
        help="Override test dataset path (default: winogender_test_dataset.json).")
    parser.add_argument(
        "--skip_existing", action="store_true",
        help="Skip runs that already have results on S3.")
    args = parser.parse_args()

    if args.run_id:
        ids = [args.run_id]
        print(f"Single-run mode: {args.run_id}")
    else:
        ids = discover_run_ids()
        print(f"Discovered {len(ids)} run(s): {ids}")

    if not ids:
        print("No runs to evaluate. Exiting.")
    else:
        run_experiments_finetuned_winogender(
            ids,
            dataset_path=args.dataset_path,
            skip_existing=args.skip_existing,
        )
