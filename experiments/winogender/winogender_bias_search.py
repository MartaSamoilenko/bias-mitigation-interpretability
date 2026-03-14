"""
DLA tracing and accumulated impact analysis for the Winogender dataset.

Uses paired templates (occupation sentence + participant sentence per pair).
Stage 1: pronoun preference + DLA on both sentences.
Stage 2: suffix coreference — compare P(suffix_occ) vs P(suffix_part) using
the preferred pronoun from the occupation sentence.
"""
import argparse
import os

import boto3
import pandas as pd
import torch
from dotenv import load_dotenv
from transformer_lens import HookedTransformer

from experiments import s3_utils

load_dotenv()

TRACING = True
ACC_ANALYSIS = True


def _pronoun_probs_stage1(model, prefix, pronoun_words, pair_id, sentence_role):
    """Run a single Stage-1 forward pass on *prefix*.

    Returns a list of row dicts (one per pronoun × token × layer) with DLA.
    """
    prefix_tokens = model.tokenizer.encode(prefix)
    cache_pos = len(prefix_tokens) - 1

    with torch.no_grad():
        _, cache = model.run_with_cache(prefix)

    rows = []
    for gender_key, pronoun_word in pronoun_words.items():
        token_ids = model.tokenizer.encode(" " + pronoun_word)

        for tok_offset, target_id in enumerate(token_ids):
            layer_acc_prob = torch.ones(model.cfg.n_layers, device="cpu")

            for layer in range(model.cfg.n_layers):
                hidden = cache[f"blocks.{layer}.hook_resid_post"][0, cache_pos]
                logits = model.unembed(model.ln_final(hidden))
                probs = torch.softmax(logits, dim=-1)

                p_tok = probs[target_id].item()
                layer_acc_prob[layer] *= p_tok

                unembed_dir = model.W_U[:, target_id]

                attn_z = cache[f"blocks.{layer}.attn.hook_z"][0, cache_pos]
                head_contribs = torch.einsum(
                    "hd, hdm, m -> h", attn_z, model.W_O[layer], unembed_dir
                )

                mlp_out = cache[f"blocks.{layer}.hook_mlp_out"][0, cache_pos]
                mlp_contrib = torch.dot(mlp_out, unembed_dir)

                row = {
                    "ID": pair_id,
                    "Sentence_Role": sentence_role,
                    "Prompt": prefix,
                    "Gender": gender_key,
                    "Candidate": pronoun_word,
                    "Token_Str": model.to_string(target_id),
                    "Token_Position": tok_offset,
                    "Is_First_Token": (tok_offset == 0),
                    "Layer": layer,
                    "Layer_Accumulated_Prob": layer_acc_prob[layer].item(),
                    "Token_Instant_Prob": p_tok,
                    "MLP_Logit_Impact": mlp_contrib.item(),
                }
                for h_idx, score in enumerate(head_contribs):
                    row[f"Head_{h_idx}"] = score.item()

                rows.append(row)

    del cache
    return rows


def _suffix_log_prob(model, full_text, suffix_start_idx):
    """Compute log P(suffix tokens | prefix+pronoun) from a full sentence."""
    tokens = model.tokenizer.encode(full_text)
    n_layers = model.cfg.n_layers

    with torch.no_grad():
        _, cache = model.run_with_cache(full_text)

    log_prob = 0.0
    for t_pos in range(suffix_start_idx, len(tokens)):
        hidden = cache[f"blocks.{n_layers - 1}.hook_resid_post"][0, t_pos - 1]
        logits_t = model.unembed(model.ln_final(hidden))
        probs_t = torch.softmax(logits_t, dim=-1)
        log_prob += torch.log(probs_t[tokens[t_pos]] + 1e-30).item()

    del cache
    return log_prob


def paired_tracing(model, dataset, pronoun_probs_path, suffix_probs_path):
    """Two-stage Winogender bias measurement on paired templates."""

    done_ids = set()
    try:
        df = s3_utils.read_csv(pronoun_probs_path)
        done_ids = set(df["ID"].unique().tolist())
    except Exception:
        pass

    stage1_rows = []
    stage2_rows = []

    for idx, pair in enumerate(dataset):
        pair_id = pair["id"]
        if pair_id in done_ids:
            continue

        print(f"[{idx}/{len(dataset)}] Processing {pair_id} ...")
        if stage1_rows and idx % 10 == 0:
            print("  Saving intermediate results ...")
            s3_utils.write_csv(pd.DataFrame(stage1_rows), pronoun_probs_path)
            s3_utils.write_csv(pd.DataFrame(stage2_rows), suffix_probs_path)

        pronouns = pair["pronouns"]
        occ = pair["sentence_occ"]
        part = pair["sentence_part"]

        # ── Stage 1: pronoun preference + DLA on both sentences ──────────
        stage1_rows.extend(
            _pronoun_probs_stage1(model, occ["prefix"], pronouns,
                                  pair_id, "occupation")
        )
        stage1_rows.extend(
            _pronoun_probs_stage1(model, part["prefix"], pronouns,
                                  pair_id, "participant")
        )

        # ── Stage 2: suffix coreference ──────────────────────────────────
        # Determine which pronoun the model prefers in the occupation context
        # (last-layer accumulated prob for the first token of each pronoun).
        occ_prefix_len = len(model.tokenizer.encode(occ["prefix"]))
        best_gender, best_pronoun, best_prob = None, None, -1.0

        for gender_key, pronoun_word in pronouns.items():
            tok_ids = model.tokenizer.encode(" " + pronoun_word)
            first_tok_id = tok_ids[0]

            hidden_key = f"blocks.{model.cfg.n_layers - 1}.hook_resid_post"
            with torch.no_grad():
                _, tmp_cache = model.run_with_cache(occ["prefix"])
            hidden = tmp_cache[hidden_key][0, occ_prefix_len - 1]
            logits = model.unembed(model.ln_final(hidden))
            p = torch.softmax(logits, dim=-1)[first_tok_id].item()
            del tmp_cache

            if p > best_prob:
                best_prob = p
                best_gender = gender_key
                best_pronoun = pronoun_word

        # Compute suffix log-prob for occupation and participant suffixes
        # using the preferred pronoun.
        for role, sent in [("occupation", occ), ("participant", part)]:
            prefix_text = sent["prefix"]
            suffix_text = sent["suffix"]
            if not suffix_text:
                continue

            full_text = f"{prefix_text} {best_pronoun} {suffix_text}"
            prefix_tok_len = len(model.tokenizer.encode(prefix_text))
            pronoun_tok_len = len(model.tokenizer.encode(" " + best_pronoun))
            suffix_start = prefix_tok_len + pronoun_tok_len

            slp = _suffix_log_prob(model, full_text, suffix_start)

            stage2_rows.append({
                "ID": pair_id,
                "Preferred_Gender": best_gender,
                "Preferred_Pronoun": best_pronoun,
                "Suffix_Role": role,
                "Suffix_Log_Prob": slp,
            })

    if stage1_rows:
        s3_utils.write_csv(pd.DataFrame(stage1_rows), pronoun_probs_path)
    if stage2_rows:
        s3_utils.write_csv(pd.DataFrame(stage2_rows), suffix_probs_path)

    return stage1_rows, stage2_rows


def accumulative_layer_impact(filename):
    """Compute accumulated DLA impact from Stage-1 pronoun probabilities.

    Only processes occupation-sentence rows (Sentence_Role == 'occupation').
    """
    print("Loading CSV from S3 ...")
    df = s3_utils.read_csv(filename)

    df = df[df["Sentence_Role"] == "occupation"].copy()

    df = df.sort_values(by=["ID", "Gender", "Layer", "Token_Position"])

    print("Calculating prefix probabilities ...")
    group_cols = ["ID", "Gender", "Layer"]
    df["Prefix_Prob"] = df.groupby(group_cols)["Layer_Accumulated_Prob"].shift(1)
    df["Prefix_Prob"] = df["Prefix_Prob"].fillna(1.0)

    print("Calculating weighted impacts ...")
    df["Weighted_MLP"] = df["MLP_Logit_Impact"] * df["Prefix_Prob"]

    head_cols = [c for c in df.columns if c.startswith("Head_")]
    weighted_head_cols = []
    for col in head_cols:
        w_col = f"Weighted_{col}"
        df[w_col] = df[col] * df["Prefix_Prob"]
        weighted_head_cols.append(w_col)

    print("Aggregating results ...")
    agg_dict = {"Weighted_MLP": "sum"}
    for col in weighted_head_cols:
        agg_dict[col] = "sum"

    final_df = df.groupby(group_cols).agg(agg_dict).reset_index()

    print("Formatting final table ...")
    mlp_data = final_df.melt(
        id_vars=group_cols, value_vars=["Weighted_MLP"],
        var_name="Component", value_name="Accumulated_Impact",
    )
    mlp_data["Component"] = "MLP"

    head_data = final_df.melt(
        id_vars=group_cols, value_vars=weighted_head_cols,
        var_name="Component", value_name="Accumulated_Impact",
    )
    head_data["Component"] = head_data["Component"].str.replace("Weighted_", "")

    return pd.concat([mlp_data, head_data], ignore_index=True)


TRAIN_DATASET_PATH = "data/winogender/winogender_paired_dataset.json"
TEST_DATASET_PATH = "data/winogender/winogender_test_dataset.json"

BASELINE_TRAIN_DIR = "outputs/gpt2-xl/winogender/baseline/train"
BASELINE_TEST_DIR = "outputs/gpt2-xl/winogender/baseline/test"
FT_TRAIN_DIR = "outputs/gpt2-xl/winogender/fine_tuned/train"
FT_TEST_DIR = "outputs/gpt2-xl/winogender/fine_tuned/test"
FT_LOGS_DIR = "outputs/gpt2-xl/winogender/fine_tuned/logs"
FT_CHECKPOINTS_PREFIX = "experiments/outputs/gpt2-xl/winogender/fine_tuned/checkpoints"

_SPLIT_DIRS = {"train": BASELINE_TRAIN_DIR, "test": BASELINE_TEST_DIR}
_SPLIT_DATASETS = {"train": TRAIN_DATASET_PATH, "test": TEST_DATASET_PATH}
_FT_SPLIT_DIRS = {"train": FT_TRAIN_DIR, "test": FT_TEST_DIR}


def run_baseline(split="train", dataset_path=None):
    if dataset_path is None:
        dataset_path = _SPLIT_DATASETS[split]

    base_dir = _SPLIT_DIRS[split]
    pronoun_path = f"{base_dir}/pronoun_probs.csv"
    suffix_path = f"{base_dir}/suffix_probs.csv"
    acc_path = f"{base_dir}/accumulated_impact.csv"

    print("Loading GPT-2 XL baseline ...")
    model = HookedTransformer.from_pretrained("gpt2-xl")
    model.eval()

    print(f"Loading Winogender dataset from S3 ({dataset_path}) ...")
    dataset = s3_utils.read_json(dataset_path)
    print(f"Loaded {len(dataset)} pairs.")

    if TRACING:
        print("Starting paired tracing ...")
        paired_tracing(model, dataset, pronoun_path, suffix_path)
        print("Tracing complete.")

    if ACC_ANALYSIS:
        print("Computing accumulated impact (occupation sentences only) ...")
        result_df = accumulative_layer_impact(pronoun_path)
        s3_utils.write_csv(result_df, acc_path)
        print(f"Saved accumulated impact to S3: {acc_path}")


def run_finetuned(run_id, split="train", dataset_path=None):
    if dataset_path is None:
        dataset_path = _SPLIT_DATASETS[split]

    s3_bucket = "modelsfinetuned"

    s3_client = boto3.client(
        "s3",
        aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
    )

    print(f"Loading GPT-2 XL and applying checkpoint for {run_id} ...")
    model = HookedTransformer.from_pretrained("gpt2-xl")
    model.eval()

    log = s3_utils.read_json(f"{FT_LOGS_DIR}/{run_id}.json")
    best_epoch = log["best_epoch"] - 1

    checkpoint_key = f"{FT_CHECKPOINTS_PREFIX}/best_model_{run_id}_epoch_{best_epoch}.pt"
    local_tmp = f"checkpoints/{run_id}.pt"
    os.makedirs("checkpoints", exist_ok=True)

    print(f"Downloading checkpoint s3://{s3_bucket}/{checkpoint_key} ...")
    s3_client.download_file(s3_bucket, checkpoint_key, local_tmp)
    model.load_state_dict(torch.load(local_tmp, weights_only=True))
    os.remove(local_tmp)
    print("Checkpoint loaded.")

    print(f"Loading Winogender dataset from S3 ({dataset_path}) ...")
    dataset = s3_utils.read_json(dataset_path)
    print(f"Loaded {len(dataset)} pairs.")

    ft_base = f"{_FT_SPLIT_DIRS[split]}/{run_id}"
    ft_pronoun_path = f"{ft_base}/pronoun_probs.csv"
    ft_suffix_path = f"{ft_base}/suffix_probs.csv"
    ft_acc_path = f"{ft_base}/accumulated_impact.csv"

    if TRACING:
        print("Starting paired tracing on fine-tuned model ...")
        paired_tracing(model, dataset, ft_pronoun_path, ft_suffix_path)
        print("Tracing complete.")

    if ACC_ANALYSIS:
        print("Computing accumulated impact for fine-tuned model ...")
        result_df = accumulative_layer_impact(ft_pronoun_path)
        s3_utils.write_csv(result_df, ft_acc_path)
        print(f"Saved accumulated impact to S3: {ft_acc_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Winogender DLA tracing")
    parser.add_argument("--run_id", type=str, default=None,
                        help="Fine-tuned run ID. Omit for baseline evaluation.")
    parser.add_argument("--split", type=str, default="train",
                        choices=["train", "test"],
                        help="Dataset split to use (default: train).")
    parser.add_argument("--dataset_path", type=str, default=None,
                        help="Custom dataset path (overrides --split default).")
    args = parser.parse_args()

    if args.run_id:
        run_finetuned(args.run_id, split=args.split, dataset_path=args.dataset_path)
    else:
        run_baseline(split=args.split, dataset_path=args.dataset_path)
