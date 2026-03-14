"""Generate Winogender fine-tuning datasets (DPO + SFT) from baseline model
pronoun probabilities.

Reads the baseline Stage-1 output (pronoun_probs.csv) to determine which
pronoun the model actually prefers per occupation, then constructs training
pairs that counteract the measured bias direction.
"""
import argparse

from dotenv import load_dotenv

import s3_utils

load_dotenv()

PRONOUN_PROBS_PATH = "outputs/gpt2-xl/winogender/pronoun_probs.csv"
DATASET_PATH = "data/winogender/winogender_paired_dataset.json"

DPO_OUTPUT_PATH = "data/winogender/fine-tune-dpo/winogender_dpo.jsonl"
SFT_OUTPUT_PATH = "data/winogender/fine-tune-sft/winogender_sft.jsonl"

N_LAYERS = 48
LAST_LAYER = N_LAYERS - 1


def load_baseline_probs():
    """Load baseline pronoun probabilities and return per-pair male/female P.

    Returns dict {pair_id: {"p_male": float, "p_female": float}}.
    """
    df = s3_utils.read_csv(PRONOUN_PROBS_PATH)

    df = df[
        (df["Sentence_Role"] == "occupation")
        & (df["Layer"] == LAST_LAYER)
        & (df["Is_First_Token"] == True)  # noqa: E712
    ]

    probs = {}
    for pair_id, group in df.groupby("ID"):
        p_male = group.loc[group["Gender"] == "male", "Token_Instant_Prob"]
        p_female = group.loc[group["Gender"] == "female", "Token_Instant_Prob"]
        probs[pair_id] = {
            "p_male": p_male.values[0] if len(p_male) else 0.0,
            "p_female": p_female.values[0] if len(p_female) else 0.0,
        }

    return probs


def generate(min_bias: float = 0.0):
    print("Loading baseline pronoun probabilities from S3 ...")
    baseline_probs = load_baseline_probs()
    print(f"  Loaded probabilities for {len(baseline_probs)} pairs.")

    print("Loading paired Winogender dataset from S3 ...")
    dataset = s3_utils.read_json(DATASET_PATH)
    print(f"  Loaded {len(dataset)} pairs.")

    dpo_pairs = []
    sft_examples = []
    n_skipped = 0
    n_male_biased = 0
    n_female_biased = 0
    magnitudes = []

    for pair in dataset:
        pair_id = pair["id"]
        probs = baseline_probs.get(pair_id)
        if probs is None:
            print(f"  WARNING: no baseline probs for {pair_id}, skipping")
            n_skipped += 1
            continue

        p_male = probs["p_male"]
        p_female = probs["p_female"]
        bias_mag = abs(p_male - p_female)

        if bias_mag < min_bias:
            n_skipped += 1
            continue

        magnitudes.append(bias_mag)
        pronouns = pair["pronouns"]

        if p_male > p_female:
            stereo_pronoun = pronouns["male"]
            anti_stereo_pronoun = pronouns["female"]
            n_male_biased += 1
        else:
            stereo_pronoun = pronouns["female"]
            anti_stereo_pronoun = pronouns["male"]
            n_female_biased += 1

        occ = pair["sentence_occ"]
        prompt = occ["prefix"]
        suffix = occ["suffix"]

        chosen = f" {anti_stereo_pronoun} {suffix}"
        rejected = f" {stereo_pronoun} {suffix}"

        dpo_pairs.append({
            "id": pair_id,
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected,
            "pair_type": "debias",
        })

        sft_examples.append({
            "id": pair_id,
            "prompt": prompt,
            "completion": chosen,
            "stereotype_completion": rejected,
        })

    print(f"\nSaving DPO data ({len(dpo_pairs)} pairs) -> {DPO_OUTPUT_PATH}")
    s3_utils.write_jsonl(dpo_pairs, DPO_OUTPUT_PATH)

    print(f"Saving SFT data ({len(sft_examples)} examples) -> {SFT_OUTPUT_PATH}")
    s3_utils.write_jsonl(sft_examples, SFT_OUTPUT_PATH)

    print("\n--- Summary ---")
    print(f"  Total pairs in dataset:   {len(dataset)}")
    print(f"  Pairs used for training:  {len(dpo_pairs)}")
    print(f"  Pairs skipped:            {n_skipped}")
    print(f"    (min_bias threshold:    {min_bias})")
    print(f"  Male-biased pairs:        {n_male_biased}")
    print(f"  Female-biased pairs:      {n_female_biased}")
    if magnitudes:
        mean_mag = sum(magnitudes) / len(magnitudes)
        max_mag = max(magnitudes)
        print(f"  Mean bias magnitude:      {mean_mag:.4f}")
        print(f"  Max  bias magnitude:      {max_mag:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate Winogender DPO/SFT fine-tuning data from "
                    "baseline model probabilities."
    )
    parser.add_argument(
        "--min_bias", type=float, default=0.0,
        help="Minimum |P(male) - P(female)| to include a pair (default: 0.0)",
    )
    args = parser.parse_args()
    generate(min_bias=args.min_bias)
