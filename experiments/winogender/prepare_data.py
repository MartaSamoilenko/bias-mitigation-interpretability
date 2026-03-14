import csv
import os
from collections import defaultdict

from experiments import s3_utils

TEMPLATES_PATH = os.path.join(os.path.dirname(__file__),
                              "winogender-schemas", "data", "templates.tsv")
OCC_STATS_PATH = os.path.join(os.path.dirname(__file__),
                              "winogender-schemas", "data", "occupations-stats.tsv")

PRONOUN_FORMS = {
    "$NOM_PRONOUN": {"male": "he",  "female": "she", "neutral": "they"},
    "$POSS_PRONOUN": {"male": "his", "female": "her", "neutral": "their"},
    "$ACC_PRONOUN": {"male": "him", "female": "her", "neutral": "them"},
}

PRONOUN_TYPE_LABELS = {
    "$NOM_PRONOUN": "nominative",
    "$POSS_PRONOUN": "possessive",
    "$ACC_PRONOUN": "accusative",
}


def load_occupation_stats():
    """Return dict mapping occupation -> {bls_pct_female, bergsma_pct_female}."""
    stats = {}
    with open(OCC_STATS_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            stats[row["occupation"]] = {
                "bls_pct_female": float(row["bls_pct_female"]),
                "bergsma_pct_female": float(row["bergsma_pct_female"]),
            }
    return stats


def detect_pronoun_placeholder(sentence):
    for placeholder in PRONOUN_FORMS:
        if placeholder in sentence:
            return placeholder
    return None


def _parse_template(row):
    """Parse a single TSV row into a sentence dict."""
    occupation = row["occupation(0)"]
    participant = row["other-participant(1)"]
    answer = int(row["answer"])
    sentence = row["sentence"]

    placeholder = detect_pronoun_placeholder(sentence)
    if placeholder is None:
        return None

    parts = sentence.split(placeholder)
    prefix = parts[0]
    suffix = parts[1] if len(parts) > 1 else ""

    prefix = prefix.replace("$OCCUPATION", occupation)
    prefix = prefix.replace("$PARTICIPANT", participant)
    prefix = prefix.rstrip()

    suffix = suffix.replace("$OCCUPATION", occupation)
    suffix = suffix.replace("$PARTICIPANT", participant)
    suffix = suffix.strip()

    return {
        "occupation": occupation,
        "participant": participant,
        "answer": answer,
        "prefix": prefix,
        "suffix": suffix,
        "placeholder": placeholder,
        "pronoun_type": PRONOUN_TYPE_LABELS[placeholder],
        "pronouns": PRONOUN_FORMS[placeholder],
    }


def build_dataset():
    occ_stats = load_occupation_stats()

    raw = defaultdict(dict)
    with open(TEMPLATES_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            parsed = _parse_template(row)
            if parsed is None:
                print(f"WARNING: no pronoun placeholder in: {row['sentence']}")
                continue
            key = (parsed["occupation"], parsed["participant"])
            raw[key][parsed["answer"]] = parsed

    dataset = []
    metadata = []

    for (occupation, participant), by_answer in sorted(raw.items()):
        if 0 not in by_answer or 1 not in by_answer:
            print(f"WARNING: incomplete pair for {occupation}/{participant}, skipping")
            continue

        occ_sent = by_answer[0]
        part_sent = by_answer[1]

        stats = occ_stats.get(occupation, {})
        bls_pct = stats.get("bls_pct_female", 50.0)
        bergsma_pct = stats.get("bergsma_pct_female", 50.0)

        same_prefix = (occ_sent["prefix"] == part_sent["prefix"])
        same_pronoun_type = (occ_sent["placeholder"] == part_sent["placeholder"])

        # Pronoun forms come from the occupation sentence (DLA target)
        pronouns = occ_sent["pronouns"]

        pair_id = f"{occupation}_{participant}"

        record = {
            "id": pair_id,
            "occupation": occupation,
            "participant": participant,
            "bls_pct_female": bls_pct,
            "bergsma_pct_female": bergsma_pct,
            "sentence_occ": {
                "prefix": occ_sent["prefix"],
                "suffix": occ_sent["suffix"],
                "pronoun_type": occ_sent["pronoun_type"],
            },
            "sentence_part": {
                "prefix": part_sent["prefix"],
                "suffix": part_sent["suffix"],
                "pronoun_type": part_sent["pronoun_type"],
            },
            "pronouns": pronouns,
            "same_prefix": same_prefix,
            "same_pronoun_type": same_pronoun_type,
        }
        dataset.append(record)

        metadata.append({
            "id": pair_id,
            "occupation": occupation,
            "participant": participant,
            "bls_pct_female": bls_pct,
            "bergsma_pct_female": bergsma_pct,
            "occ_pronoun_type": occ_sent["pronoun_type"],
            "part_pronoun_type": part_sent["pronoun_type"],
            "same_prefix": same_prefix,
            "same_pronoun_type": same_pronoun_type,
        })

    return dataset, metadata


if __name__ == "__main__":
    dataset, metadata = build_dataset()

    n_same_prefix = sum(1 for d in dataset if d["same_prefix"])
    n_same_ptype = sum(1 for d in dataset if d["same_pronoun_type"])

    print(f"Built {len(dataset)} paired examples from templates.")
    print(f"  same_prefix:       {n_same_prefix}/{len(dataset)}")
    print(f"  same_pronoun_type: {n_same_ptype}/{len(dataset)}")

    s3_utils.write_json(dataset, "data/winogender/winogender_paired_dataset.json")
    print("Saved dataset to S3: data/winogender/winogender_paired_dataset.json")

    s3_utils.write_json(metadata, "data/winogender/winogender_paired_metadata.json")
    print("Saved metadata to S3: data/winogender/winogender_paired_metadata.json")

    print("\nSample entries:")
    for ex in dataset[:3]:
        occ_s = ex["sentence_occ"]
        part_s = ex["sentence_part"]
        print(f"\n--- {ex['occupation'].upper()} / {ex['participant']} "
              f"(BLS {ex['bls_pct_female']}% female) ---")
        print(f"  same_prefix={ex['same_prefix']}  "
              f"same_pronoun_type={ex['same_pronoun_type']}")
        print(f"  Pronouns: {ex['pronouns']}")
        print(f"  Occ sentence ({occ_s['pronoun_type']}):")
        print(f"    prefix:  \"{occ_s['prefix']}\"")
        print(f"    suffix:  \"{occ_s['suffix']}\"")
        print(f"  Part sentence ({part_s['pronoun_type']}):")
        print(f"    prefix:  \"{part_s['prefix']}\"")
        print(f"    suffix:  \"{part_s['suffix']}\"")
