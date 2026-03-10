import csv
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import s3_utils

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
    stats = {}
    with open(OCC_STATS_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            stats[row["occupation"]] = float(row["bls_pct_female"])
    return stats


def detect_pronoun_placeholder(sentence):
    for placeholder in PRONOUN_FORMS:
        if placeholder in sentence:
            return placeholder
    return None


def build_dataset():
    occ_stats = load_occupation_stats()
    dataset = []
    metadata = []

    with open(TEMPLATES_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            occupation = row["occupation(0)"]
            participant = row["other-participant(1)"]
            answer = int(row["answer"])
            sentence = row["sentence"]

            placeholder = detect_pronoun_placeholder(sentence)
            if placeholder is None:
                print(f"WARNING: no pronoun placeholder found in: {sentence}")
                continue

            parts = sentence.split(placeholder)
            prefix = parts[0]
            suffix = parts[1] if len(parts) > 1 else ""

            prefix = prefix.replace("$OCCUPATION", occupation)
            prefix = prefix.replace("$PARTICIPANT", participant)
            prefix = prefix.rstrip()

            suffix = suffix.replace("$OCCUPATION", occupation)
            suffix = suffix.replace("$PARTICIPANT", participant)
            suffix = suffix.strip()

            pct_female = occ_stats.get(occupation, 50.0)
            forms = PRONOUN_FORMS[placeholder]

            if pct_female > 50:
                stereo_pronoun = forms["female"]
                anti_pronoun = forms["male"]
            else:
                stereo_pronoun = forms["male"]
                anti_pronoun = forms["female"]
            neutral_pronoun = forms["neutral"]

            targets = {
                "stereotype": stereo_pronoun,
                "anti-stereotype": anti_pronoun,
                "unrelated": neutral_pronoun,
            }
            full_sentences = {
                label: f"{prefix} {pronoun} {suffix}" if suffix else f"{prefix} {pronoun}"
                for label, pronoun in targets.items()
            }

            example_id = f"{occupation}_{participant}_{answer}"

            example = {
                "id": example_id,
                "prefix": prefix,
                "suffix": suffix,
                "full_sentences": full_sentences,
                "targets": targets,
                "occupation": occupation,
                "participant": participant,
                "answer": answer,
                "bls_pct_female": pct_female,
                "pronoun_type": PRONOUN_TYPE_LABELS[placeholder],
            }
            dataset.append(example)

            metadata.append({
                "id": example_id,
                "occupation": occupation,
                "participant": participant,
                "answer": answer,
                "bls_pct_female": pct_female,
                "pronoun_type": PRONOUN_TYPE_LABELS[placeholder],
                "suffix": suffix,
            })

    return dataset, metadata


if __name__ == "__main__":
    dataset, metadata = build_dataset()

    print(f"Built {len(dataset)} examples from templates.")
    print(f"  answer=0 (pronoun=occupation): {sum(1 for d in dataset if d['answer'] == 0)}")
    print(f"  answer=1 (pronoun=participant): {sum(1 for d in dataset if d['answer'] == 1)}")

    s3_utils.write_json(dataset, "data/winogender/winogender_dataset.json")
    print("Saved dataset to S3: data/winogender/winogender_dataset.json")

    s3_utils.write_json(metadata, "data/winogender/winogender_metadata.json")
    print("Saved metadata to S3: data/winogender/winogender_metadata.json")

    print("\nSample entries:")
    for ex in dataset[:4]:
        print(f"  [{ex['id']}] answer={ex['answer']} "
              f"occ={ex['occupation']} ({ex['bls_pct_female']}% female)")
        print(f"    prefix:  \"{ex['prefix']}\"")
        print(f"    suffix:  \"{ex['suffix']}\"")
        print(f"    targets: {ex['targets']}")
        for label, sent in ex['full_sentences'].items():
            print(f"    {label:20s}: {sent}")
