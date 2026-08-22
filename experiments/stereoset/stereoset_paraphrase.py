import os
import re
import json

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

import s3_utils

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

def generate_rephrased_blank_sentence(sentence):
    if not client:
        return f"[MOCK REPHRASE] {sentence} (BLANK moved to end)"

    prompt = f"""
    Task: Rephrase the provided sentence so that the word "BLANK" is the very last word.
    Constraint: The meaning must remain as close to the original as possible.

    Examples:
    Original: BLANK mom was there.
    Rephrased: Mom was there and looked BLANK.

    Original: He drove a BLANK car to work.
    Rephrased: The car he drove to work was BLANK.

    Original: {sentence}
    Rephrased:
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": "You are a linguistic assistant. Rephrase sentences to end with 'BLANK'."
                },
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=100
        )
        return response.choices[0].message.content.strip()

    except Exception as e:
        print(f"Error generating sentence: {e}")
        return sentence


def extract_target_word(context, full_sentence):
    pattern_str = re.escape(context)
    pattern_str = pattern_str.replace("BLANK", r"(.*)")
    pattern_str = f"^{pattern_str}$"

    match = re.search(pattern_str, full_sentence, re.IGNORECASE)

    if match:
        word = match.group(1).strip()
        if not context.endswith("BLANK") and context[-1] in ".,?!;:":
             pass
        else:
             word = word.rstrip(".,?!;:")
        return word

    parts = context.split("BLANK")
    prefix = parts[0]
    suffix = parts[1] if len(parts) > 1 else ""

    lower_sentence = full_sentence.lower()
    lower_prefix = prefix.lower()
    lower_suffix = suffix.lower()

    start_index = 0
    end_index = len(full_sentence)

    if lower_sentence.startswith(lower_prefix.strip()):
         start_index = len(prefix)
         if prefix.endswith(' ') and not full_sentence[start_index-1].isspace():
             start_index -= 1

    if suffix and lower_sentence.endswith(lower_suffix.strip()):
         end_index = -len(suffix)
         if suffix.startswith(' ') and not full_sentence[end_index].isspace():
             end_index += 1

    result = full_sentence[start_index:end_index].strip()
    return result.rstrip(".,?!;:")

def process_stereoset_chapter(chapter_data):
    original_context = chapter_data['context']

    clean_context = original_context.rstrip(".,?!;:")

    rephrased_context = original_context
    if not clean_context.endswith("BLANK"):
        print(f"Rephrasing needed for: '{original_context}'")
        rephrased_context = generate_rephrased_blank_sentence(original_context)

    targets = {}
    desired_labels = ['stereotype', 'anti-stereotype', 'unrelated']

    for item in chapter_data['sentences']:
        label = item['gold_label']
        if label in desired_labels:
            sentence_text = item['sentence']
            extracted_word = extract_target_word(original_context, sentence_text)
            targets[label] = extracted_word

    return {
        'id': chapter_data['id'],
        'bias_type': chapter_data['bias_type'],
        'original_context': original_context,
        'rephrased_context': rephrased_context,
        'targets': targets
    }


def rephrase_stereoset(input_path: str, output_path: str, bias_type: str = "gender"):
    print(f"Loading raw StereoSet from S3: {input_path}")
    raw_data = s3_utils.read_json(input_path)
    items = raw_data.get('data', {}).get('intrasentence', [])
    print(f"  Total intrasentence items: {len(items)}")

    if bias_type:
        items = [it for it in items if it.get('bias_type') == bias_type]
        print(f"  After filtering for bias_type='{bias_type}': {len(items)}")

    results = []
    for i, item in enumerate(items):
        results.append(process_stereoset_chapter(item))
        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{len(items)}")

    s3_utils.write_json(results, output_path)
    print(f"Saved {len(results)} rephrased items -> {output_path}")
    return results

def generate_dpo_triplet_dataset(input_path: str, output_path: str):
    print(f"Reading rephrased data from S3 ({input_path})...")

    # raw_data = s3_utils.read_json(input_path)
    with open(input_path, "r") as f:
        raw_data = json.load(f)

    dpo_pairs = []
    n_debias = 0
    n_lms = 0

    for item in raw_data:
        item_id = item.get("id")
        context = item.get("rephrased_context", item.get("original_context", ""))
        targets = item.get("targets", {})

        if not item_id or "BLANK" not in context or not targets:
            continue

        anti_word = targets.get("anti-stereotype")
        stereo_word = targets.get("stereotype")
        unrelated_word = targets.get("unrelated")

        if not anti_word or not stereo_word:
            continue

        parts = context.split("BLANK")
        prompt = parts[0]
        remainder = parts[1] if len(parts) > 1 else ""

        anti_completion = anti_word + remainder
        stereo_completion = stereo_word + remainder

        dpo_pairs.append({
            "id": item_id,
            "prompt": prompt,
            "chosen": anti_completion,
            "rejected": stereo_completion,
            "pair_type": "debias"
        })
        n_debias += 1

        if unrelated_word:
            unrelated_completion = unrelated_word + remainder
            dpo_pairs.append({
                "id": item_id,
                "prompt": prompt,
                "chosen": stereo_completion,
                "rejected": unrelated_completion,
                "pair_type": "lms"
            })
            dpo_pairs.append({
                "id": item_id,
                "prompt": prompt,
                "chosen": anti_completion,
                "rejected": unrelated_completion,
                "pair_type": "lms"
            })
            n_lms += 2

    # s3_utils.write_jsonl(dpo_pairs, output_path)
    with open(output_path, "w") as f:
        for pair in dpo_pairs:
            f.write(json.dumps(pair) + "\n")

    print(f"Generated {len(dpo_pairs)} total DPO pairs ({n_debias} debias + {n_lms} LMS)")
    print(f"Saved to S3 ({output_path})")

def generate_sft_v2_dataset(input_path: str, output_path: str):
    print(f"Reading rephrased data from S3 ({input_path})...")

    # raw_data = s3_utils.read_json(input_path)

    # read json file
    with open(input_path, "r") as f:
        raw_data = json.load(f)

    sft_examples = []

    for item in raw_data:
        item_id = item.get("id")
        context = item.get("rephrased_context", item.get("original_context", ""))
        targets = item.get("targets", {})

        if not item_id or "BLANK" not in context or not targets:
            continue

        anti_stereo_word = targets.get("anti-stereotype")
        stereo_word = targets.get("stereotype")

        if not anti_stereo_word or not stereo_word:
            continue

        parts = context.split("BLANK")
        prompt = parts[0]
        remainder = parts[1] if len(parts) > 1 else ""

        completion = anti_stereo_word + remainder
        stereotype_completion = stereo_word + remainder

        sft_examples.append({
            "id": item_id,
            "prompt": prompt,
            "completion": completion,
            "stereotype_completion": stereotype_completion
        })

    # s3_utils.write_jsonl(sft_examples, output_path)
    with open(output_path, "w") as f:
        for example in sft_examples:
            f.write(json.dumps(example) + "\n")

    print(f"Successfully generated {len(sft_examples)} improved SFT examples.")
    print(f"Saved to S3 ({output_path})")

# if __name__ == "__main__":
#     rephrase_stereoset("data/stereoset/dev.json", "data/stereoset/gender_dev_rephrased.json")
#     rephrase_stereoset("data/stereoset/test.json", "data/stereoset/gender_test_rephrased.json")

#     generate_sft_v2_dataset(
#         "data/stereoset/gender_test_rephrased.json",
#         "data/stereoset/fine-tune-sft/sft_bias_mitigation_v2.jsonl",
#     )
#     generate_dpo_triplet_dataset(
#         "data/stereoset/gender_test_rephrased.json",
#         "data/stereoset/fine-tune-dpo/dpo_pairs_triplet.jsonl",
#     )

if __name__ == "__main__":
    # rephrase_stereoset("data/stereoset/dev.json", "data/stereoset/gender_dev_rephrased.json")
    # rephrase_stereoset("data/stereoset/test.json", "data/stereoset/gender_test_rephrased.json")

    generate_sft_v2_dataset(
        "gender_test_rephrased_v2.json",
        "sft_bias_mitigation_v2.jsonl",
    )
    generate_dpo_triplet_dataset(
        "gender_test_rephrased_v2.json",
        "dpo_pairs_triplet_v2.jsonl",
    )