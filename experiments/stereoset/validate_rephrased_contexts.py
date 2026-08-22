"""
QA pass over experiments/stereoset/gender_test_rephrased.json.

For every entry whose `rephrased_context` differs from `original_context` (i.e. it
was actually rephrased by stereoset_paraphrase.py), ask an OpenAI reasoning model
to verify:

  1. "BLANK" is the very last word of `rephrased_context` (only trailing
     punctuation/quote marks may follow it -- nothing glued directly onto it,
     e.g. "BLANKd" or "BLANK-aged").
  2. Substituting "BLANK" with the `stereotype` target produces a meaningful,
     grammatically correct sentence.
  3. Substituting "BLANK" with the `anti-stereotype` target produces a
     meaningful, grammatically correct sentence.

If a check fails, the model is asked to fix it -- preferring to reword/reorder
`rephrased_context` alone, and only changing the `stereotype`/`anti-stereotype`
target words (to close synonyms) if no rewording can make both targets fit.
The `unrelated` target is never touched. The original meaning must be kept.

Entries that were never rephrased (rephrased_context == original_context) are
passed through unchanged and are not sent to the model.

Usage:
    python validate_rephrased_contexts.py [--limit N] [--workers N]

Outputs (written next to this script):
    gender_test_rephrased_fixed.json     Full corrected 767-item dataset.
    gender_test_rephrased_fix_log.json   Only the entries that were changed,
                                          with before/after values + explanation.
    gender_test_rephrased_checkpoint.json  Resume checkpoint (safe to delete).
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel

SCRIPT_DIR = Path(__file__).resolve().parent
load_dotenv(SCRIPT_DIR.parent / ".env")

INPUT_PATH = SCRIPT_DIR / "gender_test_rephrased.json"
OUTPUT_PATH = SCRIPT_DIR / "gender_test_rephrased_fixed.json"
LOG_PATH = SCRIPT_DIR / "gender_test_rephrased_fix_log.json"
CHECKPOINT_PATH = SCRIPT_DIR / "gender_test_rephrased_checkpoint.json"


def blank_at_end_literally(text: str) -> bool:
    """Ground-truth, code-level check (not the model's self-report) that BLANK is
    the very last word: only trailing punctuation/quote characters may follow it,
    and nothing may be glued directly onto the word itself."""
    stripped = text.rstrip()
    stripped = stripped.rstrip(".,?!;:\"'")
    return stripped.endswith("BLANK")

MODEL = "o4-mini"
REASONING_EFFORT = "medium"
MAX_WORKERS = 8
MAX_RETRIES = 4

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])


class FixResult(BaseModel):
    blank_at_end: bool
    meaningful_with_stereotype: bool
    meaningful_with_anti_stereotype: bool
    needs_fix: bool
    fixed_rephrased_context: str
    fixed_stereotype: str
    fixed_anti_stereotype: str
    explanation: str


SYSTEM_PROMPT = """You are a meticulous linguistic QA assistant for a bias-evaluation \
dataset built on StereoSet-style fill-in-the-blank sentences.

Each item has a "rephrased_context" sentence containing the placeholder word \
"BLANK", plus a "stereotype" target word/phrase and an "anti-stereotype" target \
word/phrase that are each meant to be substituted in place of "BLANK".

You must check three things about the rephrased_context:
1. "BLANK" is the very last word of the sentence. Only trailing punctuation or a \
closing quote mark may follow it (e.g. "BLANK.", "BLANK?\\"" are fine). Nothing may \
be glued directly onto the word itself, such as "BLANKd" or "BLANK-aged" -- those \
do NOT count as "BLANK at the very end".
2. Substituting "BLANK" with the stereotype target produces a grammatically \
correct, meaningful, natural-sounding sentence.
3. Substituting "BLANK" with the anti-stereotype target produces a grammatically \
correct, meaningful, natural-sounding sentence.

If any of these checks fail, fix it using this priority order:
(a) First, try to fix it by ONLY rewording or reordering the rephrased_context \
sentence, keeping it as close as possible to the current phrasing and to the \
meaning of the original_context. Do not change the targets if this works.
(b) Only if no rewording of the sentence can make it grammatical and meaningful \
for BOTH targets at once (for example the two targets are incompatible parts of \
speech, or the template requires a suffix hard-glued onto BLANK), you may change \
the stereotype and/or anti-stereotype target word(s) to a close synonym that \
preserves the same underlying meaning/concept as the original target. Prefer the \
smallest possible change.
(c) Never change or even consider the "unrelated" target -- it is provided only \
for context and does not need to grammatically fit the sentence.
(d) The overall meaning of the original sentence, and the semantic gist of each \
target, must always be preserved. Do not invent a new scenario.

CRITICAL, NON-NEGOTIABLE RULE: your fixed_rephrased_context must, after stripping \
only trailing punctuation and/or a closing quote mark, end with the literal \
substring "BLANK". This is a hard requirement, not a style preference. A common \
mistake is to "fix" an awkward sentence by reverting close to the original_context \
wording -- but original_context usually has BLANK in the middle, NOT at the end. \
Falling back to (or converging on) a phrasing where BLANK is not the last word is \
NEVER an acceptable fix, no matter how much more natural it sounds. If you catch \
yourself doing this, keep searching for an alternative phrasing that both sounds \
natural AND ends in BLANK. Double check your own fixed_rephrased_context against \
this rule before answering, and set blank_at_end truthfully based on the FIXED \
sentence you are about to return, not the original one.

Always return the FINAL rephrased_context, stereotype, and anti-stereotype in your \
response, even if you made no changes (in that case just echo the original values \
back unchanged). Set needs_fix to true if you changed anything at all (wording or \
targets), false if everything already passed all three checks as-is.
"""


def build_user_prompt(item: dict) -> str:
    targets = item["targets"]
    return f"""id: {item['id']}

original_context: {item['original_context']}
rephrased_context: {item['rephrased_context']}

targets:
- stereotype: {targets.get('stereotype')}
- anti-stereotype: {targets.get('anti-stereotype')}
- unrelated (context only, never change, does not need to fit grammatically): {targets.get('unrelated')}

Perform the three checks on rephrased_context and fix if needed, per your instructions."""


def validate_and_fix(item: dict) -> FixResult | None:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(item)},
    ]
    for attempt in range(MAX_RETRIES):
        try:
            completion = client.beta.chat.completions.parse(
                model=MODEL,
                reasoning_effort=REASONING_EFFORT,
                messages=messages,
                response_format=FixResult,
            )
            parsed = completion.choices[0].message.parsed
            if parsed is None:
                raise ValueError(f"Model returned no parsed content (refusal?): {completion.choices[0].message}")

            # Code-level enforcement: never trust the model's self-reported
            # blank_at_end blindly -- verify it ourselves and force a retry
            # with corrective feedback if it lied (e.g. reverted to a phrasing
            # close to original_context, which usually has BLANK mid-sentence).
            if not blank_at_end_literally(parsed.fixed_rephrased_context):
                messages.append({"role": "assistant", "content": parsed.model_dump_json()})
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            f"Your fixed_rephrased_context ({parsed.fixed_rephrased_context!r}) does NOT actually "
                            "end with the literal word BLANK (after stripping only trailing punctuation/quotes). "
                            "This violates the critical rule. Do not fall back to a phrasing resembling "
                            "original_context. Try again with a different sentence structure that both sounds "
                            "natural for both targets AND truly ends with BLANK."
                        ),
                    }
                )
                continue
            return parsed
        except Exception as e:
            wait = 2 ** attempt
            print(f"  [{item['id']}] attempt {attempt + 1}/{MAX_RETRIES} failed: {e}; retrying in {wait}s")
            time.sleep(wait)
    return None


def load_checkpoint(path: Path) -> dict:
    if path.exists():
        with open(path, "r") as f:
            return json.load(f)
    return {}


def save_checkpoint(checkpoint: dict, path: Path):
    tmp_path = path.with_suffix(".tmp")
    with open(tmp_path, "w") as f:
        json.dump(checkpoint, f, indent=2)
    tmp_path.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N differing items (for testing).")
    parser.add_argument("--workers", type=int, default=MAX_WORKERS, help="Number of concurrent worker threads.")
    parser.add_argument(
        "--current-path",
        type=Path,
        default=None,
        help=(
            "Optional path to a JSON file (same schema) whose rephrased_context/targets should be used as the "
            "CURRENT values to (re-)validate, instead of the base input file. Useful for a second verification "
            "pass over a previous run's output. original_context and the eligible id set are always taken from "
            "the base input file."
        ),
    )
    parser.add_argument("--output-path", type=Path, default=OUTPUT_PATH, help="Where to write the corrected full dataset.")
    parser.add_argument("--log-path", type=Path, default=LOG_PATH, help="Where to write the change log.")
    parser.add_argument("--checkpoint-path", type=Path, default=CHECKPOINT_PATH, help="Where to read/write the resume checkpoint.")
    args = parser.parse_args()

    with open(INPUT_PATH, "r") as f:
        data = json.load(f)

    print(f"Loaded {len(data)} total entries from {INPUT_PATH.name}")

    current_by_id = None
    if args.current_path is not None:
        with open(args.current_path, "r") as f:
            current_data = json.load(f)
        current_by_id = {item["id"]: item for item in current_data}
        print(f"Using current values from {args.current_path.name} for re-validation")

    to_process = [item for item in data if item["rephrased_context"] != item["original_context"]]
    unchanged_count = len(data) - len(to_process)
    print(f"  {len(to_process)} entries were rephrased (will be checked)")
    print(f"  {unchanged_count} entries are identical to original_context (skipped, passed through as-is)")

    if current_by_id is not None:
        # Swap in the current (e.g. round-1 fixed) rephrased_context/targets as
        # the basis for this pass, while keeping original_context + id set from
        # the base file.
        rebuilt = []
        for item in to_process:
            cur = current_by_id.get(item["id"])
            if cur is None:
                rebuilt.append(item)
                continue
            merged = dict(item)
            merged["rephrased_context"] = cur["rephrased_context"]
            merged["targets"] = dict(cur["targets"])
            rebuilt.append(merged)
        to_process = rebuilt

    if args.limit is not None:
        to_process = to_process[: args.limit]
        print(f"  --limit set: only processing first {len(to_process)} of the differing items")

    checkpoint = load_checkpoint(args.checkpoint_path)
    print(f"  {len(checkpoint)} entries already have a cached result in the checkpoint")

    pending = [item for item in to_process if item["id"] not in checkpoint]
    print(f"  {len(pending)} entries need an API call this run")

    failed_ids = []
    completed = 0
    total = len(pending)
    lock_save_every = 10

    if pending:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            future_to_item = {executor.submit(validate_and_fix, item): item for item in pending}
            for future in as_completed(future_to_item):
                item = future_to_item[future]
                result = future.result()
                completed += 1
                if result is None:
                    failed_ids.append(item["id"])
                    print(f"[{completed}/{total}] FAILED after retries: {item['id']}")
                else:
                    checkpoint[item["id"]] = result.model_dump()
                    tag = "CHANGED" if result.needs_fix else "ok"
                    print(f"[{completed}/{total}] {tag}: {item['id']}")
                if completed % lock_save_every == 0:
                    save_checkpoint(checkpoint, args.checkpoint_path)
        save_checkpoint(checkpoint, args.checkpoint_path)

    print("\nMerging results...")

    # "current" state (before this pass) for each processed id: either the
    # round-1 fixed values (if --current-path given) or the base raw values.
    current_lookup = {item["id"]: item for item in to_process}

    fixed_data = []
    change_log = []
    changed_count = 0
    bad_blank_count = 0

    for item in data:
        item_id = item["id"]
        if item_id not in checkpoint:
            fixed_data.append(item)
            continue

        result = checkpoint[item_id]
        current = current_lookup.get(item_id, item)
        new_item = dict(item)
        new_item["targets"] = dict(item["targets"])

        old_rephrased = current["rephrased_context"]
        old_stereotype = current["targets"].get("stereotype")
        old_anti = current["targets"].get("anti-stereotype")

        new_rephrased = result["fixed_rephrased_context"]
        new_stereotype = result["fixed_stereotype"]
        new_anti = result["fixed_anti_stereotype"]

        if not blank_at_end_literally(new_rephrased):
            bad_blank_count += 1
            print(f"  WARNING: {item_id} still does not end with BLANK after all retries: {new_rephrased!r}")

        new_item["rephrased_context"] = new_rephrased
        new_item["targets"]["stereotype"] = new_stereotype
        new_item["targets"]["anti-stereotype"] = new_anti

        fixed_data.append(new_item)

        actually_changed = (
            new_rephrased != old_rephrased
            or new_stereotype != old_stereotype
            or new_anti != old_anti
        )
        if actually_changed:
            changed_count += 1
            change_log.append(
                {
                    "id": item_id,
                    "original_context": item["original_context"],
                    "before": {
                        "rephrased_context": old_rephrased,
                        "stereotype": old_stereotype,
                        "anti-stereotype": old_anti,
                    },
                    "after": {
                        "rephrased_context": new_rephrased,
                        "stereotype": new_stereotype,
                        "anti-stereotype": new_anti,
                    },
                    "checks": {
                        "blank_at_end": result["blank_at_end"],
                        "meaningful_with_stereotype": result["meaningful_with_stereotype"],
                        "meaningful_with_anti_stereotype": result["meaningful_with_anti_stereotype"],
                    },
                    "explanation": result["explanation"],
                }
            )

    with open(args.output_path, "w") as f:
        json.dump(fixed_data, f, indent=4)
    with open(args.log_path, "w") as f:
        json.dump(change_log, f, indent=2)

    print(f"\nWrote {len(fixed_data)} entries -> {args.output_path.name}")
    print(f"Wrote {len(change_log)} changed entries -> {args.log_path.name}")
    if bad_blank_count:
        print(f"WARNING: {bad_blank_count} entries still fail the literal BLANK-at-end check -- see warnings above.")
    print("\n=== Summary ===")
    print(f"Total entries in dataset:        {len(data)}")
    print(f"Never rephrased (skipped):       {unchanged_count}")
    print(f"Checked this pass (or cached):   {len(to_process)}")
    print(f"Actually changed:                {changed_count}")
    print(f"Failed after retries (left as-is): {len(failed_ids)}")
    if failed_ids:
        print("  Failed ids:")
        for fid in failed_ids:
            print(f"    - {fid}")


if __name__ == "__main__":
    main()
