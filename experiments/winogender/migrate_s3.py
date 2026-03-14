"""Migrate Winogender S3 objects from the old layout to the new clean structure.

Usage:
    python -m experiments.winogender.migrate_s3 --dry-run
    python -m experiments.winogender.migrate_s3
    python -m experiments.winogender.migrate_s3 --delete-old
"""
import argparse
import os
import re

import boto3
from dotenv import load_dotenv

load_dotenv()

S3_BUCKET = "modelsfinetuned"
EXPERIMENTS_PREFIX = "experiments"

OLD_OUTPUTS = f"{EXPERIMENTS_PREFIX}/outputs/gpt2-xl/winogender"
OLD_CHECKPOINT_PREFIX = "gpt2-xl-finetuned-winogender"

NEW_BASELINE_TRAIN = f"{OLD_OUTPUTS}/baseline/train"
NEW_BASELINE_TEST = f"{OLD_OUTPUTS}/baseline/test"
NEW_FT_TRAIN = f"{OLD_OUTPUTS}/fine_tuned/train"
NEW_FT_TEST = f"{OLD_OUTPUTS}/fine_tuned/test"
NEW_FT_CHECKPOINTS = f"{OLD_OUTPUTS}/fine_tuned/checkpoints"


def _client():
    return boto3.client(
        "s3",
        aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
    )


def _list_keys(client, prefix):
    keys = []
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=prefix):
        for obj in page.get("Contents", []):
            keys.append(obj["Key"])
    return keys


def _rename_impact_file(filename):
    """accumulated_impact_winogender.csv -> accumulated_impact.csv"""
    if filename == "accumulated_impact_winogender.csv":
        return "accumulated_impact.csv"
    return filename


def build_migration_plan(client):
    """Return list of (old_key, new_key) tuples."""
    moves = []

    # --- Baseline outputs (flat files in outputs/gpt2-xl/winogender/) ---
    baseline_files = _list_keys(client, OLD_OUTPUTS + "/")
    for key in baseline_files:
        relative = key[len(OLD_OUTPUTS) + 1:]
        # Skip anything already in a subdirectory (fine_tuned/, finetuned/, plots/, baseline/)
        if "/" in relative:
            continue

        filename = relative
        if filename.endswith("_test.csv"):
            new_name = _rename_impact_file(filename.replace("_test.csv", ".csv"))
            new_key = f"{NEW_BASELINE_TEST}/{new_name}"
        elif filename.endswith("_custom.csv"):
            continue
        elif filename.endswith(".csv"):
            new_name = _rename_impact_file(filename)
            new_key = f"{NEW_BASELINE_TRAIN}/{new_name}"
        else:
            continue

        if key != new_key:
            moves.append((key, new_key))

    # --- Fine-tuned evaluation results (finetuned/ -> fine_tuned/train|test) ---
    ft_eval_keys = _list_keys(client, f"{OLD_OUTPUTS}/finetuned/")
    for key in ft_eval_keys:
        relative = key[len(OLD_OUTPUTS) + len("/finetuned/"):]
        parts = relative.split("/", 1)
        if len(parts) != 2:
            continue
        run_dir, filename = parts

        new_filename = _rename_impact_file(filename)

        if run_dir.endswith("_test"):
            run_id = run_dir[: -len("_test")]
            new_key = f"{NEW_FT_TEST}/{run_id}/{new_filename}"
        else:
            run_id = run_dir
            new_key = f"{NEW_FT_TRAIN}/{run_id}/{new_filename}"

        if key != new_key:
            moves.append((key, new_key))

    # --- Checkpoints (bucket root -> under experiments/) ---
    ckpt_keys = _list_keys(client, OLD_CHECKPOINT_PREFIX + "/")
    for key in ckpt_keys:
        filename = key[len(OLD_CHECKPOINT_PREFIX) + 1:]
        new_key = f"{NEW_FT_CHECKPOINTS}/{filename}"
        if key != new_key:
            moves.append((key, new_key))

    return moves


MULTIPART_THRESHOLD = 5 * 1024 * 1024 * 1024  # 5 GB
MULTIPART_CHUNKSIZE = 500 * 1024 * 1024        # 500 MB


def _copy_large_object(client, old_key, new_key, size):
    """Multipart copy for objects exceeding 5 GB."""
    mpu = client.create_multipart_upload(Bucket=S3_BUCKET, Key=new_key)
    upload_id = mpu["UploadId"]
    parts = []

    try:
        offset = 0
        part_number = 1
        while offset < size:
            end = min(offset + MULTIPART_CHUNKSIZE, size) - 1
            resp = client.upload_part_copy(
                Bucket=S3_BUCKET,
                Key=new_key,
                CopySource={"Bucket": S3_BUCKET, "Key": old_key},
                CopySourceRange=f"bytes={offset}-{end}",
                UploadId=upload_id,
                PartNumber=part_number,
            )
            parts.append({
                "ETag": resp["CopyPartResult"]["ETag"],
                "PartNumber": part_number,
            })
            offset = end + 1
            part_number += 1

        client.complete_multipart_upload(
            Bucket=S3_BUCKET, Key=new_key, UploadId=upload_id,
            MultipartUpload={"Parts": parts},
        )
    except Exception:
        client.abort_multipart_upload(
            Bucket=S3_BUCKET, Key=new_key, UploadId=upload_id)
        raise


def execute_migration(client, moves, delete_old=False):
    for i, (old_key, new_key) in enumerate(moves):
        print(f"[{i + 1}/{len(moves)}] {old_key}  ->  {new_key}")

        head = client.head_object(Bucket=S3_BUCKET, Key=old_key)
        size = head["ContentLength"]

        if size >= MULTIPART_THRESHOLD:
            print(f"  Large file ({size / 1e9:.1f} GB), using multipart copy ...")
            _copy_large_object(client, old_key, new_key, size)
        else:
            client.copy_object(
                Bucket=S3_BUCKET,
                CopySource={"Bucket": S3_BUCKET, "Key": old_key},
                Key=new_key,
            )

        if delete_old:
            client.delete_object(Bucket=S3_BUCKET, Key=old_key)
            print(f"  Deleted: {old_key}")


def main():
    parser = argparse.ArgumentParser(description="Migrate Winogender S3 layout")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print planned moves without executing")
    parser.add_argument("--delete-old", action="store_true",
                        help="Delete old keys after successful copy")
    args = parser.parse_args()

    client = _client()
    moves = build_migration_plan(client)

    if not moves:
        print("Nothing to migrate.")
        return

    if args.dry_run:
        print(f"DRY RUN — {len(moves)} object(s) would be moved:\n")
        for old_key, new_key in moves:
            print(f"  {old_key}")
            print(f"    -> {new_key}\n")
    else:
        print(f"Migrating {len(moves)} object(s) ...")
        execute_migration(client, moves, delete_old=args.delete_old)
        print("\nDone.")


if __name__ == "__main__":
    main()
