import boto3
import json
import io
import os

import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

S3_BUCKET = "modelsfinetuned"
S3_PREFIX = "experiments"

_USE_S3 = True


def set_use_s3(flag: bool):
    """Toggle between the S3 backend (default) and a local-disk backend that
    reads/writes the same relative paths directly on disk. Intended to be
    called once, near the top of a script's __main__ block, based on a
    --no-s3 CLI flag."""
    global _USE_S3
    _USE_S3 = flag


def _client():
    return boto3.client(
        's3',
        aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
    )


def s3_key(local_path: str) -> str:
    if not _USE_S3:
        return local_path

    return f"{S3_PREFIX}/{local_path}"


def read_json(path: str) -> dict:
    if not _USE_S3:
        with open(path, "r") as f:
            return json.load(f)
    obj = _client().get_object(Bucket=S3_BUCKET, Key=s3_key(path))
    return json.loads(obj['Body'].read().decode('utf-8'))


def read_jsonl(path: str) -> list:
    if not _USE_S3:
        with open(path, "r") as f:
            lines = f.read().strip().split('\n')
        return [json.loads(line) for line in lines if line.strip()]
    obj = _client().get_object(Bucket=S3_BUCKET, Key=s3_key(path))
    lines = obj['Body'].read().decode('utf-8').strip().split('\n')
    return [json.loads(line) for line in lines if line.strip()]


def read_csv(path: str) -> pd.DataFrame:
    if not _USE_S3:
        return pd.read_csv(path)
    obj = _client().get_object(Bucket=S3_BUCKET, Key=s3_key(path))
    return pd.read_csv(io.BytesIO(obj['Body'].read()))


def write_json(data, path: str):
    if not _USE_S3:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=4)
        return
    body = json.dumps(data, indent=4).encode('utf-8')
    _client().put_object(Bucket=S3_BUCKET, Key=s3_key(path), Body=body)


def write_jsonl(items: list, path: str):
    if not _USE_S3:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            f.write('\n'.join(json.dumps(item) for item in items))
        return
    body = '\n'.join(json.dumps(item) for item in items).encode('utf-8')
    _client().put_object(Bucket=S3_BUCKET, Key=s3_key(path), Body=body)


def write_csv(df: pd.DataFrame, path: str):
    if not _USE_S3:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
        return
    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    _client().put_object(Bucket=S3_BUCKET, Key=s3_key(path), Body=buf.getvalue())


def list_keys(prefix: str) -> list:
    if not _USE_S3:
        base = Path(prefix)
        if not base.exists():
            return []
        return [str(p) for p in sorted(base.rglob("*")) if p.is_file()]
    client = _client()
    keys = []
    paginator = client.get_paginator('list_objects_v2')
    for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=s3_key(prefix)):
        for obj in page.get('Contents', []):
            keys.append(obj['Key'])
    return keys


def write_bytes(data: bytes, path: str):
    if not _USE_S3:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            f.write(data)
        return
    _client().put_object(Bucket=S3_BUCKET, Key=s3_key(path), Body=data)


def save_plot(fig, path: str):
    if not _USE_S3:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, format='pdf', bbox_inches='tight')
        return
    buf = io.BytesIO()
    fig.savefig(buf, format='pdf', bbox_inches='tight')
    buf.seek(0)
    _client().put_object(Bucket=S3_BUCKET, Key=s3_key(path), Body=buf.getvalue())


def exists(path: str) -> bool:
    if not _USE_S3:
        return Path(path).is_file()
    try:
        _client().head_object(Bucket=S3_BUCKET, Key=s3_key(path))
        return True
    except _client().exceptions.ClientError:
        return False
