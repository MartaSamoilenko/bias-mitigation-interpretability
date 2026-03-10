import boto3
import json
import io
import os

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

S3_BUCKET = "modelsfinetuned"
S3_PREFIX = "experiments"


def _client():
    return boto3.client(
        's3',
        aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
    )


def s3_key(local_path: str) -> str:
    """Converts a local relative path like 'data/stereoset/test.json'
    to an S3 key like 'experiments/data/stereoset/test.json'."""
    return f"{S3_PREFIX}/{local_path}"


def read_json(path: str) -> dict:
    obj = _client().get_object(Bucket=S3_BUCKET, Key=s3_key(path))
    return json.loads(obj['Body'].read().decode('utf-8'))


def read_jsonl(path: str) -> list:
    obj = _client().get_object(Bucket=S3_BUCKET, Key=s3_key(path))
    lines = obj['Body'].read().decode('utf-8').strip().split('\n')
    return [json.loads(line) for line in lines if line.strip()]


def read_csv(path: str) -> pd.DataFrame:
    obj = _client().get_object(Bucket=S3_BUCKET, Key=s3_key(path))
    return pd.read_csv(io.BytesIO(obj['Body'].read()))


def write_json(data, path: str):
    body = json.dumps(data, indent=4).encode('utf-8')
    _client().put_object(Bucket=S3_BUCKET, Key=s3_key(path), Body=body)


def write_jsonl(items: list, path: str):
    body = '\n'.join(json.dumps(item) for item in items).encode('utf-8')
    _client().put_object(Bucket=S3_BUCKET, Key=s3_key(path), Body=body)


def write_csv(df: pd.DataFrame, path: str):
    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    _client().put_object(Bucket=S3_BUCKET, Key=s3_key(path), Body=buf.getvalue())


def list_keys(prefix: str) -> list:
    """List all S3 object keys under the given logical prefix."""
    client = _client()
    keys = []
    paginator = client.get_paginator('list_objects_v2')
    for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=s3_key(prefix)):
        for obj in page.get('Contents', []):
            keys.append(obj['Key'])
    return keys


def write_bytes(data: bytes, path: str):
    _client().put_object(Bucket=S3_BUCKET, Key=s3_key(path), Body=data)


def save_plot(fig, path: str):
    """Save a matplotlib figure to S3 as PDF."""
    buf = io.BytesIO()
    fig.savefig(buf, format='pdf', bbox_inches='tight')
    buf.seek(0)
    _client().put_object(Bucket=S3_BUCKET, Key=s3_key(path), Body=buf.getvalue())
