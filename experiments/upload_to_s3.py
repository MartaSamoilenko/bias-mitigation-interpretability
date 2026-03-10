import os
from s3_utils import _client, S3_BUCKET, S3_PREFIX


def upload_directory(local_dir: str):
    client = _client()
    for root, dirs, files in os.walk(local_dir):
        for fname in files:
            local_path = os.path.join(root, fname)
            key = f"{S3_PREFIX}/{local_path}"
            print(f"Uploading {local_path} -> s3://{S3_BUCKET}/{key}")
            client.upload_file(local_path, S3_BUCKET, key)


if __name__ == "__main__":
    upload_directory("data")
    upload_directory("outputs")
    print("Done.")
