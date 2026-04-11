"""Custom Pulumi dynamic resource for efficient multi-file GCS bucket uploads.

Treats the entire dist directory as a single resource — avoids creating one
Pulumi resource per file (which would be expensive at scale).

Adapted from the Compass project's bucket_content.py.
"""

import hashlib
import mimetypes
import os
import random
import tempfile
from dataclasses import dataclass, asdict
from typing import Any, Dict, Union

import pulumi
from pulumi.dynamic import Resource, ResourceProvider, CreateResult, CheckResult, CheckFailure, DiffResult, UpdateResult


@dataclass
class _Props:
    bucket_name: str
    no_cache_paths: list
    target_dir: str
    file_hashes: Dict[str, str]


def _parse_props(props: dict) -> _Props:
    return _Props(
        bucket_name=props.get("bucket_name"),
        no_cache_paths=props.get("no_cache_paths", []),
        target_dir=props.get("target_dir", ""),
        file_hashes=props.get("file_hashes", {}),
    )


def _get_gcs_bucket(bucket_name: str):
    import google.auth
    from google.cloud import storage
    credentials, _ = google.auth.default()
    client = storage.Client(credentials=credentials)
    return client.bucket(bucket_name)


def _compute_file_changes(old_hashes, new_hashes):
    old_files = set(old_hashes.keys())
    new_files = set(new_hashes.keys())
    new_only = new_files - old_files
    deleted_only = old_files - new_files
    changed = {f for f in old_files & new_files if old_hashes[f] != new_hashes[f]}
    return new_only, deleted_only, changed


def _will_update(changes):
    return any(len(c) > 0 for c in changes)


def _md5(file_path: str) -> str:
    h = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            h.update(chunk)
    return h.hexdigest()


def _scan_dir(directory: str) -> Dict[str, str]:
    hashes = {}
    if not os.path.exists(directory):
        pulumi.warn(f"Source directory does not exist: {directory}")
        return hashes
    for root, _, files in os.walk(directory):
        for name in files:
            abs_path = os.path.join(root, name)
            rel_path = os.path.relpath(abs_path, directory)
            checksum = _md5(abs_path)
            if checksum:
                hashes[rel_path] = checksum
    return hashes


def _concat_path(prefix: str, path: str) -> str:
    if not prefix:
        return path
    return prefix.rstrip("/") + "/" + path


def _list_bucket_paths(bucket, target_dir: str):
    existing = set()
    prefix = target_dir or ""
    try:
        for blob in bucket.list_blobs(prefix=prefix):
            name = blob.name
            if target_dir and name.startswith(prefix):
                existing.add(name[len(prefix):])
            else:
                existing.add(name)
    except Exception as e:
        pulumi.warn(f"Failed to list bucket objects: {e}")
    return existing


def _upload_file(*, bucket, source_path: str, bucket_path: str, no_cache: bool) -> bool:
    tmp = None
    try:
        mime_type, _ = mimetypes.guess_type(source_path)
        compressible = (
            (mime_type and mime_type.startswith(("text/", "application/javascript", "application/json", "image/svg+xml")))
            or source_path.endswith((".html", ".css", ".js", ".json", ".svg", ".ttf", ".woff", ".woff2"))
        )

        if compressible:
            import brotli
            with open(source_path, "rb") as f_in:
                compressed = brotli.compress(f_in.read(), quality=11)
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".br")
            tmp.write(compressed)
            tmp.close()
            upload_path = tmp.name
            content_encoding = "br"
        else:
            upload_path = source_path
            content_encoding = None

        blob = bucket.blob(bucket_path)
        if no_cache:
            blob.cache_control = "no-store"
        if content_encoding:
            blob.content_encoding = content_encoding
        blob.upload_from_filename(upload_path, mime_type)
        return True
    except Exception as e:
        pulumi.error(f"Failed to upload {bucket_path}: {e}")
        return False
    finally:
        if tmp and os.path.exists(tmp.name):
            try:
                os.unlink(tmp.name)
            except Exception:
                pass


def _random_id() -> str:
    n = random.Random().randrange(2**32, 2**64)  # nosec B311
    return f"bucket-content-{n:x}"


class _BucketContentProvider(ResourceProvider):
    def __init__(self, source_dir: str):
        super().__init__()
        self.source_dir = source_dir

    def check(self, _olds, news) -> CheckResult:
        failures = []
        if not news.get("bucket_name"):
            failures.append(CheckFailure("bucket_name", "bucket_name is required"))
        return CheckResult(news, failures)

    def create(self, props: dict) -> CreateResult:
        p = _parse_props(props)
        bucket = _get_gcs_bucket(p.bucket_name)
        ok = 0
        for rel_path in p.file_hashes:
            src = os.path.join(self.source_dir, rel_path)
            dst = _concat_path(p.target_dir, rel_path)
            if _upload_file(bucket=bucket, source_path=src, bucket_path=dst, no_cache=(rel_path in p.no_cache_paths)):
                ok += 1
        if ok != len(p.file_hashes):
            raise Exception(f"Upload failed: {len(p.file_hashes) - ok}/{len(p.file_hashes)} files")
        pulumi.info(f"Uploaded {ok} files to gs://{p.bucket_name}")
        return CreateResult(id_=_random_id(), outs=asdict(p))

    def update(self, _id, olds, news) -> UpdateResult:
        old_p = _parse_props(olds)
        new_p = _parse_props(news)
        new_files, deleted_files, changed_files = _compute_file_changes(old_p.file_hashes, new_p.file_hashes)
        bucket = _get_gcs_bucket(new_p.bucket_name)

        for f in deleted_files | changed_files:
            try:
                bucket.delete_blob(_concat_path(new_p.target_dir, f))
            except Exception as e:
                pulumi.warn(f"Failed to delete {f}: {e}")

        existing = _list_bucket_paths(bucket, new_p.target_dir)
        missing = set(new_p.file_hashes.keys()) - existing
        to_upload = new_files | changed_files | missing

        ok = 0
        for f in to_upload:
            src = os.path.join(self.source_dir, f)
            dst = _concat_path(new_p.target_dir, f)
            if _upload_file(bucket=bucket, source_path=src, bucket_path=dst, no_cache=(f in new_p.no_cache_paths)):
                ok += 1
        if ok != len(to_upload):
            raise Exception(f"Upload failed: {len(to_upload) - ok}/{len(to_upload)} files")
        return UpdateResult(asdict(new_p))

    def delete(self, _id, props):
        p = _parse_props(props)
        bucket = _get_gcs_bucket(p.bucket_name)
        for rel_path in p.file_hashes:
            blob = bucket.get_blob(_concat_path(p.target_dir, rel_path))
            if blob:
                blob.delete()

    def diff(self, _id, olds, news) -> DiffResult:
        old_p = _parse_props(olds)
        new_p = _parse_props(news)

        if new_p.bucket_name != old_p.bucket_name:
            return DiffResult(True, ["bucket_name"], [], True)
        if new_p.target_dir != old_p.target_dir:
            return DiffResult(True, ["target_dir"], [], True)

        changes = _compute_file_changes(old_p.file_hashes, new_p.file_hashes)
        if _will_update(changes):
            return DiffResult(True, [], [], False)

        try:
            bucket = _get_gcs_bucket(new_p.bucket_name)
            existing = _list_bucket_paths(bucket, new_p.target_dir)
            missing = set(new_p.file_hashes.keys()) - existing
            if missing:
                pulumi.warn(f"{len(missing)} missing objects detected — will re-upload")
                return DiffResult(True, [], [], False)
        except Exception as e:
            pulumi.warn(f"Skipping missing-object check: {e}")

        return DiffResult(False, [], [], False)


class BucketContent(Resource):
    """Upload a local directory to a GCS bucket as a single Pulumi resource.

    Only changed/new/deleted files are synced on update. Text assets are
    Brotli-compressed. Files in no_cache_paths get Cache-Control: no-store;
    everything else inherits the bucket/CDN default (long-lived cache).
    """

    def __init__(
        self,
        resource_name: str,
        *,
        bucket_name: Union[str, pulumi.Output],
        source_dir: str,
        target_dir: str = "",
        no_cache_paths: list = None,
        opts: pulumi.ResourceOptions = None,
    ):
        if not os.path.exists(source_dir):
            raise ValueError(f"source_dir does not exist: {source_dir}")

        file_hashes = _scan_dir(source_dir)
        props = {
            "bucket_name": bucket_name,
            "target_dir": target_dir,
            "no_cache_paths": no_cache_paths or [],
            "file_hashes": file_hashes,
        }
        provider = _BucketContentProvider(source_dir=source_dir)
        super().__init__(provider, resource_name, props=props, opts=opts)
