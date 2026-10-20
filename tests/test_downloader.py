"""Unit tests for resumable, SHA-256 verified download logic."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pytest

from scripts.download_bidsleep import download, sha256_of


class FakeResponse:
    """Mock requests Response object for testing streaming downloads."""

    def __init__(
        self,
        content_chunks: list[bytes],
        status_code: int = 200,
        content_length: int | None = None,
    ):
        self.chunks = content_chunks
        self.status_code = status_code
        self.headers: dict[str, str] = {}
        if content_length is not None:
            self.headers["Content-Length"] = str(content_length)
        elif content_chunks:
            self.headers["Content-Length"] = str(sum(len(c) for c in content_chunks))

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP Error {self.status_code}")

    def iter_content(self, chunk_size: int = 1024):
        yield from self.chunks

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


class FakeSession:
    """Mock requests Session tracking received headers and requests."""

    def __init__(self, responses: list[FakeResponse]):
        self.responses = list(responses)
        self.recorded_calls: list[dict[str, Any]] = []

    def get(self, url: str, headers: dict[str, str] | None = None, **kwargs) -> FakeResponse:
        self.recorded_calls.append({"url": url, "headers": headers or {}, "kwargs": kwargs})
        if not self.responses:
            raise RuntimeError("No more mocked responses available.")
        return self.responses.pop(0)


def test_corrupt_file_untouched_on_failed_download_hash(tmp_path: Path):
    """Case (a): an existing corrupt file stays untouched when new download fails its hash."""
    dest = tmp_path / "night" / "motion.csv"
    dest.parent.mkdir(parents=True, exist_ok=True)
    initial_content = b"original-corrupt-data-do-not-delete"
    dest.write_bytes(initial_content)

    part = dest.with_suffix(dest.suffix + ".part")

    # Download returns bad bytes that do not match expected hash
    bad_bytes = b"bad-streamed-data"
    fake_resp = FakeResponse([bad_bytes])
    session = FakeSession([fake_resp])

    expected_hash = hashlib.sha256(b"correct-data-never-received").hexdigest()

    with pytest.raises(RuntimeError, match="checksum mismatch"):
        download(session, "night/motion.csv", dest, expected=expected_hash)

    # Assert existing file is preserved and untouched
    assert dest.exists()
    assert dest.read_bytes() == initial_content
    # Assert partial file is unlinked
    assert not part.exists()


def test_verified_part_replaces_dest(tmp_path: Path):
    """Case (b): a verified .part replaces dest atomically."""
    dest = tmp_path / "night" / "motion.csv"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(b"old-content-to-replace")

    part = dest.with_suffix(dest.suffix + ".part")

    verified_bytes = b"verified-motion-data-streamed"
    expected_hash = hashlib.sha256(verified_bytes).hexdigest()

    fake_resp = FakeResponse([verified_bytes[:10], verified_bytes[10:]])
    session = FakeSession([fake_resp])

    download(session, "night/motion.csv", dest, expected=expected_hash)

    assert dest.exists()
    assert dest.read_bytes() == verified_bytes
    assert sha256_of(dest) == expected_hash
    assert not part.exists()


def test_leftover_part_resumes_using_range_header(tmp_path: Path):
    """Case (c): a leftover .part resumes using a Range header."""
    dest = tmp_path / "night" / "motion.csv"
    dest.parent.mkdir(parents=True, exist_ok=True)

    part = dest.with_suffix(dest.suffix + ".part")
    existing_part_data = b"0123456789"  # 10 bytes
    part.write_bytes(existing_part_data)

    remaining_data = b"-resumed-remainder-stream"
    full_data = existing_part_data + remaining_data
    expected_hash = hashlib.sha256(full_data).hexdigest()

    # Server supports range: status 206
    fake_resp = FakeResponse([remaining_data], status_code=206)
    session = FakeSession([fake_resp])

    download(session, "night/motion.csv", dest, expected=expected_hash)

    # Verify that Range header was sent with offset 10
    assert len(session.recorded_calls) == 1
    assert session.recorded_calls[0]["headers"].get("Range") == "bytes=10-"

    # Verify atomic replacement and complete verified content
    assert dest.exists()
    assert dest.read_bytes() == full_data
    assert sha256_of(dest) == expected_hash
    assert not part.exists()
