"""Verify all data/raw_v2 files against SHA256SUMS.txt."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path


def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def verify_raw_v2(base_dir: Path | str = "data/raw_v2") -> int:
    base_path = Path(base_dir)
    sums_file = base_path / "SHA256SUMS.txt"
    if not sums_file.exists():
        print(f"ERROR: {sums_file} not found.")
        return 1

    lines = sums_file.read_text(encoding="utf-8").strip().splitlines()
    total_files = 0
    ok_count = 0
    bad_count = 0

    print(f"Verifying {len(lines)} entries in {sums_file}...")

    for line in lines:
        line = line.strip()
        if not line:
            continue
        parts = line.split(None, 1)
        if len(parts) != 2:
            continue
        expected_hash, rel_path = parts[0], parts[1].strip()
        file_path = base_path / rel_path
        total_files += 1

        if not file_path.exists():
            print(f"BAD (missing): {rel_path}")
            bad_count += 1
            continue

        actual_hash = compute_sha256(file_path)
        if actual_hash.lower() == expected_hash.lower():
            print(f"ok: {rel_path}")
            ok_count += 1
        else:
            print(f"BAD (hash mismatch): {rel_path} (expected {expected_hash}, got {actual_hash})")
            bad_count += 1

    print("--------------------------------------------------")
    print(f"Verification Summary: Total={total_files}, OK={ok_count}, BAD={bad_count}")
    if bad_count > 0:
        print(f"VERIFICATION FAILED: {bad_count} file(s) bad or missing.")
        return 1
    print("VERIFICATION PASSED: All files match SHA256 checksums.")
    return 0


if __name__ == "__main__":
    sys.exit(verify_raw_v2())
