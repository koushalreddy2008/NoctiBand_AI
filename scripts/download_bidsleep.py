"""Download and verify BIDSleep v1.0.0 (resumable, SHA-256 checked).

python scripts/download_bidsleep.py --verify-only
python scripts/download_bidsleep.py --subjects Bidslab00 Bidslab01 --nights 1 2 3
python scripts/download_bidsleep.py --all
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import requests

BASE = "https://physionet.org/files/bidsleep-dataset/1.0.0/"
CHUNK = 1 << 20


PROGRESS_INTERVAL = 5 * (1 << 20)  # 5 MB


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while block := f.read(CHUNK):
            h.update(block)
    return h.hexdigest()


def fetch_manifest(session: requests.Session) -> dict[str, str]:
    r = session.get(BASE + "SHA256SUMS.txt", timeout=60)
    r.raise_for_status()
    out: dict[str, str] = {}
    for line in r.text.splitlines():
        parts = line.split(maxsplit=1)
        if len(parts) == 2:
            out[parts[1].strip().lstrip("*")] = parts[0].lower()
    return out


def download(session: requests.Session, rel: str, dest: Path, expected: str) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    pos = part.stat().st_size if part.exists() else 0
    headers = {"Range": f"bytes={pos}-"} if pos else {}
    with session.get(BASE + rel, headers=headers, stream=True, timeout=60) as r:
        if r.status_code != 416:  # 416 = .part is already complete
            r.raise_for_status()
            if pos and r.status_code != 206:  # server ignored Range: restart
                pos = 0
            content_len = r.headers.get("Content-Length")
            total = pos + int(content_len) if content_len is not None else None
            done = pos
            last_reported = done

            with part.open("ab" if pos else "wb") as f:
                for chunk in r.iter_content(CHUNK):
                    f.write(chunk)
                    done += len(chunk)
                    if done - last_reported >= PROGRESS_INTERVAL:
                        last_reported = done
                        if total:
                            pct = done / total * 100.0
                            print(
                                f"  progress: {done / (1 << 20):.1f} / "
                                f"{total / (1 << 20):.1f} MB ({pct:.1f}%)"
                            )
                        else:
                            print(f"  progress: {done / (1 << 20):.1f} MB")
    if sha256_of(part) != expected:
        part.unlink()
        raise RuntimeError(f"checksum mismatch for {rel}; partial deleted, rerun")
    part.replace(dest)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/raw")
    ap.add_argument("--subjects", nargs="*")
    ap.add_argument("--nights", nargs="*")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--verify-only", action="store_true")
    a = ap.parse_args()
    root = Path(a.root)

    with requests.Session() as s:
        files = {r: h for r, h in fetch_manifest(s).items() if r.count("/") == 2}
        if a.verify_only:
            files = {r: h for r, h in files.items() if (root / r).exists()}
        elif not a.all:
            if not a.subjects:
                ap.error("give --subjects (and optionally --nights) or --all")
            files = {r: h for r, h in files.items()
                     if r.split("/")[0] in a.subjects
                     and (not a.nights or r.split("/")[1] in a.nights)}
        bad: list[str] = []
        for i, (rel, h) in enumerate(sorted(files.items()), 1):
            dest, tag = root / rel, f"[{i}/{len(files)}]"
            if dest.exists() and sha256_of(dest) == h:
                print(tag, "ok     ", rel)
                continue
            if a.verify_only:
                print(tag, "BAD    ", rel)
                bad.append(rel)
                continue
            if dest.exists():
                print(tag, "re-downloading (hash mismatch)", rel)
            else:
                print(tag, "downloading", rel)
            download(s, rel, dest, h)
    if bad:
        print(f"{len(bad)} bad file(s)")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
