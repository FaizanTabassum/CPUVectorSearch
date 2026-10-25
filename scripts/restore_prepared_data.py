#!/usr/bin/env python3
"""Restore the validated team dataset from its pinned Git snapshot, into ignored local files."""

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def relative_path(value):
    path = PurePosixPath(value)
    if not value or path.is_absolute() or ".." in path.parts or path.as_posix() != value:
        raise ValueError(f"unsafe snapshot path: {value}")
    return path


def verify(folder, files):
    for name, expected in files.items():
        path = folder / name
        if not path.is_file() or path.is_symlink() or path.stat().st_size != expected["bytes"]:
            raise ValueError(f"missing or different snapshot file: {path}")
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected["sha256"]:
            raise ValueError(f"checksum mismatch: {path}")


def restore(repository, snapshot, output):
    if snapshot["schema_version"] != 1 or not snapshot["files"]:
        raise ValueError("unsupported or empty snapshot descriptor")
    commit = snapshot["source_commit"]
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("snapshot must pin a complete Git commit SHA")
    prefix = relative_path(snapshot["source_path"])
    for name in snapshot["files"]:
        relative_path(name)
    if output.exists():
        verify(output, snapshot["files"])
        return False
    exists = subprocess.run(["git", "cat-file", "-e", f"{commit}^{{commit}}"],
                            cwd=repository, capture_output=True)
    if exists.returncode:
        remote = snapshot["source_repository"]
        if not remote.startswith("https://github.com/"):
            raise ValueError("snapshot source must be a public HTTPS GitHub repository")
        # A shallow clone may not contain the older snapshot. Fetch without changing branches.
        subprocess.run(["git", "fetch", "--no-tags", remote, commit], cwd=repository, check=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".dataset-restore-", dir=output.parent) as temporary:
        staging = Path(temporary) / "dataset"
        staging.mkdir()
        for name in snapshot["files"]:
            target = staging / name
            target.parent.mkdir(parents=True, exist_ok=True)
            blob = subprocess.run(["git", "show", f"{commit}:{prefix.as_posix()}/{name}"],
                                  cwd=repository, check=True, capture_output=True)
            target.write_bytes(blob.stdout)
        verify(staging, snapshot["files"])
        if output.exists():
            raise ValueError("output appeared during restoration; refusing to replace it")
        staging.rename(output)
    return True


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="new directory, or an identical existing snapshot")
    args = parser.parse_args(argv)
    try:
        snapshot = json.loads((ROOT / "configs/dataset-snapshot.json").read_text())
        output = (args.output or ROOT / relative_path(snapshot["default_output"])).absolute()
        created = restore(ROOT, snapshot, output)
        print(f"{'Restored' if created else 'Already available and verified'}: {output}")
        print(f"All {len(snapshot['files'])} pinned file checksums match. No tracked files were changed.")
        print("Practice answers: reference-tuning/top10-tuning-n{10000,50000,100000}.csv")
        print("The test inputs remain reserved until the evaluation settings are frozen.")
        return 0
    except (OSError, ValueError, KeyError, TypeError, subprocess.CalledProcessError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
