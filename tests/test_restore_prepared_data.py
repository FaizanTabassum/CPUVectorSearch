"""Pinned binary restoration, repeatability, and preservation of existing local files."""
import hashlib
import importlib.util
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("restore_prepared_data", ROOT / "scripts/restore_prepared_data.py")
restore_data = importlib.util.module_from_spec(spec)
spec.loader.exec_module(restore_data)


class RestoreTests(unittest.TestCase):
    def test_pinned_revision_and_existing_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            repo = root / "repo"
            repo.mkdir()
            def git(*arguments):
                return subprocess.check_output(["git", *arguments], cwd=repo, text=True).strip()
            git("init", "-q")
            folder = repo / "data/example"
            (folder / "answers").mkdir(parents=True)
            originals = {"vectors.bin": b"\x00\xff\x01test\x00", "answers/reference.csv": b"q,id\n0,7\n"}
            for name, content in originals.items():
                (folder / name).write_bytes(content)
            git("add", ".")
            git("-c", "user.name=Test", "-c", "user.email=test@example.com", "commit", "-qm", "snapshot")
            snapshot = {"schema_version": 1, "source_commit": git("rev-parse", "HEAD"),
                        "source_repository": "https://github.com/example/example.git", "source_path": "data/example",
                        "files": {name: {"bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()}
                                  for name, content in originals.items()}}
            # Current files differ: restoration must use the pinned commit, not the worktree.
            (folder / "vectors.bin").write_bytes(b"changed later")
            output = root / "restored"
            self.assertTrue(restore_data.restore(repo, snapshot, output))
            for name, content in originals.items():
                self.assertEqual((output / name).read_bytes(), content)
            self.assertFalse(restore_data.restore(repo, snapshot, output))
            (output / "vectors.bin").write_bytes(b"keep my local edits")
            with self.assertRaises(ValueError):
                restore_data.restore(repo, snapshot, output)
            self.assertEqual((output / "vectors.bin").read_bytes(), b"keep my local edits")
            bad = dict(snapshot)
            bad["files"] = {"../escape": {"bytes": 1, "sha256": "0" * 64}}
            with self.assertRaises(ValueError):
                restore_data.restore(repo, bad, root / "unsafe")
            self.assertFalse((root / "unsafe").exists())
            bad = dict(snapshot)
            bad["files"] = {"vectors.bin": {"bytes": len(originals["vectors.bin"]), "sha256": "0" * 64}}
            with self.assertRaises(ValueError):
                restore_data.restore(repo, bad, root / "wrong-hash")
            self.assertFalse((root / "wrong-hash").exists())
            self.assertEqual(git("rev-parse", "HEAD"), snapshot["source_commit"])


if __name__ == "__main__":
    unittest.main()
