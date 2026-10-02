"""Rebuild the dnn-benchmarking workload tarballs with the corrected SDPA graphs.

usage: python apply_fixes.py <dnn-benchmarking checkout>

Run after `dvc pull`. For every member listed in manifest.json, checks that the
tarball still holds the original bytes (sha256), replaces it with the fixed
file next to this script, and rewrites the tarball with all other members,
their order and their metadata unchanged. Then run `dvc add` on each rewritten
tarball, `dvc push`, and commit the .dvc pointers.
"""

import hashlib
import io
import json
import sys
import tarfile
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main() -> int:
    repo = Path(sys.argv[1])
    rows = json.loads((HERE / "manifest.json").read_text())
    by_tar = defaultdict(dict)
    for row in rows:
        by_tar[row["tarball"]][row["member"]] = row
    for tar_rel, members in by_tar.items():
        tar_path = repo / tar_rel
        with tarfile.open(tar_path, "r:gz") as src:
            entries = [(m, src.extractfile(m).read() if m.isfile() else None) for m in src]
        seen = 0
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:gz") as dst:
            for info, data in entries:
                row = members.get(info.name)
                if row is not None:
                    if hashlib.sha256(data).hexdigest() != row["original_sha256"]:
                        print(f"{tar_rel}:{info.name} changed upstream; not touching it")
                        return 1
                    data = (HERE / row["fixed_file"]).read_bytes()
                    info.size = len(data)
                    seen += 1
                dst.addfile(info, io.BytesIO(data) if data is not None else None)
        if seen != len(members):
            print(f"{tar_rel}: found {seen} of {len(members)} listed members")
            return 1
        tar_path.write_bytes(buf.getvalue())
        print(f"{tar_rel}: replaced {seen} members")
    return 0


if __name__ == "__main__":
    sys.exit(main())
