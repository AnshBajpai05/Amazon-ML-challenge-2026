"""Build <team>_submission.zip in the layout required by the problem statement.

<team>_submission.zip
├── output/{matching_results.tsv, candidate_pairs.tsv}
├── code/business_entity_resolution/{src/, README.md, requirements.txt, reports/*.json|log}
└── Documentation_template.md
"""
from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

SKIP_DIRS = {"__pycache__", ".venv", "venv", ".ipynb_checkpoints", ".pytest_cache", "catboost_info", "cache",
             "ber_cache", "selftest_tmp"}
SKIP_SUFFIX = {".pyc", ".pyo", ".parquet", ".npy", ".pkl", ".bin", ".safetensors", ".pt", ".gz", ".joblib", ".tmp"}


def build_zip(team, out_dir, code_dir, doc_path, zip_dir) -> Path:
    out_dir, code_dir, zip_dir = Path(out_dir), Path(code_dir), Path(zip_dir)
    zip_dir.mkdir(parents=True, exist_ok=True)
    zp = zip_dir / f"{team}_submission.zip"
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
        for name in ("matching_results.tsv", "candidate_pairs.tsv"):
            z.write(out_dir / name, f"output/{name}")
        for p in sorted(code_dir.rglob("*")):
            rel = p.relative_to(code_dir)
            if not p.is_file() or any(part in SKIP_DIRS for part in rel.parts) or p.suffix in SKIP_SUFFIX:
                continue
            z.write(p, f"code/business_entity_resolution/{rel.as_posix()}")
        z.write(doc_path, "Documentation_template.md")
    return zp


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--team", required=True)
    ap.add_argument("--out", required=True, help="folder with the two output TSVs")
    ap.add_argument("--doc", required=True, help="filled Documentation_template.md")
    ap.add_argument("--zip-dir", default=".")
    a = ap.parse_args()
    print(build_zip(a.team, a.out, Path(__file__).resolve().parent.parent, a.doc, a.zip_dir))
