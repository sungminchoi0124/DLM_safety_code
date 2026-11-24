"""Write the prompt sets of the audit check to AuditCheck/outputs/data/ (git-ignored). Counts and hashes only.

    python AuditCheck/prepare_sets.py                    # downloads JBB-Behaviors (2 CSV) and XSTest (1 CSV)
    python AuditCheck/prepare_sets.py --source_dir DIR   # offline: DIR holds jbb_harmful.csv, jbb_benign.csv,
                                                         #          xstest_prompts.csv (hash-checked)
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from auditcheck import sets  # noqa: E402


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source_dir")
    p.add_argument("--data_dir", default=str(sets.DATA_DIR))
    a = p.parse_args(argv)
    built, manifest = sets.build_all(a.source_dir)
    counts = sets.write_sets(built, manifest, a.data_dir)
    print(json.dumps({"counts": counts, "sha256": {k: v["sha256"][:12] for k, v in manifest.items()}}, indent=1))
    print(f"-> {a.data_dir}")
    return counts


if __name__ == "__main__":
    main()
