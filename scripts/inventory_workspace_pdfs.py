#!/usr/bin/env python3
"""Register inventory-only records of external workspace PDFs.

Does NOT modify or process external PDF files. Registrations are purely
for auditability and workspace inventory.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path


def find_workspace_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def main() -> int:
    workspace = find_workspace_root()
    pdf_files: list[dict[str, str | int]] = []

    for fname in sorted(os.listdir(workspace)):
        if fname.endswith(".pdf"):
            fpath = workspace / fname
            stat = fpath.stat()
            sha256 = hashlib.sha256(fpath.read_bytes()).hexdigest()
            pdf_files.append({
                "filename": fname,
                "path": str(fpath),
                "size_bytes": stat.st_size,
                "sha256": sha256,
                "usage": "INVENTORY_ONLY_DO_NOT_MODIFY",
            })

    output_path = Path(__file__).resolve().parent.parent / "docs" / "workspace_pdf_inventory.json"
    data = {
        "schema_version": "1.0",
        "notice": "Inventory-only registration of external reference PDFs in parent workspace. Files remain un-mutated.",
        "total_pdf_files": len(pdf_files),
        "files": pdf_files,
    }

    output_path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    print(f"Registered {len(pdf_files)} workspace PDFs in {output_path}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
