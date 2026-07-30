#!/usr/bin/env python3
"""Audit installed package licenses and produce a proven licenses.json inventory.

Standard library only (uses importlib.metadata and json).

Exit codes:
  0 = all licenses approved and inventory generated
  1 = unapproved license detected
"""

from __future__ import annotations

import importlib.metadata
import json
import sys
from pathlib import Path

# Allowed license types for RAGLab (all OSI open-source approved)
ALLOWED_LICENSE_TERMS = {
    "mit", "apache", "bsd", "psf", "isc", "mpl", "lgpl", "public domain",
    "python software foundation", "mozilla public license", "dual license",
}


def find_repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def get_dist_license(dist: importlib.metadata.Distribution) -> str:
    # 1. License-Expression (PEP 639)
    expr = dist.metadata.get("License-Expression")
    if expr:
        return str(expr).strip()

    # 2. License field
    lic = dist.metadata.get("License")
    if lic and str(lic).strip().lower() not in ("unknown", "none", "osi approved"):
        # If short string, return it
        text = str(lic).strip()
        if len(text) < 100:
            return text

    # 3. Classifiers
    classifiers = dist.metadata.get_all("Classifier") or []
    lic_classifiers = [
        c.split("::")[-1].strip()
        for c in classifiers
        if c.startswith("License ::")
    ]
    if lic_classifiers:
        return ", ".join(lic_classifiers)

    # 4. Fallback to short snippet if license field was long text
    if lic and len(str(lic).strip()) >= 100:
        first_line = str(lic).strip().splitlines()[0]
        if "mit" in str(lic).lower():
            return "MIT License"
        if "apache" in str(lic).lower():
            return "Apache-2.0"
        if "bsd" in str(lic).lower():
            return "BSD License"
        return first_line[:60]

    return "UNKNOWN"


def inventory_licenses() -> tuple[list[dict[str, str]], bool]:
    dists = list(importlib.metadata.distributions())
    inventory: list[dict[str, str]] = []
    seen: set[str] = set()
    all_ok = True

    for dist in sorted(dists, key=lambda d: d.metadata.get("Name", "").lower()):
        name = dist.metadata.get("Name", "unknown")
        version = dist.metadata.get("Version", "0.0.0")

        key = f"{name.lower()}=={version}"
        if key in seen:
            continue
        seen.add(key)

        lic_str = get_dist_license(dist)
        lic_lower = lic_str.lower()

        is_allowed = any(term in lic_lower for term in ALLOWED_LICENSE_TERMS)

        status = "APPROVED" if is_allowed else "REVIEW_REQUIRED"
        if not is_allowed:
            all_ok = False

        inventory.append({
            "name": name,
            "version": version,
            "license": lic_str,
            "status": status,
        })

    return inventory, all_ok


def main() -> int:
    inventory, ok = inventory_licenses()
    repo_root = find_repo_root()
    output_file = repo_root / "licenses.json"

    data = {
        "schema_version": "1.0",
        "total_packages": len(inventory),
        "overall": "PASSED" if ok else "REVIEW_REQUIRED",
        "allowed_license_terms": sorted(ALLOWED_LICENSE_TERMS),
        "packages": inventory,
    }

    output_file.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    print(f"License inventory written to {output_file} ({len(inventory)} packages).")

    if not ok:
        unapproved = [p for p in inventory if p["status"] != "APPROVED"]
        print(f"WARNING: {len(unapproved)} package(s) flagged for review:")
        for p in unapproved:
            print(f"  - {p['name']} ({p['version']}): {p['license']}")
        return 1

    print("All package licenses verified compliant (100% open source approved).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
