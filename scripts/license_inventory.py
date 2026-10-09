"""Record installed distribution declarations, without copying code or license text."""

import hashlib
import json
import sys
from importlib.metadata import distributions
from pathlib import Path


def inventory():
    records = []
    for distribution in distributions():
        name = distribution.metadata["Name"]
        if name == "hansenkit":
            continue
        metadata = distribution.metadata
        license_expression = metadata.get("License-Expression")
        classifiers = [x for x in metadata.get_all("Classifier", []) if x.startswith("License ::")]
        license_files = []
        for file in distribution.files or []:
            if any(token in str(file).lower() for token in ("license", "copying", "notice")):
                path = Path(distribution.locate_file(file))
                if path.is_file():
                    license_files.append(
                        {
                            "file": str(file).replace("\\", "/"),
                            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                        }
                    )
        records.append(
            {
                "name": name,
                "version": distribution.version,
                "license_expression": license_expression,
                "license_classifiers": classifiers,
                "license_field_first_line": (metadata.get("License") or "").splitlines()[:1],
                "project_urls": metadata.get_all("Project-URL", []),
                "license_files": license_files,
                "review_status": (
                    "publisher-declarations-recorded; binary redistribution not reviewed"
                ),
            }
        )
    return sorted(records, key=lambda r: r["name"].lower())


if __name__ == "__main__":
    output = Path(sys.argv[1])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "review_date": "2026-10-10",
                "environment": "Windows CPython 3.12; uv.lock also resolves other CI environments",
                "scope": (
                    "Default, optional LightGBM and development dependencies. "
                    "Not a weight/data license approval."
                ),
                "packages": inventory(),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
