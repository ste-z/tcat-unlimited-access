"""Write a manifest that labels processed files by purpose.

The manifest is intentionally lightweight: it does not rebuild data, it only
documents what exists under data/processed and whether each file is canonical or
archived.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUT_PATH = PROCESSED_DIR / "processed_outputs_manifest.csv"


ACTIVE_TOP_LEVEL = {
    "Analysis_Ready": (
        "canonical",
        "Use these ride-event and 15-minute panel files for modeling.",
    ),
}

ARCHIVE_SECTION_NOTES = {
    "Intermediate": (
        "archived_intermediate",
        "Large route-cleaned intermediate files retained for reproducibility.",
    ),
    "Diagnostics": (
        "archived_diagnostic",
        "Repeatable diagnostic outputs retained for audit but not used as modeling inputs.",
    ),
    "Exploration": (
        "archived_exploration",
        "Older one-off exploration outputs retained for audit but not used by the pipeline.",
    ),
}


def classify(path: Path) -> tuple[str, str]:
    if path.name == "processed_outputs_manifest.csv":
        return "documentation", "Generated inventory of processed outputs."
    if path.name == "README.md":
        return "documentation", "Processed-data documentation."

    relative = path.relative_to(PROCESSED_DIR)
    top = relative.parts[0] if relative.parts else path.name
    if top in ACTIVE_TOP_LEVEL:
        return ACTIVE_TOP_LEVEL[top]
    if top == "_Archive" and len(relative.parts) > 1:
        archive_section = relative.parts[1]
        if archive_section in ARCHIVE_SECTION_NOTES:
            return ARCHIVE_SECTION_NOTES[archive_section]
    return "legacy_or_unclassified", "Review before using in analysis."


def main() -> None:
    rows = []
    for path in sorted(PROCESSED_DIR.rglob("*")):
        if not path.is_file():
            continue
        category, note = classify(path)
        rel_path = path.relative_to(PROCESSED_DIR).as_posix()
        rows.append(
            {
                "path": rel_path,
                "category": category,
                "size_mb": round(path.stat().st_size / 1_000_000, 3),
                "analysis_use": "yes" if category == "canonical" else "no",
                "note": note,
            }
        )

    manifest = pd.DataFrame(rows).sort_values(["category", "path"])
    manifest.to_csv(OUT_PATH, index=False)
    print(f"Wrote {OUT_PATH.relative_to(PROJECT_ROOT)} with {len(manifest):,} files.")
    print(manifest["category"].value_counts().rename_axis("category").rename("files").to_string())


if __name__ == "__main__":
    main()
