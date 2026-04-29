"""Build the canonical analysis-ready outputs used by the notebooks.

This script is the one-command entrypoint for the study pipeline. By default it
rebuilds the analysis-ready ridership tables and manifest. Intermediate
route-cleaned files and diagnostics live under data/processed/_Archive. Use
--include-route-cleaning only when the raw monthly ride files or GTFS inputs have
changed, and --include-diagnostics only when you want to refresh archived
diagnostic outputs.
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
from types import ModuleType


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = PROJECT_ROOT / "scripts"
ARCHIVE_COMPONENT_DIR = Path(__file__).resolve().parent


ACTIVE_STEPS = [
    (
        "route cleaning from raw ride files and GTFS",
        ARCHIVE_COMPONENT_DIR / "clean_routes_with_gtfs.py",
        "route_cleaning",
    ),
    (
        "analysis-ready ride event and panel tables",
        SCRIPT_DIR / "build_analysis_ready_ridership.py",
        "default",
    ),
    (
        "processed-output manifest",
        ARCHIVE_COMPONENT_DIR / "write_processed_manifest.py",
        "default",
    ),
]

DIAGNOSTIC_STEPS = [
    (
        "archived cleaning diagnostics",
        SCRIPT_DIR / "_archive_diagnostics" / "profile_ridership_cleaning_opportunities.py",
    ),
    (
        "archived difference-in-discontinuities treatment/control diagnostics",
        SCRIPT_DIR / "_archive_diagnostics" / "didisc_group_diagnostics.py",
    ),
    (
        "archived direction-column diagnostics",
        SCRIPT_DIR / "_archive_diagnostics" / "investigate_direction_column.py",
    ),
]


def load_module(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_step(label: str, path: Path) -> None:
    print(f"\n=== {label} ===")
    module = load_module(path)
    if not hasattr(module, "main"):
        raise AttributeError(f"{path} does not define main()")
    module.main()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rebuild canonical TCAT analysis-ready outputs."
    )
    parser.add_argument(
        "--include-route-cleaning",
        action="store_true",
        help=(
            "Also rebuild monthly route-cleaned ride files from raw ride CSVs and GTFS. "
            "Skip this for ordinary notebook work."
        ),
    )
    parser.add_argument(
        "--include-diagnostics",
        action="store_true",
        help="Also rebuild archived diagnostic outputs.",
    )
    parser.add_argument(
        "--skip-direction",
        action="store_true",
        help="When --include-diagnostics is used, skip the direction-column diagnostic step.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for label, path, step_type in ACTIVE_STEPS:
        if step_type == "route_cleaning" and not args.include_route_cleaning:
            print(f"\n=== {label} ===")
            print("Skipped; pass --include-route-cleaning to rebuild archived intermediates.")
            continue
        run_step(label, path)

    if not args.include_diagnostics:
        print("\n=== archived diagnostics ===")
        print("Skipped; pass --include-diagnostics to rebuild archived diagnostics.")
    else:
        for label, path in DIAGNOSTIC_STEPS:
            if args.skip_direction and path.name == "investigate_direction_column.py":
                print(f"\n=== {label} ===")
                print("Skipped by --skip-direction.")
                continue
            run_step(label, path)
        run_step(
            "processed-output manifest after diagnostics",
            ARCHIVE_COMPONENT_DIR / "write_processed_manifest.py",
        )


if __name__ == "__main__":
    main()
