"""Build concatenated Parquet tables from legacy aggregated APC reports."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_AGGREGATED_DIR = PROJECT_ROOT / "data" / "raw" / "TCAT_Ridership_Aggregated"
OUT_DIR = PROJECT_ROOT / "data" / "processed" / "Aggregated_Analysis"


@dataclass(frozen=True)
class ReportSpec:
    label: str
    output_name: str


REPORT_SPECS = [
    ReportSpec("Hour", "apc_by_hour.parquet"),
    ReportSpec("Trip", "apc_by_trip.parquet"),
    ReportSpec("Stop", "apc_by_stop.parquet"),
    ReportSpec("Month", "apc_by_month.parquet"),
    ReportSpec("DoTW", "apc_by_dotw.parquet"),
]


def make_unique(labels: list[str]) -> list[str]:
    seen: dict[str, int] = {}
    unique = []
    for label in labels:
        count = seen.get(label, 0)
        seen[label] = count + 1
        unique.append(label if count == 0 else f"{label}_{count + 1}")
    return unique


def clean_excel_frame(frame: pd.DataFrame) -> pd.DataFrame:
    keep_positions = []
    labels = []
    for idx, column in enumerate(frame.columns):
        label = "" if pd.isna(column) else str(column).strip()
        series = frame.iloc[:, idx]
        if label.startswith("Unnamed") and series.isna().all():
            continue
        keep_positions.append(idx)
        labels.append(label or f"unnamed_{idx + 1}")

    cleaned = frame.iloc[:, keep_positions].dropna(how="all").copy()
    cleaned.columns = make_unique(labels)
    for column in cleaned.select_dtypes(include=["object"]).columns:
        cleaned[column] = cleaned[column].astype("string")
    return cleaned


def read_report(path: Path, route: str, report: ReportSpec) -> pd.DataFrame:
    try:
        frame = pd.read_excel(path, header=1)
    except ImportError as exc:
        raise RuntimeError(
            "Reading aggregated Excel reports requires openpyxl. Install dependencies "
            "with `python -m pip install -r requirements.txt` or use `make install`."
        ) from exc

    frame = clean_excel_frame(frame)
    frame["route"] = route
    frame["source_report"] = report.label
    frame["source_file"] = path.relative_to(PROJECT_ROOT).as_posix()
    front = ["route", "source_report", "source_file"]
    return frame[front + [column for column in frame.columns if column not in front]]


def route_dirs() -> list[Path]:
    return sorted(path for path in RAW_AGGREGATED_DIR.iterdir() if path.is_dir())


def build_report(report: ReportSpec) -> pd.DataFrame:
    frames = []
    for route_dir in route_dirs():
        route = route_dir.name
        candidates = sorted(route_dir.glob(f"{route} APC by {report.label}.*"))
        candidates = [path for path in candidates if path.suffix.lower() in {".xlsx", ".xls"}]
        if not candidates:
            print(f"[WARN] Missing APC by {report.label} file for route {route}")
            continue
        if len(candidates) > 1:
            names = [path.name for path in candidates]
            print(f"[WARN] Multiple APC by {report.label} files for route {route}: {names}; using first")
        frames.append(read_report(candidates[0], route, report))

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def write_parquet(frame: pd.DataFrame, path: Path) -> None:
    try:
        frame.to_parquet(path, index=False, engine="pyarrow", compression="snappy")
    except ImportError as exc:
        raise RuntimeError(
            "Writing Parquet outputs requires pyarrow. Install dependencies with "
            "`python -m pip install -r requirements.txt` or use `make install`."
        ) from exc


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for report in REPORT_SPECS:
        frame = build_report(report)
        out_path = OUT_DIR / report.output_name
        write_parquet(frame, out_path)
        rows.append(
            {
                "path": out_path.relative_to(PROJECT_ROOT).as_posix(),
                "source_report": report.label,
                "rows": len(frame),
                "columns": len(frame.columns),
                "size_mb": round(out_path.stat().st_size / 1_000_000, 3),
            }
        )
        print(f"Wrote {out_path.relative_to(PROJECT_ROOT)} with {len(frame):,} rows")

    manifest = pd.DataFrame(rows)
    manifest.to_csv(OUT_DIR / "aggregated_parquet_manifest.csv", index=False)


if __name__ == "__main__":
    main()
