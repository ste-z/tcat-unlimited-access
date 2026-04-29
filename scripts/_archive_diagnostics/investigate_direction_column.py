"""Investigate whether direction_vhist can be used as a stop identifier.

The generated diagnostics show that direction_vhist behaves like a compass
heading. Stop analysis should use Stop_Id_vhist and Previous_Stop_Id_vhist.
"""

from pathlib import Path
import sys
from zipfile import ZipFile

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = PROJECT_ROOT / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

from build_analysis_ready_ridership import normalize_known_route_code

RAW_RIDES_DIR = PROJECT_ROOT / "data" / "raw" / "TCAT_Rides"
GTFS_DIR = PROJECT_ROOT / "data" / "raw" / "TCAT_GTFS"
OUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "_Archive"
    / "Diagnostics"
    / "Direction_Diagnostics"
)

USECOLS = [
    "ts_event",
    "ts",
    "bus",
    "Route_vhist",
    "corrected_route",
    "direction_vhist",
    "Inbound_Outbound_vhist",
    "Stop_Id_vhist",
    "Previous_Stop_Id_vhist",
    "Stop_Name_vhist",
    "lat_vhist",
    "long_vhist",
    "speed_vhist",
    "has_fare",
    "has_apc",
]

DTYPES = {
    "bus": "string",
    "Route_vhist": "string",
    "corrected_route": "string",
    "Inbound_Outbound_vhist": "string",
    "Stop_Id_vhist": "string",
    "Previous_Stop_Id_vhist": "string",
    "Stop_Name_vhist": "string",
    "has_fare": "string",
    "has_apc": "string",
}


def load_rides() -> pd.DataFrame:
    frames = []
    for path in sorted(RAW_RIDES_DIR.glob("canonical_riders_2025_??.csv")):
        frame = pd.read_csv(
            path,
            usecols=lambda col: col in USECOLS,
            dtype=DTYPES,
            parse_dates=["ts", "ts_event"],
            low_memory=False,
        )
        frame["source_month"] = path.stem[-2:]
        frames.append(frame)
    rides = pd.concat(frames, ignore_index=True)
    route_vhist = rides["Route_vhist"].fillna("").astype("string").str.strip().map(
        normalize_known_route_code
    )
    corrected_route = rides["corrected_route"].fillna("").astype("string").str.strip().map(
        normalize_known_route_code
    )
    rides["normalized_route"] = route_vhist.where(route_vhist.ne(""), corrected_route)
    rides["event_time"] = rides["ts_event"].combine_first(rides["ts"])
    rides["direction_num"] = pd.to_numeric(rides["direction_vhist"], errors="coerce")
    rides["stop_id_num"] = pd.to_numeric(rides["Stop_Id_vhist"], errors="coerce")
    rides["previous_stop_id_num"] = pd.to_numeric(rides["Previous_Stop_Id_vhist"], errors="coerce")
    rides["lat_num"] = pd.to_numeric(rides["lat_vhist"], errors="coerce")
    rides["long_num"] = pd.to_numeric(rides["long_vhist"], errors="coerce")
    rides["speed_num"] = pd.to_numeric(rides["speed_vhist"], errors="coerce")
    return rides


def load_gtfs_stop_ids() -> set[int]:
    stop_ids: set[int] = set()
    for zip_path in sorted(GTFS_DIR.glob("*.zip")):
        with ZipFile(zip_path) as zf:
            if "stops.txt" not in zf.namelist():
                continue
            with zf.open("stops.txt") as handle:
                stops = pd.read_csv(handle, usecols=["stop_id"], dtype="string")
            numeric_ids = pd.to_numeric(stops["stop_id"], errors="coerce").dropna().astype(int)
            stop_ids.update(numeric_ids.tolist())
    return stop_ids


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rides = load_rides()
    has_direction = rides["direction_num"].notna()
    direction_rows = rides.loc[has_direction].copy()

    stop_ids = set(pd.to_numeric(rides["Stop_Id_vhist"], errors="coerce").dropna().astype(int))
    gtfs_stop_ids = load_gtfs_stop_ids()

    direction_values = direction_rows["direction_num"]
    direction_int = direction_values.dropna().round().astype(int)
    stop_id_values = pd.to_numeric(direction_rows["Stop_Id_vhist"], errors="coerce")
    previous_stop_id_values = pd.to_numeric(direction_rows["Previous_Stop_Id_vhist"], errors="coerce")
    direction_equals_stop = direction_values.eq(stop_id_values)
    direction_equals_previous_stop = direction_values.eq(previous_stop_id_values)
    nonzero_stop_comparison = (
        direction_values.notna() & stop_id_values.notna() & direction_values.ne(0) & stop_id_values.ne(0)
    )
    nonzero_previous_stop_comparison = (
        direction_values.notna()
        & previous_stop_id_values.notna()
        & direction_values.ne(0)
        & previous_stop_id_values.ne(0)
    )

    summary = pd.Series(
        {
            "total_rows": len(rides),
            "direction_nonmissing_rows": has_direction.sum(),
            "direction_missing_rows": (~has_direction).sum(),
            "direction_unique_values": direction_values.nunique(dropna=True),
            "direction_min": direction_values.min(),
            "direction_max": direction_values.max(),
            "direction_all_integer": bool(np.isclose(direction_values, direction_values.round()).all()),
            "direction_all_between_0_and_359": bool(direction_values.between(0, 359).all()),
            "stop_id_nonmissing_rows_among_direction_rows": stop_id_values.notna().sum(),
            "previous_stop_id_nonmissing_rows_among_direction_rows": previous_stop_id_values.notna().sum(),
            "rows_direction_equals_stop_id": direction_equals_stop.sum(),
            "rows_direction_equals_previous_stop_id": direction_equals_previous_stop.sum(),
            "rows_direction_equals_nonzero_stop_id": (
                direction_equals_stop & nonzero_stop_comparison
            ).sum(),
            "rows_direction_equals_nonzero_previous_stop_id": (
                direction_equals_previous_stop & nonzero_previous_stop_comparison
            ).sum(),
            "share_direction_equals_stop_id_when_both_present": direction_equals_stop.sum()
            / max((direction_values.notna() & stop_id_values.notna()).sum(), 1),
            "share_direction_equals_previous_stop_id_when_both_present": direction_equals_previous_stop.sum()
            / max((direction_values.notna() & previous_stop_id_values.notna()).sum(), 1),
            "share_direction_equals_nonzero_stop_id_when_both_nonzero": (
                direction_equals_stop & nonzero_stop_comparison
            ).sum()
            / max(nonzero_stop_comparison.sum(), 1),
            "share_direction_equals_nonzero_previous_stop_id_when_both_nonzero": (
                direction_equals_previous_stop & nonzero_previous_stop_comparison
            ).sum()
            / max(nonzero_previous_stop_comparison.sum(), 1),
            "direction_values_that_are_observed_stop_ids": len(set(direction_int) & stop_ids),
            "direction_values_that_are_gtfs_stop_ids": len(set(direction_int) & gtfs_stop_ids),
            "observed_stop_id_unique_values": len(stop_ids),
            "gtfs_stop_id_unique_values": len(gtfs_stop_ids),
        },
        name="value",
    )
    summary.to_csv(OUT_DIR / "direction_column_summary.csv", header=True)

    direction_value_counts = (
        direction_rows["direction_num"]
        .value_counts(dropna=False)
        .rename_axis("direction_vhist")
        .rename("rows")
        .reset_index()
        .sort_values("direction_vhist")
    )
    direction_value_counts.to_csv(OUT_DIR / "direction_value_counts.csv", index=False)

    inbound_outbound_counts = (
        direction_rows.groupby(["normalized_route", "Inbound_Outbound_vhist"], dropna=False)
        .agg(
            rows=("direction_num", "size"),
            direction_min=("direction_num", "min"),
            direction_max=("direction_num", "max"),
            direction_unique=("direction_num", "nunique"),
        )
        .reset_index()
        .sort_values(["normalized_route", "Inbound_Outbound_vhist"])
    )
    inbound_outbound_counts.to_csv(OUT_DIR / "inbound_outbound_by_route.csv", index=False)

    sample_cols = [
        "event_time",
        "bus",
        "normalized_route",
        "Inbound_Outbound_vhist",
        "direction_num",
        "Stop_Id_vhist",
        "Previous_Stop_Id_vhist",
        "Stop_Name_vhist",
        "lat_num",
        "long_num",
        "speed_num",
    ]
    direction_rows.loc[:, sample_cols].sort_values(["event_time", "bus"]).head(200).to_csv(
        OUT_DIR / "direction_rows_sample.csv", index=False
    )

    equality_sample = direction_rows.loc[
        direction_values.eq(stop_id_values) | direction_values.eq(previous_stop_id_values),
        sample_cols,
    ].head(200)
    equality_sample.to_csv(OUT_DIR / "direction_equals_stop_id_sample.csv", index=False)

    # A compass-heading interpretation should show direction changing smoothly for a single bus trip,
    # while stop IDs jump among route stops. This sample makes that visible.
    trip_like_sample = (
        direction_rows.dropna(subset=["event_time", "bus", "normalized_route"])
        .sort_values(["bus", "normalized_route", "event_time"])
        .groupby(["bus", "normalized_route"], dropna=False)
        .head(25)
        .loc[:, sample_cols]
    )
    trip_like_sample.to_csv(OUT_DIR / "direction_sequence_sample.csv", index=False)

    print(f"Wrote direction diagnostics to {OUT_DIR.relative_to(PROJECT_ROOT)}")
    print(summary.to_string())
    print("\nInbound/outbound values:")
    print(
        direction_rows["Inbound_Outbound_vhist"]
        .fillna("<missing>")
        .value_counts()
        .rename_axis("Inbound_Outbound_vhist")
        .rename("rows")
        .reset_index()
        .to_string(index=False)
    )
    print("\nFirst direction rows sample:")
    print(direction_rows.loc[:, sample_cols].head(12).to_string(index=False))


if __name__ == "__main__":
    main()
