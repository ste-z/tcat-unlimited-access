"""Build analysis-ready TCAT ridership data directly from raw rides and GTFS.

This is the single canonical processing script for the study. It does not write
monthly cleaned ride files or route-cleaning audit tables. Instead, it resolves
routes in memory, applies the fare taxonomy, performs the analysis cleaning, and
writes only the modeling outputs in data/processed/Analysis_Ready plus the
processed-data manifest.

Route policy:
- Use Route_vhist when present.
- If Route_vhist is missing, keep corrected_route only when the date-appropriate
  GTFS feed confirms it is a real route.
- Drop through-running strings, GTFS stop IDs in corrected_route, non-GTFS
  values, and missing route values rather than inferring a route.
"""

from __future__ import annotations

import csv
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_RIDES_DIR = PROJECT_ROOT / "data" / "raw" / "TCAT_Rides"
GTFS_DIR = PROJECT_ROOT / "data" / "raw" / "TCAT_GTFS"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUT_DIR = PROCESSED_DIR / "Analysis_Ready"
MANIFEST_PATH = PROCESSED_DIR / "processed_outputs_manifest.csv"

EXCLUDED_TEXT_VALUES = {
    "Auto logoff (no activity)",
    "Direction change",
}

RAW_USECOLS = [
    "tr_seq",
    "ts",
    "ts_event",
    "ts_vhist",
    "bus",
    "route",
    "ttp",
    "amt",
    "text",
    "description",
    "media_text",
    "dropfile_parameter",
    "is_cornell",
    "has_fare",
    "has_apc",
    "canonical_rider",
    "corrected_rider",
    "is_revenue_rider",
    "service_category",
    "service_day",
    "corrected_route",
    "Route_vhist",
    "Route_Name_vhist",
    "Stop_Id_vhist",
    "Stop_Name_vhist",
]

DTYPES = {
    "tr_seq": "string",
    "bus": "string",
    "route": "string",
    "ttp": "string",
    "text": "string",
    "description": "string",
    "media_text": "string",
    "dropfile_parameter": "string",
    "is_cornell": "string",
    "has_fare": "string",
    "has_apc": "string",
    "is_revenue_rider": "string",
    "service_category": "string",
    "service_day": "string",
    "corrected_route": "string",
    "Route_vhist": "string",
    "Route_Name_vhist": "string",
    "Stop_Id_vhist": "string",
    "Stop_Name_vhist": "string",
}

TREATED_ROUTES_PRIMARY = ["30", "81", "32", "82", "51"]
CONTROL_ROUTES_PRIMARY = ["11", "14"]
TREATED_ROUTES_SENSITIVITY = ["37", "31", "52"]
CONTROL_ROUTES_SENSITIVITY = ["13", "14S"]


FARE_TAXONOMY_BY_DESCRIPTION = {
    "LA1 Youth": ("Free", "Free Fares"),
    "RA4  Transfer": ("Free", "Transfer"),
    "Cornell Card": ("Free", "Cornell"),
    "Second Left Arrow 16": ("Free", "Cornell Override"),
    "Ithaca College Pass": ("Free", "IC"),
    "Third Left Arrow 17": ("Free", "IC Override"),
    "Fourth Left Arrow 18": ("Free", "TC3 Override"),
    "RA2  Senior 60 & Up": ("Half Fare", "Ride-based Half Fares"),
    "RA3  Disabled": ("Half Fare", "Ride-based Half Fares"),
    "15 RIDE HALF-FARE TCARD": ("Half Fare", "Ride-based Half Fares"),
    "15 Ride Pass Half Fare Mobile": ("Half Fare", "Ride-based Half Fares"),
    "1 Ride Half Mobile": ("Half Fare", "Ride-based Half Fares"),
    "2 Ride Half Mobile": ("Half Fare", "Ride-based Half Fares"),
    "RA1 Adult 18-59": ("Half Fare", "Ride-based Half Fares"),
    "1 Ride Pass Mobile": ("Regular Fares", "Ride-based Regular Fares"),
    "2 Ride Pass Mobile": ("Regular Fares", "Ride-based Regular Fares"),
    "15 Ride Pass Mobile": ("Regular Fares", "Ride-based Regular Fares"),
    "15 RIDE TCARD": ("Regular Fares", "Ride-based Regular Fares"),
    "PayAsYouGo": ("Regular Fares", "Ride-based Regular Fares"),
    "Pay As You Go": ("Regular Fares", "Ride-based Regular Fares"),
    "ANNUAL PASS": ("Regular Fares", "Period-based Regular Fares"),
    "Monthly Pass Mobile": ("Regular Fares", "Period-based Regular Fares"),
    "Weekly Pass Mobile": ("Regular Fares", "Period-based Regular Fares"),
    "1 Day Pass Mobile": ("Regular Fares", "Period-based Regular Fares"),
    "2 Day Pass Mobile": ("Regular Fares", "Period-based Regular Fares"),
    "5 Day Pass Mobile": ("Regular Fares", "Period-based Regular Fares"),
    "1-DAY TCARD": ("Regular Fares", "Period-based Regular Fares"),
    "2-DAY TCARD": ("Regular Fares", "Period-based Regular Fares"),
    "5-DAY TCARD": ("Regular Fares", "Period-based Regular Fares"),
    "31-DAY TCARD": ("Regular Fares", "Period-based Regular Fares"),
    "Empty TTP": ("Unspecified", "Unspecified / No Description"),
}

FARE_FAMILY_SLUG_BY_LABEL = {
    "Free": "free",
    "Half Fare": "half_fare",
    "Regular Fares": "regular_fares",
    "Unspecified": "unspecified_fare",
    "Unmapped": "unmapped_fare",
}

FARE_CATEGORY_SLUG_BY_LABEL = {
    "Free Fares": "free_fares",
    "Transfer": "transfer",
    "Cornell": "cornell",
    "Cornell Override": "cornell_override",
    "IC": "ic",
    "IC Override": "ic_override",
    "TC3 Override": "tc3_override",
    "Ride-based Half Fares": "ride_based_half_fares",
    "Ride-based Regular Fares": "ride_based_regular_fares",
    "Period-based Regular Fares": "period_based_regular_fares",
    "Unspecified / No Description": "unspecified_fare",
    "Unmapped Fare Description": "unmapped_fare",
}


def clean_value(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    value = str(value).strip()
    if value.upper() in {"", "NULL", "<NA>", "NAN"}:
        return ""
    return value


def clean_text(series: pd.Series) -> pd.Series:
    return series.fillna("").astype("string").str.strip()


def as_bool(series: pd.Series) -> pd.Series:
    return series.astype("string").str.lower().map({"true": True, "false": False}).astype("boolean")


def normalize_known_route_code(route_code: Any) -> str:
    route_code = clean_value(route_code)
    if route_code == "145":
        return "14S"
    return route_code


def parse_service_date_value(service_day: Any, ts: Any) -> int | None:
    service_day_clean = clean_value(service_day)
    if len(service_day_clean) == 6 and service_day_clean.isdigit():
        return int(f"20{service_day_clean}")

    if pd.notna(ts):
        try:
            return int(pd.Timestamp(ts).strftime("%Y%m%d"))
        except ValueError:
            pass

    ts_clean = clean_value(ts)
    if len(ts_clean) >= 10:
        date_part = ts_clean[:10].replace("-", "")
        if len(date_part) == 8 and date_part.isdigit():
            return int(date_part)
    return None


def zip_member_name(zf: zipfile.ZipFile, basename: str) -> str:
    matches = [name for name in zf.namelist() if name.endswith(basename)]
    if not matches:
        raise FileNotFoundError(f"{basename} not found in {zf.filename}")
    return matches[0]


def read_zip_csv(zf: zipfile.ZipFile, basename: str) -> list[dict[str, str]]:
    member = zip_member_name(zf, basename)
    with zf.open(member) as handle:
        text_lines = (line.decode("utf-8-sig") for line in handle)
        return list(csv.DictReader(text_lines))


@dataclass
class GtfsFeed:
    path: Path
    feed_start_date: int
    feed_end_date: int
    feed_version: str
    route_id_to_short: dict[str, str]
    canonical_routes: set[str]
    route_tokens: tuple[str, ...]
    stop_names: dict[str, str]
    stop_routes: dict[str, set[str]]
    split_cache: dict[str, tuple[tuple[str, ...], ...]] = field(default_factory=dict)

    def covers(self, service_date: int) -> bool:
        return self.feed_start_date <= service_date <= self.feed_end_date

    def route_code_to_short(self, route_code: Any) -> str:
        route_code = clean_value(route_code)
        if not route_code:
            return ""
        if route_code in self.route_id_to_short:
            return normalize_known_route_code(self.route_id_to_short[route_code])
        if route_code in self.canonical_routes:
            return normalize_known_route_code(route_code)
        return ""

    def split_through_route(self, route_code: Any) -> tuple[tuple[str, ...], ...]:
        route_code = clean_value(route_code)
        if route_code in self.split_cache:
            return self.split_cache[route_code]
        if not route_code:
            self.split_cache[route_code] = ((),)
            return self.split_cache[route_code]

        splits: list[tuple[str, ...]] = []
        for token in self.route_tokens:
            if route_code.startswith(token):
                token_short = self.route_code_to_short(token)
                if not token_short:
                    continue
                for remaining_split in self.split_through_route(route_code[len(token) :]):
                    splits.append((token_short,) + remaining_split)
        self.split_cache[route_code] = tuple(splits)
        return self.split_cache[route_code]

    def through_route_sequences(self, route_code: Any) -> list[tuple[str, ...]]:
        seen: set[tuple[str, ...]] = set()
        sequences: list[tuple[str, ...]] = []
        for sequence in self.split_through_route(route_code):
            if len(sequence) < 2:
                continue
            if sequence not in seen:
                seen.add(sequence)
                sequences.append(sequence)
        return sequences


def load_gtfs_feed(path: Path) -> GtfsFeed:
    with zipfile.ZipFile(path) as zf:
        feed_info = read_zip_csv(zf, "feed_info.txt")[0]
        routes = read_zip_csv(zf, "routes.txt")
        stops = read_zip_csv(zf, "stops.txt")
        trips = read_zip_csv(zf, "trips.txt")

        route_id_to_short = {
            clean_value(row["route_id"]): normalize_known_route_code(row["route_short_name"])
            for row in routes
        }
        if "145" in route_id_to_short:
            route_id_to_short["145"] = "14S"

        canonical_routes = {normalize_known_route_code(short) for short in route_id_to_short.values()}
        if "145" in route_id_to_short:
            canonical_routes.add("14S")

        trip_route = {
            clean_value(row["trip_id"]): route_id_to_short.get(clean_value(row["route_id"]), "")
            for row in trips
        }
        stop_names = {
            clean_value(row["stop_id"]): clean_value(row.get("stop_name"))
            for row in stops
        }
        stop_routes: dict[str, set[str]] = defaultdict(set)

        stop_times_member = zip_member_name(zf, "stop_times.txt")
        with zf.open(stop_times_member) as handle:
            text_lines = (line.decode("utf-8-sig") for line in handle)
            for row in csv.DictReader(text_lines):
                route_short = trip_route.get(clean_value(row["trip_id"]), "")
                stop_id = clean_value(row["stop_id"])
                if route_short and stop_id:
                    stop_routes[stop_id].add(normalize_known_route_code(route_short))

    return GtfsFeed(
        path=path,
        feed_start_date=int(clean_value(feed_info["feed_start_date"])),
        feed_end_date=int(clean_value(feed_info["feed_end_date"])),
        feed_version=clean_value(feed_info.get("feed_version")),
        route_id_to_short=route_id_to_short,
        canonical_routes=canonical_routes,
        route_tokens=tuple(sorted(route_id_to_short, key=lambda code: (-len(code), code))),
        stop_names=stop_names,
        stop_routes=dict(stop_routes),
    )


def load_gtfs_feeds() -> list[GtfsFeed]:
    feeds = [load_gtfs_feed(path) for path in sorted(GTFS_DIR.glob("*.zip"))]
    return sorted(feeds, key=lambda feed: (feed.feed_start_date, feed.feed_version, feed.path.name))


def select_feed(feeds: list[GtfsFeed], service_date: int | None) -> GtfsFeed | None:
    if service_date is None:
        return None
    candidates = [feed for feed in feeds if feed.covers(service_date)]
    if not candidates:
        return None
    return max(candidates, key=lambda feed: (feed.feed_start_date, feed.feed_version, feed.path.name))


def format_sequences(sequences: list[tuple[str, ...]]) -> str:
    return " | ".join(" + ".join(sequence) for sequence in sequences)


def resolve_missing_vhist_route(
    corrected_route: Any,
    stop_id_vhist: Any,
    feed: GtfsFeed | None,
) -> tuple[str, str, str]:
    corrected_route_clean = clean_value(corrected_route)

    if feed is None:
        return "", "unresolved_no_gtfs_feed", ""
    if not corrected_route_clean:
        return "", "unresolved_missing_route_columns", ""

    corrected_route_as_route = feed.route_code_to_short(corrected_route_clean)
    if corrected_route_as_route:
        return corrected_route_as_route, "corrected_route_gtfs_route", ""

    corrected_route_is_stop = corrected_route_clean in feed.stop_names
    if corrected_route_is_stop:
        return "", "unresolved_corrected_route_is_gtfs_stop_id", ""

    through_sequences = feed.through_route_sequences(corrected_route_clean)
    through_parse = format_sequences(through_sequences)
    if through_sequences:
        return "", "unresolved_through_run_dropped", through_parse

    return "", "unresolved_corrected_route_not_in_gtfs", ""


def add_route_resolution(frame: pd.DataFrame, feeds: list[GtfsFeed]) -> pd.DataFrame:
    frame = frame.copy()
    text_clean_for_route = clean_text(frame["text"])
    frame["route_cleaning_exclusion_reason"] = np.where(
        text_clean_for_route.isin(EXCLUDED_TEXT_VALUES),
        text_clean_for_route,
        "",
    )
    frame["Route_vhist_normalized"] = clean_text(frame["Route_vhist"]).map(
        normalize_known_route_code
    )
    frame["normalized_route"] = pd.Series(pd.NA, index=frame.index, dtype="string")
    frame["normalized_route_source"] = pd.Series(pd.NA, index=frame.index, dtype="string")

    not_excluded = frame["route_cleaning_exclusion_reason"].eq("")
    has_vhist = frame["Route_vhist_normalized"].ne("")
    vhist_mask = not_excluded & has_vhist
    frame.loc[vhist_mask, "normalized_route"] = frame.loc[vhist_mask, "Route_vhist_normalized"]
    frame.loc[vhist_mask, "normalized_route_source"] = "Route_vhist_normalized"

    service_dates = {
        idx: parse_service_date_value(service_day, ts)
        for idx, service_day, ts in frame.loc[
            not_excluded & ~has_vhist,
            ["service_day", "ts"],
        ].itertuples(index=True, name=None)
    }
    feed_by_date = {
        service_date: select_feed(feeds, service_date)
        for service_date in set(service_dates.values())
    }

    for idx, corrected_route, stop_id in frame.loc[
        not_excluded & ~has_vhist,
        ["corrected_route", "Stop_Id_vhist"],
    ].itertuples(index=True, name=None):
        route, source, _through_parse = resolve_missing_vhist_route(
            corrected_route,
            stop_id,
            feed_by_date[service_dates[idx]],
        )
        frame.at[idx, "normalized_route"] = route if route else pd.NA
        frame.at[idx, "normalized_route_source"] = source

    excluded_mask = frame["route_cleaning_exclusion_reason"].ne("")
    frame.loc[excluded_mask, "normalized_route_source"] = "excluded_text_removed"
    return frame


def swiftly_route_to_short(value: Any, route_id_to_short: dict[str, str]) -> str:
    value = normalize_known_route_code(value)
    if not value:
        return ""
    if value in route_id_to_short:
        return normalize_known_route_code(route_id_to_short[value])
    return value


def route_column_diagnostics(frame: pd.DataFrame, feeds: list[GtfsFeed]) -> dict[str, pd.DataFrame]:
    canonical_routes = set().union(*(feed.canonical_routes for feed in feeds))
    route_id_to_short: dict[str, str] = {}
    for feed in feeds:
        route_id_to_short.update(feed.route_id_to_short)

    diag = pd.DataFrame(
        {
            "source_month": frame["source_month"].astype("string"),
            "route": clean_text(frame["route"]).map(normalize_known_route_code),
            "corrected_route": clean_text(frame["corrected_route"]).map(normalize_known_route_code),
            "Route_vhist": clean_text(frame["Route_vhist"]).map(normalize_known_route_code),
            "swiftly_route_id": clean_text(frame["swiftly_route_id"]).map(
                lambda value: swiftly_route_to_short(value, route_id_to_short)
            )
            if "swiftly_route_id" in frame
            else "",
        }
    )

    column_rows = []
    for column in ["route", "corrected_route", "Route_vhist", "swiftly_route_id"]:
        values = diag[column]
        nonblank = values.ne("")
        canonical = values.isin(canonical_routes)
        column_rows.append(
            {
                "route_column": column,
                "nonblank_rows": int(nonblank.sum()),
                "canonical_route_rows": int(canonical.sum()),
                "noncanonical_nonblank_rows": int((nonblank & ~canonical).sum()),
                "unique_nonblank_values": int(values.loc[nonblank].nunique()),
            }
        )
    column_coverage = pd.DataFrame(column_rows)

    pair_rows = []
    columns = ["route", "corrected_route", "Route_vhist", "swiftly_route_id"]
    for left_idx, left in enumerate(columns):
        for right in columns[left_idx + 1 :]:
            left_values = diag[left]
            right_values = diag[right]
            both_canonical = left_values.isin(canonical_routes) & right_values.isin(canonical_routes)
            disagreement = both_canonical & left_values.ne(right_values)
            pair_rows.append(
                {
                    "left_column": left,
                    "right_column": right,
                    "both_canonical_rows": int(both_canonical.sum()),
                    "agreement_rows": int((both_canonical & left_values.eq(right_values)).sum()),
                    "disagreement_rows": int(disagreement.sum()),
                    "disagreement_share_when_both_canonical": (
                        float(disagreement.sum() / both_canonical.sum())
                        if both_canonical.sum()
                        else 0.0
                    ),
                }
            )
    pairwise = pd.DataFrame(pair_rows)

    disagreement_frames = []
    for left, right in [
        ("Route_vhist", "corrected_route"),
        ("Route_vhist", "route"),
        ("corrected_route", "route"),
        ("Route_vhist", "swiftly_route_id"),
    ]:
        mask = (
            diag[left].isin(canonical_routes)
            & diag[right].isin(canonical_routes)
            & diag[left].ne(diag[right])
        )
        if not mask.any():
            continue
        top = (
            diag.loc[mask]
            .groupby([left, right], dropna=False)
            .size()
            .rename("rows")
            .reset_index()
            .sort_values("rows", ascending=False)
            .head(50)
        )
        top["left_column"] = left
        top["right_column"] = right
        top = top.rename(columns={left: "left_value", right: "right_value"})
        disagreement_frames.append(
            top[["left_column", "right_column", "left_value", "right_value", "rows"]]
        )
    top_disagreements = (
        pd.concat(disagreement_frames, ignore_index=True)
        if disagreement_frames
        else pd.DataFrame(
            columns=["left_column", "right_column", "left_value", "right_value", "rows"]
        )
    )

    missing_patterns = (
        diag.assign(
            route_is_canonical=diag["route"].isin(canonical_routes),
            corrected_route_is_canonical=diag["corrected_route"].isin(canonical_routes),
            Route_vhist_is_canonical=diag["Route_vhist"].isin(canonical_routes),
            swiftly_route_id_is_canonical=diag["swiftly_route_id"].isin(canonical_routes),
            route_is_blank=diag["route"].eq(""),
            corrected_route_is_blank=diag["corrected_route"].eq(""),
            Route_vhist_is_blank=diag["Route_vhist"].eq(""),
            swiftly_route_id_is_blank=diag["swiftly_route_id"].eq(""),
        )
        .groupby(
            [
                "route_is_canonical",
                "corrected_route_is_canonical",
                "Route_vhist_is_canonical",
                "swiftly_route_id_is_canonical",
                "route_is_blank",
                "corrected_route_is_blank",
                "Route_vhist_is_blank",
                "swiftly_route_id_is_blank",
            ],
            dropna=False,
        )
        .size()
        .rename("rows")
        .reset_index()
        .sort_values("rows", ascending=False)
    )

    return {
        "route_column_coverage": column_coverage,
        "route_column_pairwise_disagreements": pairwise,
        "route_column_top_disagreements": top_disagreements,
        "route_column_missing_patterns": missing_patterns,
    }


def load_raw_rides_with_routes() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame]]:
    feeds = load_gtfs_feeds()
    frames = []
    route_rows = []
    route_diag_frames: dict[str, list[pd.DataFrame]] = defaultdict(list)
    for path in sorted(RAW_RIDES_DIR.glob("canonical_riders_2025_??.csv")):
        frame = pd.read_csv(
            path,
            usecols=lambda col: col in RAW_USECOLS,
            dtype=DTYPES,
            parse_dates=["ts", "ts_event", "ts_vhist"],
            low_memory=False,
        )
        frame["source_month"] = path.stem[-2:]
        frame = add_route_resolution(frame, feeds)
        diagnostics = route_column_diagnostics(frame, feeds)
        for name, diag_frame in diagnostics.items():
            route_diag_frames[name].append(diag_frame.assign(source_month=path.stem[-2:]))

        source_counts = frame["normalized_route_source"].fillna("<missing>").value_counts()
        for source, rows in source_counts.items():
            route_rows.append(
                {
                    "source_month": path.stem[-2:],
                    "normalized_route_source": source,
                    "rows": int(rows),
                }
            )

        keep = (
            frame["route_cleaning_exclusion_reason"].eq("")
            & frame["normalized_route"].notna()
            & frame["normalized_route"].astype("string").str.strip().ne("")
        )
        frames.append(frame.loc[keep].copy())

    if not frames:
        raise FileNotFoundError(f"No raw ride CSVs found in {RAW_RIDES_DIR}")
    combined_diagnostics = {
        name: pd.concat(frames_for_name, ignore_index=True)
        for name, frames_for_name in route_diag_frames.items()
    }
    return pd.concat(frames, ignore_index=True), pd.DataFrame(route_rows), combined_diagnostics


def add_fare_taxonomy(
    frame: pd.DataFrame,
    description_col: str = "description_clean",
    text_col: str = "text_clean",
) -> pd.DataFrame:
    frame = frame.copy()
    description = frame[description_col].fillna("").astype("string").str.strip()
    text = frame[text_col].fillna("").astype("string").str.strip()
    taxonomy = description.map(FARE_TAXONOMY_BY_DESCRIPTION)

    frame["fare_family"] = taxonomy.map(lambda value: value[0] if isinstance(value, tuple) else pd.NA)
    frame["fare_category"] = taxonomy.map(lambda value: value[1] if isinstance(value, tuple) else pd.NA)

    blank_description = description.eq("")
    blank_got_fare = blank_description & text.eq("Got fare")
    frame.loc[blank_description, "fare_family"] = "Unspecified"
    frame.loc[blank_description, "fare_category"] = "Unspecified / No Description"
    frame.loc[blank_got_fare, "fare_family"] = "Regular Fares"
    frame.loc[blank_got_fare, "fare_category"] = "Ride-based Regular Fares"

    frame["fare_family"] = frame["fare_family"].fillna("Unmapped")
    frame["fare_category"] = frame["fare_category"].fillna("Unmapped Fare Description")
    frame["fare_family_slug"] = frame["fare_family"].map(FARE_FAMILY_SLUG_BY_LABEL).fillna("unmapped_fare")
    frame["fare_category_slug"] = frame["fare_category"].map(FARE_CATEGORY_SLUG_BY_LABEL).fillna("unmapped_fare")
    frame["fare_description_unmapped"] = frame["fare_category"].eq("Unmapped Fare Description")
    return frame


def fare_category_lookup_frame() -> pd.DataFrame:
    rows = [
        {
            "description_clean": description,
            "fare_family": family,
            "fare_category": category,
        }
        for description, (family, category) in FARE_TAXONOMY_BY_DESCRIPTION.items()
    ]
    rows.append(
        {
            "description_clean": "",
            "fare_family": "Unspecified",
            "fare_category": "Unspecified / No Description",
        }
    )
    lookup = pd.DataFrame(rows)
    lookup["fare_family_slug"] = lookup["fare_family"].map(FARE_FAMILY_SLUG_BY_LABEL)
    lookup["fare_category_slug"] = lookup["fare_category"].map(FARE_CATEGORY_SLUG_BY_LABEL)
    return lookup.sort_values(["fare_family", "fare_category", "description_clean"]).reset_index(drop=True)


def add_clean_fields(rides: pd.DataFrame) -> pd.DataFrame:
    rides = rides.copy()
    rides["route_clean"] = rides["normalized_route"].astype("string").str.strip()
    rides["route_name_clean"] = (
        clean_text(rides["Route_Name_vhist"])
        .str.replace(r"\s+", " ", regex=True)
        .replace("", pd.NA)
    )
    rides["route_name_clean"] = rides["route_name_clean"].fillna(
        "Route " + rides["route_clean"].astype("string")
    )
    rides["description_clean"] = clean_text(rides["description"])
    rides["media_text_clean"] = clean_text(rides["media_text"])
    rides["text_clean"] = clean_text(rides["text"])
    rides["dropfile_parameter_clean"] = clean_text(rides["dropfile_parameter"])
    rides["stop_id_clean"] = clean_text(rides["Stop_Id_vhist"]).replace({"": pd.NA, "0": pd.NA})
    rides["stop_name_raw_clean"] = (
        clean_text(rides["Stop_Name_vhist"])
        .str.replace(r"\s+", " ", regex=True)
        .replace("", pd.NA)
    )

    rides["has_fare_bool"] = as_bool(rides["has_fare"])
    rides["has_apc_bool"] = as_bool(rides["has_apc"])
    rides["is_cornell_bool"] = as_bool(rides["is_cornell"])
    rides["is_revenue_bool"] = as_bool(rides["is_revenue_rider"])
    rides["corrected_rider_num"] = pd.to_numeric(rides["corrected_rider"], errors="coerce").fillna(0)
    rides["canonical_rider_num"] = pd.to_numeric(rides["canonical_rider"], errors="coerce").fillna(0)
    rides["amt_num"] = pd.to_numeric(rides["amt"], errors="coerce")

    rides["event_time"] = rides["ts_event"].combine_first(rides["ts"]).combine_first(rides["ts_vhist"])
    rides["event_time_source"] = np.select(
        [rides["ts_event"].notna(), rides["ts"].notna(), rides["ts_vhist"].notna()],
        ["ts_event", "ts", "ts_vhist"],
        default="missing",
    )
    rides["service_date"] = pd.to_datetime(rides["service_day"], format="%y%m%d", errors="coerce")
    rides["event_date"] = rides["event_time"].dt.normalize()
    rides["hour"] = rides["event_time"].dt.hour
    rides["minute"] = rides["event_time"].dt.minute
    rides["dow"] = rides["event_time"].dt.dayofweek
    rides["weekday"] = rides["dow"] < 5
    rides["weekend"] = ~rides["weekday"]
    rides["minutes_since_18"] = (rides["hour"] - 18) * 60 + rides["minute"]
    rides["post_6pm"] = rides["weekday"] & rides["minutes_since_18"].ge(0)
    rides["free_policy_period"] = rides["weekend"] | rides["post_6pm"]
    rides["weekday_near_6pm"] = rides["weekday"] & rides["minutes_since_18"].between(-180, 179)
    rides["time_bin_15min"] = rides["event_time"].dt.floor("15min")

    rides["fare_group"] = np.select(
        [
            rides["description_clean"].eq("Cornell Card"),
            rides["description_clean"].eq("Second Left Arrow 16"),
            rides["description_clean"].eq(""),
        ],
        ["cornell_card", "cornell_override", "blank_or_got_fare"],
        default="other_fare",
    )
    rides = add_fare_taxonomy(rides)
    rides["auto_logoff_record"] = rides["text_clean"].eq("Auto logoff (no activity)")
    rides["direction_change_record"] = rides["text_clean"].eq("Direction change")
    rides["excluded_text_record"] = rides["auto_logoff_record"] | rides["direction_change_record"]
    rides["empty_ttp_record"] = rides["description_clean"].eq("Empty TTP")
    rides["dropfile_exception"] = rides["dropfile_parameter_clean"].eq("EXCEPTION")
    rides["fare_record_missing_description"] = (
        rides["has_fare_bool"].fillna(False)
        & rides["description_clean"].eq("")
        & rides["text_clean"].ne("")
    )
    rides["apc_only_missing_fare_description"] = (
        ~rides["has_fare_bool"].fillna(False)
        & rides["has_apc_bool"].fillna(False)
        & rides["description_clean"].eq("")
    )
    rides["invalid_analysis_record"] = rides["excluded_text_record"] | rides["empty_ttp_record"]
    rides["cornell_card_board"] = rides["fare_group"].eq("cornell_card").astype(int)
    rides["cornell_override_board"] = rides["fare_group"].eq("cornell_override").astype(int)
    rides["other_fare_board"] = rides["fare_group"].eq("other_fare").astype(int)
    rides["blank_or_got_fare_board"] = rides["fare_group"].eq("blank_or_got_fare").astype(int)
    for slug in FARE_FAMILY_SLUG_BY_LABEL.values():
        rides[f"fare_family_{slug}_board"] = rides["fare_family_slug"].eq(slug).astype(int)
    for slug in FARE_CATEGORY_SLUG_BY_LABEL.values():
        rides[f"fare_category_{slug}_board"] = rides["fare_category_slug"].eq(slug).astype(int)
    rides["off_policy_cornell_override"] = (
        rides["fare_group"].eq("cornell_override") & ~rides["free_policy_period"]
    )

    rides["route_group_primary"] = np.select(
        [
            rides["route_clean"].isin(TREATED_ROUTES_PRIMARY),
            rides["route_clean"].isin(CONTROL_ROUTES_PRIMARY),
            rides["route_clean"].isin(TREATED_ROUTES_SENSITIVITY),
            rides["route_clean"].isin(CONTROL_ROUTES_SENSITIVITY),
        ],
        [
            "treated_high_cornell_route",
            "control_low_cornell_route",
            "treated_sensitivity_route",
            "control_sensitivity_route",
        ],
        default="exclude_or_route_stop_only",
    )
    rides["route_treated_primary"] = rides["route_group_primary"].eq("treated_high_cornell_route")
    rides["route_control_primary"] = rides["route_group_primary"].eq("control_low_cornell_route")
    return rides


def add_stop_canonicalization(rides: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    stop_counts = (
        rides.dropna(subset=["stop_id_clean", "stop_name_raw_clean"])
        .groupby(["stop_id_clean", "stop_name_raw_clean"], dropna=False)
        .size()
        .rename("rows")
        .reset_index()
        .sort_values(["stop_id_clean", "rows", "stop_name_raw_clean"], ascending=[True, False, True])
    )
    stop_lookup = stop_counts.drop_duplicates("stop_id_clean").rename(
        columns={"stop_name_raw_clean": "canonical_stop_name", "rows": "canonical_name_rows"}
    )
    variants = stop_counts.groupby("stop_id_clean").agg(
        stop_name_variant_count=("stop_name_raw_clean", "nunique"),
        total_named_rows=("rows", "sum"),
        stop_name_examples=("stop_name_raw_clean", lambda s: " | ".join(s.astype(str).head(5))),
    )
    stop_lookup = (
        stop_lookup[["stop_id_clean", "canonical_stop_name", "canonical_name_rows"]]
        .merge(variants, on="stop_id_clean", how="left")
        .sort_values(["stop_name_variant_count", "total_named_rows"], ascending=[False, False])
    )
    rides = rides.merge(
        stop_lookup[["stop_id_clean", "canonical_stop_name", "stop_name_variant_count"]],
        on="stop_id_clean",
        how="left",
    )
    return rides, stop_lookup


def flag_duplicate_fare_events(rides: pd.DataFrame) -> pd.DataFrame:
    rides = rides.copy()
    keys = ["event_time", "bus", "tr_seq", "text_clean", "description_clean", "route_clean"]
    possible = rides["has_fare_bool"].fillna(False) & rides[keys].notna().all(axis=1)
    rides["duplicate_fare_event"] = False
    rides.loc[possible, "duplicate_fare_event"] = rides.loc[possible].duplicated(keys, keep="first")
    return rides


def route_stop_groups(rides: pd.DataFrame) -> pd.DataFrame:
    sample = rides.loc[
        ~rides["duplicate_fare_event"]
        & rides["weekday_near_6pm"]
        & rides["route_clean"].notna()
        & rides["stop_id_clean"].notna()
    ].copy()
    sample["pre_cutoff"] = sample["minutes_since_18"] < 0
    grouped = sample.groupby(["route_clean", "stop_id_clean"], dropna=False)
    out = grouped.agg(
        canonical_stop_name=("canonical_stop_name", "first"),
        boards=("corrected_rider_num", "sum"),
        rows=("route_clean", "size"),
        pre_boards=("pre_cutoff", lambda s: sample.loc[s.index, "corrected_rider_num"].where(s, 0).sum()),
        post_boards=("pre_cutoff", lambda s: sample.loc[s.index, "corrected_rider_num"].where(~s, 0).sum()),
        pre_cornell_card_rows=("fare_group", lambda s: ((s == "cornell_card") & sample.loc[s.index, "pre_cutoff"]).sum()),
        pre_rows=("pre_cutoff", "sum"),
        post_rows=("pre_cutoff", lambda s: (~s).sum()),
    ).reset_index()
    out["pre_cornell_card_share"] = out["pre_cornell_card_rows"] / out["pre_rows"].replace(0, np.nan)
    out["has_balanced_near6_sample"] = (out["pre_boards"] >= 50) & (out["post_boards"] >= 50)
    out["route_stop_group"] = np.select(
        [
            out["has_balanced_near6_sample"] & out["pre_cornell_card_share"].ge(0.60),
            out["has_balanced_near6_sample"] & out["pre_cornell_card_share"].le(0.15),
        ],
        ["treated_high_cornell_route_stop", "control_low_cornell_route_stop"],
        default="exclude_or_sensitivity",
    )
    return out.sort_values(["route_stop_group", "boards"], ascending=[True, False])


def panelize(rides: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    fare_dummies = {
        "cornell_card_boards": "cornell_card_board",
        "cornell_override_boards": "cornell_override_board",
        "other_fare_boards": "other_fare_board",
        "blank_or_got_fare_boards": "blank_or_got_fare_board",
        "off_policy_cornell_override_boards": "off_policy_cornell_override",
        "dropfile_exception_boards": "dropfile_exception",
        "fare_record_missing_description_boards": "fare_record_missing_description",
        "apc_only_missing_fare_description_boards": "apc_only_missing_fare_description",
    }
    family_output_names = {
        "free": "free_boards",
        "half_fare": "half_fare_boards",
        "regular_fares": "regular_fares_boards",
        "unspecified_fare": "unspecified_family_boards",
        "unmapped_fare": "unmapped_family_boards",
    }
    for slug in FARE_FAMILY_SLUG_BY_LABEL.values():
        fare_dummies[family_output_names.get(slug, f"{slug}_family_boards")] = (
            f"fare_family_{slug}_board"
        )
    for slug in FARE_CATEGORY_SLUG_BY_LABEL.values():
        fare_dummies[f"{slug}_boards"] = f"fare_category_{slug}_board"
    grouped = rides.groupby(group_cols, dropna=False)
    panel = grouped.agg(
        boards=("corrected_rider_num", "sum"),
        rows=("route_clean", "size"),
        has_stop_id_boards=("stop_id_clean", lambda s: s.notna().sum()),
    ).reset_index()
    for out_col, source_col in fare_dummies.items():
        counts = grouped[source_col].sum().rename(out_col).reset_index()
        panel = panel.merge(counts, on=group_cols, how="left")
    panel["non_cornell_card_boards"] = panel["boards"] - panel["cornell_card_boards"]
    panel["event_date"] = pd.to_datetime(panel["time_bin_15min"]).dt.normalize()
    panel["hour"] = pd.to_datetime(panel["time_bin_15min"]).dt.hour
    panel["minute"] = pd.to_datetime(panel["time_bin_15min"]).dt.minute
    panel["dow"] = pd.to_datetime(panel["time_bin_15min"]).dt.dayofweek
    panel["weekday"] = panel["dow"] < 5
    panel["weekend"] = ~panel["weekday"]
    panel["minutes_since_18"] = (panel["hour"] - 18) * 60 + panel["minute"]
    panel["post_6pm"] = panel["weekday"] & panel["minutes_since_18"].ge(0)
    panel["free_policy_period"] = panel["weekend"] | panel["post_6pm"]
    panel["weekday_near_6pm"] = panel["weekday"] & panel["minutes_since_18"].between(-180, 179)
    return panel


def build_ride_events_table(analysis_rides: pd.DataFrame) -> pd.DataFrame:
    ride_events = analysis_rides.loc[
        analysis_rides["stop_id_clean"].notna()
        & analysis_rides["canonical_stop_name"].notna()
    ].copy()
    return (
        pd.DataFrame(
            {
                "event_time": ride_events["event_time"],
                "service_date": ride_events["service_date"],
                "source_month": ride_events["source_month"],
                "route_id": ride_events["route_clean"].astype("string"),
                "route_name": ride_events["route_name_clean"].astype("string"),
                "route_group_primary": ride_events["route_group_primary"].astype("string"),
                "stop_id": ride_events["stop_id_clean"].astype("string"),
                "stop_name": ride_events["canonical_stop_name"].astype("string"),
                "fare_family": ride_events["fare_family"].astype("string"),
                "fare_category": ride_events["fare_category"].astype("string"),
                "fare_category_slug": ride_events["fare_category_slug"].astype("string"),
                "fare_description": ride_events["description_clean"].astype("string"),
                "fare_event_text": ride_events["text_clean"].astype("string"),
                "boarding_count": ride_events["corrected_rider_num"].astype("Int64"),
                "has_fare": ride_events["has_fare_bool"].astype("boolean"),
                "has_apc": ride_events["has_apc_bool"].astype("boolean"),
                "dropfile_exception": ride_events["dropfile_exception"].astype("boolean"),
                "fare_record_missing_description": ride_events[
                    "fare_record_missing_description"
                ].astype("boolean"),
                "apc_only_missing_fare_description": ride_events[
                    "apc_only_missing_fare_description"
                ].astype("boolean"),
                "off_policy_cornell_override": ride_events[
                    "off_policy_cornell_override"
                ].astype("boolean"),
            }
        )
        .sort_values(["event_time", "route_id", "stop_id"])
        .reset_index(drop=True)
    )


def write_parquet(frame: pd.DataFrame, path: Path) -> None:
    try:
        frame.to_parquet(path, index=False, engine="pyarrow", compression="snappy")
    except ImportError as exc:
        raise RuntimeError(
            "Writing Parquet outputs requires pyarrow. Install dependencies with "
            "`python -m pip install -r requirements.txt` or use the local "
            "`make install` target."
        ) from exc


def classify_processed_file(path: Path) -> tuple[str, str]:
    if path.name == "processed_outputs_manifest.csv":
        return "documentation", "Generated inventory of processed outputs."
    if path.name == "README.md":
        return "documentation", "Processed-data documentation."

    relative = path.relative_to(PROCESSED_DIR)
    top = relative.parts[0] if relative.parts else path.name
    if top == "Analysis_Ready":
        return "canonical", "Use these ride-event and 15-minute panel files for modeling."
    if top == "Aggregated_Analysis":
        return (
            "legacy_aggregated",
            "Concatenated Parquet copies of older aggregated APC reports.",
        )
    if top == "_Archive" and len(relative.parts) > 1:
        archive_section = relative.parts[1]
        if archive_section == "Intermediate":
            return "archived_intermediate", "Old intermediate files retained only until deleted."
        if archive_section == "Diagnostics":
            return "archived_diagnostic", "Diagnostic outputs retained for audit, not modeling."
        if archive_section == "Exploration":
            return "archived_exploration", "Older notebook exploration outputs retained for audit."
    return "legacy_or_unclassified", "Review before using in analysis."


def write_processed_manifest() -> None:
    rows = []
    for path in sorted(PROCESSED_DIR.rglob("*")):
        if not path.is_file():
            continue
        category, note = classify_processed_file(path)
        rows.append(
            {
                "path": path.relative_to(PROCESSED_DIR).as_posix(),
                "category": category,
                "size_mb": round(path.stat().st_size / 1_000_000, 3),
                "analysis_use": "yes" if category == "canonical" else "no",
                "note": note,
            }
        )
    manifest = pd.DataFrame(rows).sort_values(["category", "path"])
    manifest.to_csv(MANIFEST_PATH, index=False)
    print(f"Wrote {MANIFEST_PATH.relative_to(PROJECT_ROOT)} with {len(manifest):,} files.")


def write_analysis_outputs() -> pd.Series:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    raw_rides, route_resolution, route_column_diags = load_raw_rides_with_routes()
    rides = add_clean_fields(raw_rides)
    rides, stop_lookup = add_stop_canonicalization(rides)
    rides = flag_duplicate_fare_events(rides)
    route_stop_lookup = route_stop_groups(rides)
    rides = rides.merge(
        route_stop_lookup[
            [
                "route_clean",
                "stop_id_clean",
                "pre_cornell_card_share",
                "has_balanced_near6_sample",
                "route_stop_group",
            ]
        ],
        on=["route_clean", "stop_id_clean"],
        how="left",
    )
    rides["route_stop_group"] = rides["route_stop_group"].fillna("route_level_only_no_stop")

    analysis_rides = rides.loc[
        rides["event_time"].notna()
        & rides["route_clean"].notna()
        & rides["is_revenue_bool"].fillna(False)
        & ~rides["invalid_analysis_record"]
        & ~rides["duplicate_fare_event"]
    ].copy()

    route_group_lookup = (
        analysis_rides.groupby(["route_clean", "route_group_primary"], dropna=False)
        .agg(boards=("corrected_rider_num", "sum"), rows=("route_clean", "size"))
        .reset_index()
        .sort_values("boards", ascending=False)
    )

    route_panel = panelize(
        analysis_rides,
        ["time_bin_15min", "route_clean", "route_group_primary"],
    )
    route_stop_panel = panelize(
        analysis_rides.loc[analysis_rides["stop_id_clean"].notna()].copy(),
        [
            "time_bin_15min",
            "route_clean",
            "stop_id_clean",
            "canonical_stop_name",
            "route_stop_group",
        ],
    )
    ride_events = build_ride_events_table(analysis_rides)

    route_resolution_summary = (
        route_resolution.groupby("normalized_route_source", dropna=False)["rows"]
        .sum()
        .rename("rows")
        .reset_index()
        .sort_values("rows", ascending=False)
    )
    route_resolution_by_month = route_resolution.sort_values(
        ["source_month", "normalized_route_source"]
    )
    route_column_coverage = (
        route_column_diags["route_column_coverage"]
        .groupby("route_column", dropna=False)
        .agg(
            nonblank_rows=("nonblank_rows", "sum"),
            canonical_route_rows=("canonical_route_rows", "sum"),
            noncanonical_nonblank_rows=("noncanonical_nonblank_rows", "sum"),
            unique_nonblank_values=("unique_nonblank_values", "max"),
        )
        .reset_index()
        .sort_values("route_column")
    )
    route_column_pairwise = (
        route_column_diags["route_column_pairwise_disagreements"]
        .groupby(["left_column", "right_column"], dropna=False)
        .agg(
            both_canonical_rows=("both_canonical_rows", "sum"),
            agreement_rows=("agreement_rows", "sum"),
            disagreement_rows=("disagreement_rows", "sum"),
        )
        .reset_index()
        .sort_values(["left_column", "right_column"])
    )
    route_column_pairwise["disagreement_share_when_both_canonical"] = (
        route_column_pairwise["disagreement_rows"]
        / route_column_pairwise["both_canonical_rows"].replace(0, np.nan)
    ).fillna(0)
    route_column_top_disagreements = (
        route_column_diags["route_column_top_disagreements"]
        .groupby(["left_column", "right_column", "left_value", "right_value"], dropna=False)
        .agg(rows=("rows", "sum"))
        .reset_index()
        .sort_values("rows", ascending=False)
    )
    route_column_missing_patterns = (
        route_column_diags["route_column_missing_patterns"]
        .drop(columns=["source_month"], errors="ignore")
        .groupby(
            [
                "route_is_canonical",
                "corrected_route_is_canonical",
                "Route_vhist_is_canonical",
                "swiftly_route_id_is_canonical",
                "route_is_blank",
                "corrected_route_is_blank",
                "Route_vhist_is_blank",
                "swiftly_route_id_is_blank",
            ],
            dropna=False,
        )
        .agg(rows=("rows", "sum"))
        .reset_index()
        .sort_values("rows", ascending=False)
    )

    raw_route_rows = int(route_resolution["rows"].sum())
    unresolved_route_rows = int(
        route_resolution.loc[
            route_resolution["normalized_route_source"].astype(str).str.startswith("unresolved"),
            "rows",
        ].sum()
    )
    route_excluded_text_rows = int(
        route_resolution.loc[
            route_resolution["normalized_route_source"].eq("excluded_text_removed"),
            "rows",
        ].sum()
    )

    summary = pd.Series(
        {
            "raw_input_rows": raw_route_rows,
            "route_unresolved_rows": unresolved_route_rows,
            "route_excluded_text_rows": route_excluded_text_rows,
            "route_column_disagreement_route_vhist_corrected_route_rows": route_column_pairwise.loc[
                (route_column_pairwise["left_column"].eq("corrected_route"))
                & (route_column_pairwise["right_column"].eq("Route_vhist")),
                "disagreement_rows",
            ].sum(),
            "route_column_disagreement_route_vhist_raw_route_rows": route_column_pairwise.loc[
                (route_column_pairwise["left_column"].eq("route"))
                & (route_column_pairwise["right_column"].eq("Route_vhist")),
                "disagreement_rows",
            ].sum(),
            "input_rows": len(rides),
            "analysis_rows": len(analysis_rides),
            "dropped_duplicate_fare_event_rows": rides["duplicate_fare_event"].sum(),
            "dropped_empty_ttp_rows": rides["empty_ttp_record"].sum(),
            "dropped_excluded_text_rows_surviving_route_cleaning": rides["excluded_text_record"].sum(),
            "dropfile_exception_rows_kept": analysis_rides["dropfile_exception"].sum(),
            "fare_record_missing_description_rows_kept": analysis_rides[
                "fare_record_missing_description"
            ].sum(),
            "apc_only_missing_fare_description_rows_kept": analysis_rides[
                "apc_only_missing_fare_description"
            ].sum(),
            "missing_event_time_rows": rides["event_time"].isna().sum(),
            "missing_route_rows": rides["route_clean"].isna().sum(),
            "missing_stop_id_rows": rides["stop_id_clean"].isna().sum(),
            "route_15min_panel_rows": len(route_panel),
            "route_stop_15min_panel_rows": len(route_stop_panel),
            "ride_events_analysis_ready_rows": len(ride_events),
            "treated_route_stop_cells": route_stop_lookup["route_stop_group"].eq(
                "treated_high_cornell_route_stop"
            ).sum(),
            "control_route_stop_cells": route_stop_lookup["route_stop_group"].eq(
                "control_low_cornell_route_stop"
            ).sum(),
        },
        name="value",
    )

    summary.to_csv(OUT_DIR / "analysis_ready_cleaning_summary.csv", header=True)
    route_resolution_summary.to_csv(OUT_DIR / "route_resolution_summary.csv", index=False)
    route_resolution_by_month.to_csv(OUT_DIR / "route_resolution_by_month.csv", index=False)
    route_column_coverage.to_csv(OUT_DIR / "route_column_coverage.csv", index=False)
    route_column_pairwise.to_csv(
        OUT_DIR / "route_column_pairwise_disagreements.csv",
        index=False,
    )
    route_column_top_disagreements.to_csv(
        OUT_DIR / "route_column_top_disagreements.csv",
        index=False,
    )
    route_column_missing_patterns.to_csv(
        OUT_DIR / "route_column_missing_patterns.csv",
        index=False,
    )
    stop_lookup.to_csv(OUT_DIR / "stop_name_lookup.csv", index=False)
    fare_category_lookup_frame().to_csv(OUT_DIR / "fare_category_lookup.csv", index=False)

    issue_columns = [
        "excluded_text_record",
        "empty_ttp_record",
        "dropfile_exception",
        "fare_record_missing_description",
        "apc_only_missing_fare_description",
        "invalid_analysis_record",
        "off_policy_cornell_override",
    ]
    pd.DataFrame(
        [
            {
                "issue": issue,
                "input_rows": int(rides[issue].sum()),
                "analysis_rows": int(analysis_rides[issue].sum()) if issue in analysis_rides else 0,
                "input_corrected_riders": float(
                    rides.loc[rides[issue], "corrected_rider_num"].sum()
                ),
                "analysis_corrected_riders": float(
                    analysis_rides.loc[analysis_rides[issue], "corrected_rider_num"].sum()
                )
                if issue in analysis_rides
                else 0,
            }
            for issue in issue_columns
        ]
    ).to_csv(OUT_DIR / "analysis_record_issue_summary.csv", index=False)
    (
        analysis_rides.groupby(
            [
                "fare_family",
                "fare_family_slug",
                "fare_category",
                "fare_category_slug",
                "description_clean",
            ],
            dropna=False,
        )
        .agg(boards=("corrected_rider_num", "sum"), rows=("fare_category", "size"))
        .reset_index()
        .sort_values(["fare_family", "fare_category", "boards"], ascending=[True, True, False])
        .to_csv(OUT_DIR / "fare_category_description_summary.csv", index=False)
    )
    route_group_lookup.to_csv(OUT_DIR / "route_group_lookup.csv", index=False)
    route_stop_lookup.to_csv(OUT_DIR / "route_stop_group_lookup.csv", index=False)
    route_panel.to_csv(OUT_DIR / "route_15min_panel.csv", index=False)
    route_stop_panel.to_csv(OUT_DIR / "route_stop_15min_panel.csv", index=False)
    ride_events.to_csv(
        OUT_DIR / "ride_events_analysis_ready.csv.gz",
        index=False,
        compression="gzip",
    )
    write_parquet(route_panel, OUT_DIR / "route_15min_panel.parquet")
    write_parquet(route_stop_panel, OUT_DIR / "route_stop_15min_panel.parquet")
    write_parquet(ride_events, OUT_DIR / "ride_events_analysis_ready.parquet")
    return summary


def main() -> None:
    summary = write_analysis_outputs()
    write_processed_manifest()
    print(f"Wrote analysis-ready ridership outputs to {OUT_DIR.relative_to(PROJECT_ROOT)}")
    print(summary.to_string())


if __name__ == "__main__":
    main()
