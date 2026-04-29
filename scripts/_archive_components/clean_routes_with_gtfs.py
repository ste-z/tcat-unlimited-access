"""Resolve TCAT raw ride route fields against GTFS route and stop metadata."""

from __future__ import annotations

import csv
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_RIDES_DIR = PROJECT_ROOT / "data" / "raw" / "TCAT_Rides"
GTFS_DIR = PROJECT_ROOT / "data" / "raw" / "TCAT_GTFS"
OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "_Archive"
    / "Intermediate"
    / "TCAT_Rides_Route_Cleaned"
)
CLEANED_RIDES_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "_Archive"
    / "Intermediate"
    / "TCAT_Rides_Cleaned"
)

EXCLUDED_TEXT_VALUES = {
    "Auto logoff (no activity)",
    "Direction change",
}

ROUTE_METADATA_COLUMNS = [
    "Route_vhist_normalized",
    "gtfs_feed_version",
    "gtfs_feed_start_date",
    "gtfs_feed_end_date",
    "normalized_route",
    "normalized_route_source",
    "normalized_route_detail",
    "through_run_parse",
    "corrected_route_is_gtfs_stop_id",
    "corrected_route_stop_name",
    "corrected_route_stop_routes",
    "through_run_stop_candidate_routes",
    "cleaning_exclusion_reason",
]


def clean_value(value: str | None) -> str:
    if value is None:
        return ""
    value = value.strip()
    if value.upper() in {"", "NULL", "<NA>", "NAN"}:
        return ""
    return value


def normalize_known_route_code(route_code: str) -> str:
    route_code = clean_value(route_code)
    if route_code == "145":
        return "14S"
    return route_code


def parse_service_date(row: dict[str, str]) -> int | None:
    service_day = clean_value(row.get("service_day"))
    if len(service_day) == 6 and service_day.isdigit():
        return int(f"20{service_day}")

    ts = clean_value(row.get("ts"))
    if len(ts) >= 10:
        date_part = ts[:10].replace("-", "")
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

    def route_code_to_short(self, route_code: str) -> str:
        route_code = clean_value(route_code)
        if not route_code:
            return ""
        if route_code in self.route_id_to_short:
            return normalize_known_route_code(self.route_id_to_short[route_code])
        if route_code in self.canonical_routes:
            return normalize_known_route_code(route_code)
        return ""

    def split_through_route(self, route_code: str) -> tuple[tuple[str, ...], ...]:
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

    def through_route_sequences(self, route_code: str) -> list[tuple[str, ...]]:
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


def route_vhist_normalized(row: dict[str, str]) -> str:
    return normalize_known_route_code(clean_value(row.get("Route_vhist")))


def format_sequences(sequences: list[tuple[str, ...]]) -> str:
    return " | ".join(" + ".join(sequence) for sequence in sequences)


def resolve_missing_vhist_route(
    row: dict[str, str],
    feed: GtfsFeed | None,
) -> tuple[str, str, str, str, str, str, str]:
    corrected_route = clean_value(row.get("corrected_route"))
    stop_id_vhist = clean_value(row.get("Stop_Id_vhist"))

    if feed is None:
        return "", "unresolved_no_gtfs_feed", "", "", "", "", ""

    corrected_route_as_route = feed.route_code_to_short(corrected_route)
    corrected_route_is_stop = corrected_route in feed.stop_names
    corrected_route_stop_name = feed.stop_names.get(corrected_route, "")
    corrected_route_stop_routes = sorted(feed.stop_routes.get(corrected_route, set()))

    if corrected_route_as_route:
        return (
            corrected_route_as_route,
            "corrected_route_gtfs_route",
            "",
            str(corrected_route_is_stop),
            corrected_route_stop_name,
            ";".join(corrected_route_stop_routes),
            "",
        )

    through_sequences = feed.through_route_sequences(corrected_route)
    through_parse = format_sequences(through_sequences)

    if through_sequences:
        through_routes = set().union(*(set(sequence) for sequence in through_sequences))
        if stop_id_vhist:
            stop_candidate_routes = feed.stop_routes.get(stop_id_vhist, set()) & through_routes
            if len(stop_candidate_routes) == 1:
                return (
                    next(iter(stop_candidate_routes)),
                    "through_run_stop_resolved",
                    through_parse,
                    str(corrected_route_is_stop),
                    corrected_route_stop_name,
                    ";".join(corrected_route_stop_routes),
                    "",
                )
            if len(stop_candidate_routes) > 1:
                return (
                    "",
                    "unresolved_through_run_ambiguous_stop",
                    through_parse,
                    str(corrected_route_is_stop),
                    corrected_route_stop_name,
                    ";".join(corrected_route_stop_routes),
                    ";".join(sorted(stop_candidate_routes)),
                )
            return (
                "",
                "unresolved_through_run_stop_not_on_candidate_routes",
                through_parse,
                str(corrected_route_is_stop),
                corrected_route_stop_name,
                ";".join(corrected_route_stop_routes),
                "",
            )
        return (
            "",
            "unresolved_through_run_no_stop",
            through_parse,
            str(corrected_route_is_stop),
            corrected_route_stop_name,
            ";".join(corrected_route_stop_routes),
            "",
        )

    if corrected_route_is_stop:
        return (
            "",
            "unresolved_corrected_route_is_gtfs_stop_id",
            "",
            "True",
            corrected_route_stop_name,
            ";".join(corrected_route_stop_routes),
            "",
        )

    return "", "unresolved_corrected_route_not_in_gtfs", "", "False", "", "", ""


def process_rides(feeds: list[GtfsFeed]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CLEANED_RIDES_DIR.mkdir(parents=True, exist_ok=True)

    output_columns = [
        "source_file",
        "source_row_number",
        "ts",
        "service_day",
        "bus",
        "route",
        "corrected_route",
        "Route_vhist",
        "Route_vhist_normalized",
        "Stop_Id_vhist",
        "Stop_Name_vhist",
        "gtfs_feed_version",
        "gtfs_feed_start_date",
        "gtfs_feed_end_date",
        "normalized_route",
        "normalized_route_source",
        "normalized_route_detail",
        "through_run_parse",
        "corrected_route_is_gtfs_stop_id",
        "corrected_route_stop_name",
        "corrected_route_stop_routes",
        "through_run_stop_candidate_routes",
        "cleaning_exclusion_reason",
    ]

    source_counts = Counter()
    route_counts = Counter()
    unresolved_counts = Counter()
    unresolved_by_corrected_route = Counter()
    source_by_month = Counter()
    cleaned_rows_by_month = Counter()
    unresolved_rows_by_month = Counter()
    excluded_rows_by_month = Counter()
    excluded_text_counts = Counter()

    unresolved_rows_path = OUTPUT_DIR / "route_cleaning_unresolved_rows.csv"
    unresolved_full_rows_path = CLEANED_RIDES_DIR / "canonical_riders_2025_unresolved_routes.csv"
    excluded_full_rows_path = CLEANED_RIDES_DIR / "canonical_riders_2025_excluded_text_rows.csv"
    with unresolved_rows_path.open(
        "w", newline="", encoding="utf-8"
    ) as unresolved_handle, unresolved_full_rows_path.open(
        "w", newline="", encoding="utf-8"
    ) as unresolved_full_handle, excluded_full_rows_path.open(
        "w", newline="", encoding="utf-8"
    ) as excluded_full_handle:
        unresolved_writer = csv.DictWriter(unresolved_handle, fieldnames=output_columns)
        unresolved_writer.writeheader()
        unresolved_full_writer: csv.DictWriter[str] | None = None
        excluded_full_writer: csv.DictWriter[str] | None = None

        for ride_path in sorted(RAW_RIDES_DIR.glob("canonical_riders_2025_*.csv")):
            month = ride_path.stem[-2:]
            output_path = OUTPUT_DIR / f"route_cleaning_2025_{month}.csv"
            cleaned_full_path = CLEANED_RIDES_DIR / f"canonical_riders_2025_{month}_cleaned.csv"
            with ride_path.open(newline="", encoding="utf-8") as ride_handle, output_path.open(
                "w", newline="", encoding="utf-8"
            ) as output_handle, cleaned_full_path.open(
                "w", newline="", encoding="utf-8"
            ) as cleaned_full_handle:
                reader = csv.DictReader(ride_handle)
                if reader.fieldnames is None:
                    raise ValueError(f"{ride_path} has no header")

                full_output_columns = list(reader.fieldnames) + [
                    column for column in ROUTE_METADATA_COLUMNS if column not in reader.fieldnames
                ]
                if unresolved_full_writer is None:
                    unresolved_full_writer = csv.DictWriter(
                        unresolved_full_handle,
                        fieldnames=full_output_columns,
                    )
                    unresolved_full_writer.writeheader()
                if excluded_full_writer is None:
                    excluded_full_writer = csv.DictWriter(
                        excluded_full_handle,
                        fieldnames=full_output_columns,
                    )
                    excluded_full_writer.writeheader()

                writer = csv.DictWriter(output_handle, fieldnames=output_columns)
                cleaned_full_writer = csv.DictWriter(
                    cleaned_full_handle,
                    fieldnames=full_output_columns,
                )
                writer.writeheader()
                cleaned_full_writer.writeheader()

                for source_row_number, row in enumerate(reader, start=2):
                    service_date = parse_service_date(row)
                    feed = select_feed(feeds, service_date)
                    feed_version = feed.feed_version if feed else ""
                    feed_start = str(feed.feed_start_date) if feed else ""
                    feed_end = str(feed.feed_end_date) if feed else ""

                    text_value = clean_value(row.get("text"))
                    if text_value in EXCLUDED_TEXT_VALUES:
                        output_row = {
                            "source_file": ride_path.name,
                            "source_row_number": source_row_number,
                            "ts": clean_value(row.get("ts")),
                            "service_day": clean_value(row.get("service_day")),
                            "bus": clean_value(row.get("bus")),
                            "route": clean_value(row.get("route")),
                            "corrected_route": clean_value(row.get("corrected_route")),
                            "Route_vhist": clean_value(row.get("Route_vhist")),
                            "Route_vhist_normalized": route_vhist_normalized(row),
                            "Stop_Id_vhist": clean_value(row.get("Stop_Id_vhist")),
                            "Stop_Name_vhist": clean_value(row.get("Stop_Name_vhist")),
                            "gtfs_feed_version": feed_version,
                            "gtfs_feed_start_date": feed_start,
                            "gtfs_feed_end_date": feed_end,
                            "normalized_route": "",
                            "normalized_route_source": "excluded_text_removed",
                            "normalized_route_detail": "excluded_text_removed",
                            "through_run_parse": "",
                            "corrected_route_is_gtfs_stop_id": "",
                            "corrected_route_stop_name": "",
                            "corrected_route_stop_routes": "",
                            "through_run_stop_candidate_routes": "",
                            "cleaning_exclusion_reason": text_value,
                        }
                        writer.writerow(output_row)

                        route_metadata = {
                            column: output_row[column] for column in ROUTE_METADATA_COLUMNS
                        }
                        full_row = row.copy()
                        full_row.update(route_metadata)

                        source_counts["excluded_text_removed"] += 1
                        source_by_month[(month, "excluded_text_removed")] += 1
                        route_counts["<excluded>"] += 1
                        excluded_rows_by_month[month] += 1
                        excluded_text_counts[text_value] += 1
                        excluded_full_writer.writerow(full_row)
                        continue

                    vhist_route = route_vhist_normalized(row)
                    through_parse = ""
                    corrected_route_is_stop = ""
                    corrected_route_stop_name = ""
                    corrected_route_stop_routes = ""
                    through_run_stop_candidate_routes = ""

                    if vhist_route:
                        normalized_route = vhist_route
                        normalized_route_source = "Route_vhist_normalized"
                        normalized_route_detail = ""
                    else:
                        (
                            normalized_route,
                            normalized_route_source,
                            through_parse,
                            corrected_route_is_stop,
                            corrected_route_stop_name,
                            corrected_route_stop_routes,
                            through_run_stop_candidate_routes,
                        ) = resolve_missing_vhist_route(row, feed)
                        normalized_route_detail = normalized_route_source

                    output_row = {
                        "source_file": ride_path.name,
                        "source_row_number": source_row_number,
                        "ts": clean_value(row.get("ts")),
                        "service_day": clean_value(row.get("service_day")),
                        "bus": clean_value(row.get("bus")),
                        "route": clean_value(row.get("route")),
                        "corrected_route": clean_value(row.get("corrected_route")),
                        "Route_vhist": clean_value(row.get("Route_vhist")),
                        "Route_vhist_normalized": vhist_route,
                        "Stop_Id_vhist": clean_value(row.get("Stop_Id_vhist")),
                        "Stop_Name_vhist": clean_value(row.get("Stop_Name_vhist")),
                        "gtfs_feed_version": feed_version,
                        "gtfs_feed_start_date": feed_start,
                        "gtfs_feed_end_date": feed_end,
                        "normalized_route": normalized_route,
                        "normalized_route_source": normalized_route_source,
                        "normalized_route_detail": normalized_route_detail,
                        "through_run_parse": through_parse,
                        "corrected_route_is_gtfs_stop_id": corrected_route_is_stop,
                        "corrected_route_stop_name": corrected_route_stop_name,
                        "corrected_route_stop_routes": corrected_route_stop_routes,
                        "through_run_stop_candidate_routes": through_run_stop_candidate_routes,
                        "cleaning_exclusion_reason": "",
                    }
                    writer.writerow(output_row)

                    route_metadata = {
                        column: output_row[column] for column in ROUTE_METADATA_COLUMNS
                    }
                    full_row = row.copy()
                    full_row.update(route_metadata)

                    source_counts[normalized_route_source] += 1
                    source_by_month[(month, normalized_route_source)] += 1
                    route_counts[normalized_route or "<unresolved>"] += 1
                    if normalized_route_source.startswith("unresolved"):
                        unresolved_counts[normalized_route_source] += 1
                        unresolved_rows_by_month[month] += 1
                        unresolved_by_corrected_route[
                            (clean_value(row.get("corrected_route")), normalized_route_source)
                        ] += 1
                        unresolved_writer.writerow(output_row)
                        unresolved_full_writer.writerow(full_row)
                    else:
                        cleaned_rows_by_month[month] += 1
                        cleaned_full_writer.writerow(full_row)

    write_counter(
        OUTPUT_DIR / "route_cleaning_source_summary.csv",
        ["normalized_route_source", "rows"],
        source_counts,
    )
    write_counter(
        OUTPUT_DIR / "route_cleaning_normalized_route_summary.csv",
        ["normalized_route", "rows"],
        route_counts,
    )
    write_counter(
        OUTPUT_DIR / "route_cleaning_unresolved_source_summary.csv",
        ["unresolved_source", "rows"],
        unresolved_counts,
    )
    write_tuple_counter(
        OUTPUT_DIR / "route_cleaning_unresolved_by_corrected_route.csv",
        ["corrected_route", "unresolved_source", "rows"],
        unresolved_by_corrected_route,
    )
    write_tuple_counter(
        OUTPUT_DIR / "route_cleaning_source_by_month.csv",
        ["month", "normalized_route_source", "rows"],
        source_by_month,
    )
    write_counter(
        CLEANED_RIDES_DIR / "cleaned_rows_by_month.csv",
        ["month", "rows"],
        cleaned_rows_by_month,
    )
    write_counter(
        CLEANED_RIDES_DIR / "unresolved_rows_by_month.csv",
        ["month", "rows"],
        unresolved_rows_by_month,
    )
    write_counter(
        CLEANED_RIDES_DIR / "excluded_rows_by_month.csv",
        ["month", "rows"],
        excluded_rows_by_month,
    )
    write_counter(
        CLEANED_RIDES_DIR / "excluded_text_summary.csv",
        ["text", "rows"],
        excluded_text_counts,
    )


def write_counter(path: Path, columns: list[str], counter: Counter[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        for key, rows in sorted(counter.items(), key=lambda item: (-item[1], item[0])):
            writer.writerow([key, rows])


def write_tuple_counter(path: Path, columns: list[str], counter: Counter[tuple[str, ...]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        for key, rows in sorted(counter.items(), key=lambda item: (-item[1], item[0])):
            writer.writerow([*key, rows])


def write_feed_summary(feeds: list[GtfsFeed]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with (OUTPUT_DIR / "gtfs_feed_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "feed_file",
                "feed_version",
                "feed_start_date",
                "feed_end_date",
                "routes",
                "stops",
            ]
        )
        for feed in feeds:
            writer.writerow(
                [
                    feed.path.name,
                    feed.feed_version,
                    feed.feed_start_date,
                    feed.feed_end_date,
                    ";".join(sorted(feed.canonical_routes, key=lambda code: (len(code), code))),
                    len(feed.stop_names),
                ]
            )


def main() -> None:
    feeds = load_gtfs_feeds()
    write_feed_summary(feeds)
    process_rides(feeds)


if __name__ == "__main__":
    main()
