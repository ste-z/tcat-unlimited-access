"""Profile missing, invalid, and ambiguous records before final filtering.

The outputs from this script are diagnostics, not modeling inputs. They explain
the cleaning choices later implemented in build_analysis_ready_ridership.py.
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = PROJECT_ROOT / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

from build_analysis_ready_ridership import (
    add_fare_taxonomy,
    fare_category_lookup_frame,
    load_raw_rides_with_routes,
)
OUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "_Archive"
    / "Diagnostics"
    / "Ridership_Cleaning_Diagnostics"
)

USECOLS = [
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
    "normalized_route",
    "normalized_route_source",
    "Stop_Id_vhist",
    "Stop_Name_vhist",
]

DTYPES = {
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
    "normalized_route": "string",
    "normalized_route_source": "string",
    "Stop_Id_vhist": "string",
    "Stop_Name_vhist": "string",
}


def as_bool(series: pd.Series) -> pd.Series:
    return series.astype("string").str.lower().map({"true": True, "false": False}).astype("boolean")


def clean_text(series: pd.Series) -> pd.Series:
    return series.fillna("").astype("string").str.strip()


def load_cleaned_rides() -> pd.DataFrame:
    rides, _route_resolution, _route_column_diags = load_raw_rides_with_routes()
    return rides


def add_analysis_fields(rides: pd.DataFrame) -> pd.DataFrame:
    rides = rides.copy()
    rides["route_clean"] = rides["normalized_route"].astype("string").str.strip()
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

    rides["service_date"] = pd.to_datetime(rides["service_day"], format="%y%m%d", errors="coerce")
    rides["event_time"] = rides["ts_event"].combine_first(rides["ts"]).combine_first(rides["ts_vhist"])
    rides["event_time_source"] = np.select(
        [
            rides["ts_event"].notna(),
            rides["ts"].notna(),
            rides["ts_vhist"].notna(),
        ],
        ["ts_event", "ts", "ts_vhist"],
        default="missing",
    )
    rides["date"] = rides["ts"].dt.date
    rides["event_date"] = rides["event_time"].dt.date
    rides["hour"] = rides["event_time"].dt.hour
    rides["minute"] = rides["event_time"].dt.minute
    rides["dow"] = rides["event_time"].dt.dayofweek
    rides["weekday"] = rides["dow"] < 5
    rides["weekend"] = ~rides["weekday"]
    rides["minutes_since_18"] = (rides["hour"] - 18) * 60 + rides["minute"]
    rides["free_policy_period"] = rides["weekend"] | (rides["weekday"] & (rides["minutes_since_18"] >= 0))
    rides["weekday_near_6pm"] = rides["weekday"] & rides["minutes_since_18"].between(-180, 179)

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
    rides["off_policy_cornell_override"] = (
        rides["fare_group"].eq("cornell_override") & ~rides["free_policy_period"]
    )
    return rides


def modal_stop_names(rides: pd.DataFrame) -> pd.DataFrame:
    stop_counts = (
        rides.dropna(subset=["stop_id_clean", "stop_name_raw_clean"])
        .groupby(["stop_id_clean", "stop_name_raw_clean"], dropna=False)
        .size()
        .rename("rows")
        .reset_index()
        .sort_values(["stop_id_clean", "rows", "stop_name_raw_clean"], ascending=[True, False, True])
    )
    modal = stop_counts.drop_duplicates("stop_id_clean").rename(
        columns={"stop_name_raw_clean": "canonical_stop_name", "rows": "canonical_name_rows"}
    )
    variants = stop_counts.groupby("stop_id_clean").agg(
        stop_name_variant_count=("stop_name_raw_clean", "nunique"),
        total_named_rows=("rows", "sum"),
        stop_name_examples=(
            "stop_name_raw_clean",
            lambda s: " | ".join(s.astype(str).head(5)),
        ),
    )
    return (
        modal[["stop_id_clean", "canonical_stop_name", "canonical_name_rows"]]
        .merge(variants, on="stop_id_clean", how="left")
        .sort_values(["stop_name_variant_count", "total_named_rows"], ascending=[False, False])
    )


def duplicate_summary(rides: pd.DataFrame) -> pd.DataFrame:
    keys = ["event_time", "bus", "tr_seq", "text_clean", "description_clean", "route_clean"]
    keyed = rides.dropna(subset=["event_time", "bus", "tr_seq", "route_clean"]).copy()
    dup_counts = keyed.groupby(keys, dropna=False).size().rename("duplicate_rows").reset_index()
    dup_counts = dup_counts[dup_counts["duplicate_rows"] > 1]
    if dup_counts.empty:
        return pd.DataFrame(
            [{"key": ",".join(keys), "duplicate_groups": 0, "extra_rows": 0, "max_group_size": 0}]
        )
    return pd.DataFrame(
        [
            {
                "key": ",".join(keys),
                "duplicate_groups": len(dup_counts),
                "extra_rows": int((dup_counts["duplicate_rows"] - 1).sum()),
                "max_group_size": int(dup_counts["duplicate_rows"].max()),
            }
        ]
    )


def route_stop_exposure(rides: pd.DataFrame) -> pd.DataFrame:
    sample = rides.loc[
        rides["weekday"]
        & rides["minutes_since_18"].between(-180, 179)
        & rides["route_clean"].notna()
        & rides["stop_id_clean"].notna()
    ].copy()
    sample["pre_cutoff"] = sample["minutes_since_18"] < 0
    grouped = sample.groupby(["route_clean", "stop_id_clean"], dropna=False)
    out = grouped.agg(
        canonical_stop_name=("stop_name_raw_clean", lambda s: s.dropna().mode().iat[0] if not s.dropna().empty else pd.NA),
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
    out["didisc_group"] = np.select(
        [
            out["has_balanced_near6_sample"] & out["pre_cornell_card_share"].ge(0.60),
            out["has_balanced_near6_sample"] & out["pre_cornell_card_share"].le(0.15),
        ],
        ["treated_high_cornell_exposure", "control_low_cornell_exposure"],
        default="exclude_or_sensitivity",
    )
    return out.sort_values(["didisc_group", "boards"], ascending=[True, False])


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rides = add_analysis_fields(load_cleaned_rides())

    quality = pd.Series(
        {
            "rows": len(rides),
            "corrected_riders_sum": rides["corrected_rider_num"].sum(),
            "missing_ts_rows": rides["ts"].isna().sum(),
            "missing_event_time_rows": rides["event_time"].isna().sum(),
            "missing_service_date_rows": rides["service_date"].isna().sum(),
            "missing_route_rows": rides["route_clean"].isna().sum(),
            "missing_stop_id_rows": rides["stop_id_clean"].isna().sum(),
            "missing_stop_name_rows": rides["stop_name_raw_clean"].isna().sum(),
            "non_revenue_rows": (~rides["is_revenue_bool"].fillna(False)).sum(),
            "zero_corrected_rider_rows": rides["corrected_rider_num"].eq(0).sum(),
            "excluded_text_rows_surviving_cleaned_files": rides["excluded_text_record"].sum(),
            "empty_ttp_rows": rides["empty_ttp_record"].sum(),
            "dropfile_exception_rows": rides["dropfile_exception"].sum(),
            "fare_record_missing_description_rows": rides["fare_record_missing_description"].sum(),
            "apc_only_missing_fare_description_rows": rides[
                "apc_only_missing_fare_description"
            ].sum(),
            "invalid_analysis_record_rows": rides["invalid_analysis_record"].sum(),
            "off_policy_cornell_override_rows": rides["off_policy_cornell_override"].sum(),
            "negative_amount_rows": rides["amt_num"].lt(0).sum(),
            "amount_over_10_rows": rides["amt_num"].gt(10).sum(),
        },
        name="value",
    )
    quality.to_csv(OUT_DIR / "quality_summary.csv", header=True)

    route_source = (
        rides["normalized_route_source"]
        .fillna("<missing>")
        .value_counts(dropna=False)
        .rename_axis("normalized_route_source")
        .rename("rows")
        .reset_index()
    )
    route_source.to_csv(OUT_DIR / "route_source_summary.csv", index=False)

    issue_columns = [
        "excluded_text_record",
        "empty_ttp_record",
        "dropfile_exception",
        "fare_record_missing_description",
        "apc_only_missing_fare_description",
        "invalid_analysis_record",
        "off_policy_cornell_override",
    ]
    issue_summary = pd.DataFrame(
        [
            {
                "issue": issue,
                "rows": int(rides[issue].sum()),
                "corrected_riders": float(rides.loc[rides[issue], "corrected_rider_num"].sum()),
            }
            for issue in issue_columns
        ]
    )
    issue_summary.to_csv(OUT_DIR / "record_issue_summary.csv", index=False)

    issue_by_route_rows = []
    for issue in issue_columns:
        route_issue = (
            rides.loc[rides[issue]]
            .groupby("route_clean", dropna=False)
            .agg(rows=("route_clean", "size"), corrected_riders=("corrected_rider_num", "sum"))
            .reset_index()
        )
        route_issue["issue"] = issue
        issue_by_route_rows.append(route_issue)
    pd.concat(issue_by_route_rows, ignore_index=True).sort_values(
        ["issue", "rows"], ascending=[True, False]
    ).to_csv(OUT_DIR / "record_issue_by_route.csv", index=False)

    time_source = (
        rides.groupby(["event_time_source", "has_fare_bool", "has_apc_bool"], dropna=False)
        .size()
        .rename("rows")
        .reset_index()
        .sort_values("rows", ascending=False)
    )
    time_source.to_csv(OUT_DIR / "event_time_source_summary.csv", index=False)

    fare_policy = pd.crosstab(
        rides["fare_group"],
        rides["free_policy_period"],
        values=rides["corrected_rider_num"],
        aggfunc="sum",
        margins=True,
    ).rename(columns={False: "paid_policy_period", True: "free_policy_period"})
    fare_policy.to_csv(OUT_DIR / "fare_group_policy_corrected_riders.csv")

    fare_category_policy = pd.crosstab(
        rides["fare_category"],
        rides["free_policy_period"],
        values=rides["corrected_rider_num"],
        aggfunc="sum",
        margins=True,
    ).rename(columns={False: "paid_policy_period", True: "free_policy_period"})
    fare_category_policy = fare_category_policy.reindex(
        [category for category in fare_category_lookup_frame()["fare_category"].unique() if category in fare_category_policy.index]
        + [category for category in fare_category_policy.index if category not in set(fare_category_lookup_frame()["fare_category"])]
    )
    fare_category_policy.to_csv(OUT_DIR / "fare_category_policy_corrected_riders.csv")

    fare_family_policy = pd.crosstab(
        rides["fare_family"],
        rides["free_policy_period"],
        values=rides["corrected_rider_num"],
        aggfunc="sum",
        margins=True,
    ).rename(columns={False: "paid_policy_period", True: "free_policy_period"})
    fare_family_policy.to_csv(OUT_DIR / "fare_family_policy_corrected_riders.csv")

    fare_category_summary = (
        rides.groupby(
            [
                "fare_family",
                "fare_family_slug",
                "fare_category",
                "fare_category_slug",
                "description_clean",
            ],
            dropna=False,
        )
        .agg(rows=("fare_category", "size"), corrected_riders=("corrected_rider_num", "sum"))
        .reset_index()
        .sort_values(["fare_family", "fare_category", "rows"], ascending=[True, True, False])
    )
    fare_category_summary.to_csv(OUT_DIR / "fare_category_description_summary.csv", index=False)
    fare_category_lookup_frame().to_csv(OUT_DIR / "fare_category_lookup.csv", index=False)

    off_policy_override = (
        rides.loc[rides["off_policy_cornell_override"]]
        .groupby(["route_clean", "description_clean"], dropna=False)
        .agg(rows=("route_clean", "size"), corrected_riders=("corrected_rider_num", "sum"))
        .reset_index()
        .sort_values("rows", ascending=False)
    )
    off_policy_override.to_csv(OUT_DIR / "off_policy_cornell_override_by_route.csv", index=False)

    stop_names = modal_stop_names(rides)
    stop_names.to_csv(OUT_DIR / "stop_name_canonicalization_lookup.csv", index=False)

    dupes = duplicate_summary(rides)
    dupes.to_csv(OUT_DIR / "duplicate_candidate_summary.csv", index=False)
    duplicate_keys = [
        "event_time",
        "bus",
        "tr_seq",
        "text_clean",
        "description_clean",
        "route_clean",
    ]
    duplicate_mask = rides.duplicated(duplicate_keys, keep=False) & rides[duplicate_keys].notna().all(axis=1)
    rides.loc[
        duplicate_mask,
        [
            "source_month",
            "event_time",
            "ts",
            "ts_event",
            "ts_vhist",
            "bus",
            "tr_seq",
            "route_clean",
            "stop_id_clean",
            "stop_name_raw_clean",
            "text_clean",
            "description_clean",
            "media_text_clean",
            "corrected_rider_num",
        ],
    ].sort_values(duplicate_keys).head(200).to_csv(
        OUT_DIR / "duplicate_candidate_rows_sample.csv", index=False
    )

    exposure = route_stop_exposure(rides)
    exposure.to_csv(OUT_DIR / "route_stop_didisc_group_candidates.csv", index=False)

    monthly = rides.groupby("source_month", dropna=False).agg(
        rows=("source_month", "size"),
        corrected_riders=("corrected_rider_num", "sum"),
        routes=("route_clean", "nunique"),
        stop_ids=("stop_id_clean", "nunique"),
        missing_stop_id_rows=("stop_id_clean", lambda s: s.isna().sum()),
        cornell_override_rows=("fare_group", lambda s: (s == "cornell_override").sum()),
        off_policy_cornell_override_rows=("off_policy_cornell_override", "sum"),
    ).reset_index()
    monthly.to_csv(OUT_DIR / "monthly_quality_summary.csv", index=False)

    print(f"Loaded {len(rides):,} cleaned rows.")
    print(f"Wrote cleaning diagnostics to {OUT_DIR.relative_to(PROJECT_ROOT)}")
    print("\nQuality summary:")
    print(quality.to_string())
    print("\nDuplicate candidates:")
    print(dupes.to_string(index=False))
    print("\nEvent time source:")
    print(time_source.to_string(index=False))
    print("\nFare category policy summary:")
    print(fare_category_policy.to_string())
    print("\nRoute-stop DiDisc candidate counts:")
    print(
        exposure["didisc_group"]
        .value_counts()
        .rename_axis("didisc_group")
        .rename("route_stop_cells")
        .reset_index()
        .to_string(index=False)
    )
    print("\nTop stop-name variant IDs:")
    print(
        stop_names.loc[stop_names["stop_name_variant_count"] > 1]
        .head(15)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
