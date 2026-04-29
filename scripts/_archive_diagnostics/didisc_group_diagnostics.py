"""Diagnose candidate treatment/control groups for the Cornell policy design.

The design should compare high-Cornell-exposure routes or route-stop cells with
low-exposure cells around the 6 pm weekday discontinuity. Fare media are used to
measure exposure, not to define the treated outcome group.
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = PROJECT_ROOT / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

from build_analysis_ready_ridership import add_fare_taxonomy, load_raw_rides_with_routes
OUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "_Archive"
    / "Diagnostics"
    / "DiDisc_Group_Diagnostics"
)

USECOLS = [
    "ts",
    "service_day",
    "normalized_route",
    "text",
    "description",
    "media_text",
    "is_cornell",
    "amt",
    "is_revenue_rider",
    "Stop_Name_vhist",
    "Stop_Id_vhist",
    "corrected_rider",
]

DTYPES = {
    "normalized_route": "string",
    "text": "string",
    "description": "string",
    "media_text": "string",
    "is_cornell": "string",
    "Stop_Name_vhist": "string",
    "Stop_Id_vhist": "string",
    "service_day": "string",
    "is_revenue_rider": "string",
}

OLD_CORNELL_ONLY_ROUTES = ["30", "51", "82"]
OLD_NO_CORNELL_ROUTES = ["11", "13", "14", "14S", "15", "17", "21", "36", "65", "67"]


def load_cleaned_rides() -> pd.DataFrame:
    rides, _route_resolution, _route_column_diags = load_raw_rides_with_routes()
    rides["route"] = rides["normalized_route"].astype("string")
    rides["text_clean"] = rides["text"].fillna("").astype("string").str.strip()
    rides["description_clean"] = rides["description"].fillna("").astype("string").str.strip()
    rides["media_text_clean"] = rides["media_text"].fillna("").astype("string").str.strip()
    rides["is_cornell_bool"] = rides["is_cornell"].astype("string").str.lower().eq("true")
    rides["is_revenue_bool"] = rides["is_revenue_rider"].astype("string").str.lower().eq("true")

    rides["hour"] = rides["ts"].dt.hour
    rides["minute"] = rides["ts"].dt.minute
    rides["dow"] = rides["ts"].dt.dayofweek
    rides["weekday"] = rides["dow"] < 5
    rides["weekend"] = ~rides["weekday"]
    rides["minutes_since_18"] = (rides["hour"] - 18) * 60 + rides["minute"]
    rides["weekday_post6"] = rides["weekday"] & (rides["minutes_since_18"] >= 0)
    rides["weekday_pre6"] = rides["weekday"] & (rides["minutes_since_18"] < 0)
    rides["free_policy_period"] = rides["weekend"] | rides["weekday_post6"]

    cornell_card = rides["description_clean"].eq("Cornell Card")
    cornell_override = rides["description_clean"].eq("Second Left Arrow 16")
    blank_fare = rides["description_clean"].eq("")
    rides["fare_group"] = np.select(
        [cornell_card, cornell_override, blank_fare],
        ["cornell_card", "cornell_override", "blank_or_got_fare"],
        default="other_fare",
    )
    rides = add_fare_taxonomy(rides)
    return rides


def route_summary(rides: pd.DataFrame, mask: pd.Series, label: str) -> pd.DataFrame:
    sample = rides.loc[mask & rides["route"].notna()].copy()
    grouped = sample.groupby("route", dropna=False)
    out = (
        grouped.agg(
            boards=("route", "size"),
            cornell_card=("fare_group", lambda s: (s == "cornell_card").sum()),
            cornell_override=("fare_group", lambda s: (s == "cornell_override").sum()),
            other_fare=("fare_group", lambda s: (s == "other_fare").sum()),
            blank_or_got_fare=("fare_group", lambda s: (s == "blank_or_got_fare").sum()),
            unique_stops=("Stop_Id_vhist", "nunique"),
        )
        .reset_index()
        .sort_values("boards", ascending=False)
    )
    out["cornell_card_share"] = out["cornell_card"] / out["boards"]
    out["cornell_override_share"] = out["cornell_override"] / out["boards"]
    out["cornell_any_observed_share"] = (out["cornell_card"] + out["cornell_override"]) / out[
        "boards"
    ]
    out["sample"] = label
    return out


def summarize_route_sets(rides: pd.DataFrame, pre6wk: pd.DataFrame) -> pd.DataFrame:
    high_pre6 = (
        pre6wk[(pre6wk["boards"] >= 5_000) & (pre6wk["cornell_card_share"] >= 0.50)]["route"]
        .dropna()
        .astype(str)
        .tolist()
    )
    low_pre6 = (
        pre6wk[(pre6wk["boards"] >= 5_000) & (pre6wk["cornell_card_share"] <= 0.20)]["route"]
        .dropna()
        .astype(str)
        .tolist()
    )
    route_sets = {
        "old_cornell_only": OLD_CORNELL_ONLY_ROUTES,
        "old_no_cornell": OLD_NO_CORNELL_ROUTES,
        "high_pre6_cornell_card_routes": high_pre6,
        "low_pre6_cornell_card_routes": low_pre6,
    }

    periods = {
        "all": pd.Series(True, index=rides.index),
        "weekday_15_17": rides["weekday"] & rides["minutes_since_18"].between(-180, -1),
        "weekday_18_20": rides["weekday"] & rides["minutes_since_18"].between(0, 179),
        "weekend": rides["weekend"],
    }

    rows = []
    for set_name, routes in route_sets.items():
        in_set = rides["route"].astype(str).isin(routes)
        for period_name, period_mask in periods.items():
            sample = rides.loc[in_set & period_mask]
            if sample.empty:
                continue
            rows.append(
                {
                    "set": set_name,
                    "routes": ",".join(routes),
                    "period": period_name,
                    "boards": len(sample),
                    "cornell_card_share": sample["fare_group"].eq("cornell_card").mean(),
                    "cornell_override_share": sample["fare_group"].eq("cornell_override").mean(),
                    "other_fare_share": sample["fare_group"].eq("other_fare").mean(),
                    "blank_share": sample["fare_group"].eq("blank_or_got_fare").mean(),
                }
            )
    return pd.DataFrame(rows)


def summarize_stops(rides: pd.DataFrame) -> pd.DataFrame:
    sample = rides.loc[rides["weekday_pre6"] & rides["Stop_Name_vhist"].notna()].copy()
    grouped = sample.groupby(["Stop_Id_vhist", "Stop_Name_vhist"], dropna=False)
    out = grouped.agg(
        boards=("Stop_Name_vhist", "size"),
        routes=("route", lambda s: ",".join(sorted(set(s.dropna().astype(str)))[:8])),
        cornell_card=("fare_group", lambda s: (s == "cornell_card").sum()),
        cornell_override=("fare_group", lambda s: (s == "cornell_override").sum()),
    ).reset_index()
    out["cornell_card_share"] = out["cornell_card"] / out["boards"]
    out["cornell_any_observed_share"] = (out["cornell_card"] + out["cornell_override"]) / out[
        "boards"
    ]
    return out.sort_values(["cornell_card_share", "boards"], ascending=[False, False])


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rides = load_cleaned_rides()

    fare_period = pd.crosstab(
        rides["fare_group"], rides["free_policy_period"], margins=True
    ).rename(columns={False: "paid_policy_period", True: "free_policy_period"})
    fare_period.to_csv(OUT_DIR / "fare_group_by_policy_period.csv")
    pd.crosstab(
        rides["fare_category"], rides["free_policy_period"], margins=True
    ).rename(columns={False: "paid_policy_period", True: "free_policy_period"}).to_csv(
        OUT_DIR / "fare_category_by_policy_period.csv"
    )
    pd.crosstab(
        rides["fare_family"], rides["free_policy_period"], margins=True
    ).rename(columns={False: "paid_policy_period", True: "free_policy_period"}).to_csv(
        OUT_DIR / "fare_family_by_policy_period.csv"
    )

    route_all = route_summary(rides, pd.Series(True, index=rides.index), "all")
    route_pre6wk = route_summary(rides, rides["weekday_pre6"], "weekday_pre6")
    route_15_17 = route_summary(
        rides, rides["weekday"] & rides["minutes_since_18"].between(-180, -1), "weekday_15_17"
    )
    route_18_20 = route_summary(
        rides, rides["weekday"] & rides["minutes_since_18"].between(0, 179), "weekday_18_20"
    )
    route_weekend = route_summary(rides, rides["weekend"], "weekend")
    pd.concat([route_all, route_pre6wk, route_15_17, route_18_20, route_weekend]).to_csv(
        OUT_DIR / "route_cornell_exposure_summary.csv", index=False
    )

    near6 = route_15_17[
        ["route", "boards", "cornell_card_share", "cornell_override_share", "cornell_any_observed_share"]
    ].merge(
        route_18_20[
            [
                "route",
                "boards",
                "cornell_card_share",
                "cornell_override_share",
                "cornell_any_observed_share",
            ]
        ],
        on="route",
        suffixes=("_pre", "_post"),
        how="outer",
    )
    near6["boards_sum"] = near6[["boards_pre", "boards_post"]].sum(axis=1)
    near6.sort_values("boards_sum", ascending=False).to_csv(
        OUT_DIR / "route_near_6pm_summary.csv", index=False
    )

    route_sets = summarize_route_sets(rides, route_pre6wk)
    route_sets.to_csv(OUT_DIR / "candidate_route_set_diagnostics.csv", index=False)

    stops = summarize_stops(rides)
    stops.to_csv(OUT_DIR / "stop_cornell_exposure_summary.csv", index=False)

    hourly_route_sets = []
    for set_name, routes in {
        "old_cornell_only": OLD_CORNELL_ONLY_ROUTES,
        "old_no_cornell": OLD_NO_CORNELL_ROUTES,
    }.items():
        mask = (
            rides["route"].astype(str).isin(routes)
            & rides["weekday"]
            & rides["minutes_since_18"].between(-180, 179)
        )
        hourly = (
            rides.loc[mask]
            .groupby(["hour", "fare_group"])
            .size()
            .rename("boards")
            .reset_index()
        )
        hourly["set"] = set_name
        hourly_route_sets.append(hourly)
    pd.concat(hourly_route_sets).to_csv(OUT_DIR / "old_route_sets_hourly_fare_groups.csv", index=False)

    print(f"Loaded {len(rides):,} cleaned ride rows from {rides['ts'].min()} to {rides['ts'].max()}.")
    print(f"Wrote diagnostics to {OUT_DIR.relative_to(PROJECT_ROOT)}")
    print("\nFare group by policy period:")
    print(fare_period.to_string())
    print("\nRoutes ranked by Cornell Card share before 6 pm weekdays, minimum 1,000 boards:")
    cols = [
        "route",
        "boards",
        "cornell_card_share",
        "cornell_override_share",
        "cornell_any_observed_share",
    ]
    print(
        route_pre6wk.loc[route_pre6wk["boards"] >= 1_000, cols]
        .sort_values("cornell_card_share", ascending=False)
        .head(25)
        .to_string(index=False)
    )
    print("\nCandidate route set diagnostics:")
    print(route_sets.to_string(index=False))
    print("\nHigh Cornell-card stops before 6 pm weekdays, minimum 500 boards:")
    stop_cols = [
        "Stop_Id_vhist",
        "Stop_Name_vhist",
        "boards",
        "routes",
        "cornell_card_share",
        "cornell_any_observed_share",
    ]
    print(stops.loc[stops["boards"] >= 500, stop_cols].head(25).to_string(index=False))


if __name__ == "__main__":
    main()
