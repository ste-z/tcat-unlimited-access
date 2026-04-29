"""Fare-description taxonomy shared by cleaning scripts and notebooks."""

import pandas as pd


FARE_TAXONOMY_BY_DESCRIPTION = {
    # Free
    "LA1 Youth": ("Free", "Free Fares"),
    "RA4  Transfer": ("Free", "Transfer"),
    "Cornell Card": ("Free", "Cornell"),
    "Second Left Arrow 16": ("Free", "Cornell Override"),
    "Ithaca College Pass": ("Free", "IC"),
    "Third Left Arrow 17": ("Free", "IC Override"),
    "Fourth Left Arrow 18": ("Free", "TC3 Override"),
    # Half fare
    "RA2  Senior 60 & Up": ("Half Fare", "Ride-based Half Fares"),
    "RA3  Disabled": ("Half Fare", "Ride-based Half Fares"),
    "15 RIDE HALF-FARE TCARD": ("Half Fare", "Ride-based Half Fares"),
    "15 Ride Pass Half Fare Mobile": ("Half Fare", "Ride-based Half Fares"),
    "1 Ride Half Mobile": ("Half Fare", "Ride-based Half Fares"),
    "2 Ride Half Mobile": ("Half Fare", "Ride-based Half Fares"),
    "RA1 Adult 18-59": ("Half Fare", "Ride-based Half Fares"),
    # Regular fares
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
    # Sparse/empty farebox records
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

FARE_FAMILY_ORDER = list(FARE_FAMILY_SLUG_BY_LABEL)
FARE_CATEGORY_ORDER = list(FARE_CATEGORY_SLUG_BY_LABEL)


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
