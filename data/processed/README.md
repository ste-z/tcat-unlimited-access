# Processed Data

This folder is organized around one active modeling layer. Diagnostics and old
exploration outputs are archived under `_Archive/` so the processed-data root
stays easy to read.

Use `scripts/build_analysis_ready_ridership.py` to rebuild the current outputs
directly from raw ride files and GTFS. Environment setup is documented in the
repository root `README.md`; dependencies live in `requirements.txt`.

## Use For Analysis

- `Analysis_Ready/ride_events_analysis_ready.parquet`
  - One retained ride/event row per boarding record with cleaned time, route,
    stop, stop name, fare family, and fare category.
- `Analysis_Ready/route_15min_panel.parquet`
  - Route-by-15-minute panel for route-level discontinuity models.
- `Analysis_Ready/route_stop_15min_panel.parquet`
  - Route-stop-by-15-minute panel for the preferred Cornell-exposure design.
- `Aggregated_Analysis/*.parquet`
  - Concatenated legacy APC reports from the older aggregated Excel files.
- `Analysis_Ready/route_group_lookup.csv`
  - Primary route treatment/control labels.
- `Analysis_Ready/route_stop_group_lookup.csv`
  - Route-stop treatment/control labels based on pre-6 weekday Cornell Card
    share and balanced service around 6 pm.
- `Analysis_Ready/stop_name_lookup.csv`
  - Canonical stop names by `Stop_Id_vhist`.
- `Analysis_Ready/fare_category_lookup.csv`
  - Fare description to analysis category mapping.

## Archive

- `_Archive/Diagnostics/`
  - Repeatable diagnostics for cleaning issues, DiDisc group selection, and
    direction-column interpretation.
- `_Archive/Exploration/`
  - Older loose CSV outputs from notebook exploration.

## Manifest

`processed_outputs_manifest.csv` labels every file in this folder by role. Files
with `analysis_use == yes` are the canonical analysis outputs.
