# Archived Pipeline Components

These scripts were separate pipeline pieces before the build was collapsed into
one canonical script:

- `clean_routes_with_gtfs.py`
- `fare_taxonomy.py`
- `write_processed_manifest.py`
- `build_processed_outputs.py`

Their logic now lives in `scripts/build_analysis_ready_ridership.py`. Keep these
only as implementation history.
