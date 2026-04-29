# Scripts

The active raw ridership processing pipeline is intentionally one script:

```bash
python scripts/build_analysis_ready_ridership.py
```

`build_analysis_ready_ridership.py` reads raw monthly TCAT ride files and GTFS
feeds directly, then writes the canonical modeling outputs in
`data/processed/Analysis_Ready`.

Install dependencies from the repository root before running the script:

```bash
python -m pip install -r requirements.txt
```

For Steven's local `spatial` conda environment, the equivalent commands are:

```bash
make build
conda run -n spatial python scripts/build_analysis_ready_ridership.py
```

The local profile is documented in `config/local_spatial.yaml`.

The older aggregated APC reports can be concatenated into Parquet with:

```bash
python scripts/build_aggregated_ridership_parquet.py
make aggregated
```

The script includes:

- GTFS-backed route resolution.
- Fare-description taxonomy.
- Missing/invalid record flags.
- Stop-name canonicalization.
- Duplicate-looking fare-event filtering.
- Route and route-stop treatment/control labels.
- Row-level ride-event output.
- Route and route-stop 15-minute panels.
- Parquet copies of the main concatenated analysis outputs.
- `data/processed/processed_outputs_manifest.csv`.

Archived scripts live under:

- `_archive_components/`: earlier split-out route cleaning, fare taxonomy,
  manifest, and orchestration scripts.
- `_archive_diagnostics/`: repeatable diagnostic scripts, retained for audit.
- `_archive_notebook_update_helpers/`: one-off notebook editing helpers.
