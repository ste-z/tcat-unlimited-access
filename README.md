# TCAT Unlimited Access

This repository studies how unlimited transit access affects TCAT ridership,
with a focus on Cornell's free weekday-after-6 pm and weekend access policy.

## Environment Setup

Create or activate a Python environment, then install the project
requirements:

```bash
python -m pip install -r requirements.txt
```

For a fresh conda environment, you can also use:

```bash
conda env create -f environment.yml
conda activate tcat-unlimited-access
```

## Main Workflow

With an environment active, rebuild the analysis-ready ridership tables and
processed-output manifest with:

```bash
python scripts/build_analysis_ready_ridership.py
```

That command reads the raw monthly ride files and GTFS, resolves routes, applies
the fare taxonomy, performs the analysis cleaning, and writes only the canonical
modeling outputs.

To rebuild the legacy concatenated APC Parquet tables from the older aggregated
Excel reports:

```bash
python scripts/build_aggregated_ridership_parquet.py
```

To open the notebooks:

```bash
jupyter lab
```

## Key Files

- `notebooks/raw_analysis/raw_ridership_exploration_cleaning.ipynb`
  - Primary notebook for raw-data cleaning decisions, fare recoding,
    treatment/control diagnostics, and analysis-ready outputs.
- `notebooks/aggregated_analysis/aggregated_ridership_analysis.ipynb`
  - Older hour-aggregated ridership analysis, kept as legacy comparison.
- `scripts/`
  - One active processing script plus archived implementation history.
- `requirements.txt`
  - Python dependencies for the scripts and notebooks.
- `environment.yml`
  - Optional conda environment definition.
- `config/local_spatial.yaml`
  - Local run profile for the `spatial` conda environment.
- `data/processed/Analysis_Ready/`
  - Canonical modeling tables. The main concatenated outputs are available as
    Parquet and CSV.
- `data/processed/Aggregated_Analysis/`
  - Concatenated Parquet tables from the older aggregated APC Excel reports.
- `data/processed/_Archive/`
  - Archived intermediates, diagnostics, and old exploration outputs.
- `data/processed/processed_outputs_manifest.csv`
  - Inventory labeling each processed file by purpose.

The preferred DiDisc design defines treatment/control using Cornell-exposed
routes or route-stop cells, not fare media alone. `Second Left Arrow 16` is a
Cornell override/compliance signal; `Cornell Card` is best used as an exposure
proxy or placebo group because those riders already had unlimited access.
