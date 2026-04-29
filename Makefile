CONDA ?= /Users/stevenzhou/miniconda3/bin/conda
CONDA_ENV ?= spatial
PYTHON := $(CONDA) run -n $(CONDA_ENV) python
JUPYTER := $(CONDA) run -n $(CONDA_ENV) jupyter

.PHONY: build aggregated install notebook check

build:
	$(PYTHON) scripts/build_analysis_ready_ridership.py

aggregated:
	$(PYTHON) scripts/build_aggregated_ridership_parquet.py

install:
	$(PYTHON) -m pip install -r requirements.txt

notebook:
	$(JUPYTER) lab

check:
	$(PYTHON) -m py_compile \
		scripts/build_analysis_ready_ridership.py \
		scripts/build_aggregated_ridership_parquet.py
