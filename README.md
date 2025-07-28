# QXDM Log Analysis Pipeline

This repository contains a small toolkit for processing 3GPP/QXDM log files. It extracts technical sentences from large archives and computes a range of metrics using sentence embeddings.

The code has been refactored into the `ldp` Python package (located under `src/ldp`). You can run the pipeline either directly or via `python -m ldp`.

## Features

- Adaptive performance settings based on available CPU, GPU and memory
- Batch or multi-process processing modes
- Checkpointing for sentence extraction and embedding computation
- Metrics including semantic spread, redundancy index, cluster entropy and more
- Optional integration with Weights & Biases for experiment tracking

## Installation

```bash
pip install -r requirements.txt
```

Optionally install the package in editable mode:

```bash
pip install -e .
```

## Usage

Run the pipeline with:

```bash
python -m ldp --help
```

Important environment variables:

- `QXDM_ROOT` – directory containing your log files
- `BLOCK_SIZE` – number of characters processed per block (defaults adaptively)
- `BATCH_SIZE` – embedding batch size
- `OMP_NUM_THREADS` – CPU worker count

Example logs for testing can be found in `test_logs/` and `test_dataset/`.

## Repository layout

```
src/ldp/            Core package modules
notebooks/          Example Jupyter notebooks
scripts/            Helper shell scripts
```

## License

This repository is provided for research and experimentation purposes. See the individual source files for license details.
