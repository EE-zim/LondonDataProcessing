# QXDM Log Analysis Pipeline

This repository contains a small toolkit for processing 3GPP/QXDM log files. It extracts technical sentences from large archives and computes a range of metrics using sentence embeddings.

The code has been refactored into the `ldp` Python package (located under
`src/ldp`).  When running directly from the repository without installing the
package, ensure that the `src` directory is on your `PYTHONPATH`:

```bash
PYTHONPATH=src python -m ldp --help
# or
python src/ldp/main.py --help
```

If you install the project (for example with `pip install -e .`), the module can
be invoked simply as:

```bash
python -m ldp --help
```

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

Optionally install the package in editable mode, which also exposes the `ldp` console script:

```bash
pip install -e .
ldp --help  # verify the CLI is available
```

## Usage

From the repository root without installing the package:

```bash
PYTHONPATH=src python -m ldp --help
# or
python src/ldp/main.py --help
```

After installation (`pip install -e .`) the command becomes:

```bash
python -m ldp --help  # or simply `ldp --help`
```

Important environment variables:

- `QXDM_ROOT` – directory containing your log files
- `BLOCK_SIZE` – number of characters processed per block (defaults adaptively)
- `BATCH_SIZE` – embedding batch size
- `OMP_NUM_THREADS` – CPU worker count

Example logs for testing can be found in `test_logs/` and `test_dataset/`.

## Repository layout

```
src/ldp/           Core package modules
test_logs/         Sample log files for quick testing
test_dataset/      Example test dataset and processor script
```

## License

This repository is provided for research and experimentation purposes. See the individual source files for license details.
