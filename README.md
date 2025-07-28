# QXDM Log Analysis Pipeline

This project contains a set of scripts for analysing 3GPP/QXDM log files. The pipeline extracts sentences from large log archives, computes a variety of metrics and supports both Windows and Linux environments.

## Features

- Adaptive performance settings based on available CPU, GPU and memory
- Batch or multi-process processing modes
- Checkpointing for sentence extraction and embedding computation
- Metrics including semantic spread, redundancy index, cluster entropy and more
- Optional integration with Weights & Biases for experiment tracking

## Quick Start

Install the required packages and run the main pipeline script:

```bash
pip install -r requirements.txt
python src/Main.py --help
```

Example logs for testing can be found in `test_logs/`.

## License

This repository is provided for research and experimentation purposes. See the individual source files for license details.

## Dataset

The test_dataset folder is used for the test code
