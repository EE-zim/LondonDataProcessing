"""
Command-line interface for QXDM log analysis.
"""
import argparse


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser('Compute QXDM log analysis metrics')
    p.add_argument('--reset-checkpoint', action='store_true',
                   help='Delete old sentence-split checkpoint and recompute')
    p.add_argument('--checkpoint-file', default='checkpoint.pkl',
                   help='Sentence-split checkpoint (default: checkpoint.pkl)')
    p.add_argument('--embeds-file', default='embeddings.npz',
                   help='Sentence embeddings checkpoint (.npz)')
    p.add_argument('--wandb-project', default=None,
                   help='Weights & Biases project (disable if omitted)')
    p.add_argument('--wandb-run', default=None,
                   help='Explicit W&B run name (default: auto-timestamp)')
    p.add_argument('--log-sys', action='store_true', help='Log CPU/RAM/GPU to W&B')
    p.add_argument('--sys-interval', type=int, default=30,
                   help='System monitor interval (seconds)')
    return p.parse_args()
