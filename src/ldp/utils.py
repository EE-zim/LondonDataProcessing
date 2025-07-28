"""
System optimization utilities for LDP.
"""
import os
import math

import psutil
import torch

def get_optimal_settings() -> dict:
    """Determine optimal settings based on system resources."""
    info = {
        'is_windows': os.name == 'nt',
        'cpu_count': os.cpu_count() or 8,
        'memory_gb': psutil.virtual_memory().total / (1024**3),
    }
    if torch.cuda.is_available():
        mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        info['gpu_mem_gb'] = mem

    settings = {}
    # Block size
    settings['block_size'] = 500_000 if info['memory_gb'] >= 32 else 200_000
    # Concurrency
    if info['is_windows']:
        settings['n_proc'] = min(info['cpu_count'] // 2, 6)
        settings['use_batch_processing'] = True
    else:
        settings['n_proc'] = min(info['cpu_count'] - 1, 8)
        settings['use_batch_processing'] = False
    # Batch size
    if torch.cuda.is_available():
        gb = info.get('gpu_mem_gb', 0)
        settings['batch_size'] = 512 if gb >= 8 else 384 if gb >= 6 else 256
    else:
        settings['batch_size'] = 128

    # Sampling sizes
    if info['memory_gb'] >= 64:
        base, mx, thr = 50_000, 200_000, 100_000
    elif info['memory_gb'] >= 32:
        base, mx, thr = 20_000, 100_000, 50_000
    else:
        base, mx, thr = 10_000, 50_000, 25_000

    settings['base_sample_size'] = int(os.environ.get('SAMPLE_SIZE_BASE', base))
    settings['max_sample_size'] = int(os.environ.get('SAMPLE_SIZE_MAX', mx))
    settings['sample_threshold'] = int(os.environ.get('SAMPLE_THRESHOLD', thr))
    return settings

def get_smart_sample_size(dataset_size: int, settings: dict, category_name: str = "") -> int:
    """Compute adaptive sample size based on dataset_size and system settings."""
    base = settings['base_sample_size']
    mx = settings['max_sample_size']
    thr = settings['sample_threshold']

    if dataset_size <= thr:
        return dataset_size
    if dataset_size <= base * 2:
        size = min(dataset_size, int(dataset_size * 0.8))
    elif dataset_size <= base * 5:
        size = base
    else:
        factor = math.log10(max(1.0, dataset_size / base))
        size = min(mx, max(base // 2, int(base * (1 + factor))))

    size = max(1, size)
    if category_name:
        pct = (size / dataset_size) * 100
        msg = (
            f'    → Smart sampling: {size:,} of {dataset_size:,} '
            f'({pct:.1f}%) for category {category_name}'
            if size < dataset_size else
            f'    → Using all {dataset_size:,} sentences for category {category_name}'
        )
        print(msg)
    return size
