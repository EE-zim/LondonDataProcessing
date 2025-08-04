#!/usr/bin/env python3
# coding: utf-8
"""
QXDM Log Analysis Pipeline (adapted from TSpec-LLM pipeline, rev-2025-05-16)

* sentence-split & slice checkpoint      → --checkpoint-file
* sentence embedding (Sentence-BERT)     → --embeds-file
* Weights & Biases (optional)            → --wandb-project
* system monitor to W&B                  → --log-sys  --sys-interval
"""

import os
import re
import pickle
import argparse
import datetime as _dt
import multiprocessing
import threading
import time
from pathlib import Path
from typing import Tuple, Dict, List
from itertools import combinations

import numpy as np
import pandas as pd
import torch
import psutil
from tqdm.auto import tqdm
from sentence_transformers import SentenceTransformer

# When running ``main.py`` directly the package context is not initialised and
# relative imports like ``from .metric_utils`` fail.  Detect this situation and
# add the parent directory to ``sys.path`` so absolute imports work as well,
# enabling both ``python -m ldp`` and ``python src/ldp/main.py`` for local
# debugging.
if __package__ is None or __package__ == "":  # pragma: no cover - runtime setup
    import sys as _sys
    from pathlib import Path as _Path

    _sys.path.append(str(_Path(__file__).resolve().parent.parent))

from ldp.metric_utils import (
    semantic_spread,
    redundancy_index,
    cluster_entropy,
    change_mag,
    novelty_density,
)
from ldp.log_processor import QXDMLogProcessor

try:
    import pynvml
    pynvml.nvmlInit()
except Exception:
    pynvml = None

try:
    import wandb
except ImportError:           # make wandb optional
    wandb = None

# Supported log file extensions (used in file discovery and messaging)
SUPPORTED_EXTENSIONS = ('.txt', '.md', '.pdf', '.log', '.csv', '.json', '.xml')
_SUPPORTED_EXTENSIONS_STR = ', '.join(SUPPORTED_EXTENSIONS)

def is_supported_file(f: Path) -> bool:
    return f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS

def find_supported_files(root: Path) -> List[Path]:
    return [f for f in root.rglob('*') if is_supported_file(f)]


# ---------------------------------------------------------------------------
# System optimization utilities
# ---------------------------------------------------------------------------

def get_optimal_settings():
    """Determine optimal settings based on system resources"""
    system_info = {}
    system_info['is_windows'] = os.name == 'nt'
    system_info['cpu_count'] = os.cpu_count() or 8
    system_info['memory_gb'] = psutil.virtual_memory().total / (1024**3)
    
    # Check GPU memory
    if torch.cuda.is_available():
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        system_info['gpu_mem_gb'] = gpu_mem
        
    # Optimal settings based on system
    settings = {}
    
    # Block size based on available RAM (larger blocks = fewer passes)
    if system_info['memory_gb'] >= 32:
        settings['block_size'] = 500_000  # Large blocks for 32GB+ systems
    else:
        settings['block_size'] = 200_000  # Smaller blocks for <32GB systems
    
    # Processing concurrency
    if system_info['is_windows']:
        # Windows: Conservative approach to avoid spaCy multiprocessing issues
        settings['n_proc'] = min(system_info['cpu_count'] // 2, 6)  # Half of logical cores, max 6
        settings['use_batch_processing'] = True
    else:
        # Linux: Can use more processes
        settings['n_proc'] = min(system_info['cpu_count'] - 1, 8)  # Leave one core free, max 8
        settings['use_batch_processing'] = False
    
    # GPU batch size optimization
    if torch.cuda.is_available():
        # RTX 2080 has 8GB VRAM, adapt based on embedding model size
        if system_info.get('gpu_mem_gb', 0) >= 8:
            settings['batch_size'] = 512  # For 8GB+ VRAM (RTX 2080 or better)
        elif system_info.get('gpu_mem_gb', 0) >= 6:
            settings['batch_size'] = 384  # For 6GB VRAM
        else:
            settings['batch_size'] = 256  # For 4GB or less VRAM
    else:
        settings['batch_size'] = 128  # CPU mode
    
    # Smart sampling size based on available memory and processing power
    if system_info['memory_gb'] >= 64:
        # High-end systems: Use larger samples for better accuracy
        settings['base_sample_size'] = 50_000     # 50K base sample
        settings['max_sample_size'] = 200_000     # Up to 200K for very large datasets
        settings['sample_threshold'] = 100_000   # Start sampling above 100K sentences
    elif system_info['memory_gb'] >= 32:
        # Mid-range systems: Balanced approach
        settings['base_sample_size'] = 20_000     # 20K base sample
        settings['max_sample_size'] = 100_000     # Up to 100K for large datasets
        settings['sample_threshold'] = 50_000    # Start sampling above 50K sentences
    else:
        # Lower-end systems: Conservative sampling
        settings['base_sample_size'] = 10_000     # 10K base sample
        settings['max_sample_size'] = 50_000      # Up to 50K for large datasets
        settings['sample_threshold'] = 25_000    # Start sampling above 25K sentences
    
    # Allow environment variable overrides for custom sampling
    settings['base_sample_size'] = int(os.environ.get('SAMPLE_SIZE_BASE', settings['base_sample_size']))
    settings['max_sample_size'] = int(os.environ.get('SAMPLE_SIZE_MAX', settings['max_sample_size']))
    settings['sample_threshold'] = int(os.environ.get('SAMPLE_THRESHOLD', settings['sample_threshold']))
        
    return settings


def get_smart_sample_size(dataset_size: int, settings: dict, category_name: str = "") -> int:
    """
    Determine optimal sample size based on dataset size and system capabilities.
    
    Args:
        dataset_size: Number of sentences in the dataset
        settings: System optimization settings
        category_name: Name of the category (for logging)
    
    Returns:
        Optimal sample size
    """
    base_sample = settings['base_sample_size']
    max_sample = settings['max_sample_size']
    threshold = settings['sample_threshold']
    
    if dataset_size <= threshold:
        # Small datasets: use all data
        return dataset_size
    
    # For large datasets, use adaptive sampling
    if dataset_size <= base_sample * 2:
        # Medium datasets: use most of the data
        sample_size = min(dataset_size, int(dataset_size * 0.8))
    elif dataset_size <= base_sample * 5:
        # Large datasets: use base sample size
        sample_size = base_sample
    else:
        # Very large datasets: scale logarithmically with safety bounds
        import math
        log_factor = math.log10(max(1.0, dataset_size / base_sample))  # Ensure log argument > 0
        sample_size = min(max_sample, max(base_sample // 2, int(base_sample * (1 + log_factor))))
    
    # Ensure minimum sample size
    sample_size = max(1, sample_size)
    
    if category_name:
        if sample_size < dataset_size:
            percentage = (sample_size / dataset_size) * 100
            print(f'    → Smart sampling: {sample_size:,} sentences from {dataset_size:,} ({percentage:.1f}%) for category {category_name}')
        else:
            print(f'    → Using all {dataset_size:,} sentences for category {category_name} (no sampling needed)')
    
    return sample_size


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Compute QXDM log analysis metrics with checkpointing & W&B logging")
    # checkpoints
    p.add_argument('--reset-checkpoint', action='store_true',
                   help='Delete old sentence-split checkpoint and recompute')
    p.add_argument('--checkpoint-file', default='checkpoint.pkl',
                   help='Sentence-split checkpoint (default: checkpoint.pkl)')
    p.add_argument('--embeds-file', default='embeddings.npz',
                   help='Sentence embeddings checkpoint (.npz)')
    # W&B
    p.add_argument('--wandb-project', default=None, help='Weights & Biases project (disable if omitted)')
    p.add_argument('--wandb-run', default=None, help='Explicit W&B run name (default: auto-timestamp)')
    # system monitor
    p.add_argument('--log-sys', action='store_true', help='log cpu/ram/gpu to W&B')
    p.add_argument('--sys-interval', type=int, default=30, help='system monitor interval (s)')
    return p.parse_args()


# ---------------------------------------------------------------------------
# Sentence extraction & checkpointing
# ---------------------------------------------------------------------------

def get_sentences(qxdm_root: Path, block_size: int, settings: dict,
                  checkpoint_file: str, reset_checkpoint: bool
                  ) -> Tuple[List[str], Dict[str, slice]]:
    """Return (all_sentences, log_category_slices) from checkpoint or recomputation."""
    cp = Path(checkpoint_file)
    if cp.exists() and not reset_checkpoint:
        print(f'[✓] Loading sentences from {cp}')
        with cp.open('rb') as f:
            return pickle.load(f)

    if cp.exists():
        cp.unlink()
        print(f'[i] Removed old checkpoint: {cp}')

    print('[i] Starting 3GPP/QXDM log processing with specialized parser...')
    
    # Initialize the 3GPP log processor
    try:
        # Prefer full model for better sentence segmentation
        log_processor = QXDMLogProcessor(use_full_model=True)
        print('[✓] Initialized QXDM log processor with full spaCy model')
    except Exception as e:
        # Fall back to lightweight model
        print(f'[warn] Full model unavailable ({e}), using lightweight model')
        log_processor = QXDMLogProcessor(use_full_model=False)
        print('[✓] Initialized QXDM log processor with blank spaCy model')

    # Check if QXDM root exists
    if not qxdm_root.exists():
        print(f'[!] QXDM_ROOT path does not exist: {qxdm_root}')
        print('[!] Please set the correct path using: export QXDM_ROOT=/path/to/your/logs')
        return [], {}

    # For QXDM logs, look for subdirectories or log files
    log_categories = []
    if qxdm_root.is_dir():
        # Option 1: If logs are organized in subdirectories
        log_categories = sorted([d.name for d in qxdm_root.iterdir() 
                                if d.is_dir()], 
                               key=str.lower)
        
        # Option 2: If you want to process all files directly without subdirectories
        if not log_categories:
            # Check if there are any supported files in the root directory
            supported_files = find_supported_files(qxdm_root)
            if supported_files:
                log_categories = ['all_logs']  # Single category for all log files
            else:
                print(f'[!] No supported log files found in {qxdm_root}')
                print(f'[!] Supported extensions: {_SUPPORTED_EXTENSIONS_STR}')
                return [], {}
    
    print('   Log categories:', ', '.join(log_categories))

    all_sents: List[str] = []
    rel_slice: Dict[str, slice] = {}

    # Show processing configuration
    print(f'[i] Processing {len(log_categories)} log categories with 3GPP-optimized pipeline:')
    processor_stats = log_processor.get_stats()
    print(f'   → Model type: {processor_stats["model_type"]}')
    print(f'   → Pipeline components: {", ".join(processor_stats["pipeline_components"])}')
    print(f'   → Technical patterns: {sum(processor_stats["patterns_count"].values())} rules')

    for category in tqdm(log_categories, desc='Processing categories', unit='category'):
        start = len(all_sents)

        if category == 'all_logs':
            # Process all files directly from qxdm_root
            files = find_supported_files(qxdm_root)
        else:
            # Process files from subdirectory
            files = find_supported_files(qxdm_root / category)
            
        print(f'   [{category}] processing {len(files)} files with 3GPP parser')
        
        # Process files with specialized 3GPP/QXDM parser
        category_sentences = []
        for f in tqdm(files, desc=f'Processing {category} files', unit='file', 
                     bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'):
            try:
                file_sentences = log_processor.process_file(f)
                category_sentences.extend(file_sentences)
            except Exception as e:
                print(f'[warn] Error processing {f.name}: {e}')
                continue

        all_sents.extend(category_sentences)
        rel_slice[category] = slice(start, len(all_sents))
        print(f'   [{category}] → {len(category_sentences):,} technical sentences extracted')

    print(f'[✓] Collected {len(all_sents):,} sentences; saving checkpoint → {cp}')
    with cp.open('wb') as f:
        pickle.dump((all_sents, rel_slice), f)
    return all_sents, rel_slice


# ---------------------------------------------------------------------------
# Sentence embedding with checkpoint
# ---------------------------------------------------------------------------

def get_embeddings(all_sents: List[str], model_name: str, batch_size: int,
                   device: str, embeds_file: str) -> np.ndarray:
    ep = Path(embeds_file)
    if ep.exists():
        print(f'[✓] Loading embeddings from {ep}')
        with np.load(ep) as npz:
            return npz['X_all']
    
    print(f'[i] Encoding {len(all_sents):,} sentences with model: {model_name}')
    print(f'[i] Using device: {device}, batch_size: {batch_size}')
    
    # More detailed loading message
    print('[i] Loading sentence transformer model...')
    t_start = time.time()
    model = SentenceTransformer(model_name, device=device)
    print(f'[✓] Model loaded in {time.time() - t_start:.1f}s')
    
    # Estimate memory usage and adjust batch size if needed
    if device == 'cuda' and torch.cuda.is_available():
        free_mem = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()
        free_mem_gb = free_mem / (1024**3)
        print(f'[i] Available GPU memory: {free_mem_gb:.1f} GB')
        
        # Dynamic batch size adjustment based on sentence length
        if len(all_sents) > 0:
            avg_len = sum(len(s) for s in all_sents[:1000]) / min(1000, len(all_sents))  # Sample first 1000
            if avg_len > 100 and batch_size > 384:
                adjusted_batch = 384
                print(f'[i] Adjusting batch size to {adjusted_batch} due to long sentences (avg {avg_len:.0f} chars)')
                batch_size = adjusted_batch
        else:
            print('[warn] No sentences to process, returning empty embeddings')
            return np.array([]).reshape(0, 384)  # Return empty array with correct shape
    
    # More detailed progress indication with ETA
    total_batches = (len(all_sents) + batch_size - 1) // batch_size
    print(f'[i] Starting sentence encoding ({total_batches:,} batches)...')
    
    # Use enhanced progress bar with GPU OOM handling
    try:
        emb = model.encode(all_sents, batch_size=batch_size, device=device,
                           convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=True)
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f'[!] GPU OOM error. Reducing batch size from {batch_size} to {batch_size//2}...')
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            try:
                emb = model.encode(all_sents, batch_size=batch_size//2, device=device,
                                   convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=True)
            except RuntimeError as e2:
                if "out of memory" in str(e2).lower():
                    print('[!] Still OOM. Switching to CPU processing...')
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    model = model.to('cpu')
                    emb = model.encode(all_sents, batch_size=batch_size//4, device='cpu',
                                       convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=True)
                else:
                    raise e2
        else:
            raise e
    
    print('[i] Converting to numpy array and saving...')
    X_all = np.asarray(emb, dtype=np.float32)
    np.savez_compressed(ep, X_all=X_all)
    print(f'[✓] Saved embeddings → {ep}')
    return X_all


# ---------------------------------------------------------------------------
# System monitor (background thread)
# ---------------------------------------------------------------------------

def start_sys_monitor(wb_run, interval: int):
    if wb_run is None:
        return

    gpu = pynvml.nvmlDeviceGetHandleByIndex(0) if pynvml else None

    def loop():
        while True:
            payload = {
                'sys/cpu': psutil.cpu_percent(),
                'sys/ram': psutil.virtual_memory().percent,
            }
            if gpu:
                util_ = pynvml.nvmlDeviceGetUtilizationRates(gpu)
                mem_ = pynvml.nvmlDeviceGetMemoryInfo(gpu)
                payload.update({'sys/gpu': util_.gpu, 'sys/vram': mem_.used / mem_.total * 100})
            wb_run.log(payload)
            time.sleep(interval)

    t = threading.Thread(target=loop, daemon=True)
    t.start()
    print(f'[✓] System monitor started (interval={interval}s)')


# ---------------------------------------------------------------------------
# Metric computations
# ---------------------------------------------------------------------------

def compute_metrics(X_all: np.ndarray, rel_slice: Dict[str, slice],
                    wb_run, metric_settings: dict = None) -> None:
    print(f'[i] Computing metrics for {len(rel_slice)} categories...')
    
    # Use default settings if none provided
    if metric_settings is None:
        metric_settings = {
            'device': 'cpu'
        }
    
    # Create reproducible random number generator
    SEED = int(os.environ.get('RANDOM_SEED', 42))
    rng = np.random.default_rng(SEED)
    
    # Add RNG to metric settings for consistent sampling
    metric_settings['rng'] = rng
    
    # Check if we have any data
    if len(rel_slice) == 0:
        print('[!] No data found! Please check your QXDM_ROOT path and ensure it contains log files.')
        print(f'[!] Supported file extensions: {_SUPPORTED_EXTENSIONS_STR}')
        return
    
    if len(X_all) == 0:
        print('[!] No sentences were extracted! Please check your log files contain readable text.')
        return

    # Optimize numpy operations for better performance
    # Set number of threads for numpy operations
    threads = min(os.cpu_count() or 4, 8)  # Use up to 8 threads for numpy
    
    # Try to set numpy thread count if available
    if os.name != 'nt':
        try:
            import mkl
            mkl.set_num_threads(threads)
            print(f'[i] MKL threads set to {threads}')
        except ImportError:
            pass

    # Get sampling settings
    settings = get_optimal_settings()
    print('[i] Smart sampling settings:')
    print(f'   → Base sample size: {settings["base_sample_size"]:,}')
    print(f'   → Max sample size: {settings["max_sample_size"]:,}')
    print(f'   → Sampling threshold: {settings["sample_threshold"]:,}')

    # per-category metrics --------------------------------------------------
    metrics, mus, pools = [], {}, {}
    print('[i] Computing per-category metrics...')
    
    # Create category progress bar with estimated time
    categories = list(rel_slice.items())
    with tqdm(total=len(categories), desc='Category metrics', unit='cat',
              bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]') as cat_pbar:
        
        for r, sl in categories:
            X = X_all[sl]
            if X.size == 0:
                print(f'[skip] {r} empty')
                cat_pbar.update(1)
                continue

            # More compact metrics calculation
            print(f'[📊] Computing metrics for {r} ({len(X):,} sentences)')
            
            # Pre-compute means for efficiency
            mus[r] = X.mean(0)
            
            # Smart sampling based on system capabilities and dataset size
            sample_size = get_smart_sample_size(len(X), settings, r)
            if sample_size < len(X):
                indices = rng.choice(len(X), sample_size, replace=False)
                pools[r] = X[indices]
            else:
                pools[r] = X
            
            # Calculate all metrics with individual progress tracking
            with tqdm(total=3, desc=f'Metrics for {r}', leave=False,
                     bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt}') as metric_pbar:
                semantic_spread_val = semantic_spread(X, device=metric_settings['device'])
                metric_pbar.update(1)
                redundancy_index_val = redundancy_index(X, device=metric_settings['device'], rng=metric_settings['rng'])
                metric_pbar.update(1)
                cluster_entropy_val = cluster_entropy(X, device=metric_settings['device'], rng=metric_settings['rng'])
                metric_pbar.update(1)
            
            m = {
                'category': r,  # Changed from 'release' to 'category'
                'sentences': len(X),
                'semantic_spread': semantic_spread_val,
                'redundancy_index': redundancy_index_val,
                'cluster_entropy': cluster_entropy_val,
            }
            metrics.append(m)
            cat_pbar.update(1)

            if wb_run:
                # Try to extract step number, fallback to index if no numbers found
                step_match = re.findall(r'\d+', r)
                step = int(step_match[0]) if step_match else len(metrics)
                wb_run.log({f'category/{k}': v for k, v in m.items() if k != 'category'}, step=step)

    df_rel = pd.DataFrame(metrics)
    
    # Check if we have any metrics
    if df_rel.empty:
        print('[!] No metrics computed! All categories were empty.')
        # Create empty files to avoid errors
        pd.DataFrame(columns=['category', 'sentences', 'semantic_spread', 'redundancy_index', 'cluster_entropy']).to_csv('release_metrics.csv', index=False)
        pd.DataFrame(columns=['from', 'to', 'change_magnitude', 'novelty_density']).to_csv('delta_metrics.csv', index=False)
        print('[✓] Created empty CSV files')
        return
    
    df_rel.to_csv('release_metrics.csv', index=False)
    print('[✓] Saved release_metrics.csv')

    # delta metrics between all category pairs ------------------------------
    print('[i] Computing delta metrics between all category pairs...')
    delta_rows = []
    rel_list = df_rel['category'].tolist()  # Changed from 'release' to 'category'
    
    if len(rel_list) > 1:
        # Generate all possible pairs
        all_pairs = list(combinations(rel_list, 2))
        total_comparisons = len(all_pairs) * 2  # *2 because we compute both directions
        
        print(f'[i] Computing {total_comparisons} directional comparisons for {len(all_pairs)} category pairs...')
        
        # Single progress bar for all pairs
        with tqdm(total=total_comparisons, desc='Delta metrics', unit='comp',
                 bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]') as delta_pbar:
            
            for a, b in all_pairs:
                # First direction: a → b - Use pools instead of mus to avoid dimension issues
                change_magnitude_val = change_mag(pools[a], pools[b], device=metric_settings['device'])
                novelty_density_a_b = novelty_density(pools[a], pools[b], device=metric_settings['device'], rng=metric_settings['rng'])
                
                dr = {
                    'from': a, 'to': b,
                    'change_magnitude': change_magnitude_val,
                    'novelty_density': novelty_density_a_b,
                }
                delta_rows.append(dr)
                delta_pbar.update(1)
                
                # Second direction: b → a (reverse)
                novelty_density_b_a = novelty_density(pools[b], pools[a], device=metric_settings['device'], rng=metric_settings['rng'])
                
                dr_reverse = {
                    'from': b, 'to': a,
                    'change_magnitude': change_magnitude_val,  # Symmetric
                    'novelty_density': novelty_density_b_a,    # Directional
                }
                delta_rows.append(dr_reverse)
                delta_pbar.update(1)
                
                # Update W&B if enabled
                if wb_run:
                    step_a = hash(a) % 1000
                    step_b = hash(b) % 1000
                    wb_run.log({f'delta/{k}': v for k, v in dr.items() if k not in ('from', 'to')}, step=step_a)
                    wb_run.log({f'delta/{k}': v for k, v in dr_reverse.items() if k not in ('from', 'to')}, step=step_b)
    else:
        print('[i] Only one category found, skipping delta metrics')

    df_delta = pd.DataFrame(delta_rows)
    
    # Sort the results for better readability (only if we have data)
    if not df_delta.empty and 'from' in df_delta.columns:
        df_delta = df_delta.sort_values(['from', 'to']).reset_index(drop=True)
    
    df_delta.to_csv('delta_metrics.csv', index=False)
    print(f'[✓] Saved delta_metrics.csv with {len(delta_rows)} pair combinations')
    
    # Print summary of pairs computed
    if len(rel_list) > 1:
        print(f'[i] Computed metrics for all combinations of {len(rel_list)} categories:')
        for i, cat in enumerate(rel_list):
            comparisons = [f'{cat}↔{other}' for other in rel_list if other != cat]
            print(f'    {cat}: {", ".join(comparisons)}')

    # upload to W&B ----------------------------------------------------------
    if wb_run:
        art = wandb.Artifact('qxdm_log_metrics', type='dataset',
                             description='CSV metrics for QXDM log analysis')
        art.add_file('release_metrics.csv')
        art.add_file('delta_metrics.csv')
        wb_run.log_artifact(art)
        print('[✓] Uploaded CSV artifacts to W&B')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # Set random seeds for reproducibility
    SEED = int(os.environ.get('RANDOM_SEED', 42))
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
    print(f'[i] Random seed set to {SEED}')

    # Get system-optimized settings ONCE
    settings = get_optimal_settings()

    # environment hyper-parameters with adaptive defaults ------------------
    QXDM_ROOT = Path(os.environ.get('QXDM_ROOT', './QXDM_Logs')).expanduser()  # Changed default path
    BLOCK_SIZE = int(os.environ.get('BLOCK_SIZE', settings['block_size']))     # Use adaptive block size
    BATCH_SIZE = int(os.environ.get('BATCH_SIZE', settings['batch_size']))     # Use adaptive batch size
    CPUS = int(os.environ.get('OMP_NUM_THREADS', settings['n_proc']))          # Use adaptive CPU count
    
    # Improved device detection with better messaging and memory checking
    if torch.cuda.is_available():
        try:
            # Test actual CUDA functionality
            torch.cuda.init()
            DEVICE = 'cuda'
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            
            # Check available memory
            if hasattr(torch.cuda, 'mem_get_info'):  # PyTorch 2.2+
                free_mem, total_mem = torch.cuda.mem_get_info()
                free_mem_gb = free_mem / (1024**3)
                print(f'[i] 🚀 CUDA GPU available: {gpu_name} ({gpu_mem:.1f}GB total, {free_mem_gb:.1f}GB free)')
                
                # Warn if low memory
                if free_mem_gb < 2.0:
                    print(f'[warn] Low GPU memory available ({free_mem_gb:.1f}GB), consider reducing batch size')
            else:
                print(f'[i] 🚀 CUDA GPU available: {gpu_name} ({gpu_mem:.1f}GB)')
                
        except Exception as e:
            print(f'[warn] CUDA available but initialization failed: {e}')
            DEVICE = 'cpu'
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        # Check platform for Apple Silicon
        import platform
        if 'arm' in platform.processor().lower() or 'arm64' in platform.machine().lower():
            DEVICE = 'mps'
            print('[i] 🍎 Apple Silicon GPU (MPS) available')
        else:
            print('[warn] MPS reported available but not on ARM64, falling back to CPU')
            DEVICE = 'cpu'
    else:
        DEVICE = 'cpu'
        print('[i] 💻 Using CPU (no GPU available)')
    
    MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'

    # Show optimized configuration summary
    print("\n" + "="*60)
    print(f"📊 System Configuration: {os.name.upper()} | {CPUS} threads")
    print("📈 Optimized Parameters:")
    print(f"   → Processing Device: {DEVICE}")
    print(f"   → Block Size: {BLOCK_SIZE:,} chars")
    print(f"   → Batch Size: {BATCH_SIZE}")
    print(f"   → Processing Mode: {'Batch' if settings['use_batch_processing'] else 'Multiprocess'}")
    print("="*60)
    
    print(f'>>> Device={DEVICE}  BLOCK_SIZE={BLOCK_SIZE}  BATCH_SIZE={BATCH_SIZE}  CPUS={CPUS}')
    print(f'>>> QXDM_ROOT={QXDM_ROOT}')
    print(f'>>> QXDM_ROOT exists: {QXDM_ROOT.exists()}')
    
    # Early validation - exit if no data found
    if not QXDM_ROOT.exists():
        print(f'>>> [!] QXDM_ROOT does not exist: {QXDM_ROOT}')
        print('>>> [!] Please create the directory or set QXDM_ROOT environment variable')
        print('>>> [!] Exiting...')
        import sys
        sys.exit(1)
    
    files_in_root = list(QXDM_ROOT.iterdir())
    print(f'>>> Files in QXDM_ROOT: {len(files_in_root)} items')
    print(f'>>> First few items: {[f.name for f in files_in_root[:5]]}')
    
    # Check for supported files
    supported_files = find_supported_files(QXDM_ROOT)
    if not supported_files:
        print('>>> [!] No supported log files found!')
        print(f'>>> [!] Supported extensions: {_SUPPORTED_EXTENSIONS_STR}')
        print('>>> [!] Exiting...')
        import sys
        sys.exit(1)

    print(f'>>> Found {len(supported_files)} supported files')

    multiprocessing.freeze_support()

    # Weights & Biases setup ------------------------------------------------
    wb_run = None
    if args.wandb_project:
        if wandb is None:
            raise RuntimeError('wandb-project specified but wandb package not installed. `pip install wandb`')
        run_name = args.wandb_run or f"qxdm-metrics-{_dt.datetime.now():%Y%m%d-%H%M%S}"
        wb_run = wandb.init(project=args.wandb_project, name=run_name,
                            config=dict(QXDM_ROOT=str(QXDM_ROOT), BLOCK_SIZE=BLOCK_SIZE,
                                        BATCH_SIZE=BATCH_SIZE, CPUS=CPUS,
                                        DEVICE=DEVICE, MODEL_NAME=MODEL_NAME))
        if args.log_sys:
            start_sys_monitor(wb_run, args.sys_interval)

    # Prepare unified settings for all metric functions
    metric_settings = {
        'device': DEVICE
    }
    
    print(f">>> Unified metric settings: {metric_settings}")

    # processing pipeline with improved progress visualization ---------------
    print('\n' + '='*78)
    print('🚀 Starting QXDM Log Analysis Pipeline')
    print('='*78)
    
    with tqdm(total=3, desc='Overall Progress', unit='stage',
              bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} stages') as main_pbar:
        # Stage 1: Clear description with emoji and estimated time
        stage_start = time.time()
        main_pbar.set_description('📄 Stage 1/3: Extracting sentences')
        all_sents, rel_slice = get_sentences(QXDM_ROOT, BLOCK_SIZE, settings,
                                           args.checkpoint_file, args.reset_checkpoint)
        stage_time = time.time() - stage_start
        main_pbar.update(1)
        print(f'[✓] Stage 1 completed in {stage_time:.1f}s')
        
        # Stage 2
        stage_start = time.time()
        main_pbar.set_description('🧠 Stage 2/3: Computing embeddings')
        X_all = get_embeddings(all_sents, MODEL_NAME, BATCH_SIZE, DEVICE, args.embeds_file)
        stage_time = time.time() - stage_start
        main_pbar.update(1)
        print(f'[✓] Stage 2 completed in {stage_time:.1f}s')
        
        # Stage 3
        stage_start = time.time()
        main_pbar.set_description('📊 Stage 3/3: Computing metrics')
        compute_metrics(X_all, rel_slice, wb_run, metric_settings)
        stage_time = time.time() - stage_start
        main_pbar.update(1)
        print(f'[✓] Stage 3 completed in {stage_time:.1f}s')
        
        main_pbar.set_description('✅ Analysis Complete!')
    
    print('\n' + '='*78)
    print('🎉 QXDM Log Analysis Completed Successfully!')
    print('='*78)

    if wb_run:
        wb_run.finish()


if __name__ == '__main__':
    main()
