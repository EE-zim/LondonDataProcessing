"""
Metric computations for QXDM log embeddings.
"""
import os
import re
from itertools import combinations

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from .utils import get_optimal_settings, get_smart_sample_size
from .metric_utils import (
    semantic_spread,
    redundancy_index,
    cluster_entropy,
    change_mag,
    novelty_density,
)


def compute_metrics(
    X_all: np.ndarray,
    rel_slice: dict,
    wb_run,
    metric_settings: dict = None,
) -> None:
    """Compute and save per-category and delta metrics."""
    print(f'[i] Computing metrics for {len(rel_slice)} categories...')
    if metric_settings is None:
        metric_settings = {'device': 'cpu'}

    SEED = int(os.environ.get('RANDOM_SEED', 42))
    rng = np.random.default_rng(SEED)
    metric_settings['rng'] = rng

    if not rel_slice:
        print('[!] No categories found; skipping metrics.')
        return
    if X_all.size == 0:
        print('[!] No embeddings; skipping metrics.')
        return

    # Optimize numpy
    threads = min(os.cpu_count() or 4, 8)
    if os.name != 'nt':
        try:
            import mkl

            mkl.set_num_threads(threads)
            print(f'[i] MKL threads set to {threads}')
        except ImportError:
            pass

    settings = get_optimal_settings()
    print('[i] Smart sampling settings:')
    print(f'   → Base sample size: {settings["base_sample_size"]:,}')
    print(f'   → Max sample size: {settings["max_sample_size"]:,}')
    print(f'   → Sampling threshold: {settings["sample_threshold"]:,}')

    # Per-category metrics
    metrics, pools = [], {}
    categories = list(rel_slice.items())
    with tqdm(total=len(categories), desc='Category metrics', unit='cat') as cat_pbar:
        for r, sl in categories:
            X = X_all[sl]
            if X.size == 0:
                cat_pbar.update(1)
                continue

            print(f'[📊] Computing metrics for {r} ({len(X):,} sentences)')
            mus = X.mean(0)
            sample_size = get_smart_sample_size(len(X), settings, r)
            if sample_size < len(X):
                idx = rng.choice(len(X), sample_size, replace=False)
                pools[r] = X[idx]
            else:
                pools[r] = X

            with tqdm(total=3, desc=f'{r} metrics', leave=False) as mpb:
                ss = semantic_spread(X, device=metric_settings['device'])
                mpb.update(1)
                ri = redundancy_index(X, device=metric_settings['device'], rng=metric_settings['rng'])
                mpb.update(1)
                ce = cluster_entropy(X, device=metric_settings['device'], rng=metric_settings['rng'])
                mpb.update(1)

            metrics.append({
                'category': r,
                'sentences': len(X),
                'semantic_spread': ss,
                'redundancy_index': ri,
                'cluster_entropy': ce,
            })
            if wb_run:
                step = int(re.findall(r'\d+', r)[0]) if re.findall(r'\d+', r) else len(metrics)
                wb_run.log({f'category/{k}': v for k, v in metrics[-1].items() if k != 'category'}, step=step)
            cat_pbar.update(1)

    df_rel = pd.DataFrame(metrics)
    if df_rel.empty:
        df_rel.to_csv('release_metrics.csv', index=False)
        pd.DataFrame(columns=['from', 'to', 'change_magnitude', 'novelty_density']).to_csv('delta_metrics.csv', index=False)
        return

    df_rel.to_csv('release_metrics.csv', index=False)
    print('[✓] Saved release_metrics.csv')

    # Delta metrics
    delta_rows = []
    rel_list = df_rel['category'].tolist()
    if len(rel_list) > 1:
        pairs = list(combinations(rel_list, 2))
        with tqdm(total=len(pairs) * 2, desc='Delta metrics', unit='comp') as dpb:
            for a, b in pairs:
                cm = change_mag(pools[a], pools[b], device=metric_settings['device'])
                nd_ab = novelty_density(pools[a], pools[b], device=metric_settings['device'], rng=metric_settings['rng'])
                delta_rows.append({'from': a, 'to': b, 'change_magnitude': cm, 'novelty_density': nd_ab})
                dpb.update(1)

                nd_ba = novelty_density(pools[b], pools[a], device=metric_settings['device'], rng=metric_settings['rng'])
                delta_rows.append({'from': b, 'to': a, 'change_magnitude': cm, 'novelty_density': nd_ba})
                dpb.update(1)
                if wb_run:
                    step_a = hash(a) % 1000
                    step_b = hash(b) % 1000
                    wb_run.log({f'delta/{k}': v for k, v in delta_rows[-2].items() if k not in ('from', 'to')}, step=step_a)
                    wb_run.log({f'delta/{k}': v for k, v in delta_rows[-1].items() if k not in ('from', 'to')}, step=step_b)

    df_delta = pd.DataFrame(delta_rows)
    if not df_delta.empty:
        df_delta = df_delta.sort_values(['from', 'to']).reset_index(drop=True)
    df_delta.to_csv('delta_metrics.csv', index=False)
    print('[✓] Saved delta_metrics.csv')

    if wb_run:
        import wandb
        art = wandb.Artifact('qxdm_log_metrics', type='dataset')
        art.add_file('release_metrics.csv')
        art.add_file('delta_metrics.csv')
        wb_run.log_artifact(art)
        print('[✓] Uploaded artifacts to W&B')
