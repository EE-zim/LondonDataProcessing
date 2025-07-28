"""
Sentence embedding with checkpointing for QXDM logs.
"""
import time
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer


def get_embeddings(
    all_sents: list[str],
    model_name: str,
    batch_size: int,
    device: str,
    embeds_file: str,
) -> np.ndarray:
    """Encode sentences into embeddings, with npz checkpoint."""
    ep = Path(embeds_file)
    if ep.exists():
        print(f'[✓] Loading embeddings from {ep}')
        with np.load(ep) as npz:
            return npz['X_all']

    print(f'[i] Encoding {len(all_sents):,} sentences with {model_name} on {device}')
    t0 = time.time()
    model = SentenceTransformer(model_name, device=device)
    print(f'[✓] Model loaded in {time.time()-t0:.1f}s')

    if device == 'cuda' and torch.cuda.is_available():
        free = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()
        free_gb = free / (1024**3)
        print(f'[i] GPU memory free: {free_gb:.1f} GB')
        if all_sents:
            avg_len = sum(len(s) for s in all_sents[:1000]) / min(1000, len(all_sents))
            if avg_len > 100 and batch_size > 384:
                batch_size = 384
                print(f'[i] Reduced batch_size to {batch_size} for long sentences')
    
    print('[i] Starting encoding batches...')
    emb = model.encode(
        all_sents,
        batch_size=batch_size,
        device=device,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    X_all = np.asarray(emb, dtype=np.float32)
    np.savez_compressed(ep, X_all=X_all)
    print(f'[✓] Saved embeddings → {ep}')
    return X_all
