"""
Sentence extraction and checkpointing for QXDM logs.
"""
import pickle
from pathlib import Path
from typing import Tuple, Dict, List

from tqdm.auto import tqdm

from .config import find_supported_files, _SUPPORTED_EXTENSIONS_STR
from .log_processor import QXDMLogProcessor


def get_sentences(
    qxdm_root: Path,
    block_size: int,
    settings: dict,
    checkpoint_file: str,
    reset_checkpoint: bool,
) -> Tuple[List[str], Dict[str, slice]]:
    """Return (all_sentences, category_slices) from checkpoint or by recomputing."""
    cp = Path(checkpoint_file)
    if cp.exists() and not reset_checkpoint:
        print(f'[✓] Loading sentences from {cp}')
        with cp.open('rb') as f:
            return pickle.load(f)
    if cp.exists():
        cp.unlink()
        print(f'[i] Removed old checkpoint: {cp}')

    print('[i] Starting QXDM log processing with specialized parser...')
    try:
        log_processor = QXDMLogProcessor(use_full_model=True)
        print('[✓] Initialized QXDM log processor with full spaCy model')
    except Exception as e:
        print(f'[warn] Full model unavailable ({e}), using lightweight model')
        log_processor = QXDMLogProcessor(use_full_model=False)
        print('[✓] Initialized QXDM log processor with blank spaCy model')

    if not qxdm_root.exists():
        print(f'[!] QXDM_ROOT path does not exist: {qxdm_root}')
        print('[!] Please set QXDM_ROOT or check the path')
        return [], {}

    # Determine log categories or all_logs
    categories = sorted([d.name for d in qxdm_root.iterdir() if d.is_dir()], key=str.lower)
    if not categories:
        files = find_supported_files(qxdm_root)
        if files:
            categories = ['all_logs']
        else:
            print(f'[!] No supported log files found in {qxdm_root}')
            print(f'[!] Supported extensions: {_SUPPORTED_EXTENSIONS_STR}')
            return [], {}

    print('   Log categories:', ', '.join(categories))
    all_sents: List[str] = []
    rel_slice: Dict[str, slice] = {}

    stats = log_processor.get_stats()
    print(f'[i] Model: {stats["model_type"]}, components: {stats["pipeline_components"]}, '
          f'{sum(stats["patterns_count"].values())} rules')

    for cat in tqdm(categories, desc='Processing categories', unit='category'):
        start = len(all_sents)
        src = qxdm_root if cat == 'all_logs' else (qxdm_root / cat)
        files = find_supported_files(src)
        print(f'   [{cat}] processing {len(files)} files')
        sent_cat: List[str] = []
        for f in tqdm(files, desc=f'{cat} files', unit='file', leave=False):
            try:
                sent_cat.extend(log_processor.process_file(f))
            except Exception as e:
                print(f'[warn] Error processing {f.name}: {e}')
        all_sents.extend(sent_cat)
        rel_slice[cat] = slice(start, len(all_sents))
        print(f'   [{cat}] → {len(sent_cat):,} sentences')

    print(f'[✓] Collected {len(all_sents):,} sentences; saving checkpoint → {cp}')
    with cp.open('wb') as f:
        pickle.dump((all_sents, rel_slice), f)
    return all_sents, rel_slice
