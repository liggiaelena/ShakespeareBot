"""
Stage 1 — prepare.py
Reads all .txt files from data/docs/, chunks them using the same logic
as rag/ingestor.py, labels them (Tragedy=1 / Non-Tragedy=0), balances
the classes, and saves the dataset to data/processed/.
"""
import json
import os
import numpy as np

# ── Paths ─────────────────────────────────────────────────────────────────────
DOCS_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'docs')
OUT_DIR   = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'processed')
os.makedirs(OUT_DIR, exist_ok=True)

# ── Labels — identical to ShakespeareBot_NLP.ipynb ───────────────────────────
DOC_LABELS = {
    'pg16966.txt': 1,  # Shakespearean Tragedy (Bradley)
    'pg26315.txt': 0,  # Shakespeare's Family — biography
    'pg47715.txt': 1,  # Cambridge Vol. 7 — Macbeth, Romeo & Juliet, Timon
    'pg49007.txt': 1,  # Cambridge Vol. 6 — Coriolanus, Titus, Troilus
    'pg50095.txt': 0,  # Cambridge Vol. 4 — histories
    'pg23041.txt': 1,  # Cambridge Vol. 1
    'pg45128.txt': 0,  # Cambridge Vol. 2 — comedies
    'pg49008.txt': 1,  # Cambridge Vol. 8 — Antony, Cymbeline, Pericles
    'pg49297.txt': 0,  # Cambridge Vol. 5 — comedies
    'pg50559.txt': 0,  # Cambridge Vol. 3 — comedies
}


def _chunk_text(text: str, chunk_size: int = 300, overlap: int = 50) -> list:
    """Exact copy of the chunking logic from rag/ingestor.py and the notebook."""
    words = text.split()
    chunks, i = [], 0
    while i < len(words):
        chunk = ' '.join(words[i: i + chunk_size])
        if len(chunk.strip()) > 30:
            chunks.append(chunk)
        i += chunk_size - overlap
    return chunks


def main():
    all_chunks, all_labels, all_sources = [], [], []

    for filename, label in DOC_LABELS.items():
        filepath = os.path.join(DOCS_PATH, filename)
        if not os.path.exists(filepath):
            print(f'  [WARN] Not found: {filename} - skipping')
            continue
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            text = f.read()
        chunks = _chunk_text(text)
        all_chunks.extend(chunks)
        all_labels.extend([label] * len(chunks))
        all_sources.extend([filename] * len(chunks))
        print(f'  {filename}: {len(chunks):,} chunks -> {"TRAGEDY" if label else "NON-TRAGEDY"}')

    print(f'\nRaw total  : {len(all_chunks):,}')
    print(f'Tragedy (1): {sum(all_labels):,}')
    print(f'Non-Trag(0): {len(all_labels) - sum(all_labels):,}')

    # Balance classes — same logic as notebook
    np.random.seed(42)
    trag_idx = [i for i, l in enumerate(all_labels) if l == 1]
    non_idx  = [i for i, l in enumerate(all_labels) if l == 0]
    min_n    = min(len(trag_idx), len(non_idx))

    sel_t = np.random.choice(trag_idx, min_n, replace=False)
    sel_n = np.random.choice(non_idx,  min_n, replace=False)
    sel_idx = np.concatenate([sel_t, sel_n])
    np.random.shuffle(sel_idx)

    X = [all_chunks[i] for i in sel_idx]
    y = [all_labels[i] for i in sel_idx]

    print(f'\nBalanced   : {len(X):,} samples ({min_n} per class)')

    # Save chunks.json
    chunks_path = os.path.join(OUT_DIR, 'chunks.json')
    with open(chunks_path, 'w', encoding='utf-8') as f:
        json.dump({'X': X, 'y': y}, f)
    print(f'Saved -> {chunks_path}')

    # Save stats.json
    stats = {
        'total_samples': len(X),
        'tragedy_count': int(sum(y)),
        'non_tragedy_count': int(len(y) - sum(y)),
        'chunk_size': 300,
        'overlap': 50,
        'docs_loaded': len(DOC_LABELS),
    }
    stats_path = os.path.join(OUT_DIR, 'stats.json')
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f'Saved -> {stats_path}')


if __name__ == '__main__':
    main()
