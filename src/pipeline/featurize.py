"""
Stage 2 — featurize.py
Loads the balanced dataset from data/processed/chunks.json,
applies TF-IDF vectorization, performs a 75/25 stratified train/test split,
and saves the feature matrix and split indices to data/features/.
"""
import json
import os
import pickle

import numpy as np
from scipy.sparse import save_npz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR  = os.path.join(os.path.dirname(__file__), '..', '..')
IN_PATH   = os.path.join(BASE_DIR, 'data', 'processed', 'chunks.json')
OUT_DIR   = os.path.join(BASE_DIR, 'data', 'features')
os.makedirs(OUT_DIR, exist_ok=True)


def main():
    # Load balanced dataset
    with open(IN_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)
    X, y = data['X'], data['y']
    print(f'Loaded {len(X):,} samples')

    # TF-IDF — identical params to notebook
    tfidf = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        stop_words='english',
        min_df=2,
    )
    X_tfidf = tfidf.fit_transform(X)
    print(f'TF-IDF shape    : {X_tfidf.shape}')
    print(f'Vocabulary size : {len(tfidf.vocabulary_):,}')
    sparsity = 100 * (1 - X_tfidf.nnz / (X_tfidf.shape[0] * X_tfidf.shape[1]))
    print(f'Sparsity        : {sparsity:.1f}%')

    # Train/test split — 75/25, stratified, random_state=42
    X_train, X_test, y_train, y_test = train_test_split(
        X_tfidf, y, test_size=0.25, random_state=42, stratify=y
    )
    print(f'Train size : {X_train.shape[0]:,}')
    print(f'Test size  : {X_test.shape[0]:,}')

    # Save TF-IDF matrix (full, unsplit — train/test stored in split file)
    matrix_path = os.path.join(OUT_DIR, 'tfidf_matrix.npz')
    save_npz(matrix_path, X_tfidf)
    print(f'Saved -> {matrix_path}')

    # Save vectorizer
    vec_path = os.path.join(OUT_DIR, 'tfidf_vectorizer.pkl')
    with open(vec_path, 'wb') as f:
        pickle.dump(tfidf, f)
    print(f'Saved -> {vec_path}')

    # Save train/test split
    split_path = os.path.join(OUT_DIR, 'train_test_split.pkl')
    with open(split_path, 'wb') as f:
        pickle.dump({
            'X_train': X_train,
            'X_test':  X_test,
            'y_train': y_train,
            'y_test':  y_test,
        }, f)
    print(f'Saved -> {split_path}')


if __name__ == '__main__':
    main()
