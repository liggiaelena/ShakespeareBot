"""
Stage 3 — train.py
Loads the TF-IDF features and train/test split, trains three models
exactly as in ShakespeareBot_NLP.ipynb, and saves all models and
transformers to model/.

Models:
  1. MultinomialNB(alpha=0.1)          on raw TF-IDF
  2. LogisticRegression                on TruncatedSVD(100 components)
  3. LogisticRegression                on PCA(100 components) after StandardScaler
"""
import os
import pickle

import numpy as np
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import StandardScaler

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.join(os.path.dirname(__file__), '..', '..')
FEAT_DIR   = os.path.join(BASE_DIR, 'data', 'features')
MODEL_DIR  = os.path.join(BASE_DIR, 'model')
os.makedirs(MODEL_DIR, exist_ok=True)

N_COMPONENTS = 100


def _save(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)
    print(f'Saved -> {path}')


def main():
    # Load split
    split_path = os.path.join(FEAT_DIR, 'train_test_split.pkl')
    with open(split_path, 'rb') as f:
        split = pickle.load(f)
    X_train = split['X_train']
    y_train = split['y_train']
    print(f'Train set : {X_train.shape[0]:,} samples, {X_train.shape[1]:,} features')

    # ── Model 1: Naive Bayes on TF-IDF ────────────────────────────────────────
    print('\nTraining Model 1 - MultinomialNB + TF-IDF ...')
    nb = MultinomialNB(alpha=0.1)
    nb.fit(X_train, y_train)
    _save(nb, os.path.join(MODEL_DIR, 'nb_tfidf.pkl'))

    # ── Model 2: Logistic Regression on SVD ───────────────────────────────────
    print('\nTraining Model 2 - LogisticRegression + SVD ...')
    svd = TruncatedSVD(n_components=N_COMPONENTS, random_state=42)
    X_train_svd = svd.fit_transform(X_train)
    explained_svd = svd.explained_variance_ratio_.sum()
    print(f'  SVD explained variance: {explained_svd*100:.1f}%')
    _save(svd, os.path.join(MODEL_DIR, 'svd.pkl'))

    lr_svd = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
    lr_svd.fit(X_train_svd, y_train)
    _save(lr_svd, os.path.join(MODEL_DIR, 'lr_svd.pkl'))

    # ── Model 3: Logistic Regression on PCA ───────────────────────────────────
    print('\nTraining Model 3 - LogisticRegression + PCA ...')
    scaler = StandardScaler(with_mean=False)
    X_train_scaled = scaler.fit_transform(X_train)
    _save(scaler, os.path.join(MODEL_DIR, 'scaler.pkl'))

    pca = PCA(n_components=N_COMPONENTS, random_state=42)
    X_train_pca = pca.fit_transform(X_train_scaled.toarray())
    explained_pca = pca.explained_variance_ratio_.sum()
    print(f'  PCA explained variance: {explained_pca*100:.1f}%')
    _save(pca, os.path.join(MODEL_DIR, 'pca.pkl'))

    lr_pca = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
    lr_pca.fit(X_train_pca, y_train)
    _save(lr_pca, os.path.join(MODEL_DIR, 'lr_pca.pkl'))

    print('\nAll models saved.')


if __name__ == '__main__':
    main()
