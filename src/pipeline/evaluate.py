"""
Stage 4 — evaluate.py
Loads all 3 trained models and the test split, computes metrics,
saves metrics/scores.json and metrics/confusion_matrices.png.
"""
import json
import os
import pickle

import matplotlib
matplotlib.use('Agg')  # non-interactive backend — no display needed
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.join(os.path.dirname(__file__), '..', '..')
FEAT_DIR    = os.path.join(BASE_DIR, 'data', 'features')
MODEL_DIR   = os.path.join(BASE_DIR, 'model')
METRICS_DIR = os.path.join(BASE_DIR, 'metrics')
os.makedirs(METRICS_DIR, exist_ok=True)


def _load(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def main():
    # Load test split
    split = _load(os.path.join(FEAT_DIR, 'train_test_split.pkl'))
    X_test  = split['X_test']
    y_test  = split['y_test']

    # Load models and transformers
    nb      = _load(os.path.join(MODEL_DIR, 'nb_tfidf.pkl'))
    svd     = _load(os.path.join(MODEL_DIR, 'svd.pkl'))
    lr_svd  = _load(os.path.join(MODEL_DIR, 'lr_svd.pkl'))
    scaler  = _load(os.path.join(MODEL_DIR, 'scaler.pkl'))
    pca     = _load(os.path.join(MODEL_DIR, 'pca.pkl'))
    lr_pca  = _load(os.path.join(MODEL_DIR, 'lr_pca.pkl'))

    # Transform test data
    X_test_svd = svd.transform(X_test)
    X_test_pca = pca.transform(scaler.transform(X_test).toarray())

    # Predictions
    nb_pred  = nb.predict(X_test)
    svd_pred = lr_svd.predict(X_test_svd)
    pca_pred = lr_pca.predict(X_test_pca)

    # Metrics
    def metrics(y_true, y_pred, name):
        return {
            'model':     name,
            'accuracy':  round(accuracy_score(y_true, y_pred), 4),
            'precision': round(precision_score(y_true, y_pred), 4),
            'recall':    round(recall_score(y_true, y_pred), 4),
            'f1':        round(f1_score(y_true, y_pred), 4),
        }

    results = [
        metrics(y_test, nb_pred,  'NaiveBayes_TFIDF'),
        metrics(y_test, svd_pred, 'LogisticRegression_SVD'),
        metrics(y_test, pca_pred, 'LogisticRegression_PCA'),
    ]

    for r in results:
        print(f"{r['model']:30s}  acc={r['accuracy']:.4f}  "
              f"prec={r['precision']:.4f}  rec={r['recall']:.4f}  f1={r['f1']:.4f}")

    # Save scores.json
    scores_path = os.path.join(METRICS_DIR, 'scores.json')
    with open(scores_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nSaved -> {scores_path}')

    # Confusion matrices plot
    cms = [
        (confusion_matrix(y_test, nb_pred),  results[0]['accuracy'], 'Model 1\nNaive Bayes + TF-IDF',     'Reds'),
        (confusion_matrix(y_test, svd_pred), results[1]['accuracy'], 'Model 2\nLogistic Regression + SVD', 'Blues'),
        (confusion_matrix(y_test, pca_pred), results[2]['accuracy'], 'Model 3\nLogistic Regression + PCA', 'Greens'),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for ax, (cm, acc, title, cmap) in zip(axes, cms):
        sns.heatmap(cm, annot=True, fmt='d', cmap=cmap,
                    xticklabels=['Non-Tragedy', 'Tragedy'],
                    yticklabels=['Non-Tragedy', 'Tragedy'],
                    linewidths=1, linecolor='white',
                    annot_kws={'size': 18, 'weight': 'bold'}, ax=ax)
        ax.set_title(f'{title}\nAccuracy: {acc*100:.2f}%', fontsize=12, fontweight='bold')
        ax.set_xlabel('Predicted Label', fontsize=11)
        ax.set_ylabel('Actual Label', fontsize=11)
        tn, fp, fn, tp = cm.ravel()
        for (r, c), lbl in [((0,0),'TN'),((0,1),'FP'),((1,0),'FN'),((1,1),'TP')]:
            ax.text(c+0.5, r+0.75, lbl, ha='center', va='center',
                    fontsize=9, color='white', fontweight='bold', alpha=0.8)

    plt.suptitle('ShakespeareBot — Confusion Matrices (All 3 Models)',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    cm_path = os.path.join(METRICS_DIR, 'confusion_matrices.png')
    plt.savefig(cm_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved -> {cm_path}')


if __name__ == '__main__':
    main()
