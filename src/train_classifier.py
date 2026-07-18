"""
Train the ASL sign classifier from collected samples.

    python src/train_classifier.py --data data/landmarks.csv --out models/sign_model.joblib

Uses a StandardScaler + classifier pipeline (KNN by default; MLP optional) and
reports cross-validated accuracy so you know how trustworthy the model is before
demoing. KNN is a good default here: with clean, normalized landmark features
and a few dozen samples per letter it's accurate and trains instantly.
"""

import argparse
import csv

import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import joblib

from gesture_features import FEATURE_DIM


def load_dataset(path):
    X, y = [], []
    with open(path, newline="") as f:
        for row in csv.reader(f):
            if not row or len(row) != FEATURE_DIM + 1:
                continue
            y.append(row[0])
            X.append([float(v) for v in row[1:]])
    return np.array(X, dtype=np.float64), np.array(y)


def build_model(kind):
    if kind == "mlp":
        clf = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=1500,
                            early_stopping=True, random_state=0)
    else:
        clf = KNeighborsClassifier(n_neighbors=5, weights="distance")
    return make_pipeline(StandardScaler(), clf)


def main():
    ap = argparse.ArgumentParser(description="Train the ASL sign classifier.")
    ap.add_argument("--data", default="data/landmarks.csv")
    ap.add_argument("--out", default="models/sign_model.joblib")
    ap.add_argument("--model", choices=["knn", "mlp"], default="knn")
    args = ap.parse_args()

    X, y = load_dataset(args.data)
    if len(X) == 0:
        raise SystemExit(f"No samples found in {args.data}. Run collect_data.py first.")

    labels, counts = np.unique(y, return_counts=True)
    print(f"Loaded {len(X)} samples across {len(labels)} labels:")
    for lab, n in zip(labels, counts):
        print(f"  {lab}: {n}")

    model = build_model(args.model)

    # Cross-validate when every class has enough samples for the folds.
    min_count = counts.min()
    if len(labels) >= 2 and min_count >= 3:
        folds = int(min(5, min_count))
        scores = cross_val_score(model, X, y, cv=folds)
        print(f"\n{folds}-fold CV accuracy: "
              f"{scores.mean():.3f} +/- {scores.std():.3f}")
    else:
        print("\n(Not enough samples per class for cross-validation — "
              "collect more for a reliable accuracy estimate.)")

    model.fit(X, y)

    import os
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    joblib.dump({"model": model, "labels": sorted(labels.tolist())}, args.out)
    print(f"\nSaved model -> {args.out}")


if __name__ == "__main__":
    main()
