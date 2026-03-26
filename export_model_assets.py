"""One-off: load Pune-House-Price.pkl with sklearn and write model_weights.npz + model_features.json.

Run after retraining the notebook so the Streamlit app can load the model without sklearn.
Requires: pip install scikit-learn numpy
"""
import json
import pickle
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
PKL = ROOT / "Pune-House-Price.pkl"


def main():
    with open(PKL, "rb") as f:
        m = pickle.load(f)
    coef = np.asarray(m.coef_, dtype=np.float64)
    intercept = float(m.intercept_)
    cols = list(m.feature_names_in_)
    np.savez_compressed(ROOT / "model_weights.npz", coef=coef, intercept=intercept)
    with open(ROOT / "model_features.json", "w", encoding="utf-8") as f:
        json.dump(cols, f, ensure_ascii=False, indent=0)
    print(f"Wrote model_weights.npz and model_features.json ({len(cols)} features).")


if __name__ == "__main__":
    main()
