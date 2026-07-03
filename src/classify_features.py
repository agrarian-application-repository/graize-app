"""
classify_features.py
─────────────────────────────────────────────────────────────────────────────
Command-line tool: load a pre-extracted features CSV (as saved by classify_gui
or classify_gradio), dequantize the feature codes, and classify each window.

Expected CSV layout (header row required, column names are ignored for the
feature columns — only position matters):
    col 0: t0   — window start in seconds
    col 1: t1   — window end in seconds
    col 2–7:    — 6 uint8 feature codes in QUANT_RANGES order:
                  kurtosis_z, zcr_z, kurtosis_pitch, zcr_pitch,
                  subband_ratio_z, static_dynamic_ratio

Usage:
    python classify_features.py model.pkl features.csv
    python classify_features.py model.pkl features.csv --output results.csv
"""

import argparse
import os
import sys

import joblib
import pandas as pd

# Feature order and dequantization ranges — must match shape_features_window.py
# and configs/quantization_ranges.json.  Column names in the input CSV are
# ignored; the 6 feature columns are consumed in this exact order.
QUANT_RANGES = {
    "kurtosis_z":           (0.0, 1024.0),
    "zcr_z":                (0.0,    1.0),
    "kurtosis_pitch":       (0.0, 1024.0),
    "zcr_pitch":            (0.0,    1.0),
    "subband_ratio_z":      (0.0,   16.0),
    "static_dynamic_ratio": (0.0,    1.0),
}

N_FEATURES = len(QUANT_RANGES)
FEATURE_NAMES = list(QUANT_RANGES.keys())


def load_model(model_path: str) -> dict:
    artifact = joblib.load(model_path)
    return {
        "model":  artifact["model"],
        "scaler": artifact["scaler"],
    }


def dequantize(codes: pd.DataFrame) -> pd.DataFrame:
    """Convert uint8 codes (0–255) to floats using QUANT_RANGES order."""
    out = pd.DataFrame(index=codes.index)
    for col_idx, (name, (lo, hi)) in enumerate(QUANT_RANGES.items()):
        out[name] = lo + codes.iloc[:, col_idx] / 255.0 * (hi - lo)
    return out


def main():
    parser = argparse.ArgumentParser(
        description="Classify pre-extracted feature windows from a CSV file.")
    parser.add_argument("model_path",
                        help="Path to the trained model .pkl artifact.")
    parser.add_argument("features_csv",
                        help="Path to the features CSV (t0, t1, then 6 feature codes).")
    parser.add_argument("--output", "-o", default=None,
                        help="Optional path to save results as CSV (default: print to stdout).")
    args = parser.parse_args()

    if not os.path.exists(args.model_path):
        sys.exit(f"Error: model not found: {args.model_path}")
    if not os.path.exists(args.features_csv):
        sys.exit(f"Error: features CSV not found: {args.features_csv}")

    df = pd.read_csv(args.features_csv)

    n_cols = df.shape[1]
    if n_cols < 2 + N_FEATURES:
        sys.exit(f"Error: expected at least {2 + N_FEATURES} columns "
                 f"(t0, t1, {N_FEATURES} features), got {n_cols}.")

    t0 = df.iloc[:, 0]
    t1 = df.iloc[:, 1]
    
    codes = df.iloc[:, 2: 2 + N_FEATURES]#6 numbers each 1 byte (like np.uint8)
    bundle = load_model(args.model_path)
    feats = dequantize(codes)
    X = bundle["scaler"].transform(feats.values)
    preds = bundle["model"].predict(X)

    results = pd.DataFrame({
        "window":     range(1, len(preds) + 1),
        "t0":         t0.values,
        "t1":         t1.values,
        "prediction": preds,
    })

    if args.output:
        results.to_csv(args.output, index=False)
        print(f"Saved {len(results)} window(s) to {args.output}")
    else:
        print(f"{'Window':>6}  {'Start (s)':>10}  {'End (s)':>8}  Prediction")
        print(f"{'------':>6}  {'----------':>10}  {'--------':>8}  ----------")
        for _, row in results.iterrows():
            print(f"{int(row['window']):>6}  {row['t0']:>10.0f}  {row['t1']:>8.0f}  {row['prediction']}")


if __name__ == "__main__":
    main()
