"""
pipeline/train.py
Orchestrates: preprocessing → features → training → save artifacts.
Run with:  python pipeline/train.py
"""
import json, os, sys
from datetime import datetime

import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from pipeline.data_preprocessing import load_and_clean, classify_columns, build_full_pipeline
from pipeline.feature_engineering import FeatureEngineer
from pipeline.model_training import train_all


def main():
    print("=== LeadScore AI — Training Pipeline ===\n")

    # 1. Load & clean
    print("[1/5] Loading and cleaning data...")
    df = load_and_clean()
    print(f"      {len(df)} rows × {df.shape[1]} cols after cleaning")

    # 2. Split
    print("[2/5] Train/test split (80/20)...")
    X = df.drop(columns=["Converted"])
    y = df["Converted"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE,
        stratify=y, random_state=config.RANDOM_STATE,
    )
    print(f"      Train: {len(X_train)}  |  Test: {len(X_test)}")

    # 3. Feature engineering
    print("[3/5] Feature engineering...")
    fe = FeatureEngineer()
    X_train_fe = fe.fit_transform(X_train)
    X_test_fe = fe.transform(X_test)

    # 4. Preprocessing
    print("[4/5] Building & fitting preprocessor...")
    num_cols, low_cats, high_cats = classify_columns(
        X_train_fe.assign(Converted=y_train.values)
    )
    preprocessor = build_full_pipeline(num_cols, low_cats, high_cats)
    X_train_t = preprocessor.fit_transform(X_train_fe)
    X_test_t = preprocessor.transform(X_test_fe)
    print(f"      Transformed shape: {X_train_t.shape}")

    # Bundle feature engineer + preprocessor so inference needs only one .pkl
    full_pipeline = Pipeline([
        ("feature_engineer", fe),
        ("preprocessor", preprocessor),
    ])

    # 5. Train & select best
    print("[5/5] Training 6 candidate models...")
    best_model, best_name, all_metrics = train_all(
        X_train_t, y_train.values, X_test_t, y_test.values,
    )
    best_metrics = next(m for m in all_metrics if m["model_name"] == best_name)

    # Save
    os.makedirs(config.MODELS_DIR, exist_ok=True)
    joblib.dump(best_model, config.MODEL_PATH)
    joblib.dump(full_pipeline, config.PREPROCESSOR_PATH)



    meta = {
        "best_model": best_name,
        "best_accuracy": best_metrics["accuracy"],
        "best_f1": best_metrics["f1"],
        "training_date": datetime.now().isoformat(),
        "all_models": all_metrics,
    }
    with open(config.METADATA_PATH, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nWinner: {best_name}  (accuracy={best_metrics['accuracy']})")
    print(f"Saved:  {config.MODEL_PATH}")
    print(f"        {config.PREPROCESSOR_PATH}")
    print(f"        {config.METADATA_PATH}")
    print("\n=== Done ===")


if __name__ == "__main__":
    main()