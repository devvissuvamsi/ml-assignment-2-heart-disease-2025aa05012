
# utils/data.py

import pandas as pd
from typing import List

def load_and_clean_heart_csv(
    path: str,
    target_col: str,
    num_cols: List[str],
    cat_cols: List[str],
) -> pd.DataFrame:
    """
    Loads heart.csv, enforces required columns, drops NA rows on target/features,
    removes exact duplicates AND feature-only duplicates before splitting.
    Returns a clean DataFrame ready for train/val split.
    """
    df = pd.read_csv(path)

    # 1) schema checks
    required = [target_col] + num_cols + cat_cols
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}")

    # 2) basic NA policy
    df = df.dropna(subset=[target_col])
    df = df.dropna(subset=num_cols + cat_cols)

    # 3) exact duplicates (features + target)
    before = len(df)
    df = df.drop_duplicates()
    after_full = len(df)

    # 4) feature-only duplicates (keep first)
    df = df.drop_duplicates(subset=num_cols + cat_cols, keep="first")
    after_feat = len(df)

    print(f"[DEDUP] original={before}, after_full={after_full}, after_features={after_feat}")
    return df
