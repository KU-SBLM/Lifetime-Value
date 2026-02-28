#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import fbeta_score

from .config import DELTA_AROUND, CUT_STEP, F_BETA_SCORE

def fit_imputer(train_df: pd.DataFrame):
    num_cols = [
        c for c in train_df.columns
        if train_df[c].dtype != 'object' and not str(train_df[c].dtype).startswith('category')
    ]
    med = train_df[num_cols].median(numeric_only=True)
    return num_cols, med

def apply_imputer(df: pd.DataFrame, num_cols: List[str], med: pd.Series):
    df = df.copy()
    df[num_cols] = df[num_cols].fillna(med)
    df = df.fillna(0)
    return df

def _sanitize_cols(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = out.columns.astype(str).str.replace(r"\s+", "_", regex=True)
    out.columns = [col.replace('[', '').replace(']', '').replace('<', '') for col in out.columns]
    return out

def build_features(df: pd.DataFrame, target_col: str, drop_cols: List[str]):
    cols = [c for c in df.columns if c not in drop_cols and c != target_col]
    cat_cols = [c for c in cols if df[c].dtype == "object" or str(df[c].dtype).startswith("category")]
    return df[cols].copy(), cols, cat_cols

def hard_vote(preds: Dict[str, np.ndarray], cutoffs: Dict[str, float]) -> np.ndarray:
    votes = []
    for k in ["cat", "lgbm", "xgb"]:
        if k in preds:
            t = cutoffs[k]
            p = preds[k]
            votes.append((p >= t).astype(int))
    if not votes:
        return np.array([])
    votes = np.column_stack(votes)
    threshold = int(math.ceil(votes.shape[1] / 2))
    return (votes.sum(axis=1) >= threshold).astype(int)

def get_train_val_test_indices(X, y, seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    test_size_ratio = 0.2
    val_size_ratio = 0.2
    val_split_ratio = val_size_ratio / (1.0 - test_size_ratio)

    sss_test = StratifiedShuffleSplit(n_splits=1, test_size=test_size_ratio, random_state=seed)
    train_val_index, test_index = next(sss_test.split(X, y))

    X_train_val = X.iloc[train_val_index]
    y_train_val = y[train_val_index]

    sss_val = StratifiedShuffleSplit(n_splits=1, test_size=val_split_ratio, random_state=seed)
    train_index_in_tv, val_index_in_tv = next(sss_val.split(X_train_val, y_train_val))

    train_index = train_val_index[train_index_in_tv]
    val_index = train_val_index[val_index_in_tv]
    return train_index, val_index, test_index

def optimize_cutoff(y_true, y_proba, center, delta=DELTA_AROUND, step=CUT_STEP, beta=F_BETA_SCORE):
    lo = max(0.0, center - delta)
    hi = min(1.0, center + delta)
    grid = np.arange(lo, hi + 1e-9, step)

    best_t, best_score = center, -1.0
    for t in grid:
        y_pred = (y_proba >= t).astype(int)
        score = fbeta_score(y_true, y_pred, beta=beta, zero_division=0)
        if score > best_score:
            best_score = score
            best_t = t
    return float(best_t)

sanitize_cols = _sanitize_cols
