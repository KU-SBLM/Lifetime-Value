#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import math
import json
import logging
import traceback
from datetime import datetime
from logging import FileHandler, StreamHandler
from typing import Dict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")

from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
from catboost import CatBoostClassifier, Pool

from .config import (
    ANALYSIS_DIR, UNIFIED_DATA_PATH, TARGET_COL, ID_COL, SEEDS_TO_RUN,
    STAGE1_FIXED, CAT_TASK_PARAMS
)
from .utils import OrdinalCategoryEncoder
from .data_preprocessing import (
    build_features, fit_imputer, apply_imputer, sanitize_cols,
    get_train_val_test_indices, optimize_cutoff, hard_vote
)
from .models import LGBBoosterWrapper, XGBCompat


def train_and_predict_fixed_adaptive(
    Xtr, y_tr, X_val, y_val, X_test, cat_cols_idx,
    strategy: str, w_pos: float, seed: int, fixed_n_ests: Dict[str, int]
):
    if strategy == 'reweight':
        scale_pos_weight = w_pos
        class_weights = [1.0, w_pos]
        lgbm_weight_dict = {0: 1.0, 1: w_pos}
    else:
        scale_pos_weight = 1.0
        class_weights = None
        lgbm_weight_dict = None

    # LGBM
    lgbm_params = {**STAGE1_FIXED["lgbm"], 'n_estimators': fixed_n_ests['lgbm'], 'random_state': seed}
    model_lgbm = LGBBoosterWrapper(**lgbm_params).fit(
        Xtr, y_tr, None, None, fixed_n_ests['lgbm'], class_weight=lgbm_weight_dict
    )

    # XGB
    xgb_params = {**STAGE1_FIXED["xgb"], 'n_estimators': fixed_n_ests['xgb'], 'random_state': seed,
                  'scale_pos_weight': scale_pos_weight}
    model_xgb = XGBCompat(**xgb_params).fit(Xtr, y_tr, X_val, y_val, fixed_n_ests['xgb'])

    # CAT
    cat_params = {**STAGE1_FIXED["cat"], 'iterations': fixed_n_ests['cat'], 'random_seed': seed}
    if class_weights:
        cat_params['class_weights'] = class_weights
    pool_tr = Pool(Xtr, y_tr, cat_features=cat_cols_idx or None)
    model_cat = CatBoostClassifier(**cat_params, **CAT_TASK_PARAMS).fit(pool_tr, verbose=False)

    val_proba = {}
    test_proba = {}

    if X_val is not None:
        val_proba["lgbm"] = model_lgbm.predict_proba(X_val)[:, 1]
        val_proba["xgb"] = model_xgb.predict_proba(X_val)[:, 1]
        val_proba["cat"] = model_cat.predict_proba(Pool(X_val, cat_features=cat_cols_idx or None))[:, 1]

    if X_test is not None:
        test_proba["lgbm"] = model_lgbm.predict_proba(X_test)[:, 1]
        test_proba["xgb"] = model_xgb.predict_proba(X_test)[:, 1]
        test_proba["cat"] = model_cat.predict_proba(Pool(X_test, cat_features=cat_cols_idx or None))[:, 1]

    return val_proba, test_proba


def run_evaluation_mode(seed_list, df_full, config):
    """
    [Part 1] Thesis Evaluation: Adaptive Cutoff per Seed (Prior Strategy)
    + NEW: model-wise metrics + model-wise SE
    """
    logging.info("\n>>> START Part 1: Evaluation (Adaptive Cutoff - Prior Strategy)")

    fixed_n_ests = config["stage1_final_n_estimators"]
    fixed_strategy = config.get("stage1_final_strategy", "prior")

    y_all = (df_full[TARGET_COL] > 0).astype(int).values
    X_all_raw_base = df_full.drop(columns=[ID_COL, TARGET_COL])

    seedwise_rows = []
    ensemble_rows = []

    for sd in seed_list:
        logging.info(f"[Eval] Seed {sd}...")

        train_idx, val_idx, test_idx = get_train_val_test_indices(X_all_raw_base, y_all, seed=sd)
        df_train = df_full.iloc[train_idx].copy(); y_train = y_all[train_idx]
        df_val   = df_full.iloc[val_idx].copy();   y_val   = y_all[val_idx]
        df_test  = df_full.iloc[test_idx].copy();  y_test  = y_all[test_idx]

        # Preprocess
        Xtr_feat, feat_cols, cat_cols = build_features(df_train, TARGET_COL, [ID_COL])
        num_cols, med = fit_imputer(Xtr_feat)
        enc = OrdinalCategoryEncoder().fit(Xtr_feat, cat_cols)

        Xtr = apply_imputer(enc.transform(Xtr_feat), num_cols, med)
        Xva = apply_imputer(enc.transform(df_val.drop(columns=[ID_COL, TARGET_COL])), num_cols, med)
        Xte = apply_imputer(enc.transform(df_test.drop(columns=[ID_COL, TARGET_COL])), num_cols, med)

        Xtr = sanitize_cols(Xtr)
        Xva = sanitize_cols(Xva).reindex(columns=Xtr.columns, fill_value=0)
        Xte = sanitize_cols(Xte).reindex(columns=Xtr.columns, fill_value=0)
        cat_cols_idx = [Xtr.columns.get_loc(c) for c in cat_cols if c in Xtr.columns]

        train_pos_prior = float(y_train.mean())
        w_pos = (len(y_train) - y_train.sum()) / max(y_train.sum(), 1)

        val_p, test_p = train_and_predict_fixed_adaptive(
            Xtr, y_train, Xva, y_val, Xte, cat_cols_idx,
            fixed_strategy, w_pos, sd, fixed_n_ests
        )

        seed_cutoffs = {}
        for m in ["lgbm", "xgb", "cat"]:
            seed_cutoffs[m] = optimize_cutoff(
                y_val, val_p[m], center=train_pos_prior
            )

        # model-wise metrics on TEST
        for m in ["lgbm", "xgb", "cat"]:
            p = test_p[m]
            y_pred_m = (p >= seed_cutoffs[m]).astype(int)
            seedwise_rows.append({
                "seed": sd,
                "model": m,
                "auc": roc_auc_score(y_test, p),
                "f1": f1_score(y_test, y_pred_m, zero_division=0),
                "precision": precision_score(y_test, y_pred_m, zero_division=0),
                "recall": recall_score(y_test, y_pred_m, zero_division=0),
                "prior_used": train_pos_prior,
                "cutoff_used": seed_cutoffs[m],
            })

        # ensemble metrics on TEST
        y_hat_ens = hard_vote(test_p, seed_cutoffs)
        avg_proba = (test_p["lgbm"] + test_p["xgb"] + test_p["cat"]) / 3.0

        seedwise_rows.append({
            "seed": sd,
            "model": "ensemble",
            "auc": roc_auc_score(y_test, avg_proba),
            "f1": f1_score(y_test, y_hat_ens, zero_division=0),
            "precision": precision_score(y_test, y_hat_ens, zero_division=0),
            "recall": recall_score(y_test, y_hat_ens, zero_division=0),
            "prior_used": train_pos_prior,
            "cutoff_used": np.nan,
        })

        ensemble_rows.append({
            "seed": sd,
            "test_auc": roc_auc_score(y_test, avg_proba),
            "test_f1": f1_score(y_test, y_hat_ens, zero_division=0),
            "test_precision": precision_score(y_test, y_hat_ens, zero_division=0),
            "test_recall": recall_score(y_test, y_hat_ens, zero_division=0),
            "prior_used": train_pos_prior,
            "cutoff_lgbm": seed_cutoffs["lgbm"],
            "cutoff_xgb": seed_cutoffs["xgb"],
            "cutoff_cat": seed_cutoffs["cat"],
        })

    df_seedwise = pd.DataFrame(seedwise_rows)
    seedwise_path = ANALYSIS_DIR / "stage1_thesis_seedwise_results.csv"
    df_seedwise.to_csv(seedwise_path, index=False)
    logging.info(f"Seed-wise(model-wise) results saved: {seedwise_path}")

    df_raw = pd.DataFrame(ensemble_rows)
    raw_path = ANALYSIS_DIR / "stage1_thesis_all_seeds_raw.csv"
    df_raw.to_csv(raw_path, index=False)
    logging.info(f"Raw(ensemble+cutoff) results saved: {raw_path}")

    N = len(seed_list)
    summary_rows = []
    for model_name, g in df_seedwise.groupby("model"):
        for metric in ["auc", "f1", "precision", "recall"]:
            mean_v = g[metric].mean()
            std_v = g[metric].std(ddof=1)
            se_v = std_v / math.sqrt(N)
            summary_rows.append({
                "Model": model_name,
                "Metric": metric,
                "Mean": mean_v,
                "StdDev": std_v,
                "SE": se_v
            })

    df_summary = pd.DataFrame(summary_rows)
    out_path = ANALYSIS_DIR / "stage1_thesis_SE_results.csv"
    df_summary.to_csv(out_path, index=False)
    logging.info(f"Part 1 Done. Model-wise SE results saved: {out_path}")


def run_phase3_mode(seed_list, df_full, config):
    """[Part 2] Phase 3: Full Train -> Use FIXED Config Cutoff"""
    logging.info("\n>>> START Part 2: Phase 3 Prediction (Full Data)")
    logging.info("Note: Phase 3 uses FIXED cutoffs from config (No Validation Set available)")

    fixed_n_ests = config["stage1_final_n_estimators"]
    fixed_cutoffs = config["stage1_final_cutoffs"]
    fixed_strategy = config.get("stage1_final_strategy", "prior")

    y_all = (df_full[TARGET_COL] > 0).astype(int).values
    X_full_raw, feat_cols, cat_cols = build_features(df_full, TARGET_COL, [ID_COL])

    num_cols, med = fit_imputer(X_full_raw)
    enc = OrdinalCategoryEncoder().fit(X_full_raw, cat_cols)
    X_full = apply_imputer(enc.transform(X_full_raw), num_cols, med)
    X_full = sanitize_cols(X_full)
    cat_cols_idx = [X_full.columns.get_loc(c) for c in cat_cols if c in X_full.columns]

    w_pos = (len(y_all) - y_all.sum()) / max(y_all.sum(), 1)

    out_df = df_full.copy()
    pred_cols = []

    for sd in seed_list:
        logging.info(f"[Phase3] Seed {sd}...")

        _, full_proba_dict = train_and_predict_fixed_adaptive(
            X_full, y_all, None, None, X_full, cat_cols_idx,
            fixed_strategy, w_pos, sd, fixed_n_ests
        )

        p_avg = (full_proba_dict["lgbm"] + full_proba_dict["xgb"] + full_proba_dict["cat"]) / 3.0
        out_df[f"proba_{sd}"] = p_avg

        p_hard = hard_vote(full_proba_dict, fixed_cutoffs).astype(int)
        out_df[f"pred_{sd}"] = p_hard
        pred_cols.append(f"pred_{sd}")

    vote_sum = out_df[pred_cols].sum(axis=1)
    threshold = len(seed_list) // 2
    out_df["stage1_label_seed_ens"] = (vote_sum >= threshold).astype(int)
    out_df["stage1_proba_seed_ens"] = vote_sum / len(seed_list)

    out_path_csv = ANALYSIS_DIR / "stage1_seed_ensemble.csv"
    final_out = out_df[list(df_full.columns) + ["stage1_proba_seed_ens", "stage1_label_seed_ens"]]
    final_out.to_csv(out_path_csv, index=False)

    out_path_parquet = ANALYSIS_DIR / "stage1_seed_ensemble.parquet"
    final_out.to_parquet(out_path_parquet, index=False)

    logging.info(f"Part 2 Done. Files saved:\n  - {out_path_csv}\n  - {out_path_parquet}")


def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        handlers=[FileHandler(ANALYSIS_DIR / f'unified_log_{timestamp}.log'), StreamHandler(sys.stdout)]
    )
    logging.info("[MAIN] Running Unified Pipeline (Adaptive Cutoff - Prior Strategy)")

    cfg_path = ANALYSIS_DIR / "stage1_final_config.json"
    if not cfg_path.exists():
        logging.error(f"Config file missing at {cfg_path}")
        return

    with open(cfg_path) as f:
        config = json.load(f)

    df_full = pd.read_parquet(UNIFIED_DATA_PATH)

    try:
        run_evaluation_mode(SEEDS_TO_RUN, df_full, config)
    except Exception as e:
        logging.error(f"Error in Evaluation: {e}")
        traceback.print_exc()

    try:
        run_phase3_mode(SEEDS_TO_RUN, df_full, config)
    except Exception as e:
        logging.error(f"Error in Phase 3: {e}")
        traceback.print_exc()

    logging.info("\n All Tasks Completed!")


if __name__ == "__main__":
    main()