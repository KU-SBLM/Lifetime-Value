#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path
import torch

# =====================================================================================
# ---- 1. CONFIGURATION & PATHS
# =====================================================================================

SCRIPT_DIR = Path.cwd()
UNIFIED_DATA_PATH = SCRIPT_DIR.parent / 'Data' / "fin_df_5days_3stage.parquet"

ANALYSIS_DIR = SCRIPT_DIR / "stage1_for_thesis_prior"
ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

TARGET_COL = "PAY_AMT_SUM"
ID_COL = "PLAYERID"

FINAL_SEED = 2021
SEEDS_TO_RUN = list(range(FINAL_SEED, FINAL_SEED + 50))

# --- Cutoff Optimization Param (Prior Strategy) ---
F_BETA_SCORE = 1.5
DELTA_AROUND = 0.15
CUT_STEP = 0.01

# --- Fixed Hyperparameters ---
RUN_LR = 0.05
STAGE1_FIXED = {
    "lgbm": dict(
        objective="binary", learning_rate=RUN_LR, max_depth=8,
        min_child_samples=20, subsample=0.1, reg_alpha=0.1, reg_lambda=0.1, verbosity=-1,
    ),
    "xgb": dict(
        objective="binary:logistic", eval_metric="auc", learning_rate=RUN_LR,
        max_depth=9, subsample=0.1, reg_alpha=0.1, reg_lambda=0.1, max_bin=256,
        tree_method="hist", predictor="auto",
    ),
    "cat": dict(
        loss_function="Logloss", eval_metric="AUC", depth=9, learning_rate=RUN_LR, verbose=0,
    )
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CAT_TASK_PARAMS = {"task_type": "GPU"} if DEVICE == "cuda" else {}