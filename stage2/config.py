import torch
from pathlib import Path

# Experiment Control
IS_TRIAL_RUN = False 
EXP_SEEDS = list(range(2021, 2072))
EXP_MODE = 1  # 1: Seed-specific optimal cutoff | 2: Fixed 

# Core Strategy
TRAIN_STRATEGY = "normal"  # "normal": base training | "reweight": class reweighting
CUTOFF_STRATEGY = "prior"  # "fixed": 0.5 center | "prior": positive ratio center
ENSEMBLE_MODE = "soft"     # "hard": hard voting | "soft": probability average then cutoff

# Data and Column Definitions
TARGET_COL = "PAY_AMT_SUM"
ID_COL = "PLAYERID"
STAGE1_LABEL_COL = "stage1_label_seed_ens"
PRED_PAYER_COL = "pred_is_payer"
STAGE1_PROBA_COL = "stage1_proba"

# Hyperparameters
LGBM_FIXED_PARAMS = {
    'n_estimators': 174, 'num_leaves': 94, 'min_child_samples': 39, 'max_depth': 6,
    'learning_rate': 0.05, 'subsample': 1.0, 'colsample_bytree': 0.6,
    'reg_alpha': 6.0, 'reg_lambda': 12.0, 'objective': 'binary',
    'verbosity': -1, 'device': 'gpu',
    'gpu_platform_id': 0, 'gpu_device_id': 0
}

CAT_FIXED_PARAMS = {
    'depth': 6, 
    'learning_rate': 0.03, 
    'l2_leaf_reg': 0.9837, 
    'iterations': 215,
    'bagging_temperature': 1.0, 
    'random_strength': 1.0, 
    'loss_function': 'Logloss',
    'eval_metric': 'PRAUC',
    'metric_period': 100,
}

CAT_TASK_PARAMS = {
    'task_type': 'GPU',
    'devices': '0'
}

# Preprocessing & Paths
EXCLUDE_PROBA_FEATURES = True
USE_STAGE1_FEATURES = False
FIXED_VAL_EXP2 = 0.1267

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "Data"
DEFAULT_INPUT_DATA_PATH = DATA_DIR / "stage1_seed_ensemble_cutoff.parquet"
DEF_OUTPUT_DIR = PROJECT_ROOT / f"stage2_results_mode_{EXP_MODE}_{ENSEMBLE_MODE}"

METRIC_DIR = DEF_OUTPUT_DIR / "metrics"
PRED_DIR = DEF_OUTPUT_DIR / "predictions"
ART_DIR = DEF_OUTPUT_DIR / "artifacts"

F_BETA = 2.0
WHALE_Q = 0.95
BASE_SPLIT_SEED = 2025
REWEIGHT_BY_FBETA = False 

# Hardware Configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TABPFN_DEVICE = "cuda"
TABPFN_CONFIGS = 32