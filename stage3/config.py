import torch
from pathlib import Path
from typing import List


TARGET_COL = "PAY_AMT_SUM" 
ID_COL = "PLAYERID"
PRED_PAYER_COL = "stage1_label_seed_ens" 
PRED_WHALE_COL = "final_stage2_whale_pred" 

DEVICE = "cuda" if torch.cuda.is_available() else "cpu" 
CAT_TASK_PARAMS = {"task_type": "GPU"} if DEVICE == "cuda" else {}

N_JOBS = 12 

PROJECT_ROOT = Path(__file__).resolve().parent 
DATA_DIR = PROJECT_ROOT.parent / "data"
DEFAULT_INPUT_DATA_PATH = PROJECT_ROOT / "data" / "stage2_seed_ensemble.parquet"
DEF_OUTPUT_DIR = PROJECT_ROOT / "stage3_results"

SEED_RANGE = "2021..2070"

DEFAULT_TRIALS = 20 

USE_LOG_TRANSFORM = True 
LOG_EPSILON = 1e-6 
SMAPE_EPSILON = 0.1 

USE_STAGE1_FEATURES = False 
USE_STAGE2_FEATURES = False
NO_CATBOOST = False 
NO_LGBM = False 
NO_TABPFN = False 

TABPFN_DEVICE = "auto"
TABPFN_CONFIGS = 32 

LOSS_CONFIG = {
    "non_whale": {
        "lgbm_loss": "mae",     
        "lgbm_metric": "l1",      
        "cat_loss": "MAE",       
        "cat_metric": "MAE",      
        "obj_metric_func": 'mae' 
    },
    "whale": {
        "lgbm_loss": "mse", 
        "lgbm_metric": "l2",  
        "cat_loss": "RMSE",   
        "cat_metric": "RMSE",  
        "obj_metric_func": 'mse' 
    }
}


NON_WHALE_FIXED = {
    "lgbm": {
        "n_estimators": 350, 
        "max_depth": 6,
        "min_child_samples": 25, 
        "learning_rate": 0.03, 
        "reg_alpha": 0.1,   
        "reg_lambda": 0.1,  
    },
    "cat": {
        "iterations": 450,
        "depth": 6,
        "learning_rate": 0.03, 
        "l2_leaf_reg": 0.1,
        "bagging_temperature": 1
    }
}

WHALE_FIXED = {
    "lgbm": {
        "n_estimators": 300, 
        "max_depth": 5,
        "min_child_samples": 15, 
        "learning_rate": 0.03, 
        "reg_alpha": 0.1,   
        "reg_lambda": 0.1,  
    },
    "cat": {
        "iterations": 400,
        "depth": 3, 
        "learning_rate": 0.03, 
        "l2_leaf_reg": 0.1,
        "bagging_temperature": 1
    }
}

ENSEMBLE_MODE = "mean" 
SKIP_IF_EXISTS = False 

STAGE1_PROBA_COLS: List[str] = [
    "pred_2021", "pred_2022", "pred_2023", "pred_2024", "pred_2025", 
    "pred_2026", "pred_2027", "pred_2028", "pred_2029", "pred_2030",
]
STAGE2_PROBA_COLS: List[str] = [
    "pred2_2021", "pred2_2022", "pred2_2023", "pred2_2024", "pred2_2025", 
    "pred2_2026", "pred2_2027", "pred2_2028", "pred2_2029", "pred2_2030",
]
