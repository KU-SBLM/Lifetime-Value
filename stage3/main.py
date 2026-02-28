import json
import sys
import logging
import joblib 
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
from typing import List
from sklearn.model_selection import train_test_split 
from sklearn.metrics import mean_absolute_error, mean_squared_error
import re 

import config as config
from config import (
    TARGET_COL, ID_COL, PRED_PAYER_COL, PRED_WHALE_COL, 
    DEF_OUTPUT_DIR, DEFAULT_TRIALS, 
    USE_LOG_TRANSFORM, DEFAULT_INPUT_DATA_PATH, LOSS_CONFIG, SEED_RANGE
)

from utils import SectionTimer, _sanitize_cols, parse_seeds
from data_preprocessing import (
    OrdinalCategoryEncoder, build_features, fit_imputer, apply_imputer, _HAS_TABPFN 
)
from models import (
    train_and_ensemble_reg, predict_reg_model, inverse_transform, smape 
)

OUTPUT_DIR = Path(DEF_OUTPUT_DIR)
ARTIFACTS_PATH = OUTPUT_DIR / "global_artifacts"
IMPUTER_PATH = lambda s: ARTIFACTS_PATH / f"stage3_imputer_{s}.joblib"
ENCODER_PATH = lambda s: ARTIFACTS_PATH / f"stage3_encoder_{s}.joblib"
MODELS_PATH = OUTPUT_DIR / "models"
LOGS_PATH = OUTPUT_DIR / "logs"


def load_data_and_prepare_for_regression(df_full: pd.DataFrame, current_seed: int):

    if ID_COL in df_full.columns:
        df_full = df_full.set_index(ID_COL, drop=False)
    
    if PRED_PAYER_COL not in df_full.columns or PRED_WHALE_COL not in df_full.columns:
         raise KeyError(f"Missing required prediction columns ({PRED_PAYER_COL} or {PRED_WHALE_COL}).")
    
    df_payer = df_full[df_full[PRED_PAYER_COL] == 1].copy()
    df_non_payer = df_full[df_full[PRED_PAYER_COL] == 0].copy() 

    if len(df_payer) == 0:
        raise ValueError("Filtered Payer dataset is empty.")

    y_whale_payer = df_payer[PRED_WHALE_COL].astype(int)
    
    tr_ratio, va_ratio, te_ratio = 0.60, 0.20, 0.20
    hold_ratio = va_ratio + te_ratio
    
    df_tr_scope, df_hold = train_test_split(
        df_payer,
        test_size=hold_ratio,
        random_state=current_seed, 
        stratify=y_whale_payer
    )

    y_hold = df_hold[PRED_WHALE_COL].astype(int)
    val_ratio_in_hold = va_ratio / hold_ratio
    
    df_va_scope, df_test_scope = train_test_split(
        df_hold,
        test_size=(1.0 - val_ratio_in_hold),
        random_state=current_seed,
        stratify=y_hold
    )
    
    tr_nw_full = df_tr_scope[df_tr_scope[PRED_WHALE_COL] == 0].copy()
    va_nw_full = df_va_scope[df_va_scope[PRED_WHALE_COL] == 0].copy()
    test_nw = df_test_scope[df_test_scope[PRED_WHALE_COL] == 0].copy()
    
    tr_whale_full = df_tr_scope[df_tr_scope[PRED_WHALE_COL] == 1].copy()
    va_whale_full = df_va_scope[df_va_scope[PRED_WHALE_COL] == 1].copy()
    test_whale = df_test_scope[df_test_scope[PRED_WHALE_COL] == 1].copy()
    
    tr_nw_full["stage3_split"] = "train"
    va_nw_full["stage3_split"] = "val"
    test_nw["stage3_split"] = "test"
    
    tr_whale_full["stage3_split"] = "train"
    va_whale_full["stage3_split"] = "val"
    test_whale["stage3_split"] = "test"

    logging.info(f"Total data: {df_full.shape[0]} | Predicted Payer: {df_payer.shape[0]} | Non-Payer: {df_non_payer.shape[0]}")
    logging.info(f"    - Non-Whale Payer (Train/Val/Test): {len(tr_nw_full)}/{len(va_nw_full)}/{len(test_nw)}")
    logging.info(f"    - Whale Payer (Train/Val/Test): {len(tr_whale_full)}/{len(va_whale_full)}/{len(test_whale)}")
    
    return {
        "non_whale": {"tr": tr_nw_full, "va": va_nw_full, "test": test_nw},
        "whale": {"tr": tr_whale_full, "va": va_whale_full, "test": test_whale},
        "non_payer": df_non_payer, 
        "all_base": df_full 
    }


def run_single_seed(data_sets: dict, current_seed: int):
    group_keys = ["non_whale", "whale"]
    reports_per_seed = {}
    preds_per_seed = [] 

    imputer_path = IMPUTER_PATH(current_seed) 
    encoder_path = ENCODER_PATH(current_seed)

    if not imputer_path.exists():
        with SectionTimer(f"({current_seed}) Global Imputer/Encoder Fit"):
            df_for_fit = pd.concat([data_sets["non_whale"]["tr"], data_sets["whale"]["tr"]], axis=0)
            drop_cols_for_feature = [ID_COL, TARGET_COL, "stage2_tvt", "stage3_split"]
            X_tr_raw, _, cat_cols = build_features(df_for_fit, TARGET_COL, drop_cols_for_feature)
            encoder = OrdinalCategoryEncoder().fit(df_for_fit, cat_cols)
            joblib.dump(encoder, encoder_path)
            imputer = fit_imputer(encoder.transform(X_tr_raw))
            joblib.dump(imputer, imputer_path)
    else:
        imputer = joblib.load(imputer_path)
        encoder = joblib.load(encoder_path)

    num_cols_imp, med_imp = imputer

    for group_key in group_keys:
        group_data = data_sets[group_key]
        drop_cols_for_feature = [ID_COL, TARGET_COL, "stage2_tvt", "stage3_split"]
        
        X_tr_raw, _, cat_cols = build_features(group_data["tr"], TARGET_COL, drop_cols_for_feature)
        X_va_raw, _, _ = build_features(group_data["va"], TARGET_COL, drop_cols_for_feature)
        X_test_raw, _, _ = build_features(group_data["test"], TARGET_COL, drop_cols_for_feature)

        y_tr_log = np.log1p(group_data["tr"][TARGET_COL]) if USE_LOG_TRANSFORM else group_data["tr"][TARGET_COL]
        y_va_log = np.log1p(group_data["va"][TARGET_COL]) if USE_LOG_TRANSFORM else group_data["va"][TARGET_COL]
        y_va_raw = group_data["va"][TARGET_COL]
        y_test_raw = group_data["test"][TARGET_COL]

        X_tr = _sanitize_cols(apply_imputer(encoder.transform(X_tr_raw), num_cols_imp, med_imp))
        X_va = _sanitize_cols(apply_imputer(encoder.transform(X_va_raw), num_cols_imp, med_imp))
        X_test = _sanitize_cols(apply_imputer(encoder.transform(X_test_raw), num_cols_imp, med_imp))
        
        cat_indices = [X_tr.columns.get_loc(c) for c in cat_cols if c in X_tr.columns]

        with SectionTimer(f"({current_seed}|{group_key}) Evaluation Train (Split)"):
            eval_models, best_params, val_m, val_n = train_and_ensemble_reg(
                X_tr, y_tr_log, X_va, y_va_log, y_va_raw,
                cat_cols_idx=cat_indices, group_key=group_key, 
                trials=DEFAULT_TRIALS, current_seed=current_seed, has_tabpfn=_HAS_TABPFN
            )
            
            model_group_path = MODELS_PATH / f"{group_key}_seed_{current_seed}"
            model_group_path.mkdir(parents=True, exist_ok=True)
            joblib.dump(eval_models, model_group_path / "reg_models_eval.joblib")

            test_preds_log = []
            logging.info(f"--- [Individual Model Test Evaluation | {group_key.upper()} | Seed {current_seed}] ---")
            
            for m_key, m_obj in eval_models.items():
                p_test_log = predict_reg_model(m_key, m_obj, X_test, cat_indices)
                test_preds_log.append(p_test_log)
                
                p_test_raw = inverse_transform(p_test_log)
                m_test_mae = mean_absolute_error(y_test_raw.values, p_test_raw)
                logging.info(f" > Model: {m_key:7s} | Test MAE: {m_test_mae:10.2f}")
            
            ens_test_raw = inverse_transform(np.median(np.column_stack(test_preds_log), axis=1))
            test_mae = mean_absolute_error(y_test_raw.values, ens_test_raw)
            logging.info(f" > [ENSEMBLE]   | Test MAE: {test_mae:10.2f}")
            logging.info(f"[{group_key.upper()} TEST LOG] Seed {current_seed} Combined Test MAE: {test_mae:.2f}")

        with SectionTimer(f"({current_seed}|{group_key}) Full Dataset Fit & Predict"):
            df_full_group = pd.concat([group_data["tr"], group_data["va"], group_data["test"]], axis=0)
            X_full_raw, _, _ = build_features(df_full_group, TARGET_COL, drop_cols_for_feature)
            y_full_log = np.log1p(df_full_group[TARGET_COL]) if USE_LOG_TRANSFORM else df_full_group[TARGET_COL]
            
            X_full = _sanitize_cols(apply_imputer(encoder.transform(X_full_raw), num_cols_imp, med_imp))
            
            from models import train_full_dataset_reg 
            full_models = train_full_dataset_reg(X_full, y_full_log, cat_indices, group_key, current_seed)
            
            full_preds_log = []
            for m_key, m_obj in full_models.items():
                p_full_log = predict_reg_model(m_key, m_obj, X_full, cat_indices)
                full_preds_log.append(p_full_log)
            
            final_pred_raw = inverse_transform(np.median(np.column_stack(full_preds_log), axis=1))
            
            df_res = df_full_group[[ID_COL, TARGET_COL, PRED_WHALE_COL]].copy()
            df_res["ltv_pred_seed"] = final_pred_raw
            preds_per_seed.append(df_res)
            
            joblib.dump(full_models, model_group_path / "reg_models_full.joblib")

    df_non_payer = data_sets["non_payer"][[ID_COL]].copy()
    df_non_payer["ltv_pred_seed"] = 0.0
    preds_per_seed.append(df_non_payer)
    
    df_seed_pred = pd.concat(preds_per_seed, axis=0).set_index(ID_COL).sort_index()
    csv_path = OUTPUT_DIR / f"stage3_seed_{current_seed}_predictions.csv"
    df_seed_pred.to_csv(csv_path)
    logging.info(f"Saved Stage 3 Full Dataset Prediction: {csv_path.name}")
            
    return reports_per_seed 


def run_multi_seeds():
    
    seeds = parse_seeds(config.SEED_RANGE)
    if not seeds:
        logging.error("No seeds identified from Stage 2 data. Exiting.")
        sys.exit(1)
    
    logging.info(f"▶ START: Loading Stage 2 final predictions from {DEFAULT_INPUT_DATA_PATH.name}")
    try:
        df_full = pd.read_parquet(DEFAULT_INPUT_DATA_PATH)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Stage 2 final predictions file not found: {DEFAULT_INPUT_DATA_PATH}. "
            f"Ensure Stage 2 was run successfully."
        )

    if ID_COL in df_full.columns:
        df_full = df_full.set_index(ID_COL, drop=False)
        
        
    output_dir = OUTPUT_DIR
    output_dir.mkdir(exist_ok=True)
    MODELS_PATH.mkdir(exist_ok=True)
    ARTIFACTS_PATH.mkdir(exist_ok=True)
    
    ensemble_mode = config.ENSEMBLE_MODE
    skip_if_exists = config.SKIP_IF_EXISTS

    FINAL_PARQUET_PATH = output_dir / "stage3_final_predictions_all_data.parquet"
    
    if skip_if_exists and FINAL_PARQUET_PATH.exists():
        logging.info(f"Final ensemble prediction file exists at {FINAL_PARQUET_PATH}. Skipping entire stage.")
        sys.exit(0)
    
    report_all = {}
    for current_seed in seeds:
        seed_csv_path = output_dir / f"stage3_seed_{current_seed}_predictions.csv"
        
        if skip_if_exists and seed_csv_path.exists():
            logging.info(f"Skipping training for seed {current_seed}. Prediction file already exists.")
            report_all[str(current_seed)] = {"status": "skipped"}
            continue
            
        logging.info(f"--- Starting Training for Seed: {current_seed} ---")
        try:
            with SectionTimer(f"Run Stage 3 (Seed {current_seed})"):
                data_sets_s = load_data_and_prepare_for_regression(df_full, current_seed)
                reports_s = run_single_seed(data_sets_s, current_seed=current_seed)
                report_all[str(current_seed)] = reports_s
        except Exception as e:
            logging.error(f"Critical error during training for seed {current_seed}: {e}", exc_info=True)
            report_all[str(current_seed)] = {"status": "failed", "error": str(e)}
            continue 
    
    logging.info(f"Compelete Stage3")
            

def run_experiment():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    LOGS_PATH.mkdir(parents=True, exist_ok=True)
    log_file = LOGS_PATH / f"ltv_pipeline_stage3_{timestamp}_overall.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(log_file)]
    )
    try:
        run_multi_seeds()
    except SystemExit: 
        pass
    except Exception as e:
        logging.error(f"A critical error occurred during pipeline execution: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    run_experiment()