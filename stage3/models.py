import logging
import os
import numpy as np
import pandas as pd
from pathlib import Path

from catboost import CatBoostRegressor, Pool
import lightgbm as lgb
import optuna
from sklearn.metrics import mean_squared_error, mean_absolute_error

from config import (
    CAT_TASK_PARAMS,
    USE_LOG_TRANSFORM, LOG_EPSILON, SMAPE_EPSILON, 
    NO_LGBM, NO_CATBOOST, NO_TABPFN, 
    NON_WHALE_FIXED, WHALE_FIXED, LOSS_CONFIG
)

def inverse_transform(p):
    if USE_LOG_TRANSFORM:
        return np.maximum(0.0, np.expm1(p) + LOG_EPSILON) 
    return np.maximum(0.0, p)

def score_stage3_objective(y_true_raw, proba_log, group_key):
    try:
        y_pred_raw = inverse_transform(proba_log)
        obj_func = LOSS_CONFIG[group_key]["obj_metric_func"]
        if obj_func == 'mae':
            return float(mean_absolute_error(y_true_raw, y_pred_raw)) 
        elif obj_func == 'mse':
            return float(mean_squared_error(y_true_raw, y_pred_raw))
    except ValueError:
        return 0.0 

def smape(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    numerator = np.abs(y_pred - y_true)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0 + SMAPE_EPSILON
    return np.mean(numerator / denominator) * 100.0


def get_fixed_params(group_key):
    if group_key == "whale":
        return WHALE_FIXED
    else:
        return NON_WHALE_FIXED

def train_and_ensemble_reg(X_tr, y_tr_log, X_va, y_va_log, y_va_raw, 
                           cat_cols_idx, group_key, trials, has_tabpfn, current_seed):

    models = {}; best_params = {}; preds_log = {}
    config_loss = LOSS_CONFIG[group_key]
    fixed_params = get_fixed_params(group_key)

    if not NO_LGBM:
        lgb_params = fixed_params["lgbm"].copy()
        lgb_params.update({
            "objective": config_loss['lgbm_loss'],
            "metric": config_loss["lgbm_metric"],
            "random_state": current_seed,
            "n_jobs": max(1, (os.cpu_count() or 8)//4),
            "verbosity": -1,
            "force_row_wise": True
        })
        
        best_params["lgbm"] = lgb_params 
        lgbm_reg = lgb.LGBMRegressor(**lgb_params)
        
        lgbm_reg.fit(
            X_tr, y_tr_log, 
            eval_set=[(X_va, y_va_log)], 
            eval_metric=config_loss["lgbm_metric"],
            callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)]
        )
        models["lgbm"] = lgbm_reg
        preds_log["lgbm"] = lgbm_reg.predict(X_va)

    if not NO_CATBOOST:
        cat_params = fixed_params["cat"].copy()
        cat_params.update({
            "loss_function": config_loss["cat_loss"],
            "eval_metric": config_loss["cat_metric"],
            "random_seed": current_seed,
            "verbose": 0,
            "od_type": "Iter",
            "od_wait": 100
        })
        
        best_params["cat"] = cat_params
        cat_reg = CatBoostRegressor(**cat_params, **CAT_TASK_PARAMS)
        
        pool_tr = Pool(X_tr, y_tr_log, cat_features=cat_cols_idx or None)
        pool_va = Pool(X_va, y_va_log, cat_features=cat_cols_idx or None)
        
        cat_reg.fit(pool_tr, eval_set=pool_va, use_best_model=True, verbose=False)
        models["cat"] = cat_reg
        preds_log["cat"] = cat_reg.predict(pool_va)

    if has_tabpfn and not NO_TABPFN:
        from config import TABPFN_DEVICE, TABPFN_CONFIGS 
        tab_reg = make_tabpfn_regressor(device=TABPFN_DEVICE, seed=current_seed, n_ens=TABPFN_CONFIGS, input_size=X_tr.shape[1])
        if tab_reg is not None:
            tab_reg.fit(X_tr.values, y_tr_log.values) 
            models["tab"] = tab_reg
            preds_log["tab"] = tab_reg.predict(X_va.values)
            
    if not models:
        raise ValueError(f"No models were trained for group {group_key}.")
    
    logging.info(f"--- [Fixed Params Model Evaluation | {group_key.upper()} | Seed {current_seed}] ---")
    for m_name, m_preds_log in preds_log.items():
        m_preds_raw = inverse_transform(m_preds_log)
        m_mae = mean_absolute_error(y_va_raw.values, m_preds_raw)
        m_mse = mean_squared_error(y_va_raw.values, m_preds_raw)
        logging.info(f" > Model: {m_name:7s} | MAE: {m_mae:10.2f} | MSE: {m_mse:12.2f}")

    ensemble_log_proba = np.median(np.column_stack(list(preds_log.values())), axis=1)
    ensemble_raw_proba = inverse_transform(ensemble_log_proba)
    
    metric_key = config_loss['obj_metric_func']
    if metric_key == 'mae':
        val_metric_val = mean_absolute_error(y_va_raw.values, ensemble_raw_proba)
        metric_name = "MAE"
    elif metric_key == 'mse':
        val_metric_val = mean_squared_error(y_va_raw.values, ensemble_raw_proba)
        metric_name = "MSE"
    else:
        val_metric_val = mean_absolute_error(y_va_raw.values, ensemble_raw_proba)
        metric_name = "MAE"
    
    logging.info(f"[VAL|{group_key.upper()}] Ensemble {metric_name}={val_metric_val:.2f} | Models: {list(models.keys())}")
    
    return models, best_params, val_metric_val, metric_name

def make_tabpfn_regressor(device, seed, n_ens, input_size = 100):
    try:
        from tabpfn import TabPFNRegressor
    except ImportError:
        logging.warning("TabPFNRegressor import failed. TabPFN model will be skipped.")
        return None
    import utils
    return utils.construct_tabpfn(TabPFNRegressor, device=device, seed=seed, n_ens=n_ens)

def predict_reg_model(model_key, model, X, cat_cols_idx):
    if model_key == "cat":
        pool = Pool(X, cat_features=cat_cols_idx or None)
        return model.predict(pool)
    elif model_key == "lgbm":
        return model.predict(X)
    elif model_key == "tab":
        return model.predict(X.values)
    return None

def train_full_dataset_reg(X_full, y_full_log, cat_cols_idx, group_key, current_seed):
    models = {}
    fixed_params = get_fixed_params(group_key)
    config_loss = LOSS_CONFIG[group_key]

    # 1. LightGBM Full Fit
    if not NO_LGBM:
        lgb_params = fixed_params["lgbm"].copy()
        lgb_params.update({
            "objective": config_loss['lgbm_loss'],
            "metric": config_loss["lgbm_metric"],
            "random_state": current_seed,
            "n_jobs": -1,
            "verbosity": -1,
            "force_row_wise": True
        })
        lgbm_reg = lgb.LGBMRegressor(**lgb_params)
        lgbm_reg.fit(X_full, y_full_log) 
        models["lgbm"] = lgbm_reg

    # 2. CatBoost Full Fit
    if not NO_CATBOOST:
        cat_params = fixed_params["cat"].copy()
        cat_params.update({
            "loss_function": config_loss["cat_loss"],
            "eval_metric": config_loss["cat_metric"],
            "random_seed": current_seed,
            "verbose": 0
        })
        cat_reg = CatBoostRegressor(**cat_params, **CAT_TASK_PARAMS)
        pool_full = Pool(X_full, y_full_log, cat_features=cat_cols_idx or None)
        cat_reg.fit(pool_full)
        models["cat"] = cat_reg

    return models