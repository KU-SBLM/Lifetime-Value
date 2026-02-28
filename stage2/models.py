import numpy as np
import pandas as pd
import lightgbm as lgb
from catboost import CatBoostClassifier, Pool
import config
from utils import make_tabpfn_classifier

def train_lgbm_fixed(X_tr, y_tr, X_va, y_va, train_strategy, current_seed):
    params = config.LGBM_FIXED_PARAMS.copy()
    params["random_state"] = current_seed
    
    if train_strategy == "reweight":
        w_pos = (len(y_tr) - y_tr.sum()) / max(y_tr.sum(), 1)
        if config.REWEIGHT_BY_FBETA: w_pos *= (config.F_BETA ** 2)
        params["class_weight"] = {0: 1.0, 1: float(min(w_pos, 20.0))}
    
    model = lgb.LGBMClassifier(**params, n_jobs=-1) 
    model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], eval_metric="average_precision",
              callbacks=[lgb.early_stopping(50, verbose=False)])
    return model

def train_cat_fixed(X_tr, y_tr, X_va, y_va, cat_cols_idx, train_strategy, current_seed):
    params = config.CAT_FIXED_PARAMS.copy()
    params["random_seed"] = current_seed
    
    if train_strategy == "reweight":
        w_pos = (len(y_tr) - y_tr.sum()) / max(y_tr.sum(), 1)
        if config.REWEIGHT_BY_FBETA: w_pos *= (config.F_BETA ** 2)
        params["class_weights"] = [1.0, w_pos]
    
    model = CatBoostClassifier(**params, **config.CAT_TASK_PARAMS)
    pool_tr = Pool(X_tr, y_tr, cat_features=cat_cols_idx or None)
    pool_va = Pool(X_va, y_va, cat_features=cat_cols_idx or None)
    model.fit(pool_tr, eval_set=pool_va, use_best_model=True, verbose=False)
    return model

def train_tab_fixed(X_tr, y_tr, current_seed):
    model = make_tabpfn_classifier(device=config.TABPFN_DEVICE, seed=current_seed, n_ens=config.TABPFN_CONFIGS)
    if model is not None:
        X_input = X_tr.values.astype(np.float32)
        y_input = y_tr.values.astype(np.int32)        
        model.fit(X_input, y_input)
        
    return model