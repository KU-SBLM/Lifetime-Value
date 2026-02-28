import inspect
import os
import numpy as np
import pandas as pd
from sklearn.metrics import fbeta_score, precision_score
import config

try:
    from tabpfn import TabPFNClassifier
    from tabpfn.constants import ModelVersion
    TABPFN_AVAILABLE = True
except ImportError:
    TABPFN_AVAILABLE = False

os.environ["TABPFN_ALLOW_CPU_LARGE_DATASET"] = "1"

def _sanitize_cols(df):
    out = df.copy()
    out.columns = out.columns.astype(str).str.replace(r"\s+", "_", regex=True)
    return out

def tune_cutoff(y_true, proba, cutoff_strategy, train_pos_prior, beta=config.F_BETA):
    center = float(np.clip(train_pos_prior, 0.01, 0.5)) if cutoff_strategy == "prior" else 0.5
    lo, hi = max(0.0, center - 0.05), min(1.0, center + 0.05)
    grid = np.arange(lo, hi + 1e-9, 0.005)
    
    best_t = center 
    best_s = -1.0
    fallback_t = center
    max_prec = -1.0

    for t in grid:
        y_hat = (proba >= t).astype(int)
        if y_hat.sum() == 0: continue
        
        current_prec = precision_score(y_true, y_hat, zero_division=0)
        if current_prec > max_prec:
            max_prec = current_prec
            fallback_t = float(t)
            
        if current_prec < 0.3: continue
        
        s = fbeta_score(y_true, y_hat, beta=beta, zero_division=0)
        if s > best_s:
            best_t, best_s = float(t), float(s)
    
    if best_s == -1.0:
        return fallback_t, 0.0
        
    return best_t, best_s

def make_tabpfn_classifier(device, seed, n_ens):
    if not TABPFN_AVAILABLE:
        return None
        
    sig = inspect.signature(TabPFNClassifier.__init__)
    kw = {}
    if "device" in sig.parameters: kw["device"] = config.DEVICE if device == "auto" else device
    if "N_ensemble_configurations" in sig.parameters: kw["N_ensemble_configurations"] = n_ens
    if "seed" in sig.parameters: kw["seed"] = seed
    
    return TabPFNClassifier(**kw)