import sys
import joblib
import json
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import sem
from sklearn.model_selection import train_test_split
from sklearn.metrics import fbeta_score, f1_score, precision_score, recall_score, average_precision_score

import config
import utils
from data_preprocessing import OrdinalCategoryEncoder, build_features, fit_imputer, apply_imputer
from models import train_lgbm_fixed, train_cat_fixed, train_tab_fixed

def ensemble_predictions(preds, cutoffs, mode=config.ENSEMBLE_MODE):
    """
    Performs hard or soft voting ensemble based on the configured mode.
    - hard: Applies cutoff per model, then performs majority voting.
    - soft: Calculates the arithmetic mean of probabilities, then evaluates against the mean threshold.
    """
    valid_preds = [p for p in preds.values() if p is not None]
    
    if mode == "soft":
        avg_proba = np.mean(valid_preds, axis=0)
        avg_t = np.mean(list(cutoffs.values()))
        return (avg_proba >= avg_t).astype(int)
    else:
        votes = []
        for k, p in preds.items():
            t = cutoffs.get(k, 0.5)
            votes.append((p >= t).astype(int))
        votes = np.column_stack(votes)
        return (votes.mean(axis=1) >= 0.5).astype(int)

def run_experiment():
    for d in [config.METRIC_DIR, config.PRED_DIR, config.ART_DIR]:
        d.mkdir(parents=True, exist_ok=True)
    
    all_test_metrics = []
    tuned_cutoffs_history = {"lgbm": [], "cat": [], "tab": []}
    
    # Phase 1 & 2: Multi-seed model training and performance validation
    for current_seed in config.EXP_SEEDS:
        try:
            df_base = pd.read_parquet(config.DEFAULT_INPUT_DATA_PATH)
            df_filtered = df_base[df_base[config.STAGE1_LABEL_COL] == 1].copy()
            if len(df_filtered) < 20: continue

            # Calculate Whale threshold (top 5%)
            payers = df_filtered[df_filtered[config.TARGET_COL] >= 0][config.TARGET_COL]
            whale_cut_prov = float(np.quantile(payers, config.WHALE_Q)) if len(payers) > 0 else 0
            y_whale_all = (df_filtered[config.TARGET_COL] >= whale_cut_prov).astype(int)
        
            if y_whale_all.sum() < 5: continue 

            # Stratified split per seed
            df_tr, df_hold = train_test_split(df_filtered, test_size=0.4, random_state=current_seed, stratify=y_whale_all)
            y_hold = (df_hold[config.TARGET_COL] >= whale_cut_prov).astype(int)
            df_va, df_te = train_test_split(df_hold, test_size=0.5, random_state=current_seed, stratify=y_hold)

            # Preprocessing
            drop_cols = [config.ID_COL, config.TARGET_COL, config.STAGE1_LABEL_COL]
            Xtr_raw, feat_cols, cat_cols = build_features(df_tr, config.TARGET_COL, drop_cols)
            
            enc = OrdinalCategoryEncoder().fit(df_tr, cat_cols)
            Xtr, Xva, Xte = enc.transform(Xtr_raw), enc.transform(df_va[feat_cols]), enc.transform(df_te[feat_cols])
            
            num_cols, med = fit_imputer(Xtr)
            Xtr, Xva, Xte = apply_imputer(Xtr, num_cols, med), apply_imputer(Xva, num_cols, med), apply_imputer(Xte, num_cols, med)
            Xtr, Xva, Xte = utils._sanitize_cols(Xtr), utils._sanitize_cols(Xva), utils._sanitize_cols(Xte)

            y_tr = (df_tr[config.TARGET_COL] >= whale_cut_prov).astype(int)
            y_va = (df_va[config.TARGET_COL] >= whale_cut_prov).astype(int)
            y_te = (df_te[config.TARGET_COL] >= whale_cut_prov).astype(int)
            
            # Calculate positive class ratio for Prior strategy
            pos_prior = float(y_tr.mean())
            
            cur_cuts, preds_te = {}, {}
            cat_idx = [Xtr.columns.get_loc(c) for c in cat_cols]

            # 1. LightGBM
            lgbm = train_lgbm_fixed(Xtr, y_tr, Xva, y_va, config.TRAIN_STRATEGY, current_seed)
            p_va_l = lgbm.predict_proba(Xva)[:, 1]
            t_l = config.FIXED_VAL_EXP2 if config.EXP_MODE == 2 else utils.tune_cutoff(y_va, p_va_l, config.CUTOFF_STRATEGY, pos_prior)[0]
            cur_cuts["lgbm"] = t_l
            preds_te["lgbm"] = lgbm.predict_proba(Xte)[:, 1]

            # 2. CatBoost
            cat = train_cat_fixed(Xtr, y_tr, Xva, y_va, cat_idx, config.TRAIN_STRATEGY, current_seed)
            p_va_c = cat.predict_proba(Xva)[:, 1]
            t_c = config.FIXED_VAL_EXP2 if config.EXP_MODE == 2 else utils.tune_cutoff(y_va, p_va_c, config.CUTOFF_STRATEGY, pos_prior)[0]
            cur_cuts["cat"] = t_c
            preds_te["cat"] = cat.predict_proba(Xte)[:, 1]

            # 3. TabPFN
            tab = train_tab_fixed(Xtr, y_tr, current_seed)
            if tab:
                p_va_t = tab.predict_proba(Xva.values.astype(np.float32))[:, 1]
                t_t = config.FIXED_VAL_EXP2 if config.EXP_MODE == 2 else utils.tune_cutoff(y_va, p_va_t, config.CUTOFF_STRATEGY, pos_prior)[0]
                cur_cuts["tab"] = t_t
                preds_te["tab"] = tab.predict_proba(Xte.values.astype(np.float32))[:, 1]

            m_res = {"seed": current_seed}

            # Evaluate individual models
            for model_name in preds_te.keys():
                yhat_model = (preds_te[model_name] >= cur_cuts[model_name]).astype(int)
                m_res[f"{model_name}_precision"] = precision_score(y_te, yhat_model, zero_division=0)
                m_res[f"{model_name}_recall"] = recall_score(y_te, yhat_model, zero_division=0)
                m_res[f"{model_name}_f1"] = f1_score(y_te, yhat_model, zero_division=0)
                m_res[f"{model_name}_f{config.F_BETA}"] = fbeta_score(y_te, yhat_model, beta=config.F_BETA, zero_division=0)
                m_res[f"{model_name}_ap"] = average_precision_score(y_te, preds_te[model_name])

            # Evaluate ensemble
            yhat_te_ens = ensemble_predictions(preds_te, cur_cuts)
            m_res.update({
                "ens_precision": precision_score(y_te, yhat_te_ens, zero_division=0),
                "ens_recall": recall_score(y_te, yhat_te_ens, zero_division=0),
                "ens_f1": f1_score(y_te, yhat_te_ens, zero_division=0),
                f"ens_f{config.F_BETA}": fbeta_score(y_te, yhat_te_ens, beta=config.F_BETA, zero_division=0),
                "ens_ap": average_precision_score(y_te, np.mean(list(preds_te.values()), axis=0))
            })

            all_test_metrics.append(m_res)
            pd.DataFrame([m_res]).to_csv(config.METRIC_DIR / f"metric_{current_seed}.csv", index=False)
            
            # Save metadata for Phase 3
            joblib.dump({
                "enc": enc, "imputer": (num_cols, med), "feat_cols": feat_cols, 
                "whale_cut": whale_cut_prov, "cat_idx": cat_idx, "cur_cuts": cur_cuts
            }, config.ART_DIR / f"meta_{current_seed}.joblib")

        except Exception as e:
            print(f"Error in seed {current_seed}: {e}")
            continue

    # Phase 3: Final inference (Full Train & Seed Ensemble)
    df_base_full = pd.read_parquet(config.DEFAULT_INPUT_DATA_PATH)
    df_final_target = df_base_full[df_base_full[config.STAGE1_LABEL_COL] == 1].copy()
    final_ensemble_df = df_final_target[[config.ID_COL]].set_index(config.ID_COL)

    for seed in config.EXP_SEEDS:
        meta_p = config.ART_DIR / f"meta_{seed}.joblib"
        if not meta_p.exists(): continue
        meta = joblib.load(meta_p)
        
        y_full = (df_final_target[config.TARGET_COL] >= meta["whale_cut"]).astype(int)
        X_full = meta["enc"].transform(df_final_target[meta["feat_cols"]])
        X_full = apply_imputer(X_full, meta["imputer"][0], meta["imputer"][1])
        X_full = utils._sanitize_cols(X_full)
        
        lgbm_f = train_lgbm_fixed(X_full, y_full, X_full, y_full, config.TRAIN_STRATEGY, seed)
        cat_f = train_cat_fixed(X_full, y_full, X_full, y_full, meta["cat_idx"], config.TRAIN_STRATEGY, seed)
        tab_f = train_tab_fixed(X_full, y_full, seed)
        
        p_full = {
            "lgbm": lgbm_f.predict_proba(X_full)[:, 1],
            "cat": cat_f.predict_proba(X_full)[:, 1]
        }
        if tab_f:
            p_full["tab"] = tab_f.predict_proba(X_full.values.astype(np.float32))[:, 1]
        
        yhat_seed = ensemble_predictions(p_full, meta["cur_cuts"])
        
        seed_res = pd.DataFrame({config.ID_COL: df_final_target[config.ID_COL], f"pred_seed_{seed}": yhat_seed}).set_index(config.ID_COL)
        final_ensemble_df = final_ensemble_df.join(seed_res)

    pred_cols = [c for c in final_ensemble_df.columns if c.startswith("pred_seed_")]
    if pred_cols:
        final_ensemble_df["final_stage2_whale_pred"] = (final_ensemble_df[pred_cols].mean(axis=1) >= 0.5).astype(int)
        
        all_seeds_output = final_ensemble_df.reset_index()
        all_seeds_output.to_parquet(config.DEF_OUTPUT_DIR / "stage2_all_seeds_predictions.parquet", index=False)

    final_output = df_base_full.set_index(config.ID_COL).join(final_ensemble_df[["final_stage2_whale_pred"]], how='left')
    final_output["final_stage2_whale_pred"] = final_output["final_stage2_whale_pred"].fillna(0).astype(int)
    
    final_output.reset_index().to_parquet(config.DEF_OUTPUT_DIR / "stage2_final_predictions_full_ensemble.parquet", index=False)

    if all_test_metrics:
        res_df = pd.DataFrame(all_test_metrics)
        res_df.to_csv(config.DEF_OUTPUT_DIR / "all_seeds_metrics_summary.csv", index=False)
        
        metrics_only = res_df.drop(columns="seed")
        summary = pd.DataFrame({
            "mean": metrics_only.mean(),
            "se": metrics_only.apply(sem)
        })
        
        summary.to_csv(config.DEF_OUTPUT_DIR / "final_stats_report.csv")
        print("\nFinal Result Summary:\n", summary.to_string())

if __name__ == "__main__":
    run_experiment()