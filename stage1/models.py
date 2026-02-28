#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

class XGBCompat:
    def __init__(self, **params):
        self.params = params.copy()
        self.booster_ = None

    def fit(self, X_tr, y_tr, X_va, y_va, n_estimators):
        import xgboost as xgb
        dtr = xgb.DMatrix(X_tr, label=y_tr)
        train_params = self.params.copy()
        num_boost_round = int(n_estimators)

        evals = []
        if X_va is not None and y_va is not None:
            dva = xgb.DMatrix(X_va, label=y_va)
            evals = [(dva, 'validation')]

        self.booster_ = xgb.train(
            params=train_params, dtrain=dtr, num_boost_round=num_boost_round,
            evals=evals, verbose_eval=False
        )
        return self

    def predict_proba(self, X):
        import xgboost as xgb
        d = xgb.DMatrix(X)
        p1 = self.booster_.predict(d)
        p1 = np.asarray(p1, dtype=float).reshape(-1)
        p0 = 1.0 - p1
        return np.column_stack([p0, p1])


class LGBBoosterWrapper:
    def __init__(self, **params):
        self.params = params.copy()
        self.booster_ = None

    def fit(self, X_tr, y_tr, Xva, y_va, n_estimators, class_weight=None):
        import lightgbm as lgb
        lgb_params = self.params.copy()
        lgb_params.pop("class_weight", None)

        sample_weight = None
        if isinstance(class_weight, dict):
            w0 = class_weight.get(0, 1.0)
            w1 = class_weight.get(1, 1.0)
            sample_weight = np.where(y_tr == 1, w1, w0).astype(float)

        lgb_train = lgb.Dataset(X_tr, label=y_tr, weight=sample_weight)
        self.booster_ = lgb.train(lgb_params, lgb_train, num_boost_round=int(n_estimators))
        return self

    def predict_proba(self, X):
        p1 = self.booster_.predict(X, num_iteration=self.booster_.current_iteration())
        p1 = np.asarray(p1, dtype=float).reshape(-1)
        p0 = 1.0 - p1
        return np.column_stack([p0, p1])