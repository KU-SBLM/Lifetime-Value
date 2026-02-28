#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
from time import perf_counter
from typing import Dict, List
import numpy as np
import pandas as pd

class SectionTimer:
    def __init__(self, msg: str = ""):
        self.msg = msg
        self.t0 = None

    def __enter__(self):
        logging.info(f"⏱️  {self.msg} ...")
        self.t0 = perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        dt = perf_counter() - self.t0 if self.t0 else 0.0
        logging.info(f"✅ {self.msg} done in {dt:.2f}s")


class OrdinalCategoryEncoder:
    def __init__(self):
        self.maps: Dict[str, Dict] = {}
        self.cols: List[str] = []

    def fit(self, df: pd.DataFrame, cat_cols: List[str]):
        self.cols = list(cat_cols)
        for c in self.cols:
            if c not in df.columns:
                continue
            unique_vals = df[c].dropna().unique()
            try:
                sorted_cats = sorted(unique_vals, key=float)
            except (ValueError, TypeError):
                sorted_cats = map(str, unique_vals)
            self.maps[c] = {cat: i for i, cat in enumerate(sorted_cats)}
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        for c in self.cols:
            if c in out.columns:
                mapping = self.maps[c]
                s = out[c].astype(object)
                out[c] = s.apply(lambda v: mapping.get(v, -1)).astype(np.int32)
        return out