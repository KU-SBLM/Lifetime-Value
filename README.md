# Lifetime-Value

## Goal


This project implements a **Three-stage Hierarchical Ensemble Framework pipeline** designed to predict **Lifetime Value (LTV)** of users. The pipeline is optimized for highly imbalanced spending distributions and leverages both classification and regression models in a hierarchical structure.

Our work targets two core challenges in game monetization modeling:

  * Predicting long-term LTV using only early logs after launching** (launch-period behavior → lifetime value).

  * Accurately predicting LTV for extremely high-value payers** whose spending dominates overall revenue distribution.

These goals shape the entire architecture of our 3-stage pipeline, enabling robust prediction in extremely imbalanced, heavy-tailed spending environments.



## Project Overview

Predicting player LTV in games is challenging due to:

* Strong **class imbalance** (many non-spenders vs. few spenders)
* **Long-tailed distribution** among spenders
* The need to model **heterogeneous behavioral patterns**


To address this, we use a **3-stage pipeline**:

1. **Stage 1 – Payer Classification** (Non-payer vs. Payer)
2. **Stage 2 – High-value Payer Classification** (Low spender vs. High spender)
3. **Stage 3 – Two-head Regression** (Separate regressors for each segment)


## Pipeline Architecture

Below is the overall architecture of the proposed **Three-stage Hierarchical Ensemble Framework**:

<p align="center">
  <img src="ltv_pipeline.png" width="800">
</p>

The pipeline sequentially reduces distributional complexity:

- Stage 1 separates **non-payers and payers**
- Stage 2 isolates **extreme high-value (whale) users**
- Stage 3 applies **segment-specific regression models**

This hierarchical decomposition allows the model to:
- Handle extreme class imbalance
- Reduce underestimation bias for high-value users
- Improve stability in heavy-tailed distributions