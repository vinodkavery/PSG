# 📱 Ad Optimization in Mobile Gaming (PSG Case Study)

## 📑 Table of Contents
1. [Overview](#-overview)
2. [Data Generation](#-data-generation)  
   - [Generated Features](#generated-features)  
   - [Key Features of Data Generation](#key-features-of-data-generation)  
3. [Modeling Pipeline](#-modeling-pipeline)  
   - [Key Sections & Models](#key-sections--models)  
4. [Installation & Dependencies](#-installation--dependencies)  
5. [Usage](#-usage)  
6. [Results & Insights](#-results--insights)

---

## 🧩 Overview
This repository contains a comprehensive machine learning solution for optimizing ad placements in a mobile gaming app, inspired by the “Play Simple Games” case study.
The goal is to maximize ad revenue while minimizing user churn and enhancing retention through data-driven decisions.

The solution is split into two Jupyter notebooks:

- **PSG_Ad_DataGen_102025_v2.ipynb** — Creates synthetic gaming telemetry data.
- **PSG_Ad_Models_102125_v3.ipynb** — Builds and evaluates ML models for prediction, segmentation, and dynamic ad policies using XGBoost, neural networks, clustering, bandits, and reinforcement learning.

### Key Outcomes
- Predictive models for retention, churn, and ad revenue
- User segmentation for targeted strategies
- Contextual bandit and RL-based ad insertion policies
- A/B testing framework to quantify business impact (e.g., ARPDAU uplift, retention improvements)

This pipeline demonstrates end-to-end ML for ad optimization, balancing monetization with user experience.
All code is reproducible with a fixed random seed (**2025**) for consistency.

---

## 🧮 Data Generation

**Notebook:** `PSG_Ad_DataGen_102025_v2.ipynb`  
Simulates a realistic dataset of gaming telemetry (~200,000 rows) mimicking user interactions in a mobile game app.

### Generated Features
- **User Behavior:** Session durations (60–1800s), levels (1–1000), post-ad gameplay time
- **Ad Interactions:** Impressions, clicks (CTR ~3%), ad types (banner, interstitial, rewarded, native), placements (start/mid/end/idle), frequency
- **Monetization & Retention:**
  - Ad revenue per impression (~$0.02 mean)
  - Spend after ads
  - Churn probability (~0.22 mean, capped at 0.4)
  - Next-day retention (~78.5%)
  - LTV (~$5 mean)
- **Demographics:** 15 countries (e.g., US, IN, BR), Android/iOS (65%/35%)
- **Temporal Elements:** Timestamped over 30 days with time-of-day effects

### Key Features of Data Generation
- **Distributions:**
  - Poisson → sessions/ads per user
  - Normal → durations/revenue
  - Exponential → spends
  - Custom churn uplift logic (+1% per ad, reduced for rewarded ads)
- **Balancing:** Ensures minimum shares (≥15% ad type, ≥10% placements)
- **Validation Checks:**
  - Spearman correlation (frequency vs churn) ρ ≈ 0.27
  - Retention by ad type: Rewarded (~78.4%) > Interstitial (~78.6%)
  - CTR stability (~3% across quartiles)
  - Overall: Retention ~78.5%, churn ~22.3%, ARPU ~$0.029

**Output File:** `GamingData_Input_v6.csv` (~200K rows, 19 columns)

**Sample Columns:**
```
user_id | country | device_type | session_id | session_duration | level_reached |
ad_type | ad_placement | time_of_day | frequency | impressions | clicks |
game_time_post_ad | retention_next_day | spend_after_ad |
churn_probability | ad_revenue_per_impression | lifetime_value_estimate | event_time
```

This dataset serves as input for the modeling pipeline, enabling safe experimentation without real user data.

---

## 🧠 Modeling Pipeline

**Notebook:** `PSG_Ad_Models_102125_v3.ipynb`  
Processes the generated data through a full ML pipeline — configuration, logging, feature engineering, modeling, and evaluation.
Outputs are saved in `Model_Outputs_102125_v3/` (models, logs, CSVs, plots).

### Key Sections & Models

**1. Configuration & Logging**
- Sets paths, random state (2025)
- Custom run logger with timestamps, errors, Markdown tracking

**2. Data Loading & Preprocessing**
- Loads CSV, validates schema (19 cols), removes NaNs/duplicates
- Converts categories/dates, sorts by user/time
- **Output:** Clean DataFrame (~201K rows)

**3. Feature Engineering & Aggregations**
- Derived features: Day-of-week, weekend flag, CTR (clicks/impressions)
- Aggregates to user-level metrics
- Prepares categorical & numeric features for modeling

**4. Retention Prediction (XGBoost)**
- Binary classification for next-day retention
- Tuned via Bayesian optimization
- **Evaluation:** AUC (~0.85), PR-AUC, Brier score
- **Output:** Trained model, feature importances, calibration plots

**5. Churn Probability Calibration**
- Platt scaling using logistic regression on XGBoost outputs
- Improves Brier score
- **Output:** Calibrated churn model

**6. User Segmentation (Clustering)**
- K-Means on scaled behavioral features
- Optimal clusters via silhouette score
- **Output:** Cluster assignments and segment profiles

**7. Ad Revenue Prediction (Neural Network - PyTorch)**
- Regression for ad revenue per impression
- MSE loss, Adam optimizer, early stopping
- **Output:** Trained NN, R², RMSE, learning curves

**8. Ad Moment Identification**
- Rule-based + ML hybrid to find optimal insertion points (mid-session, post-level-up)

**9. Ad Decision Policy Model**
- Logistic regression for binary “show ad” decisions
- Balances revenue vs churn cost
- **Output:** Decision log CSV

**10. Expected Reward Calculation**
- Combines predicted revenue with churn-adjusted costs
- **Output:** Expected rewards CSV

**11. Policy Hyperparameter Tuning**
- Bayesian optimization for logistic model parameters

**12. Neural Network for Reward Prediction**
- Alternative NN for granular reward forecasting

**13. Contextual Bandit (LinUCB)**
- Two arms: show ad vs no ad
- Context-aware decisions with reward updates
- **Output:** CSVs, agent pickle, learning curve

**14. Reinforcement Learning Simulation**
- Episodic RL (200 episodes, batch size 1024)
- Tracks cumulative rewards
- **Output:** RL decisions, learning curves

**15. A/B/n Evaluation Framework**
- Compares baseline vs ML policies
- **KPIs:** ARPDAU, retention, coverage
- **Output:** `ab_test_kpis.csv` (ML uplift ~10–20%)

### Example Outputs
- **Models:** `xgb_retention.pkl`, `rl_agent.pkl`
- **CSVs:** Predictions, decisions, KPIs
- **Plots:** Feature importances, calibration, learning/regret curves
- **Logs:** Stepwise TXT logs with runtime & errors

---

## ⚙️ Installation & Dependencies

```bash
git clone <repo-url>
cd psg-ad-optimization
pip install -r requirements.txt
```

**Required Packages**
```
pandas
numpy
scikit-learn
xgboost
torch
matplotlib
bayes_opt
joblib
```

**Environment**
- Python 3.10+ (tested on 3.13)
- No GPU required (NNs are lightweight)

---

## 🚀 Usage

- Run `PSG_Ad_DataGen_102025_v2.ipynb` to generate synthetic input data.
- Run `PSG_Ad_Models_102125_v3.ipynb` end-to-end or cell-by-cell.
- Outputs appear under `Model_Outputs`.

**Customize parameters:**
- `RANDOM_STATE`
- File paths
- Model hyperparameters

**For production:** Wrap as scripts, deploy models via Flask/FastAPI, and integrate with A/B testing tools like Optimizely.

---

## 📊 Results & Insights

**Business Impact:** ML-based policies show ~15% ARPDAU uplift vs baseline with minimal retention loss.

**Learnings:**
- Rewarded ads boost retention
- High ad frequency increases churn
- Bandits adapt rapidly to contexts

**Limitations:**
- Synthetic dataset
- Assumes linear reward — real A/B validation needed

**Future Work:**
- Extend to multi-armed bandits for ad types
- Add causal inference (uplift modeling)
- Scale via Spark for big data environments

---

✅ This repository demonstrates an end-to-end applied ML workflow — from data simulation to policy learning — tailored for ad monetization and user experience optimization in mobile gaming.
