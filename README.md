# 🎮 PSG — When to insert an Ad?

### 🧠 Objective

This project simulates a **mobile gaming environment** and applies **machine learning** to optimize **ad-serving decisions** — deciding *when to show or skip an ad* to **maximize revenue** while **minimizing churn risk**.

---

## 📁 Repository Structure

| File                                   | Description                                                                                                                                                                |
| -------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 📘 **`PSG_Ad_DataGen.ipynb`**          | Generates a **synthetic dataset** of player sessions, ad exposures, engagement, and monetization metrics based on realistic 2025 gaming benchmarks.                        |
| 🤖 **`PSG_Ad_Models_102125_v3.ipynb`** | Trains and evaluates **machine learning models** (classification & contextual bandit preparation) to predict optimal ad-serving actions that balance **reward vs. churn**. |
| 📄 **`requirements.txt`**              | Lists all Python dependencies required to run both notebooks.                                                                                                              |
                                                                                                                            |

---

## 🧩 1. Data Generation — `PSG_Ad_DataGen.ipynb`

### 📋 Description

Creates a **simulated player–session–ad dataset** reflecting realistic puzzle gaming dynamics.

**Simulated features include:**

* 🎮 `session_duration` → 5–15 minutes
* 📺 `ads_shown_per_session` → 2–5 ads
* ⚠️ `churn_probability` → 20–30%
* 💰 `ad_revenue_per_impression` → $0.001–$0.015
* 🧮 `click_rate`, `ARPDAU`, `LTV`, `retention_rate`

**Output:**
A CSV file where each row = one user–session–ad interaction, ready for modeling.

---

## 🤖 2. Model Training & Evaluation — `PSG_Ad_Models_102125_v3.ipynb`

### ⚙️ Description

Trains ML models on the synthetic dataset to **predict churn or reward** outcomes, given user and gameplay context.

**Key Steps:**

1. Data preprocessing — scaling, encoding, and imputation
2. Model training — Logistic Regression, XGBoost, or Neural models
3. Evaluation metrics:

   * 🧾 ROC-AUC
   * 📈 Average Precision
   * 🎯 Precision–Recall Curve
   * 📊 Calibration curve
4. Future-ready structure for **Contextual Bandit / Policy Optimization**.

**Optimization Goal:**
Formulate a policy π(*state → action*) that maximizes:

> 💡 **Expected Reward = Revenue_if_Shown – (Churn_Risk × LTV)**

---

## 🧱 Setup Instructions

### 1️⃣ Clone Repository

```bash
git clone https://github.com/<your-username>/playsimple-ad-optimization.git
cd playsimple-ad-optimization
```

### 2️⃣ Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate   # macOS/Linux
venv\Scripts\activate      # Windows
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Run Notebooks

```bash
jupyter notebook
```

Then open and run:

* `PSG_Ad_DataGen.ipynb` → to generate the dataset
* `PSG_Ad_Models_102125_v3.ipynb` → to train and evaluate models

---

## 🧾 Dependencies

See [`requirements.txt`](./requirements.txt) for the complete list.
**Core libraries:**

* `pandas`, `numpy`, `scikit-learn`
* `torch`, `xgboost`, `imblearn`
* `matplotlib`, `seaborn`, `ipython`, `scipy`

---

## 🚀 Future Extensions

* 🧠 Integrate **Reinforcement Learning / Multi-Armed Bandit** algorithms
* 🔁 Add **live A/B testing simulation**
* 📈 Visualize **policy learning curves** and **reward–churn trade-offs**

---

## 👤 Author

**Vinod Kumar K**
*PSG – Ad Optimization Assignment (2025)*


---
