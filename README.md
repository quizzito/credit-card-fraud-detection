# Credit Card Fraud Detection

Detecting fraudulent transactions using ensemble classifiers with
rigorous handling of extreme class imbalance (0.17% fraud rate).

## Problem framing
284,807 real credit card transactions. 492 are fraud.
Naive accuracy is useless — the goal is maximising
**Average Precision** and **Recall** on the minority class.

## Approach

| Step | Technique | Why |
|------|-----------|-----|
| Class imbalance | SMOTE (30%) + class_weight | Avoids model ignoring fraud |
| Scaling | RobustScaler on Amount | Outlier-resistant |
| Feature engineering | Hour-of-day, log(Amount), V14×V17 | Adds circadian + interaction signal |
| Assumption checks | Mann-Whitney U, correlation matrix | Validates feature relevance |
| Threshold tuning | F1-maximising search | 0.5 is not always optimal |

## Model Results

| Model | Avg Precision | Recall | Precision | F1 | Notes |
|-------|--------------|--------|-----------|----|-------|
| Logistic Regression | 0.7120 | 0.9286 | 0.0585 | 0.1100 | Eliminated — 94 false alarms per 6 detections |
| Random Forest | 0.8482 | 0.8367 | 0.8367 | 0.8367 | Strong baseline, no SMOTE needed |
| XGBoost + SMOTE | 0.8656 | 0.8469 | 0.7281 | 0.7830 | Good recall, precision drops vs RF |
| LightGBM + SMOTE | 0.8744 | 0.8673 | 0.8173 | 0.8416 | Best overall — chosen model |
| LightGBM Tuned | — | — | — | — | Fill in after GridSearchCV |
| Isolation Forest | — | — | — | — | Unsupervised baseline — fill in |

5-Fold CV (LightGBM): Avg Precision = — ± —

### Threshold selection
Operating threshold set to 0.5 (default).
At this threshold the model catches X% of fraud (Recall)
with Y% of alerts being genuine fraud (Precision).
Lowering to 0.3 increases Recall to Z% at the cost of more false alarms.

## Key finding
[2–3 sentences: best model result, what surprised you in feature importances,
and what you would do next.]

## Setup
```bash
git clone https://github.com/YOUR_USERNAME/credit-card-fraud-detection
cd credit-card-fraud-detection
pip install -r requirements.txt
kaggle datasets download -d mlg-ulb/creditcardfraud --path ./data/raw/ --unzip
jupyter notebook notebooks/01_fraud_detection.ipynb
```

## Dataset
ULB Credit Card Fraud Detection — kaggle.com/datasets/mlg-ulb/creditcardfraud
Data not included in this repo. Download separately via Kaggle API.

