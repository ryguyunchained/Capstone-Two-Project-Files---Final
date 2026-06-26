# Model Metrics

## Problem Framing

**Task type:** Binary classification
**Target:** `conflict_escalation_6m` — whether a country experiences a conflict escalation within the next six months (1 = escalation, 0 = none)
**Dataset:** 20 countries, monthly observations 2020–2025

**Split:** Chronological — train = months before 2024-01-01, test = 2024-01-01 onward (no shuffling, to preserve the time-series structure and avoid look-ahead leakage)
**Train set:** 688 observations (217 positive / 471 negative, ~31.5% positive)
**Test set:** 360 observations (110 positive / 250 negative, ~30.6% positive)
**Features used:** ~48–49 (after dropping `regime_type`, `election_cycle` for low signal and `political_stability_index` for multicollinearity; engineered rolling/lag features added to preserve temporal signal in a classification frame)

> Note: the positive class is the minority (~31%), so accuracy alone is misleading. Recall on the conflict class (class 1) is the metric that matters most here — missing an escalation is costlier than a false alarm.

---

## Models Evaluated

| Model | Approach | Accuracy | Class 1 Recall | Class 1 F1 |
|---|---|---|---|---|
| Logistic Regression (PCA baseline) | Scaling + PCA + multivariate logreg | — | low | low |
| LightGBM (default) | `class_weight='balanced'` | 0.93 | 0.95 | 0.89 |
| **LightGBM (tuned)** | RandomizedSearchCV, 50 iters, 5-fold, scoring=f1 | **0.93** | **0.97** | **0.90** |
| SVM (RBF) | StandardScaler + SVC, balanced | 0.85 | 0.75 | 0.75 |
| SVM (tuned linear) | RandomizedSearchCV, scoring=f1 | 0.89 | 0.87 | 0.83 |

The PCA + logistic regression baseline (from the pre-processing notebook) showed strong class 0 recall but weak class 1 recall — it was poor at the one thing the model exists to do, flag escalations. Gradient boosting closed that gap immediately.

---

## Classification Metrics — Chosen Model (Tuned LightGBM)

Evaluated on the held-out test set (n = 360).

| Metric | Class 0 (No Escalation) | Class 1 (Escalation) |
|---|---|---|
| Precision | 0.99 | 0.83 |
| Recall | 0.91 | 0.97 |
| F1 Score | 0.95 | 0.90 |
| Support | 250 | 110 |

**Accuracy:** 0.93   **Macro F1:** 0.92   **Weighted F1:** 0.93

**Confusion matrix:**

|  | Predicted: No Escalation | Predicted: Escalation |
|---|---|---|
| **Actual: No Escalation** | TN = 228 | FP = 22 |
| **Actual: Escalation** | FN = 3 | TP = 107 |

The model catches 107 of 110 actual escalations (0.97 recall), missing only 3. The cost is 22 false positives — countries flagged as at-risk that did not escalate within the window. Given the use case, this asymmetry is acceptable and arguably desirable.

---

## Chosen Model

**Selected model:** Tuned LightGBM classifier

**Why:** Hyperparameter tuning pushed class 1 recall from 0.95 to 0.97 while holding overall accuracy at 0.93, producing the best recall on the conflict class of any model tested. The SVM variants traded away too much recall (0.75–0.87) to be competitive. Because a false negative (a missed escalation) is more costly than a false positive in this domain, the model that most reliably catches escalations wins.

**Hyperparameters (final, from RandomizedSearchCV):**
```
n_estimators      = 100
learning_rate     = 0.01
num_leaves        = 80
subsample         = 0.8
colsample_bytree  = 1.0
scale_pos_weight  = 2
class_weight      = 'balanced'
random_state      = 42
```

---

## Top Predictive Features

The signals most associated with a six-month escalation, per the LightGBM feature-importance output:

1. Arms imports (index, plus rolling/lag variants)
2. Social media sentiment
3. Inflation rate
4. Unemployment rate
5. [confirm remaining ranks from the feature-importance plot in Capstone Two - Modeling.ipynb]

> Read the exact ordering and any remaining features directly off the `lgb.plot_importance` chart in the modeling notebook and finalize this list.

---

## Limitations & Notes

- **Target validity is the core caveat.** EDA showed the `conflict_escalation_6m` label behaves erratically month-to-month and misses major real-world events (e.g., the Jan 6 Capitol riot) while correctly anticipating others (e.g., Russia–Ukraine, one month prior). Strong test metrics reflect the model learning the dataset's labeling pattern, which is an imperfect proxy for real-world escalation.
- **Small, narrow panel.** 20 countries over five years is a limited sample; generalization to unseen countries or regimes is unproven.
- **Class imbalance** handled via `scale_pos_weight` / balanced class weights rather than resampling.
- **Interpretation:** the model's output is best read as a risk signal to support human judgment, not a deterministic forecast.
