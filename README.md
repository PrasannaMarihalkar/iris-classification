# 🌸 Iris Classification — KNN Data Classification Pipeline
**DecodeLabs AI Internship | Project 2 | Batch 2026**

A production-quality supervised learning pipeline that classifies Iris flower species using K-Nearest Neighbors. Built to demonstrate deep understanding of the ML lifecycle — not just model accuracy, but *why every design decision exists*.

---

## Project Overview

| Property | Detail |
|----------|--------|
| **Dataset** | Iris (150 samples, 3 classes, 4 features) |
| **Algorithm** | K-Nearest Neighbors (KNN) |
| **Framework** | Scikit-Learn |
| **Primary Metric** | F1 Score (Macro + Weighted) |
| **Secondary** | Confusion Matrix, Accuracy |
| **Bonus** | KNN vs Decision Tree vs Logistic Regression |

---

## IPO Framework

```
INPUT                   PROCESS                  OUTPUT
─────────────────       ──────────────────────   ──────────────────────
Iris Dataset       →    StandardScaler        →  Confusion Matrix
(4 features)            Train/Test Split (80/20)  F1 Score
Feature Scaling         KNN (elbow-tuned K)       Per-class Metrics
                        Elbow Curve Analysis       Algorithm Comparison
```

---

## Repository Structure

```
iris-classification/
│
├── iris_knn_classification.ipynb   ← Main notebook (run this)
├── requirements.txt                 ← Python dependencies
├── README.md                        ← This file
│
└── plots/ (generated on run)
    ├── feature_distributions.png    ← Histogram per species per feature
    ├── elbow_curve.png              ← Error rate vs K — elbow identified
    ├── confusion_matrix.png         ← Raw + normalised confusion matrix
    └── algorithm_comparison.png     ← KNN vs DT vs LR bar chart
```

---

## Quickstart

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/iris-classification.git
cd iris-classification
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Launch the notebook
```bash
jupyter notebook iris_knn_classification.ipynb
```

Run all cells top-to-bottom (Kernel → Restart & Run All). Everything is self-contained.

---

## Methodology

### Why StandardScaler?
KNN classifies by computing Euclidean distance between points. Without scaling, features with larger absolute ranges dominate the distance metric — not because they're more informative, but simply because their numbers are bigger. `StandardScaler` transforms every feature to mean=0, std=1, giving all four dimensions equal weight in distance calculations.

```
z = (x - μ) / σ
```

**Critical:** The scaler is fitted on training data only, then applied to both train and test sets. Fitting on the full dataset would leak test statistics into training — a subtle but serious methodological error.

### Why the Elbow Method for K?
- **K=1**: Model memorises training data. Any noise or outlier becomes a "rule". Overfits.
- **K=large**: Model consults so many neighbours it ignores local structure. Underfits.
- **Elbow point**: Where error rate stops improving meaningfully. We test odd K values (1, 3, 5…29) to prevent voting ties, plot the error curve, and select the K at the minimum.

### Why F1 Score over Accuracy?
On balanced datasets like Iris, accuracy and F1 tell a similar story. However, F1 is the correct habit to build because:
- Accuracy is misleading on imbalanced datasets (predicting the majority class always gives high accuracy)
- F1 = harmonic mean of precision and recall — it penalises models that are strong on one but weak on the other
- The confusion matrix shows exactly *which* species are being confused, something a single accuracy number never reveals

---

## Key Results

After running the notebook you will see:
- **Elbow curve** identifying the optimal K value
- **Confusion matrix** showing per-class true/false positives and negatives
- **Classification report** with per-class precision, recall, and F1
- **Algorithm comparison** benchmarking KNN vs Decision Tree vs Logistic Regression

Typical results on this dataset:
- KNN achieves **~96–100% accuracy** with properly tuned K
- Setosa is perfectly separable; Versicolor/Virginica share a partial overlap zone
- All three algorithms perform competitively — this dataset is a good fit for distance-based methods

---

## Design Decisions

| Decision | Reasoning |
|----------|-----------|
| `test_size=0.20` | Standard 80/20 split; gives 30 test samples — enough for stable metrics |
| `stratify=y` | Preserves class proportions; prevents a lopsided test set by chance |
| `shuffle=True` | Raw CSV is sorted by class — shuffling removes ordering bias |
| `random_state=42` | Reproducibility; every run produces identical results |
| Odd K values | Prevents voting ties in binary edges of multi-class boundaries |
| `average='macro'` for F1 | Treats all classes equally regardless of size — correct for balanced datasets |

---

## Dependencies

```
scikit-learn>=1.0
pandas>=1.3
numpy>=1.21
matplotlib>=3.4
seaborn>=0.11
jupyter>=1.0
```

---

## GitHub Setup (Step-by-Step)

```bash
# 1. Initialise local repo
git init

# 2. Stage all files
git add .

# 3. First commit
git commit -m "feat: initial KNN classification pipeline with Iris dataset"

# 4. Create repo on github.com (via UI), then link
git remote add origin https://github.com/YOUR_USERNAME/iris-knn-classification.git
git branch -M main

# 5. Push
git push -u origin main
```

**Suggested commit history:**
```
feat: initial project structure and requirements
feat: data loading, exploration, and feature distributions
feat: StandardScaler preprocessing with before/after comparison  
feat: train/test split with stratification
feat: elbow curve analysis and optimal K selection
feat: KNN model training and confusion matrix evaluation
feat: algorithm comparison — KNN vs DT vs Logistic Regression
docs: professional README with methodology and results
```

---

## What This Demonstrates

This project proves competency in the complete supervised learning pipeline:

1. **Data literacy** — loading, inspecting, and visualising a structured dataset
2. **Preprocessing rigour** — understanding *why* StandardScaler matters for distance-based algorithms
3. **Principled hyperparameter tuning** — elbow method, not arbitrary K selection
4. **Evaluation depth** — confusion matrix + F1, not just accuracy
5. **Comparative thinking** — benchmarking multiple algorithms under identical conditions
6. **Reproducibility** — seeded randomness, clean code, full documentation

---

*Built for DecodeLabs Industrial Training Kit | Batch 2026*
