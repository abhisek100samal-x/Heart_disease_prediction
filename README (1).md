# Heart Disease Prediction

A machine learning project that predicts whether a patient has heart disease from routine clinical measurements. The notebook covers the full workflow: exploratory data analysis, cleaning, feature encoding and scaling, training four classifiers, hyperparameter tuning, and comparison with ROC curves.

**Best test accuracy: ~88.6%** (Random Forest), with a cross-validated ROC AUC of **0.922** for the tuned Random Forest.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Workflow](#workflow)
- [Results](#results)
- [Key Takeaways](#key-takeaways)
- [Limitations and Future Work](#limitations-and-future-work)
- [Repository Structure](#repository-structure)
- [Getting Started](#getting-started)
- [Tech Stack](#tech-stack)
- [Disclaimer](#disclaimer)

---

## Project Overview

Cardiovascular disease is one of the leading causes of death worldwide, and early detection matters. This project builds a binary classifier that takes a patient's demographic details, resting vitals, and exercise-test results and predicts whether heart disease is present (`1`) or absent (`0`).

The goal is to compare a simple, interpretable baseline (Logistic Regression) against tree-based ensemble methods (Random Forest, XGBoost) and see which generalises best.

## Dataset

- **Size:** 918 patients, 11 features + 1 target
- **Target:** `HeartDisease` (1 = disease, 0 = no disease), with 508 positive and 410 negative cases (roughly 55% / 45%)
- **Missing values:** none reported, but `Cholesterol` and `RestingBP` contain `0` values, which are physiologically impossible and treated as missing (see Workflow)

| Feature | Type | Description |
|---|---|---|
| `Age` | Numeric | Age of the patient in years |
| `Sex` | Categorical | M / F |
| `ChestPainType` | Categorical | TA (typical angina), ATA (atypical angina), NAP (non-anginal pain), ASY (asymptomatic) |
| `RestingBP` | Numeric | Resting blood pressure (mm Hg) |
| `Cholesterol` | Numeric | Serum cholesterol (mg/dl) |
| `FastingBS` | Binary | 1 if fasting blood sugar > 120 mg/dl, else 0 |
| `RestingECG` | Categorical | Normal, ST (ST-T wave abnormality), LVH (left ventricular hypertrophy) |
| `MaxHR` | Numeric | Maximum heart rate achieved |
| `ExerciseAngina` | Categorical | Exercise-induced angina (Y / N) |
| `Oldpeak` | Numeric | ST depression induced by exercise relative to rest |
| `ST_Slope` | Categorical | Slope of the peak exercise ST segment (Up, Flat, Down) |
| `HeartDisease` | Binary | **Target variable** |

> The columns match the widely used *Heart Failure Prediction* dataset on Kaggle. Download `heart.csv` from there (or your own source) and place it in the `data/` folder.

## Workflow

### 1. Exploratory Data Analysis
- Inspected shape, data types, and summary statistics
- Checked target class balance
- Plotted distributions (with KDE) of `Age`, `RestingBP`, `Cholesterol`, and `MaxHR`
- Compared heart disease rates by `Sex` and `FastingBS`
- Used a box plot (cholesterol) and violin plot (age) split by the target
- Built a correlation heatmap of the numeric features

### 2. Data Cleaning
- Replaced invalid `0` values in `Cholesterol` and `RestingBP` with the mean of the valid (non-zero) values

### 3. Preprocessing
- One-hot encoded categorical variables with `pd.get_dummies(drop_first=True)`
- Standardised the numeric columns (`Age`, `RestingBP`, `Cholesterol`, `MaxHR`, `Oldpeak`) using `StandardScaler`
- Stratified 80/20 train-test split (`random_state=42`), giving 736 training and 184 test samples

### 4. Modelling
Four models were trained and evaluated:

| Model | Configuration |
|---|---|
| Logistic Regression | `max_iter=1000` |
| Random Forest | 200 trees, `class_weight='balanced'` |
| XGBoost | 200 trees, `max_depth=4`, `learning_rate=0.05`, `subsample=0.8`, `colsample_bytree=0.8` |
| Tuned Random Forest | `GridSearchCV` (5-fold, scored on ROC AUC) over 72 parameter combinations |

**Best parameters found by grid search:** `n_estimators=200`, `max_depth=10`, `max_features='sqrt'`, `min_samples_split=5`, `min_samples_leaf=1`

## Results

Evaluated on the held-out test set (184 patients):

| Model | Accuracy | Precision (Disease) | Recall (Disease) | F1 (Disease) |
|---|---|---|---|---|
| Logistic Regression | 0.880 | 0.89 | 0.89 | 0.89 |
| **Random Forest** | **0.886** | 0.89 | 0.90 | 0.90 |
| XGBoost | 0.853 | 0.86 | 0.87 | 0.87 |
| Tuned Random Forest | 0.870 | 0.88 | 0.89 | 0.88 |

Best cross-validated ROC AUC (Tuned Random Forest, 5-fold on the training set): **0.922**

**Confusion matrices** (rows = actual, columns = predicted, order: No Disease / Disease)

| Model | True Neg | False Pos | False Neg | True Pos |
|---|---|---|---|---|
| Logistic Regression | 71 | 11 | 11 | 91 |
| Random Forest | 71 | 11 | 10 | 92 |
| XGBoost | 68 | 14 | 13 | 89 |
| Tuned Random Forest | 69 | 13 | 11 | 91 |

The notebook also plots ROC curves for all four models on a single chart for side-by-side comparison.

## Key Takeaways

- **A simple baseline is very competitive.** Logistic Regression (88.0%) is within one percentage point of the best model, which suggests the signal in this dataset is largely linear and well captured after scaling and encoding.
- **Random Forest performed best on the test set (88.6%)** and had the highest recall for the disease class (0.90), the metric that matters most in a screening scenario where missing a sick patient is costly.
- **Tuning did not improve test accuracy.** The tuned Random Forest (87.0%) scored slightly below the default one (88.6%). With only 184 test samples, a 1 to 2 point gap equals a handful of patients, so the models are effectively comparable.
- **XGBoost was the weakest here (85.3%)**, likely because of the small dataset size and untuned settings.

## Limitations and Future Work

- **Small dataset.** 918 records and a single train/test split mean the differences between models are within noise. Repeated or stratified k-fold cross-validation would give more reliable comparisons.
- **Preprocessing before splitting.** Mean imputation and scaling are fitted on the full dataset before the train/test split, which can leak a small amount of test information. Wrapping these steps in a scikit-learn `Pipeline` / `ColumnTransformer` would remove this.
- **Mean imputation is crude.** Model-based imputation (e.g. KNN or iterative imputer) or an explicit "was missing" indicator could work better for the cholesterol zeros.
- **No feature importance analysis yet.** Adding permutation importance or SHAP values would make the models more interpretable.
- **Other improvements to try:** threshold tuning to favour recall, probability calibration, tuning XGBoost, and deploying the best model as a small Streamlit app.

## Repository Structure

```
heart-disease-prediction/
├── data/
│   └── heart.csv          # dataset (download separately)
├── heart.ipynb            # main notebook: EDA, preprocessing, modelling
├── requirements.txt       # Python dependencies
└── README.md
```

## Getting Started

**1. Clone the repository**

```bash
git clone https://github.com/<your-username>/heart-disease-prediction.git
cd heart-disease-prediction
```

**2. Create a virtual environment (optional but recommended)**

```bash
python -m venv venv
source venv/bin/activate      # on Windows: venv\Scripts\activate
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

**4. Add the dataset**

Place `heart.csv` inside the `data/` folder and update the path in the notebook's first data-loading cell:

```python
df = pd.read_csv('data/heart.csv')
```

**5. Run the notebook**

```bash
jupyter notebook heart.ipynb
```

### `requirements.txt`

```
numpy
pandas
matplotlib
seaborn
scikit-learn
xgboost
jupyter
```

## Tech Stack

- **Language:** Python
- **Data handling:** NumPy, pandas
- **Visualisation:** Matplotlib, Seaborn
- **Machine learning:** scikit-learn, XGBoost

## Disclaimer

This project is for educational and portfolio purposes only. It is **not** a medical device and must not be used for real diagnosis or treatment decisions.

---

If you found this project useful, feel free to star the repository. Feedback and suggestions are welcome.
