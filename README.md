# HR Promotion Prediction & Decision Support System

An end-to-end Machine Learning solution designed to streamline the annual employee promotion cycle by identifying eligible high-performing candidates early through predictive analytics and a real-time FastAPI interface.

---

## ✨ 1. Interactive API Demo

<p align="center">
  <img width="900" alt="API Demo Interface" src="https://github.com/user-attachments/assets/be3e4e60-90b4-48d1-8659-967a1d1d9c13" />
</p>

---

## 🧭 2. Business Context & Proposed Solution

### The Problem
A large Multinational Corporation (MNC) operates across 9 broad verticals. A primary bottleneck is identifying the right personnel for promotions and preparing them in time. Promotions are traditionally announced only after rigid annual evaluations, forcing capable employees to wait months before transitioning, leading to operational delays and talent attrition.

### The Solution
This project delivers a production-ready ML pipeline to evaluate candidates at intermediate checkpoints. 
* **Model Strategy:** A custom Soft-Voting Ensemble (**LightGBM + AdaBoost**) optimized for high-precision decision-making.
* **Deployment:** Integrated into a real-time **REST API using FastAPI**, enabling HR personnel to input employee metrics and instantly receive data-driven recommendations (*"Recommend for Promotion"* vs. *"Keep under observation"*).

---

## 🧪 3. Data & Preprocessing Overview

* **Data Source:** Publicly available via [Kaggle HR Analytics Dataset](https://www.kaggle.com/datasets/arashnic/hr-analytics-jobchange).  
* **Dataset Scale:** 54,808 samples and 13 features spanning performance history and demographics.

### Data Dictionary

| Field | Description |
| :--- | :--- |
| `employee_id` | Unique ID for the employee |
| `department` | Department of the employee |
| `region` | Region of employment (unordered) |
| `education` | Education level |
| `gender` | Gender of the employee |
| `recruitment_channel` | Channel of recruitment |
| `no_of_trainings` | Number of other trainings completed in the previous year |
| `age` | Age of the employee |
| `previous_year_rating` | Employee rating for the previous year |
| `length_of_service` | Length of service in years |
| `awards_won?` | 1 if awards won during the previous year, else 0 |
| `avg_training_score` | Average score in current training evaluations |
| `is_promoted` (target) | 1 if recommended for promotion, else 0 |

### Preprocessing & AI Fairness Strategy
* **Imbalanced Handling:** Class 1 (promoted) accounts for ~8.69%. Handled via model-level class weighting. SMOTE was deliberately bypassed to prevent generating unrealistic categorical noise.
* **Missing Value Imputation:** 
  * `previous_year_rating` (~7.63% missing): Imputed with `0` (reflecting new hires without a prior year record).
  * `education` (~4.38% missing): Imputed using the most frequent strategy.
* **Bias Mitigation:** Dropped `employee_id`, `gender`, and `age`. A preliminary Chi-Square test confirmed `gender` showed no statistically significant association with promotion outcomes ($p = 0.090$), minimizing algorithmic discrimination.
* **Feature Engineering:**
  * `relative_training_score`: Employee score relative to their department's average.
  * `total_training_score`: Combined metric (`avg_training_score * no_of_trainings`).

---

## ⚡ 4. Model Architecture & Threshold Strategy

### 4.1. Preprocessing Pipeline
Built with Scikit-Learn's `Pipeline` + `ColumnTransformer`, incorporating custom feature transformers and 7 isolated feature processing sub-pipelines.

<p align="center">
  <img width="800" alt="preprocessing_pipeline" src="https://github.com/user-attachments/assets/4785306c-03b8-4e46-939a-21679b8ce0da" />
</p>

### 4.2. Why PR-AUC & Precision/Recall?
With an 8.7% positive rate, standard Accuracy and ROC-AUC are heavily inflated. We prioritized **PR-AUC** and **Class-1 Precision** because false positives (premature promotions) carry massive organizational and financial friction costs.

### 4.3. Baseline Model Comparison
Initial screening across 11 models (Traditional & Ensemble) was conducted using 5-fold cross-validation. 
* **Traditional (`class_weight='balanced'`):** Logistic Regression, Decision Tree, Gaussian Naive Bayes, K-Neighbors, SVM.
* **Ensemble (Bagging & Boosting):** Random Forest, Extra Trees, AdaBoost, XGBoost, LightGBM, CatBoost.

**Key Findings:** Boosting models (LightGBM, CatBoost, XGBoost) led on PR-AUC. Two behavioral clusters emerged:
* *Moderate Recall (~55–65%), Low Precision (~20–30%):* LightGBM, CatBoost, XGBoost, Logistic Regression.
* *High Precision (>85%), Low Recall (~30%):* AdaBoost, K-Neighbors.

### 4.4. Fine-tuning & Final Model Selection
Both LightGBM and AdaBoost were tuned with **Optuna**. Performance plateaued around a PR-AUC of ~55%, indicating that the performance ceiling was driven by feature space limitations rather than hyperparameter tuning.

To balance high-confidence decision-making, we constructed a **Soft-Voting Ensemble** combining **LightGBM (60%)** and **AdaBoost (40%)**. 

### 4.5. Final Validation Performance (Threshold = 0.70)
By shifting the threshold to **0.70**, we heavily prioritized **Precision** to align with HR's minimal-risk strategy. 

* **Precision (Class 1):** `0.9055` (The model successfully identified high-potential candidates with a ~90.5% accuracy rate when recommending promotion).
* **Recall (Class 1):** `0.3557` (Intentionally accepts missing some candidates—who will be reviewed at year-end—to guarantee absolute certainty for flagged profiles).
* **F1-Score:** `0.5108`
* **Total Flagged:** 275 high-potential candidates.

---

## 🗂️ 5. Repository Structure

```text
hr-promotion-prediction/
├── app/                        # FastAPI application
│   ├── static/                 # HTML demo frontend
│   ├── feature_engineering.py  # Custom transformer class
│   ├── hr_promotion_pipeline.pkl # Trained model pipeline (.pkl)
│   └── main.py                 # API entrypoint
├── data/
│   └── data.csv                # Raw dataset
├── notebook/
│   └── hr-promotion-predict.ipynb # EDA, training & tuning notebook
├── .gitignore
├── LICENSE
├── README.md
├── requirements.txt            # Runtime dependencies
└── requirements-dev.txt        # Development dependencies
```

## ⚙️ 6. Setup & Installation

**Requirements:** Python >= 3.10

```bash
# Clone the repository
git clone https://github.com/<your-username>/hr-promotion-prediction.git
cd hr-promotion-prediction

# Create and activate a virtual environment
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate

# Install runtime dependencies (required to run the API)
pip install -r requirements.txt
```

> If you want to reproduce the notebook (EDA, model training, hyperparameter tuning with Optuna, LazyPredict benchmarking), also install:
> ```bash
> pip install -r requirements-dev.txt
> ```

## 7. Running the API

The FastAPI app lives inside `app/`, so run uvicorn from the project root using the `app.main:app` path:

```bash
uvicorn app.main:app --reload
```

Once the server starts, it will be available at:

- **Interactive HTML demo:** http://127.0.0.1:8000/
- **Swagger UI (API docs & testing):** http://127.0.0.1:8000/docs

Both let you submit employee data and get a promotion recommendation with its predicted probability.

## 8. Future Enhancements

* **Deep-dive Error Analysis:** Analyze False Positives and False Negatives extracted from the validation set to identify complex edge cases and guide the next iteration of feature engineering.
* **Feature Richness:** Current models plateau around PR-AUC ~55%. Potential directions include interaction features between department and training metrics, tenure-based trends, and peer-relative performance scores.
* **SMOTE Variants:** Experiment with SMOTENC (for mixed categorical/numeric data) as an alternative to class-weight balancing.
* **Explainable AI (XAI):** Add SHAP values so HR can see *why* a specific employee was or wasn't recommended — crucial for decisions affecting real people.
