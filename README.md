<<<<<<< HEAD
# HR Promotion Prediction API

## 1. Interactive API Demo

<img width="1284" height="720" alt="demo-ezgif com-video-to-gif-converter" src="https://github.com/user-attachments/assets/be3e4e60-90b4-48d1-8659-967a1d1d9c13" />


## 2. Business Context & Proposed Solution

**The Problem:**
A large Multinational Corporation (MNC) has 9 broad verticals across the organization. One of their primary challenges is identifying the right people for promotion (for manager positions and below) and preparing them in time.

Currently, final promotions are only announced after annual evaluations. This rigid schedule forces capable employees to wait months before transitioning into new roles, causing operational delays and wasting talent.

**The Solution:**
To address this, this project delivers an end-to-end Machine Learning solution to evaluate and identify eligible candidates at specific checkpoints throughout the year.

Specifically, we built a robust classification pipeline that handles highly imbalanced HR data. The final model is deployed as a real-time **REST API using FastAPI**. This allows the HR department to simply input an employee's current metrics (training score, length of service, etc.) and instantly receive a data-driven recommendation (*"Recommend for Promotion"* or *"Keep under observation"*), expediting the promotion cycle without waiting for the year-end review.

Multiple attributes have been provided around each employee's past and current performance along with demographics.

## 3. Data & Preprocessing Overview
**Data Source:** The dataset used to train and evaluate this model is publicly available via [Kaggle HR Analytics Dataset](https://www.kaggle.com/datasets/arashnic/hr-ana).
The dataset contains 54,808 samples and 13 features, encompassing past/current performance and demographics.

**Data Dictionary:**

| Field | Description |
|---|---|
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

**Data Cleaning & Preprocessing Strategy:**

- **Imbalanced data handling:** class 1 (promoted) accounts for only ~8.69% of the dataset. Handled using class weights inside the model architecture. SMOTE was intentionally avoided, as generating synthetic data points for categorical variables introduces unrealistic noise.
- **Missing value imputation:**
  - `previous_year_rating` (~7.63% missing): imputed with 0 — business logic dictates that these missing values belong to newly hired employees who don't yet have a previous-year rating.
  - `education` (~4.38% missing): imputed with the most-frequent strategy to avoid dropping rows.

**Feature Selection & Bias Mitigation (AI Fairness):**

- Dropped `employee_id` (no predictive value).
- Dropped `gender` and `age`. A chi-square test confirmed `gender` has no statistically significant association with `is_promoted` (p = 0.090). Removing these demographic features reduces noise and ensures the model bases promotion recommendations strictly on merit and performance, mitigating age and gender bias.

- **Categorical encoding:** applied ordinal encoding for `education`, as higher education levels historically correlate with higher promotion readiness.
- **Feature engineering:** created two new features:
  - `relative_training_score`: ratio of the employee's score to the average score of their department.
  - `total_training_score`: `avg_training_score * no_of_trainings`, capturing absolute training volume and quality.

## 4. Model Architecture & Threshold Strategy

### 4.1. Preprocessing Pipeline

Built with scikit-learn's `Pipeline` + `ColumnTransformer`, consisting of a custom feature engineering step followed by 7 separate column groups (drop, numeric, ordinal, one-hot, target encoding, constant imputation, passthrough).

<img width="2720" height="1600" alt="preprocessing_pipeline" src="https://github.com/user-attachments/assets/4785306c-03b8-4e46-939a-21679b8ce0da" />

### 4.2. Why PR-AUC and Class-1 Precision/Recall

The dataset is highly imbalanced (~8.7% promoted), so Accuracy and ROC-AUC are misleading — a model that always predicts "not promoted" still scores ~91% accuracy. We use **PR-AUC** and **Precision/Recall of class 1** instead, since false positives (wrongly recommending promotion) and false negatives (missing deserving talent) both carry real business cost that class-0 metrics would hide.

### 4.3. Baseline Model Comparison

LazyPredict gave a quick first look, but wasn't suitable for deeper evaluation — it doesn't support custom cross-validation or imbalance handling, so it was used for reference only.

Formal benchmarking used two model groups with 5-fold cross-validation:

**Traditional** (`class_weight='balanced'`): Logistic Regression, Decision Tree, Gaussian Naive Bayes, K-Neighbors, SVM

**Ensemble** (bagging + boosting): Random Forest, Extra Trees, AdaBoost, XGBoost, LightGBM, CatBoost

<img width="570" height="167" alt="ensemble results" src="https://github.com/user-attachments/assets/5951f2af-2e67-46be-9790-3b70f515f725" />
<img width="573" height="164" alt="traditional results" src="https://github.com/user-attachments/assets/407c3db2-accd-43f3-bcdd-70e3f6dc63bd" />

**Results:** the boosting models (LightGBM, CatBoost, XGBoost) led on PR-AUC, but only slightly ahead of SVM — not every ensemble model outperformed the traditional ones. Two clear tendencies emerged:

- **Moderate recall (~55–65%), low precision (~20–30%)**: LightGBM, CatBoost, XGBoost, Logistic Regression.
- **High precision (>85%), low recall (~30%)**: AdaBoost, K-Neighbors.

We picked one model to represent each tendency for further tuning: **LightGBM** (balanced precision-recall trade-off) and **AdaBoost** (conservative, favors certainty over coverage).

### 4.4. Fine-tuning & Final Model Selection

Both models were tuned with **Optuna**. Performance improved slightly but not dramatically — most results plateaued around a PR-AUC of ~55%, suggesting the current bottleneck is feature engineering rather than hyperparameter tuning (a direction for future improvement).

<img width="352" height="245" alt="tuning result 1" src="https://github.com/user-attachments/assets/c1b25da2-27e9-433a-b77f-b5928ccd984e" />
<img width="350" height="259" alt="tuning result 2" src="https://github.com/user-attachments/assets/c2115ef2-1ac7-4618-8b16-8d3eede3f252" />
<img width="385" height="311" alt="tuning result 3" src="https://github.com/user-attachments/assets/b50ebdc0-d1bb-4a53-a7bd-723c4de6102c" />

That said, a soft-voting ensemble of LightGBM + AdaBoost, and AdaBoost alone, handle one scenario particularly well: picking a single candidate with **high confidence**.

Since HR prioritizes **precision** (a wrong promotion recommendation is costlier than a missed one at this checkpoint — unflagged candidates can still be reviewed at the next cycle), the ensemble weights and decision threshold are being re-tuned using the validation-set precision-recall curve to find a high-precision operating point that doesn't sacrifice recall unnecessarily.

> **Results pending final re-evaluation.** The final ensemble weights, threshold, and test-set performance table (Precision/Recall/F1/PR-AUC for the chosen operating point) will be added here once validated.

## 5. Repository Structure

```
hr-promotion-prediction/
├── app/                            # FastAPI application
│   ├── static/                     # HTML demo frontend
│   ├── feature_engineering.py      # Custom transformer class
│   ├── hr_promotion_pipeline.pkl   # Trained model pipeline
│   └── main.py                     # API entrypoint
├── data/
│   └── data.csv                    # Dataset
├── notebook/
│   └── hr-promotion-predict.ipynb  # Full EDA, training & tuning notebook
├── .gitignore
├── LICENSE
├── README.md
├── requirements.txt                # Runtime dependencies
└── requirements-dev.txt            # Dev/notebook dependencies (Optuna, LazyPredict...)
```

## 6. Setup & Installation

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

## 8. API Usage & Examples

**Endpoint:** `POST /predict`

**Example request (curl):**

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "employee_id": 1,
    "department": "Sales & Marketing",
    "region": "region_7",
    "education": "Bachelor'"'"'s",
    "gender": "Male",
    "recruitment_channel": "sourcing",
    "no_of_trainings": 1,
    "age": 30,
    "previous_year_rating": 4,
    "length_of_service": 5,
    "awards_won?": 0,
    "avg_training_score": 78
  }'
```

**Example response:**

```json
{
  "status": "success",
  "prediction": {
    "probability": 0.42,
    "strict_threshold": 0.7,
    "is_promoted": false,
    "action": "Keep under observation"
  }
}
```

An interactive version is available at `/docs` (Swagger UI) or the HTML demo at `/`.

## 9. Future Enhancements

**Model Performance:**
- Explore additional feature engineering — current models plateau around PR-AUC ~55%, suggesting the bottleneck is feature richness rather than hyperparameter tuning. Potential directions: interaction features between department and training metrics, tenure-based trends, peer-relative performance scores.
- Experiment with SMOTE variants (e.g. SMOTENC for mixed categorical/numeric data) as an alternative to class-weight balancing.
- Add model explainability (SHAP values) so HR can see *why* a specific employee was or wasn't recommended — important for a decision that affects real people.

**Engineering & Deployment:**
- Containerize the app with Docker for consistent deployment across environments (avoids the Python/dependency version issues encountered during local setup).
- Add a `/batch-predict` endpoint to score multiple employees at once via CSV upload.
- Add request logging and basic monitoring (e.g. track prediction distribution over time to catch data/model drift).
- Add automated tests (unit tests for the preprocessing pipeline, integration tests for the API endpoints) and a CI pipeline.

**Product:**
- Add authentication to the API before any production use, since it handles employee performance data.
- Build a lightweight internal dashboard (e.g. Streamlit) for HR to review predictions in bulk, rather than one employee at a time via the API form.
- Add a feedback loop: track actual promotion outcomes vs. predictions to periodically retrain and validate the model.

