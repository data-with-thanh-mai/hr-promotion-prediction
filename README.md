# HR Promotion Prediction API
## 1. Trải nghiệm Giao diện UI (Interactive API Demo)
## 2. Business Context & Proposed Solution

**The Problem:**
A large Multinational Corporation (MNC) has 9 broad verticals across the organization. One of their primary challenges is identifying the right people for promotion (for manager positions and below) and preparing them in time. 

Currently, final promotions are only announced after annual evaluations. This rigid schedule forces capable employees to wait months before transitioning into new roles, causing operational delays and wasting talent. 

**The Solution:**
To address this, this project delivers an end-to-end Machine Learning solution to evaluate and identify eligible candidates at specific checkpoints throughout the year. 

Specifically, we built a robust classification pipeline that handles highly imbalanced HR data. The final model is deployed as a real-time **REST API using FastAPI**. This allows the HR department to simply input an employee's current metrics (training score, length of service, etc.) and instantly receive a data-driven recommendation (*"Recommend for Promotion"* or *"Keep under observation"*), expediting the promotion cycle without waiting for the year-end review.
Multiple attributes have been provided around Employee's past and current performance along with demographics.
## 3. Data & Preprocessing Overview
The dataset contains 54,808 samples and 13 features, encompassing past/current performance and demographics.

Data Dictionary:

employee_id: Unique ID for the employee

department: Department of the employee

region: Region of employment (unordered)

education: Education Level

gender: Gender of the Employee

recruitment_channel: Channel of recruitment

no_of_trainings: Number of other trainings completed in the previous year

age: Age of the Employee

previous_year_rating: Employee Rating for the previous year

length_of_service: Length of service in years

awards_won?: 1 if awards won during the previous year, else 0

avg_training_score: Average score in current training evaluations

is_promoted: (Target) 1 if recommended for promotion, else 0

Data Cleaning & Preprocessing Strategy:

Imbalanced Data Handling: Class 1 (Promoted) accounts for only ~8.69% of the dataset. Handled using Class Weights inside the model architecture. SMOTE was intentionally avoided because generating synthetic data points for categorical variables introduces unrealistic data noise.  

Missing Value Imputation:  

previous_year_rating (~7.63% missing): Imputed with 0. Business logic dictates that these missing values belong to newly hired employees who do not have a performance rating from the previous year.  

education (~4.38% missing): Imputed with the most_frequent strategy to avoid dropping rows.

Feature Selection & Bias Mitigation (AI Fairness):

Dropped employee_id (no predictive value).

Dropped gender(hypothesis testing chi-squre5	gender	9.008336e-02	No)  :  and age : Removing these demographic features not only reduces noise but also ensures the model makes promotion recommendations based strictly on merit and performance, mitigating age and gender bias.
<img width="567" height="334" alt="image" src="https://github.com/user-attachments/assets/36a7a717-3f24-47a4-bd49-c42f542d8261" />


Categorical Encoding: Applied Ordinal Encoding for education, as higher education levels historically correlate with higher promotion readiness.

Feature Engineering: Created two new impactful features:

relative_training_score: Ratio of the employee's score to the average score of their specific department.

total_training_score: Calculated as avg_training_score * no_of_trainings to capture the absolute training volume and quality.
   
## 4. Kiến trúc Mô hình & Ngưỡng quyết định (Model Architecture & Threshold Strategy)
preprcessing pipeline 
**Preprocessing:** Custom feature engineering step + ColumnTransformer 
handling 7 column groups (drop, numeric, ordinal, one-hot, target encoding, 
constant imputation, passthrough). See diagram below.
<img width="2720" height="1600" alt="preprocessing_pipeline_column_transformer_blue" src="https://github.com/user-attachments/assets/4785306c-03b8-4e46-939a-21679b8ce0da" />


ở phần này sử dụng model của scikitlenr 
### Why PR-AUC and class-1 metrics (not Accuracy/ROC-AUC)

The dataset is highly imbalanced — only ~8.7% of employees are promoted 
(`is_promoted = 1`). This makes common metrics misleading:

**Accuracy is misleading.** A trivial model that always predicts "not 
promoted" already scores ~91.3% accuracy — while being completely useless 
for the actual business goal of finding promotion candidates. In our 
benchmark, AdaBoost has the *highest* accuracy (0.94) but recalls only 
32.6% of actual promotions — missing 2 out of every 3 employees who 
deserved one.

**ROC-AUC is overly optimistic.** It's computed from the true-negative 
rate, and with 91% of the data being class 0, true negatives are abundant 
— keeping the false-positive rate low almost by default. All models in 
our benchmark score a comfortable 0.76–0.81 ROC-AUC, but their PR-AUC 
(which ignores true negatives entirely) ranges only 0.22–0.55 — the real 
spread in performance.

**Why we track Precision/Recall for class 1 specifically:** the two 
error types have very different business costs:
- **False positive** (wrongly recommending promotion) → wasted training 
  budget, a role given to the wrong person, erodes trust in the system.
- **False negative** (missing a deserving employee) → a high performer 
  goes unrecognized, risking attrition.

Reporting Precision/Recall for class 0 and class 1 together (as most 
libraries do by default) hides this — class 0's numbers are always high 
simply because it's the majority class. **PR-AUC and class-1 
Precision/Recall are the metrics that actually reflect model quality 
for this problem.**
baseline model 
dùng lazy predict thử 1 lượng các mô hình defaul -> nhưng mà thông số chủ yếu ở đây pr-auc cho thấy sự tương qun của precision và recall của class 1, và các chỉ số preciiosn-recall calls1 -> lazy predict ở ddaya ko có tác dụng lắm do imblance data
thử 2 loại mô hình : 
traditon :  models = {
        "Logistic Regression": LogisticRegression(class_weight='balanced', max_iter=1000, random_state=random_state),
        "Decision Tree": DecisionTreeClassifier(class_weight='balanced', random_state=random_state),
        "Gaussian Naive Bayes": GaussianNB(),
        "K-Neighbors": KNeighborsClassifier(),
        "SVM": SVC(class_weight='balanced', probability=True, random_state=random_state)
    }
và embamel :(2 nhóm bagging và booging gì đó) 
odels = {
        "Random Forest": RandomForestClassifier(class_weight='balanced', random_state=random_state, n_jobs=-1),
        "Extra Trees": ExtraTreesClassifier(class_weight='balanced', random_state=random_state, n_jobs=-1),
        "AdaBoost": AdaBoostClassifier(random_state=random_state),
        "XGBoost": XGBClassifier(scale_pos_weight=scale_weight, random_state=random_state, eval_metric='logloss', n_jobs=-1),
        "LightGBM": LGBMClassifier(scale_pos_weight=scale_weight, random_state=random_state, verbose=-1, n_jobs=-1),
        "CatBoost": CatBoostClassifier(auto_class_weights='Balanced', random_state=random_state, verbose=0)
    }
   <img width="570" height="167" alt="image" src="https://github.com/user-attachments/assets/5951f2af-2e67-46be-9790-3b70f515f725" />
   <img width="573" height="164" alt="image" src="https://github.com/user-attachments/assets/407c3db2-accd-43f3-bcdd-70e3f6dc63bd" />
Kết quả cho thấy : phần lớn các mô hình them nhóm emsble chỉ performance nhỉn hơn 1 chút. tuy nhiên đnag khá thấp, chênh lẹnh pr-auc ko nhiều
có 2 xu hướng chính:
- recall trung bìnb ( cơ 60&) và precions cỡ 30%
- preciosn cao (hơn 90*) và recall tháp 30&
Mình chọn 2 mô hình đại điẹn cho 2 nhóm xu hứng này : lightxgb ( hơi ba phải) và adaboost ( có xu hướng phạt nặng dự đoán sai)

fitune bằng optuna 
## 5. Cấu trúc Thư mục Dự án (Repository Structure

## 6. Yêu cầu Hệ thống & Cài đặt (Setup & Installation)

## 7. Hướng dẫn Khởi chạy Server (Running the API)
## 8. Hướng dẫn Tương tác API (API Usage & Examples)
## 9. Lộ trình Phát triển Tiếp theo (Future Enhancements)
