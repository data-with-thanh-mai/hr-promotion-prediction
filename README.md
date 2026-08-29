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
ở phần này sử dụng model của scikitlenr 
baseline model 
fitune bằng optuna 
## 5. Cấu trúc Thư mục Dự án (Repository Structure

## 6. Yêu cầu Hệ thống & Cài đặt (Setup & Installation)

## 7. Hướng dẫn Khởi chạy Server (Running the API)
## 8. Hướng dẫn Tương tác API (API Usage & Examples)
## 9. Lộ trình Phát triển Tiếp theo (Future Enhancements)
