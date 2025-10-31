# 🧑‍💼 Employee Attrition Prediction

**Converted from:** `used_cars_Price_Predection.ipynb`  
**Conversion Date:** 2025-10-31  
**Author:** [Your Name]  

---

## 📘 Project Overview

This project aims to **predict employee attrition (whether an employee will leave or stay)** using various HR-related features such as job role, satisfaction level, income, and working environment.  
The dataset used is **`WA_Fn-UseC_-HR-Employee-Attrition.csv`**, a popular HR analytics dataset.

The project includes **data cleaning, exploratory data analysis (EDA), preprocessing, model building, class imbalance handling, model optimization, and evaluation**.

---

## 🧰 Tech Stack

| Category | Libraries Used |
|-----------|----------------|
| **Data Handling** | `pandas`, `numpy` |
| **Visualization** | `matplotlib`, `seaborn` |
| **Machine Learning** | `scikit-learn`, `imblearn` |
| **Model Evaluation** | `sklearn.metrics` |
| **Model Selection** | `GridSearchCV`, `KFold` |
| **Utilities** | `warnings` |

---

## 📂 Dataset

**File:** `WA_Fn-UseC_-HR-Employee-Attrition.csv`

| Feature | Description |
|----------|--------------|
| `Age`, `DistanceFromHome`, `MonthlyIncome`, `YearsAtCompany` | Numerical features representing employee characteristics |
| `JobRole`, `Department`, `MaritalStatus`, `BusinessTravel` | Categorical variables describing job context |
| `Attrition` | Target variable (Yes/No) |
| `EmployeeNumber`, `EmployeeCount`, `StandardHours`, `Over18` | Dropped as irrelevant |

---

## 🧹 Data Cleaning

1. **Removed Irrelevant Columns:**
   - Dropped `EmployeeNumber`, `EmployeeCount`, `StandardHours`, `Over18`
2. **Checked for Missing Values:**
   - Dataset contained **no missing values**
3. **Removed Duplicates:**
   - Verified no duplicate rows
4. **Feature Type Validation:**
   - Ensured categorical and numerical features are correctly typed
5. **Outlier Handling:**
   - Applied **IQR method** to cap extreme values in numeric columns

---

## 📊 Exploratory Data Analysis (EDA)

### 🔹 Correlation Heatmap
Identified relationships between numerical variables to detect redundant or weakly correlated features.

### 🔹 Categorical Feature Distribution
Visualized employee distribution across:
- Department
- JobRole
- Gender
- MaritalStatus
- BusinessTravel

### 🔹 Numerical Feature Distribution
Used histograms and boxplots to understand data spread and identify potential outliers.

### 🔹 Key Insights
| Observation | Insight |
|--------------|----------|
| Most employees are not leaving | The dataset is imbalanced toward “No Attrition” |
| Income and years at company correlate with attrition | High-income employees are less likely to leave |
| Work-life balance and satisfaction scores matter | Lower satisfaction relates to higher attrition |

---

## ⚙️ Data Preprocessing

1. **Target Encoding:**
   - Converted `Attrition` → `1` (Yes) and `0` (No)
2. **Categorical Columns:**
   - Identified as either **binary** or **nominal**
3. **ColumnTransformer Pipelines:**
   - **Numeric** → `SimpleImputer (median)` + `StandardScaler`  
   - **Binary** → `SimpleImputer (most_frequent)` + `OrdinalEncoder`  
   - **Nominal** → `SimpleImputer (most_frequent)` + `OneHotEncoder`
4. **Avoided Data Leakage:**
   - Fitted pipeline on training set, transformed on test set

---

## ⚖️ Handling Class Imbalance

Used **SMOTEENN** (combination of SMOTE + Edited Nearest Neighbors) to balance classes.  
After resampling, the target ratio was approximately **55% (No) vs 45% (Yes)**.

---

## 🤖 Model Building

Trained multiple classification models:

| Model | Description |
|--------|--------------|
| Logistic Regression | Baseline linear classifier |
| Decision Tree | Simple tree-based model |
| Random Forest | Ensemble bagging model |
| Gradient Boosting | Boosted ensemble for higher accuracy |
| SVM | Support Vector Machine |
| KNN | K-Nearest Neighbors |
| Naive Bayes | Probabilistic classifier |

---

## 📈 Model Evaluation (Baseline)

| Model | Accuracy | Precision | Recall | F1 Score |
|--------|-----------|------------|---------|-----------|
| Gradient Boosting | 0.89 | 0.84 | 0.83 | **0.84** |
| Random Forest | 0.87 | 0.81 | 0.79 | 0.80 |
| Logistic Regression | 0.85 | 0.80 | 0.76 | 0.78 |
| SVM | 0.82 | 0.78 | 0.73 | 0.75 |

✅ **Gradient Boosting** emerged as the best-performing base model.

---

## 🔍 Cross Validation & Hyperparameter Tuning

**Cross-validation:**  
Used 5-fold `KFold` with metrics:
- Accuracy
- Precision
- Recall
- F1

**Hyperparameter Optimization:**  
Performed `GridSearchCV` on `GradientBoostingClassifier`:

```python
param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.05, 0.1, 0.2],
    'max_depth': [2, 3, 4]
```
## Key Takeaways

Gradient Boosting with tuned hyperparameters provides strong generalization.

Handling imbalance using SMOTEENN improved minority class performance.

EDA revealed key attrition drivers — satisfaction, years at company, and income.

##  Future Enhancements

Deploy as a Streamlit or Flask web app

Integrate SHAP/LIME for model explainability

Add feature importance dashboard

Monitor model drift on real-time HR data

## 👨‍💻 Author

Developed by: [Rahma Saber Abbas]
📅 Date: October 18, 2025
📘 File: Employee_Attrition_Prediction.py
    'subsample': [0.8, 1.0]
}
