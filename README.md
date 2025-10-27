# 🏦 Mortgage Loan Approval Prediction

## 📘 Overview

This project predicts **mortgage loan approvals** using real-world application data under the **Home Mortgage Disclosure Act (HMDA)**.
The goal is to help financial institutions automatically evaluate mortgage applications and estimate approval likelihood based on applicant demographics, loan characteristics, and property details.

---

## 🎯 Objective

Develop a **machine learning model** that predicts whether a mortgage loan will be **approved or denied**, based on key financial and personal attributes such as:

* Loan amount and purpose
* Applicant income and age
* Property value and location
* Credit type and loan terms

---

## 📂 Dataset

**Source:** HMDA Public Loan Application Register
**File:** `raw_dataset.csv`
**Size:** ~1.4 million rows × 99 columns

### Key Features

| Feature                                            | Description                           |
| -------------------------------------------------- | ------------------------------------- |
| `loan_amount`                                      | Total amount requested by applicant   |
| `income`                                           | Applicant’s annual income             |
| `property_value`                                   | Estimated value of the property       |
| `loan_to_value_ratio`                              | Loan amount divided by property value |
| `derived_race`, `derived_ethnicity`, `derived_sex` | Demographic information               |
| `applicant_credit_score_type`                      | Applicant’s credit score category     |
| `action_taken`                                     | Loan outcome (approved/denied)        |

---

## 🧹 Data Processing Pipeline

### 1️⃣ Data Cleaning

* Removed irrelevant and duplicate columns (e.g., detailed ethnicity/race variants).
* Handled missing values using:

  * Mean-based imputation for income.
  * Linear Regression for `interest_rate` and `property_value`.
  * Logical imputation for `conforming_loan_limit` based on year and loan amount.

### 2️⃣ Feature Engineering

* Encoded categorical data using `OrdinalEncoder`.
* Created mappings for banks (`LEI → Bank Name`):

  * Wells Fargo Bank
  * JPMorgan Chase Bank
  * Bank of America
  * U.S. Bank
  * Citibank
* Converted string age ranges (`25–34`, `45–54`, etc.) into numeric ordinal categories.
* Simplified loan decision variable:
  `action_taken = 1` (Approved) and `0` (Denied).

### 3️⃣ Cleaned Data Outputs

| Stage        | File Name             | Description                  |
| ------------ | --------------------- | ---------------------------- |
| Raw Data     | `raw_dataset.csv`     | Original HMDA dataset        |
| Cleaned Data | `clean_dataset.csv`   | Processed and imputed data   |
| Feature Data | `feature_dataset.csv` | Encoded dataset ready for ML |
| Model Output | `output_df.csv`       | Final prediction results     |

---

## 📊 Data Visualization

Visual insights were created using **Matplotlib** and **Seaborn**:

* **Applications by Bank** — total number of loans processed by top banks.
* **Approval Rate by Gender** — shows gender-based acceptance trends.
* **Approval Rate by Age Group** — visualizes how applicant age influences loan decisions.

---

## 🤖 Model Development

### Models Used

| Model               | Accuracy  | Notes                               |
| ------------------- | --------- | ----------------------------------- |
| Logistic Regression | ~79.7%    | Baseline performance                |
| Random Forest       | ~96.4%    | Excellent recall & interpretability |
| XGBoost             | **96.5%** | Best performer overall              |

### Evaluation Metrics

* Accuracy
* Precision
* Recall
* F1-Score
* Confusion Matrix

---

## 🔬 Model Pipeline

1. **Train-Test Split (70/30)**
2. **Standardization & Encoding**
3. **Model Training (XGBoost Classifier)**
4. **Local County Similarity (KNN)** for localized approval rate analysis
5. **Institutional-Level Predictions** across top banks:

   * Wells Fargo
   * JPMorgan Chase
   * Bank of America
   * U.S. Bank
   * Citibank

---

## 🧩 Example Results

Each application is scored with overall and institution-specific approval probabilities.

| Application # | Main Prediction | Probability | County Acceptance | Accuracy | Top Bank       | Bank Prob | Acceptance Rate |
| ------------- | --------------- | ----------- | ----------------- | -------- | -------------- | --------- | --------------- |
| 1             | ✅ Approved      | 1.000       | 1.0               | 0.943    | JPMorgan Chase | 1.000     | 1.0             |
| 2             | ✅ Approved      | 1.000       | 0.8               | 0.943    | Wells Fargo    | 1.000     | 0.8             |
| 3             | ❌ Denied        | 0.022       | 0.6               | 0.943    | Citibank       | 0.022     | 0.8             |

All detailed predictions (29 columns per record) are saved in:
📄 **`output_df.csv`**

---

## 🧾 Showcase (in Jupyter)

To display the output interactively:

```python
import pandas as pd

output_df = pd.read_csv("database/output_df.csv")
output_df.insert(0, "Number of Application", range(1, len(output_df)+1))

styled_output = (
    output_df
    .style
    .set_caption("🏦 Full Mortgage Loan Approval Prediction Results")
    .background_gradient(subset=["Main Probability", "Model Accuracy"], cmap="Greens")
    .background_gradient(subset=["County Acceptance Rate (Similar Applications)"], cmap="Blues")
    .bar(subset=[col for col in output_df.columns if "Approval Probability" in col], color="#FFD700")
    .format({col: "{:.3f}" for col in output_df.columns if output_df[col].dtype != 'object'})
    .hide(axis="index")
)
styled_output
```

---

## ⚙️ Technologies Used

| Category      | Tools / Libraries                                                     |
| ------------- | --------------------------------------------------------------------- |
| Language      | Python                                                                |
| IDE           | Jupyter Notebook                                                      |
| Libraries     | `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `xgboost` |
| ML Models     | Logistic Regression, Random Forest, XGBoost                           |
| Visualization | Seaborn, Matplotlib                                                   |
| Output Format | CSV, Jupyter HTML Styling                                             |

---

## 🚀 How to Run

1. **Install dependencies**

   ```bash
   pip install pandas numpy seaborn matplotlib scikit-learn xgboost jinja2
   ```

2. **Run the notebook**

   ```bash
   jupyter notebook mortgage_loan_approval_dsa.ipynb
   ```

3. **Ensure directory structure**

   ```
   mortgage_loan_approval_dsa/
   ├── database/
   │   ├── raw_dataset.csv
   │   ├── clean_dataset.csv
   │   ├── feature_dataset.csv
   │   └── output_df.csv
   └── mortgage_loan_approval_dsa.ipynb
   ```

---

## 📚 Future Enhancements

* Integrate SHAP or LIME for model explainability.
* Build a **Streamlit dashboard** for live loan approval simulation.
* Add more financial & geographic data sources for improved generalization.
* Deploy as an API for enterprise use.

---

## 👤 **Author**

**Anand Vekariya**

📧 **Email:** [[anand.d.vekariya@gmail.com](mailto:anand.d.vekariya@gmail.com)]

💼 **LinkedIn:** [Anand Vekariya](https://www.linkedin.com/in/anand-vekariya/)

<!-- 🌐 **Portfolio:** [https://anandvekariya.netlify.app](https://anandvekariya.netlify.app) -->

---

