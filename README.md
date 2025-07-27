# 🛡️ Advanced Fraud Detection for E-commerce and Banking

## 📋 Project Overview
This repository contains a comprehensive machine learning pipeline for Adey Innovations Inc., designed to enhance the detection of fraudulent transactions in both e-commerce and banking sectors. The project tackles the critical business need to minimize financial losses from fraud while ensuring a seamless and positive user experience by reducing false positives.

By leveraging advanced data analysis, feature engineering, and machine learning, this project delivers a robust solution that can identify complex fraud patterns from raw transaction data.


## 🎯 Key Features

- ✅ **Modular Data Pipeline** - Reusable preprocessing pipeline for cleaning and preparing transaction data
- ✅ **In-depth EDA** - Comprehensive exploratory data analysis to uncover hidden fraud patterns
- ✅ **Strategic Feature Engineering** - Create powerful, predictive signals from temporal and user-behavior data
- ✅ **Class Imbalance Handling** - Clear plan for handling severe class imbalance in fraud detection
- ✅ **Well-documented Structure** - Support for model building, evaluation, and interpretation

> 💡 This project serves as a foundational step in building an intelligent system that not only detects fraud but also provides insights into the drivers of fraudulent behavior.

---

## 📁 Project Structure
fraud_detection_project/
│
├── 📂 data/
│   ├── 📂 raw/
│   │   ├── 📄 Fraud_Data.csv
│   │   ├── 📄 IpAddress_to_Country.csv
│   │   └── 📄 creditcard.csv
│   └── 📂 processed/
│       └── 📄 processed_fraud_data.csv
│
├── 📂 notebooks/
│   ├── 📓 1_Data_Processing_and_EDA.ipynb
│   └── 📓 2_Model_Training_and_Evaluation.ipynb
│
├── 📂 scripts/
│   ├── 🐍 __init__.py
│   ├── ⚙️ config.py
│   ├── 🔧 task1_pipeline.py
│   └── 🧠 task2_modeling_pipeline.py
│
├── 📂 outputs/
│   └── �️ images/
│
└── 📖 README.md

## 🚀 Setup Instructions

### 1. 📥 Clone the Repository

```bash
git clone https://github.com/matos-coder/Improved-detection-of-fraud-cases-for-e-commerce-and-bank-transactions
cd 'Improved-detection-of-fraud-cases-for-e-commerce-and-bank-transactions'
```

### 2. 🐍 Create a Virtual Environment

It is **highly recommended** to use a virtual environment to manage project dependencies.

#### Windows
```powershell
python -m venv venv
venv\Scripts\activate
```

#### macOS/Linux
```bash
python3 -m venv venv
source venv/bin/activate
```

pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn lightgbm

Or install from a requirements.txt file:

pip install -r requirements.txt

## 🎯 Task 1: Data Analysis and Preprocessing
Objective: To thoroughly clean, analyze, and enrich the e-commerce transaction dataset (Fraud_Data.csv) to prepare it for machine learning model training.

📍 Location: The entire workflow is orchestrated in notebooks/1_Data_Processing_and_EDA.ipynb, which calls modular functions from scripts/task1_pipeline.py.

### 🧹 Data Cleaning and Preprocessing
Goal: To ensure data quality and consistency. This is a critical first step as model performance is highly dependent on clean data.


#### Implementation:

| Step | Description | Justification |
|------|-------------|---------------|
| 🔄 **Handled Duplicates** | Removed identical rows | Prevents data leakage and model bias |
| 📅 **Corrected Data Types** | Converted `signup_time` and `purchase_time` to datetime | Essential for time-based calculations |
| 🌐 **Handled IP Addresses** | Converted `ip_address` from float to integer | Facilitates merging with geolocation dataset |

### 📊 Exploratory Data Analysis (EDA)

**Goal:** To understand the underlying patterns in the data, especially the differences between fraudulent and legitimate transactions.

#### 🔍 Analysis Components:

- **📈 Class Imbalance Analysis** - Visualized the severe imbalance between fraud and non-fraud classes
- **📊 Univariate Analysis** - Analyzed distributions of key numerical features
- **📈 Bivariate Analysis** - Created boxplots comparing feature distributions across classes

#### 💡 Key Insights from EDA:

| Insight | Description | Impact |
|---------|-------------|--------|
| ⚠️ **Severe Class Imbalance** | Fraudulent transactions: ~9% of dataset | Accuracy is poor metric; requires SMOTE/AUC-PR |
| ⏰ **Time-Based Patterns** | Fraud occurs faster after signup | Strong indicator of fraudulent behavior |
| 🌍 **Geographic Hotspots** | Fraud not evenly distributed globally | Certain countries show higher fraud rates |

### 🔧 Feature Engineering

**Goal:** To create new, high-signal features that explicitly capture behaviors associated with fraud.

#### 🛠️ Engineered Features:

| Feature | Description | Fraud Signal |
|---------|-------------|--------------|
| ⏱️ `time_since_signup_seconds` | Duration between signup and first purchase | Short duration = fraud risk |
| 🕐 `purchase_hour_of_day` | Hour when purchase was made | Overnight purchases = higher risk |
| 📅 `purchase_day_of_week` | Day of week for purchase | Pattern detection |
| 📱 `device_id_count` | Transactions per device | High frequency = automation |
| 👤 `user_id_count` | Transactions per user | Behavioral analysis |
| 🌍 `country` | Geographic location from IP | Regional fraud patterns |

> 💡 **Justification:** These engineered features are designed based on well-known fraud typologies, such as fraudsters creating accounts and using them immediately.

## 🎯 Task 2: Model Building and Training
Objective: To build, train, and evaluate machine learning models to accurately detect fraudulent transactions on both the e-commerce and credit card datasets.

📍 Location: The workflow is orchestrated in notebooks/2_Model_Training_and_Evaluation.ipynb, with logic encapsulated in scripts/task2_modeling_pipeline.py.

### 🧠 Model Selection
Goal: To compare a simple, interpretable baseline model against a powerful, complex ensemble model to find the best solution for the business problem.


| Model | Type | Justification |
|---|---|---|
| 📈 Logistic Regression | Baseline | A simple, fast, and highly interpretable model. Excellent for establishing a performance benchmark. |
| 🌳 LightGBM | Powerful Ensemble | A gradient boosting framework known for high performance and efficiency. It can capture complex, non-linear patterns that simpler models miss. |

### ⚙️ Data Preparation and Preprocessing
Goal: To prepare the data for the machine learning models by scaling numerical features and encoding categorical ones.


📋 Strategy:

| Component | Approach | Reasoning |
|---|---|---|
| ✂️ Train-Test Split | 80/20 split using stratify | Ensures the same percentage of fraud cases in both training and testing sets, which is crucial for imbalanced data. |
| ⚖️ Numerical Scaling | StandardScaler | Scales numerical features to have a mean of 0 and standard deviation of 1, preventing features with large ranges from dominating the model. |
| 🏷️ Categorical Encoding | OneHotEncoder | Converts categorical text data (e.g., 'Chrome', 'Ads') into a numerical format that the models can understand. |

### ⚖️ Handling Class Imbalance (SMOTE)
Goal: To address the severe class imbalance identified in Task 1 to ensure the model learns to identify the minority (fraud) class effectively.


| Component | Approach | Reasoning |
|---|---|---|
| 🎯 Chosen Technique | SMOTE (Synthetic Minority Over-sampling) | Creates new, synthetic examples of the minority class in the training data only. |
| 💡 Justification |  | By balancing the class distribution, SMOTE forces the model to learn the patterns of fraudulent transactions instead of just ignoring them. This is critical for building a useful fraud detection system. |

## 📈 Model Training and Evaluation
Goal: To train the selected models and rigorously evaluate their performance using metrics appropriate for imbalanced classification.


📊 Evaluation Metrics:

| Metric | What It Measures | Why It's Important for Fraud |
|---|---|---|
| 🎯 F1-Score | The harmonic mean of Precision and Recall. | Provides a single score that balances the cost of false positives (annoying customers) and false negatives (missing fraud). |
| 📈 AUC-PR | Area Under the Precision-Recall Curve. | The best metric for imbalanced data. It evaluates the model's ability to distinguish between classes across all possible decision thresholds. A higher score means a more robust model. |
| 🔢 Confusion Matrix | A table showing True Positives, True Negatives, False Positives, and False Negatives. | Gives a detailed breakdown of the model's prediction accuracy for both classes. |

## 🏆 Results and Justification
The performance of both models was evaluated on both the e-commerce and credit card datasets.


📊 Final Performance Summary:

| Dataset | Model | F1 Score | AUC-PR |
|---|---|---|---|
| E-commerce | Logistic Regression | 0.6133 | 0.7634 |
| E-commerce | LightGBM | 0.8407 | 0.8872 |
| Credit Card | Logistic Regression | 0.1171 | 0.7637 |
| Credit Card | LightGBM | 0.8614 | 0.8491 |

💡 Conclusion: LightGBM is the Best Model
Justification:
Across both datasets, the LightGBM model demonstrates overwhelmingly superior performance.

Higher F1-Score and AUC-PR: The significantly higher scores indicate that LightGBM is far better at correctly identifying fraudulent transactions while maintaining a low false positive rate.

Business Value: For Adey Innovations Inc., this translates directly to more fraud caught and fewer legitimate customers impacted, maximizing revenue protection and customer satisfaction.

While Logistic Regression provides a useful baseline, its performance is insufficient for a production-level fraud detection system. The advanced capabilities of LightGBM are essential for this business use case.

## 🚀 How to Run the Pipeline
📓 Task 1 - Data Processing and EDA
Execute the complete pipeline by running the Jupyter Notebook:


```bash
# Navigate to the notebook and run the cells
jupyter notebook notebooks/1_Data_Processing_and_EDA.ipynb
```

🧠 Task 2 - Model Building and Training
Execute the modeling pipeline by running the second Jupyter Notebook:

```bash
# Navigate to the notebook and run the cells
jupyter notebook notebooks/2_Model_Training_and_Evaluation.ipynb
```

This will:

- 📥 Load the processed e-commerce data and raw credit card data.
- ⚙️ Preprocess, split, and balance the data.
- 🧠 Train and evaluate both Logistic Regression and LightGBM models.
- 📊 Display performance metrics and confusion matrices for each model.
- 🏆 Provide a final summary and justification for the best model.

## 🤝 Contributing

🍴 Fork the repository
🌿 Create a feature branch (`git checkout -b feature/amazing-feature`)
💾 Commit your changes (`git commit -m 'Add some amazing feature'`)
📤 Push to the branch (`git push origin feature/amazing-feature`)
🔄 Open a Pull Request

## 📄 License
This project is licensed under the MIT License - see the LICENSE file for details.

## 📞 Contact
Adey Innovations Inc.

📧 Email: contact@adeyinnovations.com

🌐 Website: www.adeyinnovations.com

💼 LinkedIn: Adey Innovations

