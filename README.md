# 🛡️ Advanced Fraud Detection for E-commerce and Banking

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![Machine Learning](https://img.shields.io/badge/ML-Fraud%20Detection-green.svg)
![Status](https://img.shields.io/badge/status-active-success.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)


## 📋 Project Overview

This repository contains a **comprehensive machine learning pipeline** for **Adey Innovations Inc.**, designed to enhance the detection of fraudulent transactions in both **e-commerce** and **banking** sectors. The project tackles the critical business need to minimize financial losses from fraud while ensuring a seamless and positive user experience by reducing false positives.

By leveraging **advanced data analysis**, **feature engineering**, and **machine learning**, this project delivers a robust solution that can identify complex fraud patterns from raw transaction data.

### 🎯 Key Features

- ✅ **Modular Data Pipeline** - Reusable preprocessing pipeline for cleaning and preparing transaction data
- ✅ **In-depth EDA** - Comprehensive exploratory data analysis to uncover hidden fraud patterns
- ✅ **Strategic Feature Engineering** - Create powerful, predictive signals from temporal and user-behavior data
- ✅ **Class Imbalance Handling** - Clear plan for handling severe class imbalance in fraud detection
- ✅ **Well-documented Structure** - Support for model building, evaluation, and interpretation

> 💡 This project serves as a foundational step in building an intelligent system that not only detects fraud but also provides insights into the drivers of fraudulent behavior.

---

## 📁 Project Structure

```
fraud_detection_project/
│
├── 📂 data/
│   ├── 📂 raw/
│   │   ├── 📄 Fraud_Data.csv
│   │   ├── 📄 IpAddress_to_Country.csv
│   │   └── 📄 creditcard.csv
│   └── 📂 processed/
│       ├── 📂 images/
│       └── 📄 processed_fraud_data.csv
│
├── 📂 notebooks/
│   └── 📓 1_Data_Processing_and_EDA.ipynb
│
├── 📂 scripts/
│   ├── 🐍 __init__.py
│   ├── ⚙️ config.py
│   └── 🔧 task1_pipeline.py
│
├── 📂 config/
├── 📂 src/
├── 📂 tests/
└── 📖 README.md
```

---

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

### 3. 📦 Install Required Dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

Or install from requirements file:
```bash
pip install -r requirements.txt
```

---

## 🎯 Task 1: Data Analysis and Preprocessing

> **Objective:** To thoroughly clean, analyze, and enrich the e-commerce transaction dataset (`Fraud_Data.csv`) to prepare it for machine learning model training.

📍 **Location:** The entire workflow is orchestrated in `notebooks/1_Data_Processing_and_EDA.ipynb`, which calls modular functions from `scripts/task1_pipeline.py`.

### 🧹 Data Cleaning and Preprocessing

**Goal:** To ensure data quality and consistency. This is a critical first step as model performance is highly dependent on clean data.

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

### ⚖️ Handling Class Imbalance

**Goal:** To outline a clear and justified strategy for training a model on a dataset with a rare positive class.

#### 📋 Strategy:

| Component | Approach | Reasoning |
|-----------|----------|-----------|
| 🔍 **Problem Identification** | Fraud class is minority (~9%) | Naive model would achieve >90% accuracy by predicting non-fraud |
| 🎯 **Chosen Technique** | SMOTE (Synthetic Minority Over-sampling) | Generates synthetic minority samples |
| 📊 **Implementation Plan** | Apply SMOTE only to training set | Prevents data leakage in test set |

> ⚠️ **Important:** SMOTE is chosen over undersampling to avoid discarding valuable majority class data.

---

## 🚀 How to Run the Pipeline

### 📓 Task 1 - Data Processing and EDA

Execute the complete pipeline by running the Jupyter Notebook:

```bash
# Navigate to the notebook
jupyter notebook notebooks/1_Data_Processing_and_EDA.ipynb
```

**This will:**
- 📥 Load raw data
- 🧹 Execute cleaning and feature engineering
- 📊 Generate EDA plots → `data/processed/images/`
- 💾 Save processed data → `data/processed/`

---

## 🤝 Contributing

1. 🍴 Fork the repository
2. 🌿 Create a feature branch (`git checkout -b feature/amazing-feature`)
3. 💾 Commit your changes (`git commit -m 'Add some amazing feature'`)
4. 📤 Push to the branch (`git push origin feature/amazing-feature`)
5. 🔄 Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 📞 Contact

**Adey Innovations Inc.**

- 📧 Email: contact@adeyinnovations.com
- 🌐 Website: [www.adeyinnovations.com](https://www.adeyinnovations.com)
- 💼 LinkedIn: [Adey Innovations](https://linkedin.com/company/adey-innovations)



**⭐ Star this repository if you find it helpful!**

Made with ❤️ by the matias ashenafi

