# In[1]:
# Cell 1: Setup and Imports
# ==============================================================================
import pandas as pd
import sys
import os
import lightgbm as lgb
from sklearn.model_selection import train_test_split

# Add project root to path to allow module imports
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

from scripts import config
from scripts import task1_pipeline as pipe1
from scripts import task2_modeling_pipeline as pipe2
from scripts import task3_explainability as pipe3 # Our new explainability script

print("✅ Setup Complete. Libraries and modules imported.")


# In[2]:
# Cell 2: Load and Prepare E-commerce Data
# ==============================================================================
# For this task, we will focus on explaining our best model for the more
# complex e-commerce dataset.

print("--- Loading and Preparing E-commerce Data ---")
try:
    ecommerce_df = pd.read_csv(config.PROCESSED_FRAUD_DATA_PATH)
    print("✅ Processed e-commerce data loaded successfully.")
    
    # Prepare data for modeling
    X_ecom, y_ecom, num_ecom, cat_ecom = pipe2.prepare_ecommerce_data(ecommerce_df)

    # Perform Train-Test Split
    X_train_ecom, X_test_ecom, y_train_ecom, y_test_ecom = train_test_split(
        X_ecom, y_ecom, test_size=0.2, random_state=42, stratify=y_ecom
    )
    print("✅ Data prepared and split successfully.")
    
except Exception as e:
    print(f"❌ An error occurred during data loading or preparation: {e}")


# In[3]:
# Cell 3: Retrain the Best Model (LightGBM)
# ==============================================================================
# We need a trained model object to explain. Let's quickly retrain our best
# performer, LightGBM, on the e-commerce data.

print("--- Retraining the LightGBM Model ---")

# Create the preprocessor
preprocessor_ecom = pipe2.create_preprocessor(num_ecom, cat_ecom)

# Get the processed training data and apply SMOTE
X_train_processed = preprocessor_ecom.fit_transform(X_train_ecom)
X_train_resampled, y_train_resampled = pipe2.SMOTE(random_state=42).fit_resample(X_train_processed, y_train_ecom)

# Initialize and train the model
lgbm_model = lgb.LGBMClassifier(random_state=42)
lgbm_model.fit(X_train_resampled, y_train_resampled)

print("✅ Best model (LightGBM) has been retrained successfully.")


# In[4]:
# Cell 4: Generate and Interpret SHAP Explanations
# ==============================================================================
# This is the core of Task 3. We call our function to generate SHAP plots.
# The function will save the plots as images and also display them here.

pipe3.explain_model_with_shap(lgbm_model, preprocessor_ecom, X_train_ecom, X_test_ecom, "E-commerce")


# In[5]:
# Cell 5: Interpretation of SHAP Plots
# ==============================================================================
# This is where you explain what the plots reveal.

print("\n\n===== Interpretation of SHAP Plots =====")
print("""
### **1. Global Feature Importance (Summary Plot)**

The SHAP summary plot gives us a high-level view of the most important features for the model across the entire dataset.

**Key Insights:**
* **`time_since_signup_seconds` is the most impactful feature.** The plot clearly shows that low values (blue dots on the left) have a high positive SHAP value, meaning they strongly push the model's prediction towards "fraud." This confirms our EDA finding that quick purchases after signup are a major red flag.
* **`purchase_value` and `device_id_count` are also highly significant.** High purchase values and devices used for multiple transactions are strong indicators of fraud.
* **Categorical features like `country` and `source` play a role.** Certain countries and acquisition sources (e.g., `source_Ads`) contribute to the model's fraud prediction, though less than the top numerical features.

**Business Implication:** This tells us that Adey Innovations should focus its fraud prevention rules heavily on the time between signup and purchase. It also validates the importance of tracking device fingerprints.

---

### **2. Local Prediction Explanation (Force Plot)**

The Force Plot explains a **single, specific prediction**. It shows the "forces" that pushed the model's prediction for one transaction.

**How to Read It:**
* The **base value** is the average prediction over the entire dataset.
* **Red arrows** represent features that pushed the prediction **higher (towards fraud)**. The size of the arrow indicates the strength of the push.
* **Blue arrows** represent features that pushed the prediction **lower (towards legitimate)**.

**Analysis of the Example Plot:**
For the specific transaction shown, the prediction was pushed towards fraud primarily by:
1.  A **very low `time_since_signup_seconds`**.
2.  A **high `purchase_value`**.

These two factors were strong enough to overcome other features that might have suggested the transaction was legitimate (e.g., a low-risk country or a common browser).

**Business Implication:** This level of detail is invaluable for explaining to a customer or an analyst why a specific transaction was flagged. It moves the model from being a "black box" to a transparent, auditable decision-making tool.
""")

