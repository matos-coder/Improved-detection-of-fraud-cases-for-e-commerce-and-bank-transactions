# scripts/task2_modeling_pipeline.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.metrics import f1_score, precision_recall_curve, auc, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import time

def prepare_ecommerce_data(df):
    """
    Prepares the e-commerce dataset for modeling by selecting features and defining data types.
    """
    print("\n--- Preparing E-commerce Data for Modeling ---")
    try:
        # Define features (X) and target (y)
        # We drop identifier columns and the original time/IP columns
        X = df.drop(columns=['class', 'user_id', 'signup_time', 'purchase_time', 'device_id', 'ip_address'])
        y = df['class']

        # Identify categorical and numerical features for preprocessing
        categorical_features = ['source', 'browser', 'sex', 'country']
        numerical_features = X.select_dtypes(include=np.number).columns.tolist()

        print(f"✅ Target variable 'class' separated.")
        print(f"✅ Identified {len(numerical_features)} numerical features.")
        print(f"✅ Identified {len(categorical_features)} categorical features.")
        
        return X, y, numerical_features, categorical_features

    except KeyError as e:
        print(f"❌ ERROR: Column {e} not found. Make sure the processed data is correct.")
        return None, None, None, None
    except Exception as e:
        print(f"❌ An unexpected error occurred during data preparation: {e}")
        return None, None, None, None

def prepare_creditcard_data(df):
    """
    Prepares the credit card dataset for modeling.
    """
    print("\n--- Preparing Credit Card Data for Modeling ---")
    try:
        # The 'Time' column might not be a useful feature as is, but we can scale it.
        # All V1-V28 columns are already numerical and scaled. 'Amount' needs scaling.
        X = df.drop(columns=['Class'])
        y = df['Class']

        # All features are numerical in this dataset
        numerical_features = X.columns.tolist()
        categorical_features = [] # No categorical features in this dataset

        print(f"✅ Target variable 'Class' separated.")
        print(f"✅ Identified {len(numerical_features)} numerical features.")
        
        return X, y, numerical_features, categorical_features

    except KeyError as e:
        print(f"❌ ERROR: Column {e} not found. Make sure the credit card data is correct.")
        return None, None, None, None
    except Exception as e:
        print(f"❌ An unexpected error occurred during data preparation: {e}")
        return None, None, None, None

def create_preprocessing_pipeline(numerical_features, categorical_features):
    """
    Creates a scikit-learn pipeline to preprocess data:
    - Scales numerical features.
    - One-hot encodes categorical features.
    """
    print("\n--- Creating Data Preprocessing Pipeline ---")
    
    # Create a transformer for numerical features (scaling)
    numeric_transformer = StandardScaler()
    
    # Create a transformer for categorical features (one-hot encoding)
    # handle_unknown='ignore' prevents errors if a category appears in test but not train
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')
    
    # Use ColumnTransformer to apply different transformers to different columns
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    print("✅ Preprocessing pipeline created successfully.")
    return preprocessor

def train_and_evaluate_model(X_train, y_train, X_test, y_test, preprocessor, model, model_name):
    """
    Creates a full pipeline with SMOTE, preprocessing, and the model.
    Then, it trains the model and evaluates its performance.
    """
    print(f"\n===== Training and Evaluating: {model_name} =====")
    start_time = time.time()
    
    # Create the full pipeline
    # Step 1: Apply SMOTE for oversampling
    # Step 2: Apply the preprocessing pipeline (scaling and encoding)
    # Step 3: Train the model
    pipeline = ImbPipeline(steps=[('smote', SMOTE(random_state=42)),
                                  ('preprocessor', preprocessor),
                                  ('classifier', model)])
    
    print("🔄 Training the model pipeline...")
    try:
        # Train the entire pipeline on the training data
        pipeline.fit(X_train, y_train)
        print("✅ Model training complete.")
        
        # --- Evaluation ---
        print("🔄 Evaluating model on the test set...")
        
        # Make predictions on the unseen test data
        y_pred = pipeline.predict(X_test)
        
        # Get prediction probabilities for the positive class (for AUC-PR)
        y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
        
        # Calculate F1 Score
        f1 = f1_score(y_test, y_pred)
        print(f"  - F1 Score: {f1:.4f}")
        
        # Calculate Precision-Recall AUC
        precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
        pr_auc = auc(recall, precision)
        print(f"  - Area Under PR Curve (AUC-PR): {pr_auc:.4f}")
        
        # Generate and display Confusion Matrix
        print("  - Confusion Matrix:")
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap=plt.cm.Blues)
        plt.title(f'Confusion Matrix - {model_name}')
        plt.show()

        end_time = time.time()
        print(f"⏱️ Total time for {model_name}: {end_time - start_time:.2f} seconds")
        
        return f1, pr_auc

    except Exception as e:
        print(f"❌ An error occurred during model training or evaluation for {model_name}: {e}")
        return None, None