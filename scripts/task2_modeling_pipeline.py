# scripts/task2_modeling_pipeline.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb
from imblearn.over_sampling import SMOTE
from sklearn.metrics import f1_score, precision_recall_curve, auc, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import time

def prepare_ecommerce_data(df):
    """
    Prepares the e-commerce dataset for modeling by selecting features and defining data types.
    """
    print("\n--- Preparing E-commerce Data for Modeling ---")
    try:
        X = df.drop(columns=['class', 'user_id', 'signup_time', 'purchase_time', 'device_id', 'ip_address'])
        y = df['class']
        categorical_features = ['source', 'browser', 'sex', 'country']
        numerical_features = X.select_dtypes(include=np.number).columns.tolist()
        print("✅ Data preparation successful.")
        return X, y, numerical_features, categorical_features
    except KeyError as e:
        print(f"❌ ERROR: Column {e} not found during data preparation.")
        return None, None, None, None

def prepare_creditcard_data(df):
    """
    Prepares the credit card dataset for modeling.
    """
    print("\n--- Preparing Credit Card Data for Modeling ---")
    try:
        X = df.drop(columns=['Class'])
        y = df['Class']
        numerical_features = X.columns.tolist()
        categorical_features = []
        print("✅ Data preparation successful.")
        return X, y, numerical_features, categorical_features
    except KeyError as e:
        print(f"❌ ERROR: Column {e} not found during data preparation.")
        return None, None, None, None

def create_preprocessor(numerical_features, categorical_features):
    """
    Creates a ColumnTransformer to preprocess data.
    """
    print("--- Creating data preprocessor ---")
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='passthrough' # Keep other columns if any
    )
    print("✅ Preprocessor created.")
    return preprocessor

def train_and_evaluate_model(X_train, y_train, X_test, y_test, preprocessor, model, model_name):
    """
    A new, more explicit pipeline for training and evaluation.
    """
    print(f"\n===== Training and Evaluating: {model_name} =====")
    start_time = time.time()
    
    try:
        # Step 1: Fit the preprocessor on the training data and transform it
        print("🔄 [Step 1/5] Preprocessing training data...")
        X_train_processed = preprocessor.fit_transform(X_train)
        print("✅ Training data preprocessed.")

        # Step 2: Apply SMOTE to the processed training data
        print("🔄 [Step 2/5] Applying SMOTE to handle class imbalance...")
        smote = SMOTE(random_state=42)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train_processed, y_train)
        print(f"✅ SMOTE applied. New training set size: {X_train_resampled.shape[0]} samples.")

        # Step 3: Train the model on the resampled data
        print(f"🔄 [Step 3/5] Training the {model_name} model...")
        model.fit(X_train_resampled, y_train_resampled)
        print("✅ Model training complete.")

        # Step 4: Preprocess the test data using the *already fitted* preprocessor
        print("🔄 [Step 4/5] Preprocessing test data...")
        X_test_processed = preprocessor.transform(X_test)
        print("✅ Test data preprocessed.")

        # Step 5: Evaluate the model on the processed test data
        print("🔄 [Step 5/5] Evaluating model performance...")
        y_pred = model.predict(X_test_processed)
        y_pred_proba = model.predict_proba(X_test_processed)[:, 1]

        f1 = f1_score(y_test, y_pred)
        precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
        pr_auc = auc(recall, precision)
        
        print("\n--- Evaluation Results ---")
        print(f"  - F1 Score: {f1:.4f}")
        print(f"  - Area Under PR Curve (AUC-PR): {pr_auc:.4f}")

        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap=plt.cm.Blues)
        plt.title(f'Confusion Matrix - {model_name}')
        plt.show()

        end_time = time.time()
        print(f"⏱️ Total time for {model_name}: {end_time - start_time:.2f} seconds")
        return f1, pr_auc

    except Exception as e:
        print(f"❌ An error occurred during the pipeline for {model_name}: {e}")
        import traceback
        traceback.print_exc()
        return None, None