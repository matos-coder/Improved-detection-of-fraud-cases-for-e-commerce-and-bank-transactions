# scripts/task3_explainability.py

import shap
import matplotlib.pyplot as plt
import pandas as pd

def explain_model_with_shap(model, preprocessor, X_train, X_test, model_type):
    """
    Uses SHAP to explain the model's predictions.
    Generates and saves a summary plot and a force plot for a single prediction.
    """
    print(f"\n===== Generating SHAP Explanations for {model_type} Model =====")
    
    try:
        # --- Data Preparation for SHAP ---
        # 1. Preprocess the test data
        print("🔄 [Step 1/4] Preprocessing test data for SHAP...")
        X_test_processed = preprocessor.transform(X_test)
        
        # 2. Get feature names after one-hot encoding
        try:
            # For scikit-learn >= 1.0
            cat_feature_names = preprocessor.named_transformers_['cat'].get_feature_names_out()
        except AttributeError:
            # For older scikit-learn versions
            cat_feature_names = preprocessor.named_transformers_['cat'].get_feature_names()
            
        num_feature_names = preprocessor.named_transformers_['num'].feature_names_in_
        feature_names = list(num_feature_names) + list(cat_feature_names)
        
        # Convert the processed test data back to a DataFrame with proper column names
        X_test_processed_df = pd.DataFrame(X_test_processed, columns=feature_names)
        print("✅ Test data preprocessed with feature names.")

        # --- SHAP Analysis ---
        # 3. Create a SHAP explainer object
        # TreeExplainer is optimized for tree-based models like LightGBM
        print("🔄 [Step 2/4] Creating SHAP explainer and calculating SHAP values...")
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test_processed_df)
        print("✅ SHAP values calculated.")

        # --- Generate and Save Plots ---
        # 4. Generate and save the Summary Plot
        print("🔄 [Step 3/4] Generating and saving SHAP Summary Plot...")
        plt.figure()
        shap.summary_plot(shap_values, X_test_processed_df, plot_type="bar", show=False)
        plt.title(f'SHAP Feature Importance ({model_type})', fontsize=16)
        plt.tight_layout()
        # Note: You might need to adjust the path in a real project
        plt.savefig(f'outputs/images/shap_summary_{model_type.lower()}.png', dpi=300)
        print(f"✅ SHAP Summary Plot saved to 'outputs/images/shap_summary_{model_type.lower()}.png'")
        plt.close()
        
        # Display the summary plot in the notebook
        shap.summary_plot(shap_values, X_test_processed_df)


        # 5. Generate and save a Force Plot for a single prediction
        print("🔄 [Step 4/4] Generating and saving SHAP Force Plot for a single prediction...")
        # We'll explain the first prediction in the test set
        plt.figure()
        # Use shap.force_plot for a single instance
        shap.force_plot(explainer.expected_value[1], shap_values[1][0,:], X_test_processed_df.iloc[0,:], matplotlib=True, show=False)
        plt.title(f'SHAP Force Plot for a Single Prediction ({model_type})', fontsize=12)
        plt.tight_layout()
        plt.savefig(f'outputs/images/shap_force_plot_{model_type.lower()}.png', dpi=300, bbox_inches='tight')
        print(f"✅ SHAP Force Plot saved to 'outputs/images/shap_force_plot_{model_type.lower()}.png'")
        plt.close()
        
        # Display the force plot in the notebook
        return shap.force_plot(explainer.expected_value[1], shap_values[1][0,:], X_test_processed_df.iloc[0,:])


    except Exception as e:
        print(f"❌ An error occurred during SHAP analysis: {e}")
        import traceback
        traceback.print_exc()