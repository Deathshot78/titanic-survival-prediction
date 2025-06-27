import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import lightgbm as lgb
import pytorch_lightning as pl
import json
import warnings
import logging

# --- Step 1: Import from your custom scripts ---
try:
    from preprocess import preprocess_data
    from models import FTTransformerFromScratch, TitanicFTDataset
except ImportError as e:
    print(f"Error: Could not import from 'preprocess.py' or 'models.py'.")
    print("Please ensure both files exist in the same directory as this script.")
    print(f"Details: {e}")
    exit()

# --- Suppress Logs for Cleaner Output ---
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)


# --- 2. Main Execution Block ---
if __name__ == '__main__':
    pl.seed_everything(42)

    # --- Load Data and Best Hyperparameters ---
    try:
        with open('logs/best_hyperparameters.json', 'r') as f:
            all_best_params = json.load(f)
        print("Successfully loaded 'best_hyperparameters.json'")
    except FileNotFoundError:
        print("Error: 'best_hyperparameters.json' not found.")
        print("Please run the 'objective_optuna.py' script first to generate the hyperparameters.")
        exit()

    X_full, y_full, X_predict, test_passenger_ids, numerical_features, categorical_features = preprocess_data()

    # --- Train Final Models and Predict ---
    print("\n--- Training Final Models on Full Data and Generating Submissions ---")

    # --- Model 1: FT-Transformer ---
    print("\n1. Training and Predicting with final FT-Transformer...")
    best_params_ft = all_best_params['ft_transformer']
    
    # Data preparation specific to FT-Transformer
    X_train_ft_final, X_predict_ft_final = X_full.copy(), X_predict.copy()
    scaler_ft = StandardScaler()
    X_train_ft_final[numerical_features] = scaler_ft.fit_transform(X_train_ft_final[numerical_features])
    X_predict_ft_final[numerical_features] = scaler_ft.transform(X_predict_ft_final[numerical_features])
    
    cat_maps = {col: X_train_ft_final[col].astype('category').cat.categories for col in categorical_features}
    cardinalities = [len(cat_maps[col]) for col in categorical_features]
    
    for col in categorical_features:
        X_train_ft_final[col] = pd.Categorical(X_train_ft_final[col], categories=cat_maps[col]).codes
        X_predict_ft_final[col] = pd.Categorical(X_predict_ft_final[col], categories=cat_maps[col]).codes
        # Handle any categories in predict set not seen in training
        if (X_predict_ft_final[col] == -1).any():
            mode_code = cat_maps[col].get_loc(X_full[col].mode()[0])
            X_predict_ft_final[col].replace(-1, mode_code, inplace=True)
            
    final_train_ds = TitanicFTDataset(X_train_ft_final[numerical_features], X_train_ft_final[categorical_features], y_full)
    predict_ds = TitanicFTDataset(X_predict_ft_final[numerical_features], X_predict_ft_final[categorical_features])
    final_train_loader = DataLoader(final_train_ds, batch_size=32, shuffle=True)
    predict_loader = DataLoader(predict_ds, batch_size=32)
    
    # Instantiate and train the final model
    final_model_ft = FTTransformerFromScratch(
        num_numerical=len(numerical_features), 
        cat_cardinalities=cardinalities,
        embed_dim=best_params_ft['embed_dim'], 
        num_heads=best_params_ft['num_heads'],
        ff_dim=best_params_ft['embed_dim'] * best_params_ft['ff_dim_factor'],
        num_layers=best_params_ft['num_layers'], 
        lr=best_params_ft['lr'], 
        weight_decay=best_params_ft['weight_decay'], 
        dropout=best_params_ft['dropout']
    )
    final_trainer_ft = pl.Trainer(max_epochs=30, accelerator="auto", logger=False, enable_progress_bar=True, enable_model_summary=False)
    final_trainer_ft.fit(final_model_ft, final_train_loader)
    
    # Predict and save
    preds_ft_final = torch.cat(final_trainer_ft.predict(final_model_ft, predict_loader)).flatten().cpu().numpy().astype(int)
    pd.DataFrame({'PassengerId': test_passenger_ids, 'Survived': preds_ft_final}).to_csv('data/submission_ft_transformer_tuned.csv', index=False)
    print("FT-Transformer submission file created.")

    # --- Model 2: LightGBM ---
    print("\n2. Training and Predicting with final LightGBM...")
    best_params_lgbm = all_best_params['lightgbm']
    
    X_full_lgbm, X_predict_lgbm = X_full.copy(), X_predict.copy()
    for col in categorical_features:
        X_full_lgbm[col] = X_full_lgbm[col].astype('category')
        X_predict_lgbm[col] = X_predict_lgbm[col].astype('category')
        
    final_lgbm = lgb.LGBMClassifier(objective='binary', **best_params_lgbm)
    final_lgbm.fit(X_full_lgbm, y_full)
    preds_lgbm = final_lgbm.predict(X_predict_lgbm)
    pd.DataFrame({'PassengerId': test_passenger_ids, 'Survived': preds_lgbm}).to_csv('data/submission_lgbm_tuned.csv', index=False)
    print("LightGBM submission file created.")

    # --- Model 3: Logistic Regression ---
    print("\n3. Training and Predicting with final Logistic Regression...")
    best_params_logreg = all_best_params['logistic_regression']
    
    preprocessor_final = ColumnTransformer(transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])
    
    final_logreg = Pipeline(steps=[
        ('preprocessor', preprocessor_final),
        ('classifier', LogisticRegression(random_state=42, max_iter=2000, **best_params_logreg))
    ])
    final_logreg.fit(X_full, y_full)
    preds_lr = final_logreg.predict(X_predict)
    pd.DataFrame({'PassengerId': test_passenger_ids, 'Survived': preds_lr}).to_csv('data/submission_logreg_tuned.csv', index=False)
    print("Logistic Regression submission file created.")
    
    # --- Final Ensemble ---
    print("\n--- Creating Final Ensemble Submission ---")
    stacked_preds = np.vstack([preds_ft_final, preds_lgbm, preds_lr]).T
    from scipy.stats import mode
    ensemble_preds, _ = mode(stacked_preds, axis=1)
    
    submission_df = pd.DataFrame({
        'PassengerId': test_passenger_ids,
        'Survived': ensemble_preds.flatten()
    })
    submission_df.to_csv('logs/ensemble_submission_tuned.csv', index=False)

    print("\n" + "-" * 40)
    print("All submission files created successfully!")
    print("You can now upload them to the Kaggle competition.")
    print("-" * 40)
    print("Ensemble submission head:")
    print(submission_df.head())