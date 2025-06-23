import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import lightgbm as lgb
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
import optuna
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
    # Define placeholder classes if import fails
    def preprocess_data(): pass
    class FTTransformerFromScratch(pl.LightningModule): pass
    class TitanicFTDataset(torch.utils.data.Dataset): pass

# --- Suppress Logs for Cleaner Output ---
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
optuna.logging.set_verbosity(optuna.logging.WARNING)


# --- 2. DataModule for FT-Transformer ---
class FTTransformerDataModule(pl.LightningDataModule):
    """
    DataModule specifically for preparing data for the FT-Transformer.
    It handles scaling numerical features and integer-encoding categorical features.
    """
    def __init__(self, X_train, y_train, X_val, y_val, numerical_features, categorical_features, batch_size=32):
        super().__init__()
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.numerical_features = numerical_features
        self.categorical_features = categorical_features
        self.batch_size = batch_size

    def setup(self, stage=None):
        # Scale numerical features
        scaler = StandardScaler()
        self.X_train[self.numerical_features] = scaler.fit_transform(self.X_train[self.numerical_features])
        self.X_val[self.numerical_features] = scaler.transform(self.X_val[self.numerical_features])
        
        # Integer-encode categorical features
        cat_maps = {col: self.X_train[col].astype('category').cat.categories for col in self.categorical_features}
        self.cardinalities = [len(cat_maps[col]) for col in self.categorical_features]
        
        for col in self.categorical_features:
            self.X_train[col] = pd.Categorical(self.X_train[col], categories=cat_maps[col]).codes
            self.X_val[col] = pd.Categorical(self.X_val[col], categories=cat_maps[col]).codes

        # Create datasets
        self.train_ds = TitanicFTDataset(self.X_train[self.numerical_features], self.X_train[self.categorical_features], self.y_train)
        self.val_ds = TitanicFTDataset(self.X_val[self.numerical_features], self.X_val[self.categorical_features], self.y_val)

    def train_dataloader(self): return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True)
    def val_dataloader(self): return DataLoader(self.val_ds, batch_size=self.batch_size)


# --- 3. Optuna Objective Functions ---

def objective_ft_transformer(trial, datamodule, num_numerical, cardinalities):
    """Objective function for FT-Transformer, now using a DataModule."""
    embed_dim = trial.suggest_categorical("embed_dim", [128, 192, 256])
    num_heads = trial.suggest_categorical("num_heads", [4, 8])
    if embed_dim % num_heads != 0: raise optuna.exceptions.TrialPruned("embed_dim must be divisible by num_heads.")
    
    ff_dim = embed_dim * trial.suggest_categorical("ff_dim_factor", [2, 4])
    num_layers = trial.suggest_int("num_layers", 2, 5)
    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    dropout = trial.suggest_float("dropout", 0.0, 0.4)

    model = FTTransformerFromScratch(
        num_numerical=num_numerical, cat_cardinalities=cardinalities, 
        embed_dim=embed_dim, num_heads=num_heads, ff_dim=ff_dim, 
        num_layers=num_layers, lr=lr, weight_decay=weight_decay, dropout=dropout
    )
    trainer = pl.Trainer(max_epochs=30, accelerator="auto", callbacks=[EarlyStopping(monitor="val_loss", patience=5)], logger=False, enable_progress_bar=False, enable_model_summary=False)
    
    try:
        trainer.fit(model, datamodule=datamodule)
        return trainer.callback_metrics.get("val_acc", 0.0).item()
    except Exception as e:
        print(f"Trial failed for FT-Transformer with error: {e}")
        return 0.0

def objective_lgbm(trial, X_train, y_train, X_val, y_val):
    """Objective function for tuning LightGBM."""
    params = {
        'objective': 'binary', 'verbosity': -1,
        'n_estimators': trial.suggest_int('n_estimators', 100, 2000), 
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 20, 300), 
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True), 
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 1.0), 
        'subsample': trial.suggest_float('subsample', 0.4, 1.0),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100)
    }
    
    X_train_lgbm, X_val_lgbm = X_train.copy(), X_val.copy()
    for col in X_train_lgbm.select_dtypes(include=['object', 'category']).columns:
        X_train_lgbm[col] = X_train_lgbm[col].astype('category')
        X_val_lgbm[col] = X_val_lgbm[col].astype('category')
    
    model = lgb.LGBMClassifier(**params)
    model.fit(X_train_lgbm, y_train, eval_set=[(X_val_lgbm, y_val)], eval_metric='accuracy', callbacks=[lgb.early_stopping(100, verbose=False)])
    return accuracy_score(y_val, model.predict(X_val_lgbm))

def objective_logreg(trial, X_train, y_train, X_val, y_val, numerical_features, categorical_features):
    """Objective function for tuning Logistic Regression."""
    C = trial.suggest_float("C", 1e-4, 1e2, log=True); solver = trial.suggest_categorical("solver", ["liblinear", "lbfgs", "saga"])
    preprocessor = ColumnTransformer(transformers=[('num', StandardScaler(), numerical_features), ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)])
    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', LogisticRegression(random_state=42, C=C, solver=solver, max_iter=2000))])
    pipeline.fit(X_train, y_train); return accuracy_score(y_val, pipeline.predict(X_val))


# --- 4. Main Execution Block ---
if __name__ == '__main__':
    X_full, y_full, _, _, numerical_features, categorical_features = preprocess_data()
    X_train, X_val, y_train, y_val = train_test_split(
        X_full, y_full, test_size=0.2, random_state=42, stratify=y_full)

    all_best_params = {}

    # --- Tune FT-Transformer ---
    print("\n--- Tuning FT-Transformer ---")
    ft_dm = FTTransformerDataModule(X_train, y_train, X_val, y_val, numerical_features, categorical_features)
    ft_dm.setup()
    study_ft = optuna.create_study(direction="maximize")
    study_ft.optimize(lambda t: objective_ft_transformer(t, ft_dm, len(numerical_features), ft_dm.cardinalities), n_trials=30)
    all_best_params['ft_transformer'] = study_ft.best_params
    print(f"Best FT-Transformer Val Accuracy: {study_ft.best_value:.4f}")
    
    # --- Tune LightGBM ---
    print("\n--- Tuning LightGBM ---")
    study_lgbm = optuna.create_study(direction="maximize")
    study_lgbm.optimize(lambda t: objective_lgbm(t, X_train, y_train, X_val, y_val), n_trials=30)
    all_best_params['lightgbm'] = study_lgbm.best_params
    print(f"Best LightGBM Val Accuracy: {study_lgbm.best_value:.4f}")

    # --- Tune Logistic Regression ---
    print("\n--- Tuning Logistic Regression ---")
    study_logreg = optuna.create_study(direction="maximize")
    study_logreg.optimize(lambda t: objective_logreg(t, X_train, y_train, X_val, y_val, numerical_features, categorical_features), n_trials=30)
    all_best_params['logistic_regression'] = study_logreg.best_params
    print(f"Best Logistic Regression Val Accuracy: {study_logreg.best_value:.4f}")

    # --- Save Best Hyperparameters to a file ---
    with open('logs/best_hyperparameters.json', 'w') as f:
        json.dump(all_best_params, f, indent=4)
        
    print("\n----------------------------------------------------")
    print("Hyperparameter tuning complete for all models.")
    print("Best parameters saved to 'best_hyperparameters.json'")
    print("----------------------------------------------------")
