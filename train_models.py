"""
🏦 Bank Fraud Detection - Model Training Script
Run this to train all models and save them for the desktop app.
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, 
    f1_score, precision_score, recall_score, average_precision_score, accuracy_score
)
from sklearn.ensemble import RandomForestClassifier

import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier

import optuna
from optuna.samplers import TPESampler

from imblearn.over_sampling import SMOTE

import joblib
import json
import os
from datetime import datetime

# ============================================
# Configuration
# ============================================
DATA_PATH = 'data/fraud_dataset.csv'
N_TRIALS = 15  # Optuna trials per model

print('=' * 60)
print('🏦 BANK FRAUD DETECTION - MODEL TRAINING')
print('=' * 60)
print(f'📅 Started: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')

# ============================================
# Load Data
# ============================================
print('\n📊 Loading dataset...')
if not os.path.exists(DATA_PATH):
    print('⚠️ Dataset not found. Generating...')
    exec(open('generate_dataset.py').read())

df = pd.read_csv(DATA_PATH)
print(f'✅ Loaded {len(df):,} transactions')

# Class distribution
fraud_count = df['is_fraud'].sum()
normal_count = len(df) - fraud_count
fraud_pct = (fraud_count / len(df)) * 100
print(f'   Normal: {normal_count:,} ({100-fraud_pct:.2f}%)')
print(f'   Fraud:  {fraud_count:,} ({fraud_pct:.2f}%)')

# ============================================
# Preprocessing
# ============================================
print('\n🔧 Preprocessing...')

# Drop non-predictive columns
drop_cols = ['TransactionID', 'AccountID', 'TransactionDate']
df_model = df.drop(columns=drop_cols)

# Encode categorical variables
categorical_cols = ['TransactionType', 'Location', 'Channel', 'MerchantCategory', 'CustomerOccupation']
label_encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    df_model[col] = le.fit_transform(df_model[col])
    label_encoders[col] = le
    print(f'   Encoded: {col}')

# Split features and target
X = df_model.drop('is_fraud', axis=1)
y = df_model['is_fraud']
feature_names = list(X.columns)
print(f'   Features: {len(feature_names)}')

# Scale
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=feature_names)

# Train/Val/Test split
X_train, X_temp, y_train, y_temp = train_test_split(X_scaled, y, test_size=0.30, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp)

print(f'\n📊 Data Split:')
print(f'   Train: {len(X_train):,}')
print(f'   Val:   {len(X_val):,}')
print(f'   Test:  {len(X_test):,}')

# SMOTE
print('\n⚖️ Applying SMOTE...')
smote = SMOTE(random_state=42, sampling_strategy=0.5)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
print(f'   After SMOTE: Normal={sum(y_train_resampled==0):,}, Fraud={sum(y_train_resampled==1):,}')

# ============================================
# Model Training
# ============================================
results = {}
optuna.logging.set_verbosity(optuna.logging.WARNING)
sampler = TPESampler(seed=42)

def evaluate_model(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'auc_roc': roc_auc_score(y_test, y_pred_proba),
        'avg_precision': average_precision_score(y_test, y_pred_proba)
    }
    
    print(f'   AUC-ROC: {metrics["auc_roc"]:.4f} | F1: {metrics["f1"]:.4f} | Prec: {metrics["precision"]:.4f} | Rec: {metrics["recall"]:.4f}')
    return metrics, y_pred, y_pred_proba

# Random Forest
print('\n🌲 Training Random Forest...')
def objective_rf(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 300),
        'max_depth': trial.suggest_int('max_depth', 5, 20),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
        'class_weight': 'balanced', 'random_state': 42, 'n_jobs': -1
    }
    model = RandomForestClassifier(**params)
    model.fit(X_train_resampled, y_train_resampled)
    return roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])

study_rf = optuna.create_study(direction='maximize', sampler=sampler)
study_rf.optimize(objective_rf, n_trials=N_TRIALS, show_progress_bar=True)

best_rf = study_rf.best_trial.params
best_rf.update({'class_weight': 'balanced', 'random_state': 42, 'n_jobs': -1})
rf_model = RandomForestClassifier(**best_rf)
rf_model.fit(X_train_resampled, y_train_resampled)
results['Random Forest'], rf_pred, rf_proba = evaluate_model(rf_model, X_test, y_test, 'Random Forest')

# XGBoost
print('\n🚀 Training XGBoost...')
def objective_xgb(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 300),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'scale_pos_weight': sum(y_train==0)/sum(y_train==1),
        'random_state': 42, 'eval_metric': 'auc'
    }
    model = xgb.XGBClassifier(**params)
    model.fit(X_train_resampled, y_train_resampled, verbose=False)
    return roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])

study_xgb = optuna.create_study(direction='maximize', sampler=sampler)
study_xgb.optimize(objective_xgb, n_trials=N_TRIALS, show_progress_bar=True)

best_xgb = study_xgb.best_trial.params
best_xgb.update({'scale_pos_weight': sum(y_train==0)/sum(y_train==1), 'random_state': 42, 'eval_metric': 'auc'})
xgb_model = xgb.XGBClassifier(**best_xgb)
xgb_model.fit(X_train_resampled, y_train_resampled, verbose=False)
results['XGBoost'], xgb_pred, xgb_proba = evaluate_model(xgb_model, X_test, y_test, 'XGBoost')

# LightGBM
print('\n⚡ Training LightGBM...')
def objective_lgb(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 300),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 20, 100),
        'class_weight': 'balanced', 'random_state': 42, 'verbose': -1
    }
    model = lgb.LGBMClassifier(**params)
    model.fit(X_train_resampled, y_train_resampled)
    return roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])

study_lgb = optuna.create_study(direction='maximize', sampler=sampler)
study_lgb.optimize(objective_lgb, n_trials=N_TRIALS, show_progress_bar=True)

best_lgb = study_lgb.best_trial.params
best_lgb.update({'class_weight': 'balanced', 'random_state': 42, 'verbose': -1})
lgb_model = lgb.LGBMClassifier(**best_lgb)
lgb_model.fit(X_train_resampled, y_train_resampled)
results['LightGBM'], lgb_pred, lgb_proba = evaluate_model(lgb_model, X_test, y_test, 'LightGBM')

# CatBoost
print('\n🐱 Training CatBoost...')
def objective_cat(trial):
    params = {
        'iterations': trial.suggest_int('iterations', 100, 300),
        'depth': trial.suggest_int('depth', 4, 8),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'auto_class_weights': 'Balanced', 'random_state': 42, 'verbose': False
    }
    model = CatBoostClassifier(**params)
    model.fit(X_train_resampled, y_train_resampled, verbose=False)
    return roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])

study_cat = optuna.create_study(direction='maximize', sampler=sampler)
study_cat.optimize(objective_cat, n_trials=N_TRIALS, show_progress_bar=True)

best_cat = study_cat.best_trial.params
best_cat.update({'auto_class_weights': 'Balanced', 'random_state': 42, 'verbose': False})
cat_model = CatBoostClassifier(**best_cat)
cat_model.fit(X_train_resampled, y_train_resampled, verbose=False)
results['CatBoost'], cat_pred, cat_proba = evaluate_model(cat_model, X_test, y_test, 'CatBoost')

# ============================================
# Results
# ============================================
print('\n' + '=' * 60)
print('🏆 MODEL COMPARISON')
print('=' * 60)

comparison_df = pd.DataFrame(results).T.round(4)
comparison_df = comparison_df.sort_values('auc_roc', ascending=False)
print(comparison_df.to_string())

best_model_name = comparison_df.index[0]
print(f'\n🥇 Best Model: {best_model_name} (AUC-ROC: {comparison_df.loc[best_model_name, "auc_roc"]:.4f})')

# ============================================
# Save Models
# ============================================
print('\n💾 Saving models...')
os.makedirs('models', exist_ok=True)

models_to_save = {
    'random_forest': rf_model,
    'xgboost': xgb_model,
    'lightgbm': lgb_model,
    'catboost': cat_model
}

for name, model in models_to_save.items():
    joblib.dump(model, f'models/{name}_model.pkl')
    print(f'   ✅ models/{name}_model.pkl')

best_model = {'Random Forest': rf_model, 'XGBoost': xgb_model, 'LightGBM': lgb_model, 'CatBoost': cat_model}[best_model_name]
joblib.dump(best_model, 'models/best_model.pkl')
print(f'   🏆 models/best_model.pkl')

joblib.dump(scaler, 'models/scaler.pkl')
joblib.dump(label_encoders, 'models/label_encoders.pkl')
print('   ✅ models/scaler.pkl')
print('   ✅ models/label_encoders.pkl')

with open('models/feature_names.json', 'w') as f:
    json.dump(feature_names, f)
print('   ✅ models/feature_names.json')

metrics_dict = {
    'best_model': best_model_name,
    'results': {k: {m: float(v) for m, v in metrics.items()} for k, metrics in results.items()},
    'training_date': datetime.now().isoformat(),
    'dataset_size': len(df),
    'fraud_rate': float(fraud_pct)
}

with open('models/model_metrics.json', 'w') as f:
    json.dump(metrics_dict, f, indent=2)
print('   ✅ models/model_metrics.json')

# ============================================
# Done
# ============================================
print('\n' + '=' * 60)
print('🎉 TRAINING COMPLETE!')
print('=' * 60)
print(f'\n📅 Finished: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
print(f'\n🚀 Run the desktop app:')
print('   python app_desktop.py')
print('=' * 60)
