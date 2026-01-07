"""
🧪 Bank Fraud Detection - Comprehensive Test Suite
Deep QA testing of all components: data, models, preprocessing, predictions
"""

import pandas as pd
import numpy as np
import json
import os
import sys
import joblib
from datetime import datetime

# Test results tracking
TESTS_PASSED = 0
TESTS_FAILED = 0
TEST_RESULTS = []

def log_test(name, passed, details=""):
    global TESTS_PASSED, TESTS_FAILED
    status = "✅ PASS" if passed else "❌ FAIL"
    if passed:
        TESTS_PASSED += 1
    else:
        TESTS_FAILED += 1
    result = f"{status} | {name}"
    if details:
        result += f" | {details}"
    TEST_RESULTS.append((name, passed, details))
    print(result)

print("=" * 70)
print("🧪 BANK FRAUD DETECTION - COMPREHENSIVE QA TEST SUITE")
print("=" * 70)
print(f"📅 Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print()

# ============================================
# 1. DATA TESTS
# ============================================
print("=" * 70)
print("📊 1. DATA QUALITY TESTS")
print("=" * 70)

# Test 1.1: Dataset exists
try:
    df = pd.read_csv('data/fraud_dataset.csv')
    log_test("Dataset file exists", True, f"{len(df):,} rows")
except Exception as e:
    log_test("Dataset file exists", False, str(e))
    df = None

if df is not None:
    # Test 1.2: Correct number of columns
    expected_cols = 15
    log_test("Correct number of columns", len(df.columns) == expected_cols, 
            f"Expected {expected_cols}, got {len(df.columns)}")
    
    # Test 1.3: Required columns exist
    required_cols = ['TransactionID', 'AccountID', 'TransactionAmount', 'TransactionDate', 
                    'hour', 'TransactionType', 'Location', 'Channel', 'MerchantCategory',
                    'CustomerAge', 'CustomerOccupation', 'TransactionDuration', 
                    'LoginAttempts', 'AccountBalance', 'is_fraud']
    missing_cols = [c for c in required_cols if c not in df.columns]
    log_test("All required columns exist", len(missing_cols) == 0, 
            f"Missing: {missing_cols}" if missing_cols else "All present")
    
    # Test 1.4: No null values in critical columns
    critical_cols = ['TransactionAmount', 'is_fraud', 'hour', 'CustomerAge']
    null_counts = {c: df[c].isnull().sum() for c in critical_cols if c in df.columns}
    no_nulls = all(v == 0 for v in null_counts.values())
    log_test("No null values in critical columns", no_nulls, str(null_counts))
    
    # Test 1.5: Transaction amounts are positive
    positive_amounts = (df['TransactionAmount'] > 0).all()
    log_test("All transaction amounts are positive", positive_amounts,
            f"Min: ${df['TransactionAmount'].min():.2f}, Max: ${df['TransactionAmount'].max():.2f}")
    
    # Test 1.6: Fraud rate is reasonable (between 1% and 10%)
    fraud_rate = df['is_fraud'].mean() * 100
    reasonable_fraud_rate = 1 <= fraud_rate <= 10
    log_test("Fraud rate is reasonable (1-10%)", reasonable_fraud_rate, f"{fraud_rate:.2f}%")
    
    # Test 1.7: Hour values are valid (0-23)
    valid_hours = df['hour'].between(0, 23).all()
    log_test("Hour values are valid (0-23)", valid_hours, 
            f"Range: {df['hour'].min()}-{df['hour'].max()}")
    
    # Test 1.8: Customer ages are realistic (18-100)
    valid_ages = df['CustomerAge'].between(18, 100).all()
    log_test("Customer ages are realistic (18-100)", valid_ages,
            f"Range: {df['CustomerAge'].min()}-{df['CustomerAge'].max()}")
    
    # Test 1.9: Unique transaction IDs
    unique_ids = df['TransactionID'].nunique() == len(df)
    log_test("Transaction IDs are unique", unique_ids,
            f"Unique: {df['TransactionID'].nunique()}, Total: {len(df)}")
    
    # Test 1.10: Categorical values are valid
    valid_channels = set(['Online', 'ATM', 'Branch', 'Mobile', 'POS'])
    channel_valid = set(df['Channel'].unique()).issubset(valid_channels)
    log_test("Channel values are valid", channel_valid,
            f"Found: {df['Channel'].unique().tolist()}")

print()

# ============================================
# 2. MODEL FILES TESTS
# ============================================
print("=" * 70)
print("🤖 2. MODEL FILES TESTS")
print("=" * 70)

# Test 2.1: Best model file exists
best_model_path = 'models/best_model.pkl'
best_model_exists = os.path.exists(best_model_path)
log_test("Best model file exists", best_model_exists, best_model_path)

# Test 2.2: All model files exist
model_files = ['random_forest_model.pkl', 'xgboost_model.pkl', 'lightgbm_model.pkl', 'catboost_model.pkl']
for mf in model_files:
    exists = os.path.exists(f'models/{mf}')
    log_test(f"Model file: {mf}", exists)

# Test 2.3: Scaler file exists
scaler_path = 'models/scaler.pkl'
scaler_exists = os.path.exists(scaler_path)
log_test("Scaler file exists", scaler_exists)

# Test 2.4: Label encoders file exists
encoders_path = 'models/label_encoders.pkl'
encoders_exists = os.path.exists(encoders_path)
log_test("Label encoders file exists", encoders_exists)

# Test 2.5: Feature names file exists
features_path = 'models/feature_names.json'
features_exists = os.path.exists(features_path)
log_test("Feature names file exists", features_exists)

# Test 2.6: Model metrics file exists
metrics_path = 'models/model_metrics.json'
metrics_exists = os.path.exists(metrics_path)
log_test("Model metrics file exists", metrics_exists)

print()

# ============================================
# 3. MODEL LOADING TESTS
# ============================================
print("=" * 70)
print("📦 3. MODEL LOADING TESTS")
print("=" * 70)

# Load model
try:
    model = joblib.load('models/best_model.pkl')
    log_test("Best model loads successfully", True, type(model).__name__)
except Exception as e:
    log_test("Best model loads successfully", False, str(e))
    model = None

# Load scaler
try:
    scaler = joblib.load('models/scaler.pkl')
    log_test("Scaler loads successfully", True, type(scaler).__name__)
except Exception as e:
    log_test("Scaler loads successfully", False, str(e))
    scaler = None

# Load label encoders
try:
    label_encoders = joblib.load('models/label_encoders.pkl')
    log_test("Label encoders load successfully", True, f"{len(label_encoders)} encoders")
except Exception as e:
    log_test("Label encoders load successfully", False, str(e))
    label_encoders = {}

# Load feature names
try:
    with open('models/feature_names.json', 'r') as f:
        feature_names = json.load(f)
    log_test("Feature names load successfully", True, f"{len(feature_names)} features")
except Exception as e:
    log_test("Feature names load successfully", False, str(e))
    feature_names = []

# Load metrics
try:
    with open('models/model_metrics.json', 'r') as f:
        metrics = json.load(f)
    log_test("Model metrics load successfully", True, f"Best: {metrics.get('best_model', 'N/A')}")
except Exception as e:
    log_test("Model metrics load successfully", False, str(e))
    metrics = {}

print()

# ============================================
# 4. MODEL QUALITY TESTS
# ============================================
print("=" * 70)
print("📈 4. MODEL QUALITY TESTS")
print("=" * 70)

if metrics:
    best = metrics.get('best_model', 'Unknown')
    results = metrics.get('results', {})
    
    if best in results:
        model_metrics = results[best]
        
        # Test 4.1: AUC-ROC > 0.95
        auc = model_metrics.get('auc_roc', 0)
        log_test("AUC-ROC > 0.95", auc > 0.95, f"AUC-ROC: {auc:.4f}")
        
        # Test 4.2: Precision > 0.85
        precision = model_metrics.get('precision', 0)
        log_test("Precision > 0.85", precision > 0.85, f"Precision: {precision:.4f}")
        
        # Test 4.3: Recall > 0.90
        recall = model_metrics.get('recall', 0)
        log_test("Recall > 0.90", recall > 0.90, f"Recall: {recall:.4f}")
        
        # Test 4.4: F1 > 0.90
        f1 = model_metrics.get('f1', 0)
        log_test("F1-Score > 0.90", f1 > 0.90, f"F1: {f1:.4f}")
        
        # Test 4.5: Accuracy > 0.99
        accuracy = model_metrics.get('accuracy', 0)
        log_test("Accuracy > 0.99", accuracy > 0.99, f"Accuracy: {accuracy:.4f}")
    
    # Test 4.6: All 4 models trained
    num_models = len(results)
    log_test("All 4 models trained", num_models == 4, f"Found: {num_models} models")
    
    # Test 4.7: Best model is LightGBM (expected)
    log_test("Best model is LightGBM", best == "LightGBM", f"Best: {best}")

print()

# ============================================
# 5. PREDICTION TESTS
# ============================================
print("=" * 70)
print("🔮 5. PREDICTION TESTS")
print("=" * 70)

if model and scaler and label_encoders:
    # Test 5.1: Model can make predictions
    try:
        # Create sample normal transaction
        sample_normal = {
            'TransactionAmount': 50.00,
            'hour': 14,
            'TransactionType': 'Debit',
            'Location': 'Chicago',
            'Channel': 'ATM',
            'MerchantCategory': 'Grocery',
            'CustomerAge': 35,
            'CustomerOccupation': 'Engineer',
            'TransactionDuration': 45,
            'LoginAttempts': 1,
            'AccountBalance': 5000.00
        }
        
        # Prepare data
        df_sample = pd.DataFrame([sample_normal])
        for col in ['TransactionType', 'Location', 'Channel', 'MerchantCategory', 'CustomerOccupation']:
            if col in label_encoders:
                try:
                    df_sample[col] = label_encoders[col].transform(df_sample[col])
                except:
                    df_sample[col] = 0
        
        df_sample = df_sample[feature_names]
        df_scaled = scaler.transform(df_sample)
        
        pred = model.predict(df_scaled)[0]
        prob = model.predict_proba(df_scaled)[0][1]
        
        log_test("Model predicts normal transaction as safe", pred == 0, 
                f"Pred: {pred}, Prob: {prob*100:.2f}%")
    except Exception as e:
        log_test("Model predicts normal transaction", False, str(e))
    
    # Test 5.2: Model detects fraud patterns
    try:
        # Create sample fraud transaction
        sample_fraud = {
            'TransactionAmount': 5000.00,  # High amount
            'hour': 2,  # Late night
            'TransactionType': 'Transfer',
            'Location': 'Chicago',
            'Channel': 'Online',
            'MerchantCategory': 'Electronics',
            'CustomerAge': 25,
            'CustomerOccupation': 'Student',
            'TransactionDuration': 5,  # Very quick
            'LoginAttempts': 5,  # Multiple attempts
            'AccountBalance': 500.00  # Low balance
        }
        
        df_sample = pd.DataFrame([sample_fraud])
        for col in ['TransactionType', 'Location', 'Channel', 'MerchantCategory', 'CustomerOccupation']:
            if col in label_encoders:
                try:
                    df_sample[col] = label_encoders[col].transform(df_sample[col])
                except:
                    df_sample[col] = 0
        
        df_sample = df_sample[feature_names]
        df_scaled = scaler.transform(df_sample)
        
        pred = model.predict(df_scaled)[0]
        prob = model.predict_proba(df_scaled)[0][1]
        
        log_test("Model flags suspicious transaction", pred == 1 or prob > 0.5, 
                f"Pred: {pred}, Prob: {prob*100:.2f}%")
    except Exception as e:
        log_test("Model flags suspicious transaction", False, str(e))
    
    # Test 5.3: Probability is in valid range
    try:
        log_test("Probability in valid range [0,1]", 0 <= prob <= 1, f"Prob: {prob:.4f}")
    except:
        log_test("Probability in valid range [0,1]", False, "No probability available")
    
    # Test 5.4: Batch prediction works
    try:
        batch_data = pd.DataFrame([sample_normal] * 10 + [sample_fraud] * 5)
        for col in ['TransactionType', 'Location', 'Channel', 'MerchantCategory', 'CustomerOccupation']:
            if col in label_encoders:
                try:
                    batch_data[col] = label_encoders[col].transform(batch_data[col])
                except:
                    batch_data[col] = 0
        
        batch_data = batch_data[feature_names]
        batch_scaled = scaler.transform(batch_data)
        
        preds = model.predict(batch_scaled)
        probs = model.predict_proba(batch_scaled)[:, 1]
        
        log_test("Batch prediction works", len(preds) == 15, f"Predictions: {len(preds)}")
    except Exception as e:
        log_test("Batch prediction works", False, str(e))

print()

# ============================================
# 6. PREPROCESSING TESTS
# ============================================
print("=" * 70)
print("🔧 6. PREPROCESSING TESTS")
print("=" * 70)

# Test 6.1: All categorical columns have encoders
categorical_cols = ['TransactionType', 'Location', 'Channel', 'MerchantCategory', 'CustomerOccupation']
for col in categorical_cols:
    has_encoder = col in label_encoders
    log_test(f"Encoder exists for {col}", has_encoder)

# Test 6.2: Feature names match expected count
expected_features = 11
log_test("Correct number of features", len(feature_names) == expected_features,
        f"Expected {expected_features}, got {len(feature_names)}")

# Test 6.3: Scaler is fitted
if scaler:
    has_mean = hasattr(scaler, 'mean_')
    log_test("Scaler is fitted", has_mean, "Has mean_" if has_mean else "Not fitted")

print()

# ============================================
# 7. APP FILE TESTS
# ============================================
print("=" * 70)
print("🖥️ 7. APPLICATION FILES TESTS")
print("=" * 70)

# Test 7.1: Main app file exists
app_exists = os.path.exists('app_desktop.py')
log_test("Desktop app file exists", app_exists)

# Test 7.2: Training script exists
train_exists = os.path.exists('train_models.py')
log_test("Training script exists", train_exists)

# Test 7.3: Dataset generator exists
gen_exists = os.path.exists('generate_dataset.py')
log_test("Dataset generator exists", gen_exists)

# Test 7.4: Requirements file exists
req_exists = os.path.exists('requirements.txt')
log_test("Requirements file exists", req_exists)

# Test 7.5: Build spec file exists
build_exists = os.path.exists('build.spec')
log_test("Build spec file exists", build_exists)

# Test 7.6: Notebook exists
notebook_exists = os.path.exists('notebooks/fraud_detection_ml.ipynb')
log_test("Jupyter notebook exists", notebook_exists)

# Test 7.7: README exists
readme_exists = os.path.exists('README.md')
log_test("README file exists", readme_exists)

print()

# ============================================
# 8. EDGE CASE TESTS
# ============================================
print("=" * 70)
print("⚠️ 8. EDGE CASE TESTS")
print("=" * 70)

if model and scaler and label_encoders:
    # Test 8.1: Very high amount
    try:
        edge_sample = {
            'TransactionAmount': 100000.00,
            'hour': 12, 'TransactionType': 'Transfer', 'Location': 'Chicago',
            'Channel': 'Online', 'MerchantCategory': 'Travel', 'CustomerAge': 40,
            'CustomerOccupation': 'Businessman', 'TransactionDuration': 30,
            'LoginAttempts': 1, 'AccountBalance': 200000.00
        }
        df_edge = pd.DataFrame([edge_sample])
        for col in categorical_cols:
            if col in label_encoders:
                try:
                    df_edge[col] = label_encoders[col].transform(df_edge[col])
                except:
                    df_edge[col] = 0
        df_edge = df_edge[feature_names]
        df_scaled = scaler.transform(df_edge)
        pred = model.predict(df_scaled)[0]
        log_test("Handles very high amount", True, f"Amount: $100K, Pred: {pred}")
    except Exception as e:
        log_test("Handles very high amount", False, str(e))
    
    # Test 8.2: Minimum age
    try:
        edge_sample['CustomerAge'] = 18
        edge_sample['TransactionAmount'] = 50
        df_edge = pd.DataFrame([edge_sample])
        for col in categorical_cols:
            if col in label_encoders:
                try:
                    df_edge[col] = label_encoders[col].transform(df_edge[col])
                except:
                    df_edge[col] = 0
        df_edge = df_edge[feature_names]
        df_scaled = scaler.transform(df_edge)
        pred = model.predict(df_scaled)[0]
        log_test("Handles minimum age (18)", True, f"Pred: {pred}")
    except Exception as e:
        log_test("Handles minimum age (18)", False, str(e))
    
    # Test 8.3: Maximum login attempts
    try:
        edge_sample['LoginAttempts'] = 10
        df_edge = pd.DataFrame([edge_sample])
        for col in categorical_cols:
            if col in label_encoders:
                try:
                    df_edge[col] = label_encoders[col].transform(df_edge[col])
                except:
                    df_edge[col] = 0
        df_edge = df_edge[feature_names]
        df_scaled = scaler.transform(df_edge)
        pred = model.predict(df_scaled)[0]
        prob = model.predict_proba(df_scaled)[0][1]
        log_test("Handles high login attempts", True, f"Attempts: 10, Prob: {prob*100:.1f}%")
    except Exception as e:
        log_test("Handles high login attempts", False, str(e))

print()

# ============================================
# 9. CONSISTENCY TESTS
# ============================================
print("=" * 70)
print("🔄 9. CONSISTENCY TESTS")
print("=" * 70)

if model:
    # Test 9.1: Same input gives same output
    try:
        consistent_sample = {
            'TransactionAmount': 100.00, 'hour': 10, 
            'TransactionType': 'Debit', 'Location': 'Chicago',
            'Channel': 'ATM', 'MerchantCategory': 'Grocery', 'CustomerAge': 30,
            'CustomerOccupation': 'Doctor', 'TransactionDuration': 60,
            'LoginAttempts': 1, 'AccountBalance': 3000.00
        }
        
        df_test = pd.DataFrame([consistent_sample])
        for col in categorical_cols:
            if col in label_encoders:
                try:
                    df_test[col] = label_encoders[col].transform(df_test[col])
                except:
                    df_test[col] = 0
        df_test = df_test[feature_names]
        df_scaled = scaler.transform(df_test)
        
        pred1 = model.predict(df_scaled)[0]
        prob1 = model.predict_proba(df_scaled)[0][1]
        pred2 = model.predict(df_scaled)[0]
        prob2 = model.predict_proba(df_scaled)[0][1]
        
        consistent = (pred1 == pred2) and (abs(prob1 - prob2) < 0.0001)
        log_test("Predictions are deterministic", consistent, f"Prob1: {prob1:.4f}, Prob2: {prob2:.4f}")
    except Exception as e:
        log_test("Predictions are deterministic", False, str(e))

print()

# ============================================
# SUMMARY
# ============================================
print("=" * 70)
print("📊 TEST SUMMARY")
print("=" * 70)
total = TESTS_PASSED + TESTS_FAILED
print(f"\n✅ Passed: {TESTS_PASSED}/{total}")
print(f"❌ Failed: {TESTS_FAILED}/{total}")
print(f"📈 Success Rate: {(TESTS_PASSED/total)*100:.1f}%")

if TESTS_FAILED > 0:
    print("\n❌ Failed Tests:")
    for name, passed, details in TEST_RESULTS:
        if not passed:
            print(f"   - {name}: {details}")

print()
print("=" * 70)
print(f"🏁 Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 70)

# Exit with error code if any tests failed
sys.exit(0 if TESTS_FAILED == 0 else 1)
