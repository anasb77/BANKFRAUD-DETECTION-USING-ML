"""
Synthetic Fraud Detection Dataset Generator
Generates realistic transaction data with interpretable features for ML training.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import os

# Seed for reproducibility
np.random.seed(42)
random.seed(42)

# ============================================
# Configuration
# ============================================
NUM_TRANSACTIONS = 50000  # 50K transactions
FRAUD_RATE = 0.035  # 3.5% fraud rate (realistic)

# Feature definitions
LOCATIONS = [
    'New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix', 'Philadelphia',
    'San Antonio', 'San Diego', 'Dallas', 'San Jose', 'Austin', 'Jacksonville',
    'Fort Worth', 'Columbus', 'Charlotte', 'San Francisco', 'Indianapolis',
    'Seattle', 'Denver', 'Washington', 'Boston', 'Nashville', 'Detroit',
    'Portland', 'Las Vegas', 'Memphis', 'Louisville', 'Baltimore', 'Milwaukee',
    'Albuquerque', 'Tucson', 'Fresno', 'Sacramento', 'Kansas City', 'Atlanta',
    'Miami', 'Omaha', 'Raleigh', 'Colorado Springs', 'Virginia Beach'
]

OCCUPATIONS = ['Student', 'Engineer', 'Doctor', 'Retired', 'Businessman', 'Teacher', 'Lawyer', 'Artist', 'Nurse', 'Unemployed']
TRANSACTION_TYPES = ['Debit', 'Credit', 'Transfer', 'Withdrawal', 'Deposit']
CHANNELS = ['Online', 'ATM', 'Branch', 'Mobile', 'POS']
MERCHANT_CATEGORIES = ['Grocery', 'Restaurant', 'Gas Station', 'Electronics', 'Clothing', 'Travel', 'Entertainment', 'Healthcare', 'Utilities', 'Other']

# ============================================
# Generator Functions
# ============================================

def generate_normal_transaction():
    """Generate a normal (non-fraud) transaction"""
    age = np.random.choice([
        np.random.randint(18, 30),  # Young
        np.random.randint(30, 50),  # Middle
        np.random.randint(50, 80),  # Senior
    ], p=[0.3, 0.45, 0.25])
    
    occupation = np.random.choice(OCCUPATIONS, p=[0.15, 0.2, 0.1, 0.15, 0.1, 0.1, 0.05, 0.05, 0.05, 0.05])
    
    # Amount based on occupation (realistic patterns)
    if occupation == 'Student':
        amount = np.random.exponential(50) + 5
    elif occupation in ['Doctor', 'Lawyer', 'Businessman']:
        amount = np.random.exponential(200) + 20
    elif occupation == 'Retired':
        amount = np.random.exponential(100) + 10
    else:
        amount = np.random.exponential(80) + 10
    
    amount = min(amount, 5000)  # Cap at 5000
    
    # Normal transactions happen during business hours
    hour = np.random.choice(range(6, 23), p=[
        0.02, 0.05, 0.08, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.08, 
        0.05, 0.04, 0.03, 0.02, 0.01, 0.01, 0.01
    ])
    
    return {
        'TransactionAmount': round(amount, 2),
        'hour': hour,
        'TransactionType': np.random.choice(TRANSACTION_TYPES, p=[0.35, 0.25, 0.15, 0.15, 0.10]),
        'Location': np.random.choice(LOCATIONS),
        'Channel': np.random.choice(CHANNELS, p=[0.35, 0.25, 0.15, 0.15, 0.10]),
        'CustomerAge': age,
        'CustomerOccupation': occupation,
        'MerchantCategory': np.random.choice(MERCHANT_CATEGORIES),
        'TransactionDuration': np.random.randint(10, 180),
        'LoginAttempts': np.random.choice([1, 1, 1, 1, 2], p=[0.9, 0.025, 0.025, 0.025, 0.025]),
        'AccountBalance': round(np.random.exponential(5000) + 100, 2),
        'is_fraud': 0
    }


def generate_fraud_transaction():
    """Generate a fraudulent transaction with suspicious patterns"""
    # Fraud patterns
    fraud_patterns = np.random.choice([
        'high_amount',      # Unusually high amount
        'unusual_time',     # Late night transaction
        'multiple_attempts', # Multiple login attempts
        'new_location',     # Unusual location
        'rapid_succession', # Quick transactions
        'suspicious_combo'  # Multiple red flags
    ], p=[0.2, 0.15, 0.15, 0.15, 0.15, 0.2])
    
    tx = generate_normal_transaction()
    tx['is_fraud'] = 1
    
    if fraud_patterns == 'high_amount':
        tx['TransactionAmount'] = round(np.random.uniform(2000, 10000), 2)
        
    elif fraud_patterns == 'unusual_time':
        tx['hour'] = np.random.choice([0, 1, 2, 3, 4, 5, 23])
        tx['TransactionAmount'] = round(np.random.uniform(500, 3000), 2)
        
    elif fraud_patterns == 'multiple_attempts':
        tx['LoginAttempts'] = np.random.choice([3, 4, 5, 6, 7])
        tx['TransactionAmount'] = round(np.random.uniform(300, 2000), 2)
        
    elif fraud_patterns == 'new_location':
        # Use less common location
        tx['Location'] = np.random.choice(['Anchorage', 'Honolulu', 'Fargo', 'Cheyenne', 'Pierre'])
        tx['TransactionAmount'] = round(np.random.uniform(400, 2500), 2)
        
    elif fraud_patterns == 'rapid_succession':
        tx['TransactionDuration'] = np.random.randint(1, 15)  # Very quick
        tx['TransactionAmount'] = round(np.random.uniform(200, 1500), 2)
        
    else:  # suspicious_combo
        tx['TransactionAmount'] = round(np.random.uniform(1000, 8000), 2)
        tx['hour'] = np.random.choice([0, 1, 2, 3, 4, 5, 23])
        tx['LoginAttempts'] = np.random.choice([2, 3, 4, 5])
        tx['TransactionDuration'] = np.random.randint(1, 30)
    
    # Additional fraud indicators
    if np.random.random() < 0.4:
        tx['Channel'] = 'Online'  # Online fraud is common
    
    return tx


def generate_dataset(num_transactions, fraud_rate):
    """Generate complete dataset"""
    transactions = []
    num_fraud = int(num_transactions * fraud_rate)
    num_normal = num_transactions - num_fraud
    
    print(f"Generating {num_normal:,} normal transactions...")
    for _ in range(num_normal):
        transactions.append(generate_normal_transaction())
    
    print(f"Generating {num_fraud:,} fraudulent transactions...")
    for _ in range(num_fraud):
        transactions.append(generate_fraud_transaction())
    
    # Shuffle
    random.shuffle(transactions)
    
    # Create DataFrame
    df = pd.DataFrame(transactions)
    
    # Add IDs and dates
    df['TransactionID'] = [f'TX{str(i).zfill(6)}' for i in range(1, len(df) + 1)]
    df['AccountID'] = [f'AC{str(np.random.randint(1, 10000)).zfill(5)}' for _ in range(len(df))]
    
    # Generate dates (last 2 years)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=730)
    dates = [start_date + timedelta(
        days=np.random.randint(0, 730),
        hours=row['hour'],
        minutes=np.random.randint(0, 60),
        seconds=np.random.randint(0, 60)
    ) for _, row in df.iterrows()]
    df['TransactionDate'] = dates
    
    # Reorder columns
    columns = [
        'TransactionID', 'AccountID', 'TransactionAmount', 'TransactionDate', 
        'hour', 'TransactionType', 'Location', 'Channel', 'MerchantCategory',
        'CustomerAge', 'CustomerOccupation', 'TransactionDuration', 
        'LoginAttempts', 'AccountBalance', 'is_fraud'
    ]
    df = df[columns]
    
    return df


# ============================================
# Main
# ============================================
if __name__ == "__main__":
    print("=" * 60)
    print("🏦 Synthetic Fraud Detection Dataset Generator")
    print("=" * 60)
    
    # Generate dataset
    df = generate_dataset(NUM_TRANSACTIONS, FRAUD_RATE)
    
    # Save to CSV
    output_path = 'data/fraud_dataset.csv'
    os.makedirs('data', exist_ok=True)
    df.to_csv(output_path, index=False)
    
    # Summary
    print(f"\n✅ Dataset generated successfully!")
    print(f"📁 Saved to: {output_path}")
    print(f"\n📊 Dataset Summary:")
    print(f"   Total transactions: {len(df):,}")
    print(f"   Normal transactions: {(df['is_fraud'] == 0).sum():,}")
    print(f"   Fraudulent transactions: {(df['is_fraud'] == 1).sum():,}")
    print(f"   Fraud rate: {df['is_fraud'].mean()*100:.2f}%")
    print(f"\n📋 Features:")
    for col in df.columns:
        print(f"   - {col}")
    print(f"\n📈 Sample rows:")
    print(df.head(10).to_string())
