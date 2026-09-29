import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Create directories
dirs = ['data/raw', 'data/cleaned', 'data/processed', 'data/final', 'reports']
for d in dirs:
    os.makedirs(d, exist_ok=True)

# 1. DATASET AUDIT
file_path = "drug_abuse_dataset (2).csv"
df = pd.read_csv(file_path)

# Save raw
df.to_csv('data/raw/raw_dataset.csv', index=False)

audit_data = []
for col in df.columns:
    audit_data.append({
        'Column': col,
        'DataType': df[col].dtype,
        'MissingValues': df[col].isna().sum(),
        'MissingPercentage': (df[col].isna().sum() / len(df)) * 100,
        'UniqueValues': df[col].nunique(),
        'ExampleValues': str(df[col].unique()[:5])
    })

audit_df = pd.DataFrame(audit_data)
audit_df.to_csv('reports/dataset_audit.csv', index=False)

print("Rows:", len(df))
print("Columns:", len(df.columns))
print("Duplicates:", df.duplicated().sum())

print("\nTarget Distribution:")
print(df['Drug_Abuse_Risk'].value_counts())
print("\nTarget Percentage:")
print(df['Drug_Abuse_Risk'].value_counts(normalize=True)*100)

# Check target relation with substance use variables
vars_to_check = ['Type_of_Substance_Used', 'Frequency_of_Use', 'Duration_of_Use', 
                 'Craving_Level', 'Withdrawal_Symptoms', 'Previous_Rehabilitation_Counselling',
                 'Previous_Substance_Exposure', 'Age_First_Substance_Exposure', 
                 'Substance_Exposure_Frequency', 'Polysubstance_Exposure',
                 'Substance_Availability_Exposure', 'Substance_Use_Intention']

for var in vars_to_check:
    if var in df.columns:
        print(f"\n--- {var} vs Target ---")
        print(pd.crosstab(df[var], df['Drug_Abuse_Risk']))

