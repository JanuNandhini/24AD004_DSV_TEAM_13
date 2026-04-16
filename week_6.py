# ================================
# 1. IMPORT LIBRARIES
# ================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder

# ================================
# 2. LOAD DATASET
# ================================
df = pd.read_csv(r"C:\Users\welcome\Downloads\final_cleaned_crime_data.csv")

print("Initial Shape:", df.shape)
print(df.head())

# ================================
# 3. IDENTIFY COLUMN TYPES
# ================================
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = df.select_dtypes(include=['object']).columns.tolist()

print("\nNumerical Columns:", num_cols)
print("\nCategorical Columns:", cat_cols)

# ================================
# 4. HANDLE SKEWNESS
# ================================
# Apply log1p only to highly skewed numerical columns
skewness = df[num_cols].skew().sort_values(ascending=False)

# Select columns with high skewness (> 0.75)
skewed_cols = skewness[skewness > 0.75].index.tolist()

print("\nSkewed Columns:", skewed_cols)

# Apply log transformation
for col in skewed_cols:
    df[col] = np.log1p(df[col])

# ================================
# 5. ENCODING
# ================================

# --- A. LABEL ENCODING (High Cardinality) ---
label_encoded_df = df.copy()

for col in cat_cols:
    if df[col].nunique() > 10:
        le = LabelEncoder()
        label_encoded_df[col] = le.fit_transform(df[col].astype(str))

# --- B. ONE-HOT ENCODING (Low Cardinality) ---
one_hot_cols = [col for col in cat_cols if df[col].nunique() <= 10]

df_encoded = pd.get_dummies(label_encoded_df, columns=one_hot_cols, drop_first=True)

print("\nAfter Encoding Shape:", df_encoded.shape)

# ================================
# 6. FEATURE SCALING
# ================================

# Separate numerical columns again after encoding
num_cols_final = df_encoded.select_dtypes(include=[np.number]).columns.tolist()

# --- A. STANDARDIZATION ---
scaler_std = StandardScaler()
df_std = df_encoded.copy()
df_std[num_cols_final] = scaler_std.fit_transform(df_encoded[num_cols_final])

# --- B. MIN-MAX SCALING ---
scaler_minmax = MinMaxScaler()
df_minmax = df_encoded.copy()
df_minmax[num_cols_final] = scaler_minmax.fit_transform(df_encoded[num_cols_final])

# ================================
# 7. VISUALIZATION (BEFORE VS AFTER)
# ================================

# Pick a few important numeric columns for comparison
sample_cols = num_cols[:4] if len(num_cols) >= 4 else num_cols

for col in sample_cols:
    plt.figure(figsize=(15,5))

    # BEFORE
    plt.subplot(1,3,1)
    sns.histplot(df[col], kde=True)
    plt.title(f'{col} - Before')

    # STANDARD SCALED
    plt.subplot(1,3,2)
    sns.histplot(df_std[col], kde=True)
    plt.title(f'{col} - Standard Scaled')

    # MINMAX SCALED
    plt.subplot(1,3,3)
    sns.histplot(df_minmax[col], kde=True)
    plt.title(f'{col} - MinMax Scaled')

    plt.tight_layout()
    plt.savefig(f'comparison_{col}.png')
    plt.show()

# ================================
# 8. SAVE FINAL DATASETS
# ================================

df_std.to_csv('final_standard_scaled_data.csv', index=False)
df_minmax.to_csv('final_minmax_scaled_data.csv', index=False)

print("\nFinal Standard Scaled Shape:", df_std.shape)
print("Final MinMax Scaled Shape:", df_minmax.shape)

print("\n✅ Feature Engineering Completed Successfully!")