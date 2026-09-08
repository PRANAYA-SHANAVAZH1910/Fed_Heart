"""
Generic Data Preprocessing Pipeline using scikit-learn
--------------------------------------------------------
Handles:
  - Loading a CSV dataset
  - Splitting features/target
  - Missing value imputation (numeric + categorical)
  - Categorical encoding (One-Hot)
  - Feature scaling (Standardization)
  - Train/test split
  - A reusable ColumnTransformer + Pipeline you can plug into any model

Just change DATA_PATH, TARGET_COLUMN, and the column lists below to fit
your dataset.
"""

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# ---------------------------------------------------------------------
# 1. CONFIG — edit these for your dataset
# ---------------------------------------------------------------------
DATA_PATH = "/Users/pranaya.shanavazh/Desktop/Research Paper/framingham_heart_study.csv"  # path to your CSV file
TARGET_COLUMN = "TenYearCHD"       # name of the column you want to predict
TEST_SIZE = 0.2

# ---------------------------------------------------------------------
# 2. LOAD DATA
# ---------------------------------------------------------------------
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"Loaded dataset with shape: {df.shape}")
    print(df.head())
    return df


# ---------------------------------------------------------------------
# 3. IDENTIFY COLUMN TYPES AUTOMATICALLY
# ---------------------------------------------------------------------
def get_column_types(df: pd.DataFrame, target_column: str):
    features = df.drop(columns=[target_column])
    numeric_cols = features.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = features.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    print(f"Numeric columns: {numeric_cols}")
    print(f"Categorical columns: {categorical_cols}")
    return numeric_cols, categorical_cols


# ---------------------------------------------------------------------
# 4. BUILD PREPROCESSING PIPELINE
# ---------------------------------------------------------------------
def build_preprocessor(numeric_cols, categorical_cols) -> ColumnTransformer:
    # Numeric pipeline: fill missing values with median, then scale
    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    # Categorical pipeline: fill missing values with most frequent, then one-hot encode
    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_pipeline, numeric_cols),
        ("cat", categorical_pipeline, categorical_cols),
    ])

    return preprocessor


# ---------------------------------------------------------------------
# 5. MAIN WORKFLOW
# ---------------------------------------------------------------------
def main():
    # Load
    df = load_data(DATA_PATH)

    # Drop rows where target is missing (can't train/evaluate on these)
    df = df.dropna(subset=[TARGET_COLUMN])

    # Split features/target
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    # Identify column types
    numeric_cols, categorical_cols = get_column_types(df, TARGET_COLUMN)

    # Train/test split: take the LAST 20% of rows (in file order) as the test set,
    # instead of a random split. This makes the split reproducible and consistent
    # across runs, rather than randomly sampled each time.
    split_idx = int(len(df) * (1 - TEST_SIZE))
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]

    X_train = train_df.drop(columns=[TARGET_COLUMN])
    y_train = train_df[TARGET_COLUMN]
    X_test = test_df.drop(columns=[TARGET_COLUMN])
    y_test = test_df[TARGET_COLUMN]

    print(f"Train rows: {len(X_train)}, Test rows: {len(X_test)} (last {int(TEST_SIZE*100)}% of the data)")

    # Build preprocessing pipeline
    preprocessor = build_preprocessor(numeric_cols, categorical_cols)

    # Fit on training data only, then transform both sets (avoids data leakage)
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    print(f"\nProcessed training set shape: {X_train_processed.shape}")
    print(f"Processed test set shape: {X_test_processed.shape}")

    return X_train_processed, X_test_processed, y_train, y_test, preprocessor


if __name__ == "__main__":
    X_train, X_test, y_train, y_test, preprocessor = main()
    import numpy as np

print("\n--- Verification ---")
print("Any NaNs left in training data?", np.isnan(X_train).any())
print("Mean of first column (should be ~0 after scaling):", X_train[:, 0].mean())
print("Std of first column (should be ~1 after scaling):", X_train[:, 0].std())

# Get the feature names after preprocessing (handles the one-hot encoded names too)
feature_names = preprocessor.get_feature_names_out()

# Convert back into DataFrames for viewing
X_train_df = pd.DataFrame(X_train, columns=feature_names)
X_test_df = pd.DataFrame(X_test, columns=feature_names)

print("\n--- Preprocessed Training Data (first 10 rows) ---")
print(X_train_df.head(10))
X_train_df.to_csv("preprocessed_train_data.csv", index=False)
print("Saved to preprocessed_train_data.csv")

    # Example: plug straight into a model
    # from sklearn.ensemble import RandomForestClassifier
    # model = RandomForestClassifier(random_state=42)
    # model.fit(X_train, y_train)
    # print("Test accuracy:", model.score(X_test, y_test))

