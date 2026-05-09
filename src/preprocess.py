"""
preprocess.py — Reusable preprocessing pipeline
Author: Swathi K S
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE


FEATURES = [
    'age', 'gender', 'currentSmoker', 'cigsPerDay', 'BPMeds',
    'prevalentStroke', 'prevalentHyp', 'diabetes', 'totChol',
    'sysBP', 'diaBP', 'BMI', 'heartRate', 'glucose'
]
TARGET = 'TenYearCHD'


def load_and_clean(filepath: str) -> pd.DataFrame:
    """Load the Framingham dataset and perform initial cleaning."""
    df = pd.read_csv(filepath)
    df.drop(columns=['education'], errors='ignore', inplace=True)
    df.rename(columns={'male': 'gender'}, inplace=True)
    return df


def get_preprocessed_splits(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """
    Full preprocessing pipeline:
    1. Stratified train/test split
    2. Median imputation (fit on train only)
    3. SMOTE oversampling on train
    4. StandardScaler normalization (fit on train only)

    Returns: X_train_sc, X_test_sc, y_train_sm, y_test, imputer, scaler
    """
    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Impute
    imputer = SimpleImputer(strategy='median')
    X_train_imp = imputer.fit_transform(X_train)
    X_test_imp  = imputer.transform(X_test)

    # SMOTE
    smote = SMOTE(random_state=random_state)
    X_train_sm, y_train_sm = smote.fit_resample(X_train_imp, y_train)

    # Scale
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train_sm)
    X_test_sc  = scaler.transform(X_test_imp)

    return X_train_sc, X_test_sc, y_train_sm, y_test, imputer, scaler


def preprocess_single(input_dict: dict, imputer, scaler) -> np.ndarray:
    """Preprocess a single patient record for inference."""
    row = pd.DataFrame([input_dict])[FEATURES]
    row_imp = imputer.transform(row)
    row_sc  = scaler.transform(row_imp)
    return row_sc
