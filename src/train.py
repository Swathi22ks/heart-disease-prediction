"""
train.py — Train and save all models
Run: python src/train.py
Author: Swathi K S
"""

import pickle
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import xgboost as xgb

from preprocess import load_and_clean, get_preprocessed_splits

DATA_PATH   = 'data/framingham.csv'
MODELS_PATH = 'models/'


def get_models():
    return {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Decision Tree':       DecisionTreeClassifier(max_depth=5, random_state=42),
        'Random Forest':       RandomForestClassifier(n_estimators=200, random_state=42),
        'Gradient Boosting':   GradientBoostingClassifier(n_estimators=200, random_state=42),
        'XGBoost':             xgb.XGBClassifier(eval_metric='logloss', random_state=42),
    }


def train_all():
    print("Loading data...")
    df = load_and_clean(DATA_PATH)
    X_train, X_test, y_train, y_test, imputer, scaler = get_preprocessed_splits(df)

    models = get_models()
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)

    # Save best model (highest AUC-ROC = Logistic Regression)
    best_model = models['Logistic Regression']
    pickle.dump(best_model, open(f'{MODELS_PATH}best_model.pkl', 'wb'))
    pickle.dump(scaler,     open(f'{MODELS_PATH}scaler.pkl',     'wb'))
    pickle.dump(imputer,    open(f'{MODELS_PATH}imputer.pkl',    'wb'))

    print("✅ Models saved to /models/")
    return models, imputer, scaler, X_test, y_test


if __name__ == '__main__':
    train_all()
