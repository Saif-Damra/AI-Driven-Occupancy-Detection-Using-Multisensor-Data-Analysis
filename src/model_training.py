"""
Model Training Module for Occupancy Detection
===============================================
Trains and evaluates multiple ML models with GridSearchCV
for hyperparameter tuning and cross-validation.

Models:
- Random Forest
- Gradient Boosting
- XGBoost
- SVM
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix,
)

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("⚠️  XGBoost not installed. Skipping XGBoost model.")


def get_param_grids() -> dict:
    """Return parameter grids for GridSearchCV."""
    grids = {
        "Random Forest": {
            "model": RandomForestClassifier(random_state=42),
            "params": {
                "n_estimators": [50, 100, 200],
                "max_depth": [None, 10, 20],
                "max_features": ["sqrt", "log2"],
                "criterion": ["gini", "entropy"],
            },
        },
        "Gradient Boosting": {
            "model": GradientBoostingClassifier(random_state=42),
            "params": {
                "n_estimators": [50, 100, 200],
                "learning_rate": [0.01, 0.1, 0.2],
                "max_depth": [3, 5, 7],
            },
        },
        "SVM": {
            "model": SVC(random_state=42),
            "params": {
                "C": [0.1, 1, 10],
                "kernel": ["rbf", "poly", "linear"],
                "gamma": ["scale", "auto"],
            },
        },
    }
    
    if HAS_XGBOOST:
        grids["XGBoost"] = {
            "model": XGBClassifier(
                random_state=42, use_label_encoder=False,
                eval_metric="mlogloss", verbosity=0,
            ),
            "params": {
                "n_estimators": [50, 100, 200],
                "learning_rate": [0.01, 0.1, 0.2],
                "max_depth": [3, 5, 7],
            },
        }
    
    return grids


def train_with_grid_search(X_train, y_train, X_test, y_test, cv: int = 5):
    """
    Train all models with GridSearchCV for hyperparameter tuning.
    
    Returns
    -------
    tuple: (results_df, detailed_results)
    """
    print("\n" + "=" * 60)
    print("🤖 MODEL TRAINING WITH GRIDSEARCHCV")
    print(f"   Method: {cv}-Fold Cross Validation")
    print("=" * 60)
    
    grids = get_param_grids()
    results = {}
    detailed_results = {}
    
    for name, config in grids.items():
        print(f"\n📌 Training: {name}...")
        print("-" * 40)
        
        # GridSearchCV
        grid_search = GridSearchCV(
            config["model"], config["params"],
            cv=cv, scoring="accuracy", n_jobs=-1, verbose=0,
        )
        grid_search.fit(X_train, y_train)
        
        # Best params
        print(f"   Best Parameters: {grid_search.best_params_}")
        print(f"   Best CV Score: {grid_search.best_score_:.4f}")
        
        # Predict on test set
        y_pred = grid_search.predict(X_test)
        
        # Metrics (macro average for multiclass)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="macro")
        prec = precision_score(y_test, y_pred, average="macro")
        rec = recall_score(y_test, y_pred, average="macro")
        
        print(f"   Test Accuracy:  {acc:.4f}")
        print(f"   Test Precision: {prec:.4f}")
        print(f"   Test Recall:    {rec:.4f}")
        print(f"   Test F1-Score:  {f1:.4f}")
        
        results[name] = {
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1-Score": f1,
        }
        
        detailed_results[name] = {
            "y_test": y_test,
            "y_pred": y_pred,
            "confusion_matrix": confusion_matrix(y_test, y_pred),
            "classification_report": classification_report(y_test, y_pred),
            "best_params": grid_search.best_params_,
            "best_score": grid_search.best_score_,
            "model": grid_search.best_estimator_,
        }
    
    # Build comparison DataFrames
    results_df = pd.DataFrame(results)
    
    # Find best model
    best_model = max(results.keys(), key=lambda k: results[k]["F1-Score"])
    best_f1 = results[best_model]["F1-Score"]
    
    print("\n" + "=" * 60)
    print("📊 MODEL COMPARISON RESULTS")
    print("=" * 60)
    print(results_df.round(4).to_string())
    print(f"\n🏆 Best Model: {best_model} (F1-Score: {best_f1:.4f})")
    print("=" * 60)
    
    # Print detailed classification reports
    print("\n" + "=" * 60)
    print("🔍 DETAILED CLASSIFICATION REPORTS")
    print("=" * 60)
    for name, result in detailed_results.items():
        print(f"\n📌 {name}:")
        print(f"   Best Parameters: {result['best_params']}")
        print(f"   Confusion Matrix:\n{result['confusion_matrix']}")
        print(f"\n   Classification Report:\n{result['classification_report']}")
    
    return results_df, detailed_results
