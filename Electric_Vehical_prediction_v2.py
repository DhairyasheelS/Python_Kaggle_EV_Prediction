import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier

def LoadDataset(train_csv, test_csv):
    train = pd.read_csv(train_csv)
    test = pd.read_csv(test_csv)
    print("Dataset loaded successfully!!")
    return train, test

def AdvancedFeatureEngineering(X_train, X_test):
    """
    ADVANCED FEATURE ENGINEERING

    Strategies to gain 0.005+ ROC-AUC:
    1. Polynomial features for key variables
    2. More interaction features
    3. Binning/discretization
    4. Target encoding proxies
    5. Statistical aggregations
    """

    print("\n[ADVANCED FEATURES] Creating additional features...")

    # ========== 1. POLYNOMIAL FEATURES ==========
    # Square of key predictors
    X_train['Income_Squared'] = X_train['Annual_Income_USD'] ** 2
    X_test['Income_Squared'] = X_test['Annual_Income_USD'] ** 2

    X_train['Age_Squared'] = X_train['Age'] ** 2
    X_test['Age_Squared'] = X_test['Age'] ** 2

    X_train['Commute_Squared'] = X_train['Daily_Commute_km'] ** 2
    X_test['Commute_Squared'] = X_test['Daily_Commute_km'] ** 2

    # ========== 2. MORE INTERACTION FEATURES ==========

    # Age groups (younger people more likely to adopt EV) - convert to int
    X_train['Age_Group'] = pd.cut(X_train['Age'], bins=[0, 35, 50, 70], labels=[0, 1, 2]).astype(int)
    X_test['Age_Group'] = pd.cut(X_test['Age'], bins=[0, 35, 50, 70], labels=[0, 1, 2]).astype(int)

    # Income brackets - convert to int
    X_train['Income_Bracket'] = pd.cut(X_train['Annual_Income_USD'],
                                        bins=[0, 60000, 100000, 200000], labels=[0, 1, 2]).astype(int)
    X_test['Income_Bracket'] = pd.cut(X_test['Annual_Income_USD'],
                                       bins=[0, 60000, 100000, 200000], labels=[0, 1, 2]).astype(int)

    # Charging accessibility (home + work)
    X_train['Has_Home_Charging'] = (X_train['Home_Charging_Possible'] == 'Yes').astype(int)
    X_test['Has_Home_Charging'] = (X_test['Home_Charging_Possible'] == 'Yes').astype(int)

    # Environmental + Subsidy interaction
    X_train['Eco_Subsidy_Interaction'] = (
        X_train['Environmental_Concern_Level'] *
        (X_train['Subsidy_Available'] == 'Yes').astype(int)
    )
    X_test['Eco_Subsidy_Interaction'] = (
        X_test['Environmental_Concern_Level'] *
        (X_test['Subsidy_Available'] == 'Yes').astype(int)
    )

    # Total charging stations (weighted by importance)
    X_train['Total_Charging_Weighted'] = (
        X_train['Charging_Stations_Near_Home'] * 1.5 +
        X_train['Charging_Stations_Near_Work']
    )
    X_test['Total_Charging_Weighted'] = (
        X_test['Charging_Stations_Near_Home'] * 1.5 +
        X_test['Charging_Stations_Near_Work']
    )

    # Commute intensity ratio
    X_train['Commute_Income_Ratio'] = X_train['Daily_Commute_km'] / (X_train['Annual_Income_USD'] / 10000)
    X_test['Commute_Income_Ratio'] = X_test['Daily_Commute_km'] / (X_test['Annual_Income_USD'] / 10000)

    # Urban + high charging stations
    X_train['Urban_High_Charging'] = (
        (X_train['City_Type'] == 'Urban').astype(int) *
        (X_train['Charging_Stations_Near_Home'] > 5).astype(int)
    )
    X_test['Urban_High_Charging'] = (
        (X_test['City_Type'] == 'Urban').astype(int) *
        (X_test['Charging_Stations_Near_Home'] > 5).astype(int)
    )

    # Environmental concern binary
    X_train['High_Eco_Concern'] = (X_train['Environmental_Concern_Level'] >= 4).astype(int)
    X_test['High_Eco_Concern'] = (X_test['Environmental_Concern_Level'] >= 4).astype(int)

    # Affordability index (income vs cars owned)
    X_train['Affordability_Index'] = X_train['Annual_Income_USD'] / (X_train['Number_of_Cars_Owned'] * 30000)
    X_test['Affordability_Index'] = X_test['Annual_Income_USD'] / (X_test['Number_of_Cars_Owned'] * 30000)

    # Range anxiety + commute interaction
    X_train['Anxiety_Commute'] = (
        (X_train['Range_Anxiety_Level'] == 'High').astype(int) * X_train['Daily_Commute_km']
    )
    X_test['Anxiety_Commute'] = (
        (X_test['Range_Anxiety_Level'] == 'High').astype(int) * X_test['Daily_Commute_km']
    )

    print(f"  [+] Created {X_train.shape[1] - 13} additional features")
    print(f"  New shape - Train: {X_train.shape}, Test: {X_test.shape}")

    return X_train, X_test

def PreProcessData(train_data, test_data):
    """Enhanced preprocessing with advanced feature engineering"""

    print("="*60)
    print("ENHANCED PREPROCESSING PIPELINE")
    print("="*60)

    # Store test IDs
    test_ids = test_data['id'].copy()

    # Separate target
    y_train = train_data['Will_Buy_EV'].map({'No': 0, 'Yes': 1})

    # Drop id and target
    X_train = train_data.drop(['id', 'Will_Buy_EV'], axis=1).copy()
    X_test = test_data.drop(['id'], axis=1).copy()

    print(f"Initial shape - Train: {X_train.shape}, Test: {X_test.shape}")

    # ========== ADVANCED FEATURE ENGINEERING ==========
    X_train, X_test = AdvancedFeatureEngineering(X_train, X_test)

    # ========== ENCODING ==========
    print("\n[ENCODING] Encoding categorical variables...")

    # Binary encoding
    binary_map = {'Yes': 1, 'No': 0}
    X_train['Home_Charging_Possible'] = X_train['Home_Charging_Possible'].map(binary_map)
    X_test['Home_Charging_Possible'] = X_test['Home_Charging_Possible'].map(binary_map)
    X_train['Subsidy_Available'] = X_train['Subsidy_Available'].map(binary_map)
    X_test['Subsidy_Available'] = X_test['Subsidy_Available'].map(binary_map)

    # Ordinal encoding
    anxiety_map = {'Low': 0, 'Medium': 1, 'High': 2}
    X_train['Range_Anxiety_Level'] = X_train['Range_Anxiety_Level'].map(anxiety_map)
    X_test['Range_Anxiety_Level'] = X_test['Range_Anxiety_Level'].map(anxiety_map)

    # Label encoding
    for col in ['Gender', 'City_Type', 'Current_Car_Type']:
        le = LabelEncoder()
        X_train[col] = le.fit_transform(X_train[col])
        X_test[col] = le.transform(X_test[col])

    # ========== SCALING ==========
    print("\n[SCALING] Standardizing numerical features...")

    numerical_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()

    scaler = StandardScaler()
    X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
    X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])

    print(f"\n✓ Final shape - Train: {X_train.shape}, Test: {X_test.shape}")

    return X_train, y_train, X_test, test_ids

def TrainOptimizedModel(X_train, y_train, X_test, n_folds=7):
    """
    OPTIMIZED TRAINING STRATEGY

    Improvements:
    1. Increased folds (5→7) for better generalization
    2. Optimized hyperparameters
    3. Weighted ensemble (not simple average)
    4. More training iterations with early stopping
    """

    print("="*60)
    print("OPTIMIZED MODEL TRAINING")
    print("="*60)

    # Calculate class weight
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    scale_pos_weight = neg_count / pos_count

    print(f"\nClass imbalance ratio: {scale_pos_weight:.4f}")

    # Initialize CV
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    # Storage
    oof_xgb = np.zeros(len(X_train))
    oof_lgb = np.zeros(len(X_train))
    oof_cat = np.zeros(len(X_train))

    test_preds_xgb = np.zeros(len(X_test))
    test_preds_lgb = np.zeros(len(X_test))
    test_preds_cat = np.zeros(len(X_test))

    cv_scores = {'xgboost': [], 'lightgbm': [], 'catboost': []}

    print(f"\n[CROSS-VALIDATION] {n_folds}-Fold Stratified CV")

    # ========== CV LOOP ==========
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train), 1):

        print(f"\n{'='*20} FOLD {fold}/{n_folds} {'='*20}")

        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

        # ========== XGBoost (OPTIMIZED) ==========
        xgb_params = {
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'max_depth': 7,  # Increased from 6
            'learning_rate': 0.015,  # Decreased for finer steps
            'subsample': 0.85,
            'colsample_bytree': 0.85,
            'min_child_weight': 1,
            'gamma': 0.1,  # Added regularization
            'reg_alpha': 0.1,  # L1 regularization
            'reg_lambda': 1.0,  # L2 regularization
            'scale_pos_weight': scale_pos_weight,
            'tree_method': 'hist',
            'random_state': 42,
            'n_jobs': -1
        }

        model_xgb = xgb.XGBClassifier(**xgb_params, n_estimators=2000)
        model_xgb.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            verbose=False
        )

        oof_xgb[val_idx] = model_xgb.predict_proba(X_val)[:, 1]
        test_preds_xgb += model_xgb.predict_proba(X_test)[:, 1] / n_folds

        xgb_score = roc_auc_score(y_val, oof_xgb[val_idx])
        cv_scores['xgboost'].append(xgb_score)
        print(f"XGBoost: {xgb_score:.6f}")

        # ========== LightGBM (OPTIMIZED) ==========
        lgb_params = {
            'objective': 'binary',
            'metric': 'auc',
            'boosting_type': 'gbdt',
            'max_depth': 8,  # Increased
            'learning_rate': 0.015,
            'num_leaves': 127,  # Increased
            'subsample': 0.85,
            'colsample_bytree': 0.85,
            'min_child_samples': 20,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'is_unbalance': True,
            'random_state': 42,
            'n_jobs': -1,
            'verbose': -1
        }

        model_lgb = lgb.LGBMClassifier(**lgb_params, n_estimators=2000)
        model_lgb.fit(
            X_tr, y_tr,
            eval_X=X_val,
            eval_y=y_val,
            callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)]
        )

        oof_lgb[val_idx] = model_lgb.predict_proba(X_val)[:, 1]
        test_preds_lgb += model_lgb.predict_proba(X_test)[:, 1] / n_folds

        lgb_score = roc_auc_score(y_val, oof_lgb[val_idx])
        cv_scores['lightgbm'].append(lgb_score)
        print(f"LightGBM: {lgb_score:.6f}")

        # ========== CatBoost (OPTIMIZED) ==========
        model_cat = CatBoostClassifier(
            iterations=2000,
            learning_rate=0.015,
            depth=8,  # Increased
            l2_leaf_reg=3.0,  # Regularization
            eval_metric='AUC',
            auto_class_weights='Balanced',
            random_seed=42,
            verbose=False
        )

        model_cat.fit(
            X_tr, y_tr,
            eval_set=(X_val, y_val),
            early_stopping_rounds=100,
            verbose=False
        )

        oof_cat[val_idx] = model_cat.predict_proba(X_val)[:, 1]
        test_preds_cat += model_cat.predict_proba(X_test)[:, 1] / n_folds

        cat_score = roc_auc_score(y_val, oof_cat[val_idx])
        cv_scores['catboost'].append(cat_score)
        print(f"CatBoost: {cat_score:.6f}")

        # Ensemble
        oof_fold = (oof_xgb[val_idx] + oof_lgb[val_idx] + oof_cat[val_idx]) / 3
        fold_score = roc_auc_score(y_val, oof_fold)
        print(f"🎯 Fold {fold} Ensemble: {fold_score:.6f}")

    # ========== WEIGHTED ENSEMBLE ==========
    print("\n" + "="*60)
    print("CROSS-VALIDATION RESULTS")
    print("="*60)

    avg_xgb = np.mean(cv_scores['xgboost'])
    avg_lgb = np.mean(cv_scores['lightgbm'])
    avg_cat = np.mean(cv_scores['catboost'])

    print(f"\nXGBoost:  {avg_xgb:.6f} ± {np.std(cv_scores['xgboost']):.6f}")
    print(f"LightGBM: {avg_lgb:.6f} ± {np.std(cv_scores['lightgbm']):.6f}")
    print(f"CatBoost: {avg_cat:.6f} ± {np.std(cv_scores['catboost']):.6f}")

    # Calculate weights based on performance
    total_score = avg_xgb + avg_lgb + avg_cat
    w_xgb = avg_xgb / total_score
    w_lgb = avg_lgb / total_score
    w_cat = avg_cat / total_score

    print(f"\nWeights: XGB={w_xgb:.3f}, LGB={w_lgb:.3f}, CAT={w_cat:.3f}")

    # Weighted ensemble
    oof_ensemble = (oof_xgb * w_xgb + oof_lgb * w_lgb + oof_cat * w_cat)
    test_preds = (test_preds_xgb * w_xgb + test_preds_lgb * w_lgb + test_preds_cat * w_cat)

    ensemble_cv = roc_auc_score(y_train, oof_ensemble)

    print(f"\n{'='*60}")
    print(f"🏆 WEIGHTED ENSEMBLE CV: {ensemble_cv:.6f}")
    print(f"{'='*60}\n")

    if ensemble_cv >= 0.947:
        print("✅ TARGET ACHIEVED!")
    else:
        gap = 0.947 - ensemble_cv
        print(f"Gap to target: {gap:.6f}")

    return test_preds, oof_ensemble, cv_scores

def main():

    train, test = LoadDataset("train.csv", "test.csv")

    X_train, y_train, X_test, test_ids = PreProcessData(train, test)

    test_predictions, oof, cv_scores = TrainOptimizedModel(X_train, y_train, X_test, n_folds=3)

    # Create submission
    submission = pd.DataFrame({
        'id': test_ids,
        'Will_Buy_EV': test_predictions
    })

    submission.to_csv('submission_v2.csv', index=False)
    print("\n✅ Enhanced submission created: submission_v2.csv")
    print(f"Shape: {submission.shape}")

if __name__ == "__main__":
    main()
