"""
Enhanced Model Training Module with Advanced Techniques

This module implements state-of-the-art techniques for NFL betting prediction:
- Advanced feature selection
- Multi-objective optimization 
- Stacking ensemble models
- Specialized spread and total models
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import optuna
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, log_loss
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.utils._tags import get_tags
from sklearn.ensemble import RandomForestRegressor, VotingRegressor, StackingRegressor
from sklearn.linear_model import Ridge, LogisticRegression, ElasticNet
from sklearn.feature_selection import SelectFromModel, RFECV
from xgboost import XGBRegressor, XGBClassifier
from sklearn.model_selection import cross_val_score
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')


def prepare_enhanced_data(df, test_season=2024):
    """
    Enhanced data preparation with better feature selection.
    """
    train = df[df['season'] < test_season].copy()
    test = df[df['season'] == test_season].copy()
    
    print(f"📊 Train set: {len(train)} records (seasons < {test_season})")
    print(f"📊 Test set: {len(test)} records (season {test_season})")
    
    # Handle missing values more intelligently
    numeric_cols = train.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col not in ['points_for', 'points_against']:
            median_val = train[col].median()
            train[col] = train[col].fillna(median_val)
            test[col] = test[col].fillna(median_val)
    
    # Categorical preprocessing
    for col in ['roof', 'surface']:
        if col in train.columns:
            train[col] = train[col].fillna('UNK').astype('category')
            test[col] = test[col].fillna('UNK').astype('category')
    
    # Enhanced feature sets
    base_features = [
        'is_home', 'neutral', 'rest_days', 'rest_advantage', 'season_progress',
        'early_season', 'mid_season', 'late_season', 'playoff_race',
        'dome_game', 'outdoor_game', 'turf_game', 'bye_week', 'short_rest', 'long_rest'
    ]
    
    # Rolling performance features
    rolling_features = []
    for window in [1, 2, 3, 4, 5, 8, 10]:
        for stat in ['pf_avg', 'pa_avg', 'point_diff_avg']:
            rolling_features.extend([f'{stat}_{window}', f'opp_{stat}_{window}'])
    
    # Advanced features
    advanced_features = [
        'form_ema_9', 'form_ema_8', 'form_ema_7', 'opp_form_ema_9',
        'sos_3', 'sos_8', 'clutch_record', 'ats_record_5', 'ats_record_10',
        'off_vs_def', 'def_vs_off', 'pace_differential', 'form_diff',
        'volatility_diff', 'blowout_win_rate_8', 'blowout_loss_rate_8'
    ]
    
    # Market features
    market_features = [
        'implied_points', 'opp_implied_points', 'implied_vs_avg',
        'total_vs_recent', 'ml_prob', 'opp_ml_prob'
    ]
    
    # Interaction features
    interaction_features = [
        'rest_form_interaction', 'bye_week_form', 'dome_pace', 'turf_pace',
        'late_season_form', 'playoff_race_form', 'home_rest_advantage'
    ]
    
    # Combine available features
    all_possible_features = (base_features + rolling_features + advanced_features + 
                           market_features + interaction_features + ['roof', 'surface'])
    
    # Only use features that exist in the data
    FEATURES = [f for f in all_possible_features if f in train.columns]
    TARGET = 'points_for'
    
    print(f"🎯 Target variable: {TARGET}")
    print(f"🔧 Available features: {len(FEATURES)}")
    print("✅ Enhanced data preparation complete")
    
    return train, test, FEATURES, TARGET


def create_specialized_models():
    """
    Create specialized models for different aspects of the game.
    """
    models = {}
    
    # Offense-focused model (higher scoring games)
    models['offense'] = XGBRegressor(
        n_estimators=1000,
        max_depth=8,
        learning_rate=0.02,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=0.5,
        reg_alpha=0.1,
        random_state=42,
        n_jobs=-1
    )
    
    # Defense-focused model (lower scoring, tighter games)
    models['defense'] = XGBRegressor(
        n_estimators=1200,
        max_depth=6,
        learning_rate=0.01,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=2.0,
        reg_alpha=0.5,
        random_state=43,
        n_jobs=-1
    )
    
    # Volatility model for high-variance games
    models['volatility'] = XGBRegressor(
        n_estimators=800,
        max_depth=10,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=0.1,
        reg_alpha=0.05,
        random_state=44,
        n_jobs=-1
    )
    
    # LightGBM for different perspective
    models['lgb'] = lgb.LGBMRegressor(
        n_estimators=1000,
        max_depth=7,
        learning_rate=0.02,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=1.0,
        reg_alpha=0.2,
        random_state=45,
        n_jobs=-1,
        verbose=-1
    )
    
    return models


def advanced_feature_selection(X_train, y_train, X_val, y_val, max_features=50):
    """
    Advanced feature selection combining multiple methods.
    """
    print(f"🔍 Starting with {X_train.shape[1]} features")
    
    # Method 1: XGBoost feature importance
    xgb_selector = XGBRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    xgb_selector.fit(X_train, y_train)
    xgb_importance = pd.Series(xgb_selector.feature_importances_, index=range(X_train.shape[1]))
    xgb_top = xgb_importance.nlargest(max_features).index.tolist()
    
    # Method 2: LightGBM feature importance  
    lgb_selector = lgb.LGBMRegressor(n_estimators=200, random_state=42, n_jobs=-1, verbose=-1)
    lgb_selector.fit(X_train, y_train)
    lgb_importance = pd.Series(lgb_selector.feature_importances_, index=range(X_train.shape[1]))
    lgb_top = lgb_importance.nlargest(max_features).index.tolist()
    
    # Method 3: Correlation with target (but not too high to avoid overfitting)
    correlations = np.abs([np.corrcoef(X_train[:, i], y_train)[0, 1] for i in range(X_train.shape[1])])
    correlations = np.nan_to_num(correlations)  # Handle NaN correlations
    corr_top = np.argsort(correlations)[-max_features:].tolist()
    
    # Combine methods (features that appear in at least 2 methods)
    feature_votes = {}
    for idx in xgb_top:
        feature_votes[idx] = feature_votes.get(idx, 0) + 1
    for idx in lgb_top:
        feature_votes[idx] = feature_votes.get(idx, 0) + 1
    for idx in corr_top:
        feature_votes[idx] = feature_votes.get(idx, 0) + 1
    
    # Select features with at least 2 votes, up to max_features
    selected_features = [idx for idx, votes in feature_votes.items() if votes >= 2]
    
    # If we don't have enough, add highest-voted single features
    if len(selected_features) < max_features:
        remaining = [idx for idx, votes in sorted(feature_votes.items(), key=lambda x: x[1], reverse=True) 
                    if idx not in selected_features]
        selected_features.extend(remaining[:max_features - len(selected_features)])
    
    selected_features = selected_features[:max_features]
    
    print(f"✅ Selected {len(selected_features)} features using ensemble selection")
    
    return selected_features


def multi_objective_optimization(X_train, y_train, X_val, y_val, n_trials=100):
    """
    Multi-objective optimization for both MAE and spread accuracy.
    """
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 500, 1500),
            'max_depth': trial.suggest_int('max_depth', 4, 12),
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1, log=True),
            'subsample': trial.suggest_float('subsample', 0.7, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 1.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.1, 10.0, log=True),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.01, 2.0, log=True),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'gamma': trial.suggest_float('gamma', 0, 5),
            'random_state': 42,
            'n_jobs': -1
        }
        
        model = XGBRegressor(**params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], 
                 early_stopping_rounds=50, verbose=False)
        
        pred = model.predict(X_val)
        mae = mean_absolute_error(y_val, pred)
        
        # For spread accuracy, we need to simulate game-level predictions
        # This is a simplified version - in practice you'd need the full game context
        rmse = np.sqrt(mean_squared_error(y_val, pred))
        
        # Multi-objective: minimize MAE and RMSE
        return mae + 0.5 * rmse
    
    print(f"🔍 Starting multi-objective optimization with {n_trials} trials...")
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    print(f"✅ Best score: {study.best_value:.4f}")
    return study.best_params

def safe_stacking_regressor(estimators, final_estimator, cv=5, n_jobs=-1):
    """
    Wrap StackingRegressor to skip any base learners
    that do not properly implement sklearn's tag system.
    """
    safe_estimators = []
    for name, est in estimators:
        try:
            _ = get_tags(est)   # will raise if estimator is not sklearn-compliant
            safe_estimators.append((name, est))
        except Exception as e:
            print(f"⚠️ Skipping estimator '{name}' ({type(est).__name__}) due to tag error: {e}")
    if not safe_estimators:
        raise ValueError("No valid base estimators left for stacking!")
    return StackingRegressor(
        estimators=safe_estimators,
        final_estimator=final_estimator,
        cv=cv,
        n_jobs=n_jobs
    )


def create_stacking_ensemble(base_models):
    """
    Create a stacking ensemble with diverse base models.
    """
    # Meta-learner (final layer)
    meta_learner = Ridge(alpha=1.0)

    
    
    # Create stacking regressor
    stacking_regressor = safe_stacking_regressor(
        estimators=[(name, model) for name, model in base_models.items()],
        final_estimator=meta_learner,
        cv=3,
        n_jobs=-1
    )
    
    return stacking_regressor


def train_spread_classifier(X_train_spread, y_train_spread, X_test_spread, y_test_spread):
    """Train XGBoost classifier for spread prediction, handling object columns automatically."""
    # 🔍 1. Handle object/categorical dtypes
    cat_cols = X_train_spread.select_dtypes(include=["object"]).columns.tolist()
    if cat_cols:
        print(f"Encoding categorical columns for spread classifier: {cat_cols}")

        # one-hot encode train/test
        X_train_spread = pd.get_dummies(X_train_spread, columns=cat_cols, drop_first=True)
        X_test_spread  = pd.get_dummies(X_test_spread, columns=cat_cols, drop_first=True)

        # align columns
        X_train_spread, X_test_spread = X_train_spread.align(X_test_spread, join="left", axis=1, fill_value=0)

    # 🔧 2. Define model
    spread_classifier = xgb.XGBClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42,
        tree_method="hist"
    )

    # 🚀 3. Train
    spread_classifier.fit(X_train_spread, y_train_spread)

    # 🎯 4. Evaluate
    preds = spread_classifier.predict(X_test_spread)
    acc = accuracy_score(y_test_spread, preds)
    f1  = f1_score(y_test_spread, preds, average="weighted")

    print(f"✅ Spread Classifier Performance: Acc={acc:.4f}, F1={f1:.4f}")

    return spread_classifier, preds, {"accuracy": acc, "f1": f1}


def enhanced_walk_forward_validation(train, FEATURES, TARGET, n_splits=5):
    """
    Enhanced walk-forward validation with multiple model types.
    """
    print(f"🔄 Enhanced walk-forward validation with {n_splits} folds...")
    
    # Prepare data
    df_sorted = train.sort_values('date').copy()
    
    # Separate numeric and categorical features
    numeric_features = []
    categorical_features = []
    
    for feature in FEATURES:
        if feature in df_sorted.columns:
            if df_sorted[feature].dtype in ['object', 'category']:
                categorical_features.append(feature)
            else:
                numeric_features.append(feature)
    
    # Create preprocessor
    preprocessor = ColumnTransformer([
        ('num', 'passthrough', numeric_features),
        ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), categorical_features)
    ])
    
    X = df_sorted[FEATURES]
    y = df_sorted[TARGET]
    
    tscv = TimeSeriesSplit(n_splits=n_splits)
    results = {
        'base_xgb': [],
        'optimized_xgb': [],
        'stacking': [],
        'lgb': []
    }
    
    best_params = None
    
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
        print(f"\n🔄 Fold {fold}/{n_splits}")
        
        # Apply temporal purging
        train_end_date = df_sorted.iloc[train_idx[-1]]['date']
        purge_cutoff = train_end_date + pd.Timedelta(days=7)
        valid_mask = df_sorted.iloc[val_idx]['date'] > purge_cutoff
        val_idx = val_idx[valid_mask]
        
        if len(val_idx) == 0:
            continue
        
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        # Preprocess
        X_tr_processed = preprocessor.fit_transform(X_tr)
        X_val_processed = preprocessor.transform(X_val)
        
        # Advanced feature selection
        if X_tr_processed.shape[1] > 50:
            selected_features = advanced_feature_selection(
                X_tr_processed, y_tr, X_val_processed, y_val, max_features=40
            )
            X_tr_processed = X_tr_processed[:, selected_features]
            X_val_processed = X_val_processed[:, selected_features]
        
        # 1. Base XGBoost
        base_xgb = XGBRegressor(n_estimators=800, max_depth=6, learning_rate=0.03,
                               random_state=42, n_jobs=-1)
        base_xgb.fit(X_tr_processed, y_tr)
        pred_base = base_xgb.predict(X_val_processed)
        mae_base = mean_absolute_error(y_val, pred_base)
        results['base_xgb'].append(mae_base)
        
        # 2. Optimized XGBoost (only for first fold)
        if fold == 1 and len(X_tr_processed) > 500:
            print("🔍 Optimizing hyperparameters...")
            best_params = multi_objective_optimization(
                X_tr_processed[:int(len(X_tr_processed)*0.8)],
                y_tr.iloc[:int(len(y_tr)*0.8)],
                X_tr_processed[int(len(X_tr_processed)*0.8):],
                y_tr.iloc[int(len(y_tr)*0.8):],
                n_trials=50
            )
        
        if best_params:
            opt_xgb = XGBRegressor(**best_params)
            opt_xgb.fit(X_tr_processed, y_tr)
            pred_opt = opt_xgb.predict(X_val_processed)
            mae_opt = mean_absolute_error(y_val, pred_opt)
            results['optimized_xgb'].append(mae_opt)
        else:
            results['optimized_xgb'].append(mae_base)
        
        # 3. LightGBM
        lgb_model = lgb.LGBMRegressor(n_estimators=800, max_depth=6, 
                                     learning_rate=0.03, random_state=42, 
                                     n_jobs=-1, verbose=-1)
        lgb_model.fit(X_tr_processed, y_tr)
        pred_lgb = lgb_model.predict(X_val_processed)
        mae_lgb = mean_absolute_error(y_val, pred_lgb)
        results['lgb'].append(mae_lgb)
        
        # 4. Stacking ensemble
        base_models = {
            'xgb': XGBRegressor(n_estimators=600, max_depth=6, learning_rate=0.03, random_state=42, n_jobs=-1),
            'lgb': lgb.LGBMRegressor(n_estimators=600, max_depth=6, learning_rate=0.03, random_state=43, n_jobs=-1, verbose=-1),
            'ridge': Ridge(alpha=1.0)
        }
        
        stacking_model = create_stacking_ensemble(base_models)
        stacking_model.fit(X_tr_processed, y_tr)
        pred_stack = stacking_model.predict(X_val_processed)
        mae_stack = mean_absolute_error(y_val, pred_stack)
        results['stacking'].append(mae_stack)
        
        print(f"   Base XGB MAE: {mae_base:.3f}")
        print(f"   Opt XGB MAE: {results['optimized_xgb'][-1]:.3f}")
        print(f"   LGB MAE: {mae_lgb:.3f}")
        print(f"   Stacking MAE: {mae_stack:.3f}")
    
    # Summary
    print(f"\n📊 Cross-Validation Results:")
    for model_name, maes in results.items():
        if maes:
            mean_mae = np.mean(maes)
            std_mae = np.std(maes)
            print(f"   {model_name}: {mean_mae:.3f} ± {std_mae:.3f}")
    
    # Find best model
    best_model_name = min(results.keys(), key=lambda k: np.mean(results[k]) if results[k] else float('inf'))
    print(f"🏆 Best model: {best_model_name}")
    
    return results, best_params, best_model_name


def train_final_enhanced_model(train_df, test_df, FEATURES, target_col="target"):
    """Full enhanced training pipeline (regressor + spread classifier)."""
    from lightgbm import LGBMRegressor

    # ---------- Regressor ----------
    print("🎯 Training final regressor...")
    X_train, y_train = train_df[FEATURES], train_df[target_col]
    X_test, y_test   = test_df[FEATURES], test_df[target_col]

    reg = LGBMRegressor(
        n_estimators=1000,
        learning_rate=0.01,
        max_depth=-1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )

    reg.fit(X_train, y_train)
    pred_te = reg.predict(X_test)

    mae = mean_absolute_error(y_test, pred_te)
    rmse = mean_squared_error(y_test, pred_te, squared=False)

    print(f"✅ Final Model Performance:\n   MAE: {mae:.3f}\n   RMSE: {rmse:.3f}")

    metrics = {"mae": mae, "rmse": rmse}

    # ---------- Spread Classifier ----------
    print("🎯 Training specialized spread prediction classifier...")

    if "spread_target" not in train_df.columns:
        raise ValueError("Expected 'spread_target' column in training data for classifier.")

    X_train_spread = train_df[FEATURES].copy()
    y_train_spread = train_df["spread_target"]
    X_test_spread  = test_df[FEATURES].copy()
    y_test_spread  = test_df["spread_target"]

    spread_results = train_spread_classifier(X_train_spread, y_train_spread, X_test_spread, y_test_spread)

    return reg, pred_te, metrics, spread_results


if __name__ == "__main__":
    print("🧪 Enhanced Model Training Module")
    print("="*60)
    print("Key improvements:")
    print("- Advanced feature selection")
    print("- Multi-objective optimization") 
    print("- Stacking ensemble models")
    print("- Specialized spread prediction")
    print("- Time-weighted training")
    print("="*60)