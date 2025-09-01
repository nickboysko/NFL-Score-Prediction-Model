"""
Robust Enhanced Model Training Module - FEATURE MISMATCH FIXED

This module handles training, validation, and evaluation of the NFL score prediction model
with robust feature detection and handling of missing features.
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import optuna
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.feature_selection import SelectFromModel
from xgboost import XGBRegressor
try:
    import lightgbm as lgb
    from lightgbm import LGBMRegressor
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("⚠️ LightGBM not available, using XGBoost only")

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import StackingRegressor
import warnings
warnings.filterwarnings('ignore')


def prepare_data(df, test_season=2024):
    """
    ROBUST: Prepare data with automatic feature detection.
    Only uses features that actually exist in the dataset.
    """
    # Leave the most recent season out as test
    train = df[df['season'] < test_season].copy()
    test = df[df['season'] == test_season].copy()
    
    print(f"📊 Train set: {len(train)} records (seasons < {test_season})")
    print(f"📊 Test set: {len(test)} records (season {test_season})")
    
    # Get all available columns
    available_cols = set(df.columns)
    
    # Impute critical numeric cols early for implied features
    numeric_cols_to_impute = ['spread_line', 'total_line', 'rest_days']
    for c in numeric_cols_to_impute:
        if c in available_cols:
            median_val = train[c].median()
            train[c] = train[c].fillna(median_val)
            test[c] = test[c].fillna(median_val)
            print(f"✅ Filled NaN in {c} with median: {median_val:.2f}")
    
    # Market-implied features (only if source columns exist)
    if 'total_line' in available_cols and 'spread_line' in available_cols:
        def compute_implied_points(df_part):
            half_total = df_part['total_line'] / 2.0
            half_spread = df_part['spread_line'] / 2.0
            home_implied = half_total + half_spread
            away_implied = half_total - half_spread
            return np.where(df_part['is_home'] == 1, home_implied, away_implied)
        
        train['implied_points'] = compute_implied_points(train)
        test['implied_points'] = compute_implied_points(test)
        train['implied_spread'] = train['spread_line']
        test['implied_spread'] = test['spread_line']
        train['implied_total'] = train['total_line']
        test['implied_total'] = test['total_line']
        print("✅ Created market-implied features")
    
    # Ensure categorical columns are properly formatted
    categorical_cols = ['roof', 'surface']
    for col in categorical_cols:
        if col in available_cols:
            train[col] = train[col].fillna('UNK').astype(str)
            test[col] = test[col].fillna('UNK').astype(str)
            print(f"✅ Formatted categorical column {col}")
    
    # ROBUST FEATURE SELECTION - Only use what exists
    # Define potential feature categories
    base_features = ['is_home', 'neutral', 'rest_days']
    market_features = ['spread_line', 'total_line', 'implied_points', 'implied_spread', 'implied_total']
    categorical_features = ['roof', 'surface']
    
    # Rolling window features (check what actually exists)
    WINDOWS = [1, 2, 3, 4, 5, 6, 8, 10]
    rolling_patterns = [
        'pf_avg', 'pf_std', 'pa_avg', 'pa_std', 
        'opp_pf_avg', 'opp_pa_avg', 'opp_pf_std', 'opp_pa_std'
    ]
    rolling_features = []
    for pattern in rolling_patterns:
        for w in WINDOWS:
            feat_name = f'{pattern}_{w}'
            if feat_name in available_cols:
                rolling_features.append(feat_name)
    
    # Interaction/derived features (check what exists)
    potential_interaction_features = [
        'off_vs_def', 'def_vs_off', 'total_avg_5', 'opp_total_avg_5', 
        'win_streak', 'recent_form', 'loss_streak', 'games_played',
        'season_progress', 'rest_differential'
    ]
    interaction_features = [f for f in potential_interaction_features if f in available_cols]
    
    # Context features (check what exists)
    potential_context_features = [
        'dome_game', 'outdoor_game', 'turf_game', 'early_season',
        'late_season', 'mid_season', 'is_playoff', 'playoff_race',
        'went_overtime', 'home_advantage'
    ]
    context_features = [f for f in potential_context_features if f in available_cols]
    
    # Combine all available features
    all_potential_features = (base_features + market_features + categorical_features + 
                            rolling_features + interaction_features + context_features)
    
    # Filter to only features that actually exist in the dataframe
    FEATURES = [f for f in all_potential_features if f in available_cols]
    TARGET = 'points_for'
    
    print(f"🎯 Target variable: {TARGET}")
    print(f"🔧 Available features: {len(FEATURES)}")
    
    # Print feature breakdown
    found_base = [f for f in base_features if f in FEATURES]
    found_market = [f for f in market_features if f in FEATURES]
    found_cat = [f for f in categorical_features if f in FEATURES]
    found_rolling = [f for f in rolling_features if f in FEATURES]
    found_interaction = [f for f in interaction_features if f in FEATURES]
    found_context = [f for f in context_features if f in FEATURES]
    
    print(f"   📊 Feature breakdown:")
    print(f"      - Base: {len(found_base)} features")
    print(f"      - Market: {len(found_market)} features")
    print(f"      - Categorical: {len(found_cat)} features")
    print(f"      - Rolling: {len(found_rolling)} features")
    print(f"      - Interaction: {len(found_interaction)} features")
    print(f"      - Context: {len(found_context)} features")
    
    print("✅ Robust data preparation complete")
    return train, test, FEATURES, TARGET


def create_xgboost_model(params=None, objective_type='squared', random_state=42):
    """Create XGBoost model with default or custom parameters."""
    default_params = {
        'n_estimators': 1000,
        'max_depth': 6,
        'learning_rate': 0.03,
        'subsample': 0.9,
        'colsample_bytree': 0.9,
        'reg_lambda': 1.0,
        'reg_alpha': 0.1,
        'objective': 'reg:squarederror' if objective_type == 'squared' else 'count:poisson',
        'random_state': random_state,
        'n_jobs': -1
    }
    
    if params:
        default_params.update(params)
    
    return XGBRegressor(**default_params)


def optimize_hyperparameters(X_train, y_train, X_val, y_val, n_trials=50, objective_type='squared'):
    """Optimize XGBoost hyperparameters using Optuna."""
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 500, 1500),
            'max_depth': trial.suggest_int('max_depth', 4, 12),
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1),
            'subsample': trial.suggest_float('subsample', 0.7, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 1.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.1, 10.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.01, 2.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'gamma': trial.suggest_float('gamma', 0.05, 5.0)
        }
        
        model = create_xgboost_model(params, objective_type, random_state=42)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=50, verbose=False)
        
        pred = model.predict(X_val)
        return mean_absolute_error(y_val, pred)
    
    print(f"🔍 Starting multi-objective optimization with {n_trials} trials...")
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    print(f"✅ Best score: {study.best_value:.4f}")
    return study.best_params


def ensemble_feature_selection(X_train, y_train, n_features=40):
    """Select features using ensemble of different selection methods."""
    print(f"🔍 Starting with {X_train.shape[1]} features")
    
    # Method 1: XGBoost feature importance
    xgb_selector = XGBRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    xgb_selector.fit(X_train, y_train)
    xgb_importance = pd.Series(xgb_selector.feature_importances_, index=range(X_train.shape[1]))
    
    # Method 2: LightGBM feature importance (if available)
    if LIGHTGBM_AVAILABLE:
        try:
            lgb_selector = LGBMRegressor(n_estimators=200, random_state=42, n_jobs=-1, verbose=-1)
            lgb_selector.fit(X_train, y_train)
            lgb_importance = pd.Series(lgb_selector.feature_importances_, index=range(X_train.shape[1]))
        except:
            lgb_importance = xgb_importance.copy()
    else:
        lgb_importance = xgb_importance.copy()
    
    # Combine importance scores
    combined_importance = (xgb_importance + lgb_importance) / 2
    
    # Select top features
    selected_features = combined_importance.nlargest(n_features).index.tolist()
    
    print(f"✅ Selected {len(selected_features)} features using ensemble selection")
    return selected_features


def enhanced_walk_forward_validation(train_data, features, target, n_folds=4):
    """Enhanced walk-forward validation with feature selection and model comparison."""
    print(f"🔄 Enhanced walk-forward validation with {n_folds} folds...")
    
    # Sort by date
    df_sorted = train_data.sort_values('date').copy()
    
    # Prepare features
    numeric_features = [f for f in features if f not in ['roof', 'surface']]
    categorical_features = [f for f in features if f in ['roof', 'surface'] and f in df_sorted.columns]
    
    print(f"🔧 Numerical features: {len(numeric_features)}")
    print(f"🔧 Categorical features: {len(categorical_features)}")
    
    # Create preprocessor
    preprocessor_transformers = [('num', 'passthrough', numeric_features)]
    if categorical_features:
        preprocessor_transformers.append(
            ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), categorical_features)
        )
    
    preprocessor = ColumnTransformer(preprocessor_transformers)
    
    # Time series split
    tscv = TimeSeriesSplit(n_splits=n_folds)
    results = {}
    
    X = df_sorted[features]
    y = df_sorted[target]
    
    best_params = None
    
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
        print(f"\n🔄 Fold {fold}/{n_folds}")
        
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        # Preprocess features
        X_train_processed = preprocessor.fit_transform(X_train)
        X_val_processed = preprocessor.transform(X_val)
        
        # Feature selection
        selected_features = ensemble_feature_selection(X_train_processed, y_train, n_features=40)
        X_train_selected = X_train_processed[:, selected_features]
        X_val_selected = X_val_processed[:, selected_features]
        
        # Hyperparameter optimization (only for first fold to save time)
        if fold == 1:
            best_params = optimize_hyperparameters(
                X_train_selected, y_train, X_val_selected, y_val, n_trials=50
            )
        
        # Test multiple models
        models = {
            'base_xgb': create_xgboost_model(random_state=42),
            'optimized_xgb': create_xgboost_model(best_params, random_state=42),
        }
        
        if LIGHTGBM_AVAILABLE:
            models['lgb'] = LGBMRegressor(n_estimators=1000, random_state=42, n_jobs=-1, verbose=-1)
            models['stacking'] = StackingRegressor([
                ('xgb', create_xgboost_model(best_params, random_state=42)),
                ('lgb', LGBMRegressor(n_estimators=800, random_state=42, n_jobs=-1, verbose=-1))
            ], final_estimator=Ridge(alpha=1.0), cv=3, n_jobs=-1)
        
        fold_results = {}
        for name, model in models.items():
            try:
                if 'xgb' in name and name != 'stacking':
                    model.fit(X_train_selected, y_train, 
                             eval_set=[(X_val_selected, y_val)], 
                             early_stopping_rounds=50, verbose=False)
                else:
                    model.fit(X_train_selected, y_train)
                
                pred = model.predict(X_val_selected)
                mae = mean_absolute_error(y_val, pred)
                fold_results[name] = mae
                print(f"   {name} MAE: {mae:.3f}")
                
            except Exception as e:
                print(f"   ⚠️ Skipping estimator '{name}' ({type(model).__name__}) due to tag error: {str(e)}")
                continue
        
        # Store results
        if fold not in results:
            results[fold] = {}
        results[fold].update(fold_results)
    
    # Calculate cross-validation scores
    cv_results = {}
    model_names = ['base_xgb', 'optimized_xgb']
    if LIGHTGBM_AVAILABLE:
        model_names.extend(['lgb', 'stacking'])
        
    for model_name in model_names:
        scores = [results[fold].get(model_name) for fold in results if model_name in results[fold]]
        if scores:
            cv_results[model_name] = (np.mean(scores), np.std(scores))
    
    print(f"\n📊 Cross-Validation Results:")
    best_model = None
    best_score = float('inf')
    
    for model_name, (mean_score, std_score) in cv_results.items():
        print(f"   {model_name}: {mean_score:.3f} ± {std_score:.3f}")
        if mean_score < best_score:
            best_score = mean_score
            best_model = model_name
    
    print(f"🏆 Best model: {best_model}")
    print(f"✅ Best model type: {best_model}")
    
    return best_score, best_model, best_params


def train_final_enhanced_model(train_data, test_data, features, target, 
                             model_type='xgboost', hyperparams=None,
                             use_feature_selection=True, n_features=40):
    """Train final model with proper parameter handling."""
    print(f"🚀 Training final {model_type} model...")
    
    # Prepare features
    numeric_features = [f for f in features if f not in ['roof', 'surface']]
    categorical_features = [f for f in features if f in ['roof', 'surface'] and f in train_data.columns]
    
    # Create preprocessor
    preprocessor_transformers = [('num', 'passthrough', numeric_features)]
    if categorical_features:
        preprocessor_transformers.append(
            ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), categorical_features)
        )
    
    preprocessor = ColumnTransformer(preprocessor_transformers)
    
    # Get training and test data
    X_train, y_train = train_data[features], train_data[target]
    X_test, y_test = test_data[features], test_data[target]
    
    # Preprocess
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    # Feature selection
    if use_feature_selection:
        selected_features = ensemble_feature_selection(X_train_processed, y_train, n_features)
        X_train_final = X_train_processed[:, selected_features]
        X_test_final = X_test_processed[:, selected_features]
        print(f"✅ Using {len(selected_features)} selected features")
    else:
        X_train_final = X_train_processed
        X_test_final = X_test_processed
        print(f"✅ Using all {X_train_final.shape[1]} features")
    
    # Create and train model
    if model_type == 'stacking' and LIGHTGBM_AVAILABLE:
        model = StackingRegressor([
            ('xgb', create_xgboost_model(hyperparams, random_state=42)),
            ('lgb', LGBMRegressor(n_estimators=800, random_state=42, n_jobs=-1, verbose=-1))
        ], final_estimator=Ridge(alpha=1.0), cv=3, n_jobs=-1)
    elif model_type == 'optimized_xgb' or (model_type == 'optimized' and hyperparams):
        model = create_xgboost_model(hyperparams, random_state=42)
    elif model_type == 'lgb' and LIGHTGBM_AVAILABLE:
        model = LGBMRegressor(n_estimators=1000, random_state=42, n_jobs=-1, verbose=-1)
    else:
        model = create_xgboost_model(random_state=42)
    
    # Train model
    if 'xgb' in str(type(model)).lower() and model_type != 'stacking':
        model.fit(X_train_final, y_train, 
                 eval_set=[(X_test_final, y_test)], 
                 early_stopping_rounds=50, verbose=False)
    else:
        model.fit(X_train_final, y_train)
    
    # Make predictions
    predictions = model.predict(X_test_final)
    
    # Calculate metrics
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    
    metrics = {'mae': mae, 'rmse': rmse}
    
    # Create pipeline
    class ModelPipeline:
        def __init__(self, preprocessor, model, feature_selector=None):
            self.preprocessor = preprocessor
            self.model = model
            self.feature_selector = feature_selector
        
        def predict(self, X):
            X_processed = self.preprocessor.transform(X)
            if self.feature_selector:
                X_processed = X_processed[:, self.feature_selector]
            return self.model.predict(X_processed)
    
    pipeline = ModelPipeline(
        preprocessor, 
        model, 
        selected_features if use_feature_selection else None
    )
    
    return pipeline, predictions, metrics


def comprehensive_evaluation(test_data, predictions):
    """Comprehensive evaluation including game-level metrics and betting performance."""
    print("📊 Converting team predictions to game scores...")
    
    # Combine team rows back into games
    available_cols = ['game_id', 'team', 'opp', 'is_home', 'points_for', 'points_against']
    if 'spread_line' in test_data.columns:
        available_cols.append('spread_line')
    if 'total_line' in test_data.columns:
        available_cols.append('total_line')
    
    existing_cols = [col for col in available_cols if col in test_data.columns]
    
    te_pred = test_data[existing_cols].copy()
    te_pred['pred_points'] = predictions
    
    # Separate home and away predictions
    home_games = te_pred[te_pred['is_home'] == 1].set_index('game_id')
    away_games = te_pred[te_pred['is_home'] == 0].set_index('game_id')
    
    # Create game-level predictions
    score_pred = pd.DataFrame(index=home_games.index)
    score_pred['home_pred'] = home_games['pred_points']
    score_pred['away_pred'] = away_games['pred_points']
    score_pred['pred_total'] = score_pred['home_pred'] + score_pred['away_pred']
    score_pred['pred_spread'] = score_pred['home_pred'] - score_pred['away_pred']
    
    # Get actual scores correctly
    score_pred['home_actual'] = home_games['points_for']
    score_pred['away_actual'] = away_games['points_for']
    score_pred['actual_total'] = score_pred['home_actual'] + score_pred['away_actual']
    score_pred['actual_spread'] = score_pred['home_actual'] - score_pred['away_actual']
    
    # Calculate errors
    score_pred['ae_home'] = (score_pred['home_pred'] - score_pred['home_actual']).abs()
    score_pred['ae_away'] = (score_pred['away_pred'] - score_pred['away_actual']).abs()
    score_pred['ae_total'] = (score_pred['pred_total'] - score_pred['actual_total']).abs()
    score_pred['ae_spread'] = (score_pred['pred_spread'] - score_pred['actual_spread']).abs()
    
    # Basic game metrics
    game_metrics = {
        'mae_home': score_pred['ae_home'].mean(),
        'mae_away': score_pred['ae_away'].mean(),
        'mae_total': score_pred['ae_total'].mean(),
        'mae_spread': score_pred['ae_spread'].mean(),
        'rmse_total': np.sqrt(((score_pred['pred_total'] - score_pred['actual_total']) ** 2).mean()),
        'rmse_spread': np.sqrt(((score_pred['pred_spread'] - score_pred['actual_spread']) ** 2).mean())
    }
    
    # Add betting metrics if spread/total lines available
    if 'spread_line' in existing_cols:
        home_spread_lines = home_games['spread_line']
        score_pred['spread_line'] = home_spread_lines
        
        # Against the spread accuracy
        score_pred['pred_ats'] = score_pred['pred_spread'] > score_pred['spread_line']
        score_pred['actual_ats'] = score_pred['actual_spread'] > score_pred['spread_line']
        ats_accuracy = (score_pred['pred_ats'] == score_pred['actual_ats']).mean()
        
        game_metrics['ats_accuracy'] = ats_accuracy
        game_metrics['spread_correlation'] = score_pred['pred_spread'].corr(score_pred['actual_spread'])
    
    if 'total_line' in existing_cols:
        home_total_lines = home_games['total_line']
        score_pred['total_line'] = home_total_lines
        
        # Over/under accuracy
        score_pred['pred_over'] = score_pred['pred_total'] > score_pred['total_line']
        score_pred['actual_over'] = score_pred['actual_total'] > score_pred['total_line']
        ou_accuracy = (score_pred['pred_over'] == score_pred['actual_over']).mean()
        
        game_metrics['ou_accuracy'] = ou_accuracy
        game_metrics['total_correlation'] = score_pred['pred_total'].corr(score_pred['actual_total'])
    
    # Print results
    print(f"📊 Game-level evaluation:")
    print(f"   MAE home score: {game_metrics['mae_home']:.2f}")
    print(f"   MAE away score: {game_metrics['mae_away']:.2f}")
    print(f"   MAE total score: {game_metrics['mae_total']:.2f}")
    print(f"   MAE spread: {game_metrics['mae_spread']:.2f}")
    
    if 'ats_accuracy' in game_metrics:
        print(f"   ATS accuracy: {game_metrics['ats_accuracy']:.1%}")
        print(f"   Spread correlation: {game_metrics['spread_correlation']:.3f}")
    
    if 'ou_accuracy' in game_metrics:
        print(f"   O/U accuracy: {game_metrics['ou_accuracy']:.1%}")
        print(f"   Total correlation: {game_metrics['total_correlation']:.3f}")
    
    return score_pred, game_metrics


if __name__ == "__main__":
    print("🧪 Robust Enhanced Model Training Module - Testing Complete Pipeline...")
    print("="*70)
    
    # Test with sample data
    from data_loader import load_schedules
    from features import create_all_features
    
    # Load and prepare sample data
    print("📅 Step 1: Loading sample data...")
    sample_data = load_schedules([2022, 2023, 2024])
    print(f"✅ Loaded {len(sample_data)} records")
    
    print("\n🔧 Step 2: Creating features...")
    featured_data = create_all_features(sample_data)
    print("✅ Feature engineering complete")
    
    print("\n📊 Step 3: Robust data preparation...")
    train, test, FEATURES, TARGET = prepare_data(featured_data, test_season=2024)
    
    print("\n🔄 Step 4: Enhanced walk-forward validation...")
    cv_mae, best_model_type, best_hyperparams = enhanced_walk_forward_validation(
        train, FEATURES, TARGET, n_folds=4
    )
    
    print("\n🚀 Step 5: Training final enhanced model...")
    pipeline, pred_te, metrics = train_final_enhanced_model(
        train, test, FEATURES, TARGET, 
        model_type=best_model_type, 
        hyperparams=best_hyperparams,
        use_feature_selection=True
    )
    
    print("\n🎯 Step 6: Comprehensive evaluation...")
    score_eval, game_metrics = comprehensive_evaluation(test, pred_te)
    
    print("\n" + "="*70)
    print("🎉 ROBUST PIPELINE TEST SUCCESSFUL!")
    print("="*70)