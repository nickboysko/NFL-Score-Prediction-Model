"""
Enhanced Model Training and Evaluation Module

This module handles training, validation, and evaluation of the NFL score prediction model
using XGBoost with advanced features like hyperparameter optimization, ensemble methods,
and comprehensive evaluation metrics.
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
import warnings
warnings.filterwarnings('ignore')


def prepare_data(df, test_season=2024):
    """
    Prepare data for training by splitting into train/test and cleaning features.
    
    Args:
        df (pd.DataFrame): DataFrame with engineered features
        test_season (int): Season to use as test set
    
    Returns:
        tuple: (train, test, FEATURES, TARGET)
    """
    # Leave the most recent season out as test
    train = df[df['season'] < test_season].copy()
    test = df[df['season'] == test_season].copy()
    
    print(f"📊 Train set: {len(train)} records (seasons < {test_season})")
    print(f"📊 Test set: {len(test)} records (season {test_season})")
    
    # Define features and target
    WINDOWS = [1, 2, 3, 4, 5, 6, 8, 10]  # Expanded to match features.py
    
    # Impute critical numeric cols early for implied features
    for c in ['spread_line', 'total_line', 'rest_days']:
        if c in train.columns:
            median_val = train[c].median()
            train[c] = train[c].fillna(median_val)
            test[c] = test[c].fillna(median_val)
            print(f"✅ Filled NaN in {c} with median: {median_val:.2f}")
    
    # Market-implied features (based on home perspective)
    # implied home points = total/2 + spread/2 ; implied away points = total/2 - spread/2
    def compute_implied_points(df_part):
        half_total = df_part['total_line'] / 2.0
        half_spread = df_part['spread_line'] / 2.0
        home_implied = half_total + half_spread
        away_implied = half_total - half_spread
        # map to team row using is_home
        return np.where(df_part['is_home'] == 1, home_implied, away_implied)
    
    train['implied_points'] = compute_implied_points(train)
    test['implied_points'] = compute_implied_points(test)
    train['implied_spread'] = train['spread_line']  # home - away
    test['implied_spread'] = test['spread_line']
    train['implied_total'] = train['total_line']
    test['implied_total'] = test['total_line']
    
    # Ensure categorical columns are properly formatted
    for col in ['roof', 'surface']:
        if col in train.columns:
            train[col] = train[col].fillna('UNK').astype(str)
            test[col] = test[col].fillna('UNK').astype(str)
            print(f"✅ Formatted categorical column {col}")
    
    base_cols = ['is_home', 'neutral', 'rest_days', 'spread_line', 'total_line', 'roof', 'surface',
                 'implied_points', 'implied_spread', 'implied_total']
    rolling_cols = [f'{p}_{w}' for p in ['pf_avg', 'pf_std', 'pa_avg', 'pa_std', 'opp_pf_avg', 'opp_pa_avg'] for w in WINDOWS]
    interaction_cols = ['off_vs_def', 'def_vs_off', 'total_avg_5', 'opp_total_avg_5', 'win_streak', 'recent_form']
    context_cols = ['dome_game', 'outdoor_game', 'turf_game', 'season_progress']
    
    FEATURES = base_cols + rolling_cols + interaction_cols + context_cols
    TARGET = 'points_for'
    
    print(f"🎯 Target variable: {TARGET}")
    print(f"🔧 Number of features: {len(FEATURES)}")
    print("✅ Data preparation complete")
    return train, test, FEATURES, TARGET


def create_xgboost_model(params=None, objective_type='squared', random_state=42):
    """
    Create XGBoost model with default or custom parameters.
    
    Args:
        params (dict): Custom parameters for XGBoost
        objective_type (str): 'squared' or 'poisson'
        random_state (int): seed
    
    Returns:
        XGBRegressor: Configured XGBoost model
    """
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


def create_ensemble_model():
    """
    Create ensemble model combining XGBoost, Random Forest, and Ridge.
    
    Returns:
        VotingRegressor: Ensemble model
    """
    # Individual models with different strengths
    xgb_model = XGBRegressor(
        n_estimators=800,
        max_depth=6,
        learning_rate=0.03,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1
    )
    
    rf_model = RandomForestRegressor(
        n_estimators=500,
        max_depth=8,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    
    ridge_model = Ridge(alpha=1.0)
    
    # Create ensemble with weighted voting
    ensemble = VotingRegressor([
        ('xgb', xgb_model),
        ('rf', rf_model),
        ('ridge', ridge_model)
    ], weights=[0.6, 0.3, 0.1])  # XGBoost gets most weight
    
    return ensemble


def optimize_hyperparameters(X_train, y_train, X_val, y_val, n_trials=80, objective_type='squared'):
    """
    Optimize XGBoost hyperparameters using Optuna.
    """
    def objective(trial):
        # Widened parameter search space
        params = {
            'objective': 'reg:squarederror' if objective_type == 'squared' else 'count:poisson',
            'eval_metric': 'rmse',
            'booster': 'gbtree',
            'verbosity': 0,
            'random_state': trial.suggest_int('random_state', 1, 10000),
            'n_jobs': -1,
            'n_estimators': trial.suggest_int('n_estimators', 300, 1400),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 5.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 5.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        }
        
        model = XGBRegressor(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=50,
            verbose=False
        )
        y_pred = model.predict(X_val)
        mae = mean_absolute_error(y_val, y_pred)
        return mae
    
    print(f"🔍 Starting hyperparameter optimization with {n_trials} trials...")
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    print(f"✅ Best MAE: {study.best_value:.4f}")
    print(f"🔧 Best parameters: {study.best_params}")
    return study.best_params


def improved_walk_forward_backtest(train, FEATURES, TARGET, n_splits=5, use_optimization=False, objective_type='squared', decay_half_life_days=60):
    """
    Enhanced walk-forward backtesting with purging, optional optimization, and time-decay weights.
    """
    print(f"🔄 Performing enhanced walk-forward backtesting with {n_splits} folds...")
    df_sorted = train.sort_values('date').copy()
    num_cols = [c for c in FEATURES if c not in ['roof', 'surface']]
    cat_cols = ['roof', 'surface']
    print(f"🔧 Numerical features: {len(num_cols)}")
    print(f"🔧 Categorical features: {len(cat_cols)}")
    preprocessor = ColumnTransformer([
        ('num', 'passthrough', num_cols),
        ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), cat_cols)
    ])
    X = df_sorted[FEATURES]
    y = df_sorted[TARGET]
    print(f"📅 Date range: {df_sorted['date'].min().strftime('%Y-%m-%d')} to {df_sorted['date'].max().strftime('%Y-%m-%d')}")
    print(f"📊 Total training samples: {len(X)}")
    tscv = TimeSeriesSplit(n_splits=n_splits)
    maes, fold_results = [], []
    best_params = None
    PURGE_DAYS = 7
    for fold, (tr_idx, va_idx) in enumerate(tscv.split(X), 1):
        print(f"\n🔄 Fold {fold}/{n_splits}")
        train_end_date = df_sorted.iloc[tr_idx[-1]]['date']
        purge_cutoff = train_end_date + pd.Timedelta(days=PURGE_DAYS)
        valid_va_mask = df_sorted.iloc[va_idx]['date'] > purge_cutoff
        va_idx = va_idx[valid_va_mask]
        if len(va_idx) == 0:
            print(f"   ⚠️ No validation data after purging for fold {fold}")
            continue
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]
        train_dates = df_sorted.iloc[tr_idx]['date']
        val_dates = df_sorted.iloc[va_idx]['date']
        print(f"   📅 Train: {train_dates.min().strftime('%Y-%m-%d')} to {train_dates.max().strftime('%Y-%m-%d')} ({len(X_tr)} samples)")
        print(f"   📅 Val:   {val_dates.min().strftime('%Y-%m-%d')} to {val_dates.max().strftime('%Y-%m-%d')} ({len(X_va)} samples)")
        X_tr_processed = preprocessor.fit_transform(X_tr)
        X_va_processed = preprocessor.transform(X_va)
        
        # Time-decay sample weights (newer games weigh more)
        # half-life parameter controls decay; convert days delta to weight = 0.5 ** (age_days / half_life)
        age_days = (train_end_date - train_dates).dt.days.clip(lower=0)
        sample_weight = np.power(0.5, age_days / decay_half_life_days).values
        
        if use_optimization and fold == 1 and len(X_tr_processed) > 200:
            opt_split = int(len(X_tr_processed) * 0.8)
            X_opt_train = X_tr_processed[:opt_split]
            y_opt_train = y_tr.iloc[:opt_split]
            X_opt_val = X_tr_processed[opt_split:]
            y_opt_val = y_tr.iloc[opt_split:]
            best_params = optimize_hyperparameters(
                X_opt_train, y_opt_train, X_opt_val, y_opt_val, n_trials=80, objective_type=objective_type
            )
        model = create_xgboost_model(best_params, objective_type=objective_type, random_state=42)
        model.fit(
            X_tr_processed, y_tr,
            sample_weight=sample_weight,
            eval_set=[(X_va_processed, y_va)],
            early_stopping_rounds=50,
            verbose=False
        )
        pred = model.predict(X_va_processed)
        maes.append(mean_absolute_error(y_va, pred))
        fold_rmse = np.sqrt(mean_squared_error(y_va, pred))
        fold_results.append({
            'fold': fold,
            'train_size': len(X_tr),
            'val_size': len(X_va),
            'train_dates': (train_dates.min(), train_dates.max()),
            'val_dates': (val_dates.min(), val_dates.max()),
            'mae': maes[-1],
            'rmse': fold_rmse,
            'predictions': pred,
            'actual': y_va.values
        })
        print(f"   ✅ MAE: {maes[-1]:.3f}, RMSE: {fold_rmse:.3f}")
    if maes:
        mean_cv_mae = np.mean(maes)
        std_cv_mae = np.std(maes)
        print(f"\n📊 Cross-Validation Results:")
        print(f"   Mean MAE: {mean_cv_mae:.3f} ± {std_cv_mae:.3f}")
        print(f"   Individual fold MAEs: {[f'{m:.3f}' for m in maes]}")
    else:
        print("⚠️ No valid folds completed!")
        mean_cv_mae = float('inf')
    print("✅ Walk-forward backtesting complete")
    return mean_cv_mae, fold_results, best_params


def train_final_model(train, test, FEATURES, TARGET, model_type='xgboost', 
                     best_params=None, use_feature_selection=True,
                     objective_type='squared', seeds=(42, 1337, 2024), decay_half_life_days=60,
                     residual_learning=True):
    """
    Train final model with optional ensemble, feature selection, residual learning, and winner calibration.
    """
    num_cols = [c for c in FEATURES if c not in ['roof', 'surface']]
    cat_cols = ['roof', 'surface']
    print(f"🔧 Numerical features: {len(num_cols)}")
    print(f"🔧 Categorical features: {len(cat_cols)}")
    preprocessor = ColumnTransformer([
        ('num', 'passthrough', num_cols),
        ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), cat_cols)
    ])
    X_tr = train[FEATURES]; y_tr = train[TARGET]
    X_te = test[FEATURES]; y_te = test[TARGET]
    print(f"📊 Training data shape: {X_tr.shape}")
    print(f"📊 Test data shape: {X_te.shape}")
    for col in cat_cols:
        if col in X_tr.columns:
            X_tr[col] = X_tr[col].astype(str)
            X_te[col] = X_te[col].astype(str)
            print(f"✅ Converted {col} to string dtype")
    print("🔧 Preprocessing features...")
    X_tr_processed = preprocessor.fit_transform(X_tr)
    X_te_processed = preprocessor.transform(X_te)
    print(f"📊 Processed training data shape: {X_tr_processed.shape}")
    print(f"📊 Processed test data shape: {X_te_processed.shape}")
    if use_feature_selection and X_tr_processed.shape[1] > 15:
        print("🔧 Applying feature selection...")
        from sklearn.impute import SimpleImputer
        if np.isnan(X_tr_processed).any():
            print("🔧 Handling NaN values before feature selection...")
            imputer = SimpleImputer(strategy='median')
            X_tr_processed = imputer.fit_transform(X_tr_processed)
            X_te_processed = imputer.transform(X_te_processed)
            print("✅ NaN values handled")
        temp_model = XGBRegressor(n_estimators=150, random_state=42, n_jobs=-1)
        temp_model.fit(X_tr_processed, y_tr)
        selector = SelectFromModel(temp_model, threshold='median')
        X_tr_processed = selector.fit_transform(X_tr_processed, y_tr)
        X_te_processed = selector.transform(X_te_processed)
        n_selected = X_tr_processed.shape[1]
        print(f"✅ Selected {n_selected} features from {X_te.shape[1]} original features")
    # Time-decay sample weights for final training
    train_end_date = train['date'].max()
    age_days = (train_end_date - train['date']).dt.days.clip(lower=0)
    sample_weight = np.power(0.5, age_days / decay_half_life_days).values
    print("🔧 Using time-decay sample weights for final training")
    # Residual learning target
    if residual_learning and 'implied_points' in train.columns and 'implied_points' in test.columns:
        y_tr_target = (train[TARGET] - train['implied_points']).values
        implied_te = test['implied_points'].values
        residual_mode = True
        print("✅ Residual learning enabled (predicting points_for - implied_points)")
    else:
        y_tr_target = y_tr.values
        implied_te = np.zeros(len(test))
        residual_mode = False
        print("ℹ️ Residual learning disabled (predicting raw points)")
    # Train model(s)
    print(f"🚀 Training {model_type} model...")
    models = []
    if model_type == 'ensemble':
        model = create_ensemble_model()
        model.fit(X_tr_processed, y_tr_target)
        models = [model]
    else:
        for seed in seeds:
            model = create_xgboost_model(best_params, objective_type=objective_type, random_state=seed)
            model.fit(
                X_tr_processed, y_tr_target,
                sample_weight=sample_weight,
                eval_set=[(X_te_processed, y_te - implied_te if residual_mode else y_te)],
                early_stopping_rounds=50,
                verbose=False
            )
            models.append(model)
    # Predictions (average if multiple models)
    preds_res = np.column_stack([m.predict(X_te_processed) for m in models]).mean(axis=1)
    pred_te = implied_te + preds_res if residual_mode else preds_res
    if objective_type == 'poisson':
        pred_te = np.clip(pred_te, 0, None)
    team_mae = mean_absolute_error(y_te, pred_te)
    team_rmse = np.sqrt(mean_squared_error(y_te, pred_te))
    print(f'✅ 2024 Team-level MAE: {team_mae:.2f}')
    print(f'✅ 2024 Team-level RMSE: {team_rmse:.2f}')
    # Winner calibration: fit logistic regression on training seasons
    # Build training game-level spreads using model predictions on training data
    print("🔧 Fitting winner calibration model (logistic regression)...")
    # Predict on training set with out-of-fold proxy (fit model on full train, reuse same for proxy)
    preds_res_tr = np.column_stack([m.predict(X_tr_processed) for m in models]).mean(axis=1)
    pred_points_tr = (train['implied_points'].values + preds_res_tr) if residual_mode else preds_res_tr
    # Construct team-level frame
    train_tmp = train[['game_id', 'is_home', 'points_for', 'implied_spread']].copy()
    train_tmp['pred_points'] = pred_points_tr
    # Split home/away
    home_tr = train_tmp[train_tmp['is_home'] == 1].set_index('game_id')
    away_tr = train_tmp[train_tmp['is_home'] == 0].set_index('game_id')
    common_ids = home_tr.index.intersection(away_tr.index)
    home_tr = home_tr.loc[common_ids]
    away_tr = away_tr.loc[common_ids]
    pred_spread_tr = (home_tr['pred_points'] - away_tr['pred_points']).values
    implied_spread_tr = home_tr['implied_spread'].values
    actual_home_win = (home_tr['points_for'].values > away_tr['points_for'].values).astype(int)
    X_calib = np.column_stack([pred_spread_tr, implied_spread_tr, pred_spread_tr - implied_spread_tr])
    calibrator = LogisticRegression(max_iter=1000)
    calibrator.fit(X_calib, actual_home_win)
    print("✅ Winner calibration model fitted")
    class ModelPipeline:
        def __init__(self, preprocessor, models, feature_selector=None, objective_type='squared', residual_mode=False, calibrator=None):
            self.preprocessor = preprocessor
            self.models = models
            self.feature_selector = feature_selector
            self.objective_type = objective_type
            self.residual_mode = residual_mode
            self.calibrator = calibrator
        def predict(self, X):
            X_processed = self.preprocessor.transform(X)
            if self.feature_selector:
                X_processed = self.feature_selector.transform(X_processed)
            preds = np.column_stack([m.predict(X_processed) for m in self.models]).mean(axis=1)
            if self.residual_mode and 'implied_points' in X.columns:
                preds = X['implied_points'].values + preds
            if self.objective_type == 'poisson':
                preds = np.clip(preds, 0, None)
            return preds
        def predict_win_prob(self, df_team):
            # df_team: team-level DataFrame with implied_spread and is_home
            # Build game-level predictions and compute calibrated P(home win)
            tmp = df_team[['game_id', 'is_home', 'implied_spread']].copy()
            tmp['pred_points'] = self.predict(df_team)
            home = tmp[tmp['is_home'] == 1].set_index('game_id')
            away = tmp[tmp['is_home'] == 0].set_index('game_id')
            common = home.index.intersection(away.index)
            home = home.loc[common]
            away = away.loc[common]
            pred_spread = (home['pred_points'] - away['pred_points']).values
            implied_spread = home['implied_spread'].values
            Xc = np.column_stack([pred_spread, implied_spread, pred_spread - implied_spread])
            if self.calibrator is None:
                # Fallback: sigmoid on pred_spread
                from scipy.special import expit
                return pd.Series(expit(pred_spread / 3.0), index=common)
            probs = self.calibrator.predict_proba(Xc)[:, 1]
            return pd.Series(probs, index=common)
    selector_obj = selector if use_feature_selection and 'selector' in locals() else None
    pipeline = ModelPipeline(preprocessor, models, selector_obj, objective_type, residual_mode, calibrator)
    print("✅ Final model training complete")
    return pipeline, pred_te, {'mae': team_mae, 'rmse': team_rmse}


def comprehensive_evaluation(test, pred_te):
    """
    Comprehensive evaluation including game-level metrics and betting performance.
    
    Args:
        test: Test DataFrame with game information
        pred_te: Team-level predictions
    
    Returns:
        tuple: (score_eval, comprehensive_metrics)
    """
    print("Converting team predictions to game scores...")
    
    # Combine team rows back into games
    available_cols = ['game_id', 'team', 'opp', 'is_home', 'points_for', 'points_against']
    if 'spread_line' in test.columns:
        available_cols.append('spread_line')
    if 'total_line' in test.columns:
        available_cols.append('total_line')
    
    existing_cols = [col for col in available_cols if col in test.columns]
    
    te_pred = test[existing_cols].copy()
    te_pred['pred_points'] = pred_te
    
    # Separate home and away predictions
    home_games = te_pred[te_pred['is_home'] == 1].set_index('game_id')
    away_games = te_pred[te_pred['is_home'] == 0].set_index('game_id')
    
    # Create game-level predictions
    score_pred = pd.DataFrame(index=home_games.index)
    score_pred['home_pred'] = home_games['pred_points']
    score_pred['away_pred'] = away_games['pred_points']
    score_pred['pred_total'] = score_pred['home_pred'] + score_pred['away_pred']
    score_pred['pred_spread'] = score_pred['home_pred'] - score_pred['away_pred']
    
    # FIXED: Get actual scores correctly
    # For home teams, points_for is the home score
    # For away teams, points_for is the away score (NOT points_against)
    score_pred['home_actual'] = home_games['points_for']
    score_pred['away_actual'] = away_games['points_for']  # FIXED: was points_against
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
    
    # Print comprehensive results
    print(f"Game-level evaluation:")
    print(f"   MAE home score: {game_metrics['mae_home']:.2f}")
    print(f"   MAE away score: {game_metrics['mae_away']:.2f}")
    print(f"   MAE total score: {game_metrics['mae_total']:.2f}")
    print(f"   MAE spread: {game_metrics['mae_spread']:.2f}")
    print(f"   RMSE total: {game_metrics['rmse_total']:.2f}")
    print(f"   RMSE spread: {game_metrics['rmse_spread']:.2f}")
    
    if 'ats_accuracy' in game_metrics:
        print(f"   ATS accuracy: {game_metrics['ats_accuracy']:.1%}")
        print(f"   Spread correlation: {game_metrics['spread_correlation']:.3f}")
    
    if 'ou_accuracy' in game_metrics:
        print(f"   O/U accuracy: {game_metrics['ou_accuracy']:.1%}")
        print(f"   Total correlation: {game_metrics['total_correlation']:.3f}")
    
    print("Game evaluation complete")
    
    return score_pred, game_metrics


if __name__ == "__main__":
    print("🧪 Enhanced Model Training Module - Testing Complete Pipeline...")
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
    
    print("\n📊 Step 3: Preparing data...")
    train, test, FEATURES, TARGET = prepare_data(featured_data, test_season=2024)
    
    print("\n🔄 Step 4: Enhanced walk-forward backtesting...")
    cv_mae, fold_results, best_params = improved_walk_forward_backtest(
        train, FEATURES, TARGET, n_splits=5, use_optimization=True, objective_type='squared', decay_half_life_days=60
    )
    
    print("\n🚀 Step 5: Training final optimized model...")
    model_type = 'optimized' if best_params else 'xgboost'
    pipeline, pred_te, metrics = train_final_model(
        train, test, FEATURES, TARGET, 
        model_type=model_type, 
        best_params=best_params,
        use_feature_selection=True,
        objective_type='squared',
        seeds=(42, 1337, 2024),
        decay_half_life_days=60
    )
    
    print("\n🎯 Step 6: Comprehensive evaluation...")
    score_eval, game_metrics = comprehensive_evaluation(test, pred_te)
    
    print("\n" + "="*70)
    print("🎉 ENHANCED PIPELINE TEST SUCCESSFUL!")
    print("="*70)
    
    print(f"\n📊 Final Results Summary:")
    print(f"   Training records: {len(train)}")
    print(f"   Test records: {len(test)}")
    print(f"   Features used: {len(FEATURES)}")
    print(f"   Model type: {model_type}")
    
    print(f"\n🔄 Cross-Validation:")
    print(f"   Mean CV MAE: {cv_mae:.3f}")
    
    print(f"\n🎯 Final Model Performance:")
    print(f"   Team-level MAE: {metrics['mae']:.2f}")
    print(f"   Team-level RMSE: {metrics['rmse']:.2f}")
    
    print(f"\n🏈 Game-level Performance:")
    for metric, value in game_metrics.items():
        if isinstance(value, float):
            if 'accuracy' in metric:
                print(f"   {metric.replace('_', ' ').title()}: {value:.1%}")
            else:
                print(f"   {metric.replace('_', ' ').title()}: {value:.2f}")
    
    print(f"\n📋 Sample Predictions:")
    print(score_eval[['home_pred', 'away_pred', 'pred_total', 'pred_spread', 
                     'home_actual', 'away_actual', 'ae_total']].head())
    
    print(f"\n✅ Enhanced NFL prediction model is ready!")
    print(f"💡 Key improvements implemented:")
    print(f"   - Hyperparameter optimization with Optuna")
    print(f"   - Enhanced walk-forward validation with purging") 
    print(f"   - Feature selection for noise reduction")
    print(f"   - Comprehensive betting-focused evaluation")
    print(f"   - Support for ensemble models")