"""
Betting-Focused Model Training Module

This module implements specific improvements for betting accuracy:
- Separate models for spread and total predictions
- Market-aware loss functions
- Ensemble methods with betting-specific calibration
- Advanced validation focused on betting metrics
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import optuna
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, log_loss
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import VotingRegressor
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.feature_selection import SelectKBest, f_regression
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')


class BettingLoss:
    """Custom loss functions that optimize for betting accuracy."""
    
    @staticmethod
    def spread_accuracy_objective(y_true, y_pred, sample_weight=None):
        """Custom objective that penalizes wrong side of spread predictions more."""
        # Convert to spread format (positive = home favored)
        spread_errors = y_pred - y_true
        
        # Penalize being on wrong side more than just being off by amount
        wrong_side_penalty = np.where(
            (y_true > 0) != (y_pred > 0),  # Different sides of zero
            2.0,  # Double penalty for wrong side
            1.0   # Normal penalty for right side but wrong magnitude
        )
        
        loss = spread_errors ** 2 * wrong_side_penalty
        
        if sample_weight is not None:
            loss *= sample_weight
            
        return loss.mean()
    
    @staticmethod
    def total_accuracy_objective(y_true, y_pred, total_line, sample_weight=None):
        """Custom objective for over/under accuracy."""
        # Penalize being on wrong side of total line
        pred_over = y_pred > total_line
        actual_over = y_true > total_line
        
        wrong_side = pred_over != actual_over
        base_error = (y_pred - y_true) ** 2
        
        # Higher penalty for wrong side of total
        loss = np.where(wrong_side, base_error * 1.5, base_error)
        
        if sample_weight is not None:
            loss *= sample_weight
            
        return loss.mean()


def prepare_betting_data(df, test_season=2024):
    """
    Prepare data specifically optimized for betting predictions.
    """
    # Filter out games without betting lines
    df_clean = df.dropna(subset=['spread_line', 'total_line']).copy()
    
    train = df_clean[df_clean['season'] < test_season].copy()
    test = df_clean[df_clean['season'] == test_season].copy()
    
    print(f"📊 Betting data - Train: {len(train)} records, Test: {len(test)} records")
    
    # Enhanced feature set for betting
    WINDOWS = [2, 3, 4, 5, 6, 8, 10]
    
    # Core betting features
    base_features = [
        'is_home', 'spread_line', 'total_line', 'rest_days', 'neutral',
        'implied_points', 'implied_spread', 'implied_total'
    ]
    
    # Market-aware features
    market_features = [
        'spread_total_ratio', 'heavy_favorite', 'pick_em', 'high_total', 'low_total',
        'market_over_performance', 'spread_expectation_diff', 'key_number_spread', 'key_number_total'
    ]
    
    # Team strength features
    strength_features = []
    for w in WINDOWS:
        strength_features.extend([
            f'pf_avg_{w}', f'pa_avg_{w}', f'opp_pf_avg_{w}', f'opp_pa_avg_{w}',
            f'weighted_pf_{w}', f'weighted_pa_{w}', f'consistency_pf_{w}', f'consistency_pa_{w}'
        ])
    
    # Situational features
    situation_features = [
        'rest_bucket', 'div_game', 'cross_country', 'west_to_east', 'east_to_west',
        'must_win', 'meaningless', 'short_week', 'monday_night', 'prime_time',
        'cold_weather', 'dome_boost', 'weather_game'
    ]
    
    # Matchup features
    matchup_features = [
        'off_vs_def', 'def_vs_off', 'pace_matchup', 'defensive_game', 'shootout_potential',
        'off_vs_def_percentile', 'elite_vs_poor'
    ]
    
    # Trend features
    trend_features = [
        'recent_cover_rate_3', 'recent_cover_rate_5', 'recent_over_rate_3', 'recent_over_rate_5',
        'scoring_trend', 'allow_trend', 'recent_momentum'
    ]
    
    # Combine all features
    all_features = base_features + market_features + strength_features + situation_features + matchup_features + trend_features
    
    # Filter to existing columns
    existing_features = [f for f in all_features if f in train.columns]
    missing_features = [f for f in all_features if f not in train.columns]
    
    if missing_features:
        print(f"⚠️ Missing features: {missing_features[:10]}{'...' if len(missing_features) > 10 else ''}")
    
    print(f"🔧 Using {len(existing_features)} betting features")
    
    return train, test, existing_features


def optimize_betting_model(X_train, y_train, X_val, y_val, model_type='spread', n_trials=100):
    """
    Optimize model specifically for spread or total predictions.
    """
    def objective(trial):
        if model_type == 'lightgbm':
            params = {
                'objective': 'regression',
                'metric': 'mae',
                'boosting_type': 'gbdt',
                'verbosity': -1,
                'seed': 42,
                'n_estimators': trial.suggest_int('n_estimators', 300, 1500),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'num_leaves': trial.suggest_int('num_leaves', 20, 300),
                'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
                'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0),
                'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
            }
            
            model = lgb.LGBMRegressor(**params)
        else:  # XGBoost
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 300, 1500),
                'max_depth': trial.suggest_int('max_depth', 3, 12),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 20),
                'random_state': 42,
                'n_jobs': -1
            }
            
            model = xgb.XGBRegressor(**params)
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        
        # Use MAE as primary metric
        mae = mean_absolute_error(y_val, y_pred)
        return mae
    
    print(f"🔍 Optimizing {model_type} model with {n_trials} trials...")
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    print(f"✅ Best MAE: {study.best_value:.4f}")
    return study.best_params


class BettingEnsemble:
    """Ensemble specifically designed for betting predictions."""
    
    def __init__(self, models_config=None):
        if models_config is None:
            self.models_config = {
                'xgb_fast': {'type': 'xgb', 'n_estimators': 500, 'learning_rate': 0.1, 'max_depth': 6},
                'xgb_deep': {'type': 'xgb', 'n_estimators': 1000, 'learning_rate': 0.03, 'max_depth': 8},
                'lgb_fast': {'type': 'lgb', 'n_estimators': 500, 'learning_rate': 0.1, 'num_leaves': 31},
                'ridge': {'type': 'ridge', 'alpha': 1.0}
            }
        else:
            self.models_config = models_config
        
        self.models = {}
        self.weights = None
    
    def fit(self, X, y, sample_weight=None):
        """Fit all models in ensemble."""
        for name, config in self.models_config.items():
            if config['type'] == 'xgb':
                model = xgb.XGBRegressor(
                    n_estimators=config.get('n_estimators', 500),
                    learning_rate=config.get('learning_rate', 0.1),
                    max_depth=config.get('max_depth', 6),
                    random_state=42,
                    n_jobs=-1
                )
            elif config['type'] == 'lgb':
                model = lgb.LGBMRegressor(
                    n_estimators=config.get('n_estimators', 500),
                    learning_rate=config.get('learning_rate', 0.1),
                    num_leaves=config.get('num_leaves', 31),
                    random_state=42,
                    verbosity=-1
                )
            elif config['type'] == 'ridge':
                from sklearn.preprocessing import StandardScaler
                from sklearn.pipeline import Pipeline
                model = Pipeline([
                    ('scaler', StandardScaler()),
                    ('ridge', Ridge(alpha=config.get('alpha', 1.0)))
                ])
            
            print(f"  Training {name}...")
            if sample_weight is not None and config['type'] in ['xgb', 'lgb']:
                model.fit(X, y, sample_weight=sample_weight)
            else:
                model.fit(X, y)
            
            self.models[name] = model
        
        # Equal weights for now - could optimize these
        self.weights = {name: 1.0/len(self.models) for name in self.models}
        
    def predict(self, X):
        """Make weighted ensemble prediction."""
        predictions = []
        for name, model in self.models.items():
            pred = model.predict(X)
            predictions.append(pred * self.weights[name])
        
        return np.sum(predictions, axis=0)


def betting_walk_forward_validation(train, features, n_splits=5, model_type='ensemble'):
    """
    Walk-forward validation specifically for betting accuracy.
    """
    print(f"🔄 Betting-focused walk-forward validation with {n_splits} splits...")
    
    df_sorted = train.sort_values('date').copy()
    
    # Prepare features
    categorical_cols = [col for col in features if df_sorted[col].dtype == 'object']
    numerical_cols = [col for col in features if col not in categorical_cols]
    
    preprocessor = ColumnTransformer([
        ('num', 'passthrough', numerical_cols),
        ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), categorical_cols)
    ])
    
    X = df_sorted[features]
    
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    spread_results = []
    total_results = []
    
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
        print(f"\n🔄 Fold {fold}/{n_splits}")
        
        # Purge data to avoid lookahead bias
        train_end_date = df_sorted.iloc[train_idx[-1]]['date']
        purge_cutoff = train_end_date + pd.Timedelta(days=3)
        valid_mask = df_sorted.iloc[val_idx]['date'] > purge_cutoff
        val_idx = val_idx[valid_mask]
        
        if len(val_idx) == 0:
            continue
            
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        train_df = df_sorted.iloc[train_idx]
        val_df = df_sorted.iloc[val_idx]
        
        # Preprocess features
        X_train_processed = preprocessor.fit_transform(X_train)
        X_val_processed = preprocessor.transform(X_val)
        
        # Train spread model (predict point differential)
        spread_target = train_df['points_for'] - train_df['points_against']
        
        if model_type == 'ensemble':
            spread_model = BettingEnsemble()
            spread_model.fit(X_train_processed, spread_target)
        else:
            spread_model = xgb.XGBRegressor(n_estimators=800, max_depth=6, learning_rate=0.03, random_state=42)
            spread_model.fit(X_train_processed, spread_target)
        
        # Train total model
        total_target = train_df['points_for'] + train_df['points_against']
        
        if model_type == 'ensemble':
            total_model = BettingEnsemble()
            total_model.fit(X_train_processed, total_target)
        else:
            total_model = xgb.XGBRegressor(n_estimators=800, max_depth=6, learning_rate=0.03, random_state=42)
            total_model.fit(X_train_processed, total_target)
        
        # Make predictions
        spread_pred = spread_model.predict(X_val_processed)
        total_pred = total_model.predict(X_val_processed)
        
        # Calculate betting accuracy
        val_spread_actual = val_df['points_for'] - val_df['points_against']
        val_total_actual = val_df['points_for'] + val_df['points_against']
        
        # Spread accuracy (against the spread)
        spread_line_adj = np.where(val_df['is_home'], val_df['spread_line'], -val_df['spread_line'])
        pred_covers = spread_pred > spread_line_adj
        actual_covers = val_spread_actual > spread_line_adj
        spread_accuracy = (pred_covers == actual_covers).mean()
        
        # Total accuracy (over/under)
        pred_over = total_pred > val_df['total_line']
        actual_over = val_total_actual > val_df['total_line']
        total_accuracy = (pred_over == actual_over).mean()
        
        spread_results.append({
            'fold': fold,
            'spread_accuracy': spread_accuracy,
            'spread_mae': mean_absolute_error(val_spread_actual, spread_pred),
            'total_accuracy': total_accuracy,
            'total_mae': mean_absolute_error(val_total_actual, total_pred)
        })
        
        print(f"   Spread accuracy: {spread_accuracy:.1%}")
        print(f"   Total accuracy: {total_accuracy:.1%}")
    
    # Calculate average metrics
    if spread_results:
        avg_spread_acc = np.mean([r['spread_accuracy'] for r in spread_results])
        avg_total_acc = np.mean([r['total_accuracy'] for r in spread_results])
        
        print(f"\n📊 Cross-Validation Betting Results:")
        print(f"   Average spread accuracy: {avg_spread_acc:.1%}")
        print(f"   Average total accuracy: {avg_total_acc:.1%}")
        
        return spread_results, avg_spread_acc, avg_total_acc
    
    return [], 0.0, 0.0


def train_betting_models(train, test, features, optimize_params=True):
    """
    Train separate optimized models for spread and total predictions.
    """
    print("🚀 Training betting-focused models...")
    
    # Prepare data
    categorical_cols = [col for col in features if train[col].dtype == 'object']
    numerical_cols = [col for col in features if col not in categorical_cols]
    
    preprocessor = ColumnTransformer([
        ('num', 'passthrough', numerical_cols),
        ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), categorical_cols)
    ])
    
    X_train = preprocessor.fit_transform(train[features])
    X_test = preprocessor.transform(test[features])
    
    # Targets
    spread_train = train['points_for'] - train['points_against']
    total_train = train['points_for'] + train['points_against']
    spread_test = test['points_for'] - test['points_against']
    total_test = test['points_for'] + test['points_against']
    
    # Sample weights (more recent games get higher weight)
    train_dates = pd.to_datetime(train['date'])
    max_date = train_dates.max()
    days_ago = (max_date - train_dates).dt.days
    sample_weights = np.exp(-days_ago / 365)  # Exponential decay with 1-year half-life
    
    results = {}
    
    # Train spread model
    print("\n📈 Training spread prediction model...")
    
    if optimize_params:
        # Split for optimization
        split_idx = int(len(X_train) * 0.8)
        X_opt_train, X_opt_val = X_train[:split_idx], X_train[split_idx:]
        y_opt_train, y_opt_val = spread_train.iloc[:split_idx], spread_train.iloc[split_idx:]
        
        best_spread_params = optimize_betting_model(X_opt_train, y_opt_train, X_opt_val, y_opt_val, 'xgb')
    else:
        best_spread_params = {'n_estimators': 1000, 'max_depth': 6, 'learning_rate': 0.03}
    
    # Create ensemble for spread
    spread_ensemble = BettingEnsemble({
        'xgb_optimized': {'type': 'xgb', **best_spread_params},
        'xgb_conservative': {'type': 'xgb', 'n_estimators': 800, 'max_depth': 4, 'learning_rate': 0.05},
        'lgb_fast': {'type': 'lgb', 'n_estimators': 500, 'learning_rate': 0.1, 'num_leaves': 31},
        'ridge': {'type': 'ridge', 'alpha': 1.0}
    })
    
    spread_ensemble.fit(X_train, spread_train, sample_weight=sample_weights)
    spread_pred = spread_ensemble.predict(X_test)
    
    # Train total model
    print("\n📊 Training total prediction model...")
    
    if optimize_params:
        best_total_params = optimize_betting_model(X_opt_train, total_train.iloc[:split_idx], 
                                                 X_opt_val, total_train.iloc[split_idx:], 'xgb')
    else:
        best_total_params = {'n_estimators': 1000, 'max_depth': 6, 'learning_rate': 0.03}
    
    # Create ensemble for totals  
    total_ensemble = BettingEnsemble({
        'xgb_optimized': {'type': 'xgb', **best_total_params},
        'xgb_conservative': {'type': 'xgb', 'n_estimators': 800, 'max_depth': 4, 'learning_rate': 0.05},
        'lgb_fast': {'type': 'lgb', 'n_estimators': 500, 'learning_rate': 0.1, 'num_leaves': 31},
        'ridge': {'type': 'ridge', 'alpha': 1.0}
    })
    
    total_ensemble.fit(X_train, total_train, sample_weight=sample_weights)
    total_pred = total_ensemble.predict(X_test)
    
    # Evaluate betting performance
    print("\n🎯 Evaluating betting performance...")
    
    # Spread betting accuracy
    test_spread_line_adj = np.where(test['is_home'], test['spread_line'], -test['spread_line'])
    pred_covers = spread_pred > test_spread_line_adj
    actual_covers = spread_test > test_spread_line_adj
    spread_accuracy = (pred_covers == actual_covers).mean()
    
    # Total betting accuracy
    pred_over = total_pred > test['total_line']
    actual_over = total_test > test['total_line']
    total_accuracy = (pred_over == actual_over).mean()
    
    # Additional metrics
    spread_mae = mean_absolute_error(spread_test, spread_pred)
    total_mae = mean_absolute_error(total_test, total_pred)
    
    # Confidence-based accuracy (only bet when confident)
    spread_confidence = np.abs(spread_pred - test_spread_line_adj)
    total_confidence = np.abs(total_pred - test['total_line'])
    
    # Top 25% confidence bets
    spread_conf_thresh = np.percentile(spread_confidence, 75)
    total_conf_thresh = np.percentile(total_confidence, 75)
    
    high_conf_spread_mask = spread_confidence >= spread_conf_thresh
    high_conf_total_mask = total_confidence >= total_conf_thresh
    
    if high_conf_spread_mask.sum() > 0:
        spread_conf_accuracy = (pred_covers[high_conf_spread_mask] == actual_covers[high_conf_spread_mask]).mean()
    else:
        spread_conf_accuracy = 0.0
    
    if high_conf_total_mask.sum() > 0:
        total_conf_accuracy = (pred_over[high_conf_total_mask] == actual_over[high_conf_total_mask]).mean()
    else:
        total_conf_accuracy = 0.0
    
    results = {
        'spread_accuracy': spread_accuracy,
        'total_accuracy': total_accuracy,
        'spread_mae': spread_mae,
        'total_mae': total_mae,
        'spread_conf_accuracy': spread_conf_accuracy,
        'total_conf_accuracy': total_conf_accuracy,
        'high_conf_spread_pct': high_conf_spread_mask.mean(),
        'high_conf_total_pct': high_conf_total_mask.mean(),
        'spread_predictions': spread_pred,
        'total_predictions': total_pred
    }
    
    print(f"\n✅ Final Betting Results:")
    print(f"   Spread accuracy: {spread_accuracy:.1%}")
    print(f"   Total accuracy: {total_accuracy:.1%}")
    print(f"   High-confidence spread accuracy: {spread_conf_accuracy:.1%} ({high_conf_spread_mask.mean():.1%} of bets)")
    print(f"   High-confidence total accuracy: {total_conf_accuracy:.1%} ({high_conf_total_mask.mean():.1%} of bets)")
    
    return {
        'spread_model': spread_ensemble,
        'total_model': total_ensemble,
        'preprocessor': preprocessor,
        'results': results
    }


def analyze_betting_edges(test_df, spread_pred, total_pred):
    """
    Analyze potential betting edges and profitable opportunities.
    """
    print("\n🔍 Analyzing betting edges...")
    
    results = test_df.copy()
    results['spread_pred'] = spread_pred
    results['total_pred'] = total_pred
    
    # Calculate actual outcomes
    results['actual_spread'] = results['points_for'] - results['points_against']
    results['actual_total'] = results['points_for'] + results['points_against']
    
    # Adjust spread predictions for team perspective
    results['spread_line_adj'] = np.where(results['is_home'], results['spread_line'], -results['spread_line'])
    
    # Calculate edges (prediction vs line)
    results['spread_edge'] = results['spread_pred'] - results['spread_line_adj']
    results['total_edge'] = results['total_pred'] - results['total_line']
    
    # Identify profitable bets (hypothetical)
    results['spread_bet_correct'] = (
        ((results['spread_edge'] > 0) & (results['actual_spread'] > results['spread_line_adj'])) |
        ((results['spread_edge'] < 0) & (results['actual_spread'] < results['spread_line_adj']))
    )
    
    results['total_bet_correct'] = (
        ((results['total_edge'] > 0) & (results['actual_total'] > results['total_line'])) |
        ((results['total_edge'] < 0) & (results['actual_total'] < results['total_line']))
    )
    
    # Analyze by edge size
    edge_thresholds = [1, 2, 3, 4, 5]
    
    print("📊 Betting Edge Analysis:")
    print("   Spread betting by edge size:")
    for thresh in edge_thresholds:
        mask = np.abs(results['spread_edge']) >= thresh
        if mask.sum() > 0:
            accuracy = results[mask]['spread_bet_correct'].mean()
            print(f"     Edge >= {thresh}: {accuracy:.1%} accuracy ({mask.sum()} bets)")
    
    print("   Total betting by edge size:")
    for thresh in edge_thresholds:
        mask = np.abs(results['total_edge']) >= thresh
        if mask.sum() > 0:
            accuracy = results[mask]['total_bet_correct'].mean()
            print(f"     Edge >= {thresh}: {accuracy:.1%} accuracy ({mask.sum()} bets)")
    
    return results


if __name__ == "__main__":
    print("🧪 Testing Betting-Focused Model...")
    
    # This would integrate with your existing pipeline
    print("💡 To use this module:")
    print("1. Run enhanced feature engineering first")
    print("2. Call prepare_betting_data() to get betting-specific features")
    print("3. Use betting_walk_forward_validation() for CV")
    print("4. Train final models with train_betting_models()")
    print("5. Analyze edges with analyze_betting_edges()")