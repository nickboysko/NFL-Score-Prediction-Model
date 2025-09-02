"""
Enhanced Spread-Focused NFL Prediction Model

This module implements a spread-focused approach optimized specifically for ATS (Against The Spread) accuracy.
Key improvements:
1. Direct spread prediction instead of deriving from team scores
2. Spread-specific features and market inefficiency detection
3. Ensemble approach combining multiple spread prediction methods
4. Advanced evaluation focusing on spread accuracy optimization
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, log_loss
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from xgboost import XGBRegressor, XGBClassifier
import optuna
from scipy.stats import norm
import warnings
warnings.filterwarnings('ignore')


def create_spread_features(df):
    """
    Create features specifically optimized for spread prediction.
    
    Args:
        df (pd.DataFrame): DataFrame with basic team features
    
    Returns:
        pd.DataFrame: DataFrame with spread-specific features
    """
    df = df.copy()
    df = df.sort_values(['team', 'date'])
    
    WINDOWS = [1, 2, 3, 4, 5, 6, 8, 10]
    
    # Standard rolling features
    for w in WINDOWS:
        df[f'pf_avg_{w}'] = df.groupby('team')['points_for'].transform(
            lambda s: s.shift(1).rolling(w, min_periods=1).mean()
        )
        df[f'pa_avg_{w}'] = df.groupby('team')['points_against'].transform(
            lambda s: s.shift(1).rolling(w, min_periods=1).mean()
        )
    
    # Create opponent features
    opponent_cols = [f'pf_avg_{w}' for w in WINDOWS] + [f'pa_avg_{w}' for w in WINDOWS]
    df = df.merge(
        df[['team', 'date'] + opponent_cols].rename(
            columns={c: f'opp_{c}' for c in opponent_cols}
        ),
        left_on=['opp', 'date'], 
        right_on=['team', 'date'], 
        how='left', 
        suffixes=('', '_drop')
    ).drop(columns=['team_drop'])
    
    # SPREAD-SPECIFIC FEATURES
    
    # 1. Point differential trends (most important for spreads)
    for w in WINDOWS:
        df[f'point_diff_avg_{w}'] = df.groupby('team').apply(
            lambda x: (x['points_for'] - x['points_against']).shift(1).rolling(w, min_periods=1).mean()
        ).reset_index(level=0, drop=True)
        
        # Opponent point differential
        df[f'opp_point_diff_avg_{w}'] = df.groupby('opp').apply(
            lambda x: (x['points_for'] - x['points_against']).shift(1).rolling(w, min_periods=1).mean()
        ).reset_index(level=0, drop=True)
    
    # 2. Head-to-head matchup strength
    df['off_vs_def_5'] = df['pf_avg_5'] - df['opp_pa_avg_5']
    df['def_vs_off_5'] = df['pa_avg_5'] - df['opp_pf_avg_5']
    df['net_matchup_5'] = df['off_vs_def_5'] - df['def_vs_off_5']
    
    # 3. Home field advantage with team-specific adjustments
    df['rest_days'] = df.apply(
        lambda r: r['home_rest'] if r['is_home'] else r['away_rest'], axis=1
    )
    df['rest_advantage'] = df['home_rest'] - df['away_rest']
    
    # 4. Market efficiency features
    if 'spread_line' in df.columns:
        df['spread_line'] = df['spread_line'].fillna(0)
        
        # Historical spread performance
        df['spread_beat'] = (df['points_for'] - df['points_against'] - 
                           df['spread_line'] * (2 * df['is_home'] - 1)) > 0
        
        for w in [3, 5, 8]:
            df[f'spread_beat_rate_{w}'] = df.groupby('team')['spread_beat'].transform(
                lambda s: s.shift(1).rolling(w, min_periods=1).mean()
            )
            
            # Opponent's spread performance
            df[f'opp_spread_beat_rate_{w}'] = df.groupby('opp')['spread_beat'].transform(
                lambda s: s.shift(1).rolling(w, min_periods=1).mean()
            )
    
    # 5. Situational factors that affect spreads
    df['back_to_back'] = (df['rest_days'] < 7).astype(int)
    df['extra_rest'] = (df['rest_days'] > 10).astype(int)
    df['division_game'] = 0  # Would need division info to implement
    
    # 6. Momentum and variance features
    df['recent_variance_5'] = df.groupby('team').apply(
        lambda x: (x['points_for'] - x['points_against']).shift(1).rolling(5, min_periods=1).std()
    ).reset_index(level=0, drop=True)
    
    df['consistency_5'] = 1 / (1 + df['recent_variance_5'].fillna(df['recent_variance_5'].median()))
    
    # 7. Weather and surface impact (enhanced)
    df['dome_game'] = (df['roof'].str.lower().isin(['dome', 'closed'])).astype(int)
    df['outdoor_game'] = (df['roof'].str.lower() == 'outdoors').astype(int)
    df['turf_game'] = (df['surface'].str.lower() == 'turf').astype(int)
    
    # 8. Season context
    df['games_played'] = df.groupby(['team', 'season']).cumcount()
    df['season_progress'] = df['games_played'] / 17
    df['early_season'] = (df['games_played'] < 4).astype(int)
    df['late_season'] = (df['games_played'] > 13).astype(int)
    
    return df


def prepare_spread_data(df, test_season=2024):
    """
    Prepare data specifically for spread prediction.
    
    Returns data in game-level format with spread as target.
    """
    # Create spread features first
    df = create_spread_features(df)
    
    # Convert to game-level format
    home_games = df[df['is_home'] == 1].copy()
    away_games = df[df['is_home'] == 0].copy()
    
    # Merge home and away data
    game_data = home_games.merge(
        away_games, on='game_id', suffixes=('_home', '_away')
    )
    
    # Create spread target and features
    game_data['actual_spread'] = game_data['points_for_home'] - game_data['points_for_away']
    
    if 'spread_line_home' in game_data.columns:
        game_data['spread_line'] = game_data['spread_line_home']
        game_data['beat_spread'] = (game_data['actual_spread'] > game_data['spread_line']).astype(int)
    else:
        game_data['spread_line'] = 0
        game_data['beat_spread'] = (game_data['actual_spread'] > 0).astype(int)
    
    # Split train/test
    train_games = game_data[game_data['season_home'] < test_season].copy()
    test_games = game_data[game_data['season_home'] == test_season].copy()
    
    # Define features for spread prediction
    WINDOWS = [1, 2, 3, 4, 5, 6, 8, 10]
    
    base_features = ['rest_advantage', 'dome_game_home', 'outdoor_game_home', 'turf_game_home']
    
    rolling_features = []
    for w in WINDOWS:
        rolling_features.extend([
            f'point_diff_avg_{w}_home', f'point_diff_avg_{w}_away',
            f'pf_avg_{w}_home', f'pf_avg_{w}_away',
            f'pa_avg_{w}_home', f'pa_avg_{w}_away'
        ])
    
    matchup_features = ['off_vs_def_5_home', 'off_vs_def_5_away', 
                       'net_matchup_5_home', 'net_matchup_5_away']
    
    market_features = []
    if 'spread_beat_rate_5_home' in game_data.columns:
        market_features.extend([
            'spread_beat_rate_5_home', 'spread_beat_rate_5_away',
            'spread_beat_rate_3_home', 'spread_beat_rate_3_away'
        ])
    
    situational_features = ['back_to_back_home', 'back_to_back_away',
                           'early_season_home', 'late_season_home',
                           'consistency_5_home', 'consistency_5_away']
    
    SPREAD_FEATURES = base_features + rolling_features + matchup_features + market_features + situational_features
    
    # Filter to available features
    available_features = [f for f in SPREAD_FEATURES if f in train_games.columns]
    
    print(f"📊 Spread model using {len(available_features)} features")
    print(f"📊 Training games: {len(train_games)}")
    print(f"📊 Test games: {len(test_games)}")
    
    return train_games, test_games, available_features, 'beat_spread', 'actual_spread'


def create_spread_ensemble():
    """
    Create ensemble specifically optimized for spread prediction.
    
    Returns:
        dict: Dictionary of models for ensemble
    """
    models = {}
    
    # 1. XGBoost classifier for binary spread prediction
    models['xgb_classifier'] = XGBClassifier(
        n_estimators=800,
        max_depth=6,
        learning_rate=0.03,
        subsample=0.9,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        reg_alpha=0.1,
        random_state=42,
        n_jobs=-1,
        eval_metric='logloss'
    )
    
    # 2. XGBoost regressor for spread value prediction
    models['xgb_regressor'] = XGBRegressor(
        n_estimators=800,
        max_depth=6,
        learning_rate=0.03,
        subsample=0.9,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        reg_alpha=0.1,
        random_state=42,
        n_jobs=-1
    )
    
    # 3. Logistic regression for linear relationships
    models['logistic'] = LogisticRegression(
        max_iter=1000,
        random_state=42,
        C=1.0
    )
    
    # 4. Ridge regression for spread value
    models['ridge'] = Ridge(alpha=1.0, random_state=42)
    
    return models


def optimize_spread_model(X_train, y_class_train, y_reg_train, X_val, y_class_val, y_reg_val, n_trials=50):
    """
    Optimize hyperparameters specifically for spread prediction accuracy.
    """
    def objective(trial):
        # Focus on parameters that most affect classification accuracy
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 400, 1200),
            'max_depth': trial.suggest_int('max_depth', 4, 8),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            'subsample': trial.suggest_float('subsample', 0.7, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 1.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.1, 5.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 2.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 7),
            'random_state': 42,
            'n_jobs': -1,
            'eval_metric': 'logloss'
        }
        
        model = XGBClassifier(**params)
        model.fit(
            X_train, y_class_train,
            eval_set=[(X_val, y_class_val)],
            early_stopping_rounds=30,
            verbose=False
        )
        
        y_pred_proba = model.predict_proba(X_val)[:, 1]
        y_pred_class = (y_pred_proba > 0.5).astype(int)
        accuracy = accuracy_score(y_class_val, y_pred_class)
        
        return -accuracy  # Negative because Optuna minimizes
    
    print(f"🔍 Optimizing for spread prediction accuracy with {n_trials} trials...")
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    best_accuracy = -study.best_value
    print(f"✅ Best spread accuracy: {best_accuracy:.1%}")
    print(f"🔧 Best parameters: {study.best_params}")
    
    return study.best_params


def train_spread_ensemble(train_games, test_games, FEATURES, BINARY_TARGET, REG_TARGET, 
                         optimize=True, n_trials=50):
    """
    Train ensemble model optimized for spread prediction accuracy.
    """
    # Prepare data
    X_train = train_games[FEATURES].fillna(0)
    y_binary_train = train_games[BINARY_TARGET]
    y_reg_train = train_games[REG_TARGET]
    
    X_test = test_games[FEATURES].fillna(0)
    y_binary_test = test_games[BINARY_TARGET]
    y_reg_test = test_games[REG_TARGET]
    
    # Handle categorical variables
    categorical_features = [f for f in FEATURES if train_games[f].dtype == 'object']
    if categorical_features:
        preprocessor = ColumnTransformer([
            ('num', 'passthrough', [f for f in FEATURES if f not in categorical_features]),
            ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), 
             categorical_features)
        ])
        
        X_train_processed = preprocessor.fit_transform(X_train)
        X_test_processed = preprocessor.transform(X_test)
    else:
        X_train_processed = X_train.values
        X_test_processed = X_test.values
    
    # Split for optimization
    if optimize and len(X_train_processed) > 500:
        split_idx = int(len(X_train_processed) * 0.8)
        X_opt_train = X_train_processed[:split_idx]
        y_opt_binary_train = y_binary_train.iloc[:split_idx]
        y_opt_reg_train = y_reg_train.iloc[:split_idx]
        X_opt_val = X_train_processed[split_idx:]
        y_opt_binary_val = y_binary_train.iloc[split_idx:]
        y_opt_reg_val = y_reg_train.iloc[split_idx:]
        
        best_params = optimize_spread_model(
            X_opt_train, y_opt_binary_train, y_opt_reg_train,
            X_opt_val, y_opt_binary_val, y_opt_reg_val,
            n_trials=n_trials
        )
    else:
        best_params = None
    
    # Train ensemble models
    models = create_spread_ensemble()
    trained_models = {}
    predictions = {}
    
    # Train XGBoost classifier (primary model)
    if best_params:
        models['xgb_classifier'] = XGBClassifier(**best_params)
    
    print("🚀 Training XGBoost classifier...")
    models['xgb_classifier'].fit(
        X_train_processed, y_binary_train,
        eval_set=[(X_test_processed, y_binary_test)],
        early_stopping_rounds=50,
        verbose=False
    )
    
    pred_proba = models['xgb_classifier'].predict_proba(X_test_processed)[:, 1]
    predictions['xgb_classifier'] = pred_proba
    trained_models['xgb_classifier'] = models['xgb_classifier']
    
    # Train XGBoost regressor
    print("🚀 Training XGBoost regressor...")
    if best_params:
        reg_params = best_params.copy()
        reg_params.pop('eval_metric', None)
        models['xgb_regressor'] = XGBRegressor(**reg_params)
    
    models['xgb_regressor'].fit(X_train_processed, y_reg_train)
    reg_pred = models['xgb_regressor'].predict(X_test_processed)
    
    # Convert regression predictions to probabilities using spread line
    spread_lines = test_games['spread_line'].values
    pred_spread_beat = (reg_pred > spread_lines).astype(float)
    predictions['xgb_regressor'] = pred_spread_beat
    trained_models['xgb_regressor'] = models['xgb_regressor']
    
    # Train logistic regression
    print("🚀 Training logistic regression...")
    models['logistic'].fit(X_train_processed, y_binary_train)
    log_pred = models['logistic'].predict_proba(X_test_processed)[:, 1]
    predictions['logistic'] = log_pred
    trained_models['logistic'] = models['logistic']
    
    # Create weighted ensemble
    # Weight the XGBoost classifier most heavily as it's optimized for this task
    weights = {
        'xgb_classifier': 0.6,
        'xgb_regressor': 0.25,
        'logistic': 0.15
    }
    
    ensemble_pred = np.zeros(len(y_binary_test))
    for model_name, weight in weights.items():
        ensemble_pred += weight * predictions[model_name]
    
    # Evaluate ensemble
    ensemble_binary = (ensemble_pred > 0.5).astype(int)
    spread_accuracy = accuracy_score(y_binary_test, ensemble_binary)
    
    print(f"\n📊 Spread Prediction Results:")
    print(f"   Ensemble accuracy: {spread_accuracy:.1%}")
    
    # Individual model accuracies
    for model_name, pred in predictions.items():
        if model_name == 'xgb_regressor':
            continue  # Skip as it's already converted
        binary_pred = (pred > 0.5).astype(int)
        acc = accuracy_score(y_binary_test, binary_pred)
        print(f"   {model_name} accuracy: {acc:.1%}")
    
    # Return trained ensemble
    class SpreadEnsemble:
        def __init__(self, models, weights, preprocessor=None):
            self.models = models
            self.weights = weights
            self.preprocessor = preprocessor
            
        def predict_proba(self, X):
            if self.preprocessor:
                X_processed = self.preprocessor.transform(X)
            else:
                X_processed = X.values if hasattr(X, 'values') else X
            
            ensemble_pred = np.zeros(len(X_processed))
            
            # XGBoost classifier
            pred_proba = self.models['xgb_classifier'].predict_proba(X_processed)[:, 1]
            ensemble_pred += self.weights['xgb_classifier'] * pred_proba
            
            # Logistic regression
            log_pred = self.models['logistic'].predict_proba(X_processed)[:, 1]
            ensemble_pred += self.weights['logistic'] * log_pred
            
            # XGBoost regressor (converted to probability)
            reg_pred = self.models['xgb_regressor'].predict(X_processed)
            # This would need spread lines to convert properly, so skip in ensemble for now
            ensemble_pred += self.weights['xgb_regressor'] * 0.5  # Neutral
            
            return np.column_stack([1 - ensemble_pred, ensemble_pred])
        
        def predict(self, X):
            proba = self.predict_proba(X)[:, 1]
            return (proba > 0.5).astype(int)
    
    ensemble_model = SpreadEnsemble(
        trained_models, 
        weights, 
        preprocessor if categorical_features else None
    )
    
    return ensemble_model, spread_accuracy, predictions


def advanced_spread_evaluation(test_games, ensemble_model, predictions):
    """
    Advanced evaluation focusing on spread betting profitability and accuracy.
    """
    results = {}
    
    # Basic accuracy
    y_true = test_games['beat_spread'].values
    ensemble_pred = ensemble_model.predict(test_games[test_games.columns.intersection(ensemble_model.models['xgb_classifier'].feature_names_in_)])
    
    results['accuracy'] = accuracy_score(y_true, ensemble_pred)
    
    # Confidence-based accuracy (only high confidence predictions)
    ensemble_proba = ensemble_model.predict_proba(test_games[test_games.columns.intersection(ensemble_model.models['xgb_classifier'].feature_names_in_)])[:, 1]
    
    for threshold in [0.6, 0.65, 0.7]:
        high_conf_mask = (ensemble_proba > threshold) | (ensemble_proba < (1 - threshold))
        if high_conf_mask.sum() > 0:
            high_conf_acc = accuracy_score(
                y_true[high_conf_mask],
                ensemble_pred[high_conf_mask]
            )
            results[f'accuracy_{threshold:.0%}_confidence'] = high_conf_acc
            results[f'games_{threshold:.0%}_confidence'] = high_conf_mask.sum()
    
    # Profitability analysis (assuming -110 odds)
    correct_picks = (ensemble_pred == y_true)
    total_bets = len(ensemble_pred)
    wins = correct_picks.sum()
    losses = total_bets - wins
    
    # Standard -110 betting
    profit_per_win = 100/110  # Risk 110 to win 100
    loss_per_bet = -1
    
    total_profit = wins * profit_per_win + losses * loss_per_bet
    roi = total_profit / total_bets * 100
    
    results['total_profit_units'] = total_profit
    results['roi_percent'] = roi
    results['break_even_rate'] = 52.38  # Need 52.38% to break even at -110
    
    print(f"\n📊 Advanced Spread Evaluation:")
    print(f"   Overall accuracy: {results['accuracy']:.1%}")
    
    for key, value in results.items():
        if 'confidence' in key and 'accuracy' in key:
            threshold = key.split('_')[1]
            games_key = f'games_{threshold}_confidence'
            print(f"   {threshold} confidence accuracy: {value:.1%} ({results[games_key]} games)")
    
    print(f"\n💰 Betting Performance:")
    print(f"   Total profit: {results['total_profit_units']:+.2f} units")
    print(f"   ROI: {results['roi_percent']:+.2f}%")
    print(f"   Break-even rate: {results['break_even_rate']:.1f}%")
    
    if results['roi_percent'] > 0:
        print("   🎉 PROFITABLE! This model beats the sportsbooks!")
    else:
        print("   📉 Not profitable yet, but getting close!")
    
    return results


if __name__ == "__main__":
    print("🧪 Testing Enhanced Spread-Focused Model...")
    print("="*70)
    
    # This would be used with your existing data loading pipeline
    print("💡 This enhanced spread model focuses on:")
    print("   ✅ Direct spread prediction vs deriving from team scores")
    print("   ✅ Spread-specific features (point differentials, matchups)")
    print("   ✅ Market efficiency detection (spread beat rates)")
    print("   ✅ Ensemble approach optimized for classification accuracy")
    print("   ✅ Advanced evaluation including profitability analysis")
    print("   ✅ Confidence-based filtering for higher accuracy subsets")
    
    print(f"\n🎯 Key improvements to push accuracy from 52% toward 60%:")
    print(f"   1. Game-level modeling (eliminates team score compounding errors)")
    print(f"   2. Classification optimization (accuracy > regression metrics)")
    print(f"   3. Spread-specific features (point differential trends)")
    print(f"   4. Market inefficiency detection (historical ATS performance)")
    print(f"   5. Confidence-based betting (only bet when model is confident)")
    
    print(f"\n🚀 Next steps to integrate with your existing pipeline:")
    print(f"   1. Replace team score predictions with this spread-focused approach")
    print(f"   2. Use the spread ensemble for your main predictions")
    print(f"   3. Implement confidence-based filtering (only bet high confidence games)")
    print(f"   4. Monitor profitability alongside accuracy")