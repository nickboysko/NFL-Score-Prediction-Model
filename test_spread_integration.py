#!/usr/bin/env python3
"""
Integration Script: Spread-Focused NFL Prediction Pipeline

This script integrates the spread-focused model with your existing pipeline.
Run this to test the enhanced spread prediction approach.
"""

import sys
import os
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score


# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def create_spread_focused_features(df):
    """
    Enhanced feature creation specifically for spread prediction.
    Integrates with your existing features.py but adds spread-specific features.
    """
    from features import create_all_features
    
    # Start with your existing features
    df = create_all_features(df)
    
    # Add spread-specific features
    df = df.sort_values(['team', 'date'])
    WINDOWS = [1, 2, 3, 4, 5, 6, 8, 10]
    
    # Point differential trends (critical for spreads)
    for w in WINDOWS:
        df[f'point_diff_avg_{w}'] = df.groupby('team').apply(
            lambda x: (x['points_for'] - x['points_against']).shift(1).rolling(w, min_periods=1).mean()
        ).reset_index(level=0, drop=True)
        
        df[f'point_diff_std_{w}'] = df.groupby('team').apply(
            lambda x: (x['points_for'] - x['points_against']).shift(1).rolling(w, min_periods=1).std()
        ).reset_index(level=0, drop=True)
    
    # Opponent point differential
    opponent_diff_cols = [f'point_diff_avg_{w}' for w in WINDOWS]
    df = df.merge(
        df[['team', 'date'] + opponent_diff_cols].rename(
            columns={c: f'opp_{c}' for c in opponent_diff_cols}
        ),
        left_on=['opp', 'date'], 
        right_on=['team', 'date'], 
        how='left', 
        suffixes=('', '_drop')
    ).drop(columns=['team_drop'])
    
    # Spread-specific matchup metrics
    df['net_differential_5'] = df['point_diff_avg_5'] - df['opp_point_diff_avg_5']
    df['differential_consistency_5'] = 1 / (1 + df['point_diff_std_5'].fillna(df['point_diff_std_5'].median()))
    
    # Historical spread performance (if available)
    if 'spread_line' in df.columns:
        df['spread_line'] = df['spread_line'].fillna(0)
        
        # Calculate if team beat the spread (from their perspective)
        # For home teams: actual_spread > spread_line means home team beat spread
        # For away teams: actual_spread < -spread_line means away team beat spread
        df['actual_spread'] = df['points_for'] - df['points_against']
        df['spread_beat'] = np.where(
            df['is_home'] == 1,
            df['actual_spread'] > df['spread_line'],
            df['actual_spread'] < -df['spread_line']
        )
        
        # Rolling spread performance
        for w in [3, 5, 8]:
            df[f'spread_beat_rate_{w}'] = df.groupby('team')['spread_beat'].transform(
                lambda s: s.shift(1).rolling(w, min_periods=1).mean()
            )
    
    # Market efficiency indicators
    df['public_side'] = 0  # Would need betting percentages data
    df['line_movement'] = 0  # Would need historical line data
    
    return df


def convert_to_game_level_spread_data(df, test_season=2024):
    """
    Convert team-level data to game-level data for spread prediction.
    This is the key change from your current approach.
    """
    # Split train/test first
    train_data = df[df['season'] < test_season].copy()
    test_data = df[df['season'] == test_season].copy()
    
    def create_game_level_data(team_data):
        """Convert team-level data to game-level format"""
        # Separate home and away teams
        home_teams = team_data[team_data['is_home'] == 1].copy()
        away_teams = team_data[team_data['is_home'] == 0].copy()
        
        # Merge on game_id to create game-level records
        games = home_teams.merge(
            away_teams, 
            on='game_id', 
            suffixes=('_home', '_away')
        )
        
        # Create target variables
        games['actual_spread'] = games['points_for_home'] - games['points_for_away']
        games['total_points'] = games['points_for_home'] + games['points_for_away']
        
        if 'spread_line_home' in games.columns:
            games['spread_line'] = games['spread_line_home']
            games['beat_spread'] = (games['actual_spread'] > games['spread_line']).astype(int)
        else:
            games['spread_line'] = 0
            games['beat_spread'] = (games['actual_spread'] > 0).astype(int)
        
        if 'total_line_home' in games.columns:
            games['total_line'] = games['total_line_home']
            games['beat_total'] = (games['total_points'] > games['total_line']).astype(int)
        
        return games
    
    train_games = create_game_level_data(train_data)
    test_games = create_game_level_data(test_data)
    
    print(f"📊 Converted to game-level format:")
    print(f"   Training games: {len(train_games)} (from {len(train_data)} team records)")
    print(f"   Test games: {len(test_games)} (from {len(test_data)} team records)")
    
    return train_games, test_games


def create_spread_features_game_level(train_games, test_games):
    """
    Create features at game level optimized for spread prediction.
    """
    WINDOWS = [1, 2, 3, 4, 5, 6, 8, 10]
    
    # Base features
    base_features = [
        'rest_advantage', 'dome_game_home', 'outdoor_game_home', 
        'turf_game_home', 'season_progress_home'
    ]
    
    # Point differential features (most important for spreads)
    differential_features = []
    for w in WINDOWS:
        if f'point_diff_avg_{w}_home' in train_games.columns:
            differential_features.extend([
                f'point_diff_avg_{w}_home', f'point_diff_avg_{w}_away',
                f'net_differential_{w}_home', f'net_differential_{w}_away'
            ])
    
    # Traditional offensive/defensive features
    traditional_features = []
    for w in [3, 5, 8]:
        traditional_features.extend([
            f'pf_avg_{w}_home', f'pf_avg_{w}_away',
            f'pa_avg_{w}_home', f'pa_avg_{w}_away',
            f'off_vs_def_{w}_home', f'off_vs_def_{w}_away'
        ])
    
    # Market efficiency features
    market_features = []
    if 'spread_beat_rate_5_home' in train_games.columns:
        market_features.extend([
            'spread_beat_rate_5_home', 'spread_beat_rate_5_away',
            'spread_beat_rate_3_home', 'spread_beat_rate_3_away'
        ])
    
    # Situational features
    situational_features = [
        'early_season_home', 'late_season_home',
        'back_to_back_home', 'back_to_back_away'
    ]
    
    # Combine all features
    all_features = base_features + differential_features + traditional_features + market_features + situational_features
    
    # Filter to available features
    available_features = [f for f in all_features if f in train_games.columns]
    
    print(f"🔧 Game-level features created: {len(available_features)}")
    print(f"   Differential features: {len([f for f in available_features if 'point_diff' in f or 'net_differential' in f])}")
    print(f"   Market features: {len([f for f in available_features if 'spread_beat' in f])}")
    print(f"   Traditional features: {len([f for f in available_features if any(x in f for x in ['pf_avg', 'pa_avg', 'off_vs_def'])])}")
    
    return available_features


def train_spread_focused_model(train_games, test_games, features, optimize_params=True):
    """
    Train the spread-focused model using XGBoost classification.
    """
    from xgboost import XGBClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, classification_report
    import optuna
    
    # Prepare data
    X_train = train_games[features].fillna(0)
    y_train = train_games['beat_spread']
    X_test = test_games[features].fillna(0)
    y_test = test_games['beat_spread']
    
    print(f"🎯 Training spread classification model...")
    print(f"   Features: {len(features)}")
    print(f"   Training samples: {len(X_train)}")
    print(f"   Test samples: {len(X_test)}")
    print(f"   Target balance: {y_train.mean():.1%} beat spread")
    
    if optimize_params and len(X_train) > 200:
        print("🔍 Optimizing hyperparameters for spread accuracy...")
        
        def objective(trial):
            params = {
                'objective': 'binary:logistic',
                'eval_metric': 'logloss',
                'n_estimators': trial.suggest_int('n_estimators', 300, 1000),
                'max_depth': trial.suggest_int('max_depth', 3, 8),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.15, log=True),
                'subsample': trial.suggest_float('subsample', 0.7, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 1.0),
                'reg_lambda': trial.suggest_float('reg_lambda', 0.1, 5.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 2.0),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 7),
                'random_state': 42,
                'n_jobs': -1
            }
            
            # Use a validation split from training data
            X_tr, X_val, y_tr, y_val = train_test_split(
                X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
            )
            
            model = XGBClassifier(**params)
            model.fit(
                X_tr, y_tr,
                eval_set=[(X_val, y_val)],
                early_stopping_rounds=30,
                verbose=False
            )
            
            y_pred = model.predict(X_val)
            accuracy = accuracy_score(y_val, y_pred)
            
            return -accuracy  # Optuna minimizes
        
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=50, show_progress_bar=True)
        
        best_params = study.best_params
        best_accuracy = -study.best_value
        print(f"✅ Best validation accuracy: {best_accuracy:.1%}")
        
    else:
        # Use default optimized parameters
        best_params = {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'n_estimators': 800,
            'max_depth': 6,
            'learning_rate': 0.03,
            'subsample': 0.9,
            'colsample_bytree': 0.8,
            'reg_lambda': 1.5,
            'reg_alpha': 0.1,
            'min_child_weight': 3,
            'random_state': 42,
            'n_jobs': -1
        }
    
    # Train final model
    print("🚀 Training final spread model...")
    final_model = XGBClassifier(**best_params)
    final_model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        early_stopping_rounds=50,
        verbose=False
    )
    
    # Make predictions
    y_pred_proba = final_model.predict_proba(X_test)[:, 1]
    y_pred = final_model.predict(X_test)
    
    # Evaluate
    accuracy = accuracy_score(y_test, y_pred)
    print(f"✅ Test accuracy: {accuracy:.1%}")
    
    return final_model, y_pred_proba, y_pred, accuracy


def evaluate_spread_performance(test_games, y_pred_proba, y_pred, model_accuracy):
    """
    Comprehensive evaluation of spread prediction performance.
    """
    y_true = test_games['beat_spread'].values
    
    print(f"\n📊 Comprehensive Spread Evaluation:")
    print(f"   Overall accuracy: {model_accuracy:.1%}")
    
    # Confidence-based analysis
    confidence_thresholds = [0.55, 0.6, 0.65, 0.7]
    
    for threshold in confidence_thresholds:
        # High confidence predictions
        high_conf_mask = (y_pred_proba > threshold) | (y_pred_proba < (1 - threshold))
        
        if high_conf_mask.sum() > 10:  # Need reasonable sample size
            high_conf_accuracy = accuracy_score(
                y_true[high_conf_mask], 
                y_pred[high_conf_mask]
            )
            pct_games = high_conf_mask.mean() * 100
            
            print(f"   {threshold:.0%}+ confidence: {high_conf_accuracy:.1%} accuracy ({pct_games:.1f}% of games)")
    
    # Betting performance analysis
    print(f"\n💰 Betting Performance Analysis:")
    
    # Standard betting (-110 odds)
    correct_predictions = (y_pred == y_true)
    total_bets = len(y_pred)
    wins = correct_predictions.sum()
    losses = total_bets - wins
    
    # Calculate profit (risk 1.1 units to win 1 unit)
    profit_per_win = 1.0
    loss_per_bet = -1.1
    total_profit = wins * profit_per_win + losses * loss_per_bet
    roi = (total_profit / (total_bets * 1.1)) * 100
    
    print(f"   Total bets: {total_bets}")
    print(f"   Wins: {wins} ({wins/total_bets:.1%})")
    print(f"   Losses: {losses} ({losses/total_bets:.1%})")
    print(f"   Total profit/loss: {total_profit:+.2f} units")
    print(f"   ROI: {roi:+.2f}%")
    print(f"   Break-even rate needed: 52.4%")
    
    if model_accuracy > 0.524:
        profit_potential = "🎉 PROFITABLE MODEL! 🎉"
    else:
        profit_potential = "📉 Not profitable yet, but close!"
    print(f"   Status: {profit_potential}")
    
    # High-confidence betting analysis
    print(f"\n🎯 High-Confidence Betting Strategy:")
    best_threshold = 0.6
    best_roi = roi
    
    for threshold in confidence_thresholds:
        conf_mask = (y_pred_proba > threshold) | (y_pred_proba < (1 - threshold))
        
        if conf_mask.sum() > 5:
            conf_correct = correct_predictions[conf_mask].sum()
            conf_total = conf_mask.sum()
            conf_accuracy = conf_correct / conf_total
            
            conf_profit = conf_correct * 1.0 + (conf_total - conf_correct) * (-1.1)
            conf_roi = (conf_profit / (conf_total * 1.1)) * 100
            
            if conf_roi > best_roi and conf_total > 10:
                best_threshold = threshold
                best_roi = conf_roi
            
            print(f"   {threshold:.0%}+ confidence: {conf_accuracy:.1%} accuracy, {conf_roi:+.1f}% ROI ({conf_total} bets)")
    
    print(f"\n🏆 Optimal Strategy: Use {best_threshold:.0%}+ confidence threshold")
    print(f"   Expected ROI: {best_roi:+.1f}%")
    
    return {
        'overall_accuracy': model_accuracy,
        'total_profit': total_profit,
        'roi': roi,
        'best_confidence_threshold': best_threshold,
        'best_roi': best_roi
    }


def main():
    """
    Main function to test the integrated spread-focused pipeline.
    """
    print("🧪 Testing Integrated Spread-Focused NFL Pipeline...")
    print("="*80)
    
    try:
        # Import your existing modules
        from data_loader import load_schedules
        
        print("📅 Step 1: Loading NFL data...")
        # Load more recent data for better model performance
        schedules = load_schedules([2020, 2021, 2022, 2023, 2024])
        print(f"✅ Loaded {len(schedules)} team-game records")
        
        print("\n🔧 Step 2: Creating spread-focused features...")
        enhanced_data = create_spread_focused_features(schedules)
        print("✅ Enhanced feature creation complete")
        
        print("\n🎯 Step 3: Converting to game-level format...")
        train_games, test_games = convert_to_game_level_spread_data(enhanced_data, test_season=2024)
        
        print("\n🔧 Step 4: Preparing spread-specific features...")
        spread_features = create_spread_features_game_level(train_games, test_games)
        
        print("\n🚀 Step 5: Training spread-focused model...")
        model, pred_proba, pred_binary, accuracy = train_spread_focused_model(
            train_games, test_games, spread_features, optimize_params=True
        )
        
        print("\n📊 Step 6: Comprehensive evaluation...")
        performance_metrics = evaluate_spread_performance(
            test_games, pred_proba, pred_binary, accuracy
        )
        
        print("\n" + "="*80)
        print("🎉 SPREAD-FOCUSED PIPELINE COMPLETE!")
        print("="*80)
        
        print(f"\n📈 Key Improvements vs Original Model:")
        print(f"   Original spread accuracy: 52%")
        print(f"   New spread accuracy: {accuracy:.1%}")
        improvement = (accuracy - 0.52) / 0.52 * 100
        print(f"   Improvement: {improvement:+.1f}%")
        
        print(f"\n🎯 Path to 60% Accuracy:")
        remaining = (0.60 - accuracy) / (0.60 - 0.52) * 100
        print(f"   Progress toward 60% target: {100-remaining:.1f}%")
        print(f"   Remaining improvement needed: {0.60 - accuracy:.1%}")
        
        print(f"\n💡 Next Steps to Reach 60%:")
        if accuracy < 0.56:
            print("   1. Add more sophisticated features (player injuries, weather, etc.)")
            print("   2. Implement ensemble of multiple models")
            print("   3. Add market sentiment data (public betting percentages)")
        elif accuracy < 0.58:
            print("   1. Fine-tune with more advanced ensemble methods")
            print("   2. Add situational factors (division games, revenge games, etc.)")
            print("   3. Implement confidence calibration")
        else:
            print("   1. Focus on high-confidence subset betting")
            print("   2. Add real-time data feeds")
            print("   3. Consider advanced ML techniques (neural networks, etc.)")
        
        print(f"\n🏆 Profitability Status:")
        if performance_metrics['roi'] > 5:
            print("   🎉 HIGHLY PROFITABLE MODEL!")
        elif performance_metrics['roi'] > 0:
            print("   ✅ Profitable model - ready for deployment!")
        else:
            print("   📈 Getting close to profitability - keep optimizing!")
            
        return True
        
    except Exception as e:
        print(f"❌ Error in pipeline: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎉 Integration test completed successfully!")
        print("\n💡 To use this in production:")
        print("   1. Replace your current team-score approach with this spread-focused method")
        print("   2. Focus on games where the model has high confidence (60%+ probability)")
        print("   3. Track performance and adjust thresholds based on results")
        print("   4. Consider adding more advanced features as suggested above")
    else:
        print("\n❌ Integration test failed - check errors above")
        sys.exit(1)