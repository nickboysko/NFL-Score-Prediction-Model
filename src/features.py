"""
Enhanced Feature Engineering Module for NFL Score Prediction

This module creates advanced features for the NFL score prediction model,
including situational features, momentum indicators, and betting market features.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def create_advanced_team_features(df):
    """
    Create advanced team-level features including momentum, situational factors.
    """
    df = df.copy()
    df = df.sort_values(['team', 'date'])
    
    # Expanded rolling windows
    WINDOWS = [1, 2, 3, 4, 5, 6, 8, 10, 16]
    
    # Basic rolling stats
    for w in WINDOWS:
        # Offensive stats
        df[f'pf_avg_{w}'] = df.groupby('team')['points_for'].transform(
            lambda s: s.shift(1).rolling(w, min_periods=1).mean()
        )
        df[f'pf_std_{w}'] = df.groupby('team')['points_for'].transform(
            lambda s: s.shift(1).rolling(w, min_periods=1).std()
        )
        
        # Defensive stats
        df[f'pa_avg_{w}'] = df.groupby('team')['points_against'].transform(
            lambda s: s.shift(1).rolling(w, min_periods=1).mean()
        )
        df[f'pa_std_{w}'] = df.groupby('team')['points_against'].transform(
            lambda s: s.shift(1).rolling(w, min_periods=1).std()
        )
        
        # Point differential
        df[f'point_diff_avg_{w}'] = df.groupby('team').apply(
            lambda x: (x['points_for'] - x['points_against']).shift(1).rolling(w, min_periods=1).mean()
        ).reset_index(level=0, drop=True)
    
    # Advanced momentum features
    df['won'] = (df['points_for'] > df['points_against']).astype(int)
    
    # Recent form with exponential decay
    for decay in [0.9, 0.8, 0.7]:
        weights = np.array([decay**i for i in range(10)])[::-1]
        weights = weights / weights.sum()
        
        df[f'form_ema_{int(decay*10)}'] = df.groupby('team')['won'].transform(
            lambda s: s.shift(1).rolling(10, min_periods=1).apply(
                lambda x: np.average(x, weights=weights[-len(x):])
            )
        )
    
    # Strength of schedule
    df['opp_win_pct'] = df.groupby(['season', 'opp'])['won'].transform('mean')
    df['sos_3'] = df.groupby('team')['opp_win_pct'].transform(
        lambda s: s.shift(1).rolling(3, min_periods=1).mean()
    )
    df['sos_8'] = df.groupby('team')['opp_win_pct'].transform(
        lambda s: s.shift(1).rolling(8, min_periods=1).mean()
    )
    
    # Performance vs spread (if spread data exists)
    if 'spread_line' in df.columns:
        # ATS performance
        df['home_spread_cover'] = np.where(
            df['is_home'] == 1,
            (df['points_for'] - df['points_against']) > df['spread_line'],
            (df['points_against'] - df['points_for']) > -df['spread_line']
        )
        
        df['ats_record_5'] = df.groupby('team')['home_spread_cover'].transform(
            lambda s: s.shift(1).rolling(5, min_periods=1).mean()
        )
        df['ats_record_10'] = df.groupby('team')['home_spread_cover'].transform(
            lambda s: s.shift(1).rolling(10, min_periods=1).mean()
        )
    
    # Clutch performance (close games)
    df['margin'] = (df['points_for'] - df['points_against']).abs()
    df['close_game'] = (df['margin'] <= 7).astype(int)
    df['clutch_record'] = df.groupby('team').apply(
        lambda x: (x['won'] * x['close_game']).shift(1).rolling(10, min_periods=1).sum() / 
                  x['close_game'].shift(1).rolling(10, min_periods=1).sum().clip(lower=1)
    ).reset_index(level=0, drop=True)
    
    # Blowout tendency
    df['blowout_win'] = ((df['points_for'] - df['points_against']) >= 14).astype(int)
    df['blowout_loss'] = ((df['points_against'] - df['points_for']) >= 14).astype(int)
    
    for stat in ['blowout_win', 'blowout_loss']:
        df[f'{stat}_rate_8'] = df.groupby('team')[stat].transform(
            lambda s: s.shift(1).rolling(8, min_periods=1).mean()
        )
    
    # Offensive/Defensive efficiency trends
    df['off_efficiency'] = df['points_for'] / (df['points_for'] + df['points_against']).clip(0.01, 0.99)
    df['def_efficiency'] = df['points_against'] / (df['points_for'] + df['points_against']).clip(0.01, 0.99)
    
    for eff in ['off_efficiency', 'def_efficiency']:
        df[f'{eff}_trend_5'] = df.groupby('team')[eff].transform(
            lambda s: s.shift(1).rolling(5, min_periods=1).apply(
                lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0
            )
        )
    
    return df


def create_situational_features(df):
    """
    Create situational features based on game context.
    """
    df = df.copy()
    
    # Rest and travel factors
    df['rest_days'] = df.apply(
        lambda r: r['home_rest'] if r['is_home'] else r['away_rest'], axis=1
    )
    df['rest_advantage'] = df['home_rest'] - df['away_rest']
    
    # Rest categories
    df['short_rest'] = (df['rest_days'] <= 5).astype(int)
    df['long_rest'] = (df['rest_days'] >= 10).astype(int)
    df['bye_week'] = (df['rest_days'] >= 13).astype(int)
    
    # Game timing features
    df['date'] = pd.to_datetime(df['gameday'])
    df['month'] = df['date'].dt.month
    df['day_of_week'] = df['date'].dt.dayofweek
    
    # Season timing
    df['games_played'] = df.groupby(['team', 'season']).cumcount()
    df['season_progress'] = df['games_played'] / 17
    
    # Time-based categories
    df['early_season'] = (df['week'] <= 4).astype(int)
    df['mid_season'] = ((df['week'] > 4) & (df['week'] <= 13)).astype(int)
    df['late_season'] = ((df['week'] > 13) & (df['week'] <= 18)).astype(int)
    df['playoff_race'] = ((df['week'] >= 15) & (df['game_type'] == 'REG')).astype(int)
    
    # Weather/surface proxies
    for col in ['roof', 'surface']:
        if col in df.columns:
            df[col] = df[col].fillna('UNK')
    
    df['dome_game'] = df['roof'].str.lower().isin(['dome', 'closed']).astype(int)
    df['outdoor_game'] = (df['roof'].str.lower() == 'outdoors').astype(int)
    df['turf_game'] = (df['surface'].str.lower() == 'turf').astype(int)
    
    # Neutral site
    df['neutral'] = df['location'].str.contains('Neutral', case=False, na=False).astype(int)
    
    # Division games (assuming team names follow NFL format)
    df['division_game'] = 0  # Would need division mappings to implement fully
    
    # Prime time games (rough approximation)
    if 'gametime' in df.columns:
        df['prime_time'] = df['gametime'].str.contains('20:20|20:15|17:00', na=False).astype(int)
    else:
        df['prime_time'] = 0
    
    return df


def create_matchup_features(df):
    """
    Create features based on team vs opponent matchups.
    """
    df = df.copy()
    
    # Get opponent stats by merging
    stat_cols = [
        col for col in df.columns 
        if any(x in col for x in ['pf_', 'pa_', 'point_diff_', 'form_ema'])
    ]
    
    # Create opponent features
    opponent_stats = df[['team', 'date'] + stat_cols].copy()
    opponent_stats.columns = ['team', 'date'] + [f'opp_{col}' for col in stat_cols]
    
    df = df.merge(
        opponent_stats.rename(columns={'team': 'opp'}),
        on=['opp', 'date'],
        how='left'
    )
    
    # Matchup-specific features
    df['off_vs_def'] = df['pf_avg_5'] - df['opp_pa_avg_5']
    df['def_vs_off'] = df['pa_avg_5'] - df['opp_pf_avg_5']
    
    # Pace matchups
    df['total_pace'] = df['pf_avg_5'] + df['pa_avg_5']
    df['opp_total_pace'] = df['opp_pf_avg_5'] + df['opp_pa_avg_5']
    df['pace_differential'] = df['total_pace'] - df['opp_total_pace']
    
    # Form differential
    df['form_diff'] = df['form_ema_9'] - df['opp_form_ema_9']
    
    # Volatility matchup
    df['volatility'] = (df['pf_std_5'] + df['pa_std_5']) / 2
    df['opp_volatility'] = (df['opp_pf_std_5'] + df['opp_pa_std_5']) / 2
    df['volatility_diff'] = df['volatility'] - df['opp_volatility']
    
    return df


def create_market_features(df):
    """
    Create features based on betting market information.
    """
    df = df.copy()
    
    # Market-implied features
    for col in ['spread_line', 'total_line']:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())
    
    if 'spread_line' in df.columns and 'total_line' in df.columns:
        # Implied scores
        half_total = df['total_line'] / 2.0
        half_spread = df['spread_line'] / 2.0
        
        home_implied = half_total + half_spread
        away_implied = half_total - half_spread
        
        df['implied_points'] = np.where(df['is_home'] == 1, home_implied, away_implied)
        df['opp_implied_points'] = np.where(df['is_home'] == 1, away_implied, home_implied)
        
        # Market expectations vs recent performance
        df['implied_vs_avg'] = df['implied_points'] - df['pf_avg_5']
        df['opp_implied_vs_avg'] = df['opp_implied_points'] - df['opp_pf_avg_5']
        
        # Total line analysis
        df['implied_total'] = df['total_line']
        df['recent_total_avg'] = df['pf_avg_3'] + df['pa_avg_3']
        df['total_vs_recent'] = df['implied_total'] - df['recent_total_avg']
    
    # Moneyline features (if available)
    if 'home_moneyline' in df.columns and 'away_moneyline' in df.columns:
        # Convert moneylines to implied probabilities
        df['home_ml_prob'] = np.where(
            df['home_moneyline'] > 0,
            100 / (df['home_moneyline'] + 100),
            -df['home_moneyline'] / (-df['home_moneyline'] + 100)
        )
        df['away_ml_prob'] = np.where(
            df['away_moneyline'] > 0,
            100 / (df['away_moneyline'] + 100),
            -df['away_moneyline'] / (-df['away_moneyline'] + 100)
        )
        
        df['ml_prob'] = np.where(df['is_home'] == 1, df['home_ml_prob'], df['away_ml_prob'])
        df['opp_ml_prob'] = np.where(df['is_home'] == 1, df['away_ml_prob'], df['home_ml_prob'])
    
    return df


def create_interaction_features(df):
    """
    Create interaction features between different categories.
    """
    df = df.copy()
    
    # Rest × Form interactions
    df['rest_form_interaction'] = df['rest_days'] * df['form_ema_9']
    df['bye_week_form'] = df['bye_week'] * df['form_ema_9']
    
    # Surface × Pace interactions
    df['dome_pace'] = df['dome_game'] * df['total_pace']
    df['turf_pace'] = df['turf_game'] * df['total_pace']
    
    # Season timing × Performance
    df['late_season_form'] = df['late_season'] * df['form_ema_9']
    df['playoff_race_form'] = df['playoff_race'] * df['form_ema_9']
    
    # Home field × Rest
    df['home_rest_advantage'] = df['is_home'] * df['rest_advantage']
    
    # Market × Recent performance
    if 'implied_points' in df.columns:
        df['market_form_diff'] = (df['implied_points'] - df['pf_avg_3']) * df['form_ema_9']
    
    return df


def create_all_features(df):
    """
    Create all enhanced features.
    """
    print("🔧 Creating advanced team features...")
    df = create_advanced_team_features(df)
    
    print("🔧 Creating situational features...")
    df = create_situational_features(df)
    
    print("🔧 Creating matchup features...")
    df = create_matchup_features(df)
    
    print("🔧 Creating market features...")
    df = create_market_features(df)
    
    print("🔧 Creating interaction features...")
    df = create_interaction_features(df)
    
    print("✅ Enhanced feature engineering complete!")
    print(f"📊 Final shape: {df.shape}")
    
    # Count new features
    original_cols = ['game_id', 'season', 'week', 'gameday', 'gametime', 'game_type', 
                    'team', 'opp', 'points_for', 'points_against', 'is_home', 'date', 
                    'season_week', 'location', 'overtime', 'spread_line', 'total_line', 
                    'home_moneyline', 'away_moneyline', 'home_rest', 'away_rest', 'surface', 'roof']
    
    new_features = len([col for col in df.columns if col not in original_cols])
    print(f"🎯 Total new features created: {new_features}")
    
    return df


if __name__ == "__main__":
    print("🧪 Enhanced Feature Engineering Module")
    
    # Test with sample data
    from data_loader import load_schedules
    
    print("📅 Loading sample data...")
    sample_data = load_schedules([2023])
    
    print("🔧 Creating enhanced features...")
    featured_data = create_all_features(sample_data)
    
    print("\n📋 Sample of new features:")
    new_feature_cols = [col for col in featured_data.columns if any(x in col for x in 
                       ['form_ema', 'ats_record', 'clutch', 'efficiency', 'off_vs_def', 'pace', 'implied'])]
    
    sample_team = featured_data['team'].iloc[20]  # Skip early season games
    team_sample = featured_data[featured_data['team'] == sample_team].iloc[5:10]
    
    display_cols = ['team', 'opp', 'date', 'points_for'] + new_feature_cols[:8]
    print(team_sample[display_cols].to_string(index=False))