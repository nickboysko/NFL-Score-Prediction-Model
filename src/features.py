"""
Feature Engineering Module for NFL Score Prediction

This module creates features for the NFL score prediction model,
including team-level rolling statistics, opponent features, and game context.
"""

import pandas as pd
import numpy as np


def create_team_features(df):
    """
    Create team-level rolling features and statistics.
    
    Args:
        df (pd.DataFrame): DataFrame with team-game data
    
    Returns:
        pd.DataFrame: DataFrame with engineered features
    """
    df = df.copy()
    
    # Sort for proper rolling calculations
    df = df.sort_values(['team', 'date'])
    
    # Expanded rolling windows for feature calculation
    WINDOWS = [1, 2, 3, 4, 5, 6, 8, 10]
    
    # Calculate rolling offensive and defensive statistics
    for w in WINDOWS:
        # Offensive form (points scored)
        df[f'pf_avg_{w}'] = df.groupby('team')['points_for'].transform(
            lambda s: s.shift(1).rolling(w, min_periods=1).mean()
        )
        df[f'pf_std_{w}'] = df.groupby('team')['points_for'].transform(
            lambda s: s.shift(1).rolling(w, min_periods=1).std()
        )
        
        # Defensive form (points allowed)
        df[f'pa_avg_{w}'] = df.groupby('team')['points_against'].transform(
            lambda s: s.shift(1).rolling(w, min_periods=1).mean()
        )
        df[f'pa_std_{w}'] = df.groupby('team')['points_against'].transform(
            lambda s: s.shift(1).rolling(w, min_periods=1).std()
        )
    
    # Create opponent features by merging team stats
    opponent_cols = [f'pf_avg_{w}' for w in WINDOWS] + [f'pa_avg_{w}' for w in WINDOWS]
    
    # Merge opponent features
    df = df.merge(
        df[['team', 'date'] + opponent_cols].rename(
            columns={c: f'opp_{c}' for c in opponent_cols}
        ),
        left_on=['opp', 'date'], 
        right_on=['team', 'date'], 
        how='left', 
        suffixes=('', '_drop')
    ).drop(columns=['team_drop'])
    
    # Game context features
    df['rest_days'] = df.apply(
        lambda r: r['home_rest'] if r['is_home'] else r['away_rest'], 
        axis=1
    )
    
    # Neutral site flag
    df['neutral'] = (df['location'].str.contains('Neutral', case=False, na=False)).astype(int)
    
    # Encode roof and surface features
    for col in ['roof', 'surface']:
        df[col] = df[col].fillna('UNK')
    
    # Calculate win/loss from points (for win streaks)
    df['won'] = (df['points_for'] > df['points_against']).astype(int)
    
    # Win streaks (using calculated win/loss)
    df['win_streak'] = df.groupby('team').apply(
        lambda x: x['won'].shift(1).rolling(5, min_periods=1).sum()
    ).reset_index(level=0, drop=True)
    
    # Loss streaks
    df['loss_streak'] = df.groupby('team').apply(
        lambda x: ((1 - x['won']).shift(1).rolling(5, min_periods=1).sum())
    ).reset_index(level=0, drop=True)
    
    # Recent performance trends
    df['recent_form'] = df.groupby('team').apply(
        lambda x: x['won'].shift(1).rolling(5, min_periods=1).mean()
    ).reset_index(level=0, drop=True)
    
    # Season progression
    df['games_played'] = df.groupby(['team', 'season']).cumcount()
    df['season_progress'] = df['games_played'] / 17  # Percentage through season
    
    # Rest advantage
    df['rest_differential'] = df['home_rest'] - df['away_rest']
    
    # Enhanced surface and roof impact
    df['dome_game'] = (df['roof'].str.lower().isin(['dome', 'closed'])).astype(int)
    df['outdoor_game'] = (df['roof'].str.lower() == 'outdoors').astype(int)
    df['turf_game'] = (df['surface'].str.lower() == 'turf').astype(int)
    
    # Add offensive vs defensive matchups
    df['off_vs_def'] = df['pf_avg_5'] - df['opp_pa_avg_5']  # How team's offense matches opponent's defense
    df['def_vs_off'] = df['pa_avg_5'] - df['opp_pf_avg_5']  # How team's defense matches opponent's offense
    
    # Pace of play
    df['total_avg_5'] = (df['pf_avg_5'] + df['pa_avg_5'])  # Team's average total points in games
    df['opp_total_avg_5'] = (df['opp_pf_avg_5'] + df['opp_pa_avg_5'])  # Opponent's average total
    
    return df


def create_game_features(df):
    """
    Create game-level features and context.
    
    Args:
        df (pd.DataFrame): DataFrame with team-game data
    
    Returns:
        pd.DataFrame: DataFrame with game-level features
    """
    df = df.copy()
    
    # Week-based features
    df['early_season'] = (df['week'] <= 4).astype(int)
    df['late_season'] = (df['week'] >= 15).astype(int)
    df['mid_season'] = ((df['week'] > 4) & (df['week'] < 15)).astype(int)
    
    # Game type features
    df['is_playoff'] = (df['game_type'] == 'POST').astype(int)
    df['is_regular_season'] = (df['game_type'] == 'REG').astype(int)
    
    # Playoff race implications (weeks 15-18)
    df['playoff_race'] = ((df['week'] >= 15) & (df['game_type'] == 'REG')).astype(int)
    
    # Overtime games
    df['went_overtime'] = (df['overtime'] == 1).astype(int)
    
    # Home field advantage (could be enhanced with historical data)
    df['home_advantage'] = df['is_home'].astype(int)
    
    return df


def create_all_features(df):
    """
    Create all features by running both team and game feature functions.
    
    Args:
        df (pd.DataFrame): Raw DataFrame from data_loader.load_schedules()
    
    Returns:
        pd.DataFrame: DataFrame with all engineered features
    """
    print("Creating team-level features...")
    df = create_team_features(df)
    
    print("Creating game-level features...")
    df = create_game_features(df)
    
    print("Feature engineering complete!")
    print(f"Final shape: {df.shape}")
    print(f"Total features created: {len([col for col in df.columns if col not in ['game_id', 'season', 'week', 'gameday', 'gametime', 'game_type', 'team', 'opp', 'points_for', 'points_against', 'is_home', 'date', 'season_week', 'location', 'overtime', 'spread_line', 'total_line', 'home_moneyline', 'away_moneyline', 'home_rest', 'away_rest', 'surface', 'roof']])}")
    
    return df


if __name__ == "__main__":
    print("Feature engineering module - testing feature creation...")
    
    # Test with sample data
    from data_loader import load_schedules
    
    # Load a small sample for testing
    print("Loading sample data...")
    sample_data = load_schedules([2023])
    
    print("Creating features...")
    featured_data = create_all_features(sample_data)
    
    print("\nSample of engineered features:")
    feature_cols = [col for col in featured_data.columns if any(x in col for x in ['pf_avg', 'pa_avg', 'opp_', 'rest_days', 'neutral', 'win_streak', 'recent_form'])]
    print(featured_data[['team', 'opp', 'date'] + feature_cols].head())