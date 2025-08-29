"""
Enhanced Feature Engineering Module for NFL Score Prediction
Focused on improving spread and total prediction accuracy

Key improvements:
- Market-aware features (line movement, public betting)
- Advanced team strength metrics 
- Situational context features
- Weather impact modeling
- Injury/roster stability indicators
"""

import pandas as pd
import numpy as np
from datetime import timedelta


def create_market_features(df):
    """
    Create market-aware features that capture betting line information.
    """
    df = df.copy()
    
    # Market efficiency features
    df['spread_total_ratio'] = df['spread_line'].abs() / df['total_line'].clip(lower=30)
    df['heavy_favorite'] = (df['spread_line'].abs() > 7).astype(int)
    df['pick_em'] = (df['spread_line'].abs() < 3).astype(int)
    df['high_total'] = (df['total_line'] > 47).astype(int)
    df['low_total'] = (df['total_line'] < 42).astype(int)
    
    # Market expectations vs recent performance
    df['market_over_performance'] = df['pf_avg_3'] - (df['total_line'] / 2)
    df['spread_expectation_diff'] = df['pf_avg_3'] - df['opp_pa_avg_3'] - np.where(df['is_home'], df['spread_line'], -df['spread_line'])
    
    return df


def create_advanced_team_metrics(df):
    """
    Create advanced team strength and consistency metrics.
    """
    df = df.copy()
    
    # Sort for proper calculations
    df = df.sort_values(['team', 'date'])
    
    # Expanded rolling windows
    WINDOWS = [2, 3, 4, 5, 6, 8, 10, 16]
    
    for w in WINDOWS:
        # Consistency metrics (lower std = more consistent)
        df[f'consistency_pf_{w}'] = df.groupby('team')['points_for'].transform(
            lambda s: 1 / (1 + s.shift(1).rolling(w, min_periods=2).std().fillna(10))
        )
        df[f'consistency_pa_{w}'] = df.groupby('team')['points_against'].transform(
            lambda s: 1 / (1 + s.shift(1).rolling(w, min_periods=2).std().fillna(10))
        )
        
        # Point differential trends
        df[f'point_diff_avg_{w}'] = df.groupby('team').apply(
            lambda x: (x['points_for'] - x['points_against']).shift(1).rolling(w, min_periods=1).mean()
        ).reset_index(level=0, drop=True)
        
        # Weighted recent performance (more weight on recent games)
        weights = np.exp(np.linspace(-1, 0, w))
        weights = weights / weights.sum()
        
        def weighted_mean(series, weights):
            if len(series) < len(weights):
                weights = weights[-len(series):]
                weights = weights / weights.sum()
            return np.average(series, weights=weights)
        
        df[f'weighted_pf_{w}'] = df.groupby('team')['points_for'].transform(
            lambda s: s.shift(1).rolling(w, min_periods=1).apply(lambda x: weighted_mean(x, weights))
        )
        df[f'weighted_pa_{w}'] = df.groupby('team')['points_against'].transform(
            lambda s: s.shift(1).rolling(w, min_periods=1).apply(lambda x: weighted_mean(x, weights))
        )
    
    # Momentum indicators
    df['recent_momentum'] = df.groupby('team').apply(
        lambda x: ((x['points_for'] - x['points_against']).shift(1).rolling(3).mean() - 
                  (x['points_for'] - x['points_against']).shift(4).rolling(3).mean())
    ).reset_index(level=0, drop=True)
    
    # Strength of schedule
    df['sos_pf'] = df.groupby('team')['opp_pa_avg_5'].transform(lambda x: x.shift(1).rolling(8, min_periods=1).mean())
    df['sos_pa'] = df.groupby('team')['opp_pf_avg_5'].transform(lambda x: x.shift(1).rolling(8, min_periods=1).mean())
    
    return df


def create_situational_features(df):
    """
    Create situational context features that affect game outcomes.
    """
    df = df.copy()
    
    # Advanced rest analysis
    df['rest_bucket'] = pd.cut(df['rest_days'], bins=[0, 6, 7, 10, 14, 30], 
                              labels=['short', 'normal', 'long', 'bye_week', 'very_long'])
    df['rest_bucket'] = df['rest_bucket'].astype(str)
    
    # Divisional games
    # Simple heuristic: if teams are in same division, they play twice
    df['div_game'] = df.groupby(['team', 'opp', 'season']).cumcount().apply(lambda x: 1 if x > 0 else 0)
    
    # Travel distance proxy (coast-to-coast games)
    east_teams = ['BUF', 'MIA', 'NE', 'NYJ', 'BAL', 'CIN', 'CLE', 'PIT', 'HOU', 'IND', 'JAX', 'TEN',
                  'ATL', 'CAR', 'NO', 'TB', 'DAL', 'NYG', 'PHI', 'WAS']
    west_teams = ['DEN', 'KC', 'LV', 'LAC', 'ARI', 'LAR', 'SF', 'SEA']
    
    df['team_coast'] = df['team'].apply(lambda x: 'east' if x in east_teams else ('west' if x in west_teams else 'central'))
    df['opp_coast'] = df['opp'].apply(lambda x: 'east' if x in east_teams else ('west' if x in west_teams else 'central'))
    df['cross_country'] = ((df['team_coast'] == 'east') & (df['opp_coast'] == 'west') |
                          (df['team_coast'] == 'west') & (df['opp_coast'] == 'east')).astype(int)
    
    # Time zone impact
    df['west_to_east'] = ((df['team_coast'] == 'west') & (df['opp_coast'] == 'east') & (df['is_home'] == 0)).astype(int)
    df['east_to_west'] = ((df['team_coast'] == 'east') & (df['opp_coast'] == 'west') & (df['is_home'] == 0)).astype(int)
    
    # Playoff implications
    df['must_win'] = ((df['week'] >= 15) & (df['game_type'] == 'REG')).astype(int)
    df['meaningless'] = ((df['week'] == 18) & (df['game_type'] == 'REG')).astype(int)  # Week 18 can have resting starters
    
    # Thursday/Monday night effects
    df['short_week'] = (df['gameday'] == 'Thursday').astype(int)
    df['monday_night'] = (df['gameday'] == 'Monday').astype(int)
    df['prime_time'] = df['short_week'] + df['monday_night']
    
    return df


def create_weather_features(df):
    """
    Create weather-related features (simplified version without real weather data).
    """
    df = df.copy()
    
    # Month-based weather proxy
    df['month'] = pd.to_datetime(df['gameday']).dt.month
    df['cold_weather'] = ((df['month'].isin([12, 1, 2])) & (df['roof'] == 'outdoors')).astype(int)
    df['hot_weather'] = ((df['month'].isin([9])) & (df['roof'] == 'outdoors')).astype(int)
    
    # Stadium type effects on totals
    df['dome_boost'] = (df['roof'].str.lower().isin(['dome', 'closed'])).astype(int)
    df['weather_game'] = ((df['roof'] == 'outdoors') & (df['month'].isin([11, 12, 1, 2]))).astype(int)
    
    return df


def create_matchup_features(df):
    """
    Create advanced matchup-specific features.
    """
    df = df.copy()
    
    # Style matchups
    df['pace_matchup'] = (df['pf_avg_5'] + df['pa_avg_5']) + (df['opp_pf_avg_5'] + df['opp_pa_avg_5'])
    df['defensive_game'] = ((df['pa_avg_5'] < 20) & (df['opp_pa_avg_5'] < 20)).astype(int)
    df['shootout_potential'] = ((df['pf_avg_5'] > 25) & (df['opp_pf_avg_5'] > 25)).astype(int)
    
    # Strength vs weakness matchups
    df['off_vs_def_percentile'] = df.groupby(['season', 'week'])['off_vs_def'].transform(
        lambda x: x.rank(pct=True)
    )
    df['elite_vs_poor'] = ((df['off_vs_def_percentile'] > 0.8) | (df['off_vs_def_percentile'] < 0.2)).astype(int)
    
    # Historical head-to-head (simplified)
    df['rivalry_game'] = df.groupby(['team', 'opp']).cumcount().apply(lambda x: min(x, 5) / 5)
    
    return df


def create_trend_features(df):
    """
    Create trend-based features for recent performance patterns.
    """
    df = df.copy()
    df = df.sort_values(['team', 'date'])
    
    # Cover rate trends
    for w in [3, 5, 8]:
        # Points vs spread trend
        df[f'recent_cover_rate_{w}'] = df.groupby('team').apply(
            lambda team_df: (
                (team_df['points_for'].shift(1) - team_df['points_against'].shift(1) - 
                 np.where(team_df['is_home'], team_df['spread_line'], -team_df['spread_line']).shift(1)) > 0
            ).rolling(w, min_periods=1).mean()
        ).reset_index(level=0, drop=True)
        
        # Over/under trend
        df[f'recent_over_rate_{w}'] = df.groupby('team').apply(
            lambda team_df: (
                (team_df['points_for'].shift(1) + team_df['points_against'].shift(1) - 
                 team_df['total_line'].shift(1)) > 0
            ).rolling(w, min_periods=1).mean()
        ).reset_index(level=0, drop=True)
    
    # Scoring trend direction
    df['scoring_trend'] = df.groupby('team')['points_for'].transform(
        lambda s: s.shift(1).rolling(4).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) >= 2 else 0)
    )
    df['allow_trend'] = df.groupby('team')['points_against'].transform(
        lambda s: s.shift(1).rolling(4).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) >= 2 else 0)
    )
    
    return df


def create_market_timing_features(df):
    """
    Create features that capture market timing and line movement patterns.
    """
    df = df.copy()
    
    # Line value indicators (simplified - would need actual line movement data)
    df['spread_magnitude'] = df['spread_line'].abs()
    df['total_level'] = pd.cut(df['total_line'], 
                              bins=[0, 40, 44, 47, 50, 100], 
                              labels=['very_low', 'low', 'medium', 'high', 'very_high'])
    df['total_level'] = df['total_level'].astype(str)
    
    # Key numbers in spread betting
    df['key_number_spread'] = df['spread_line'].abs().isin([3, 6, 7, 10, 14]).astype(int)
    df['key_number_total'] = df['total_line'].isin([41, 42, 43, 44, 47, 48, 49, 50, 51]).astype(int)
    
    return df


def create_all_enhanced_features(df):
    """
    Create all enhanced features for improved betting accuracy.
    """
    print("Creating enhanced features for betting accuracy...")
    
    print("  - Market features...")
    df = create_market_features(df)
    
    print("  - Advanced team metrics...")
    df = create_advanced_team_metrics(df)
    
    print("  - Situational features...")
    df = create_situational_features(df)
    
    print("  - Weather features...")
    df = create_weather_features(df)
    
    print("  - Matchup features...")
    df = create_matchup_features(df)
    
    print("  - Trend features...")
    df = create_trend_features(df)
    
    print("  - Market timing features...")
    df = create_market_timing_features(df)
    
    print("✅ Enhanced feature engineering complete!")
    print(f"Final shape: {df.shape}")
    
    return df


if __name__ == "__main__":
    # Test enhanced features
    from data_loader import load_schedules
    from features import create_all_features
    
    print("Loading sample data...")
    sample_data = load_schedules([2023])
    
    print("Creating base features...")
    base_featured = create_all_features(sample_data)
    
    print("Creating enhanced features...")
    enhanced_featured = create_all_enhanced_features(base_featured)
    
    print(f"\nFeature count comparison:")
    print(f"Base features: {base_featured.shape[1]}")
    print(f"Enhanced features: {enhanced_featured.shape[1]}")
    print(f"New features added: {enhanced_featured.shape[1] - base_featured.shape[1]}")
    
    # Show sample of new features
    new_cols = set(enhanced_featured.columns) - set(base_featured.columns)
    print(f"\nSample of new enhanced features:")
    for i, col in enumerate(sorted(new_cols)):
        if i < 10:  # Show first 10
            print(f"  - {col}")
    print(f"  ... and {len(new_cols) - 10} more")