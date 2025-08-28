"""
NFL Data Loader Module

This module handles loading and preprocessing NFL schedule data,
including converting to team-game format for modeling.
"""

import nfl_data_py as nfl
import pandas as pd


def load_schedules(years=None):
    """
    Load NFL schedules for specified years and convert to team-game format.
    
    Args:
        years (list, optional): List of years to load. Defaults to 2009-2024.
    
    Returns:
        pd.DataFrame: DataFrame in team-game format with columns:
            - game_id, season, week, gameday, gametime, game_type
            - team, opp (opponent), points_for, points_against
            - is_home, date, season_week
            - location, overtime, spread_line, total_line
            - home_moneyline, away_moneyline, home_rest, away_rest
            - surface, roof
    """
    if years is None:
        years = list(range(2009, 2025))
    
    # Load schedules for specified years
    sched = nfl.import_schedules(years)
    
    # Keep regular season (+ post if you want)
    sched = sched.query("game_type in ['REG', 'POST']").copy()
    
    # Basic columns we'll use
    keep_cols = [
        'game_id', 'season', 'week', 'gameday', 'gametime', 'game_type',
        'home_team', 'away_team', 'home_score', 'away_score', 'location', 'overtime',
        'spread_line', 'total_line', 'home_moneyline', 'away_moneyline',
        'home_rest', 'away_rest', 'surface', 'roof'
    ]
    sched = sched[keep_cols]
    
    # Convert to long format (team-game)
    # Home team records
    home = sched.rename(columns={
        'home_team': 'team',
        'away_team': 'opp',
        'home_score': 'points_for',
        'away_score': 'points_against'
    })
    home['is_home'] = 1
    
    # Away team records
    away = sched.rename(columns={
        'away_team': 'team',
        'home_team': 'opp',
        'away_score': 'points_for',
        'home_score': 'points_against'
    })
    away['is_home'] = 0
    
    # Combine home and away records
    df = pd.concat([home, away], ignore_index=True)
    
    # Add convenience columns
    df['date'] = pd.to_datetime(df['gameday'])
    df['season_week'] = df['season'].astype(str) + '-' + df['week'].astype(str)
    
    return df


if __name__ == "__main__":
    # Example usage
    print("Loading NFL schedules...")
    schedules = load_schedules()
    print(f"Loaded {len(schedules)} team-game records")
    print(f"Years: {schedules['season'].min()}-{schedules['season'].max()}")
    print(f"Teams: {schedules['team'].nunique()}")
    print("\nSample data:")
    print(schedules.head())
