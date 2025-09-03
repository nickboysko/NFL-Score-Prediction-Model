"""
NFL Data Loader + Feature Engineering

This module loads and preprocesses NFL schedule data into team-game format,
and optionally computes rolling features for modeling.
"""

import nfl_data_py as nfl
import pandas as pd


def load_schedules(years=None):
    if years is None:
        years = list(range(2009, 2025))
    
    sched = nfl.import_schedules(years)
    sched = sched.query("game_type in ['REG', 'POST']").copy()
    
    keep_cols = [
        'game_id', 'season', 'week', 'gameday', 'gametime', 'game_type',
        'home_team', 'away_team', 'home_score', 'away_score', 'location', 'overtime',
        'spread_line', 'total_line', 'home_moneyline', 'away_moneyline',
        'home_rest', 'away_rest', 'surface', 'roof'
    ]
    sched = sched[keep_cols]
    
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
    
    df = pd.concat([home, away], ignore_index=True)
    df['date'] = pd.to_datetime(df['gameday'])
    df['season_week'] = df['season'].astype(str) + '-' + df['week'].astype(str)
    
    return df


def add_features(df):
    """Add rolling averages and matchup features for modeling."""
    df = df.sort_values(['team', 'date']).copy()
    
    # rolling points for / against
    for window in [3, 5, 8]:
        df[f'pf_avg_{window}'] = df.groupby('team')['points_for'].transform(
            lambda x: x.shift(1).rolling(window, min_periods=1).mean()
        )
        df[f'pa_avg_{window}'] = df.groupby('team')['points_against'].transform(
            lambda x: x.shift(1).rolling(window, min_periods=1).mean()
        )
        df[f'pd_avg_{window}'] = df[f'pf_avg_{window}'] - df[f'pa_avg_{window}']
    
    # rest advantage
    df['rest_advantage'] = df['home_rest'].fillna(0) - df['away_rest'].fillna(0)
    df.loc[df['is_home'] == 0, 'rest_advantage'] *= -1
    
    # signed spread from perspective of current team
    df['spread_line_signed'] = df['spread_line'] * df['is_home'].apply(lambda x: 1 if x == 1 else -1)
    
    # opponent-joined diffs
    opp_cols = [c for c in df.columns if c.endswith('_avg_3') or c.endswith('_avg_5') or c.endswith('_avg_8')]
    opp = df[['game_id', 'team'] + opp_cols].rename(columns={c: f'opp_{c}' for c in opp_cols})
    df = df.merge(opp, on=['game_id', 'team'], how='left', suffixes=('', '_dup'))
    
    for col in opp_cols:
        metric = col.replace('_avg_', '')
        df[f'diff_{metric}'] = df[col] - df[f'opp_{col}']
    
    return df

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add rolling averages, spread features, and target variable.
    """
    games = df.copy()

    # Signed spread (from perspective of team)
    games['spread_line_signed'] = games.apply(
        lambda row: row['spread_line'] if row['is_home'] == 1 else -row['spread_line'],
        axis=1
    )

    # Point differential
    games['point_diff'] = games['points_for'] - games['points_against']

    # Rolling averages (PF/PA/PD)
    for window in [3, 5, 8]:
        games[f'pf_avg_{window}'] = (
            games.groupby('team')['points_for']
            .transform(lambda x: x.shift().rolling(window).mean())
        )
        games[f'pa_avg_{window}'] = (
            games.groupby('team')['points_against']
            .transform(lambda x: x.shift().rolling(window).mean())
        )
        games[f'pd_avg_{window}'] = games[f'pf_avg_{window}'] - games[f'pa_avg_{window}']

    # Opponent differences
    for window in [3, 5, 8]:
        games[f'diff_pf_avg_{window}'] = games[f'pf_avg_{window}'] - games.groupby('opp')[f'pf_avg_{window}'].transform('shift')
        games[f'diff_pa_avg_{window}'] = games[f'pa_avg_{window}'] - games.groupby('opp')[f'pa_avg_{window}'].transform('shift')
        games[f'diff_pd_avg_{window}'] = games[f'pd_avg_{window}'] - games.groupby('opp')[f'pd_avg_{window}'].transform('shift')

    # Rest advantage (home_rest - away_rest)
    games['rest_advantage'] = (
        games.groupby('game_id')['home_rest'].transform('first') -
        games.groupby('game_id')['away_rest'].transform('first')
    )
    games.loc[games['is_home'] == 0, 'rest_advantage'] *= -1

    # 🎯 Target variable: did team beat the spread?
    games['beat_spread'] = (
        games['point_diff'] + games['spread_line_signed']
    ) > 0

    return games


def build_and_save_features(csv_path="games_with_features.csv"):
    """Load schedules, add features, save to CSV."""
    df = load_schedules()
    df = add_features(df)
    df.to_csv(csv_path, index=False)
    print(f"Saved {len(df)} rows with features -> {csv_path}")


if __name__ == "__main__":
    print("Loading and building NFL schedules with features...")
    schedules = load_schedules()
    games = build_features(schedules)
    games.to_csv("games_with_features.csv", index=False)
    print(f"Saved {len(games)} rows to games_with_features.csv")
