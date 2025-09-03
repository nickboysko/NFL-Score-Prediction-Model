#!/usr/bin/env python3
"""
Advanced, Leakage-Safe NFL ATS Model (Home-ATS framing)

Key changes:
- Correct ATS labeling: drop pushes (actual_spread == spread_line_signed)
- Strict temporal isolation: team-rolling stats use shift(1)
- Opponent merge uses only pregame features per team (post-shift)
- Matchup features = home_minus_away diffs
- Validation-based threshold tuning for accuracy
- ROI metrics vs -110 vig for high-confidence subset

Targets:
- beat_spread_home: 1 if HOME covered, 0 if not (pushes removed)
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Any

from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


# -----------------------------
# Feature engineering (team-level)
# -----------------------------
def _rolling_means(s: pd.Series, w: int) -> pd.Series:
    return s.shift(1).rolling(w, min_periods=1).mean()

def create_team_rolling_features(df: pd.DataFrame, windows: List[int] = [3, 5, 8]) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values(['team', 'date']).reset_index(drop=True)
    df['point_diff_game'] = df['points_for'] - df['points_against']
    for w in windows:
        df[f'pf_avg_{w}'] = df.groupby('team')['points_for'].transform(lambda s: _rolling_means(s, w))
        df[f'pa_avg_{w}'] = df.groupby('team')['points_against'].transform(lambda s: _rolling_means(s, w))
        df[f'pd_avg_{w}'] = df.groupby('team')['point_diff_game'].transform(lambda s: _rolling_means(s, w))
    return df


# -----------------------------
# Opponent + game-level transform
# -----------------------------
def to_game_level_with_diffs(team_df: pd.DataFrame) -> pd.DataFrame:
    df = team_df.copy()
    home = df[df['is_home'] == 1].copy()
    away = df[df['is_home'] == 0].copy()

    games = home.merge(away, on='game_id', suffixes=('_home', '_away'), how='inner')

    # Margin + ATS label
    games['actual_spread'] = games['points_for_home'] - games['points_for_away']
    games['spread_line_signed'] = games.get('spread_line_home', 0.0)
    games['ats_margin'] = games['actual_spread'] + games['spread_line_signed']

    games = games.loc[games['ats_margin'] != 0].copy()  # drop pushes
    games['beat_spread_home'] = (games['ats_margin'] > 0).astype(int)

    # Compact matchup features
    feat_windows = [3, 5, 8]
    feat_list = [f'pf_avg_{w}' for w in feat_windows] + \
                [f'pa_avg_{w}' for w in feat_windows] + \
                [f'pd_avg_{w}' for w in feat_windows]

    for f in feat_list:
        hf, af = f + '_home', f + '_away'
        if hf in games and af in games:
            games[f'diff_{f}'] = games[hf] - games[af]

    # Rest advantage
    if 'rest_days_home' in games and 'rest_days_away' in games:
        games['rest_advantage'] = games['rest_days_home'] - games['rest_days_away']

    candidate_features = [c for c in games.columns if c.startswith('diff_')]
    if 'rest_advantage' in games: candidate_features.append('rest_advantage')
    candidate_features.append('spread_line_signed')
    games['_candidate_features'] = [candidate_features] * len(games)
    return games


# -----------------------------
# Feature selection
# -----------------------------
def select_features(games: pd.DataFrame, min_nonnull: float = 0.9) -> List[str]:
    numeric_cols = games.select_dtypes(include=[np.number]).columns.tolist()
    ban = {'points_for_home', 'points_for_away', 'actual_spread', 'ats_margin', 'beat_spread_home'}
    feats = []
    for c in numeric_cols:
        if c in ban: continue
        if c.startswith('diff_') or c in {'rest_advantage', 'spread_line_signed'}:
            if games[c].notna().mean() >= min_nonnull and games[c].std(skipna=True) > 1e-6:
                feats.append(c)
    return feats[:25]


# -----------------------------
# Train + tune threshold
# -----------------------------
def train_xgb_with_threshold(
    train_games: pd.DataFrame,
    test_games: pd.DataFrame,
    features: List[str],
    random_state: int = 42
) -> Tuple[XGBClassifier, float, np.ndarray, np.ndarray, float]:
    X_train, y_train = train_games[features].fillna(0.0).values, train_games['beat_spread_home'].values
    X_test, y_test = test_games[features].fillna(0.0).values, test_games['beat_spread_home'].values

    params = dict(
        objective='binary:logistic', eval_metric='logloss',
        n_estimators=1000, max_depth=4, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.8,
        reg_lambda=2.0, reg_alpha=0.2, min_child_weight=5,
        random_state=random_state, n_jobs=-1
    )

    X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=random_state, stratify=y_train)
    model = XGBClassifier(**params)
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], early_stopping_rounds=75, verbose=False)

    val_proba = model.predict_proba(X_val)[:, 1]
    best_thr, best_acc = 0.5, 0.0
    for thr in np.linspace(0.35, 0.65, 61):
        acc = accuracy_score(y_val, (val_proba >= thr).astype(int))
        if acc > best_acc: best_acc, best_thr = acc, thr

    test_proba = model.predict_proba(X_test)[:, 1]
    test_pred = (test_proba >= best_thr).astype(int)
    test_acc = accuracy_score(y_test, test_pred)

    return model, best_thr, test_proba, test_pred, test_acc


# -----------------------------
# ROI helper
# -----------------------------
def roi_report(probs: np.ndarray, y_true: np.ndarray, threshold: float,
               wager: float = 100.0, odds: int = -110) -> Dict[str, Any]:
    mask = (probs >= threshold) | (probs <= (1 - threshold))
    n_bets = int(mask.sum())
    if n_bets == 0: return dict(n_bets=0, hit_rate=np.nan, units=0.0, roi=np.nan)

    preds, y = (probs[mask] >= 0.5).astype(int), y_true[mask]
    hits = int((preds == y).sum()); hit_rate = hits / n_bets

    win_unit = wager * (100 / abs(odds)) if odds < 0 else wager * (odds / 100)
    units = hits * win_unit - (n_bets - hits) * wager
    roi = units / (n_bets * wager)
    return dict(n_bets=n_bets, hit_rate=hit_rate, units=units, roi=roi)


# -----------------------------
# Public pipeline API
# -----------------------------
def prepare_games_for_seasons(schedules: pd.DataFrame, test_seasons: List[int]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    schedules = schedules.sort_values(['date', 'game_id']).reset_index(drop=True)
    train_df, test_df = schedules[~schedules['season'].isin(test_seasons)].copy(), schedules[schedules['season'].isin(test_seasons)].copy()
    train_games, test_games = to_game_level_with_diffs(create_team_rolling_features(train_df)), to_game_level_with_diffs(create_team_rolling_features(test_df))
    return train_games, test_games


def run_training(schedules: pd.DataFrame,
                 data_years: List[int] = list(range(2009, 2024)),
                 test_seasons: List[int] = [2024],
                 random_state: int = 42) -> Dict[str, Any]:
    train_games, test_games = prepare_games_for_seasons(schedules[schedules['season'].isin(data_years + test_seasons)], test_seasons)
    if train_games['beat_spread_home'].nunique() < 2 or test_games['beat_spread_home'].nunique() < 2:
        raise ValueError("Not enough class diversity after dropping pushes. Check spread_line inputs.")

    features = select_features(train_games) or [c for c in train_games if c.startswith('diff_')][:12]
    model, best_thr, test_proba, test_pred, test_acc = train_xgb_with_threshold(train_games, test_games, features, random_state)
    roi_stats_60, roi_stats_65 = roi_report(test_proba, test_games['beat_spread_home'].values, max(0.60, best_thr)), roi_report(test_proba, test_games['beat_spread_home'].values, max(0.65, best_thr))

    return dict(model=model, features=features, best_threshold=best_thr,
                test_accuracy=float(test_acc), test_size=int(len(test_games)),
                roi_60=roi_stats_60, roi_65=roi_stats_65)
