# backtest_spread_pipeline.py

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

def load_games():
    # Replace with your real loader
    games = pd.read_csv("games_with_features.csv")
    return games

def run_backtest(games, feature_cols, target_col="beat_spread", stake=100, odds=-110):
    results = []
    seasons = sorted(games['season'].unique())

    for test_season in seasons[5:]:  # leave first 5 seasons for warmup
        train = games[games['season'] < test_season].dropna(subset=[target_col])
        test = games[games['season'] == test_season].dropna(subset=[target_col])

        X_train, y_train = train[feature_cols], train[target_col]
        X_test, y_test = test[feature_cols], test[target_col]

        model = RandomForestClassifier(
            n_estimators=400, max_depth=6, random_state=42
        )
        model.fit(X_train, y_train)

        proba = model.predict_proba(X_test)[:, 1]
        preds = (proba >= 0.55).astype(int)  # fixed threshold
        acc = accuracy_score(y_test, preds)

        # ROI calc (only bet when prob >= 0.60 or <= 0.40)
        bet_mask = (proba >= 0.60) | (proba <= 0.40)
        bets = test[bet_mask].copy()
        bet_preds = preds[bet_mask]
        bet_true = y_test[bet_mask]

        wins = (bet_preds == bet_true).sum()
        losses = len(bets) - wins

        profit = wins * stake * (100/abs(odds)) - losses * stake
        roi = profit / (len(bets) * stake) if len(bets) > 0 else 0

        results.append({
            "season": test_season,
            "games": len(test),
            "bets": len(bets),
            "acc": acc,
            "bet_acc": wins / len(bets) if len(bets) > 0 else None,
            "profit": profit,
            "roi": roi,
        })

    return pd.DataFrame(results)

if __name__ == "__main__":
    games = load_games()

    feature_cols = [
        "spread_line_signed",
        "diff_pf_avg_3", "diff_pa_avg_3", "diff_pd_avg_3",
        "diff_pf_avg_5", "diff_pa_avg_5", "diff_pd_avg_5",
        "diff_pf_avg_8", "diff_pa_avg_8", "diff_pd_avg_8",
        "rest_advantage",
    ]

    results = run_backtest(games, feature_cols)
    print("\n===== Year-by-Year Backtest =====")
    print(results)

    avg_acc = results["acc"].mean()
    avg_roi = results["roi"].mean()
    total_profit = results["profit"].sum()

    print("\n===== Summary =====")
    print(f"Average accuracy: {avg_acc:.3f}")
    print(f"Average ROI: {avg_roi:.3%}")
    print(f"Total profit: ${total_profit:.0f}")
