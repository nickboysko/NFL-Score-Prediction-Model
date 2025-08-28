#!/usr/bin/env python3
"""
Test script for the NFL feature engineering.
Run this to verify your feature creation is working correctly.
"""

import sys
import os

# Add src to path so we can import our modules
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data_loader import load_schedules
from features import create_all_features


def main():
    """Test the feature engineering functionality."""
    print("🧪 Testing NFL Feature Engineering...")
    print("=" * 60)
    
    try:
        # Test loading schedules for a smaller range first
        print("📅 Loading schedules for 2023...")
        schedules = load_schedules([2023])
        
        print(f"✅ Successfully loaded {len(schedules)} team-game records")
        print(f"📊 Original shape: {schedules.shape}")
        print(f"🏈 Teams: {schedules['team'].nunique()}")
        
        # Test feature engineering
        print("\n🔧 Creating features...")
        featured_data = create_all_features(schedules)
        
        print(f"\n✅ Feature engineering complete!")
        print(f"📊 Final shape: {featured_data.shape}")
        print(f"🎯 New features added: {featured_data.shape[1] - schedules.shape[1]}")
        
        # Show the new feature columns
        original_cols = set(schedules.columns)
        new_cols = set(featured_data.columns) - original_cols
        print(f"\n🆕 New feature columns:")
        for col in sorted(new_cols):
            print(f"   - {col}")
        
        # Show sample of engineered features
        print(f"\n📋 Sample of engineered features:")
        feature_cols = ['team', 'opp', 'date', 'points_for', 'points_against']
        feature_cols += [col for col in featured_data.columns if any(x in col for x in ['pf_avg', 'pa_avg', 'opp_', 'rest_days', 'neutral'])]
        
        # Filter to show a specific team for clarity
        sample_team = featured_data['team'].iloc[0]
        team_data = featured_data[featured_data['team'] == sample_team].head(5)
        print(f"\n🏈 Sample data for {sample_team}:")
        print(team_data[feature_cols].to_string(index=False))
        
        # Check for any NaN values in new features
        new_feature_cols = list(new_cols)
        nan_check = featured_data[new_feature_cols].isnull().sum()
        print(f"\n🔍 NaN check for new features:")
        print(nan_check[nan_check > 0] if nan_check.sum() > 0 else "✅ No NaN values in new features")
        
        print("\n🎉 Feature engineering test completed successfully!")
        
    except Exception as e:
        print(f"❌ Error testing feature engineering: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
