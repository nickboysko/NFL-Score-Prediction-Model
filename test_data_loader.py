#!/usr/bin/env python3
"""
Test script for the NFL data loader.
Run this to verify your setup is working correctly.
"""

import sys
import os

# Add src to path so we can import our modules
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data_loader import load_schedules


def main():
    """Test the data loader functionality."""
    print("🧪 Testing NFL Data Loader...")
    print("=" * 50)
    
    try:
        # Test loading schedules for a smaller range first
        print("📅 Loading schedules for 2023-2024...")
        schedules = load_schedules([2023, 2024])
        
        print(f"✅ Successfully loaded {len(schedules)} team-game records")
        print(f"📊 Years: {schedules['season'].min()}-{schedules['season'].max()}")
        print(f"🏈 Teams: {schedules['team'].nunique()}")
        print(f"🎯 Games: {len(schedules) // 2}")  # Divide by 2 since each game has 2 records
        
        print("\n📋 Sample data:")
        print(schedules.head(10))
        
        print("\n🔍 Data info:")
        print(f"Columns: {list(schedules.columns)}")
        print(f"Data types:\n{schedules.dtypes}")
        
        print("\n🎉 Data loader test completed successfully!")
        
    except Exception as e:
        print(f"❌ Error testing data loader: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
