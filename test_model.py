#!/usr/bin/env python3
"""
Test script for the enhanced NFL Modeling Pipeline

This script tests the complete pipeline including:
- Data loading and feature engineering
- Enhanced walk-forward backtesting with purging
- Hyperparameter optimization
- Final model training with feature selection
- Comprehensive evaluation including betting metrics
"""

import sys
import os

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def main():
    """Main test function for the enhanced NFL modeling pipeline."""
    print("🧪 Testing Enhanced NFL Modeling Pipeline...")
    print("="*70)
    
    try:
        print("📦 Importing modules...")
        
        # Import data loading and feature engineering
        from data_loader import load_schedules
        from features import create_all_features
        print("✅ Data and features modules imported")
        
        # Import the enhanced model functions
        from model import (
            prepare_data, 
            improved_walk_forward_backtest, 
            train_final_model, 
            comprehensive_evaluation
        )
        print("✅ Enhanced model functions imported")
        
        print("\n📅 Step 1: Loading NFL schedules...")
        schedules = load_schedules()
        print(f"✅ Loaded {len(schedules)} team-game records")
        
        print("\n🔧 Step 2: Creating features...")
        featured_data = create_all_features(schedules)
        print("✅ Feature engineering complete")
        
        print("\n📊 Step 3: Preparing data for modeling...")
        train, test, FEATURES, TARGET = prepare_data(featured_data, test_season=2024)
        
        print("\n🔄 Step 4: Enhanced walk-forward backtesting...")
        cv_mae, fold_results, best_params = improved_walk_forward_backtest(
            train, FEATURES, TARGET, n_splits=3, use_optimization=True
        )
        print(f"✅ Enhanced walk-forward backtesting complete")
        
        print("\n🚀 Step 5: Training final optimized model...")
        model_type = 'optimized' if best_params else 'xgboost'
        pipeline, pred_te, metrics = train_final_model(
            train, test, FEATURES, TARGET, 
            model_type=model_type, 
            best_params=best_params,
            use_feature_selection=True
        )
        
        print("\n🎯 Step 6: Comprehensive evaluation...")
        score_eval, game_metrics = comprehensive_evaluation(test, pred_te)
        
        print("\n" + "="*70)
        print("🎉 ENHANCED PIPELINE TEST SUCCESSFUL!")
        print("="*70)
        
        print(f"\n📊 Final Results Summary:")
        print(f"   Training records: {len(train)}")
        print(f"   Test records: {len(test)}")
        print(f"   Features used: {len(FEATURES)}")
        print(f"   Model type: {model_type}")
        
        print(f"\n🔄 Cross-Validation:")
        print(f"   Mean CV MAE: {cv_mae:.3f}")
        
        print(f"\n🎯 Final Model Performance:")
        print(f"   Team-level MAE: {metrics['mae']:.2f}")
        print(f"   Team-level RMSE: {metrics['rmse']:.2f}")
        
        print(f"\n🏈 Game-level Performance:")
        for metric, value in game_metrics.items():
            if isinstance(value, float):
                if 'accuracy' in metric:
                    print(f"   {metric.replace('_', ' ').title()}: {value:.1%}")
                else:
                    print(f"   {metric.replace('_', ' ').title()}: {value:.2f}")
        
        print(f"\n📋 Sample Predictions:")
        if len(score_eval) > 0:
            display_cols = ['home_pred', 'away_pred', 'pred_total', 'pred_spread', 
                           'home_actual', 'away_actual', 'ae_total']
            available_cols = [col for col in display_cols if col in score_eval.columns]
            print(score_eval[available_cols].head())
        
        print(f"\n✅ Enhanced NFL prediction model is ready!")
        print(f"💡 Key improvements implemented:")
        print(f"   - Hyperparameter optimization with Optuna")
        print(f"   - Enhanced walk-forward validation with purging") 
        print(f"   - Feature selection for noise reduction")
        print(f"   - Comprehensive betting-focused evaluation")
        print(f"   - Support for ensemble models")
        
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        print("💡 Make sure you're running from the correct directory and all dependencies are installed")
        print("💡 Check that all required modules exist in the src/ directory")
        
        # Debug: list available functions in features module
        try:
            import features
            print(f"\n🔍 Available functions in features module:")
            for attr in dir(features):
                if not attr.startswith('_'):
                    print(f"   - {attr}")
        except:
            print("   Could not import features module at all")
            
        return False
        
    except Exception as e:
        print(f"❌ Error in modeling pipeline: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎉 All tests passed successfully!")
    else:
        print("\n❌ Some tests failed. Check the error messages above.")
        sys.exit(1)