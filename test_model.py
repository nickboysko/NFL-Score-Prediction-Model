#!/usr/bin/env python3
"""
Enhanced NFL Betting Prediction Model - FIXED VERSION
🎯 Target: 55%+ spread accuracy with advanced techniques
"""

import sys
import os
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def main():
    """Main function for enhanced NFL modeling pipeline."""
    print("🚀 NFL Betting Prediction Model - ENHANCED VERSION")
    print("🎯 Target: 55%+ spread accuracy with advanced techniques")
    print("="*80)
    
    try:
        print("📦 Importing enhanced modules...")
        # Test imports first
        try:
            from features import create_all_features
            print("ℹ️ Using standard features module")
        except ImportError:
            from features_enhanced import create_all_features
            print("ℹ️ Using enhanced features module")
        
        try:
            from model_enhanced import (
                prepare_data, 
                enhanced_walk_forward_validation, 
                train_final_enhanced_model, 
                comprehensive_evaluation
            )
            print("✅ Using enhanced model module")
        except ImportError:
            from model import (
                prepare_data, 
                enhanced_walk_forward_validation, 
                train_final_enhanced_model, 
                comprehensive_evaluation
            )
            print("✅ Using standard model module")
        
        from data_loader import load_schedules
        
        print("\n📅 Step 1: Loading NFL data...")
        schedules = load_schedules()
        print(f"✅ Loaded {len(schedules)} team-game records")
        print(f"📊 Seasons: {schedules['season'].min()}-{schedules['season'].max()}")
        
        print("\n🔧 Step 2: Enhanced feature engineering...")
        print("🔧 Creating advanced team features...")
        print("🔧 Creating situational features...")
        print("🔧 Creating matchup features...")
        print("🔧 Creating market features...")
        print("🔧 Creating interaction features...")
        featured_data = create_all_features(schedules)
        print("✅ Enhanced feature engineering complete!")
        print(f"📊 Final shape: {featured_data.shape}")
        
        # Count new features
        original_cols = set(['game_id', 'season', 'week', 'gameday', 'gametime', 'game_type',
                           'team', 'opp', 'points_for', 'points_against', 'is_home', 'date', 
                           'season_week', 'location', 'overtime', 'spread_line', 'total_line',
                           'home_moneyline', 'away_moneyline', 'home_rest', 'away_rest',
                           'surface', 'roof'])
        new_features = len(featured_data.columns) - len(original_cols)
        print(f"🎯 Total new features created: {new_features}")
        print("✅ Feature engineering complete")
        
        print("\n📊 Step 3: Advanced data preparation...")
        train, test, FEATURES, TARGET = prepare_data(featured_data, test_season=2024)
        
        print(f"\n🎯 Features being used: {len(FEATURES)}")
        print("   Key feature categories:")
        
        # Categorize features for display
        market_features = [f for f in FEATURES if any(x in f for x in ['spread', 'total', 'implied'])]
        form_features = [f for f in FEATURES if any(x in f for x in ['pf_avg', 'pa_avg', 'win_streak', 'recent_form'])]
        matchup_features = [f for f in FEATURES if any(x in f for x in ['opp_', 'off_vs', 'def_vs'])]
        situational_features = [f for f in FEATURES if any(x in f for x in ['rest', 'neutral', 'dome', 'turf', 'season_progress'])]
        performance_features = [f for f in FEATURES if f not in market_features + form_features + matchup_features + situational_features]
        
        print(f"     - Market: {len(market_features)} features")
        print(f"     - Form: {len(form_features)} features") 
        print(f"     - Matchup: {len(matchup_features)} features")
        print(f"     - Situational: {len(situational_features)} features")
        print(f"     - Performance: {len(performance_features)} features")
        
        print("✅ Enhanced data preparation complete")
        
        print("\n🔄 Step 4: Enhanced walk-forward validation...")
        cv_mae, best_model_type, best_hyperparams = enhanced_walk_forward_validation(
            train, FEATURES, TARGET, n_folds=4
        )
        print(f"🏆 Best model: {best_model_type}")
        print(f"✅ Best model type: {best_model_type}")
        
        print("\n🚀 Step 5: Training final enhanced model...")
        pipeline, pred_te, metrics = train_final_enhanced_model(
            train, test, FEATURES, TARGET, 
            model_type=best_model_type, 
            hyperparams=best_hyperparams,  # FIXED: Changed from best_params
            use_feature_selection=True
        )
        
        print("\n🎯 Step 6: Comprehensive evaluation...")
        score_eval, game_metrics = comprehensive_evaluation(test, pred_te)
        
        print("\n" + "="*80)
        print("🎉 ENHANCED PIPELINE SUCCESS!")
        print("="*80)
        
        print(f"\n📊 Final Results Summary:")
        print(f"   Training records: {len(train)}")
        print(f"   Test records: {len(test)}")
        print(f"   Features used: {len(FEATURES)}")
        print(f"   Model type: {best_model_type}")
        
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
        
        return True
        
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        print("\n💡 Troubleshooting suggestions:")
        print("   1. Ensure all dependencies are installed: pip install lightgbm scipy")
        print("   2. Check that enhanced feature/model modules are properly saved")
        print("   3. Verify data loading is working correctly")
        print("   4. Try running with fewer features if memory is an issue")
        return False
        
    except Exception as e:
        print(f"❌ Error in enhanced training pipeline: {e}")
        import traceback
        traceback.print_exc()
        
        print("\n💡 Troubleshooting suggestions:")
        print("   1. Ensure all dependencies are installed: pip install lightgbm scipy")
        print("   2. Check that enhanced feature/model modules are properly saved")
        print("   3. Verify data loading is working correctly")
        print("   4. Try running with fewer features if memory is an issue")
        return False


if __name__ == "__main__":
    success = main()
    if success:
        print("\n✅ Training completed successfully")
    else:
        print("\n❌ Training failed - check error messages above")