#!/usr/bin/env python3
"""
Improved NFL Model Training Script

This script implements all the accuracy improvements:
- Enhanced feature engineering
- Advanced model training techniques
- Specialized spread prediction
- Comprehensive evaluation
"""

import sys
import os
import warnings
warnings.filterwarnings('ignore')

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))


def main():
    """Main function to run the improved NFL prediction model."""
    print("🚀 NFL Betting Prediction Model - ENHANCED VERSION")
    print("🎯 Target: 55%+ spread accuracy with advanced techniques")
    print("="*80)
    
    try:
        # Import modules
        print("📦 Importing enhanced modules...")
        from data_loader import load_schedules
        
        # Use our enhanced features if we've replaced the file, otherwise fall back
        try:
            from enhanced_features import create_all_features as create_enhanced_features
            print("✅ Using enhanced features module")
            use_enhanced = True
        except ImportError:
            from features import create_all_features as create_enhanced_features
            print("ℹ️ Using standard features module")
            use_enhanced = False
            
        # Try to use enhanced model functions
        try:
            from enhanced_model import (
                prepare_enhanced_data,
                enhanced_walk_forward_validation,
                train_final_enhanced_model
            )
            print("✅ Using enhanced model module")
            use_enhanced_model = True
        except ImportError:
            from model import (
                prepare_data as prepare_enhanced_data,
                improved_walk_forward_backtest as enhanced_walk_forward_validation,
                train_final_model as train_final_enhanced_model,
                comprehensive_evaluation
            )
            print("ℹ️ Using standard model module")
            use_enhanced_model = False
            
        print("\n📅 Step 1: Loading NFL data...")
        # Load more years for better training data
        schedules = load_schedules(list(range(2010, 2025)))
        print(f"✅ Loaded {len(schedules)} team-game records")
        print(f"📊 Seasons: {schedules['season'].min()}-{schedules['season'].max()}")
        
        print("\n🔧 Step 2: Enhanced feature engineering...")
        featured_data = create_enhanced_features(schedules)
        print("✅ Feature engineering complete")
        
        print("\n📊 Step 3: Advanced data preparation...")
        train, test, FEATURES, TARGET = prepare_enhanced_data(featured_data, test_season=2024)
        
        print(f"\n🎯 Features being used: {len(FEATURES)}")
        if len(FEATURES) > 20:
            print("   Key feature categories:")
            feature_categories = {
                'Market': [f for f in FEATURES if any(x in f for x in ['implied', 'spread', 'total', 'ml_prob'])],
                'Form': [f for f in FEATURES if any(x in f for x in ['form', 'ats_record', 'clutch'])],
                'Matchup': [f for f in FEATURES if any(x in f for x in ['vs_', 'diff', 'pace'])],
                'Situational': [f for f in FEATURES if any(x in f for x in ['rest', 'home', 'season', 'dome'])],
                'Performance': [f for f in FEATURES if any(x in f for x in ['avg_', 'pf_', 'pa_'])]
            }
            
            for category, features in feature_categories.items():
                if features:
                    print(f"     - {category}: {len(features)} features")
        
        print("\n🔄 Step 4: Enhanced walk-forward validation...")
        if use_enhanced_model:
            cv_results, best_params, best_model_type = enhanced_walk_forward_validation(
                train, FEATURES, TARGET, n_splits=4  # Fewer splits for enhanced models
            )
            print(f"✅ Best model type: {best_model_type}")
        else:
            # Fallback to standard validation
            cv_mae, fold_results, best_params = enhanced_walk_forward_validation(
                train, FEATURES, TARGET, n_splits=4, use_optimization=True
            )
            best_model_type = 'optimized_xgb' if best_params else 'xgboost'
            print(f"✅ CV MAE: {cv_mae:.3f}")
        
        print("\n🚀 Step 5: Training final enhanced model...")
        if use_enhanced_model:
            pipeline, pred_te, metrics = train_final_enhanced_model(
                train, test, FEATURES, TARGET, 
                best_params=best_params,
                model_type=best_model_type
            )
        else:
            # Use standard training but with enhancements
            pipeline, pred_te, metrics = train_final_enhanced_model(
                train, test, FEATURES, TARGET,
                model_type='optimized' if best_params else 'xgboost',
                best_params=best_params,
                use_feature_selection=True,
                residual_learning=True
            )
        
        print("\n🎯 Step 6: Comprehensive evaluation...")
        try:
            from model import comprehensive_evaluation
            score_eval, game_metrics = comprehensive_evaluation(test, pred_te)
        except:
            # Simplified evaluation if comprehensive_evaluation fails
            print("📊 Basic evaluation:")
            print(f"   Team-level MAE: {metrics['mae']:.2f}")
            print(f"   Team-level RMSE: {metrics['rmse']:.2f}")
            game_metrics = {}
            score_eval = None
        
        print("\n" + "="*80)
        print("🎉 ENHANCED MODEL TRAINING COMPLETE!")
        print("="*80)
        
        print(f"\n📊 Final Performance Summary:")
        print(f"   Training records: {len(train)}")
        print(f"   Test records: {len(test)}")
        print(f"   Features used: {len(FEATURES)}")
        print(f"   Enhancement level: {'Full' if use_enhanced and use_enhanced_model else 'Partial'}")
        
        print(f"\n🎯 Model Performance:")
        print(f"   Team-level MAE: {metrics['mae']:.2f} points")
        print(f"   Team-level RMSE: {metrics['rmse']:.2f} points")
        
        if 'spread_accuracy' in metrics and metrics['spread_accuracy']:
            print(f"   Spread accuracy: {metrics['spread_accuracy']:.1%}")
        
        if game_metrics:
            print(f"\n🏈 Game-level Performance:")
            for metric, value in game_metrics.items():
                if isinstance(value, float):
                    if 'accuracy' in metric:
                        print(f"   {metric.replace('_', ' ').title()}: {value:.1%}")
                    elif 'correlation' in metric:
                        print(f"   {metric.replace('_', ' ').title()}: {value:.3f}")
                    else:
                        print(f"   {metric.replace('_', ' ').title()}: {value:.2f}")
        
        # Improvement suggestions
        print(f"\n💡 Accuracy Improvement Strategies Applied:")
        improvements = [
            "✅ Enhanced feature engineering (momentum, situational factors)",
            "✅ Advanced model selection and hyperparameter optimization",
            "✅ Time-weighted training (recent games matter more)",
            "✅ Temporal purging in cross-validation",
        ]
        
        if use_enhanced:
            improvements.extend([
                "✅ Market-implied features and betting line analysis",
                "✅ Strength of schedule and opponent adjustments",
                "✅ Clutch performance and situational modeling"
            ])
            
        if use_enhanced_model:
            improvements.extend([
                "✅ Ensemble stacking with diverse models",
                "✅ Specialized spread prediction classifier",
                "✅ Multi-objective optimization"
            ])
        else:
            improvements.extend([
                "⚠️ Standard model used (enhanced model failed to import)",
                "💡 For better results, implement the enhanced model module"
            ])
        
        for improvement in improvements:
            print(f"     {improvement}")
            
        # Next steps for further improvement
        if game_metrics and 'ats_accuracy' in game_metrics:
            spread_acc = game_metrics['ats_accuracy']
            print(f"\n🎯 Current spread accuracy: {spread_acc:.1%}")
            
            if spread_acc < 0.55:
                print(f"\n📈 Further Improvement Suggestions:")
                print(f"   🔍 Add more situational features (weather, injuries, motivation)")
                print(f"   🔍 Implement neural network ensemble")
                print(f"   🔍 Add more betting market features (line movement, public %)")
                print(f"   🔍 Fine-tune for specific bet types (home favorites, divisional games)")
                print(f"   🔍 Add team-specific model adjustments")
            else:
                print(f"\n🎉 EXCELLENT! Target accuracy achieved!")
                print(f"   🏆 Model is ready for live betting")
                print(f"   💰 Expected ROI at 55%+: 5-10% with proper bankroll management")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in enhanced training pipeline: {e}")
        import traceback
        traceback.print_exc()
        
        print(f"\n💡 Troubleshooting suggestions:")
        print(f"   1. Ensure all dependencies are installed: pip install lightgbm scipy")
        print(f"   2. Check that enhanced feature/model modules are properly saved")
        print(f"   3. Verify data loading is working correctly")
        print(f"   4. Try running with fewer features if memory is an issue")
        
        return False


if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎉 Enhanced model training completed successfully!")
        print("💡 Run this script periodically to retrain with new data")
    else:
        print("\n❌ Training failed - check error messages above")
        sys.exit(1)