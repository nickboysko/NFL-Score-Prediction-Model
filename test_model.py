#!/usr/bin/env python3
"""
Improved NFL Prediction Model - Main Training Script
Focused on achieving 55%+ betting accuracy

This script integrates all improvements:
- Enhanced feature engineering for betting markets
- Separate models for spread and total predictions  
- Advanced ensemble methods
- Betting-focused validation and evaluation
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def main():
    """Main training pipeline for improved betting accuracy."""
    print("🚀 NFL Betting Prediction Model - Enhanced Training Pipeline")
    print("🎯 Target: 55%+ spread and total accuracy")
    print("="*80)
    
    try:
        # Import modules
        from data_loader import load_schedules
        from features import create_all_features
        
        # Import enhanced modules (assuming they're saved as files)
        print("📦 Loading enhanced modules...")
        
        print("\n📅 Step 1: Loading NFL data...")
        # Load more years for better training
        years = list(range(2010, 2025))  # Extended range
        schedules = load_schedules(years)
        print(f"✅ Loaded {len(schedules)} team-game records from {min(years)}-{max(years)}")
        
        print("\n🔧 Step 2: Creating base features...")
        featured_data = create_all_features(schedules)
        print("✅ Base feature engineering complete")
        
        print("\n⚡ Step 3: Creating enhanced betting features...")
        # This would use the enhanced_features.py functions
        try:
            from enhanced_features import create_all_enhanced_features
            enhanced_data = create_all_enhanced_features(featured_data)
            print("✅ Enhanced feature engineering complete")
        except ImportError:
            print("⚠️ Enhanced features module not found, using base features")
            enhanced_data = featured_data
        
        print(f"📊 Final feature count: {enhanced_data.shape[1]} columns")
        
        print("\n📊 Step 4: Preparing betting-focused data...")
        try:
            from betting_focused_model import prepare_betting_data
            train, test, betting_features = prepare_betting_data(enhanced_data, test_season=2024)
            print(f"✅ Using {len(betting_features)} betting-optimized features")
        except ImportError:
            print("⚠️ Betting model module not found, using standard preparation")
            from model import prepare_data
            train, test, betting_features, _ = prepare_data(enhanced_data, test_season=2024)
        
        print("\n🔄 Step 5: Betting-focused validation...")
        try:
            from betting_focused_model import betting_walk_forward_validation
            cv_results, avg_spread_acc, avg_total_acc = betting_walk_forward_validation(
                train, betting_features, n_splits=5, model_type='ensemble'
            )
            print(f"✅ CV Results - Spread: {avg_spread_acc:.1%}, Total: {avg_total_acc:.1%}")
        except ImportError:
            print("⚠️ Using standard validation")
            from model import improved_walk_forward_backtest
            cv_mae, _, _ = improved_walk_forward_backtest(
                train, betting_features, 'points_for', n_splits=5
            )
            print(f"✅ CV MAE: {cv_mae:.3f}")
        
        print("\n🚀 Step 6: Training optimized betting models...")
        try:
            from betting_focused_model import train_betting_models, analyze_betting_edges
            
            betting_models = train_betting_models(
                train, test, betting_features, optimize_params=True
            )
            
            results = betting_models['results']
            
            print(f"\n🎯 FINAL BETTING RESULTS:")
            print(f"   📈 Spread Accuracy: {results['spread_accuracy']:.1%}")
            print(f"   📊 Total Accuracy: {results['total_accuracy']:.1%}")
            print(f"   💎 High-Confidence Spread: {results['spread_conf_accuracy']:.1%}")
            print(f"   💎 High-Confidence Total: {results['total_conf_accuracy']:.1%}")
            
            # Analyze betting edges
            edge_analysis = analyze_betting_edges(
                test, results['spread_predictions'], results['total_predictions']
            )
            
            # Check if we hit target
            target_met = (results['spread_accuracy'] >= 0.55) or (results['total_accuracy'] >= 0.55)
            
            if target_met:
                print(f"\n🎉 TARGET ACHIEVED! 55%+ accuracy reached")
            else:
                print(f"\n📈 Progress made - continue optimizing for 55% target")
                
                # Suggestions for further improvement
                print(f"\n💡 Improvement suggestions:")
                print(f"   - Add real-time injury data")
                print(f"   - Include weather API data")  
                print(f"   - Implement line movement tracking")
                print(f"   - Add public betting percentage data")
                print(f"   - Fine-tune ensemble weights")
                
        except ImportError:
            print("⚠️ Using fallback standard model training")
            from model import train_final_model, comprehensive_evaluation
            
            pipeline, pred_te, metrics = train_final_model(
                train, test, betting_features, 'points_for',
                model_type='ensemble', use_feature_selection=True
            )
            
            score_eval, game_metrics = comprehensive_evaluation(test, pred_te)
            
            print(f"\n🎯 Standard Model Results:")
            print(f"   Team MAE: {metrics['mae']:.2f}")
            if 'ats_accuracy' in game_metrics:
                print(f"   Spread Accuracy: {game_metrics['ats_accuracy']:.1%}")
            if 'ou_accuracy' in game_metrics:
                print(f"   Total Accuracy: {game_metrics['ou_accuracy']:.1%}")
        
        print("\n📊 Step 7: Feature importance analysis...")
        # Analyze which features are most important
        try:
            feature_importance = analyze_feature_importance(betting_models, betting_features)
            print("✅ Feature importance analysis complete")
        except:
            print("⚠️ Feature importance analysis skipped")
        
        print("\n💾 Step 8: Model persistence...")
        # Save models and results
        save_models_and_results(betting_models if 'betting_models' in locals() else None)
        
        print("\n" + "="*80)
        print("🏁 TRAINING COMPLETE!")
        print("="*80)
        
        print(f"\n📋 Summary:")
        print(f"   Training samples: {len(train):,}")
        print(f"   Test samples: {len(test):,}")
        print(f"   Features used: {len(betting_features)}")
        print(f"   Training time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        if 'results' in locals():
            print(f"\n🏆 Best Results:")
            print(f"   Spread: {results['spread_accuracy']:.1%}")
            print(f"   Total: {results['total_accuracy']:.1%}")
            
            if results['spread_accuracy'] >= 0.55 or results['total_accuracy'] >= 0.55:
                print("🎯 SUCCESS: 55%+ accuracy achieved!")
            else:
                print("📈 Continue optimizing to reach 55% target")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in training pipeline: {e}")
        import traceback
        traceback.print_exc()
        return False


def analyze_feature_importance(models, feature_names):
    """Analyze which features contribute most to betting accuracy."""
    print("🔍 Analyzing feature importance...")
    
    try:
        # Get feature importance from spread model
        spread_model = models['spread_model']
        if hasattr(spread_model, 'models') and 'xgb_optimized' in spread_model.models:
            xgb_model = spread_model.models['xgb_optimized']
            if hasattr(xgb_model, 'feature_importances_'):
                importances = xgb_model.feature_importances_
                
                # Create importance DataFrame
                importance_df = pd.DataFrame({
                    'feature': feature_names,
                    'importance': importances
                }).sort_values('importance', ascending=False)
                
                print("📊 Top 15 Most Important Features:")
                for i, row in importance_df.head(15).iterrows():
                    print(f"   {row['feature']}: {row['importance']:.4f}")
                
                return importance_df
    except Exception as e:
        print(f"⚠️ Could not analyze feature importance: {e}")
    
    return None


def save_models_and_results(models):
    """Save trained models and results for later use."""
    try:
        import joblib
        import os
        
        # Create results directory
        results_dir = 'model_results'
        os.makedirs(results_dir, exist_ok=True)
        
        if models is not None:
            # Save models
            model_path = os.path.join(results_dir, f'betting_models_{datetime.now().strftime("%Y%m%d_%H%M")}.pkl')
            joblib.dump(models, model_path)
            print(f"💾 Models saved to: {model_path}")
        
        print("✅ Results saved successfully")
        
    except Exception as e:
        print(f"⚠️ Could not save models: {e}")


def quick_test():
    """Quick test with subset of data for development."""
    print("🧪 Running quick test with 2023-2024 data...")
    
    from data_loader import load_schedules
    from features import create_all_features
    
    # Load just recent years for testing
    schedules = load_schedules([2023, 2024])
    featured_data = create_all_features(schedules)
    
    print(f"✅ Quick test data loaded: {len(featured_data)} records")
    print(f"   Features: {featured_data.shape[1]}")
    print(f"   Games with betting lines: {featured_data.dropna(subset=['spread_line', 'total_line']).shape[0]}")
    
    return featured_data


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='NFL Betting Prediction Model Training')
    parser.add_argument('--quick', action='store_true', help='Run quick test with subset of data')
    parser.add_argument('--years', nargs='+', type=int, help='Specific years to train on')
    
    args = parser.parse_args()
    
    if args.quick:
        print("🏃 Running in quick test mode...")
        test_data = quick_test()
        print("Quick test completed!")
    else:
        print("🏋️ Running full training pipeline...")
        success = main()
        if success:
            print("\n🎉 Training completed successfully!")
            print("💡 Next steps:")
            print("   1. Review accuracy results")
            print("   2. Test predictions on new games")
            print("   3. Implement live betting integration")
            print("   4. Continue optimizing features")
        else:
            print("\n❌ Training failed - check logs above")
            sys.exit(1)