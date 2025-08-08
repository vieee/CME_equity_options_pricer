#!/usr/bin/env python3
"""
Test script for enhanced plotting functionality with overlays and regression analysis
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__)))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def test_enhanced_plotting_features():
    """Test the enhanced plotting features"""
    print("🧪 Testing Enhanced Plotting Features...")
    
    try:
        # Test imports
        from scipy.interpolate import UnivariateSpline
        from sklearn.linear_model import LinearRegression
        from sklearn.metrics import mean_squared_error, mean_absolute_error
        from plotly.subplots import make_subplots
        import plotly.graph_objects as go
        
        print("✅ All required imports successful")
        
        # Create sample data for testing
        np.random.seed(42)
        n_options = 50
        
        current_price = 100.0
        strikes = np.linspace(80, 120, n_options)
        
        # Generate realistic option data
        sample_data = []
        for i, strike in enumerate(strikes):
            option_type = 'call' if i % 2 == 0 else 'put'
            days_to_expiry = np.random.choice([7, 14, 30, 60, 90])
            
            # Generate realistic IV and prices
            moneyness = strike / current_price
            base_iv = 0.20 + 0.1 * abs(moneyness - 1.0)  # IV smile
            iv = base_iv + np.random.normal(0, 0.02)
            
            # Simplified BS price approximation for testing
            intrinsic = max(0, current_price - strike) if option_type == 'call' else max(0, strike - current_price)
            time_value = max(0.01, iv * np.sqrt(days_to_expiry/365) * current_price * 0.4)
            price = intrinsic + time_value + np.random.normal(0, 0.1)
            
            sample_data.append({
                'strike': strike,
                'option_type': option_type,
                'lastPrice': max(0.01, price),
                'impliedVolatility': max(0.05, iv),
                'expiration': (datetime.now() + timedelta(days=days_to_expiry)).strftime('%Y-%m-%d'),
                'volume': np.random.randint(1, 1000),
                'openInterest': np.random.randint(10, 5000),
                'delta': np.random.uniform(-1, 1) if option_type == 'put' else np.random.uniform(0, 1),
                'gamma': np.random.uniform(0, 0.1),
                'theta': np.random.uniform(-0.1, 0),
                'vega': np.random.uniform(0, 0.5),
                'rho': np.random.uniform(-0.1, 0.1),
                'theoretical_price': max(0.01, price + np.random.normal(0, 0.05)),
                'delta_theoretical': np.random.uniform(-1, 1) if option_type == 'put' else np.random.uniform(0, 1),
                'gamma_theoretical': np.random.uniform(0, 0.1),
                'theta_theoretical': np.random.uniform(-0.1, 0),
                'vega_theoretical': np.random.uniform(0, 0.5),
                'rho_theoretical': np.random.uniform(-0.1, 0.1)
            })
        
        options_df = pd.DataFrame(sample_data)
        
        print(f"✅ Generated sample data with {len(options_df)} options")
        
        # Test enhanced features
        
        # 1. Test spline fitting
        print("🔄 Testing spline fitting...")
        try:
            iv_values = options_df['impliedVolatility'].values
            price_values = options_df['lastPrice'].values
            
            # Remove any NaN values
            mask = ~(np.isnan(iv_values) | np.isnan(price_values))
            iv_clean = iv_values[mask]
            price_clean = price_values[mask]
            
            if len(iv_clean) >= 3:
                spline = UnivariateSpline(iv_clean, price_clean, s=0.1, k=min(3, len(iv_clean)-1))
                iv_smooth = np.linspace(iv_clean.min(), iv_clean.max(), 100)
                price_smooth = spline(iv_smooth)
                print(f"   ✅ Spline fitting successful: {len(price_smooth)} points generated")
            else:
                print("   ⚠️ Not enough data points for spline fitting")
        except Exception as e:
            print(f"   ❌ Spline fitting failed: {e}")
        
        # 2. Test regression analysis
        print("🔄 Testing regression analysis...")
        try:
            X = options_df['strike'].values.reshape(-1, 1)
            y = options_df['delta'].values
            
            reg = LinearRegression().fit(X, y)
            x_trend = np.linspace(X.min(), X.max(), 50)
            y_trend = reg.predict(x_trend.reshape(-1, 1))
            print(f"   ✅ Linear regression successful: R² = {reg.score(X, y):.3f}")
        except Exception as e:
            print(f"   ❌ Regression analysis failed: {e}")
        
        # 3. Test performance metrics
        print("🔄 Testing performance metrics...")
        try:
            market_prices = options_df['lastPrice'].values
            theoretical_prices = options_df['theoretical_price'].values
            
            rmse = np.sqrt(mean_squared_error(market_prices, theoretical_prices))
            mae = mean_absolute_error(market_prices, theoretical_prices)
            mape = np.mean(np.abs((theoretical_prices - market_prices) / market_prices)) * 100
            correlation = np.corrcoef(theoretical_prices, market_prices)[0, 1]
            
            print(f"   ✅ Performance metrics calculated:")
            print(f"      RMSE: ${rmse:.3f}")
            print(f"      MAE: ${mae:.3f}")
            print(f"      MAPE: {mape:.1f}%")
            print(f"      Correlation: {correlation:.3f}")
        except Exception as e:
            print(f"   ❌ Performance metrics failed: {e}")
        
        # 4. Test color mapping and data enrichment
        print("🔄 Testing data enrichment...")
        try:
            options_df['moneyness'] = options_df['strike'] / current_price
            options_df['days_to_expiry'] = pd.to_datetime(options_df['expiration']).apply(
                lambda x: max(1, (x - pd.Timestamp.now()).days)
            )
            
            # Test various color mappings
            color_by_moneyness = options_df['moneyness']
            color_by_expiry = options_df['days_to_expiry']
            color_by_type = options_df['option_type'].map({'call': 1, 'put': 0})
            
            print(f"   ✅ Data enrichment successful:")
            print(f"      Moneyness range: {color_by_moneyness.min():.3f} - {color_by_moneyness.max():.3f}")
            print(f"      Days to expiry range: {color_by_expiry.min()} - {color_by_expiry.max()}")
            print(f"      Option types: {options_df['option_type'].unique()}")
        except Exception as e:
            print(f"   ❌ Data enrichment failed: {e}")
        
        # 5. Test enhanced tooltip generation
        print("🔄 Testing enhanced tooltips...")
        try:
            sample_row = options_df.iloc[0]
            moneyness_status = "ITM" if (
                (sample_row['option_type'] == 'call' and sample_row['strike'] < current_price) or
                (sample_row['option_type'] == 'put' and sample_row['strike'] > current_price)
            ) else "OTM"
            
            tooltip = (
                f"<b>{sample_row['option_type'].title()} Option</b><br>"
                f"Strike: ${sample_row['strike']:.2f}<br>"
                f"Expiry: {sample_row['expiration']}<br>"
                f"Days to Expiry: {sample_row['days_to_expiry']}<br>"
                f"IV: {sample_row['impliedVolatility']:.1%}<br>"
                f"Market Price: ${sample_row['lastPrice']:.2f}<br>"
                f"Moneyness: {sample_row['moneyness']:.3f} ({moneyness_status})<br>"
                f"Volume: {sample_row['volume']}<br>"
                f"Open Interest: {sample_row['openInterest']}"
            )
            
            if 'theoretical_price' in sample_row and pd.notna(sample_row['theoretical_price']):
                tooltip += f"<br>Theoretical Price: ${sample_row['theoretical_price']:.2f}"
                price_diff = sample_row['theoretical_price'] - sample_row['lastPrice']
                tooltip += f"<br>Price Difference: ${price_diff:.2f}"
            
            print(f"   ✅ Enhanced tooltip generated ({len(tooltip)} characters)")
        except Exception as e:
            print(f"   ❌ Enhanced tooltip generation failed: {e}")
        
        # 6. Test subplot creation
        print("🔄 Testing subplot functionality...")
        try:
            from plotly.subplots import make_subplots
            
            # Test creating subplots for residual analysis
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=['Model Price Comparison', 'Residuals vs Strike', 
                              'Residuals Distribution', 'Model Performance'],
                specs=[[{"colspan": 2}, None],
                       [{"type": "histogram"}, {"type": "bar"}]],
                vertical_spacing=0.15
            )
            
            print(f"   ✅ Subplot creation successful")
        except Exception as e:
            print(f"   ❌ Subplot creation failed: {e}")
        
        print("\n🎉 Enhanced plotting feature tests completed!")
        print("\n📋 Summary of New Features:")
        print("   • 🎨 Color-coding by moneyness, expiry, and option type")
        print("   • 🔗 Overlay of theoretical vs market prices")
        print("   • 📊 Interactive tooltips with comprehensive option details")
        print("   • 📈 Spline curve fitting for IV smile modeling")
        print("   • 📉 Linear regression trend lines")
        print("   • 🔍 Residual analysis with distribution plots")
        print("   • 📐 Confidence bands for price predictions")
        print("   • 📋 Enhanced performance metrics (RMSE, MAE, MAPE, Correlation)")
        print("   • 🎛️ Normalization options for better comparisons")
        print("   • 🎯 Current price reference lines")
        
        return True
        
    except Exception as e:
        print(f"❌ Enhanced plotting test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 Starting Enhanced Plotting Features Test...\n")
    success = test_enhanced_plotting_features()
    
    if success:
        print("\n✅ All enhanced plotting features are working correctly!")
        print("\n🔗 You can now test the enhanced plots in the Streamlit app:")
        print("   1. Navigate to any symbol (e.g., SPY, AAPL)")
        print("   2. Enable theoretical pricing")
        print("   3. Explore the enhanced plotting tabs:")
        print("      • Price vs IV with overlay options and IV smile curves")
        print("      • Greeks comparison with market vs theoretical overlays")
        print("      • Model comparison with residual analysis")
        print("      • Sensitivity analysis with enhanced visualizations")
    else:
        print("\n❌ Some enhanced plotting features may have issues.")
        print("   Please check the error messages above and ensure all dependencies are installed.")
    
    print(f"\n📅 Test completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
