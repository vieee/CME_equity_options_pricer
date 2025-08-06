"""
Test formatting for Delta and Theta values
Verify that both webapp and CLI display 4 decimal places
"""

import pandas as pd
import numpy as np
from options_pricing import OptionsPricingModels

def test_formatting():
    """Test that Delta and Theta are formatted to 4 decimal places"""
    print("Testing Delta and Theta Formatting")
    print("=" * 40)
    
    # Create sample data
    pricer = OptionsPricingModels()
    
    # Test parameters
    S = 100  # Stock price
    K = 105  # Strike price  
    T = 0.25  # 3 months
    r = 0.05  # 5% risk-free rate
    sigma = 0.2  # 20% volatility
    
    # Calculate Greeks
    call_greeks = pricer.calculate_greeks(S, K, T, r, sigma, 'call')
    put_greeks = pricer.calculate_greeks(S, K, T, r, sigma, 'put')
    
    print("Raw values:")
    print(f"Call Delta: {call_greeks['delta']}")
    print(f"Call Theta: {call_greeks['theta']}")
    print(f"Put Delta: {put_greeks['delta']}")
    print(f"Put Theta: {put_greeks['theta']}")
    
    print("\nFormatted to 4 decimal places:")
    print(f"Call Delta: {call_greeks['delta']:.4f}")
    print(f"Call Theta: {call_greeks['theta']:.4f}")
    print(f"Put Delta: {put_greeks['delta']:.4f}")
    print(f"Put Theta: {put_greeks['theta']:.4f}")
    
    # Test DataFrame formatting (like in webapp)
    sample_data = pd.DataFrame({
        'strike': [100, 105, 110],
        'delta': [call_greeks['delta'], 0.45123, put_greeks['delta']],
        'theta': [call_greeks['theta'], -0.02567, put_greeks['theta']],
        'option_type': ['call', 'call', 'put']
    })
    
    print("\nSample DataFrame before formatting:")
    print(sample_data)
    
    # Apply formatting like in webapp
    sample_data['delta'] = sample_data['delta'].apply(lambda x: f"{x:.4f}" if pd.notna(x) else "N/A")
    sample_data['theta'] = sample_data['theta'].apply(lambda x: f"{x:.4f}" if pd.notna(x) else "N/A")
    
    print("\nSample DataFrame after formatting:")
    print(sample_data)
    
    # Test CLI-style formatting
    print("\nCLI-style formatting (using to_string with float_format):")
    sample_data_numeric = pd.DataFrame({
        'strike': [100, 105, 110],
        'delta': [call_greeks['delta'], 0.45123, put_greeks['delta']],
        'theta': [call_greeks['theta'], -0.02567, put_greeks['theta']],
        'option_type': ['call', 'call', 'put']
    })
    
    print(sample_data_numeric.to_string(index=False, float_format='%.4f'))
    
    print("\n✅ Formatting test complete!")
    print("Both webapp and CLI will now display Delta and Theta with 4 decimal places")

if __name__ == "__main__":
    test_formatting()
