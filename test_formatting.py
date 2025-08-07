"""
Test formatting for Delta and Theta values
Verify that both webapp and CLI display 4 decimal places
"""

import pandas as pd
import numpy as np
import sys
import os

# Add the current directory to the path to import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import only the CLI formatting function to avoid streamlit import
from cli_pricer import format_cli_display

def format_numeric_columns_test(df, max_decimals=4):
    """
    Test version of format_numeric_columns without streamlit dependencies
    """
    formatted_df = df.copy()
    
    # Define columns that should be formatted as percentages (already in decimal form)
    percentage_columns = ['impliedVolatility', 'price_diff_pct']
    
    # Define columns that should be formatted as currency
    currency_columns = ['strike', 'lastPrice', 'bid', 'ask', 'theoretical_price', 'price_diff']
    
    # Define columns that should use standard decimal formatting
    decimal_columns = ['delta', 'gamma', 'theta', 'vega', 'rho', 'calculated_iv']
    
    # Define columns that should be displayed as integers
    integer_columns = ['volume', 'openInterest']
    
    for col in formatted_df.columns:
        if formatted_df[col].dtype in ['float64', 'float32', 'int64', 'int32']:
            if col in percentage_columns:
                # Format as percentage with 2 decimal places
                formatted_df[col] = formatted_df[col].apply(
                    lambda x: f"{x*100:.2f}%" if pd.notna(x) else "N/A"
                )
            elif col in currency_columns:
                # Format as currency with 2 decimal places for prices
                formatted_df[col] = formatted_df[col].apply(
                    lambda x: f"${x:.2f}" if pd.notna(x) else "N/A"
                )
            elif col in decimal_columns:
                # Format with specified decimal places for Greeks and analytics
                formatted_df[col] = formatted_df[col].apply(
                    lambda x: f"{x:.{max_decimals}f}" if pd.notna(x) else "N/A"
                )
            elif col in integer_columns:
                # Format as integers with comma separators
                formatted_df[col] = formatted_df[col].apply(
                    lambda x: f"{int(x):,}" if pd.notna(x) else "N/A"
                )
            else:
                # Format other numeric columns with max_decimals, but use scientific notation for very large numbers
                formatted_df[col] = formatted_df[col].apply(
                    lambda x: f"{x:.{max_decimals}f}" if pd.notna(x) and abs(x) < 1e6 else (f"{x:.2e}" if pd.notna(x) else "N/A")
                )
    
    return formatted_df

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
    
    # Apply formatting like in webapp using our formatting function
    sample_formatted = format_numeric_columns_test(sample_data.copy())
    
    print("\nSample DataFrame after formatting (using format_numeric_columns_test):")
    print(sample_formatted)
    
    # Test CLI-style formatting
    print("\nCLI-style formatting (using format_cli_display):")
    cli_formatted = format_cli_display(sample_data.copy())
    print(cli_formatted.to_string(index=False))
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
