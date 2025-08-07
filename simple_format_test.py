"""
Simple test to verify formatting functions work correctly
"""

import pandas as pd
import numpy as np

def format_numeric_columns_simple(df, max_decimals=4):
    """
    Simple version of format_numeric_columns for testing
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
                # Format other numeric columns with max_decimals
                formatted_df[col] = formatted_df[col].apply(
                    lambda x: f"{x:.{max_decimals}f}" if pd.notna(x) and abs(x) < 1e6 else (f"{x:.2e}" if pd.notna(x) else "N/A")
                )
    
    return formatted_df

def test_formatting():
    """Test the formatting function"""
    print("Testing Comprehensive Numeric Formatting")
    print("="*60)
    
    # Create sample data with various types of values that need formatting
    sample_data = pd.DataFrame({
        'strike': [100.123456, 105.789123, 110.456789],
        'lastPrice': [5.256789123, 2.109876543, 0.856789123],
        'bid': [5.10987654, 2.05678912, 0.80123456],
        'ask': [5.40987654, 2.15678912, 0.90123456],
        'delta': [0.123456789, 0.456789123, -0.789123456],
        'gamma': [0.001234567, 0.002345678, 0.003456789],
        'theta': [-0.012345678, -0.023456789, -0.034567891],
        'vega': [0.987654321, 0.876543210, 0.765432109],
        'impliedVolatility': [0.25123456, 0.28987654, 0.32456789],
        'volume': [1000, 2500, 500],
        'openInterest': [5000, 12500, 2500],
        'theoretical_price': [5.123456789, 2.087654321, 0.823456789],
        'option_type': ['call', 'call', 'put']
    })
    
    print("Sample DataFrame BEFORE formatting:")
    print(sample_data.to_string(index=False))
    
    # Apply formatting
    formatted_data = format_numeric_columns_simple(sample_data.copy())
    
    print("\nSample DataFrame AFTER formatting:")
    print(formatted_data.to_string(index=False))
    
    print("\nFormatting Summary:")
    print("- Currency values (strike, lastPrice, bid, ask, theoretical_price): 2 decimal places with $ sign")
    print("- Greeks (delta, gamma, theta, vega): 4 decimal places")
    print("- Percentages (impliedVolatility): 2 decimal places with % sign")
    print("- Integers (volume, openInterest): Comma separators")
    print("- All other values: 4 decimal places maximum")

if __name__ == "__main__":
    test_formatting()
