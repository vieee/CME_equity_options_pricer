"""
Data formatting utilities for display and output
"""
import pandas as pd
import numpy as np
from typing import Union, Any

class DataFormatter:
    """Handles formatting of financial data for display"""
    
    @staticmethod
    def format_currency(value: Union[float, int], decimals: int = 2) -> str:
        """Format value as currency"""
        if pd.isna(value):
            return "N/A"
        return f"${value:,.{decimals}f}"
    
    @staticmethod
    def format_percentage(value: Union[float, int], decimals: int = 2) -> str:
        """Format value as percentage"""
        if pd.isna(value):
            return "N/A"
        return f"{value:.{decimals}f}%"
    
    @staticmethod
    def format_number(value: Union[float, int], decimals: int = 4) -> str:
        """Format number with specified decimal places"""
        if pd.isna(value):
            return "N/A"
        if abs(value) >= 1e6:
            return f"{value:.2e}"
        return f"{value:.{decimals}f}"
    
    @staticmethod
    def format_large_number(value: Union[float, int]) -> str:
        """Format large numbers with K, M, B suffixes"""
        if pd.isna(value):
            return "N/A"
        
        if abs(value) >= 1e9:
            return f"{value/1e9:.1f}B"
        elif abs(value) >= 1e6:
            return f"{value/1e6:.1f}M"
        elif abs(value) >= 1e3:
            return f"{value/1e3:.1f}K"
        else:
            return f"{value:,.0f}"
    
    @classmethod
    def format_options_dataframe(cls, df: pd.DataFrame) -> pd.DataFrame:
        """Format entire options dataframe for display"""
        if df.empty:
            return df
        
        display_df = df.copy()
        
        # Currency columns
        currency_cols = ['strike', 'lastPrice', 'bid', 'ask', 'theoretical_price', 'price_diff']
        for col in currency_cols:
            if col in display_df.columns:
                display_df[col] = display_df[col].apply(
                    lambda x: cls.format_currency(x, 2) if pd.notna(x) else "N/A"
                )
        
        # Percentage columns
        percentage_cols = ['impliedVolatility', 'price_diff_pct']
        for col in percentage_cols:
            if col in display_df.columns:
                display_df[col] = display_df[col].apply(
                    lambda x: cls.format_percentage(x, 2) if pd.notna(x) else "N/A"
                )
        
        # Greeks columns (4 decimal places)
        greeks_cols = ['delta', 'gamma', 'theta', 'vega', 'rho', 'calculated_iv']
        for col in greeks_cols:
            if col in display_df.columns:
                display_df[col] = display_df[col].apply(
                    lambda x: cls.format_number(x, 4) if pd.notna(x) else "N/A"
                )
        
        # Integer columns with commas
        integer_cols = ['volume', 'openInterest']
        for col in integer_cols:
            if col in display_df.columns:
                display_df[col] = display_df[col].apply(
                    lambda x: f"{int(x):,}" if pd.notna(x) else "N/A"
                )
        
        return display_df

class CLIFormatter(DataFormatter):
    """Extended formatter for command-line interface"""
    
    @classmethod
    def format_cli_table(cls, df: pd.DataFrame) -> str:
        """Format dataframe for CLI display"""
        if df.empty:
            return "No data available"
        
        formatted_df = cls.format_options_dataframe(df)
        return formatted_df.to_string(index=False, max_rows=50)
