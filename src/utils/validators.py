"""
Data validation utilities
"""
import pandas as pd
import numpy as np
from typing import Union, Optional
from datetime import datetime, date

class DataValidator:
    """Validates financial data inputs"""
    
    @staticmethod
    def validate_symbol(symbol: str) -> bool:
        """Validate stock symbol format"""
        if not symbol or not isinstance(symbol, str):
            return False
        return symbol.replace('.', '').replace('-', '').isalnum() and len(symbol) <= 10
    
    @staticmethod
    def validate_price(price: Union[float, int]) -> bool:
        """Validate price is positive number"""
        try:
            return float(price) > 0
        except (ValueError, TypeError):
            return False
    
    @staticmethod
    def validate_rate(rate: Union[float, int]) -> bool:
        """Validate interest rate (typically between -10% and 50%)"""
        try:
            rate_val = float(rate)
            return -0.1 <= rate_val <= 0.5
        except (ValueError, TypeError):
            return False
    
    @staticmethod
    def validate_volatility(vol: Union[float, int]) -> bool:
        """Validate volatility (typically between 1% and 500%)"""
        try:
            vol_val = float(vol)
            return 0.01 <= vol_val <= 5.0
        except (ValueError, TypeError):
            return False
    
    @staticmethod
    def validate_time_to_expiry(expiry_date: Union[str, date, datetime]) -> Optional[float]:
        """Validate and calculate time to expiry in years"""
        try:
            if isinstance(expiry_date, str):
                expiry_dt = datetime.strptime(expiry_date, '%Y-%m-%d').date()
            elif isinstance(expiry_date, datetime):
                expiry_dt = expiry_date.date()
            elif isinstance(expiry_date, date):
                expiry_dt = expiry_date
            else:
                return None
            
            today = datetime.now().date()
            days_to_expiry = (expiry_dt - today).days
            
            if days_to_expiry < 0:
                return 0  # Expired option
            
            return days_to_expiry / 365.25  # Convert to years
        except (ValueError, TypeError):
            return None
    
    @classmethod
    def validate_options_input(cls, symbol: str, strike: float, current_price: float, 
                              expiry: Union[str, date], risk_free_rate: float, 
                              volatility: float) -> dict:
        """Validate all options pricing inputs"""
        errors = []
        
        if not cls.validate_symbol(symbol):
            errors.append("Invalid symbol format")
        
        if not cls.validate_price(strike):
            errors.append("Strike price must be positive")
        
        if not cls.validate_price(current_price):
            errors.append("Current price must be positive")
        
        if not cls.validate_rate(risk_free_rate):
            errors.append("Risk-free rate must be between -10% and 50%")
        
        if not cls.validate_volatility(volatility):
            errors.append("Volatility must be between 1% and 500%")
        
        time_to_expiry = cls.validate_time_to_expiry(expiry)
        if time_to_expiry is None:
            errors.append("Invalid expiry date format")
        
        return {
            'valid': len(errors) == 0,
            'errors': errors,
            'time_to_expiry': time_to_expiry
        }
