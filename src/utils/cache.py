"""
Caching utilities for performance optimization
"""
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Any, Callable, Optional
import functools
import hashlib
import pickle

class CacheManager:
    """Manages caching for expensive operations"""
    
    @staticmethod
    def get_cache_key(*args, **kwargs) -> str:
        """Generate cache key from arguments"""
        key_data = str(args) + str(sorted(kwargs.items()))
        return hashlib.md5(key_data.encode()).hexdigest()
    
    @staticmethod
    def streamlit_cache_data(ttl: int = 300):
        """Streamlit cache decorator with TTL"""
        return st.cache_data(ttl=ttl)
    
    @staticmethod
    def memory_cache(ttl_seconds: int = 300):
        """In-memory cache decorator"""
        def decorator(func: Callable) -> Callable:
            cache = {}
            
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                cache_key = CacheManager.get_cache_key(*args, **kwargs)
                current_time = datetime.now()
                
                # Check if cached result exists and is not expired
                if cache_key in cache:
                    result, timestamp = cache[cache_key]
                    if (current_time - timestamp).seconds < ttl_seconds:
                        return result
                
                # Execute function and cache result
                result = func(*args, **kwargs)
                cache[cache_key] = (result, current_time)
                
                # Clean old entries (basic cleanup)
                if len(cache) > 100:  # Limit cache size
                    oldest_key = min(cache.keys(), 
                                   key=lambda k: cache[k][1])
                    del cache[oldest_key]
                
                return result
            
            return wrapper
        return decorator

# Streamlit-specific cache decorators
@st.cache_data(ttl=300)
def cached_stock_info(symbol: str) -> Optional[dict]:
    """Cache stock information to avoid repeated API calls"""
    try:
        import yfinance as yf
        ticker = yf.Ticker(symbol)
        info = ticker.info
        return {
            'shortName': info.get('shortName', symbol),
            'currentPrice': info.get('currentPrice', 0),
            'marketCap': info.get('marketCap', 0),
            'volume': info.get('volume', 0),
            'previousClose': info.get('previousClose', 0)
        }
    except Exception:
        return None

@st.cache_data(ttl=300)
def cached_options_data(symbol: str) -> tuple:
    """Cache options data to improve performance"""
    try:
        import yfinance as yf
        ticker = yf.Ticker(symbol)
        options_dates = ticker.options
        if not options_dates:
            return None, None
        
        current_price = ticker.info.get('currentPrice', 0)
        if current_price == 0:
            # Fallback to history
            hist = ticker.history(period="1d")
            if not hist.empty:
                current_price = hist['Close'].iloc[-1]
        
        all_options = []
        for date in options_dates[:3]:  # Limit to first 3 expiration dates for performance
            try:
                calls = ticker.option_chain(date).calls
                puts = ticker.option_chain(date).puts
                
                calls['option_type'] = 'call'
                calls['expiration'] = date
                puts['option_type'] = 'put'
                puts['expiration'] = date
                
                all_options.extend([calls, puts])
            except Exception:
                continue
        
        if all_options:
            options_df = pd.concat(all_options, ignore_index=True)
            return options_df, current_price
        
        return None, None
    except Exception:
        return None, None

@st.cache_data(ttl=600)  # Cache for 10 minutes
def cached_historical_data(symbol: str, period: str = "1y") -> Optional[pd.DataFrame]:
    """Cache historical price data"""
    try:
        import yfinance as yf
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period=period)
        return hist if not hist.empty else None
    except Exception:
        return None
