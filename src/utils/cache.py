"""
Caching utilities for expensive operations
"""

import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Callable, Union, Tuple
import functools

# Simple in-memory cache for non-streamlit operations
_cache = {}

def simple_cache(ttl_seconds: int = 300):
    """
    Simple cache decorator for functions that returns cached results
    for a specified time-to-live period
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Create cache key from function name and arguments
            cache_key = f"{func.__name__}_{hash(str(args) + str(sorted(kwargs.items())))}"
            current_time = datetime.now()
            
            # Check if we have a cached result that's still valid
            if cache_key in _cache:
                result, timestamp = _cache[cache_key]
                if (current_time - timestamp).seconds < ttl_seconds:
                    return result
            
            # Execute function and cache result
            result = func(*args, **kwargs)
            _cache[cache_key] = (result, current_time)
            
            # Clean old entries (basic cleanup)
            if len(_cache) > 100:  # Limit cache size
                oldest_key = min(_cache.keys(), 
                               key=lambda k: _cache[k][1])
                del _cache[oldest_key]
            
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
        
        # Get current price with fallback logic
        current_price = info.get('currentPrice', 0)
        previous_close = info.get('previousClose', 0)
        
        # If current price is 0 or None, use previous close
        if current_price == 0 or current_price is None:
            current_price = previous_close
        
        # If still 0, try to get from recent history
        if current_price == 0:
            try:
                hist = ticker.history(period="1d")
                if not hist.empty:
                    current_price = float(hist['Close'].iloc[-1])
                    # Update previous close if we got it from history
                    if previous_close == 0:
                        previous_close = current_price
            except Exception:
                pass
        
        return {
            'symbol': symbol.upper(),
            'shortName': info.get('shortName', symbol),
            'currentPrice': current_price,
            'marketCap': info.get('marketCap', 0),
            'volume': info.get('volume', 0),
            'previousClose': previous_close,
            'dayLow': info.get('dayLow', 0),
            'dayHigh': info.get('dayHigh', 0),
            'fiftyTwoWeekLow': info.get('fiftyTwoWeekLow', 0),
            'fiftyTwoWeekHigh': info.get('fiftyTwoWeekHigh', 0),
            'dividendYield': info.get('dividendYield', 0),
            'beta': info.get('beta', 1.0),
            'trailingPE': info.get('trailingPE', 0),
            'sector': info.get('sector', 'Unknown'),
            'industry': info.get('industry', 'Unknown')
        }
    except Exception:
        return None

@st.cache_data(ttl=300)
def cached_options_data(symbol: str) -> tuple:
    """Cache options data to improve performance"""
    try:
        import yfinance as yf
        ticker = yf.Ticker(symbol)
        
        # Get options dates with better error handling
        try:
            options_dates = ticker.options
        except Exception as e:
            print(f"Error fetching options dates for {symbol}: {e}")
            return None, None
            
        if not options_dates:
            print(f"No options dates available for {symbol}")
            return None, None
        
        # Get current price with better error handling and fallbacks
        current_price = 0
        previous_close = 0
        try:
            info = ticker.info
            current_price = info.get('currentPrice', 0)
            previous_close = info.get('previousClose', 0)
            
            # If current price is 0 or None, use previous close
            if current_price == 0 or current_price is None:
                current_price = previous_close
                
        except Exception as e:
            print(f"Error fetching info for {symbol}: {e}")
        
        if current_price == 0:
            # Final fallback to history
            try:
                hist = ticker.history(period="1d")
                if not hist.empty:
                    current_price = float(hist['Close'].iloc[-1])
                else:
                    print(f"No recent price data available for {symbol}")
                    return None, None
            except Exception as e:
                print(f"Error fetching historical data for {symbol}: {e}")
                return None, None
        
        all_options = []
        success_count = 0
        for i, date in enumerate(options_dates[:3]):  # Limit to first 3 expiration dates for performance
            try:
                option_chain = ticker.option_chain(date)
                calls = option_chain.calls
                puts = option_chain.puts
                
                if not calls.empty and not puts.empty:
                    calls['option_type'] = 'call'
                    calls['expiration'] = date
                    puts['option_type'] = 'put'
                    puts['expiration'] = date
                    
                    all_options.extend([calls, puts])
                    success_count += 1
            except Exception as e:
                print(f"Error fetching options chain for {symbol} date {date}: {e}")
                continue
        
        if all_options and success_count > 0:
            options_df = pd.concat(all_options, ignore_index=True)
            print(f"Successfully fetched {len(options_df)} options for {symbol}")
            return options_df, current_price
        
        print(f"No valid options data found for {symbol}")
        return None, None
    except Exception as e:
        print(f"Unexpected error in cached_options_data for {symbol}: {e}")
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

def clear_all_caches():
    """Clear all caches"""
    global _cache
    _cache.clear()
    st.cache_data.clear()
    print("All caches cleared")

def get_cache_stats():
    """Get cache statistics"""
    return {
        'simple_cache_size': len(_cache),
        'streamlit_cache_enabled': True
    }
