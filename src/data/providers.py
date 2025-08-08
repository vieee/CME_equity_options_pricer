"""
Stock and Options Data Provider - Refactored and Enhanced
Handles data fetching from various financial APIs
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Optional, List, Tuple, Union
import warnings
warnings.filterwarnings('ignore')

class StockDataProvider:
    """Provides stock market data from various sources"""
    
    def __init__(self):
        """Initialize the stock data provider"""
        self.data_sources = ['yfinance']  # Can be extended with more sources
    
    def get_stock_info(self, symbol: str) -> Optional[Dict]:
        """
        Get basic stock information
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Dictionary with stock information or None if failed
        """
        try:
            import yfinance as yf
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            return {
                'symbol': symbol.upper(),
                'shortName': info.get('shortName', symbol),
                'longName': info.get('longName', symbol),
                'currentPrice': info.get('currentPrice', 0),
                'marketCap': info.get('marketCap', 0),
                'volume': info.get('volume', 0),
                'previousClose': info.get('previousClose', 0),
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
        except Exception as e:
            print(f"Error fetching stock info for {symbol}: {e}")
            return None
    
    def get_historical_data(self, symbol: str, period: str = "1y") -> Optional[pd.DataFrame]:
        """
        Get historical stock price data
        
        Args:
            symbol: Stock symbol
            period: Time period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
            
        Returns:
            DataFrame with historical price data or None if failed
        """
        try:
            import yfinance as yf
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period=period)
            
            if hist.empty:
                return None
            
            # Clean and prepare data
            hist = hist.reset_index()
            hist['Symbol'] = symbol.upper()
            
            # Calculate additional metrics
            hist['Daily_Return'] = hist['Close'].pct_change()
            hist['Volatility_20d'] = hist['Daily_Return'].rolling(window=20).std() * np.sqrt(252)
            
            return hist
        except Exception as e:
            print(f"Error fetching historical data for {symbol}: {e}")
            return None

class OptionsDataProvider:
    """Provides options market data and chain information"""
    
    def __init__(self):
        """Initialize the options data provider"""
        self.max_expiry_dates = 6  # Limit for performance
    
    def get_options_chain(self, symbol: str, max_expiry_dates: Optional[int] = None) -> Tuple[Optional[pd.DataFrame], Optional[float]]:
        """
        Get options chain data for a symbol
        
        Args:
            symbol: Stock symbol
            max_expiry_dates: Maximum number of expiry dates to fetch
            
        Returns:
            Tuple of (options_dataframe, current_stock_price)
        """
        try:
            import yfinance as yf
            ticker = yf.Ticker(symbol)
            
            # Get current stock price
            current_price = self._get_current_price(ticker)
            if current_price is None:
                return None, None
            
            # Get options expiry dates
            options_dates = ticker.options
            if not options_dates:
                return None, current_price
            
            # Limit expiry dates for performance
            max_dates = max_expiry_dates or self.max_expiry_dates
            limited_dates = options_dates[:max_dates]
            
            all_options = []
            for date in limited_dates:
                try:
                    # Get calls and puts for this expiry
                    option_chain = ticker.option_chain(date)
                    calls = option_chain.calls.copy()
                    puts = option_chain.puts.copy()
                    
                    # Add metadata
                    calls['option_type'] = 'call'
                    calls['expiration'] = date
                    puts['option_type'] = 'put'
                    puts['expiration'] = date
                    
                    # Add stock price for reference
                    calls['stock_price'] = current_price
                    puts['stock_price'] = current_price
                    
                    all_options.extend([calls, puts])
                
                except Exception as e:
                    print(f"Error fetching options for {date}: {e}")
                    continue
            
            if not all_options:
                return None, current_price
            
            # Combine all options data
            options_df = pd.concat(all_options, ignore_index=True)
            
            # Clean and enhance data
            options_df = self._clean_options_data(options_df, symbol, current_price)
            
            return options_df, current_price
            
        except Exception as e:
            print(f"Error fetching options chain for {symbol}: {e}")
            return None, None
    
    def _get_current_price(self, ticker) -> Optional[float]:
        """Get current stock price with fallback methods"""
        try:
            # Try info first
            info = ticker.info
            current_price = info.get('currentPrice', 0)
            
            if current_price > 0:
                return current_price
            
            # Fallback to recent history
            hist = ticker.history(period="1d")
            if not hist.empty:
                return float(hist['Close'].iloc[-1])
            
            return None
        except Exception:
            return None
    
    def _clean_options_data(self, df: pd.DataFrame, symbol: str, current_price: float) -> pd.DataFrame:
        """Clean and enhance options data"""
        if df.empty:
            return df
        
        # Add symbol
        df['symbol'] = symbol.upper()
        
        # Calculate time to expiry
        df['days_to_expiry'] = (pd.to_datetime(df['expiration']) - datetime.now()).dt.days
        df['time_to_expiry'] = df['days_to_expiry'] / 365.25
        
        # Calculate moneyness
        df['moneyness'] = df['strike'] / current_price
        
        # Classify ITM/OTM/ATM
        def classify_moneyness(row):
            if row['option_type'] == 'call':
                if row['strike'] < current_price:
                    return 'ITM'
                elif abs(row['strike'] - current_price) / current_price < 0.02:
                    return 'ATM'
                else:
                    return 'OTM'
            else:  # put
                if row['strike'] > current_price:
                    return 'ITM'
                elif abs(row['strike'] - current_price) / current_price < 0.02:
                    return 'ATM'
                else:
                    return 'OTM'
        
        df['itm_otm'] = df.apply(classify_moneyness, axis=1)
        
        # Calculate intrinsic value
        def calc_intrinsic_value(row):
            if row['option_type'] == 'call':
                return max(0, current_price - row['strike'])
            else:
                return max(0, row['strike'] - current_price)
        
        df['intrinsic_value'] = df.apply(calc_intrinsic_value, axis=1)
        
        # Calculate time value (if lastPrice is available)
        if 'lastPrice' in df.columns:
            df['time_value'] = df['lastPrice'] - df['intrinsic_value']
            df['time_value'] = df['time_value'].clip(lower=0)  # Time value can't be negative
        
        # Sort by expiration and strike
        df = df.sort_values(['expiration', 'strike', 'option_type']).reset_index(drop=True)
        
        return df
    
    def get_options_by_expiry(self, symbol: str, expiry_date: str) -> Tuple[Optional[pd.DataFrame], Optional[float]]:
        """
        Get options for a specific expiry date
        
        Args:
            symbol: Stock symbol
            expiry_date: Expiry date in YYYY-MM-DD format
            
        Returns:
            Tuple of (options_dataframe, current_stock_price)
        """
        try:
            import yfinance as yf
            ticker = yf.Ticker(symbol)
            
            current_price = self._get_current_price(ticker)
            if current_price is None:
                return None, None
            
            option_chain = ticker.option_chain(expiry_date)
            calls = option_chain.calls.copy()
            puts = option_chain.puts.copy()
            
            calls['option_type'] = 'call'
            calls['expiration'] = expiry_date
            puts['option_type'] = 'put'
            puts['expiration'] = expiry_date
            
            # Add stock price
            calls['stock_price'] = current_price
            puts['stock_price'] = current_price
            
            options_df = pd.concat([calls, puts], ignore_index=True)
            options_df = self._clean_options_data(options_df, symbol, current_price)
            
            return options_df, current_price
            
        except Exception as e:
            print(f"Error fetching options for {symbol} expiry {expiry_date}: {e}")
            return None, None

class MarketDataProvider:
    """Combined provider for stock and options data"""
    
    def __init__(self):
        """Initialize the market data provider"""
        self.stock_provider = StockDataProvider()
        self.options_provider = OptionsDataProvider()
    
    def get_complete_market_data(self, symbol: str) -> Dict:
        """
        Get complete market data for a symbol
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Dictionary with all market data
        """
        result = {
            'symbol': symbol.upper(),
            'stock_info': None,
            'historical_data': None,
            'options_data': None,
            'current_price': None,
            'error': None
        }
        
        try:
            # Get stock information
            result['stock_info'] = self.stock_provider.get_stock_info(symbol)
            
            # Get historical data
            result['historical_data'] = self.stock_provider.get_historical_data(symbol, "6mo")
            
            # Get options data
            options_df, current_price = self.options_provider.get_options_chain(symbol)
            result['options_data'] = options_df
            result['current_price'] = current_price
            
            # Use current price from options if stock info failed
            if result['stock_info'] and current_price:
                result['stock_info']['currentPrice'] = current_price
            
        except Exception as e:
            result['error'] = str(e)
        
        return result
    
    def get_available_expiry_dates(self, symbol: str) -> List[str]:
        """Get available options expiry dates for a symbol"""
        try:
            import yfinance as yf
            ticker = yf.Ticker(symbol)
            return list(ticker.options)
        except Exception:
            return []

# Factory functions for compatibility
def create_stock_provider() -> StockDataProvider:
    """Create stock data provider instance"""
    return StockDataProvider()

def create_options_provider() -> OptionsDataProvider:
    """Create options data provider instance"""
    return OptionsDataProvider()

def create_market_provider() -> MarketDataProvider:
    """Create market data provider instance"""
    return MarketDataProvider()
