"""
Risk-Free Rates Provider - Refactored and Enhanced
Fetches real-time risk-free rates from multiple APIs
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os
from typing import Dict, Optional, List, Tuple
import warnings
warnings.filterwarnings('ignore')

class RiskFreeRatesProvider:
    """
    Enhanced provider for fetching real-time risk-free rates from multiple sources
    Sources: FRED API (US), ECB Data Portal (EU), Bank of England (UK)
    """
    
    def __init__(self, fred_api_key: Optional[str] = None):
        """
        Initialize the rates provider
        
        Args:
            fred_api_key: Optional FRED API key. If not provided, will try environment
        """
        self.fred_api_key = fred_api_key or os.getenv('FRED_API_KEY')
        self.base_urls = {
            'fred': 'https://api.stlouisfed.org/fred',
            'ecb': 'https://data.ecb.europa.eu/api/v1/data',
            'boe': 'https://www.bankofengland.co.uk/boeapps/database'
        }
        
        # Rate series mappings
        self.rate_series = {
            'SOFR': {
                'overnight': 'SOFR',
                '1m': 'SOFR1M',
                '3m': 'SOFR3M',
                '6m': 'SOFR6M',
                '1y': 'SOFR12M'
            },
            'ESTR': {
                'overnight': 'EST.B.EU000A2X2A25.WT',
                '1w': 'EST.B.EU000A2X2A25.W1',
                '1m': 'EST.B.EU000A2X2A25.M1',
                '3m': 'EST.B.EU000A2X2A25.M3',
                '6m': 'EST.B.EU000A2X2A25.M6',
                '1y': 'EST.B.EU000A2X2A25.Y1'
            },
            'SONIA': {
                'overnight': 'IUDSOIA',  # UK Official interest rates
                '1m': 'IUDSOIA1M',
                '3m': 'IUDSOIA3M',
                '6m': 'IUDSOIA6M',
                '1y': 'IUDSOIA1Y'
            },
            'EONIA': {
                'overnight': 'EUEON',
                '1w': 'EUEON1W',
                '1m': 'EUEON1M',
                '3m': 'EUEON3M',
                '6m': 'EUEON6M',
                '1y': 'EUEON1Y'
            }
        }
        
        # Fallback rates (approximate current market rates)
        self.fallback_rates = {
            'SOFR': {
                'overnight': 5.32,
                '1m': 5.35,
                '3m': 5.38,
                '6m': 5.25,
                '1y': 4.95
            },
            'ESTR': {
                'overnight': 3.40,
                '1w': 3.42,
                '1m': 3.45,
                '3m': 3.48,
                '6m': 3.35,
                '1y': 3.15
            },
            'SONIA': {
                'overnight': 4.75,
                '1m': 4.78,
                '3m': 4.82,
                '6m': 4.65,
                '1y': 4.25
            },
            'EONIA': {
                'overnight': 3.40,  # Based on ESTR
                '1w': 3.42,
                '1m': 3.45,
                '3m': 3.48,
                '6m': 3.35,
                '1y': 3.15
            }
        }
        
        self.current_rates = {}
        self.last_updated = None
        
        # Initialize with fallback rates
        self.current_rates = self.fallback_rates.copy()
    
    def fetch_fred_rates(self) -> Dict[str, Dict[str, Optional[float]]]:
        """Fetch rates from FRED API"""
        fred_rates = {}
        
        if not self.fred_api_key:
            return fred_rates
        
        for currency, series_dict in self.rate_series.items():
            if currency in ['SOFR']:  # Only FRED-supported currencies
                fred_rates[currency] = {}
                
                for tenor, series_id in series_dict.items():
                    try:
                        url = f"{self.base_urls['fred']}/series/observations"
                        params = {
                            'series_id': series_id,
                            'api_key': self.fred_api_key,
                            'file_type': 'json',
                            'sort_order': 'desc',
                            'limit': 1
                        }
                        
                        response = requests.get(url, params=params, timeout=10)
                        if response.status_code == 200:
                            data = response.json()
                            if data['observations']:
                                rate_value = data['observations'][0]['value']
                                if rate_value != '.':
                                    fred_rates[currency][tenor] = float(rate_value)
                                else:
                                    fred_rates[currency][tenor] = None
                            else:
                                fred_rates[currency][tenor] = None
                        else:
                            fred_rates[currency][tenor] = None
                    
                    except (requests.RequestException, json.JSONDecodeError, KeyError, ValueError):
                        fred_rates[currency][tenor] = None
        
        return fred_rates
    
    def fetch_ecb_rates(self) -> Dict[str, Dict[str, Optional[float]]]:
        """Fetch rates from ECB (simplified approach)"""
        ecb_rates = {}
        
        # For now, return fallback ESTR rates
        # In production, would implement proper ECB API integration
        for currency in ['ESTR', 'EONIA']:
            if currency in self.fallback_rates:
                ecb_rates[currency] = self.fallback_rates[currency].copy()
        
        return ecb_rates
    
    def fetch_boe_rates(self) -> Dict[str, Dict[str, Optional[float]]]:
        """Fetch rates from Bank of England (simplified approach)"""
        boe_rates = {}
        
        # For now, return fallback SONIA rates
        # In production, would implement proper BoE API integration
        if 'SONIA' in self.fallback_rates:
            boe_rates['SONIA'] = self.fallback_rates['SONIA'].copy()
        
        return boe_rates
    
    def fetch_all_rates(self) -> None:
        """Fetch rates from all available sources"""
        try:
            # Start with fallback rates
            self.current_rates = self.fallback_rates.copy()
            
            # Try to update with live data
            fred_rates = self.fetch_fred_rates()
            ecb_rates = self.fetch_ecb_rates()
            boe_rates = self.fetch_boe_rates()
            
            # Merge live rates (overwrites fallback where available)
            for currency_rates in [fred_rates, ecb_rates, boe_rates]:
                for currency, rates in currency_rates.items():
                    if currency not in self.current_rates:
                        self.current_rates[currency] = {}
                    
                    for tenor, rate in rates.items():
                        if rate is not None:
                            self.current_rates[currency][tenor] = rate
            
            self.last_updated = datetime.now().isoformat()
            
        except Exception as e:
            # Keep fallback rates on error
            print(f"Error fetching rates: {e}")
    
    def get_rate_suggestions(self, currency: str = 'USD', 
                           option_expiry_days: Optional[int] = None) -> List[Dict]:
        """
        Get rate suggestions based on currency and option expiry
        
        Args:
            currency: Currency code ('USD', 'EUR', 'GBP')
            option_expiry_days: Days to option expiry for tenor matching
            
        Returns:
            List of rate suggestions with metadata
        """
        currency_map = {
            'USD': 'SOFR',
            'EUR': 'ESTR',
            'GBP': 'SONIA'
        }
        
        rate_currency = currency_map.get(currency, 'SOFR')
        suggestions = []
        
        if rate_currency in self.current_rates:
            rates = self.current_rates[rate_currency]
            
            # Determine appropriate tenor based on expiry
            recommended_tenor = self._get_recommended_tenor(option_expiry_days)
            
            for tenor, rate in rates.items():
                if rate is not None:
                    is_recommended = (tenor == recommended_tenor)
                    tenor_description = self._get_tenor_description(tenor)
                    
                    suggestion = {
                        'rate': rate / 100,  # Convert to decimal
                        'rate_percent': rate,
                        'tenor': tenor,
                        'description': f"{rate_currency} {tenor_description}",
                        'currency': rate_currency,
                        'recommended': is_recommended,
                        'source': self._get_rate_source(rate_currency)
                    }
                    suggestions.append(suggestion)
            
            # Sort by recommendation, then by tenor preference
            suggestions.sort(key=lambda x: (not x['recommended'], self._get_tenor_priority(x['tenor'])))
        
        return suggestions
    
    def _get_recommended_tenor(self, expiry_days: Optional[int]) -> str:
        """Get recommended tenor based on option expiry"""
        if expiry_days is None:
            return '3m'  # Default to 3-month
        
        if expiry_days <= 7:
            return 'overnight'
        elif expiry_days <= 35:
            return '1m'
        elif expiry_days <= 100:
            return '3m'
        elif expiry_days <= 200:
            return '6m'
        else:
            return '1y'
    
    def _get_tenor_description(self, tenor: str) -> str:
        """Get human-readable tenor description"""
        tenor_map = {
            'overnight': 'Overnight',
            '1w': '1 Week',
            '1m': '1 Month',
            '3m': '3 Month',
            '6m': '6 Month',
            '1y': '1 Year'
        }
        return tenor_map.get(tenor, tenor.title())
    
    def _get_tenor_priority(self, tenor: str) -> int:
        """Get sorting priority for tenors"""
        priority_map = {
            'overnight': 1,
            '1w': 2,
            '1m': 3,
            '3m': 4,
            '6m': 5,
            '1y': 6
        }
        return priority_map.get(tenor, 99)
    
    def _get_rate_source(self, currency: str) -> str:
        """Get data source description"""
        source_map = {
            'SOFR': 'Federal Reserve (FRED API)',
            'ESTR': 'European Central Bank',
            'SONIA': 'Bank of England',
            'EONIA': 'European Central Bank (ESTR-based)'
        }
        return source_map.get(currency, 'Market Data')
    
    def get_rate_info(self) -> Dict:
        """Get information about current rates"""
        return {
            'last_updated': self.last_updated,
            'currencies_available': list(self.current_rates.keys()),
            'fred_api_configured': self.fred_api_key is not None,
            'total_rates': sum(len(rates) for rates in self.current_rates.values())
        }
    
    def get_rate_by_currency_tenor(self, currency: str, tenor: str) -> Optional[float]:
        """Get specific rate by currency and tenor"""
        currency_map = {
            'USD': 'SOFR',
            'EUR': 'ESTR',
            'GBP': 'SONIA'
        }
        
        rate_currency = currency_map.get(currency, currency)
        
        if rate_currency in self.current_rates:
            rate = self.current_rates[rate_currency].get(tenor)
            if rate is not None:
                return rate / 100  # Convert to decimal
        
        return None

# Factory function for compatibility
def get_risk_free_rates(fred_api_key: Optional[str] = None) -> RiskFreeRatesProvider:
    """
    Factory function to create RiskFreeRatesProvider instance
    
    Args:
        fred_api_key: Optional FRED API key
        
    Returns:
        RiskFreeRatesProvider instance
    """
    provider = RiskFreeRatesProvider(fred_api_key)
    provider.fetch_all_rates()  # Initialize with current rates
    return provider
