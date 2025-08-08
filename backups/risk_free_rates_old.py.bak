"""
Risk-Free Rates Module
Fetches real-time risk-free rates from public APIs for various currencies and tenors.
Sources: FRED API (US), ECB Data Portal (EU), Bank of England (UK)
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os
from typing import Dict, Optional, List
import warnings
warnings.filterwarnings('ignore')

class RiskFreeRatesProvider:
    """
    Provider for fetching real-time risk-free rates from multiple sources
    """
    
    def __init__(self, fred_api_key: Optional[str] = None):
        """
        Initialize the rates provider
        
        Args:
            fred_api_key: Optional FRED API key. If not provided, will try to get from environment
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
                'overnight': 'EST.B.EU000A2X2A25.WT',  # ECB series code for ESTR
                '1w': 'EST.B.EU000A2X2A25.W1',
                '1m': 'EST.B.EU000A2X2A25.M1',
                '3m': 'EST.B.EU000A2X2A25.M3',
                '6m': 'EST.B.EU000A2X2A25.M6',
                '1y': 'EST.B.EU000A2X2A25.Y1'
            },
            'SONIA': {
                'overnight': 'IUDSOIA',
                '1m': 'IUDSONM',
                '3m': 'IUDSO3M',
                '6m': 'IUDSO6M',
                '1y': 'IUDSO12'
            }
        }
        
        self.current_rates = {}
        self.last_updated = None
    
    def fetch_fred_rate(self, series_id: str) -> Optional[float]:
        """
        Fetch rate from FRED API
        
        Args:
            series_id: FRED series identifier
            
        Returns:
            Latest rate value or None if failed
        """
        if not self.fred_api_key:
            return None
            
        try:
            # Get the latest observation
            url = f"{self.base_urls['fred']}/series/observations"
            params = {
                'series_id': series_id,
                'api_key': self.fred_api_key,
                'file_type': 'json',
                'limit': 1,
                'sort_order': 'desc'
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            observations = data.get('observations', [])
            
            if observations and observations[0]['value'] != '.':
                return float(observations[0]['value'])
                
        except Exception as e:
            print(f"Error fetching FRED rate {series_id}: {str(e)}")
            
        return None
    
    def fetch_ecb_rate(self, series_id: str) -> Optional[float]:
        """
        Fetch rate from ECB Data Portal
        
        Args:
            series_id: ECB series identifier
            
        Returns:
            Latest rate value or None if failed
        """
        try:
            # ECB API endpoint for latest data
            url = f"{self.base_urls['ecb']}/{series_id}"
            params = {
                'format': 'jsondata',
                'lastNObservations': 1
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            # Parse ECB JSON structure
            if 'dataSets' in data and data['dataSets']:
                dataset = data['dataSets'][0]
                if 'series' in dataset and dataset['series']:
                    series = list(dataset['series'].values())[0]
                    if 'observations' in series and series['observations']:
                        obs_key = list(series['observations'].keys())[-1]  # Get latest
                        value = series['observations'][obs_key][0]
                        if value is not None:
                            return float(value)
                            
        except Exception as e:
            print(f"Error fetching ECB rate {series_id}: {str(e)}")
            
        return None
    
    def fetch_boe_rate(self, series_id: str) -> Optional[float]:
        """
        Fetch rate from Bank of England (using alternative approach)
        
        Args:
            series_id: BOE series identifier
            
        Returns:
            Latest rate value or None if failed
        """
        try:
            # Bank of England statistical API
            url = f"https://www.bankofengland.co.uk/boeapps/database/_iadb-fromshowcolumns.asp"
            params = {
                'csv.x': 'yes',
                'Datefrom': '01/Jan/2020',
                'Dateto': 'now',
                'SeriesCodes': series_id,
                'CSVF': 'TN',
                'UsingCodes': 'Y',
                'VPD': 'Y',
                'VFD': 'N'
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            # Parse CSV response
            lines = response.text.strip().split('\n')
            if len(lines) > 1:
                # Get the last line with data
                for line in reversed(lines[1:]):
                    parts = line.split(',')
                    if len(parts) >= 2 and parts[1].strip():
                        try:
                            return float(parts[1].strip())
                        except ValueError:
                            continue
                            
        except Exception as e:
            print(f"Error fetching BOE rate {series_id}: {str(e)}")
            
        return None
    
    def get_fallback_rates(self) -> Dict[str, Dict[str, float]]:
        """
        Get fallback rates when APIs are unavailable
        
        Returns:
            Dictionary of fallback rates by currency and tenor
        """
        return {
            'SOFR': {
                'overnight': 5.30,
                '1m': 5.35,
                '3m': 5.40,
                '6m': 5.45,
                '1y': 5.50
            },
            'ESTR': {
                'overnight': 3.25,
                '1w': 3.30,
                '1m': 3.35,
                '3m': 3.40,
                '6m': 3.45,
                '1y': 3.50
            },
            'SONIA': {
                'overnight': 4.75,
                '1m': 4.80,
                '3m': 4.85,
                '6m': 4.90,
                '1y': 4.95
            },
            'EONIA': {
                'overnight': 3.20,  # EONIA is being phased out, use ESTR-based approximation
                '1m': 3.25,
                '3m': 3.30,
                '6m': 3.35,
                '1y': 3.40
            }
        }
    
    def fetch_all_rates(self, use_fallback: bool = True) -> Dict[str, Dict[str, float]]:
        """
        Fetch all available rates from APIs
        
        Args:
            use_fallback: Whether to use fallback rates when API fails
            
        Returns:
            Dictionary of rates by currency and tenor
        """
        rates = {}
        
        # Fetch SOFR rates (US)
        sofr_rates = {}
        if self.fred_api_key:
            for tenor, series_id in self.rate_series['SOFR'].items():
                rate = self.fetch_fred_rate(series_id)
                if rate is not None:
                    sofr_rates[tenor] = rate
        
        # Fetch ESTR rates (EU)
        estr_rates = {}
        for tenor, series_id in self.rate_series['ESTR'].items():
            rate = self.fetch_ecb_rate(series_id)
            if rate is not None:
                estr_rates[tenor] = rate
        
        # Fetch SONIA rates (UK)
        sonia_rates = {}
        for tenor, series_id in self.rate_series['SONIA'].items():
            rate = self.fetch_boe_rate(series_id)
            if rate is not None:
                sonia_rates[tenor] = rate
        
        # Use fetched rates or fallback
        fallback_rates = self.get_fallback_rates()
        
        rates['SOFR'] = sofr_rates if sofr_rates else (fallback_rates['SOFR'] if use_fallback else {})
        rates['ESTR'] = estr_rates if estr_rates else (fallback_rates['ESTR'] if use_fallback else {})
        rates['SONIA'] = sonia_rates if sonia_rates else (fallback_rates['SONIA'] if use_fallback else {})
        
        # EONIA is deprecated, use ESTR as approximation
        if rates['ESTR']:
            rates['EONIA'] = {k: v - 0.05 for k, v in rates['ESTR'].items()}  # Approximate EONIA as ESTR - 5bps
        elif use_fallback:
            rates['EONIA'] = fallback_rates['EONIA']
        
        self.current_rates = rates
        self.last_updated = datetime.now()
        
        return rates
    
    def get_rate_for_tenor(self, currency: str, tenor: str) -> Optional[float]:
        """
        Get specific rate for currency and tenor
        
        Args:
            currency: Rate currency (SOFR, ESTR, SONIA, EONIA)
            tenor: Rate tenor (overnight, 1m, 3m, 6m, 1y)
            
        Returns:
            Rate value or None if not available
        """
        if not self.current_rates:
            self.fetch_all_rates()
        
        return self.current_rates.get(currency, {}).get(tenor)
    
    def get_rate_suggestions(self, base_currency: str = 'USD') -> List[Dict[str, any]]:
        """
        Get rate suggestions based on base currency and typical option expiries
        
        Args:
            base_currency: Base currency for the options
            
        Returns:
            List of rate suggestions with descriptions
        """
        if not self.current_rates:
            self.fetch_all_rates()
        
        suggestions = []
        
        # Map currencies to their risk-free rates
        currency_map = {
            'USD': 'SOFR',
            'EUR': 'ESTR', 
            'GBP': 'SONIA'
        }
        
        rate_currency = currency_map.get(base_currency, 'SOFR')
        
        # Common option expiries and their corresponding rate tenors
        expiry_map = {
            'overnight': 'Overnight (for very short-term options)',
            '1m': '1 Month (for monthly expiries)',
            '3m': '3 Month (for quarterly expiries)',
            '6m': '6 Month (for semi-annual expiries)',
            '1y': '1 Year (for annual expiries)'
        }
        
        rates_data = self.current_rates.get(rate_currency, {})
        
        for tenor, description in expiry_map.items():
            rate = rates_data.get(tenor)
            if rate is not None:
                suggestions.append({
                    'rate': rate / 100,  # Convert percentage to decimal
                    'rate_percent': rate,
                    'description': f"{rate_currency} {description}",
                    'currency': rate_currency,
                    'tenor': tenor,
                    'source': self._get_source_name(rate_currency)
                })
        
        return suggestions
    
    def _get_source_name(self, rate_currency: str) -> str:
        """Get the source name for a rate currency"""
        sources = {
            'SOFR': 'Federal Reserve Bank of New York',
            'ESTR': 'European Central Bank',
            'SONIA': 'Bank of England',
            'EONIA': 'European Central Bank (deprecated, ESTR-based)'
        }
        return sources.get(rate_currency, 'Unknown')
    
    def get_rate_info(self) -> Dict[str, any]:
        """
        Get information about available rates and their sources
        
        Returns:
            Dictionary with rate information
        """
        info = {
            'rates_available': list(self.current_rates.keys()),
            'last_updated': self.last_updated.isoformat() if self.last_updated else None,
            'sources': {
                'SOFR': {
                    'name': 'Secured Overnight Financing Rate',
                    'source': 'Federal Reserve Bank of New York',
                    'api': 'FRED API',
                    'currency': 'USD'
                },
                'ESTR': {
                    'name': 'Euro Short-Term Rate',
                    'source': 'European Central Bank',
                    'api': 'ECB Data Portal',
                    'currency': 'EUR'
                },
                'SONIA': {
                    'name': 'Sterling Overnight Index Average',
                    'source': 'Bank of England',
                    'api': 'Bank of England Database',
                    'currency': 'GBP'
                },
                'EONIA': {
                    'name': 'Euro Overnight Index Average (deprecated)',
                    'source': 'European Central Bank',
                    'api': 'Approximated from ESTR',
                    'currency': 'EUR',
                    'note': 'EONIA is being phased out, ESTR is the replacement'
                }
            },
            'fred_api_configured': self.fred_api_key is not None
        }
        
        return info

# Convenience function for easy import
def get_risk_free_rates(fred_api_key: Optional[str] = None) -> RiskFreeRatesProvider:
    """
    Get a configured risk-free rates provider
    
    Args:
        fred_api_key: Optional FRED API key
        
    Returns:
        Configured RiskFreeRatesProvider instance
    """
    return RiskFreeRatesProvider(fred_api_key=fred_api_key)
