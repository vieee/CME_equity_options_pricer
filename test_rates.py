"""
Test script for risk-free rates functionality
"""

import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from risk_free_rates import get_risk_free_rates

def test_risk_free_rates():
    """Test the risk-free rates functionality"""
    print("Testing Risk-Free Rates Module")
    print("=" * 50)
    
    # Initialize rates provider
    rates_provider = get_risk_free_rates()
    
    # Test fallback rates
    print("1. Testing fallback rates...")
    fallback_rates = rates_provider.get_fallback_rates()
    print("Fallback rates loaded successfully:")
    for currency, rates in fallback_rates.items():
        print(f"  {currency}: {rates}")
    
    print("\n2. Testing rate suggestions...")
    suggestions = rates_provider.get_rate_suggestions('USD')
    if suggestions:
        print("Rate suggestions for USD:")
        for suggestion in suggestions:
            print(f"  • {suggestion['rate_percent']:.2f}% - {suggestion['description']}")
    else:
        print("No rate suggestions available")
    
    print("\n3. Testing rate info...")
    rate_info = rates_provider.get_rate_info()
    print(f"Available rates: {rate_info['rates_available']}")
    print(f"FRED API configured: {rate_info['fred_api_configured']}")
    
    print("\n4. Testing specific rate lookup...")
    sofr_overnight = rates_provider.get_rate_for_tenor('SOFR', 'overnight')
    print(f"SOFR overnight rate: {sofr_overnight}%")
    
    print("\n✅ All tests completed successfully!")

if __name__ == "__main__":
    test_risk_free_rates()
