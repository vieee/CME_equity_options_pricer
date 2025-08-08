"""
Test suite for Options Pricing Models
"""
import unittest
import numpy as np
import sys
import os

# Add src to path
current_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(current_dir)
sys.path.append(os.path.join(parent_dir, 'src'))

from models.pricing import OptionsPricingEngine
from utils.validators import DataValidator
from utils.formatters import DataFormatter

class TestOptionsPricing(unittest.TestCase):
    """Test options pricing calculations"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.pricing_engine = OptionsPricingEngine(risk_free_rate=0.05)
        self.S = 100  # Stock price
        self.K = 100  # Strike price
        self.T = 0.25  # Time to expiry (3 months)
        self.r = 0.05  # Risk-free rate
        self.sigma = 0.20  # Volatility
    
    def test_black_scholes_call(self):
        """Test Black-Scholes call option pricing"""
        call_price = self.pricing_engine.black_scholes_call(
            self.S, self.K, self.T, self.r, self.sigma
        )
        
        # Call price should be positive
        self.assertGreater(call_price, 0)
        
        # ATM call should have positive time value
        self.assertGreater(call_price, 0)
    
    def test_black_scholes_put(self):
        """Test Black-Scholes put option pricing"""
        put_price = self.pricing_engine.black_scholes_put(
            self.S, self.K, self.T, self.r, self.sigma
        )
        
        # Put price should be positive
        self.assertGreater(put_price, 0)
    
    def test_put_call_parity(self):
        """Test put-call parity relationship"""
        call_price = self.pricing_engine.black_scholes_call(
            self.S, self.K, self.T, self.r, self.sigma
        )
        put_price = self.pricing_engine.black_scholes_put(
            self.S, self.K, self.T, self.r, self.sigma
        )
        
        # Put-call parity: C - P = S - K*e^(-r*T)
        parity_left = call_price - put_price
        parity_right = self.S - self.K * np.exp(-self.r * self.T)
        
        self.assertAlmostEqual(parity_left, parity_right, places=5)
    
    def test_greeks_calculation(self):
        """Test Greeks calculations"""
        greeks = self.pricing_engine.calculate_greeks(
            self.S, self.K, self.T, self.r, self.sigma, 'call'
        )
        
        # Delta should be between 0 and 1 for calls
        self.assertGreaterEqual(greeks['delta'], 0)
        self.assertLessEqual(greeks['delta'], 1)
        
        # Gamma should be positive
        self.assertGreater(greeks['gamma'], 0)
        
        # Vega should be positive
        self.assertGreater(greeks['vega'], 0)
    
    def test_binomial_pricing(self):
        """Test binomial tree pricing"""
        call_price = self.pricing_engine.binomial_option_price(
            self.S, self.K, self.T, self.r, self.sigma, 'call', steps=50
        )
        
        # Binomial price should be positive
        self.assertGreater(call_price, 0)
        
        # Should be close to Black-Scholes price
        bs_price = self.pricing_engine.black_scholes_call(
            self.S, self.K, self.T, self.r, self.sigma
        )
        
        # Allow 5% difference due to discretization
        self.assertAlmostEqual(call_price, bs_price, delta=bs_price * 0.05)
    
    def test_monte_carlo_pricing(self):
        """Test Monte Carlo pricing"""
        call_price = self.pricing_engine.monte_carlo_option_price(
            self.S, self.K, self.T, self.r, self.sigma, 'call', simulations=10000
        )
        
        # Monte Carlo price should be positive
        self.assertGreater(call_price, 0)
        
        # Should be reasonably close to Black-Scholes price
        bs_price = self.pricing_engine.black_scholes_call(
            self.S, self.K, self.T, self.r, self.sigma
        )
        
        # Allow 10% difference due to random sampling
        self.assertAlmostEqual(call_price, bs_price, delta=bs_price * 0.1)

class TestDataValidation(unittest.TestCase):
    """Test data validation functions"""
    
    def test_symbol_validation(self):
        """Test symbol validation"""
        self.assertTrue(DataValidator.validate_symbol("AAPL"))
        self.assertTrue(DataValidator.validate_symbol("SPY"))
        self.assertFalse(DataValidator.validate_symbol(""))
        self.assertFalse(DataValidator.validate_symbol("TOOLONGSYMBOL"))
        self.assertFalse(DataValidator.validate_symbol("123!@#"))
    
    def test_price_validation(self):
        """Test price validation"""
        self.assertTrue(DataValidator.validate_price(100.0))
        self.assertTrue(DataValidator.validate_price(0.01))
        self.assertFalse(DataValidator.validate_price(0))
        self.assertFalse(DataValidator.validate_price(-10))
        self.assertFalse(DataValidator.validate_price("invalid"))
    
    def test_rate_validation(self):
        """Test rate validation"""
        self.assertTrue(DataValidator.validate_rate(0.05))
        self.assertTrue(DataValidator.validate_rate(0.0))
        self.assertTrue(DataValidator.validate_rate(-0.05))
        self.assertFalse(DataValidator.validate_rate(0.6))  # 60% rate
        self.assertFalse(DataValidator.validate_rate(-0.2))  # -20% rate
    
    def test_volatility_validation(self):
        """Test volatility validation"""
        self.assertTrue(DataValidator.validate_volatility(0.25))
        self.assertTrue(DataValidator.validate_volatility(0.01))
        self.assertTrue(DataValidator.validate_volatility(1.0))
        self.assertFalse(DataValidator.validate_volatility(0))
        self.assertFalse(DataValidator.validate_volatility(6.0))  # 600% volatility

class TestDataFormatting(unittest.TestCase):
    """Test data formatting functions"""
    
    def test_currency_formatting(self):
        """Test currency formatting"""
        self.assertEqual(DataFormatter.format_currency(123.456), "$123.46")
        self.assertEqual(DataFormatter.format_currency(1000000), "$1,000,000.00")
        self.assertEqual(DataFormatter.format_currency(np.nan), "N/A")
    
    def test_percentage_formatting(self):
        """Test percentage formatting"""
        self.assertEqual(DataFormatter.format_percentage(12.345), "12.35%")
        self.assertEqual(DataFormatter.format_percentage(0.1), "0.10%")
        self.assertEqual(DataFormatter.format_percentage(np.nan), "N/A")
    
    def test_large_number_formatting(self):
        """Test large number formatting"""
        self.assertEqual(DataFormatter.format_large_number(1234), "1,234")
        self.assertEqual(DataFormatter.format_large_number(1234567), "1.2M")
        self.assertEqual(DataFormatter.format_large_number(1234567890), "1.2B")
        self.assertEqual(DataFormatter.format_large_number(np.nan), "N/A")

if __name__ == '__main__':
    unittest.main()
