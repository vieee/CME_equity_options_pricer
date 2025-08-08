"""
Options Pricing Models - Refactored and Enhanced
Implementation of various benchmark pricing techniques
"""

import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import brentq
import math
from datetime import datetime, timedelta
from typing import Dict, Union, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

class OptionsPricingEngine:
    """
    Comprehensive options pricing engine with multiple models:
    - Black-Scholes-Merton Model
    - Binomial Tree Model  
    - Monte Carlo Simulation
    - Greeks calculations
    - Implied Volatility calculations
    """
    
    def __init__(self, risk_free_rate: float = 0.05):
        """
        Initialize the pricing engine
        
        Args:
            risk_free_rate: Default risk-free rate
        """
        self.risk_free_rate = risk_free_rate
    
    def black_scholes_call(self, S: float, K: float, T: float, r: float, 
                          sigma: float, q: float = 0) -> float:
        """
        Black-Scholes-Merton formula for European call options
        
        Args:
            S: Current stock price
            K: Strike price
            T: Time to expiration (in years)
            r: Risk-free rate
            sigma: Volatility
            q: Dividend yield (default 0)
            
        Returns:
            Call option price
        """
        if T <= 0:
            return max(0, S - K)
        
        if sigma <= 0:
            return max(0, S - K) if S > K else 0
        
        try:
            d1 = (np.log(S/K) + (r - q + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
            d2 = d1 - sigma*np.sqrt(T)
            
            call_price = S*np.exp(-q*T)*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
            return max(0, call_price)
        except (ZeroDivisionError, ValueError, OverflowError):
            return max(0, S - K) if S > K else 0
    
    def black_scholes_put(self, S: float, K: float, T: float, r: float, 
                         sigma: float, q: float = 0) -> float:
        """
        Black-Scholes-Merton formula for European put options
        
        Args:
            S: Current stock price
            K: Strike price
            T: Time to expiration (in years)
            r: Risk-free rate
            sigma: Volatility
            q: Dividend yield (default 0)
            
        Returns:
            Put option price
        """
        if T <= 0:
            return max(0, K - S)
        
        if sigma <= 0:
            return max(0, K - S) if K > S else 0
        
        try:
            d1 = (np.log(S/K) + (r - q + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
            d2 = d1 - sigma*np.sqrt(T)
            
            put_price = K*np.exp(-r*T)*norm.cdf(-d2) - S*np.exp(-q*T)*norm.cdf(-d1)
            return max(0, put_price)
        except (ZeroDivisionError, ValueError, OverflowError):
            return max(0, K - S) if K > S else 0
    
    def calculate_greeks(self, S: float, K: float, T: float, r: float, 
                        sigma: float, option_type: str = 'call', 
                        q: float = 0) -> Dict[str, float]:
        """
        Calculate option Greeks (Delta, Gamma, Theta, Vega, Rho)
        
        Args:
            S: Current stock price
            K: Strike price
            T: Time to expiration (in years)
            r: Risk-free rate
            sigma: Volatility
            option_type: 'call' or 'put'
            q: Dividend yield (default 0)
            
        Returns:
            Dictionary containing all Greeks
        """
        if T <= 0 or sigma <= 0:
            return {'delta': 0, 'gamma': 0, 'theta': 0, 'vega': 0, 'rho': 0}
        
        try:
            d1 = (np.log(S/K) + (r - q + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
            d2 = d1 - sigma*np.sqrt(T)
            
            # Delta
            if option_type.lower() == 'call':
                delta = np.exp(-q*T) * norm.cdf(d1)
            else:
                delta = -np.exp(-q*T) * norm.cdf(-d1)
            
            # Gamma (same for calls and puts)
            gamma = np.exp(-q*T) * norm.pdf(d1) / (S * sigma * np.sqrt(T))
            
            # Theta
            theta_common = -(S * norm.pdf(d1) * sigma * np.exp(-q*T)) / (2 * np.sqrt(T))
            if option_type.lower() == 'call':
                theta = theta_common - r*K*np.exp(-r*T)*norm.cdf(d2) + q*S*np.exp(-q*T)*norm.cdf(d1)
            else:
                theta = theta_common + r*K*np.exp(-r*T)*norm.cdf(-d2) - q*S*np.exp(-q*T)*norm.cdf(-d1)
            theta = theta / 365  # Convert to daily theta
            
            # Vega (same for calls and puts)
            vega = S * np.exp(-q*T) * norm.pdf(d1) * np.sqrt(T) / 100  # Divide by 100 for 1% change
            
            # Rho
            if option_type.lower() == 'call':
                rho = K * T * np.exp(-r*T) * norm.cdf(d2) / 100
            else:
                rho = -K * T * np.exp(-r*T) * norm.cdf(-d2) / 100
            
            return {
                'delta': delta,
                'gamma': gamma,
                'theta': theta,
                'vega': vega,
                'rho': rho
            }
        except (ZeroDivisionError, ValueError, OverflowError):
            return {'delta': 0, 'gamma': 0, 'theta': 0, 'vega': 0, 'rho': 0}
    
    def binomial_option_price(self, S: float, K: float, T: float, r: float, 
                             sigma: float, option_type: str = 'call', 
                             steps: int = 100, q: float = 0) -> float:
        """
        Binomial tree option pricing model
        
        Args:
            S: Current stock price
            K: Strike price
            T: Time to expiration (in years)
            r: Risk-free rate
            sigma: Volatility
            option_type: 'call' or 'put'
            steps: Number of time steps
            q: Dividend yield (default 0)
            
        Returns:
            Option price
        """
        if T <= 0:
            if option_type.lower() == 'call':
                return max(0, S - K)
            else:
                return max(0, K - S)
        
        if sigma <= 0 or steps <= 0:
            if option_type.lower() == 'call':
                return max(0, S - K) if S > K else 0
            else:
                return max(0, K - S) if K > S else 0
        
        try:
            dt = T / steps
            u = np.exp(sigma * np.sqrt(dt))
            d = 1 / u
            p = (np.exp((r - q) * dt) - d) / (u - d)
            
            # Stock price tree
            stock_tree = np.zeros((steps + 1, steps + 1))
            for i in range(steps + 1):
                for j in range(i + 1):
                    stock_tree[j, i] = S * (u ** (i - j)) * (d ** j)
            
            # Option value tree
            option_tree = np.zeros((steps + 1, steps + 1))
            
            # Terminal values
            for j in range(steps + 1):
                if option_type.lower() == 'call':
                    option_tree[j, steps] = max(0, stock_tree[j, steps] - K)
                else:
                    option_tree[j, steps] = max(0, K - stock_tree[j, steps])
            
            # Backward induction
            for i in range(steps - 1, -1, -1):
                for j in range(i + 1):
                    option_tree[j, i] = np.exp(-r * dt) * (
                        p * option_tree[j, i + 1] + (1 - p) * option_tree[j + 1, i + 1]
                    )
            
            return option_tree[0, 0]
        except (ZeroDivisionError, ValueError, OverflowError):
            if option_type.lower() == 'call':
                return max(0, S - K) if S > K else 0
            else:
                return max(0, K - S) if K > S else 0
    
    def monte_carlo_option_price(self, S: float, K: float, T: float, r: float, 
                                sigma: float, option_type: str = 'call', 
                                simulations: int = 10000, q: float = 0) -> float:
        """
        Monte Carlo option pricing simulation
        
        Args:
            S: Current stock price
            K: Strike price
            T: Time to expiration (in years)
            r: Risk-free rate
            sigma: Volatility
            option_type: 'call' or 'put'
            simulations: Number of Monte Carlo simulations
            q: Dividend yield (default 0)
            
        Returns:
            Option price
        """
        if T <= 0:
            if option_type.lower() == 'call':
                return max(0, S - K)
            else:
                return max(0, K - S)
        
        if sigma <= 0 or simulations <= 0:
            if option_type.lower() == 'call':
                return max(0, S - K) if S > K else 0
            else:
                return max(0, K - S) if K > S else 0
        
        try:
            # Generate random paths
            np.random.seed(42)  # For reproducibility
            dt = T
            drift = (r - q - 0.5 * sigma**2) * dt
            diffusion = sigma * np.sqrt(dt)
            
            # Simulate final stock prices
            random_shocks = np.random.normal(0, 1, simulations)
            final_prices = S * np.exp(drift + diffusion * random_shocks)
            
            # Calculate payoffs
            if option_type.lower() == 'call':
                payoffs = np.maximum(final_prices - K, 0)
            else:
                payoffs = np.maximum(K - final_prices, 0)
            
            # Discount back to present value
            option_price = np.exp(-r * T) * np.mean(payoffs)
            return option_price
        except (ZeroDivisionError, ValueError, OverflowError):
            if option_type.lower() == 'call':
                return max(0, S - K) if S > K else 0
            else:
                return max(0, K - S) if K > S else 0
    
    def calculate_implied_volatility(self, market_price: float, S: float, K: float, 
                                   T: float, r: float, option_type: str = 'call', 
                                   q: float = 0) -> Optional[float]:
        """
        Calculate implied volatility using Brent's method
        
        Args:
            market_price: Observed market price
            S: Current stock price
            K: Strike price
            T: Time to expiration (in years)
            r: Risk-free rate
            option_type: 'call' or 'put'
            q: Dividend yield (default 0)
            
        Returns:
            Implied volatility or None if calculation fails
        """
        if T <= 0 or market_price <= 0:
            return None
        
        def price_diff(sigma):
            if option_type.lower() == 'call':
                theoretical_price = self.black_scholes_call(S, K, T, r, sigma, q)
            else:
                theoretical_price = self.black_scholes_put(S, K, T, r, sigma, q)
            return theoretical_price - market_price
        
        try:
            # Use Brent's method to find root
            implied_vol = brentq(price_diff, 0.001, 5.0, xtol=1e-6, maxiter=100)
            return implied_vol
        except (ValueError, RuntimeError):
            return None
    
    def calculate_pricing_breakdown(self, S: float, K: float, T: float, r: float, 
                                  sigma: float, option_type: str = 'call', 
                                  q: float = 0) -> Dict[str, Union[float, Dict]]:
        """
        Calculate comprehensive pricing breakdown with all models
        
        Args:
            S: Current stock price
            K: Strike price
            T: Time to expiration (in years)
            r: Risk-free rate
            sigma: Volatility
            option_type: 'call' or 'put'
            q: Dividend yield (default 0)
            
        Returns:
            Dictionary with pricing results from all models
        """
        breakdown = {
            'inputs': {
                'stock_price': S,
                'strike_price': K,
                'time_to_expiry': T,
                'risk_free_rate': r,
                'volatility': sigma,
                'dividend_yield': q,
                'option_type': option_type
            },
            'black_scholes': 0.0,
            'binomial_tree': 0.0,
            'monte_carlo': 0.0,
            'greeks': {},
            'moneyness': '',
            'time_value': 0.0,
            'intrinsic_value': 0.0
        }
        
        try:
            # Calculate prices with different models
            if option_type.lower() == 'call':
                breakdown['black_scholes'] = self.black_scholes_call(S, K, T, r, sigma, q)
                breakdown['intrinsic_value'] = max(0, S - K)
            else:
                breakdown['black_scholes'] = self.black_scholes_put(S, K, T, r, sigma, q)
                breakdown['intrinsic_value'] = max(0, K - S)
            
            breakdown['binomial_tree'] = self.binomial_option_price(S, K, T, r, sigma, option_type, 50, q)
            breakdown['monte_carlo'] = self.monte_carlo_option_price(S, K, T, r, sigma, option_type, 5000, q)
            
            # Calculate Greeks
            breakdown['greeks'] = self.calculate_greeks(S, K, T, r, sigma, option_type, q)
            
            # Calculate derived values
            breakdown['time_value'] = breakdown['black_scholes'] - breakdown['intrinsic_value']
            
            # Determine moneyness
            if option_type.lower() == 'call':
                if S > K:
                    breakdown['moneyness'] = 'ITM'  # In-the-money
                elif abs(S - K) / K < 0.02:  # Within 2%
                    breakdown['moneyness'] = 'ATM'  # At-the-money
                else:
                    breakdown['moneyness'] = 'OTM'  # Out-of-the-money
            else:
                if S < K:
                    breakdown['moneyness'] = 'ITM'
                elif abs(S - K) / K < 0.02:
                    breakdown['moneyness'] = 'ATM'
                else:
                    breakdown['moneyness'] = 'OTM'
            
        except Exception as e:
            # Return breakdown with error info
            breakdown['error'] = str(e)
        
        return breakdown

# Legacy compatibility class
class OptionsPricingModels(OptionsPricingEngine):
    """Legacy compatibility class - redirects to new OptionsPricingEngine"""
    pass
