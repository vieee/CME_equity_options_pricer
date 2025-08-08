"""
Options Pricing Models for CME Equity Options
Implementation of various benchmark pricing techniques
"""

import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import brentq
import math
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class OptionsPricingModels:
    """
    Comprehensive options pricing models including:
    - Black-Scholes-Merton Model
    - Binomial Tree Model
    - Monte Carlo Simulation
    - Black-76 Model (for futures options)
    - Implied Volatility calculations
    """
    
    def __init__(self, risk_free_rate=0.05):
        self.risk_free_rate = risk_free_rate
    
    def black_scholes_call(self, S, K, T, r, sigma, q=0):
        """
        Black-Scholes-Merton formula for European call options
        
        Parameters:
        S: Current stock price
        K: Strike price
        T: Time to expiration (in years)
        r: Risk-free rate
        sigma: Volatility
        q: Dividend yield (default 0)
        """
        if T <= 0:
            return max(0, S - K)
        
        d1 = (np.log(S/K) + (r - q + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
        d2 = d1 - sigma*np.sqrt(T)
        
        call_price = S*np.exp(-q*T)*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
        return max(0, call_price)
    
    def black_scholes_put(self, S, K, T, r, sigma, q=0):
        """
        Black-Scholes-Merton formula for European put options
        """
        if T <= 0:
            return max(0, K - S)
        
        d1 = (np.log(S/K) + (r - q + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
        d2 = d1 - sigma*np.sqrt(T)
        
        put_price = K*np.exp(-r*T)*norm.cdf(-d2) - S*np.exp(-q*T)*norm.cdf(-d1)
        return max(0, put_price)
    
    def calculate_greeks(self, S, K, T, r, sigma, option_type='call', q=0):
        """
        Calculate option Greeks (Delta, Gamma, Theta, Vega, Rho)
        """
        if T <= 0:
            return {'delta': 0, 'gamma': 0, 'theta': 0, 'vega': 0, 'rho': 0}
        
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
    
    def binomial_option_price(self, S, K, T, r, sigma, n=100, option_type='call', american=False):
        """
        Binomial tree model for option pricing
        
        Parameters:
        S: Current stock price
        K: Strike price
        T: Time to expiration (in years)
        r: Risk-free rate
        sigma: Volatility
        n: Number of time steps
        option_type: 'call' or 'put'
        american: True for American options, False for European
        """
        if T <= 0:
            if option_type.lower() == 'call':
                return max(0, S - K)
            else:
                return max(0, K - S)
        
        dt = T / n
        u = np.exp(sigma * np.sqrt(dt))
        d = 1 / u
        p = (np.exp(r * dt) - d) / (u - d)
        
        # Initialize asset prices at maturity
        prices = np.zeros(n + 1)
        for i in range(n + 1):
            prices[i] = S * (u ** (n - i)) * (d ** i)
        
        # Initialize option values at maturity
        option_values = np.zeros(n + 1)
        for i in range(n + 1):
            if option_type.lower() == 'call':
                option_values[i] = max(0, prices[i] - K)
            else:
                option_values[i] = max(0, K - prices[i])
        
        # Work backwards through the tree
        for j in range(n - 1, -1, -1):
            for i in range(j + 1):
                # European option value
                european_value = np.exp(-r * dt) * (p * option_values[i] + (1 - p) * option_values[i + 1])
                
                if american:
                    # American option value (early exercise)
                    current_price = S * (u ** (j - i)) * (d ** i)
                    if option_type.lower() == 'call':
                        intrinsic_value = max(0, current_price - K)
                    else:
                        intrinsic_value = max(0, K - current_price)
                    
                    option_values[i] = max(european_value, intrinsic_value)
                else:
                    option_values[i] = european_value
        
        return option_values[0]
    
    def monte_carlo_option_price(self, S, K, T, r, sigma, n_simulations=10000, option_type='call'):
        """
        Monte Carlo simulation for European option pricing
        """
        if T <= 0:
            if option_type.lower() == 'call':
                return max(0, S - K)
            else:
                return max(0, K - S)
        
        # Generate random price paths
        Z = np.random.standard_normal(n_simulations)
        ST = S * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)
        
        # Calculate payoffs
        if option_type.lower() == 'call':
            payoffs = np.maximum(ST - K, 0)
        else:
            payoffs = np.maximum(K - ST, 0)
        
        # Discount back to present value
        option_price = np.exp(-r * T) * np.mean(payoffs)
        return option_price
    
    def implied_volatility(self, market_price, S, K, T, r, option_type='call', q=0):
        """
        Calculate implied volatility using Brent's method
        """
        if T <= 0:
            return 0
        
        def objective_function(sigma):
            if option_type.lower() == 'call':
                theoretical_price = self.black_scholes_call(S, K, T, r, sigma, q)
            else:
                theoretical_price = self.black_scholes_put(S, K, T, r, sigma, q)
            return theoretical_price - market_price
        
        try:
            # Try to find implied volatility between 0.01% and 500%
            iv = brentq(objective_function, 0.0001, 5.0, xtol=1e-6)
            return iv
        except:
            return 0
    
    def black76_option_price(self, F, K, T, r, sigma, option_type='call'):
        """
        Black-76 model for options on futures
        
        Parameters:
        F: Forward/Futures price
        K: Strike price
        T: Time to expiration
        r: Risk-free rate
        sigma: Volatility
        """
        if T <= 0:
            if option_type.lower() == 'call':
                return max(0, F - K)
            else:
                return max(0, K - F)
        
        d1 = (np.log(F/K) + 0.5*sigma**2*T) / (sigma*np.sqrt(T))
        d2 = d1 - sigma*np.sqrt(T)
        
        if option_type.lower() == 'call':
            price = np.exp(-r*T) * (F*norm.cdf(d1) - K*norm.cdf(d2))
        else:
            price = np.exp(-r*T) * (K*norm.cdf(-d2) - F*norm.cdf(-d1))
        
        return max(0, price)
    
    def calculate_time_to_expiration(self, expiration_date):
        """
        Calculate time to expiration in years
        """
        if isinstance(expiration_date, str):
            exp_date = datetime.strptime(expiration_date, '%Y-%m-%d')
        else:
            exp_date = expiration_date
        
        today = datetime.now()
        time_diff = exp_date - today
        return max(0, time_diff.days / 365.25)
    
    def price_options_dataframe(self, options_df, current_price, risk_free_rate=None):
        """
        Price all options in a DataFrame using Black-Scholes model
        """
        if risk_free_rate is None:
            risk_free_rate = self.risk_free_rate
        
        priced_options = options_df.copy()
        
        # Add theoretical prices and Greeks
        theoretical_prices = []
        deltas = []
        gammas = []
        thetas = []
        vegas = []
        rhos = []
        implied_vols = []
        
        for idx, row in options_df.iterrows():
            try:
                # Calculate time to expiration
                T = self.calculate_time_to_expiration(row['expiration'])
                
                # Get option parameters
                K = row['strike']
                option_type = row['option_type']
                market_price = row.get('lastPrice', row.get('bid', 0))
                
                if T > 0 and market_price > 0:
                    # Calculate theoretical price (using 20% volatility as default)
                    if option_type.lower() == 'call':
                        theoretical_price = self.black_scholes_call(current_price, K, T, risk_free_rate, 0.2)
                    else:
                        theoretical_price = self.black_scholes_put(current_price, K, T, risk_free_rate, 0.2)
                    
                    # Calculate Greeks
                    greeks = self.calculate_greeks(current_price, K, T, risk_free_rate, 0.2, option_type)
                    
                    # Calculate implied volatility
                    iv = self.implied_volatility(market_price, current_price, K, T, risk_free_rate, option_type)
                    
                    theoretical_prices.append(theoretical_price)
                    deltas.append(greeks['delta'])
                    gammas.append(greeks['gamma'])
                    thetas.append(greeks['theta'])
                    vegas.append(greeks['vega'])
                    rhos.append(greeks['rho'])
                    implied_vols.append(iv)
                else:
                    # Handle expired or invalid options
                    theoretical_prices.append(0)
                    deltas.append(0)
                    gammas.append(0)
                    thetas.append(0)
                    vegas.append(0)
                    rhos.append(0)
                    implied_vols.append(0)
                    
            except Exception as e:
                # Handle any errors in calculations
                theoretical_prices.append(0)
                deltas.append(0)
                gammas.append(0)
                thetas.append(0)
                vegas.append(0)
                rhos.append(0)
                implied_vols.append(0)
        
        # Add calculated values to DataFrame
        priced_options['theoretical_price'] = theoretical_prices
        priced_options['delta'] = deltas
        priced_options['gamma'] = gammas
        priced_options['theta'] = thetas
        priced_options['vega'] = vegas
        priced_options['rho'] = rhos
        priced_options['calculated_iv'] = implied_vols
        
        return priced_options

def demonstrate_pricing_models():
    """
    Demonstration of different pricing models
    """
    # Example parameters
    S = 100  # Current stock price
    K = 105  # Strike price
    T = 0.25  # 3 months to expiration
    r = 0.05  # 5% risk-free rate
    sigma = 0.2  # 20% volatility
    
    pricer = OptionsPricingModels(r)
    
    print("Options Pricing Model Comparison")
    print("=" * 50)
    print(f"Stock Price: ${S}")
    print(f"Strike Price: ${K}")
    print(f"Time to Expiration: {T} years")
    print(f"Risk-free Rate: {r*100}%")
    print(f"Volatility: {sigma*100}%")
    print()
    
    # Call option prices
    print("CALL OPTION PRICES:")
    bs_call = pricer.black_scholes_call(S, K, T, r, sigma)
    binomial_call = pricer.binomial_option_price(S, K, T, r, sigma, option_type='call')
    mc_call = pricer.monte_carlo_option_price(S, K, T, r, sigma, option_type='call')
    
    print(f"Black-Scholes: ${bs_call:.4f}")
    print(f"Binomial Tree: ${binomial_call:.4f}")
    print(f"Monte Carlo:   ${mc_call:.4f}")
    
    # Put option prices
    print("\nPUT OPTION PRICES:")
    bs_put = pricer.black_scholes_put(S, K, T, r, sigma)
    binomial_put = pricer.binomial_option_price(S, K, T, r, sigma, option_type='put')
    mc_put = pricer.monte_carlo_option_price(S, K, T, r, sigma, option_type='put')
    
    print(f"Black-Scholes: ${bs_put:.4f}")
    print(f"Binomial Tree: ${binomial_put:.4f}")
    print(f"Monte Carlo:   ${mc_put:.4f}")
    
    # Greeks
    print("\nGREEKS (Call Option):")
    greeks = pricer.calculate_greeks(S, K, T, r, sigma, 'call')
    for greek, value in greeks.items():
        print(f"{greek.capitalize()}: {value:.4f}")

if __name__ == "__main__":
    demonstrate_pricing_models()
