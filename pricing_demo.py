"""
Demonstration script for different options pricing models
Run this to see benchmark pricing techniques in action
"""

from options_pricing import OptionsPricingModels, demonstrate_pricing_models
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def compare_pricing_models():
    """
    Compare different pricing models for various option scenarios
    """
    print("="*60)
    print("CME EQUITY OPTIONS PRICING - BENCHMARK COMPARISON")
    print("="*60)
    
    # Initialize pricer
    pricer = OptionsPricingModels(risk_free_rate=0.05)
    
    # Test scenarios
    scenarios = [
        {"name": "At-The-Money Call", "S": 100, "K": 100, "T": 0.25, "sigma": 0.2, "type": "call"},
        {"name": "Out-of-Money Call", "S": 100, "K": 110, "T": 0.25, "sigma": 0.2, "type": "call"},
        {"name": "In-The-Money Put", "S": 100, "K": 110, "T": 0.25, "sigma": 0.2, "type": "put"},
        {"name": "High Volatility Option", "S": 100, "K": 105, "T": 0.1, "sigma": 0.4, "type": "call"},
        {"name": "Long-term Option", "S": 100, "K": 105, "T": 1.0, "sigma": 0.25, "type": "call"},
    ]
    
    results = []
    
    for scenario in scenarios:
        print(f"\nScenario: {scenario['name']}")
        print("-" * 40)
        print(f"Stock: ${scenario['S']}, Strike: ${scenario['K']}, Time: {scenario['T']} years, Vol: {scenario['sigma']*100}%")
        
        # Black-Scholes
        if scenario['type'] == 'call':
            bs_price = pricer.black_scholes_call(scenario['S'], scenario['K'], scenario['T'], 
                                               pricer.risk_free_rate, scenario['sigma'])
        else:
            bs_price = pricer.black_scholes_put(scenario['S'], scenario['K'], scenario['T'], 
                                              pricer.risk_free_rate, scenario['sigma'])
        
        # Binomial Tree
        binomial_price = pricer.binomial_option_price(scenario['S'], scenario['K'], scenario['T'],
                                                    pricer.risk_free_rate, scenario['sigma'], 
                                                    n=100, option_type=scenario['type'])
        
        # Monte Carlo
        mc_price = pricer.monte_carlo_option_price(scenario['S'], scenario['K'], scenario['T'],
                                                 pricer.risk_free_rate, scenario['sigma'],
                                                 n_simulations=10000, option_type=scenario['type'])
        
        # Calculate Greeks
        greeks = pricer.calculate_greeks(scenario['S'], scenario['K'], scenario['T'],
                                       pricer.risk_free_rate, scenario['sigma'], scenario['type'])
        
        print(f"Black-Scholes:  ${bs_price:.4f}")
        print(f"Binomial Tree:  ${binomial_price:.4f}")
        print(f"Monte Carlo:    ${mc_price:.4f}")
        print(f"Delta: {greeks['delta']:.4f}, Gamma: {greeks['gamma']:.4f}, Theta: {greeks['theta']:.4f}")
        
        results.append({
            'Scenario': scenario['name'],
            'Black-Scholes': bs_price,
            'Binomial': binomial_price,
            'Monte Carlo': mc_price,
            'Delta': greeks['delta'],
            'Gamma': greeks['gamma'],
            'Theta': greeks['theta'],
            'Vega': greeks['vega']
        })
    
    # Create summary DataFrame
    df_results = pd.DataFrame(results)
    print(f"\n{'SUMMARY TABLE':^60}")
    print("=" * 60)
    print(df_results.to_string(index=False, float_format='%.4f'))
    
    return df_results

def implied_volatility_demo():
    """
    Demonstrate implied volatility calculation
    """
    print(f"\n{'IMPLIED VOLATILITY DEMONSTRATION':^60}")
    print("=" * 60)
    
    pricer = OptionsPricingModels()
    
    # Market prices (simulated)
    market_scenarios = [
        {"market_price": 5.50, "S": 100, "K": 105, "T": 0.25, "type": "call"},
        {"market_price": 2.30, "S": 100, "K": 95, "T": 0.25, "type": "put"},
        {"market_price": 8.75, "S": 100, "K": 100, "T": 0.5, "type": "call"},
    ]
    
    for i, scenario in enumerate(market_scenarios, 1):
        print(f"\nScenario {i}:")
        print(f"Market Price: ${scenario['market_price']:.2f}")
        print(f"Stock: ${scenario['S']}, Strike: ${scenario['K']}, Time: {scenario['T']} years")
        
        # Calculate implied volatility
        iv = pricer.implied_volatility(scenario['market_price'], scenario['S'], scenario['K'],
                                     scenario['T'], pricer.risk_free_rate, scenario['type'])
        
        print(f"Implied Volatility: {iv*100:.2f}%")
        
        # Verify by calculating theoretical price with this IV
        if scenario['type'] == 'call':
            theoretical = pricer.black_scholes_call(scenario['S'], scenario['K'], scenario['T'],
                                                  pricer.risk_free_rate, iv)
        else:
            theoretical = pricer.black_scholes_put(scenario['S'], scenario['K'], scenario['T'],
                                                 pricer.risk_free_rate, iv)
        
        print(f"Verification - Theoretical Price: ${theoretical:.4f}")
        print(f"Difference: ${abs(theoretical - scenario['market_price']):.6f}")

def american_vs_european_demo():
    """
    Compare American vs European option pricing using binomial model
    """
    print(f"\n{'AMERICAN VS EUROPEAN OPTIONS':^60}")
    print("=" * 60)
    
    pricer = OptionsPricingModels()
    
    # Deep in-the-money put (likely to be exercised early)
    S, K, T, sigma = 80, 100, 0.5, 0.3
    
    print(f"Deep ITM Put Example:")
    print(f"Stock: ${S}, Strike: ${K}, Time: {T} years, Vol: {sigma*100}%")
    
    # European put
    european_put = pricer.binomial_option_price(S, K, T, pricer.risk_free_rate, sigma,
                                              n=100, option_type='put', american=False)
    
    # American put
    american_put = pricer.binomial_option_price(S, K, T, pricer.risk_free_rate, sigma,
                                              n=100, option_type='put', american=True)
    
    # Intrinsic value
    intrinsic = max(0, K - S)
    
    print(f"European Put: ${european_put:.4f}")
    print(f"American Put: ${american_put:.4f}")
    print(f"Intrinsic Value: ${intrinsic:.4f}")
    print(f"Early Exercise Premium: ${american_put - european_put:.4f}")
    print(f"Time Value (European): ${european_put - intrinsic:.4f}")

if __name__ == "__main__":
    # Run all demonstrations
    demonstrate_pricing_models()
    print("\n" + "="*60 + "\n")
    
    compare_pricing_models()
    print("\n" + "="*60 + "\n")
    
    implied_volatility_demo()
    print("\n" + "="*60 + "\n")
    
    american_vs_european_demo()
    
    print(f"\n{'BENCHMARK TECHNIQUES SUMMARY':^60}")
    print("=" * 60)
    print("""
1. BLACK-SCHOLES-MERTON MODEL:
   - Best for: European options, liquid markets
   - Assumptions: Constant volatility, no dividends, efficient markets
   - Use case: Quick theoretical pricing, Greeks calculation

2. BINOMIAL TREE MODEL:
   - Best for: American options, dividend-paying stocks
   - Advantages: Handles early exercise, path-dependent features
   - Use case: American options, complex payoffs

3. MONTE CARLO SIMULATION:
   - Best for: Complex derivatives, path-dependent options
   - Advantages: Handles any payoff structure, multiple factors
   - Use case: Exotic options, risk management

4. IMPLIED VOLATILITY:
   - Best for: Market sentiment analysis, relative value
   - Use case: Trading decisions, volatility surface construction

5. BLACK-76 MODEL:
   - Best for: Futures options (CME products)
   - Use case: Commodity options, interest rate derivatives
   
RECOMMENDED APPROACH FOR CME OPTIONS:
- Use Black-Scholes for quick estimates
- Use Binomial for American-style options
- Calculate implied volatility for market analysis
- Use Greeks for risk management
""")
