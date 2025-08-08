"""
Enhanced Command-line CME Options Pricer with Theoretical Pricing
Features: Multiple pricing models, Greeks calculation, mispricing analysis
"""

import yfinance as yf
import pandas as pd
import argparse
from datetime import datetime
import sys
import numpy as np
from options_pricing import OptionsPricingModels
from risk_free_rates import get_risk_free_rates

def format_cli_display(df, max_decimals=4):
    """Format dataframe for CLI display with controlled decimal places"""
    formatted_df = df.copy()
    
    # Define columns that should use specific formatting
    percentage_columns = ['impliedVolatility', 'price_diff_pct']
    currency_columns = ['strike', 'lastPrice', 'bid', 'ask', 'theoretical_price', 'price_diff']
    decimal_columns = ['delta', 'gamma', 'theta', 'vega', 'rho', 'calculated_iv']
    integer_columns = ['volume', 'openInterest']
    
    for col in formatted_df.columns:
        if formatted_df[col].dtype in ['float64', 'float32']:
            if col in percentage_columns:
                # Format as percentage with 2 decimal places
                formatted_df[col] = formatted_df[col].apply(
                    lambda x: f"{x*100:.2f}%" if pd.notna(x) else "N/A"
                )
            elif col in currency_columns:
                # Format as currency with 2 decimal places
                formatted_df[col] = formatted_df[col].apply(
                    lambda x: f"${x:.2f}" if pd.notna(x) else "N/A"
                )
            elif col in decimal_columns:
                # Format with specified decimal places for Greeks
                formatted_df[col] = formatted_df[col].apply(
                    lambda x: f"{x:.{max_decimals}f}" if pd.notna(x) else "N/A"
                )
            else:
                # Format other numeric columns with max_decimals
                formatted_df[col] = formatted_df[col].apply(
                    lambda x: f"{x:.{max_decimals}f}" if pd.notna(x) and abs(x) < 1e6 else (f"{x:.2e}" if pd.notna(x) else "N/A")
                )
        elif formatted_df[col].dtype in ['int64', 'int32']:
            if col in integer_columns:
                # Format volume and open interest with comma separators
                formatted_df[col] = formatted_df[col].apply(
                    lambda x: f"{x:,}" if pd.notna(x) else "N/A"
                )
            else:
                # Keep other integers as is
                formatted_df[col] = formatted_df[col].apply(
                    lambda x: f"{x}" if pd.notna(x) else "N/A"
                )
    
    return formatted_df

def fetch_options_data(symbol, expiration_date=None, enable_pricing=True, pricing_model="Black-Scholes", risk_free_rate=0.05):
    """Fetch and display options data with optional theoretical pricing"""
    try:
        ticker = yf.Ticker(symbol)
        
        # Get basic info
        info = ticker.info
        current_price = info.get('currentPrice', info.get('regularMarketPrice', 0))
        
        print(f"\n{'='*70}")
        print(f"CME OPTIONS PRICER - {symbol.upper()}")
        print(f"{'='*70}")
        print(f"Current Stock Price: ${current_price:.2f}")
        
        if enable_pricing:
            print(f"Pricing Model: {pricing_model}")
            print(f"Risk-free Rate: {risk_free_rate*100:.1f}%")
        
        # Get available expiration dates
        exp_dates = ticker.options
        if not exp_dates:
            print(f"No options data available for {symbol}")
            return
        
        print(f"Available expiration dates: {', '.join(exp_dates[:5])}")
        
        # Use specified expiration or first available
        target_exp = expiration_date if expiration_date in exp_dates else exp_dates[0]
        print(f"Analyzing expiration: {target_exp}")
        
        # Get options chain
        option_chain = ticker.option_chain(target_exp)
        
        # Initialize pricer if enabled
        pricer = None
        if enable_pricing:
            pricer = OptionsPricingModels(risk_free_rate)
        
        # Process and display calls
        print(f"\n{'CALL OPTIONS':^120}")
        print('=' * 120)
        calls = option_chain.calls[['strike', 'lastPrice', 'bid', 'ask', 'volume', 'openInterest', 'impliedVolatility']].head(10)
        
        if enable_pricing and pricer:
            calls_with_pricing = add_theoretical_pricing(calls, current_price, target_exp, 'call', pricer, pricing_model, risk_free_rate)
            calls_formatted = format_cli_display(calls_with_pricing)
            print(calls_formatted.to_string(index=False))
        else:
            calls_formatted = format_cli_display(calls)
            print(calls_formatted.to_string(index=False))
        
        # Process and display puts
        print(f"\n{'PUT OPTIONS':^120}")
        print('=' * 120)
        puts = option_chain.puts[['strike', 'lastPrice', 'bid', 'ask', 'volume', 'openInterest', 'impliedVolatility']].head(10)
        
        if enable_pricing and pricer:
            puts_with_pricing = add_theoretical_pricing(puts, current_price, target_exp, 'put', pricer, pricing_model, risk_free_rate)
            puts_formatted = format_cli_display(puts_with_pricing)
            print(puts_formatted.to_string(index=False))
        else:
            puts_formatted = format_cli_display(puts)
            print(puts_formatted.to_string(index=False))
        
        # Enhanced summary statistics
        print(f"\n{'MARKET SUMMARY':^70}")
        print('=' * 70)
        print(f"Total Call Volume: {calls['volume'].sum():,.0f}")
        print(f"Total Put Volume: {puts['volume'].sum():,.0f}")
        print(f"Call/Put Volume Ratio: {calls['volume'].sum() / max(puts['volume'].sum(), 1):.2f}")
        
        # ATM analysis
        analyze_atm_options(calls, puts, current_price, enable_pricing, pricer, target_exp, pricing_model, risk_free_rate)
        
        # Mispricing analysis
        if enable_pricing:
            analyze_mispricing(calls, puts, current_price, target_exp, pricer, pricing_model, risk_free_rate)
        
    except Exception as e:
        print(f"Error fetching options data: {str(e)}")

def add_theoretical_pricing(options_df, current_price, expiration_date, option_type, pricer, pricing_model, risk_free_rate):
    """Add theoretical pricing columns to options DataFrame"""
    df = options_df.copy()
    
    theoretical_prices = []
    deltas = []
    gammas = []
    thetas = []
    price_diffs = []
    
    for _, row in df.iterrows():
        try:
            T = pricer.calculate_time_to_expiration(expiration_date)
            K = row['strike']
            
            if T > 0:
                # Calculate theoretical price based on model
                if pricing_model == "Black-Scholes":
                    if option_type == 'call':
                        theo_price = pricer.black_scholes_call(current_price, K, T, risk_free_rate, 0.2)
                    else:
                        theo_price = pricer.black_scholes_put(current_price, K, T, risk_free_rate, 0.2)
                elif pricing_model == "Binomial":
                    theo_price = pricer.binomial_option_price(current_price, K, T, risk_free_rate, 0.2, option_type=option_type, american=True)
                elif pricing_model == "Monte Carlo":
                    theo_price = pricer.monte_carlo_option_price(current_price, K, T, risk_free_rate, 0.2, option_type=option_type)
                else:
                    theo_price = 0
                
                # Calculate Greeks
                greeks = pricer.calculate_greeks(current_price, K, T, risk_free_rate, 0.2, option_type)
                
                theoretical_prices.append(theo_price)
                deltas.append(greeks['delta'])
                gammas.append(greeks['gamma'])
                thetas.append(greeks['theta'])
                price_diffs.append(row['lastPrice'] - theo_price)
            else:
                theoretical_prices.append(0)
                deltas.append(0)
                gammas.append(0)
                thetas.append(0)
                price_diffs.append(0)
        except:
            theoretical_prices.append(0)
            deltas.append(0)
            gammas.append(0)
            thetas.append(0)
            price_diffs.append(0)
    
    # Add new columns
    df['theoretical'] = theoretical_prices
    df['delta'] = deltas
    df['gamma'] = gammas
    df['theta'] = thetas
    df['mkt_vs_theo'] = price_diffs
    
    return df

def analyze_atm_options(calls, puts, current_price, enable_pricing, pricer, expiration_date, pricing_model, risk_free_rate):
    """Analyze at-the-money options"""
    print(f"\n{'ATM ANALYSIS':^70}")
    print('-' * 70)
    
    # Find ATM options (within 2% of current price)
    atm_calls = calls[abs(calls['strike'] - current_price) <= current_price * 0.02]
    atm_puts = puts[abs(puts['strike'] - current_price) <= current_price * 0.02]
    
    if not atm_calls.empty:
        print(f"ATM Call Strike: ${atm_calls['strike'].iloc[0]:.0f}")
        print(f"ATM Call Market Price: ${atm_calls['lastPrice'].iloc[0]:.4f}")
        print(f"ATM Call Implied Vol: {atm_calls['impliedVolatility'].iloc[0]:.2%}")
        
        if enable_pricing and pricer:
            T = pricer.calculate_time_to_expiration(expiration_date)
            if T > 0:
                theo_price = pricer.black_scholes_call(current_price, atm_calls['strike'].iloc[0], T, risk_free_rate, 0.2)
                print(f"ATM Call Theoretical ({pricing_model}): ${theo_price:.4f}")
                print(f"ATM Call Price Difference: ${atm_calls['lastPrice'].iloc[0] - theo_price:.4f}")
    
    if not atm_puts.empty:
        print(f"ATM Put Strike: ${atm_puts['strike'].iloc[0]:.0f}")
        print(f"ATM Put Market Price: ${atm_puts['lastPrice'].iloc[0]:.4f}")
        print(f"ATM Put Implied Vol: {atm_puts['impliedVolatility'].iloc[0]:.2%}")
        
        if enable_pricing and pricer:
            T = pricer.calculate_time_to_expiration(expiration_date)
            if T > 0:
                theo_price = pricer.black_scholes_put(current_price, atm_puts['strike'].iloc[0], T, risk_free_rate, 0.2)
                print(f"ATM Put Theoretical ({pricing_model}): ${theo_price:.4f}")
                print(f"ATM Put Price Difference: ${atm_puts['lastPrice'].iloc[0] - theo_price:.4f}")

def analyze_mispricing(calls, puts, current_price, expiration_date, pricer, pricing_model, risk_free_rate):
    """Analyze potential mispricings"""
    print(f"\n{'MISPRICING ANALYSIS':^70}")
    print('-' * 70)
    
    # Add theoretical pricing to both calls and puts
    calls_priced = add_theoretical_pricing(calls, current_price, expiration_date, 'call', pricer, pricing_model, risk_free_rate)
    puts_priced = add_theoretical_pricing(puts, current_price, expiration_date, 'put', pricer, pricing_model, risk_free_rate)
    
    # Find significant mispricings (>10% difference)
    threshold = 0.1
    
    mispriced_calls = calls_priced[
        (calls_priced['theoretical'] > 0) & 
        (abs(calls_priced['mkt_vs_theo'] / calls_priced['theoretical']) > threshold)
    ]
    
    mispriced_puts = puts_priced[
        (puts_priced['theoretical'] > 0) & 
        (abs(puts_priced['mkt_vs_theo'] / puts_priced['theoretical']) > threshold)
    ]
    
    if not mispriced_calls.empty:
        print("🔴 POTENTIALLY MISPRICED CALLS (>10% difference):")
        for _, row in mispriced_calls.iterrows():
            pct_diff = (row['mkt_vs_theo'] / row['theoretical']) * 100
            status = "OVERPRICED" if pct_diff > 0 else "UNDERPRICED"
            print(f"  ${row['strike']:3.0f} Strike: Market ${row['lastPrice']:.4f} vs Theory ${row['theoretical']:.4f} ({pct_diff:+.1f}% {status})")
    
    if not mispriced_puts.empty:
        print("🔴 POTENTIALLY MISPRICED PUTS (>10% difference):")
        for _, row in mispriced_puts.iterrows():
            pct_diff = (row['mkt_vs_theo'] / row['theoretical']) * 100
            status = "OVERPRICED" if pct_diff > 0 else "UNDERPRICED"
            print(f"  ${row['strike']:3.0f} Strike: Market ${row['lastPrice']:.4f} vs Theory ${row['theoretical']:.4f} ({pct_diff:+.1f}% {status})")
    
    if mispriced_calls.empty and mispriced_puts.empty:
        print("✅ No significant mispricings detected (using 10% threshold)")
    
    # Portfolio Greeks summary
    if not calls_priced.empty or not puts_priced.empty:
        print(f"\n{'PORTFOLIO GREEKS (1 contract each)':^70}")
        print('-' * 70)
        
        total_delta = calls_priced['delta'].sum() + puts_priced['delta'].sum()
        total_gamma = calls_priced['gamma'].sum() + puts_priced['gamma'].sum()
        total_theta = calls_priced['theta'].sum() + puts_priced['theta'].sum()
        
        print(f"Total Portfolio Delta: {total_delta:.4f}")
        print(f"Total Portfolio Gamma: {total_gamma:.4f}")
        print(f"Total Portfolio Theta: {total_theta:.4f} (daily decay)")

def show_risk_free_rate_suggestions():
    """Display current risk-free rate suggestions"""
    print(f"\n{'CURRENT RISK-FREE RATES':^70}")
    print('=' * 70)
    
    try:
        rates_provider = get_risk_free_rates()
        rates_provider.fetch_all_rates()
        
        # Get suggestions for USD (most common for equity options)
        suggestions = rates_provider.get_rate_suggestions('USD')
        
        if suggestions:
            print("💡 Live market rates (you can use these for more accurate pricing):")
            print()
            for i, suggestion in enumerate(suggestions, 1):
                print(f"{i}. {suggestion['rate_percent']:.2f}% - {suggestion['description']}")
                print(f"   Source: {suggestion['source']}")
                print()
        else:
            print("⚠️  Could not fetch live rates. Using fallback rates:")
            fallback = rates_provider.get_fallback_rates()
            for currency, rates in fallback.items():
                print(f"{currency} rates:")
                for tenor, rate in rates.items():
                    print(f"  {tenor}: {rate:.2f}%")
                print()
        
        # Show rate info
        rate_info = rates_provider.get_rate_info()
        if rate_info['fred_api_configured']:
            print("✅ FRED API configured for live US rates")
        else:
            print("⚠️  FRED API not configured - using fallback rates")
            print("   Set FRED_API_KEY environment variable for live SOFR rates")
        
        print(f"Last updated: {rate_info['last_updated'] or 'Never'}")
        
    except Exception as e:
        print(f"❌ Error fetching rates: {str(e)}")
        print("Using default 5% rate")
    
    print('=' * 70)

def main():
    parser = argparse.ArgumentParser(description='Enhanced CME Options Pricer with Theoretical Pricing')
    parser.add_argument('symbol', nargs='?', help='Stock symbol (e.g., SPY, AAPL, TSLA)')
    parser.add_argument('--expiration', '-e', help='Expiration date (YYYY-MM-DD format)')
    parser.add_argument('--list', '-l', action='store_true', help='List available expiration dates only')
    parser.add_argument('--rates', action='store_true', help='Show current risk-free rate suggestions')
    parser.add_argument('--no-pricing', action='store_true', help='Disable theoretical pricing calculations')
    parser.add_argument('--model', '-m', choices=['Black-Scholes', 'Binomial', 'Monte Carlo'], 
                       default='Black-Scholes', help='Pricing model (default: Black-Scholes)')
    parser.add_argument('--rate', '-r', type=float, default=5.0, 
                       help='Risk-free rate as percentage (default: 5.0%%)')
    
    args = parser.parse_args()
    
    # Handle rates command
    if args.rates:
        show_risk_free_rate_suggestions()
        return
    
    # Require symbol for other operations
    if not args.symbol:
        parser.print_help()
        print("\nError: symbol is required unless using --rates")
        return
    
    symbol = args.symbol.upper()
    risk_free_rate = args.rate / 100  # Convert percentage to decimal
    
    if args.list:
        # List available expiration dates
        try:
            ticker = yf.Ticker(symbol)
            exp_dates = ticker.options
            print(f"\nAvailable expiration dates for {symbol}:")
            print("=" * 40)
            for i, date in enumerate(exp_dates, 1):
                print(f"{i:2d}. {date}")
            print(f"\nTotal: {len(exp_dates)} expiration dates available")
        except Exception as e:
            print(f"Error: {str(e)}")
        return
    
    enable_pricing = not args.no_pricing
    pricing_model = args.model
    
    print(f"\nCME Options Pricer - Command Line Interface")
    print(f"Symbol: {symbol}")
    if enable_pricing:
        print(f"Theoretical Pricing: {pricing_model} model @ {args.rate}% risk-free rate")
        
        # Show brief rate suggestions unless user is using default rate
        if args.rate == 5.0:  # Default rate
            print("\n💡 Tip: Run 'python cli_pricer.py --rates' to see current market rates")
    else:
        print("Theoretical Pricing: Disabled")
    
    fetch_options_data(symbol, args.expiration, enable_pricing, pricing_model, risk_free_rate)

if __name__ == "__main__":
    main()
