"""
Command Line Interface for Options Pricing
"""
import sys
import os
import argparse
from datetime import datetime

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.models.pricing import OptionsPricingEngine
from src.data.rates import get_risk_free_rates
from src.data.providers import MarketDataProvider
from src.utils.formatters import CLIFormatter
from src.utils.validators import DataValidator

def main():
    """Main CLI function"""
    parser = argparse.ArgumentParser(description='CME Equity Options Pricer - CLI')
    parser.add_argument('symbol', help='Stock symbol (e.g., AAPL)')
    parser.add_argument('--strike', type=float, help='Strike price')
    parser.add_argument('--expiry', help='Expiry date (YYYY-MM-DD)')
    parser.add_argument('--option-type', choices=['call', 'put'], default='call', help='Option type')
    parser.add_argument('--rate', type=float, default=0.05, help='Risk-free rate (default: 0.05)')
    parser.add_argument('--volatility', type=float, default=0.25, help='Volatility (default: 0.25)')
    parser.add_argument('--price-only', action='store_true', help='Show only theoretical price')
    parser.add_argument('--show-chain', action='store_true', help='Show entire options chain')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not DataValidator.validate_symbol(args.symbol):
        print(f"Error: Invalid symbol '{args.symbol}'")
        return 1
    
    if not DataValidator.validate_rate(args.rate):
        print(f"Error: Invalid risk-free rate '{args.rate}'")
        return 1
    
    if not DataValidator.validate_volatility(args.volatility):
        print(f"Error: Invalid volatility '{args.volatility}'")
        return 1
    
    # Initialize providers
    print(f"Fetching data for {args.symbol}...")
    market_provider = MarketDataProvider()
    pricing_engine = OptionsPricingEngine(args.rate)
    rates_provider = get_risk_free_rates()
    
    try:
        # Get market data
        market_data = market_provider.get_complete_market_data(args.symbol)
        
        if market_data['error']:
            print(f"Error fetching data: {market_data['error']}")
            return 1
        
        stock_info = market_data['stock_info']
        current_price = market_data['current_price']
        
        if not stock_info or not current_price:
            print(f"Error: Could not fetch data for {args.symbol}")
            return 1
        
        print(f"\n{stock_info['longName']} ({args.symbol})")
        print(f"Current Price: ${current_price:.2f}")
        print(f"Risk-free Rate: {args.rate*100:.2f}%")
        print(f"Volatility: {args.volatility*100:.2f}%")
        print("-" * 50)
        
        if args.show_chain:
            # Show entire options chain
            options_df = market_data['options_data']
            if options_df is not None and not options_df.empty:
                print(f"\nOptions Chain for {args.symbol}:")
                print(CLIFormatter.format_cli_table(options_df.head(20)))  # Show first 20
            else:
                print("No options data available")
        
        elif args.strike and args.expiry:
            # Calculate specific option price
            validation = DataValidator.validate_options_input(
                args.symbol, args.strike, current_price, args.expiry, args.rate, args.volatility
            )
            
            if not validation['valid']:
                for error in validation['errors']:
                    print(f"Error: {error}")
                return 1
            
            time_to_expiry = validation['time_to_expiry']
            
            if args.option_type == 'call':
                theoretical_price = pricing_engine.black_scholes_call(
                    current_price, args.strike, time_to_expiry, args.rate, args.volatility
                )
            else:
                theoretical_price = pricing_engine.black_scholes_put(
                    current_price, args.strike, time_to_expiry, args.rate, args.volatility
                )
            
            if args.price_only:
                print(f"{theoretical_price:.4f}")
            else:
                print(f"\n{args.option_type.title()} Option Pricing:")
                print(f"Strike: ${args.strike:.2f}")
                print(f"Expiry: {args.expiry}")
                print(f"Time to Expiry: {time_to_expiry:.4f} years")
                print(f"Theoretical Price: ${theoretical_price:.4f}")
                
                # Calculate Greeks
                greeks = pricing_engine.calculate_greeks(
                    current_price, args.strike, time_to_expiry, args.rate, args.volatility, args.option_type
                )
                
                print(f"\nGreeks:")
                print(f"Delta: {greeks['delta']:.4f}")
                print(f"Gamma: {greeks['gamma']:.4f}")
                print(f"Theta: {greeks['theta']:.4f}")
                print(f"Vega: {greeks['vega']:.4f}")
                print(f"Rho: {greeks['rho']:.4f}")
        
        else:
            print("\nUse --show-chain to see options chain, or specify --strike and --expiry for pricing")
            print("Example: python cli.py AAPL --strike 150 --expiry 2024-12-20 --option-type call")
    
    except Exception as e:
        print(f"Error: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
