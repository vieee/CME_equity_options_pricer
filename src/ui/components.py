"""
Streamlit UI Components for Options Pricing Application
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, Optional, List, Any
from datetime import datetime
import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.utils.formatters import DataFormatter
from src.utils.validators import DataValidator
from src.models.pricing import OptionsPricingEngine
from src.data.rates import RiskFreeRatesProvider
from src.data.providers import MarketDataProvider

class StreamlitComponents:
    """Reusable Streamlit UI components"""
    
    @staticmethod
    def render_header():
        """Render application header"""
        st.title("📈 CME Equity Options Pricer")
        st.markdown("Real-time equity options data and advanced pricing models")
        st.markdown("---")
    
    @staticmethod
    def render_sidebar(rates_provider: RiskFreeRatesProvider) -> Dict[str, Any]:
        """
        Render sidebar with input parameters
        
        Returns:
            Dictionary with user inputs
        """
        st.sidebar.header("📊 Options Parameters")
        
        # Popular symbols
        popular_symbols = ['SPY', 'QQQ', 'IWM', 'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META']
        
        symbol = st.sidebar.selectbox(
            "Select Stock Symbol",
            options=popular_symbols,
            index=0
        )
        
        custom_symbol = st.sidebar.text_input("Or enter custom symbol:")
        if custom_symbol:
            symbol = custom_symbol.upper()
        
        # Risk-free rates section
        st.sidebar.header("📊 Risk-Free Rates")
        
        if st.sidebar.button("🔄 Refresh Rates", help="Fetch latest rates from APIs"):
            with st.spinner("Fetching latest rates..."):
                rates_provider.fetch_all_rates()
            st.sidebar.success("Rates updated!")
        
        # Display current rates summary
        with st.sidebar.expander("Current Rates Overview", expanded=False):
            rates_info = rates_provider.get_rate_info()
            
            if rates_info['last_updated']:
                st.write(f"**Last Updated:** {rates_info['last_updated'][:19]}")
            
            current_rates = rates_provider.current_rates
            if current_rates:
                st.write("**Overnight Rates:**")
                for currency, rates in current_rates.items():
                    overnight_rate = rates.get('overnight')
                    if overnight_rate is not None:
                        st.write(f"• {currency}: {overnight_rate:.2f}%")
            
            if rates_info['fred_api_configured']:
                st.success("✅ FRED API configured")
            else:
                st.warning("⚠️ FRED API key not configured")
                st.info("Set FRED_API_KEY environment variable for US rates")
        
        # API configuration help
        with st.sidebar.expander("API Configuration", expanded=False):
            st.write("**FRED API Key (Optional):**")
            st.write("For live US rates (SOFR), get a free API key from:")
            st.link_button("Get FRED API Key", "https://fred.stlouisfed.org/docs/api/api_key.html")
            
            st.write("**Supported Rates:**")
            st.write("• **SOFR** (USD) - US risk-free rate")
            st.write("• **ESTR** (EUR) - Euro risk-free rate") 
            st.write("• **SONIA** (GBP) - UK risk-free rate")
            st.write("• **EONIA** (EUR) - Legacy rate (ESTR-based)")
            
            st.caption("Note: Fallback rates are used when APIs are unavailable")
        
        return {'symbol': symbol}
    
    @staticmethod
    def render_stock_metrics(stock_info: Dict) -> None:
        """Render stock metrics in 4 columns"""
        if not stock_info:
            st.info("💡 Enter a symbol and the stock information will appear here")
            return
        
        st.subheader(f"📊 {stock_info['symbol']} - Stock Overview")
        
        # Create 4 columns for stock metrics
        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
        
        with metric_col1:
            st.metric(
                label="💰 Current Price", 
                value=DataFormatter.format_currency(stock_info['currentPrice']),
                help="Real-time stock price"
            )
        
        with metric_col2:
            # Calculate change from previous close
            if stock_info['previousClose'] > 0:
                price_change = stock_info['currentPrice'] - stock_info['previousClose']
                price_change_pct = (price_change / stock_info['previousClose']) * 100
                st.metric(
                    label="📈 Daily Change", 
                    value=DataFormatter.format_currency(price_change),
                    delta=f"{price_change_pct:.2f}%",
                    help="Change from previous close"
                )
            else:
                st.metric("📈 Daily Change", "N/A")
        
        with metric_col3:
            # Market Cap
            if stock_info['marketCap'] > 0:
                st.metric(
                    label="🏢 Market Cap", 
                    value=DataFormatter.format_large_number(stock_info['marketCap']),
                    help="Total market value"
                )
            else:
                st.metric("🏢 Market Cap", "N/A")
        
        with metric_col4:
            # Volume
            if stock_info['volume'] > 0:
                st.metric(
                    label="📊 Volume", 
                    value=DataFormatter.format_large_number(stock_info['volume']),
                    help="Trading volume today"
                )
            else:
                st.metric("📊 Volume", "N/A")
    
    @staticmethod
    def render_options_filters(options_df: pd.DataFrame, current_price: float) -> pd.DataFrame:
        """Render options filtering controls and return filtered dataframe"""
        st.subheader("🔍 Filter Options")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            option_type = st.selectbox("Option Type", ["All", "call", "put"])
        
        with col2:
            expiration_dates = sorted(options_df['expiration'].unique())
            selected_expiration = st.selectbox("Expiration Date", ["All"] + expiration_dates)
        
        with col3:
            moneyness = st.selectbox("Moneyness", ["All", "ITM", "ATM", "OTM"])
        
        # Apply filters
        filtered_df = options_df.copy()
        
        if option_type != "All":
            filtered_df = filtered_df[filtered_df['option_type'] == option_type]
        
        if selected_expiration != "All":
            filtered_df = filtered_df[filtered_df['expiration'] == selected_expiration]
        
        if moneyness != "All":
            if moneyness == "ITM":
                filtered_df = filtered_df[
                    ((filtered_df['option_type'] == 'call') & (filtered_df['strike'] < current_price)) |
                    ((filtered_df['option_type'] == 'put') & (filtered_df['strike'] > current_price))
                ]
            elif moneyness == "ATM":
                tolerance = current_price * 0.02  # 2% tolerance
                filtered_df = filtered_df[
                    abs(filtered_df['strike'] - current_price) <= tolerance
                ]
            elif moneyness == "OTM":
                filtered_df = filtered_df[
                    ((filtered_df['option_type'] == 'call') & (filtered_df['strike'] > current_price)) |
                    ((filtered_df['option_type'] == 'put') & (filtered_df['strike'] < current_price))
                ]
        
        # Show active filters status
        active_filters = []
        if option_type != "All":
            active_filters.append(f"Type: {option_type}")
        if selected_expiration != "All":
            active_filters.append(f"Expiry: {selected_expiration}")
        if moneyness != "All":
            active_filters.append(f"Moneyness: {moneyness}")
        
        if active_filters:
            st.info(f"🔍 Active Filters: {' | '.join(active_filters)} | Showing {len(filtered_df)} options")
        else:
            st.info(f"📊 Showing all {len(filtered_df)} options")
        
        return filtered_df
    
    @staticmethod
    def render_pricing_controls(rates_provider: RiskFreeRatesProvider, 
                              options_df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """Render pricing analysis controls"""
        st.subheader("🧮 Options Pricing Analysis")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            enable_pricing = st.checkbox("Enable Theoretical Pricing", value=True)
        
        with col2:
            # Risk-free rate section with suggestions
            st.write("**Risk-free Rate (%)**")
            
            # Get rate suggestions for all currencies
            all_rate_suggestions = []
            currencies = ['USD', 'EUR', 'GBP']
            
            for currency in currencies:
                rate_suggestions = rates_provider.get_rate_suggestions(currency)
                all_rate_suggestions.extend(rate_suggestions)
            
            # Also add EONIA separately
            if 'EONIA' in rates_provider.current_rates:
                eonia_rates = rates_provider.current_rates['EONIA']
                for tenor, rate in eonia_rates.items():
                    if rate is not None:
                        all_rate_suggestions.append({
                            'rate': rate / 100,
                            'rate_percent': rate,
                            'description': f"EONIA {tenor.replace('_', ' ').title()} (Legacy EUR rate)",
                            'currency': 'EONIA',
                            'tenor': tenor,
                            'source': 'European Central Bank (ESTR-based)'
                        })
            
            default_rate = 5.0
            if all_rate_suggestions:
                # Use the 3-month rate as default if available
                for suggestion in all_rate_suggestions:
                    if '3m' in suggestion['tenor'].lower():
                        default_rate = suggestion['rate_percent']
                        break
                else:
                    default_rate = all_rate_suggestions[0]['rate_percent']
            
            risk_free_rate = st.number_input(
                "Manual Rate (%)", 
                min_value=0.0, 
                max_value=20.0, 
                value=default_rate, 
                step=0.01,
                help="Enter custom rate or use live market rates"
            ) / 100
        
        with col3:
            volatility_override = st.number_input(
                "Override Volatility (%)",
                min_value=1.0,
                max_value=200.0,
                value=25.0,
                step=0.1,
                help="Override implied volatility for calculations"
            ) / 100
        
        with col4:
            # Strike price dropdown (if options data available)
            selected_strike = None
            if options_df is not None and not options_df.empty:
                available_strikes = sorted(options_df['strike'].unique())
                
                if available_strikes:
                    strike_options = ["All Strikes"] + [f"${strike:.2f}" for strike in available_strikes]
                    selected_strike_display = st.selectbox(
                        "Filter by Strike:",
                        strike_options,
                        help="Filter options by specific strike price"
                    )
                    
                    if selected_strike_display != "All Strikes":
                        selected_strike = float(selected_strike_display.replace("$", ""))
        
        return {
            'enable_pricing': enable_pricing,
            'risk_free_rate': risk_free_rate,
            'volatility_override': volatility_override,
            'selected_strike': selected_strike,
            'rate_suggestions': all_rate_suggestions
        }
    
    @staticmethod
    def render_rate_suggestions(rate_suggestions: List[Dict]) -> None:
        """Render live market rate suggestions in expander"""
        if not rate_suggestions:
            return
        
        with st.expander("📊 Live Market Rates", expanded=False):
            st.write("**Current Risk-Free Rates:**")
            
            # Create dropdown for quick rate selection
            rate_options = {}
            rate_display_list = ["Select a rate..."]
            
            for suggestion in rate_suggestions:
                display_text = f"{suggestion['rate_percent']:.2f}% - {suggestion['description']}"
                rate_display_list.append(display_text)
                rate_options[display_text] = suggestion['rate_percent']
            
            selected_rate_display = st.selectbox(
                "Quick Rate Selection:",
                rate_display_list,
                key="rate_dropdown"
            )
            
            if selected_rate_display != "Select a rate..." and selected_rate_display in rate_options:
                if st.button("Apply Selected Rate", key="apply_dropdown_rate"):
                    st.session_state['selected_risk_free_rate'] = rate_options[selected_rate_display]
                    st.rerun()
            
            st.divider()
            
            for suggestion in rate_suggestions:
                col_rate, col_use = st.columns([3, 1])
                with col_rate:
                    st.write(f"**{suggestion['rate_percent']:.2f}%** - {suggestion['description']}")
                    st.caption(f"Source: {suggestion['source']}")
                with col_use:
                    if st.button(f"Use", key=f"use_rate_{suggestion['currency']}_{suggestion['tenor']}", 
                               help=f"Use {suggestion['rate_percent']:.2f}%"):
                        st.session_state['selected_risk_free_rate'] = suggestion['rate_percent']
                        st.rerun()
    
    @staticmethod
    def render_options_table(options_df: pd.DataFrame) -> None:
        """Render formatted options data table"""
        if options_df.empty:
            st.warning("No options data to display")
            return
        
        # Format the dataframe for display
        display_df = DataFormatter.format_options_dataframe(options_df)
        
        # Select relevant columns for display
        display_columns = [
            'option_type', 'strike', 'expiration', 'lastPrice', 'bid', 'ask',
            'volume', 'openInterest', 'impliedVolatility'
        ]
        
        # Add theoretical pricing columns if available
        if 'theoretical_price' in display_df.columns:
            display_columns.extend(['theoretical_price', 'price_diff', 'price_diff_pct'])
        
        # Add Greeks if available
        greeks_cols = ['delta', 'gamma', 'theta', 'vega', 'rho']
        for col in greeks_cols:
            if col in display_df.columns:
                display_columns.append(col)
        
        # Filter to available columns
        available_columns = [col for col in display_columns if col in display_df.columns]
        
        if available_columns:
            st.dataframe(
                display_df[available_columns],
                use_container_width=True,
                height=400
            )
        else:
            st.dataframe(display_df, use_container_width=True, height=400)
    
    @staticmethod
    def render_pricing_breakdown(breakdown: Dict) -> None:
        """Render step-by-step pricing breakdown"""
        with st.expander("🔍 Step-by-Step Pricing Breakdown", expanded=False):
            if 'error' in breakdown:
                st.error(f"Pricing calculation error: {breakdown['error']}")
                return
            
            # Input parameters
            st.write("### 📝 Input Parameters")
            inputs = breakdown.get('inputs', {})
            
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Stock Price (S):** {DataFormatter.format_currency(inputs.get('stock_price', 0))}")
                st.write(f"**Strike Price (K):** {DataFormatter.format_currency(inputs.get('strike_price', 0))}")
                st.write(f"**Time to Expiry (T):** {inputs.get('time_to_expiry', 0):.4f} years")
            
            with col2:
                st.write(f"**Risk-free Rate (r):** {DataFormatter.format_percentage(inputs.get('risk_free_rate', 0) * 100)}")
                st.write(f"**Volatility (σ):** {DataFormatter.format_percentage(inputs.get('volatility', 0) * 100)}")
                st.write(f"**Option Type:** {inputs.get('option_type', 'call').title()}")
            
            st.divider()
            
            # Pricing results
            st.write("### 💰 Pricing Results")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Black-Scholes Price",
                    DataFormatter.format_currency(breakdown.get('black_scholes', 0))
                )
            
            with col2:
                st.metric(
                    "Binomial Tree Price",
                    DataFormatter.format_currency(breakdown.get('binomial_tree', 0))
                )
            
            with col3:
                st.metric(
                    "Monte Carlo Price",
                    DataFormatter.format_currency(breakdown.get('monte_carlo', 0))
                )
            
            st.divider()
            
            # Value breakdown
            st.write("### 📊 Value Breakdown")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Intrinsic Value",
                    DataFormatter.format_currency(breakdown.get('intrinsic_value', 0))
                )
            
            with col2:
                st.metric(
                    "Time Value",
                    DataFormatter.format_currency(breakdown.get('time_value', 0))
                )
            
            with col3:
                st.metric(
                    "Moneyness",
                    breakdown.get('moneyness', 'N/A')
                )
            
            # Greeks
            greeks = breakdown.get('greeks', {})
            if greeks:
                st.divider()
                st.write("### 📈 Greeks")
                
                col1, col2, col3, col4, col5 = st.columns(5)
                
                with col1:
                    st.metric("Delta", DataFormatter.format_number(greeks.get('delta', 0)))
                
                with col2:
                    st.metric("Gamma", DataFormatter.format_number(greeks.get('gamma', 0)))
                
                with col3:
                    st.metric("Theta", DataFormatter.format_number(greeks.get('theta', 0)))
                
                with col4:
                    st.metric("Vega", DataFormatter.format_number(greeks.get('vega', 0)))
                
                with col5:
                    st.metric("Rho", DataFormatter.format_number(greeks.get('rho', 0)))
