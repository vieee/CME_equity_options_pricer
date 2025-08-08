"""
Streamlit UI Components for Options Pricing Application
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
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
        
        # Safely get the symbol with fallback
        symbol = stock_info.get('symbol', stock_info.get('shortName', 'Unknown'))
        st.subheader(f"📊 {symbol} - Stock Overview")
        
        # Create 4 columns for stock metrics
        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
        
        with metric_col1:
            # Determine if we're showing current or previous close price
            current_price = stock_info['currentPrice']
            previous_close = stock_info['previousClose']
            
            # Check if current price equals previous close (likely using fallback)
            is_market_hours = abs(current_price - previous_close) > 0.01 if previous_close > 0 else True
            
            price_label = "💰 Current Price" if is_market_hours else "💰 Price (Previous Close)"
            price_help = "Real-time stock price" if is_market_hours else "Previous close price (markets may be closed)"
            
            st.metric(
                label=price_label, 
                value=DataFormatter.format_currency(current_price),
                help=price_help
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
            
            # Determine default rate - check for pending rate update first
            default_rate = 5.0
            
            # Check if there's a pending rate update
            if 'pending_rate_update' in st.session_state:
                default_rate = st.session_state['pending_rate_update']
                # Clear the pending update and set it as selected
                st.session_state['selected_risk_free_rate'] = default_rate
                del st.session_state['pending_rate_update']
            elif 'selected_risk_free_rate' in st.session_state:
                default_rate = st.session_state['selected_risk_free_rate']
            elif all_rate_suggestions:
                # Use the 3-month rate as default if available
                for suggestion in all_rate_suggestions:
                    if '3m' in suggestion['tenor'].lower():
                        default_rate = suggestion['rate_percent']
                        break
                else:
                    default_rate = all_rate_suggestions[0]['rate_percent']
            
            # Create a callback function for rate updates
            def on_rate_change():
                # This will be called when number_input changes
                if 'manual_rate_input' in st.session_state:
                    st.session_state['selected_risk_free_rate'] = st.session_state['manual_rate_input']
            
            risk_free_rate_percent = st.number_input(
                "Manual Rate (%)", 
                min_value=0.0, 
                max_value=20.0, 
                value=default_rate, 
                step=0.01,
                key="manual_rate_input",
                on_change=on_rate_change,
                help="Enter custom rate or use live market rates"
            )
            
            risk_free_rate = risk_free_rate_percent / 100
        
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
                    
                    # Check for existing selection in session state
                    default_index = 0
                    if 'selected_strike_filter' in st.session_state:
                        stored_strike = st.session_state['selected_strike_filter']
                        if stored_strike is not None:
                            strike_display = f"${stored_strike:.2f}"
                            if strike_display in strike_options:
                                default_index = strike_options.index(strike_display)
                    
                    selected_strike_display = st.selectbox(
                        "Filter by Strike:",
                        strike_options,
                        index=default_index,
                        key="strike_filter_selector",
                        help="Filter options by specific strike price"
                    )
                    
                    if selected_strike_display != "All Strikes":
                        selected_strike = float(selected_strike_display.replace("$", ""))
                        # Store in session state and trigger rerun if changed
                        if st.session_state.get('selected_strike_filter') != selected_strike:
                            st.session_state['selected_strike_filter'] = selected_strike
                            st.info(f"🔄 Updating filter to ${selected_strike:.2f}...")
                            st.rerun()
                    else:
                        # Clear strike filter if "All Strikes" selected
                        if 'selected_strike_filter' in st.session_state and st.session_state['selected_strike_filter'] is not None:
                            st.session_state['selected_strike_filter'] = None
                            st.info("🔄 Clearing strike filter...")
                            st.rerun()
        
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
            
            # Show current rate status
            if 'selected_risk_free_rate' in st.session_state:
                current_rate = st.session_state['selected_risk_free_rate']
                st.info(f"🎯 Active Rate: {current_rate:.2f}%")
            
            if 'pending_rate_update' in st.session_state:
                pending_rate = st.session_state['pending_rate_update']
                st.warning(f"⏳ Pending Rate Update: {pending_rate:.2f}% (will apply on next refresh)")
            
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
                    # Set pending rate update which will be applied on next rerun
                    new_rate = rate_options[selected_rate_display]
                    st.session_state['pending_rate_update'] = new_rate
                    st.success(f"✅ Applied rate: {new_rate:.2f}%")
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
                        # Set pending rate update which will be applied on next rerun
                        new_rate = suggestion['rate_percent']
                        st.session_state['pending_rate_update'] = new_rate
                        st.success(f"✅ Applied rate: {new_rate:.2f}%")
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

    @staticmethod
    def render_option_analysis_plots(options_df: pd.DataFrame, current_price: float, 
                                   pricing_controls: Dict, rates_provider) -> None:
        """Render comprehensive option analysis plots"""
        st.subheader("📊 Option Analysis Plots")
        
        if options_df.empty:
            st.warning("No options data available for plotting")
            return
        
        # Create tabs for different plot types
        plot_tab1, plot_tab2, plot_tab3, plot_tab4 = st.tabs([
            "📈 Price vs IV", "🎯 Greeks Comparison", "💰 Model Comparison", "📊 Sensitivity Analysis"
        ])
        
        with plot_tab1:
            StreamlitComponents._render_price_vs_iv_plot(options_df, current_price)
        
        with plot_tab2:
            StreamlitComponents._render_greeks_comparison_plot(options_df, current_price, pricing_controls)
        
        with plot_tab3:
            StreamlitComponents._render_model_comparison_plot(options_df, current_price, pricing_controls)
        
        with plot_tab4:
            StreamlitComponents._render_sensitivity_analysis_plot(options_df, current_price, pricing_controls)
    
    @staticmethod
    def _render_price_vs_iv_plot(options_df: pd.DataFrame, current_price: float) -> None:
        """Render enhanced option price vs implied volatility plot with overlays and regression"""
        if 'impliedVolatility' not in options_df.columns or 'lastPrice' not in options_df.columns:
            st.warning("Implied volatility or price data not available")
            return
        
        # Filter out invalid data
        plot_df = options_df[
            (options_df['impliedVolatility'] > 0) & 
            (options_df['impliedVolatility'] < 5) &  # Remove extreme values
            (options_df['lastPrice'] > 0) &
            (pd.notna(options_df['impliedVolatility'])) &
            (pd.notna(options_df['lastPrice']))
        ].copy()
        
        if plot_df.empty:
            st.warning("No valid price/IV data for plotting")
            return
        
        # Add calculated fields for enhanced analysis
        plot_df['moneyness'] = plot_df['strike'] / current_price
        plot_df['days_to_expiry'] = pd.to_datetime(plot_df['expiration']).apply(
            lambda x: max(1, (x - pd.Timestamp.now()).days)
        )
        
        # Color coding options
        color_options = ["Moneyness", "Days to Expiry", "Option Type"]
        color_by = st.selectbox("Color code by:", color_options, key="iv_color_selector")
        
        # Plotting options
        plot_col1, plot_col2 = st.columns([3, 1])
        
        with plot_col2:
            show_regression = st.checkbox("Show IV Smile Curve", value=True)
            overlay_theoretical = st.checkbox("Overlay Theoretical Prices", 
                                            value='theoretical_price' in plot_df.columns)
            separate_plots = st.checkbox("Separate Calls/Puts", value=True)
        
        with plot_col1:
            if separate_plots:
                # Separate plots for calls and puts
                subcol1, subcol2 = st.columns(2)
                
                with subcol1:
                    st.write("**Call Options**")
                    calls_df = plot_df[plot_df['option_type'] == 'call']
                    if not calls_df.empty:
                        fig_calls = StreamlitComponents._create_enhanced_iv_plot(
                            calls_df, current_price, color_by, show_regression, 
                            overlay_theoretical, "Call Options"
                        )
                        st.plotly_chart(fig_calls, use_container_width=True)
                    else:
                        st.info("No call options data")
                
                with subcol2:
                    st.write("**Put Options**")
                    puts_df = plot_df[plot_df['option_type'] == 'put']
                    if not puts_df.empty:
                        fig_puts = StreamlitComponents._create_enhanced_iv_plot(
                            puts_df, current_price, color_by, show_regression, 
                            overlay_theoretical, "Put Options"
                        )
                        st.plotly_chart(fig_puts, use_container_width=True)
                    else:
                        st.info("No put options data")
            else:
                # Combined plot
                st.write("**Options: Price vs Implied Volatility**")
                fig_combined = StreamlitComponents._create_enhanced_iv_plot(
                    plot_df, current_price, color_by, show_regression, 
                    overlay_theoretical, "All Options", show_option_type=True
                )
                st.plotly_chart(fig_combined, use_container_width=True)
    
    @staticmethod
    def _create_enhanced_iv_plot(df: pd.DataFrame, current_price: float, color_by: str, 
                               show_regression: bool, overlay_theoretical: bool, 
                               title: str, show_option_type: bool = False) -> go.Figure:
        """Create enhanced IV plot with all requested features"""
        from scipy.interpolate import UnivariateSpline
        from sklearn.linear_model import LinearRegression
        
        fig = go.Figure()
        
        # Determine color mapping
        if color_by == "Moneyness":
            color_values = df['moneyness']
            colorscale = 'RdYlBu_r'
            colorbar_title = "Moneyness (K/S)"
        elif color_by == "Days to Expiry":
            color_values = df['days_to_expiry']
            colorscale = 'Viridis'
            colorbar_title = "Days to Expiry"
        else:  # Option Type
            color_values = df['option_type'].map({'call': 1, 'put': 0})
            colorscale = 'RdBu'
            colorbar_title = "Option Type"
        
        # Enhanced tooltips with comprehensive information
        hover_text = []
        for _, row in df.iterrows():
            moneyness_status = "ITM" if (
                (row['option_type'] == 'call' and row['strike'] < current_price) or
                (row['option_type'] == 'put' and row['strike'] > current_price)
            ) else "OTM"
            
            tooltip = (
                f"<b>{row['option_type'].title()} Option</b><br>"
                f"Strike: ${row['strike']:.2f}<br>"
                f"Expiry: {row['expiration']}<br>"
                f"Days to Expiry: {row['days_to_expiry']}<br>"
                f"IV: {row['impliedVolatility']:.1%}<br>"
                f"Market Price: ${row['lastPrice']:.2f}<br>"
                f"Moneyness: {row['moneyness']:.3f} ({moneyness_status})<br>"
                f"Volume: {row.get('volume', 'N/A')}<br>"
                f"Open Interest: {row.get('openInterest', 'N/A')}"
            )
            
            if 'theoretical_price' in row and pd.notna(row['theoretical_price']):
                tooltip += f"<br>Theoretical Price: ${row['theoretical_price']:.2f}"
                price_diff = row['theoretical_price'] - row['lastPrice']
                tooltip += f"<br>Price Difference: ${price_diff:.2f}"
            
            hover_text.append(tooltip)
        
        # Main scatter plot for market prices
        symbol_map = {'call': 'circle', 'put': 'triangle-up'} if show_option_type else None
        
        fig.add_trace(go.Scatter(
            x=df['impliedVolatility'] * 100,
            y=df['lastPrice'],
            mode='markers',
            marker=dict(
                size=10,
                color=color_values,
                colorscale=colorscale,
                colorbar=dict(title=colorbar_title),
                showscale=True,
                symbol=[symbol_map.get(ot, 'circle') for ot in df['option_type']] if symbol_map else 'circle',
                line=dict(width=1, color='white')
            ),
            text=hover_text,
            hovertemplate='%{text}<extra></extra>',
            name='Market Prices'
        ))
        
        # Overlay theoretical prices if available and requested
        if overlay_theoretical and 'theoretical_price' in df.columns:
            theoretical_hover = []
            for _, row in df.iterrows():
                theo_tooltip = (
                    f"<b>Theoretical {row['option_type'].title()}</b><br>"
                    f"Strike: ${row['strike']:.2f}<br>"
                    f"IV: {row['impliedVolatility']:.1%}<br>"
                    f"Theoretical Price: ${row['theoretical_price']:.2f}<br>"
                    f"Market Price: ${row['lastPrice']:.2f}<br>"
                    f"Difference: ${row['theoretical_price'] - row['lastPrice']:.2f}"
                )
                theoretical_hover.append(theo_tooltip)
            
            fig.add_trace(go.Scatter(
                x=df['impliedVolatility'] * 100,
                y=df['theoretical_price'],
                mode='markers',
                marker=dict(
                    size=8,
                    color='red',
                    symbol='x',
                    line=dict(width=2)
                ),
                text=theoretical_hover,
                hovertemplate='%{text}<extra></extra>',
                name='Theoretical Prices',
                opacity=0.7
            ))
        
        # Add regression/spline curve for IV smile
        if show_regression and len(df) >= 3:
            try:
                # Sort by IV for smooth curve
                df_sorted = df.sort_values('impliedVolatility')
                iv_values = df_sorted['impliedVolatility'].values
                price_values = df_sorted['lastPrice'].values
                
                # Remove any NaN values
                mask = ~(np.isnan(iv_values) | np.isnan(price_values))
                iv_clean = iv_values[mask]
                price_clean = price_values[mask]
                
                if len(iv_clean) >= 3:
                    # Use spline interpolation for smooth curve
                    spline = UnivariateSpline(iv_clean, price_clean, s=0.1, k=min(3, len(iv_clean)-1))
                    iv_smooth = np.linspace(iv_clean.min(), iv_clean.max(), 100)
                    price_smooth = spline(iv_smooth)
                    
                    fig.add_trace(go.Scatter(
                        x=iv_smooth * 100,
                        y=price_smooth,
                        mode='lines',
                        line=dict(color='orange', width=3, dash='dash'),
                        name='IV Smile Curve',
                        hovertemplate='IV: %{x:.1f}%<br>Fitted Price: $%{y:.2f}<extra></extra>'
                    ))
            except Exception as e:
                st.warning(f"Could not fit IV smile curve: {str(e)}")
        
        # Enhanced layout
        fig.update_layout(
            title=f"{title}: Price vs Implied Volatility",
            xaxis_title="Implied Volatility (%)",
            yaxis_title="Option Price ($)",
            height=500,
            hovermode='closest',
            legend=dict(
                x=0.02, y=0.98,
                bgcolor='rgba(255,255,255,0.8)',
                bordercolor='rgba(0,0,0,0.2)',
                borderwidth=1
            ),
            template='plotly_white'
        )
        
        return fig
    
    @staticmethod
    def _render_greeks_comparison_plot(options_df: pd.DataFrame, current_price: float, 
                                     pricing_controls: Dict) -> None:
        """Render enhanced Greeks comparison with market vs theoretical overlays"""
        greeks_cols = ['delta', 'gamma', 'theta', 'vega', 'rho']
        available_greeks = [col for col in greeks_cols if col in options_df.columns]
        theoretical_greeks = [col for col in greeks_cols if f"{col}_theoretical" in options_df.columns]
        
        if not available_greeks and not theoretical_greeks:
            st.warning("No Greeks data available for plotting")
            return
        
        # Filter valid data
        plot_df = options_df[
            (options_df['lastPrice'] > 0) &
            (pd.notna(options_df['strike']))
        ].copy()
        
        if plot_df.empty:
            st.warning("No valid options data for Greeks plotting")
            return
        
        # Add calculated fields for enhanced analysis
        plot_df['moneyness'] = plot_df['strike'] / current_price
        plot_df['days_to_expiry'] = pd.to_datetime(plot_df['expiration']).apply(
            lambda x: max(1, (x - pd.Timestamp.now()).days)
        )
        
        # Greek selection and display options
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            all_greeks = list(set(available_greeks + theoretical_greeks))
            selected_greeks = st.multiselect(
                "Select Greeks to compare:", 
                all_greeks,
                default=all_greeks[:3] if len(all_greeks) >= 3 else all_greeks,
                key="greeks_selector"
            )
        
        with col2:
            color_by = st.selectbox(
                "Color by:",
                ["Moneyness", "Days to Expiry", "Option Type"],
                key="greeks_color_selector"
            )
            overlay_mode = st.checkbox("Overlay Market vs Theoretical", value=True)
        
        with col3:
            show_regression = st.checkbox("Show Trend Lines", value=False)
            normalize_greeks = st.checkbox("Normalize Values", value=False)
        
        if not selected_greeks:
            st.info("Please select at least one Greek to display")
            return
        
        # Create subplots for selected Greeks
        n_greeks = len(selected_greeks)
        cols_per_row = min(2, n_greeks)
        n_rows = (n_greeks + cols_per_row - 1) // cols_per_row
        
        fig = make_subplots(
            rows=n_rows, 
            cols=cols_per_row,
            subplot_titles=[f"{greek.title()} Analysis" for greek in selected_greeks],
            vertical_spacing=0.1,
            horizontal_spacing=0.1
        )
        
        # Color mapping
        if color_by == "Moneyness":
            color_values = plot_df['moneyness']
            colorscale = 'RdYlBu_r'
        elif color_by == "Days to Expiry":
            color_values = plot_df['days_to_expiry']
            colorscale = 'Viridis'
        else:  # Option Type
            color_values = plot_df['option_type'].map({'call': 1, 'put': 0})
            colorscale = 'RdBu'
        
        for i, greek in enumerate(selected_greeks):
            row = (i // cols_per_row) + 1
            col = (i % cols_per_row) + 1
            
            # Market Greeks
            if greek in plot_df.columns:
                market_values = plot_df[greek].copy()
                if normalize_greeks and market_values.std() > 0:
                    market_values = (market_values - market_values.mean()) / market_values.std()
                
                # Enhanced hover text
                hover_text = []
                for _, row_data in plot_df.iterrows():
                    tooltip = (
                        f"<b>{row_data['option_type'].title()} Option</b><br>"
                        f"Strike: ${row_data['strike']:.2f}<br>"
                        f"Expiry: {row_data['expiration']}<br>"
                        f"Market {greek.title()}: {row_data[greek]:.4f}<br>"
                        f"Moneyness: {row_data['moneyness']:.3f}<br>"
                        f"Price: ${row_data['lastPrice']:.2f}"
                    )
                    
                    # Add theoretical comparison if available
                    theo_col = f"{greek}_theoretical"
                    if theo_col in row_data and pd.notna(row_data[theo_col]):
                        tooltip += f"<br>Theoretical {greek.title()}: {row_data[theo_col]:.4f}"
                        diff = row_data[theo_col] - row_data[greek]
                        tooltip += f"<br>Difference: {diff:.4f}"
                    
                    hover_text.append(tooltip)
                
                # Market values scatter
                fig.add_trace(
                    go.Scatter(
                        x=plot_df['strike'],
                        y=market_values,
                        mode='markers',
                        marker=dict(
                            size=8,
                            color=color_values,
                            colorscale=colorscale,
                            showscale=(i == 0),  # Show colorbar only once
                            symbol='circle',
                            line=dict(width=1, color='white')
                        ),
                        text=hover_text,
                        hovertemplate='%{text}<extra></extra>',
                        name=f'Market {greek.title()}',
                        legendgroup=f'market_{greek}',
                        showlegend=(i == 0)
                    ),
                    row=row, col=col
                )
            
            # Theoretical Greeks overlay
            theo_col = f"{greek}_theoretical"
            if overlay_mode and theo_col in plot_df.columns:
                theo_values = plot_df[theo_col].copy()
                valid_mask = pd.notna(theo_values)
                
                if valid_mask.sum() > 0:
                    if normalize_greeks and theo_values[valid_mask].std() > 0:
                        theo_mean = theo_values[valid_mask].mean()
                        theo_std = theo_values[valid_mask].std()
                        theo_values = (theo_values - theo_mean) / theo_std
                    
                    theo_hover = []
                    for idx, row_data in plot_df[valid_mask].iterrows():
                        theo_tooltip = (
                            f"<b>Theoretical {greek.title()}</b><br>"
                            f"Strike: ${row_data['strike']:.2f}<br>"
                            f"Value: {row_data[theo_col]:.4f}<br>"
                            f"Option: {row_data['option_type'].title()}"
                        )
                        theo_hover.append(theo_tooltip)
                    
                    fig.add_trace(
                        go.Scatter(
                            x=plot_df[valid_mask]['strike'],
                            y=theo_values[valid_mask],
                            mode='markers',
                            marker=dict(
                                size=6,
                                color='red',
                                symbol='x',
                                line=dict(width=2)
                            ),
                            text=theo_hover,
                            hovertemplate='%{text}<extra></extra>',
                            name=f'Theoretical {greek.title()}',
                            legendgroup=f'theoretical_{greek}',
                            showlegend=(i == 0),
                            opacity=0.7
                        ),
                        row=row, col=col
                    )
            
            # Add trend lines if requested
            if show_regression and greek in plot_df.columns:
                valid_data = plot_df.dropna(subset=[greek, 'strike'])
                if len(valid_data) >= 3:
                    try:
                        from sklearn.linear_model import LinearRegression
                        X = valid_data['strike'].values.reshape(-1, 1)
                        y = valid_data[greek].values
                        
                        reg = LinearRegression().fit(X, y)
                        x_trend = np.linspace(X.min(), X.max(), 50)
                        y_trend = reg.predict(x_trend.reshape(-1, 1))
                        
                        fig.add_trace(
                            go.Scatter(
                                x=x_trend.flatten(),
                                y=y_trend,
                                mode='lines',
                                line=dict(color='orange', width=2, dash='dash'),
                                name=f'{greek.title()} Trend',
                                legendgroup=f'trend_{greek}',
                                showlegend=(i == 0),
                                hovertemplate=f'Strike: $%{{x:.2f}}<br>{greek.title()} Trend: %{{y:.4f}}<extra></extra>'
                            ),
                            row=row, col=col
                        )
                    except Exception as e:
                        pass  # Skip trend line if regression fails
            
            # Update subplot axes
            fig.update_xaxes(title_text="Strike Price ($)", row=row, col=col)
            y_title = f"{greek.title()}" + (" (Normalized)" if normalize_greeks else "")
            fig.update_yaxes(title_text=y_title, row=row, col=col)
        
        # Add current price reference lines
        for i in range(n_greeks):
            row = (i // cols_per_row) + 1
            col = (i % cols_per_row) + 1
            fig.add_vline(x=current_price, line_dash="dash", line_color="gray", 
                         annotation_text="Current Price", row=row, col=col)
        
        # Update layout
        fig.update_layout(
            title="Greeks Analysis: Market vs Theoretical Comparison",
            height=400 * n_rows,
            hovermode='closest',
            template='plotly_white',
            legend=dict(
                x=1.02, y=1,
                bgcolor='rgba(255,255,255,0.8)',
                bordercolor='rgba(0,0,0,0.2)',
                borderwidth=1
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    @staticmethod
    def _render_model_comparison_plot(options_df: pd.DataFrame, current_price: float, 
                                    pricing_controls: Dict) -> None:
        """Render enhanced comparison of different pricing models with overlays and analysis"""
        from src.models.pricing import OptionsPricingEngine
        from sklearn.metrics import mean_squared_error, mean_absolute_error
        
        if not pricing_controls.get('enable_pricing', False):
            st.info("Enable theoretical pricing to see model comparisons")
            return
        
        # Enhanced model selection and visualization options
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            st.write("**Option Pricing Model Comparison**")
            
        with col2:
            show_residuals = st.checkbox("Show Residual Analysis", value=False)
            show_confidence = st.checkbox("Show Confidence Bands", value=True)
            
        with col3:
            color_by_expiry = st.checkbox("Color by Expiry", value=True)
            normalize_prices = st.checkbox("Normalize by Market Price", value=False)
        
        # Filter and prepare data
        valid_df = options_df[
            (options_df['lastPrice'] > 0) &
            (pd.notna(options_df['strike'])) &
            (pd.notna(options_df['lastPrice']))
        ].copy()
        
        if valid_df.empty:
            st.warning("No valid options data for model comparison")
            return
        
        # Calculate days to expiry for color coding
        valid_df['days_to_expiry'] = pd.to_datetime(valid_df['expiration']).apply(
            lambda x: max(1, (x - pd.Timestamp.now()).days)
        )
        valid_df['time_to_expiry'] = valid_df['days_to_expiry'] / 365.0
        
        # Sample data if too many options
        sample_size = min(50, len(valid_df))
        sample_df = valid_df.sample(n=sample_size, random_state=42).copy()
        
        pricing_engine = OptionsPricingEngine()
        
        # Calculate prices using different models
        model_results = {
            'Black-Scholes': {'prices': [], 'errors': []},
            'Binomial Tree': {'prices': [], 'errors': []},
            'Monte Carlo': {'prices': [], 'errors': []}
        }
        
        strikes = []
        market_prices = []
        option_types = []
        days_to_expiry = []
        
        progress_bar = st.progress(0)
        
        for i, (_, row) in enumerate(sample_df.iterrows()):
            try:
                S = current_price
                K = row['strike']
                T = max(row['time_to_expiry'], 0.001)  # Minimum 1 day
                r = pricing_controls.get('risk_free_rate', 0.05)
                sigma = pricing_controls.get('volatility_override', 0.25)
                option_type = row['option_type']
                market_price = row['lastPrice']
                
                # Black-Scholes
                if option_type == 'call':
                    bs_price = pricing_engine.black_scholes_call(S, K, T, r, sigma)
                else:
                    bs_price = pricing_engine.black_scholes_put(S, K, T, r, sigma)
                
                # Binomial Tree
                binomial_price = pricing_engine.binomial_option_price(
                    S, K, T, r, sigma, option_type, steps=100
                )
                
                # Monte Carlo (reduced simulations for faster computation)
                mc_price = pricing_engine.monte_carlo_option_price(
                    S, K, T, r, sigma, option_type, simulations=5000
                )
                
                # Store results
                model_results['Black-Scholes']['prices'].append(bs_price)
                model_results['Black-Scholes']['errors'].append(bs_price - market_price)
                
                model_results['Binomial Tree']['prices'].append(binomial_price)
                model_results['Binomial Tree']['errors'].append(binomial_price - market_price)
                
                model_results['Monte Carlo']['prices'].append(mc_price)
                model_results['Monte Carlo']['errors'].append(mc_price - market_price)
                
                strikes.append(K)
                market_prices.append(market_price)
                option_types.append(option_type)
                days_to_expiry.append(row['days_to_expiry'])
                
                # Update progress
                progress_bar.progress((i + 1) / len(sample_df))
                
            except Exception as e:
                continue
        
        progress_bar.empty()
        
        if not strikes:
            st.warning("No valid options for model comparison")
            return
        
        # Prepare plotting data
        plot_data = pd.DataFrame({
            'strike': strikes,
            'market_price': market_prices,
            'option_type': option_types,
            'days_to_expiry': days_to_expiry
        })
        
        for model, results in model_results.items():
            plot_data[f'{model}_price'] = results['prices']
            plot_data[f'{model}_error'] = results['errors']
        
        # Create enhanced comparison visualization
        if show_residuals:
            # Residual analysis subplot
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=['Model Price Comparison', 'Residuals vs Strike', 
                              'Residuals Distribution', 'Model Performance'],
                specs=[[{"colspan": 2}, None],
                       [{"type": "histogram"}, {"type": "bar"}]],
                vertical_spacing=0.15
            )
        else:
            fig = go.Figure()
        
        # Color mapping
        if color_by_expiry:
            color_values = plot_data['days_to_expiry']
            colorscale = 'Viridis'
            colorbar_title = 'Days to Expiry'
        else:
            color_values = plot_data['option_type'].map({'call': 1, 'put': 0})
            colorscale = 'RdBu'
            colorbar_title = 'Option Type'
        
        # Main price comparison plot
        colors = {'Black-Scholes': 'red', 'Binomial Tree': 'blue', 'Monte Carlo': 'green'}
        
        # Market prices (baseline)
        hover_text = []
        for _, row in plot_data.iterrows():
            tooltip = (
                f"<b>{row['option_type'].title()} Option</b><br>"
                f"Strike: ${row['strike']:.2f}<br>"
                f"Market Price: ${row['market_price']:.2f}<br>"
                f"Days to Expiry: {row['days_to_expiry']}<br>"
            )
            hover_text.append(tooltip)
        
        fig.add_trace(go.Scatter(
            x=plot_data['strike'],
            y=plot_data['market_price'] if not normalize_prices else [1.0] * len(plot_data),
            mode='markers',
            marker=dict(
                size=10,
                color=color_values,
                colorscale=colorscale,
                colorbar=dict(title=colorbar_title),
                showscale=True,
                symbol='circle',
                line=dict(width=2, color='white')
            ),
            text=hover_text,
            hovertemplate='%{text}<extra></extra>',
            name='Market Prices'
        ), row=1, col=1 if show_residuals else None)
        
        # Model predictions with enhanced tooltips
        for model in ['Black-Scholes', 'Binomial Tree', 'Monte Carlo']:
            model_hover = []
            for _, row in plot_data.iterrows():
                tooltip = (
                    f"<b>{model} {row['option_type'].title()}</b><br>"
                    f"Strike: ${row['strike']:.2f}<br>"
                    f"Model Price: ${row[f'{model}_price']:.2f}<br>"
                    f"Market Price: ${row['market_price']:.2f}<br>"
                    f"Error: ${row[f'{model}_error']:.2f}<br>"
                    f"Error %: {(row[f'{model}_error']/row['market_price']*100):.1f}%"
                )
                model_hover.append(tooltip)
            
            y_values = (plot_data[f'{model}_price'] / plot_data['market_price'] 
                       if normalize_prices else plot_data[f'{model}_price'])
            
            fig.add_trace(go.Scatter(
                x=plot_data['strike'],
                y=y_values,
                mode='markers+lines',
                name=model,
                marker=dict(color=colors[model], size=6),
                line=dict(width=2, color=colors[model], dash='dash'),
                text=model_hover,
                hovertemplate='%{text}<extra></extra>',
                opacity=0.8
            ), row=1, col=1 if show_residuals else None)
        
        # Add confidence bands if requested
        if show_confidence and not show_residuals:
            for model in ['Black-Scholes']:  # Show bands for one model to avoid clutter
                sorted_data = plot_data.sort_values('strike')
                x_vals = sorted_data['strike']
                y_vals = sorted_data[f'{model}_price']
                
                # Calculate simple confidence band (±10% of price)
                upper_band = y_vals * 1.1
                lower_band = y_vals * 0.9
                
                fig.add_trace(go.Scatter(
                    x=list(x_vals) + list(x_vals[::-1]),
                    y=list(upper_band) + list(lower_band[::-1]),
                    fill='toself',
                    fillcolor=f'rgba(255,0,0,0.1)',
                    line=dict(color='rgba(255,255,255,0)'),
                    name=f'{model} Confidence',
                    showlegend=False,
                    hoverinfo='skip'
                ))
        
        # Residual analysis subplots
        if show_residuals:
            # Residuals vs Strike
            for model in ['Black-Scholes', 'Binomial Tree', 'Monte Carlo']:
                fig.add_trace(go.Scatter(
                    x=plot_data['strike'],
                    y=plot_data[f'{model}_error'],
                    mode='markers',
                    name=f'{model} Residuals',
                    marker=dict(color=colors[model], size=5),
                    showlegend=False
                ), row=2, col=1)
            
            # Residuals distribution
            for model in ['Black-Scholes']:  # Show one distribution to avoid clutter
                fig.add_trace(go.Histogram(
                    x=plot_data[f'{model}_error'],
                    name=f'{model} Error Dist',
                    nbinsx=20,
                    opacity=0.7,
                    showlegend=False
                ), row=2, col=2)
            
            # Model performance metrics
            metrics_data = []
            for model in ['Black-Scholes', 'Binomial Tree', 'Monte Carlo']:
                rmse = np.sqrt(mean_squared_error(plot_data['market_price'], plot_data[f'{model}_price']))
                mae = mean_absolute_error(plot_data['market_price'], plot_data[f'{model}_price'])
                metrics_data.append({'Model': model, 'RMSE': rmse, 'MAE': mae})
            
            metrics_df = pd.DataFrame(metrics_data)
            
            fig.add_trace(go.Bar(
                x=metrics_df['Model'],
                y=metrics_df['RMSE'],
                name='RMSE',
                showlegend=False,
                marker_color=['red', 'blue', 'green']
            ), row=2, col=2)
        
        # Add current price reference line
        fig.add_vline(x=current_price, line_dash="dash", line_color="gray",
                     annotation_text="Current Price")
        
        # Update layout
        title = "Enhanced Option Pricing Models Comparison"
        if normalize_prices:
            title += " (Normalized by Market Price)"
        
        fig.update_layout(
            title=title,
            height=800 if show_residuals else 600,
            hovermode='closest',
            template='plotly_white',
            legend=dict(
                x=0.02, y=0.98,
                bgcolor='rgba(255,255,255,0.8)',
                bordercolor='rgba(0,0,0,0.2)',
                borderwidth=1
            )
        )
        
        # Update axes labels
        if show_residuals:
            fig.update_xaxes(title_text="Strike Price ($)", row=1, col=1)
            fig.update_yaxes(title_text="Option Price ($)" + (" (Normalized)" if normalize_prices else ""), row=1, col=1)
            fig.update_xaxes(title_text="Strike Price ($)", row=2, col=1)
            fig.update_yaxes(title_text="Price Error ($)", row=2, col=1)
            fig.update_xaxes(title_text="Price Error ($)", row=2, col=2)
            fig.update_yaxes(title_text="Frequency", row=2, col=2)
        else:
            fig.update_xaxes(title_text="Strike Price ($)")
            fig.update_yaxes(title_text="Option Price ($)" + (" (Normalized)" if normalize_prices else ""))
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Enhanced model performance metrics
        st.write("**📊 Model Performance Analysis**")
        
        perf_col1, perf_col2, perf_col3, perf_col4 = st.columns(4)
        
        market_prices_arr = np.array(plot_data['market_price'])
        
        for i, model in enumerate(['Black-Scholes', 'Binomial Tree', 'Monte Carlo']):
            model_prices_arr = np.array(plot_data[f'{model}_price'])
            errors = model_prices_arr - market_prices_arr
            
            rmse = np.sqrt(np.mean(errors ** 2))
            mae = np.mean(np.abs(errors))
            mape = np.mean(np.abs(errors / market_prices_arr)) * 100
            correlation = np.corrcoef(model_prices_arr, market_prices_arr)[0, 1]
            
            col = [perf_col1, perf_col2, perf_col3][i]
            
            with col:
                st.metric(f"**{model}**", "")
                st.metric("RMSE", f"${rmse:.3f}")
                st.metric("MAE", f"${mae:.3f}")
                st.metric("MAPE", f"{mape:.1f}%")
                st.metric("Correlation", f"{correlation:.3f}")
        
        # Statistical analysis
        with perf_col4:
            st.metric("**Market Stats**", "")
            st.metric("Avg Price", f"${np.mean(market_prices_arr):.2f}")
            st.metric("Price Std", f"${np.std(market_prices_arr):.2f}")
            st.metric("Min Price", f"${np.min(market_prices_arr):.2f}")
            st.metric("Max Price", f"${np.max(market_prices_arr):.2f}")
    
    @staticmethod
    def _render_sensitivity_analysis_plot(options_df: pd.DataFrame, current_price: float, 
                                        pricing_controls: Dict) -> None:
        """Render sensitivity analysis plots"""
        from src.models.pricing import OptionsPricingEngine
        
        if not pricing_controls.get('enable_pricing', False):
            st.info("Enable theoretical pricing to see sensitivity analysis")
            return
        
        st.write("**Option Price Sensitivity Analysis**")
        
        # Select an option for analysis
        if len(options_df) == 0:
            st.warning("No options available for sensitivity analysis")
            return
        
        # Use ATM option or closest to ATM
        options_df['distance_to_atm'] = abs(options_df['strike'] - current_price)
        atm_option = options_df.loc[options_df['distance_to_atm'].idxmin()]
        
        st.info(f"Analyzing {atm_option['option_type']} option with strike ${atm_option['strike']:.2f}")
        
        pricing_engine = OptionsPricingEngine()
        
        # Base parameters
        K = atm_option['strike']
        T = atm_option.get('time_to_expiry', 0.25)
        r = pricing_controls.get('risk_free_rate', 0.05)
        sigma = pricing_controls.get('volatility_override', 0.25)
        option_type = atm_option['option_type']
        
        if T <= 0:
            st.warning("Invalid time to expiry for sensitivity analysis")
            return
        
        # Create sensitivity plots
        sens_col1, sens_col2 = st.columns(2)
        
        with sens_col1:
            # Stock price sensitivity
            st.write("**Stock Price Sensitivity**")
            
            price_range = np.linspace(current_price * 0.8, current_price * 1.2, 50)
            option_prices = []
            
            for S in price_range:
                if option_type == 'call':
                    price = pricing_engine.black_scholes_call(S, K, T, r, sigma)
                else:
                    price = pricing_engine.black_scholes_put(S, K, T, r, sigma)
                option_prices.append(price)
            
            fig_price = go.Figure()
            fig_price.add_trace(go.Scatter(
                x=price_range,
                y=option_prices,
                mode='lines',
                name='Option Price',
                line=dict(color='blue', width=3)
            ))
            
            fig_price.add_vline(x=current_price, line_dash="dash", 
                               annotation_text="Current Price")
            
            fig_price.update_layout(
                title=f"{option_type.title()} Option Price vs Stock Price",
                xaxis_title="Stock Price ($)",
                yaxis_title="Option Price ($)",
                height=400
            )
            
            st.plotly_chart(fig_price, use_container_width=True)
        
        with sens_col2:
            # Volatility sensitivity
            st.write("**Volatility Sensitivity**")
            
            vol_range = np.linspace(0.1, 1.0, 50)
            vol_option_prices = []
            
            for vol in vol_range:
                if option_type == 'call':
                    price = pricing_engine.black_scholes_call(current_price, K, T, r, vol)
                else:
                    price = pricing_engine.black_scholes_put(current_price, K, T, r, vol)
                vol_option_prices.append(price)
            
            fig_vol = go.Figure()
            fig_vol.add_trace(go.Scatter(
                x=vol_range * 100,
                y=vol_option_prices,
                mode='lines',
                name='Option Price',
                line=dict(color='green', width=3)
            ))
            
            fig_vol.add_vline(x=sigma * 100, line_dash="dash", 
                             annotation_text="Current Vol")
            
            fig_vol.update_layout(
                title=f"{option_type.title()} Option Price vs Volatility",
                xaxis_title="Volatility (%)",
                yaxis_title="Option Price ($)",
                height=400
            )
            
            st.plotly_chart(fig_vol, use_container_width=True)
