"""
CME Equity Options Pricer
A comprehensive tool to fetch and analyze equity options data from various free APIs
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
import json
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from alpha_vantage.timeseries import TimeSeries
from alpha_vantage.fundamentaldata import FundamentalData
import os
from dotenv import load_dotenv
import warnings
from options_pricing import OptionsPricingModels
from risk_free_rates import get_risk_free_rates
warnings.filterwarnings('ignore')

# Load environment variables
load_dotenv()

# Performance optimization: Cache heavy computations
@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_cached_stock_info(symbol):
    """Cache stock information to avoid repeated API calls"""
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        return {
            'shortName': info.get('shortName', symbol),
            'currentPrice': info.get('currentPrice', 0),
            'marketCap': info.get('marketCap', 0),
            'volume': info.get('volume', 0),
            'previousClose': info.get('previousClose', 0)
        }
    except:
        return None

@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_cached_options_data(symbol):
    """Cache options data to improve performance"""
    try:
        ticker = yf.Ticker(symbol)
        options_dates = ticker.options
        if not options_dates:
            return None, None
        
        current_price = ticker.info.get('currentPrice', 0)
        if current_price == 0:
            # Fallback to history
            hist = ticker.history(period="1d")
            if not hist.empty:
                current_price = hist['Close'].iloc[-1]
        
        all_options = []
        for date in options_dates[:3]:  # Limit to first 3 expiration dates for performance
            try:
                calls = ticker.option_chain(date).calls
                puts = ticker.option_chain(date).puts
                
                calls['option_type'] = 'call'
                calls['expiration'] = date
                puts['option_type'] = 'put'
                puts['expiration'] = date
                
                all_options.extend([calls, puts])
            except:
                continue
        
        if all_options:
            options_df = pd.concat(all_options, ignore_index=True)
            return options_df, current_price
        
        return None, None
    except Exception as e:
        st.error(f"Error fetching options data: {str(e)}")
        return None, None

# JavaScript for double-clicking on implied volatility to set override value
DOUBLE_CLICK_JS = """
<script>
// Function to store data in localStorage and navigate back to the app
function sendToStreamlit(key, value) {
    // Store the value in localStorage
    localStorage.setItem('double_clicked_iv', value.toString());
    
    // Update the URL with the volatility value as a fallback
    window.location.href = `?volatility=${value}`;
}

// Function to handle double-click on dataframe cells
function setupDataframeDoubleClick() {
    // Add event listener with a slight delay to ensure the dataframe is rendered
    setTimeout(function() {
        // Find all dataframe tables
        const tables = document.querySelectorAll('[data-testid="stDataFrame"] table');
        
        tables.forEach(table => {
            // Add event listener to the table
            table.addEventListener('dblclick', function(e) {
                // Get the clicked cell
                const cell = e.target.closest('td');                if (!cell) return;
                
                // Find the header row
                const headerRow = table.querySelector('thead tr');
                if (!headerRow) return;
                
                // Get all header cells
                const headerCells = Array.from(headerRow.querySelectorAll('th'))
                    .map(th => th.textContent.trim());
                
                // Find the column index of the clicked cell
                const cellIndex = Array.from(cell.parentNode.children).indexOf(cell);
                
                // Get the header text for the clicked column
                const colName = headerCells[cellIndex];
                
                // Check if this is an implied volatility cell
                if (colName === 'impliedVolatility') {
                    // Get the cell value
                    const value = parseFloat(cell.textContent) * 100; // Convert to percentage
                    
                    // If value is valid, send it to Streamlit
                    if (!isNaN(value)) {
                        console.log("Double-clicked on impliedVolatility cell with value:", value);
                        
                        // Show visual feedback
                        const originalBackground = cell.style.backgroundColor;
                        cell.style.backgroundColor = "yellow";
                        setTimeout(() => { cell.style.backgroundColor = originalBackground; }, 500);
                          // Send to Streamlit and reload page
                        sendToStreamlit('double_clicked_iv', value);
                        
                        // Show toast notification
                        const toast = document.createElement('div');
                        toast.style.position = 'fixed';
                        toast.style.bottom = '20px';
                        toast.style.right = '20px';
                        toast.style.backgroundColor = '#4CAF50';
                        toast.style.color = 'white';
                        toast.style.padding = '15px';
                        toast.style.borderRadius = '5px';
                        toast.style.zIndex = '9999';
                        toast.style.boxShadow = '0 4px 8px rgba(0,0,0,0.2)';
                        toast.innerHTML = `Setting volatility override to: ${value.toFixed(2)}%`;
                        document.body.appendChild(toast);
                        
                        // Remove toast after 3 seconds
                        setTimeout(() => {
                            document.body.removeChild(toast);
                        }, 3000);
                    }
                }
            });
        });
    }, 1500); // Wait 1.5 second for the dataframe to render
}

// Check if there's a stored value from localStorage on page load
window.addEventListener('load', function() {
    const storedValue = localStorage.getItem('double_clicked_iv');
    if (storedValue) {
        // Find the volatility input field and set its value
        setTimeout(function() {
            const inputFields = document.querySelectorAll('input[type="number"]');
            inputFields.forEach(input => {
                // Check if this input is the volatility input
                const label = input.closest('div').querySelector('label');
                if (label && label.textContent.includes('Override Volatility')) {
                    // Set the value and trigger input event to update Streamlit
                    input.value = storedValue;
                    input.dispatchEvent(new Event('input', { bubbles: true }));
                    
                    // Clear the stored value to avoid reapplying
                    localStorage.removeItem('double_clicked_iv');
                    
                    // Add visual feedback
                    input.style.backgroundColor = '#ffffe0';
                    setTimeout(() => { input.style.backgroundColor = ''; }, 2000);
                }
            });
        }, 1500);
    }
});

// Setup the event listener when the DOM is loaded
document.addEventListener('DOMContentLoaded', setupDataframeDoubleClick);

// Also setup when Streamlit has done an update
const observer = new MutationObserver(function(mutations) {
    setupDataframeDoubleClick();
});

// Start observing the document with the configured parameters
observer.observe(document.body, { childList: true, subtree: true });
</script>
"""

class OptionsDataFetcher:
    """Class to fetch options data from various free APIs"""
    
    def __init__(self):
        self.alpha_vantage_key = os.getenv('ALPHA_VANTAGE_API_KEY', 'demo')
        
    def get_yahoo_options_data(self, symbol):
        """Fetch options data from Yahoo Finance"""
        try:
            ticker = yf.Ticker(symbol)
            
            # Get basic stock info
            info = ticker.info
            current_price = info.get('currentPrice', info.get('regularMarketPrice', 0))
            
            # Get options expiration dates
            exp_dates = ticker.options
            
            if not exp_dates:
                return None, f"No options data available for {symbol}"
            
            options_data = []
            
            # Get options data for each expiration date
            for exp_date in exp_dates[:5]:  # Limit to first 5 expiration dates
                try:
                    option_chain = ticker.option_chain(exp_date)
                    
                    # Process calls
                    calls = option_chain.calls
                    calls['option_type'] = 'call'
                    calls['expiration'] = exp_date
                    
                    # Process puts
                    puts = option_chain.puts
                    puts['option_type'] = 'put'
                    puts['expiration'] = exp_date
                    
                    # Combine calls and puts
                    combined = pd.concat([calls, puts], ignore_index=True)
                    options_data.append(combined)
                    
                except Exception as e:
                    continue
            
            if options_data:
                all_options = pd.concat(options_data, ignore_index=True)
                return all_options, current_price
            else:
                return None, f"Could not fetch options data for {symbol}"
                
        except Exception as e:
            return None, f"Error fetching data: {str(e)}"
    
    def get_stock_data(self, symbol, period="1y"):
        """Get historical stock data"""
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period=period)
            return hist
        except Exception as e:
            st.error(f"Error fetching stock data: {str(e)}")
            return None
    
    def calculate_implied_volatility_approx(self, option_price, stock_price, strike, time_to_expiry, risk_free_rate=0.05):
        """Approximate implied volatility calculation using Newton-Raphson method"""
        try:
            if time_to_expiry <= 0 or option_price <= 0:
                return 0
            
            # Initial guess
            iv = 0.2
            
            for i in range(100):  # Maximum iterations
                d1 = (np.log(stock_price / strike) + (risk_free_rate + 0.5 * iv**2) * time_to_expiry) / (iv * np.sqrt(time_to_expiry))
                d2 = d1 - iv * np.sqrt(time_to_expiry)
                
                # Black-Scholes call price
                from scipy.stats import norm
                call_price = stock_price * norm.cdf(d1) - strike * np.exp(-risk_free_rate * time_to_expiry) * norm.cdf(d2)
                
                # Vega (sensitivity to volatility)
                vega = stock_price * norm.pdf(d1) * np.sqrt(time_to_expiry)
                
                if abs(vega) < 1e-8:
                    break
                
                # Newton-Raphson update
                iv_new = iv - (call_price - option_price) / vega
                
                if abs(iv_new - iv) < 1e-6:
                    return max(0, iv_new)
                
                iv = max(0.001, iv_new)  # Prevent negative volatility
            
            return iv
        except:
            return 0

def format_numeric_columns(df, max_decimals=4):
    """
    Format numeric columns to display with a maximum number of decimal places
    while keeping the original data numeric for calculations
    """
    formatted_df = df.copy()
    
    # Define columns that should be formatted as percentages (already in decimal form)
    percentage_columns = ['impliedVolatility', 'price_diff_pct']
    
    # Define columns that should be formatted as currency
    currency_columns = ['strike', 'lastPrice', 'bid', 'ask', 'theoretical_price', 'price_diff']
    
    # Define columns that should use standard decimal formatting
    decimal_columns = ['delta', 'gamma', 'theta', 'vega', 'rho', 'calculated_iv']
    
    # Define columns that should be displayed as integers
    integer_columns = ['volume', 'openInterest']
    
    for col in formatted_df.columns:
        if formatted_df[col].dtype in ['float64', 'float32', 'int64', 'int32']:
            if col in percentage_columns:
                # Format as percentage with 2 decimal places
                formatted_df[col] = formatted_df[col].apply(
                    lambda x: f"{x*100:.2f}%" if pd.notna(x) else "N/A"
                )
            elif col in currency_columns:
                # Format as currency with 2 decimal places for prices
                formatted_df[col] = formatted_df[col].apply(
                    lambda x: f"${x:.2f}" if pd.notna(x) else "N/A"
                )
            elif col in decimal_columns:
                # Format with specified decimal places for Greeks and analytics
                formatted_df[col] = formatted_df[col].apply(
                    lambda x: f"{x:.{max_decimals}f}" if pd.notna(x) else "N/A"
                )
            elif col in integer_columns:
                # Format as integers with comma separators
                formatted_df[col] = formatted_df[col].apply(
                    lambda x: f"{int(x):,}" if pd.notna(x) else "N/A"
                )
            else:
                # Format other numeric columns with max_decimals, but use scientific notation for very large numbers
                formatted_df[col] = formatted_df[col].apply(
                    lambda x: f"{x:.{max_decimals}f}" if pd.notna(x) and abs(x) < 1e6 else (f"{x:.2e}" if pd.notna(x) else "N/A")
                )
    
    return formatted_df

def create_options_dashboard():
    """Create Streamlit dashboard for options data"""
    
    st.set_page_config(
        page_title="CME Equity Options Pricer",
        page_icon="📈",
        layout="wide"
    )
    # Initialize session state variables if they don't exist
    if 'double_clicked_iv' not in st.session_state:
        st.session_state.double_clicked_iv = 20.0  # Default value
    
    # Create a key for rerun triggering
    if 'rerun_key' not in st.session_state:
        st.session_state.rerun_key = 0
    
    # Function to handle volatility value updates
    def update_volatility(value):
        st.session_state.double_clicked_iv = value
        st.session_state.rerun_key += 1  # Force rerun
    
    # Use session state to store and retrieve double-clicked volatility from URL parameters
    try:
        # Try to get query parameters using newer method
        query_params = st.query_params
        if 'volatility' in query_params:
            try:
                vol_value = float(query_params['volatility'])
                if vol_value > 0 and vol_value <= 100:
                    update_volatility(vol_value)
                    st.sidebar.success(f"Volatility value {vol_value:.2f}% set from table double-click")
                    # Clear the query parameter
                    query_params.clear()
            except ValueError:
                pass
    except:
        # Fall back to experimental methods for older Streamlit versions
        try:
            query_params = st.experimental_get_query_params()
            if 'volatility' in query_params:
                try:
                    vol_value = float(query_params['volatility'][0])
                    if vol_value > 0 and vol_value <= 100:
                        update_volatility(vol_value)
                        st.sidebar.success(f"Volatility value {vol_value:.2f}% set from table double-click")
                        # Clear the query parameter
                        st.experimental_set_query_params()
                except ValueError:
                    pass
        except:
            pass
    
    st.title("📈 CME Equity Options Pricer")
    st.markdown("Real-time equity options data from free APIs")
    
    # Sidebar for inputs
    st.sidebar.header("Options Parameters")
    
    # Popular CME-related symbols
    popular_symbols = ['SPY', 'QQQ', 'IWM', 'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META']
    
    symbol = st.sidebar.selectbox(
        "Select Stock Symbol",
        options=popular_symbols,
        index=0
    )
    
    custom_symbol = st.sidebar.text_input("Or enter custom symbol:")
    if custom_symbol:
        symbol = custom_symbol.upper()
    
    # Risk-Free Rates Section
    st.sidebar.header("📊 Risk-Free Rates")
    
    # Initialize rates provider if not already done
    if 'rates_provider' not in st.session_state:
        st.session_state.rates_provider = get_risk_free_rates()
    
    # Manual refresh button
    if st.sidebar.button("🔄 Refresh Rates", help="Fetch latest rates from APIs"):
        with st.spinner("Fetching latest rates..."):
            st.session_state.rates_provider.fetch_all_rates()
        st.sidebar.success("Rates updated!")
    
    # Display current rates summary
    with st.sidebar.expander("Current Rates Overview", expanded=False):
        rates_info = st.session_state.rates_provider.get_rate_info()
        
        if rates_info['last_updated']:
            st.write(f"**Last Updated:** {rates_info['last_updated'][:19]}")
        
        # Show overnight rates for each currency
        current_rates = st.session_state.rates_provider.current_rates
        if current_rates:
            st.write("**Overnight Rates:**")
            for currency, rates in current_rates.items():
                overnight_rate = rates.get('overnight')
                if overnight_rate is not None:
                    st.write(f"• {currency}: {overnight_rate:.2f}%")
        
        # API configuration status
        if rates_info['fred_api_configured']:
            st.success("✅ FRED API configured")
        else:
            st.warning("⚠️ FRED API key not configured")
            st.info("Set FRED_API_KEY environment variable for US rates")
    
    # API Key configuration help
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
      
      # Initialize data fetcher
    fetcher = OptionsDataFetcher()
    pricer = OptionsPricingModels()
    
    # Test function to validate pricing methods (for debugging)
    def test_pricing_methods():
        """Test that all pricing methods work correctly"""
        try:
            pricer = OptionsPricingModels()
            S, K, T, r, sigma = 100, 105, 0.25, 0.05, 0.2
            
            # Test Black-Scholes
            call_price = pricer.black_scholes_call(S, K, T, r, sigma)
            put_price = pricer.black_scholes_put(S, K, T, r, sigma)
            greeks = pricer.calculate_greeks(S, K, T, r, sigma, 'call')
            
            # Test Binomial
            bin_call = pricer.binomial_option_price(S, K, T, r, sigma, option_type='call')
            
            # Test Monte Carlo
            mc_call = pricer.monte_carlo_option_price(S, K, T, r, sigma, option_type='call', n_simulations=1000)
            
            return True, "All pricing methods working"
        except Exception as e:
            return False, f"Pricing methods error: {str(e)}"
    
    # Main content with better space utilization
    st.markdown("---")  # Add visual separation
    
    # Stock Information Bar - Distributed across the top
    if symbol:
        stock_info = get_cached_stock_info(symbol)
        if stock_info:
            st.subheader(f"📊 {symbol} - Stock Overview")
            
            # Create 4 columns for stock metrics
            metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
            
            with metric_col1:
                st.metric(
                    label="💰 Current Price", 
                    value=f"${stock_info['currentPrice']:.2f}",
                    help="Real-time stock price"
                )
            
            with metric_col2:
                # Calculate change from previous close
                if stock_info['previousClose'] > 0:
                    price_change = stock_info['currentPrice'] - stock_info['previousClose']
                    price_change_pct = (price_change / stock_info['previousClose']) * 100
                    st.metric(
                        label="📈 Daily Change", 
                        value=f"${price_change:.2f}",
                        delta=f"{price_change_pct:.2f}%",
                        help="Change from previous close"
                    )
                else:
                    st.metric("📈 Daily Change", "N/A")
            
            with metric_col3:
                # Market Cap
                if stock_info['marketCap'] > 0:
                    market_cap_b = stock_info['marketCap'] / 1e9
                    st.metric(
                        label="🏢 Market Cap", 
                        value=f"${market_cap_b:.1f}B",
                        help="Total market value"
                    )
                else:
                    st.metric("🏢 Market Cap", "N/A")
            
            with metric_col4:
                # Volume
                if stock_info['volume'] > 0:
                    volume_m = stock_info['volume'] / 1e6
                    st.metric(
                        label="📊 Volume", 
                        value=f"{volume_m:.1f}M",
                        help="Trading volume today"
                    )
                else:
                    st.metric("📊 Volume", "N/A")
        else:
            st.info("💡 Enter a symbol above and the stock information will appear here")
    
    # Options Chain Section - Full width
    st.markdown("---")  # Add separator
    st.subheader(f"📋 Options Chain for {symbol}")
    
    # Fetch button centered
    col_left, col_center, col_right = st.columns([1, 2, 1])
    with col_center:
        # Auto-fetch on symbol change for better UX
        if st.button("🔄 Fetch Options Data", type="primary", use_container_width=True) or (symbol != st.session_state.get('last_symbol', '')):
            st.session_state['last_symbol'] = symbol
            with st.spinner("Fetching options data..."):
                # Use cached data for better performance
                options_data, current_price = get_cached_options_data(symbol)
                
                if options_data is not None:
                    st.session_state['options_data'] = options_data
                    st.session_state['current_price'] = current_price
                    st.session_state['symbol'] = symbol
                    # Center-aligned success message
                    st.markdown(
                        f'<div style="text-align: center;"><span style="color: #28a745; background-color: #d4edda; '
                        f'border: 1px solid #c3e6cb; border-radius: 0.375rem; padding: 0.75rem 1rem; '
                        f'display: inline-block; font-weight: 500;">✅ Loaded {len(options_data)} options contracts</span></div>',
                        unsafe_allow_html=True
                    )
                else:
                    # Center-aligned error message for consistency
                    st.markdown(
                        f'<div style="text-align: center;"><span style="color: #721c24; background-color: #f8d7da; '
                        f'border: 1px solid #f5c6cb; border-radius: 0.375rem; padding: 0.75rem 1rem; '
                        f'display: inline-block; font-weight: 500;">{current_price}</span></div>',
                        unsafe_allow_html=True
                    )
    
    # Display options data
    if 'options_data' in st.session_state:
        options_df = st.session_state['options_data']
        current_price = st.session_state['current_price']
        
        # Filter options
        st.markdown("---")  # Add separator
        st.subheader("Filter Options")
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
        
        # Apply strike filter if selected
        if 'selected_strike' in st.session_state and st.session_state['selected_strike'] is not None:
            filtered_df = filtered_df[filtered_df['strike'] == st.session_state['selected_strike']]
          # Display filtered data
        if len(filtered_df) > 0:
            # Show active filters status
            active_filters = []
            if option_type != "All":
                active_filters.append(f"Type: {option_type}")
            if selected_expiration != "All":
                active_filters.append(f"Expiry: {selected_expiration}")
            if moneyness != "All":
                active_filters.append(f"Moneyness: {moneyness}")
            if 'selected_strike' in st.session_state and st.session_state['selected_strike'] is not None:
                active_filters.append(f"Strike: ${st.session_state['selected_strike']:.2f}")
            
            if active_filters:
                st.info(f"🔍 Active Filters: {' | '.join(active_filters)} | Showing {len(filtered_df)} options")
            else:
                st.info(f"📊 Showing all {len(filtered_df)} options")
            # Add pricing analysis section
            st.markdown("---")  # Add separator
            st.subheader("🧮 Options Pricing Analysis")
            
            # Initialize risk-free rates provider
            if 'rates_provider' not in st.session_state:
                st.session_state.rates_provider = get_risk_free_rates()
            
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
                    rate_suggestions = st.session_state.rates_provider.get_rate_suggestions(currency)
                    all_rate_suggestions.extend(rate_suggestions)
                
                # Also add EONIA separately since it's not in the standard currency map
                if 'EONIA' in st.session_state.rates_provider.current_rates:
                    eonia_rates = st.session_state.rates_provider.current_rates['EONIA']
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
                
                if all_rate_suggestions:
                    # Create expander for rate suggestions
                    with st.expander("📊 Live Market Rates", expanded=False):
                        st.write("**Current Risk-Free Rates:**")
                        
                        # Create dropdown for quick rate selection
                        rate_options = {}
                        rate_display_list = ["Select a rate..."]
                        
                        for suggestion in all_rate_suggestions:
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
                        
                        for suggestion in all_rate_suggestions:
                            col_rate, col_use = st.columns([3, 1])
                            with col_rate:
                                st.write(f"**{suggestion['rate_percent']:.2f}%** - {suggestion['description']}")
                                st.caption(f"Source: {suggestion['source']}")
                            with col_use:
                                if st.button(f"Use", key=f"use_rate_{suggestion['currency']}_{suggestion['tenor']}", help=f"Use {suggestion['rate_percent']:.2f}%"):
                                    st.session_state['selected_risk_free_rate'] = suggestion['rate_percent']
                                    st.rerun()
                
                # Manual input with suggested default
                default_rate = st.session_state.get('selected_risk_free_rate', 5.0)
                if all_rate_suggestions and 'selected_risk_free_rate' not in st.session_state:
                    # Use the 3-month rate as default if available
                    for suggestion in all_rate_suggestions:
                        if '3m' in suggestion['tenor'].lower() or '3 month' in suggestion['description'].lower():
                            default_rate = suggestion['rate_percent']
                            break
                    else:
                        # If no 3-month rate, use the first available
                        default_rate = all_rate_suggestions[0]['rate_percent']
                
                # Create two columns for Manual Rate and Strike dropdown
                rate_col1, rate_col2 = st.columns(2)
                
                with rate_col1:
                    risk_free_rate = st.number_input(
                        "Manual Rate (%)", 
                        min_value=0.0, 
                        max_value=20.0, 
                        value=default_rate, 
                        step=0.01,
                        help="Enter custom rate or use live market rates above"
                    ) / 100
                
                with rate_col2:
                    # Strike prices dropdown (if options data is available)
                    if 'options_data' in st.session_state:
                        options_df = st.session_state['options_data']
                        if not options_df.empty and 'strike' in options_df.columns:
                            unique_strikes = sorted(options_df['strike'].unique())
                            strike_display_list = ["All Strikes"] + [f"${strike:.2f}" for strike in unique_strikes]
                            
                            selected_strike_display = st.selectbox(
                                "Available Strikes:",
                                strike_display_list,
                                key="strike_dropdown",
                                help="Filter by specific strike price"
                            )
                            
                            # Store selected strike for potential filtering
                            if selected_strike_display != "All Strikes":
                                selected_strike_value = float(selected_strike_display.replace('$', ''))
                                st.session_state['selected_strike'] = selected_strike_value
                            else:
                                st.session_state['selected_strike'] = None
                            
                            # Clear strike filter button
                            if 'selected_strike' in st.session_state and st.session_state['selected_strike'] is not None:
                                if st.button("Clear Strike Filter", key="clear_strike_filter", help="Show all strikes"):
                                    st.session_state['selected_strike'] = None
                                    st.rerun()
                    else:
                        st.write("💡 Strike filter available after fetching options data")
                
                # Show last update time if available
                if hasattr(st.session_state.rates_provider, 'last_updated') and st.session_state.rates_provider.last_updated:
                    st.caption(f"Rates updated: {st.session_state.rates_provider.last_updated.strftime('%H:%M UTC')}")
            
            with col3:
                pricing_model = st.selectbox("Pricing Model", ["Black-Scholes", "Binomial Tree", "Monte Carlo"])
            with col4:
                volatility_override = st.number_input("Override Volatility (%)", min_value=0.0, max_value=100.0, value=st.session_state.double_clicked_iv, step=1.0) / 100
            
            if enable_pricing:
                # Cache expensive pricing calculations
                @st.cache_data(ttl=60)  # Cache for 1 minute since pricing is more dynamic
                def calculate_options_pricing(df_dict, current_price, risk_free_rate, volatility_override, pricing_model):
                    """Cache pricing calculations to improve performance"""
                    df = pd.DataFrame(df_dict)
                    
                    # Calculate pricing and Greeks
                    for index, row in df.iterrows():
                        try:
                            S = current_price
                            K = row['strike']
                            days_to_expiry = (pd.to_datetime(row['expiration']) - datetime.now()).days
                            T = max(days_to_expiry / 365.0, 0.001)  # Prevent division by zero
                            r = risk_free_rate
                            sigma = volatility_override if volatility_override > 0 else row.get('impliedVolatility', 0.25)
                            sigma = max(sigma, 0.001)  # Prevent zero volatility
                            option_type = row['option_type']
                            
                            # Calculate theoretical price and Greeks based on model
                            if pricing_model == "Black-Scholes":
                                # For Black-Scholes, calculate price separately
                                if option_type == 'call':
                                    theoretical_price = pricer.black_scholes_call(S, K, T, r, sigma)
                                else:
                                    theoretical_price = pricer.black_scholes_put(S, K, T, r, sigma)
                                greeks = pricer.calculate_greeks(S, K, T, r, sigma, option_type)
                            elif pricing_model == "Binomial Tree":
                                theoretical_price = pricer.binomial_option_price(S, K, T, r, sigma, option_type=option_type, n=100)
                                greeks = pricer.calculate_greeks(S, K, T, r, sigma, option_type)  # Use analytical Greeks for speed
                            else:  # Monte Carlo
                                theoretical_price = pricer.monte_carlo_option_price(S, K, T, r, sigma, option_type=option_type, n_simulations=10000)
                                greeks = pricer.calculate_greeks(S, K, T, r, sigma, option_type)  # Use analytical Greeks for speed
                            
                            # Update dataframe with valid results
                            df.at[index, 'theoretical_price'] = max(0, theoretical_price) if theoretical_price and not np.isnan(theoretical_price) else 0
                            df.at[index, 'days_to_expiry'] = days_to_expiry
                            
                            # Update Greeks
                            for greek, value in greeks.items():
                                if value and not np.isnan(value):
                                    df.at[index, greek] = value
                                else:
                                    df.at[index, greek] = 0
                            
                            # Calculate price difference
                            market_price = row.get('lastPrice', 0)
                            if market_price > 0 and theoretical_price > 0:
                                df.at[index, 'price_diff'] = market_price - theoretical_price
                                df.at[index, 'price_diff_pct'] = ((market_price - theoretical_price) / theoretical_price) * 100
                            else:
                                df.at[index, 'price_diff'] = 0
                                df.at[index, 'price_diff_pct'] = 0
                                
                        except Exception as e:
                            # If calculation fails for this row, set default values
                            df.at[index, 'theoretical_price'] = 0
                            df.at[index, 'days_to_expiry'] = days_to_expiry if 'days_to_expiry' in locals() else 0
                            df.at[index, 'delta'] = 0
                            df.at[index, 'gamma'] = 0  
                            df.at[index, 'theta'] = 0
                            df.at[index, 'vega'] = 0
                            df.at[index, 'rho'] = 0
                            df.at[index, 'price_diff'] = 0
                            df.at[index, 'price_diff_pct'] = 0
                            print(f"Error calculating option {index}: {str(e)}")  # For debugging
                    
                    return df
                
                with st.spinner("Calculating theoretical prices and Greeks..."):
                    try:
                        # Convert dataframe to dict for caching compatibility
                        df_dict = filtered_df.to_dict('records')
                        enhanced_df = calculate_options_pricing(
                            df_dict, current_price, risk_free_rate, 
                            volatility_override, pricing_model
                        )
                        
                        # Use the enhanced dataframe for display
                        
        # Prepare display columns with pricing data
                        base_columns = ['strike', 'lastPrice', 'bid', 'ask', 'volume', 'openInterest', 
                                       'impliedVolatility', 'option_type', 'expiration']
                        pricing_columns = ['theoretical_price', 'delta', 'gamma', 'theta', 'vega', 'calculated_iv', 
                                         'price_diff', 'price_diff_pct', 'days_to_expiry']
                        
                        all_columns = base_columns + [col for col in pricing_columns if col in enhanced_df.columns]
                        available_columns = [col for col in all_columns if col in enhanced_df.columns]
                        display_df = enhanced_df[available_columns].copy()
                        
                        # Add mispricing analysis
                        if 'price_diff_pct' in display_df.columns:
                            display_df['mispricing'] = display_df['price_diff_pct'].apply(
                                lambda x: 'Overpriced' if x > 5 else ('Underpriced' if x < -5 else 'Fair Value') if not pd.isna(x) else 'N/A'
                            )
                          # Sort by strike price
                        display_df = display_df.sort_values('strike')
                        
                        # Create a numeric version of the dataframe for calculations
                        numeric_df = display_df.copy()
                        
                        # Format all numeric columns for display
                        display_df = format_numeric_columns(display_df)
                        
                        # Display the enhanced options table
                        st.markdown("---")  # Add separator
                        st.subheader(f"📊 Options Chain with {pricing_model} Pricing")
                        st.dataframe(
                            display_df,
                            use_container_width=True,
                            height=500,
                            hide_index=True
                        )
                        
                        # Pricing comparison charts
                        if 'theoretical_price' in numeric_df.columns and 'lastPrice' in numeric_df.columns:
                            
                            # Step-by-step pricing breakdown
                            st.markdown("---")  # Add separator
                            st.subheader("🔍 Option Pricing Breakdown")
                            
                            # Create an expander for the detailed breakdown
                            with st.expander("📚 How Option Prices Are Calculated", expanded=False):
                                
                                # Get the first option for demonstration
                                sample_option = numeric_df.iloc[0] if not numeric_df.empty else None
                                
                                if sample_option is not None:
                                    # Extract key parameters
                                    S = current_price
                                    K = sample_option['strike']
                                    T = sample_option.get('days_to_expiry', 30) / 365.0
                                    r = risk_free_rate / 100
                                    sigma = (volatility_override if volatility_override > 0 else sample_option.get('impliedVolatility', 0.25))
                                    option_type = sample_option['option_type']
                                    
                                    st.markdown("### 📊 Pricing Model Parameters")
                                    
                                    # Create three columns for parameter display
                                    col1, col2, col3 = st.columns(3)
                                    
                                    with col1:
                                        st.metric("📈 Stock Price (S)", f"${S:.2f}")
                                        st.metric("🎯 Strike Price (K)", f"${K:.2f}")
                                    
                                    with col2:
                                        st.metric("📅 Time to Expiry (T)", f"{T:.4f} years ({sample_option.get('days_to_expiry', 30)} days)")
                                        st.metric("💰 Risk-Free Rate (r)", f"{r*100:.2f}%")
                                    
                                    with col3:
                                        st.metric("📊 Volatility (σ)", f"{sigma*100:.2f}%")
                                        st.metric("🔄 Option Type", option_type.title())
                                    
                                    st.markdown("---")
                                    
                                    # Show calculation steps based on the pricing model
                                    if pricing_model == "Black-Scholes":
                                        st.markdown("### 🧮 Black-Scholes Calculation Steps")
                                        
                                        # Calculate d1 and d2
                                        d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
                                        d2 = d1 - sigma*np.sqrt(T)
                                        
                                        st.markdown("**Step 1: Calculate d1 and d2**")
                                        st.code(f"""
d1 = [ln(S/K) + (r + σ²/2) × T] / (σ × √T)
d1 = [ln({S:.2f}/{K:.2f}) + ({r:.4f} + {sigma:.4f}²/2) × {T:.4f}] / ({sigma:.4f} × √{T:.4f})
d1 = {d1:.4f}

d2 = d1 - σ × √T
d2 = {d1:.4f} - {sigma:.4f} × √{T:.4f}
d2 = {d2:.4f}
                                        """)
                                        
                                        # Calculate normal distributions
                                        from scipy.stats import norm
                                        N_d1 = norm.cdf(d1)
                                        N_d2 = norm.cdf(d2)
                                        
                                        st.markdown("**Step 2: Calculate Normal Distribution Values**")
                                        st.code(f"""
N(d1) = Normal CDF of d1 = {N_d1:.4f}
N(d2) = Normal CDF of d2 = {N_d2:.4f}
                                        """)
                                        
                                        # Final calculation
                                        if option_type == 'call':
                                            theoretical_price = S * N_d1 - K * np.exp(-r * T) * N_d2
                                            st.markdown("**Step 3: Call Option Price Formula**")
                                            st.code(f"""
Call Price = S × N(d1) - K × e^(-r×T) × N(d2)
Call Price = {S:.2f} × {N_d1:.4f} - {K:.2f} × e^(-{r:.4f}×{T:.4f}) × {N_d2:.4f}
Call Price = ${theoretical_price:.4f}
                                            """)
                                        else:
                                            theoretical_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
                                            st.markdown("**Step 3: Put Option Price Formula**")
                                            st.code(f"""
Put Price = K × e^(-r×T) × N(-d2) - S × N(-d1)
Put Price = {K:.2f} × e^(-{r:.4f}×{T:.4f}) × {norm.cdf(-d2):.4f} - {S:.2f} × {norm.cdf(-d1):.4f}
Put Price = ${theoretical_price:.4f}
                                            """)
                                    
                                    elif pricing_model == "Binomial":
                                        st.markdown("### 🌳 Binomial Tree Calculation Steps")
                                        st.markdown("**Step 1: Calculate Tree Parameters**")
                                        
                                        dt = T / 100  # 100 steps
                                        u = np.exp(sigma * np.sqrt(dt))
                                        d = 1 / u
                                        p = (np.exp(r * dt) - d) / (u - d)
                                        
                                        st.code(f"""
Time Step (dt) = T / Steps = {T:.4f} / 100 = {dt:.6f}
Up Factor (u) = e^(σ × √dt) = e^({sigma:.4f} × √{dt:.6f}) = {u:.4f}
Down Factor (d) = 1/u = {d:.4f}
Risk-Neutral Probability (p) = (e^(r×dt) - d) / (u - d) = {p:.4f}
                                        """)
                                        
                                        st.markdown("**Step 2: Build Price Tree**")
                                        st.markdown("The binomial tree models possible stock price paths over time, with each node representing a possible future stock price.")
                                        
                                        st.markdown("**Step 3: Calculate Option Values**")
                                        st.markdown("Working backward from expiration, calculate option values at each node using risk-neutral valuation.")
                                        
                                        theoretical_price = sample_option.get('theoretical_price', 0)
                                        st.code(f"""
Final Option Price = ${theoretical_price:.4f}
                                        """)
                                    
                                    elif pricing_model == "Monte Carlo":
                                        st.markdown("### 🎲 Monte Carlo Simulation Steps")
                                        st.markdown("**Step 1: Generate Random Price Paths**")
                                        st.code(f"""
Number of Simulations: 100,000
Random Walk Formula: S(t) = S(0) × e^((r - σ²/2)×t + σ×√t×Z)
Where Z ~ Normal(0,1)
                                        """)
                                        
                                        st.markdown("**Step 2: Calculate Payoffs**")
                                        if option_type == 'call':
                                            st.code(f"""
Call Payoff = max(S(T) - K, 0)
Call Payoff = max(Final_Price - {K:.2f}, 0)
                                            """)
                                        else:
                                            st.code(f"""
Put Payoff = max(K - S(T), 0)
Put Payoff = max({K:.2f} - Final_Price, 0)
                                            """)
                                        
                                        st.markdown("**Step 3: Discount Average Payoff**")
                                        theoretical_price = sample_option.get('theoretical_price', 0)
                                        st.code(f"""
Option Price = e^(-r×T) × Average(Payoffs)
Option Price = e^(-{r:.4f}×{T:.4f}) × Average_Payoff
Option Price = ${theoretical_price:.4f}
                                        """)
                                    
                                    # Show the impact of Greeks
                                    st.markdown("---")
                                    st.markdown("### 📊 Greeks Analysis")
                                    st.markdown("The Greeks measure how the option price changes with respect to different factors:")
                                    
                                    if 'delta' in sample_option:
                                        greeks_col1, greeks_col2 = st.columns(2)
                                        
                                        with greeks_col1:
                                            st.markdown(f"""
**Delta (D)**: {sample_option.get('delta', 0):.4f}
- Price change per $1 stock move
- Current impact: ${abs(sample_option.get('delta', 0)):.4f} per $1 stock change

**Gamma (G)**: {sample_option.get('gamma', 0):.4f}
- Delta change per $1 stock move
- Acceleration factor for price changes
                                            """)
                                        
                                        with greeks_col2:
                                            st.markdown(f"""
**Theta (T)**: {sample_option.get('theta', 0):.4f}
- Daily time decay
- Option loses ${abs(sample_option.get('theta', 0)):.4f} per day

**Vega (V)**: {sample_option.get('vega', 0):.4f}
- Price change per 1% volatility change
- Impact: ${sample_option.get('vega', 0):.4f} per 1% vol change
                                            """)
                                    
                                    # Show market vs theoretical comparison
                                    market_price = sample_option.get('lastPrice', 0)
                                    theoretical_price = sample_option.get('theoretical_price', 0)
                                    price_diff = market_price - theoretical_price
                                    price_diff_pct = (price_diff / theoretical_price * 100) if theoretical_price > 0 else 0
                                    
                                    st.markdown("---")
                                    st.markdown("### 💡 Market vs Theoretical Summary")
                                    
                                    result_col1, result_col2, result_col3 = st.columns(3)
                                    
                                    with result_col1:
                                        st.metric("🏪 Market Price", f"${market_price:.4f}")
                                    
                                    with result_col2:
                                        st.metric("🧮 Theoretical Price", f"${theoretical_price:.4f}")
                                    
                                    with result_col3:
                                        st.metric("📊 Difference", 
                                                f"${price_diff:.4f}", 
                                                f"{price_diff_pct:.1f}%",
                                                delta_color="inverse" if abs(price_diff_pct) > 5 else "normal")
                                    
                                    if abs(price_diff_pct) > 10:
                                        st.warning(f"⚠️ Significant mispricing detected! Market price differs by {price_diff_pct:.1f}% from theoretical value.")
                                    elif abs(price_diff_pct) > 5:
                                        st.info(f"ℹ️ Moderate mispricing: {price_diff_pct:.1f}% difference from theoretical value.")
                                    else:
                                        st.success("✅ Market price is reasonably aligned with theoretical value.")
                                
                                else:
                                    st.warning("No option data available for pricing breakdown.")
                            
                            st.markdown("---")  # Add separator
                            st.subheader("💰 Market vs Theoretical Prices")
                            
                            # Create comparison chart
                            fig_pricing = go.Figure()
                            
                            # Separate calls and puts for better visualization
                            calls_df = numeric_df[numeric_df['option_type'] == 'call']
                            puts_df = numeric_df[numeric_df['option_type'] == 'put']
                            
                            if not calls_df.empty:
                                fig_pricing.add_trace(go.Scatter(
                                    x=calls_df['strike'],
                                    y=calls_df['lastPrice'],
                                    mode='markers+lines',
                                    name='Market Price (Calls)',
                                    marker=dict(color='blue', size=8),
                                    line=dict(color='blue')
                                ))
                                
                                fig_pricing.add_trace(go.Scatter(
                                    x=calls_df['strike'],
                                    y=calls_df['theoretical_price'],
                                    mode='markers+lines',
                                    name=f'Theoretical (Calls) - {pricing_model}',
                                    marker=dict(color='lightblue', size=8),
                                    line=dict(color='lightblue', dash='dash')
                                ))
                            
                            if not puts_df.empty:
                                fig_pricing.add_trace(go.Scatter(
                                    x=puts_df['strike'],
                                    y=puts_df['lastPrice'],
                                    mode='markers+lines',
                                    name='Market Price (Puts)',
                                    marker=dict(color='red', size=8),
                                    line=dict(color='red')
                                ))
                                
                                fig_pricing.add_trace(go.Scatter(
                                    x=puts_df['strike'],
                                    y=puts_df['theoretical_price'],
                                    mode='markers+lines',
                                    name=f'Theoretical (Puts) - {pricing_model}',
                                    marker=dict(color='pink', size=8),
                                    line=dict(color='pink', dash='dash')
                                ))
                            
                            fig_pricing.add_vline(x=current_price, line_dash="dot", line_color="green", 
                                                annotation_text="Current Stock Price")
                            
                            fig_pricing.update_layout(
                                title=f"Market vs {pricing_model} Prices Comparison",
                                xaxis_title="Strike Price ($)",
                                yaxis_title="Option Price ($)",
                                hovermode='x unified',
                                height=500
                            )
                            
                            st.plotly_chart(fig_pricing, use_container_width=True)
                            
                            # Mispricing analysis
                            if 'mispricing' in numeric_df.columns:
                                st.markdown("---")  # Add separator
                                st.subheader("🎯 Mispricing Analysis")
                                
                                mispricing_summary = numeric_df['mispricing'].value_counts()
                                col1, col2, col3 = st.columns(3)
                                
                                with col1:
                                    st.metric("Overpriced Options", mispricing_summary.get('Overpriced', 0))
                                with col2:
                                    st.metric("Fair Value Options", mispricing_summary.get('Fair Value', 0))
                                with col3:
                                    st.metric("Underpriced Options", mispricing_summary.get('Underpriced', 0))
                                
                                # Price difference distribution
                                valid_diff = numeric_df['price_diff_pct'].dropna()
                                if not valid_diff.empty:
                                    fig_diff = px.histogram(
                                        valid_diff,
                                        nbins=20,
                                        title="Distribution of Price Differences (%)",
                                        labels={'value': 'Price Difference (%)', 'count': 'Number of Options'}
                                    )
                                    fig_diff.add_vline(x=0, line_dash="dash", line_color="red", 
                                                     annotation_text="Perfect Pricing")
                                    st.plotly_chart(fig_diff, use_container_width=True)
    
                    except Exception as e:
                        st.error(f"Error in pricing calculation: {str(e)}")
                        st.info("Falling back to basic display...")
                        enable_pricing = False
            
            if not enable_pricing:
                # Basic display without pricing
                display_columns = ['strike', 'lastPrice', 'bid', 'ask', 'volume', 'openInterest', 
                                 'impliedVolatility', 'option_type', 'expiration']
                
                available_columns = [col for col in display_columns if col in filtered_df.columns]
                display_df = filtered_df[available_columns].copy()
                
                # Sort by strike price
                display_df = display_df.sort_values('strike')
                
                # Create a numeric version of the dataframe for calculations
                numeric_df = display_df.copy()
                
                # Format all numeric columns for display
                display_df = format_numeric_columns(display_df)
                
                # Display the enhanced options table
                st.subheader("📊 Options Chain")
                st.dataframe(
                    display_df,
                    use_container_width=True,
                    height=400,
                    hide_index=True
                )
            
            # Greeks Analysis (if pricing is enabled)
            if enable_pricing and 'delta' in numeric_df.columns:
                st.markdown("---")  # Add separator
                st.subheader("📈 Greeks Analysis")
                
                greeks_tabs = st.tabs(["Delta & Gamma", "Theta & Vega", "Greeks Summary"])
                
                with greeks_tabs[0]:  # Delta & Gamma
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if 'delta' in numeric_df.columns:
                            fig_delta = px.line(
                                numeric_df,
                                x='strike',
                                y='delta',
                                color='option_type',
                                title="Delta by Strike Price",
                                labels={'delta': 'Delta', 'strike': 'Strike Price'}
                            )
                            fig_delta.add_vline(x=current_price, line_dash="dash", annotation_text="Current Price")
                            st.plotly_chart(fig_delta, use_container_width=True)
                    
                    with col2:
                        if 'gamma' in numeric_df.columns:
                            fig_gamma = px.line(
                                numeric_df,
                                x='strike',
                                y='gamma',
                                color='option_type',
                                title="Gamma by Strike Price",
                                labels={'gamma': 'Gamma', 'strike': 'Strike Price'}
                            )
                            fig_gamma.add_vline(x=current_price, line_dash="dash", annotation_text="Current Price")
                            st.plotly_chart(fig_gamma, use_container_width=True)
                
                with greeks_tabs[1]:  # Theta & Vega
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if 'theta' in numeric_df.columns:
                            fig_theta = px.line(
                                numeric_df,
                                x='strike',
                                y='theta',
                                color='option_type',
                                title="Theta by Strike Price (Daily Decay)",
                                labels={'theta': 'Theta', 'strike': 'Strike Price'}
                            )
                            fig_theta.add_vline(x=current_price, line_dash="dash", annotation_text="Current Price")
                            st.plotly_chart(fig_theta, use_container_width=True)
                    
                    with col2:
                        if 'vega' in numeric_df.columns:
                            fig_vega = px.line(
                                numeric_df,
                                x='strike',
                                y='vega',
                                color='option_type',
                                title="Vega by Strike Price",
                                labels={'vega': 'Vega', 'strike': 'Strike Price'}
                            )
                            fig_vega.add_vline(x=current_price, line_dash="dash", annotation_text="Current Price")
                            st.plotly_chart(fig_vega, use_container_width=True)
                with greeks_tabs[2]:  # Greeks Summary
                    if all(col in numeric_df.columns for col in ['delta', 'gamma', 'theta', 'vega']):
                        # Portfolio Greeks
                        st.subheader("Portfolio Greeks Summary")
                          # Calculate portfolio-level Greeks (assuming 1 contract each)
                        # Use sum() with skipna=True to handle NaN values
                        total_delta = numeric_df['delta'].sum(skipna=True)
                        total_gamma = numeric_df['gamma'].sum(skipna=True)
                        total_theta = numeric_df['theta'].sum(skipna=True)
                        total_vega = numeric_df['vega'].sum(skipna=True)
                        
                        # Ensure we have valid numbers, not NaN
                        total_delta = 0.0 if pd.isna(total_delta) else total_delta
                        total_gamma = 0.0 if pd.isna(total_gamma) else total_gamma
                        total_theta = 0.0 if pd.isna(total_theta) else total_theta
                        total_vega = 0.0 if pd.isna(total_vega) else total_vega
                        
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Total Delta", f"{total_delta:.4f}")
                        with col2:
                            st.metric("Total Gamma", f"{total_gamma:.4f}")
                        with col3:
                            st.metric("Total Theta", f"{total_theta:.4f}")
                        with col4:
                            st.metric("Total Vega", f"{total_vega:.4f}")
                          # Greeks by option type
                        # Use the numeric dataframe for calculations
                        greeks_by_type = numeric_df.groupby('option_type')[['delta', 'gamma', 'theta', 'vega']].sum()
                        
                        # Format the Greeks summary table using our formatting function
                        greeks_formatted = format_numeric_columns(greeks_by_type)
                        
                        st.subheader("Greeks by Option Type")
                        st.dataframe(greeks_formatted, use_container_width=True)
            
            # Original Options Analytics            
            st.markdown("---")  # Add separator
            st.subheader("📊 Market Analytics")
            
            # Volatility smile
            if 'impliedVolatility' in numeric_df.columns and selected_expiration != "All":
                fig_vol = px.scatter(
                    numeric_df, 
                    x='strike', 
                    y='impliedVolatility',
                    color='option_type',
                    title=f"Implied Volatility Smile - {selected_expiration}",
                    labels={'strike': 'Strike Price', 'impliedVolatility': 'Implied Volatility'}
                )
                fig_vol.add_vline(x=current_price, line_dash="dash", annotation_text="Current Price")
                st.plotly_chart(fig_vol, use_container_width=True)
            
            # Volume analysis
            if 'volume' in numeric_df.columns:
                vol_data = numeric_df.groupby(['option_type', 'strike'])['volume'].sum().reset_index()
                fig_volume = px.bar(
                    vol_data,
                    x='strike',
                    y='volume',
                    color='option_type',
                    title="Options Volume by Strike",
                    labels={'strike': 'Strike Price', 'volume': 'Volume'}
                )
                st.plotly_chart(fig_volume, use_container_width=True)
            
        else:
            st.warning("No options data available for the selected filters.")
    
    # Historical stock chart
    if 'symbol' in st.session_state:
        st.markdown("---")  # Add separator
        st.subheader(f"Historical Price Chart - {st.session_state['symbol']}")
        
        period = st.selectbox("Time Period", ["1mo", "3mo", "6mo", "1y", "2y"])
        
        # Use cached data for better performance
        @st.cache_data(ttl=3600)  # Cache for 1 hour
        def get_cached_historical_data(symbol, period):
            return fetcher.get_stock_data(symbol, period)
        
        stock_data = get_cached_historical_data(st.session_state['symbol'], period)
        if stock_data is not None:
            with st.spinner("Generating chart..."):
                fig_stock = go.Figure(data=go.Candlestick(
                    x=stock_data.index,
                    open=stock_data['Open'],
                    high=stock_data['High'],
                    low=stock_data['Low'],
                    close=stock_data['Close']
                ))
                fig_stock.update_layout(
                    title=f"{st.session_state['symbol']} Price Chart",
                    yaxis_title="Price ($)",
                    xaxis_title="Date"
                )
                st.plotly_chart(fig_stock, use_container_width=True)    # Inject the JavaScript for double-click functionality
    st.components.v1.html(DOUBLE_CLICK_JS, height=0, width=0)
    
    # Add a localStorage check component at the end
    st.components.v1.html("""
    <script>
        // Check localStorage for volatility values
        document.addEventListener('DOMContentLoaded', function() {
            const storedValue = localStorage.getItem('double_clicked_iv');
            if (storedValue) {
                const value = parseFloat(storedValue);
                if (!isNaN(value)) {
                    // Clear the stored value
                    localStorage.removeItem('double_clicked_iv');
                    
                    // Redirect with the value as a query parameter
                    window.location.href = '?volatility=' + value;
                }
            }
        });
    </script>
    """, height=0, width=0)

def main():
    """Main function to run the Streamlit app"""
    create_options_dashboard()

if __name__ == "__main__":
    main()