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
warnings.filterwarnings('ignore')

# Load environment variables
load_dotenv()

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
      # Initialize data fetcher
    fetcher = OptionsDataFetcher()
    pricer = OptionsPricingModels()
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader(f"Options Chain for {symbol}")
        
        if st.button("Fetch Options Data", type="primary"):
            with st.spinner("Fetching options data..."):
                options_data, current_price = fetcher.get_yahoo_options_data(symbol)
                
                if options_data is not None:
                    st.session_state['options_data'] = options_data
                    st.session_state['current_price'] = current_price
                    st.session_state['symbol'] = symbol
                else:
                    st.error(current_price)  # Error message
    
    with col2:
        st.subheader("Stock Information")
        
        if 'current_price' in st.session_state:
            st.metric("Current Price", f"${st.session_state['current_price']:.2f}")
            
            # Get additional stock data
            stock_data = fetcher.get_stock_data(symbol, "1mo")
            if stock_data is not None and len(stock_data) > 1:
                price_change = stock_data['Close'].iloc[-1] - stock_data['Close'].iloc[-2]
                price_change_pct = (price_change / stock_data['Close'].iloc[-2]) * 100
                st.metric("Daily Change", f"${price_change:.2f}", f"{price_change_pct:.2f}%")
    
    # Display options data
    if 'options_data' in st.session_state:
        options_df = st.session_state['options_data']
        current_price = st.session_state['current_price']
        
        # Filter options
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
          # Display filtered data
        if len(filtered_df) > 0:            # Add pricing analysis section
            st.subheader("🧮 Options Pricing Analysis")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                enable_pricing = st.checkbox("Enable Theoretical Pricing", value=True)
            with col2:
                risk_free_rate = st.number_input("Risk-free Rate (%)", min_value=0.0, max_value=10.0, value=5.0, step=0.1) / 100
            with col3:
                pricing_model = st.selectbox("Pricing Model", ["Black-Scholes", "Binomial Tree", "Monte Carlo"])
            with col4:
                volatility_override = st.number_input("Override Volatility (%)", min_value=0.0, max_value=100.0, value=st.session_state.double_clicked_iv, step=1.0) / 100
            
            if enable_pricing:
                with st.spinner("Calculating theoretical prices and Greeks..."):
                    try:
                        # Price the options using the selected model
                        priced_df = pricer.price_options_dataframe(filtered_df, current_price, risk_free_rate)
                        
                        # Calculate prices with different models if requested
                        if pricing_model != "Black-Scholes":
                            theoretical_prices = []
                            for idx, row in filtered_df.iterrows():
                                try:
                                    T = pricer.calculate_time_to_expiration(row['expiration'])
                                    K = row['strike']
                                    option_type = row['option_type']
                                    
                                    if T > 0:
                                        if pricing_model == "Binomial Tree":
                                            price = pricer.binomial_option_price(
                                                current_price, K, T, risk_free_rate, volatility_override, 
                                                n=100, option_type=option_type, american=True
                                            )
                                        elif pricing_model == "Monte Carlo":
                                            price = pricer.monte_carlo_option_price(
                                                current_price, K, T, risk_free_rate, volatility_override,
                                                n_simulations=10000, option_type=option_type
                                            )
                                        else:
                                            price = 0
                                        theoretical_prices.append(price)
                                    else:
                                        theoretical_prices.append(0)
                                except:
                                    theoretical_prices.append(0)
                            
                            # Update theoretical prices with selected model
                            priced_df['theoretical_price'] = theoretical_prices
                        
                        # Prepare display columns with pricing data
                        base_columns = ['strike', 'lastPrice', 'bid', 'ask', 'volume', 'openInterest', 
                                       'impliedVolatility', 'option_type', 'expiration']
                        pricing_columns = ['theoretical_price', 'delta', 'gamma', 'theta', 'vega', 'calculated_iv']
                        
                        all_columns = base_columns + [col for col in pricing_columns if col in priced_df.columns]
                        available_columns = [col for col in all_columns if col in priced_df.columns]
                        display_df = priced_df[available_columns].copy()
                        
                        # Add price difference analysis
                        if 'theoretical_price' in display_df.columns and 'lastPrice' in display_df.columns:
                            display_df['price_diff'] = display_df['lastPrice'] - display_df['theoretical_price']
                            display_df['price_diff_pct'] = (display_df['price_diff'] / display_df['theoretical_price'].replace(0, np.nan)) * 100
                            display_df['mispricing'] = display_df['price_diff_pct'].apply(
                                lambda x: 'Overpriced' if x > 5 else ('Underpriced' if x < -5 else 'Fair Value') if not pd.isna(x) else 'N/A'
                            )
                          # Sort by strike price
                        display_df = display_df.sort_values('strike')
                        
                        # Create a numeric version of the dataframe for calculations
                        numeric_df = display_df.copy()
                        
                        # Format Delta and Theta to 4 decimal places
                        if 'delta' in display_df.columns:
                            display_df['delta'] = display_df['delta'].apply(lambda x: f"{x:.4f}" if pd.notna(x) else "N/A")
                        if 'theta' in display_df.columns:
                            display_df['theta'] = display_df['theta'].apply(lambda x: f"{x:.4f}" if pd.notna(x) else "N/A")
                        
                        # Display the enhanced options table
                        st.subheader(f"📊 Options Chain with {pricing_model} Pricing")
                        st.dataframe(
                            display_df,
                            use_container_width=True,
                            height=500,
                            hide_index=True
                        )
                        
                        # Pricing comparison charts
                        if 'theoretical_price' in display_df.columns and 'lastPrice' in display_df.columns:
                            st.subheader("💰 Market vs Theoretical Prices")
                            
                            # Create comparison chart
                            fig_pricing = go.Figure()
                            
                            # Separate calls and puts for better visualization
                            calls_df = display_df[display_df['option_type'] == 'call']
                            puts_df = display_df[display_df['option_type'] == 'put']
                            
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
                            if 'mispricing' in display_df.columns:
                                st.subheader("🎯 Mispricing Analysis")
                                
                                mispricing_summary = display_df['mispricing'].value_counts()
                                col1, col2, col3 = st.columns(3)
                                
                                with col1:
                                    st.metric("Overpriced Options", mispricing_summary.get('Overpriced', 0))
                                with col2:
                                    st.metric("Fair Value Options", mispricing_summary.get('Fair Value', 0))
                                with col3:
                                    st.metric("Underpriced Options", mispricing_summary.get('Underpriced', 0))
                                
                                # Price difference distribution
                                valid_diff = display_df['price_diff_pct'].dropna()
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
                
                # Format Delta and Theta to 4 decimal places
                if 'delta' in display_df.columns:
                    display_df['delta'] = display_df['delta'].apply(lambda x: f"{x:.4f}" if pd.notna(x) else "N/A")
                if 'theta' in display_df.columns:
                    display_df['theta'] = display_df['theta'].apply(lambda x: f"{x:.4f}" if pd.notna(x) else "N/A")
                
                # Display the enhanced options table
                st.subheader("📊 Options Chain")
                st.dataframe(
                    display_df,
                    use_container_width=True,
                    height=400,
                    hide_index=True
                )
            
            # Greeks Analysis (if pricing is enabled)
            if enable_pricing and 'delta' in display_df.columns:
                st.subheader("📈 Greeks Analysis")
                
                greeks_tabs = st.tabs(["Delta & Gamma", "Theta & Vega", "Greeks Summary"])
                
                with greeks_tabs[0]:  # Delta & Gamma
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if 'delta' in display_df.columns:
                            fig_delta = px.line(
                                display_df,
                                x='strike',
                                y='delta',
                                color='option_type',
                                title="Delta by Strike Price",
                                labels={'delta': 'Delta', 'strike': 'Strike Price'}
                            )
                            fig_delta.add_vline(x=current_price, line_dash="dash", annotation_text="Current Price")
                            st.plotly_chart(fig_delta, use_container_width=True)
                    
                    with col2:
                        if 'gamma' in display_df.columns:
                            fig_gamma = px.line(
                                display_df,
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
                        if 'theta' in display_df.columns:
                            fig_theta = px.line(
                                display_df,
                                x='strike',
                                y='theta',
                                color='option_type',
                                title="Theta by Strike Price (Daily Decay)",
                                labels={'theta': 'Theta', 'strike': 'Strike Price'}
                            )
                            fig_theta.add_vline(x=current_price, line_dash="dash", annotation_text="Current Price")
                            st.plotly_chart(fig_theta, use_container_width=True)
                    
                    with col2:
                        if 'vega' in display_df.columns:
                            fig_vega = px.line(
                                display_df,
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
                        
                        # Format the Greeks summary table
                        greeks_formatted = greeks_by_type.copy()
                        greeks_formatted['delta'] = greeks_formatted['delta'].apply(lambda x: f"{x:.4f}")
                        greeks_formatted['theta'] = greeks_formatted['theta'].apply(lambda x: f"{x:.4f}")
                        greeks_formatted['gamma'] = greeks_formatted['gamma'].apply(lambda x: f"{x:.4f}")
                        greeks_formatted['vega'] = greeks_formatted['vega'].apply(lambda x: f"{x:.4f}")
                        
                        st.subheader("Greeks by Option Type")
                        st.dataframe(greeks_formatted, use_container_width=True)
            
            # Original Options Analytics
            st.subheader("📊 Market Analytics")
            
            # Volatility smile
            if 'impliedVolatility' in display_df.columns and selected_expiration != "All":
                fig_vol = px.scatter(
                    display_df, 
                    x='strike', 
                    y='impliedVolatility',
                    color='option_type',
                    title=f"Implied Volatility Smile - {selected_expiration}",
                    labels={'strike': 'Strike Price', 'impliedVolatility': 'Implied Volatility'}
                )
                fig_vol.add_vline(x=current_price, line_dash="dash", annotation_text="Current Price")
                st.plotly_chart(fig_vol, use_container_width=True)
            
            # Volume analysis
            if 'volume' in display_df.columns:
                vol_data = display_df.groupby(['option_type', 'strike'])['volume'].sum().reset_index()
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
        st.subheader(f"Historical Price Chart - {st.session_state['symbol']}")
        
        period = st.selectbox("Time Period", ["1mo", "3mo", "6mo", "1y", "2y"])
        
        stock_data = fetcher.get_stock_data(st.session_state['symbol'], period)
        if stock_data is not None:
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