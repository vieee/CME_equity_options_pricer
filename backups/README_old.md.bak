# CME Equity Options Pricer

A comprehensive Python application to fetch and analyze equity options data using free APIs. This tool provides real-time options chains, implied volatility analysis, and interactive visualizations.

## Features

- **Real-time Options Data**: Fetch live options chains from Yahoo Finance
- **Multiple Data Sources**: Support for Yahoo Finance and Alpha Vantage APIs
- **Interactive Dashboard**: Streamlit-based web interface with interactive features
- **Advanced Options Pricing**: Multiple benchmark pricing models including:
  - Black-Scholes-Merton Model
  - Binomial Tree Model (for American options)
  - Monte Carlo Simulation
  - Black-76 Model (for futures options)
  - Implied Volatility calculations
- **Options Analytics**: 
  - Implied volatility smile visualization
  - Volume analysis by strike price
  - Moneyness filtering (ITM, ATM, OTM)
  - Historical price charts
  - Greeks calculation (Delta, Gamma, Theta, Vega, Rho)
  - Market vs Theoretical price comparison
  - Mispricing identification
  - Portfolio-level Greeks summary
- **Interactive UI Features**:
  - Double-click on implied volatility to set as override
  - Real-time filtering and analysis tools
  - Clean, index-free tables with consistent formatting
  - Delta/Theta formatted to 4 decimal places for precision
- **Popular Symbols**: Pre-configured with popular CME-related symbols (SPY, QQQ, etc.)
- **Dual Interface**: Both web UI (Streamlit) and CLI options for flexibility
- **Live Risk-Free Rates**: Access real-time SOFR, ESTR, SONIA, and EONIA rates from central bank APIs
- **Rate Tenor Matching**: Automatic rate suggestions based on option expiration periods

## Installation

1. **Clone or navigate to the project directory**:
   ```bash
   cd c:\Users\Deepak\Documents\projects\finance_projects\CME_equity_options_pricer
   ```

2. **Install required packages**:
   ```bash
   pip install -r requirements.txt
   ```
   
   All dependencies are managed in requirements.txt, including:
   - streamlit for the interactive web UI
   - yfinance for options data
   - pandas, numpy for data manipulation
   - plotly for interactive visualizations
   - scipy for advanced mathematical functions

3. **Set up environment variables** (optional):
   ```bash
   # Create .env file for API keys (optional)
   echo ALPHA_VANTAGE_API_KEY=your_alpha_vantage_key > .env
   echo FRED_API_KEY=your_fred_api_key >> .env
   ```
   
   **API Key Requirements:**
   - **Alpha Vantage**: Optional, for additional data sources (sign up at https://www.alphavantage.co/support/#api-key)
   - **FRED API**: Optional but recommended for live US SOFR rates (free at https://fred.stlouisfed.org/docs/api/api_key.html)
   - **No API keys needed**: ECB (ESTR) and Bank of England (SONIA) rates work without keys
   - **Fallback rates**: Application works fully without any API keys using market estimates

## Usage

### Running the Streamlit Dashboard

```bash
streamlit run app.py
```

This will open a web browser with the interactive dashboard where you can:

1. Select a stock symbol from popular options or enter a custom symbol
2. Fetch real-time options data
3. Enable theoretical pricing with different models (Black-Scholes, Binomial, Monte Carlo)
4. Filter options by type (calls/puts), expiration date, and moneyness
5. View implied volatility smiles and volume analysis
6. Analyze Greeks (Delta, Gamma, Theta, Vega) with 4-decimal precision
7. Compare market prices vs theoretical prices to identify mispricing
8. Analyze historical price charts
9. **Double-click on any implied volatility value** to set it as your override volatility
10. View portfolio-level Greek summaries for selected options

### Running the Command Line Tool

```bash
python cli_pricer.py SYMBOL [--expiration DATE] [--list]
```

Examples:
```bash
python cli_pricer.py SPY                    # Get SPY options data
python cli_pricer.py AAPL --list           # List available expiration dates for AAPL
python cli_pricer.py TSLA -e 2024-01-19    # Get TSLA options for specific expiration
```

### Running the Pricing Demo

```bash
python pricing_demo.py
```

This demonstrates various pricing models and benchmark techniques with sample data.

### Supported Symbols

The application comes pre-configured with popular symbols:
- **ETFs**: SPY, QQQ, IWM
- **Popular Stocks**: AAPL, MSFT, GOOGL, AMZN, TSLA, NVDA, META

You can also enter any custom symbol that has options trading.

## Data Sources

### Primary Source: Yahoo Finance (yfinance)
- **Free**: No API key required
- **Data Available**: Options chains, historical prices, basic stock info
- **Rate Limits**: Reasonable for personal use

### Secondary Source: Alpha Vantage (optional)
- **Free Tier**: 25 requests per day
- **API Key Required**: Sign up at https://www.alphavantage.co/support/#api-key
- **Data Available**: Extended fundamental data

### Risk-Free Rates Data Sources

The application fetches live risk-free rates from multiple central bank APIs:

#### **SOFR (US Rates) - FRED API**
- **Source**: Federal Reserve Economic Data (FRED)
- **API Key**: Optional but recommended for live data
- **Signup**: Free at https://fred.stlouisfed.org/docs/api/api_key.html
- **Fallback**: Market-based estimates when API key not provided

#### **ESTR (European Rates) - ECB**
- **Source**: European Central Bank Data Portal
- **API Key**: Not required
- **Access**: Publicly available real-time data

#### **SONIA (UK Rates) - Bank of England**
- **Source**: Bank of England Statistical Database
- **API Key**: Not required
- **Access**: Publicly available real-time data

#### **EONIA (Legacy European Rates)**
- **Source**: Derived from ESTR (ESTR - 5 basis points)
- **API Key**: Not required

**The application works immediately without any API keys** and provides live rates for European and UK markets, with fallback estimates for US rates.

## API Limitations

- **Yahoo Finance**: No official rate limits but should be used responsibly
- **Alpha Vantage**: 25 requests/day on free tier, 500 requests/minute on premium

## Features Breakdown

### Options Chain Analysis
- View complete options chains for selected symbols
- Filter by calls/puts, expiration dates, and moneyness
- Real-time bid/ask spreads and volume data

### Implied Volatility Analysis
- Volatility smile visualization
- Strike price vs implied volatility plots
- Moneyness-based analysis

### Volume Analysis
- Options volume by strike price
- Call vs put volume comparison
- Open interest analysis

### Historical Analysis
- Candlestick charts for underlying stocks
- Multiple time period options (1mo to 2y)
- Price trend analysis

## Technical Requirements

- Python 3.7+
- Internet connection for API calls
- Modern web browser for Streamlit interface

## Troubleshooting

### Common Issues

1. **No options data available**: Some symbols may not have active options trading
2. **API errors**: Check internet connection and API key configuration
3. **Import errors**: Ensure all packages are installed via `pip install -r requirements.txt`
4. **Formatting errors in metrics**: If you see errors with metrics formatting, ensure you're using the latest version with all fixes

### Error Messages

- "No options data available for [SYMBOL]": The symbol doesn't have options or they're not actively traded
- "Error fetching data": Network or API issue, try again later
- "Failed to format metrics": Typically occurs if Delta/Theta were formatted as strings before calculation (fixed in latest version)

## Options Pricing Models

This application implements several industry-standard benchmark pricing techniques:

### 1. Black-Scholes-Merton Model
- **Best for**: European options, liquid markets
- **Assumptions**: Constant volatility, log-normal price distribution
- **Use case**: Quick theoretical pricing, Greeks calculation
- **Formula**: Uses closed-form solutions for calls and puts

### 2. Binomial Tree Model
- **Best for**: American options, dividend-paying stocks
- **Advantages**: Handles early exercise, flexible assumptions
- **Use case**: American-style options, complex payoffs
- **Method**: Discrete-time lattice approach

### 3. Monte Carlo Simulation
- **Best for**: Complex derivatives, path-dependent options
- **Advantages**: Handles any payoff structure, multiple risk factors
- **Use case**: Exotic options, risk management scenarios
- **Method**: Stochastic simulation with multiple price paths

### 4. Black-76 Model
- **Best for**: Options on futures (common for CME products)
- **Use case**: Commodity options, interest rate derivatives
- **Difference**: Uses forward prices instead of spot prices

### 5. Implied Volatility Analysis
- **Purpose**: Extract market's expectation of future volatility
- **Use case**: Trading decisions, volatility surface construction
- **Method**: Numerical inversion of Black-Scholes formula

### Greeks Calculation
The application calculates all major Greeks:
- **Delta**: Price sensitivity to underlying asset movement
- **Gamma**: Rate of change of Delta
- **Theta**: Time decay (daily)
- **Vega**: Sensitivity to volatility changes
- **Rho**: Sensitivity to interest rate changes

## Interactive Features

- **Double-Click Functionality**: Double-click on any implied volatility value in the options chain to set it as your override volatility
- **Interactive Filters**: Filter options by type, expiration date, and moneyness in real-time
- **Portfolio Greeks**: View total portfolio Greeks for selected options
- **Mispricing Analysis**: Identify potentially mispriced options through market vs. theoretical price comparison
- **Delta/Theta Formatting**: All tables display Delta and Theta to 4 decimal places for precision trading

## Recent Updates

### UI Improvements
- **Improved Double-Click Feature**: Enhanced the double-click functionality on implied volatility values to set as override volatility with visual feedback and toast notifications
- **Better Data Formatting**: Consistent 4-decimal place formatting for Delta and Theta values across all tables
- **Visual Feedback**: Added highlights and notifications for user interactions
- **Index Hiding**: All tables now hide the index column for cleaner presentation
- **Live Risk-Free Rates**: Real-time risk-free rate suggestions from FRED, ECB, and Bank of England APIs

### Risk-Free Rates Integration
- **Multi-Currency Support**: SOFR (USD), ESTR (EUR), SONIA (GBP), and EONIA (EUR) rates
- **Live API Integration**: Real-time rates from Federal Reserve, ECB, and Bank of England
- **Multiple Tenors**: Overnight, 1M, 3M, 6M, and 1Y rates for accurate option pricing
- **Fallback Rates**: Automatic fallback to market-based estimates when APIs are unavailable
- **Smart Suggestions**: Contextual rate recommendations based on option expiry periods

### Technical Improvements
- **Robust Error Handling**: Better handling of NaN and string values in calculations
- **Performance Optimizations**: More efficient data processing for large options chains
- **Fixed Portfolio Greeks Calculations**: Ensured accurate summation of Greek values in portfolio summary
- **Improved Visualization**: Enhanced charts for Greeks, pricing comparisons, and volatility analysis

### Bug Fixes
- Fixed calculation issues with formatted data in Portfolio Greeks summary
- Resolved issues with the double-click functionality for impliedVolatility values
- Fixed edge cases in data formatting and numerical calculations
- Improved error handling for API requests and data processing

## Future Enhancements

- Add more data sources (IEX Cloud, Polygon.io free tier)
- Implement more exotic options pricing models
- Add options strategy builders and analyzers
- Export functionality for data analysis
- Real-time streaming updates
- Portfolio tracking and optimization features

## License

This project is for educational and personal use. Please respect the terms of service of the APIs used.

## Contributing

Feel free to submit issues and enhancement requests!
