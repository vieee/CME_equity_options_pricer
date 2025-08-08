# CME Equity Options Pricer

A comprehensive, modular options pricing application with both Streamlit web interface and command-line interface.

## 🏗️ Architecture

### Modular Structure
```
├── src/                          # Source code
│   ├── core/                     # Core business logic
│   ├── models/                   # Financial models (pricing, Greeks)
│   │   └── pricing.py           # Options pricing engine
│   ├── data/                     # Data providers and fetchers
│   │   ├── providers.py         # Stock/options data providers
│   │   └── rates.py            # Risk-free rates provider
│   ├── utils/                    # Utilities and helpers
│   │   ├── cache.py             # Caching utilities
│   │   ├── formatters.py        # Data formatting
│   │   └── validators.py        # Input validation
│   └── ui/                       # User interface components
│       └── components.py        # Streamlit UI components
├── config/                       # Configuration
│   └── settings.py              # Application settings
├── tests/                        # Test suite
│   └── test_pricing.py          # Pricing model tests
├── main.py                       # Main Streamlit application
├── cli.py                        # Command-line interface
└── requirements_new.txt          # Dependencies
```

## 🚀 Features

### Core Functionality
- **Multiple Pricing Models**: Black-Scholes, Binomial Tree, Monte Carlo
- **Greeks Calculation**: Delta, Gamma, Theta, Vega, Rho
- **Live Market Data**: Real-time stock and options data via yfinance
- **Risk-Free Rates**: Live rates from FRED, ECB, Bank of England APIs
- **Data Validation**: Comprehensive input validation and error handling
- **Performance Optimized**: Caching, lazy loading, reduced computations

### User Interfaces
- **Web Interface**: Full-featured Streamlit application
- **Command Line**: CLI for automated analysis and scripting
- **Modular Components**: Reusable UI components for easy customization

### Advanced Analytics
- **Options Chain Analysis**: Filter by type, expiry, moneyness
- **Theoretical vs Market Pricing**: Compare model prices with market prices
- **Step-by-Step Breakdown**: Detailed pricing calculations
- **Interactive Charts**: Visualizations for Greeks and price sensitivity

## 📦 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup
1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd CME_equity_options_pricer
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements_new.txt
   ```

3. **Optional: Configure API keys**
   ```bash
   # For enhanced US rates data
   export FRED_API_KEY="your_fred_api_key"
   
   # Or create a .env file
   echo "FRED_API_KEY=your_fred_api_key" > .env
   ```

## 🖥️ Usage

### Web Application
```bash
streamlit run main.py
```
Access at: `http://localhost:8501`

### Command Line Interface
```bash
# Get options chain
python cli.py AAPL --show-chain

# Price specific option
python cli.py AAPL --strike 150 --expiry 2024-12-20 --option-type call

# Price with custom parameters
python cli.py AAPL --strike 150 --expiry 2024-12-20 --option-type put --rate 0.045 --volatility 0.30

# Get only the theoretical price (for scripting)
python cli.py AAPL --strike 150 --expiry 2024-12-20 --price-only
```

### Python API
```python
from src.models.pricing import OptionsPricingEngine
from src.data.providers import MarketDataProvider
from src.data.rates import get_risk_free_rates

# Initialize components
pricing_engine = OptionsPricingEngine()
market_provider = MarketDataProvider()
rates_provider = get_risk_free_rates()

# Get market data
market_data = market_provider.get_complete_market_data("AAPL")
current_price = market_data['current_price']

# Calculate option price
call_price = pricing_engine.black_scholes_call(
    S=current_price, K=150, T=0.25, r=0.05, sigma=0.25
)

# Calculate Greeks
greeks = pricing_engine.calculate_greeks(
    S=current_price, K=150, T=0.25, r=0.05, sigma=0.25, option_type='call'
)
```

## 🧪 Testing

Run the test suite:
```bash
python -m pytest tests/
# or
python tests/test_pricing.py
```

### Test Coverage
- Options pricing models (Black-Scholes, Binomial, Monte Carlo)
- Put-call parity validation
- Greeks calculations
- Data validation functions
- Formatting utilities

## ⚙️ Configuration

### Application Settings
Edit `config/settings.py` to customize:
- API timeouts and retry settings
- UI layout and display options
- Pricing model parameters
- Cache TTL settings

### Environment Variables
- `FRED_API_KEY`: Federal Reserve Economic Data API key
- `ALPHA_VANTAGE_API_KEY`: Alpha Vantage API key (optional)
- `DEBUG`: Enable debug mode (True/False)

## 📊 Data Sources

### Market Data
- **Primary**: Yahoo Finance (yfinance) - Free, real-time data
- **Options**: Full options chains with Greeks
- **Historical**: Price history and volatility calculations

### Risk-Free Rates
- **SOFR** (USD): Federal Reserve (FRED API)
- **ESTR** (EUR): European Central Bank
- **SONIA** (GBP): Bank of England
- **EONIA** (EUR): Legacy rate (ESTR-based)

### Fallback Mechanisms
- Automatic fallback to cached/default rates when APIs are unavailable
- Graceful degradation for offline usage

## 🔧 Development

### Code Structure
- **Separation of Concerns**: Clear separation between data, models, UI, and utilities
- **Type Hints**: Comprehensive type annotations
- **Error Handling**: Robust error handling and validation
- **Documentation**: Inline documentation and docstrings

### Adding New Features
1. **Models**: Add new pricing models in `src/models/`
2. **Data Providers**: Extend data sources in `src/data/`
3. **UI Components**: Create reusable components in `src/ui/`
4. **Tests**: Add tests in `tests/` directory

### Performance Optimization
- Streamlit caching for expensive operations
- Lazy loading of data providers
- Efficient vectorized calculations
- Memory-conscious data handling

## 📈 Advanced Features

### Pricing Models
- **Black-Scholes-Merton**: European options with dividends
- **Binomial Trees**: American-style exercise capabilities
- **Monte Carlo**: Path-dependent options support
- **Implied Volatility**: Reverse-engineering market expectations

### Analytics
- **Greeks Visualization**: Interactive charts for sensitivity analysis
- **Scenario Analysis**: What-if calculations with parameter variations
- **Model Comparison**: Side-by-side comparison of different pricing models
- **Risk Metrics**: Value-at-Risk and other risk measures

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- **Yahoo Finance** for market data
- **Federal Reserve** for US interest rates
- **European Central Bank** for EUR rates
- **Bank of England** for GBP rates
- **Streamlit** for the web framework
- **SciPy** for numerical computations
