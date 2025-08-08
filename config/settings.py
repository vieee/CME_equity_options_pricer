"""
Application configuration and settings
"""
import os
from dataclasses import dataclass, field
from typing import Optional

# Load environment variables from .env if available
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    # If python-dotenv isn't installed, env vars from the OS will still be used
    pass

@dataclass
class APIConfig:
    """API configuration settings"""
    fred_api_key: Optional[str] = os.getenv('FRED_API_KEY')
    alpha_vantage_key: Optional[str] = os.getenv('ALPHA_VANTAGE_API_KEY')
    request_timeout: int = 30
    max_retries: int = 3

@dataclass
class PricingConfig:
    """Options pricing model configuration"""
    default_risk_free_rate: float = 0.05
    default_volatility: float = 0.25
    monte_carlo_simulations: int = 10000
    binomial_tree_steps: int = 100
    cache_ttl_seconds: int = 300

@dataclass
class UIConfig:
    """User interface configuration"""
    page_title: str = "CME Equity Options Pricer"
    page_icon: str = "📈"
    layout: str = "wide"
    max_options_display: int = 500
    decimal_places: int = 4

@dataclass
class AppConfig:
    """Main application configuration"""
    api: APIConfig = field(default_factory=APIConfig)
    pricing: PricingConfig = field(default_factory=PricingConfig)
    ui: UIConfig = field(default_factory=UIConfig)
    debug: bool = field(default_factory=lambda: os.getenv('DEBUG', 'False').lower() == 'true')

# Global configuration instance
config = AppConfig()
