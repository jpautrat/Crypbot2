"""
Institutional-Grade Trading Bot Configuration
Optimized for Kraken.com with real-world live data
"""
import os
from dataclasses import dataclass
from typing import Dict, List, Optional
import torch

@dataclass
class TradingConfig:
    # Exchange Configuration
    EXCHANGE = 'kraken'
    BASE_SYMBOL = 'BTC/USD'
    SUPPORTED_PAIRS = [
        'BTC/USD', 'ETH/USD', 'ADA/USD', 'DOT/USD', 'LINK/USD',
        'XRP/USD', 'LTC/USD', 'BCH/USD', 'XLM/USD', 'ATOM/USD'
    ]
    
    # Hardware Optimization
    CPU_CORES = 16
    GPU_ENABLED = torch.cuda.is_available()
    GPU_DEVICE = 'cuda:0' if GPU_ENABLED else 'cpu'
    MAX_MEMORY_GB = 28  # Leave 4GB for system
    
    # Trading Parameters
    MIN_ORDER_SIZE_BTC = 0.00005
    MAX_OPEN_ORDERS = 80
    TRADING_FEE = 0.0026  # Kraken Pro fee (0.26%)
    PROFIT_THRESHOLD = 0.003  # 0.3% minimum profit after fees
    MAX_POSITION_SIZE = 0.02  # 2% of portfolio per position
    
    # Data Configuration
    TIMEFRAMES = ['1m', '5m', '15m', '1h', '4h', '1d']
    PRIMARY_TIMEFRAME = '1m'
    CANDLE_LIMIT = 1000
    WEBSOCKET_ENABLED = True
    
    # Model Configuration
    MODEL_ENSEMBLE_WEIGHTS = {
        'lstm': 0.3,
        'transformer': 0.25,
        'xgboost': 0.2,
        'random_forest': 0.15,
        'autoencoder': 0.1
    }
    
    # Risk Management
    MAX_DAILY_LOSS = 0.05  # 5% max daily loss
    MAX_DRAWDOWN = 0.15    # 15% max drawdown
    VAR_CONFIDENCE = 0.95  # 95% VaR confidence
    POSITION_SIZING_METHOD = 'kelly'  # kelly, fixed, volatility
    
    # Performance Optimization
    BATCH_SIZE = 64
    PREDICTION_CACHE_TTL = 30  # seconds
    DATA_BUFFER_SIZE = 10000
    ASYNC_WORKERS = 8
    
    # Logging and Monitoring
    LOG_LEVEL = 'INFO'
    METRICS_RETENTION_DAYS = 30
    PERFORMANCE_TRACKING = True
    
    @classmethod
    def get_api_credentials(cls) -> Dict[str, Optional[str]]:
        """Get API credentials from environment"""
        return {
            'api_key': os.getenv('KRAKEN_API_KEY') or os.getenv('API_KEY'),
            'api_secret': os.getenv('KRAKEN_API_SECRET') or os.getenv('API_SECRET')
        }
    
    @classmethod
    def validate_hardware(cls) -> Dict[str, bool]:
        """Validate hardware capabilities"""
        import psutil
        
        return {
            'cpu_cores': psutil.cpu_count() >= cls.CPU_CORES,
            'memory': psutil.virtual_memory().total >= cls.MAX_MEMORY_GB * 1024**3,
            'gpu': cls.GPU_ENABLED
        }