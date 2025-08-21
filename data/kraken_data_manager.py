"""
High-Performance Kraken Data Manager
Real-time data acquisition and processing for institutional trading
"""
import asyncio
import websockets
import json
import time
import threading
from collections import deque
from typing import Dict, List, Optional, Callable
import pandas as pd
import numpy as np
import ccxt
from datetime import datetime, timedelta
import logging
from concurrent.futures import ThreadPoolExecutor
import queue

from config.settings import TradingConfig

logger = logging.getLogger(__name__)

class KrakenDataManager:
    def __init__(self, config: TradingConfig):
        self.config = config
        self.exchange = self._initialize_exchange()
        
        # Data storage
        self.ohlcv_data = {}
        self.ticker_data = {}
        self.orderbook_data = {}
        self.trade_data = deque(maxlen=config.DATA_BUFFER_SIZE)
        
        # Threading and async
        self.executor = ThreadPoolExecutor(max_workers=config.ASYNC_WORKERS)
        self.data_queue = queue.Queue(maxsize=10000)
        self.websocket_active = False
        
        # Caching
        self.last_fetch_times = {}
        self.cached_data = {}
        
        # Performance metrics
        self.metrics = {
            'api_calls': 0,
            'websocket_messages': 0,
            'data_points_processed': 0,
            'latency_ms': deque(maxlen=1000)
        }
        
    def _initialize_exchange(self) -> ccxt.kraken:
        """Initialize Kraken exchange with optimal settings"""
        credentials = self.config.get_api_credentials()
        
        exchange_config = {
            'enableRateLimit': True,
            'rateLimit': 1000,  # Kraken allows 1 call per second for public API
            'timeout': 10000,
            'options': {
                'adjustForTimeDifference': True,
            }
        }
        
        if credentials['api_key'] and credentials['api_secret']:
            exchange_config.update({
                'apiKey': credentials['api_key'],
                'secret': credentials['api_secret']
            })
            
        return ccxt.kraken(exchange_config)
    
    async def start_websocket_feeds(self, symbols: List[str], callbacks: Dict[str, Callable]):
        """Start real-time websocket data feeds"""
        if not self.config.WEBSOCKET_ENABLED:
            logger.info("WebSocket disabled in config")
            return
            
        self.websocket_active = True
        
        # Kraken WebSocket API endpoints
        public_ws_url = "wss://ws.kraken.com"
        
        try:
            async with websockets.connect(public_ws_url) as websocket:
                # Subscribe to ticker data
                ticker_subscription = {
                    "event": "subscribe",
                    "pair": symbols,
                    "subscription": {"name": "ticker"}
                }
                await websocket.send(json.dumps(ticker_subscription))
                
                # Subscribe to OHLC data
                ohlc_subscription = {
                    "event": "subscribe",
                    "pair": symbols,
                    "subscription": {"name": "ohlc", "interval": 1}
                }
                await websocket.send(json.dumps(ohlc_subscription))
                
                # Subscribe to order book data
                book_subscription = {
                    "event": "subscribe",
                    "pair": symbols,
                    "subscription": {"name": "book", "depth": 25}
                }
                await websocket.send(json.dumps(book_subscription))
                
                logger.info(f"WebSocket subscriptions sent for {len(symbols)} symbols")
                
                # Process incoming messages
                async for message in websocket:
                    if not self.websocket_active:
                        break
                        
                    try:
                        data = json.loads(message)
                        await self._process_websocket_message(data, callbacks)
                        self.metrics['websocket_messages'] += 1
                    except Exception as e:
                        logger.error(f"WebSocket message processing error: {e}")
                        
        except Exception as e:
            logger.error(f"WebSocket connection error: {e}")
            self.websocket_active = False
    
    async def _process_websocket_message(self, data: Dict, callbacks: Dict[str, Callable]):
        """Process incoming websocket messages"""
        if isinstance(data, list) and len(data) >= 2:
            channel_id = data[0]
            message_data = data[1]
            channel_name = data[2] if len(data) > 2 else None
            
            # Process ticker data
            if channel_name == "ticker":
                symbol = data[3]
                ticker_info = {
                    'symbol': symbol,
                    'bid': float(message_data['b'][0]),
                    'ask': float(message_data['a'][0]),
                    'last': float(message_data['c'][0]),
                    'volume': float(message_data['v'][1]),
                    'timestamp': time.time()
                }
                self.ticker_data[symbol] = ticker_info
                
                if 'ticker' in callbacks:
                    await callbacks['ticker'](ticker_info)
            
            # Process OHLC data
            elif channel_name == "ohlc-1":
                symbol = data[3]
                ohlc_info = {
                    'symbol': symbol,
                    'timestamp': float(message_data[1]),
                    'open': float(message_data[2]),
                    'high': float(message_data[3]),
                    'low': float(message_data[4]),
                    'close': float(message_data[5]),
                    'volume': float(message_data[7])
                }
                
                if symbol not in self.ohlcv_data:
                    self.ohlcv_data[symbol] = deque(maxlen=self.config.CANDLE_LIMIT)
                self.ohlcv_data[symbol].append(ohlc_info)
                
                if 'ohlc' in callbacks:
                    await callbacks['ohlc'](ohlc_info)
            
            # Process order book data
            elif channel_name == "book-25":
                symbol = data[3]
                if 'bs' in message_data:  # Snapshot
                    self.orderbook_data[symbol] = {
                        'bids': [[float(x[0]), float(x[1])] for x in message_data['bs']],
                        'asks': [[float(x[0]), float(x[1])] for x in message_data['as']],
                        'timestamp': time.time()
                    }
                elif 'b' in message_data or 'a' in message_data:  # Update
                    if symbol in self.orderbook_data:
                        if 'b' in message_data:
                            # Update bids
                            for bid in message_data['b']:
                                price, volume = float(bid[0]), float(bid[1])
                                self._update_orderbook_side(self.orderbook_data[symbol]['bids'], price, volume, 'desc')
                        if 'a' in message_data:
                            # Update asks
                            for ask in message_data['a']:
                                price, volume = float(ask[0]), float(ask[1])
                                self._update_orderbook_side(self.orderbook_data[symbol]['asks'], price, volume, 'asc')
                        
                        self.orderbook_data[symbol]['timestamp'] = time.time()
                
                if 'orderbook' in callbacks:
                    await callbacks['orderbook'](self.orderbook_data.get(symbol))
    
    def _update_orderbook_side(self, side: List[List[float]], price: float, volume: float, sort_order: str):
        """Update order book side with new price/volume"""
        # Remove existing price level
        side[:] = [x for x in side if x[0] != price]
        
        # Add new price level if volume > 0
        if volume > 0:
            side.append([price, volume])
            
        # Sort order book
        reverse = sort_order == 'desc'
        side.sort(key=lambda x: x[0], reverse=reverse)
        
        # Keep only top 25 levels
        side[:] = side[:25]
    
    def fetch_historical_data(self, symbol: str, timeframe: str = '1m', 
                            limit: int = None, force_refresh: bool = False) -> pd.DataFrame:
        """Fetch historical OHLCV data with intelligent caching"""
        start_time = time.time()
        
        limit = limit or self.config.CANDLE_LIMIT
        cache_key = f"{symbol}_{timeframe}_{limit}"
        
        # Check cache
        if not force_refresh and cache_key in self.cached_data:
            cache_time = self.last_fetch_times.get(cache_key, 0)
            if time.time() - cache_time < 60:  # 1 minute cache
                logger.debug(f"Using cached data for {cache_key}")
                return self.cached_data[cache_key].copy()
        
        try:
            # Fetch from exchange
            candles = self.exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
            
            # Convert to DataFrame
            df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # Cache the data
            self.cached_data[cache_key] = df.copy()
            self.last_fetch_times[cache_key] = time.time()
            
            # Update metrics
            self.metrics['api_calls'] += 1
            self.metrics['data_points_processed'] += len(df)
            latency = (time.time() - start_time) * 1000
            self.metrics['latency_ms'].append(latency)
            
            logger.debug(f"Fetched {len(df)} candles for {symbol} in {latency:.2f}ms")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching historical data for {symbol}: {e}")
            # Return cached data if available
            if cache_key in self.cached_data:
                logger.warning(f"Returning stale cached data for {cache_key}")
                return self.cached_data[cache_key].copy()
            raise
    
    def get_current_ticker(self, symbol: str) -> Dict:
        """Get current ticker data"""
        # Try websocket data first
        if symbol in self.ticker_data:
            ticker_age = time.time() - self.ticker_data[symbol]['timestamp']
            if ticker_age < 5:  # Use if less than 5 seconds old
                return self.ticker_data[symbol]
        
        # Fallback to REST API
        try:
            ticker = self.exchange.fetch_ticker(symbol)
            self.metrics['api_calls'] += 1
            return ticker
        except Exception as e:
            logger.error(f"Error fetching ticker for {symbol}: {e}")
            return self.ticker_data.get(symbol, {})
    
    def get_orderbook(self, symbol: str, limit: int = 25) -> Dict:
        """Get current order book data"""
        # Try websocket data first
        if symbol in self.orderbook_data:
            book_age = time.time() - self.orderbook_data[symbol]['timestamp']
            if book_age < 2:  # Use if less than 2 seconds old
                return self.orderbook_data[symbol]
        
        # Fallback to REST API
        try:
            orderbook = self.exchange.fetch_order_book(symbol, limit)
            self.metrics['api_calls'] += 1
            return orderbook
        except Exception as e:
            logger.error(f"Error fetching orderbook for {symbol}: {e}")
            return self.orderbook_data.get(symbol, {})
    
    def get_multiple_timeframes(self, symbol: str, timeframes: List[str]) -> Dict[str, pd.DataFrame]:
        """Efficiently fetch multiple timeframes"""
        results = {}
        
        # Use ThreadPoolExecutor for parallel fetching
        future_to_timeframe = {}
        
        with ThreadPoolExecutor(max_workers=len(timeframes)) as executor:
            for tf in timeframes:
                future = executor.submit(self.fetch_historical_data, symbol, tf)
                future_to_timeframe[future] = tf
            
            for future in future_to_timeframe:
                tf = future_to_timeframe[future]
                try:
                    results[tf] = future.result(timeout=10)
                except Exception as e:
                    logger.error(f"Error fetching {tf} data for {symbol}: {e}")
                    results[tf] = pd.DataFrame()
        
        return results
    
    def get_performance_metrics(self) -> Dict:
        """Get data manager performance metrics"""
        latencies = list(self.metrics['latency_ms'])
        
        return {
            'api_calls_total': self.metrics['api_calls'],
            'websocket_messages_total': self.metrics['websocket_messages'],
            'data_points_processed': self.metrics['data_points_processed'],
            'avg_latency_ms': np.mean(latencies) if latencies else 0,
            'p95_latency_ms': np.percentile(latencies, 95) if latencies else 0,
            'websocket_active': self.websocket_active,
            'cached_datasets': len(self.cached_data)
        }
    
    def stop_websocket(self):
        """Stop websocket connections"""
        self.websocket_active = False
        logger.info("WebSocket connections stopped")