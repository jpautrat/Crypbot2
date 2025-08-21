"""
Enhanced Institutional-Grade Kraken Trading Bot
Multi-model ensemble with advanced risk management and real-time optimization
"""
import asyncio
import logging
import time
import signal
import sys
from typing import Dict, List, Optional
from datetime import datetime
import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor
import threading

# Import our enhanced modules
from config.settings import TradingConfig
from data.kraken_data_manager import KrakenDataManager
from models.model_manager import ModelManager
from trading.advanced_strategy import AdvancedTradingStrategy
from risk.risk_manager import RiskManager

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading_bot.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class InstitutionalTradingBot:
    """Enhanced institutional-grade trading bot"""
    
    def __init__(self, config: TradingConfig, simulation_mode: bool = True):
        self.config = config
        self.simulation_mode = simulation_mode
        
        # Initialize components
        self.data_manager = KrakenDataManager(config)
        self.model_manager = ModelManager(config)
        self.risk_manager = RiskManager(config)
        self.strategy = AdvancedTradingStrategy(config, self.model_manager, self.data_manager)
        
        # Bot state
        self.running = False
        self.positions = {}
        self.orders = {}
        self.performance_metrics = {}
        
        # Threading
        self.executor = ThreadPoolExecutor(max_workers=config.ASYNC_WORKERS)
        self.websocket_task = None
        
        # Performance tracking
        self.start_time = time.time()
        self.total_trades = 0
        self.successful_trades = 0
        self.total_pnl = 0.0
        
        logger.info(f"Institutional Trading Bot initialized in {'SIMULATION' if simulation_mode else 'LIVE'} mode")
    
    async def initialize(self):
        """Initialize all bot components"""
        try:
            # Validate hardware
            hardware_status = self.config.validate_hardware()
            logger.info(f"Hardware validation: {hardware_status}")
            
            if not hardware_status['cpu_cores']:
                logger.warning(f"CPU cores below recommended: {self.config.CPU_CORES}")
            
            if not hardware_status['gpu']:
                logger.warning("GPU not available - using CPU for ML models")
            
            # Load models
            models_dir = "models/pretrained"
            if os.path.exists(models_dir):
                loaded_models = self.model_manager.load_models_from_directory(models_dir)
                logger.info(f"Loaded {loaded_models} pre-trained models")
            else:
                logger.warning(f"Models directory {models_dir} not found - creating placeholder models")
                # Create some basic models for demonstration
                await self._create_demo_models()
            
            # Initialize data feeds
            if self.config.WEBSOCKET_ENABLED:
                await self._start_websocket_feeds()
            
            # Initialize portfolio tracking
            await self._initialize_portfolio()
            
            logger.info("Bot initialization completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Bot initialization failed: {e}")
            return False
    
    async def _create_demo_models(self):
        """Create demonstration models when pre-trained models are not available"""
        try:
            from models.model_manager import XGBoostModel, RandomForestModel
            from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
            import pickle
            import os
            
            # Create models directory
            os.makedirs("models/pretrained", exist_ok=True)
            
            # Create a simple Random Forest model
            rf_model = RandomForestClassifier(n_estimators=50, random_state=42)
            
            # Create dummy training data
            import numpy as np
            X_dummy = np.random.randn(1000, 8)  # 8 features
            y_dummy = np.random.randint(0, 2, 1000)  # Binary classification
            
            rf_model.fit(X_dummy, y_dummy)
            
            # Save the model
            with open("models/pretrained/demo_random_forest.pkl", "wb") as f:
                pickle.dump(rf_model, f)
            
            # Create a simple XGBoost-like model using GradientBoosting
            gb_model = GradientBoostingClassifier(n_estimators=50, random_state=42)
            gb_model.fit(X_dummy, y_dummy)
            
            with open("models/pretrained/demo_xgboost.pkl", "wb") as f:
                pickle.dump(gb_model, f)
            
            # Load the demo models
            demo_rf = RandomForestModel("demo_random_forest", self.config)
            demo_rf.load_model("models/pretrained/demo_random_forest.pkl")
            self.model_manager.ensemble.add_model(demo_rf)
            
            demo_xgb = XGBoostModel("demo_xgboost", self.config)
            demo_xgb.load_model("models/pretrained/demo_xgboost.pkl")
            self.model_manager.ensemble.add_model(demo_xgb)
            
            logger.info("Created and loaded demonstration models")
            
        except Exception as e:
            logger.error(f"Error creating demo models: {e}")
    
    async def _start_websocket_feeds(self):
        """Start WebSocket data feeds"""
        try:
            # Define callbacks for different data types
            callbacks = {
                'ticker': self._on_ticker_update,
                'ohlc': self._on_ohlc_update,
                'orderbook': self._on_orderbook_update
            }
            
            # Start WebSocket feeds for supported pairs
            symbols = self.config.SUPPORTED_PAIRS
            self.websocket_task = asyncio.create_task(
                self.data_manager.start_websocket_feeds(symbols, callbacks)
            )
            
            logger.info(f"Started WebSocket feeds for {len(symbols)} symbols")
            
        except Exception as e:
            logger.error(f"Error starting WebSocket feeds: {e}")
    
    async def _on_ticker_update(self, ticker_data: Dict):
        """Handle ticker updates"""
        try:
            symbol = ticker_data['symbol']
            price = ticker_data['last']
            
            # Update risk manager with new price
            if symbol in self.positions:
                position = self.positions[symbol]
                market_prices = {symbol: price}
                self.risk_manager.update_portfolio_value(self.positions, market_prices)
            
        except Exception as e:
            logger.error(f"Error processing ticker update: {e}")
    
    async def _on_ohlc_update(self, ohlc_data: Dict):
        """Handle OHLC updates"""
        try:
            symbol = ohlc_data['symbol']
            
            # Update volatility estimates for risk management
            # This would typically use more historical data
            price_data = self.data_manager.fetch_historical_data(symbol, '1h', 100)
            if not price_data.empty:
                self.risk_manager.update_volatility_estimates(symbol, price_data)
            
        except Exception as e:
            logger.error(f"Error processing OHLC update: {e}")
    
    async def _on_orderbook_update(self, orderbook_data: Dict):
        """Handle order book updates"""
        try:
            # Could be used for market microstructure analysis
            # For now, just log the update
            if orderbook_data:
                symbol = orderbook_data.get('symbol', 'Unknown')
                logger.debug(f"Order book updated for {symbol}")
            
        except Exception as e:
            logger.error(f"Error processing orderbook update: {e}")
    
    async def _initialize_portfolio(self):
        """Initialize portfolio tracking"""
        try:
            if not self.simulation_mode:
                # Get actual balances from exchange
                balance = self.data_manager.exchange.fetch_balance()
                
                for currency, amount in balance['total'].items():
                    if amount > 0:
                        self.positions[currency] = {
                            'quantity': amount,
                            'avg_price': 0.0,  # Would need to track this
                            'timestamp': time.time()
                        }
            else:
                # Initialize simulation portfolio
                self.positions = {
                    'USD': {'quantity': 10000.0, 'avg_price': 1.0, 'timestamp': time.time()}
                }
            
            logger.info(f"Portfolio initialized: {list(self.positions.keys())}")
            
        except Exception as e:
            logger.error(f"Error initializing portfolio: {e}")
            # Default simulation portfolio
            self.positions = {
                'USD': {'quantity': 10000.0, 'avg_price': 1.0, 'timestamp': time.time()}
            }
    
    async def execute_trade(self, signal, current_price: float) -> bool:
        """Execute a trade based on trading signal"""
        try:
            symbol = signal.symbol
            action = signal.action
            
            if action == 'hold':
                return True
            
            # Calculate trade size
            if 'USD' not in self.positions:
                logger.warning("No USD balance available for trading")
                return False
            
            usd_balance = self.positions['USD']['quantity']
            trade_value = min(signal.position_size * usd_balance, usd_balance * 0.95)  # Keep 5% cash
            
            if trade_value < self.config.MIN_ORDER_SIZE_BTC * current_price:
                logger.info(f"Trade size too small: ${trade_value:.2f}")
                return False
            
            # Check risk limits
            proposed_trade = {
                'symbol': symbol,
                'quantity': trade_value / current_price,
                'price': current_price,
                'action': action
            }
            
            risk_ok, violations = self.risk_manager.check_risk_limits(proposed_trade)
            if not risk_ok:
                logger.warning(f"Trade rejected due to risk violations: {violations}")
                return False
            
            # Execute the trade
            if self.simulation_mode:
                success = await self._execute_simulation_trade(signal, current_price, trade_value)
            else:
                success = await self._execute_live_trade(signal, current_price, trade_value)
            
            if success:
                self.total_trades += 1
                logger.info(f"Trade executed: {action} {trade_value:.2f} USD of {symbol} @ ${current_price:.2f}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error executing trade: {e}")
            return False
    
    async def _execute_simulation_trade(self, signal, current_price: float, trade_value: float) -> bool:
        """Execute trade in simulation mode"""
        try:
            symbol = signal.symbol
            action = signal.action
            quantity = trade_value / current_price
            
            if action == 'buy':
                # Buy crypto with USD
                if self.positions['USD']['quantity'] >= trade_value:
                    self.positions['USD']['quantity'] -= trade_value
                    
                    if symbol not in self.positions:
                        self.positions[symbol] = {'quantity': 0.0, 'avg_price': 0.0, 'timestamp': time.time()}
                    
                    # Update average price
                    old_quantity = self.positions[symbol]['quantity']
                    old_avg_price = self.positions[symbol]['avg_price']
                    
                    new_quantity = old_quantity + quantity
                    new_avg_price = ((old_quantity * old_avg_price) + (quantity * current_price)) / new_quantity
                    
                    self.positions[symbol]['quantity'] = new_quantity
                    self.positions[symbol]['avg_price'] = new_avg_price
                    self.positions[symbol]['timestamp'] = time.time()
                    
                    return True
                    
            elif action == 'sell':
                # Sell crypto for USD
                if symbol in self.positions and self.positions[symbol]['quantity'] >= quantity:
                    self.positions[symbol]['quantity'] -= quantity
                    self.positions['USD']['quantity'] += trade_value
                    
                    # Calculate PnL
                    avg_cost = self.positions[symbol]['avg_price']
                    pnl = (current_price - avg_cost) * quantity
                    self.total_pnl += pnl
                    
                    if pnl > 0:
                        self.successful_trades += 1
                    
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error in simulation trade: {e}")
            return False
    
    async def _execute_live_trade(self, signal, current_price: float, trade_value: float) -> bool:
        """Execute trade in live mode"""
        try:
            symbol = signal.symbol
            action = signal.action
            quantity = trade_value / current_price
            
            if action == 'buy':
                order = self.data_manager.exchange.create_market_buy_order(symbol, quantity)
            elif action == 'sell':
                order = self.data_manager.exchange.create_market_sell_order(symbol, quantity)
            else:
                return False
            
            # Update positions based on filled order
            if order['status'] == 'closed':
                # Update local position tracking
                await self._update_position_from_order(order)
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error in live trade execution: {e}")
            return False
    
    async def _update_position_from_order(self, order: Dict):
        """Update position tracking from executed order"""
        try:
            symbol = order['symbol']
            side = order['side']
            amount = order['amount']
            price = order['average'] or order['price']
            
            if symbol not in self.positions:
                self.positions[symbol] = {'quantity': 0.0, 'avg_price': 0.0, 'timestamp': time.time()}
            
            if side == 'buy':
                old_quantity = self.positions[symbol]['quantity']
                old_avg_price = self.positions[symbol]['avg_price']
                
                new_quantity = old_quantity + amount
                new_avg_price = ((old_quantity * old_avg_price) + (amount * price)) / new_quantity
                
                self.positions[symbol]['quantity'] = new_quantity
                self.positions[symbol]['avg_price'] = new_avg_price
                
            elif side == 'sell':
                self.positions[symbol]['quantity'] -= amount
                
                # Calculate realized PnL
                avg_cost = self.positions[symbol]['avg_price']
                pnl = (price - avg_cost) * amount
                self.total_pnl += pnl
                
                if pnl > 0:
                    self.successful_trades += 1
            
            self.positions[symbol]['timestamp'] = time.time()
            
        except Exception as e:
            logger.error(f"Error updating position from order: {e}")
    
    async def trading_loop(self):
        """Main trading loop"""
        logger.info("Starting main trading loop")
        
        last_strategy_update = 0
        last_risk_update = 0
        last_performance_log = 0
        
        while self.running:
            try:
                current_time = time.time()
                
                # Get trading signals for primary symbol
                primary_symbol = self.config.BASE_SYMBOL
                signal = self.strategy.get_trading_decision(primary_symbol)
                
                # Get current market price
                ticker = self.data_manager.get_current_ticker(primary_symbol)
                current_price = ticker.get('last', 0)
                
                if current_price > 0:
                    # Execute trade if signal is strong enough
                    if signal.strength > 0.5 and signal.confidence > 0.6:
                        await self.execute_trade(signal, current_price)
                    
                    # Update risk management
                    if current_time - last_risk_update > 60:  # Update every minute
                        market_prices = {primary_symbol: current_price}
                        self.risk_manager.update_portfolio_value(self.positions, market_prices)
                        last_risk_update = current_time
                    
                    # Update strategy performance
                    if current_time - last_strategy_update > 300:  # Update every 5 minutes
                        # This would typically involve comparing predictions to actual outcomes
                        self.model_manager.ensemble.adjust_weights_based_on_performance()
                        last_strategy_update = current_time
                    
                    # Log performance
                    if current_time - last_performance_log > 600:  # Log every 10 minutes
                        await self._log_performance()
                        last_performance_log = current_time
                
                # Sleep for the configured interval
                await asyncio.sleep(0.1)  # 10 times per second
                
            except Exception as e:
                logger.error(f"Error in trading loop: {e}")
                await asyncio.sleep(1)  # Wait before retrying
    
    async def _log_performance(self):
        """Log current performance metrics"""
        try:
            # Get risk metrics
            risk_metrics = self.risk_manager.get_risk_metrics()
            
            # Calculate runtime
            runtime_hours = (time.time() - self.start_time) / 3600
            
            # Calculate success rate
            success_rate = (self.successful_trades / self.total_trades * 100) if self.total_trades > 0 else 0
            
            # Get model performance
            model_status = self.model_manager.get_status()
            
            logger.info("=== PERFORMANCE SUMMARY ===")
            logger.info(f"Runtime: {runtime_hours:.2f} hours")
            logger.info(f"Total Trades: {self.total_trades}")
            logger.info(f"Success Rate: {success_rate:.1f}%")
            logger.info(f"Total PnL: ${self.total_pnl:.2f}")
            logger.info(f"Portfolio Value: ${risk_metrics.portfolio_value:.2f}")
            logger.info(f"Daily PnL: ${risk_metrics.daily_pnl:.2f}")
            logger.info(f"Current Drawdown: {self.risk_manager.current_drawdown:.2%}")
            logger.info(f"Sharpe Ratio: {risk_metrics.sharpe_ratio:.3f}")
            logger.info(f"Loaded Models: {len(model_status['loaded_models'])}")
            logger.info("========================")
            
        except Exception as e:
            logger.error(f"Error logging performance: {e}")
    
    async def start(self):
        """Start the trading bot"""
        try:
            # Initialize the bot
            if not await self.initialize():
                logger.error("Bot initialization failed")
                return False
            
            self.running = True
            
            # Start the main trading loop
            await self.trading_loop()
            
        except KeyboardInterrupt:
            logger.info("Bot stopped by user")
        except Exception as e:
            logger.error(f"Bot error: {e}")
        finally:
            await self.stop()
    
    async def stop(self):
        """Stop the trading bot"""
        logger.info("Stopping trading bot...")
        
        self.running = False
        
        # Stop WebSocket feeds
        if self.websocket_task:
            self.websocket_task.cancel()
            self.data_manager.stop_websocket()
        
        # Final performance log
        await self._log_performance()
        
        # Generate final risk report
        risk_report = self.risk_manager.generate_risk_report()
        logger.info(f"Final Risk Report: {json.dumps(risk_report, indent=2)}")
        
        logger.info("Bot stopped successfully")

def signal_handler(signum, frame):
    """Handle shutdown signals"""
    logger.info(f"Received signal {signum}, shutting down...")
    sys.exit(0)

async def main():
    """Main entry point"""
    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Enhanced Institutional Trading Bot')
    parser.add_argument('--live', action='store_true', help='Enable live trading')
    parser.add_argument('--config', type=str, help='Configuration file path')
    parser.add_argument('--models-dir', type=str, default='models/pretrained', help='Pre-trained models directory')
    
    args = parser.parse_args()
    
    # Load configuration
    config = TradingConfig()
    
    # Create and start the bot
    simulation_mode = not args.live
    bot = InstitutionalTradingBot(config, simulation_mode)
    
    logger.info(f"Starting Enhanced Institutional Trading Bot")
    logger.info(f"Mode: {'LIVE TRADING' if args.live else 'SIMULATION'}")
    logger.info(f"GPU Available: {config.GPU_ENABLED}")
    logger.info(f"CPU Cores: {config.CPU_CORES}")
    
    await bot.start()

if __name__ == "__main__":
    asyncio.run(main())