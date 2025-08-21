"""
Advanced Institutional Trading Strategy
Multi-timeframe, multi-model approach with sophisticated risk management
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import logging
from dataclasses import dataclass
from enum import Enum
import time

from config.settings import TradingConfig
from models.model_manager import ModelManager
from data.kraken_data_manager import KrakenDataManager

logger = logging.getLogger(__name__)

class SignalStrength(Enum):
    VERY_WEAK = 0.1
    WEAK = 0.3
    MODERATE = 0.5
    STRONG = 0.7
    VERY_STRONG = 0.9

class MarketRegime(Enum):
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    SIDEWAYS = "sideways"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"

@dataclass
class TradingSignal:
    action: str  # 'buy', 'sell', 'hold'
    strength: float  # 0.0 to 1.0
    confidence: float  # 0.0 to 1.0
    price_target: Optional[float]
    stop_loss: Optional[float]
    position_size: float
    reasoning: List[str]
    timestamp: float
    timeframe: str
    symbol: str

class AdvancedTradingStrategy:
    """Institutional-grade trading strategy with multiple models and timeframes"""
    
    def __init__(self, config: TradingConfig, model_manager: ModelManager, data_manager: KrakenDataManager):
        self.config = config
        self.model_manager = model_manager
        self.data_manager = data_manager
        
        # Strategy parameters
        self.timeframes = config.TIMEFRAMES
        self.primary_timeframe = config.PRIMARY_TIMEFRAME
        
        # Market analysis
        self.current_regime = MarketRegime.SIDEWAYS
        self.volatility_threshold = 0.02  # 2% daily volatility threshold
        
        # Signal aggregation
        self.signal_history = []
        self.max_signal_history = 100
        
        # Performance tracking
        self.strategy_performance = {
            'total_signals': 0,
            'successful_signals': 0,
            'failed_signals': 0,
            'avg_return': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0
        }
    
    def analyze_market_regime(self, data: Dict[str, pd.DataFrame]) -> MarketRegime:
        """Analyze current market regime across timeframes"""
        try:
            # Use daily data for regime analysis
            daily_data = data.get('1d', data.get('4h', data.get('1h')))
            if daily_data is None or len(daily_data) < 20:
                return MarketRegime.SIDEWAYS
            
            # Calculate trend indicators
            daily_data = daily_data.copy()
            daily_data['sma_20'] = daily_data['close'].rolling(20).mean()
            daily_data['sma_50'] = daily_data['close'].rolling(50).mean()
            daily_data['returns'] = daily_data['close'].pct_change()
            
            # Volatility analysis
            volatility = daily_data['returns'].rolling(20).std() * np.sqrt(365)  # Annualized
            current_volatility = volatility.iloc[-1] if not volatility.empty else 0
            
            # Trend analysis
            current_price = daily_data['close'].iloc[-1]
            sma_20 = daily_data['sma_20'].iloc[-1]
            sma_50 = daily_data['sma_50'].iloc[-1]
            
            # Determine regime
            if current_volatility > self.volatility_threshold * 2:
                regime = MarketRegime.HIGH_VOLATILITY
            elif current_volatility < self.volatility_threshold * 0.5:
                regime = MarketRegime.LOW_VOLATILITY
            elif current_price > sma_20 > sma_50:
                regime = MarketRegime.TRENDING_UP
            elif current_price < sma_20 < sma_50:
                regime = MarketRegime.TRENDING_DOWN
            else:
                regime = MarketRegime.SIDEWAYS
            
            self.current_regime = regime
            logger.debug(f"Market regime: {regime.value}, Volatility: {current_volatility:.4f}")
            
            return regime
            
        except Exception as e:
            logger.error(f"Error analyzing market regime: {e}")
            return MarketRegime.SIDEWAYS
    
    def calculate_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate comprehensive technical indicators"""
        df = data.copy()
        
        # Price-based indicators
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        
        # Moving averages
        for period in [5, 10, 20, 50, 100, 200]:
            df[f'sma_{period}'] = df['close'].rolling(period).mean()
            df[f'ema_{period}'] = df['close'].ewm(span=period).mean()
        
        # Bollinger Bands
        df['bb_middle'] = df['close'].rolling(20).mean()
        bb_std = df['close'].rolling(20).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # RSI
        df['rsi'] = self._calculate_rsi(df['close'])
        
        # MACD
        df['macd'], df['macd_signal'], df['macd_histogram'] = self._calculate_macd(df['close'])
        
        # Stochastic
        df['stoch_k'], df['stoch_d'] = self._calculate_stochastic(df)
        
        # Volume indicators
        df['volume_sma_20'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma_20']
        df['price_volume'] = df['close'] * df['volume']
        df['vwap'] = df['price_volume'].rolling(20).sum() / df['volume'].rolling(20).sum()
        
        # Volatility indicators
        df['atr'] = self._calculate_atr(df)
        df['volatility'] = df['returns'].rolling(20).std()
        
        # Support and resistance levels
        df['resistance'] = df['high'].rolling(20).max()
        df['support'] = df['low'].rolling(20).min()
        df['price_position'] = (df['close'] - df['support']) / (df['resistance'] - df['support'])
        
        return df
    
    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate MACD"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal).mean()
        macd_histogram = macd - macd_signal
        return macd, macd_signal, macd_histogram
    
    def _calculate_stochastic(self, data: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> Tuple[pd.Series, pd.Series]:
        """Calculate Stochastic oscillator"""
        lowest_low = data['low'].rolling(k_period).min()
        highest_high = data['high'].rolling(k_period).max()
        k_percent = 100 * ((data['close'] - lowest_low) / (highest_high - lowest_low))
        d_percent = k_percent.rolling(d_period).mean()
        return k_percent, d_percent
    
    def _calculate_atr(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        high_low = data['high'] - data['low']
        high_close = np.abs(data['high'] - data['close'].shift())
        low_close = np.abs(data['low'] - data['close'].shift())
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        atr = true_range.rolling(period).mean()
        return atr
    
    def generate_multi_timeframe_signals(self, symbol: str) -> Dict[str, TradingSignal]:
        """Generate trading signals across multiple timeframes"""
        signals = {}
        
        try:
            # Get data for all timeframes
            multi_tf_data = self.data_manager.get_multiple_timeframes(symbol, self.timeframes)
            
            # Analyze market regime
            regime = self.analyze_market_regime(multi_tf_data)
            
            # Generate signals for each timeframe
            for timeframe, data in multi_tf_data.items():
                if len(data) < 50:  # Need sufficient data
                    continue
                
                # Calculate technical indicators
                data_with_indicators = self.calculate_technical_indicators(data)
                
                # Get ML model predictions
                ml_prediction = self.model_manager.get_prediction(data_with_indicators)
                
                # Generate signal for this timeframe
                signal = self._generate_timeframe_signal(
                    symbol, timeframe, data_with_indicators, ml_prediction, regime
                )
                
                signals[timeframe] = signal
            
            return signals
            
        except Exception as e:
            logger.error(f"Error generating multi-timeframe signals: {e}")
            return {}
    
    def _generate_timeframe_signal(self, symbol: str, timeframe: str, data: pd.DataFrame, 
                                 ml_prediction: Dict, regime: MarketRegime) -> TradingSignal:
        """Generate trading signal for a specific timeframe"""
        try:
            current_data = data.iloc[-1]
            reasoning = []
            
            # Initialize signal components
            technical_score = 0.0
            ml_score = ml_prediction.get('prediction', 0.0)
            confidence = ml_prediction.get('confidence', 0.5)
            
            # Technical analysis scoring
            
            # Trend following signals
            if current_data['close'] > current_data['sma_20']:
                technical_score += 0.2
                reasoning.append("Price above SMA20")
            
            if current_data['sma_20'] > current_data['sma_50']:
                technical_score += 0.15
                reasoning.append("SMA20 above SMA50")
            
            # Momentum signals
            if current_data['rsi'] < 30:
                technical_score += 0.3  # Oversold
                reasoning.append("RSI oversold")
            elif current_data['rsi'] > 70:
                technical_score -= 0.3  # Overbought
                reasoning.append("RSI overbought")
            
            # MACD signals
            if current_data['macd'] > current_data['macd_signal']:
                technical_score += 0.1
                reasoning.append("MACD bullish")
            
            # Bollinger Bands
            if current_data['bb_position'] < 0.2:
                technical_score += 0.2  # Near lower band
                reasoning.append("Near BB lower band")
            elif current_data['bb_position'] > 0.8:
                technical_score -= 0.2  # Near upper band
                reasoning.append("Near BB upper band")
            
            # Volume confirmation
            if current_data['volume_ratio'] > 1.5:
                technical_score += 0.1
                reasoning.append("High volume")
            
            # Combine technical and ML scores
            combined_score = (technical_score * 0.6) + (ml_score * 0.4)
            
            # Adjust for market regime
            regime_adjustment = self._get_regime_adjustment(regime, combined_score)
            final_score = combined_score * regime_adjustment
            
            # Determine action and strength
            if final_score > 0.3:
                action = 'buy'
                strength = min(final_score, 1.0)
            elif final_score < -0.3:
                action = 'sell'
                strength = min(abs(final_score), 1.0)
            else:
                action = 'hold'
                strength = 0.0
            
            # Calculate position size based on Kelly criterion
            position_size = self._calculate_position_size(strength, confidence, current_data['volatility'])
            
            # Calculate price targets and stop loss
            current_price = current_data['close']
            atr = current_data['atr']
            
            if action == 'buy':
                price_target = current_price + (atr * 2)
                stop_loss = current_price - (atr * 1.5)
            elif action == 'sell':
                price_target = current_price - (atr * 2)
                stop_loss = current_price + (atr * 1.5)
            else:
                price_target = None
                stop_loss = None
            
            # Add ML reasoning
            if ml_prediction.get('individual_predictions'):
                for model_name, pred in ml_prediction['individual_predictions'].items():
                    reasoning.append(f"{model_name}: {pred['prediction']:.3f}")
            
            return TradingSignal(
                action=action,
                strength=strength,
                confidence=confidence,
                price_target=price_target,
                stop_loss=stop_loss,
                position_size=position_size,
                reasoning=reasoning,
                timestamp=time.time(),
                timeframe=timeframe,
                symbol=symbol
            )
            
        except Exception as e:
            logger.error(f"Error generating signal for {timeframe}: {e}")
            return TradingSignal(
                action='hold',
                strength=0.0,
                confidence=0.0,
                price_target=None,
                stop_loss=None,
                position_size=0.0,
                reasoning=[f"Error: {str(e)}"],
                timestamp=time.time(),
                timeframe=timeframe,
                symbol=symbol
            )
    
    def _get_regime_adjustment(self, regime: MarketRegime, score: float) -> float:
        """Adjust signal strength based on market regime"""
        adjustments = {
            MarketRegime.TRENDING_UP: 1.2 if score > 0 else 0.8,
            MarketRegime.TRENDING_DOWN: 1.2 if score < 0 else 0.8,
            MarketRegime.SIDEWAYS: 0.7,  # Reduce signal strength in sideways markets
            MarketRegime.HIGH_VOLATILITY: 0.8,  # Be more cautious in high volatility
            MarketRegime.LOW_VOLATILITY: 1.1   # Slightly more aggressive in low volatility
        }
        
        return adjustments.get(regime, 1.0)
    
    def _calculate_position_size(self, strength: float, confidence: float, volatility: float) -> float:
        """Calculate position size using Kelly criterion and risk management"""
        try:
            # Base Kelly calculation
            win_rate = confidence
            avg_win = strength * 0.02  # Assume 2% average win
            avg_loss = 0.01  # Assume 1% average loss
            
            if avg_loss > 0:
                kelly_fraction = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
            else:
                kelly_fraction = 0.0
            
            # Apply safety constraints
            kelly_fraction = max(0.0, min(kelly_fraction, 0.25))  # Cap at 25%
            
            # Adjust for volatility
            volatility_adjustment = 1.0 / (1.0 + volatility * 10)
            
            # Final position size
            position_size = kelly_fraction * volatility_adjustment * self.config.MAX_POSITION_SIZE
            
            return min(position_size, self.config.MAX_POSITION_SIZE)
            
        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return 0.01  # Default small position
    
    def aggregate_signals(self, signals: Dict[str, TradingSignal]) -> TradingSignal:
        """Aggregate signals from multiple timeframes into a single decision"""
        if not signals:
            return TradingSignal(
                action='hold',
                strength=0.0,
                confidence=0.0,
                price_target=None,
                stop_loss=None,
                position_size=0.0,
                reasoning=["No signals available"],
                timestamp=time.time(),
                timeframe='aggregated',
                symbol='BTC/USD'
            )
        
        # Timeframe weights (longer timeframes have more weight)
        timeframe_weights = {
            '1m': 0.05,
            '5m': 0.1,
            '15m': 0.15,
            '1h': 0.25,
            '4h': 0.3,
            '1d': 0.15
        }
        
        # Calculate weighted scores
        buy_score = 0.0
        sell_score = 0.0
        total_weight = 0.0
        all_reasoning = []
        avg_confidence = 0.0
        
        for timeframe, signal in signals.items():
            weight = timeframe_weights.get(timeframe, 0.1)
            
            if signal.action == 'buy':
                buy_score += signal.strength * weight
            elif signal.action == 'sell':
                sell_score += signal.strength * weight
            
            total_weight += weight
            avg_confidence += signal.confidence * weight
            all_reasoning.extend([f"{timeframe}: {r}" for r in signal.reasoning])
        
        if total_weight > 0:
            buy_score /= total_weight
            sell_score /= total_weight
            avg_confidence /= total_weight
        
        # Determine final action
        if buy_score > sell_score and buy_score > 0.4:
            action = 'buy'
            strength = buy_score
        elif sell_score > buy_score and sell_score > 0.4:
            action = 'sell'
            strength = sell_score
        else:
            action = 'hold'
            strength = 0.0
        
        # Use primary timeframe for price targets
        primary_signal = signals.get(self.primary_timeframe)
        if primary_signal:
            price_target = primary_signal.price_target
            stop_loss = primary_signal.stop_loss
            position_size = primary_signal.position_size
        else:
            price_target = None
            stop_loss = None
            position_size = 0.0
        
        return TradingSignal(
            action=action,
            strength=strength,
            confidence=avg_confidence,
            price_target=price_target,
            stop_loss=stop_loss,
            position_size=position_size,
            reasoning=all_reasoning,
            timestamp=time.time(),
            timeframe='aggregated',
            symbol=list(signals.values())[0].symbol if signals else 'BTC/USD'
        )
    
    def get_trading_decision(self, symbol: str) -> TradingSignal:
        """Get final trading decision for a symbol"""
        try:
            # Generate signals across timeframes
            multi_tf_signals = self.generate_multi_timeframe_signals(symbol)
            
            # Aggregate into final decision
            final_signal = self.aggregate_signals(multi_tf_signals)
            
            # Store signal history
            self.signal_history.append(final_signal)
            if len(self.signal_history) > self.max_signal_history:
                self.signal_history = self.signal_history[-self.max_signal_history:]
            
            # Update performance tracking
            self.strategy_performance['total_signals'] += 1
            
            logger.info(f"Trading decision for {symbol}: {final_signal.action} "
                       f"(strength: {final_signal.strength:.3f}, confidence: {final_signal.confidence:.3f})")
            
            return final_signal
            
        except Exception as e:
            logger.error(f"Error getting trading decision: {e}")
            return TradingSignal(
                action='hold',
                strength=0.0,
                confidence=0.0,
                price_target=None,
                stop_loss=None,
                position_size=0.0,
                reasoning=[f"Error: {str(e)}"],
                timestamp=time.time(),
                timeframe='error',
                symbol=symbol
            )
    
    def update_performance(self, signal: TradingSignal, actual_return: float):
        """Update strategy performance metrics"""
        try:
            if actual_return > 0:
                self.strategy_performance['successful_signals'] += 1
            else:
                self.strategy_performance['failed_signals'] += 1
            
            # Update average return (simple moving average)
            total_signals = self.strategy_performance['total_signals']
            current_avg = self.strategy_performance['avg_return']
            self.strategy_performance['avg_return'] = (current_avg * (total_signals - 1) + actual_return) / total_signals
            
            logger.debug(f"Updated performance: avg_return={self.strategy_performance['avg_return']:.4f}")
            
        except Exception as e:
            logger.error(f"Error updating performance: {e}")
    
    def get_strategy_status(self) -> Dict:
        """Get current strategy status and performance"""
        return {
            'current_regime': self.current_regime.value,
            'performance': self.strategy_performance,
            'recent_signals': len(self.signal_history),
            'model_status': self.model_manager.get_status()
        }