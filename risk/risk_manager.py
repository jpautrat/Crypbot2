"""
Institutional Risk Management System
Advanced portfolio risk management with real-time monitoring
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
import time

from config.settings import TradingConfig

logger = logging.getLogger(__name__)

@dataclass
class RiskMetrics:
    portfolio_value: float
    daily_pnl: float
    daily_return: float
    volatility: float
    var_95: float
    expected_shortfall: float
    max_drawdown: float
    sharpe_ratio: float
    sortino_ratio: float
    beta: float
    correlation_btc: float
    leverage: float
    margin_usage: float

@dataclass
class PositionRisk:
    symbol: str
    position_size: float
    market_value: float
    unrealized_pnl: float
    daily_var: float
    position_limit: float
    risk_contribution: float
    correlation_risk: float

class RiskManager:
    """Advanced risk management system for institutional trading"""
    
    def __init__(self, config: TradingConfig):
        self.config = config
        
        # Risk limits
        self.max_daily_loss = config.MAX_DAILY_LOSS
        self.max_drawdown = config.MAX_DRAWDOWN
        self.var_confidence = config.VAR_CONFIDENCE
        self.max_position_size = config.MAX_POSITION_SIZE
        
        # Portfolio tracking
        self.portfolio_history = []
        self.position_history = {}
        self.daily_returns = []
        self.drawdown_history = []
        
        # Risk state
        self.current_drawdown = 0.0
        self.daily_pnl = 0.0
        self.portfolio_value = 0.0
        self.high_water_mark = 0.0
        
        # Circuit breakers
        self.trading_halted = False
        self.halt_reason = ""
        self.halt_timestamp = 0
        
        # Correlation matrix for multi-asset risk
        self.correlation_matrix = {}
        self.volatility_estimates = {}
        
    def update_portfolio_value(self, positions: Dict[str, Dict], market_prices: Dict[str, float]):
        """Update current portfolio value and positions"""
        try:
            total_value = 0.0
            position_values = {}
            
            # Calculate position values
            for symbol, position in positions.items():
                if symbol in market_prices:
                    market_value = position['quantity'] * market_prices[symbol]
                    position_values[symbol] = {
                        'quantity': position['quantity'],
                        'avg_price': position.get('avg_price', market_prices[symbol]),
                        'market_price': market_prices[symbol],
                        'market_value': market_value,
                        'unrealized_pnl': market_value - (position['quantity'] * position.get('avg_price', market_prices[symbol]))
                    }
                    total_value += market_value
            
            # Add cash position
            cash_balance = positions.get('USD', {}).get('quantity', 0)
            total_value += cash_balance
            
            # Update portfolio tracking
            previous_value = self.portfolio_value
            self.portfolio_value = total_value
            
            if previous_value > 0:
                self.daily_pnl = total_value - previous_value
                daily_return = self.daily_pnl / previous_value
                self.daily_returns.append(daily_return)
                
                # Keep only last 252 days (1 year)
                if len(self.daily_returns) > 252:
                    self.daily_returns = self.daily_returns[-252:]
            
            # Update high water mark and drawdown
            if total_value > self.high_water_mark:
                self.high_water_mark = total_value
                self.current_drawdown = 0.0
            else:
                self.current_drawdown = (self.high_water_mark - total_value) / self.high_water_mark
            
            self.drawdown_history.append(self.current_drawdown)
            if len(self.drawdown_history) > 252:
                self.drawdown_history = self.drawdown_history[-252:]
            
            # Store portfolio snapshot
            portfolio_snapshot = {
                'timestamp': time.time(),
                'total_value': total_value,
                'daily_pnl': self.daily_pnl,
                'drawdown': self.current_drawdown,
                'positions': position_values
            }
            
            self.portfolio_history.append(portfolio_snapshot)
            if len(self.portfolio_history) > 1000:
                self.portfolio_history = self.portfolio_history[-1000:]
            
            logger.debug(f"Portfolio updated: Value=${total_value:.2f}, PnL=${self.daily_pnl:.2f}, DD={self.current_drawdown:.2%}")
            
        except Exception as e:
            logger.error(f"Error updating portfolio value: {e}")
    
    def calculate_var(self, confidence: float = None) -> float:
        """Calculate Value at Risk using historical simulation"""
        confidence = confidence or self.var_confidence
        
        if len(self.daily_returns) < 30:
            return 0.0
        
        try:
            returns_array = np.array(self.daily_returns)
            var_percentile = (1 - confidence) * 100
            var_return = np.percentile(returns_array, var_percentile)
            var_dollar = abs(var_return * self.portfolio_value)
            
            return var_dollar
            
        except Exception as e:
            logger.error(f"Error calculating VaR: {e}")
            return 0.0
    
    def calculate_expected_shortfall(self, confidence: float = None) -> float:
        """Calculate Expected Shortfall (Conditional VaR)"""
        confidence = confidence or self.var_confidence
        
        if len(self.daily_returns) < 30:
            return 0.0
        
        try:
            returns_array = np.array(self.daily_returns)
            var_percentile = (1 - confidence) * 100
            var_threshold = np.percentile(returns_array, var_percentile)
            
            # Expected shortfall is the mean of returns below VaR threshold
            tail_returns = returns_array[returns_array <= var_threshold]
            if len(tail_returns) > 0:
                es_return = np.mean(tail_returns)
                es_dollar = abs(es_return * self.portfolio_value)
                return es_dollar
            else:
                return self.calculate_var(confidence)
                
        except Exception as e:
            logger.error(f"Error calculating Expected Shortfall: {e}")
            return 0.0
    
    def calculate_sharpe_ratio(self, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio"""
        if len(self.daily_returns) < 30:
            return 0.0
        
        try:
            returns_array = np.array(self.daily_returns)
            excess_returns = returns_array - (risk_free_rate / 252)  # Daily risk-free rate
            
            if np.std(excess_returns) > 0:
                sharpe = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)
                return sharpe
            else:
                return 0.0
                
        except Exception as e:
            logger.error(f"Error calculating Sharpe ratio: {e}")
            return 0.0
    
    def calculate_sortino_ratio(self, risk_free_rate: float = 0.02) -> float:
        """Calculate Sortino ratio (downside deviation)"""
        if len(self.daily_returns) < 30:
            return 0.0
        
        try:
            returns_array = np.array(self.daily_returns)
            excess_returns = returns_array - (risk_free_rate / 252)
            
            # Only consider negative returns for downside deviation
            downside_returns = excess_returns[excess_returns < 0]
            
            if len(downside_returns) > 0:
                downside_deviation = np.std(downside_returns)
                if downside_deviation > 0:
                    sortino = np.mean(excess_returns) / downside_deviation * np.sqrt(252)
                    return sortino
            
            return 0.0
            
        except Exception as e:
            logger.error(f"Error calculating Sortino ratio: {e}")
            return 0.0
    
    def calculate_position_risk(self, symbol: str, quantity: float, market_price: float) -> PositionRisk:
        """Calculate risk metrics for a specific position"""
        try:
            market_value = quantity * market_price
            
            # Get volatility estimate
            volatility = self.volatility_estimates.get(symbol, 0.02)  # Default 2% daily vol
            
            # Calculate daily VaR for this position
            daily_var = market_value * volatility * 1.65  # 95% confidence
            
            # Position limit based on portfolio size
            position_limit = self.portfolio_value * self.max_position_size
            
            # Risk contribution (simplified)
            risk_contribution = daily_var / self.portfolio_value if self.portfolio_value > 0 else 0
            
            # Correlation risk (placeholder - would need market data)
            correlation_risk = 0.0
            
            # Unrealized PnL (would need average cost basis)
            unrealized_pnl = 0.0  # Placeholder
            
            return PositionRisk(
                symbol=symbol,
                position_size=quantity,
                market_value=market_value,
                unrealized_pnl=unrealized_pnl,
                daily_var=daily_var,
                position_limit=position_limit,
                risk_contribution=risk_contribution,
                correlation_risk=correlation_risk
            )
            
        except Exception as e:
            logger.error(f"Error calculating position risk for {symbol}: {e}")
            return PositionRisk(
                symbol=symbol,
                position_size=0,
                market_value=0,
                unrealized_pnl=0,
                daily_var=0,
                position_limit=0,
                risk_contribution=0,
                correlation_risk=0
            )
    
    def check_risk_limits(self, proposed_trade: Dict) -> Tuple[bool, List[str]]:
        """Check if proposed trade violates risk limits"""
        violations = []
        
        try:
            # Check daily loss limit
            if self.daily_pnl < -self.max_daily_loss * self.portfolio_value:
                violations.append(f"Daily loss limit exceeded: {self.daily_pnl:.2f}")
            
            # Check maximum drawdown
            if self.current_drawdown > self.max_drawdown:
                violations.append(f"Maximum drawdown exceeded: {self.current_drawdown:.2%}")
            
            # Check position size limits
            symbol = proposed_trade.get('symbol')
            quantity = proposed_trade.get('quantity', 0)
            price = proposed_trade.get('price', 0)
            
            if symbol and quantity and price:
                position_value = abs(quantity * price)
                max_position_value = self.portfolio_value * self.max_position_size
                
                if position_value > max_position_value:
                    violations.append(f"Position size limit exceeded for {symbol}: ${position_value:.2f} > ${max_position_value:.2f}")
            
            # Check if trading is halted
            if self.trading_halted:
                violations.append(f"Trading halted: {self.halt_reason}")
            
            return len(violations) == 0, violations
            
        except Exception as e:
            logger.error(f"Error checking risk limits: {e}")
            return False, [f"Risk check error: {str(e)}"]
    
    def update_volatility_estimates(self, symbol: str, price_data: pd.DataFrame):
        """Update volatility estimates for risk calculations"""
        try:
            if len(price_data) < 20:
                return
            
            # Calculate returns
            returns = price_data['close'].pct_change().dropna()
            
            # EWMA volatility estimate
            volatility = returns.ewm(span=20).std().iloc[-1]
            
            self.volatility_estimates[symbol] = volatility
            
            logger.debug(f"Updated volatility estimate for {symbol}: {volatility:.4f}")
            
        except Exception as e:
            logger.error(f"Error updating volatility for {symbol}: {e}")
    
    def trigger_circuit_breaker(self, reason: str):
        """Trigger emergency trading halt"""
        self.trading_halted = True
        self.halt_reason = reason
        self.halt_timestamp = time.time()
        
        logger.critical(f"CIRCUIT BREAKER TRIGGERED: {reason}")
    
    def reset_circuit_breaker(self):
        """Reset circuit breaker (manual intervention required)"""
        self.trading_halted = False
        self.halt_reason = ""
        self.halt_timestamp = 0
        
        logger.warning("Circuit breaker reset - trading resumed")
    
    def get_risk_metrics(self) -> RiskMetrics:
        """Get comprehensive risk metrics"""
        try:
            var_95 = self.calculate_var(0.95)
            expected_shortfall = self.calculate_expected_shortfall(0.95)
            sharpe_ratio = self.calculate_sharpe_ratio()
            sortino_ratio = self.calculate_sortino_ratio()
            
            # Calculate volatility
            volatility = np.std(self.daily_returns) * np.sqrt(252) if len(self.daily_returns) > 1 else 0.0
            
            # Calculate max drawdown
            max_drawdown = max(self.drawdown_history) if self.drawdown_history else 0.0
            
            # Daily return
            daily_return = self.daily_returns[-1] if self.daily_returns else 0.0
            
            return RiskMetrics(
                portfolio_value=self.portfolio_value,
                daily_pnl=self.daily_pnl,
                daily_return=daily_return,
                volatility=volatility,
                var_95=var_95,
                expected_shortfall=expected_shortfall,
                max_drawdown=max_drawdown,
                sharpe_ratio=sharpe_ratio,
                sortino_ratio=sortino_ratio,
                beta=0.0,  # Would need benchmark data
                correlation_btc=0.0,  # Would need BTC correlation
                leverage=0.0,  # Not applicable for spot trading
                margin_usage=0.0  # Not applicable for spot trading
            )
            
        except Exception as e:
            logger.error(f"Error calculating risk metrics: {e}")
            return RiskMetrics(
                portfolio_value=self.portfolio_value,
                daily_pnl=0.0,
                daily_return=0.0,
                volatility=0.0,
                var_95=0.0,
                expected_shortfall=0.0,
                max_drawdown=0.0,
                sharpe_ratio=0.0,
                sortino_ratio=0.0,
                beta=0.0,
                correlation_btc=0.0,
                leverage=0.0,
                margin_usage=0.0
            )
    
    def generate_risk_report(self) -> Dict:
        """Generate comprehensive risk report"""
        metrics = self.get_risk_metrics()
        
        # Risk status
        risk_status = "LOW"
        if self.current_drawdown > 0.05:
            risk_status = "MEDIUM"
        if self.current_drawdown > 0.10:
            risk_status = "HIGH"
        if self.trading_halted:
            risk_status = "CRITICAL"
        
        # Recent performance
        recent_returns = self.daily_returns[-30:] if len(self.daily_returns) >= 30 else self.daily_returns
        win_rate = len([r for r in recent_returns if r > 0]) / len(recent_returns) if recent_returns else 0
        
        return {
            'timestamp': time.time(),
            'risk_status': risk_status,
            'metrics': {
                'portfolio_value': metrics.portfolio_value,
                'daily_pnl': metrics.daily_pnl,
                'daily_return': metrics.daily_return,
                'volatility_annualized': metrics.volatility,
                'var_95_daily': metrics.var_95,
                'expected_shortfall': metrics.expected_shortfall,
                'max_drawdown': metrics.max_drawdown,
                'current_drawdown': self.current_drawdown,
                'sharpe_ratio': metrics.sharpe_ratio,
                'sortino_ratio': metrics.sortino_ratio
            },
            'limits': {
                'max_daily_loss': self.max_daily_loss,
                'max_drawdown': self.max_drawdown,
                'max_position_size': self.max_position_size
            },
            'performance': {
                'win_rate_30d': win_rate,
                'total_trades': len(self.daily_returns),
                'profitable_days': len([r for r in self.daily_returns if r > 0]),
                'losing_days': len([r for r in self.daily_returns if r < 0])
            },
            'circuit_breaker': {
                'active': self.trading_halted,
                'reason': self.halt_reason,
                'timestamp': self.halt_timestamp
            }
        }
    
    def stress_test_portfolio(self, scenarios: Dict[str, float]) -> Dict[str, float]:
        """Run stress tests on current portfolio"""
        stress_results = {}
        
        try:
            for scenario_name, shock_percentage in scenarios.items():
                # Apply shock to portfolio value
                shocked_value = self.portfolio_value * (1 + shock_percentage)
                loss = self.portfolio_value - shocked_value
                loss_percentage = loss / self.portfolio_value if self.portfolio_value > 0 else 0
                
                stress_results[scenario_name] = {
                    'shock_applied': shock_percentage,
                    'portfolio_loss': loss,
                    'loss_percentage': loss_percentage,
                    'surviving_value': shocked_value
                }
            
            return stress_results
            
        except Exception as e:
            logger.error(f"Error in stress testing: {e}")
            return {}