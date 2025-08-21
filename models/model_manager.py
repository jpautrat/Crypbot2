"""
Institutional Model Management System
Handles loading, inference, and ensemble of pre-trained models
"""
import os
import pickle
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any
import logging
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
import time
from dataclasses import dataclass

from config.settings import TradingConfig

logger = logging.getLogger(__name__)

@dataclass
class ModelPrediction:
    model_name: str
    prediction: float
    confidence: float
    timestamp: float
    features_used: List[str]
    processing_time_ms: float

class BaseModel(ABC):
    """Base class for all trading models"""
    
    def __init__(self, name: str, config: TradingConfig):
        self.name = name
        self.config = config
        self.model = None
        self.is_loaded = False
        self.last_prediction_time = 0
        self.prediction_cache = {}
        
    @abstractmethod
    def load_model(self, model_path: str) -> bool:
        """Load the pre-trained model"""
        pass
    
    @abstractmethod
    def preprocess_features(self, data: pd.DataFrame) -> np.ndarray:
        """Preprocess data for model input"""
        pass
    
    @abstractmethod
    def predict(self, features: np.ndarray) -> Tuple[float, float]:
        """Make prediction and return (prediction, confidence)"""
        pass
    
    def get_prediction(self, data: pd.DataFrame, use_cache: bool = True) -> ModelPrediction:
        """Get prediction with caching and timing"""
        start_time = time.time()
        
        # Check cache
        cache_key = hash(str(data.tail(10).values.tobytes()))
        if use_cache and cache_key in self.prediction_cache:
            cache_age = time.time() - self.prediction_cache[cache_key]['timestamp']
            if cache_age < self.config.PREDICTION_CACHE_TTL:
                cached = self.prediction_cache[cache_key]
                return ModelPrediction(
                    model_name=self.name,
                    prediction=cached['prediction'],
                    confidence=cached['confidence'],
                    timestamp=cached['timestamp'],
                    features_used=cached['features_used'],
                    processing_time_ms=0  # Cached
                )
        
        # Make new prediction
        try:
            features = self.preprocess_features(data)
            prediction, confidence = self.predict(features)
            
            processing_time = (time.time() - start_time) * 1000
            
            result = ModelPrediction(
                model_name=self.name,
                prediction=prediction,
                confidence=confidence,
                timestamp=time.time(),
                features_used=list(data.columns),
                processing_time_ms=processing_time
            )
            
            # Cache result
            self.prediction_cache[cache_key] = {
                'prediction': prediction,
                'confidence': confidence,
                'timestamp': result.timestamp,
                'features_used': result.features_used
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error in {self.name} prediction: {e}")
            return ModelPrediction(
                model_name=self.name,
                prediction=0.0,
                confidence=0.0,
                timestamp=time.time(),
                features_used=[],
                processing_time_ms=(time.time() - start_time) * 1000
            )

class XGBoostModel(BaseModel):
    """XGBoost model for crypto price prediction"""
    
    def load_model(self, model_path: str) -> bool:
        try:
            if model_path.endswith('.pkl'):
                with open(model_path, 'rb') as f:
                    self.model = pickle.load(f)
            else:
                self.model = joblib.load(model_path)
            
            self.is_loaded = True
            logger.info(f"XGBoost model loaded from {model_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to load XGBoost model: {e}")
            return False
    
    def preprocess_features(self, data: pd.DataFrame) -> np.ndarray:
        """Preprocess features for XGBoost"""
        # Calculate technical indicators
        df = data.copy()
        
        # Price-based features
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        
        # Moving averages
        for window in [5, 10, 20, 50]:
            df[f'ma_{window}'] = df['close'].rolling(window).mean()
            df[f'ma_ratio_{window}'] = df['close'] / df[f'ma_{window}']
        
        # Volatility features
        df['volatility_5'] = df['returns'].rolling(5).std()
        df['volatility_20'] = df['returns'].rolling(20).std()
        
        # Volume features
        df['volume_ma_10'] = df['volume'].rolling(10).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma_10']
        
        # Price position features
        df['high_low_ratio'] = df['high'] / df['low']
        df['close_position'] = (df['close'] - df['low']) / (df['high'] - df['low'])
        
        # Momentum features
        df['rsi'] = self._calculate_rsi(df['close'])
        df['macd'], df['macd_signal'] = self._calculate_macd(df['close'])
        
        # Select features (adjust based on your specific model requirements)
        feature_columns = [
            'returns', 'log_returns', 'ma_ratio_5', 'ma_ratio_10', 'ma_ratio_20',
            'volatility_5', 'volatility_20', 'volume_ratio', 'high_low_ratio',
            'close_position', 'rsi', 'macd'
        ]
        
        # Get the last row of features
        features = df[feature_columns].iloc[-1:].values
        
        # Handle NaN values
        features = np.nan_to_num(features, nan=0.0)
        
        return features
    
    def predict(self, features: np.ndarray) -> Tuple[float, float]:
        """Make XGBoost prediction"""
        if not self.is_loaded:
            return 0.0, 0.0
        
        try:
            # Get prediction
            prediction = self.model.predict(features)[0]
            
            # Get feature importance as confidence proxy
            if hasattr(self.model, 'feature_importances_'):
                confidence = np.mean(self.model.feature_importances_)
            else:
                confidence = 0.7  # Default confidence
            
            return float(prediction), float(confidence)
            
        except Exception as e:
            logger.error(f"XGBoost prediction error: {e}")
            return 0.0, 0.0
    
    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series]:
        """Calculate MACD indicator"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal).mean()
        return macd, macd_signal

class RandomForestModel(BaseModel):
    """Random Forest model for crypto prediction"""
    
    def load_model(self, model_path: str) -> bool:
        try:
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            
            self.is_loaded = True
            logger.info(f"Random Forest model loaded from {model_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to load Random Forest model: {e}")
            return False
    
    def preprocess_features(self, data: pd.DataFrame) -> np.ndarray:
        """Preprocess features for Random Forest"""
        # Similar to XGBoost but may have different feature requirements
        return self._calculate_tree_features(data)
    
    def predict(self, features: np.ndarray) -> Tuple[float, float]:
        """Make Random Forest prediction"""
        if not self.is_loaded:
            return 0.0, 0.0
        
        try:
            # Get prediction and probability
            prediction = self.model.predict(features)[0]
            
            if hasattr(self.model, 'predict_proba'):
                probabilities = self.model.predict_proba(features)[0]
                confidence = np.max(probabilities)
            else:
                confidence = 0.7
            
            return float(prediction), float(confidence)
            
        except Exception as e:
            logger.error(f"Random Forest prediction error: {e}")
            return 0.0, 0.0
    
    def _calculate_tree_features(self, data: pd.DataFrame) -> np.ndarray:
        """Calculate features optimized for tree-based models"""
        df = data.copy()
        
        # Basic price features
        df['returns'] = df['close'].pct_change()
        df['high_low_pct'] = (df['high'] - df['low']) / df['close']
        df['open_close_pct'] = (df['close'] - df['open']) / df['open']
        
        # Moving averages and ratios
        for window in [5, 10, 20]:
            df[f'ma_{window}'] = df['close'].rolling(window).mean()
            df[f'price_above_ma_{window}'] = (df['close'] > df[f'ma_{window}']).astype(int)
        
        # Volume features
        df['volume_sma_10'] = df['volume'].rolling(10).mean()
        df['volume_above_avg'] = (df['volume'] > df['volume_sma_10']).astype(int)
        
        # Volatility
        df['volatility'] = df['returns'].rolling(10).std()
        
        # Select final features
        feature_columns = [
            'returns', 'high_low_pct', 'open_close_pct',
            'price_above_ma_5', 'price_above_ma_10', 'price_above_ma_20',
            'volume_above_avg', 'volatility'
        ]
        
        features = df[feature_columns].iloc[-1:].values
        features = np.nan_to_num(features, nan=0.0)
        
        return features

class LSTMModel(BaseModel):
    """LSTM model for time series prediction"""
    
    def __init__(self, name: str, config: TradingConfig):
        super().__init__(name, config)
        self.device = torch.device(config.GPU_DEVICE)
        self.sequence_length = 60  # 60-day lookback
        self.scaler = None
    
    def load_model(self, model_path: str) -> bool:
        try:
            if model_path.endswith('.h5'):
                # TensorFlow/Keras model
                import tensorflow as tf
                self.model = tf.keras.models.load_model(model_path)
                self.framework = 'tensorflow'
            else:
                # PyTorch model
                self.model = torch.load(model_path, map_location=self.device)
                self.model.eval()
                self.framework = 'pytorch'
            
            self.is_loaded = True
            logger.info(f"LSTM model loaded from {model_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to load LSTM model: {e}")
            return False
    
    def preprocess_features(self, data: pd.DataFrame) -> np.ndarray:
        """Preprocess features for LSTM"""
        # Prepare sequence data
        df = data.copy()
        
        # Calculate features
        df['returns'] = df['close'].pct_change()
        df['high_low_ratio'] = df['high'] / df['low']
        df['volume_change'] = df['volume'].pct_change()
        
        # Technical indicators
        df['sma_5'] = df['close'].rolling(5).mean()
        df['sma_20'] = df['close'].rolling(20).mean()
        df['rsi'] = self._calculate_rsi(df['close'])
        
        # Select features for LSTM
        feature_columns = ['close', 'volume', 'returns', 'high_low_ratio', 'volume_change', 'sma_5', 'sma_20', 'rsi']
        
        # Get the last sequence_length rows
        features = df[feature_columns].tail(self.sequence_length).values
        
        # Handle NaN values
        features = np.nan_to_num(features, nan=0.0)
        
        # Normalize features (simple min-max scaling)
        if features.shape[0] > 0:
            features = (features - features.min(axis=0)) / (features.max(axis=0) - features.min(axis=0) + 1e-8)
        
        # Reshape for LSTM: (batch_size, sequence_length, features)
        features = features.reshape(1, self.sequence_length, len(feature_columns))
        
        return features
    
    def predict(self, features: np.ndarray) -> Tuple[float, float]:
        """Make LSTM prediction"""
        if not self.is_loaded:
            return 0.0, 0.0
        
        try:
            if self.framework == 'tensorflow':
                prediction = self.model.predict(features, verbose=0)[0][0]
                confidence = 0.8  # Default confidence for TF models
            else:
                # PyTorch
                with torch.no_grad():
                    features_tensor = torch.FloatTensor(features).to(self.device)
                    prediction = self.model(features_tensor).cpu().numpy()[0][0]
                    confidence = 0.8
            
            return float(prediction), float(confidence)
            
        except Exception as e:
            logger.error(f"LSTM prediction error: {e}")
            return 0.0, 0.0
    
    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

class ModelEnsemble:
    """Ensemble of multiple models with weighted predictions"""
    
    def __init__(self, config: TradingConfig):
        self.config = config
        self.models: Dict[str, BaseModel] = {}
        self.weights = config.MODEL_ENSEMBLE_WEIGHTS
        self.performance_history = {}
        
    def add_model(self, model: BaseModel) -> bool:
        """Add a model to the ensemble"""
        if model.is_loaded:
            self.models[model.name] = model
            self.performance_history[model.name] = []
            logger.info(f"Added {model.name} to ensemble")
            return True
        return False
    
    def get_ensemble_prediction(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Get weighted ensemble prediction"""
        if not self.models:
            return {'prediction': 0.0, 'confidence': 0.0, 'individual_predictions': {}}
        
        # Get predictions from all models in parallel
        with ThreadPoolExecutor(max_workers=len(self.models)) as executor:
            future_to_model = {
                executor.submit(model.get_prediction, data): name 
                for name, model in self.models.items()
            }
            
            predictions = {}
            for future in future_to_model:
                model_name = future_to_model[future]
                try:
                    prediction = future.result(timeout=5)
                    predictions[model_name] = prediction
                except Exception as e:
                    logger.error(f"Error getting prediction from {model_name}: {e}")
        
        # Calculate weighted ensemble prediction
        weighted_prediction = 0.0
        weighted_confidence = 0.0
        total_weight = 0.0
        
        individual_predictions = {}
        
        for model_name, prediction in predictions.items():
            weight = self.weights.get(model_name, 0.1)
            weighted_prediction += prediction.prediction * weight
            weighted_confidence += prediction.confidence * weight
            total_weight += weight
            
            individual_predictions[model_name] = {
                'prediction': prediction.prediction,
                'confidence': prediction.confidence,
                'processing_time_ms': prediction.processing_time_ms
            }
        
        if total_weight > 0:
            weighted_prediction /= total_weight
            weighted_confidence /= total_weight
        
        return {
            'prediction': weighted_prediction,
            'confidence': weighted_confidence,
            'individual_predictions': individual_predictions,
            'ensemble_weight_sum': total_weight,
            'timestamp': time.time()
        }
    
    def update_model_performance(self, model_name: str, actual_outcome: float, predicted_outcome: float):
        """Update model performance tracking"""
        if model_name in self.performance_history:
            error = abs(actual_outcome - predicted_outcome)
            self.performance_history[model_name].append(error)
            
            # Keep only last 100 predictions
            if len(self.performance_history[model_name]) > 100:
                self.performance_history[model_name] = self.performance_history[model_name][-100:]
    
    def get_model_performance_stats(self) -> Dict[str, Dict[str, float]]:
        """Get performance statistics for all models"""
        stats = {}
        
        for model_name, errors in self.performance_history.items():
            if errors:
                stats[model_name] = {
                    'mean_error': np.mean(errors),
                    'std_error': np.std(errors),
                    'min_error': np.min(errors),
                    'max_error': np.max(errors),
                    'prediction_count': len(errors)
                }
            else:
                stats[model_name] = {
                    'mean_error': 0.0,
                    'std_error': 0.0,
                    'min_error': 0.0,
                    'max_error': 0.0,
                    'prediction_count': 0
                }
        
        return stats
    
    def adjust_weights_based_on_performance(self):
        """Dynamically adjust model weights based on recent performance"""
        stats = self.get_model_performance_stats()
        
        # Calculate inverse error weights (lower error = higher weight)
        new_weights = {}
        total_inverse_error = 0.0
        
        for model_name in self.models.keys():
            if model_name in stats and stats[model_name]['prediction_count'] > 10:
                mean_error = stats[model_name]['mean_error']
                inverse_error = 1.0 / (mean_error + 1e-6)  # Add small epsilon to avoid division by zero
                new_weights[model_name] = inverse_error
                total_inverse_error += inverse_error
            else:
                new_weights[model_name] = self.weights.get(model_name, 0.1)
        
        # Normalize weights
        if total_inverse_error > 0:
            for model_name in new_weights:
                new_weights[model_name] /= total_inverse_error
            
            # Smooth the weight adjustment (blend with original weights)
            alpha = 0.3  # Adjustment factor
            for model_name in self.weights:
                if model_name in new_weights:
                    self.weights[model_name] = (1 - alpha) * self.weights[model_name] + alpha * new_weights[model_name]
            
            logger.info(f"Adjusted model weights: {self.weights}")

class ModelManager:
    """Central manager for all trading models"""
    
    def __init__(self, config: TradingConfig):
        self.config = config
        self.ensemble = ModelEnsemble(config)
        self.model_paths = {}
        
    def load_models_from_directory(self, models_dir: str) -> int:
        """Load all available models from directory"""
        loaded_count = 0
        
        if not os.path.exists(models_dir):
            logger.warning(f"Models directory {models_dir} does not exist")
            return 0
        
        # Define model types and their file patterns
        model_configs = [
            ('xgboost_btc', XGBoostModel, ['*.pkl', '*xgboost*.pkl', '*xgb*.pkl']),
            ('random_forest_btc', RandomForestModel, ['*random_forest*.pkl', '*rf*.pkl']),
            ('lstm_btc', LSTMModel, ['*.h5', '*lstm*.h5', '*lstm*.pth'])
        ]
        
        for model_name, model_class, patterns in model_configs:
            model_path = self._find_model_file(models_dir, patterns)
            if model_path:
                try:
                    model = model_class(model_name, self.config)
                    if model.load_model(model_path):
                        if self.ensemble.add_model(model):
                            loaded_count += 1
                            self.model_paths[model_name] = model_path
                except Exception as e:
                    logger.error(f"Failed to load {model_name}: {e}")
        
        logger.info(f"Loaded {loaded_count} models successfully")
        return loaded_count
    
    def _find_model_file(self, directory: str, patterns: List[str]) -> Optional[str]:
        """Find model file matching patterns"""
        import glob
        
        for pattern in patterns:
            files = glob.glob(os.path.join(directory, pattern))
            if files:
                return files[0]  # Return first match
        
        return None
    
    def get_prediction(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Get ensemble prediction"""
        return self.ensemble.get_ensemble_prediction(data)
    
    def update_performance(self, actual_outcome: float, predicted_outcome: float):
        """Update all model performance metrics"""
        for model_name in self.ensemble.models.keys():
            self.ensemble.update_model_performance(model_name, actual_outcome, predicted_outcome)
    
    def get_status(self) -> Dict[str, Any]:
        """Get model manager status"""
        return {
            'loaded_models': list(self.ensemble.models.keys()),
            'model_weights': self.ensemble.weights,
            'performance_stats': self.ensemble.get_model_performance_stats(),
            'model_paths': self.model_paths
        }