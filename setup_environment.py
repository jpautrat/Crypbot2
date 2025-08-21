"""
Environment Setup Script for Enhanced Trading Bot
Validates hardware, installs dependencies, and prepares the system
"""
import os
import sys
import subprocess
import platform
import psutil
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnvironmentSetup:
    def __init__(self):
        self.system_info = self.get_system_info()
        self.requirements_met = True
        
    def get_system_info(self):
        """Get comprehensive system information"""
        try:
            import GPUtil
            gpus = GPUtil.getGPUs()
            gpu_info = [{"name": gpu.name, "memory": gpu.memoryTotal} for gpu in gpus]
        except:
            gpu_info = []
        
        return {
            "platform": platform.platform(),
            "python_version": sys.version,
            "cpu_count": psutil.cpu_count(),
            "memory_gb": psutil.virtual_memory().total / (1024**3),
            "gpu_info": gpu_info
        }
    
    def validate_hardware(self):
        """Validate hardware requirements"""
        logger.info("Validating hardware requirements...")
        
        # CPU validation
        if self.system_info["cpu_count"] < 8:
            logger.warning(f"CPU cores: {self.system_info['cpu_count']} (recommended: 16+)")
        else:
            logger.info(f"✓ CPU cores: {self.system_info['cpu_count']}")
        
        # Memory validation
        if self.system_info["memory_gb"] < 16:
            logger.error(f"RAM: {self.system_info['memory_gb']:.1f}GB (minimum: 16GB)")
            self.requirements_met = False
        else:
            logger.info(f"✓ RAM: {self.system_info['memory_gb']:.1f}GB")
        
        # GPU validation
        if not self.system_info["gpu_info"]:
            logger.warning("No GPU detected - ML models will run on CPU")
        else:
            for gpu in self.system_info["gpu_info"]:
                logger.info(f"✓ GPU: {gpu['name']} ({gpu['memory']}MB)")
        
        return self.requirements_met
    
    def check_python_version(self):
        """Check Python version compatibility"""
        logger.info("Checking Python version...")
        
        version = sys.version_info
        if version.major != 3 or version.minor < 8:
            logger.error(f"Python {version.major}.{version.minor} detected. Minimum required: Python 3.8")
            self.requirements_met = False
            return False
        
        logger.info(f"✓ Python {version.major}.{version.minor}.{version.micro}")
        return True
    
    def install_cuda_support(self):
        """Install CUDA support for PyTorch and TensorFlow"""
        logger.info("Setting up CUDA support...")
        
        try:
            # Check if NVIDIA GPU is available
            if not self.system_info["gpu_info"]:
                logger.info("No NVIDIA GPU detected, skipping CUDA setup")
                return True
            
            # Install PyTorch with CUDA support
            logger.info("Installing PyTorch with CUDA support...")
            subprocess.run([
                sys.executable, "-m", "pip", "install", 
                "torch", "torchvision", "torchaudio", 
                "--index-url", "https://download.pytorch.org/whl/cu118"
            ], check=True)
            
            # Install TensorFlow with GPU support
            logger.info("Installing TensorFlow with GPU support...")
            subprocess.run([
                sys.executable, "-m", "pip", "install", 
                "tensorflow[and-cuda]"
            ], check=True)
            
            logger.info("✓ CUDA support installed")
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to install CUDA support: {e}")
            return False
    
    def install_requirements(self):
        """Install Python requirements"""
        logger.info("Installing Python requirements...")
        
        try:
            # Upgrade pip first
            subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"], check=True)
            
            # Install requirements
            requirements_file = "requirements_enhanced.txt"
            if os.path.exists(requirements_file):
                subprocess.run([
                    sys.executable, "-m", "pip", "install", "-r", requirements_file
                ], check=True)
                logger.info("✓ Requirements installed")
            else:
                logger.error(f"Requirements file {requirements_file} not found")
                return False
            
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to install requirements: {e}")
            return False
    
    def create_directories(self):
        """Create necessary directories"""
        logger.info("Creating directory structure...")
        
        directories = [
            "models/pretrained",
            "data/cache",
            "logs",
            "config",
            "backups",
            "reports"
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
            logger.info(f"✓ Created directory: {directory}")
    
    def create_env_template(self):
        """Create .env template file"""
        logger.info("Creating .env template...")
        
        env_template = """# Kraken API Credentials
KRAKEN_API_KEY=your_kraken_api_key_here
KRAKEN_API_SECRET=your_kraken_api_secret_here

# Alternative naming (for compatibility)
API_KEY=your_kraken_api_key_here
API_SECRET=your_kraken_api_secret_here

# Trading Configuration
TRADING_MODE=simulation
MAX_POSITION_SIZE=0.02
PROFIT_THRESHOLD=0.003

# Logging
LOG_LEVEL=INFO
LOG_FILE=trading_bot.log

# Performance
ENABLE_GPU=true
BATCH_SIZE=64
ASYNC_WORKERS=8
"""
        
        if not os.path.exists(".env"):
            with open(".env", "w") as f:
                f.write(env_template)
            logger.info("✓ Created .env template")
        else:
            logger.info("✓ .env file already exists")
    
    def test_imports(self):
        """Test critical imports"""
        logger.info("Testing critical imports...")
        
        critical_imports = [
            "ccxt",
            "pandas",
            "numpy",
            "sklearn",
            "torch",
            "tensorflow",
            "xgboost",
            "websockets",
            "asyncio"
        ]
        
        failed_imports = []
        
        for module in critical_imports:
            try:
                __import__(module)
                logger.info(f"✓ {module}")
            except ImportError as e:
                logger.error(f"✗ {module}: {e}")
                failed_imports.append(module)
        
        if failed_imports:
            logger.error(f"Failed to import: {failed_imports}")
            return False
        
        return True
    
    def test_gpu_availability(self):
        """Test GPU availability for ML frameworks"""
        logger.info("Testing GPU availability...")
        
        try:
            import torch
            if torch.cuda.is_available():
                logger.info(f"✓ PyTorch CUDA available: {torch.cuda.get_device_name(0)}")
            else:
                logger.warning("PyTorch CUDA not available")
        except Exception as e:
            logger.error(f"PyTorch GPU test failed: {e}")
        
        try:
            import tensorflow as tf
            if tf.config.list_physical_devices('GPU'):
                logger.info("✓ TensorFlow GPU available")
            else:
                logger.warning("TensorFlow GPU not available")
        except Exception as e:
            logger.error(f"TensorFlow GPU test failed: {e}")
    
    def run_setup(self):
        """Run complete environment setup"""
        logger.info("Starting environment setup for Enhanced Trading Bot")
        logger.info("=" * 60)
        
        # System information
        logger.info("System Information:")
        for key, value in self.system_info.items():
            logger.info(f"  {key}: {value}")
        logger.info("=" * 60)
        
        # Validation steps
        steps = [
            ("Python Version", self.check_python_version),
            ("Hardware Requirements", self.validate_hardware),
            ("Directory Structure", self.create_directories),
            ("Environment Template", self.create_env_template),
            ("Python Requirements", self.install_requirements),
            ("CUDA Support", self.install_cuda_support),
            ("Import Testing", self.test_imports),
            ("GPU Testing", self.test_gpu_availability)
        ]
        
        failed_steps = []
        
        for step_name, step_func in steps:
            logger.info(f"\n--- {step_name} ---")
            try:
                if not step_func():
                    failed_steps.append(step_name)
            except Exception as e:
                logger.error(f"Error in {step_name}: {e}")
                failed_steps.append(step_name)
        
        # Summary
        logger.info("\n" + "=" * 60)
        logger.info("SETUP SUMMARY")
        logger.info("=" * 60)
        
        if failed_steps:
            logger.error(f"Failed steps: {failed_steps}")
            logger.error("Please resolve the issues above before running the bot")
            return False
        else:
            logger.info("✓ All setup steps completed successfully!")
            logger.info("\nNext steps:")
            logger.info("1. Edit .env file with your Kraken API credentials")
            logger.info("2. Download pre-trained models to models/pretrained/")
            logger.info("3. Run: python enhanced_bot.py --help")
            return True

if __name__ == "__main__":
    setup = EnvironmentSetup()
    success = setup.run_setup()
    sys.exit(0 if success else 1)