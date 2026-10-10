import yaml
from pathlib import Path
import logging

class Settings:
    """
    Configuration manager for the Breast Cancer MLOps project.
    Loads settings from a YAML file and provides typed access.
    """
    def __init__(self, config_path: str = "configs/config.yaml"):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self._setup_logging()

    def _load_config(self):
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found at {self.config_path}")
        with open(self.config_path, "r") as f:
            return yaml.safe_load(f)

    def _setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler("logs/pipeline.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger("breast_cancer_ml")

    def get(self, key_path: str, default=None):
        """Retrieve value from nested config using dot notation (e.g., 'data.raw_path')."""
        keys = key_path.split(".")
        value = self.config
        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default

# Singleton instance
settings = Settings()
logger = settings.logger
