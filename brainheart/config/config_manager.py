import json
from pathlib import Path
from typing import Any


class ConfigManager:
    def __init__(self, config_fn: str = "config.local.json"):
        self.config_fn = Path(__file__).parent.absolute() / config_fn
        self.config = self._load_config()

    def _load_config(self):
        """Load configuration from JSON file"""
        try:
            with open(self.config_fn, 'r') as f:
                config = json.load(f)
            return config
        except FileNotFoundError:
            print(f"Config file not found: {self.config_fn}")
            return {}
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON: {e}")
            return {}

    def get_path(self, key: str, default: Any = None) -> Any:
        """Get configuration value with optional default"""
        return Path(self.config.get(key, default))


if __name__ == "__main__":
    # Load config and get data_path
    local_config = ConfigManager()

    fn = local_config.get_path("data_path", "test_value")
    print(fn)

