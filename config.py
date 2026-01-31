"""
RRMC Configuration System.

Loads and merges YAML configs with environment variable expansion
and CLI override support.
"""

import os
import re
from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml


class Config:
    """
    Configuration object with dot-notation access.

    Example:
        cfg = Config({"model": {"name": "gpt-4"}, "max_turns": 10})
        print(cfg.model.name)  # "gpt-4"
        print(cfg.max_turns)   # 10
    """

    def __init__(self, data: Dict[str, Any]):
        for key, value in data.items():
            if isinstance(value, dict):
                setattr(self, key, Config(value))
            else:
                setattr(self, key, value)

    def __repr__(self):
        return f"Config({self.to_dict()})"

    def to_dict(self) -> Dict[str, Any]:
        """Convert config back to dictionary."""
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, Config):
                result[key] = value.to_dict()
            else:
                result[key] = value
        return result

    def get(self, key: str, default: Any = None) -> Any:
        """Get value with optional default."""
        return getattr(self, key, default)


def expand_env_vars(value: Any) -> Any:
    """
    Expand environment variables in string values.

    Supports ${VAR_NAME} syntax.
    """
    if isinstance(value, str):
        pattern = r'\$\{([^}]+)\}'
        matches = re.findall(pattern, value)
        for var_name in matches:
            env_value = os.environ.get(var_name, '')
            value = value.replace(f'${{{var_name}}}', env_value)
        return value
    elif isinstance(value, dict):
        return {k: expand_env_vars(v) for k, v in value.items()}
    elif isinstance(value, list):
        return [expand_env_vars(item) for item in value]
    return value


def load_yaml(path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load a single YAML file.

    Args:
        path: Path to YAML file

    Returns:
        Dictionary of config values
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, 'r') as f:
        data = yaml.safe_load(f) or {}

    return expand_env_vars(data)


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deep merge two dictionaries.

    Override values take precedence. Nested dicts are merged recursively.
    """
    result = base.copy()

    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value

    return result


def load_config(
    experiment: Optional[str] = None,
    overrides: Optional[Dict[str, Any]] = None,
    config_dir: Optional[Union[str, Path]] = None,
) -> Config:
    """
    Load and merge configuration from multiple sources.

    Merge order (later overrides earlier):
    1. configs/base.yaml (defaults)
    2. configs/providers/{provider}.yaml (API settings)
    3. configs/experiments/{experiment}.yaml (experiment-specific)
    4. CLI overrides

    Args:
        experiment: Experiment name or path to experiment config
        overrides: CLI overrides as dict
        config_dir: Base config directory (default: ./configs)

    Returns:
        Merged Config object
    """
    if config_dir is None:
        config_dir = Path(__file__).parent / "configs"
    else:
        config_dir = Path(config_dir)

    # Start with base config
    base_path = config_dir / "base.yaml"
    if base_path.exists():
        config = load_yaml(base_path)
    else:
        config = {}

    # Load provider config (default: openrouter)
    provider = config.get("provider", "openrouter")
    provider_path = config_dir / "providers" / f"{provider}.yaml"
    if provider_path.exists():
        provider_config = load_yaml(provider_path)
        config = deep_merge(config, provider_config)

    # Load experiment config
    if experiment:
        # Check if it's a path or an experiment name (supports nested experiment names like
        # "dc_methods/fixed_turns" -> configs/experiments/dc_methods/fixed_turns.yaml).
        exp_arg = str(experiment)
        exp_as_path = Path(exp_arg)

        # Treat as a filesystem path if:
        # - it explicitly looks like a yaml path, OR
        # - it is absolute, OR
        # - it exists on disk (relative or absolute)
        if exp_arg.endswith(".yaml") or exp_as_path.is_absolute() or exp_as_path.exists():
            exp_path = exp_as_path
        else:
            exp_path = config_dir / "experiments" / f"{exp_arg}.yaml"

        if exp_path.exists():
            exp_config = load_yaml(exp_path)
            config = deep_merge(config, exp_config)
        else:
            raise FileNotFoundError(f"Experiment config not found: {exp_path}")

    # Apply CLI overrides
    if overrides:
        # Filter out None values
        overrides = {k: v for k, v in overrides.items() if v is not None}
        config = deep_merge(config, overrides)

    return Config(config)


def get_method_config(method_name: str, config_dir: Optional[Path] = None) -> Dict[str, Any]:
    """
    Load method-specific configuration.

    Args:
        method_name: Name of the method (e.g., "robust_mi")
        config_dir: Config directory

    Returns:
        Method config dict
    """
    if config_dir is None:
        config_dir = Path(__file__).parent / "configs"

    method_path = config_dir / "methods" / f"{method_name}.yaml"
    if method_path.exists():
        return load_yaml(method_path)
    return {}


def list_experiments(config_dir: Optional[Path] = None) -> list:
    """List available experiment configs."""
    if config_dir is None:
        config_dir = Path(__file__).parent / "configs"

    exp_dir = config_dir / "experiments"
    if not exp_dir.exists():
        return []

    # Include nested experiments (e.g. experiments/dc_methods/*.yaml)
    names = []
    for p in exp_dir.rglob("*.yaml"):
        rel = p.relative_to(exp_dir)
        # Drop ".yaml" while preserving subdirectories
        names.append(str(rel.with_suffix("")).replace("\\", "/"))
    return sorted(names)
