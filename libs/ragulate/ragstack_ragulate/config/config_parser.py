from __future__ import annotations

from typing import TYPE_CHECKING, Any

import yaml
from cerberus import Validator

from .config_schema_0_1 import _VERSION_0_1, ConfigSchema0Dot1
from .objects import Config

if TYPE_CHECKING:
    from .base_config_schema import BaseConfigSchema


class ConfigParser:
    """Config parser."""

    _config_schema: BaseConfigSchema
    _valid: bool
    _errors: Any
    _document: Any

    def __init__(self, config_schema: BaseConfigSchema, config: dict[str, Any]):
        self._config_schema = config_schema
        validator = Validator(config_schema.schema())
        self.is_valid = validator.validate(config)
        self.errors = validator.errors
        self._document = validator.document

    def get_config(self) -> Config:
        """Return the config."""
        if not self.is_valid:
            return Config()
        return self._config_schema.parse_document(self._document)

    @classmethod
    def from_file(cls, file_path: str) -> ConfigParser:
        """Create a ConfigParser from a file."""
        with open(file_path) as file:
            config = yaml.safe_load(file)

            version = config.get("version", _VERSION_0_1)
            if version == _VERSION_0_1:
                return cls(config_schema=ConfigSchema0Dot1(), config=config)
            raise ValueError(f"config file version {version} is not supported")
