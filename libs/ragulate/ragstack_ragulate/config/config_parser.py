from typing import Any, Dict

import yaml
from cerberus import Validator

from .base_config_schema import BaseConfigSchema
from .config_schema_0_1 import ConfigSchema_0_1
from .objects import Config


class ConfigParser:
    _config_schema: BaseConfigSchema
    _valid: bool
    _errors: Any
    _document: Any

    def __init__(self, config_schema: BaseConfigSchema, config: Dict[str, Any]):
        self._config_schema = config_schema
        validator = Validator(config_schema.schema())
        self.is_valid = validator.validate(config)
        self.errors = validator.errors
        self._document = validator.document

    def get_config(self) -> Config:
        if not self.is_valid:
            return None
        return self._config_schema.parse_document(self._document)

    @classmethod
    def from_file(cls, file_path: str) -> "ConfigParser":
        with open(file_path, "r") as file:
            config = yaml.safe_load(file)

            version = config.get("version", 0.1)
            if version == 0.1:
                return cls(config_schema=ConfigSchema_0_1(), config=config)
            else:
                raise ValueError(f"config file version {version} is not supported")
