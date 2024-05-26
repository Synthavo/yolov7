import logging
from copy import deepcopy
from pathlib import Path

from ruamel.yaml import YAML

from yolov7.model.configuration import Configuration as ConfigurationModel

log = logging.getLogger(__name__)


class Configuration(ConfigurationModel):
    @staticmethod
    def from_string(s: str):
        return YAML(typ="safe").load(s)

    @staticmethod
    def from_file(p: Path):
        with open(p) as f:
            return Configuration.from_string(f)

    @staticmethod
    def is_file(something):
        try:
            return Path(something).is_file()
        except Exception:
            return False

    def __init__(self, raw=None):
        if raw is None:
            raw = {}

        if Configuration.is_file(raw):
            kwargs = Configuration.from_file(raw)
        elif isinstance(raw, str):
            kwargs = Configuration.from_string(raw)
        elif isinstance(raw, dict):
            kwargs = raw
        else:
            raise Exception(f"Configuration format not recognized: {raw}")

        super().__init__(**kwargs)


_conf = None


def get_config() -> Configuration:
    global _conf
    if _conf is None:
        log.warning("Configuration loaded with defaults, not with config file!")
        _conf = Configuration()
    return deepcopy(_conf)


def save_config(raw):
    global _conf
    log.info(f"Use service config: {raw}")
    _conf = Configuration(raw=raw)
    return _conf
