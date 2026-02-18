from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path

APP_DIR = Path.home() / "DogBarkMonitor"
CONFIG_PATH = APP_DIR / "config.json"


@dataclass
class AppConfig:
    sample_rate: int = 16000
    channels: int = 1
    window_seconds: float = 0.96
    hop_seconds: float = 0.48
    detection_threshold: float = 0.65
    consecutive_required: int = 2
    cooldown_seconds: float = 2.0
    merge_gap_seconds: float = 2.0
    auto_start: bool = False
    enable_thunder_monitoring: bool = True
    thunder_detection_threshold: float = 0.55
    thunder_consecutive_required: int = 2
    thunder_cooldown_seconds: float = 2.0


DEFAULT_CONFIG = AppConfig()


def ensure_dirs() -> None:
    (APP_DIR / "exports").mkdir(parents=True, exist_ok=True)
    (APP_DIR / "logs").mkdir(parents=True, exist_ok=True)


def load_config() -> AppConfig:
    ensure_dirs()
    if not CONFIG_PATH.exists():
        save_config(DEFAULT_CONFIG)
        return DEFAULT_CONFIG

    data = json.loads(CONFIG_PATH.read_text())
    base = asdict(DEFAULT_CONFIG)
    base.update(data)
    return AppConfig(**base)


def save_config(config: AppConfig) -> None:
    ensure_dirs()
    CONFIG_PATH.write_text(json.dumps(asdict(config), indent=2))
