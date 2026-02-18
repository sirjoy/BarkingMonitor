from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
import sys

from PyQt6.QtWidgets import QApplication

from audio_engine import AudioEngine
from config import APP_DIR, ensure_dirs, load_config
from database import BarkDatabase
from detector import BarkDetector, ThunderDetector
from gui import MainWindow


def setup_logging() -> None:
    ensure_dirs()
    log_file = APP_DIR / "logs" / "app.log"
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers.clear()
    formatter = logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s")

    fh = RotatingFileHandler(log_file, maxBytes=2_000_000, backupCount=3)
    fh.setFormatter(formatter)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(formatter)

    root.addHandler(fh)
    root.addHandler(sh)


def main() -> int:
    setup_logging()
    config = load_config()
    db = BarkDatabase()
    detector = BarkDetector(threshold=config.detection_threshold)
    
    thunder_detector = None
    if config.enable_thunder_monitoring:
        thunder_detector = ThunderDetector(threshold=config.thunder_detection_threshold)
    
    engine = AudioEngine(
        config=config, 
        detector=detector, 
        db=db,
        thunder_detector=thunder_detector
    )

    app = QApplication(sys.argv)
    win = MainWindow(app=app, config=config, db=db, engine=engine)
    win.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
