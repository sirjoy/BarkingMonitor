# Dog Bark Monitor (macOS)

Production-ready Python desktop app for continuous dog bark detection using **YAMNet** and local analytics storage.

## Features
- Real-time microphone monitoring at 16kHz mono.
- Dog bark detection with TensorFlow Hub YAMNet.
- Confirmed event logic with consecutive detections and cooldown.
- Daily analytics, hourly histogram, time-series trends, duration distribution.
- SQLite persistence + JSON/CSV export.
- PyQt6 GUI with Monitoring, Analytics, and Data Management tabs.
- Configurable threshold/cooldown and optional auto-start.

## Project Structure
- `main.py` — app entrypoint + logging init
- `config.py` — app configuration and directory bootstrapping
- `detector.py` — YAMNet model wrapper
- `audio_engine.py` — real-time capture and event lifecycle
- `database.py` — SQLite persistence + export helpers
- `analytics.py` — summary/aggregation utilities
- `gui.py` — PyQt6 UI and chart rendering

## Requirements
- macOS (Apple Silicon compatible)
- Python 3.10+
- Microphone access permissions

## Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Run
```bash
python main.py
```

## Microphone Permissions (macOS)
1. Run the app once.
2. Approve microphone access in the prompt.
3. If blocked later: **System Settings → Privacy & Security → Microphone** and enable Terminal/Python (or packaged app).

## Data Layout
Created automatically at:
```text
~/DogBarkMonitor/
  database.sqlite
  config.json
  exports/
  logs/
```

## Event Logic
- Window: 0.96s
- Hop: 0.48s
- Threshold default: 0.65
- Confirm bark after 2 consecutive positive detections
- 2s cooldown after confirmed detection
- Detections within 2s are merged into one event

## Export
Use Data Management tab:
- Export selected day → JSON + CSV
- Export date range → JSON + CSV

## Delete Operations
- Delete selected day
- Delete all data (with confirmation)

## Packaging with PyInstaller
```bash
pip install pyinstaller
pyinstaller --name DogBarkMonitor --windowed --onefile main.py
```

Generated app appears in `dist/`.

## Notes on Performance
- YAMNet is loaded once at startup.
- Audio stream + model inference run off the GUI thread.
- Stream restart loop handles disconnect/interruption scenarios.
