# Dog Bark Monitor (macOS)

Production-ready Python desktop app for continuous dog bark detection using **YAMNet** and local analytics storage.

## Features
- Real-time microphone monitoring at 16kHz mono.
- Dog bark detection with TensorFlow Hub YAMNet.
- Thunder/thunderstorm detection for environmental correlation analysis.
- Confirmed event logic with consecutive detections and cooldown.
- Daily analytics, hourly histogram, time-series trends, duration distribution.
- Bark-thunder temporal correlation analysis with configurable time windows.
- SQLite persistence + JSON/CSV export.
- PyQt6 GUI with Monitoring, Analytics, and Data Management tabs.
- Configurable threshold/cooldown for both bark and thunder detection.
- Optional auto-start and thunder monitoring toggle.

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

### Using uv (Recommended)
```bash
# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment and install dependencies
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -r requirements.txt
```

### Using pip
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Run

### Using uv
```bash
uv run main.py
```

### Using python directly
```bash
python main.py
```

## Microphone Permissions (macOS)
1. Run the app once.
2. Approve microphone access in the prompt.
3. If blocked later: **System Settings → Privacy & Security → Microphone** and enable Terminal/Python (or packaged app).

## Configuration

The app stores configuration in `~/DogBarkMonitor/config.json`:

### Bark Detection Settings
- `detection_threshold`: Bark detection confidence threshold (default: 0.65)
- `consecutive_required`: Number of consecutive detections needed (default: 2)
- `cooldown_seconds`: Cooldown after bark detection (default: 2.0)

### Thunder Detection Settings
- `enable_thunder_monitoring`: Enable/disable thunder detection (default: true)
- `thunder_detection_threshold`: Thunder detection confidence threshold (default: 0.55)
- `thunder_consecutive_required`: Number of consecutive thunder detections needed (default: 2)
- `thunder_cooldown_seconds`: Cooldown after thunder detection (default: 2.0)

### General Settings
- `auto_start`: Auto-start listening on launch (default: false)

## Data Layout
Created automatically at:
```text
~/DogBarkMonitor/
  database.sqlite         # Bark and thunder event storage
  config.json             # User settings
  exports/                # Exported JSON/CSV files
  logs/                   # Application logs
```

## Event Logic

### Bark Detection
- Window: 0.96s
- Hop: 0.48s
- Threshold default: 0.65
- Confirm bark after 2 consecutive positive detections
- 2s cooldown after confirmed detection
- Detections within cooldown period are merged into one event

### Thunder Detection
- Same audio processing pipeline as bark detection
- Independent threshold (default: 0.55) and cooldown (default: 2.0s)
- Can be disabled via `enable_thunder_monitoring` setting
- Uses YAMNet classes: Thunder (index 429) and Thunderstorm (index 430)

## Analytics

The Analytics tab provides three analysis views:

### Barking Events
- Daily summary statistics (total events, avg confidence, duration)
- Hourly histogram showing bark frequency by hour of day
- Timeline chart of all bark events
- Duration distribution histogram

### Thunder Events
- Daily summary statistics for thunder detections
- Hourly histogram showing thunder frequency by hour of day
- Timeline chart of all thunder events
- Duration distribution histogram

### Correlation Analysis
- Temporal correlation between bark and thunder events
- Configurable time window for matching events (default: ±30 minutes)
- Statistics showing bark events before/after thunder
- Visual timeline showing correlated events

## Export & Data Management

The Data Management tab provides:

### Export Operations
- Export selected day → JSON + CSV (separate tabs for barks and thunder)
- Export date range → JSON + CSV
- Opens exports folder after successful export

### Delete Operations
- Delete selected day (barks or thunder separately)
- Delete all data (with confirmation)
- Separate controls for bark and thunder data

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
