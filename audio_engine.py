from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Callable, Optional

import numpy as np
import sounddevice as sd

from config import AppConfig
from database import BarkDatabase, BarkEvent, ThunderEvent
from detector import BarkDetector, ThunderDetector

LOGGER = logging.getLogger(__name__)


@dataclass
class RuntimeStats:
    listening: bool = False
    latest_confidence: float = 0.0
    today_count: int = 0
    current_day: str = ""
    latest_thunder_confidence: float = 0.0
    today_thunder_count: int = 0


class AudioEngine:
    def __init__(
        self,
        config: AppConfig,
        detector: BarkDetector,
        db: BarkDatabase,
        on_event: Optional[Callable[[BarkEvent], None]] = None,
        thunder_detector: Optional[ThunderDetector] = None,
        on_thunder_event: Optional[Callable[[ThunderEvent], None]] = None,
    ) -> None:
        self.config = config
        self.detector = detector
        self.thunder_detector = thunder_detector
        self.db = db
        self.on_event = on_event
        self.on_thunder_event = on_thunder_event
        self.stats = RuntimeStats()

        self._audio_queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=20)
        self._stop_evt = threading.Event()
        self._pause_evt = threading.Event()
        self._stream = None
        self._threads: list[threading.Thread] = []

        self._lock = threading.Lock()
        self._consecutive = 0
        self._cooldown_until = datetime.min
        self._active_event: Optional[dict] = None
        self._last_day_check = datetime.min
        
        self._thunder_consecutive = 0
        self._thunder_cooldown_until = datetime.min
        self._active_thunder_event: Optional[dict] = None

        self.window_samples = int(config.sample_rate * config.window_seconds)
        self.hop_samples = int(config.sample_rate * config.hop_seconds)
        today = date.today().isoformat()
        self.stats.today_count = self.db.count_for_day(today)
        self.stats.today_thunder_count = self.db.count_thunder_for_day(today)
        self.stats.current_day = today

    def start(self) -> None:
        if self.stats.listening:
            return
        self._stop_evt.clear()
        self._pause_evt.clear()
        self.stats.listening = True
        self._threads = [
            threading.Thread(target=self._stream_worker, daemon=True),
            threading.Thread(target=self._process_worker, daemon=True),
        ]
        for t in self._threads:
            t.start()

    def pause(self) -> None:
        self._pause_evt.set()
        self.stats.listening = False

    def resume(self) -> None:
        self._pause_evt.clear()
        self.stats.listening = True

    def stop(self) -> None:
        self._stop_evt.set()
        self.stats.listening = False
        self._finalize_active_event()
        self._finalize_active_thunder_event()
        
        for t in self._threads:
            t.join(timeout=2.0)
            if t.is_alive():
                LOGGER.warning("Thread %s did not stop in time", t.name)
        
        if self._stream is not None:
            self._stream.abort(ignore_errors=True)
            self._stream = None

    def _audio_callback(self, indata, frames, time_info, status):
        if status:
            LOGGER.warning("Audio callback status: %s", status)
        if self._pause_evt.is_set():
            return
        chunk = np.copy(indata[:, 0])
        try:
            self._audio_queue.put_nowait(chunk)
        except queue.Full:
            LOGGER.warning("Audio queue full; dropping chunk")

    def _stream_worker(self) -> None:
        while not self._stop_evt.is_set():
            try:
                with sd.InputStream(
                    samplerate=self.config.sample_rate,
                    channels=self.config.channels,
                    dtype="float32",
                    callback=self._audio_callback,
                    blocksize=self.hop_samples,
                    latency="low",
                ) as stream:
                    self._stream = stream
                    LOGGER.info("Microphone stream started")
                    while not self._stop_evt.is_set():
                        time.sleep(0.25)
            except Exception as exc:
                LOGGER.exception("Microphone stream interrupted: %s", exc)
                time.sleep(1.0)

    def _process_worker(self) -> None:
        ring = np.zeros(self.window_samples, dtype=np.float32)
        filled = 0
        while not self._stop_evt.is_set():
            now = datetime.now()
            if (now - self._last_day_check).total_seconds() >= 60:
                self._check_day_change()
                self._last_day_check = now
            
            try:
                hop = self._audio_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            if len(hop) != self.hop_samples:
                hop = np.resize(hop, self.hop_samples)

            ring = np.roll(ring, -self.hop_samples)
            ring[-self.hop_samples :] = hop
            filled = min(self.window_samples, filled + self.hop_samples)
            if filled < self.window_samples:
                continue

            result = self.detector.detect(ring)
            self.stats.latest_confidence = result.confidence
            self._handle_detection(result.confidence, result.is_bark)
            
            if self.thunder_detector:
                thunder_result = self.thunder_detector.detect(ring)
                self.stats.latest_thunder_confidence = thunder_result.confidence
                self._handle_thunder_detection(thunder_result.confidence, thunder_result.is_bark)

    def _handle_detection(self, confidence: float, is_bark: bool) -> None:
        now = datetime.now()
        with self._lock:
            if now < self._cooldown_until:
                self._consecutive = 0
                return

            if is_bark:
                self._consecutive += 1
                if self._consecutive >= self.config.consecutive_required:
                    self._register_detection(now, confidence)
                    self._consecutive = 0
                    self._cooldown_until = now + timedelta(seconds=self.config.cooldown_seconds)
            else:
                self._consecutive = 0

            self._maybe_finalize_event(now)

    def _register_detection(self, ts: datetime, confidence: float) -> None:
        if self._active_event and (ts - self._active_event["last_detection"]).total_seconds() <= self.config.merge_gap_seconds:
            self._active_event["last_detection"] = ts
            self._active_event["confidences"].append(confidence)
            return

        self._finalize_active_event()
        self._active_event = {
            "start": ts,
            "last_detection": ts,
            "confidences": [confidence],
        }

    def _maybe_finalize_event(self, now: datetime) -> None:
        if not self._active_event:
            return
        gap = (now - self._active_event["last_detection"]).total_seconds()
        if gap > self.config.merge_gap_seconds:
            self._finalize_active_event()

    def _check_day_change(self) -> None:
        """Check if the day has changed and reset today's count if needed.
        
        This method resets the daily event counter when a new day is detected.
        """
        today = date.today().isoformat()
        if today != self.stats.current_day:
            LOGGER.info("Day changed from %s to %s - resetting today_count", self.stats.current_day, today)
            self.stats.current_day = today
            self.stats.today_count = self.db.count_for_day(today)
            self.stats.today_thunder_count = self.db.count_thunder_for_day(today)

    def _finalize_active_event(self) -> None:
        if not self._active_event:
            return
        start = self._active_event["start"]
        end = self._active_event["last_detection"]
        duration = max((end - start).total_seconds(), self.config.window_seconds)
        avg_conf = float(np.mean(self._active_event["confidences"]))
        event = BarkEvent(
            start_ts=start.isoformat(timespec="seconds"),
            end_ts=end.isoformat(timespec="seconds"),
            duration_sec=duration,
            avg_confidence=avg_conf,
        )
        self._check_day_change()
        self.db.add_event(event)
        self.stats.today_count += 1
        if self.on_event:
            self.on_event(event)
        LOGGER.info("Bark event saved: %s", event)
        self._active_event = None

    def _handle_thunder_detection(self, confidence: float, is_thunder: bool) -> None:
        """Handle thunder detection result with separate state management.
        
        Args:
            confidence: Detection confidence score
            is_thunder: Whether thunder was detected
        """
        now = datetime.now()
        with self._lock:
            if now < self._thunder_cooldown_until:
                self._thunder_consecutive = 0
                return

            if is_thunder:
                self._thunder_consecutive += 1
                if self._thunder_consecutive >= self.config.thunder_consecutive_required:
                    self._register_thunder_detection(now, confidence)
                    self._thunder_consecutive = 0
                    self._thunder_cooldown_until = now + timedelta(seconds=self.config.thunder_cooldown_seconds)
            else:
                self._thunder_consecutive = 0

            self._maybe_finalize_thunder_event(now)

    def _register_thunder_detection(self, ts: datetime, confidence: float) -> None:
        """Register a thunder detection, merging with active event if within gap.
        
        Args:
            ts: Detection timestamp
            confidence: Detection confidence
        """
        if self._active_thunder_event and (ts - self._active_thunder_event["last_detection"]).total_seconds() <= self.config.merge_gap_seconds:
            self._active_thunder_event["last_detection"] = ts
            self._active_thunder_event["confidences"].append(confidence)
            return

        self._finalize_active_thunder_event()
        self._active_thunder_event = {
            "start": ts,
            "last_detection": ts,
            "confidences": [confidence],
        }

    def _maybe_finalize_thunder_event(self, now: datetime) -> None:
        """Finalize active thunder event if gap exceeded.
        
        Args:
            now: Current timestamp
        """
        if not self._active_thunder_event:
            return
        gap = (now - self._active_thunder_event["last_detection"]).total_seconds()
        if gap > self.config.merge_gap_seconds:
            self._finalize_active_thunder_event()

    def _finalize_active_thunder_event(self) -> None:
        """Finalize and save the active thunder event to database."""
        if not self._active_thunder_event:
            return
        start = self._active_thunder_event["start"]
        end = self._active_thunder_event["last_detection"]
        duration = max((end - start).total_seconds(), self.config.window_seconds)
        avg_conf = float(np.mean(self._active_thunder_event["confidences"]))
        event = ThunderEvent(
            start_ts=start.isoformat(timespec="seconds"),
            end_ts=end.isoformat(timespec="seconds"),
            duration_sec=duration,
            avg_confidence=avg_conf,
        )
        self._check_day_change()
        self.db.add_thunder_event(event)
        self.stats.today_thunder_count += 1
        if self.on_thunder_event:
            self.on_thunder_event(event)
        LOGGER.info("Thunder event saved: %s", event)
        self._active_thunder_event = None
