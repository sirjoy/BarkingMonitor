from __future__ import annotations

import logging
import os
import shutil
from dataclasses import dataclass

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub

LOGGER = logging.getLogger(__name__)

_YAMNET_URL = "https://tfhub.dev/google/yamnet/1"
_YAMNET_MODEL = None
_CLASS_NAMES = None


def _get_tfhub_cache_dir() -> str:
    """Return the TF Hub module cache directory.

    Returns:
        Absolute path to the TF Hub cache directory.
    """
    return os.environ.get(
        "TFHUB_CACHE_DIR",
        os.path.join(os.environ.get("TMPDIR", "/tmp"), "tfhub_modules"),
    )


def _is_cache_corrupt(cache_dir: str) -> bool:
    """Check whether the TF Hub cache directory contains an incomplete model.

    A valid SavedModel must contain at least a ``saved_model.pb`` or
    ``saved_model.pbtxt`` file at its root.  Any sub-directory that lacks
    both files is considered corrupt.

    Args:
        cache_dir: Path to the TF Hub modules cache directory.

    Returns:
        ``True`` if a corrupt model directory is detected, ``False`` otherwise.
    """
    if not os.path.isdir(cache_dir):
        return False
    for entry in os.scandir(cache_dir):
        if not entry.is_dir():
            continue
        files = {f.name for f in os.scandir(entry.path)}
        if "saved_model.pb" not in files and "saved_model.pbtxt" not in files:
            LOGGER.warning(
                "Corrupt TF Hub cache detected at '%s' (missing saved_model.pb).",
                entry.path,
            )
            return True
    return False


def _clear_tfhub_cache(cache_dir: str) -> None:
    """Remove the TF Hub cache directory so models are re-downloaded.

    Args:
        cache_dir: Path to the TF Hub modules cache directory to remove.
    """
    if os.path.isdir(cache_dir):
        shutil.rmtree(cache_dir)
        LOGGER.info("Cleared TF Hub cache at '%s'.", cache_dir)


def _load_yamnet_with_recovery() -> tuple:
    """Load the YAMNet model, automatically recovering from a corrupt cache.

    On first failure with a ``ValueError`` (corrupt/unknown model type), the
    cache is cleared and the download is retried once.

    Returns:
        Tuple of (model, class_names list).

    Raises:
        RuntimeError: If the model cannot be loaded after the recovery attempt.
    """
    cache_dir = _get_tfhub_cache_dir()

    if _is_cache_corrupt(cache_dir):
        LOGGER.warning("Corrupt TF Hub cache found before loading. Clearing and retrying.")
        _clear_tfhub_cache(cache_dir)

    for attempt in range(1, 3):
        try:
            LOGGER.info("Loading YAMNet model (attempt %d)...", attempt)
            model = hub.load(_YAMNET_URL)
            class_map = model.class_map_path().numpy().decode("utf-8")
            class_names = pd.read_csv(class_map)["display_name"].tolist()
            LOGGER.info("YAMNet loaded with %d classes", len(class_names))
            return model, class_names
        except ValueError as exc:
            LOGGER.error("Failed to load YAMNet model: %s", exc)
            if attempt == 1:
                LOGGER.warning("Attempting recovery: clearing TF Hub cache and retrying.")
                _clear_tfhub_cache(cache_dir)
            else:
                raise RuntimeError(
                    "YAMNet model could not be loaded after cache recovery."
                ) from exc
    raise RuntimeError("YAMNet model could not be loaded.")  # DELETE - unreachable guard


def _get_yamnet_model() -> tuple:
    """Return the YAMNet model singleton, loading it on first call.

    Returns:
        Tuple of (model, class_names list).
    """
    global _YAMNET_MODEL, _CLASS_NAMES
    if _YAMNET_MODEL is None:
        _YAMNET_MODEL, _CLASS_NAMES = _load_yamnet_with_recovery()
    return _YAMNET_MODEL, _CLASS_NAMES


@dataclass
class DetectionResult:
    confidence: float
    is_bark: bool


class BarkDetector:
    """YAMNet-backed detector for dog bark class."""

    def __init__(self, threshold: float = 0.65) -> None:
        self.threshold = threshold
        self.model, class_names = _get_yamnet_model()
        self.bark_index = class_names.index("Bark")
        LOGGER.info("BarkDetector initialized. Bark class index=%s", self.bark_index)

    def detect(self, audio: np.ndarray) -> DetectionResult:
        waveform = tf.convert_to_tensor(audio, dtype=tf.float32)
        scores, _, _ = self.model(waveform)
        scores_np = scores.numpy()
        bark_conf = float(np.max(scores_np[:, self.bark_index]))
        return DetectionResult(confidence=bark_conf, is_bark=bark_conf >= self.threshold)


class ThunderDetector:
    """YAMNet-backed detector for thunder/thunderstorm classes."""

    def __init__(self, threshold: float = 0.55) -> None:
        self.threshold = threshold
        self.model, class_names = _get_yamnet_model()
        
        self.thunder_indices = []
        if "Thunder" in class_names:
            self.thunder_indices.append(class_names.index("Thunder"))
        if "Thunderstorm" in class_names:
            self.thunder_indices.append(class_names.index("Thunderstorm"))
        
        if not self.thunder_indices:
            LOGGER.warning("Thunder/Thunderstorm classes not found in YAMNet")
        
        LOGGER.info("ThunderDetector initialized. Thunder class indices=%s", self.thunder_indices)

    def detect(self, audio: np.ndarray) -> DetectionResult:
        if not self.thunder_indices:
            return DetectionResult(confidence=0.0, is_bark=False)
        
        waveform = tf.convert_to_tensor(audio, dtype=tf.float32)
        scores, _, _ = self.model(waveform)
        scores_np = scores.numpy()
        
        thunder_conf = 0.0
        for idx in self.thunder_indices:
            conf = float(np.max(scores_np[:, idx]))
            thunder_conf = max(thunder_conf, conf)
        
        is_detected = thunder_conf >= self.threshold
        if thunder_conf > 0.1:
            LOGGER.debug("Thunder detection: confidence=%.3f, threshold=%.3f, detected=%s", 
                        thunder_conf, self.threshold, is_detected)
        
        return DetectionResult(confidence=thunder_conf, is_bark=is_detected)
