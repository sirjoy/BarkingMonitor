from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub

LOGGER = logging.getLogger(__name__)


_YAMNET_MODEL = None
_CLASS_NAMES = None


def _get_yamnet_model():
    """Load YAMNet model singleton to avoid loading multiple times.
    
    Returns:
        Tuple of (model, class_names list)
    """
    global _YAMNET_MODEL, _CLASS_NAMES
    if _YAMNET_MODEL is None:
        LOGGER.info("Loading YAMNet model from TensorFlow Hub...")
        _YAMNET_MODEL = hub.load("https://tfhub.dev/google/yamnet/1")
        class_map = _YAMNET_MODEL.class_map_path().numpy().decode("utf-8")
        _CLASS_NAMES = pd.read_csv(class_map)["display_name"].tolist()
        LOGGER.info("YAMNet loaded with %d classes", len(_CLASS_NAMES))
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
