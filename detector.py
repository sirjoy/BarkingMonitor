from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub

LOGGER = logging.getLogger(__name__)


@dataclass
class DetectionResult:
    confidence: float
    is_bark: bool


class BarkDetector:
    """YAMNet-backed detector for dog bark class."""

    def __init__(self, threshold: float = 0.65) -> None:
        self.threshold = threshold
        LOGGER.info("Loading YAMNet model from TensorFlow Hub...")
        self.model = hub.load("https://tfhub.dev/google/yamnet/1")
        class_map = self.model.class_map_path().numpy().decode("utf-8")
        class_names = pd.read_csv(class_map)["display_name"].tolist()
        self.bark_index = class_names.index("Dog bark")
        LOGGER.info("YAMNet loaded. Dog bark class index=%s", self.bark_index)

    def detect(self, audio: np.ndarray) -> DetectionResult:
        waveform = tf.convert_to_tensor(audio, dtype=tf.float32)
        scores, _, _ = self.model(waveform)
        scores_np = scores.numpy()
        bark_conf = float(np.max(scores_np[:, self.bark_index]))
        return DetectionResult(confidence=bark_conf, is_bark=bark_conf >= self.threshold)
