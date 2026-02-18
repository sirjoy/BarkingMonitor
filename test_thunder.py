"""Test thunder detection to verify it's working correctly."""
import logging
import numpy as np
from detector import ThunderDetector

logging.basicConfig(level=logging.INFO)

# Create thunder detector
detector = ThunderDetector(threshold=0.55)

# Create test audio (random noise as placeholder)
test_audio = np.random.randn(15360).astype(np.float32) * 0.1

# Try detection
result = detector.detect(test_audio)
print(f"Thunder detection result:")
print(f"  Confidence: {result.confidence:.4f}")
print(f"  Detected: {result.is_bark}")
print(f"  Thunder indices: {detector.thunder_indices}")
print(f"  Threshold: {detector.threshold}")
