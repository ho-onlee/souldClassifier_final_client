from souldClassifier_final_client.src.sound_to_tensor import extract_enhanced_features
import numpy as np

def test_extract_enhanced_features():
    # Generate a synthetic audio signal (sine wave + noise)
    sr = 16000
    duration = 1.0  # seconds
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    signal = 0.5 * np.sin(2 * np.pi * 440 * t) + 0.05 * np.random.randn(len(t))

    features, mfcc = extract_enhanced_features(signal, sr=sr)

    assert len(features) > 0, "Features should not be empty"
    assert len(mfcc) > 0, "MFCC should not be empty"