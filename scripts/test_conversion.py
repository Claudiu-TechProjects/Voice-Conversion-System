"""Quick test — LightVC conversion"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

from voice_conversion.models.lightvc import LightVCConverter

print("=== LightVC Conversion Test ===")
converter = LightVCConverter()
converter.load_model()
print(f"Model loaded: {converter.is_loaded}")
print(f"Checkpoint info: epoch={converter.checkpoint_info.get('epoch')}")

# Find a test audio file
from voice_conversion.config import AUDIO_DATASET
test_files = list(AUDIO_DATASET.rglob("*.wav"))[:2]
if len(test_files) >= 2:
    source = str(test_files[0])
    reference = str(test_files[1])
    print(f"\nSource: {source}")
    print(f"Reference: {reference}")

    result = converter.convert(
        source_audio=source,
        target_references=[reference]
    )
    print(f"\nConversion OK!")
    print(f"  Duration: {result.get_duration():.2f}s")
    print(f"  Time: {result.conversion_time:.2f}s")
    print(f"  Audio shape: {result.converted_audio.shape}")
    print(f"  Audio range: [{result.converted_audio.min():.3f}, {result.converted_audio.max():.3f}]")

    # Save test output
    out_path = result.save("test_lightvc_output.wav")
    print(f"  Saved to: {out_path}")
else:
    print("Not enough test files found")
