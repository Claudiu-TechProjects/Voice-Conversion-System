"""Quick training test — diagnostic"""
import sys, os, traceback
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

from voice_conversion.config import LIGHTVC_CFG, AUDIO_DATASET

print(f"Audio dataset path: {AUDIO_DATASET}")
print(f"Exists: {AUDIO_DATASET.exists()}")
if AUDIO_DATASET.exists():
    import os
    files = list(AUDIO_DATASET.rglob("*.mp3")) + list(AUDIO_DATASET.rglob("*.wav"))
    print(f"Audio files found: {len(files)}")
    if files:
        print(f"  First: {files[0]}")
        print(f"  Last:  {files[-1]}")

print("\n--- Testing LightVCTrainer ---")
try:
    from voice_conversion.training.trainer import LightVCTrainer

    cfg = LIGHTVC_CFG
    cfg.num_epochs = 2  # just 2 epochs
    cfg.n_pseudo_speakers = 3
    cfg.batch_size = 4

    trainer = LightVCTrainer(cfg)
    print("Setup...")
    ok = trainer.setup()
    print(f"Setup result: {ok}")

    if ok:
        print(f"Train loader batches: {len(trainer.train_loader)}")
        print(f"Val loader batches: {len(trainer.val_loader)}")

        # Test one batch
        batch = next(iter(trainer.train_loader))
        print(f"Batch keys: {list(batch.keys())}")
        print(f"  mel_src shape: {batch['mel_src'].shape}")
        print(f"  mel_tgt shape: {batch['mel_tgt'].shape}")
        print(f"  speaker_id: {batch['speaker_id']}")

        print("\nStarting mini-training (2 epochs)...")
        trainer.train(num_epochs=2)
        print("\nTraining completed!")
        print(f"Progress: {trainer.get_progress()}")

except Exception as e:
    print(f"\nERROR: {e}")
    traceback.print_exc()
