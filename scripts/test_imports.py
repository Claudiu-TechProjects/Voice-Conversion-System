import sys
sys.path.insert(0, '.')

print("1. config...")
from voice_conversion.config import AUDIO_CFG, KNN_VC_CFG, LIGHTVC_CFG
print(f"   LIGHTVC_CFG bottleneck={LIGHTVC_CFG.bottleneck_dim}, epochs={LIGHTVC_CFG.num_epochs}")

print("2. data modules...")
from voice_conversion.data.speaker_clustering import SpeakerClusterer
from voice_conversion.data.vc_dataset import MelExtractor, VoiceConversionDataset
print("   OK")

print("3. LightVC model...")
import torch
from voice_conversion.models.lightvc import LightVCModel, LightVCConverter
model = LightVCModel(n_mels=80, content_channels=128, bottleneck_dim=32,
                     speaker_emb_dim=192, decoder_channels=256, num_speakers=5)
n = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"   Params: {n:,}")
mel = torch.randn(2, 80, 64)
spk = torch.tensor([0, 1])
out = model(mel, spk)
print(f"   Forward OK: mel_pred={out['mel_pred'].shape}")

print("4. trainer...")
from voice_conversion.training.trainer import LightVCTrainer
print("   OK")

print("\nAll imports successful!")
