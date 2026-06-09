"""
VoiceConversion Dataset — PyTorch
====================================

Dataset care servește perechi (sursă, target) pentru antrenarea LightVC.
Fiecare item conține:
  - mel spectrogram sursă (același vorbitor, utterance diferit)
  - mel spectrogram target (vorbitor diferit)
  - speaker embedding target (precalculat)
  - speaker_id (index numeric)
"""

import json
import random
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import torch
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import Dataset, DataLoader, random_split

logger = logging.getLogger(__name__)

# =====================================================================
# PARAMETRI AUDIO
# =====================================================================

SAMPLE_RATE = 16000
N_FFT = 1024
HOP_LENGTH = 256
WIN_LENGTH = 1024
N_MELS = 80
FMIN = 0.0
FMAX = 8000.0
MAX_FRAMES = 256   # ~4 secunde la 16kHz cu hop=256
MIN_FRAMES = 32    # ~0.5s minim


class MelExtractor:
    """Extractor mel spectrogram consistent cu HiFi-GAN."""

    def __init__(self, sample_rate: int = SAMPLE_RATE):
        self.mel_transform = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=N_FFT,
            win_length=WIN_LENGTH,
            hop_length=HOP_LENGTH,
            n_mels=N_MELS,
            f_min=FMIN,
            f_max=FMAX,
            power=1.0,
            norm="slaney",
            mel_scale="slaney"
        )
        self.amplitude_to_db = T.AmplitudeToDB(stype="amplitude", top_db=80)

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Args:
            waveform: [1, T] tensor
        Returns:
            mel: [n_mels, T'] tensor, valori normalizate în [-1, 1]
        """
        mel = self.mel_transform(waveform)  # [1, n_mels, T']
        mel = self.amplitude_to_db(mel)
        mel = mel.squeeze(0)  # [n_mels, T']
        # Normalizare la [-1, 1]
        mel = (mel + 40.0) / 40.0  # mapăm [-80, 0] → [-1, 1] aproximativ
        mel = mel.clamp(-1.0, 1.0)
        return mel


def load_audio_mono(path: str, target_sr: int = SAMPLE_RATE) -> Optional[torch.Tensor]:
    """Încarcă audio mono resampled."""
    try:
        waveform, sr = torchaudio.load(path)
        if sr != target_sr:
            resampler = T.Resample(sr, target_sr)
            waveform = resampler(waveform)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        return waveform
    except Exception as e:
        logger.debug(f"Eroare incarcare {path}: {e}")
        return None


# =====================================================================
# DATASET
# =====================================================================

class VoiceConversionDataset(Dataset):
    """
    Dataset pentru antrenarea LightVC.

    Pentru fiecare item returnează:
        - mel_src: mel spectrogram sursă [n_mels, T]
        - mel_tgt: mel spectrogram target (același vorbitor ca target_emb) [n_mels, T]
        - speaker_id: index numeric al vorbitorului target
        - target_speaker_name: string pentru logging

    Strategia de perechi:
        - src și tgt vin din VORBITORI DIFERIȚI (many-to-many conversion)
        - La antrenare: src=random_utterance_speaker_A, tgt=random_utterance_speaker_B
        - Modelul trebuie să transforme conținut din A în stilul lui B
    """

    def __init__(
        self,
        pseudo_speakers_path: str,
        max_frames: int = MAX_FRAMES,
        min_frames: int = MIN_FRAMES,
        augment: bool = True,
        cache_mels: bool = True
    ):
        self.max_frames = max_frames
        self.min_frames = min_frames
        self.augment = augment
        self.mel_extractor = MelExtractor()

        # Încarcă structura pseudo-vorbitori
        with open(pseudo_speakers_path, 'r') as f:
            data = json.load(f)

        self.speakers: Dict[str, List[str]] = data["speakers"]
        self.speaker_names = sorted(self.speakers.keys())
        self.speaker_to_idx = {name: i for i, name in enumerate(self.speaker_names)}
        self.num_speakers = len(self.speaker_names)

        # Cache mel spectrograms pentru viteză
        self.mel_cache: Dict[str, torch.Tensor] = {} if cache_mels else None

        # Construiește lista plată de (audio_path, speaker_idx)
        self.samples: List[Tuple[str, int]] = []
        for speaker_name, files in self.speakers.items():
            speaker_idx = self.speaker_to_idx[speaker_name]
            for f in files:
                if Path(f).exists():
                    self.samples.append((f, speaker_idx))

        # Indexare per speaker pentru sampling rapid
        self.speaker_samples: Dict[int, List[str]] = {}
        for path, spk_idx in self.samples:
            if spk_idx not in self.speaker_samples:
                self.speaker_samples[spk_idx] = []
            self.speaker_samples[spk_idx].append(path)

        logger.info(
            f"Dataset: {len(self.samples)} sample-uri, "
            f"{self.num_speakers} vorbitori"
        )

    def _get_mel(self, path: str) -> Optional[torch.Tensor]:
        """Extrage sau returnează din cache mel spectrogram."""
        if self.mel_cache is not None and path in self.mel_cache:
            return self.mel_cache[path]

        waveform = load_audio_mono(path)
        if waveform is None:
            return None

        # Augmentare timp de antrenare
        if self.augment:
            waveform = self._augment_waveform(waveform)

        mel = self.mel_extractor(waveform)  # [n_mels, T]

        # Verifică lungime minimă
        if mel.shape[1] < self.min_frames:
            return None

        # Cache (fără augmentare)
        if self.mel_cache is not None and not self.augment:
            self.mel_cache[path] = mel

        return mel

    def _augment_waveform(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Augmentări avansate pentru robustitate la limba română.
        
        Aplică aleatoriu una sau mai multe transformări:
        - Perturbație de volum (±6dB)
        - Pitch shifting (±2 semitonuri)  
        - Noise injection (zgomot gaussian, SNR 20-40dB)
        - Time stretching (variație de viteză ±10%)
        """
        # 1. Perturbație de volum ±6dB
        if random.random() < 0.5:
            gain = random.uniform(0.5, 2.0)
            waveform = waveform * gain
            waveform = waveform.clamp(-1.0, 1.0)

        # 2. Pitch shifting (±2 semitonuri)
        if random.random() < 0.3:
            try:
                n_steps = random.uniform(-2.0, 2.0)
                waveform = torchaudio.functional.pitch_shift(
                    waveform, SAMPLE_RATE, n_steps
                )
            except Exception:
                pass  # Fallback: skip dacă pitch_shift nu e disponibil

        # 3. Noise injection (zgomot gaussian ușor, SNR 20-40dB)
        if random.random() < 0.4:
            snr_db = random.uniform(20.0, 40.0)
            signal_power = waveform.norm(p=2)
            noise = torch.randn_like(waveform)
            noise_power = noise.norm(p=2)
            if noise_power > 0:
                snr_linear = 10 ** (snr_db / 20.0)
                scale = signal_power / (noise_power * snr_linear)
                waveform = waveform + noise * scale
                waveform = waveform.clamp(-1.0, 1.0)

        # 4. Time stretching (variație de viteză ±10%)
        if random.random() < 0.3:
            try:
                stretch_factor = random.uniform(0.9, 1.1)
                effects = [["tempo", str(stretch_factor)]]
                waveform_np = waveform.numpy()
                augmented, _ = torchaudio.sox_effects.apply_effects_tensor(
                    waveform, SAMPLE_RATE, effects
                )
                waveform = augmented
            except Exception:
                pass  # Fallback: skip dacă sox nu e disponibil

        return waveform

    def _pad_or_crop(self, mel: torch.Tensor) -> torch.Tensor:
        """Standardizează lungimea mel spectrogram."""
        T = mel.shape[1]
        if T >= self.max_frames:
            # Crop aleator
            start = random.randint(0, T - self.max_frames)
            return mel[:, start:start + self.max_frames]
        else:
            # Pad cu zero
            pad = torch.zeros(mel.shape[0], self.max_frames - T)
            return torch.cat([mel, pad], dim=1)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Optional[Dict]:
        src_path, src_speaker_idx = self.samples[idx]

        # Alege un vorbitor target DIFERIT
        other_speakers = [k for k in self.speaker_samples.keys() if k != src_speaker_idx]
        if not other_speakers:
            # Fallback: same speaker (auto-conversie)
            tgt_speaker_idx = src_speaker_idx
        else:
            tgt_speaker_idx = random.choice(other_speakers)

        # Alege un utterance random din target speaker
        tgt_path = random.choice(self.speaker_samples[tgt_speaker_idx])

        # Extrage mel
        mel_src = self._get_mel(src_path)
        mel_tgt = self._get_mel(tgt_path)

        if mel_src is None or mel_tgt is None:
            # Returnează un item valid de rezervă
            return self.__getitem__((idx + 1) % len(self.samples))

        # Standardizare lungime
        mel_src = self._pad_or_crop(mel_src)
        mel_tgt = self._pad_or_crop(mel_tgt)

        return {
            "mel_src": mel_src,                          # [80, T]
            "mel_tgt": mel_tgt,                          # [80, T]
            "speaker_id": torch.tensor(tgt_speaker_idx, dtype=torch.long),
            "src_speaker_id": torch.tensor(src_speaker_idx, dtype=torch.long),
        }


def create_dataloaders(
    pseudo_speakers_path: str,
    batch_size: int = 16,
    val_ratio: float = 0.1,
    num_workers: int = 0,
    max_frames: int = MAX_FRAMES
) -> Tuple[DataLoader, DataLoader]:
    """
    Creează DataLoader-uri pentru antrenare și validare.

    Returns:
        (train_loader, val_loader)
    """
    dataset = VoiceConversionDataset(
        pseudo_speakers_path=pseudo_speakers_path,
        max_frames=max_frames,
        augment=True
    )

    # Split train/val
    n_val = max(1, int(len(dataset) * val_ratio))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )

    # Val dataset fără augmentare
    val_ds.dataset.augment = False

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
        pin_memory=torch.cuda.is_available()
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False
    )

    logger.info(f"DataLoader: {len(train_ds)} train, {len(val_ds)} val, batch={batch_size}")
    return train_loader, val_loader
