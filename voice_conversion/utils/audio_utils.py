"""
Utilități Audio — Procesare Fișiere Audio
==========================================

Funcții pentru:
- Încărcare și salvare audio (wav, mp3, flac)
- Resampling la sample rate standard
- Normalizare, trimming, padding
- Extracție mel spectrograme
- Vizualizare waveform și spectrograme
"""

import torch
import torchaudio
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Union, List
import logging

logger = logging.getLogger(__name__)


def load_audio(
    filepath: Union[str, Path],
    target_sr: int = 16000,
    mono: bool = True,
    normalize: bool = True,
    max_duration: Optional[float] = None
) -> Tuple[torch.Tensor, int]:
    """
    Încarcă un fișier audio și îl procesează.

    Args:
        filepath: Calea către fișierul audio
        target_sr: Sample rate țintă (default: 16000 Hz)
        mono: Conversie la mono (default: True)
        normalize: Normalizare la [-1, 1] (default: True)
        max_duration: Durata maximă în secunde (None = fără limită)

    Returns:
        Tuple[torch.Tensor, int]: (waveform [1, T], sample_rate)

    Raises:
        FileNotFoundError: Fișierul nu există
        ValueError: Fișierul audio e prea scurt
    """
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"Fișier audio inexistent: {filepath}")

    # Încărcare audio
    waveform, sr = torchaudio.load(str(filepath))

    # Conversie mono
    if mono and waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    # Resampling
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(
            orig_freq=sr,
            new_freq=target_sr
        )
        waveform = resampler(waveform)
        sr = target_sr

    # Trimming la max_duration
    if max_duration is not None:
        max_samples = int(max_duration * sr)
        if waveform.shape[1] > max_samples:
            waveform = waveform[:, :max_samples]

    # Normalizare
    if normalize:
        max_val = waveform.abs().max()
        if max_val > 0:
            waveform = waveform / max_val

    return waveform, sr


def save_audio(
    waveform: torch.Tensor,
    filepath: Union[str, Path],
    sample_rate: int = 16000,
    normalize: bool = True
) -> Path:
    """
    Salvează un tensor audio ca fișier WAV.

    Args:
        waveform: Tensor audio [1, T] sau [T]
        filepath: Calea de salvare
        sample_rate: Sample rate
        normalize: Normalizare înainte de salvare

    Returns:
        Path: Calea fișierului salvat
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    # Asigurare dimensiuni corecte
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)

    if normalize:
        max_val = waveform.abs().max()
        if max_val > 0:
            waveform = waveform / max_val * 0.95  # Headroom

    torchaudio.save(str(filepath), waveform.cpu(), sample_rate)
    logger.info(f"💾 Audio salvat: {filepath} ({waveform.shape[1] / sample_rate:.2f}s)")

    return filepath


def get_audio_info(filepath: Union[str, Path]) -> dict:
    """
    Returnează informații despre un fișier audio.

    Returns:
        dict cu: duration, sample_rate, channels, num_samples
    """
    filepath = Path(filepath)
    info = torchaudio.info(str(filepath))

    return {
        "filepath": str(filepath),
        "filename": filepath.name,
        "duration": info.num_frames / info.sample_rate,
        "sample_rate": info.sample_rate,
        "channels": info.num_channels,
        "num_samples": info.num_frames,
        "format": filepath.suffix.lstrip(".")
    }


def compute_mel_spectrogram(
    waveform: torch.Tensor,
    sample_rate: int = 16000,
    n_fft: int = 1024,
    hop_length: int = 256,
    n_mels: int = 80,
    fmin: float = 0.0,
    fmax: float = 8000.0
) -> torch.Tensor:
    """
    Calculează mel spectrograma unui audio.

    Args:
        waveform: Tensor audio [1, T]
        sample_rate: Sample rate
        n_fft, hop_length, n_mels, fmin, fmax: Parametri mel

    Returns:
        torch.Tensor: Mel spectrogram [1, n_mels, T']
    """
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        f_min=fmin,
        f_max=fmax
    )

    mel = mel_transform(waveform)

    # Log mel spectrogram
    mel = torch.log(torch.clamp(mel, min=1e-5))

    return mel


def trim_silence(
    waveform: torch.Tensor,
    sample_rate: int = 16000,
    threshold_db: float = -40.0,
    min_silence_duration: float = 0.1
) -> torch.Tensor:
    """
    Elimină tăcerea de la începutul și sfârșitul audio-ului.

    Args:
        waveform: Tensor audio [1, T]
        sample_rate: Sample rate
        threshold_db: Pragul de tăcere în dB
        min_silence_duration: Durata minimă a tăcerii (secunde)

    Returns:
        torch.Tensor: Audio fără tăcere [1, T']
    """
    # Detectare activitate vocală simplă bazată pe energie
    frame_length = int(0.025 * sample_rate)  # 25ms frames
    hop = int(0.010 * sample_rate)  # 10ms hop

    signal = waveform.squeeze()
    energy = []

    for i in range(0, len(signal) - frame_length, hop):
        frame = signal[i:i + frame_length]
        frame_energy = 20 * torch.log10(frame.abs().mean() + 1e-10)
        energy.append(frame_energy.item())

    energy = np.array(energy)
    threshold = threshold_db

    # Găsește primul și ultimul frame activ
    active_frames = np.where(energy > threshold)[0]

    if len(active_frames) == 0:
        return waveform

    start_frame = max(0, active_frames[0] - 5)
    end_frame = min(len(energy), active_frames[-1] + 5)

    start_sample = start_frame * hop
    end_sample = min(end_frame * hop + frame_length, len(signal))

    return signal[start_sample:end_sample].unsqueeze(0)


def list_audio_files(
    directory: Union[str, Path],
    extensions: List[str] = None
) -> List[Path]:
    """
    Listează toate fișierele audio dintr-un director.

    Args:
        directory: Calea directorului
        extensions: Extensii acceptate (default: wav, mp3, flac, ogg)

    Returns:
        Lista de Path-uri
    """
    if extensions is None:
        extensions = [".wav", ".mp3", ".flac", ".ogg", ".m4a"]

    directory = Path(directory)
    audio_files = []

    for ext in extensions:
        audio_files.extend(directory.glob(f"**/*{ext}"))

    return sorted(audio_files)
