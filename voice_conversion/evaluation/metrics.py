"""
Metrici de Evaluare — Voice Conversion
========================================

Metrici obiective pentru evaluarea calității conversiei vocii:

1. MCD  — Mel Cepstral Distortion (distorsiune spectrală)
2. PESQ — Perceptual Evaluation of Speech Quality
3. F0   — Metrici de pitch (RMSE, Pearson Correlation)
4. SNR  — Signal-to-Noise Ratio
5. Speaker Similarity — Cosine similarity pe embeddings ECAPA-TDNN

Referințe:
- MCD: Kubichek (1993) "Mel-Cepstral Distance Measure for Objective
  Speech Quality Assessment"
- PESQ: ITU-T P.862 (2001)
- Speaker Similarity: Desplanques et al. (2020) "ECAPA-TDNN"
"""

import numpy as np
import torch
import torchaudio
from pathlib import Path
from typing import Union, Optional, Dict, Tuple
import logging

logger = logging.getLogger(__name__)


# =====================================================================
# 1. MCD — Mel Cepstral Distortion
# =====================================================================

def compute_mcd(
    reference_audio: Union[str, Path, torch.Tensor],
    converted_audio: Union[str, Path, torch.Tensor],
    sample_rate: int = 16000,
    n_mfcc: int = 13,
    n_fft: int = 1024,
    hop_length: int = 256,
    n_mels: int = 80
) -> float:
    """
    Calculează Mel Cepstral Distortion (MCD) între audio de referință
    și audio convertit.

    MCD măsoară distanța spectrală în spațiul cepstral, fiind una din
    cele mai utilizate metrici pentru evaluarea voice conversion.

    Formula:
        MCD = (10√2 / ln10) × √(Σ(mc_ref_i - mc_conv_i)²)
        unde mc = coeficienți mel-cepstrali (excluzând c0)

    Args:
        reference_audio: Audio de referință (target speaker, same content)
        converted_audio: Audio convertit
        sample_rate: Sample rate
        n_mfcc: Număr de coeficienți MFCC (default: 13, se exclude c0)
        n_fft: Dimensiunea FFT
        hop_length: Hop length
        n_mels: Număr de mel filterbanks

    Returns:
        float: MCD în dB (mai mic = mai bine, < 6.0 dB este bun)
    """
    # Încărcare audio dacă e path
    if isinstance(reference_audio, (str, Path)):
        ref_wav, ref_sr = torchaudio.load(str(reference_audio))
        if ref_sr != sample_rate:
            ref_wav = torchaudio.transforms.Resample(ref_sr, sample_rate)(ref_wav)
    else:
        ref_wav = reference_audio.clone()

    if isinstance(converted_audio, (str, Path)):
        conv_wav, conv_sr = torchaudio.load(str(converted_audio))
        if conv_sr != sample_rate:
            conv_wav = torchaudio.transforms.Resample(conv_sr, sample_rate)(conv_wav)
    else:
        conv_wav = converted_audio.clone()

    # Asigurare mono
    if ref_wav.dim() > 1 and ref_wav.shape[0] > 1:
        ref_wav = ref_wav.mean(0, keepdim=True)
    if conv_wav.dim() > 1 and conv_wav.shape[0] > 1:
        conv_wav = conv_wav.mean(0, keepdim=True)

    # Aliniere lungime (DTW simplificat - trunchiere la lungimea minimă)
    min_len = min(ref_wav.shape[-1], conv_wav.shape[-1])
    ref_wav = ref_wav[..., :min_len]
    conv_wav = conv_wav[..., :min_len]

    # Extracție MFCC
    mfcc_transform = torchaudio.transforms.MFCC(
        sample_rate=sample_rate,
        n_mfcc=n_mfcc + 1,  # +1 pentru a exclude c0
        melkwargs={
            "n_fft": n_fft,
            "hop_length": hop_length,
            "n_mels": n_mels
        }
    )

    ref_mfcc = mfcc_transform(ref_wav)[:, 1:, :]   # Exclude c0
    conv_mfcc = mfcc_transform(conv_wav)[:, 1:, :]

    # Aliniere temporală (trunchiere la minimul de frames)
    min_frames = min(ref_mfcc.shape[-1], conv_mfcc.shape[-1])
    ref_mfcc = ref_mfcc[..., :min_frames]
    conv_mfcc = conv_mfcc[..., :min_frames]

    # Calcul MCD
    diff = ref_mfcc - conv_mfcc
    frame_mcd = torch.sqrt((diff ** 2).sum(dim=1))  # [1, T]

    # Constanta MCD: 10 * sqrt(2) / ln(10) ≈ 4.3423
    K = 10.0 * np.sqrt(2.0) / np.log(10.0)
    mcd = K * frame_mcd.mean().item()

    return mcd


# =====================================================================
# 2. PESQ — Perceptual Evaluation of Speech Quality
# =====================================================================

def compute_pesq(
    reference_audio: Union[str, Path, torch.Tensor],
    converted_audio: Union[str, Path, torch.Tensor],
    sample_rate: int = 16000,
    mode: str = "wb"
) -> float:
    """
    Calculează PESQ (Perceptual Evaluation of Speech Quality).

    PESQ prezice scorul MOS (Mean Opinion Score) al unui semnal audio
    degradat în comparație cu unul de referință.

    Args:
        reference_audio: Audio de referință
        converted_audio: Audio convertit (degradat)
        sample_rate: Sample rate (16000 pentru wb, 8000 pentru nb)
        mode: 'wb' (wideband) sau 'nb' (narrowband)

    Returns:
        float: Scor PESQ (-0.5 → 4.5, mai mare = mai bine)
    """
    try:
        from pesq import pesq as pesq_fn
    except ImportError:
        logger.warning("⚠️ Biblioteca 'pesq' nu este instalată. "
                       "Instalare: pip install pesq")
        return float("nan")

    # Încărcare audio
    if isinstance(reference_audio, (str, Path)):
        ref_wav, ref_sr = torchaudio.load(str(reference_audio))
        if ref_sr != sample_rate:
            ref_wav = torchaudio.transforms.Resample(ref_sr, sample_rate)(ref_wav)
    else:
        ref_wav = reference_audio.clone()

    if isinstance(converted_audio, (str, Path)):
        conv_wav, conv_sr = torchaudio.load(str(converted_audio))
        if conv_sr != sample_rate:
            conv_wav = torchaudio.transforms.Resample(conv_sr, sample_rate)(conv_wav)
    else:
        conv_wav = converted_audio.clone()

    # Conversie numpy
    ref_np = ref_wav.squeeze().numpy()
    conv_np = conv_wav.squeeze().numpy()

    # Aliniere lungime
    min_len = min(len(ref_np), len(conv_np))
    ref_np = ref_np[:min_len]
    conv_np = conv_np[:min_len]

    try:
        score = pesq_fn(sample_rate, ref_np, conv_np, mode)
        return float(score)
    except Exception as e:
        logger.warning(f"⚠️ Eroare la PESQ: {e}")
        return float("nan")


# =====================================================================
# 3. F0 — Metrici de Pitch
# =====================================================================

def extract_f0(
    audio: Union[str, Path, torch.Tensor],
    sample_rate: int = 16000,
    fmin: float = 50.0,
    fmax: float = 500.0
) -> np.ndarray:
    """
    Extrage curba F0 (frecvența fundamentală / pitch) din audio.

    Folosește algoritmul PYIN pentru extracție robustă.

    Args:
        audio: Audio (path sau tensor)
        sample_rate: Sample rate
        fmin: F0 minim (Hz)
        fmax: F0 maxim (Hz)

    Returns:
        np.ndarray: F0 values (Hz), NaN pentru segmente unvoiced
    """
    try:
        import librosa
    except ImportError:
        logger.warning("⚠️ Biblioteca 'librosa' nu este instalată.")
        return np.array([])

    # Încărcare
    if isinstance(audio, (str, Path)):
        y, sr = librosa.load(str(audio), sr=sample_rate)
    else:
        y = audio.squeeze().numpy()
        sr = sample_rate

    # Extracție F0 cu PYIN
    f0, voiced_flag, voiced_probs = librosa.pyin(
        y,
        fmin=fmin,
        fmax=fmax,
        sr=sr
    )

    return f0


def compute_f0_metrics(
    reference_audio: Union[str, Path, torch.Tensor],
    converted_audio: Union[str, Path, torch.Tensor],
    sample_rate: int = 16000
) -> Dict[str, float]:
    """
    Calculează metrici F0 (pitch) între referință și audio convertit.

    Metrici calculate:
    - F0 RMSE: Root Mean Square Error al frecvenței fundamentale
    - F0 PCC: Pearson Correlation Coefficient al F0
    - F0 Mean Diff: Diferența medie de pitch

    Args:
        reference_audio: Audio de referință
        converted_audio: Audio convertit

    Returns:
        Dict cu metrici: f0_rmse, f0_pcc, f0_mean_diff
    """
    ref_f0 = extract_f0(reference_audio, sample_rate)
    conv_f0 = extract_f0(converted_audio, sample_rate)

    if len(ref_f0) == 0 or len(conv_f0) == 0:
        return {
            "f0_rmse": float("nan"),
            "f0_pcc": float("nan"),
            "f0_mean_diff": float("nan")
        }

    # Aliniere lungime
    min_len = min(len(ref_f0), len(conv_f0))
    ref_f0 = ref_f0[:min_len]
    conv_f0 = conv_f0[:min_len]

    # Filtrare segmente voiced (ambele trebuie să fie voiced)
    voiced_mask = ~np.isnan(ref_f0) & ~np.isnan(conv_f0)

    if voiced_mask.sum() < 10:
        return {
            "f0_rmse": float("nan"),
            "f0_pcc": float("nan"),
            "f0_mean_diff": float("nan")
        }

    ref_voiced = ref_f0[voiced_mask]
    conv_voiced = conv_f0[voiced_mask]

    # RMSE
    f0_rmse = np.sqrt(np.mean((ref_voiced - conv_voiced) ** 2))

    # Pearson Correlation
    if np.std(ref_voiced) > 0 and np.std(conv_voiced) > 0:
        f0_pcc = np.corrcoef(ref_voiced, conv_voiced)[0, 1]
    else:
        f0_pcc = 0.0

    # Mean difference
    f0_mean_diff = np.mean(np.abs(ref_voiced - conv_voiced))

    return {
        "f0_rmse": float(f0_rmse),
        "f0_pcc": float(f0_pcc),
        "f0_mean_diff": float(f0_mean_diff)
    }


# =====================================================================
# 4. SNR — Signal-to-Noise Ratio
# =====================================================================

def compute_snr(
    audio: Union[str, Path, torch.Tensor],
    sample_rate: int = 16000
) -> float:
    """
    Estimează Signal-to-Noise Ratio (SNR) al audio-ului.

    Estimarea se bazează pe raportul energie vocală / energie fundal,
    folosind segmentare bazată pe energie.

    Args:
        audio: Audio (path sau tensor)
        sample_rate: Sample rate

    Returns:
        float: SNR în dB (mai mare = mai bine, > 15 dB este bun)
    """
    if isinstance(audio, (str, Path)):
        wav, sr = torchaudio.load(str(audio))
        if sr != sample_rate:
            wav = torchaudio.transforms.Resample(sr, sample_rate)(wav)
    else:
        wav = audio.clone()

    signal = wav.squeeze().numpy()

    # Segmentare în frames
    frame_length = int(0.025 * sample_rate)
    hop = int(0.010 * sample_rate)

    energies = []
    for i in range(0, len(signal) - frame_length, hop):
        frame = signal[i:i + frame_length]
        energy = np.mean(frame ** 2)
        energies.append(energy)

    energies = np.array(energies)

    if len(energies) == 0:
        return float("nan")

    # Separare speech / noise bazat pe threshold de energie
    threshold = np.percentile(energies, 30)  # Bottom 30% = noise

    noise_energy = np.mean(energies[energies <= threshold])
    signal_energy = np.mean(energies[energies > threshold])

    if noise_energy <= 0 or signal_energy <= 0:
        return float("nan")

    snr = 10 * np.log10(signal_energy / noise_energy)
    return float(snr)


# =====================================================================
# 5. Speaker Similarity — Cosine Similarity (ECAPA-TDNN)
# =====================================================================

def compute_speaker_similarity(
    reference_audio: Union[str, Path],
    converted_audio: Union[str, Path],
    speaker_model=None
) -> float:
    """
    Calculează similaritatea de vorbitor între audio de referință
    (target speaker) și audio-ul convertit.

    Folosește ECAPA-TDNN pentru extracție de embeddings și
    cosine similarity pentru comparare.

    Args:
        reference_audio: Audio target speaker
        converted_audio: Audio convertit
        speaker_model: Model speaker recognition (opțional, se încarcă automat)

    Returns:
        float: Cosine similarity (-1 → 1, mai mare = mai bine, > 0.7 bun)
    """
    if speaker_model is None:
        try:
            # Încearcă mai multe importuri SpeechBrain
            try:
                from speechbrain.pretrained import SpeakerRecognition
            except ImportError:
                from speechbrain.inference.speaker import SpeakerRecognition

            speaker_model = SpeakerRecognition.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                savedir="./models_cache/spkrec-ecapa-voxceleb"
            )
        except Exception as e:
            logger.warning(f"⚠️ Nu s-a putut încărca ECAPA-TDNN: {e}")
            return float("nan")

    try:
        # Reparație specifică Windows: transformăm "\" în "/" (posix)
        # deoarece SpeechBrain/torchaudio pot trata "\" ca escape characters
        ref_path = Path(reference_audio).resolve().as_posix()
        conv_path = Path(converted_audio).resolve().as_posix()
        
        score, prediction = speaker_model.verify_files(
            ref_path,
            conv_path
        )
        return float(score.item())
    except Exception as e:
        logger.warning(f"⚠️ Eroare speaker similarity: {e}")
        return float("nan")


# =====================================================================
# WRAPPER COMPLET — Toate metricile
# =====================================================================

def compute_all_metrics(
    source_audio: Union[str, Path],
    converted_audio: Union[str, Path],
    target_reference: Union[str, Path],
    sample_rate: int = 16000,
    speaker_model=None
) -> Dict[str, float]:
    """
    Calculează toate metricile de evaluare pentru voice conversion.

    Args:
        source_audio: Audio original (sursă)
        converted_audio: Audio convertit
        target_reference: Audio referință target speaker
        sample_rate: Sample rate
        speaker_model: Model ECAPA-TDNN (opțional)

    Returns:
        Dict cu toate metricile:
        - mcd: Mel Cepstral Distortion (dB)
        - pesq: PESQ score
        - speaker_similarity: Cosine similarity
        - f0_rmse: F0 RMSE (Hz)
        - f0_pcc: F0 Pearson Correlation
        - snr: SNR (dB)
    """
    logger.info("📊 Calcul metrici complete...")

    metrics = {}

    # MCD (convertit vs target - cât de similar spectral cu target-ul)
    logger.info("   [1/5] MCD...")
    metrics["mcd"] = compute_mcd(target_reference, converted_audio, sample_rate)

    # PESQ (convertit vs sursă - calitatea generală a semnalului)
    logger.info("   [2/5] PESQ...")
    metrics["pesq"] = compute_pesq(source_audio, converted_audio, sample_rate)

    # Speaker Similarity (convertit vs target - identitate speaker)
    logger.info("   [3/5] Speaker Similarity...")
    metrics["speaker_similarity"] = compute_speaker_similarity(
        target_reference, converted_audio, speaker_model
    )

    # F0 Metrics (convertit vs target - pitch)
    logger.info("   [4/5] F0 Metrics...")
    f0_metrics = compute_f0_metrics(target_reference, converted_audio, sample_rate)
    metrics.update(f0_metrics)

    # SNR (calitatea audio-ului convertit)
    logger.info("   [5/5] SNR...")
    metrics["snr"] = compute_snr(converted_audio, sample_rate)

    # Formatare
    logger.info("\n📋 Rezultate evaluare:")
    for name, value in metrics.items():
        if isinstance(value, float) and not np.isnan(value):
            logger.info(f"   {name:25s}: {value:.4f}")
        else:
            logger.info(f"   {name:25s}: N/A")

    return metrics
