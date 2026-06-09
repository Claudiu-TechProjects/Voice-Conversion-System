"""
Curățare Dataset Common Voice RO pentru XTTS v2 Fine-Tuning
============================================================

Filtrează clipurile validate din Common Voice și produce un subset
curat în format LJSpeech (wav + metadata.csv) potrivit pentru
fine-tuning-ul XTTS v2.

Criteriile de filtrare:
- Doar clipuri validate (up_votes >= 2, down_votes == 0)
- Durata audio între 2-15 secunde
- SNR minim 15dB (eliminare zgomot)
- Fără clipping (< 1% sample-uri saturate)
- Conversie MP3 → WAV mono 22050Hz
"""

import os
import logging
import csv
from pathlib import Path
from typing import Optional
import numpy as np

logger = logging.getLogger(__name__)


def _check_clipping(audio: np.ndarray, threshold: float = 0.99, max_ratio: float = 0.01) -> bool:
    """Verifică dacă audio-ul are clipping (sample-uri saturate)."""
    clipped = np.sum(np.abs(audio) >= threshold)
    ratio = clipped / len(audio) if len(audio) > 0 else 0
    return ratio < max_ratio


def _estimate_snr(audio: np.ndarray, sr: int, frame_length: int = 2048) -> float:
    """Estimează SNR-ul audio-ului folosind raportul energie semnal/zgomot."""
    import librosa
    
    # Calculăm energia per frame
    frames = librosa.util.frame(audio, frame_length=frame_length, hop_length=frame_length // 2)
    energy = np.sum(frames ** 2, axis=0)
    
    if len(energy) == 0:
        return 0.0
    
    # Top 10% frames = semnal, bottom 10% = zgomot
    sorted_energy = np.sort(energy)
    n = len(sorted_energy)
    noise_energy = np.mean(sorted_energy[:max(1, n // 10)])
    signal_energy = np.mean(sorted_energy[-max(1, n // 10):])
    
    if noise_energy <= 0:
        return 60.0  # Foarte curat
    
    snr = 10 * np.log10(signal_energy / noise_energy)
    return float(snr)


def clean_cv_dataset(
    cv_root: Optional[str] = None,
    output_dir: Optional[str] = None,
    max_clips: int = 1000,
    min_duration: float = 2.0,
    max_duration: float = 15.0,
    min_snr: float = 15.0,
    target_sr: int = 22050,
    progress_callback=None
):
    """
    Curăță și pregătește un subset din Common Voice RO pentru XTTS fine-tuning.
    
    Args:
        cv_root: Calea către folderul Common Voice RO (conține clips/ și validated.tsv)
        output_dir: Directorul de ieșire (va conține wavs/ și metadata.csv)
        max_clips: Numărul maxim de clipuri de selectat
        min_duration: Durata minimă acceptată (secunde)
        max_duration: Durata maximă acceptată (secunde)
        min_snr: SNR minim acceptat (dB)
        target_sr: Sample rate de ieșire
        progress_callback: Funcție opțională de raportare progres
    
    Returns:
        dict cu statistici despre procesare
    """
    import librosa
    import soundfile as sf
    import pandas as pd
    
    # Setăm căile implicite
    project_root = Path(__file__).parent.parent.parent
    
    if cv_root is None:
        # Căutăm automat folderul Common Voice
        cv_candidates = list((project_root / "dataset").glob("cv-corpus-*/ro"))
        if not cv_candidates:
            raise FileNotFoundError(
                "Nu am găsit folderul Common Voice RO. "
                "Asigură-te că arhiva este extrasă în dataset/cv-corpus-*/ro/"
            )
        cv_root = cv_candidates[0]
    else:
        cv_root = Path(cv_root)
    
    if output_dir is None:
        output_dir = project_root / "dataset" / "xtts_clean_ro"
    else:
        output_dir = Path(output_dir)
    
    clips_dir = cv_root / "clips"
    validated_tsv = cv_root / "validated.tsv"
    
    if not validated_tsv.exists():
        raise FileNotFoundError(f"Nu găsesc {validated_tsv}")
    if not clips_dir.exists():
        raise FileNotFoundError(f"Nu găsesc folderul cu clipuri: {clips_dir}")
    
    # Creare directoare output
    wavs_dir = output_dir / "wavs"
    wavs_dir.mkdir(parents=True, exist_ok=True)
    
    # Citire CSV validat
    logger.info(f"Citire {validated_tsv}...")
    df = pd.read_csv(validated_tsv, sep='\t')
    total_validated = len(df)
    logger.info(f"Total clipuri validate: {total_validated}")
    
    # Filtru 1: Doar clipuri cu voturi pozitive, fără negative
    df = df[df['up_votes'] >= 2]
    if 'down_votes' in df.columns:
        df = df[df['down_votes'] == 0]
    logger.info(f"După filtrare voturi: {len(df)} clipuri")
    
    # Amestecăm aleator și limităm la un maxim rezonabil de candidați
    df = df.sample(frac=1, random_state=42).head(max_clips * 3)
    
    # Procesare clip cu clip
    stats = {
        "total_validated": total_validated,
        "candidates": len(df),
        "accepted": 0,
        "rejected_missing": 0,
        "rejected_duration": 0,
        "rejected_snr": 0,
        "rejected_clipping": 0,
        "rejected_error": 0,
        "status": "running",
        "progress_pct": 0
    }
    
    metadata_rows = []
    
    for idx, (_, row) in enumerate(df.iterrows()):
        if stats["accepted"] >= max_clips:
            break
        
        clip_path = clips_dir / row['path']
        if not clip_path.exists():
            stats["rejected_missing"] += 1
            continue
        
        try:
            # Încărcare audio
            audio, sr = librosa.load(str(clip_path), sr=target_sr, mono=True)
            duration = len(audio) / sr
            
            # Filtru 2: Durata
            if duration < min_duration or duration > max_duration:
                stats["rejected_duration"] += 1
                continue
            
            # Filtru 3: SNR
            snr = _estimate_snr(audio, sr)
            if snr < min_snr:
                stats["rejected_snr"] += 1
                continue
            
            # Filtru 4: Clipping
            if not _check_clipping(audio):
                stats["rejected_clipping"] += 1
                continue
            
            # Normalizare
            peak = np.max(np.abs(audio))
            if peak > 0:
                audio = audio / peak * 0.95
            
            # Salvare WAV
            out_name = f"cv_ro_{stats['accepted']:05d}.wav"
            out_path = wavs_dir / out_name
            sf.write(str(out_path), audio, target_sr)
            
            # Adăugare metadata (format LJSpeech: filename|transcription)
            sentence = str(row.get('sentence', '')).strip()
            if sentence:
                metadata_rows.append((out_name.replace('.wav', ''), sentence))
                stats["accepted"] += 1
            
        except Exception as e:
            stats["rejected_error"] += 1
            logger.debug(f"Eroare la {clip_path.name}: {e}")
            continue
        
        # Progress
        progress = int((idx + 1) / len(df) * 100)
        stats["progress_pct"] = min(progress, 100)
        
        if progress_callback and idx % 20 == 0:
            progress_callback(stats)
        
        if idx % 50 == 0:
            logger.info(
                f"  Progres: {idx+1}/{len(df)} procesat, "
                f"{stats['accepted']} acceptate / {max_clips} țintă"
            )
    
    # Scriere metadata.csv
    metadata_path = output_dir / "metadata.csv"
    with open(metadata_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f, delimiter='|')
        for row in metadata_rows:
            writer.writerow(row)
    
    stats["status"] = "done"
    stats["progress_pct"] = 100
    stats["output_dir"] = str(output_dir)
    stats["metadata_file"] = str(metadata_path)
    
    logger.info(f"\n{'='*60}")
    logger.info(f"  CURĂȚARE COMMON VOICE RO — FINALIZAT")
    logger.info(f"{'='*60}")
    logger.info(f"  Clipuri acceptate: {stats['accepted']}/{max_clips}")
    logger.info(f"  Respinse (lipsă):  {stats['rejected_missing']}")
    logger.info(f"  Respinse (durată): {stats['rejected_duration']}")
    logger.info(f"  Respinse (SNR):    {stats['rejected_snr']}")
    logger.info(f"  Respinse (clip):   {stats['rejected_clipping']}")
    logger.info(f"  Respinse (eroare): {stats['rejected_error']}")
    logger.info(f"  Output:  {output_dir}")
    logger.info(f"{'='*60}\n")
    
    return stats


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    result = clean_cv_dataset(max_clips=100)
    print(f"Rezultat: {result}")
