"""
YourTTS Dataset Preparation — Optimizat pentru Limba Română
=============================================================

Pregătește dataset-ul Common Voice RO pentru fine-tuning YourTTS:
1. Citește train.tsv ȘI validated.tsv din cv-corpus-25.0
2. Convertește MP3 → WAV 16kHz mono
3. VAD Trimming (elimină liniștea de la capete) — CRITIC pentru alignment
4. Normalizare RMS audio
5. Curățare text (caractere invalide, numere, etc.)
6. Selectare top N vorbitori cu cele mai multe clipuri
7. Generare metadata compatibil Coqui TTS
"""

import os
import re
import csv
import json
import logging
import unicodedata
from pathlib import Path
from typing import Dict, List, Tuple
from collections import Counter

import librosa
import soundfile as sf
import numpy as np

logger = logging.getLogger(__name__)


# =====================================================================
# TEXT CLEANING — Esențial pentru antrenare corectă
# =====================================================================

# Dicționar numere → text (română)
_NUM_TO_TEXT_RO = {
    '0': 'zero', '1': 'unu', '2': 'doi', '3': 'trei', '4': 'patru',
    '5': 'cinci', '6': 'șase', '7': 'șapte', '8': 'opt', '9': 'nouă',
    '10': 'zece', '11': 'unsprezece', '12': 'doisprezece',
    '13': 'treisprezece', '14': 'paisprezece', '15': 'cincisprezece',
    '16': 'șaisprezece', '17': 'șaptesprezece', '18': 'optsprezece',
    '19': 'nouăsprezece', '20': 'douăzeci', '30': 'treizeci',
    '40': 'patruzeci', '50': 'cincizeci', '100': 'o sută',
    '1000': 'o mie',
}

# Caractere permise în text (litere românești + punctuație de bază)
_ALLOWED_CHARS = set(
    "abcdefghijklmnopqrstuvwxyz"
    "ăâîșț"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "ĂÂÎȘȚ"
    " .,!?;:-–—'\"()"
)


def clean_text(text: str) -> str:
    """
    Curăță textul pentru antrenare YourTTS pe limba română.
    
    - Normalizează unicode (cedillă → virgulă sub literă)
    - Înlocuiește numerele simple cu text
    - Elimină caractere non-românești
    - Normalizează spații și punctuație
    """
    if not text:
        return ""
    
    # 1. Normalizare Unicode: ș cu cedillă → ș cu virgulă
    text = text.replace('Ş', 'Ș').replace('ş', 'ș')
    text = text.replace('Ţ', 'Ț').replace('ţ', 'ț')
    
    # 2. Înlocuire numere simple (1-20, zeci, 100, 1000)
    def replace_number(match):
        num = match.group(0)
        if num in _NUM_TO_TEXT_RO:
            return _NUM_TO_TEXT_RO[num]
        # Numere compuse simple (21-99)
        if len(num) == 2 and num.isdigit():
            tens = int(num[0]) * 10
            ones = int(num[1])
            tens_str = _NUM_TO_TEXT_RO.get(str(tens), '')
            ones_str = _NUM_TO_TEXT_RO.get(str(ones), '')
            if tens_str and ones_str and ones > 0:
                return f"{tens_str} și {ones_str}"
            elif tens_str:
                return tens_str
        return num  # Nu putem converti, lăsăm ca atare
    
    text = re.sub(r'\b\d{1,4}\b', replace_number, text)
    
    # 3. Eliminare numere rămase (care n-au putut fi convertite)
    if re.search(r'\d', text):
        return ""  # Skip complet propoziția cu numere complexe
    
    # 4. Înlocuire ghilimele fancy și apostrofuri
    text = text.replace('"', '"').replace('"', '"')
    text = text.replace('„', '"').replace('«', '"').replace('»', '"')
    text = text.replace(''', "'").replace(''', "'").replace('`', "'")
    
    # 5. Normalizare liniuțe
    text = text.replace('–', '-').replace('—', '-')
    
    # 6. Eliminare caractere non-permise
    cleaned = []
    for ch in text:
        if ch in _ALLOWED_CHARS:
            cleaned.append(ch)
        elif ch in '\t\n\r':
            cleaned.append(' ')
        # else: skip
    text = ''.join(cleaned)
    
    # 7. Normalizare spații
    text = re.sub(r'\s+', ' ', text).strip()
    
    # 8. Verificare minimă
    # Trebuie să aibă cel puțin 3 cuvinte și 10 caractere
    words = text.split()
    if len(words) < 3 or len(text) < 10:
        return ""
    
    # 9. Lowercase
    text = text.lower()
    
    return text


# =====================================================================
# DATASET PREPARER
# =====================================================================

class YourTTSDatasetPreparer:
    """
    Prepară dataset-ul Common Voice RO pentru fine-tuning YourTTS.
    
    Optimizări față de versiunea anterioară:
    - Folosește validated.tsv + train.tsv (de 4x mai multe date)
    - VAD Trimming cu librosa (elimină liniștea)
    - Normalizare RMS (volum consistent)
    - Curățare text avansată (numere, caractere speciale)
    - Speaker map corect
    
    Structura output:
        dataset/yourtts_prepared/
        ├── wavs/
        │   ├── spk00_0001.wav
        │   └── ...
        ├── metadata_train.csv   (audio_file|text|speaker_name)
        ├── metadata_val.csv
        ├── speaker_map.json
        └── stats.json
    """
    
    def __init__(self, config=None):
        if config is None:
            from voice_conversion.config import YOURTTS_CFG, PROJECT_ROOT
            self.cfg = YOURTTS_CFG
            self.project_root = PROJECT_ROOT
        else:
            self.cfg = config
            self.project_root = Path(__file__).parent.parent.parent
        
        self.corpus_dir = self.project_root / "dataset" / "cv-corpus-25.0-2026-03-09" / "ro"
        self.clips_dir = self.corpus_dir / "clips"
        self.output_dir = self.project_root / self.cfg.prepared_dataset_dir
        self.wavs_dir = self.output_dir / "wavs"
    
    def prepare(self, val_ratio: float = 0.1) -> Dict:
        """Pipeline complet de pregătire dataset."""
        logger.info("=" * 60)
        logger.info("YourTTS Dataset Preparation — Optimizat RO")
        logger.info("=" * 60)
        
        # 1. Citire validated.tsv + train.tsv (maximizare date)
        entries = self._read_all_corpus_entries()
        logger.info(f"Citite {len(entries)} intrări totale (validated + train)")
        
        # 2. Curățare text + filtrare
        entries = self._clean_and_filter_text(entries)
        logger.info(f"După curățare text: {len(entries)} intrări valide")
        
        # 3. Selectare top speakers
        entries = self._select_top_speakers(entries)
        
        # 4. Creare directoare
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.wavs_dir.mkdir(parents=True, exist_ok=True)
        
        # 5. Conversie MP3 → WAV + VAD trim + normalizare RMS
        processed = self._convert_trim_normalize(entries)
        logger.info(f"Procesate {len(processed)} clipuri valide")
        
        if not processed:
            raise ValueError("Nu s-a putut procesa niciun clip!")
        
        # 6. Split train/val (stratificat per speaker)
        train_entries, val_entries = self._split_train_val(processed, val_ratio)
        
        # 7. Generare metadata CSV
        self._write_metadata(train_entries, self.output_dir / "metadata_train.csv")
        self._write_metadata(val_entries, self.output_dir / "metadata_val.csv")
        
        # 8. Speaker map CORECT
        speaker_map = {}
        for e in processed:
            sid = e["speaker_id"]
            sname = e["speaker_name"]
            if sid not in speaker_map:
                speaker_map[sid] = sname
        
        with open(self.output_dir / "speaker_map.json", 'w', encoding='utf-8') as f:
            json.dump(speaker_map, f, indent=2, ensure_ascii=False)
        
        # 9. Statistici
        stats = {
            "total_clips": len(processed),
            "train_clips": len(train_entries),
            "val_clips": len(val_entries),
            "num_speakers": len(speaker_map),
            "speakers": {
                name: sum(1 for e in processed if e["speaker_name"] == name)
                for name in set(speaker_map.values())
            },
            "total_duration_hours": round(
                sum(e["duration"] for e in processed) / 3600, 2
            ),
            "avg_clip_duration": round(
                np.mean([e["duration"] for e in processed]), 2
            ),
            "output_dir": str(self.output_dir)
        }
        
        with open(self.output_dir / "stats.json", 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Dataset pregătit: {stats['train_clips']} train, "
                     f"{stats['val_clips']} val, {stats['num_speakers']} vorbitori, "
                     f"{stats['total_duration_hours']}h audio total")
        
        return stats
    
    # -----------------------------------------------------------------
    # CITIRE DATE
    # -----------------------------------------------------------------
    
    def _read_all_corpus_entries(self) -> List[Dict]:
        """Citește validated.tsv + train.tsv (deduplicate)."""
        seen_paths = set()
        entries = []
        
        # Prioritate: validated.tsv (calitate mai bună), apoi train.tsv
        for tsv_name in ["validated.tsv", "train.tsv"]:
            tsv_path = self.corpus_dir / tsv_name
            if not tsv_path.exists():
                logger.warning(f"{tsv_name} nu există, skip")
                continue
            
            count_before = len(entries)
            with open(tsv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f, delimiter='\t')
                for row in reader:
                    path_val = row.get("path", "")
                    if path_val in seen_paths:
                        continue
                    
                    mp3_path = self.clips_dir / path_val
                    sentence = row.get("sentence", "")
                    
                    if mp3_path.exists() and sentence.strip():
                        seen_paths.add(path_val)
                        entries.append({
                            "client_id": row.get("client_id", "unknown"),
                            "mp3_path": str(mp3_path),
                            "sentence": sentence.strip(),
                        })
            
            logger.info(f"  {tsv_name}: +{len(entries) - count_before} intrări noi")
        
        return entries
    
    # -----------------------------------------------------------------
    # CURĂȚARE TEXT
    # -----------------------------------------------------------------
    
    def _clean_and_filter_text(self, entries: List[Dict]) -> List[Dict]:
        """Curăță textul și elimină intrările invalide."""
        valid = []
        for e in entries:
            cleaned = clean_text(e["sentence"])
            if cleaned:
                e["sentence"] = cleaned
                valid.append(e)
        return valid
    
    # -----------------------------------------------------------------
    # SELECTARE VORBITORI
    # -----------------------------------------------------------------
    
    def _select_top_speakers(self, entries: List[Dict]) -> List[Dict]:
        """Selectează top N vorbitori după nr. de clipuri."""
        speaker_counts = Counter(e["client_id"] for e in entries)
        
        # Top speakers cu minim 20 clipuri (pentru a avea suficiente date per vorbitor)
        min_clips = 20
        eligible = {
            sid: count for sid, count in speaker_counts.items()
            if count >= min_clips
        }
        
        top_speakers = [
            sid for sid, _ in sorted(
                eligible.items(), key=lambda x: x[1], reverse=True
            )[:self.cfg.max_speakers]
        ]
        
        logger.info(f"Selectați top {len(top_speakers)} vorbitori din "
                     f"{len(speaker_counts)} total ({len(eligible)} eligibili cu ≥{min_clips} clipuri)")
        
        # Filtrare + asignare speaker_name + limitare clipuri per speaker
        filtered = []
        speaker_clip_count = Counter()
        
        for e in entries:
            if e["client_id"] in top_speakers:
                spk_idx = top_speakers.index(e["client_id"])
                
                if speaker_clip_count[spk_idx] >= self.cfg.max_clips_per_speaker:
                    continue
                
                e["speaker_name"] = f"spk{spk_idx:02d}"
                e["speaker_id"] = e["client_id"]
                filtered.append(e)
                speaker_clip_count[spk_idx] += 1
        
        for spk_idx in range(len(top_speakers)):
            name = f"spk{spk_idx:02d}"
            count = speaker_clip_count[spk_idx]
            logger.info(f"  {name}: {count} clipuri")
        
        return filtered
    
    # -----------------------------------------------------------------
    # CONVERSIE + VAD TRIM + NORMALIZARE
    # -----------------------------------------------------------------
    
    def _convert_trim_normalize(self, entries: List[Dict]) -> List[Dict]:
        """
        Convertește MP3 → WAV 16kHz mono cu:
        - VAD Trimming (librosa.effects.trim) — elimină liniștea
        - Normalizare RMS — volum consistent
        - Filtrare durată
        """
        processed = []
        skipped = 0
        total = len(entries)
        
        for i, entry in enumerate(entries):
            try:
                # 1. Încărcare MP3 → 16kHz mono
                audio, sr = librosa.load(
                    entry["mp3_path"],
                    sr=self.cfg.sample_rate,
                    mono=True
                )
                
                # 2. VAD Trimming — CRITIC: elimină liniștea de la capete
                # top_db=25 e agresiv dar sigur (elimină orice sub -25dB)
                audio_trimmed, _ = librosa.effects.trim(audio, top_db=25)
                
                # 3. Verificare durată DUPĂ trim
                duration = len(audio_trimmed) / self.cfg.sample_rate
                
                if duration < self.cfg.min_clip_duration:
                    skipped += 1
                    continue
                if duration > self.cfg.max_clip_duration:
                    skipped += 1
                    continue
                
                # 4. Normalizare RMS (volum consistent între vorbitori)
                rms = np.sqrt(np.mean(audio_trimmed ** 2))
                if rms > 0:
                    target_rms = 0.08  # ~-22 dBFS
                    audio_trimmed = audio_trimmed * (target_rms / rms)
                    # Clip pentru a evita saturarea
                    audio_trimmed = np.clip(audio_trimmed, -0.99, 0.99)
                
                # 5. Salvare WAV
                clip_idx = len([
                    p for p in processed
                    if p["speaker_name"] == entry["speaker_name"]
                ]) + 1
                
                wav_filename = f"{entry['speaker_name']}_{clip_idx:04d}.wav"
                wav_path = self.wavs_dir / wav_filename
                
                sf.write(str(wav_path), audio_trimmed, self.cfg.sample_rate)
                
                processed.append({
                    "wav_path": str(wav_path),
                    "wav_filename": wav_filename,
                    "text": entry["sentence"],
                    "speaker_name": entry["speaker_name"],
                    "speaker_id": entry["speaker_id"],
                    "duration": duration
                })
                
                if (i + 1) % 200 == 0:
                    logger.info(f"  Procesat {i + 1}/{total} "
                                f"({len(processed)} valide, {skipped} skip)")
                
            except Exception as e:
                logger.debug(f"Skip {entry.get('mp3_path', '?')}: {e}")
                skipped += 1
                continue
        
        logger.info(f"Conversie completă: {len(processed)} valide, "
                     f"{skipped} respinse")
        return processed
    
    # -----------------------------------------------------------------
    # SPLIT TRAIN/VAL
    # -----------------------------------------------------------------
    
    def _split_train_val(
        self,
        entries: List[Dict],
        val_ratio: float
    ) -> Tuple[List[Dict], List[Dict]]:
        """Split stratificat per speaker."""
        import random
        random.seed(42)
        
        by_speaker = {}
        for e in entries:
            spk = e["speaker_name"]
            if spk not in by_speaker:
                by_speaker[spk] = []
            by_speaker[spk].append(e)
        
        train, val = [], []
        for spk, spk_entries in by_speaker.items():
            random.shuffle(spk_entries)
            n_val = max(1, int(len(spk_entries) * val_ratio))
            val.extend(spk_entries[:n_val])
            train.extend(spk_entries[n_val:])
        
        logger.info(f"Split: {len(train)} train, {len(val)} val")
        return train, val
    
    # -----------------------------------------------------------------
    # SCRIERE METADATA
    # -----------------------------------------------------------------
    
    def _write_metadata(self, entries: List[Dict], path: Path):
        """Scrie metadata CSV în format Coqui: audio_file|text|speaker_name"""
        with open(path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f, delimiter='|')
            for e in entries:
                writer.writerow([
                    e["wav_filename"],
                    e["text"],
                    e["speaker_name"]
                ])
        
        logger.info(f"Metadata salvată: {path} ({len(entries)} intrări)")
