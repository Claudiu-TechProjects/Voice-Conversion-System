"""
Speaker Clustering — Pseudo-vorbitori din Common Voice
=======================================================

Problema: CSV-ul conține un singur speaker_id fictiv ("speaker_common_voice").
Soluție: Clustering K-Means pe embeddings ECAPA-TDNN pentru a crea
         pseudo-vorbitori distincți necesari antrenării LightVC.

Utilizare:
    clusterer = SpeakerClusterer(audio_dir="dataset/common_voices20_audio")
    clusterer.run(n_clusters=10)
    # -> salvează dataset/pseudo_speakers.json
"""

import os
import json
import logging
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict

import torch
import torchaudio

logger = logging.getLogger(__name__)


class SpeakerClusterer:
    """
    Clusterizează fișiere audio în pseudo-vorbitori folosind
    embeddings ECAPA-TDNN + K-Means.
    """

    def __init__(
        self,
        audio_dir: str,
        output_path: Optional[str] = None,
        device: str = "auto"
    ):
        self.audio_dir = Path(audio_dir)
        self.output_path = Path(output_path) if output_path else \
            self.audio_dir.parent / "pseudo_speakers.json"

        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.speaker_model = None
        self.audio_files: List[Path] = []
        self.embeddings: Optional[np.ndarray] = None
        self.cluster_labels: Optional[np.ndarray] = None

    def _load_speaker_model(self):
        """Încarcă ECAPA-TDNN din SpeechBrain."""
        try:
            from speechbrain.inference.speaker import EncoderClassifier
            logger.info("Incarcare ECAPA-TDNN pentru speaker embeddings...")
            self.speaker_model = EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                savedir=str(Path(__file__).parent.parent.parent / "models_cache" / "ecapa"),
                run_opts={"device": self.device}
            )
            logger.info("ECAPA-TDNN incarcat.")
        except Exception as e:
            logger.warning(f"SpeechBrain indisponibil: {e}. Folosim MFCC fallback.")
            self.speaker_model = None

    def _collect_audio_files(self, max_files: Optional[int] = None) -> List[Path]:
        """Colectează fișiere WAV din directorul audio."""
        files = sorted(self.audio_dir.glob("*.wav"))
        if max_files:
            files = files[:max_files]
        logger.info(f"Gasit {len(files)} fisiere WAV.")
        return files

    def _extract_embedding_ecapa(self, audio_path: Path) -> Optional[np.ndarray]:
        """Extrage embedding ECAPA-TDNN dintr-un fișier audio."""
        try:
            waveform, sr = torchaudio.load(str(audio_path))
            if sr != 16000:
                resampler = torchaudio.transforms.Resample(sr, 16000)
                waveform = resampler(waveform)
            # Mono
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            # Minim 0.5s
            if waveform.shape[1] < 8000:
                return None

            with torch.no_grad():
                emb = self.speaker_model.encode_batch(waveform)
            return emb.squeeze().cpu().numpy()
        except Exception:
            return None

    def _extract_embedding_mfcc(self, audio_path: Path) -> Optional[np.ndarray]:
        """Fallback: embeddings MFCC când SpeechBrain nu e disponibil."""
        try:
            waveform, sr = torchaudio.load(str(audio_path))
            if sr != 16000:
                resampler = torchaudio.transforms.Resample(sr, 16000)
                waveform = resampler(waveform)
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            if waveform.shape[1] < 8000:
                return None

            mfcc_transform = torchaudio.transforms.MFCC(
                sample_rate=16000, n_mfcc=40,
                melkwargs={"n_fft": 512, "hop_length": 160, "n_mels": 80}
            )
            mfcc = mfcc_transform(waveform)  # [1, 40, T]
            # Statistici (mean + std) ca embedding
            emb = torch.cat([
                mfcc.mean(dim=-1),
                mfcc.std(dim=-1)
            ], dim=-1).squeeze().numpy()  # [80]
            return emb
        except Exception:
            return None

    def extract_embeddings(self, max_files: Optional[int] = None) -> np.ndarray:
        """Extrage embeddings pentru toate fișierele audio."""
        self._load_speaker_model()
        self.audio_files = self._collect_audio_files(max_files)

        embeddings = []
        valid_files = []
        use_ecapa = self.speaker_model is not None

        logger.info(f"Extragere embeddings ({('ECAPA-TDNN' if use_ecapa else 'MFCC fallback')})...")

        for i, audio_path in enumerate(self.audio_files):
            if i % 50 == 0:
                logger.info(f"  Progres: {i}/{len(self.audio_files)}")

            if use_ecapa:
                emb = self._extract_embedding_ecapa(audio_path)
            else:
                emb = self._extract_embedding_mfcc(audio_path)

            if emb is not None:
                embeddings.append(emb)
                valid_files.append(audio_path)

        self.audio_files = valid_files
        self.embeddings = np.array(embeddings)
        logger.info(f"Embeddings extrase: {self.embeddings.shape} pentru {len(valid_files)} fisiere.")
        return self.embeddings

    def cluster(self, n_clusters: int = 10) -> np.ndarray:
        """Aplică K-Means clustering pe embeddings."""
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler

        if self.embeddings is None:
            raise RuntimeError("Apelați extract_embeddings() mai întâi!")

        # Normalizare
        scaler = StandardScaler()
        emb_scaled = scaler.fit_transform(self.embeddings)

        # K-Means cu restartare multiplă
        n_clusters = min(n_clusters, len(self.audio_files) // 5)
        n_clusters = max(n_clusters, 2)

        logger.info(f"K-Means clustering: {n_clusters} clustere pe {len(self.audio_files)} sample-uri...")
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self.cluster_labels = kmeans.fit_predict(emb_scaled)

        # Log distribuție
        unique, counts = np.unique(self.cluster_labels, return_counts=True)
        for u, c in zip(unique, counts):
            logger.info(f"  Speaker {u:02d}: {c} sample-uri")

        return self.cluster_labels

    def save(self) -> Dict:
        """Salvează rezultatele clustering în JSON."""
        if self.cluster_labels is None:
            raise RuntimeError("Apelați cluster() mai întâi!")

        # Construiește structura: {speaker_id: [audio_files]}
        speakers: Dict[str, List[str]] = {}
        for audio_path, label in zip(self.audio_files, self.cluster_labels):
            speaker_id = f"pseudo_speaker_{label:02d}"
            if speaker_id not in speakers:
                speakers[speaker_id] = []
            speakers[speaker_id].append(str(audio_path))

        # Statistici
        result = {
            "num_speakers": len(speakers),
            "total_files": len(self.audio_files),
            "speakers": speakers,
            "metadata": {
                "n_clusters": len(speakers),
                "embedding_method": "ecapa" if self.speaker_model else "mfcc",
                "audio_dir": str(self.audio_dir)
            }
        }

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        logger.info(f"Pseudo-vorbitori salvati in: {self.output_path}")
        logger.info(f"  {len(speakers)} vorbitori, {len(self.audio_files)} fisiere total")
        return result

    def run(
        self,
        n_clusters: int = 10,
        max_files: Optional[int] = None
    ) -> Dict:
        """Pipeline complet: extrage embeddings -> clustering -> salvare."""
        self.extract_embeddings(max_files=max_files)
        self.cluster(n_clusters=n_clusters)
        return self.save()

    @staticmethod
    def load(output_path: str) -> Dict:
        """Încarcă pseudo-vorbitorii din fișierul JSON."""
        with open(output_path, 'r', encoding='utf-8') as f:
            return json.load(f)
