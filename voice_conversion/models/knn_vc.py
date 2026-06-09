"""
kNN-VC — k-Nearest Neighbors Voice Conversion
===============================================

Implementare bazată pe: Baas et al. (2023)
"kNN-VC: Voice Conversion via k-Nearest Neighbors"

Arhitectura:
  1. WavLM-Large (frozen) — extrage features frame-level (layer 6, 1024-dim)
  2. k-NN Matching — înlocuiește fiecare frame sursă cu media celor k
     cei mai apropiați vecini din features-urile vocii țintă
  3. HiFi-GAN Vocoder — sintetizează waveform din features-urile convertite

Avantaje:
  - Any-to-any: orice vorbitor → orice vorbitor
  - Nu necesită antrenare pentru conversie (non-parametric)
  - Calitate excelentă cu prematched vocoder
  - Funcționează cu orice limbă

Utilizare:
    converter = KnnVoiceConverter()
    result = converter.convert("sursa.wav", ["referinta1.wav", "referinta2.wav"])
    # result.converted_audio = tensor audio convertit
    # result.save("output.wav")
"""

import torch
import torchaudio
import numpy as np
from pathlib import Path
from typing import List, Union, Optional, Dict
from dataclasses import dataclass
import logging
import time

from voice_conversion.config import (
    KNN_VC_CFG, CONVERTED_AUDIO_DIR, get_device
)
from voice_conversion.utils.audio_utils import (
    load_audio, save_audio, get_audio_info
)

logger = logging.getLogger(__name__)


@dataclass
class ConversionResult:
    """
    Rezultatul unei conversii de voce.

    Conține audio-ul convertit, metadate și metode de salvare.
    """
    converted_audio: torch.Tensor
    """Tensor audio convertit [1, T]."""

    sample_rate: int = 16000
    """Sample rate al audio-ului convertit."""

    source_path: Optional[str] = None
    """Calea fișierului audio sursă."""

    target_paths: Optional[List[str]] = None
    """Căile fișierelor audio de referință (target speaker)."""

    topk: int = 4
    """Valoarea k utilizată pentru kNN matching."""

    conversion_time: float = 0.0
    """Timpul de conversie (secunde)."""

    device_used: str = "cpu"
    """Device-ul folosit pentru conversie."""

    def save(self, filepath: Union[str, Path] = None) -> Path:
        """
        Salvează audio-ul convertit ca fișier WAV.

        Args:
            filepath: Calea fișierului (auto-generată dacă None)

        Returns:
            Path: Calea fișierului salvat
        """
        if filepath is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            source_name = Path(self.source_path).stem if self.source_path else "unknown"
            filepath = CONVERTED_AUDIO_DIR / f"converted_{source_name}_{timestamp}.wav"

        return save_audio(self.converted_audio, filepath, self.sample_rate)

    def get_duration(self) -> float:
        """Durata audio-ului convertit în secunde."""
        return self.converted_audio.shape[-1] / self.sample_rate

    def to_dict(self) -> Dict:
        """Serializare ca dicționar (pentru API)."""
        return {
            "duration": self.get_duration(),
            "sample_rate": self.sample_rate,
            "source": self.source_path,
            "targets": self.target_paths,
            "topk": self.topk,
            "conversion_time_ms": round(self.conversion_time * 1000, 1),
            "device": self.device_used
        }


class KnnVoiceConverter:
    """
    Convertor de voce bazat pe kNN-VC (model pre-antrenat).

    Această clasă oferă o interfață simplă pentru conversia vocii
    folosind modelul kNN-VC pre-antrenat, disponibil prin torch.hub.

    Arhitectura internă:
        WavLM-Large → k-NN Matching → HiFi-GAN Vocoder

    Exemplu:
        >>> converter = KnnVoiceConverter()
        >>> result = converter.convert(
        ...     source_audio="speaker_a.wav",
        ...     target_references=["speaker_b_1.wav", "speaker_b_2.wav"],
        ...     topk=4
        ... )
        >>> result.save("converted_output.wav")
    """

    def __init__(
        self,
        config: Optional[object] = None,
        device: str = "auto"
    ):
        """
        Inițializare kNN Voice Converter.

        Args:
            config: Obiect KnnVCConfig (default: configurare globală)
            device: Device pentru inference ('auto', 'cuda', 'cpu')
        """
        self.config = config or KNN_VC_CFG

        # kNN-VC folosește WavLM + HiFi-GAN prin torch.hub
        # Acestea NU suportă DirectML (AMD GPU) — forțăm CPU
        requested = device if device != "auto" else self.config.device
        resolved = get_device(requested)
        if isinstance(resolved, str) and resolved in ("cuda", "cpu"):
            self.device = resolved
        elif hasattr(resolved, 'type') and resolved.type == "cuda":
            self.device = resolved
        else:
            # DirectML sau alt device non-CUDA → fallback CPU
            logger.info("kNN-VC nu suportă DirectML/AMD GPU — se folosește CPU")
            self.device = "cpu"

        self.model = None
        self._loaded = False

        logger.info("kNN-VC Converter inițializat")
        logger.info(f"   Device: {self.device}")

    def load_model(self) -> None:
        """
        Încarcă modelul kNN-VC de pe torch.hub.

        La prima rulare, descarcă automat:
        - WavLM-Large (~1.2GB)
        - HiFi-GAN Vocoder prematched (~55MB)
        """
        if self._loaded:
            logger.info("✓ Modelul este deja încărcat")
            return

        logger.info("📥 Încărcare model kNN-VC...")
        logger.info(f"   Repository: {self.config.hub_repo}")
        logger.info(f"   Prematched vocoder: {self.config.prematched}")
        logger.info("   (Prima încărcare descarcă ~1.3GB, durează câteva minute)")

        try:
            self.model = torch.hub.load(
                self.config.hub_repo,
                self.config.model_name,
                prematched=self.config.prematched,
                trust_repo=self.config.trust_repo,
                device=self.device
            )
            self._loaded = True
            logger.info("✅ Model kNN-VC încărcat cu succes!")

        except Exception as e:
            logger.error(f"❌ Eroare la încărcarea modelului: {e}")
            logger.error("   Verifică conexiunea la internet și PyTorch.")
            raise

    def convert(
        self,
        source_audio: Union[str, Path, torch.Tensor],
        target_references: Union[str, Path, List[Union[str, Path]]],
        topk: Optional[int] = None,
        output_path: Optional[Union[str, Path]] = None
    ) -> ConversionResult:
        """
        Convertește vocea din audio-ul sursă în vocea target.

        Pipeline:
            1. Extrage features WavLM din audio-ul sursă
            2. Construiește matching set din referințele target
            3. Pentru fiecare frame sursă, găsește k-NN din matching set
            4. Sintetizează waveform cu HiFi-GAN

        Args:
            source_audio: Audio sursă (cale fișier sau tensor)
            target_references: Referință(e) vorbitor țintă (una sau mai multe)
            topk: Număr de vecini k (default: din config, 4)
            output_path: Cale opțională pentru salvare automată

        Returns:
            ConversionResult: Obiect cu audio convertit și metadate
        """
        # Asigurare model încărcat
        if not self._loaded:
            self.load_model()

        topk = topk or self.config.topk
        start_time = time.time()

        # Pregătire paths
        source_path = str(source_audio) if not isinstance(source_audio, torch.Tensor) else None

        if isinstance(target_references, (str, Path)):
            target_references = [target_references]
        target_paths = [str(p) for p in target_references]

        logger.info(f"\n🎤 Conversie voce:")
        logger.info(f"   Sursă: {source_path or 'tensor'}")
        logger.info(f"   Referințe target: {len(target_paths)} fișier(e)")
        logger.info(f"   k = {topk}")

        # Step 1: Extrage features din sursa
        logger.info("   [1/3] Extracție features sursă (WavLM)...")
        if isinstance(source_audio, torch.Tensor):
            # Salvare temporară dacă e tensor
            import tempfile
            tmp_path = Path(tempfile.mktemp(suffix=".wav"))
            save_audio(source_audio, tmp_path)
            query_seq = self.model.get_features(str(tmp_path))
            tmp_path.unlink()
        else:
            query_seq = self.model.get_features(str(source_audio))

        # Step 2: Construiește matching set din referințe
        logger.info("   [2/3] Construire matching set din referințe...")
        matching_set = self.model.get_matching_set(target_paths)

        # Step 3: kNN matching + vocoding
        logger.info("   [3/3] kNN matching + sinteză audio...")
        converted_wav = self.model.match(query_seq, matching_set, topk=topk)

        conversion_time = time.time() - start_time

        logger.info(f"   ✅ Conversie completă în {conversion_time:.2f}s")

        # Construire rezultat
        result = ConversionResult(
            converted_audio=converted_wav.unsqueeze(0).cpu(),
            sample_rate=16000,
            source_path=source_path,
            target_paths=target_paths,
            topk=topk,
            conversion_time=conversion_time,
            device_used=self.device
        )

        # Salvare automată dacă specificat
        if output_path:
            result.save(output_path)

        return result

    def get_features(self, audio_path: Union[str, Path]) -> torch.Tensor:
        """
        Extrage features WavLM dintr-un fișier audio.

        Util pentru analiză și vizualizare.

        Args:
            audio_path: Calea fișierului audio

        Returns:
            torch.Tensor: Features [T, 1024]
        """
        if not self._loaded:
            self.load_model()

        return self.model.get_features(str(audio_path))

    def build_speaker_matching_set(
        self,
        audio_paths: List[Union[str, Path]]
    ) -> torch.Tensor:
        """
        Construiește un matching set pentru un vorbitor (pre-calculare).

        Util pentru a pre-calcula matching set-uri pentru vorbitori frecvenți,
        economisind timp la conversii repetate.

        Args:
            audio_paths: Lista cu fișiere audio ale vorbitorului

        Returns:
            torch.Tensor: Matching set pre-calculat
        """
        if not self._loaded:
            self.load_model()

        paths = [str(p) for p in audio_paths]
        return self.model.get_matching_set(paths)

    def batch_convert(
        self,
        source_audios: List[Union[str, Path]],
        target_references: List[Union[str, Path]],
        topk: Optional[int] = None,
        output_dir: Optional[Union[str, Path]] = None
    ) -> List[ConversionResult]:
        """
        Conversie batch pentru mai multe fișiere sursă.

        Optimizat: calculează matching set o singură dată.

        Args:
            source_audios: Lista de fișiere audio sursă
            target_references: Referințe pentru vorbitor țintă
            topk: Număr de vecini k
            output_dir: Director pentru salvarea output-urilor

        Returns:
            Lista de ConversionResult
        """
        if not self._loaded:
            self.load_model()

        topk = topk or self.config.topk
        output_dir = Path(output_dir) if output_dir else CONVERTED_AUDIO_DIR

        logger.info(f"\n🔄 Conversie batch: {len(source_audios)} fișiere")

        # Pre-calculare matching set (o singură dată)
        logger.info("   Construire matching set...")
        target_paths = [str(p) for p in target_references]
        matching_set = self.model.get_matching_set(target_paths)

        results = []
        for i, source in enumerate(source_audios, 1):
            logger.info(f"   [{i}/{len(source_audios)}] Conversie: {Path(source).name}")

            start_time = time.time()
            query_seq = self.model.get_features(str(source))
            converted_wav = self.model.match(query_seq, matching_set, topk=topk)
            conversion_time = time.time() - start_time

            result = ConversionResult(
                converted_audio=converted_wav.unsqueeze(0).cpu(),
                sample_rate=16000,
                source_path=str(source),
                target_paths=target_paths,
                topk=topk,
                conversion_time=conversion_time,
                device_used=self.device
            )

            # Salvare
            out_path = output_dir / f"converted_{Path(source).stem}.wav"
            result.save(out_path)
            results.append(result)

        logger.info(f"   ✅ Batch completat: {len(results)} conversii")
        return results

    def get_model_info(self) -> Dict:
        """Returnează informații despre model."""
        return {
            "name": "kNN-VC",
            "paper": "Baas et al. (2023) - Voice Conversion via k-Nearest Neighbors",
            "encoder": "WavLM-Large (frozen, layer 6, 1024-dim)",
            "matching": f"k-NN (k={self.config.topk})",
            "vocoder": f"HiFi-GAN V1 (prematched={self.config.prematched})",
            "type": "Non-parametric, any-to-any",
            "device": self.device,
            "loaded": self._loaded
        }

    def __repr__(self):
        return (
            f"KnnVoiceConverter("
            f"loaded={self._loaded}, "
            f"device={self.device}, "
            f"topk={self.config.topk})"
        )
