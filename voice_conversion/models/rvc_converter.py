"""
RVC (Retrieval-based Voice Conversion) — Converter
====================================================

Post-procesare cu RVC care aplică timbrul vocal specific al
speaker-ului țintă peste audio-ul generat de XTTS.

Folosește rvc-python pentru inferență cu RMVPE pitch extraction.

Pipeline:
    Audio TTS → RVC (model per-speaker) → Audio cu timbru corect
"""

import os
import logging
import tempfile
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Tuple

logger = logging.getLogger(__name__)


class RVCConverter:
    """
    Wrapper peste rvc-python pentru aplicarea timbrului vocal.
    Funcționează cu modele .pth antrenate per-speaker.
    """
    
    def __init__(self, config=None):
        if config is None:
            from voice_conversion.config import RVC_CFG, PROJECT_ROOT
            self.cfg = RVC_CFG
            self.project_root = PROJECT_ROOT
        else:
            self.cfg = config
            self.project_root = Path(__file__).parent.parent.parent
        
        self.checkpoint_dir = self.project_root / self.cfg.checkpoint_dir
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self._rvc = None
        self._is_available = None
    
    def is_available(self) -> bool:
        """Verifică dacă rvc-python este instalat."""
        if self._is_available is not None:
            return self._is_available
        
        try:
            from rvc_python.infer import RVCInference
            self._is_available = True
        except ImportError:
            logger.warning(
                "rvc-python nu este instalat. "
                "Rulează: pip install rvc-python"
            )
            self._is_available = False
        
        return self._is_available
    
    def get_speaker_model_path(self, speaker_id: str) -> Optional[Path]:
        """Returnează calea modelului RVC pentru un speaker, dacă există."""
        speaker_dir = self.checkpoint_dir / speaker_id
        model_path = speaker_dir / "model.pth"
        if model_path.exists():
            return model_path
        return None
    
    def has_model(self, speaker_id: str) -> bool:
        """Verifică dacă un speaker are model RVC antrenat."""
        return self.get_speaker_model_path(speaker_id) is not None
    
    def list_trained_speakers(self) -> list:
        """Listează toți speakerii cu model RVC antrenat."""
        speakers = []
        if self.checkpoint_dir.exists():
            for d in self.checkpoint_dir.iterdir():
                if d.is_dir() and (d / "model.pth").exists():
                    speakers.append(d.name)
        return speakers
    
    def convert(
        self,
        input_audio_path: str,
        speaker_id: str,
        output_path: Optional[str] = None
    ) -> str:
        """
        Aplică timbrul vocal RVC peste audio-ul de intrare.
        
        Args:
            input_audio_path: Calea audio-ului de intrare (ex: output XTTS)
            speaker_id: ID-ul speaker-ului cu model RVC antrenat
            output_path: Calea de ieșire (opțional, se generează automat)
            
        Returns:
            Calea fișierului audio procesat cu RVC
        """
        if not self.is_available():
            raise RuntimeError("rvc-python nu este instalat")
        
        model_path = self.get_speaker_model_path(speaker_id)
        if model_path is None:
            raise FileNotFoundError(
                f"Nu există model RVC pentru speaker-ul '{speaker_id}'. "
                "Antrenează mai întâi modelul din pagina de antrenare."
            )
        
        if output_path is None:
            input_p = Path(input_audio_path)
            output_path = str(input_p.parent / f"{input_p.stem}_rvc{input_p.suffix}")
        
        try:
            from rvc_python.infer import RVCInference
            
            # Inițializare RVC (pe CPU pentru AMD)
            rvc = RVCInference(device="cpu")
            rvc.load_model(str(model_path))
            
            # Setare parametri
            rvc.set_params(
                f0method=self.cfg.f0_method,
                index_rate=self.cfg.index_rate,
                filter_radius=self.cfg.filter_radius,
                rms_mix_rate=self.cfg.rms_mix_rate,
                protect=self.cfg.protect
            )
            
            # Inferență
            logger.info(f"RVC: Conversie timbru vocal pentru speaker '{speaker_id}'")
            rvc.infer_file(input_audio_path, output_path)
            
            logger.info(f"RVC: Output salvat: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Eroare RVC inferență: {e}")
            # Fallback: returnăm fișierul original
            logger.warning("RVC indisponibil, se returnează audio-ul original")
            return input_audio_path
    
    def get_info(self) -> Dict:
        """Returnează informații despre starea RVC."""
        return {
            "available": self.is_available(),
            "trained_speakers": self.list_trained_speakers(),
            "f0_method": self.cfg.f0_method,
            "index_rate": self.cfg.index_rate,
            "checkpoint_dir": str(self.checkpoint_dir)
        }
