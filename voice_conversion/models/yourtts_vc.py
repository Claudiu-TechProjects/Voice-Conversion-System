"""
YourTTS Voice Converter — Inferență Voice-to-Voice
====================================================

Conversie voce-în-voce cu YourTTS:
1. Audio sursă → STT (Whisper) → Text
2. Referințe speaker → Speaker Embedding
3. Text + Speaker Embedding → TTS YourTTS → Audio convertit

Modelul suportă:
- Zero-shot: cu modelul pre-antrenat (fără fine-tuning)
- Fine-tuned: cu modelul antrenat pe Common Voice RO
"""

import time
import logging
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict

logger = logging.getLogger(__name__)


class YourTTSConverter:
    """
    Conversie voce-în-voce cu YourTTS.
    
    Folosește pipeline-ul:
    Voce sursă → STT → Text → TTS (cu vocea țintă) → Audio convertit
    """
    
    def __init__(self, config=None):
        if config is None:
            from voice_conversion.config import YOURTTS_CFG, PROJECT_ROOT
            self.cfg = YOURTTS_CFG
            self.project_root = PROJECT_ROOT
        else:
            self.cfg = config
            self.project_root = Path(__file__).parent.parent.parent
        
        self.checkpoint_dir = self.project_root / self.cfg.checkpoint_dir
        self.tts_model = None
        self._loaded = False
    
    def _get_finetuned_path(self):
        """Caută calea către best_model.pth și config.json în subfoldere."""
        # 1. Caută direct în directorul root (dacă a fost copiat manual)
        best_model = self.checkpoint_dir / "best_model.pth"
        config_json = self.checkpoint_dir / "config.json"
        
        if best_model.exists() and config_json.exists():
            return best_model, config_json
            
        # 2. Caută în subdirectoarele generate de Coqui Trainer (ex: yourtts_ro-June-03...)
        if self.checkpoint_dir.exists():
            # Sortează folderele după timpul modificării (cel mai recent)
            subdirs = sorted(
                [d for d in self.checkpoint_dir.iterdir() if d.is_dir()],
                key=lambda x: x.stat().st_mtime,
                reverse=True
            )
            for d in subdirs:
                bm = d / "best_model.pth"
                cj = d / "config.json"
                if bm.exists() and cj.exists():
                    return bm, cj
                    
        return None, None

    def _ensure_loaded(self):
        """Încarcă modelul YourTTS (lazy loading)."""
        if self._loaded:
            return
        
        t0 = time.time()
        logger.info("Încărcare model YourTTS...")
        
        from TTS.api import TTS
        
        # Caută modelul fine-tuned
        best_model_path, config_path = self._get_finetuned_path()
        
        if best_model_path:
            logger.info(f"Folosesc modelul fine-tuned: {best_model_path}")
            try:
                self.tts_model = TTS(
                    model_path=str(best_model_path),
                    config_path=str(config_path)
                )
                self._is_finetuned = True
            except Exception as e:
                logger.warning(
                    f"Nu pot încărca modelul fine-tuned ({e}), "
                    "folosesc pre-antrenat."
                )
                self._load_pretrained()
        else:
            self._load_pretrained()
        
        elapsed = time.time() - t0
        logger.info(f"YourTTS încărcat în {elapsed:.1f}s")
        self._loaded = True
    
    def _load_pretrained(self):
        """Încarcă modelul YourTTS pre-antrenat din Coqui hub."""
        from TTS.api import TTS
        
        logger.info("Descărcare/încărcare model YourTTS pre-antrenat...")
        self.tts_model = TTS("tts_models/multilingual/multi-dataset/your_tts")
        self._is_finetuned = False
        logger.info("Model YourTTS pre-antrenat încărcat.")
    
    def convert(
        self,
        source_audio_path: str,
        target_references: List[str],
        output_path: Optional[str] = None,
        language: str = "ro"
    ) -> Dict:
        """
        Conversie voce-în-voce.
        
        Args:
            source_audio_path: Calea audio-ului sursă
            target_references: Lista fișierelor audio de referință speaker
            output_path: Calea unde se salvează rezultatul
            language: Limba textului (pentru STT + TTS)
        
        Returns:
            Dict cu: output_path, duration, conversion_time, source_text
        """
        self._ensure_loaded()
        
        t0 = time.time()
        
        source_path = Path(source_audio_path)
        if not source_path.exists():
            raise FileNotFoundError(f"Fișier sursă inexistent: {source_path}")
        
        # 1. STT — Transcriere audio sursă
        logger.info(f"YourTTS conversie: {source_path.name}")
        source_text = self._transcribe(str(source_path), language)
        logger.info(f"  STT: '{source_text}'")
        
        if not source_text or len(source_text.strip()) < 2:
            raise ValueError("STT nu a putut transcrie audio-ul sursă!")
        
        # 2. Selectare referință speaker (prima disponibilă)
        speaker_ref = None
        for ref in target_references:
            ref_path = Path(ref)
            if ref_path.exists():
                speaker_ref = str(ref_path)
                break
        
        if not speaker_ref:
            raise ValueError("Nicio referință speaker validă!")
        
        # 3. TTS — Sinteză cu vocea țintă
        if output_path is None:
            output_path = str(
                self.project_root / "webapp" / "outputs" /
                f"yourtts_{int(time.time())}.wav"
            )
        
        # Ajustăm limba pentru modelul pre-antrenat
        # YourTTS pre-antrenat suportă: en, pt-br, fr
        # Pentru română, folosim 'en' ca fallback (modelul face embedding)
        tts_lang = language if language in ["en", "pt-br", "fr"] else "en"
        
        try:
            if self._is_finetuned:
                # Bypass wrapper-ul TTS pentru modelul fine-tuned pentru a evita eroarea cu is_multi_lingual
                wav = self.tts_model.synthesizer.tts(
                    text=source_text,
                    speaker_wav=speaker_ref
                )
                self.tts_model.synthesizer.save_wav(wav, output_path)
            else:
                # Modelul pre-antrenat suportă multi-lingual
                self.tts_model.tts_to_file(
                    text=source_text,
                    file_path=output_path,
                    speaker_wav=speaker_ref,
                    language=tts_lang
                )
        except Exception as e:
            logger.error(f"Eroare TTS YourTTS: {e}")
            # Fallback doar pentru pre-antrenat
            if not self._is_finetuned and tts_lang != "en":
                logger.info("Retry cu limba 'en'...")
                self.tts_model.tts_to_file(
                    text=source_text,
                    file_path=output_path,
                    speaker_wav=speaker_ref,
                    language="en"
                )
            else:
                raise
        
        # 4. Calculare durată
        import librosa
        duration = librosa.get_duration(path=output_path)
        
        conversion_time = time.time() - t0
        
        logger.info(
            f"  Conversie completă: {conversion_time:.1f}s, "
            f"{duration:.1f}s audio"
        )
        
        return {
            "output_path": output_path,
            "duration": duration,
            "conversion_time": conversion_time,
            "source_text": source_text,
            "method": "finetuned" if self._is_finetuned else "pretrained"
        }
    
    def _transcribe(self, audio_path: str, language: str = "ro") -> str:
        """Transcriere audio cu Whisper STT."""
        try:
            # Folosim STT-ul din aplicație dacă e disponibil
            from webapp.backend.app import get_whisper_stt
            stt = get_whisper_stt()
            result = stt.transcribe(audio_path)
            return result.get("text", "")
        except Exception:
            pass
        
        # Fallback: Whisper direct
        try:
            import whisper
            model = whisper.load_model("base")
            result = model.transcribe(audio_path, language=language)
            return result.get("text", "")
        except Exception as e:
            logger.error(f"Eroare STT: {e}")
            return ""
    
    def is_available(self) -> bool:
        """Verifică dacă modelul poate fi încărcat."""
        try:
            from TTS.api import TTS
            return True
        except ImportError:
            return False
    
    def has_finetuned_model(self) -> bool:
        """Verifică dacă există un model fine-tuned."""
        bm, _ = self._get_finetuned_path()
        return bm is not None
    
    def get_model_info(self) -> Dict:
        """Returnează informații despre modelul disponibil."""
        finetuned = self.has_finetuned_model()
        
        info = {
            "available": self.is_available(),
            "finetuned": finetuned,
            "model_type": "fine-tuned" if finetuned else "pre-trained",
            "checkpoint_dir": str(self.checkpoint_dir)
        }
        
        if finetuned:
            # Citire metadata checkpoint
            best_model_path, _ = self._get_finetuned_path()
            try:
                import torch
                ck = torch.load(
                    str(best_model_path),
                    map_location="cpu",
                    weights_only=False
                )
                info["epoch"] = ck.get("epoch", 0)
                info["loss"] = ck.get("loss", None)
                info["num_speakers"] = ck.get("config", {}).get(
                    "num_speakers", 0
                )
            except Exception:
                pass
        
        return info
