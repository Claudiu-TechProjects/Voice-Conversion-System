"""
Whisper STT — Speech-to-Text cu OpenAI Whisper
================================================

Transcripție automată audio → text folosind Whisper.
Suportă limba română, rulează local (fără API key).

Utilizare:
    stt = WhisperSTT(model_size="base")
    result = stt.transcribe("audio.wav")
    print(result["text"])
"""

import logging
import time
from pathlib import Path
from typing import Optional, Dict

logger = logging.getLogger(__name__)

# Singleton model (evită re-încărcarea)
_whisper_model = None
_whisper_model_size = None


class WhisperSTT:
    """
    Speech-to-Text cu OpenAI Whisper.

    Suportă limba română nativ. Modelul se încarcă lazy
    și rămâne în memorie pentru transcripții ulterioare.
    """

    # Dimensiuni model și RAM necesar aproximativ
    MODEL_SIZES = {
        "tiny": {"params": "39M", "ram": "~1 GB", "speed": "~32x"},
        "base": {"params": "74M", "ram": "~1 GB", "speed": "~16x"},
        "small": {"params": "244M", "ram": "~2 GB", "speed": "~6x"},
        "medium": {"params": "769M", "ram": "~5 GB", "speed": "~2x"},
    }

    def __init__(self, model_size: str = "base"):
        """
        Args:
            model_size: Dimensiunea modelului Whisper.
                        'tiny', 'base', 'small', 'medium'
        """
        self.model_size = model_size
        self._ensure_model_loaded()

    def _ensure_model_loaded(self):
        """Încarcă modelul Whisper (singleton, o singură dată)."""
        global _whisper_model, _whisper_model_size

        if _whisper_model is not None and _whisper_model_size == self.model_size:
            self.model = _whisper_model
            return

        try:
            import whisper
            logger.info(f"Încărcare Whisper model '{self.model_size}'...")
            t0 = time.time()
            _whisper_model = whisper.load_model(self.model_size)
            _whisper_model_size = self.model_size
            self.model = _whisper_model
            logger.info(
                f"Whisper '{self.model_size}' încărcat în {time.time()-t0:.1f}s "
                f"({self.MODEL_SIZES.get(self.model_size, {}).get('params', '?')} parametri)"
            )
        except ImportError:
            logger.error("openai-whisper nu este instalat! pip install openai-whisper")
            raise
        except Exception as e:
            logger.error(f"Eroare la încărcarea Whisper: {e}")
            raise

    def transcribe(
        self,
        audio_path: str,
        language: Optional[str] = "ro",
        task: str = "transcribe"
    ) -> Dict:
        """
        Transcrie un fișier audio în text.

        Args:
            audio_path: Calea către fișierul audio (WAV, MP3, etc.)
            language:   Codul limbii ('ro' pentru română, None pentru autodetect)
            task:       'transcribe' sau 'translate' (traducere în engleză)

        Returns:
            dict cu:
                - text: textul transcris
                - language: limba detectată
                - duration: durata audio-ului
                - transcription_time: timpul de procesare
                - segments: lista de segmente cu timestamps
        """
        audio_path = str(audio_path)

        if not Path(audio_path).exists():
            return {
                "text": "",
                "error": f"Fișier inexistent: {audio_path}",
                "language": language or "unknown",
                "duration": 0,
                "transcription_time": 0,
                "segments": []
            }

        t0 = time.time()

        try:
            options = {
                "task": task,
                "fp16": False,  # CPU nu suportă fp16
                "temperature": 0.0,  # Deterministă, mai precisă
            }
            if language:
                options["language"] = language

            # Prompt inițial care ancorează Whisper pe limba română
            # Ajută la recunoașterea corectă a diacriticelor și a cuvintelor românești
            if language == "ro":
                options["initial_prompt"] = (
                    "Aceasta este o transcriere în limba română. "
                    "Bună ziua, mă numesc și vorbesc în limba română. "
                    "Transcrierea trebuie să conțină diacritice corecte: ă, â, î, ș, ț."
                )

            result = self.model.transcribe(audio_path, **options)

            transcription_time = time.time() - t0
            text = result.get("text", "").strip()
            detected_lang = result.get("language", language or "unknown")

            # Extrage durata din segmente
            segments = result.get("segments", [])
            duration = segments[-1]["end"] if segments else 0

            logger.info(
                f"STT: '{Path(audio_path).name}' → "
                f"'{text[:50]}{'...' if len(text) > 50 else ''}' "
                f"({detected_lang}, {transcription_time:.1f}s)"
            )

            return {
                "text": text,
                "language": detected_lang,
                "duration": round(duration, 2),
                "transcription_time": round(transcription_time, 2),
                "segments": [
                    {
                        "start": round(s["start"], 2),
                        "end": round(s["end"], 2),
                        "text": s["text"].strip()
                    }
                    for s in segments
                ]
            }

        except Exception as e:
            logger.error(f"Eroare transcripție Whisper: {e}")
            return {
                "text": "",
                "error": str(e),
                "language": language or "unknown",
                "duration": 0,
                "transcription_time": round(time.time() - t0, 2),
                "segments": []
            }

    def compute_wer(self, reference: str, hypothesis: str) -> float:
        """
        Calculează Word Error Rate între două texte.

        WER = (S + D + I) / N
            S = substituții, D = ștergeri, I = inserții
            N = număr de cuvinte în referință

        Returns:
            float: WER (0.0 = perfect, 1.0 = complet greșit)
        """
        ref_words = reference.lower().split()
        hyp_words = hypothesis.lower().split()

        if not ref_words:
            return 0.0 if not hyp_words else 1.0

        # Levenshtein la nivel de cuvânt
        n = len(ref_words)
        m = len(hyp_words)
        d = [[0] * (m + 1) for _ in range(n + 1)]

        for i in range(n + 1):
            d[i][0] = i
        for j in range(m + 1):
            d[0][j] = j

        for i in range(1, n + 1):
            for j in range(1, m + 1):
                if ref_words[i-1] == hyp_words[j-1]:
                    d[i][j] = d[i-1][j-1]
                else:
                    d[i][j] = min(
                        d[i-1][j] + 1,    # deletion
                        d[i][j-1] + 1,    # insertion
                        d[i-1][j-1] + 1   # substitution
                    )

        wer = d[n][m] / n
        return round(min(wer, 1.0), 4)

    @staticmethod
    def get_available_models() -> Dict:
        """Returnează modelele disponibile și cerințele lor."""
        return WhisperSTT.MODEL_SIZES
