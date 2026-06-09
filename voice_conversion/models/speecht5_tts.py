"""
SpeechT5 TTS — Text-to-Speech cu Voice Cloning
================================================

Generează vorbire din text cu vocea unui vorbitor selectat.
Flux: Text + Referință vocală → SpeechT5 TTS → HiFi-GAN → WAV

Componente:
  - microsoft/speecht5_tts: model encoder-decoder text→mel
  - microsoft/speecht5_hifigan: vocoder mel→waveform
  - speechbrain/spkrec-xvect-voxceleb: extractor amprentă vocală (512-dim)

Utilizare:
    tts = SpeechT5TTS()
    audio_np, sr = tts.synthesize("Bună ziua!", "referinta_vorbitor.wav")
"""

import os
import shutil
import time
import logging
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict, Any, Union, Tuple

logger = logging.getLogger(__name__)

# Patch pentru Windows symlinks (SpeechBrain necesită symlinks, Windows le blochează)
_original_symlink = os.symlink
def _patched_symlink(src, dst, target_is_directory=False, **kwargs):
    if os.path.isdir(src):
        if not os.path.exists(dst):
            shutil.copytree(src, dst)
    else:
        if not os.path.exists(dst):
            shutil.copy(src, dst)
os.symlink = _patched_symlink


class SpeechT5TTS:
    """
    Text-to-Speech cu voice cloning folosind SpeechT5 de la Microsoft.

    Extrage amprenta vocală (x-vector) dintr-un fișier audio de referință
    și generează vorbire nouă din text cu acea voce.
    """

    def __init__(self, device: str = "cpu"):
        """
        Args:
            device: 'cpu' sau 'cuda'. DirectML nu este suportat.
        """
        self.device = device
        self.processor = None
        self.model = None
        self.vocoder = None
        self.spk_model = None
        self.is_loaded = False

    def load_model(self):
        """Încarcă toate componentele modelului (lazy loading)."""
        if self.is_loaded:
            return

        logger.info("Încărcare SpeechT5 TTS...")
        t0 = time.time()

        try:
            import torch
            from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
            from speechbrain.inference.speaker import EncoderClassifier

            # Model TTS
            self.processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
            self.model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts").to(self.device)
            
            # Vocoder (reutilizăm același HiFi-GAN deja descărcat)
            self.vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan").to(self.device)

            # Extractor amprentă vocală (X-Vector, 512-dim)
            logger.info("Încărcare SpeechBrain X-Vector...")
            self.spk_model = EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-xvect-voxceleb",
                savedir="pretrained_models/spkrec-xvect-voxceleb",
                run_opts={"device": self.device}
            )

            self.is_loaded = True
            logger.info(f"SpeechT5 TTS încărcat în {time.time()-t0:.1f}s pe {self.device}")

        except Exception as e:
            logger.error(f"Eroare la încărcarea SpeechT5 TTS: {e}")
            raise

    def _extract_speaker_embedding(self, reference_path: str) -> "torch.Tensor":
        """
        Extrage embedding-ul vocii (x-vector) dintr-un fișier audio de referință.

        Args:
            reference_path: Calea către fișierul audio al vorbitorului

        Returns:
            Tensor cu embedding-ul vocii (1, 512)
        """
        import torch
        import torchaudio

        wav, sr = torchaudio.load(reference_path)
        if sr != 16000:
            wav = torchaudio.functional.resample(wav, sr, 16000)
        if wav.ndim > 1 and wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)

        with torch.no_grad():
            embedding = self.spk_model.encode_batch(wav.to(self.device))
            embedding = embedding.squeeze(1)  # (1, 512)

        return embedding

    def synthesize(
        self,
        text: str,
        speaker_reference: Union[str, Path, List[str]],
        max_length: int = 600
    ) -> Tuple[np.ndarray, int]:
        """
        Generează audio din text cu vocea vorbitorului de referință.

        Args:
            text:               Textul de sintetizat
            speaker_reference:  Cale fișier audio referință sau listă de fișiere
            max_length:         Lungimea maximă a secvenței generate

        Returns:
            Tuple (audio_numpy, sample_rate)
        """
        if not self.is_loaded:
            self.load_model()

        import torch

        # Selectează primul fișier de referință dacă e o listă
        if isinstance(speaker_reference, list):
            ref_path = str(speaker_reference[0])
        else:
            ref_path = str(speaker_reference)

        logger.info(f"TTS Voice Clone: \"{text[:60]}{'...' if len(text) > 60 else ''}\"")
        logger.info(f"   Referință vorbitor: {Path(ref_path).name}")

        t0 = time.time()

        # 1. Tokenizare text
        inputs = self.processor(text=text, return_tensors="pt").to(self.device)

        # 2. Extragere embedding voce
        speaker_embeddings = self._extract_speaker_embedding(ref_path)

        # 3. Generare audio
        with torch.no_grad():
            speech = self.model.generate_speech(
                inputs["input_ids"],
                speaker_embeddings,
                vocoder=self.vocoder
            )

        audio_np = speech.cpu().numpy()
        sr = 16000  # SpeechT5 generează la 16kHz

        elapsed = time.time() - t0
        duration = len(audio_np) / sr

        logger.info(
            f"   TTS completat în {elapsed:.2f}s → {duration:.1f}s audio"
        )

        return audio_np, sr

    def get_model_info(self) -> Dict:
        """Returnează informații despre model."""
        return {
            "name": "SpeechT5 TTS",
            "full_name": "Microsoft SpeechT5 Text-to-Speech + Voice Cloning",
            "type": "Encoder-Decoder TTS",
            "training_required": False,
            "architecture": {
                "text_encoder": "SpeechT5 Text Encoder",
                "speaker_encoder": "SpeechBrain X-Vector (512-dim)",
                "decoder": "SpeechT5 Speech Decoder",
                "vocoder": "HiFi-GAN"
            },
            "paper": "Ao et al. (2022) - SpeechT5",
            "training_data": "LibriTTS (English)",
            "zero_shot": True,
            "device": self.device,
            "is_loaded": self.is_loaded,
            "limitations": "Antrenat pe engleză; pronunția în română poate fi imperfectă"
        }
