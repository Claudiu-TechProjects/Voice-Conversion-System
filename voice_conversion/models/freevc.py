"""
SpeechT5 — Voice Conversion Pre-antrenat (Microsoft via HuggingFace)
====================================================================

Înlocuiește FreeVC, deoarece Coqui TTS necesită Microsoft C++ Build Tools
pe Windows (compilator C++), ceea ce restricționează mediile locale.
SpeechT5 rulează nativ în Python și oferă o performanță excelentă.

Arhitectură:
  Audio sursă → SpeechT5 Processor → Model Encoder-Decoder → HiFiGAN → WAV
  Audio referință țintă → SpeechBrain X-Vector (512-dim embedding) → SpeechT5

Utilizare:
    converter = FreeVCConverter()  # Nume păstrat pentru compatibilitate
    result = converter.convert("sursa.wav", ["referinta.wav"])
    result.save("output.wav")
"""

import time
import logging
import numpy as np
import os
import shutil
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Union

logger = logging.getLogger(__name__)

# Patch pentru Windows symlinks (pentru SpeechBrain)
_original_symlink = os.symlink
def _patched_symlink(src, dst, target_is_directory=False, **kwargs):
    if os.path.isdir(src):
        if not os.path.exists(dst):
            shutil.copytree(src, dst)
    else:
        if not os.path.exists(dst):
            shutil.copy(src, dst)
os.symlink = _patched_symlink


@dataclass
class FreeVCResult:
    """Rezultatul conversiei SpeechT5."""
    converted_audio: np.ndarray
    sample_rate: int = 16000
    conversion_time: float = 0.0
    source_path: str = ""
    model_name: str = "SpeechT5-VC"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def save(self, output_path) -> Path:
        output_path = Path(output_path)
        import soundfile as sf
        sf.write(str(output_path), self.converted_audio, self.sample_rate)
        return output_path

    def get_duration(self) -> float:
        return len(self.converted_audio) / self.sample_rate


class FreeVCConverter:
    """
    Convertor de voce bazat pe SpeechT5 (înlocuitor FreeVC).
    (Numele clasei e păstrat pt. compatibilitate backend)
    """

    MODEL_NAME = "microsoft/speecht5_vc"

    def __init__(self, device: str = "auto"):
        import torch
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        elif device in ("cuda", "cpu"):
            self.device = device
        else:
            self.device = "cpu"

        self.processor = None
        self.model = None
        self.vocoder = None
        self.spk_model = None
        self.is_loaded = False

    def load_model(self):
        if self.is_loaded:
            return

        logger.info(f"Încărcare SpeechT5 ({self.MODEL_NAME})...")
        t0 = time.time()

        try:
            import torch
            from transformers import SpeechT5Processor, SpeechT5ForSpeechToSpeech, SpeechT5HifiGan
            from speechbrain.inference.speaker import EncoderClassifier

            # Încărcare componente SpeechT5
            self.processor = SpeechT5Processor.from_pretrained(self.MODEL_NAME)
            
            # Incarcam modelul finetuned daca exista, altfel pe cel standard
            checkpoint_dir = Path("checkpoints/freevc_finetuned")
            
            # if (checkpoint_dir / "pytorch_model.bin").exists() or (checkpoint_dir / "model.safetensors").exists():
            #     logger.info("!!! Se incarca varianta SpeechT5 (FreeVC) FINE-TUNED pe lb. romana !!!")
            #     self.model = SpeechT5ForSpeechToSpeech.from_pretrained(str(checkpoint_dir)).to(self.device)
            # else:
            #     self.model = SpeechT5ForSpeechToSpeech.from_pretrained(self.MODEL_NAME).to(self.device)
            
            # Fortam incarcarea modelului preantrenat:
            self.model = SpeechT5ForSpeechToSpeech.from_pretrained(self.MODEL_NAME).to(self.device)
                
            self.vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan").to(self.device)
            
            # Încărcare extractor amprentă vocală (X-Vector)
            logger.info("Încărcare extractor amprentă vocală SpeechBrain...")
            self.spk_model = EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-xvect-voxceleb",
                savedir="pretrained_models/spkrec-xvect-voxceleb",
                run_opts={"device": self.device}
            )

            self.is_loaded = True
            logger.info(f"SpeechT5 încărcat cu succes în {time.time()-t0:.1f}s pe {self.device}")
        except Exception as e:
            logger.error(f"Eroare la încărcarea SpeechT5: {e}")
            raise

    def convert(
        self,
        source_audio: Union[str, Path],
        target_references: List[Union[str, Path]],
        **kwargs
    ) -> FreeVCResult:
        
        if not self.is_loaded:
            self.load_model()

        source_path = str(source_audio)
        
        if not target_references:
            raise ValueError("SpeechT5 necesită cel puțin un fișier referință target")

        logger.info(f"SpeechT5 conversie: {Path(source_path).name} → {len(target_references)} referințe agregate")
        t0 = time.time()

        try:
            import torch
            import torchaudio

            # 1. Procesare sursă (SpeechT5 vrea 16kHz mono)
            source_wav, sr_s = torchaudio.load(source_path)
            if sr_s != 16000:
                source_wav = torchaudio.functional.resample(source_wav, sr_s, 16000)
            if source_wav.ndim > 1:
                source_wav = source_wav.mean(dim=0, keepdim=True)
            source_wav = source_wav.squeeze()

            inputs = self.processor(audio=source_wav, sampling_rate=16000, return_tensors="pt")
            inputs = inputs.to(self.device)

            # 2. Extragere Embeddings (Prin Concatenarea fișierelor audio)
            # În loc de media matematică a embedding-urilor (care nivelează vocile), 
            # concatenăm toate fișierele audio ale referinței într-unul singur mai lung,
            # pentru ca analizatorul să aibă un context "cursiv" și robust despre timbru.
            target_wavs = []
            for target_path in target_references:
                target_wav, sr_t = torchaudio.load(str(target_path))
                if sr_t != 16000:
                    target_wav = torchaudio.functional.resample(target_wav, sr_t, 16000)
                if target_wav.ndim > 1:
                    target_wav = target_wav.mean(dim=0, keepdim=True)
                target_wavs.append(target_wav)
                
            if target_wavs:
                # Concatenăm cap la cap toate formele de undă pe axa timpului (dim=1)
                concatenated_wav = torch.cat(target_wavs, dim=1)
                
                with torch.no_grad():
                    # Extragem o singură amprentă vocală puternică din tot calupul audio
                    emb = self.spk_model.encode_batch(concatenated_wav.to(self.device))
                    spk_embeddings = emb.squeeze(1)
            else:
                raise ValueError("Niciun fișier de referință nu a putut fi încărcat pentru extragerea timbrului.")

            # 3. Generare Audio
            with torch.no_grad():
                # Extragem spectrograma omițând vocoder-ul pentru a evita bug-ul de dimensiune tensor (1877 vs 1876)
                spectrogram = self.model.generate_speech(
                    inputs["input_values"],
                    spk_embeddings,
                    vocoder=None
                )
                
                # spectrogram are forma [seq_len, 80]
                # Adăugăm dimensiunea de batch necesară pentru vocoder: [1, seq_len, 80]
                if spectrogram.ndim == 2:
                    spectrogram = spectrogram.unsqueeze(0)
                
                # Asigurăm că lungimea spectrogramei este multiplu de 2
                seq_len = spectrogram.shape[1]
                if seq_len % 2 != 0:
                    import torch.nn.functional as F
                    spectrogram = F.pad(spectrogram, (0, 0, 0, 1))
                
                # Generăm unda audio cu vocoder-ul folosind spectrograma corectată
                speech = self.vocoder(spectrogram)
                
                if speech.ndim > 1:
                    speech = speech.squeeze()

            audio_np = speech.cpu().numpy()
            conversion_time = time.time() - t0

            logger.info(
                f"SpeechT5 conversie completă: {conversion_time:.2f}s, "
                f"{len(audio_np)/16000:.1f}s audio"
            )

            return FreeVCResult(
                converted_audio=audio_np,
                sample_rate=16000,
                conversion_time=conversion_time,
                source_path=source_path,
                model_name="SpeechT5",
                metadata={
                    "target_reference": target_path,
                    "device": self.device,
                    "model": self.MODEL_NAME
                }
            )

        except Exception as e:
            logger.error(f"Eroare conversie SpeechT5: {e}")
            raise

    def get_model_info(self) -> Dict:
        return {
            "name": "SpeechT5",
            "full_name": "Microsoft SpeechT5 Voice Conversion",
            "type": "Encoder-Decoder",
            "training_required": False,
            "architecture": {
                "content_encoder": "SpeechT5 Base",
                "speaker_encoder": "SpeechBrain X-Vector",
                "decoder": "SpeechT5 Decoder",
                "vocoder": "HiFi-GAN"
            },
            "paper": "Ao et al. (2022) - SpeechT5",
            "training_data": "LibriSpeech / VoxPopuli",
            "zero_shot": True,
            "device": self.device,
            "is_loaded": self.is_loaded
        }
