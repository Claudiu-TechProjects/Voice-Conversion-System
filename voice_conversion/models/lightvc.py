"""
LightVC — Model de Voice Conversion Antrenabil
================================================

Arhitectură:
  Audio → Mel → [Content Encoder] → content_code (32-dim)
                                          ↓ + speaker_emb (192-dim)
               [Speaker Encoder ECAPA, frozen] → speaker_emb
                                          ↓
                               [Decoder] → Mel reconstituit
                                          ↓
                              [HiFi-GAN, frozen] → WAV

Principiu: Information Bottleneck — forțăm content_code să fie
           atât de mic (32-dim) încât nu poate reține identitatea
           vorbitorului, doar conținutul lingvistic.

Referință: Inspirat din AutoVC (Qian et al., 2019) + FreeVC (Li et al., 2022)
"""

import json
import time
import logging
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

logger = logging.getLogger(__name__)


# =====================================================================
# CONTENT ENCODER
# =====================================================================

class Conv1dBNReLU(nn.Module):
    """Bloc de bază: Conv1d + BatchNorm + ReLU."""
    def __init__(self, in_ch: int, out_ch: int, kernel: int = 5,
                 stride: int = 1, padding: int = 2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel, stride=stride, padding=padding),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ContentEncoder(nn.Module):
    """
    Encoder pentru conținut lingvistic.

    Intrare: mel spectrogram [B, 80, T]
    Ieșire:  content code [B, bottleneck_dim, T]

    Bottleneck mic (32-dim) forțează modelul să elimine informațiile
    de identitate (pitch absolut, timbru) și să rețină doar conținutul.
    """

    def __init__(
        self,
        n_mels: int = 80,
        channels: int = 256,
        bottleneck_dim: int = 32,
        num_conv_layers: int = 3
    ):
        super().__init__()

        # Proiecție inițială
        self.input_proj = Conv1dBNReLU(n_mels, channels)

        # Straturi convoluționale
        self.conv_layers = nn.ModuleList([
            Conv1dBNReLU(channels, channels)
            for _ in range(num_conv_layers - 1)
        ])

        # LSTM bidirecțional pentru context temporal
        self.lstm = nn.LSTM(
            input_size=channels,
            hidden_size=channels // 2,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.1
        )

        # Bottleneck — squeeze la dimensiunea mică
        self.bottleneck = nn.Sequential(
            nn.Conv1d(channels, bottleneck_dim, kernel_size=1),
            nn.Tanh()  # limitează range-ul
        )

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """
        Args:
            mel: [B, n_mels, T]
        Returns:
            code: [B, bottleneck_dim, T]
        """
        x = self.input_proj(mel)  # [B, channels, T]

        for conv in self.conv_layers:
            x = conv(x) + x  # residual

        # LSTM (batch_first expects [B, T, C])
        x = x.transpose(1, 2)  # [B, T, channels]
        
        # CPU Fallback pentru DirectML (AMD GPU) deoarece LSTM nu e suportat nativ
        x_device = x.device
        if x_device.type == "privateuseone":
            x_cpu = x.cpu()
            self.lstm.cpu()
            x_cpu, _ = self.lstm(x_cpu)
            self.lstm.to(x_device)
            x = x_cpu.to(x_device)
        else:
            x, _ = self.lstm(x)
            
        x = x.transpose(1, 2)  # [B, channels, T]

        # Bottleneck
        code = self.bottleneck(x)  # [B, bottleneck_dim, T]
        return code


# =====================================================================
# DECODER
# =====================================================================

class Decoder(nn.Module):
    """
    Decoder: content_code + speaker_emb → mel spectrogram.

    Primește content code și speaker embedding, le combină și
    reconstruiește mel spectrogram-ul în stilul vorbitorului target.
    """

    def __init__(
        self,
        bottleneck_dim: int = 32,
        speaker_emb_dim: int = 192,
        channels: int = 512,
        n_mels: int = 80,
        num_conv_layers: int = 5
    ):
        super().__init__()

        input_dim = bottleneck_dim + speaker_emb_dim

        # Proiecție la spațiul decoderului
        self.input_proj = nn.Sequential(
            nn.Conv1d(input_dim, channels, kernel_size=1),
            nn.ReLU(inplace=True)
        )

        # Straturi convoluționale cu receptive field crescut
        self.conv_layers = nn.ModuleList()
        for i in range(num_conv_layers):
            dilation = 2 ** (i % 4)
            padding = dilation * 2
            self.conv_layers.append(nn.Sequential(
                nn.Conv1d(
                    channels, channels,
                    kernel_size=5,
                    padding=padding,
                    dilation=dilation
                ),
                nn.BatchNorm1d(channels),
                nn.ReLU(inplace=True)
            ))

        # LSTM pentru coerență temporală
        self.lstm = nn.LSTM(
            input_size=channels,
            hidden_size=channels // 2,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.1
        )

        # Proiecție la mel
        self.output_proj = nn.Conv1d(channels, n_mels, kernel_size=1)

        # PostNet pentru rafinare (5 conv-uri ca în Tacotron 2)
        self.postnet = nn.Sequential(
            nn.Conv1d(n_mels, 512, kernel_size=5, padding=2),
            nn.BatchNorm1d(512), nn.Tanh(),
            nn.Conv1d(512, 512, kernel_size=5, padding=2),
            nn.BatchNorm1d(512), nn.Tanh(),
            nn.Conv1d(512, 512, kernel_size=5, padding=2),
            nn.BatchNorm1d(512), nn.Tanh(),
            nn.Conv1d(512, 512, kernel_size=5, padding=2),
            nn.BatchNorm1d(512), nn.Tanh(),
            nn.Conv1d(512, n_mels, kernel_size=5, padding=2),
            nn.BatchNorm1d(n_mels)
        )

    def forward(
        self,
        content_code: torch.Tensor,
        speaker_emb: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            content_code: [B, bottleneck_dim, T]
            speaker_emb:  [B, speaker_emb_dim] sau [B, speaker_emb_dim, 1]
        Returns:
            mel_out: [B, n_mels, T]
        """
        B, _, T = content_code.shape

        # Expand speaker embedding la toată secvența
        if speaker_emb.dim() == 2:
            spk = speaker_emb.unsqueeze(-1).expand(-1, -1, T)  # [B, spk_dim, T]
        else:
            spk = speaker_emb.expand(-1, -1, T)

        # Concatenare content + speaker
        x = torch.cat([content_code, spk], dim=1)  # [B, btl+spk, T]
        x = self.input_proj(x)  # [B, channels, T]

        # Conv layers cu residual
        for conv in self.conv_layers:
            x = conv(x) + x

        # LSTM
        x = x.transpose(1, 2)  # [B, T, channels]
        
        # CPU Fallback pentru DirectML (AMD GPU) deoarece LSTM nu e suportat nativ
        x_device = x.device
        if x_device.type == "privateuseone":
            x_cpu = x.cpu()
            self.lstm.cpu()
            x_cpu, _ = self.lstm(x_cpu)
            self.lstm.to(x_device)
            x = x_cpu.to(x_device)
        else:
            x, _ = self.lstm(x)
            
        x = x.transpose(1, 2)  # [B, channels, T]

        # Output mel
        mel_before = self.output_proj(x)  # [B, n_mels, T]
        mel_after = mel_before + self.postnet(mel_before)  # refinement residual

        return mel_after


# =====================================================================
# LIGHTVC MODEL
# =====================================================================

class LightVCModel(nn.Module):
    """
    Modelul complet LightVC.

    Combină ContentEncoder + SpeakerEncoder (frozen ECAPA) + Decoder.
    """

    def __init__(
        self,
        n_mels: int = 80,
        content_channels: int = 256,
        bottleneck_dim: int = 32,
        speaker_emb_dim: int = 192,
        decoder_channels: int = 512,
        num_speakers: int = 10
    ):
        super().__init__()

        self.content_encoder = ContentEncoder(
            n_mels=n_mels,
            channels=content_channels,
            bottleneck_dim=bottleneck_dim
        )

        self.decoder = Decoder(
            bottleneck_dim=bottleneck_dim,
            speaker_emb_dim=speaker_emb_dim,
            channels=decoder_channels,
            n_mels=n_mels
        )

        # Speaker embedding lookup (pentru antrenare fără ECAPA)
        # Alternativă light când ECAPA nu e disponibil
        self.speaker_embedding = nn.Embedding(num_speakers, speaker_emb_dim)

        self.bottleneck_dim = bottleneck_dim
        self.speaker_emb_dim = speaker_emb_dim
        self.n_mels = n_mels

    def forward(
        self,
        mel_src: torch.Tensor,
        speaker_id: torch.Tensor,
        speaker_emb: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass pentru antrenare.

        Args:
            mel_src:     [B, n_mels, T] — mel sursa
            speaker_id:  [B] — index vorbitor target
            speaker_emb: [B, speaker_emb_dim] — embedding precomputat (opțional)

        Returns:
            dict cu mel_pred, content_code
        """
        # Extragere conținut
        content_code = self.content_encoder(mel_src)  # [B, btl, T]

        # Speaker embedding
        if speaker_emb is None:
            spk_emb = self.speaker_embedding(speaker_id)  # [B, spk_dim]
        else:
            spk_emb = speaker_emb  # [B, spk_dim]

        # Decodare
        mel_pred = self.decoder(content_code, spk_emb)  # [B, n_mels, T]

        return {
            "mel_pred": mel_pred,
            "content_code": content_code,
            "speaker_emb": spk_emb
        }

    def encode_content(self, mel: torch.Tensor) -> torch.Tensor:
        """Extrage conținut dintr-un mel spectrogram."""
        with torch.no_grad():
            return self.content_encoder(mel)

    def decode(
        self,
        content_code: torch.Tensor,
        speaker_emb: torch.Tensor
    ) -> torch.Tensor:
        """Decodează content + speaker → mel."""
        with torch.no_grad():
            return self.decoder(content_code, speaker_emb)


# =====================================================================
# CONVERSION RESULT (shared interface cu KnnVoiceConverter)
# =====================================================================

@dataclass
class LightVCResult:
    """Rezultatul conversiei LightVC — aceeași interfață ca ConversionResult."""
    converted_audio: np.ndarray
    sample_rate: int
    conversion_time: float
    source_path: str
    model_name: str = "LightVC"
    metadata: Dict[str, Any] = None

    def save(self, output_path) -> Path:
        output_path = Path(output_path)
        import soundfile as sf
        sf.write(str(output_path), self.converted_audio, self.sample_rate)
        return output_path

    def get_duration(self) -> float:
        return len(self.converted_audio) / self.sample_rate


# =====================================================================
# LIGHTVC CONVERTER — Interfață identică cu KnnVoiceConverter
# =====================================================================

class LightVCConverter:
    """
    Wrapper pentru inferență LightVC.

    Interfață identică cu KnnVoiceConverter pentru a putea fi
    folosit interschimbabil în backend.
    """

    def __init__(self, device: str = "auto"):
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.model: Optional[LightVCModel] = None
        self.speaker_model = None  # ECAPA-TDNN
        self.vocoder = None        # HiFi-GAN (shared cu kNN-VC)
        self.is_loaded = False
        self.checkpoint_info: Dict = {}

        # Mel extractor (identic cu training)
        self._mel_extractor = None

    def _get_mel_extractor(self):
        """Lazy init mel extractor."""
        if self._mel_extractor is None:
            from voice_conversion.data.vc_dataset import MelExtractor
            self._mel_extractor = MelExtractor()
        return self._mel_extractor

    def load_model(self, checkpoint_path: Optional[str] = None):
        """
        Încarcă modelul din checkpoint.

        Args:
            checkpoint_path: Cale către fișierul .pth. Dacă None,
                           caută 'checkpoints/lightvc/best_model.pth'.
        """
        from voice_conversion.config import PROJECT_ROOT, LIGHTVC_CFG

        if checkpoint_path is None:
            checkpoint_path = PROJECT_ROOT / LIGHTVC_CFG.checkpoint_dir / "best_model.pth"

        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Checkpoint LightVC negăsit: {checkpoint_path}\n"
                f"Antrenează modelul mai întâi cu: python scripts/train_lightvc.py"
            )

        logger.info(f"Incarcare LightVC din: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        # Reconstruieste modelul din config salvat
        model_cfg = checkpoint.get("model_config", {})
        self.model = LightVCModel(
            n_mels=model_cfg.get("n_mels", 80),
            content_channels=model_cfg.get("content_channels", 256),
            bottleneck_dim=model_cfg.get("bottleneck_dim", 32),
            speaker_emb_dim=model_cfg.get("speaker_emb_dim", 192),
            decoder_channels=model_cfg.get("decoder_channels", 512),
            num_speakers=model_cfg.get("num_speakers", 10)
        ).to(self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        self.checkpoint_info = {
            "epoch": checkpoint.get("epoch", 0),
            "best_val_loss": checkpoint.get("best_val_loss", None),
            "train_loss_history": checkpoint.get("train_loss_history", []),
            "val_loss_history": checkpoint.get("val_loss_history", []),
            "training_time_hours": checkpoint.get("training_time_hours", 0),
            "num_speakers": model_cfg.get("num_speakers", 10),
            "checkpoint_path": str(checkpoint_path)
        }

        # Încearcă să încarce ECAPA-TDNN pentru speaker embeddings
        self._load_speaker_model()

        # Încearcă să încarce HiFi-GAN vocoder (shared cu kNN-VC)
        self._load_vocoder()

        self.is_loaded = True
        logger.info(
            f"LightVC incarcat: epoch={self.checkpoint_info['epoch']}, "
            f"val_loss={self.checkpoint_info['best_val_loss']}"
        )

    def _load_speaker_model(self):
        """Încarcă ECAPA-TDNN pentru speaker embeddings."""
        try:
            import os
            # Pe Windows, SpeechBrain folosește symlink-uri care necesită privilegii.
            # Setăm HF_HUB_ENABLE_HF_TRANSFER=0 și folosim copy strategy.
            os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '0'

            from speechbrain.inference.speaker import EncoderClassifier
            from voice_conversion.config import PROJECT_ROOT

            cache_dir = PROJECT_ROOT / "models_cache" / "ecapa"
            cache_dir.mkdir(parents=True, exist_ok=True)

            # Verifică dacă fișierele sunt deja copiate
            hparams_file = cache_dir / "hyperparams.yaml"
            if not hparams_file.exists():
                # Descarcă manual fără symlink-uri
                from huggingface_hub import hf_hub_download
                for fname in ["hyperparams.yaml", "embedding_model.ckpt",
                              "classifier.ckpt", "label_encoder.txt",
                              "mean_var_norm_emb.ckpt"]:
                    try:
                        downloaded = hf_hub_download(
                            repo_id="speechbrain/spkrec-ecapa-voxceleb",
                            filename=fname,
                            local_dir=str(cache_dir),
                            local_dir_use_symlinks=False
                        )
                        logger.debug(f"ECAPA fișier descărcat: {fname}")
                    except Exception as e:
                        logger.debug(f"ECAPA fișier opțional lipsă: {fname}: {e}")

            self.speaker_model = EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                savedir=str(cache_dir),
                run_opts={"device": self.device}
            )
            logger.info("ECAPA-TDNN încărcat pentru speaker embeddings.")
        except Exception as e:
            logger.warning(
                f"ECAPA-TDNN indisponibil: {e}. "
                f"Se folosește speaker embedding lookup."
            )
            self.speaker_model = None

    def _load_vocoder(self):
        """Încearcă să partajeze vocodul HiFi-GAN cu kNN-VC."""
        try:
            # Încearcă să obțină vocodul de la kNN-VC dacă e deja încărcat
            self.vocoder = None  # Va fi setat extern dacă e disponibil
            logger.info("Vocoder: se va folosi sinteză directă din mel.")
        except Exception:
            self.vocoder = None

    def _extract_audio_features(self, audio_path: str) -> tuple:
        """
        Extrage mel spectrogram și speaker embedding dintr-un fișier audio.
        Returns: (mel_tensor, speaker_emb)
        """
        import soundfile as sf

        audio, sr = sf.read(audio_path)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        waveform = torch.FloatTensor(audio).unsqueeze(0)  # [1, T]
        if sr != 16000:
            resampler = torchaudio.transforms.Resample(sr, 16000)
            waveform = resampler(waveform)

        mel_extractor = self._get_mel_extractor()
        mel = mel_extractor(waveform)  # [n_mels, T]
        mel = mel.unsqueeze(0).to(self.device)  # [1, n_mels, T]

        # Speaker embedding
        speaker_emb = None
        if self.speaker_model is not None:
            try:
                with torch.no_grad():
                    emb = self.speaker_model.encode_batch(waveform.to(self.device))
                speaker_emb = emb.squeeze(1)  # [1, 192]
            except Exception:
                pass

        return mel, speaker_emb

    def _mel_to_audio(self, mel: torch.Tensor) -> np.ndarray:
        """
        Convertește mel spectrogram în formă de undă audio.
        Folosim pseudo-inversa mel filterbank + Griffin-Lim.
        """
        mel_sq = mel.squeeze(0)  # [n_mels, T]

        # De-normalizare: inversul normalizării din dataset
        # Dataset: mel = (mel_db + 40) / 40, clamp(-1, 1)
        # Inversul: mel_db = mel * 40 - 40
        mel_db = mel_sq * 40.0 - 40.0  # acum în dB, range ~ [-80, 0]

        # dB → amplitudine liniară
        mel_linear = torch.pow(10.0, mel_db / 20.0)  # [n_mels, T]
        mel_linear = mel_linear.clamp(min=1e-5)

        # Construiește mel filterbank și pseudo-inversa
        n_fft = 1024
        n_mels = mel_linear.shape[0]
        sample_rate = 16000

        mel_fb = torchaudio.functional.melscale_fbanks(
            n_freqs=n_fft // 2 + 1,
            f_min=0.0,
            f_max=8000.0,
            n_mels=n_mels,
            sample_rate=sample_rate,
            norm="slaney",
            mel_scale="slaney"
        )  # [n_freqs, n_mels]

        # Pseudo-inversă: mel_fb_pinv [n_mels, n_freqs] → [n_freqs, n_mels]
        # mel_linear [n_mels, T] → spectrogram [n_freqs, T]
        mel_fb_pinv = torch.linalg.pinv(mel_fb.T)  # [n_freqs, n_mels]
        spec_linear = torch.mm(mel_fb_pinv, mel_linear)  # [n_freqs, T]
        spec_linear = spec_linear.clamp(min=0.0)

        # Griffin-Lim: spectrogramă magnitude → audio
        griffin_lim = torchaudio.transforms.GriffinLim(
            n_fft=n_fft,
            n_iter=64,
            win_length=1024,
            hop_length=256,
            power=1.0
        )

        audio = griffin_lim(spec_linear.unsqueeze(0))  # [1, T_audio]
        audio_np = audio.squeeze(0).numpy()

        # Normalizare volum
        peak = np.abs(audio_np).max()
        if peak > 0:
            audio_np = audio_np / peak * 0.9

        return audio_np

    def convert(
        self,
        source_audio: str,
        target_references: List[str],
        speaker_id: Optional[int] = None,
        **kwargs
    ) -> LightVCResult:
        """
        Convertește vocea sursă în vocea din referințele target.

        Args:
            source_audio:      cale fișier audio sursă
            target_references: liste de fișiere audio referință target
            speaker_id:        index speaker (dacă ECAPA nu e disponibil)

        Returns:
            LightVCResult
        """
        if not self.is_loaded:
            raise RuntimeError("Modelul nu este încărcat. Apelați load_model() mai întâi.")

        t0 = time.time()

        # Extrage mel sursă
        mel_src, _ = self._extract_audio_features(source_audio)

        # Extrage speaker embedding din referințe target
        target_embs = []
        for ref_path in target_references[:5]:  # max 5 referințe
            _, emb = self._extract_audio_features(ref_path)
            if emb is not None:
                target_embs.append(emb)

        if target_embs:
            # Media embedding-urilor din toate referințele
            speaker_emb = torch.stack(target_embs).mean(dim=0)  # [1, 192]
        else:
            # Fallback: speaker embedding lookup
            if speaker_id is None:
                speaker_id = 0
            speaker_emb = self.model.speaker_embedding(
                torch.tensor([speaker_id], device=self.device)
            )  # [1, 192]

        # Inferență — procesare în chunck-uri dacă e prea lung
        with torch.no_grad():
            chunk_size = 256
            T = mel_src.shape[-1]

            if T <= chunk_size:
                content_code = self.model.encode_content(mel_src)
                mel_out = self.model.decode(content_code, speaker_emb)
            else:
                # Procesare chunk cu overlap
                hop = chunk_size // 2
                outputs = []
                for start in range(0, T, hop):
                    end = min(start + chunk_size, T)
                    chunk = mel_src[:, :, start:end]
                    if chunk.shape[-1] < 16:
                        break
                    # Pad dacă necesar
                    if chunk.shape[-1] < chunk_size:
                        pad = torch.zeros(1, chunk.shape[1], chunk_size - chunk.shape[-1],
                                          device=self.device)
                        chunk = torch.cat([chunk, pad], dim=-1)

                    code = self.model.encode_content(chunk)
                    out = self.model.decode(code, speaker_emb)

                    # Crop la lungimea reală (fără padding)
                    real_len = min(hop, end - start)
                    if start == 0:
                        outputs.append(out[:, :, :real_len])
                    else:
                        outputs.append(out[:, :, chunk_size - real_len:chunk_size])

                mel_out = torch.cat(outputs, dim=-1)

        # Mel → audio
        audio_np = self._mel_to_audio(mel_out.cpu())

        conversion_time = time.time() - t0

        return LightVCResult(
            converted_audio=audio_np,
            sample_rate=16000,
            conversion_time=conversion_time,
            source_path=str(source_audio),
            model_name="LightVC",
            metadata={
                "num_refs": len(target_references),
                "used_ecapa": len(target_embs) > 0,
                "mel_frames": mel_src.shape[-1]
            }
        )

    @property
    def is_trained(self) -> bool:
        return self.is_loaded

    def get_info(self) -> Dict:
        """Informații despre modelul curent."""
        return {
            "name": "LightVC",
            "full_name": "Light Voice Conversion (antrenat pe date proprii)",
            "type": "Encoder-Decoder cu Information Bottleneck",
            "encoder": "ContentEncoder (Conv1D + LSTM + Bottleneck 32-dim)",
            "speaker_encoder": "ECAPA-TDNN (SpeechBrain, frozen)",
            "decoder": "Decoder (Conv1D dilated + LSTM + PostNet)",
            "vocoder": "Griffin-Lim / HiFi-GAN",
            "training_required": True,
            "is_loaded": self.is_loaded,
            **self.checkpoint_info
        }
