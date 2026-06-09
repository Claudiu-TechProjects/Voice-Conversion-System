"""
Configurare Globală — Sistem de Conversie a Vocii
==================================================

Parametrii centralizați pentru:
- Procesare audio (sample rate, mel, etc.)
- Modele (kNN-VC, AutoVC)
- Căi fișiere
- Evaluare
"""

from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional
import torch


# =====================================================================
# CĂILE PROIECTULUI
# =====================================================================

# Directorul rădăcină al proiectului
PROJECT_ROOT = Path(__file__).parent.parent
DATASET_ROOT = PROJECT_ROOT / "dataset"
AUDIO_DATASET = DATASET_ROOT / "cv-corpus-25.0-2026-03-09" / "ro" / "clips"

# Directoare output
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
RESULTS_DIR = PROJECT_ROOT / "results"
CONVERTED_AUDIO_DIR = RESULTS_DIR / "converted"
EVALUATION_DIR = RESULTS_DIR / "evaluation"
FIGURES_DIR = RESULTS_DIR / "figures"
WEBAPP_UPLOADS = PROJECT_ROOT / "webapp" / "uploads"
WEBAPP_OUTPUTS = PROJECT_ROOT / "webapp" / "outputs"

# Creare automată directoare
for _dir in [CHECKPOINTS_DIR, RESULTS_DIR, CONVERTED_AUDIO_DIR,
             EVALUATION_DIR, FIGURES_DIR, WEBAPP_UPLOADS, WEBAPP_OUTPUTS]:
    _dir.mkdir(parents=True, exist_ok=True)


# =====================================================================
# CONFIGURARE AUDIO
# =====================================================================

@dataclass
class AudioConfig:
    """Parametri pentru procesare audio."""

    sample_rate: int = 16000
    """Sample rate standard (Hz). 16kHz este standard pentru voice processing."""

    n_fft: int = 1024
    """Dimensiunea FFT window."""

    hop_length: int = 256
    """Hop length pentru STFT (16ms la 16kHz)."""

    win_length: int = 1024
    """Window length pentru STFT."""

    n_mels: int = 80
    """Număr de mel filterbanks."""

    fmin: float = 0.0
    """Frecvența minimă mel (Hz)."""

    fmax: float = 8000.0
    """Frecvența maximă mel (Hz)."""

    max_duration: float = 15.0
    """Durata maximă a audio-ului procesat (secunde)."""

    min_duration: float = 0.5
    """Durata minimă a audio-ului acceptat (secunde)."""

    normalize: bool = True
    """Normalizare audio la [-1, 1]."""


# =====================================================================
# CONFIGURARE kNN-VC
# =====================================================================

@dataclass
class KnnVCConfig:
    """Parametri pentru modelul kNN-VC pre-antrenat."""

    hub_repo: str = "bshall/knn-vc"
    """Repository torch.hub pentru kNN-VC."""

    model_name: str = "knn_vc"
    """Numele modelului din hub."""

    prematched: bool = True
    """Folosește vocoder antrenat pe date prematched (calitate mai bună)."""

    topk: int = 4
    """Numărul de vecini k pentru kNN matching. 
    4 oferă cel mai bun trade-off calitate/identitate speaker."""

    device: str = "auto"
    """Device: 'auto', 'cuda', 'cpu'."""

    trust_repo: bool = True
    """Trust torch.hub repository."""


# =====================================================================
# CONFIGURARE LightVC (MODEL ANTRENABIL)
# =====================================================================

@dataclass
class LightVCConfig:
    """Parametri pentru modelul LightVC antrenabil."""

    # Arhitectura model
    n_mels: int = 80
    """Număr de mel filterbanks (trebuie să coincidă cu AudioConfig)."""

    content_channels: int = 256
    """Canale în content encoder."""

    bottleneck_dim: int = 32
    """Dimensiunea bottleneck-ului de conținut (info-poor representation)."""

    speaker_emb_dim: int = 192
    """Dimensiunea embedding-ului de vorbitor (ECAPA-TDNN output)."""

    decoder_channels: int = 512
    """Dimensiunea decoderului."""

    # Antrenare
    batch_size: int = 16
    """Dimensiunea batch-ului pentru antrenare."""

    learning_rate: float = 1e-4
    """Learning rate Adam."""

    num_epochs: int = 300
    """Număr maxim de epoci de antrenare."""

    val_interval: int = 10
    """Evaluare pe validation set la fiecare N epoci."""

    checkpoint_interval: int = 25
    """Salvare checkpoint la fiecare N epoci."""

    # Loss weights
    lambda_recon: float = 1.0
    """Ponderea pierderii de reconstrucție mel."""

    lambda_content: float = 0.1
    """Ponderea pierderii de consistență conținut."""

    lambda_speaker: float = 0.1
    """Ponderea pierderii de similaritate speaker."""

    # Date
    n_pseudo_speakers: int = 20
    """Număr de pseudo-vorbitori pentru clustering (20-30 recomandat pentru română)."""

    max_frames: int = 256
    """Lungimea maximă a secvenței (frames)."""

    val_ratio: float = 0.1
    """Fracția din dataset pentru validare."""

    # Augmentare audio avansată
    augment_pitch: bool = True
    """Activare pitch shifting (±2 semitonuri) la antrenare."""

    augment_noise: bool = True
    """Activare noise injection (SNR 20-40dB) la antrenare."""

    augment_time_stretch: bool = True
    """Activare time stretching (±10%) la antrenare."""

    # Căi
    checkpoint_dir: str = "checkpoints/lightvc"
    """Director pentru checkpoints."""

    pseudo_speakers_file: str = "dataset/pseudo_speakers.json"
    """Fișierul JSON cu pseudo-vorbitorii."""


# =====================================================================
# CONFIGURARE FREEVC (SpeechT5) FINE-TUNING
# =====================================================================

@dataclass
class FreeVCConfig:
    """Parametri de antrenare (Fine-Tuning) pentru SpeechT5 (FreeVC)."""
    
    batch_size: int = 8
    """Batch size optim (8) pentru a folosi la maxim placa fără să dăm crash de VRAM."""
    
    gradient_accumulation_steps: int = 2
    """Acumulare parțială pentru a păstra pasul de învățare la 16."""
    
    learning_rate: float = 2e-6
    """Learning rate extrem de mic (2e-6) pentru fine-tuning stabil fără warmup."""
    
    num_epochs: int = 50
    """Număr de epoci recomandat pentru ajustare pe limba română."""
    
    checkpoint_dir: str = "checkpoints/freevc_finetuned"
    """Director pentru salvarea noului model adaptat."""
    
    log_interval: int = 10
    """La câți pași să logheze progresul."""


# =====================================================================
# CONFIGURARE XTTS v2 FINE-TUNING PE ROMÂNĂ
# =====================================================================

@dataclass
class XTTSFineTuneConfig:
    """Parametri de fine-tuning pentru XTTS v2 pe Common Voice RO."""
    
    batch_size: int = 4
    """Batch size mic pentru a evita OOM pe CPU/GPU limitat."""
    
    learning_rate: float = 5e-6
    """Learning rate foarte mic pentru fine-tuning stabil."""
    
    num_epochs: int = 5
    """Epoci de antrenare (5-10 sunt suficiente pentru adaptare fonetică)."""
    
    max_clips: int = 1000
    """Număr maxim de clipuri curate din Common Voice RO."""
    
    min_duration: float = 2.0
    """Durata minimă a clipurilor acceptate (secunde)."""
    
    max_duration: float = 15.0
    """Durata maximă a clipurilor acceptate (secunde)."""
    
    min_snr: float = 15.0
    """SNR minim acceptat (dB) pentru filtrare calitate."""
    
    target_sr: int = 22050
    """Sample rate de ieșire pentru clipurile curate."""
    
    checkpoint_dir: str = "checkpoints/xtts_ro_finetuned"
    """Director pentru salvarea modelului XTTS fine-tuned."""
    
    clean_dataset_dir: str = "dataset/xtts_clean_ro"
    """Director pentru clipurile curate (format LJSpeech)."""
    
    log_interval: int = 5
    """La câți pași să logheze progresul."""


# =====================================================================
# CONFIGURARE RVC (Retrieval-based Voice Conversion)
# =====================================================================

@dataclass
class RVCConfig:
    """Parametri pentru RVC — post-procesare timbru vocal per-speaker."""
    
    f0_method: str = "rmvpe"
    """Metoda de extracție pitch: 'rmvpe' (cel mai precis), 'harvest', 'crepe'."""
    
    index_rate: float = 0.75
    """Rata de indexare (0-1). Mai mare = mai aproape de vocea țintă."""
    
    filter_radius: int = 3
    """Rază filtrare median pentru pitch (1-7). Mai mare = mai neted."""
    
    rms_mix_rate: float = 0.25
    """Mix între volumul original și cel convertit (0-1)."""
    
    protect: float = 0.33
    """Protecție consoane (0-0.5). Mai mare = consoane mai clare."""
    
    checkpoint_dir: str = "checkpoints/rvc_speakers"
    """Director rădăcină pentru modelele RVC per-speaker."""
    
    training_epochs: int = 200
    """Epoci de antrenare RVC per speaker."""
    
    training_batch_size: int = 8
    """Batch size pentru antrenarea RVC."""


# =====================================================================
# CONFIGURARE YourTTS (Fine-Tuning pe Română)
# =====================================================================

@dataclass
class YourTTSConfig:
    """Parametri pentru fine-tuning YourTTS pe Common Voice RO."""
    
    batch_size: int = 4
    """Batch size mic pentru a evita OOM pe CPU."""
    
    eval_batch_size: int = 2
    """Batch size pentru evaluare."""
    
    learning_rate: float = 1e-4
    """Learning rate pentru fine-tuning."""
    
    num_epochs: int = 100
    """Epoci de antrenare (100 pentru adaptare bună pe română)."""
    
    max_speakers: int = 20
    """Număr maxim de vorbitori din corpus (top 20 după nr. clipuri)."""
    
    max_clips_per_speaker: int = 300
    """Număr maxim de clipuri per vorbitor (echilibrare dataset)."""
    
    min_clip_duration: float = 2.0
    """Durata minimă a unui clip (2s — sub aceasta alignmentul eșuează)."""
    
    max_clip_duration: float = 12.0
    """Durata maximă (12s — clipurile lungi consumă VRAM)."""
    
    sample_rate: int = 22050
    """Sample rate nativ VITS (22050 Hz — NU 16000!)."""
    
    eval_interval: int = 10
    """Evaluare pe validation set la fiecare N epoci."""
    
    checkpoint_interval: int = 10
    """Salvare checkpoint la fiecare N epoci."""
    
    checkpoint_dir: str = "checkpoints/yourtts_ro"
    """Director pentru checkpoint-urile modelului fine-tuned."""
    
    prepared_dataset_dir: str = "dataset/yourtts_prepared"
    """Director pentru dataset-ul pregătit (WAV + metadata)."""
    
    log_interval: int = 10
    """La câți pași să logheze progresul."""



# =====================================================================
# CONFIGURARE EVALUARE
# =====================================================================

@dataclass
class EvaluationConfig:
    """Parametri pentru sistemul de evaluare."""

    compute_mcd: bool = True
    """Calcul Mel Cepstral Distortion."""

    compute_pesq: bool = True
    """Calcul PESQ (Perceptual Evaluation of Speech Quality)."""

    compute_speaker_sim: bool = True
    """Calcul similaritate speaker (cosine similarity ECAPA-TDNN)."""

    compute_f0: bool = True
    """Calcul metrici F0 (pitch): RMSE și Pearson correlation."""

    compute_snr: bool = True
    """Calcul Signal-to-Noise Ratio."""

    n_mfcc: int = 13
    """Număr de coeficienți MFCC pentru MCD."""

    pesq_mode: str = "wb"
    """Mod PESQ: 'wb' (wideband, 16kHz) sau 'nb' (narrowband, 8kHz)."""


# =====================================================================
# INSTANȚE GLOBALE
# =====================================================================

AUDIO_CFG = AudioConfig()
KNN_VC_CFG = KnnVCConfig()
LIGHTVC_CFG = LightVCConfig()
XTTS_FT_CFG = XTTSFineTuneConfig()
RVC_CFG = RVCConfig()
YOURTTS_CFG = YourTTSConfig()
EVAL_CFG = EvaluationConfig()


def get_device(preference: str = "auto"):
    """
    Detectează cel mai bun device disponibil.
    Compatibil cu AMD ROCm și Windows DirectML (pentru AMD GPUs).
    Prioritate: CUDA > DirectML (GPU dedicat) > CPU
    """
    if preference == "auto":
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            print(f"[GPU] CUDA detectat: {device_name}")
            return "cuda"
        
        try:
            import torch_directml
            if torch_directml.is_available():
                device_count = torch_directml.device_count()
                best_device_idx = 0
                
                # Căutăm placa dedicată: conține "RX" în nume
                # (ex: "AMD Radeon RX 9070 XT" vs "AMD Radeon(TM) Graphics")
                for i in range(device_count):
                    name = torch_directml.device_name(i).upper()
                    if "RX" in name or "9070" in name or "7900" in name:
                        best_device_idx = i
                        break
                        
                device_name = torch_directml.device_name(best_device_idx).strip('\x00')
                print(f"[GPU] AMD GPU (DirectML) detectat: {device_name} (Device {best_device_idx})")
                return torch_directml.device(best_device_idx)
        except ImportError:
            pass
            
        print("[CPU] GPU indisponibil, se folosește CPU")
        return "cpu"
    return preference
