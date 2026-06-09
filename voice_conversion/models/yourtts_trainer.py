"""
YourTTS Trainer — Fine-Tuning pe Common Voice RO (Optimizat)
=============================================================

Antrenează modelul YourTTS (VITS multi-speaker) pe corpusul
Common Voice românesc. Folosește Coqui TTS Trainer oficial.

Pipeline:
1. Pregătire dataset (VAD trim + normalizare + text cleaning)
2. Configurare VITS cu caractere românești (ă, â, î, ș, ț)
3. Fine-tuning cu Coqui Trainer
4. Salvare checkpoint
"""

import os
import time
import json
import logging
import threading
from pathlib import Path
from typing import Optional, Dict

logger = logging.getLogger(__name__)

# Progres global accesibil din API
_yourtts_training_progress = {
    "status": "idle",
    "epoch": 0,
    "total_epochs": 0,
    "message": "",
    "elapsed_hours": 0,
    "progress_pct": 0,
    "loss": None,
    "loss_history": []
}
_yourtts_training_lock = threading.Lock()


def get_yourtts_training_progress() -> dict:
    """Returnează progresul curent al antrenării YourTTS."""
    with _yourtts_training_lock:
        return dict(_yourtts_training_progress)


def _update_progress(**kwargs):
    """Actualizează progresul intern."""
    with _yourtts_training_lock:
        if "loss" in kwargs and kwargs["loss"] is not None:
            _yourtts_training_progress["loss_history"].append(kwargs["loss"])
        _yourtts_training_progress.update(kwargs)


# =====================================================================
# CARACTERE ROMÂNEȘTI COMPLETE
# =====================================================================

# Setul complet de caractere pentru limba română (lowercase — textul e lowercased la curățare)
ROMANIAN_CHARACTERS = (
    "abcdefghijklmnopqrstuvwxyz"  # ASCII de bază
    "ăâîșț"                       # Diacritice românești lowercase
)

ROMANIAN_PUNCTUATIONS = "!\"'(),-.:;? "


# =====================================================================
# FORMATTER CUSTOM PENTRU DATASET-UL NOSTRU
# =====================================================================

def yourtts_ro_formatter(root_path, meta_file, **kwargs):
    """
    Formatter custom pentru dataset-ul nostru.
    Format CSV: wav_filename|text|speaker_name
    """
    txt_file = os.path.join(root_path, meta_file)
    items = []
    with open(txt_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            cols = line.split("|")
            if len(cols) < 3:
                continue
            wav_file = os.path.join(root_path, "wavs", cols[0])
            text = cols[1].strip()
            speaker_name = cols[2].strip()
            
            if not os.path.exists(wav_file):
                continue
            
            items.append({
                "text": text,
                "audio_file": wav_file,
                "speaker_name": speaker_name,
                "root_path": root_path
            })
    return items


class YourTTSTrainer:
    """
    Antrenează YourTTS pe Common Voice RO cu Coqui TTS Trainer.
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
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.prepared_dir = self.project_root / self.cfg.prepared_dataset_dir
    
    def train(self) -> Dict:
        """Pipeline complet de antrenare YourTTS."""
        t_start = time.time()
        
        try:
            _update_progress(
                status="preparing",
                message="Pregătire dataset (VAD trim + normalizare + text cleaning)...",
                epoch=0,
                total_epochs=self.cfg.num_epochs,
                progress_pct=0
            )
            
            # 1. Pregătire dataset dacă nu există
            if not (self.prepared_dir / "metadata_train.csv").exists():
                logger.info("Dataset nepregătit, rulez prepararea optimizată...")
                from voice_conversion.models.yourtts_dataset import YourTTSDatasetPreparer
                preparer = YourTTSDatasetPreparer(self.cfg)
                preparer.project_root = self.project_root
                stats = preparer.prepare()
                logger.info(f"Dataset pregătit: {stats}")
            else:
                logger.info("Dataset deja pregătit, se sare prepararea.")
                with open(self.prepared_dir / "stats.json", 'r') as f:
                    stats = json.load(f)
            
            _update_progress(
                status="training",
                message="Configurare model VITS cu caractere românești...",
                progress_pct=5
            )
            
            # 2. Antrenare reală
            self._run_coqui_training(t_start)
            
            elapsed = (time.time() - t_start) / 3600
            
            _update_progress(
                status="done",
                epoch=self.cfg.num_epochs,
                total_epochs=self.cfg.num_epochs,
                progress_pct=100,
                elapsed_hours=round(elapsed, 2),
                message=f"YourTTS fine-tuned cu succes în {elapsed:.2f}h"
            )
            
            return {
                "status": "done",
                "elapsed_hours": round(elapsed, 2),
                "checkpoint_dir": str(self.checkpoint_dir)
            }
            
        except Exception as e:
            logger.error(f"Eroare antrenare YourTTS: {e}", exc_info=True)
            _update_progress(status="error", message=f"Eroare: {str(e)}")
            raise
    
    def _run_coqui_training(self, t_start: float):
        """Antrenare VITS cu Coqui TTS Trainer oficial."""
        import torch
        from TTS.tts.configs.vits_config import VitsConfig
        from TTS.tts.configs.shared_configs import CharactersConfig, BaseDatasetConfig, BaseAudioConfig
        from TTS.tts.models.vits import Vits
        from TTS.tts.datasets import load_tts_samples
        from trainer import Trainer, TrainerArgs
        
        # Citire speaker map
        with open(self.prepared_dir / "speaker_map.json", 'r') as f:
            speaker_map = json.load(f)
        
        num_speakers = len(speaker_map)
        logger.info(f"Fine-tuning VITS cu {num_speakers} vorbitori și caractere românești")
        
        # ──────────────────────────────────────────────
        # Configurare caractere românești
        # ──────────────────────────────────────────────
        characters_config = CharactersConfig(
            characters=ROMANIAN_CHARACTERS,
            punctuations=ROMANIAN_PUNCTUATIONS,
            pad="<PAD>",
            eos="<EOS>",
            bos="<BOS>",
            blank="<BLNK>",
            is_unique=True,
            is_sorted=True,
        )
        
        # ──────────────────────────────────────────────
        # Configurare dataset
        # ──────────────────────────────────────────────
        dataset_config = BaseDatasetConfig(
            formatter="",  # vom folosi formatter-ul nostru custom
            meta_file_train="metadata_train.csv",
            meta_file_val="metadata_val.csv",
            path=str(self.prepared_dir) + "/",
            language="ro",
        )

        # Audio config cu sample_rate nativ VITS = 22050
        audio_config = BaseAudioConfig(
            sample_rate=self.cfg.sample_rate,  # 22050
            win_length=1024,
            hop_length=256,
            num_mels=80,
            fft_size=1024,
            mel_fmin=0,
            mel_fmax=None,
        )
        
        # ──────────────────────────────────────────────
        # Configurare model VITS
        # ──────────────────────────────────────────────
        config = VitsConfig(
            output_path=str(self.checkpoint_dir),
            run_name="yourtts_ro",
            
            # Audio
            audio=audio_config,
            
            # Caractere românești
            characters=characters_config,
            
            # Multi-speaker
            use_speaker_embedding=True,
            num_speakers=num_speakers,
            
            # Limbă
            use_language_embedding=False,
            
            # Antrenare
            batch_size=self.cfg.batch_size,
            eval_batch_size=self.cfg.eval_batch_size,
            num_loader_workers=0,  # Windows compatibility
            num_eval_loader_workers=0,
            
            # Epochs & learning rate
            epochs=self.cfg.num_epochs,
            lr_disc=self.cfg.learning_rate,
            lr_gen=self.cfg.learning_rate,
            
            # Logging
            print_step=25,
            print_eval=True,
            save_step=500,
            save_checkpoints=True,
            save_all_best=True,
            
            # Dataset
            datasets=[dataset_config],
            
            # Text processing
            text_cleaner="basic_cleaners",
            use_phonemes=False,  # Fără phonemizer deocamdată (espeak-ng nu e instalat)
        )
        
        # ──────────────────────────────────────────────
        # Detectare device (AMD GPU DirectML)
        # ──────────────────────────────────────────────
        device_name = "CPU"
        use_cuda = False
        
        if torch.cuda.is_available():
            use_cuda = True
            device_name = torch.cuda.get_device_name(0)
        else:
            try:
                import torch_directml
                if torch_directml.is_available():
                    for i in range(torch_directml.device_count()):
                        name = torch_directml.device_name(i).upper()
                        if "RX" in name or "9070" in name:
                            device_name = torch_directml.device_name(i).strip('\x00')
                            break
                    logger.info(f"GPU AMD detectat: {device_name}")
                    logger.info("Nota: Coqui Trainer folosește CPU pe Windows AMD. "
                                "GPU va fi folosit pentru operațiuni individuale.")
            except ImportError:
                pass
        
        logger.info(f"Device antrenare: {device_name} (CUDA: {use_cuda})")
        _update_progress(message=f"Antrenare pe {device_name}...")
        
        # ──────────────────────────────────────────────
        # Încărcare date cu formatter custom
        # ──────────────────────────────────────────────
        train_samples, eval_samples = load_tts_samples(
            dataset_config,
            formatter=yourtts_ro_formatter,
            eval_split=True,
            eval_split_max_size=200,
            eval_split_size=0.1,
        )
        
        logger.info(f"Train samples: {len(train_samples)}, Eval samples: {len(eval_samples)}")
        
        if not train_samples:
            raise ValueError("Nu s-au încărcat date de antrenare! Verifică metadata_train.csv")
        
        # ──────────────────────────────────────────────
        # Inițializare model
        # ──────────────────────────────────────────────
        model = Vits.init_from_config(config)
        
        # ──────────────────────────────────────────────
        # Trainer Args
        # ──────────────────────────────────────────────
        trainer_args = TrainerArgs(
            restore_path=None,
            skip_train_epoch=False,
            start_with_eval=False,
            use_accelerate=False,
        )
        
        # ──────────────────────────────────────────────
        # Callback pentru progres
        # ──────────────────────────────────────────────
        class ProgressCallback:
            """Actualizează progresul pentru UI."""
            def __init__(self, total_epochs, t_start):
                self.total_epochs = total_epochs
                self.t_start = t_start
                self.current_epoch = 0
            
            def on_epoch_end(self, trainer_obj):
                self.current_epoch += 1
                elapsed = (time.time() - self.t_start) / 3600
                pct = int(5 + (self.current_epoch / self.total_epochs) * 90)
                
                # Extrage loss dacă disponibil
                loss_val = None
                if hasattr(trainer_obj, 'keep_avg_train') and trainer_obj.keep_avg_train:
                    avg = trainer_obj.keep_avg_train
                    if hasattr(avg, 'avg_values'):
                        loss_val = avg.avg_values.get('loss', None)
                
                loss_str = f"{loss_val:.4f}" if loss_val is not None else "N/A"
                
                _update_progress(
                    epoch=self.current_epoch,
                    total_epochs=self.total_epochs,
                    progress_pct=pct,
                    loss=loss_val,
                    elapsed_hours=round(elapsed, 2),
                    message=(
                        f"Epoca {self.current_epoch}/{self.total_epochs} — "
                        f"Loss: {loss_str} — "
                        f"Timp: {elapsed:.2f}h"
                    )
                )
        
        progress_cb = ProgressCallback(self.cfg.num_epochs, t_start)
        
        # ──────────────────────────────────────────────
        # Pornire antrenare
        # ──────────────────────────────────────────────
        logger.info("=" * 60)
        logger.info("PORNIRE ANTRENARE VITS (YourTTS) PE ROMÂNĂ")
        logger.info(f"  Vorbitori: {num_speakers}")
        logger.info(f"  Caractere: {ROMANIAN_CHARACTERS}")
        logger.info(f"  Train: {len(train_samples)} | Eval: {len(eval_samples)}")
        logger.info(f"  Epoci: {self.cfg.num_epochs} | Batch: {self.cfg.batch_size}")
        logger.info(f"  Sample Rate: {self.cfg.sample_rate} Hz")
        logger.info(f"  Device: {device_name}")
        logger.info(f"  Output: {self.checkpoint_dir}")
        logger.info("=" * 60)
        
        _update_progress(
            status="training",
            message="Antrenare VITS în desfășurare...",
            progress_pct=5
        )
        
        trainer = Trainer(
            trainer_args,
            config,
            output_path=str(self.checkpoint_dir),
            model=model,
            train_samples=train_samples,
            eval_samples=eval_samples,
            gpu=(0 if use_cuda else None),
        )
        
        # Hook pentru progres în timp real la fiecare X pași
        original_train_step = trainer.train_step
        
        def custom_train_step(batch, batch_n_steps, step, loader_start_time):
            # Execută pasul real
            ret = original_train_step(batch, batch_n_steps, step, loader_start_time)
            
            # Actualizare UI la fiecare 20 de pași (pentru grafic în timp real)
            if step % 20 == 0:
                elapsed = (time.time() - t_start) / 3600
                current_ep = trainer.epochs_done
                # Prevenim împărțirea la 0 dacă total_epochs e 0, deși nu e cazul
                pct = int(5 + (current_ep / max(1, self.cfg.num_epochs)) * 90)
                
                loss_val = None
                loss_dict = ret[1] if isinstance(ret, tuple) and len(ret) > 1 else {}
                
                if loss_dict:
                    # YourTTS (VITS) folosește 2 loss-uri: loss_0 (Discriminator), loss_1 (Generator)
                    if 'loss_1' in loss_dict:
                        loss_val = float(loss_dict['loss_1'])
                    elif 'loss' in loss_dict:
                        loss_val = float(loss_dict['loss'])
                
                loss_str = f"{loss_val:.4f}" if loss_val is not None else "N/A"
                
                _update_progress(
                    epoch=current_ep,
                    total_epochs=self.cfg.num_epochs,
                    progress_pct=pct,
                    loss=loss_val,
                    elapsed_hours=round(elapsed, 2),
                    message=(
                        f"Epoca {current_ep}/{self.cfg.num_epochs} (Pas {step}/{batch_n_steps}) — "
                        f"Loss: {loss_str} — "
                        f"Timp: {elapsed:.2f}h"
                    )
                )
            return ret
            
        trainer.train_step = custom_train_step
        
        try:
            trainer.fit()
        except KeyboardInterrupt:
            logger.info("Antrenare oprită de utilizator.")
            _update_progress(status="done", message="Antrenare oprită manual.")
        finally:
            progress_cb.on_epoch_end(trainer)
        
        logger.info("Antrenare YourTTS completă!")
