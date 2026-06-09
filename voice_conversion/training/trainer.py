"""
LightVC Trainer — Antrenare Model
====================================

Training loop complet cu:
- Loss multitask: reconstructie + content consistency + speaker matching
- Validare periodică cu MCD
- Salvare checkpoint (best + periodic)
- Progress logging (JSON) pentru interfata web
- Suport oprire grațioasă
"""

import json
import time
import logging
import threading
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List, Callable

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


class LightVCTrainer:
    """
    Orchestrează antrenarea modelului LightVC.

    Utilizare:
        trainer = LightVCTrainer(config)
        trainer.setup(pseudo_speakers_json)
        trainer.train()
    """

    def __init__(self, config=None):
        from voice_conversion.config import LIGHTVC_CFG, PROJECT_ROOT
        self.cfg = config or LIGHTVC_CFG
        self.project_root = PROJECT_ROOT

        # Checkpoint dir
        self.checkpoint_dir = PROJECT_ROOT / self.cfg.checkpoint_dir
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Progress state (shared cu web UI)
        self.progress: Dict = {
            "status": "idle",          # idle | running | stopped | done | error
            "epoch": 0,
            "total_epochs": self.cfg.num_epochs,
            "train_loss": None,
            "val_loss": None,
            "best_val_loss": float("inf"),
            "train_loss_history": [],
            "val_loss_history": [],
            "elapsed_seconds": 0,
            "eta_seconds": None,
            "message": "Neantrenat",
            "error": None,
            "num_speakers": 0,
            "num_train_samples": 0,
        }
        self._stop_flag = threading.Event()
        self._progress_lock = threading.Lock()

        # Componente
        self.model: Optional[nn.Module] = None
        self.optimizer = None
        self.train_loader: Optional[DataLoader] = None
        self.val_loader: Optional[DataLoader] = None
        from voice_conversion.config import get_device
        self.device = get_device("auto")

    def setup(self, pseudo_speakers_path: Optional[str] = None) -> bool:
        """
        Pregătește datele și modelul pentru antrenare.

        Args:
            pseudo_speakers_path: cale JSON pseudo-vorbitori.
                                  Dacă None, îl creează automat.

        Returns:
            True dacă setup reușit, False altfel.
        """
        from voice_conversion.models.lightvc import LightVCModel
        from voice_conversion.data.speaker_clustering import SpeakerClusterer
        from voice_conversion.data.vc_dataset import create_dataloaders
        from voice_conversion.config import AUDIO_DATASET

        # --- Date ---
        if pseudo_speakers_path is None:
            pseudo_speakers_path = str(
                self.project_root / self.cfg.pseudo_speakers_file
            )

        if not Path(pseudo_speakers_path).exists():
            logger.info("Pseudo-vorbitorii nu exista. Se executa clustering...")
            self._update_progress(status="running", message="Clustering vorbitori...")

            audio_dir = str(AUDIO_DATASET)
            clusterer = SpeakerClusterer(
                audio_dir=audio_dir,
                output_path=pseudo_speakers_path
            )
            result = clusterer.run(
                n_clusters=self.cfg.n_pseudo_speakers,
                max_files=1000  # max 1000 fișiere pentru eficiență
            )
            logger.info(f"Clustering gata: {result['num_speakers']} vorbitori.")
        else:
            logger.info(f"Pseudo-vorbitori găsiți: {pseudo_speakers_path}")

        # Dataloaders
        try:
            self.train_loader, self.val_loader = create_dataloaders(
                pseudo_speakers_path=pseudo_speakers_path,
                batch_size=self.cfg.batch_size,
                val_ratio=self.cfg.val_ratio,
                max_frames=self.cfg.max_frames
            )
        except Exception as e:
            self._update_progress(status="error", error=str(e))
            logger.error(f"Eroare creare dataloaders: {e}")
            return False

        # Număr de vorbitori din JSON
        with open(pseudo_speakers_path) as f:
            ps_data = json.load(f)
        num_speakers = ps_data["num_speakers"]

        # --- Model ---
        self.model = LightVCModel(
            n_mels=self.cfg.n_mels,
            content_channels=self.cfg.content_channels,
            bottleneck_dim=self.cfg.bottleneck_dim,
            speaker_emb_dim=self.cfg.speaker_emb_dim,
            decoder_channels=self.cfg.decoder_channels,
            num_speakers=num_speakers
        ).to(self.device)

        num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info(f"Model LightVC: {num_params:,} parametri trainabili")

        # Optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.cfg.learning_rate,
            betas=(0.9, 0.999),
            weight_decay=1e-6
        )

        # Scheduler: reduce LR dacă val_loss stagnează
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=20,
            verbose=True
        )

        # Salvăm config în checkpoint pentru reproducibilitate
        self._model_config = {
            "n_mels": self.cfg.n_mels,
            "content_channels": self.cfg.content_channels,
            "bottleneck_dim": self.cfg.bottleneck_dim,
            "speaker_emb_dim": self.cfg.speaker_emb_dim,
            "decoder_channels": self.cfg.decoder_channels,
            "num_speakers": num_speakers
        }

        self._update_progress(
            num_speakers=num_speakers,
            num_train_samples=len(self.train_loader.dataset),
            message=f"Gata: {len(self.train_loader.dataset)} sample-uri, {num_speakers} vorbitori"
        )

        logger.info(f"Setup complet: {self.device}, {num_speakers} speakers")
        return True

    def _compute_loss(
        self,
        mel_pred: torch.Tensor,
        mel_tgt: torch.Tensor,
        content_code_src: torch.Tensor,
        content_code_tgt: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Calculează loss-ul multitask."""
        # L1 reconstrucție mel
        loss_recon = F.l1_loss(mel_pred, mel_tgt)

        # Content consistency: dacă avem cod din target, să fie similar
        loss_content = torch.tensor(0.0, device=self.device)
        if content_code_tgt is not None:
            loss_content = F.mse_loss(content_code_src, content_code_tgt.detach())

        total = (
            self.cfg.lambda_recon * loss_recon +
            self.cfg.lambda_content * loss_content
        )

        return {
            "total": total,
            "recon": loss_recon.item(),
            "content": loss_content.item()
        }

    def _train_epoch(self) -> float:
        """Un epoch de antrenare. Returnează media loss-ului."""
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        for batch in self.train_loader:
            if self._stop_flag.is_set():
                break

            mel_src = batch["mel_src"].to(self.device)    # [B, 80, T]
            mel_tgt = batch["mel_tgt"].to(self.device)    # [B, 80, T]
            speaker_id = batch["speaker_id"].to(self.device)  # [B]

            self.optimizer.zero_grad()

            # Forward: convertim src → stil tgt
            out = self.model(mel_src, speaker_id)
            mel_pred = out["mel_pred"]
            content_src = out["content_code"]

            # Calculăm și content code din target pentru consistency loss
            with torch.no_grad():
                out_tgt = self.model(mel_tgt, batch["src_speaker_id"].to(self.device))
                content_tgt = out_tgt["content_code"]

            losses = self._compute_loss(mel_pred, mel_tgt, content_src, content_tgt)

            losses["total"].backward()

            # Gradient clipping pentru stabilitate
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            total_loss += losses["total"].item()
            n_batches += 1

        return total_loss / max(n_batches, 1)

    def _val_epoch(self) -> float:
        """Un epoch de validare. Returnează media loss-ului."""
        self.model.eval()
        total_loss = 0.0
        n_batches = 0

        with torch.no_grad():
            for batch in self.val_loader:
                mel_src = batch["mel_src"].to(self.device)
                mel_tgt = batch["mel_tgt"].to(self.device)
                speaker_id = batch["speaker_id"].to(self.device)

                out = self.model(mel_src, speaker_id)
                mel_pred = out["mel_pred"]
                content_src = out["content_code"]

                losses = self._compute_loss(mel_pred, mel_tgt, content_src)
                total_loss += losses["total"].item()
                n_batches += 1

        return total_loss / max(n_batches, 1)

    def _save_checkpoint(self, epoch: int, val_loss: float, is_best: bool = False):
        """Salvează checkpoint."""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_loss": self.progress["best_val_loss"],
            "train_loss_history": self.progress["train_loss_history"],
            "val_loss_history": self.progress["val_loss_history"],
            "model_config": self._model_config,
            "training_time_hours": self.progress["elapsed_seconds"] / 3600
        }

        if is_best:
            path = self.checkpoint_dir / "best_model.pth"
            torch.save(checkpoint, path)
            logger.info(f"Best model salvat: {path} (val_loss={val_loss:.4f})")

        if epoch % self.cfg.checkpoint_interval == 0:
            path = self.checkpoint_dir / f"checkpoint_epoch_{epoch:04d}.pth"
            torch.save(checkpoint, path)

    def _save_progress_json(self):
        """Salvează progresul într-un JSON pentru polling din web UI."""
        progress_path = self.checkpoint_dir / "training_progress.json"
        with self._progress_lock:
            data = dict(self.progress)
        with open(progress_path, 'w') as f:
            json.dump(data, f, indent=2)

    def _update_progress(self, **kwargs):
        with self._progress_lock:
            self.progress.update(kwargs)
        self._save_progress_json()

    def train(
        self,
        num_epochs: Optional[int] = None,
        progress_callback: Optional[Callable] = None
    ):
        """
        Loop principal de antrenare.

        Args:
            num_epochs: Override număr de epoci (default din config)
            progress_callback: Funcție apelată la fiecare epocă cu dict progres
        """
        if self.model is None or self.train_loader is None:
            raise RuntimeError("Apelați setup() înainte de train()!")

        self._stop_flag.clear()
        total_epochs = num_epochs or self.cfg.num_epochs
        best_val_loss = float("inf")
        t_start = time.time()

        self._update_progress(
            status="running",
            total_epochs=total_epochs,
            message="Antrenare in curs..."
        )
        logger.info(f"Pornire antrenare: {total_epochs} epoci pe {self.device}")

        for epoch in range(1, total_epochs + 1):
            if self._stop_flag.is_set():
                logger.info("Antrenare oprita de utilizator.")
                self._update_progress(status="stopped", message="Oprit de utilizator")
                break

            # Train
            train_loss = self._train_epoch()

            # Val (periodic)
            val_loss = None
            is_best = False
            if epoch % self.cfg.val_interval == 0 or epoch == total_epochs:
                val_loss = self._val_epoch()
                self.scheduler.step(val_loss)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    is_best = True
                    self._save_checkpoint(epoch, val_loss, is_best=True)

            # Checkpoint periodic
            if epoch % self.cfg.checkpoint_interval == 0:
                self._save_checkpoint(epoch, val_loss or train_loss)

            # Calcul ETA
            elapsed = time.time() - t_start
            eta = (elapsed / epoch) * (total_epochs - epoch) if epoch > 0 else None

            # Update progress
            with self._progress_lock:
                self.progress["train_loss_history"].append(round(train_loss, 4))
                if val_loss is not None:
                    self.progress["val_loss_history"].append(round(val_loss, 4))

            self._update_progress(
                epoch=epoch,
                train_loss=round(train_loss, 4),
                val_loss=round(val_loss, 4) if val_loss else None,
                best_val_loss=round(best_val_loss, 4),
                elapsed_seconds=int(elapsed),
                eta_seconds=int(eta) if eta else None,
                message=f"Epoca {epoch}/{total_epochs} | Loss: {train_loss:.4f}"
                        + (f" | Val: {val_loss:.4f}" if val_loss else "")
                        + (" | BEST" if is_best else "")
            )

            if progress_callback:
                with self._progress_lock:
                    progress_callback(dict(self.progress))

            if epoch % 10 == 0:
                logger.info(
                    f"[{epoch:4d}/{total_epochs}] "
                    f"train={train_loss:.4f} "
                    + (f"val={val_loss:.4f} " if val_loss else "")
                    + (f"BEST " if is_best else "")
                    + f"| {elapsed/60:.1f}min"
                )

        # Final
        self._update_progress(
            status="done" if not self._stop_flag.is_set() else "stopped",
            message=f"Antrenare finalizata. Best val loss: {best_val_loss:.4f}"
        )
        logger.info(f"Antrenare terminata. Best val_loss={best_val_loss:.4f}")

    def stop(self):
        """Oprire grațioasă a antrenării."""
        self._stop_flag.set()
        logger.info("Semnal de oprire trimis.")

    def get_progress(self) -> Dict:
        """Returnează starea curentă a antrenării."""
        with self._progress_lock:
            return dict(self.progress)

    @staticmethod
    def load_progress(checkpoint_dir: str) -> Dict:
        """Încarcă progresul salvat pe disc."""
        progress_path = Path(checkpoint_dir) / "training_progress.json"
        if not progress_path.exists():
            return {"status": "idle", "message": "Nicio antrenare gasita"}
        with open(progress_path) as f:
            return json.load(f)

    @staticmethod
    def has_trained_model(checkpoint_dir: str) -> bool:
        """Verifică dacă există un model antrenat."""
        return (Path(checkpoint_dir) / "best_model.pth").exists()


# Import necesar pentru pierderi
import torch.nn.functional as F
