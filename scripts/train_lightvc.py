"""
Script CLI — Antrenare LightVC
================================

Utilizare:
    # Antrenare standard (200 epoci)
    python scripts/train_lightvc.py

    # Quick test (10 epoci)
    python scripts/train_lightvc.py --epochs 10 --quick

    # Configurare custom
    python scripts/train_lightvc.py --epochs 100 --batch-size 8 --speakers 8

    # Resume din checkpoint
    python scripts/train_lightvc.py --resume checkpoints/lightvc/checkpoint_epoch_0050.pth
"""

import sys
import time
import logging
import argparse
import threading
from pathlib import Path

# Adăugare root în sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)


def print_progress(progress: dict):
    """Afișează progres în consolă."""
    epoch = progress.get("epoch", 0)
    total = progress.get("total_epochs", 1)
    train_loss = progress.get("train_loss", 0) or 0
    val_loss = progress.get("val_loss")
    elapsed = progress.get("elapsed_seconds", 0) or 0
    eta = progress.get("eta_seconds")

    bar_len = 30
    filled = int(bar_len * epoch / max(total, 1))
    bar = "=" * filled + "-" * (bar_len - filled)

    eta_str = ""
    if eta:
        if eta > 3600:
            eta_str = f" | ETA: {eta//3600}h{(eta%3600)//60}m"
        elif eta > 60:
            eta_str = f" | ETA: {eta//60}m{eta%60}s"
        else:
            eta_str = f" | ETA: {eta}s"

    elapsed_str = f"{elapsed//60}m{elapsed%60}s"

    val_str = f" | Val: {val_loss:.4f}" if val_loss else ""
    best_str = f" (best)" if val_loss and val_loss <= progress.get("best_val_loss", 999) else ""

    print(
        f"\r[{bar}] {epoch:4d}/{total} "
        f"| Loss: {train_loss:.4f}{val_str}{best_str} "
        f"| {elapsed_str}{eta_str}  ",
        end="", flush=True
    )


def main():
    parser = argparse.ArgumentParser(
        description="Antrenare model LightVC de voice conversion"
    )
    parser.add_argument(
        "--epochs", type=int, default=None,
        help="Număr de epoci (default: din config, 200)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=None,
        help="Dimensiunea batch-ului (default: 16)"
    )
    parser.add_argument(
        "--speakers", type=int, default=None,
        help="Număr de pseudo-vorbitori pentru clustering (default: 10)"
    )
    parser.add_argument(
        "--audio-dir", type=str, default=None,
        help="Director cu fișiere WAV (default: dataset/common_voices20_audio)"
    )
    parser.add_argument(
        "--checkpoint-dir", type=str, default=None,
        help="Director pentru salvare checkpoints (default: checkpoints/lightvc)"
    )
    parser.add_argument(
        "--pseudo-speakers", type=str, default=None,
        help="Fișier JSON pseudo-vorbitori (dacă deja există)"
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Mod rapid: 10 epoci, 50 fișiere (pentru test)"
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Resume din checkpoint (cale .pth)"
    )
    parser.add_argument(
        "--lr", type=float, default=None,
        help="Learning rate (default: 1e-4)"
    )
    args = parser.parse_args()

    # --- Configurare ---
    from voice_conversion.config import LIGHTVC_CFG, PROJECT_ROOT
    from voice_conversion.training.trainer import LightVCTrainer

    cfg = LIGHTVC_CFG

    if args.epochs:
        cfg.num_epochs = args.epochs
    if args.batch_size:
        cfg.batch_size = args.batch_size
    if args.speakers:
        cfg.n_pseudo_speakers = args.speakers
    if args.checkpoint_dir:
        cfg.checkpoint_dir = args.checkpoint_dir
    if args.lr:
        cfg.learning_rate = args.lr
    if args.quick:
        cfg.num_epochs = 10
        cfg.n_pseudo_speakers = 5
        cfg.val_interval = 5
        cfg.checkpoint_interval = 10

    # --- Audio dir ---
    audio_dir = args.audio_dir or str(PROJECT_ROOT / "dataset" / "common_voices20_audio")
    pseudo_speakers_path = args.pseudo_speakers or str(PROJECT_ROOT / cfg.pseudo_speakers_file)

    print("=" * 60)
    print("  LightVC — Antrenare Model de Voice Conversion")
    print("=" * 60)
    print(f"  Epoci:         {cfg.num_epochs}")
    print(f"  Batch size:    {cfg.batch_size}")
    print(f"  Pseudo-speakeri: {cfg.n_pseudo_speakers}")
    print(f"  LR:            {cfg.learning_rate}")
    print(f"  Checkpoint dir: {cfg.checkpoint_dir}")
    print(f"  Audio dir:     {audio_dir}")
    print("=" * 60)
    print()

    # --- Setup ---
    trainer = LightVCTrainer(cfg)

    print("Configurare dataset si model...")
    ok = trainer.setup(pseudo_speakers_path=pseudo_speakers_path)
    if not ok:
        print("EROARE: Setup esuat. Verificati datele si dependentele.")
        sys.exit(1)

    progress = trainer.get_progress()
    print(f"Dataset: {progress['num_train_samples']} sample-uri, "
          f"{progress['num_speakers']} vorbitori")
    print()

    # --- Resume ---
    if args.resume:
        resume_path = Path(args.resume)
        if resume_path.exists():
            print(f"Resume din: {resume_path}")
            import torch
            checkpoint = torch.load(resume_path, map_location="cpu", weights_only=False)
            trainer.model.load_state_dict(checkpoint["model_state_dict"])
            trainer.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            print(f"  Continuat din epoca {checkpoint['epoch']}")
        else:
            print(f"AVERTISMENT: Checkpoint {resume_path} nu exista. Antrenare de la zero.")

    # --- Antrenare ---
    print(f"Pornire antrenare ({cfg.num_epochs} epoci)...")
    print("(Apasati CTRL+C pentru oprire gratiosa)\n")

    t_start = time.time()

    try:
        trainer.train(
            num_epochs=cfg.num_epochs,
            progress_callback=print_progress
        )
        print()  # newline după progress bar

    except KeyboardInterrupt:
        print("\n\nOprire gratiosa...")
        trainer.stop()
        time.sleep(1)

    # --- Sumar final ---
    total_time = time.time() - t_start
    progress = trainer.get_progress()

    print()
    print("=" * 60)
    print("  ANTRENARE TERMINATA")
    print("=" * 60)
    print(f"  Status:        {progress['status']}")
    print(f"  Epoci:         {progress['epoch']}/{cfg.num_epochs}")
    print(f"  Best val loss: {progress['best_val_loss']:.4f}")
    print(f"  Timp total:    {total_time/60:.1f} minute")
    print(f"  Checkpoint:    {cfg.checkpoint_dir}/best_model.pth")
    print()

    if progress['status'] == 'done':
        print("Modelul este gata! Porneste serverul si converteste vocea.")
        print(f"  python run_server.py")
    elif progress['status'] == 'stopped':
        print("Antrenarea a fost oprita. Poti relua cu --resume:")
        print(f"  python scripts/train_lightvc.py --resume {cfg.checkpoint_dir}/best_model.pth")

    print("=" * 60)


if __name__ == "__main__":
    main()
