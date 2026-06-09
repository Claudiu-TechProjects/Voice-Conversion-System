import os
import json
import time
import random
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import SpeechT5Processor, SpeechT5ForSpeechToSpeech
import soundfile as sf
import librosa

from voice_conversion.config import FreeVCConfig, get_device

logger = logging.getLogger(__name__)

# =====================================================================
# DATASET PENTRU SPEECHT5 (FreeVC)
# =====================================================================

class SpeechT5VCDataset(Dataset):
    """
    Dataset pentru Fine-Tuning SpeechT5 (Voice Conversion).
    Pregătește: input_values (sursa), labels (target mel), speaker_embeddings (target).
    """
    def __init__(self, csv_path: str, wav_dir: str, max_duration: float = 4.0):
        self.processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_vc")
        self.max_duration = max_duration
        self.sample_rate = 16000
        
        # Incarcam modelul SpeechBrain pentru speaker embeddings (X-Vector de 512, obligatoriu pentru SpeechT5)
        from speechbrain.inference.speaker import EncoderClassifier
        self.spk_model = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-xvect-voxceleb",
            savedir="pretrained_models/spkrec-xvect-voxceleb",
            run_opts={"device": "cpu"} # pe CPU pentru pre-procesare sigura
        )
        
        self.samples = []
        self.speaker_samples = {}
        
        with open(csv_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                cols = line.split('|')
                if len(cols) < 3:
                    continue
                
                wav_file = Path(wav_dir) / cols[0]
                speaker_name = cols[2].strip()
                
                if wav_file.exists():
                    self.samples.append((str(wav_file), speaker_name))
                    if speaker_name not in self.speaker_samples:
                        self.speaker_samples[speaker_name] = []
                    self.speaker_samples[speaker_name].append(str(wav_file))

    def __len__(self):
        return len(self.samples)

    def _load_audio(self, path: str):
        audio, sr = librosa.load(path, sr=self.sample_rate)
        max_samples = int(self.max_duration * self.sample_rate)
        if len(audio) > max_samples:
            start = random.randint(0, len(audio) - max_samples)
            audio = audio[start:start + max_samples]
        return audio

    def _get_speaker_embedding(self, audio_array):
        waveform = torch.FloatTensor(audio_array).unsqueeze(0)
        with torch.no_grad():
            emb = self.spk_model.encode_batch(waveform)
        return emb.squeeze() # [512]

    def __getitem__(self, idx):
        src_path, src_spk_idx = self.samples[idx]
        
        # Alege target speaker diferit
        other_speakers = [k for k in self.speaker_samples.keys() if k != src_spk_idx]
        tgt_spk_idx = random.choice(other_speakers) if other_speakers else src_spk_idx
        tgt_path = random.choice(self.speaker_samples[tgt_spk_idx])
        
        # Incarca audio
        src_audio = self._load_audio(src_path)
        tgt_audio = self._load_audio(tgt_path)
        
        # Procesare cu SpeechT5Processor
        inputs = self.processor(
            audio=src_audio,
            audio_target=tgt_audio,
            sampling_rate=self.sample_rate,
            return_tensors="pt"
        )
        
        # Extragere speaker embedding din target cu X-Vector (512-dim nativ)
        spk_emb = self._get_speaker_embedding(tgt_audio)
        
        return {
            "input_values": inputs["input_values"][0],
            "labels": inputs["labels"][0],
            "speaker_embeddings": spk_emb
        }

from torch.nn.utils.rnn import pad_sequence

def collate_fn(batch):
    # Padding manual pentru lungimi variabile
    input_values = [item["input_values"] for item in batch]
    labels = [item["labels"] for item in batch]
    speaker_embeddings = [item["speaker_embeddings"] for item in batch]
    
    padded_inputs = pad_sequence(input_values, batch_first=True, padding_value=0.0)
    
    # Folosim 0.0 pentru labels, deoarece SpeechT5 le introduce în Decoder Prenet!
    # Dacă folosim -100.0, rețeaua va exploda matematic.
    padded_labels = pad_sequence(labels, batch_first=True, padding_value=0.0)
    
    # Creăm un mask boolean care este True unde NU e padding
    labels_lengths = torch.tensor([label.size(0) for label in labels])
    max_len = padded_labels.size(1)
    labels_mask = torch.arange(max_len).expand(len(labels), max_len) < labels_lengths.unsqueeze(1)
    
    # Creăm attention_mask pentru input_values (encoder)
    input_lengths = torch.tensor([x.size(0) for x in input_values])
    max_input_len = padded_inputs.size(1)
    attention_mask = torch.arange(max_input_len).expand(len(input_values), max_input_len) < input_lengths.unsqueeze(1)
    
    return {
        "input_values": padded_inputs,
        "attention_mask": attention_mask.long(),
        "labels": padded_labels,
        "labels_mask": labels_mask,
        "speaker_embeddings": torch.stack(speaker_embeddings)
    }

# =====================================================================
# ANTRENOR (CUSTOM TRAINING LOOP)
# =====================================================================

class FreeVCTrainer:
    def __init__(self):
        self.cfg = FreeVCConfig()
        self.device = get_device("auto")
        self.checkpoint_dir = Path(self.cfg.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.progress_file = self.checkpoint_dir / "training_progress.json"

    def _save_progress(self, progress_dict: dict):
        with open(self.progress_file, "w") as f:
            json.dump(progress_dict, f, indent=4)

    def train(self):
        logger.info(f"Start Fine-Tuning FreeVC (SpeechT5) pe device: {self.device}")
        
        self._save_progress({"status": "running", "message": "Încărcare dataset..."})
        
        try:
            # Folosim setul de date procesat anterior (pentru YourTTS/CV-Corpus)
            from voice_conversion.config import YOURTTS_CFG
            prepared_dir = Path(YOURTTS_CFG.prepared_dataset_dir)
            csv_path = prepared_dir / "metadata_train.csv"
            wav_dir = prepared_dir / "wavs"
            
            if not csv_path.exists():
                raise FileNotFoundError(f"Nu s-a găsit setul de date {csv_path}. Rulați întâi pregătirea YourTTS.")

            dataset = SpeechT5VCDataset(csv_path=str(csv_path), wav_dir=str(wav_dir))
            
            if len(dataset) == 0:
                raise ValueError(f"Dataset gol! Verificați {csv_path}.")

            dataloader = DataLoader(
                dataset, 
                batch_size=self.cfg.batch_size, 
                shuffle=True, 
                collate_fn=collate_fn
            )
            
            logger.info(f"Încărcare model SpeechT5ForSpeechToSpeech... ({len(dataset)} mostre)")
            model = SpeechT5ForSpeechToSpeech.from_pretrained("microsoft/speecht5_vc").to(self.device)
            
            # EXTREM DE IMPORTANT: Înghețăm extractorul de trăsături pentru a preveni explozia gradienților (Loss = NaN)
            model.freeze_feature_encoder()
            
            # Înghețăm și componentele care generează incompatibilități pe DirectML (pos_conv_embed)
            for param in model.speecht5.encoder.prenet.parameters():
                param.requires_grad = False
                
            model.train()
            
            optimizer = torch.optim.AdamW(model.parameters(), lr=self.cfg.learning_rate)
            
            start_time = time.time()
            best_loss = float('inf')
            
            for epoch in range(1, self.cfg.num_epochs + 1):
                epoch_loss = 0.0
                steps = 0
                
                for batch_idx, batch in enumerate(dataloader):
                    input_values = batch["input_values"].to(self.device)
                    attention_mask = batch["attention_mask"].to(self.device)
                    labels = batch["labels"].to(self.device)
                    labels_mask = batch["labels_mask"].to(self.device)
                    speaker_embeddings = batch["speaker_embeddings"].to(self.device)
                    
                    # Activăm autocast pentru a spori stabilitatea dacă suntem pe GPU
                    with torch.amp.autocast('cuda' if 'cuda' in str(self.device) else 'cpu', enabled=False):
                        outputs = model(
                            input_values=input_values,
                            speaker_embeddings=speaker_embeddings,
                            labels=labels
                        )
                        
                        spectrogram_pred = outputs.spectrogram
                    
                    # Evităm crash dacă dimensiunile diferă cu 1 cadru
                    seq_len = min(spectrogram_pred.size(1), labels.size(1))
                    mask_len = min(seq_len, labels_mask.size(1))
                    
                    spectrogram_pred = spectrogram_pred[:, :mask_len, :]
                    labels_matched = labels[:, :mask_len, :]
                    labels_mask = labels_mask[:, :mask_len]
                    
                    # Dacă toate cadrele sunt padding dintr-un motiv anume, sărim
                    if not labels_mask.any():
                        logger.warning("Batch cu mască goală, se sare.")
                        continue
                        
                    # Aplicăm masca corect
                    mask = labels_mask.unsqueeze(-1).expand(-1, -1, 80)
                    
                    loss = torch.nn.functional.l1_loss(
                        spectrogram_pred[mask], 
                        labels_matched[mask]
                    )
                    
                    # Gradient Accumulation
                    loss = loss / self.cfg.gradient_accumulation_steps
                    loss.backward()
                    
                    if (batch_idx + 1) % self.cfg.gradient_accumulation_steps == 0:
                        # Tăiem gradienții prea mari ca să prevenim Loss = NaN (Explozia gradienților)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        optimizer.step()
                        optimizer.zero_grad()
                        
                    loss_val = loss.item() * self.cfg.gradient_accumulation_steps
                    if torch.isnan(torch.tensor(loss_val)):
                        loss_val = 999.0 # Evitam crash la JSON
                        
                    epoch_loss += loss_val
                    steps += 1
                    
                    if steps % self.cfg.log_interval == 0:
                        logger.info(f"Epoch {epoch} | Step {steps}/{len(dataloader)} | Loss: {loss_val:.4f}")
                
                avg_loss = epoch_loss / max(1, steps)
                elapsed_hours = (time.time() - start_time) / 3600
                
                logger.info(f"--- Epoch {epoch} completă. Avg Loss: {avg_loss:.4f} ---")
                
                # Curățăm nan pentru JSON
                safe_avg_loss = float(avg_loss) if not torch.isnan(torch.tensor(avg_loss)) else 999.0
                
                self._save_progress({
                    "status": "running",
                    "epoch": epoch,
                    "total_epochs": self.cfg.num_epochs,
                    "avg_loss": safe_avg_loss,
                    "elapsed_hours": elapsed_hours
                })
                
                if safe_avg_loss < best_loss and safe_avg_loss != 999.0:
                    best_loss = avg_loss
                    # Mutam temporar modelul pe CPU pentru salvare, 
                    # evitand bug-ul safetensors cu DirectML (OpaqueTensorImpl)
                    model.to("cpu")
                    model.save_pretrained(self.checkpoint_dir, safe_serialization=False)
                    model.to(self.device)
                    
                    logger.info(f"Model salvat! Nou best loss: {best_loss:.4f}")
                    
            self._save_progress({
                "status": "finished",
                "epoch": self.cfg.num_epochs,
                "best_loss": best_loss,
                "total_time_hours": (time.time() - start_time) / 3600
            })
            logger.info("Fine-Tuning complet!")
            
        except Exception as e:
            logger.error(f"Eroare în timpul antrenării FreeVC: {e}")
            self._save_progress({"status": "error", "error": str(e)})

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    trainer = FreeVCTrainer()
    trainer.train()
