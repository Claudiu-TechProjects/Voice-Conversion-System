"""
kNN-VC (Multilingv) — Experimental
====================================

Utilizează extractorul de trăsături facebook/wav2vec2-xls-r-300m 
(care suportă limba română mult mai bine) în loc de WavLM (engleză).

NOTĂ: Vocoder-ul utilizat la final rămâne cel antrenat pe WavLM, 
deci deși matching-ul fonetic pe română va fi precis, 
timbrul generat va suna distorsionat până la atașarea unui vocoder XLS-R.
"""

import torch
import torchaudio
import numpy as np
import time
import logging
from pathlib import Path
from typing import List, Union, Optional
from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor

from voice_conversion.models.knn_vc import ConversionResult
from voice_conversion.config import CONVERTED_AUDIO_DIR

logger = logging.getLogger(__name__)

class KnnVoiceConverterMultilingual:
    def __init__(self, device: str = "cpu"):
        self.device = device
        self._loaded = False
        self.encoder = None
        self.feature_extractor = None
        self.vocoder = None
        self.model_name = "facebook/wav2vec2-xls-r-300m"
        
    def load_model(self):
        if self._loaded:
            return
            
        logger.info(f"📥 Încărcare kNN-VC Multilingual ({self.model_name})...")
        t0 = time.time()
        
        try:
            # 1. Încărcare Encoder XLS-R
            self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(self.model_name)
            self.encoder = Wav2Vec2Model.from_pretrained(self.model_name).to(self.device)
            self.encoder.eval()
            
            # 2. Încărcare Vocoder (împrumutat de la vechiul knn-vc)
            # Pentru a nu dubla codul complex de HiFi-GAN, folosim hub-ul bshall.
            # Va emite avertisment la utilizare pentru că nu e calibrat pe XLS-R.
            logger.info("   Se împrumută Vocoderul de la modelul WavLM vechi...")
            old_model = torch.hub.load("bshall/knn-vc", "knn_vc", prematched=True, trust_repo=True, device=self.device)
            self.vocoder = old_model.hifigan.eval().to(self.device)
            del old_model
            
            self._loaded = True
            logger.info(f"✅ kNN-VC Multilingual încărcat cu succes în {time.time()-t0:.1f}s pe {self.device}")
            logger.warning("   ATENȚIE: Vocoderul nu este calibrat pentru trăsăturile XLS-R! Sunetul va fi distorsionat.")
            
        except Exception as e:
            logger.error(f"Eroare la încărcarea kNN-VC Multilingual: {e}")
            raise

    def get_features(self, audio_path: Union[str, Path]) -> torch.Tensor:
        """Extrage trăsăturile XLS-R dintr-un fișier audio."""
        wav, sr = torchaudio.load(str(audio_path))
        if sr != 16000:
            wav = torchaudio.functional.resample(wav, sr, 16000)
            
        # Convertim stereo în mono
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
            
        inputs = self.feature_extractor(wav.squeeze(0).numpy(), return_tensors="pt", sampling_rate=16000)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.encoder(**inputs)
            # Luăm ultimul hidden state, care are dim=1024
            features = outputs.last_hidden_state
            
        return features # Shape: [1, T, 1024]

    def convert(
        self,
        source_audio: Union[str, Path],
        target_references: Union[str, Path, List[Union[str, Path]]],
        topk: int = 4,
        output_path: Optional[Union[str, Path]] = None
    ) -> ConversionResult:
        
        if not self._loaded:
            self.load_model()
            
        t_start = time.time()
        logger.info(f"🔄 Începe conversia multilingvă (k={topk}) pentru {Path(source_audio).name}...")
        
        if not isinstance(target_references, list):
            target_references = [target_references]
            
        # 1. Extragere trăsături Sursă
        src_feats = self.get_features(source_audio) # [1, T_src, 1024]
        
        # 2. Extragere trăsături Țintă (combinate din toate referințele)
        tgt_feats_list = []
        for ref in target_references:
            feat = self.get_features(ref)
            tgt_feats_list.append(feat)
            
        # Concatenăm frame-urile țintă pe axa timpului
        tgt_feats = torch.cat(tgt_feats_list, dim=1) # [1, T_tgt, 1024]
        
        # 3. Matching k-Nearest Neighbors
        # Squeeze dim 0
        src = src_feats.squeeze(0) # [T_src, 1024] sau [1024, T_src]?
        tgt = tgt_feats.squeeze(0) # [T_tgt, 1024]
        
        logger.info(f"   Shape SRC: {src.shape}")
        logger.info(f"   Shape TGT: {tgt.shape}")
        
        # Dacă modelul XLS-R a scos (1, T, 1024), atunci src e (T, 1024).
        # Dacă a scos (1, 1024, T), atunci e (1024, T).
        # cdist are nevoie de (N, D) și (M, D) unde D este același (adică 1024).
        if src.shape[0] == 1024 and src.shape[1] != 1024:
            src = src.transpose(0, 1)
            tgt = tgt.transpose(0, 1)
            logger.info(f"   Am transpus automat la SRC: {src.shape}")
            
        # Calculăm distanța Euclidiană între fiecare frame sursă și toate frame-urile țintă
        distances = torch.cdist(src, tgt) # [T_src, T_tgt]
        
        # Găsim cei mai apropiați top_k
        _, indices = torch.topk(distances, topk, dim=1, largest=False) # [T_src, topk]
        
        # Selectăm frame-urile țintă corespunzătoare
        # tgt[indices] are shape [T_src, topk, 1024]
        # Facem media lor pe axa topk
        matched_feats = tgt[indices].mean(dim=1) # [T_src, 1024]
        
        # Reshape pentru vocoder.
        # Vechiul vocoder aștepta features (1, 256, T) sau (1, 1024, T)?
        matched_feats = matched_feats.unsqueeze(0).transpose(1, 2)
        
        logger.info(f"   Shape matched_feats pt Vocoder: {matched_feats.shape}")
        
        # 4. Vocoder (Sinteză)
        logger.info("   Generare formă de undă prin HiFi-GAN (Distorsionat din cauza missmatch-ului)...")
        try:
            with torch.no_grad():
                wav_out = self.vocoder(matched_feats) # [1, 1, T_out]
        except Exception as e:
            logger.error(f"Eroare în Vocoder! Încercăm să dăm (1, T, 1024)...")
            try:
                with torch.no_grad():
                    wav_out = self.vocoder(matched_feats.transpose(1, 2))
            except Exception as e2:
                raise e # Throw original if both fail
            
        wav_out = wav_out.squeeze(1).cpu() # [1, T_out]
        
        conversion_time = time.time() - t_start
        logger.info(f"✅ Conversie finalizată în {conversion_time:.2f}s")
        
        result = ConversionResult(
            converted_audio=wav_out,
            sample_rate=16000,
            source_path=str(source_audio),
            target_paths=[str(p) for p in target_references],
            topk=topk,
            conversion_time=conversion_time,
            device_used=self.device
        )
        
        if output_path is None:
            output_path = result.save()
        else:
            result.save(output_path)
            
        return result
