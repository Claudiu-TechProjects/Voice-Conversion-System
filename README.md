# Voice Conversion 

Platformă web pentru conversia vocii folosind 4 modele de Deep Learning: **kNN-VC**, **FreeVC**, **YourTTS** și **Whisper + XTTS-v2 + RVC**, plus un modul de recunoaștere a vorbitorului bazat pe **ECAPA-TDNN**.

Aplicația permite înrolarea vorbitorilor, conversia vocii cu oricare dintre cele 4 modele, compararea simultană a rezultatelor și identificarea biometrică a vorbitorului.

## Modele

| Model | Tip | Flux |
|-------|-----|------|
| **kNN-VC** | Neparametric | WavLM-Large → k-NN matching (k=4) → HiFi-GAN vocoder |
| **FreeVC** (SpeechT5) | Encoder-Decoder | SpeechT5 encoder + X-Vector (512-dim) → SpeechT5 decoder → HiFi-GAN |
| **YourTTS** | TTS multilingv | Whisper STT → text → YourTTS (VITS) + Speaker Embedding → audio |
| **Whisper+XTTS+RVC** | Cascadă (3 modele) | Whisper STT → XTTS-v2 (GPT auto-regresiv) → RVC (HuBERT + retrieval) |
| **ECAPA-TDNN** | Recunoaștere vorbitor | Audio → embedding 192-dim → Cosine Similarity → identificare |

## Tehnologii și Framework-uri

| Categorie | Tehnologii |
|-----------|------------|
| **Backend** | Python 3.11, FastAPI, Uvicorn |
| **Frontend** | HTML, CSS, JavaScript, Web Audio API |
| **Deep Learning** | PyTorch 2.4.1, torchaudio |
| **TTS / Voice Cloning** | Coqui TTS (YourTTS, XTTS-v2), HuggingFace Transformers (SpeechT5) |
| **Speaker Embeddings** | SpeechBrain (ECAPA-TDNN, X-Vector) |
| **Speech-to-Text** | OpenAI Whisper |
| **Rafinare vocală** | rvc-python, FAISS (nearest-neighbor search), HuBERT |
| **Audio processing** | librosa, soundfile, pyworld, praat-parselmouth, torchcrepe |
| **Evaluare** | WER (Whisper), MOS-N, SMOS, Cosine Similarity, F0 RMSE |
| **Date** | Mozilla Common Voice RO (cv-corpus-25.0) |

## Antrenare

Fine-tuning-ul modelelor (YourTTS, XTTS-v2, FreeVC) folosește:

- **PyTorch 2.4.1** — framework de antrenare
- **Coqui TTS Trainer** — training loop pentru YourTTS și XTTS-v2
- **GPU NVIDIA (CUDA)** — recomandat pentru antrenare (RTX 3060+ / 6+ GB VRAM)
- **torch-directml** — suport experimental AMD GPU (inferență, antrenare limitată)
- **CPU** — fallback funcțional dar lent
- **AdamW optimizer** + gradient clipping (max_norm=1.0) + gradient accumulation
- **Dataset:** Common Voice RO — curățat cu VAD, filtru SNR, normalizare volum







# Ghid de Instalare 

## Varianta 1 — Pornire Automată 
Dublu-click pe **`start_usb.bat`**. Scriptul face totul automat:
1. Verifică dacă Python este instalat
2. Creează mediul virtual `venv_app/`
3. Instalează dependențele din `requirements_final.txt`
4. Pornește serverul pe **http://localhost:8000**
---
## Varianta 2 — Instalare Manuală
### Pas 1: Creare mediu virtual
```bash
python -m venv sb_env
```
### Pas 2: Activare mediu virtual
```bash
.\sb_env\Scripts\activate
```
### Pas 3: Actualizare pip
```bash
python -m pip install --upgrade pip
```
### Pas 4: Instalare dependențe
```bash
pip install -r requirements_final.txt
```
> **Notă:** Instalarea durează câteva minute (190+ pachete, inclusiv PyTorch ~2 GB).
### Pas 5: Pornire server
```bash
python -m uvicorn webapp.backend.app:app --host 0.0.0.0 --port 8000
```
### Pas 6: Deschide în browser
```
http://localhost:8000
```
---
## Prima Rulare
La prima utilizare a fiecărui model, acesta se descarcă automat:
| Model | Dimensiune | Sursă |
|-------|-----------|-------|
| kNN-VC (WavLM + HiFi-GAN) | ~1.3 GB | torch.hub |
| FreeVC (SpeechT5) | ~500 MB | HuggingFace |
| YourTTS | ~500 MB | Coqui TTS |
| XTTS-v2 | ~1.8 GB | Coqui TTS |
| Whisper (base) | ~140 MB | OpenAI |
| ECAPA-TDNN | ~80 MB | SpeechBrain |
Modelele se salvează local și nu mai trebuie descărcate a doua oară.
---
## Structura Minimă Necesară
```
lucru1/
├── webapp/
│   ├── backend/app.py
│   ├── frontend/
│   ├── uploads/
│   └── outputs/
├── voice_conversion/
├── ffmpeg.exe
├── requirements_final.txt
└── start_usb.bat
```
> Folderele `uploads/`, `outputs/`, `checkpoints/` și `pretrained_models/` se creează automat dacă nu există.
