"""
Web Application Backend — Voice Conversion System (v2)
=======================================================

"""

import os
import sys

try:
    import imageio_ffmpeg
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    ffmpeg_dir = os.path.dirname(ffmpeg_exe)
    os.environ["PATH"] = f"{ffmpeg_dir}{os.pathsep}{os.environ.get('PATH', '')}"
except ImportError:
    pass

import uuid
import time
import json
import math
import asyncio
import logging
import threading
from pathlib import Path
from typing import List, Optional
from datetime import datetime

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import torch

# Path setup
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from voice_conversion.models.knn_vc import KnnVoiceConverter, ConversionResult
from voice_conversion.models.knn_vc_multilingual import KnnVoiceConverterMultilingual
from voice_conversion.models.lightvc import LightVCConverter
from voice_conversion.models.freevc import FreeVCConverter
from voice_conversion.models.xtts_v2 import XTTSModel
from voice_conversion.evaluation.metrics import compute_all_metrics
from voice_conversion.evaluation.stt import WhisperSTT
from voice_conversion.utils.audio_utils import get_audio_info, load_audio, save_audio
from voice_conversion.config import WEBAPP_UPLOADS, WEBAPP_OUTPUTS, PROJECT_ROOT, LIGHTVC_CFG

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def safe_float(value, default=0.0):
    """Sanitizează float pentru JSON (inf/nan -> default)."""
    if value is None:
        return default
    try:
        f = float(value)
        if math.isnan(f) or math.isinf(f):
            return default
        return round(f, 4)
    except (TypeError, ValueError):
        return default


def convert_to_wav(input_path: Path) -> Path:
    """
    Convertește orice format audio suportat la WAV 16kHz mono.
    Funcționează cu: WebM, M4A, MP3, OGG, FLAC, etc.
    """
    output_path = input_path.with_suffix('.wav')
    if input_path.suffix.lower() == '.wav':
        return input_path


    try:
        from pydub import AudioSegment
        audio = AudioSegment.from_file(str(input_path))
        audio = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)
        audio.export(str(output_path), format='wav')
        logger.info(f"Convertit {input_path.suffix} -> WAV cu pydub")
        return output_path
    except Exception as e:
        logger.debug(f"pydub nu a reușit: {e}")

    try:
        import torchaudio
        waveform, sr = torchaudio.load(str(input_path))
        if sr != 16000:
            waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        torchaudio.save(str(output_path), waveform, 16000)
        logger.info(f"Convertit {input_path.suffix} -> WAV cu torchaudio")
        return output_path
    except Exception as e:
        logger.debug(f"torchaudio nu a reușit: {e}")
    try:
        import soundfile as sf
        import numpy as np
        data, sr = sf.read(str(input_path))
        if data.ndim > 1:
            data = data.mean(axis=1)
        if sr != 16000:
            import torchaudio
            import torch
            t = torch.FloatTensor(data).unsqueeze(0)
            t = torchaudio.transforms.Resample(sr, 16000)(t)
            data = t.squeeze().numpy()
        sf.write(str(output_path), data, 16000)
        logger.info(f"Convertit {input_path.suffix} -> WAV cu soundfile")
        return output_path
    except Exception as e:
        logger.warning(f"Nicio metodă de conversie nu a funcționat pentru {input_path.suffix}: {e}")
        return input_path  # returnează originalul, poate merge direct

# =====================================================================
# INIȚIALIZARE APP
# =====================================================================

app = FastAPI(
    title="Voice Conversion System",
    description="",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

FRONTEND_DIR = Path(__file__).parent.parent / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

WEBAPP_UPLOADS.mkdir(parents=True, exist_ok=True)
WEBAPP_OUTPUTS.mkdir(parents=True, exist_ok=True)
SPEAKERS_DIR = WEBAPP_UPLOADS / "speakers"
SPEAKERS_DIR.mkdir(parents=True, exist_ok=True)

# =====================================================================
# STATE GLOBAL
# =====================================================================

# Modele (lazy loading)
knn_converter: Optional[KnnVoiceConverter] = None
#mknn_converter: Optional[KnnVoiceConverterMultilingual] = None
#lightvc_converter: Optional[LightVCConverter] = None
freevc_converter: Optional[FreeVCConverter] = None
whisper_stt: Optional[WhisperSTT] = None
tts_model: Optional[XTTSModel] = None
speaker_recognizer = None  # SpeechBrain ECAPA-TDNN (lazy)

# Speakers DB
speakers_db: dict = {}
SPEAKERS_DB_PATH = SPEAKERS_DIR / "speakers_db.json"

# Istoric persistent
conversion_history: List[dict] = []
HISTORY_PATH = WEBAPP_UPLOADS / "conversion_history.json"

# Trainer (pentru antrenare LightVC)
#_trainer = None
#_trainer_thread: Optional[threading.Thread] = None

#LIGHTVC_CHECKPOINT_DIR = PROJECT_ROOT / LIGHTVC_CFG.checkpoint_dir


def get_knn_converter() -> KnnVoiceConverter:
    global knn_converter
    if knn_converter is None:
        logger.info("Initializare kNN-VC...")
        knn_converter = KnnVoiceConverter(device="auto")
        knn_converter.load_model()
    return knn_converter


#def get_mknn_converter() -> KnnVoiceConverterMultilingual:
#    global mknn_converter
#    if mknn_converter is None:
#        import torch
#        logger.info("Initializare kNN-VC Multilingual...")
#        mknn_converter = KnnVoiceConverterMultilingual(device="cuda" if torch.cuda.is_available() else "cpu")
#        mknn_converter.load_model()
#    return mknn_converter


#def get_lightvc_converter() -> LightVCConverter:
#    global lightvc_converter
#    if lightvc_converter is None:
#        lightvc_converter = LightVCConverter(device="auto")
#    if not lightvc_converter.is_loaded:
#        checkpoint = LIGHTVC_CHECKPOINT_DIR / "best_model.pth"
#        if checkpoint.exists():
#            lightvc_converter.load_model(str(checkpoint))
#    return lightvc_converter


def get_freevc_converter() -> FreeVCConverter:
    global freevc_converter
    
    from pathlib import Path
    checkpoint_dir = Path("checkpoints/freevc_finetuned")
    ckpt_file = checkpoint_dir / "pytorch_model.bin"
    if not ckpt_file.exists():
        ckpt_file = checkpoint_dir / "model.safetensors"
        
    disk_mtime = ckpt_file.stat().st_mtime if ckpt_file.exists() else 0

    if freevc_converter is not None:
        loaded_mtime = getattr(freevc_converter, "disk_mtime", 0)
        if disk_mtime != loaded_mtime:
            logger.info("Versiune nouă de model FreeVC/SpeechT5 detectată pe disc. Reîncărcare...")
            freevc_converter = None
            import gc; gc.collect()

    if freevc_converter is None:
        logger.info("Initializare FreeVC...")
        freevc_converter = FreeVCConverter(device="auto")
        freevc_converter.load_model()
        freevc_converter.disk_mtime = disk_mtime
        
    return freevc_converter


def get_whisper_stt() -> WhisperSTT:
    global whisper_stt
    if whisper_stt is None:
        whisper_stt = WhisperSTT(model_size="medium")
    return whisper_stt


def get_tts_model() -> XTTSModel:
    global tts_model
    if tts_model is None:
        import torch
        tts_model = XTTSModel(device="cuda" if torch.cuda.is_available() else "cpu")
        tts_model.load_model()
    return tts_model

rvc_converter = None
def get_rvc_converter():
    global rvc_converter
    if rvc_converter is None:
        from voice_conversion.models.rvc_converter import RVCConverter
        rvc_converter = RVCConverter()
    return rvc_converter


def get_speaker_recognizer():
    """Lazy loader pentru SpeechBrain ECAPA-TDNN speaker recognition."""
    global speaker_recognizer
    if speaker_recognizer is None:
        logger.info("Inițializare SpeechBrain ECAPA-TDNN pentru recunoaștere vorbitor...")
        from speechbrain.inference.speaker import SpeakerRecognition
        speaker_recognizer = SpeakerRecognition.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir="pretrained_models/spkrec-ecapa-voxceleb"
        )
        logger.info("SpeechBrain ECAPA-TDNN încărcat cu succes.")
    return speaker_recognizer


def load_speakers_db():
    global speakers_db
    if SPEAKERS_DB_PATH.exists():
        with open(SPEAKERS_DB_PATH, 'r', encoding='utf-8') as f:
            speakers_db = json.load(f)
    logger.info(f"Vorbitori incarcati: {len(speakers_db)}")


def save_speakers_db():
    with open(SPEAKERS_DB_PATH, 'w', encoding='utf-8') as f:
        json.dump(speakers_db, f, indent=2, ensure_ascii=False)


def load_history():
    """Încarcă istoricul conversiilor de pe disc."""
    global conversion_history
    if HISTORY_PATH.exists():
        try:
            with open(HISTORY_PATH, 'r', encoding='utf-8') as f:
                conversion_history = json.load(f)
            logger.info(f"Istoric încărcat: {len(conversion_history)} conversii")
        except Exception as e:
            logger.warning(f"Eroare la încărcarea istoricului: {e}")
            conversion_history = []


def save_history():
    """Salvează istoricul conversiilor pe disc."""
    try:
        with open(HISTORY_PATH, 'w', encoding='utf-8') as f:
            json.dump(conversion_history, f, indent=2, ensure_ascii=False, default=str)
    except Exception as e:
        logger.warning(f"Eroare la salvarea istoricului: {e}")


load_speakers_db()
load_history()


# =====================================================================
# ENDPOINT — SPEECH-TO-TEXT INDIVIDUAL
# =====================================================================

@app.post("/api/transcribe")
async def transcribe_audio(
    audio: UploadFile = File(...)
):
    """Transcrie un fișier audio folosind Whisper (independent de conversie)."""
    transcription_id = uuid.uuid4().hex[:12]
    audio_path = WEBAPP_UPLOADS / f"{transcription_id}_stt_{audio.filename}"
    
    content = await audio.read()
    with open(audio_path, 'wb') as f:
        f.write(content)

    audio_path = convert_to_wav(audio_path)

    try:
        stt = get_whisper_stt()
        result = stt.transcribe(str(audio_path))
        return {
            "success": True,
            "text": result["text"],
            "language": result["language"],
            "duration": result["duration"]
        }
    except Exception as e:
        logger.error(f"Eroare transcripție: {e}")
        raise HTTPException(500, f"Eroare STT: {str(e)}")


# =====================================================================
# ENDPOINTS — INFO & HISTORY
# =====================================================================

@app.get("/")
async def serve_frontend():
    index_path = FRONTEND_DIR / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    return {"message": "Frontend indisponibil. Accesează /docs."}


# =====================================================================
# ENDPOINTS — VORBITORI
# =====================================================================

@app.get("/api/speakers")
async def list_speakers():
    speakers_list = []
    for speaker_id, data in speakers_db.items():
        speakers_list.append({
            "id": speaker_id,
            "name": data.get("name", speaker_id),
            "num_references": len(data.get("audio_files", [])),
            "added_date": data.get("added_date", ""),
            "description": data.get("description", "")
        })
    return {"speakers": speakers_list, "total": len(speakers_list)}


@app.post("/api/speakers")
async def add_speaker(
    name: str = Form(...),
    description: str = Form(""),
    files: List[UploadFile] = File(...)
):
    speaker_id = f"speaker_{uuid.uuid4().hex[:8]}"
    speaker_dir = SPEAKERS_DIR / speaker_id
    speaker_dir.mkdir(parents=True, exist_ok=True)

    audio_files = []
    for file in files:
        if not file.filename.lower().endswith(('.wav', '.mp3', '.flac', '.ogg', '.m4a')):
            continue
        filepath = speaker_dir / f"{uuid.uuid4().hex[:8]}_{file.filename}"
        content = await file.read()
        with open(filepath, 'wb') as f:
            f.write(content)
        audio_files.append(str(filepath))

    if not audio_files:
        raise HTTPException(400, "Nu au fost trimise fisiere audio valide!")

    speakers_db[speaker_id] = {
        "name": name,
        "description": description,
        "audio_files": audio_files,
        "added_date": datetime.now().isoformat(),
        "num_references": len(audio_files)
    }
    save_speakers_db()

    return {
        "success": True,
        "speaker_id": speaker_id,
        "name": name,
        "num_references": len(audio_files)
    }


@app.delete("/api/speakers/{speaker_id}")
async def remove_speaker(speaker_id: str):
    if speaker_id not in speakers_db:
        raise HTTPException(404, "Vorbitor negasit!")
    import shutil
    speaker_dir = SPEAKERS_DIR / speaker_id
    if speaker_dir.exists():
        shutil.rmtree(speaker_dir)
    name = speakers_db[speaker_id].get("name", speaker_id)
    del speakers_db[speaker_id]
    save_speakers_db()
    return {"success": True, "message": f"Vorbitor '{name}' sters."}


# =====================================================================
# ENDPOINT — RECUNOAȘTERE VORBITOR (Speaker Recognition)
# =====================================================================

@app.post("/api/speakers/recognize")
async def recognize_speaker(
    audio: UploadFile = File(...)
):
    """Recunoaște vorbitorul din audio comparând cu toți vorbitorii înrolați."""
    if not speakers_db:
        raise HTTPException(400, "Nu există vorbitori înrolați! Adaugă cel puțin un vorbitor.")

    # Salvăm fișierul uploadat
    rec_id = uuid.uuid4().hex[:12]
    audio_path = WEBAPP_UPLOADS / f"{rec_id}_recognize_{audio.filename}"
    content = await audio.read()
    with open(audio_path, 'wb') as f:
        f.write(content)
    audio_path = convert_to_wav(audio_path)

    try:
        import torch
        recognizer = get_speaker_recognizer()

        # Extrage embedding din audio-ul de test (.as_posix() - evita erorile de parsare Windows \U)
        audio_abs = Path(audio_path).resolve().as_posix()
        test_embedding = recognizer.encode_batch(
            recognizer.load_audio(audio_abs).unsqueeze(0)
        ).squeeze()

        results = []
        for speaker_id, speaker_data in speakers_db.items():
            ref_files = speaker_data.get("audio_files", [])
            if not ref_files:
                continue

            # Extrage embedding-uri din toate referințele vorbitorului
            ref_embeddings = []
            for ref_path in ref_files:
                ref_abs = Path(ref_path).resolve().as_posix()
                if not Path(ref_path).exists():
                    continue
                try:
                    ref_emb = recognizer.encode_batch(
                        recognizer.load_audio(ref_abs).unsqueeze(0)
                    ).squeeze()
                    ref_embeddings.append(ref_emb)
                except Exception as e:
                    logger.warning(f"Nu pot procesa referința {ref_path}: {e}")
                    continue

            if not ref_embeddings:
                continue

            # Media embedding-urilor de referință
            mean_ref = torch.stack(ref_embeddings).mean(dim=0)

            # Similaritate cosinus
            similarity = torch.nn.functional.cosine_similarity(
                test_embedding.unsqueeze(0),
                mean_ref.unsqueeze(0)
            ).item()

            # Scor procentual (cosine similarity [-1, 1] → [0, 100])
            score_pct = max(0, min(100, (similarity + 1) / 2 * 100))

            results.append({
                "speaker_id": speaker_id,
                "name": speaker_data.get("name", speaker_id),
                "score": round(score_pct, 1),
                "raw_similarity": round(similarity, 4),
                "num_references": len(ref_files),
                "is_match": similarity > 0.25  # prag SpeechBrain implicit
            })

        # Sortăm descrescător după scor
        results.sort(key=lambda x: x["score"], reverse=True)

        best = results[0] if results else None
        return {
            "success": True,
            "best_match": best,
            "all_results": results,
            "threshold": 0.25
        }

    except Exception as e:
        logger.error(f"Eroare recunoaștere vorbitor: {e}", exc_info=True)
        raise HTTPException(500, f"Eroare la recunoaștere: {str(e)}")
    finally:
        # Curăță fișierul temporar
        if audio_path.exists():
            audio_path.unlink(missing_ok=True)


# =====================================================================
# ENDPOINTS — CONVERSIE kNN-VC
# =====================================================================

@app.post("/api/convert")
async def convert_voice_knn(
    source: UploadFile = File(...),
    speaker_id: str = Form(...),
    topk: int = Form(4)
):
    """Conversie cu kNN-VC (pre-antrenat)."""
    if speaker_id not in speakers_db:
        raise HTTPException(404, "Vorbitor target negasit!")

    speaker_data = speakers_db[speaker_id]
    target_refs = speaker_data["audio_files"]
    if not target_refs:
        raise HTTPException(400, "Vorbitorul nu are referinte audio!")

    conversion_id = uuid.uuid4().hex[:12]
    source_path = WEBAPP_UPLOADS / f"{conversion_id}_source_{source.filename}"
    content = await source.read()
    with open(source_path, 'wb') as f:
        f.write(content)

    # Conversie automată la WAV dacă nu e deja
    source_path = convert_to_wav(source_path)

    try:
        vc = get_knn_converter()
        result = vc.convert(
            source_audio=str(source_path),
            target_references=target_refs,
            topk=topk
        )

        output_filename = f"{conversion_id}_knnvc.wav"
        output_path = WEBAPP_OUTPUTS / output_filename
        result.save(output_path)

        duration = safe_float(result.get_duration())
        conv_time = safe_float(result.conversion_time * 1000, 0)

        entry = {
            "id": conversion_id,
            "model": "knn-vc",
            "timestamp": datetime.now().isoformat(),
            "source_filename": source.filename,
            "target_speaker": speaker_data["name"],
            "target_speaker_id": speaker_id,
            "topk": topk,
            "conversion_time_ms": conv_time,
            "output_path": str(output_path),
            "source_path": str(source_path),
            "duration": duration
        }

        # STT: transcripție sursă și output
        try:
            stt = get_whisper_stt()
            src_stt = stt.transcribe(str(source_path))
            out_stt = stt.transcribe(str(output_path))
            entry["source_text"] = src_stt["text"]
            entry["converted_text"] = out_stt["text"]
            if src_stt["text"] and out_stt["text"]:
                entry["wer"] = safe_float(stt.compute_wer(src_stt["text"], out_stt["text"]))
        except Exception as stt_err:
            logger.warning(f"STT indisponibil: {stt_err}")

        conversion_history.append(entry)
        save_history()

        response = {
            "success": True,
            "conversion_id": conversion_id,
            "model": "knn-vc",
            "output_url": f"/api/audio/{output_filename}",
            "source_url": f"/api/audio/{source_path.name}",
            "conversion_time_ms": conv_time,
            "duration": duration,
            "target_speaker": speaker_data["name"],
        }
        # Adaugă transcripții dacă disponibile
        if "source_text" in entry:
            response["source_text"] = entry["source_text"]
            response["converted_text"] = entry.get("converted_text", "")
            response["wer"] = entry.get("wer", None)

        return response

    except Exception as e:
        logger.error(f"Eroare conversie kNN-VC: {e}")
        raise HTTPException(500, f"Eroare: {str(e)}")


# =====================================================================
# ENDPOINT — CONVERSIE kNN-VC (XLS-R Multilingv - Experimental)
# =====================================================================

@app.post("/api/convert/mknn")
async def convert_mknn_voice(
    source: UploadFile = File(...),
    target_speaker: str = Form(...),
    topk: int = Form(4)
):
    try:
        if target_speaker not in speakers_db:
            raise HTTPException(404, "Vorbitorul țintă nu a fost găsit")
            
        target_refs = speakers_db[target_speaker].get("audio_files", [])
        if not target_refs:
            raise HTTPException(400, "Vorbitorul țintă nu are fișiere de referință")

        unique_id = uuid.uuid4().hex[:8]
        source_ext = Path(source.filename).suffix
        source_path = WEBAPP_UPLOADS / f"source_mknn_{unique_id}{source_ext}"
        content = await source.read()
        with open(source_path, "wb") as f:
            f.write(content)

        converter = get_mknn_converter()
        logger.info(f"Rulare conversie Multilingvă kNN pentru sursa: {source.filename}")
        
        # Salvăm fișierul convertit cu prefix special
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_filename = f"mknn_converted_{timestamp}.wav"
        output_filepath = WEBAPP_OUTPUTS / output_filename
        
        result = converter.convert(
            source_audio=str(source_path),
            target_references=target_refs,
            topk=topk
        )
        result.save(output_filepath)

        record = {
            "id": unique_id,
            "date": datetime.now().isoformat(),
            "model": "mknn-vc (xls-r)",
            "source_file": source.filename,
            "target_speaker": target_speaker,
            "duration": round(result.get_duration(), 1),
            "output_file": output_filename
        }
        conversion_history.append(record)
        save_history()

        return {
            "success": True,
            "duration": result.get_duration(),
            "sample_rate": result.sample_rate,
            "source": str(source_path),
            "targets": target_refs,
            "topk": topk,
            "conversion_time_ms": round(result.conversion_time * 1000, 1),
            "device": result.device_used,
            "output_url": f"/api/audio/{output_filename}"
        }

    except Exception as e:
        logger.error(f"Eroare conversie mKNN-VC: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =====================================================================
# ENDPOINTS — CONVERSIE LightVC
# =====================================================================

#@app.post("/api/lightvc/convert")
#async def convert_voice_lightvc(
#    source: UploadFile = File(...),
#    speaker_id: str = Form(...),
#):
#    """Conversie cu LightVC (model antrenat de utilizator)."""
    # Verificăm că modelul e antrenat
 #   checkpoint = LIGHTVC_CHECKPOINT_DIR / "best_model.pth"
 #   if not checkpoint.exists():
  #      raise HTTPException(
 #           503,
 #          "Modelul LightVC nu a fost antrenat inca. "
 #           "Mergi la pagina 'Antrenare Model' si porneste antrenarea."
 #       )

 ##   if speaker_id not in speakers_db:
  ##      raise HTTPException(404, "Vorbitor target negasit!")

      # speaker_data = speakers_db[speaker_id]
    #target_refs = speaker_data["audio_files"]

    #conversion_id = uuid.uuid4().hex[:12]
    #source_path = WEBAPP_UPLOADS / f"{conversion_id}_source_{source.filename}"
    #content = await source.read()
    #with open(source_path, 'wb') as f:
    #    f.write(content)

    # Conversie automată la WAV
   # source_path = convert_to_wav(source_path)

   # try:
   #     vc = get_lightvc_converter()
   #     result = vc.convert(
   #         source_audio=str(source_path),
   #         target_references=target_refs
   #     )

     #   output_filename = f"{conversion_id}_lightvc.wav"
     #   output_path = WEBAPP_OUTPUTS / output_filename
     #   result.save(output_path)

     #   duration = safe_float(result.get_duration())
     #   conv_time = safe_float(result.conversion_time * 1000, 0)

   #     entry = {
  #          "id": conversion_id,
  #          "model": "lightvc",
   #         "timestamp": datetime.now().isoformat(),
  #          "source_filename": source.filename,
   #         "target_speaker": speaker_data["name"],
   #         "target_speaker_id": speaker_id,
   #         "conversion_time_ms": conv_time,
   #         "output_path": str(output_path),
   #         "source_path": str(source_path),
   #         "duration": duration
   #     }

   #     # STT
   #     try:
   #         stt = get_whisper_stt()
   #         src_stt = stt.transcribe(str(source_path))
   #         out_stt = stt.transcribe(str(output_path))
   #         entry["source_text"] = src_stt["text"]
   #         entry["converted_text"] = out_stt["text"]
   #         if src_stt["text"] and out_stt["text"]:
   #             entry["wer"] = safe_float(stt.compute_wer(src_stt["text"], out_stt["text"]))
   #     except Exception as stt_err:
   #         logger.warning(f"STT indisponibil: {stt_err}")

   #     conversion_history.append(entry)
   #     save_history()

   #     response = {
   #         "success": True,
   #         "conversion_id": conversion_id,
   #         "model": "lightvc",
   #         "output_url": f"/api/audio/{output_filename}",
   #         "source_url": f"/api/audio/{source_path.name}",
   #         "conversion_time_ms": conv_time,
   #         "duration": duration,
   #     }
   #     if "source_text" in entry:
   #         response["source_text"] = entry["source_text"]
   #         response["converted_text"] = entry.get("converted_text", "")
   #         response["wer"] = entry.get("wer", None)

   #     return response

   # except Exception as e:
   #     logger.error(f"Eroare conversie LightVC: {e}")
   #     raise HTTPException(500, f"Eroare: {str(e)}")


# =====================================================================
# ENDPOINTS — CONVERSIE FreeVC
# =====================================================================

@app.post("/api/freevc/convert")
async def convert_voice_freevc(
    source: UploadFile = File(...),
    speaker_id: str = Form(...)
):
    """Conversie cu FreeVC (pre-antrenat, Coqui TTS)."""
    if speaker_id not in speakers_db:
        raise HTTPException(404, "Vorbitor target negasit!")

    speaker_data = speakers_db[speaker_id]
    target_refs = speaker_data["audio_files"]

    conversion_id = uuid.uuid4().hex[:12]
    source_path = WEBAPP_UPLOADS / f"{conversion_id}_source_{source.filename}"
    content = await source.read()
    with open(source_path, 'wb') as f:
        f.write(content)

    source_path = convert_to_wav(source_path)

    try:
        vc = get_freevc_converter()
        result = vc.convert(
            source_audio=str(source_path),
            target_references=target_refs
        )

        output_filename = f"{conversion_id}_freevc.wav"
        output_path = WEBAPP_OUTPUTS / output_filename
        result.save(output_path)

        duration = safe_float(result.get_duration())
        conv_time = safe_float(result.conversion_time * 1000, 0)

        entry = {
            "id": conversion_id,
            "model": "freevc",
            "timestamp": datetime.now().isoformat(),
            "source_filename": source.filename,
            "target_speaker": speaker_data["name"],
            "target_speaker_id": speaker_id,
            "conversion_time_ms": conv_time,
            "output_path": str(output_path),
            "source_path": str(source_path),
            "duration": duration
        }

        # STT
        try:
            stt = get_whisper_stt()
            src_stt = stt.transcribe(str(source_path))
            out_stt = stt.transcribe(str(output_path))
            entry["source_text"] = src_stt["text"]
            entry["converted_text"] = out_stt["text"]
            if src_stt["text"] and out_stt["text"]:
                entry["wer"] = safe_float(stt.compute_wer(src_stt["text"], out_stt["text"]))
        except Exception as stt_err:
            logger.warning(f"STT indisponibil: {stt_err}")

        conversion_history.append(entry)
        save_history()

        response = {
            "success": True,
            "conversion_id": conversion_id,
            "model": "freevc",
            "output_url": f"/api/audio/{output_filename}",
            "source_url": f"/api/audio/{source_path.name}",
            "conversion_time_ms": conv_time,
            "duration": duration,
        }
        if "source_text" in entry:
            response["source_text"] = entry["source_text"]
            response["converted_text"] = entry.get("converted_text", "")
            response["wer"] = entry.get("wer", None)

        return response

    except Exception as e:
        logger.error(f"Eroare conversie FreeVC: {e}")
        raise HTTPException(500, f"Eroare: {str(e)}")


# =====================================================================
# ENDPOINTS — CONVERSIE RVC (Direct Voice-to-Voice)
# =====================================================================

@app.post("/api/rvc/convert")
async def convert_voice_rvc(
    source: UploadFile = File(...),
    speaker_id: str = Form(...)
):
    """Conversie directa Voice-to-Voice cu RVC."""
    if speaker_id not in speakers_db:
        raise HTTPException(404, "Vorbitor target negasit!")

    speaker_data = speakers_db[speaker_id]
    
    conversion_id = uuid.uuid4().hex[:12]
    source_path = WEBAPP_UPLOADS / f"{conversion_id}_source_{source.filename}"
    content = await source.read()
    with open(source_path, 'wb') as f:
        f.write(content)
        
    source_path = convert_to_wav(source_path)

    try:
        rvc = get_rvc_converter()
        if not rvc.has_model(speaker_id):
            raise ValueError(f"Vorbitorul '{speaker_data['name']}' nu are un model RVC antrenat!")
            
        t0 = time.time()
        output_filename = f"{conversion_id}_rvc.wav"
        output_path = WEBAPP_OUTPUTS / output_filename
        
        result_path = rvc.convert(
            input_audio_path=str(source_path),
            speaker_id=speaker_id,
            output_path=str(output_path)
        )
        
        conv_time = safe_float((time.time() - t0) * 1000)
        
        import librosa
        duration = safe_float(librosa.get_duration(path=result_path))

        entry = {
            "id": conversion_id,
            "model": "rvc",
            "timestamp": datetime.now().isoformat(),
            "source_filename": source.filename,
            "target_speaker": speaker_data["name"],
            "target_speaker_id": speaker_id,
            "conversion_time_ms": conv_time,
            "output_path": str(result_path),
            "source_path": str(source_path),
            "duration": duration
        }

        # STT
        try:
            stt = get_whisper_stt()
            src_stt = stt.transcribe(str(source_path))
            out_stt = stt.transcribe(str(result_path))
            entry["source_text"] = src_stt["text"]
            entry["converted_text"] = out_stt["text"]
            if src_stt["text"] and out_stt["text"]:
                entry["wer"] = safe_float(stt.compute_wer(src_stt["text"], out_stt["text"]))
        except Exception as stt_err:
            logger.warning(f"STT indisponibil: {stt_err}")

        conversion_history.append(entry)
        save_history()

        response = {
            "success": True,
            "conversion_id": conversion_id,
            "model": "rvc",
            "output_url": f"/api/audio/{Path(result_path).name}",
            "source_url": f"/api/audio/{source_path.name}",
            "conversion_time_ms": conv_time,
            "duration": duration,
        }
        if "source_text" in entry:
            response["source_text"] = entry["source_text"]
            response["converted_text"] = entry.get("converted_text", "")
            response["wer"] = entry.get("wer", None)

        return response

    except Exception as e:
        logger.error(f"Eroare conversie RVC: {e}")
        raise HTTPException(500, f"Eroare: {str(e)}")


# =====================================================================
# ENDPOINTS — CONVERSIE YourTTS (Voice-to-Voice prin STT→TTS)
# =====================================================================

_yourtts_converter = None

def get_yourtts_converter():
    """Lazy loading YourTTS converter."""
    global _yourtts_converter
    if _yourtts_converter is None:
        from voice_conversion.models.yourtts_vc import YourTTSConverter
        _yourtts_converter = YourTTSConverter()
    return _yourtts_converter


@app.post("/api/yourtts/convert")
async def convert_voice_yourtts(
    source: UploadFile = File(...),
    speaker_id: str = Form(...)
):
    """Conversie Voice-to-Voice cu YourTTS (STT → TTS cu vocea țintă)."""
    if speaker_id not in speakers_db:
        raise HTTPException(404, "Vorbitor target negăsit!")

    speaker_data = speakers_db[speaker_id]
    target_refs = speaker_data["audio_files"]

    conversion_id = uuid.uuid4().hex[:12]
    source_path = WEBAPP_UPLOADS / f"{conversion_id}_source_{source.filename}"
    content = await source.read()
    with open(source_path, 'wb') as f:
        f.write(content)

    source_path = convert_to_wav(source_path)

    try:
        converter = get_yourtts_converter()
        
        output_filename = f"{conversion_id}_yourtts.wav"
        output_path = WEBAPP_OUTPUTS / output_filename
        
        result = converter.convert(
            source_audio_path=str(source_path),
            target_references=target_refs,
            output_path=str(output_path),
            language="ro"
        )

        conv_time = safe_float(result["conversion_time"] * 1000)
        duration = safe_float(result["duration"])

        entry = {
            "id": conversion_id,
            "model": "yourtts",
            "timestamp": datetime.now().isoformat(),
            "source_filename": source.filename,
            "target_speaker": speaker_data["name"],
            "target_speaker_id": speaker_id,
            "conversion_time_ms": conv_time,
            "output_path": str(output_path),
            "source_path": str(source_path),
            "duration": duration
        }

        # STT comparativ
        try:
            stt = get_whisper_stt()
            src_stt = stt.transcribe(str(source_path))
            out_stt = stt.transcribe(str(output_path))
            entry["source_text"] = src_stt["text"]
            entry["converted_text"] = out_stt["text"]
            if src_stt["text"] and out_stt["text"]:
                entry["wer"] = safe_float(
                    stt.compute_wer(src_stt["text"], out_stt["text"])
                )
        except Exception as stt_err:
            logger.warning(f"STT indisponibil: {stt_err}")

        conversion_history.append(entry)
        save_history()

        response = {
            "success": True,
            "conversion_id": conversion_id,
            "model": "yourtts",
            "output_url": f"/api/audio/{output_filename}",
            "source_url": f"/api/audio/{source_path.name}",
            "conversion_time_ms": conv_time,
            "duration": duration,
        }
        if "source_text" in entry:
            response["source_text"] = entry["source_text"]
            response["converted_text"] = entry.get("converted_text", "")
            response["wer"] = entry.get("wer", None)

        return response

    except Exception as e:
        logger.error(f"Eroare conversie YourTTS: {e}")
        raise HTTPException(500, f"Eroare: {str(e)}")


@app.get("/api/yourtts/status")
async def get_yourtts_status():
    """Status model YourTTS: disponibilitate, fine-tuned, progres antrenare."""
    try:
        from voice_conversion.models.yourtts_vc import YourTTSConverter
        converter = YourTTSConverter()
        info = converter.get_model_info()
    except Exception:
        info = {"available": False, "finetuned": False}
    
    # Progres antrenare
    try:
        from voice_conversion.models.yourtts_trainer import (
            get_yourtts_training_progress
        )
        training = get_yourtts_training_progress()
    except Exception:
        training = {"status": "idle"}
    
    return {**info, "training": training}


@app.post("/api/yourtts/train")
async def start_yourtts_training(
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """Pornire fine-tuning YourTTS pe Common Voice RO."""
    from voice_conversion.models.yourtts_trainer import (
        get_yourtts_training_progress, YourTTSTrainer
    )
    
    status = get_yourtts_training_progress()
    if status.get("status") in ["preparing", "training"]:
        raise HTTPException(409, "O antrenare YourTTS rulează deja!")
    
    def _train():
        trainer = YourTTSTrainer()
        trainer.train()
    
    background_tasks.add_task(_train)
    return {"success": True, "message": "Fine-tuning YourTTS pornit!"}


# =====================================================================
# ENDPOINT — VOICE CLONING (STT → TTS)
# =====================================================================

@app.post("/api/tts/clone")
async def clone_voice_via_tts(
    source: UploadFile = File(...),
    speaker_id: str = Form(...)
):
    """
    Flux STT → TTS:
    1. Transcrie audio-ul sursă (Whisper)
    2. Generează audio nou din text cu vocea target (SpeechT5)
    """
    try:
        if speaker_id not in speakers_db:
            raise HTTPException(404, "Vorbitorul nu există")
        
        speaker_data = speakers_db[speaker_id]
        speaker_refs = speaker_data.get("audio_files", [])
        if not speaker_refs:
            raise HTTPException(400, "Vorbitorul nu are înregistrări")
            
        ref_audio_path = Path(speaker_refs[0])

        # Salvare sursă temporară
        unique_id = uuid.uuid4().hex[:8]
        source_ext = Path(source.filename).suffix
        source_path = WEBAPP_UPLOADS / f"tts_source_{unique_id}{source_ext}"
        
        content = await source.read()
        with open(source_path, "wb") as f:
            f.write(content)

        # 1. Transcriere (Whisper STT)
        logger.info(f"Începe transcrierea pentru {source_path.name}")
        stt = get_whisper_stt()
        stt_result = stt.transcribe(str(source_path), language="ro")
        text = stt_result.get("text", "")
        
        if not text:
            raise HTTPException(400, "Nu s-a putut transcrie text din audio-ul furnizat.")

        # 2. Generare (XTTS-v2)
        logger.info(f"Începe generarea XTTS pentru textul transcris: {text[:30]}...")
        tts = get_tts_model()
        audio_np, sr = tts.synthesize(text, ref_audio_path, language="ro")

        # 3. Salvare rezultat
        output_filename = f"tts_clone_{unique_id}.wav"
        output_path = WEBAPP_OUTPUTS / output_filename
        
        import soundfile as sf
        sf.write(str(output_path), audio_np, sr)

        # Pregătire istoric
        entry = {
            "id": unique_id,
            "timestamp": datetime.now().isoformat(),
            "model": "VoiceClone-TTS",
            "source_audio": f"/api/audio/{source_path.name}",
            "target_speaker": speaker_id,
            "converted_audio": f"/api/audio/{output_filename}",
            "source_text": text,
            "stt_time": stt_result.get("transcription_time", 0)
        }
        
        conversion_history.append(entry)
        save_history()

        return {
            "success": True,
            "source_text": text,
            "output_url": entry["converted_audio"],
            "source_url": entry["source_audio"],
            "speaker_name": speaker_id,
            "stt_time": entry["stt_time"],
            "message": "Conversie STT-TTS finalizată cu succes!"
        }

    except Exception as e:
        logger.error(f"Eroare Voice Cloning TTS: {e}")
        raise HTTPException(500, f"Eroare: {str(e)}")


# =====================================================================
# ENDPOINT — COMPARATIE 3 MODELE
# =====================================================================

@app.post("/api/compare")
async def compare_models(
    source: UploadFile = File(...),
    speaker_id: str = Form(...),
    topk: int = Form(4)
):
    """
    Rulează AMBELE modele pe același input și returnează rezultatele
    pentru pagina de comparație.
    """
    if speaker_id not in speakers_db:
        raise HTTPException(404, "Vorbitor target negasit!")

    speaker_data = speakers_db[speaker_id]
    target_refs = speaker_data["audio_files"]

    # Salvare sursă
    conversion_id = uuid.uuid4().hex[:12]
    source_content = await source.read()
    source_path = WEBAPP_UPLOADS / f"{conversion_id}_source_{source.filename}"
    with open(source_path, 'wb') as f:
        f.write(source_content)

    # Conversie automată la WAV
    source_path = convert_to_wav(source_path)

    # --- Transcripție STT pentru sursă ---
    source_text = ""
    stt_enabled = False
    stt = None
    try:
        stt = get_whisper_stt()
        src_stt = stt.transcribe(str(source_path))
        source_text = src_stt["text"]
        stt_enabled = True
    except Exception as e:
        logger.warning(f"STT nu e disponibil pentru sursă: {e}")

    results = {
        "comparison_id": conversion_id,
        "source_url": f"/api/audio/{source_path.name}",
        "source_text": source_text,
        "target_speaker": speaker_data["name"],
        "knn_vc": None,
        "lightvc": None,
        "freevc": None,
        "voice_clone": None,
        "timestamp": datetime.now().isoformat()
    }

    # Helper pentru STT pe output
    def add_stt_to_result(model_key, output_path_str):
        if stt_enabled and source_text:
            try:
                out_stt = stt.transcribe(output_path_str)
                results[model_key]["converted_text"] = out_stt["text"]
                if out_stt["text"]:
                    results[model_key]["wer"] = safe_float(stt.compute_wer(source_text, out_stt["text"]))
            except Exception:
                pass

    # --- kNN-VC ---
    try:
        knn_vc = get_knn_converter()
        t0 = time.time()
        knn_result = knn_vc.convert(
            source_audio=str(source_path),
            target_references=target_refs,
            topk=topk
        )
        knn_time = time.time() - t0

        knn_output = f"{conversion_id}_knnvc.wav"
        knn_output_path = WEBAPP_OUTPUTS / knn_output
        knn_result.save(knn_output_path)

        results["knn_vc"] = {
            "output_url": f"/api/audio/{knn_output}",
            "conversion_time_ms": safe_float(knn_time * 1000, 0),
            "status": "success"
        }
        add_stt_to_result("knn_vc", str(knn_output_path))
    except Exception as e:
        results["knn_vc"] = {"status": "error", "error": str(e)}

    # --- YourTTS ---
    try:
        from voice_conversion.models.yourtts_vc import YourTTSConverter
        yourtts_model = get_yourtts_converter()
        
        yourtts_output = f"{conversion_id}_yourtts.wav"
        yourtts_output_path = WEBAPP_OUTPUTS / yourtts_output
        
        t0 = time.time()
        yourtts_result = yourtts_model.convert(
            source_audio_path=str(source_path),
            target_references=target_refs,
            output_path=str(yourtts_output_path),
            language="ro"
        )
        yourtts_time = time.time() - t0

        results["yourtts"] = {
            "output_url": f"/api/audio/{yourtts_output}",
            "conversion_time_ms": safe_float(yourtts_time * 1000, 0),
            "status": "success",
            "transcribed_text": yourtts_result.get("source_text", "")
        }
        add_stt_to_result("yourtts", str(yourtts_output_path))
    except Exception as e:
        logger.error(f"Eroare comparatie YourTTS: {e}")
        results["yourtts"] = {"status": "error", "error": str(e)}

    # --- FreeVC ---
    try:
        fvc = get_freevc_converter()
        t0 = time.time()
        fvc_result = fvc.convert(
            source_audio=str(source_path),
            target_references=target_refs
        )
        fvc_time = time.time() - t0

        fvc_output = f"{conversion_id}_freevc.wav"
        fvc_output_path = WEBAPP_OUTPUTS / fvc_output
        fvc_result.save(fvc_output_path)

        results["freevc"] = {
            "output_url": f"/api/audio/{fvc_output}",
            "conversion_time_ms": safe_float(fvc_time * 1000, 0),
            "status": "success"
        }
        add_stt_to_result("freevc", str(fvc_output_path))
    except Exception as e:
        results["freevc"] = {"status": "error", "error": str(e)}

    # --- Voice Cloning (Pipeline Modern: Whisper STT → XTTS v2 → RVC) ---
    try:
        if source_text and source_text.strip():
            tts = get_tts_model()
            t0 = time.time()
            audio_np, clone_sr = tts.synthesize(
                text=source_text,
                speaker_reference=target_refs,
                language="ro",
                speaker_id=speaker_id,  # Permite RVC post-processing
                apply_rvc=True
            )
            clone_time = time.time() - t0

            clone_output = f"{conversion_id}_voiceclone.wav"
            clone_output_path = WEBAPP_OUTPUTS / clone_output

            # Convertim numpy → torch tensor pentru save_audio
            clone_tensor = torch.from_numpy(audio_np).unsqueeze(0).float()
            save_audio(clone_tensor, str(clone_output_path), sample_rate=clone_sr)

            results["voice_clone"] = {
                "output_url": f"/api/audio/{clone_output}",
                "conversion_time_ms": safe_float(clone_time * 1000, 0),
                "status": "success",
                "transcribed_text": source_text,
                "is_finetuned": getattr(tts, 'is_finetuned', False),
                "rvc_applied": getattr(tts, '_rvc', None) is not None
            }
            add_stt_to_result("voice_clone", str(clone_output_path))
        else:
            results["voice_clone"] = {
                "status": "error",
                "error": "Nu s-a putut transcrie textul sursă (STT indisponibil)"
            }
    except Exception as e:
        logger.warning(f"Voice Cloning eșuat în comparație: {e}")
        results["voice_clone"] = {"status": "error", "error": str(e)}

    # --- Metrici pentru toate modelele ---
    try:
        target_ref = target_refs[0] if target_refs else None
        for model_key in ["knn_vc", "yourtts", "freevc", "voice_clone"]:
            r = results.get(model_key)
            if r and r.get("status") == "success":
                output_path = WEBAPP_OUTPUTS / f"{conversion_id}_{model_key.replace('_', '')}.wav"
                if output_path.exists() and target_ref:
                    metrics = compute_all_metrics(
                        source_audio=str(source_path),
                        converted_audio=str(output_path),
                        target_reference=target_ref
                    )
                    r["metrics"] = {
                        k: safe_float(v) if isinstance(v, float) else v
                        for k, v in metrics.items()
                    }
    except Exception as e:
        logger.warning(f"Metrici indisponibile: {e}")

    results["source"] = {
        "output_url": f"/api/audio/{source_path.name}",
        "filename": source_path.name
    }

    return results


# =====================================================================
# ENDPOINTS — EVALUARE
# =====================================================================

@app.post("/api/evaluate")
async def evaluate_conversion(
    conversion_id: str = Form(...),
    target_reference_index: int = Form(0)
):
    entry = next((h for h in conversion_history if h["id"] == conversion_id), None)
    if entry is None:
        raise HTTPException(404, "Conversie negasita!")

    speaker_data = speakers_db.get(entry["target_speaker_id"])
    if not speaker_data:
        raise HTTPException(404, "Vorbitor negasit!")

    refs = speaker_data["audio_files"]
    target_ref = refs[min(target_reference_index, len(refs) - 1)]

    try:
        metrics = compute_all_metrics(
            source_audio=entry["source_path"],
            converted_audio=entry["output_path"],
            target_reference=target_ref
        )
        return {
            "conversion_id": conversion_id,
            "metrics": {
                "mcd": {"value": safe_float(metrics.get("mcd", 0)), "unit": "dB",
                        "label": "MCD", "direction": "lower"},
                "pesq": {"value": safe_float(metrics.get("pesq", 0)), "unit": "",
                         "label": "PESQ", "direction": "higher"},
                "speaker_similarity": {"value": safe_float(metrics.get("speaker_similarity", 0)),
                                       "unit": "", "label": "Speaker Sim.", "direction": "higher"},
                "f0_rmse": {"value": safe_float(metrics.get("f0_rmse", 0)), "unit": "Hz",
                            "label": "F0 RMSE", "direction": "lower"},
                "f0_pcc": {"value": safe_float(metrics.get("f0_pcc", 0)), "unit": "",
                           "label": "F0 PCC", "direction": "higher"},
                "snr": {"value": safe_float(metrics.get("snr", 0)), "unit": "dB",
                        "label": "SNR", "direction": "higher"}
            }
        }
    except Exception as e:
        raise HTTPException(500, f"Eroare evaluare: {str(e)}")


# =====================================================================
# ENDPOINT — ÎNREGISTRARE AUDIO DIN BROWSER
# =====================================================================

@app.post("/api/upload-recording")
async def upload_recording(
    audio: UploadFile = File(...),
    purpose: str = Form("source"),
    speaker_name: str = Form(""),
    speaker_id: str = Form("")
):
    """
    Primește o înregistrare audio din browser (MediaRecorder, format WebM/WAV)
    și o convertește automat la WAV 16kHz mono.

    Args:
        audio: Fișier audio (WebM din MediaRecorder sau alt format)
        purpose: 'source' (pentru conversie) sau 'reference' (pentru speaker)
        speaker_name: Nume vorbitor (doar pentru purpose='reference')
        speaker_id: ID vorbitor existent (doar pentru purpose='add_reference')
    """
    recording_id = uuid.uuid4().hex[:12]

    # Determină extensia originală
    orig_ext = Path(audio.filename).suffix if audio.filename else '.webm'
    raw_path = WEBAPP_UPLOADS / f"recording_{recording_id}{orig_ext}"

    content = await audio.read()
    with open(raw_path, 'wb') as f:
        f.write(content)

    logger.info(f"\U0001F3A4 Înregistrare primită: {len(content)} bytes, format={orig_ext}")

    # Conversie la WAV
    wav_path = convert_to_wav(raw_path)

    # Verifică că fișierul WAV e valid
    try:
        import soundfile as sf
        info = sf.info(str(wav_path))
        duration = info.duration
        logger.info(f"   Durată: {duration:.1f}s, SR: {info.samplerate}")
    except Exception:
        duration = 0

    if purpose == 'reference' and speaker_name:
        # Creează speaker nou cu această înregistrare
        spk_id = f"speaker_{uuid.uuid4().hex[:8]}"
        speaker_dir = SPEAKERS_DIR / spk_id
        speaker_dir.mkdir(parents=True, exist_ok=True)

        import shutil
        final_path = speaker_dir / f"{recording_id}_recording.wav"
        shutil.copy2(str(wav_path), str(final_path))

        speakers_db[spk_id] = {
            "name": speaker_name,
            "description": "\u00cenregistrat din browser",
            "audio_files": [str(final_path)],
            "added_date": datetime.now().isoformat(),
            "num_references": 1
        }
        save_speakers_db()

        return {
            "success": True,
            "purpose": "reference",
            "speaker_id": spk_id,
            "speaker_name": speaker_name,
            "recording_url": f"/api/audio/{wav_path.name}",
            "duration": safe_float(duration)
        }

    elif purpose == 'add_reference' and speaker_id:
        # Adaugă referință la speaker existent
        if speaker_id not in speakers_db:
            raise HTTPException(404, "Vorbitor negăsit!")

        speaker_dir = SPEAKERS_DIR / speaker_id
        speaker_dir.mkdir(parents=True, exist_ok=True)

        import shutil
        final_path = speaker_dir / f"{recording_id}_recording.wav"
        shutil.copy2(str(wav_path), str(final_path))

        speakers_db[speaker_id]["audio_files"].append(str(final_path))
        speakers_db[speaker_id]["num_references"] = len(speakers_db[speaker_id]["audio_files"])
        save_speakers_db()

        return {
            "success": True,
            "purpose": "add_reference",
            "speaker_id": speaker_id,
            "recording_url": f"/api/audio/{wav_path.name}",
            "total_references": len(speakers_db[speaker_id]["audio_files"]),
            "duration": safe_float(duration)
        }

    else:
        # purpose == 'source' — fișier sursă pentru conversie
        return {
            "success": True,
            "purpose": "source",
            "recording_url": f"/api/audio/{wav_path.name}",
            "filename": wav_path.name,
            "duration": safe_float(duration)
        }

# =====================================================================
# ENDPOINTS — LIGHTVC TRAINING
# =====================================================================

#@app.get("/api/lightvc/status")
#async def lightvc_status():
   # """Status model LightVC: antrenat/neantrenat, informații checkpoint."""
   # checkpoint = LIGHTVC_CHECKPOINT_DIR / "best_model.pth"
   # progress_file = LIGHTVC_CHECKPOINT_DIR / "training_progress.json"

   # is_trained = checkpoint.exists()
   # training_progress = {}
   # if progress_file.exists():
   #      with open(progress_file) as f:
   #         training_progress = json.load(f)

    # Info checkpoint dacă există
   # checkpoint_info = {}
   # if is_trained:
   #      try:
   #          import torch
   #          ck = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
   #          checkpoint_info = {
   #              "epoch": ck.get("epoch", 0),
   #              "best_val_loss": safe_float(ck.get("best_val_loss")),
   #              "num_speakers": ck.get("model_config", {}).get("num_speakers", 0),
   #              "training_time_hours": safe_float(ck.get("training_time_hours", 0))
   #          }
   #      except Exception:
   #          pass

    # Sanitizare float-uri din training_progress (poate conține inf/nan)
   # def sanitize_progress(d):
   #     if isinstance(d, dict):
   #         return {k: sanitize_progress(v) for k, v in d.items()}
   #     elif isinstance(d, list):
   #         return [sanitize_progress(v) for v in d]
   #     elif isinstance(d, float):
   #         return safe_float(d)
   #     return d

   # training_progress = sanitize_progress(training_progress)

   # return {
   #      "is_trained": is_trained,
   #      "is_training": training_progress.get("status") == "running",
   #      "training_progress": training_progress,
   #      "checkpoint_info": checkpoint_info,
   #      "checkpoint_path": str(checkpoint) if is_trained else None
   #  }


#@app.post("/api/lightvc/train")
#async def start_training(
   # background_tasks: BackgroundTasks,
   # epochs: int = Form(100),
   # n_speakers: int = Form(10),
   # batch_size: int = Form(16)
#):
   # """Pornire antrenare LightVC în background."""
   # global _trainer, _trainer_thread

    # Verifică dacă o antrenare e deja în curs
   # if _trainer_thread and _trainer_thread.is_alive():
   #     raise HTTPException(409, "O antrenare este deja in curs!")

   # from voice_conversion.config import LIGHTVC_CFG
   # from voice_conversion.training.trainer import LightVCTrainer

   # cfg = LIGHTVC_CFG
   # cfg.num_epochs = epochs
   # cfg.n_pseudo_speakers = n_speakers
   #    cfg.batch_size = batch_size

    #_trainer = LightVCTrainer(cfg)

   # def run_training():
    #    ok = _trainer.setup()
   #     if ok:
   #         _trainer.train()

    #_trainer_thread = threading.Thread(target=run_training, daemon=True)
    #_trainer_thread.start()

    #return {
    #    "success": True,
    #    "message": f"Antrenare pornita: {epochs} epoci, {n_speakers} vorbitori",
    #    "epochs": epochs
    #}


#@app.get("/api/lightvc/train/progress")
#async def get_training_progress():
    #"""Starea curentă a antrenării (polling)."""
    #global _trainer

    #if _trainer is not None:
    #    return _trainer.get_progress()

    # Citire din fișier (persistence după restart)
   # from voice_conversion.training.trainer import LightVCTrainer
    #return LightVCTrainer.load_progress(str(LIGHTVC_CHECKPOINT_DIR))


#@app.post("/api/lightvc/train/stop")
#async def stop_training():
 #   """Oprire antrenare."""
    #global _trainer
    #if _trainer:
    #    _trainer.stop()
    #    return {"success": True, "message": "Semnal de oprire trimis."}
    
    # Dacă ajungem aici înseamnă că nu există un _trainer activ în memorie
    # Dar poate fi un status blocat în JSON
   # progress_file = LIGHTVC_CHECKPOINT_DIR / "training_progress.json"
   # if progress_file.exists():
   #     try:
   #         with open(progress_file, 'r') as f:
   #             data = json.load(f)
   #         if data.get("status") == "running":
   #             data["status"] = "stopped"
   #             data["message"] = "Antrenare oprita (resetat manual)"
   #             with open(progress_file, 'w') as f:
   #                 json.dump(data, f, indent=2)
   #             return {"success": True, "message": "Statusul blocat a fost resetat la oprit."}
   #     except Exception as e:
    #        logger.error(f"Eroare la resetarea statusului blocat: {e}")
            
    #raise HTTPException(404, "Nicio antrenare activa.")


# =====================================================================
# ENDPOINTS — FREEVC (SpeechT5) TRAINING
# =====================================================================

_freevc_trainer_thread = None

@app.get("/api/freevc/status")
async def freevc_status():
    """Starea antrenamentului pentru FreeVC."""
    checkpoint_dir = Path("checkpoints/freevc_finetuned")
    progress_file = checkpoint_dir / "training_progress.json"

    is_trained = (checkpoint_dir / "pytorch_model.bin").exists() or (checkpoint_dir / "model.safetensors").exists()
    
    training_progress = {}
    if progress_file.exists():
        try:
            with open(progress_file) as f:
                training_progress = json.load(f)
        except Exception:
            pass

    return {
        "is_trained": is_trained,
        "is_training": training_progress.get("status") == "running",
        "training_progress": training_progress
    }


@app.post("/api/freevc/train")
async def start_freevc_training(background_tasks: BackgroundTasks):
    """Pornire antrenare FreeVC în background."""
    global _freevc_trainer_thread

    if _freevc_trainer_thread and _freevc_trainer_thread.is_alive():
        raise HTTPException(409, "O antrenare FreeVC este deja in curs!")

    from voice_conversion.models.freevc_trainer import FreeVCTrainer

    def run_training():
        trainer = FreeVCTrainer()
        trainer.train()

    _freevc_trainer_thread = threading.Thread(target=run_training, daemon=True)
    _freevc_trainer_thread.start()

    return {
        "success": True,
        "message": "Antrenare SpeechT5 (FreeVC) pornită."
    }

# =====================================================================
# ENDPOINTS — ANTRENARE XTTS v2 
# =====================================================================

@app.get("/api/xtts/finetune/status")
async def get_xtts_finetune_status():
    from voice_conversion.models.xtts_trainer import get_xtts_training_progress
    return get_xtts_training_progress()


@app.post("/api/xtts/finetune")
async def start_xtts_finetuning(background_tasks: BackgroundTasks):
    from voice_conversion.models.xtts_trainer import get_xtts_training_progress, XTTSTrainer
    
    status = get_xtts_training_progress()
    if status.get("status") in ["training", "cleaning", "preparing"]:
        raise HTTPException(409, "Un proces XTTS rulează deja!")
        
    def _train_task():
        trainer = XTTSTrainer()
        trainer.train()
        
    background_tasks.add_task(_train_task)
    return {"success": True, "message": "Fine-Tuning XTTS v2 a pornit în background."}


# =====================================================================
# ENDPOINTS — ANTRENARE RVC PER SPEAKER
# =====================================================================

@app.get("/api/rvc/status")
async def get_rvc_status():
    from voice_conversion.models.rvc_trainer import get_rvc_training_progress
    return get_rvc_training_progress()


@app.post("/api/rvc/train")
async def start_rvc_training(
    speaker_id: str = Form(...),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    if speaker_id not in speakers_db:
        raise HTTPException(404, "Vorbitor negăsit")
        
    speaker_data = speakers_db[speaker_id]
    audio_files = speaker_data.get("audio_files", [])
    
    if not audio_files:
        raise HTTPException(400, "Vorbitorul nu are fișiere audio!")
        
    from voice_conversion.models.rvc_trainer import get_rvc_training_progress, RVCTrainer
    status = get_rvc_training_progress()
    
    if status.get("status") in ["preparing", "training"]:
        raise HTTPException(409, "O antrenare RVC rulează deja!")
        
    def _train_rvc():
        trainer = RVCTrainer()
        trainer.train(speaker_id=speaker_id, audio_files=audio_files)
        
    background_tasks.add_task(_train_rvc)
    return {"success": True, "message": f"Antrenare RVC pornită pentru {speaker_data['name']}."}


# =====================================================================
# ENDPOINTS — AUDIO & HISTORY & SYSTEM INFO
# =====================================================================

@app.get("/api/audio/{filename}")
async def serve_audio(filename: str):
    for directory in [WEBAPP_OUTPUTS, WEBAPP_UPLOADS]:
        filepath = directory / filename
        if filepath.exists():
            return FileResponse(str(filepath), media_type="audio/wav",
                                headers={"Accept-Ranges": "bytes"})
    raise HTTPException(404, "Fisier negasit!")


@app.get("/api/history")
async def get_history():
    return {
        "history": list(reversed(conversion_history[-50:])),
        "total": len(conversion_history)
    }


@app.get("/api/system-info")
async def get_system_info():
    import torch
    
    # Verificare CUDA (NVIDIA)
    gpu_available = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if gpu_available else "N/A"
    device_type = "cuda" if gpu_available else "cpu"
    
    # Verificare DirectML (AMD/Intel) pe Windows
    if not gpu_available:
        try:
            import torch_directml
            if torch_directml.is_available():
                gpu_available = True
                gpu_name = "AMD GPU (DirectML)"
                device_type = "privateuseone"
        except ImportError:
            pass
            
    #lightvc_trained = (LIGHTVC_CHECKPOINT_DIR / "best_model.pth").exists()

    return {
        "knn_vc": {
            "status": "ready" if knn_converter else "pre-antrenat",
            "info": "bshall/knn-vc"
        },
        #"mknn_vc": {
        #    "status": "ready" if mknn_converter else "pre-antrenat",
        #    "info": "xls-r-300m (experimental)"
        #},
        #"lightvc": {
        #    "name": "LightVC",
        #    "status": "trained" if lightvc_trained else "not_trained",
        #    "type": "antrenat-de-utilizator"
        #},
        "gpu_available": gpu_available,
        "gpu_name": gpu_name,
        "device": device_type,
        "speakers_registered": len(speakers_db),
        "total_conversions": len(conversion_history),
        "version": "2.0.0"
    }


@app.get("/api/model-info/knn-vc")
async def knn_vc_info():
    return {
        "name": "kNN-VC",
        "full_name": "k-Nearest Neighbors Voice Conversion",
        "paper": "Baas et al. (2023) — arXiv:2305.18975",
        "type": "Non-parametric, Any-to-Any",
        "training_required": False,
        "architecture": {
            "encoder": "WavLM-Large (layer 6, 1024-dim)",
            "matching": "k-Nearest Neighbors (topk=4)",
            "vocoder": "HiFi-GAN V1 (prematched)"
        },
        "performance": {
            "WER": "6.29%",
            "EER": "35.73%",
            "dataset": "LibriSpeech test-clean"
        }
    }


#@app.get("/api/model-info/lightvc")
#async def lightvc_info():
#    lvc = get_lightvc_converter()
#    return {
#        "name": "LightVC",
#        "full_name": "Light Voice Conversion (antrenat pe Common Voice RO)",
#        "type": "Encoder-Decoder cu Information Bottleneck",
#        "training_required": True,
#        "architecture": {
#            "content_encoder": "Conv1D × 3 + BiLSTM + Bottleneck 32-dim",
#            "speaker_encoder": "ECAPA-TDNN (SpeechBrain, frozen)",
#            "decoder": "Conv1D dilated × 5 + BiLSTM + PostNet",
#            "vocoder": "Griffin-Lim / HiFi-GAN"
 #       },
 #       **lvc.get_info()
#    }


# =====================================================================
# ENDPOINT — GRAFICE ACUSTICE ON-DEMAND (DIN RAM)
# =====================================================================

@app.get("/api/plot")
async def get_audio_plot(filename: str, plot_type: str = "spectrogram"):
    """
    Generează on-demand grafice acustice din RAM fără a le salva pe disc.
    plot_type: 'spectrogram', 'mel', 'mfcc'
    """
    # Caută fișierul în outputs, apoi în uploads
    audio_path = WEBAPP_OUTPUTS / filename
    if not audio_path.exists():
        audio_path = WEBAPP_UPLOADS / filename
        
    if not audio_path.exists():
        raise HTTPException(404, f"Fișierul audio nu există: {filename}")
    
    try:
        from voice_conversion.evaluation.plots import generate_spectrogram, generate_mel_spectrogram, generate_mfcc
        
        audio_str = audio_path.resolve().as_posix()
        
        if plot_type == "mel":
            buf = generate_mel_spectrogram(audio_str)
        elif plot_type == "mfcc":
            buf = generate_mfcc(audio_str)
        else:
            buf = generate_spectrogram(audio_str)
            
        return StreamingResponse(buf, media_type="image/png")
    except Exception as e:
        logger.error(f"Eroare generare grafic {plot_type}: {e}")
        raise HTTPException(500, f"Eroare generare grafic: {str(e)}")


# =====================================================================
# STARTUP
# =====================================================================

@app.on_event("startup")
async def startup_event():
    #lightvc_trained = (LIGHTVC_CHECKPOINT_DIR / "best_model.pth").exists()
    logger.info("\n" + "=" * 60)
    logger.info("  VOICE CONVERSION SYSTEM v2.0 — WEB SERVER")
    logger.info("=" * 60)
    logger.info(f"  Frontend:  http://localhost:8000")
    logger.info(f"  API Docs:  http://localhost:8000/docs")
    logger.info(f"  Vorbitori: {len(speakers_db)}")
    logger.info(f"  kNN-VC:    ready (pre-antrenat)")
   # logger.info(f"  LightVC:   {'trained' if lightvc_trained else 'not trained yet'}")
    logger.info("=" * 60 + "\n")


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True, log_level="info")
