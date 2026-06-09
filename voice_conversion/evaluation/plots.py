import os
import io
import librosa
import librosa.display
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Setăm matplotlib pe un backend non-interactiv (server-safe)
# Asta previne deschiderea de ferestre GUI și scurgerile de memorie
matplotlib.use('Agg')

def get_audio_data(audio_path, sr=16000):
    """Încarcă semnalul audio dintr-un fișier."""
    path_str = Path(audio_path).resolve().as_posix()
    y, sr = librosa.load(path_str, sr=sr)
    return y, sr

def generate_spectrogram(audio_path: str) -> io.BytesIO:
    """Generează o spectrogramă liniară."""
    y, sr = get_audio_data(audio_path)
    
    # Calculăm Short-Time Fourier Transform (STFT)
    D = librosa.stft(y)
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    
    fig, ax = plt.subplots(figsize=(8, 3))
    img = librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='hz', ax=ax, cmap='magma')
    ax.set_title('Spectrogramă Liniară')
    ax.set_xlabel('Timp (s)')
    ax.set_ylabel('Frecvență (Hz)')
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=120, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return buf

def generate_mel_spectrogram(audio_path: str) -> io.BytesIO:
    """Generează o spectrogramă Mel (apropiată de percepția umană)."""
    y, sr = get_audio_data(audio_path)
    
    # Calculăm Mel Spectrogram
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
    S_dB = librosa.power_to_db(S, ref=np.max)
    
    fig, ax = plt.subplots(figsize=(8, 3))
    img = librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', sr=sr, fmax=8000, ax=ax, cmap='viridis')
    ax.set_title('Mel-Spectrogramă (Percepție Umană)')
    ax.set_xlabel('Timp (s)')
    ax.set_ylabel('Frecvență Mel')
    fig.colorbar(img, ax=ax, format='%+2.0f dB')
    
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=120, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return buf

def generate_mfcc(audio_path: str) -> io.BytesIO:
    """Generează coeficienții cepstrali (MFCC), esențiali pentru recunoașterea vocii."""
    y, sr = get_audio_data(audio_path)
    
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    
    fig, ax = plt.subplots(figsize=(8, 3))
    img = librosa.display.specshow(mfccs, x_axis='time', ax=ax, cmap='coolwarm')
    ax.set_title('MFCC (Amprenta Vocală)')
    ax.set_xlabel('Timp (s)')
    ax.set_ylabel('Coeficienți MFCC')
    fig.colorbar(img, ax=ax)
    
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=120, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return buf
