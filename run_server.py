"""
Script de Pornire — Voice Conversion Web App
=============================================

Pornește serverul FastAPI care servește:
- Frontend web (http://localhost:8000)
- API REST (http://localhost:8000/docs)

Utilizare:
    python run_server.py
"""

import sys
import os
from pathlib import Path

# Configurare path-uri
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# Setare variabile de mediu
os.environ.setdefault('PYTHONDONTWRITEBYTECODE', '1')


def check_dependencies():
    """Verifică dependențele necesare."""
    missing = []
    
    try:
        import torch
        print(f"✅ PyTorch {torch.__version__}")
        if torch.cuda.is_available():
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
        else:
            print(f"   ⚠️  GPU indisponibil, se folosește CPU")
    except ImportError:
        missing.append("torch")

    try:
        import torchaudio
        print(f"✅ torchaudio {torchaudio.__version__}")
    except ImportError:
        missing.append("torchaudio")

    try:
        import fastapi
        print(f"✅ FastAPI {fastapi.__version__}")
    except ImportError:
        missing.append("fastapi")

    try:
        import uvicorn
        print(f"✅ uvicorn disponibil")
    except ImportError:
        missing.append("uvicorn[standard]")

    try:
        import librosa
        print(f"✅ librosa {librosa.__version__}")
    except ImportError:
        missing.append("librosa")

    if missing:
        print(f"\n❌ Dependențe lipsă: {', '.join(missing)}")
        print(f"\nInstalare:")
        print(f"  pip install {' '.join(missing)}")
        print(f"\nSau instalează toate dependențele:")
        print(f"  pip install -r requirements_vc.txt")
        return False

    return True


def main():
    print("\n" + "=" * 60)
    print("  🎤 VOICE CONVERSION SYSTEM — WEB SERVER")
    print("=" * 60 + "\n")
    
    # Verificare dependențe
    print("📋 Verificare dependențe...\n")
    if not check_dependencies():
        sys.exit(1)
    
    print("\n" + "-" * 60)
    print("  🚀 Pornire server...")
    print("-" * 60)
    print(f"\n  🌐 Frontend: http://localhost:8000")
    print(f"  📚 API Docs: http://localhost:8000/docs")
    print(f"  📂 Uploads:  {PROJECT_ROOT / 'webapp' / 'uploads'}")
    print(f"\n  Ctrl+C pentru oprire\n")
    
    import uvicorn
    
    # Pornire server
    uvicorn.run(
        "webapp.backend.app:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info",
        access_log=True
    )


if __name__ == "__main__":
    main()
