"""
Demo Script — Conversie Voce cu kNN-VC
=======================================

Demonstrare conversie voce din linia de comandă.

Utilizare:
    python scripts/demo_conversion.py --source audio_sursa.wav \
                                       --target ref1.wav ref2.wav \
                                       --output convertit.wav

Sau interactiv (fără argumente):
    python scripts/demo_conversion.py
"""

import sys
import argparse
from pathlib import Path

# Setup path
sys.path.insert(0, str(Path(__file__).parent.parent))

from voice_conversion.models.knn_vc import KnnVoiceConverter
from voice_conversion.evaluation.metrics import compute_all_metrics
from voice_conversion.evaluation.evaluator import VoiceConversionEvaluator
from voice_conversion.utils.audio_utils import get_audio_info, list_audio_files
from voice_conversion.config import CONVERTED_AUDIO_DIR, RESULTS_DIR


def demo_interactive():
    """Mod interactiv — ghidare pas cu pas."""
    print("\n" + "=" * 60)
    print("  🎤 DEMO CONVERSIE VOCE — kNN-VC")
    print("=" * 60)
    
    print("""
    Acest demo convertește vocea dintr-un fișier audio sursă
    ca să sune ca un alt vorbitor (target).

    Ai nevoie de:
    1. Un fișier audio SURSĂ (vocea de convertit)
    2. Unul sau mai multe fișiere audio REFERINȚĂ (vocea țintă)
    """)
    
    # Input
    source = input("  📎 Calea fișierului audio SURSĂ: ").strip().strip('"')
    if not Path(source).exists():
        print(f"  ❌ Fișierul nu există: {source}")
        return
    
    targets_str = input("  📎 Calea fișierelor REFERINȚĂ (separate prin virgulă): ").strip()
    targets = [t.strip().strip('"') for t in targets_str.split(",")]
    
    valid_targets = [t for t in targets if Path(t).exists()]
    if not valid_targets:
        print(f"  ❌ Niciun fișier de referință valid!")
        return
    
    topk = input("  🔢 Parametru k (default: 4): ").strip() or "4"
    topk = int(topk)
    
    output = input("  💾 Fișier output (default: auto): ").strip() or None
    
    # Conversie
    run_conversion(source, valid_targets, topk, output)


def run_conversion(source, targets, topk=4, output=None, evaluate=True):
    """Rulează o conversie completă."""
    
    print(f"\n{'─' * 60}")
    print(f"  Sursă:     {Path(source).name}")
    print(f"  Referințe:  {[Path(t).name for t in targets]}")
    print(f"  k:          {topk}")
    print(f"{'─' * 60}\n")
    
    # Info sursă
    info = get_audio_info(source)
    print(f"  📊 Audio sursă: {info['duration']:.2f}s, {info['sample_rate']}Hz\n")
    
    # Inițializare converter
    print("  🔧 Inițializare kNN-VC...")
    converter = KnnVoiceConverter()
    converter.load_model()
    
    # Conversie
    print("\n  🔄 Conversie în curs...")
    result = converter.convert(
        source_audio=source,
        target_references=targets,
        topk=topk,
        output_path=output
    )
    
    # Salvare
    if output is None:
        saved_path = result.save()
    else:
        saved_path = Path(output)
    
    print(f"\n  ✅ Conversie completă!")
    print(f"  💾 Salvat: {saved_path}")
    print(f"  ⏱️  Timp: {result.conversion_time:.2f}s")
    print(f"  📏 Durată output: {result.get_duration():.2f}s")
    
    # Evaluare
    if evaluate:
        print(f"\n{'─' * 60}")
        print(f"  📊 EVALUARE CALITATE")
        print(f"{'─' * 60}\n")
        
        evaluator = VoiceConversionEvaluator()
        eval_result = evaluator.evaluate_single(
            source=source,
            converted=str(saved_path),
            target_ref=targets[0],
            label="kNN-VC Demo",
            model_name="kNN-VC"
        )
        
        # Afișare rezultate
        print(f"\n  {'='*50}")
        print(f"  {'Metrică':<25s} {'Valoare':>10s} {'Calitate':>10s}")
        print(f"  {'─'*50}")
        
        thresholds = {
            "mcd": (6.0, "lower"),
            "pesq": (3.0, "higher"),
            "speaker_similarity": (0.7, "higher"),
            "f0_rmse": (20.0, "lower"),
            "f0_pcc": (0.8, "higher"),
            "snr": (15.0, "higher")
        }
        
        metric_labels = {
            "mcd": "MCD (dB)",
            "pesq": "PESQ",
            "speaker_similarity": "Speaker Similarity",
            "f0_rmse": "F0 RMSE (Hz)",
            "f0_pcc": "F0 Correlation",
            "f0_mean_diff": "F0 Mean Diff (Hz)",
            "snr": "SNR (dB)"
        }
        
        for metric, label in metric_labels.items():
            val = eval_result.get(metric)
            if val is None or (isinstance(val, float) and val != val):  # NaN check
                continue
            
            quality = ""
            if metric in thresholds:
                thresh, direction = thresholds[metric]
                if direction == "lower":
                    quality = "✅ Bine" if val <= thresh else "⚠️ Slab"
                else:
                    quality = "✅ Bine" if val >= thresh else "⚠️ Slab"
            
            print(f"  {label:<25s} {val:>10.4f} {quality:>10s}")
        
        print(f"  {'='*50}")
        
        # Generare raport
        report_dir = evaluator.generate_report(include_plots=True)
        print(f"\n  📄 Raport salvat în: {report_dir}")
    
    print(f"\n{'=' * 60}\n")
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Demo conversie voce cu kNN-VC"
    )
    parser.add_argument("--source", "-s", type=str,
                       help="Fișier audio sursă")
    parser.add_argument("--target", "-t", type=str, nargs="+",
                       help="Fișier(e) audio referință target")
    parser.add_argument("--output", "-o", type=str, default=None,
                       help="Fișier output (default: auto)")
    parser.add_argument("--topk", "-k", type=int, default=4,
                       help="Parametru k pentru kNN (default: 4)")
    parser.add_argument("--no-eval", action="store_true",
                       help="Nu evalua calitatea")
    
    args = parser.parse_args()
    
    if args.source and args.target:
        run_conversion(
            source=args.source,
            targets=args.target,
            topk=args.topk,
            output=args.output,
            evaluate=not args.no_eval
        )
    else:
        demo_interactive()


if __name__ == "__main__":
    main()
