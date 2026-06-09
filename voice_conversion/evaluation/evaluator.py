"""
Evaluator Complet — Orchestrare Evaluare Voice Conversion
==========================================================

Clasă de nivel înalt care orchestrează:
- Evaluarea individuală a conversiilor
- Evaluarea batch (mai multe perechi sursă-target)
- Generare rapoarte cu tabele și grafice
- Export rezultate în format CSV pentru teză
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Union, Optional
from datetime import datetime
import json
import logging

from voice_conversion.evaluation.metrics import compute_all_metrics
from voice_conversion.config import EVALUATION_DIR, FIGURES_DIR

logger = logging.getLogger(__name__)


class VoiceConversionEvaluator:
    """
    Evaluator complet pentru sistemul de conversie a vocii.

    Colectează rezultatele evaluărilor, generează rapoarte
    și creează vizualizări pentru lucrarea de master.

    Exemplu:
        evaluator = VoiceConversionEvaluator()
        evaluator.evaluate_single(
            source="sursa.wav",
            converted="convertit.wav",
            target_ref="target.wav",
            label="kNN-VC k=4"
        )
        evaluator.generate_report()
    """

    def __init__(self, speaker_model=None):
        """
        Inițializare evaluator.

        Args:
            speaker_model: Model ECAPA-TDNN pre-încărcat (opțional)
        """
        self.results = []
        self.speaker_model = speaker_model
        self._model_loaded = False

        logger.info("📊 VoiceConversionEvaluator inițializat")

    def _ensure_speaker_model(self):
        """Încarcă modelul speaker recognition dacă nu e încărcat."""
        if self.speaker_model is not None:
            return

        if self._model_loaded:
            return

        try:
            try:
                from speechbrain.pretrained import SpeakerRecognition
            except ImportError:
                from speechbrain.inference.speaker import SpeakerRecognition

            self.speaker_model = SpeakerRecognition.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                savedir="./models_cache/spkrec-ecapa-voxceleb"
            )
            self._model_loaded = True
            logger.info("✅ ECAPA-TDNN încărcat pentru evaluare speaker similarity")
        except Exception as e:
            logger.warning(f"⚠️ Nu s-a putut încărca ECAPA-TDNN: {e}")
            self._model_loaded = True  # Nu mai încerca

    def evaluate_single(
        self,
        source: Union[str, Path],
        converted: Union[str, Path],
        target_ref: Union[str, Path],
        label: str = "",
        model_name: str = "kNN-VC",
        metadata: Optional[Dict] = None
    ) -> Dict:
        """
        Evaluează o singură conversie.

        Args:
            source: Audio sursă
            converted: Audio convertit
            target_ref: Audio referință target speaker
            label: Etichetă descriptivă
            model_name: Numele modelului folosit
            metadata: Metadate adiționale

        Returns:
            Dict cu toate metricile
        """
        self._ensure_speaker_model()

        logger.info(f"\n🔍 Evaluare: {label or Path(converted).name}")

        metrics = compute_all_metrics(
            source_audio=str(source),
            converted_audio=str(converted),
            target_reference=str(target_ref),
            speaker_model=self.speaker_model
        )

        # Adaugă metadate
        result = {
            "timestamp": datetime.now().isoformat(),
            "model": model_name,
            "label": label,
            "source": str(source),
            "converted": str(converted),
            "target_ref": str(target_ref),
            **metrics,
            **(metadata or {})
        }

        self.results.append(result)
        return result

    def evaluate_batch(
        self,
        conversions: List[Dict],
        model_name: str = "kNN-VC"
    ) -> pd.DataFrame:
        """
        Evaluare batch a mai multor conversii.

        Args:
            conversions: Lista de dict-uri cu cheile:
                         source, converted, target_ref, label (opțional)
            model_name: Numele modelului

        Returns:
            pd.DataFrame cu rezultate
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"  EVALUARE BATCH — {model_name}")
        logger.info(f"  {len(conversions)} conversii")
        logger.info(f"{'='*60}\n")

        for i, conv in enumerate(conversions, 1):
            logger.info(f"[{i}/{len(conversions)}]")
            self.evaluate_single(
                source=conv["source"],
                converted=conv["converted"],
                target_ref=conv["target_ref"],
                label=conv.get("label", f"sample_{i}"),
                model_name=model_name,
                metadata=conv.get("metadata", {})
            )

        return self.get_results_dataframe()

    def get_results_dataframe(self) -> pd.DataFrame:
        """Returnează rezultatele ca DataFrame."""
        if not self.results:
            return pd.DataFrame()

        return pd.DataFrame(self.results)

    def get_summary_statistics(self) -> Dict:
        """
        Calculează statistici sumarizate pentru toate evaluările.

        Returns:
            Dict cu mean, std, min, max pentru fiecare metrică
        """
        df = self.get_results_dataframe()
        if df.empty:
            return {}

        metric_cols = ["mcd", "pesq", "speaker_similarity",
                       "f0_rmse", "f0_pcc", "snr"]

        summary = {}
        for col in metric_cols:
            if col in df.columns:
                valid = df[col].dropna()
                if len(valid) > 0:
                    summary[col] = {
                        "mean": float(valid.mean()),
                        "std": float(valid.std()),
                        "min": float(valid.min()),
                        "max": float(valid.max()),
                        "median": float(valid.median()),
                        "count": int(len(valid))
                    }

        return summary

    def generate_report(
        self,
        output_dir: Union[str, Path] = None,
        include_plots: bool = True
    ) -> Path:
        """
        Generează raport complet de evaluare.

        Creează:
        - Tabel CSV cu toate rezultatele
        - Tabel sumarizat (medie ± std)
        - Grafice (dacă include_plots=True)
        - Raport text

        Args:
            output_dir: Director output
            include_plots: Include grafice matplotlib

        Returns:
            Path: Directorul cu raportul generat
        """
        output_dir = Path(output_dir or EVALUATION_DIR)
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        df = self.get_results_dataframe()
        if df.empty:
            logger.warning("⚠️ Nu există rezultate de evaluat!")
            return output_dir

        # 1. CSV cu toate rezultatele
        csv_path = output_dir / f"evaluation_results_{timestamp}.csv"
        df.to_csv(csv_path, index=False)
        logger.info(f"📄 CSV salvat: {csv_path}")

        # 2. Statistici sumarizate
        summary = self.get_summary_statistics()
        summary_path = output_dir / f"evaluation_summary_{timestamp}.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        logger.info(f"📄 Summary salvat: {summary_path}")

        # 3. Raport text
        report_path = output_dir / f"evaluation_report_{timestamp}.txt"
        self._write_text_report(report_path, df, summary)
        logger.info(f"📄 Raport salvat: {report_path}")

        # 4. Grafice
        if include_plots:
            try:
                self._generate_plots(output_dir, df, timestamp)
            except Exception as e:
                logger.warning(f"⚠️ Eroare la generarea graficelor: {e}")

        logger.info(f"\n✅ Raport complet generat în: {output_dir}")
        return output_dir

    def _write_text_report(self, filepath: Path, df: pd.DataFrame, summary: Dict):
        """Scrie raport text detaliat."""
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("=" * 70 + "\n")
            f.write("  RAPORT DE EVALUARE — SISTEM CONVERSIE VOCE\n")
            f.write(f"  Data: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 70 + "\n\n")

            f.write(f"Total evaluări: {len(df)}\n")
            if 'model' in df.columns:
                f.write(f"Modele evaluate: {df['model'].unique().tolist()}\n")
            f.write("\n")

            # Tabel sumarizat
            f.write("-" * 70 + "\n")
            f.write("STATISTICI SUMARIZATE\n")
            f.write("-" * 70 + "\n\n")

            metric_labels = {
                "mcd": ("MCD (dB)", "↓ mai mic = mai bine"),
                "pesq": ("PESQ", "↑ mai mare = mai bine"),
                "speaker_similarity": ("Speaker Sim.", "↑ mai mare = mai bine"),
                "f0_rmse": ("F0 RMSE (Hz)", "↓ mai mic = mai bine"),
                "f0_pcc": ("F0 PCC", "↑ mai mare = mai bine"),
                "snr": ("SNR (dB)", "↑ mai mare = mai bine")
            }

            f.write(f"{'Metrică':<25s} {'Media':>10s} {'± Std':>10s} "
                    f"{'Min':>10s} {'Max':>10s} {'Direcție':<25s}\n")
            f.write("-" * 90 + "\n")

            for metric, stats in summary.items():
                label, direction = metric_labels.get(
                    metric, (metric, ""))
                f.write(
                    f"{label:<25s} "
                    f"{stats['mean']:>10.4f} "
                    f"{stats['std']:>10.4f} "
                    f"{stats['min']:>10.4f} "
                    f"{stats['max']:>10.4f} "
                    f"{direction:<25s}\n"
                )

            f.write("\n" + "=" * 70 + "\n")

    def _generate_plots(self, output_dir: Path, df: pd.DataFrame, timestamp: str):
        """Generează grafice matplotlib pentru raport."""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import seaborn as sns

        sns.set_theme(style="whitegrid", palette="husl")

        figures_dir = output_dir / "figures"
        figures_dir.mkdir(exist_ok=True)

        metric_info = {
            "mcd": {"label": "MCD (dB)", "lower_better": True},
            "pesq": {"label": "PESQ Score", "lower_better": False},
            "speaker_similarity": {"label": "Speaker Similarity", "lower_better": False},
            "f0_rmse": {"label": "F0 RMSE (Hz)", "lower_better": True},
            "f0_pcc": {"label": "F0 Pearson Correlation", "lower_better": False},
            "snr": {"label": "SNR (dB)", "lower_better": False}
        }

        # 1. Bar chart cu metrici medii
        available_metrics = [m for m in metric_info if m in df.columns
                             and df[m].notna().any()]

        if available_metrics:
            fig, axes = plt.subplots(2, 3, figsize=(18, 10))
            axes = axes.flatten()

            for idx, metric in enumerate(available_metrics):
                if idx >= len(axes):
                    break

                info = metric_info[metric]
                values = df[metric].dropna()

                color = '#e74c3c' if info["lower_better"] else '#2ecc71'
                axes[idx].bar(range(len(values)), values, color=color, alpha=0.7)
                axes[idx].axhline(y=values.mean(), color='black',
                                  linestyle='--', alpha=0.5,
                                  label=f'Mean: {values.mean():.3f}')
                axes[idx].set_title(info["label"], fontsize=14, fontweight='bold')
                axes[idx].legend()
                axes[idx].set_xlabel("Sample")
                axes[idx].set_ylabel(info["label"])

            # Ascunde axe neutilizate
            for idx in range(len(available_metrics), len(axes)):
                axes[idx].set_visible(False)

            plt.suptitle("Metrici Evaluare Voice Conversion",
                         fontsize=16, fontweight='bold')
            plt.tight_layout()
            plt.savefig(figures_dir / f"metrics_overview_{timestamp}.png",
                        dpi=300, bbox_inches='tight')
            plt.close()

            logger.info(f"📈 Grafic salvat: metrics_overview_{timestamp}.png")

        # 2. Box plot comparativ
        if len(available_metrics) >= 2:
            fig, ax = plt.subplots(figsize=(12, 6))

            plot_data = []
            for metric in available_metrics:
                values = df[metric].dropna()
                for v in values:
                    plot_data.append({
                        "Metric": metric_info[metric]["label"],
                        "Value": v
                    })

            plot_df = pd.DataFrame(plot_data)
            sns.boxplot(data=plot_df, x="Metric", y="Value", ax=ax)
            ax.set_title("Distribuția Metricilor", fontsize=14, fontweight='bold')
            plt.xticks(rotation=15)
            plt.tight_layout()
            plt.savefig(figures_dir / f"metrics_boxplot_{timestamp}.png",
                        dpi=300, bbox_inches='tight')
            plt.close()

            logger.info(f"📈 Grafic salvat: metrics_boxplot_{timestamp}.png")

    def __repr__(self):
        return f"VoiceConversionEvaluator(results={len(self.results)})"
