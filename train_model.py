"""Train and evaluate the FMA Small Random Forest genre classifier."""

from ml_model import MODEL_PATH, train_from_fma


def main() -> None:
    """main: Latih model FMA dan cetak pembagian data serta hasil evaluasinya."""
    def report_progress(fraction: float, message: str) -> None:
        """report_progress: Cetak perkembangan tahap pelatihan secara langsung ke terminal."""
        print(f"[{fraction:6.1%}] {message}", flush=True)

    results = train_from_fma(progress=report_progress)
    split_counts = results["data_splits"]
    total = split_counts["total"]
    print("\nPembagian dataset FMA Small (80:10:10):")
    for label, key in (
        ("Training", "training"),
        ("Validasi", "validation"),
        ("Testing", "testing"),
    ):
        count = split_counts[key]
        print(f"  {label:<10} {count:>5,} track ({count / total:.2%})")
    print(f"  {'Total':<10} {total:>5,} track (100.00%)")
    print(f"Genre: {', '.join(results['classes'])}")
    print(f"Skenario terbaik: {results['scenario']}")
    print(
        "Macro F1 validasi: "
        f"{results['validation_metrics']['f1_macro']:.2%}"
    )
    print(f"Aturan pemilihan: {results['selection_rule']}")
    print(f"Hyperparameter terbaik: {results['best_params']}")
    for code, scenario in results["scenario_results"].items():
        metrics = scenario["test_metrics"]
        print(f"\nHasil skenario {code} ({scenario['name']}):")
        print(f"  Accuracy          : {metrics['accuracy']:.2%}")
        print(f"  Precision (macro) : {metrics['precision']:.2%}")
        print(f"  Recall (macro)    : {metrics['recall']:.2%}")
        print(f"  F1-score (macro)  : {metrics['f1_macro']:.2%}")
    print(f"Model tersimpan di: {MODEL_PATH}")


if __name__ == "__main__":
    main()
