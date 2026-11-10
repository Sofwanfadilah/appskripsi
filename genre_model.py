"""Audio feature extraction and Random Forest music-genre classification."""

from __future__ import annotations

import io
import json
import math
import os
import pickle
import shutil
import tempfile
import zipfile
from contextlib import nullcontext
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from scipy.fft import dct
from scipy.signal import find_peaks, resample_poly
from scipy.stats import kurtosis, skew
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
import soundfile


PROJECT_ROOT = Path(__file__).resolve().parent
FMA_ROOT = PROJECT_ROOT / "fma_small"
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODEL_DIR / "random_forest_genre_model.pkl"
RHYTHM_CACHE_PATH = FMA_ROOT / "rhythm_features.csv"
DATASET_DIR = FMA_ROOT / "datasets"
FINAL_DATASET_PATH = DATASET_DIR / "fma_small_final.csv"
DATASET_AUDIT_PATH = DATASET_DIR / "dataset_audit.csv"
EXPERIMENT_RESULTS_PATH = MODEL_DIR / "feature_scenario_results.json"
STATS = ("mean", "std", "skew", "kurtosis", "median", "min", "max")
GENRES = (
    "Electronic",
    "Experimental",
    "Folk",
    "Hip-Hop",
    "Instrumental",
    "International",
    "Pop",
    "Rock",
)
FEATURE_GROUPS = {
    "A": {
        "name": "Energi/amplitudo",
        "prefixes": ("rmse_", "zcr_"),
    },
    "B": {
        "name": "Spektral",
        "prefixes": (
            "spectral_centroid_",
            "spectral_bandwidth_",
            "spectral_rolloff_",
            "spectral_contrast_",
        ),
    },
    "C": {
        "name": "Timbre (MFCC)",
        "prefixes": ("mfcc_",),
    },
    "D": {
        "name": "Harmonik",
        "prefixes": (
            "chroma_stft_",
            "chroma_cqt_",
            "chroma_cens_",
            "tonnetz_",
        ),
    },
    "E": {
        "name": "Ritmis",
        "prefixes": ("onset_strength_", "bpm"),
    },
}
SCENARIOS = {
    "S1": ("C",),
    "S2": ("A", "B", "D", "E"),
    "S3": ("A", "B", "C", "D", "E"),
}
DATASET_METADATA_COLUMNS = ("track_id", "artist_id", "genre_top", "split")
DATASET_SPECS = {
    "S1": ("dataset_S1_mfcc.csv", "MFCC saja"),
    "S2": ("dataset_S2_fitur_audio.csv", "Fitur audio saja"),
    "S3": ("dataset_S3_mfcc_dan_audio.csv", "MFCC + fitur audio"),
}


def _statistics(values: np.ndarray) -> dict[str, float]:
    """_statistics: Hitung statistik deskriptif untuk satu deret nilai fitur."""
    values = np.asarray(values, dtype=float).reshape(-1)
    if values.size == 0:
        raise ValueError("Ekstraksi fitur menghasilkan deret kosong.")

    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "skew": float(np.nan_to_num(skew(values), nan=0.0)),
        "kurtosis": float(np.nan_to_num(kurtosis(values), nan=0.0)),
        "median": float(np.median(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def _load_waveform(audio: str | Path | bytes) -> tuple[np.ndarray, int]:
    """_load_waveform: Baca, ubah ke mono, resampling, dan normalisasi audio 30 detik."""
    source = io.BytesIO(audio) if isinstance(audio, bytes) else str(audio)
    waveform, sample_rate = soundfile.read(
        source,
        dtype="float32",
        always_2d=True,
    )
    if waveform.size == 0:
        raise ValueError("File audio tidak memiliki sampel yang dapat dianalisis.")
    waveform = np.mean(waveform, axis=1)
    target_rate = 22050
    if sample_rate != target_rate:
        divisor = math.gcd(sample_rate, target_rate)
        waveform = resample_poly(
            waveform,
            target_rate // divisor,
            sample_rate // divisor,
        )
    waveform = waveform[: target_rate * 30]
    if waveform.size < target_rate * 30:
        waveform = np.pad(waveform, (0, target_rate * 30 - waveform.size))
    peak = float(np.max(np.abs(waveform)))
    if peak > 0:
        waveform = waveform / peak
    return waveform.astype(np.float32, copy=False), target_rate


def _stft_magnitude(
    waveform: np.ndarray,
    n_fft: int = 2048,
    hop_length: int = 512,
) -> np.ndarray:
    """_stft_magnitude: Hitung magnitudo STFT dari waveform dengan window Hann."""
    padded = np.pad(
        waveform,
        (n_fft // 2, n_fft // 2),
        mode="reflect",
    )
    frames = np.lib.stride_tricks.sliding_window_view(padded, n_fft)[::hop_length]
    window = np.hanning(n_fft + 1)[:-1]
    return np.abs(np.fft.rfft(frames * window, axis=1)).T


def _mel_filterbank(
    sample_rate: int,
    n_fft: int,
    n_mels: int = 128,
) -> np.ndarray:
    """_mel_filterbank: Bangun filter bank Mel segitiga untuk spektrum audio."""
    def hz_to_mel(frequency: np.ndarray) -> np.ndarray:
        """hz_to_mel: Ubah frekuensi Hertz ke skala Mel."""
        return 2595.0 * np.log10(1.0 + frequency / 700.0)

    def mel_to_hz(mel: np.ndarray) -> np.ndarray:
        """mel_to_hz: Ubah skala Mel kembali menjadi frekuensi Hertz."""
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

    mel_points = np.linspace(
        hz_to_mel(np.array([0.0]))[0],
        hz_to_mel(np.array([sample_rate / 2]))[0],
        n_mels + 2,
    )
    frequencies = np.fft.rfftfreq(n_fft, 1 / sample_rate)
    mel_frequencies = mel_to_hz(mel_points)
    lower = (frequencies[None, :] - mel_frequencies[:-2, None]) / (
        mel_frequencies[1:-1, None] - mel_frequencies[:-2, None]
    )
    upper = (mel_frequencies[2:, None] - frequencies[None, :]) / (
        mel_frequencies[2:, None] - mel_frequencies[1:-1, None]
    )
    filters = np.maximum(0.0, np.minimum(lower, upper))
    filters *= (
        2.0
        / (mel_frequencies[2:] - mel_frequencies[:-2])[:, np.newaxis]
    )
    return filters


def _chroma_from_magnitude(
    magnitude: np.ndarray,
    sample_rate: int,
    n_fft: int,
) -> np.ndarray:
    """_chroma_from_magnitude: Kelompokkan magnitudo spektrum ke dalam 12 kelas pitch chroma."""
    frequencies = np.fft.rfftfreq(n_fft, 1 / sample_rate)
    valid = frequencies > 0
    midi = np.rint(69 + 12 * np.log2(frequencies[valid] / 440.0)).astype(int)
    chroma = np.zeros((12, magnitude.shape[1]), dtype=float)
    for pitch_class in range(12):
        pitch_bins = midi % 12 == pitch_class
        if pitch_bins.any():
            chroma[pitch_class] = magnitude[valid][pitch_bins].sum(axis=0)
    norms = chroma.max(axis=0, keepdims=True)
    return np.divide(chroma, norms, out=np.zeros_like(chroma), where=norms > 0)


def _cqt_chroma(
    waveform: np.ndarray,
    sample_rate: int,
    hop_length: int = 512,
) -> np.ndarray:
    """_cqt_chroma: Perkirakan fitur chroma beresolusi konstan dari waveform."""
    n_fft = 8192
    magnitude = _stft_magnitude(waveform, n_fft, hop_length)
    frequencies = np.fft.rfftfreq(n_fft, 1 / sample_rate)
    valid = frequencies > 0
    frequencies = frequencies[valid]
    spectrum = magnitude[valid]
    quality_factor = 1.0 / (2 ** (1 / 12) - 1)
    note_frequencies = 440 * 2.0 ** (
        (np.arange(24, 108, dtype=float) - 69) / 12
    )
    chroma = np.zeros((12, spectrum.shape[1]), dtype=float)
    for midi, note_frequency in zip(range(24, 108), note_frequencies):
        bandwidth = max(note_frequency / quality_factor, frequencies[1])
        weights = np.exp(
            -0.5 * ((frequencies - note_frequency) / bandwidth) ** 2
        )
        total_weight = float(weights.sum())
        if total_weight:
            chroma[midi % 12] += weights @ spectrum / total_weight
    norms = chroma.max(axis=0, keepdims=True)
    return np.divide(chroma, norms, out=np.zeros_like(chroma), where=norms > 0)


def _cens(chroma: np.ndarray) -> np.ndarray:
    """_cens: Normalisasi, kuantisasi, dan haluskan matriks chroma menjadi CENS."""
    normalized = chroma / np.maximum(
        np.linalg.norm(chroma, ord=1, axis=0, keepdims=True),
        np.finfo(float).eps,
    )
    quantized = (
        (normalized >= 0.05).astype(float)
        + (normalized >= 0.1).astype(float)
        + (normalized >= 0.2).astype(float)
        + (normalized >= 0.4).astype(float)
    )
    smoothed = np.vstack(
        [
            np.convolve(row, np.ones(41) / 41, mode="same")
            for row in quantized
        ]
    )
    norms = np.linalg.norm(smoothed, axis=0, keepdims=True)
    return np.divide(
        smoothed,
        norms,
        out=np.zeros_like(smoothed),
        where=norms > 0,
    )


def _tonnetz(chroma: np.ndarray) -> np.ndarray:
    """_tonnetz: Proyeksikan fitur chroma ke representasi tonal Tonnetz enam dimensi."""
    pitch_angles = 2 * np.pi * np.arange(12) / 12
    basis = np.vstack(
        [
            np.cos(7 * pitch_angles),
            np.sin(7 * pitch_angles),
            np.cos(3 * pitch_angles),
            np.sin(3 * pitch_angles),
            np.cos(4 * pitch_angles),
            np.sin(4 * pitch_angles),
        ]
    )
    return basis @ chroma / np.sqrt(6)


def _onset_strength(
    magnitude: np.ndarray,
    sample_rate: int,
    n_fft: int = 2048,
) -> np.ndarray:
    """_onset_strength: Hitung onset envelope dari kenaikan energi spektrogram Mel log."""
    mel_spectrogram = _mel_filterbank(sample_rate, n_fft) @ magnitude**2
    log_spectrogram = np.log1p(10000 * mel_spectrogram)
    positive_difference = np.maximum(
        0.0,
        log_spectrogram[:, 1:] - log_spectrogram[:, :-1],
    )
    return np.pad(positive_difference.mean(axis=0), (1, 0))


def _tempo_from_onset(onset_envelope: np.ndarray, sample_rate: int) -> float:
    """_tempo_from_onset: Perkirakan BPM dari puncak autokorelasi onset envelope."""
    envelope = onset_envelope - onset_envelope.mean()
    autocorrelation = np.correlate(envelope, envelope, mode="full")[
        len(envelope) - 1 :
    ]
    frames_per_second = sample_rate / 512
    minimum_lag = max(1, math.ceil(frames_per_second * 60 / 240))
    maximum_lag = min(
        len(autocorrelation) - 1,
        int(frames_per_second * 60 / 40),
    )
    if maximum_lag <= minimum_lag or np.allclose(envelope, 0):
        return 0.0
    peaks, _ = find_peaks(autocorrelation[minimum_lag : maximum_lag + 1])
    candidate_lags = peaks + minimum_lag
    if candidate_lags.size == 0:
        candidate_lags = np.arange(minimum_lag, maximum_lag + 1)
    candidate_tempos = 60 * frames_per_second / candidate_lags
    tempo_prior = np.exp(
        -0.5 * (np.log(candidate_tempos / 120.0) / 0.5) ** 2
    )
    best_lag = int(
        candidate_lags[
            np.argmax(autocorrelation[candidate_lags] * tempo_prior)
        ]
    )
    return float(60 * frames_per_second / best_lag)


def _stat_features(name: str, values: np.ndarray) -> dict[str, float]:
    """_stat_features: Ubah satu atau beberapa deret fitur menjadi kolom statistik bernama."""
    values = np.asarray(values)
    if values.ndim == 1:
        values = values[np.newaxis, :]
    features: dict[str, float] = {}
    for index, row in enumerate(values, start=1):
        prefix = f"{name}_{index:02d}_" if values.shape[0] > 1 else f"{name}_"
        for statistic, value in _statistics(row).items():
            features[f"{prefix}{statistic}"] = value
    return features


def _is_blocked_numba_import(error: ImportError) -> bool:
    """_is_blocked_numba_import: Kenali ImportError Numba yang khusus disebabkan kebijakan Windows."""
    message = str(error)
    return (
        "_box" in message
        and "Application Control policy has blocked this file" in message
    )


@lru_cache(maxsize=1)
def _librosa_backend():
    """_librosa_backend: Muat librosa; kembalikan None jika backend native diblokir atau gagal."""
    try:
        import librosa

        librosa.onset.onset_strength(
            y=np.zeros(4096, dtype=np.float32),
            sr=22050,
            hop_length=512,
        )
    except ImportError as error:
        if _is_blocked_numba_import(error):
            return None
        raise
    except OSError as error:
        if "llvmlite.dll" not in str(error):
            raise
        return None
    return librosa


def _extract_with_librosa(
    waveform: np.ndarray,
    sample_rate: int,
    librosa,
) -> dict[str, float]:
    """_extract_with_librosa: Ekstrak fitur audio dengan librosa saat backend tersedia."""
    n_fft = 2048
    hop_length = 512
    features: dict[str, float] = {}
    feature_arrays = [
        (
            "mfcc",
            librosa.feature.mfcc(
                y=waveform,
                sr=sample_rate,
                n_mfcc=20,
                n_fft=n_fft,
                hop_length=hop_length,
            ),
        ),
        (
            "rmse",
            librosa.feature.rms(
                y=waveform,
                frame_length=n_fft,
                hop_length=hop_length,
            ),
        ),
        (
            "zcr",
            librosa.feature.zero_crossing_rate(
                y=waveform,
                frame_length=n_fft,
                hop_length=hop_length,
            ),
        ),
        (
            "spectral_centroid",
            librosa.feature.spectral_centroid(
                y=waveform,
                sr=sample_rate,
                n_fft=n_fft,
                hop_length=hop_length,
            ),
        ),
        (
            "spectral_bandwidth",
            librosa.feature.spectral_bandwidth(
                y=waveform,
                sr=sample_rate,
                n_fft=n_fft,
                hop_length=hop_length,
            ),
        ),
        (
            "spectral_rolloff",
            librosa.feature.spectral_rolloff(
                y=waveform,
                sr=sample_rate,
                n_fft=n_fft,
                hop_length=hop_length,
            ),
        ),
        (
            "spectral_contrast",
            librosa.feature.spectral_contrast(
                y=waveform,
                sr=sample_rate,
                n_fft=n_fft,
                hop_length=hop_length,
            ),
        ),
        (
            "chroma_stft",
            librosa.feature.chroma_stft(
                y=waveform,
                sr=sample_rate,
                n_fft=n_fft,
                hop_length=hop_length,
            ),
        ),
        (
            "chroma_cqt",
            librosa.feature.chroma_cqt(
                y=waveform,
                sr=sample_rate,
                hop_length=hop_length,
            ),
        ),
        (
            "chroma_cens",
            librosa.feature.chroma_cens(
                y=waveform,
                sr=sample_rate,
                hop_length=hop_length,
            ),
        ),
    ]
    harmonic = librosa.effects.harmonic(y=waveform)
    feature_arrays.append(
        (
            "tonnetz",
            librosa.feature.tonnetz(y=harmonic, sr=sample_rate),
        )
    )
    onset_envelope = librosa.onset.onset_strength(
        y=waveform,
        sr=sample_rate,
        hop_length=hop_length,
    )
    feature_arrays.append(("onset_strength", onset_envelope))
    for name, values in feature_arrays:
        features.update(_stat_features(name, values))
    tempo = librosa.beat.beat_track(
        onset_envelope=onset_envelope,
        sr=sample_rate,
        hop_length=hop_length,
    )[0]
    features["bpm"] = float(np.asarray(tempo).reshape(-1)[0])
    return features


def extract_audio_features(audio: str | Path | bytes) -> dict[str, float]:
    """extract_audio_features: Ekstrak 526 fitur; gunakan jalur NumPy/SciPy jika Numba diblokir Windows."""
    waveform, sample_rate = _load_waveform(audio)
    try:
        librosa = _librosa_backend()
        if librosa is not None:
            return _extract_with_librosa(waveform, sample_rate, librosa)
    except ImportError as error:
        if not _is_blocked_numba_import(error):
            raise
    n_fft = 2048
    hop_length = 512
    magnitude = _stft_magnitude(waveform, n_fft, hop_length)
    frequencies = np.fft.rfftfreq(n_fft, 1 / sample_rate)
    magnitudes = magnitude.sum(axis=0)
    safe_magnitudes = np.maximum(magnitudes, np.finfo(float).eps)
    centroid = (frequencies[:, None] * magnitude).sum(axis=0) / safe_magnitudes
    bandwidth = np.sqrt(
        (
            (np.abs(frequencies[:, None] - centroid) ** 2) * magnitude
        ).sum(axis=0)
        / safe_magnitudes
    )
    cumulative = np.cumsum(magnitude, axis=0)
    rolloff_bins = np.argmax(
        cumulative >= 0.85 * cumulative[-1],
        axis=0,
    )
    rolloff = frequencies[rolloff_bins]

    contrast = []
    for band in range(7):
        low = 200 * 2**band
        high = min(200 * 2 ** (band + 1), sample_rate / 2)
        band_spectrum = magnitude[(frequencies >= low) & (frequencies < high)]
        if band_spectrum.size == 0:
            band_spectrum = magnitude[-1:]
        peak_value = np.percentile(band_spectrum, 98, axis=0)
        valley_value = np.percentile(band_spectrum, 2, axis=0)
        contrast.append(
            20
            * np.log10(
                np.maximum(peak_value, 1e-10)
                / np.maximum(valley_value, 1e-10)
            )
        )

    mel_spectrogram = _mel_filterbank(sample_rate, n_fft) @ magnitude**2
    log_mel = 10 * np.log10(np.maximum(mel_spectrogram, 1e-10))
    log_mel = np.maximum(log_mel, log_mel.max() - 80)
    mfcc = dct(log_mel, type=2, axis=0, norm="ortho")[:20]
    chroma_cqt = _cqt_chroma(waveform, sample_rate, hop_length)
    chroma_stft = _chroma_from_magnitude(magnitude, sample_rate, n_fft)
    onset_envelope = _onset_strength(magnitude, sample_rate, n_fft)
    zero_crossings = np.abs(
        np.diff(np.signbit(np.pad(waveform, (n_fft // 2, n_fft // 2))))
    )
    zero_crossing_frames = np.lib.stride_tricks.sliding_window_view(
        zero_crossings,
        n_fft,
    )[::hop_length].mean(axis=1)
    rms_frames = np.lib.stride_tricks.sliding_window_view(
        np.pad(waveform, (n_fft // 2, n_fft // 2)) ** 2,
        n_fft,
    )[::hop_length].mean(axis=1)
    harmonic_approximation = chroma_cqt
    tonnetz = _tonnetz(harmonic_approximation)

    features: dict[str, float] = {}
    for name, values in (
        ("mfcc", mfcc),
        ("rmse", np.sqrt(rms_frames)),
        ("zcr", zero_crossing_frames),
        ("spectral_bandwidth", bandwidth),
        ("spectral_centroid", centroid),
        ("spectral_rolloff", rolloff),
        ("spectral_contrast", np.asarray(contrast)),
        ("chroma_stft", chroma_stft),
        ("chroma_cqt", chroma_cqt),
        ("chroma_cens", _cens(chroma_cqt)),
        ("tonnetz", tonnetz),
        ("onset_strength", onset_envelope),
    ):
        features.update(_stat_features(name, values))
    features["bpm"] = _tempo_from_onset(onset_envelope, sample_rate)
    return features


def get_feature_columns() -> list[str]:
    """get_feature_columns: Kembalikan nama seluruh kolom fitur yang dipakai dalam eksperimen."""
    return [
        *(
            f"{feature}_{index:02d}_{statistic}"
            for feature, count in (
                ("mfcc", 20),
                ("chroma_stft", 12),
                ("chroma_cqt", 12),
                ("chroma_cens", 12),
                ("spectral_contrast", 7),
                ("tonnetz", 6),
            )
            for index in range(1, count + 1)
            for statistic in STATS
        ),
        *(
            f"{feature}_{statistic}"
            for feature in (
                "rmse",
                "zcr",
                "spectral_bandwidth",
                "spectral_centroid",
                "spectral_rolloff",
            )
            for statistic in STATS
        ),
        *(f"onset_strength_{statistic}" for statistic in STATS),
        "bpm",
    ]


def get_feature_groups() -> dict[str, list[str]]:
    """get_feature_groups: Kelompokkan kolom fitur ke grup energi, spektral, MFCC, harmonik, ritmis."""
    columns = get_feature_columns()
    return {
        group: [
            column
            for column in columns
            if any(
                column.startswith(prefix)
                for prefix in definition["prefixes"]
            )
        ]
        for group, definition in FEATURE_GROUPS.items()
    }


def extract_fma_metadata() -> list[Path]:
    """extract_fma_metadata: Pastikan metadata FMA tersedia dengan mengekstraknya dari arsip bila perlu."""
    metadata_directory = FMA_ROOT / "fma_metadata"
    metadata_filenames = ("tracks.csv", "genres.csv", "features.csv")
    metadata_path = FMA_ROOT / "tracks.csv"
    archive_path = FMA_ROOT / "fma_metadata.zip"
    if not archive_path.exists() and not (
        metadata_path.is_file() and metadata_path.stat().st_size
    ):
        raise FileNotFoundError(
            "Metadata genre FMA tidak ditemukan. Letakkan tracks.csv atau "
            f"fma_metadata.zip di folder {FMA_ROOT}."
        )

    extracted_paths: list[Path] = []
    archive_context = (
        zipfile.ZipFile(archive_path)
        if archive_path.exists()
        else nullcontext(None)
    )
    with archive_context as archive:
        archive_members = (
            {
                name.replace("\\", "/").rsplit("/", 1)[-1]: name
                for name in archive.namelist()
                if name.replace("\\", "/").rsplit("/", 1)[-1]
                in metadata_filenames
            }
            if archive is not None
            else {}
        )
        for filename in metadata_filenames:
            destination = metadata_directory / filename
            if destination.is_file() and destination.stat().st_size:
                extracted_paths.append(destination)
                continue

            member = archive_members.get(filename)
            if member is None and filename == "tracks.csv" and metadata_path.is_file():
                source_path: Path | None = metadata_path
            elif member is not None:
                source_path = None
            else:
                raise ValueError(f"Arsip metadata tidak berisi {filename}.")

            metadata_directory.mkdir(parents=True, exist_ok=True)
            temporary_path: Path | None = None
            try:
                with tempfile.NamedTemporaryFile(
                    mode="wb",
                    dir=metadata_directory,
                    prefix=f".{filename}.",
                    suffix=".tmp",
                    delete=False,
                ) as extracted_file:
                    temporary_path = Path(extracted_file.name)
                    if member is not None:
                        if archive is None:
                            raise FileNotFoundError(
                                "Arsip metadata FMA tidak tersedia."
                            )
                        with archive.open(member) as archived_file:
                            shutil.copyfileobj(archived_file, extracted_file)
                    elif source_path is not None:
                        with source_path.open("rb") as source_file:
                            shutil.copyfileobj(source_file, extracted_file)
                temporary_path.replace(destination)
            finally:
                if temporary_path is not None and temporary_path.exists():
                    temporary_path.unlink()
            extracted_paths.append(destination)

    return extracted_paths


def _load_track_metadata() -> pd.DataFrame:
    """_load_track_metadata: Muat metadata trek FMA dari berkas yang tersedia atau arsip metadata."""
    metadata_path = FMA_ROOT / "tracks.csv"
    if metadata_path.is_file() and metadata_path.stat().st_size:
        return pd.read_csv(metadata_path, header=[0, 1], index_col=0)
    extracted_metadata_path = FMA_ROOT / "fma_metadata" / "tracks.csv"
    if extracted_metadata_path.is_file() and extracted_metadata_path.stat().st_size:
        return pd.read_csv(
            extracted_metadata_path,
            header=[0, 1],
            index_col=0,
        )
    return pd.read_csv(
        next(
            (
                path
                for path in extract_fma_metadata()
                if path.name == "tracks.csv"
            ),
            extracted_metadata_path,
        ),
        header=[0, 1],
        index_col=0,
    )


def _load_precomputed_features(track_ids: pd.Index) -> pd.DataFrame:
    """_load_precomputed_features: Muat fitur bawaan FMA untuk ID trek yang diminta."""
    archive_path = FMA_ROOT / "fma_metadata.zip"
    features_path = FMA_ROOT / "features.csv"
    extracted_features_path = FMA_ROOT / "fma_metadata" / "features.csv"
    if features_path.exists():
        with features_path.open("rb") as feature_file:
            return _read_precomputed_features(feature_file, track_ids)
    if extracted_features_path.is_file():
        with extracted_features_path.open("rb") as feature_file:
            return _read_precomputed_features(feature_file, track_ids)
    if not archive_path.exists():
        raise FileNotFoundError(
            "Berkas fitur bawaan FMA tidak ditemukan. Gunakan "
            f"fma_metadata.zip di {FMA_ROOT}."
        )

    with zipfile.ZipFile(archive_path) as archive:
        csv_path = next(
            (
                name
                for name in archive.namelist()
                if name.replace("\\", "/").endswith("/features.csv")
                or name == "features.csv"
            ),
            None,
        )
        if csv_path is None:
            raise ValueError("Arsip metadata tidak berisi features.csv.")
        with archive.open(csv_path) as feature_file:
            return _read_precomputed_features(feature_file, track_ids)


def _read_precomputed_features(
    feature_file,
    track_ids: pd.Index,
) -> pd.DataFrame:
    """_read_precomputed_features: Baca, pilih, dan normalkan kolom fitur bawaan FMA dari CSV."""
    header = pd.read_csv(
        feature_file,
        header=[0, 1, 2],
        index_col=0,
        nrows=0,
    )
    selected_positions = list(range(1, len(header.columns) + 1))
    if len(selected_positions) != 518:
        raise ValueError(
            "Kolom fitur audio FMA pada features.csv tidak lengkap."
        )

    feature_file.seek(0)
    selected = pd.read_csv(
        feature_file,
        header=None,
        skiprows=4,
        index_col=0,
        usecols=[0, *selected_positions],
    )
    selected.columns = [
        (
            f"{column[0]}_{column[2]}_{column[1]}"
            if column[0]
            in (
                "mfcc",
                "chroma_stft",
                "chroma_cqt",
                "chroma_cens",
                "spectral_contrast",
                "tonnetz",
            )
            else f"{column[0]}_{column[1]}"
        )
        for column in header.columns
    ]
    numeric_ids = pd.to_numeric(selected.index, errors="coerce")
    selected = selected.loc[numeric_ids.notna()].copy()
    selected.index = numeric_ids[numeric_ids.notna()].astype(int)
    selected.index.name = "track_id"
    selected = selected.reindex(track_ids)
    base_feature_columns = [
        column for column in get_feature_columns() if not column.startswith(
            ("onset_strength_", "bpm")
        )
    ]
    selected = selected.reindex(columns=base_feature_columns)
    return selected.apply(pd.to_numeric, errors="coerce")


def _load_fma_tracks() -> tuple[pd.DataFrame, pd.Series]:
    """_load_fma_tracks: Pilih trek FMA Small berlabel dan pasangkan dengan file audionya."""
    metadata = _load_track_metadata()
    required_columns = (("track", "genre_top"), ("set", "subset"))
    missing_columns = [column for column in required_columns if column not in metadata]
    if missing_columns:
        raise ValueError(
            "Kolom genre/subset tidak ditemukan pada metadata tracks.csv: "
            + ", ".join(f"{first}.{second}" for first, second in missing_columns)
        )

    genre = metadata[("track", "genre_top")]
    subset = metadata[("set", "subset")]
    small_tracks = metadata.loc[
        subset.eq("small") & genre.isin(GENRES)
    ].copy()
    if small_tracks.empty:
        raise ValueError("Metadata tidak memuat label genre untuk FMA Small.")

    audio_paths: dict[int, Path] = {}
    for audio_path in FMA_ROOT.glob("*/*.mp3"):
        try:
            track_id = int(audio_path.stem)
        except ValueError:
            continue
        audio_paths[track_id] = audio_path

    track_ids = small_tracks.index.to_series().astype(int)
    small_tracks = small_tracks.loc[track_ids.isin(audio_paths)].copy()
    if small_tracks.empty:
        raise ValueError(
            "Tidak ada file MP3 FMA Small yang cocok dengan ID pada metadata."
        )

    if ("artist", "id") in small_tracks.columns:
        groups = small_tracks[("artist", "id")].fillna(
            pd.Series(small_tracks.index, index=small_tracks.index)
        )
    else:
        groups = pd.Series(small_tracks.index, index=small_tracks.index)
    small_tracks["__artist_group"] = groups.astype(str)
    small_tracks["__audio_path"] = [
        str(audio_paths[int(track_id)]) for track_id in small_tracks.index
    ]
    return small_tracks, genre.loc[small_tracks.index].astype(str)


def _extract_rhythm_features(audio_path: str | Path) -> dict[str, float]:
    """_extract_rhythm_features: Ekstrak onset envelope dan BPM dari satu file audio."""
    waveform, sample_rate = _load_waveform(audio_path)
    librosa = _librosa_backend()
    if librosa is not None:
        onset_envelope = librosa.onset.onset_strength(
            y=waveform,
            sr=sample_rate,
            hop_length=512,
        )
        tempo = librosa.beat.beat_track(
            onset_envelope=onset_envelope,
            sr=sample_rate,
            hop_length=512,
        )[0]
        features = {
            f"onset_strength_{statistic}": value
            for statistic, value in _statistics(onset_envelope).items()
        }
        features["bpm"] = float(np.asarray(tempo).reshape(-1)[0])
        return features
    magnitude = _stft_magnitude(waveform)
    onset_envelope = _onset_strength(magnitude, sample_rate)
    features = {
        f"onset_strength_{statistic}": value
        for statistic, value in _statistics(onset_envelope).items()
    }
    features["bpm"] = _tempo_from_onset(onset_envelope, sample_rate)
    return features


def _load_or_extract_features(
    tracks: pd.DataFrame,
    progress: Callable[[float, str], None] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """_load_or_extract_features: Muat cache fitur ritmis dan ekstrak fitur untuk trek yang belum tercache."""
    _librosa_backend()
    track_ids = tracks.index.astype(int)
    columns = [
        *(f"onset_strength_{statistic}" for statistic in STATS),
        "bpm",
    ]
    cached = pd.DataFrame(index=track_ids, columns=columns, dtype=float)
    if RHYTHM_CACHE_PATH.is_file():
        cache_df = pd.read_csv(RHYTHM_CACHE_PATH)
        if not {"track_id", *columns}.issubset(cache_df.columns):
            raise ValueError(
                f"Cache ritmis {RHYTHM_CACHE_PATH} tidak memiliki kolom yang "
                "diperlukan."
            )
        cache_df = cache_df.drop_duplicates("track_id", keep="last").set_index(
            "track_id"
        )
        cache_df.index = pd.to_numeric(cache_df.index, errors="raise").astype(int)
        common_ids = track_ids.intersection(cache_df.index)
        cached.loc[common_ids, columns] = cache_df.loc[common_ids, columns].to_numpy()

    cached = cached.apply(pd.to_numeric, errors="coerce")
    cached = cached.replace([np.inf, -np.inf], np.nan)
    uncached = cached.index[~cached.notna().all(axis=1)]
    errors: list[str] = []
    total = max(len(uncached), 1)
    max_workers = min(8, os.cpu_count() or 1)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_track_ids = {
            executor.submit(
                _extract_rhythm_features,
                tracks.loc[track_id, "__audio_path"],
            ): track_id
            for track_id in uncached
        }
        for position, future in enumerate(
            as_completed(future_track_ids),
            start=1,
        ):
            track_id = future_track_ids[future]
            try:
                cached.loc[track_id, columns] = future.result()
            except (
                EOFError,
                OSError,
                RuntimeError,
                ValueError,
                soundfile.LibsndfileError,
            ) as error:
                errors.append(f"{int(track_id)}: {error}")
            if progress is not None:
                progress(
                    position / total,
                    f"Mengekstrak fitur ritmis {position:,}/{len(uncached):,}",
                )
            if position % 250 == 0:
                _write_csv_atomically(
                    cached.rename_axis("track_id").reset_index(),
                    RHYTHM_CACHE_PATH,
                )

    features = cached.replace([np.inf, -np.inf], np.nan).dropna()
    if features.empty:
        detail = f" Contoh kegagalan: {errors[0]}" if errors else ""
        raise ValueError(f"Tidak ada fitur ritmis yang berhasil dihitung.{detail}")
    _write_csv_atomically(
        cached.rename_axis("track_id").reset_index(),
        RHYTHM_CACHE_PATH,
    )
    return features, errors


def _write_csv_atomically(data: pd.DataFrame, destination: Path) -> None:
    """_write_csv_atomically: Tulis DataFrame ke CSV melalui file sementara lalu ganti secara atomik."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as output_file:
            temporary_path = Path(output_file.name)
            data.to_csv(output_file, index=False)
        temporary_path.replace(destination)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def _write_scenario_datasets(
    dataset: pd.DataFrame,
    feature_groups: dict[str, list[str]],
    only_outdated: bool = False,
) -> dict[str, str]:
    """_write_scenario_datasets: Simpan CSV untuk tiap skenario kombinasi kelompok fitur."""
    scenario_paths: dict[str, str] = {}
    for code, group_codes in SCENARIOS.items():
        feature_columns = [
            column
            for group_code in group_codes
            for column in feature_groups[group_code]
        ]
        filename, _ = DATASET_SPECS[code]
        scenario_path = DATASET_DIR / filename
        expected_columns = [*DATASET_METADATA_COLUMNS, *feature_columns]
        if only_outdated and scenario_path.is_file():
            existing_columns = pd.read_csv(scenario_path, nrows=0).columns
            if list(existing_columns) == expected_columns:
                scenario_paths[code] = str(scenario_path)
                continue
        _write_csv_atomically(
            dataset[expected_columns],
            scenario_path,
        )
        scenario_paths[code] = str(scenario_path)
    return scenario_paths


def build_final_dataset(
    progress: Callable[[float, str], None] | None = None,
) -> dict[str, object]:
    """build_final_dataset: Bangun dataset FMA tervalidasi beserta ringkasan audit dan file skenario."""
    if progress is not None:
        progress(0.01, "Memeriksa metadata, label, dan pembagian resmi FMA")
    tracks, labels = _load_fma_tracks()
    track_ids = tracks.index.astype(int)
    base_features = _load_precomputed_features(track_ids)
    if progress is not None:
        progress(0.05, "Memuat 518 fitur bawaan FMA")

    rhythm_features, extraction_errors = _load_or_extract_features(
        tracks,
        (
            lambda fraction, message: progress(
                0.05 + fraction * 0.85,
                message,
            )
            if progress is not None
            else None
        ),
    )
    all_features = base_features.join(rhythm_features, how="inner")
    all_features = all_features.reindex(columns=get_feature_columns())
    all_features = all_features.replace([np.inf, -np.inf], np.nan)
    valid_rows = all_features.notna().all(axis=1)
    valid_ids = all_features.index[valid_rows]
    audit_rows = [
        {
            "track_id": int(track_id),
            "reason": "Fitur FMA atau ritmis kosong/tidak valid",
        }
        for track_id in all_features.index[~valid_rows]
    ]
    for extraction_error in extraction_errors:
        track_id_text, _, reason = extraction_error.partition(":")
        try:
            track_id = int(track_id_text)
        except ValueError:
            continue
        audit_rows.append(
            {
                "track_id": track_id,
                "reason": f"Ekstraksi audio ritmis gagal: {reason.strip()}",
            }
        )
    all_features = all_features.loc[valid_ids]
    valid_tracks = tracks.loc[valid_ids]

    split_column = ("set", "split")
    artist_column = ("artist", "id")
    if split_column not in valid_tracks.columns or artist_column not in valid_tracks.columns:
        raise ValueError(
            "Metadata FMA harus memiliki kolom set.split dan artist.id."
        )
    metadata = pd.DataFrame(
        {
            "track_id": valid_ids.astype(int),
            "artist_id": valid_tracks[artist_column].to_numpy(),
            "genre_top": labels.loc[valid_ids].to_numpy(),
            "split": valid_tracks[split_column].to_numpy(),
        },
        index=valid_ids,
    )
    if metadata.isna().any().any():
        raise ValueError(
            "Dataset akhir masih memiliki identitas, label, artis, atau split "
            "yang kosong."
        )
    dataset = pd.concat([metadata, all_features], axis=1).reset_index(drop=True)
    if dataset.empty:
        raise ValueError("Tidak ada trek yang lolos audit untuk dataset akhir.")
    expected_columns = len(DATASET_METADATA_COLUMNS) + 526
    if dataset.shape[1] != expected_columns:
        raise ValueError(
            f"Dataset akhir memiliki {dataset.shape[1]} kolom; "
            f"seharusnya {expected_columns}."
        )
    if dataset["track_id"].duplicated().any():
        raise ValueError("ID trek ganda ditemukan pada dataset akhir.")

    _write_csv_atomically(dataset, FINAL_DATASET_PATH)
    audit = pd.DataFrame(audit_rows, columns=("track_id", "reason"))
    _write_csv_atomically(audit, DATASET_AUDIT_PATH)
    feature_groups = get_feature_groups()
    scenario_paths = _write_scenario_datasets(dataset, feature_groups)

    split_counts = dataset["split"].value_counts().to_dict()
    genre_counts = dataset["genre_top"].value_counts().sort_index().to_dict()
    summary: dict[str, object] = {
        "dataset_path": str(FINAL_DATASET_PATH),
        "audit_path": str(DATASET_AUDIT_PATH),
        "scenario_paths": scenario_paths,
        "total": len(dataset),
        "columns": len(dataset.columns),
        "feature_counts": {
            group: len(columns) for group, columns in feature_groups.items()
        },
        "split_counts": split_counts,
        "genre_counts": genre_counts,
        "excluded_audio_tracks": extraction_errors,
        "excluded_tracks": len(audit),
        "trained_tracks": len(dataset),
    }
    if progress is not None:
        progress(1.0, "Dataset akhir dan tiga skenario berhasil disimpan")
    return summary


def _metrics(labels: pd.Series, predictions: np.ndarray) -> dict[str, float]:
    """_metrics: Hitung accuracy, balanced accuracy, macro precision, recall, dan F1."""
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "precision": float(
            precision_score(labels, predictions, average="macro", zero_division=0)
        ),
        "recall": float(
            recall_score(labels, predictions, average="macro", zero_division=0)
        ),
        "f1_macro": float(
            f1_score(labels, predictions, average="macro", zero_division=0)
        ),
    }


def train_from_fma(
    progress: Callable[[float, str], None] | None = None,
) -> dict[str, object]:
    """train_from_fma: Latih dan evaluasi model untuk setiap skenario kombinasi fitur FMA."""
    if FINAL_DATASET_PATH.is_file():
        existing_header = pd.read_csv(FINAL_DATASET_PATH, nrows=0).columns
        expected_columns = [
            *DATASET_METADATA_COLUMNS,
            *get_feature_columns(),
        ]
        if list(existing_header) == expected_columns:
            metadata = pd.read_csv(
                FINAL_DATASET_PATH,
                usecols=list(DATASET_METADATA_COLUMNS),
            )
            dataset_summary: dict[str, object] = {
                "dataset_path": str(FINAL_DATASET_PATH),
                "total": len(metadata),
                "columns": len(expected_columns),
                "feature_counts": {
                    group: len(columns)
                    for group, columns in get_feature_groups().items()
                },
                "split_counts": metadata["split"].value_counts().to_dict(),
                "genre_counts": (
                    metadata["genre_top"].value_counts().sort_index().to_dict()
                ),
                "excluded_audio_tracks": [],
                "trained_tracks": len(metadata),
            }
        else:
            dataset_summary = build_final_dataset(progress)
    else:
        dataset_summary = build_final_dataset(progress)
    dataset = pd.read_csv(FINAL_DATASET_PATH)
    feature_groups = get_feature_groups()
    dataset_summary["scenario_paths"] = _write_scenario_datasets(
        dataset,
        feature_groups,
        only_outdated=True,
    )
    labels = dataset["genre_top"]
    artists = dataset["artist_id"]
    split = dataset["split"]
    train_mask = split.eq("training")
    validation_mask = split.eq("validation")
    test_mask = split.eq("test")
    if not train_mask.any() or not validation_mask.any() or not test_mask.any():
        raise ValueError(
            "Dataset FMA harus memiliki split training, validation, dan test."
        )

    split_groups = {
        name: set(artists.loc[mask].dropna())
        for name, mask in (
            ("training", train_mask),
            ("validation", validation_mask),
            ("test", test_mask),
        )
    }
    if (
        split_groups["training"] & split_groups["validation"]
        or split_groups["training"] & split_groups["test"]
        or split_groups["validation"] & split_groups["test"]
    ):
        raise ValueError("ID artis tumpang tindih pada pembagian resmi FMA.")

    y_train = labels.loc[train_mask]
    y_validation = labels.loc[validation_mask]
    y_test = labels.loc[test_mask]
    artist_train = artists.loc[train_mask]
    scenario_results: dict[str, dict[str, object]] = {}
    selected_models: dict[str, RandomForestClassifier] = {}
    selected_metrics: dict[str, dict[str, float]] = {}
    scenario_items = list(SCENARIOS.items())

    for scenario_index, (scenario_code, group_codes) in enumerate(
        scenario_items,
        start=1,
    ):
        feature_columns = [
            column
            for group_code in group_codes
            for column in feature_groups[group_code]
        ]
        X_train = dataset.loc[train_mask, feature_columns]
        X_validation = dataset.loc[validation_mask, feature_columns]
        X_test = dataset.loc[test_mask, feature_columns]

        if progress is not None:
            progress(
                min(0.10 + 0.85 * (scenario_index - 1) / len(scenario_items), 0.95),
                f"Melatih skenario {scenario_code} "
                f"({scenario_index}/{len(scenario_items)})",
            )

        baseline_models = {
            "M0": DummyClassifier(strategy="most_frequent"),
            "M1": RandomForestClassifier(
                n_estimators=300,
                max_features="sqrt",
                criterion="gini",
                class_weight=None,
                random_state=42,
                n_jobs=1,
            ),
            "M2": RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                criterion="gini",
                class_weight=None,
                random_state=42,
                n_jobs=1,
            ),
        }
        model_comparisons: dict[str, dict[str, object]] = {}
        fitted_models: dict[str, object] = {}
        for model_code, model in baseline_models.items():
            model.fit(X_train, y_train)
            fitted_models[model_code] = model
            validation_metrics = _metrics(
                y_validation,
                model.predict(X_validation),
            )
            model_comparisons[model_code] = {
                "validation_metrics": validation_metrics,
                "params": model.get_params(),
            }

        search = GridSearchCV(
            estimator=RandomForestClassifier(
                n_estimators=300,
                criterion="gini",
                class_weight=None,
                random_state=42,
                n_jobs=1,
            ),
            param_grid={
                "max_depth": [None, 5, 10, 20],
                "min_samples_leaf": [1, 2, 5, 10],
                "max_features": ["sqrt", "log2", 0.3],
            },
            scoring="f1_macro",
            cv=StratifiedGroupKFold(
                n_splits=4,
                shuffle=True,
                random_state=42,
            ),
            n_jobs=1,
            pre_dispatch=1,
            refit=True,
            error_score="raise",
            verbose=2,
        )
        search.fit(X_train, y_train, groups=artist_train)
        fitted_models["M3"] = search.best_estimator_
        model_comparisons["M3"] = {
            "validation_metrics": _metrics(
                y_validation,
                search.best_estimator_.predict(X_validation),
            ),
            "params": search.best_params_,
            "cross_validation_f1_macro_mean": float(search.best_score_),
            "cross_validation_f1_macro_std": float(
                search.cv_results_["std_test_score"][search.best_index_]
            ),
        }

        m1_f1 = model_comparisons["M1"]["validation_metrics"]["f1_macro"]
        m3_f1 = model_comparisons["M3"]["validation_metrics"]["f1_macro"]
        cv_std = model_comparisons["M3"]["cross_validation_f1_macro_std"]
        selected_code = "M3" if m3_f1 - m1_f1 > cv_std else "M1"
        selected_model = fitted_models[selected_code]
        selected_models[scenario_code] = selected_model
        selected_validation_metrics = model_comparisons[selected_code][
            "validation_metrics"
        ]
        selected_metrics[scenario_code] = selected_validation_metrics
        selected_params = model_comparisons[selected_code]["params"]
        test_predictions = selected_model.predict(X_test)
        test_metrics = _metrics(y_test, test_predictions)
        classes = list(selected_model.classes_)
        matrix = confusion_matrix(y_test, test_predictions, labels=classes)
        report = classification_report(
            y_test,
            test_predictions,
            labels=classes,
            output_dict=True,
            zero_division=0,
        )
        scenario_results[scenario_code] = {
            "name": DATASET_SPECS[scenario_code][1],
            "groups": list(group_codes),
            "feature_count": len(feature_columns),
            "selected_model": selected_code,
            "validation_metrics": selected_validation_metrics,
            "test_metrics": test_metrics,
            "model_comparisons": model_comparisons,
            "best_params": selected_params,
            "confusion_matrix": matrix.tolist(),
            "classification_report": report,
        }

    selected_scenario = max(
        selected_metrics,
        key=lambda scenario: selected_metrics[scenario]["f1_macro"],
    )
    final_model = selected_models[selected_scenario]
    final_features = [
        column
        for group_code in SCENARIOS[selected_scenario]
        for column in feature_groups[group_code]
    ]
    final_test_metrics = scenario_results[selected_scenario]["test_metrics"]
    final_results: dict[str, object] = {
        "artifact_version": 3,
        "model": final_model,
        "features": final_features,
        "feature_groups": list(SCENARIOS[selected_scenario]),
        "scenario": selected_scenario,
        "classes": list(final_model.classes_),
        "metrics": final_test_metrics,
        "validation_metrics": selected_metrics[selected_scenario],
        "best_params": scenario_results[selected_scenario]["best_params"],
        "selection_rule": (
            "Skenario dipilih berdasarkan macro F1 validasi tertinggi; "
            "M3 dipilih jika peningkatan validasi atas M1 melebihi simpangan "
            "baku cross-validation."
        ),
        "data_splits": {
            "training": int(train_mask.sum()),
            "validation": int(validation_mask.sum()),
            "testing": int(test_mask.sum()),
            "total": len(dataset),
        },
        "genre_counts": labels.value_counts().to_dict(),
        "scenario_results": scenario_results,
        "trained_tracks": len(dataset),
        "dataset_path": str(FINAL_DATASET_PATH),
        "trained_at": datetime.now().isoformat(timespec="seconds"),
    }
    full_model = selected_models["S3"]
    full_feature_columns = [
        column
        for group_code in SCENARIOS["S3"]
        for column in feature_groups[group_code]
    ]
    full_test = dataset.loc[test_mask, full_feature_columns]
    full_test_predictions = full_model.predict(full_test)
    full_test_f1 = f1_score(
        y_test,
        full_test_predictions,
        average="macro",
        zero_division=0,
    )
    rng = np.random.default_rng(42)
    group_permutation_importance: dict[str, float] = {}
    group_mean_decrease_impurity: dict[str, float] = {}
    feature_importances = dict(
        zip(full_feature_columns, full_model.feature_importances_)
    )
    for group_code, columns in feature_groups.items():
        drops = []
        for _ in range(5):
            shuffled = full_test.copy()
            permutation = rng.permutation(len(shuffled))
            shuffled.loc[:, columns] = shuffled[columns].to_numpy()[permutation]
            permuted_f1 = f1_score(
                y_test,
                full_model.predict(shuffled),
                average="macro",
                zero_division=0,
            )
            drops.append(float(full_test_f1 - permuted_f1))
        group_permutation_importance[group_code] = float(np.mean(drops))
        group_mean_decrease_impurity[group_code] = float(
            sum(feature_importances[column] for column in columns)
        )
    scenario_results["S3"]["group_permutation_importance"] = (
        group_permutation_importance
    )
    scenario_results["S3"]["group_mean_decrease_impurity"] = (
        group_mean_decrease_impurity
    )
    final_results["group_permutation_importance"] = (
        group_permutation_importance
    )
    final_results["group_mean_decrease_impurity"] = group_mean_decrease_impurity

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    with MODEL_PATH.open("wb") as model_file:
        pickle.dump(final_results, model_file)
    experiment_summary = {
        "dataset": dataset_summary,
        "selected_scenario": selected_scenario,
        "scenarios": scenario_results,
    }
    with EXPERIMENT_RESULTS_PATH.open("w", encoding="utf-8") as results_file:
        json.dump(experiment_summary, results_file, ensure_ascii=False, indent=2)
    if progress is not None:
        progress(1.0, "Evaluasi seluruh skenario selesai dan model disimpan")
    return final_results


def train_default_model() -> dict[str, object]:
    """train_default_model: Train the model selected by the documented feature-scenario experiment."""
    return train_from_fma()


def load_model() -> dict[str, object] | None:
    """load_model: Muat artefak model yang kompatibel atau latih ulang bila diperlukan."""
    if not MODEL_PATH.exists():
        if FINAL_DATASET_PATH.exists():
            try:
                return train_default_model()
            except (OSError, ValueError):
                return None
        return None

    try:
        with MODEL_PATH.open("rb") as model_file:
            artifact = pickle.load(model_file)
    except (EOFError, OSError, pickle.PickleError, ValueError):
        if FINAL_DATASET_PATH.exists():
            try:
                return train_default_model()
            except (OSError, ValueError):
                return None
        return None

    required_keys = {
        "artifact_version",
        "model",
        "features",
        "classes",
        "metrics",
        "scenario",
        "genre_counts",
    }
    if not isinstance(artifact, dict) or not required_keys.issubset(artifact):
        if FINAL_DATASET_PATH.exists():
            try:
                return train_default_model()
            except (OSError, ValueError):
                return None
        return None
    if artifact["artifact_version"] != 3:
        if FINAL_DATASET_PATH.exists():
            try:
                return train_default_model()
            except (OSError, ValueError):
                return None
        return None
    return artifact


def predict_genre(
    artifact: dict[str, object],
    features: dict[str, float],
) -> dict[str, object]:
    """predict_genre: Prediksi genre audio beserta keyakinan dan probabilitas tiap kelas."""
    feature_columns = artifact["features"]
    model = artifact["model"]
    row = pd.DataFrame(
        [{column: features[column] for column in feature_columns}],
        columns=feature_columns,
    )
    probabilities = model.predict_proba(row)[0]
    classes = list(model.classes_)
    best_index = int(np.argmax(probabilities))
    return {
        "genre": classes[best_index],
        "confidence": float(probabilities[best_index]),
        "probabilities": {
            genre: float(probability)
            for genre, probability in zip(classes, probabilities)
        },
    }
