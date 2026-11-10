"""
Dataset Loader untuk berbagai format EEG
Mendukung: EDF, GDF, SET, dan format lainnya
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, Optional, List
import warnings

warnings.filterwarnings('ignore')

DATASET_ROOT = Path(__file__).resolve().parent / "Data"

# =========================================================
# FUNGSI UMUM
# =========================================================

# [FUNGSI 02] extract_band_powers
def extract_band_powers(signal: np.ndarray, sampling_rate: float = 200) -> Dict[str, float]:
    """extract_band_powers: 
    Ekstrak power dari berbagai band frekuensi EEG
    """
    try:
        # Handle multidimensional arrays - ambil rata-rata dari semua channel
        if signal.ndim > 1:
            signal = np.mean(signal, axis=1)
        
        # Normalisasi signal
        signal = signal - np.mean(signal)
        
        # [HITUNG EEG 04] Daya spektral = kuadrat amplitudo FFT / jumlah sampel.
        spectrum = np.abs(np.fft.rfft(signal)) ** 2 / len(signal)
        frequencies = np.fft.rfftfreq(len(signal), d=1 / sampling_rate)
        
        # Band definitions
        bands = {
            "delta_power": (0.5, 4),
            "theta_power": (4, 8),
            "alpha_power": (8, 13),
            "beta_power": (13, 30),
            "gamma_power": (30, 100),
        }
        
        powers = {}
        for name, (low, high) in bands.items():
            # [HITUNG EEG 05] Band power = rata-rata daya FFT dalam rentang frekuensi.
            mask = (frequencies >= low) & (frequencies < high)
            powers[name] = float(np.mean(spectrum[mask])) if mask.any() else 0.0
        
        return powers
    except Exception as e:
        print(f"Error extracting band powers: {e}")
        return {
            "delta_power": 0.0,
            "theta_power": 0.0,
            "alpha_power": 0.0,
            "beta_power": 0.0,
            "gamma_power": 0.0,
        }


# [FUNGSI 03] get_dominant_wave
def get_dominant_wave(powers: Dict[str, float]) -> str:
    """get_dominant_wave: Tentukan gelombang dominan berdasarkan power"""
    return max(powers, key=powers.get).replace("_power", "").title()


# =========================================================
# MOTOR IMAGERY DATASET (EDF FILES)
# =========================================================

# [FUNGSI 04] load_motor_imagery_edf
def load_motor_imagery_edf():
    """load_motor_imagery_edf: Load EEG Motor Movement Imagery Dataset (EDF format)"""
    try:
        import mne
    except ImportError:
        return [], "mne library tidak terinstall. Silakan jalankan: pip install mne"
    
    dataset_path = DATASET_ROOT / "eeg-motor-movementimagery-dataset-1.0.0" / "files"
    
    if not dataset_path.exists():
        return [], f"Dataset folder tidak ditemukan: {dataset_path}"
    
    records = []
    
    # Scan semua EDF files
    edf_files = list(dataset_path.glob("S*/S*R*.edf"))
    
    if not edf_files:
        return [], "Tidak ada file EDF ditemukan"
    
    for file_path in edf_files[:50]:  # Limit untuk testing
        try:
            # Load EDF file
            raw = mne.io.read_raw_edf(str(file_path), preload=True)
            signal = raw.get_data()
            
            # Extract features
            powers = extract_band_powers(signal, sampling_rate=raw.info['sfreq'])
            brain_wave = get_dominant_wave(powers)
            
            record = {
                "frequency_sound": 0.0,  # Motor imagery tidak punya frekuensi suara
                "delta_power": powers["delta_power"],
                "theta_power": powers["theta_power"],
                "alpha_power": powers["alpha_power"],
                "beta_power": powers["beta_power"],
                "gamma_power": powers["gamma_power"],
                "brain_wave": brain_wave,
                "source": f"Motor Imagery:{file_path.name}",
            }
            records.append(record)
        except Exception as e:
            print(f"Error loading {file_path.name}: {e}")
            continue
    
    return records, f"Berhasil load {len(records)} file dari Motor Imagery Dataset"


# =========================================================
# GDF DATASET (ppz3r5j2n2-1)
# =========================================================

# [FUNGSI 05] load_gdf_dataset
def load_gdf_dataset():
    """load_gdf_dataset: Load GDF dataset (ppz3r5j2n2-1)"""
    try:
        import mne
    except ImportError:
        return [], "mne library tidak terinstall"
    
    dataset_path = DATASET_ROOT / "ppz3r5j2n2-1"
    
    if not dataset_path.exists():
        return [], f"Dataset folder tidak ditemukan: {dataset_path}"
    
    records = []
    conditions = ["1. Resting State", "2. Theta BB", "3. Beta BB", "4. Alpha BB"]
    
    for condition in conditions:
        condition_path = dataset_path / condition
        if not condition_path.exists():
            continue
        
        gdf_files = list(condition_path.glob("*.gdf"))
        
        for file_path in gdf_files[:10]:  # Limit untuk testing
            try:
                # Load GDF file
                raw = mne.io.read_raw_gdf(str(file_path), preload=True)
                signal = raw.get_data()
                
                # Extract features
                powers = extract_band_powers(signal, sampling_rate=raw.info['sfreq'])
                brain_wave = get_dominant_wave(powers)
                
                record = {
                    "frequency_sound": 0.0,
                    "delta_power": powers["delta_power"],
                    "theta_power": powers["theta_power"],
                    "alpha_power": powers["alpha_power"],
                    "beta_power": powers["beta_power"],
                    "gamma_power": powers["gamma_power"],
                    "brain_wave": brain_wave,
                    "source": f"GDF {condition}:{file_path.name}",
                }
                records.append(record)
            except Exception as e:
                print(f"Error loading {file_path.name}: {e}")
                continue
    
    return records, f"Berhasil load {len(records)} file dari GDF Dataset"


# =========================================================
# BIDS FORMAT DATASET (v1.0.0)
# =========================================================

# [FUNGSI 06] load_bids_dataset
def load_bids_dataset():
    """load_bids_dataset: Load satu rekaman penuh per file dari BIDS dataset v1.0.0."""
    try:
        import mne
    except ImportError:
        return [], "MNE library tidak terinstall. Silakan jalankan: pip install mne"
    
    dataset_path = DATASET_ROOT / "v1.0.0"
    
    if not dataset_path.exists():
        return [], f"Dataset folder tidak ditemukan: {dataset_path}"
    
    records = []
    song_tempos = {
        1: 121.95,
        2: 119.0,
        3: 161.0,
        4: None,
        5: 123.0,
        6: 126.0,
        7: 70.0,
        8: 194.0,
        9: 117.0,
        10: 197.0,
        11: 120.0,
        12: 146.0,
    }
    
    # Scan semua subject folders
    subject_folders = sorted([d for d in dataset_path.iterdir() if d.is_dir() and d.name.startswith("sub-")])
    
    for subject_folder in subject_folders:
        # Scan session folders
        session_folders = sorted([d for d in subject_folder.iterdir() if d.is_dir() and d.name.startswith("ses-")])
        
        for session_folder in session_folders:
            eeg_folder = session_folder / "eeg"
            if not eeg_folder.exists():
                continue
            
            # Cari SET files (EEGLAB format)
            set_files = sorted(eeg_folder.glob("*.set"))
            
            for set_file in set_files:
                try:
                    # Read SET file menggunakan mne
                    raw = mne.io.read_raw_eeglab(str(set_file), preload=True)
                    signal = raw.get_data()
                    powers = extract_band_powers(
                        signal.T, sampling_rate=float(raw.info["sfreq"])
                    )
                    brain_wave = get_dominant_wave(powers)
                    recording_id = set_file.stem
                    if recording_id.endswith("_eeg"):
                        recording_id = recording_id[:-4]
                    relative_path = set_file.relative_to(dataset_path).as_posix()
                    session_number = int(session_folder.name.removeprefix("ses-"))
                    records.append({
                        "frequency_sound": 0.0,
                        "delta_power": powers["delta_power"],
                        "theta_power": powers["theta_power"],
                        "alpha_power": powers["alpha_power"],
                        "beta_power": powers["beta_power"],
                        "gamma_power": powers["gamma_power"],
                        "brain_wave": brain_wave,
                        "source": f"BIDS Dataset:{relative_path}",
                        "subject_id": subject_folder.name,
                        "recording_id": recording_id,
                        "epoch_id": None,
                        "audio_dominant_frequency": None,
                        "audio_spectral_centroid": None,
                        "audio_bpm": song_tempos.get(session_number),
                    })
                except Exception as e:
                    print(f"Error loading {set_file.name}: {e}")
                    continue
    
    return records, f"Berhasil memuat {len(records)} rekaman dari BIDS Dataset"


# =========================================================
# AUDITORY DATASET (existing)
# =========================================================

# [FUNGSI 07] load_auditory_dataset
def load_auditory_dataset():
    """load_auditory_dataset: Load Auditory Evoked Potential Dataset (CSV format)"""
    dataset_path = DATASET_ROOT / "auditory-evoked-potential-eeg-biometric-dataset-1.0.0" / "auditory-evoked-potential-eeg-biometric-dataset-1.0.0"
    signal_dir = dataset_path / "Filtered_Data"
    
    if not signal_dir.exists():
        return [], f"Dataset folder tidak ditemukan: {signal_dir}"
    
    records = []
    csv_files = sorted(signal_dir.glob("s*_ex*.csv"))
    
    for file_path in csv_files:
        try:
            signal_df = pd.read_csv(file_path, usecols=["P4", "Cz", "F8", "T7"])
            signal = signal_df.to_numpy(dtype=float)
            
            # Extract features
            powers = extract_band_powers(signal, sampling_rate=200)
            brain_wave = get_dominant_wave(powers)
            
            # Extract frequency dari nama file
            experiment = int(file_path.stem.split("_ex")[1].split("_")[0])
            
            record = {
                "frequency_sound": float(experiment),
                "delta_power": powers["delta_power"],
                "theta_power": powers["theta_power"],
                "alpha_power": powers["alpha_power"],
                "beta_power": powers["beta_power"],
                "gamma_power": powers["gamma_power"],
                "brain_wave": brain_wave,
                "source": f"Auditory Dataset:{file_path.name}",
            }
            records.append(record)
        except Exception as e:
            print(f"Error loading {file_path.name}: {e}")
            continue
    
    return records, f"Berhasil load {len(records)} file dari Auditory Dataset"


# =========================================================
# MASTER LOADER
# =========================================================

# [FUNGSI 08] load_all_datasets
def load_all_datasets() -> Tuple[List[Dict], Dict[str, str]]:
    """load_all_datasets: Load semua dataset yang tersedia"""
    all_records = []
    status = {}
    
    # Load Auditory Dataset
    records, msg = load_auditory_dataset()
    all_records.extend(records)
    status["Auditory Dataset"] = msg
    
    # Load Motor Imagery Dataset
    records, msg = load_motor_imagery_edf()
    all_records.extend(records)
    status["Motor Imagery Dataset"] = msg
    
    # Load GDF Dataset
    records, msg = load_gdf_dataset()
    all_records.extend(records)
    status["GDF Dataset"] = msg
    
    # Load BIDS Dataset
    records, msg = load_bids_dataset()
    all_records.extend(records)
    status["BIDS Dataset"] = msg
    
    return all_records, status


# [FUNGSI 09] get_available_datasets
def get_available_datasets() -> List[str]:
    """get_available_datasets: Get list dari dataset yang tersedia"""
    available = []
    
    if (DATASET_ROOT / "auditory-evoked-potential-eeg-biometric-dataset-1.0.0").exists():
        available.append("Auditory Dataset")
    
    if (DATASET_ROOT / "eeg-motor-movementimagery-dataset-1.0.0").exists():
        available.append("Motor Imagery Dataset")
    
    if (DATASET_ROOT / "ppz3r5j2n2-1").exists():
        available.append("GDF Dataset")
    
    if (DATASET_ROOT / "v1.0.0").exists():
        available.append("BIDS Dataset")
    
    return available
