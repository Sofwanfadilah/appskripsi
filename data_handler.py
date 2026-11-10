"""
Module untuk Data Handler
Menangani database operations dan data loading
"""

import sqlite3
import pandas as pd
from pathlib import Path
from io import BytesIO
from datetime import datetime
import hashlib
import numpy as np
import soundfile as sf
from scipy.signal import find_peaks

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
DATASET_ROOT = PROJECT_ROOT / "Data" / "auditory-evoked-potential-eeg-biometric-dataset-1.0.0" / "auditory-evoked-potential-eeg-biometric-dataset-1.0.0"
BIDS_ROOT = PROJECT_ROOT / "Data" / "v1.0.0"

DATABASE_PATH = DATA_DIR / "eeg_database.db"

BRAIN_WAVE_BANDS = {
    "Delta": (0.5, 4),
    "Theta": (4, 8),
    "Alpha": (8, 13),
    "Beta": (13, 30),
    "Gamma": (30, 100),
}
BIDS_SOURCE_PREFIXES = ("BIDS Dataset:", "BIDS v1.0.0:")
RESEARCH_DATASET_COLUMNS = (
    "id",
    "subject_id",
    "recording_id",
    "epoch_id",
    "delta_power",
    "theta_power",
    "alpha_power",
    "beta_power",
    "gamma_power",
    "dominant_brain_wave",
)


class EEGDatabase:
    """Kelas untuk menangani database EEG"""
    
    # [FUNGSI 10] __init__
    def __init__(self, db_path=None):
        """__init__: Inisialisasi database EEG dan siapkan tabel yang diperlukan."""
        if db_path is None:
            db_path = DATABASE_PATH
        self.db_path = db_path
        self.create_tables()
    
    # [FUNGSI 11] get_connection
    def get_connection(self):
        """get_connection: Mendapatkan koneksi database"""
        return sqlite3.connect(str(self.db_path), check_same_thread=False)
    
    # [FUNGSI 12] create_tables
    def create_tables(self):
        """create_tables: Membuat tabel database jika belum ada"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS eeg_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT,
            frequency_sound REAL,
            delta_power REAL,
            theta_power REAL,
            alpha_power REAL,
            beta_power REAL,
            gamma_power REAL,
            brain_wave TEXT,
            source TEXT,
            audio_filename TEXT,
            audio_data BLOB,
            subject_id TEXT,
            recording_id TEXT,
            epoch_id TEXT,
            audio_dominant_frequency REAL,
            audio_spectral_centroid REAL,
            audio_bpm REAL
        )
        """)
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS audio_samples (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            file_hash TEXT NOT NULL UNIQUE,
            file_size_bytes INTEGER NOT NULL,
            sample_rate_hz INTEGER NOT NULL,
            channel_count INTEGER NOT NULL,
            analysis_duration_seconds REAL NOT NULL,
            dominant_frequency_hz REAL,
            spectral_centroid_hz REAL,
            tempo_bpm REAL,
            energy REAL,
            zero_crossing_rate REAL,
            audio_power_1_4 REAL,
            audio_power_4_8 REAL,
            audio_power_8_13 REAL,
            audio_power_13_30 REAL,
            audio_power_30_45 REAL,
            left_carrier_frequency_hz REAL,
            right_carrier_frequency_hz REAL,
            binaural_beat_hz REAL,
            binaural_confidence REAL,
            binaural_windows_analyzed INTEGER NOT NULL,
            brain_wave_label TEXT,
            label_method TEXT NOT NULL,
            processed_at TEXT NOT NULL
        )
        """)

        columns = {
            row[1] for row in cursor.execute("PRAGMA table_info(eeg_data)").fetchall()
        }
        new_columns = {
            "audio_filename": "TEXT",
            "audio_data": "BLOB",
            "subject_id": "TEXT",
            "recording_id": "TEXT",
            "epoch_id": "TEXT",
            "audio_dominant_frequency": "REAL",
            "audio_spectral_centroid": "REAL",
            "audio_bpm": "REAL",
        }
        for name, data_type in new_columns.items():
            if name not in columns:
                cursor.execute(f"ALTER TABLE eeg_data ADD COLUMN {name} {data_type}")
        audio_sample_columns = {
            row[1]
            for row in cursor.execute(
                "PRAGMA table_info(audio_samples)"
            ).fetchall()
        }
        if "binaural_windows_analyzed" not in audio_sample_columns:
            cursor.execute(
                """
                ALTER TABLE audio_samples
                ADD COLUMN binaural_windows_analyzed INTEGER NOT NULL DEFAULT 0
                """
            )
        
        conn.commit()
        conn.close()
    
    # [FUNGSI 13] load_all_data
    def load_all_data(self):
        """load_all_data: Memuat semua data dari database"""
        conn = self.get_connection()
        query = "SELECT * FROM eeg_data ORDER BY id DESC"
        df = pd.read_sql_query(query, conn)
        conn.close()
        return df

    # [FUNGSI 14] save_audio_sample
    def save_audio_sample(self, sample):
        """save_audio_sample: Menyimpan fitur audio tanpa mencampurnya dengan label EEG."""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT OR IGNORE INTO audio_samples (
                filename, file_hash, file_size_bytes, sample_rate_hz,
                channel_count, analysis_duration_seconds,
                dominant_frequency_hz, spectral_centroid_hz, tempo_bpm,
                energy, zero_crossing_rate, audio_power_1_4, audio_power_4_8,
                audio_power_8_13, audio_power_13_30, audio_power_30_45,
                left_carrier_frequency_hz, right_carrier_frequency_hz,
                binaural_beat_hz, binaural_confidence,
                binaural_windows_analyzed, brain_wave_label, label_method,
                processed_at
            )
            VALUES (
                :filename, :file_hash, :file_size_bytes, :sample_rate_hz,
                :channel_count, :analysis_duration_seconds,
                :dominant_frequency_hz, :spectral_centroid_hz, :tempo_bpm,
                :energy, :zero_crossing_rate, :audio_power_1_4,
                :audio_power_4_8, :audio_power_8_13, :audio_power_13_30,
                :audio_power_30_45, :left_carrier_frequency_hz,
                :right_carrier_frequency_hz, :binaural_beat_hz,
                :binaural_confidence, :binaural_windows_analyzed,
                :brain_wave_label, :label_method, :processed_at
            )
            """,
            sample,
        )
        inserted = cursor.rowcount == 1
        conn.commit()
        conn.close()
        return inserted

    # [FUNGSI 15] has_audio_sample
    def has_audio_sample(self, file_hash):
        """has_audio_sample: Mengecek apakah file audio dengan hash yang sama sudah diimpor."""
        conn = self.get_connection()
        row = conn.execute(
            "SELECT 1 FROM audio_samples WHERE file_hash = ?",
            (file_hash,),
        ).fetchone()
        conn.close()
        return row is not None

    # [FUNGSI 16] load_audio_samples
    def load_audio_samples(self):
        """load_audio_samples: Memuat fitur lagu hasil impor, termasuk label binaural bila valid."""
        conn = self.get_connection()
        df = pd.read_sql_query(
            "SELECT * FROM audio_samples ORDER BY id DESC",
            conn,
        )
        conn.close()
        return df

    # [FUNGSI 17] save_audio_sample_to_dataset
    def save_audio_sample_to_dataset(self, sample):
        """save_audio_sample_to_dataset: Salin hasil analisis lagu ke dataset utama dengan sumber yang jelas."""
        source = (
            f"Audio Input: {sample['filename']} "
            f"[sha256:{sample['file_hash']}]"
        )
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT 1 FROM eeg_data WHERE source = ?", (source,))
        if cursor.fetchone() is not None:
            conn.close()
            return False

        brain_wave = sample.get("brain_wave_label") or "Tidak terklasifikasi"
        cursor.execute(
            """
            INSERT INTO eeg_data (
                created_at, frequency_sound, brain_wave, source,
                audio_filename, recording_id, audio_dominant_frequency,
                audio_spectral_centroid, audio_bpm
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                sample.get("processed_at")
                or datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                sample.get("binaural_beat_hz"),
                brain_wave,
                source,
                sample["filename"],
                sample["filename"],
                sample.get("dominant_frequency_hz"),
                sample.get("spectral_centroid_hz"),
                sample.get("tempo_bpm"),
            ),
        )
        conn.commit()
        conn.close()
        return True

    # [FUNGSI 18] sync_audio_samples_to_dataset
    def sync_audio_samples_to_dataset(self):
        """sync_audio_samples_to_dataset: Pastikan semua hasil lagu tersimpan juga muncul di dataset utama."""
        samples = self.load_audio_samples()
        return sum(
            self.save_audio_sample_to_dataset(sample)
            for sample in samples.to_dict(orient="records")
        )
    
    # [FUNGSI 19] save_eeg_data
    def save_eeg_data(self, frequency_sound, delta_power, theta_power, 
                      alpha_power, beta_power, gamma_power, brain_wave, 
                      source="Manual Input", audio_filename=None, audio_data=None,
                      subject_id=None, recording_id=None, epoch_id=None,
                      audio_dominant_frequency=None, audio_spectral_centroid=None,
                      audio_bpm=None):
        """save_eeg_data: Menyimpan data EEG ke database"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        cursor.execute("""
        INSERT INTO eeg_data (
            created_at, frequency_sound, delta_power, theta_power,
            alpha_power, beta_power, gamma_power, brain_wave, source,
            audio_filename, audio_data, subject_id, recording_id, epoch_id,
            audio_dominant_frequency, audio_spectral_centroid, audio_bpm
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            created_at, frequency_sound, delta_power, theta_power,
            alpha_power, beta_power, gamma_power, brain_wave, source,
            audio_filename, audio_data, subject_id, recording_id, epoch_id,
            audio_dominant_frequency, audio_spectral_centroid, audio_bpm
        ))
        
        conn.commit()
        conn.close()

    # [FUNGSI 20] delete_eeg_data_by_source
    def delete_eeg_data_by_source(self, source):
        """delete_eeg_data_by_source: Menghapus baris rekaman dan epoch lama untuk satu sumber."""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            DELETE FROM eeg_data
            WHERE source = ?
               OR substr(source, 1, ?) = ?
            """,
            (source, len(source) + 1, f"{source}#"),
        )
        deleted = cursor.rowcount
        conn.commit()
        conn.close()
        return deleted

    # [FUNGSI 21] load_final_dataset
    def load_final_dataset(self, include_audio_features=False):
        """load_final_dataset: Memuat dataset dengan kolom identitas dan fitur yang siap diekspor."""
        df = self.load_all_data()

        for column in (
            "subject_id",
            "recording_id",
            "epoch_id",
            "audio_dominant_frequency",
            "audio_spectral_centroid",
            "audio_bpm",
        ):
            if column not in df:
                df[column] = None

        audio_feature_cache = {}
        for index, row in df.iterrows():
            source = str(row.get("source") or "")
            if "#epoch-" in source:
                base_source, epoch_suffix = source.split("#epoch-", maxsplit=1)
            else:
                base_source, epoch_suffix = source, None

            if (
                pd.isna(row.get("subject_id"))
                and base_source.startswith(BIDS_SOURCE_PREFIXES)
            ):
                relative_path = base_source.split(":", maxsplit=1)[1]
                parts = Path(relative_path).parts
                subject_id = next(
                    (part for part in parts if part.startswith("sub-")), None
                )
                if subject_id is None:
                    file_subject = Path(relative_path).name.split("_", maxsplit=1)[0]
                    if file_subject.startswith("sub-"):
                        subject_id = file_subject
                if subject_id:
                    df.at[index, "subject_id"] = subject_id

            if pd.isna(row.get("recording_id")):
                if base_source.startswith(BIDS_SOURCE_PREFIXES):
                    recording_name = Path(
                        base_source.split(":", maxsplit=1)[1]
                    ).stem
                    if recording_name.endswith("_eeg"):
                        recording_name = recording_name[:-4]
                else:
                    recording_name = row.get("audio_filename")
                    if pd.isna(recording_name):
                        recording_name = f"record-{row['id']:06d}"
                df.at[index, "recording_id"] = recording_name

            if pd.isna(row.get("epoch_id")) and epoch_suffix:
                df.at[index, "epoch_id"] = epoch_suffix

            audio_filename = row.get("audio_filename")
            audio_blob = row.get("audio_data")
            has_audio = (
                (pd.notna(audio_filename) and bool(audio_filename))
                or (audio_blob is not None and not pd.isna(audio_blob))
            )
            if has_audio:
                missing_features = pd.isna(row.get("audio_spectral_centroid"))
                if include_audio_features and missing_features and audio_blob is not None:
                    audio_bytes = bytes(audio_blob)
                    audio_key = hashlib.sha256(audio_bytes).digest()
                    if audio_key not in audio_feature_cache:
                        audio_feature_cache[audio_key] = (
                            EEGSignalProcessor.extract_audio_features(
                                BytesIO(audio_bytes)
                            )
                        )
                    features = audio_feature_cache[audio_key]
                    df.at[index, "audio_dominant_frequency"] = features["dominant_frequency"]
                    df.at[index, "audio_spectral_centroid"] = features["spectral_centroid"]
                    df.at[index, "audio_bpm"] = features["tempo_bpm"]
                    self._update_audio_features(
                        row["id"],
                        features["dominant_frequency"],
                        features["spectral_centroid"],
                        features["tempo_bpm"],
                    )

        df["dominant_brain_wave"] = df["brain_wave"]
        if "audio_data" in df:
            df = df.drop(columns=["audio_data"])
        return df

    # [FUNGSI 22] load_research_dataset
    def load_research_dataset(self):
        """load_research_dataset: Return BIDS EEG band-power features and labels for model training."""
        df = self.load_final_dataset()
        bids_mask = df["source"].astype(str).str.startswith(
            BIDS_SOURCE_PREFIXES
        )
        research_df = df.loc[bids_mask].copy()
        return research_df.reindex(columns=RESEARCH_DATASET_COLUMNS)

    # [FUNGSI 23] _update_audio_features
    def _update_audio_features(
        self, data_id, dominant_frequency, spectral_centroid, bpm
    ):
        """_update_audio_features: Menyimpan fitur audio yang dihitung saat menyiapkan dataset akhir."""
        conn = self.get_connection()
        conn.execute(
            """
            UPDATE eeg_data
            SET audio_dominant_frequency = ?,
                audio_spectral_centroid = ?,
                audio_bpm = ?
            WHERE id = ?
            """,
            (dominant_frequency, spectral_centroid, bpm, data_id),
        )
        conn.commit()
        conn.close()
    
    # [FUNGSI 24] delete_eeg_data
    def delete_eeg_data(self, data_id):
        """delete_eeg_data: Menghapus data EEG dari database"""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM eeg_data WHERE id = ?", (data_id,))
        conn.commit()
        conn.close()
    
    # [FUNGSI 25] get_data_count
    def get_data_count(self):
        """get_data_count: Mendapatkan jumlah total data"""
        df = self.load_all_data()
        return len(df)
    
    # [FUNGSI 26] get_data_by_brain_wave
    def get_data_by_brain_wave(self):
        """get_data_by_brain_wave: Mendapatkan distribusi data per brain wave"""
        df = self.load_all_data()
        if len(df) == 0:
            return {}
        return df["brain_wave"].value_counts().to_dict()


class EEGSignalProcessor:
    """Kelas untuk memproses sinyal EEG"""
    
    @staticmethod
    # [FUNGSI 27] extract_band_powers
    def extract_band_powers(signal, sampling_rate=200):
        """extract_band_powers: Ekstrak power dari setiap band EEG"""
        # [HITUNG EEG 01] Band power EEG = rata-rata daya FFT pada rentang band.
        centered = signal - np.mean(signal, axis=0, keepdims=True)
        spectrum = np.abs(np.fft.rfft(centered, axis=0)) ** 2 / len(centered)
        frequencies = np.fft.rfftfreq(len(centered), d=1 / sampling_rate)
        
        bands = {
            "delta_power": (0.5, 4),
            "theta_power": (4, 8),
            "alpha_power": (8, 13),
            "beta_power": (13, 30),
            "gamma_power": (30, 100),
        }
        
        powers = {}
        for name, (low, high) in bands.items():
            mask = (frequencies >= low) & (frequencies < high)
            powers[name] = float(np.mean(spectrum[mask])) if mask.any() else 0.0
        
        return powers
    
    @staticmethod
    # [FUNGSI 28] extract_audio_frequency
    def extract_audio_frequency(audio_file):
        """extract_audio_frequency: Ekstrak frekuensi dominan dari file audio"""
        features = EEGSignalProcessor.extract_audio_features(audio_file)
        return features["dominant_frequency"], features["band_powers"]

    @staticmethod
    # [FUNGSI 29] extract_audio_features
    def extract_audio_features(audio_file, max_duration_seconds=None):
        """extract_audio_features: Ekstrak karakteristik audio, opsional dari cuplikan berdurasi tetap."""
        audio_file.seek(0)
        frames = -1
        if max_duration_seconds is not None:
            if max_duration_seconds <= 0:
                raise ValueError("Durasi analisis audio harus lebih dari 0.")
            info = sf.info(audio_file)
            frames = max(
                1, int(info.samplerate * max_duration_seconds)
            )
            audio_file.seek(0)
        samples, sample_rate = sf.read(
            audio_file,
            frames=frames,
            dtype="float32",
            always_2d=False,
        )
        
        if samples.ndim > 1:
            samples = np.mean(samples, axis=1)
        
        if samples.size == 0:
            raise ValueError("File audio kosong atau tidak dapat dibaca.")
        
        spectrum = np.abs(np.fft.rfft(samples))
        frequencies = np.fft.rfftfreq(samples.size, d=1 / sample_rate)
        
        if spectrum.size < 2 or not np.any(spectrum[1:]):
            raise ValueError("Frekuensi dominan tidak dapat dihitung dari file audio.")
        
        # [HITUNG AUDIO 01] Daya spektrum = kuadrat amplitudo FFT / jumlah sampel.
        power_spectrum = spectrum ** 2 / samples.size
        
        bands = {
            "delta_power": (0.5, 4),
            "theta_power": (4, 8),
            "alpha_power": (8, 13),
            "beta_power": (13, 30),
            "gamma_power": (30, 100),
        }
        
        # [HITUNG AUDIO 02] Band power = rata-rata daya pada tiap rentang frekuensi.
        powers = {}
        for name, (low, high) in bands.items():
            mask = (frequencies >= low) & (frequencies < high)
            powers[name] = float(np.mean(power_spectrum[mask])) if mask.any() else 0.0
        
        # [HITUNG AUDIO 03] Frekuensi dominan = bin FFT dengan amplitudo terbesar (DC dikecualikan).
        dominant_freq = float(frequencies[1:][np.argmax(spectrum[1:])])
        max_frequency = min(float(sample_rate / 2), 5000.0)
        domain_mask = frequencies <= max_frequency
        domain_frequencies = frequencies[domain_mask]
        domain_amplitudes = spectrum[domain_mask]
        if len(domain_frequencies) > 2000:
            selected_indices = np.linspace(
                0, len(domain_frequencies) - 1, 2000, dtype=int
            )
            domain_frequencies = domain_frequencies[selected_indices]
            domain_amplitudes = domain_amplitudes[selected_indices]

        frame_length = min(2048, samples.size)
        hop_length = max(frame_length // 2, 1)
        frame_starts = range(0, max(samples.size - frame_length + 1, 1), hop_length)
        frames = np.array([
            samples[start:start + frame_length]
            for start in frame_starts
            if samples[start:start + frame_length].size == frame_length
        ])
        if frames.size == 0:
            frames = samples.reshape(1, -1)

        window = np.hanning(frames.shape[1])
        windowed = frames * window
        frame_rms = np.sqrt(np.mean(frames ** 2, axis=1))
        frame_spectrum = np.abs(np.fft.rfft(windowed, axis=1))
        frame_frequencies = np.fft.rfftfreq(frames.shape[1], d=1 / sample_rate)
        spectrum_sum = np.sum(frame_spectrum, axis=1)
        # [HITUNG AUDIO 04] Centroid = jumlah(amplitudo x frekuensi) / jumlah(amplitudo).
        centroid_values = np.divide(
            frame_spectrum @ frame_frequencies,
            spectrum_sum,
            out=np.zeros_like(spectrum_sum),
            where=spectrum_sum > 0,
        )
        # [HITUNG AUDIO 05] RMS energi dan rasio zero-crossing audio.
        energy = float(np.sqrt(np.mean(samples ** 2)))
        centroid = float(np.mean(centroid_values))
        zero_crossing_rate = float(np.mean(np.abs(np.diff(np.signbit(samples)))))
        tempo = EEGSignalProcessor._estimate_tempo(frame_rms, sample_rate, hop_length)
        atmosphere, mood = EEGSignalProcessor._classify_audio_mood(
            tempo, energy, centroid, zero_crossing_rate
        )

        return {
            "dominant_frequency": dominant_freq,
            "spectral_centroid": centroid,
            "band_powers": powers,
            "frequency_domain": {
                "frequencies": domain_frequencies.tolist(),
                "amplitudes": domain_amplitudes.tolist(),
            },
            "tempo_bpm": tempo,
            "energy": energy,
            "zero_crossing_rate": zero_crossing_rate,
            "atmosphere": atmosphere,
            "mood": mood,
        }

    @staticmethod
    # [FUNGSI 30] analyze_binaural_beat
    def analyze_binaural_beat(
        audio_file,
        max_duration_seconds=None,
        window_seconds=5,
        max_windows=24,
        consistency_tolerance_hz=0.75,
    ):
        """analyze_binaural_beat: Estimasi beat dari selisih carrier dominan kanal kiri dan kanan."""
        if (
            (max_duration_seconds is not None and max_duration_seconds <= 0)
            or window_seconds <= 0
            or max_windows < 1
        ):
            raise ValueError("Durasi analisis harus lebih dari 0.")
        if consistency_tolerance_hz <= 0:
            raise ValueError("Toleransi konsistensi harus lebih dari 0.")

        audio_file.seek(0)
        info = sf.info(audio_file)
        sample_rate = int(info.samplerate)
        frame_count = info.frames
        if max_duration_seconds is not None:
            frame_count = min(
                frame_count,
                max(1, int(sample_rate * max_duration_seconds)),
            )
        audio_file.seek(0)
        if frame_count == 0:
            raise ValueError("File audio kosong atau tidak dapat dibaca.")

        result = {
            "sample_rate_hz": int(sample_rate),
            "channel_count": int(info.channels),
            "analysis_duration_seconds": float(frame_count / sample_rate),
            "left_carrier_frequency_hz": None,
            "right_carrier_frequency_hz": None,
            "binaural_beat_hz": None,
            "binaural_confidence": None,
            "binaural_windows_analyzed": 0,
            "brain_wave_label": None,
        }
        if info.channels < 2:
            return result

        frame_length = min(
            int(sample_rate * window_seconds),
            frame_count,
        )
        if frame_length < 2:
            return result
        window = np.hanning(frame_length)
        min_bin = max(int(np.ceil(80 * frame_length / sample_rate)), 1)
        max_bin = min(
            int(np.floor(1500 * frame_length / sample_rate)),
            frame_length // 2,
        )
        if max_bin <= min_bin:
            return result

        available_windows = max(1, frame_count // frame_length)
        window_count = min(max_windows, available_windows)
        starts = np.linspace(
            0,
            max(frame_count - frame_length, 0),
            num=window_count,
            dtype=int,
        )
        candidate_pairs = {}
        windows_analyzed = 0
        with sf.SoundFile(audio_file, closefd=False) as audio_stream:
            for window_index, start in enumerate(starts):
                audio_stream.seek(int(start))
                frame = audio_stream.read(
                    frames=frame_length,
                    dtype="float32",
                    always_2d=True,
                )
                if len(frame) < frame_length:
                    continue
                frame = frame[:, :2]
                frame = frame - np.mean(frame, axis=0, keepdims=True)
                spectrum = np.abs(np.fft.rfft(frame * window[:, None], axis=0))
                peaks_by_channel = []
                for channel in range(2):
                    band_spectrum = spectrum[min_bin:max_bin + 1, channel]
                    max_peak = float(np.max(band_spectrum))
                    baseline = max(
                        float(np.median(band_spectrum)),
                        np.finfo(float).eps,
                    )
                    peak_indices, _ = find_peaks(
                        band_spectrum,
                        distance=max(
                            1, int(round(0.5 * frame_length / sample_rate))
                        ),
                    )
                    peak_indices = [
                        int(index) for index in peak_indices
                        if (
                            band_spectrum[index] / baseline >= 5
                            and band_spectrum[index] >= max_peak * 0.01
                        )
                    ]
                    peak_indices = sorted(
                        peak_indices,
                        key=lambda index: band_spectrum[index],
                        reverse=True,
                    )[:30]
                    channel_peaks = []
                    for relative_index in peak_indices:
                        peak_index = relative_index + min_bin
                        peak_frequency = peak_index * sample_rate / frame_length
                        left_amp, peak_amp, right_amp = spectrum[
                            peak_index - 1:peak_index + 2, channel
                        ]
                        denominator = left_amp - 2 * peak_amp + right_amp
                        if denominator != 0:
                            offset = 0.5 * (left_amp - right_amp) / denominator
                            if np.isfinite(offset) and abs(offset) <= 0.5:
                                peak_frequency += (
                                    offset * sample_rate / frame_length
                                )
                        channel_peaks.append(
                            (
                                float(peak_frequency),
                                float(spectrum[peak_index, channel] / baseline),
                            )
                        )
                    peaks_by_channel.append(channel_peaks)

                window_pairs = {}
                for left_frequency, left_strength in peaks_by_channel[0]:
                    for right_frequency, right_strength in peaks_by_channel[1]:
                        # [HITUNG BINAURAL 01] Beat = absolut selisih carrier kiri dan kanan.
                        beat_frequency = abs(left_frequency - right_frequency)
                        if not 0.5 <= beat_frequency <= 100:
                            continue
                        key = (round(left_frequency), round(right_frequency))
                        pair = (
                            left_frequency,
                            right_frequency,
                            beat_frequency,
                            min(left_strength, right_strength),
                        )
                        if (
                            key not in window_pairs
                            or pair[3] > window_pairs[key][3]
                        ):
                            window_pairs[key] = pair

                for key, pair in window_pairs.items():
                    candidate_pairs.setdefault(key, {})[window_index] = pair
                windows_analyzed += 1

        result["binaural_windows_analyzed"] = windows_analyzed
        if not candidate_pairs:
            return result

        tolerance = max(1, int(np.ceil(consistency_tolerance_hz)))
        candidates = []
        for left_center, right_center in candidate_pairs:
            matched_pairs = {}
            for left_offset in range(-tolerance, tolerance + 1):
                for right_offset in range(-tolerance, tolerance + 1):
                    pairs = candidate_pairs.get(
                        (
                            left_center + left_offset,
                            right_center + right_offset,
                        ),
                        {},
                    )
                    for index, pair in pairs.items():
                        if (
                            index not in matched_pairs
                            or pair[3] > matched_pairs[index][3]
                        ):
                            matched_pairs[index] = pair
            if len(matched_pairs) < 3:
                continue
            confidence = len(matched_pairs) / windows_analyzed
            if confidence < 0.6:
                continue
            matched_values = list(matched_pairs.values())
            quality = float(np.median([pair[3] for pair in matched_values]))
            candidates.append(
                (
                    confidence,
                    quality,
                    float(np.median([pair[2] for pair in matched_values])),
                    float(np.median([pair[0] for pair in matched_values])),
                    float(np.median([pair[1] for pair in matched_values])),
                )
            )
        if not candidates:
            return result

        confidence, _, beat_frequency, left_frequency, right_frequency = max(
            candidates,
            key=lambda candidate: (candidate[0], candidate[1]),
        )
        # [HITUNG BINAURAL 02] Carrier/beat akhir = median kandidat konsisten antar-jendela.
        result.update(
            {
                "left_carrier_frequency_hz": left_frequency,
                "right_carrier_frequency_hz": right_frequency,
                "binaural_beat_hz": beat_frequency,
                "binaural_confidence": confidence,
            }
        )
        result["brain_wave_label"] = next(
            (
                label
                for label, (low, high) in BRAIN_WAVE_BANDS.items()
                if low <= beat_frequency < high
                or (label == "Gamma" and beat_frequency <= high)
            ),
            None,
        )
        return result

    @staticmethod
    # [FUNGSI 31] _estimate_tempo
    def _estimate_tempo(frame_rms, sample_rate, hop_length):
        """_estimate_tempo: Perkiraan tempo berbasis autokorelasi envelope energi."""
        # [HITUNG AUDIO 06] BPM = 60 x sample_rate / (lag puncak autokorelasi x hop_length).
        if len(frame_rms) < 4 or not np.any(frame_rms):
            return None
        envelope = frame_rms - np.mean(frame_rms)
        autocorrelation = np.correlate(envelope, envelope, mode="full")[len(envelope) - 1:]
        min_lag = max(int(sample_rate * 60 / (200 * hop_length)), 1)
        max_lag = min(int(sample_rate * 60 / (40 * hop_length)), len(autocorrelation) - 1)
        if max_lag <= min_lag:
            return None
        lag = min_lag + int(np.argmax(autocorrelation[min_lag:max_lag + 1]))
        return round(float(60 * sample_rate / (lag * hop_length)), 1)

    @staticmethod
    # [FUNGSI 32] _classify_audio_mood
    def _classify_audio_mood(tempo, energy, centroid, zero_crossing_rate):
        """_classify_audio_mood: Klasifikasi deskriptif sederhana, bukan diagnosis atau label genre."""
        if tempo is None:
            tempo = 0
        if tempo < 75:
            atmosphere = "tenang dan lapang"
        elif tempo < 120:
            atmosphere = "stabil dan seimbang"
        else:
            atmosphere = "dinamis dan berenergi"

        if energy < 0.03 and centroid < 1800:
            mood = "lembut / kontemplatif"
        elif energy > 0.15 or centroid > 3500 or zero_crossing_rate > 0.15:
            mood = "cerah / stimulatif"
        else:
            mood = "netral / fokus"
        return atmosphere, mood
    
    @staticmethod
    # [FUNGSI 33] load_database_eeg_signal
    def load_database_eeg_signal(source):
        """load_database_eeg_signal: Memuat sinyal EEG asli dari database"""
        bids_prefix = "BIDS Dataset:"
        legacy_bids_prefix = "BIDS v1.0.0:"
        if str(source).startswith(legacy_bids_prefix):
            source = bids_prefix + str(source)[len(legacy_bids_prefix):]
        if str(source).startswith(bids_prefix):
            try:
                import mne
            except ImportError as error:
                raise ImportError("MNE diperlukan untuk membaca sinyal BIDS.") from error

            signal_path = BIDS_ROOT / str(source)[len(bids_prefix):]
            if not signal_path.exists():
                raise FileNotFoundError(f"File sinyal BIDS tidak ditemukan: {signal_path.name}")
            raw = mne.io.read_raw_eeglab(str(signal_path), preload=True, verbose=False)
            return pd.DataFrame(raw.get_data().T, columns=raw.ch_names)

        prefix = "Dataset EEG:"
        if not str(source).startswith(prefix):
            raise ValueError("Rekaman ini tidak memiliki file sinyal EEG asli.")
        
        signal_path = DATASET_ROOT / "Filtered_Data" / str(source)[len(prefix):]
        if not signal_path.exists():
            raise FileNotFoundError(f"File sinyal EEG tidak ditemukan: {signal_path.name}")
        
        return pd.read_csv(signal_path, usecols=["P4", "Cz", "F8", "T7"])
