"""Streamlit application for FMA Small music-genre classification."""

import hashlib
import pickle

import pandas as pd
import soundfile
import streamlit as st

from ml_model import (
    FINAL_DATASET_PATH,
    MODEL_PATH,
    extract_audio_features,
    load_model,
    predict_genre,
)
from ui_components import (
    apply_custom_styling,
    display_header,
    display_sidebar_menu,
)


st.set_page_config(
    page_title=(
        "Evaluasi Pengaruh Kombinasi Fitur Audio terhadap Kinerja Random Forest "
        "pada Klasifikasi Genre Musik"
    ),
    page_icon="M",
    layout="wide",
)

apply_custom_styling()


@st.cache_resource
def get_model(modified_at: int):
    """get_model: Memuat model klasifikasi dan menyimpannya dalam cache resource Streamlit."""
    return load_model()


@st.cache_data
def get_genre_counts(modified_at: int) -> pd.Series:
    """get_genre_counts: Membaca dan menghitung jumlah trek untuk setiap genre pada dataset."""
    dataset = pd.read_csv(
        FINAL_DATASET_PATH,
        usecols=["genre_top"],
    )
    return dataset["genre_top"].value_counts().sort_index()


def display_analysis(result: dict[str, object]) -> None:
    """display_analysis: Menampilkan ringkasan fitur audio dan hasil prediksi pada halaman."""
    features = result["features"]
    prediction = result.get("prediction")

    bpm_col, bandwidth_col, centroid_col, genre_col = st.columns(4)
    bpm_col.metric("BPM", f"{features['bpm']:.1f}")
    bandwidth_col.metric(
        "Spectral bandwidth",
        f"{features['spectral_bandwidth_mean']:.2f} Hz",
    )
    centroid_col.metric(
        "Spectral centroid",
        f"{features['spectral_centroid_mean']:.2f} Hz",
    )
    genre_col.metric(
        "Genre",
        prediction["genre"] if prediction else "Model belum dilatih",
    )
    if prediction:
        genre_col.caption(f"Keyakinan prediksi: {prediction['confidence']:.1%}")

    st.subheader("MFCC")
    mfcc_rows = [
        {
            "Koefisien": f"MFCC {index:02d}",
            "Rata-rata": features[f"mfcc_{index:02d}_mean"],
            "Simpangan baku": features[f"mfcc_{index:02d}_std"],
        }
        for index in range(1, 21)
    ]
    st.dataframe(pd.DataFrame(mfcc_rows), width="stretch", hide_index=True)

    if prediction:
        st.subheader("Probabilitas genre")
        probabilities = (
            pd.Series(prediction["probabilities"], name="Probabilitas")
            .sort_values(ascending=False)
            .to_frame()
        )
        st.bar_chart(probabilities)


display_header()
menu = display_sidebar_menu()

model_artifact = None
if menu == "Input":
    model_timestamp = MODEL_PATH.stat().st_mtime_ns if MODEL_PATH.exists() else 0
    try:
        model_artifact = get_model(model_timestamp)
    except (EOFError, OSError, pickle.UnpicklingError, ValueError) as error:
        st.error(f"Model genre tidak dapat dimuat: {error}")


if menu == "Dashboard":
    st.header("Dashboard")
    if not FINAL_DATASET_PATH.is_file():
        st.error(f"Dataset tidak ditemukan: {FINAL_DATASET_PATH}")
    else:
        try:
            genre_counts = get_genre_counts(
                FINAL_DATASET_PATH.stat().st_mtime_ns,
            )
            total_tracks = int(genre_counts.sum())
            st.metric("Jumlah data", f"{total_tracks:,}")
            st.subheader("Distribusi genre")
            genre_distribution = genre_counts.rename_axis("Genre").reset_index(
                name="Jumlah data",
            )
            st.bar_chart(
                genre_distribution,
                x="Genre",
                y="Jumlah data",
                x_label="Genre",
                y_label="Jumlah data",
            )
        except (OSError, ValueError, pd.errors.ParserError) as error:
            st.error(f"Ringkasan dataset tidak dapat dimuat: {error}")


elif menu == "Input":
    st.header("Input audio")
    st.markdown(
        '<p class="section-intro">Unggah lagu untuk mengekstrak BPM, MFCC, '
        'spectral bandwidth, dan spectral centroid, lalu memprediksi genrenya.</p>',
        unsafe_allow_html=True,
    )
    uploaded_audio = st.file_uploader(
        "Pilih file audio",
        type=["mp3", "wav", "flac", "ogg"],
    )

    if uploaded_audio is not None:
        audio_bytes = uploaded_audio.getvalue()
        audio_hash = hashlib.sha256(audio_bytes).hexdigest()
        st.audio(audio_bytes, width="stretch")
        if st.button("Analisis audio", type="primary"):
            st.session_state.pop("last_analysis", None)
            try:
                with st.spinner("Mengekstrak fitur dan memprediksi genre..."):
                    features = extract_audio_features(audio_bytes)
                    prediction = (
                        predict_genre(model_artifact, features)
                        if model_artifact is not None
                        else None
                    )
                st.session_state["last_analysis"] = {
                    "filename": uploaded_audio.name,
                    "audio_hash": audio_hash,
                    "features": features,
                    "prediction": prediction,
                }
            except (
                OSError,
                RuntimeError,
                ValueError,
                soundfile.LibsndfileError,
            ) as error:
                st.error(f"Audio tidak dapat dianalisis: {error}")

        result = st.session_state.get("last_analysis")
        if result is not None and result["audio_hash"] == audio_hash:
            display_analysis(result)
            if result["prediction"] is None:
                st.warning(
                    "Fitur berhasil dihitung, tetapi model genre belum tersedia. "
                    "Latih model melalui Dashboard."
                )


elif menu == "Output":
    st.header("Output")
    result = st.session_state.get("last_analysis")
    if result is None:
        st.info("Belum ada hasil. Unggah dan analisis lagu melalui menu Input.")
    else:
        st.caption(f"Hasil analisis untuk: {result['filename']}")
        display_analysis(result)
        if result["prediction"] is None:
            st.warning(
                "Genre belum dapat diprediksi karena model Random Forest belum dilatih."
            )
