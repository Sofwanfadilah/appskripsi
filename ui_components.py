"""
Module untuk UI Components dan Styling
"""

import streamlit as st


# =========================================================
# KONFIGURASI CSS STYLING
# =========================================================

# [FUNGSI 60] apply_custom_styling
def apply_custom_styling():
    """apply_custom_styling: Menerapkan gaya visual aplikasi yang tenang dan informatif."""
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Source+Serif+4:opsz,wght@8..60,500;8..60,600&display=swap');
        :root { 
            --ink: #1d2b2a; 
            --muted: #6d7b77; 
            --paper: #f4f1eb; 
            --line: #d9ded8; 
            --accent: #b45f43;
            --mint: #dce9e3; 
            --panel: #ffffff; 
        }
        .stApp { background: var(--paper) !important; color: var(--ink) !important; }
        [data-testid="stAppViewContainer"], [data-testid="stMain"] { background-color: var(--paper) !important; color: var(--ink) !important; background-image: linear-gradient(120deg, rgba(255,255,255,.65), transparent 42%); }
        [data-testid="stMain"] p, [data-testid="stMain"] label, [data-testid="stMain"] h1, [data-testid="stMain"] h2, [data-testid="stMain"] h3, [data-testid="stMain"] span { color: var(--ink); }
        .block-container { max-width: 1240px; padding-top: 2.3rem; padding-bottom: 4rem; }
        [data-testid="stSidebar"] { background: #263a37; border-right: 1px solid #405650; }
        [data-testid="stSidebar"] * { color: #edf1ed; }
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { color: #9eb1a9; font-size: .78rem; }
        [data-testid="stSidebar"] [role="radiogroup"] { gap: .35rem; }
        [data-testid="stSidebar"] [role="radiogroup"] label { border-left: 2px solid transparent; border-radius: 6px; padding: .72rem .8rem; transition: background .2s ease, border-color .2s ease; }
        [data-testid="stSidebar"] [role="radiogroup"] label:hover, [data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) { background: #334b46; border-left-color: #d18b6c; }
        [data-testid="stMetric"] { background: rgba(255,255,255,.8); border: 1px solid var(--line); border-top: 3px solid var(--accent); padding: 1.05rem 1.15rem; border-radius: 10px; box-shadow: 0 8px 22px rgba(29,43,42,.06); }
        [data-testid="stMetricLabel"] { color: var(--muted); text-transform: uppercase; font-size: .7rem; letter-spacing: .08em; }
        [data-testid="stMetricValue"] { font-family: 'DM Sans', sans-serif; color: var(--ink); }
        h1, h2, h3 { color: var(--ink); font-family: 'Source Serif 4', serif; letter-spacing: -.02em; }
        p, label, [data-testid="stMarkdownContainer"] { font-family: 'DM Sans', sans-serif; letter-spacing: 0; }
        [data-testid="stHeader"] { background: transparent; }
        .hero { position: relative; overflow: hidden; background: #263a37; color: white; padding: 2.55rem 2.7rem; border-bottom: 4px solid var(--accent); margin-bottom: 2rem; border-radius: 12px; box-shadow: 0 12px 28px rgba(29,43,42,.13); }
        .hero:after { content: ''; position: absolute; width: 220px; height: 220px; right: -65px; top: -95px; border: 1px solid rgba(255,255,255,.18); border-radius: 50%; box-shadow: 0 0 0 22px rgba(255,255,255,.05), 0 0 0 44px rgba(255,255,255,.04); }
        .hero h1 { color: white; margin: 0; font-size: 2.45rem; letter-spacing: -.03em; }
        .hero p { margin: .65rem 0 0; color: #c4d4ce; max-width: 610px; line-height: 1.6; }
        .section-intro { color: var(--muted); font-size: .95rem; margin-top: -.45rem; margin-bottom: 1.4rem; }
        .upload-panel { background: rgba(255,255,255,.72); border: 1px solid var(--line); border-radius: 12px; padding: 1.35rem 1.45rem .75rem; box-shadow: 0 8px 22px rgba(29,43,42,.05); }
        .stButton > button { 
            border-radius: 7px; 
            border: 1px solid #b4473f; 
            font-weight: 600; 
            min-height: 2.7rem; 
            transition: transform .15s ease, box-shadow .15s ease, background .15s ease;
            background: #b4473f !important;
            color: white !important;
        }
        .stButton > button:hover { 
            transform: translateY(-1px); 
            box-shadow: 0 5px 12px rgba(180, 71, 63, .28);
            background: #963934 !important;
        }
        [data-testid="stFileUploader"] { border: 1px solid #639878; background: #e6f2e8; border-radius: 9px; padding: .35rem; }
        [data-testid="stFileUploaderDropzone"] { background: #d8ecdc; border: 1px dashed #5f9874; border-radius: 7px; }
        [data-testid="stFileUploaderDropzone"] button { background: #3f8058 !important; border-color: #3f8058 !important; color: white !important; }
        [data-testid="stFileUploaderDropzone"] button:hover { background: #2f6544 !important; }
        [data-baseweb="input"], [data-baseweb="select"] > div { background: var(--panel); border-radius: 6px; border-color: #b8c2bd; }
        [data-baseweb="input"] input, [data-baseweb="select"] input, [data-baseweb="select"] span { color: var(--ink) !important; -webkit-text-fill-color: var(--ink); }
        [data-testid="stFileUploader"] *, [data-testid="stFileUploaderDropzone"] * { color: var(--ink) !important; }
        [data-testid="stTabs"] button { color: var(--muted) !important; }
        [data-testid="stTabs"] button[aria-selected="true"] { color: var(--ink) !important; }
        [data-testid="stAlert"] * { color: var(--ink) !important; }
        [data-testid="stDataFrame"] { border: 1px solid var(--line); border-radius: 8px; overflow: hidden; background: var(--panel); }
        hr { border-color: var(--line); }
        .stAlert { border-radius: 8px; }
        [data-testid="stVerticalBlock"] > [data-testid="stElementContainer"] h2 { margin-top: .8rem; }
        @media (max-width: 720px) {
            .block-container { padding: 1.2rem .85rem 2.5rem; }
            .hero { padding: 1.7rem 1.25rem; margin-bottom: 1.3rem; }
            .hero h1 { font-size: 1.8rem; }
            [data-testid="stMetric"] { padding: .8rem; }
        }
    </style>
    """, unsafe_allow_html=True)


# [FUNGSI 61] display_header
def display_header():
    """display_header: Menampilkan identitas aplikasi klasifikasi genre musik."""
    st.markdown("""
    <div class="hero">
        <h1>Evaluasi Pengaruh Kombinasi Fitur Audio</h1>
        <p>Terhadap kinerja Random Forest pada klasifikasi genre musik menggunakan data FMA Small.</p>
    </div>
    """, unsafe_allow_html=True)


# [FUNGSI 62] display_sidebar_menu
def display_sidebar_menu():
    """display_sidebar_menu: Menampilkan navigasi dashboard, input audio, dan output prediksi."""
    st.sidebar.title("Music Genre Lab")
    st.sidebar.caption("RANDOM FOREST · FMA SMALL")
    
    menu = st.sidebar.radio(
        "NAVIGASI",
        [
            "Dashboard",
            "Input",
            "Output",
        ]
    )
    
    return menu


# [FUNGSI 63] display_metrics
def display_metrics(col1, col2, model_status, dominant_wave):
    """display_metrics: Menampilkan metrik di dashboard"""
    with col1:
        st.metric("Status Model", model_status)
    
    with col2:
        st.metric("Gelombang Dominan Dataset", dominant_wave)


# [FUNGSI 64] display_data_table
def display_data_table(df, columns_to_show=None):
    """display_data_table: Menampilkan tabel data"""
    if columns_to_show:
        df = df[columns_to_show]
    
    st.dataframe(df, width="stretch", hide_index=True)


# [FUNGSI 65] get_brain_wave_interpretation
def get_brain_wave_interpretation(brain_wave):
    """get_brain_wave_interpretation: Mendapatkan interpretasi brain wave"""
    interpretations = {
        "Delta": {
            "title": "Aktivitas Delta",
            "description": """
Aktivitas EEG menunjukkan dominasi pada pita Delta, yaitu gelombang dengan frekuensi paling rendah.

Pada kondisi normal, Delta sering ditemukan saat tubuh dalam keadaan tidur yang sangat dalam, pemulihan fisiologis, atau kondisi relaksasi ekstrem. Jika gelombang Delta sangat dominan pada kondisi sadar, hasil ini dapat menandakan adanya penurunan alertness atau kemungkinan artefak yang perlu dikaji ulang.

Secara klinis, Delta lebih terkait dengan pemulihan, tidur, dan pergantian keadaan dari aktivitas kognitif ke kondisi istirahat yang dalam.
""",
            "activities": [
                "Tidur dalam dan pemulihan tubuh",
                "Relaksasi sangat dalam",
                "Penurunan aktivitas kognitif dan kewaspadaan",
                "Perlu pengecekan apakah kondisi ini terjadi pada keadaan sadar atau ada artefak sinyal"
            ]
        },
        "Theta": {
            "title": "Aktivitas Theta",
            "description": """
Aktivitas EEG menunjukkan dominasi pada pita Theta, yaitu transisi antara kondisi relaksasi dan keadaan tidur ringan.

Theta biasanya berkaitan dengan keadaan santai, mengantuk, imajinasi, dan proses internal seperti refleksi mental. Nilai Theta yang tinggi dapat menunjukkan bahwa sistem saraf sedang dalam mode yang lebih pasif atau sedang beralih ke periode istirahat.

Pada konteks kognitif, Theta sering berkaitan dengan konsentrasi yang lebih internal daripada fokus eksternal.
""",
            "activities": [
                "Relaksasi mendalam",
                "Kondisi mengantuk atau transisi tidur",
                "Pemikiran internal dan refleksi",
                "Fokus yang lebih bersifat internal daripada eksternal"
            ]
        },
        "Alpha": {
            "title": "Aktivitas Alpha",
            "description": """
Aktivitas EEG menunjukkan dominasi pada pita Alpha.

Gelombang Alpha umumnya ditemukan saat seseorang dalam keadaan tenang, rileks, dan tetap sadar tanpa beban kognitif yang berat. Kondisi ini sering dijumpai saat istirahat, menutup mata, atau berada dalam keadaan yang santai namun tetap waspada.

Dominasi Alpha biasanya diartikan sebagai tanda relaksasi yang sehat dan keseimbangan antara keadaan tenang dan kemampuan berpikir yang masih aktif.
""",
            "activities": [
                "Relaksasi yang sehat",
                "Tetap sadar dan tenang",
                "Istirahat mental tanpa tekanan tinggi",
                "Kondisi fokus yang stabil namun tidak terlalu intens"
            ]
        },
        "Beta": {
            "title": "Aktivitas Beta",
            "description": """
Aktivitas EEG menunjukkan dominasi pada pita Beta.

Beta merupakan gelombang yang sering muncul saat kondisi mental aktif, fokus, berpikir, atau bersiap untuk merespons stimulus. Nilai Beta yang tinggi biasanya mencerminkan kewaspadaan, perhatian, dan aktivitas kognitif yang intens.

Pada kondisi tertentu, dominasi Beta dapat menjadi indikator bahwa seseorang sedang dalam proses pemecahan masalah, berpikir cepat, atau dalam keadaan observasi yang tinggi.
""",
            "activities": [
                "Kewaspadaan tinggi",
                "Fokus dan konsentrasi aktif",
                "Berpikir analitis dan responsif",
                "Aktivitas mental yang intens"
            ]
        },
        "Gamma": {
            "title": "Aktivitas Gamma",
            "description": """
Aktivitas EEG menunjukkan dominasi pada pita Gamma, yaitu gelombang frekuensi tinggi yang menandakan aktivitas neural yang sangat cepat.

Gamma biasanya berkaitan dengan integrasi informasi, proses kognitif kompleks, dan koordinasi aktivitas neuron. Bila Gamma sangat dominan, bisa mengindikasikan aktivitas otak yang tinggi atau proses pemrosesan informasi yang intens.

Namun, interpretasi Gamma perlu dilakukan hati-hati karena nilai yang tinggi juga dapat dipengaruhi oleh artefak, noise, atau kualitas sinyal yang kurang optimal.
""",
            "activities": [
                "Pemrosesan informasi yang cepat",
                "Aktivitas neural tinggi",
                "Keterlibatan proses kognitif kompleks",
                "Perlu pengecekan kualitas sinyal agar tidak terjadi artefak"
            ]
        }
    }
    
    return interpretations.get(
        brain_wave,
        {
            "title": "Tidak Diketahui",
            "description": "Interpretasi tidak tersedia karena gelombang otak yang terdeteksi tidak terdaftar dalam kategori standar. Perlu dilakukan pengecekan terhadap data input atau kualitas sinyal.",
            "activities": [
                "Data belum memenuhi kategori EEG standar",
                "Perlu validasi ulang input",
                "Kualitas sinyal perlu ditinjau"
            ]
        }
    )


# Konstanta band frequency
BAND_RANGES = {
    "Delta": "0.5-4 Hz",
    "Theta": "4-8 Hz",
    "Alpha": "8-13 Hz",
    "Beta": "13-30 Hz",
    "Gamma": "30-100 Hz",
}
