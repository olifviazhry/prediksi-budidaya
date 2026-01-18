import streamlit as st
import numpy as np
import joblib
import matplotlib.pyplot as plt
import pandas as pd

# =========================
# Konfigurasi halaman
# =========================
st.set_page_config(
    page_title="Prediksi Produksi Budidaya",
    layout="centered"
)

# =========================
# Load model & scaler
# =========================
@st.cache_resource
def load_artifacts():
    model = joblib.load("model_regresi_linear.pkl")
    scaler_X = joblib.load("scaler_X.pkl")
    scaler_y = joblib.load("scaler_y.pkl")
    le = joblib.load("label_encoder_kecamatan.pkl")
    return model, scaler_X, scaler_y, le

model, scaler_X, scaler_y, le = load_artifacts()

# =========================
# Judul
# =========================
st.title("🎣🐟 Dashboard Prediksi Hasil Produksi Budidaya")
st.write(
    "Sistem ini digunakan untuk memprediksi hasil produksi budidaya "
    "berdasarkan jumlah komoditas, pelaku budidaya, luas lahan, jumlah benih, "
    "dan wilayah kecamatan."
)

# =====================================================
# PREDIKSI DARI FILE EXCEL
# =====================================================

st.subheader("📂 Prediksi dari File Excel")

uploaded_file = st.file_uploader(
    "Upload file Excel (.xlsx)",
    type=["xlsx"]
)

if uploaded_file is not None:
    df = pd.read_excel(uploaded_file)

    # Normalisasi kolom
    df.columns = df.columns.str.strip().str.lower()

    kolom_wajib = [
        "jumlah_komoditas",
        "pelaku_budidaya",
        "luas_lahan",
        "jumlah_benih",
        "kode_kec"
    ]

    # =========================
    # 1️⃣ VALIDASI KOLOM
    # =========================
    kolom_hilang = set(kolom_wajib) - set(df.columns)
    if kolom_hilang:
        st.error("❌ Kolom wajib tidak ditemukan:")
        st.write(kolom_hilang)
        st.stop()

    st.write("📄 Data yang diupload:")
    st.dataframe(df)

    # =========================
    # 2️⃣ VALIDASI KECAMATAN
    # =========================
    df["kode_kec"] = df["kecamatan"].astype(str).str.strip()

    kec_excel = set(df["kecamatan"].unique())
    kec_model = set(le.classes_)

    kec_tidak_dikenali = kec_excel - kec_model

    if kec_tidak_dikenali:
        st.error("❌ Kecamatan tidak dikenali oleh sistem")
        st.write("Kecamatan bermasalah:")
        st.write(kec_tidak_dikenali)
        st.stop()

    # =========================
    # 3️⃣ VALIDASI DATA NUMERIK
    # =========================
    kolom_numerik = [
        "jumlah_komoditas",
        "pelaku_budidaya",
        "luas_lahan",
        "jumlah_benih"
    ]

    for col in kolom_numerik:
        if df[col].isnull().any():
            st.error(f"❌ Terdapat nilai kosong (NaN) pada kolom: {col}")
            st.stop()

        if not pd.api.types.is_numeric_dtype(df[col]):
            st.error(f"❌ Kolom {col} harus berupa angka")
            st.stop()

    # =========================
    # 4️⃣ ENCODING + PREDIKSI
    # =========================
    df["kode_kec"] = le.transform(df["kecamatan"])

    X = df[
        [
            "jumlah_komoditas",
            "pelaku_budidaya",
            "luas_lahan",
            "jumlah_benih",
            "kode_kec"
        ]
    ].astype(float)

    try:
        X_scaled = scaler_X.transform(X)
        y_scaled = model.predict(X_scaled)
        y_pred = scaler_y.inverse_transform(
            y_scaled.reshape(-1, 1)
        ).flatten()

    except Exception as e:
        st.error("❌ Terjadi kesalahan saat proses prediksi")
        st.code(str(e))
        st.stop()

    df["hasil_prediksi_kg"] = y_pred.astype(int)

    st.success("✅ Prediksi dari file Excel berhasil")
    st.dataframe(df)
    
# =====================================================
# INPUT MANUAL
# =====================================================
st.subheader("✍️ Prediksi Manual")

with st.form("form_prediksi"):
    col1, col2 = st.columns(2)

    with col1:
        jumlah_komoditas = st.number_input(
            "Jumlah Komoditas", min_value=0, step=1
        )
        pelaku_budidaya = st.number_input(
            "Jumlah Pelaku Budidaya", min_value=0, step=1
        )
        luas_lahan = st.number_input(
            "Luas Lahan (Ha)", min_value=0.0, step=0.1
        )

    with col2:
        jumlah_benih = st.number_input(
            "Jumlah Benih", min_value=0, step=1000
        )
        kecamatan = st.selectbox(
            "Nama Kecamatan", options=list(le.classes_)
        )

    submit = st.form_submit_button("🔍 Prediksi")

# =====================================================
# PROSES PREDIKSI MANUAL
# =====================================================
if submit:
    kode_kec = le.transform([kecamatan])[0]

    X_new = np.array([[
        jumlah_komoditas,
        pelaku_budidaya,
        luas_lahan,
        jumlah_benih,
        kode_kec
    ]])

    X_new_scaled = scaler_X.transform(X_new)
    y_pred_scaled = model.predict(X_new_scaled)
    y_pred_actual = scaler_y.inverse_transform(
        y_pred_scaled.reshape(-1, 1)
    )[0][0]

    st.success("✅ Prediksi berhasil")
    st.metric(
        "Hasil Prediksi Produksi",
        f"{int(y_pred_actual):,} kg"
    )

    # =========================
    # Grafik Tren Produksi
    # =========================
    st.subheader("📈 Tren Produksi terhadap Luas Lahan")

    luas_range = np.linspace(
        max(1, luas_lahan * 0.5),
        luas_lahan * 1.5,
        10
    )

    hasil = []

    for ll in luas_range:
        X_temp = np.array([[
            jumlah_komoditas,
            pelaku_budidaya,
            ll,
            jumlah_benih,
            kode_kec
        ]])

        X_temp_scaled = scaler_X.transform(X_temp)
        y_temp_scaled = model.predict(X_temp_scaled)
        y_temp_actual = scaler_y.inverse_transform(
            y_temp_scaled.reshape(-1, 1)
        )[0][0]

        hasil.append(y_temp_actual)

    fig, ax = plt.subplots()
    ax.plot(luas_range, hasil, marker="o")
    ax.set_xlabel("Luas Lahan (Ha)")
    ax.set_ylabel("Produksi (kg)")
    ax.set_title("Tren Produksi Budidaya")

    st.pyplot(fig)


