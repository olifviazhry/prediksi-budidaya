import streamlit as st
import numpy as np
import joblib
import matplotlib.pyplot as plt
import pandas as pd
import pickle

# =========================
# LOAD MODEL & TOOLS
# =========================
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler_X.pkl", "rb") as f:
    scaler_X = pickle.load(f)

with open("scaler_y.pkl", "rb") as f:
    scaler_y = pickle.load(f)

with open("label_encoder_kecamatan.pkl", "rb") as f:
    le = pickle.load(f)

# =========================
# JUDUL APLIKASI
# =========================
st.title("📊 Prediksi Produksi Perikanan")
st.write("Input data menggunakan **nama kecamatan**, sistem akan mengonversi otomatis.")

# =========================
# UPLOAD FILE EXCEL
# =========================
st.subheader("📂 Prediksi dari File Excel")

uploaded_file = st.file_uploader(
    "Upload file Excel (.xlsx)",
    type=["xlsx"]
)

if uploaded_file is not None:
    df = pd.read_excel(uploaded_file)

    # -------------------------
    # NORMALISASI NAMA KOLOM
    # -------------------------
    df.columns = df.columns.str.strip().str.lower()

    kolom_wajib = [
        "jumlah_komoditas",
        "pelaku_budidaya",
        "luas_lahan",
        "jumlah_benih",
        "kecamatan"
    ]

    # -------------------------
    # VALIDASI KOLOM
    # -------------------------
    kolom_hilang = set(kolom_wajib) - set(df.columns)
    if kolom_hilang:
        st.error("❌ Kolom wajib tidak ditemukan:")
        st.write(kolom_hilang)
        st.stop()

    st.write("📄 Data yang diupload:")
    st.dataframe(df)

    # -------------------------
    # NORMALISASI KECAMATAN
    # -------------------------
    df["kecamatan"] = (
        df["kecamatan"]
        .astype(str)
        .str.strip()
        .str.title()
    )

    # -------------------------
    # VALIDASI KECAMATAN
    # -------------------------
    kec_excel = set(df["kecamatan"].unique())
    kec_model = set(le.classes_)

    kec_tidak_dikenali = kec_excel - kec_model

    if kec_tidak_dikenali:
        st.error("❌ Kecamatan tidak dikenali oleh sistem")
        st.write("Kecamatan bermasalah:")
        st.write(kec_tidak_dikenali)
        st.stop()

    # -------------------------
    # ENCODING KECAMATAN
    # -------------------------
    df["kode_kec"] = le.transform(df["kecamatan"])

    # -------------------------
    # VALIDASI DATA NUMERIK
    # -------------------------
    kolom_numerik = [
        "jumlah_komoditas",
        "pelaku_budidaya",
        "luas_lahan",
        "jumlah_benih"
    ]

    for col in kolom_numerik:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        if df[col].isnull().any():
            st.error(f"❌ Kolom {col} harus berupa angka dan tidak boleh kosong")
            st.stop()

    # -------------------------
    # PREDIKSI
    # -------------------------
    X = df[
        [
            "jumlah_komoditas",
            "pelaku_budidaya",
            "luas_lahan",
            "jumlah_benih",
            "kode_kec"
        ]
    ]

    X_scaled = scaler_X.transform(X)
    y_scaled = model.predict(X_scaled)
    y_pred = scaler_y.inverse_transform(
        y_scaled.reshape(-1, 1)
    ).flatten()

    df["hasil_prediksi_kg"] = y_pred.astype(int)

    # -------------------------
    # OUTPUT
    # -------------------------
    st.success("✅ Prediksi berhasil dilakukan")
    st.dataframe(df)

    # -------------------------
    # DOWNLOAD HASIL
    # -------------------------
    output_file = "hasil_prediksi.xlsx"
    df.to_excel(output_file, index=False)

    with open(output_file, "rb") as f:
        st.download_button(
            "⬇️ Download Hasil Prediksi",
            f,
            file_name=output_file,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )



