import streamlit as st
import numpy as np
import joblib
import matplotlib.pyplot as plt

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
st.title("🌱 Dashboard Prediksi Produksi Budidaya")
st.write("Masukkan data berikut untuk memprediksi hasil produksi budidaya.")

# =========================
# Form Input
# =========================
with st.form("form_prediksi"):
    col1, col2 = st.columns(2)

    with col1:
        jumlah_komoditas = st.number_input("Jumlah Komoditas", min_value=0, step=1)
        pelaku_budidaya = st.number_input("Jumlah Pelaku Budidaya", min_value=0, step=1)
        luas_lahan = st.number_input("Luas Lahan (Ha)", min_value=0.0, step=0.1)

    with col2:
        jumlah_benih = st.number_input("Jumlah Benih", min_value=0, step=1000)
        kecamatan = st.selectbox("Nama Kecamatan", options=list(le.classes_))

    submit = st.form_submit_button("🔍 Prediksi")

# =========================
# Proses Prediksi
# =========================
if submit:
    kecamatan_encoded = le.transform([kecamatan])[0]

    X_new = np.array([[
        jumlah_komoditas,
        pelaku_budidaya,
        luas_lahan,
        jumlah_benih,
        kecamatan_encoded
    ]])

    X_new_scaled = scaler_X.transform(X_new)
    y_pred_scaled = model.predict(X_new_scaled)
    y_pred_actual = scaler_y.inverse_transform(
        y_pred_scaled.reshape(-1, 1)
    )[0][0]

    # =========================
    # Output Angka
    # =========================
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
            kecamatan_encoded
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