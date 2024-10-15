import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

# Load dataset (pastikan file CSV berada di path yang benar)
@st.cache
def load_data():
    file_path = "house_price_regression_dataset.csv"  # Sesuaikan path dataset Anda
    data = pd.read_csv(file_path)
    return data

# Load model (pastikan file pkl berada di path yang benar)
@st.cache(allow_output_mutation=True)
def load_model():
    with open('house_price_model.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

# Load data dan model
data = load_data()
model = load_model()

# Scaler untuk normalisasi input data
scaler = StandardScaler()
X = data.drop(columns='House_Price')
y = data['House_Price']
scaler.fit(X)  # Fitting scaler hanya pada fitur (X)

# Streamlit app
st.title("Prediksi Harga Rumah")

# Form input pengguna
st.header("Masukkan Data Rumah")

square_footage = st.number_input("Luas Rumah (Square Footage)", min_value=500, max_value=10000, value=1000)
num_bedrooms = st.slider("Jumlah Kamar Tidur", min_value=1, max_value=10, value=3)
num_bathrooms = st.slider("Jumlah Kamar Mandi", min_value=1, max_value=5, value=2)
year_built = st.slider("Tahun Dibangun", min_value=1900, max_value=2023, value=2000)
lot_size = st.number_input("Ukuran Tanah (Lot Size)", min_value=500, max_value=50000, value=2000)
garage_size = st.slider("Jumlah Garasi", min_value=0, max_value=5, value=1)
neighborhood_quality = st.slider("Kualitas Lingkungan (1-10)", min_value=1, max_value=10, value=5)

# Button untuk melakukan prediksi
if st.button("Prediksi Harga"):
    try:
        # Siapkan input data dari pengguna
        input_data = pd.DataFrame({
            'Square_Footage': [square_footage],
            'Num_Bedrooms': [num_bedrooms],
            'Num_Bathrooms': [num_bathrooms],
            'Year_Built': [year_built],
            'Lot_Size': [lot_size],
            'Garage_Size': [garage_size],
            'Neighborhood_Quality': [neighborhood_quality]
        })

        # Normalisasi input data
        scaled_input_data = scaler.transform(input_data)

        # Lakukan prediksi harga rumah
        predicted_price = model.predict(scaled_input_data)[0]

        # Tampilkan hasil prediksi
        st.success(f"Harga Rumah yang Diprediksi: Rp {predicted_price:,.2f}")

        # Opsional: Tampilkan detail input pengguna
        st.write("### Detail Input:")
        st.dataframe(input_data)
    
    except Exception as e:
        st.error(f"Terjadi kesalahan: {e}")
