import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from keras.models import load_model
from sklearn.feature_extraction.text import TfidfVectorizer

# Membaca data dari file Excel
data = pd.read_excel('dataset/data hp.xlsx')

# Memuat model
model = load_model('model.h5')

# Streamlit UI
st.title("Rekomendasi Smartphone dengan Model Deep Learning")
st.write("Masukkan preferensi Anda untuk mendapatkan rekomendasi smartphone:")

# Input pengguna
input_name = st.text_input("Masukkan nama hp yang diinginkan").lower()
input_rom = st.slider("Masukkan kapasitas ROM yang diinginkan (dalam GB)", 8, 512, 64)
input_ram = st.slider("Masukkan kapasitas RAM yang diinginkan (dalam GB)", 2, 16, 4)

# Cek apakah semua input sudah diisi
if input_name and input_rom and input_ram:
    # TF-IDF Vectorization untuk nama smartphone
    tfidf = TfidfVectorizer(stop_words='english')
    name_vectors = tfidf.fit_transform(data['Name'].str.lower())

    # Query pengguna
    query_vector = tfidf.transform([input_name]).toarray()
    query_numerical = [[input_rom, input_ram, 0]]  # Ratings diatur ke 0
    query_combined = pd.concat([
        pd.DataFrame(query_vector),
        pd.DataFrame(query_numerical)
    ], axis=1).values

    # Gunakan model untuk prediksi
    predicted_similarity = model.predict(query_combined)

    # Tambahkan hasil prediksi ke dataset
    data['Predicted Similarity'] = predicted_similarity.flatten()

    # Filter hasil
    recommended = data.sort_values(by='Predicted Similarity', ascending=False)

    # Tampilkan hasil dalam tabel
    st.write("### Hasil Rekomendasi:")
    st.dataframe(recommended[['Name', 'ROM(GB)', 'RAM(GB)', 'Ratings', 'Price', 'Predicted Similarity']])
else:
    st.write("Silakan masukkan semua preferensi untuk mendapatkan rekomendasi.")
