import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Membaca data dari file Excel
data = pd.read_excel('dataset/data hp.xlsx')

# Pastikan data dimuat dengan benar
st.write("### Data yang dimuat:")
st.dataframe(data) 

# Streamlit UI
st.title("Rekomendasi Smartphone")
st.write("Masukkan preferensi Anda untuk mendapatkan rekomendasi smartphone:")

# Input pengguna
input_name = st.text_input("Masukkan nama hp yang diinginkan").lower()
input_rom = st.slider("Masukkan kapasitas ROM yang diinginkan (dalam GB)", 8, 512, 64)  # Slider untuk ROM
input_ram = st.slider("Masukkan kapasitas RAM yang diinginkan (dalam GB)", 2, 16, 4)  # Slider untuk RAM

# Cek apakah semua input sudah diisi
if input_name and input_rom and input_ram:
    # Content-Based Filtering
    # TF-IDF Vectorization untuk nama smartphone
    tfidf = TfidfVectorizer(stop_words='english')  # Mengabaikan kata-kata umum (stopwords)
    name_vectors = tfidf.fit_transform(data['Name'].str.lower())

    # Gabungkan fitur TF-IDF dengan ROM, RAM, dan Ratings
    numerical_features = data[['ROM(GB)', 'RAM(GB)', 'Ratings']].values
    combined_features = pd.concat([
        pd.DataFrame(name_vectors.toarray()),
        pd.DataFrame(numerical_features)
    ], axis=1).values

    # Query pengguna
    query_vector = tfidf.transform([input_name]).toarray()
    query_numerical = [[input_rom, input_ram, 0]]  # Ratings diatur ke 0 untuk pencarian
    query_combined = pd.concat([
        pd.DataFrame(query_vector),
        pd.DataFrame(query_numerical)
    ], axis=1).values

    # Hitung kesamaan menggunakan cosine similarity
    similarities = cosine_similarity(query_combined, combined_features)

    # Tambahkan skor kesamaan ke dataset
    data['Similarity'] = similarities.flatten()

    # Filter hasil berdasarkan nama, ROM, dan RAM
    data_filtered = data[
        (data['Name'].str.lower().str.contains(input_name)) &  # Filter nama berbasis input
        (data['ROM(GB)'] == input_rom) &
        (data['RAM(GB)'] == input_ram)
    ]

    # Periksa apakah ada data yang cocok
    if not data_filtered.empty:
        # Urutkan hasil berdasarkan kesamaan tertinggi
        recommended = data_filtered.sort_values(by='Similarity', ascending=False)

        # Tampilkan hasil dalam format deskriptif
        st.write("### Hasil Rekomendasi:")
        for idx, row in recommended.iterrows():
            with st.expander(f"Smartphone: {row['Name']}"):
                st.write(f"- **ROM**: {row['ROM(GB)']} GB")
                st.write(f"- **RAM**: {row['RAM(GB)']} GB")
                st.write(f"- **Ratings**: {row['Ratings']:.1f}")
                st.write(f"- **Price**: ${row['Price']:,}")
                st.write(f"- **Similarity Score**: {row['Similarity']:.4f}")
    else:
        st.write("Tidak ada smartphone yang memenuhi kriteria pencarian Anda.")
else:
    st.write("Silakan masukkan semua preferensi untuk mendapatkan rekomendasi.")
    
