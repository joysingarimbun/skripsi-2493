import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import re
from io import StringIO
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

st.set_page_config(page_title="Aplikasi Klasifikasi Tweet", layout="wide")

# ===============================#
# Fungsi Preprocessing           #
# ===============================#

# Inisialisasi Sastrawi
stopword_factory = StopWordRemoverFactory()
stopword_remover = stopword_factory.create_stop_word_remover()

stemmer_factory = StemmerFactory()
stemmer = stemmer_factory.create_stemmer()

# Load kamus normalisasi dari file CSV yang tersedia pada sistem
@st.cache_data
def load_slang_dictionary():
    """
    Load kamus normalisasi dari file CSV yang sudah tersedia pada sistem
    Format CSV yang diharapkan:
    - Kolom 1: kata slang/tidak baku
    - Kolom 2: kata baku/normalisasi
    """
    file_path = "kamus_slang.csv"
    
    try:
        # Baca file CSV kamus normalisasi
        df_kamus = pd.read_csv(file_path)
        
        # Pastikan file memiliki minimal 2 kolom
        if df_kamus.shape[1] >= 2:
            # Ambil kolom pertama sebagai slang dan kolom kedua sebagai normalisasi
            slang_col = df_kamus.columns[0]
            normal_col = df_kamus.columns[1]
            
            # Buat dictionary dari dataframe
            slang_dict = dict(zip(df_kamus[slang_col].astype(str).str.lower(), 
                                df_kamus[normal_col].astype(str).str.lower()))
            
            return slang_dict
        else:
            st.error(f"File {file_path} harus memiliki minimal 2 kolom")
            return {}
            
    except FileNotFoundError:
        st.error(f"File kamus normalisasi '{file_path}' tidak ditemukan pada sistem.")
        return {}
    except Exception as e:
        st.error(f"Error saat membaca file kamus: {str(e)}")
        return {}

# Inisialisasi kamus normalisasi
slang_dict = load_slang_dictionary()

def remove_elongation(text):
    """Menghapus elongation (huruf berulang lebih dari 2 kali)"""
    # Mengganti huruf yang berulang lebih dari 2 kali menjadi 2 huruf
    return re.sub(r'(.)\1{2,}', r'\1\1', text)

def normalize_slang(text, slang_dict):
    """Normalisasi kata slang menggunakan kamus"""
    words = text.split()
    normalized_words = []
    for word in words:
        if word.lower() in slang_dict:
            normalized_words.append(slang_dict[word.lower()])
        else:
            normalized_words.append(word)
    return ' '.join(normalized_words)

def preprocess_text(text):
    """
    Preprocessing teks dengan urutan:
    1. Casefolding
    2. Data Cleaning
    3. Penghapusan elongation
    4. Normalisasi Slang
    5. Tokenisasi
    6. Penghapusan Stopword menggunakan Sastrawi
    7. Stemming menggunakan Sastrawi
    """
    # 1. Casefolding
    text = text.lower()
    
    # 2. Data Cleaning
    # Hapus URL
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # Hapus mention (@username)
    text = re.sub(r'@\w+', '', text)
    # Hapus hashtag (#hashtag)
    text = re.sub(r'#\w+', '', text)
    # Hapus angka
    text = re.sub(r'\d+', '', text)
    # Hapus tanda baca dan karakter khusus, kecuali spasi
    text = re.sub(r'[^\w\s]', '', text)
    # Hapus extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # 3. Penghapusan elongation
    text = remove_elongation(text)
    
    # 4. Normalisasi Slang
    text = normalize_slang(text, slang_dict)
    
    # 5. Tokenisasi (implisit dalam split())
    words = text.split()
    
    # 6. Penghapusan Stopword menggunakan Sastrawi
    text = ' '.join(words)
    text = stopword_remover.remove(text)
    
    # 7. Stemming menggunakan Sastrawi
    text = stemmer.stem(text)
    
    return text

# ==========================================#
# Load model yang sudah di Latih dan Uji    #
# ==========================================#
@st.cache_resource
def load_model():
    return joblib.load("optimized_svm_ovr_model.pkl")

model = load_model()

# ===============================#
# Load informasi model           #
# ===============================#
@st.cache_data
def load_training_info():
    return {
        "train_size": 10168,
        "test_size": 2542,
        "preprocessing_steps": [
            "1. Casefolding",
            "2. Data Cleaning (URL, mention, hashtag, angka, tanda baca)",
            "3. Penghapusan Elongation",
            "4. Normalisasi Slang menggunakan kamus normalisasi",
            "5. Tokenisasi",
            "6. Penghapusan Stopword menggunakan Sastrawi",
            "7. Stemming menggunakan Sastrawi"
        ],
        "label_dist": {"positif": 3516 , "negatif": 6657, "netral": 2537},
        "classification_report": "Akurasi: 0.8474\nF1-Score weightened : 0.8473",
    }

# ===============================#
# Sidebar Navigasi               #
# ===============================#
menu = st.sidebar.radio("Navigasi", ["Halaman Utama", "Overview Model", "Mulai Analisis"])

# ===============================
# Halaman Utama (Petunjuk Pemakaian)
# ===============================
if menu == "Halaman Utama":
    st.title("ANALISIS SENTIMEN PUBLIK TERHADAP APLIKASI LAYANAN PERPAJAKAN CORETAX PADA PLATFORM X MENGGUNAKAN METODE SUPPORT VECTOR MACHINE")
    st.markdown("---")
    st.header("📋 Petunjuk Pemakaian Aplikasi")
    st.markdown("---")
    
    # Pengenalan Aplikasi
    st.header("🔍 Tentang Aplikasi")
    st.markdown("""
    Aplikasi ini adalah sistem analisis sentimen untuk tweet yang menggunakan model **Support Vector Machine (SVM)** 
    yang telah dioptimalkan. Aplikasi dapat mengklasifikasikan tweet ke dalam tiga kategori sentimen:
    - **Positif** 😊
    - **Negatif** 😞  
    - **Netral** 😐
    """)
    
    st.markdown("---")
    
    # Navigasi Menu
    st.header("🧭 Navigasi Menu")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📌 Halaman Utama")
        st.markdown("""
        - Berisi petunjuk pemakaian lengkap
        - Cara menggunakan setiap fitur
        - Format data yang diperlukan
        """)
        
        st.subheader("📊 Overview Model")
        st.markdown("""
        - Informasi dataset
        - Tahapan preprocessing
        - Distribusi label data
        - Evaluasi performa model
        - Confusion matrix
        """)
    
    with col2:
        st.subheader("🚀 Mulai Analisis")
        st.markdown("""
        - Upload dataset CSV untuk analisis
        - Proses klasifikasi sentimen
        - Visualisasi hasil
        - Download hasil klasifikasi
        """)
    
    st.markdown("---")
    
    # Cara Menggunakan Aplikasi
    st.header("📖 Cara Menggunakan Aplikasi")
    
    st.subheader("1️⃣ Persiapan Dataset")
    st.markdown("""
    **Format File yang Diperlukan:**
    - File berformat **CSV** (.csv)
    - Harus memiliki kolom bernama **'full_text'**
    - Kolom 'full_text' berisi teks tweet yang akan dianalisis
    
    **Contoh struktur dataset:**
    """)
    
    # Contoh dataset
    sample_data = pd.DataFrame({
        'full_text': [
            'Saya sangat senang dengan layanan ini!',
            'Produk ini mengecewakan sekali',
            'Cuaca hari ini cerah'
        ],
        'user_id': ['user1', 'user2', 'user3'],
        'created_at': ['2024-01-01', '2024-01-02', '2024-01-03']
    })
    st.dataframe(sample_data)
    
    st.subheader("2️⃣ Upload Dataset")
    st.markdown("""
    1. Klik menu **"Mulai Analisis"** di sidebar
    2. Klik tombol **"Upload dataset CSV"**
    3. Pilih file CSV dari komputer Anda
    4. Aplikasi akan menampilkan 5 baris pertama data
    5. Periksa jumlah baris dan kolom data
    """)
    
    st.subheader("3️⃣ Proses Analisis")
    st.markdown("""
    1. Setelah dataset ter-upload, klik tombol **"Mulai Analisis"**
    2. Aplikasi akan melakukan preprocessing otomatis:
       - Casefolding (mengubah ke huruf kecil)
       - Data cleaning (hapus URL, mention, hashtag, angka, tanda baca)
       - Penghapusan elongation (huruf berulang)
       - Normalisasi slang menggunakan kamus yang tersedia pada sistem
       - Tokenisasi
       - Penghapusan stopwords bahasa Indonesia (Sastrawi)
       - Stemming bahasa Indonesia (Sastrawi)
    3. Model SVM akan memprediksi sentimen setiap tweet
    4. Hasil akan ditampilkan dalam diagram pie
    """)
    
    st.subheader("4️⃣ Download Hasil")
    st.markdown("""
    1. Setelah klasifikasi selesai, klik **"Unduh Hasil Analisis"**
    2. File CSV baru akan terdownload dengan kolom tambahan:
       - **'cleaned_text'**: teks yang sudah melalui preprocessing
       - **'label'**: hasil prediksi sentimen ('positif', 'negatif', atau 'netral')
    3. Teks asli pada kolom 'full_text' tetap dipertahankan
    """)
    
    st.markdown("---")


# ===============================
# Halaman Overview Model
# ===============================
elif menu == "Overview Model":
    st.title("ANALISIS SENTIMEN PUBLIK TERHADAP APLIKASI LAYANAN PERPAJAKAN CORETAX PADA PLATFORM X MENGGUNAKAN METODE SUPPORT VECTOR MACHINE")
    st.markdown("---")
    st.header("📊 Overview Model")

    info = load_training_info()
    st.subheader("📊 Ringkasan Model")
    st.write(f"Jumlah Data Training : {info['train_size']} Data Tweets")
    st.write(f"Jumlah Data Testing : {info['test_size']} Data Tweets")
    
    st.subheader("⚙️ Tahapan Preprocessing:")
    for step in info['preprocessing_steps']:
        st.write("-", step)
    
    st.subheader("📈 Distribusi Label Pada Dataset :")
    st.bar_chart(info['label_dist'])
    
    st.subheader("📉 Evaluasi Model:")
    st.text(info['classification_report'])

    # Confusion Matrix (simulasi) - Ukuran diperkecil
    st.subheader("🔍 Confusion Matrix:")
    cm = [[1222, 90, 19], [81, 328, 99], [15, 84, 604]]
    fig, ax = plt.subplots(figsize=(6, 4))  # Diperkecil dari (4, 3) dan disesuaikan proporsi
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Negatif", "Netral", "Positif"],
                yticklabels=["Negatif", "Netral", "Positif"], ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    plt.tight_layout()  # Mengurangi whitespace
    st.pyplot(fig)

# ===============================
# Halaman Mulai Analisis
# ===============================
elif menu == "Mulai Analisis":
    st.title("ANALISIS SENTIMEN PUBLIK TERHADAP APLIKASI LAYANAN PERPAJAKAN CORETAX PADA PLATFORM X MENGGUNAKAN METODE SUPPORT VECTOR MACHINE")
    st.markdown("---")
    st.header("🚀 Analisis Dataset Baru")

    uploaded_file = st.file_uploader("Upload dataset CSV", type="csv")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.session_state['uploaded_df_raw'] = df
        st.write("5 Baris Pertama:")
        st.write(df.head())
        st.write(f"Jumlah Data: {df.shape[0]} baris | {df.shape[1]} kolom")

    if st.button("Mulai Analisis"):
        if 'uploaded_df_raw' in st.session_state:
            df = st.session_state['uploaded_df_raw'].copy()  # Buat copy untuk menjaga data asli

            if 'full_text' not in df.columns:
                st.error("Dataset harus memiliki kolom 'full_text'")
            else:
                # Preprocessing sebelum klasifikasi
                st.info("Melakukan preprocessing teks sebelum klasifikasi...")
                
                # Buat kolom baru untuk teks yang sudah dipreprocess
                df['cleaned_text'] = df['full_text'].astype(str).apply(preprocess_text)

                # Gunakan teks yang sudah dipreprocess untuk prediksi
                texts = df['cleaned_text']
                predicted = model.predict(texts)

                df['label'] = predicted
                st.session_state['classified_df'] = df

                st.success("Analisis selesai!")

                # Pie Chart - Ukuran diperkecil
                label_counts = df['label'].value_counts()
                fig, ax = plt.subplots(figsize=(6, 6))  # Diperkecil dari (4, 4)
                ax.pie(label_counts, labels=label_counts.index, autopct='%1.1f%%', startangle=90)
                ax.axis('equal')
                ax.set_title("Distribusi Label Hasil Klasifikasi")
                plt.tight_layout()  # Mengurangi whitespace
                st.subheader("📊 Distribusi Label Hasil Klasifikasi")
                st.pyplot(fig)
        else:
            st.warning("Silakan upload dataset terlebih dahulu.")

    if st.button("Unduh Hasil Analisis"):
        if 'classified_df' in st.session_state:
            df_result = st.session_state['classified_df']
            csv_buffer = StringIO()
            df_result.to_csv(csv_buffer, index=False)
            csv_data = csv_buffer.getvalue()

            st.download_button(
                label="Klik untuk Unduh CSV",
                data=csv_data,
                file_name='hasil_klasifikasi.csv',
                mime='text/csv'
            )
            st.success(f"Berhasil diunduh! Jumlah data: {df_result.shape[0]} baris.")
            st.info("File hasil berisi kolom asli + 'cleaned_text' (teks preprocessing) + 'label' (prediksi sentimen)")
        else:
            st.warning("Belum ada data yang diklasifikasikan.")