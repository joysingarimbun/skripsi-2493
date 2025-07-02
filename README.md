# Aplikasi Web Analisis Sentimen Layanan Perpajakan Coretax

Aplikasi ini merupakan sistem analisis sentimen terhadap layanan perpajakan **Coretax** dengan pendekatan pembelajaran mesin. Aplikasi ini dikembangkan untuk mengklasifikasikan opini masyarakat (positif, negatif, netral) secara otomatis dari data Twitter menggunakan metode berbasis leksikon dan machine learning. Antarmuka pengguna dibangun menggunakan framework **Streamlit** sehingga mudah diakses melalui web.

## ✨ Fitur Utama

- **Pengambilan Data**: Menggunakan *tweet-harvest* untuk melakukan crawling data tweet terkait Coretax.
- **Pelabelan Otomatis**: Sentimen data dilabeli secara otomatis menggunakan pendekatan *lexicon-based* dengan *INSET* (Indonesia Sentiment Lexicon).
- **Pra-pemrosesan Teks**:
  - Normalisasi kata alay menggunakan kamus alay
  - *Stemming* menggunakan library **Sastrawi**
- **Pelatihan Model**:
  - Ekstraksi fitur teks menggunakan **TF-IDF**
  - Klasifikasi sentimen menggunakan **Support Vector Machine (SVM)**
- **Visualisasi Wordcloud**: Menampilkan kata-kata yang sering muncul berdasarkan kategori sentimen
- **Aplikasi Web**: Dibangun dengan **Streamlit** untuk memudahkan penggunaan analisis secara interaktif
