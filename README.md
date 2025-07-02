# Aplikasi Web Analisis Sentimen Layanan Perpajakan Coretax

Aplikasi ini merupakan sistem analisis sentimen terhadap layanan perpajakan **Coretax** dengan pendekatan pembelajaran mesin. Aplikasi ini dikembangkan untuk mengklasifikasikan opini masyarakat (positif, negatif, netral) secara otomatis dari data Twitter menggunakan metode berbasis leksikon dan machine learning. Antarmuka pengguna dibangun menggunakan framework **Streamlit** sehingga mudah diakses melalui web.

## Crawling Data

- Menggunakan *tweet-harvest* untuk melakukan crawling data tweet terkait Coretax.
- Memasukkan Twitter/X Auth Token agar tweet-harvest dapat mengakses X/twitter.
- Memasukkan nama file "crawling-coretax.csv"
- Memasukkan query berupa kata kunci topik coretax dan rentang waktu dari 1 januari 2025 sampai dengan 30 April 2025
- Hasil dari proses crawling dapat dilihat pada tautan berikut https://drive.google.com/file/d/1iKWzzYhZLNxq2Fu-qXlou-59YfeW-uE7/view?usp=sharing

## Pra-pemrosesan Data
- Pra-pemrosesan data meliputi penyaringan dan deduplikasi data, casefolding, pembersihan data, penghapusan elongation, normalisasi slang dengan menggunakan kamus alay, tokenisasi, penghapusan stopword, stemming dengan menggunakan sastrawi
- Hasil dari Pra-pemrosesan data dapat dilihat pada tautan berikut https://drive.google.com/file/d/1tQUEEApF_ahTA0lpF_MnObigIwLDPBhD/view?usp=sharing 

## Pelabelan Data
- Pelabelan data menggunakan pendekatan Lexicon based dengan acuan Kamus Lexicon Inset
- Proses dilakukan dengan cara mencocokkan setiap kata yang ada pada data tweet dengan kata yang ada pada daftar kata positif dan kata negatif. Setelah itu, dilakukan proses penghitungan untuk menentukan label tweets dengan cara, jika kemunculan kata positif lebih banyak daripada kemunculan kata negatif pada data tweet, maka tweet tersebut akan diberikan label positif dan jika kemunculan kata negatif lebih banyak daripada kemunculan kata positif pada data tweet, maka tweet tersebut akan diberikan label negatif. Jika kedua kondisi sebelumnya tidak terpenuhi maka akan diberikan label netral 
- Hasil dari proses pelabelan dapat dilihat pada tautan berikut https://drive.google.com/file/d/11mHwlkljw4FMRV0_ManlCHLDRiean45G/view?usp=sharing

## Pelatihan dan Pengujian Model
- Model menggunakan TF-IDF untuk ekstraksi fitur dan Menggunakan SVM untuk klasifikasi
- Model dilatih menggunakan data yang sudah di Pra-pemrosesan dan dilabeli
- Model menggunakan pendekatan One vs Rest
- Hyperparameter tuning dilakukan untuk mengoptimalkan model
- Akurasi Akhir model 84%
- Model akhir disimpan dengan format .pkl agar dapat digunaka pada aplikasi web 

## Pengambangan Aplikasi Web
- Pengembangan aplikasi web menggunakan Streanlit
- Fitur utama web dapat melakukan analisis sentimen terhadap dataset yang diunggah
- Demo aplikasi Web dapat dilihat pada tautan berikut

## Cara Menjalankan Program
Cara menjalankan Seluruh kode program dapat dilihat pada video demo berikut:  
