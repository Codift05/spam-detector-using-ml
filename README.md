# 🛡️ Spam Chat Detector - Machine Learning Project

Proyek Machine Learning untuk mendeteksi spam dalam pesan chat menggunakan Natural Language Processing (NLP) dan algoritma klasifikasi.

## 📋 Deskripsi

Aplikasi ini menggunakan teknik Machine Learning untuk mengklasifikasikan pesan chat sebagai spam atau bukan spam (ham). Model dilatih menggunakan dataset pesan dengan preprocessing teks yang komprehensif dan feature extraction menggunakan TF-IDF.

## 🎯 Fitur

- ✅ **Deteksi Spam Otomatis** - Klasifikasi pesan spam dengan akurasi tinggi
- ✅ **Text Preprocessing** - Lowercase, remove punctuation, stopword removal
- ✅ **TF-IDF Vectorization** - Feature extraction yang powerful
- ✅ **Multiple Models** - Perbandingan Logistic Regression & Naive Bayes
- ✅ **Web Interface** - Aplikasi web interaktif dengan Streamlit
- ✅ **Probability Score** - Menampilkan tingkat keyakinan prediksi
- ✅ **Visualisasi** - Confusion matrix dan perbandingan performa model

## 📁 Struktur Proyek

```
spam_chat_detector_using_ML/
│
├── data/
│   └── spam.csv                 # Dataset pesan spam dan ham
│
├── model/
│   ├── model.pkl                # Model terlatih (generated)
│   └── tfidf.pkl                # TF-IDF vectorizer (generated)
│
├── app/
│   └── app.py                   # Aplikasi Streamlit
│
├── notebook/
│   └── training.ipynb           # Jupyter notebook untuk training
│
├── README.md                    # Dokumentasi proyek
└── requirements.txt             # Python dependencies

```

## 🚀 Instalasi

### 1. Clone atau Download Proyek

```bash
cd spam_chat_detector_using_ML
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Download NLTK Data (Otomatis saat run training/app)

NLTK data akan didownload otomatis saat pertama kali menjalankan training atau aplikasi.

## 📊 Training Model

### Menjalankan Training Notebook

1. Buka Jupyter Notebook atau VS Code dengan Jupyter extension
2. Buka file `notebook/training.ipynb`
3. Jalankan semua cell secara berurutan

Training notebook akan:
- Load dataset dari `data/spam.csv`
- Melakukan preprocessing teks (lowercase, remove punctuation, stopword removal)
- Extract features menggunakan TF-IDF
- Melatih 2 model: **Logistic Regression** dan **Naive Bayes**
- Mengevaluasi dan membandingkan performa kedua model
- Menyimpan model terbaik ke `model/model.pkl` dan `model/tfidf.pkl`

### Output Training

Setelah training selesai, Anda akan mendapatkan:
- **Model comparison**: Perbandingan accuracy, precision, recall, f1-score
- **Confusion matrix**: Visualisasi untuk setiap model
- **Classification report**: Detail performa per kelas
- **Saved models**: File `.pkl` untuk deployment

## 🌐 Menjalankan Web App

### Start Streamlit App

```bash
streamlit run app/app.py
```

Atau dari root directory:

```bash
cd app
streamlit run app.py
```

### Menggunakan Aplikasi

1. **Buka browser** - Aplikasi akan otomatis terbuka di `http://localhost:8501`
2. **Masukkan teks pesan** di kotak input
3. **Klik tombol "🔍 Deteksi Spam"**
4. **Lihat hasil prediksi**:
   - Label: SPAM atau PESAN AMAN
   - Tingkat keyakinan (confidence score)
   - Detail probabilitas untuk setiap kelas
   - Teks setelah preprocessing

### Contoh Penggunaan

**Spam Message:**
```
URGENT! You've won a $5000 prize. Click here to claim now!
```

**Ham (Normal) Message:**
```
Hey, are we still meeting for lunch tomorrow?
```

## 🧠 Teknologi & Algoritma

### Machine Learning
- **Logistic Regression** - Model linear untuk klasifikasi binary
- **Naive Bayes** (MultinomialNB) - Probabilistic classifier
- **TF-IDF Vectorizer** - Feature extraction dari teks

### Text Preprocessing
- **Lowercase conversion** - Normalisasi teks
- **Punctuation removal** - Menghilangkan tanda baca
- **Stopword removal** - Menghapus kata-kata umum yang tidak informatif
- **Tokenization** - Memecah teks menjadi token

### Libraries
- **scikit-learn** - Machine learning framework
- **NLTK** - Natural Language Toolkit
- **pandas** - Data manipulation
- **matplotlib/seaborn** - Visualisasi
- **Streamlit** - Web framework

## 📈 Evaluasi Model

Model dievaluasi menggunakan metrik:

- **Accuracy** - Proporsi prediksi yang benar
- **Precision** - Proporsi prediksi spam yang benar
- **Recall** - Proporsi spam yang terdeteksi
- **F1-Score** - Harmonic mean dari precision dan recall
- **Confusion Matrix** - Visualisasi true/false positives/negatives

## 📊 Dataset

Dataset `spam.csv` berisi:
- **text**: Konten pesan
- **label**: Klasifikasi (spam/ham)
- **Total samples**: 100 pesan (50 spam, 50 ham)

Format:
```csv
text,label
"Congratulations! You've won $1000...",spam
"Hey, are we meeting tomorrow?",ham
```

## 🛠️ Customization

### Menambah Data Training

1. Tambahkan data ke `data/spam.csv`
2. Format: `text,label`
3. Jalankan ulang training notebook

### Mengubah Model Parameters

Edit di `training.ipynb`:
```python
# Logistic Regression
lr_model = LogisticRegression(max_iter=1000, C=1.0, random_state=42)

# TF-IDF
tfidf = TfidfVectorizer(max_features=3000, ngram_range=(1, 2))
```

### Custom Styling Streamlit

Edit CSS di `app/app.py` dalam blok `st.markdown()`.

## 🐛 Troubleshooting

### Error: Model files not found
**Solusi**: Jalankan training notebook terlebih dahulu untuk generate `model.pkl` dan `tfidf.pkl`

### Error: NLTK data not found
**Solusi**: Data akan didownload otomatis. Jika gagal, manual download:
```python
import nltk
nltk.download('stopwords')
nltk.download('punkt')
```

### Streamlit not starting
**Solusi**: 
```bash
pip install --upgrade streamlit
streamlit run app/app.py --server.port 8501
```

## 📝 To-Do / Future Improvements

- [ ] Tambah lebih banyak data training
- [ ] Implementasi deep learning (LSTM/BERT)
- [ ] Support multi-language detection
- [ ] API endpoint untuk integrasi
- [ ] Batch prediction untuk multiple messages
- [ ] Model versioning dan A/B testing
- [ ] Deploy ke cloud (Streamlit Cloud/Heroku)

## 📄 Requirements

```
numpy>=1.24.3
pandas>=2.0.3
scikit-learn>=1.3.0
nltk>=3.8.1
matplotlib>=3.7.2
seaborn>=0.12.2
streamlit>=1.26.0
```

## 👨‍💻 Author

Proyek ini dibuat sebagai contoh implementasi Machine Learning untuk deteksi spam menggunakan Python.

## 📜 License

MIT License - Silakan gunakan untuk pembelajaran dan pengembangan.

## 🤝 Contributing

Contributions, issues, dan feature requests sangat diterima!

---

**Happy Coding! 🚀**

*Dibuat dengan ❤️ menggunakan Python, scikit-learn, dan Streamlit*
