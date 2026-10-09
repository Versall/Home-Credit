# Home Credit Default Risk - GCI Global 2026

Repositori ini berisi starter kit dan panduan untuk mengikuti kompetisi **Home Credit Default Risk** yang diselenggarakan dalam **GCI Global 2026**. Kompetisi ini merupakan adaptasi dari kompetisi [Kaggle Home Credit Default Risk (2018)](https://www.kaggle.com/c/home-credit-default-risk) menggunakan subset data.

---

## 📖 Daftar Isi

- [Home Credit Default Risk - GCI Global 2026](#home-credit-default-risk---gci-global-2026)
  - [📖 Daftar Isi](#-daftar-isi)
  - [🎯 Tentang Kompetisi](#-tentang-kompetisi)
  - [📁 Struktur Repositori](#-struktur-repositori)
  - [📊 Dataset](#-dataset)
  - [📏 Metrik Evaluasi](#-metrik-evaluasi)
  - [📤 Format Submission](#-format-submission)
  - [📜 Aturan Kompetisi](#-aturan-kompetisi)
  - [🚀 Cara Memulai](#-cara-memulai)
  - [💡 Tips Meningkatkan Model](#-tips-meningkatkan-model)
  - [📅 Timeline](#-timeline)
  - [🏆 Leaderboard](#-leaderboard)
  - [📚 Referensi](#-referensi)
  - [📄 Lisensi](#-lisensi)

---

## 🎯 Tentang Kompetisi

**Home Credit** berusaha memperluas akses layanan keuangan kepada masyarakat yang belum memiliki riwayat kredit (unbanked population). Tujuan dari kompetisi ini adalah memprediksi **probabilitas seorang pelanggan akan gagal bayar (default)** pada pinjamannya berdasarkan berbagai data pelanggan seperti:

- Data aplikasi pinjaman
- Riwayat kredit sebelumnya
- Data transaksi sebelumnya
- Data kartu kredit
- Dan lain-lain

Dengan model prediksi yang baik, Home Credit dapat meminimalkan risiko kerugian sekaligus tetap melayani pelanggan yang layak mendapatkan pinjaman.

---

## 📁 Struktur Repositori

```
Home-Credit/
├── README.md                          # Dokumentasi ini
├── README.ipynb                       # Notebook panduan kompetisi
├── tutorial.ipynb                     # Notebook tutorial untuk memulai
├── HomeCredit_columns_description.xlsx # Deskripsi kolom dataset
└── input/
    ├── train.csv                      # Data training (dengan label TARGET)
    ├── test.csv                       # Data testing (tanpa label)
    └── sample_submission.csv          # Contoh format submission
```

---

## 📊 Dataset

Dataset yang digunakan adalah **subset** dari dataset asli Home Credit Default Risk.

| File | Deskripsi |
|------|-----------|
| `train.csv` | Data pelatihan yang sudah memiliki label `TARGET` (0 = tidak default, 1 = default) |
| `test.csv` | Data pengujian tanpa label, digunakan untuk prediksi akhir |
| `sample_submission.csv` | Contoh format file submission yang benar |
| `HomeCredit_columns_description.xlsx` | Penjelasan detail setiap kolom dalam dataset |

> ⚠️ **Hanya boleh menggunakan dataset yang disediakan di folder `input/`.** Penggunaan data eksternal dilarang.

---

## 📏 Metrik Evaluasi

Kompetisi ini menggunakan metrik **AUC (Area Under the ROC Curve)** untuk mengukur performa model.

- **AUC** mengukur kemampuan model dalam membedakan antara kelas positif (default) dan negatif (tidak default).
- Nilai AUC berkisar antara **0.5** (random) hingga **1.0** (sempurna).
- Prediksi berupa **probabilitas** (bukan label kelas).

---

## 📤 Format Submission

File submission harus berupa CSV dengan format berikut:

```
SK_ID_CURR,TARGET
171202,0.5
171203,0.12
171204,0.87
...
```

**Ketentuan:**
- Berisi **seluruh** `SK_ID_CURR` dari `test.csv`
- Kolom `TARGET` berisi **probabilitas** (nilai antara 0 dan 1)
- Nama file: `<username>.csv`
- Dikirim melalui platform **Omnicampus**

---

## 📜 Aturan Kompetisi

1. ✅ **Data Eksternal Dilarang** — Hanya dataset yang disediakan yang boleh digunakan.
2. ✅ **Dilarang Hand-Labeling** — Semua prediksi harus dihasilkan oleh model, bukan keputusan manual.
3. ✅ **Reproducibility** — Kode harus dapat direproduksi, gunakan seed pada random number generator.
4. ✅ **Private Sharing Diperbolehkan** — Kompetisi ini bersifat tutorial untuk pemula, berbagi kode diperbolehkan.
5. ✅ **Cantumkan Sumber** — Jika mereferensikan notebook atau kode orang lain, wajib mencantumkan sumbernya.

> Peserta yang ingin dipertimbangkan untuk **Honors** atau **Outstanding Student** wajib mengumpulkan kode yang dapat mereproduksi CSV yang dikirim.

---

## 🚀 Cara Memulai

1. **Clone repositori ini**

   ```bash
   git clone https://github.com/Versall/Home-Credit.git
   cd Home-Credit
   ```

2. **Install dependencies**

   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn lightgbm xgboost
   ```

3. **Buka notebook tutorial**

   ```bash
   jupyter notebook tutorial.ipynb
   ```

4. **Pahami data**

   - Baca `HomeCredit_columns_description.xlsx` untuk memahami setiap kolom.
   - Lakukan **Exploratory Data Analysis (EDA)** pada `train.csv`.

5. **Bangun model**

   - Lakukan preprocessing (handling missing values, encoding, scaling).
   - Latih model (contoh: Logistic Regression, Random Forest, LightGBM, XGBoost).
   - Evaluasi dengan cross-validation menggunakan metrik AUC.

6. **Buat submission**

   - Prediksi `test.csv` dan simpan hasilnya dalam format CSV.
   - Upload ke Omnicampus.

---

## 💡 Tips Meningkatkan Model

1. **Pahami data lebih dalam** — Lakukan visualisasi dan analisis korelasi antar fitur.
2. **Feature engineering** — Tutorial hanya menggunakan 5 fitur. Coba tambahkan lebih banyak fitur dan buat fitur baru dari kombinasi fitur yang ada.
3. **Handling missing values** — Gunakan strategi imputasi yang tepat (mean, median, model-based, dll).
4. **Feature selection** — Pilih fitur yang paling informatif untuk mengurangi overfitting.
5. **Ensemble methods** — Gabungkan beberapa model (stacking, blending, voting) untuk hasil yang lebih baik.
6. **Hyperparameter tuning** — Gunakan Grid Search, Random Search, atau Optuna.
7. **Cross-validation** — Gunakan Stratified K-Fold untuk evaluasi yang lebih stabil.
8. **Class imbalance** — Atasi ketidakseimbangan kelas dengan SMOTE, class weights, atau teknik lainnya.

---

## 📅 Timeline

| Tanggal | Agenda |
|---------|--------|
| 2 Oktober 2026 | Kompetisi dimulai |
| 20 November 2026, 01:00 UTC | Batas akhir submission |
| 17 Desember 2026 | Pengumuman peringkat akhir |

---

## 🏆 Leaderboard

- **Public Leaderboard** — Skor dihitung pada subset data uji selama kompetisi berlangsung.
- **Private Leaderboard** — Skor akhir dihitung pada seluruh data uji setelah kompetisi selesai.

---

## 📚 Referensi

- [Kaggle Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk)
- [scikit-learn Documentation](https://scikit-learn.org/)
- [LightGBM Documentation](https://lightgbm.readthedocs.io/)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)

---

## 📄 Lisensi

Repositori ini dibuat untuk keperluan edukasi dalam kompetisi **GCI Global 2026**. Silakan gunakan, modifikasi, dan bagikan dengan tetap mencantumkan sumber.

---

**Selamat berkompetisi dan semoga sukses! 🎉**