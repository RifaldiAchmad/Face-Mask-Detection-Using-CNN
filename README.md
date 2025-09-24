# Deteksi Masker Wajah dengan Convolutional Neural Network (CNN)

Proyek ini bertujuan untuk membangun model *deep learning* yang mampu mengklasifikasikan citra wajah untuk mendeteksi apakah seseorang mengenakan masker atau tidak. Model ini dilatih menggunakan arsitektur Convolutional Neural Network (CNN) dan dioptimalkan untuk inferensi cepat menggunakan TensorFlow Lite.

![Face Mask Detection](https://i.imgur.com/gC26a4H.png)

## 📖 Latar Belakang

Di tengah pentingnya protokol kesehatan publik, deteksi penggunaan masker secara otomatis menjadi sangat relevan. Proyek ini merupakan implementasi dari *computer vision* untuk menyelesaikan masalah praktis di dunia nyata, seperti pemantauan di area publik atau pintu masuk gedung untuk memastikan kepatuhan terhadap protokol kesehatan.

## 🎯 Tujuan Proyek

* Membangun model klasifikasi gambar biner (*binary classification*) untuk membedakan antara wajah **dengan masker** dan **tanpa masker**.
* Menggunakan *transfer learning* dengan arsitektur **MobileNet** sebagai *base model* untuk efisiensi dan akurasi yang tinggi.
* Menerapkan *image data augmentation* untuk meningkatkan variasi data latih dan robustisitas model.
* Mengonversi model Keras ke format **TensorFlow Lite (`.tflite`)** untuk memungkinkan implementasi pada perangkat *edge* atau aplikasi *mobile*.
* Menyajikan alur kerja *end-to-end*, mulai dari persiapan data, pelatihan model, evaluasi, hingga penyimpanan model untuk inferensi.

## 📊 Dataset

Dataset yang digunakan adalah **"Face Mask 44k Images Dataset"** yang tersedia di Kaggle.

* **Sumber**: [Kaggle - Face Mask 44k](https://www.kaggle.com/datasets/istiakhasan/facemask44k)
* **Total Gambar**: ~44,950 citra.
* **Kelas**:
    * `with_mask`: ~22,471 gambar.
    * `without_mask`: ~22,479 gambar.
* **Resolusi**: Semua gambar memiliki resolusi seragam **224x224 piksel**.

## ⚙️ Alur Kerja Proyek

1.  **Persiapan Data**:
    * Dataset diunduh dari Kaggle menggunakan API.
    * Data gambar dibagi menjadi set data latih (*training*) dan validasi (*validation*) dengan rasio 80:20.
2.  **Augmentasi Gambar**:
    * `ImageDataGenerator` dari Keras digunakan untuk menerapkan augmentasi secara *real-time* pada data latih.
    * Teknik augmentasi yang diterapkan meliputi: rotasi, pergeseran lebar & tinggi, *shear*, *zoom*, dan flip horizontal untuk memperkaya data.
3.  **Pembangunan Model (Transfer Learning)**:
    * **Base Model**: MobileNet, yang telah dilatih pada dataset ImageNet, digunakan sebagai dasar. Bobot *pre-trained*-nya dibekukan (*frozen*) untuk mempertahankan fitur-fitur yang telah dipelajari.
    * **Custom Layers**: Beberapa layer ditambahkan di atas *base model*, termasuk `GlobalAveragePooling2D`, `Dropout` untuk regularisasi, dan `Dense` layer dengan aktivasi **sigmoid** sebagai output akhir.
4.  **Pelatihan Model**:
    * Model dikompilasi dengan **Adam optimizer** dan *binary cross-entropy* sebagai *loss function*.
    * *Callbacks* `EarlyStopping` dan `ReduceLROnPlateau` digunakan untuk mencegah *overfitting* dan menyesuaikan *learning rate* secara dinamis.
    * Model dilatih selama 20 epoch dengan hasil akurasi dan validasi yang sangat baik.
5.  **Evaluasi**:
    * Grafik akurasi dan *loss* untuk data latih dan validasi diplot untuk memvisualisasikan performa model selama pelatihan.
    * Akurasi akhir yang dicapai pada set data validasi mencapai **~99%**.
6.  **Konversi ke TensorFlow Lite**:
    * Model Keras yang telah dilatih disimpan dalam format `.h5`.
    * Model kemudian dikonversi menjadi format `.tflite` untuk optimasi ukuran dan kecepatan inferensi.
7.  **Inferensi (Testing)**:
    * Model `.tflite` diuji pada gambar baru untuk membuktikan fungsionalitasnya dalam memprediksi kelas (`with_mask` atau `without_mask`).

## 🛠️ Teknologi yang Digunakan

* **Python**
* **TensorFlow & Keras**: Untuk membangun dan melatih model deep learning.
* **Scikit-learn**: Untuk evaluasi model.
* **NumPy**: Untuk operasi numerik.
* **Matplotlib & PIL**: Untuk visualisasi dan pemrosesan gambar.
* **Google Colab**: Sebagai lingkungan pengembangan dengan akses GPU gratis.

## 📈 Hasil

Model yang dilatih berhasil mencapai akurasi yang sangat tinggi, menunjukkan kemampuannya untuk mengklasifikasikan gambar dengan tepat.

* **Akurasi Pelatihan**: ~99.42%
* **Akurasi Validasi**: ~99.30%
* **Loss Pelatihan**: ~0.0163
* **Loss Validasi**: ~0.0199

Berikut adalah grafik performa model:

**Grafik Akurasi Model**
![Akurasi Model](https://i.imgur.com/G5g2mJg.png)

**Grafik Loss Model**
![Loss Model](https://i.imgur.com/0F9f2QZ.png)

## 🚀 Cara Menjalankan Proyek

1.  **Clone Repository**:
    ```bash
    git clone [https://github.com/username/nama-repository.git](https://github.com/username/nama-repository.git)
    cd nama-repository
    ```

2.  **Siapkan `kaggle.json`**:
    * Pastikan Anda memiliki file `kaggle.json` dari akun Kaggle Anda.
    * Letakkan file tersebut di direktori yang sama atau ikuti instruksi di notebook untuk mengunggahnya.

3.  **Buka Notebook**:
    * Buka file `notebook.ipynb` menggunakan Jupyter Notebook atau Google Colab.

4.  **Jalankan Semua Sel**:
    * Eksekusi sel-sel di dalam notebook secara berurutan untuk mengunduh dataset, melatih model, dan melihat hasil inferensi.

## 📄 Lisensi

Proyek ini berada di bawah Lisensi MIT. Lihat file `LICENSE` untuk detail lebih lanjut.
