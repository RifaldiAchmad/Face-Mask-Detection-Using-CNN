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

1.  **Persiapan Data**: Dataset diunduh dari Kaggle dan dibagi menjadi set data latih (*training*) dan validasi (*validation*) dengan rasio 80:20.
2.  **Augmentasi Gambar**: `ImageDataGenerator` Keras digunakan untuk menerapkan rotasi, pergeseran, *zoom*, dan flip pada gambar latih secara *real-time* untuk memperkaya data.
3.  **Pembangunan Model (Transfer Learning)**: Arsitektur MobileNet (telah dilatih pada ImageNet) digunakan sebagai *base model*. Beberapa *layer* kustom (termasuk `GlobalAveragePooling2D`, `Dropout`, dan `Dense`) ditambahkan di atasnya untuk klasifikasi.
4.  **Pelatihan Model**: Model dikompilasi dengan *optimizer* Adam dan dilatih selama 20 epoch. *Callbacks* `EarlyStopping` dan `ReduceLROnPlateau` digunakan untuk mencegah *overfitting*.
5.  **Evaluasi**: Performa model dievaluasi menggunakan metrik akurasi dan *loss*, yang divisualisasikan melalui grafik.
6.  **Konversi ke TensorFlow Lite**: Model Keras yang telah dilatih disimpan dan dikonversi ke format `.tflite` untuk optimasi ukuran dan kecepatan.
7.  **Inferensi (Testing)**: Model `.tflite` diuji pada gambar baru untuk memvalidasi kemampuannya dalam memprediksi kelas.

## 🛠️ Teknologi yang Digunakan

* **Python**
* **TensorFlow & Keras**: Untuk membangun dan melatih model deep learning.
* **Scikit-learn**: Untuk evaluasi model.
* **NumPy**: Untuk operasi numerik.
* **Matplotlib & PIL**: Untuk visualisasi dan pemrosesan gambar.
* **Google Colab**: Sebagai lingkungan pengembangan dengan akses GPU gratis.

## 📈 Hasil

Model berhasil mencapai akurasi yang sangat tinggi, menunjukkan kemampuannya untuk mengklasifikasikan gambar dengan tepat.

* **Akurasi Pelatihan**: ~99.42%
* **Akurasi Validasi**: ~99.30%
* **Loss Pelatihan**: ~0.0163
* **Loss Validasi**: ~0.0199

**Grafik Performa Model**
![Akurasi & Loss](https://i.imgur.com/KxO0xJ0.png)

## 🚀 Cara Menjalankan Proyek

1.  **Clone Repository**:
    ```bash
    git clone [https://github.com/username/nama-repository.git](https://github.com/username/nama-repository.git)
    cd nama-repository
    ```
2.  **Siapkan `kaggle.json`**: Pastikan Anda memiliki file `kaggle.json` dari akun Kaggle Anda untuk mengunduh dataset.
3.  **Buka Notebook**: Buka file `notebook.ipynb` menggunakan Jupyter Notebook atau Google Colab.
4.  **Jalankan Semua Sel**: Eksekusi sel-sel di dalam notebook secara berurutan untuk menjalankan seluruh alur kerja.

---

<details>
<summary>🇬🇧 CLICK HERE FOR ENGLISH VERSION</summary>

# Face Mask Detection with Convolutional Neural Network (CNN)

This project aims to build a deep learning model capable of classifying facial images to detect whether a person is wearing a mask. The model is trained using a Convolutional Neural Network (CNN) architecture and optimized for fast inference using TensorFlow Lite.

## 📖 Background

Amidst the importance of public health protocols, automatic detection of face mask usage has become highly relevant. This project is a computer vision implementation to solve a practical, real-world problem, such as monitoring in public areas or building entrances to ensure compliance with health protocols.

## 🎯 Project Objectives

* To build a binary image classification model to distinguish between faces **with mask** and **without mask**.
* To use *transfer learning* with the **MobileNet** architecture as a base model for high efficiency and accuracy.
* To implement *image data augmentation* to increase the variety of training data and model robustness.
* To convert the Keras model to the **TensorFlow Lite (`.tflite`)** format to enable implementation on edge devices or mobile applications.
* To present an end-to-end workflow, from data preparation, model training, evaluation, to model saving for inference.

## 📊 Dataset

The dataset used is the **"Face Mask 44k Images Dataset"** available on Kaggle.

* **Source**: [Kaggle - Face Mask 44k](https://www.kaggle.com/datasets/istiakhasan/facemask44k)
* **Total Images**: ~44,950 images.
* **Classes**:
    * `with_mask`: ~22,471 images.
    * `without_mask`: ~22,479 images.
* **Resolution**: All images have a uniform resolution of **224x224 pixels**.

## ⚙️ Project Workflow

1.  **Data Preparation**: The dataset is downloaded from Kaggle and split into training and validation sets with an 80:20 ratio.
2.  **Image Augmentation**: The Keras `ImageDataGenerator` is used to apply real-time rotation, shifting, zooming, and flipping to the training images to enrich the data.
3.  **Model Building (Transfer Learning)**: The MobileNet architecture (pre-trained on ImageNet) is used as the base model. Several custom layers (including `GlobalAveragePooling2D`, `Dropout`, and `Dense`) are added on top for classification.
4.  **Model Training**: The model is compiled with the Adam optimizer and trained for 20 epochs. `EarlyStopping` and `ReduceLROnPlateau` callbacks are used to prevent overfitting.
5.  **Evaluation**: The model's performance is evaluated using accuracy and loss metrics, which are visualized through plots.
6.  **Conversion to TensorFlow Lite**: The trained Keras model is saved and then converted to the `.tflite` format for size and speed optimization.
7.  **Inference (Testing)**: The `.tflite` model is tested on new images to validate its prediction capabilities.

## 🛠️ Tech Stack

* **Python**
* **TensorFlow & Keras**: For building and training the deep learning model.
* **Scikit-learn**: For model evaluation.
* **NumPy**: For numerical operations.
* **Matplotlib & PIL**: For visualization and image processing.
* **Google Colab**: As the development environment with free GPU access.

## 📈 Results

The model achieved very high accuracy, demonstrating its ability to classify images correctly.

* **Training Accuracy**: ~99.42%
* **Validation Accuracy**: ~99.30%
* **Training Loss**: ~0.0163
* **Validation Loss**: ~0.0199

**Model Performance Plots**
![Accuracy & Loss](https://i.imgur.com/KxO0xJ0.png)

## 🚀 How to Run

1.  **Clone the Repository**:
    ```bash
    git clone [https://github.com/username/repository-name.git](https://github.com/username/repository-name.git)
    cd repository-name
    ```
2.  **Prepare `kaggle.json`**: Ensure you have your `kaggle.json` file from your Kaggle account to download the dataset.
3.  **Open the Notebook**: Open the `notebook.ipynb` file using Jupyter Notebook or Google Colab.
4.  **Run All Cells**: Execute the cells in the notebook sequentially to run the entire workflow.

</details>
