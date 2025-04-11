# FFNN From Scratch - IF3270 Machine Learning

Repositori ini merupakan hasil dari Tugas Besar I mata kuliah IF3270 Pembelajaran Mesin. Proyek ini bertujuan untuk mengimplementasikan **Feedforward Neural Network (FFNN)** *from scratch*.

## 🚀 Deskripsi Singkat

FFNN yang kami implementasikan memiliki fitur-fitur sebagai berikut:
- Arsitektur fleksibel (jumlah layer dan neuron dapat disesuaikan)
- Pilihan fungsi aktivasi: Linear, ReLU, Sigmoid, Tanh, Softmax
- Pilihan loss function: Mean Squared Error (MSE), Binary Cross-Entropy, Categorical Cross-Entropy, Swish, dan ELU
- Inisialisasi bobot: Zero, Random Uniform, Random Normal (dengan seed untuk reproducibility), He, dan Xavier
- Forward dan backward propagation untuk batch input
- Visualisasi model (struktur, bobot, gradien, distribusi bobot/gradien per layer)
- Metode regularisasi: L1 dan L2
- Metode pelatihan menggunakan Gradient Descent
- Save dan load model
- RMS Normalization

---

## 🔠 Cara Setup dan Menjalankan Program

1. **Clone repository**
   ```bash
   git clone https://github.com/ninoaddict/FFNN-From-Scratch.git
   cd src
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Gunakan Model dalam notebook Anda**
   Usage
   ```python
   # Inisialisasi model FFNN
   ffnn_model = FFNN(
       layer_sizes=[784, 64, 64, 64, 10],
       activations=["relu"] * 3 + ["softmax"],
       seed=42,
       weight_init="he",
       use_rmsnorm=False
   )

   # Melatih model
   ffnn_training_loss, ffnn_val_loss = ffnn_model.train(
       X_train, y_train, X_val, y_val,
       loss_function="categorical_cross_entropy",
       learning_rate=0.05,
       epochs=20,
       batch_size=64,
       verbose=1
   )

   # Prediksi
   ffnn_proba = ffnn_model.predict(X_val)
   ffnn_preds = np.argmax(ffnn_proba, axis=1)
   ```

4. **Jalankan Notebook Anda**

   Jika menggunakan Jupyter Notebook:
   ```bash
   jupyter notebook
   ```
   Lalu buka file berikut:
   ```
   src/main.ipynb
   ```

---

## 👥 Pembagian Tugas Anggota Kelompok

| Nama | NIM | Tugas |
|------|-----|-------|
| Adril Putra Merin | 13522068 | Implementasi Program FFNN, Pengujian Program FFNN, Laporan |
| Marvin Scifo Hutahaean | 13522110 | Laporan |
| Berto Richardo Togatorop | 13522118 | Laporan |

---

