# Seleksi GAIB 

## Deskripsi Singkat
Repositori ini berisi implementasi _from scratch_ dari beberapa model yang ada pada library scikit-learn dan tensorflow. Model yang diimplementasikan diuji keakuratannya dengan menggunakan metric accuracy. Alasannya adalah karena accuracy menghitung persentase antara prediksi yang benar dengan total keseluruhan prediksi yang dibuat oleh model sehingga hal ini lebih mudah dipahami dan diinterpretasikan.

Accuracy = $\frac{\text{True Positives} + \text{True Negatives}}{\text{Total Samples}}$


## Cara penggunaan
1. Pastikan python telah terinstall
2. Pindah ke folder `src` dan install semua library yang diperlukan dengan menjalankan command berikut:
   
   ```
   pip install -r requirements.txt
   ```
3. Pastikan semua features yang ada pada dataset sudah dalam format numerikal (ex: int, float) agar mencegah terjadinya error (perhatikan contoh penggunaan pada notebook).

Berikut adalah model yang diimplementasikan pada repositori beriikut: 

**Supervised Learning (Bagian 2):**
- [v] KNN
- [v] Logistic Regression
- [v] Gaussian Naive Bayes
- [v] CART
- [v] SVM
- [v] ANN

**Bonus yang diimplementasikan:**
- Penambahan fungsi aktivasi lain pada ANN, seperti leaky ReLU dan exponential ReLU

**Unsupervised Learning (Bagian 3):**
- [v] K-MEANS
- [v] DBSCAN
- [v] PCA
  
**Bonus yang diimplementasikan:**
- Penambahan metode inisialisasi K-Means++ pada model K-MEANS

**Reinforcement Learning (Bagian 4):**
- [v] Q-LEARNING
- [v] SARSA


| Author  | NIM |
| ------------- | ------------- |
| Muhammad Althariq Fairuz  | 13522027  |
