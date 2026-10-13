Role: Senior MLOps & High-Performance Computing Engineer
Task: Membangun skrip benchmarking terstruktur untuk membandingkan performa 3 Arsitektur (1 ML, 2 DL) pada klasifikasi 3-Kelas EKG dengan protokol Anti-Kebocoran Data (Anti-Data Leakage) yang ketat dan eksperimen komparasi Resampling (No-SMOTE vs In-Fold Resampling).


# DATASET & DUAL-BRANCH INPUTS
1. Branch A (Tabular ML): `dataset_hrv_tabular.csv` (134.922 sampel, 6 fitur, label, record).
2. Branch B (Sequence DL): `dataset_hrv_sequence_23p.npz` (X: [134922, 30, 2], mask: [134922, 30], y: [134922], records: [134922]).
3. Kelas Target: 0 (Normal) ~72%, 1 (AFIB) ~24%, 2 (AFL) ~3.7%.


# STRICT ANTI-DATA LEAKAGE PROTOCOL
1. Patient-Wise Splitting: WAJIB gunakan `StratifiedGroupKFold(n_splits=5)` dengan `groups=records`. Pasien di Train Fold TIDAK BOLEH muncul di Validation Fold.
2. Isolated Resampling & Scaling:
   - Branch A: Fitur Scaler dan SMOTE HANYA boleh di-fit pada Train Fold di setiap iterasi CV menggunakan `imblearn.pipeline.Pipeline`. Validation Fold harus tetap MURNI.
   - Branch B: Z-Score Normalization (mean & std) HANYA dihitung dari Train Fold, lalu di-transform ke Validation Fold.


# EXPERIMENTAL MATRIX (SKENARIOS TO BENCHMARK)

#  A. Branch A (Tabular ML - LightGBM / HistGradientBoosting)
- Skenario A1 (No SMOTE + Cost-Sensitive): Model dilatih pada data murni dengan parameter `class_weight='balanced'`.
- Skenario A2 (In-Fold SMOTE): Terapkan SMOTE di dalam Pipeline CV hanya pada Train Fold, diuji pada Validation Fold murni.

#  B. Branch B (Sequence DL - 1D-ResNet & Micro-Transformer)
- Model DL 1: Lightweight 1D-ResNet (3 Residual Blocks, Conv1D, Batch Normalization, Swish, Global Avg Pool).
- Model DL 2: Micro-Transformer Encoder (Embedding -> Positional Encoding -> 1 Transformer Block (nhead=2, dim_feedforward=64) -> Linear). Menggunakan tensor `mask` untuk `src_key_padding_mask`.
- Skenario B1 (Weighted Loss): Latih dengan `nn.CrossEntropyLoss(weight=class_weights)` di mana $w_c = \frac{N_{total}}{K \times N_c}$.
- Skenario B2 (Class-Aware Sampler): Latih menggunakan `WeightedRandomSampler` pada PyTorch DataLoader untuk menyeimbangkan batch pelatihan secara alami.


# EVALUATION & PERFORMANCE METRICS
1. Predictive Metrics (Evaluasi 3-Kelas):
   - Macro F1-Score & Weighted F1-Score.
   - Precision & Recall spesifik per kelas (terutama kelas minoritas AFL / Label 2).
   - Normalized Confusion Matrix (Tampilkan Recall pada diagonal utama).
2. HPC & Computational Efficiency Metrics:
   - Training Time per Epoch/Fold (dalam detik).
   - Inference Latency: Waktu (milidetik) untuk melakukan prediksi pada 1.000 sampel saat simulasi deployment.

Tolong sediakan struktur kode Python berbasis OOP yang modular, rapi, dan menyertakan tabel ringkasan komparasi di akhir eksekusi.