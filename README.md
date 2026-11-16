# SIHEDAF — klasifikasi ritme jantung dari interval RR

Klasifikasi window EKG 10 detik menjadi **Normal / AFIB / AFL** dari urutan interval RR (IBI),
untuk perangkat EKG single-lead 50 Hz. Data latih: MIT-BIH Atrial Fibrillation Database (AFDB,
23 pasien). Repo ini menyimpan setiap iterasi eksperimen sebagai folder versi tersendiri
beserta laporannya.

> Untuk AI agent / anggota tim yang ingin menghasilkan output serupa: baca **[AGENTS.md](AGENTS.md)**.

## Versi dan hasil

| Versi | Isi | Hasil utama (data uji detektor 50 Hz)* | Laporan |
|---|---|---|---|
| [v1_lama](v1_lama/) | Pipeline awal: `ekstraksi_hrv.py`, `model.ipynb`, `final.joblib`, backend `find_peaks` | Diklaim Macro-F1 0,826. Diukur ulang: **0,744**; dengan front-end aslinya **0,418** | dibahas di laporan v3 (hal. 2–3) |
| [v2_revisi](v2_revisi/) | Modul bersama `hrv_core.py`, detektor R-peak 50 Hz + cek kualitas, 3 arsitektur DL, bias kelas, `predictor.py` + `model_export/` | MicroTransformer, bias `[0.0, 2.0, -3.0]`: **0,779** [0,63–0,82] (notebook melaporkan 0,792, satu partisi) | [PDF](v2_revisi/laporan/laporan_hasil_sihedaf_v2.pdf) |
| [v3_evaluasi_robust](v3_evaluasi_robust/) | Protokol evaluasi: 5×5 fold pasien, CI bootstrap pasien, perbandingan berpasangan; v1 dilatih ulang untuk dibandingkan | Kandidat model sama kuat; batasnya data AFL; bias v2 terpotong grid | [PDF](v3_evaluasi_robust/laporan/laporan_evaluasi_robust.pdf) |
| [v3.1_kalibrasi_produksi](v3.1_kalibrasi_produksi/) | Menerapkan keputusan v3 tanpa training: bias baru, **paket produksi aktif**, panduan backend | Bias `[0.0, -0.5, -4.5]`: **0,832** [0,67–0,92], AFL F1 0,647; lebih baik dari v2 produksi (P = 1,00; bersarang 0,993, lihat v3.2) | [PDF](v3.1_kalibrasi_produksi/laporan/laporan_kalibrasi_produksi.pdf) |
| [v3.2_uji_takikardia](v3.2_uji_takikardia/) | Evaluasi saja (menindaklanjuti review eksternal): uji stres ritme teratur 60–170 bpm, bootstrap bersarang, tanpa 04936, simulasi aturan produk | **Semua versi menebak ritme teratur ≥ ±115 bpm sebagai AFIB/AFL** (95% Normal ≥120 bpm jadi alarm). v3.1 vs v2 tetap lebih baik (P = 0,993 bersarang); vs v1 tidak terbedakan tanpa 04936 (P = 0,94) | [PDF](v3.2_uji_takikardia/laporan/laporan_uji_takikardia.pdf) |
| [v3.3_takikardia_asli](v3.3_takikardia_asli/) | Evaluasi saja (riset 6 November 2026): alarm palsu pada **EKG latihan fisik asli** (EPFL, Wrist, EPHNOGRAM; 41 subjek), detektor perangkat vs R-peak manual, aturan produk, abstain conformal, metrik episode gaya EC57 | **Paket v3.1 mengalarmkan 98,4% sinus ≥ 120 bpm** [97–100]. Hanya model yang pernah melihat sinus takikardia membaik (v8c 60%, v7a 64%). Detektor 50 Hz saat aktivitas memicu alarm AF juga di < 100 bpm (21%). Abstain conformal tidak menolong (model yakin) | [PDF](v3.3_takikardia_asli/laporan/laporan_v3_3_takikardia_asli.pdf) |
| [v3.4_pembersih_detak](v3.4_pembersih_detak/) | Versi minor, pra-registrasi: front-end di atas detektor R-peak produksi, ide dari pembaca EKG Tugas Akhir pemilik repo. F1 menyelaraskan setiap R-peak ke templat QRS median (memperbaiki lompatan R/S); F2 menambah gerbang morfologi. Diuji di AFDB (latih ulang resep v3.1), latihan fisik v3.3, dan CinC 2017, MIT-BIH, LTAFDB, CPSC 2021 | Jitter EPFL 10,7 → 2,8 ms, alarm < 100 bpm saat aktivitas 20,4% → 17,7% (F1) / 12,0% (F2), AFDB Macro-F1 0,832 → 0,842 (bersarang tidak terbedakan), spesifisitas naik di 4 data eksternal. **Tidak direkomendasikan:** Se F1 turun tipis di MIT-BIH (−0,7 poin, P = 0,038), Se F2 turun 1–3 poin di keempatnya. Paket tetap v3.1 | [PDF](v3.4_pembersih_detak/laporan/laporan_v3_4_pembersih_detak.pdf) |
| [v4_smoothing_ensemble](v4_smoothing_ensemble/) | Smoothing antar-window & ensemble di atas OOF v3, tanpa training | Tidak ada yang lebih baik dari v3.1. Smoothing: alarm palsu 0,51 → 0,09/jam tapi episode AFL terdeteksi 74% → 38% | [PDF](v4_smoothing_ensemble/laporan/laporan_smoothing_ensemble.pdf) |
| [v5_training_afl](v5_training_afl/) | Dua resep training MicroTransformer: window AFL rapat, bobot kelas lunak | Keduanya setara dengan v3.1. Bobot lunak: bias AFL −2,25 (bukan −4,5), tanpa bias 0,793 vs 0,717 | [PDF](v5_training_afl/laporan/laporan_training_afl.pdf) |
| [v6_cabang_ekg](v6_cabang_ekg/) | Cabang sinyal EKG mentah 50 Hz: CNN EKG saja, dua cabang EKG + RR, late fusion | EKG saja 0,651 (lebih buruk); dua cabang 0,747 (cenderung buruk); late fusion 0,838 (setara). 07910 & 08215 tetap 0 | [PDF](v6_cabang_ekg/laporan/laporan_cabang_ekg.pdf) |

| [v7_augmentasi_takikardia](v7_augmentasi_takikardia/) | MicroTransformer + sinus takikardia sintetis (detak Normal asli pasien latih yang dipercepat): 100–130 bpm (7a), 100–160 bpm (7b) | 7a: ambang alarm 115 → **135 bpm**, Macro-F1 0,825 (setara), tetapi recall AFL 0,592 → 0,555 (08378: 0,96 → 0,71). 7b rusak (0,700). Belum dikemas | [PDF](v7_augmentasi_takikardia/laporan/laporan_augmentasi_takikardia.pdf) |

| [v8_chapman_shaoxing](v8_chapman_shaoxing/) | Data eksternal PhysioNet `ecg-arrhythmia` (Chapman–Shaoxing & Ningbo, 42 ribu pasien, lead II); dijalankan di Colab/Kaggle | **Sinus takikardia asli: paket v3.1 alarm 42% (98–100% di ≥120 bpm).** 8c (+ Chapman tanpa label AFL): alarm ST 9%, AFDB 0,799 (setara), tetapi AFL 2:1 & SVT hilang (08378: 0,98 → 0,51). Label AFL Ningbo tidak andal. Belum dikemas | [PDF](v8_chapman_shaoxing/laporan/laporan_chapman_shaoxing.pdf) |

| [v9_gelombang_ekg](v9_gelombang_ekg/) | Cabang bentuk gelombang EKG (CNN v6) dilatih dengan AFDB + 34 ribu pasien Chapman berlabel andal; Kaggle 2× T4 | **Chapman: alarm sinus takikardia 42% → 2%, Macro-F1 0,520 → 0,739.** AFDB: 0,687 (P = 0,07), AFL palsu melonjak, 08378 → 0,00, bias tidak stabil. Batasnya kini pergeseran domain. Belum dikemas | [PDF](v9_gelombang_ekg/laporan/laporan_gelombang_ekg.pdf) |
| [v10_icentia_wearable](v10_icentia_wearable/) | Data wearable Icentia11k (patch EKG satu lead, 510 pasien AFL, akademik saja) untuk cabang bentuk gelombang; Kaggle CPU + 2× T4 | **Icentia: Macro-F1 0,477 → 0,666, alarm Normal ≥ 120 bpm 83% → 13%; Chapman: lebih baik dari 9a (P = 1,00).** AFDB: 0,646, lebih buruk dari v31 (P = 0,01): AF kasar disebut AFL. Batasnya kini definisi AFL vs AFIB antar dataset. Belum dikemas | [PDF](v10_icentia_wearable/laporan/laporan_icentia_wearable.pdf) |
| [v11_model_tim](v11_model_tim/) | Permintaan pembimbing: 6 model berbasis RR yang berbeda dari v1 (BiGRU, TCN, InceptionTime, MiniRocket, XGBoost, SVM), satu folder per anggota tim, protokol v3 yang sama; 31 setelan termasuk balancing (under/over/SMOTE) dan bobot kelas | **Tidak ada yang lebih baik dari v3.1.** Dengan resep yang sama, BiGRU terbaik (0,821, setara v3.1); TCN, InceptionTime, XGBoost, dan MiniRocket lebih buruk darinya. Dengan bobot kelas penuh, BiGRU dan TCN 0,830, setara v3.1, juga tanpa 04936; bootstrap bersarang vs v1 P = 0,98. Balancing tidak lebih baik dari bobot kelas. Alarm Normal ≥ 120 bpm 72–92%. Belum dikemas | [PDF](v11_model_tim/laporan/laporan_v11_model_tim.pdf) |
| [v11.1_dua_kelas](v11.1_dua_kelas/) | Evaluasi saja: model yang ada (v1, v3.1, BiGRU/TCN/XGBoost v11) dinilai sebagai 2 kelas (Normal vs AF/AFL, "AF₁" di literatur) dengan protokol lengkap, termasuk butir 9–10 | **Macro-F1 2 kelas 0,916–0,938 (target ≥ 0,90 tercapai).** BiGRU bawaan 0,938, lebih baik dari v3.1 (P = 1,00; bersarang 0,96) dan v1 (bersarang 0,98), tetapi tidak pasti tanpa 04936 (P = 0,92). Keputusan harus "3 kelas lalu digabung": bias 2 kelas tersendiri membuat alarm pada ritme normal teratur 80–95 bpm (sampai 22%). Alarm Normal ≥ 120 bpm tetap 72–95%. Belum dikemas | [PDF](v11.1_dua_kelas/laporan/laporan_dua_kelas.pdf) |
| [v12_shdb_atrium](v12_shdb_atrium/) | Cabang aktivitas atrium (pembatalan QRST + fitur fisiologis) dari EKG 50 Hz + data Holter SHDB-AF (98 rekaman, 12 subjek AFL); gerbang cek fitur sebelum training | **Berhenti di gerbang**: frekuensi atrium dominan dan organisasi tidak memisahkan AFL dari AFIB searah di AFDB dan SHDB-AF, juga setelah revisi pembatalan QRST (AUROC AFL vs AFIB 0,46–0,61). Training tidak dijalankan | [README](v12_shdb_atrium/README.md) |
| [v13_ecgfounder](v13_ecgfounder/) | Model fondasi EKG ECGFounder 1-lead (MIT, 1,8 juta pasien) dipakai beku pada EKG 50 Hz: zero-shot dari head aslinya, probe per fold, fusi dengan v3.1; kontrol 250 Hz; uji sinus takikardia asli, PTB-XL, Icentia | **Alarm sinus takikardia asli 98,4% → 5,4% (zero-shot)**; lebih baik dari v3.1 di PTB-XL (0,642 vs 0,622) dan Icentia (0,549 vs 0,476). Tetapi AFDB 3 kelas tidak lebih baik (fusi 0,775, P = 0,19; zero-shot 0,663, lebih buruk) dan AFL 2:1 (08378) hilang | [PDF](v13_ecgfounder/laporan/laporan_v13_ecgfounder.pdf) |
| [v14_konteks_menit](v14_konteks_menit/) | Tahap kedua (XGBoost) dengan dinamika detak 2–10 menit (cara mulai, kedataran, lama keadaan cepat-teratur) di atas OOF v3.1; hipotesis dikunci sebelum training; uji EPHNOGRAM dan uji stres konteks sintetis | **Keempat hipotesis tidak lolos**: Macro-F1 0,781 vs 0,832 (P = 0,033), Se episode AFL 0,69 → 0,44, alarm sinus asli ≥ 120 bpm 98% → 96%. Hanya episode AFIB palsu turun (11,9 → 3,1/jam) | [PDF](v14_konteks_menit/laporan/laporan_v14_konteks_menit.pdf) |
| [v15_kuantisasi_rr](v15_kuantisasi_rr/) | Pra-registrasi hipotesis v12 (kuantisasi RR, konsistensi pra-QRS) pada data baru: CPSC 2021, lalu PTB-XL sebagai analisis tambahan; tahap gabungan dengan v3.1 | **Hasil utama: tidak cukup data** (CPSC 2021: AFL hanya 2 subjek). PTB-XL: AUROC AFL vs AFIB 0,92 / 0,86 (lolos ambang); gabungan 3 kelas memperburuk AFDB (0,756, P = 0,037) | [PDF](v15_kuantisasi_rr/laporan/laporan_v15_kuantisasi_rr.pdf) |
| [v16_hierarki_ecgfounder](v16_hierarki_ecgfounder/) | Pra-registrasi: ECGFounder (tanpa penyetelan) memutuskan sinus vs AF, RR v3.1 atau fitur atrium v15 memutuskan AFIB vs AFL | Alarm sinus takikardia asli 98,4% → **0,1%**, episode AFIB palsu 11,9 → 1,2/jam; tetapi tahap 1 menyebut 85% AFL sebagai sinus → 3 kelas **lebih buruk** (0,650, P = 0,020), 2 kelas setara (0,951 vs 0,926, P = 0,92); bukan kandidat | [PDF](v16_hierarki_ecgfounder/laporan/laporan_v16_hierarki_ecgfounder.pdf) |
| [v16.1_dua_kelas](v16.1_dua_kelas/) | Versi minor, tanpa training: keputusan sinus vs AF tahap 1 v16 (ECGFounder tanpa penyetelan) dinilai sebagai detektor AF 2 kelas; tanpa 04936, PTB-XL, Icentia, per pasien, episode, kurva ambang, fungsi prediksi `DetektorAF` (paritas + latensi) | AFDB 2 kelas 0,951 vs 0,926 (P = 0,92), PTB-XL 0,944 vs 0,817 (lebih baik), alarm sinus takikardia 0,1%, episode AF palsu 11,4 → 1,0/jam; tetapi sensitivitas lebih rendah dan **Icentia lebih buruk** (0,823 vs 0,858) → belum kandidat | [PDF](v16.1_dua_kelas/laporan/laporan_v16_1_dua_kelas.pdf) |
| [v16.2_cpsc2021](v16.2_cpsc2021/) | Pra-registrasi: ambang t = −1 detektor AF 2 kelas ECGFounder (D-1) dikunci dari kurva v16.1, lalu diuji di **CPSC 2021** (1.424 rekaman, 105 subjek, belum pernah dinilai); label Normal mengikuti konvensi resmi (penyimpangan tercatat, kedua analisis dilaporkan) | **Kandidat detektor AF 2 kelas untuk uji perangkat**: Macro-F1 0,953 vs v3.1 0,881 (lebih baik), sensitivitas 0,955 vs 0,951 (sama), spesifisitas 0,958 vs 0,853, episode AF palsu 25,7 → 7,0/jam; lolos juga di analisis yang dikunci. Bukan pengganti paket 3 kelas | [PDF](v16.2_cpsc2021/laporan/laporan_v16_2_cpsc2021.pdf) |
| [v16.3_replikasi](v16.3_replikasi/) | Pra-registrasi: D-1 (ECGFounder, ambang −1) diuji ulang tanpa penyetelan di **CinC 2017** (EKG genggam satu lead), **MIT-BIH Arrhythmia**, **LTAFDB** (Holter 24 jam) | **Replikasi sebagian (2/3)**: Macro-F1 lebih baik di ketiga dataset dan sensitivitas lebih baik di MIT-BIH (0,983 vs 0,916) dan LTAFDB (0,974 vs 0,948), tetapi **lebih rendah di CinC 2017** (0,830 vs 0,884), seperti di Icentia. Kelemahan baru: sinus bradikardia LTAFDB 28% dialarmkan. Tetap kandidat uji perangkat, dengan sensitivitas sebagai kriteria utama | [PDF](v16.3_replikasi/laporan/laporan_v16_3_replikasi.pdf) |
| [v16.4_nsrdb_georgia](v16.4_nsrdb_georgia/) | Pra-registrasi: D-1 dan v3.1 di dua data yang belum pernah dipakai: **NSRDB** (18 orang sehat, Holter 24 jam) dan **Georgia** (10.344 EKG klinis 12-lead 10 s, lead I, label dokter) | NSRDB: alarm palsu 15,0% → 5,6% window (18/18 subjek lebih rendah), tetapi tanpa aturan durasi masih ±21 episode palsu/jam. Georgia: Macro-F1 0,691 → 0,843, alarm sinus takikardia 50% → 2%, **tetapi Se 0,770 → 0,706 (H2 gagal)**. Kekurangan Se D-1 konsisten di tiga data non-Holter; di Holter setara/lebih baik | [PDF](v16.4_nsrdb_georgia/laporan/laporan_v16_4_nsrdb_georgia.pdf) |
| [v16.5_episode_30s](v16.5_episode_30s/) | Pra-registrasi: aturan episode klinis AF ≥ 30 s di atas keputusan per window D-1 dan v3.1; alarm palsu per jam di NSRDB, **EDB** (baru), latihan fisik; Se episode di AFDB, LTAFDB, MIT-BIH, CPSC 2021 | Alarm palsu D-1 orang sehat 20,6 → 1,23/jam, saat olahraga 0,13 (v3.1 + 30 s: 8,8); Se episode AF ≥ 30 s 0,91–0,98. **Target (≤ 1,0/jam, Se ≥ 0,90) belum terpenuhi**: S1 gagal tipis di NSRDB | [PDF](v16.5_episode_30s/laporan/laporan_v16_5_episode_30s.pdf) |
| [v16.6_episode_celah](v16.6_episode_celah/) | Pra-registrasi: durasi minimum × toleransi celah dipilih di data v16.5, divalidasi sekali di **SHDB-AF** (98 Holter 24 jam, 93 subjek), **afpdb**, **incartdb** (baru), svdb (deskriptif) | Terpilih 45 s tanpa celah → **Se SHDB-AF 0,700, gagal**; celah menaikkan alarm palsu. Sekunder: **D-1 + 30 s (S1) lolos di data baru**: Se 0,942, alarm palsu 0,87/jam (SHDB-AF), 0,89 (afpdb), 0,77 (incartdb); v3.1 + 30 s 1,80/2,21/5,98. S1 = sistem terbaik saat ini untuk deteksi AF 2 kelas | [PDF](v16.6_episode_celah/laporan/laporan_v16_6_episode_celah.pdf) |
| [v16.7_konfirmasi_wearable](v16.7_konfirmasi_wearable/) | Pra-registrasi, **uji konfirmasi primer** S1 = D-1 + episode AF ≥ 30 s di **Icentia11k** (600 pasien wearable, 534 jam, segmen utuh yang belum pernah diproses) dan uji stres derau **nstdb** | **S1 terkonfirmasi**: Se episode AF ≥ 30 s 0,940 [0,916–0,960] (v3.1 + 30 s 0,899), alarm palsu 0,50/jam (v3.1 + 30 s 2,33); 4/4 kriteria primer lolos. Derau: 1,01/jam pada SNR 12 dB, 5–8/jam pada ≤ 6 dB → perlu gerbang kualitas | [PDF](v16.7_konfirmasi_wearable/laporan/laporan_v16_7_konfirmasi_wearable.pdf) |
| [v16.8_gerbang_kualitas](v16.8_gerbang_kualitas/) | Pra-registrasi: gerbang kualitas sinyal (median korelasi detak ke templat QRS, v3.4) untuk S1; ambang dipilih di nstdb/LTAFDB/MIT-BIH, divalidasi di afpdb + derau otot/elektroda nstdb dan Icentia11k | **Gagal**: derau otot 6 dB 1,36 → 1,34 alarm palsu/jam (target ≤ 1,0); hanya derau elektroda (jenis derau tahap A) yang turun (0 dB: 6,05 → 3,11). Biaya di Icentia kecil (Se 0,940 → 0,931). S1 tetap sistem terbaik | [PDF](v16.8_gerbang_kualitas/laporan/laporan_v16_8_gerbang_kualitas.pdf) |
| [v16.9_sinus_bradikardia](v16.9_sinus_bradikardia/) | Pra-registrasi, evaluasi dari cache: S1 pada sinus bradikardia berlabel (LTAFDB) dan pada window non-AF berdetak lambat di 10 dataset | **Kelemahan nyata tetapi sempit**: saat sinus bradikardia S1 8,89 alarm palsu/jam (v3.1 + 30 s: 0,48), 90% dari 3 dari 25 subjek; detak lambat juga menaikkan alarm S1 di EDB dan Icentia11k (pola U, < 50 bpm > 1/jam di 8 dari 10 dataset). Aturan detak tidak bisa: 11–16% window AF berdetak < 60 bpm. S1 tetap sistem terbaik | [PDF](v16.9_sinus_bradikardia/laporan/laporan_v16_9_sinus_bradikardia.pdf) |
| [v17_fusi_wearable](v17_fusi_wearable/) | Pra-registrasi: fusi 3 parameter (d1 ECGFounder + log-odds AF v3.1) dilatih di **CinC 2017** (CC BY 3.0), diuji di **Icentia11k** (wearable, hanya evaluasi), dijaga di Holter dan sinus takikardia asli | **Tidak menggantikan D-1**: sensitivitas Icentia 0,921 vs D-1 0,843 (Macro-F1 0,913 vs 0,870), tetapi alarm sinus takikardia asli 57% (batas 10%) dan lebih buruk dari D-1 di CPSC 2021 dan MIT-BIH. Titik kerja harus dipilih dari data perangkat | [PDF](v17_fusi_wearable/laporan/laporan_v17_fusi_wearable.pdf) |

\*Rata-rata out-of-fold, pasien uji tidak pernah ikut dilatih. Ini bukan uji pada perangkat asli.

## Keputusan proyek yang berlaku

- **Paket produksi aktif: `v3.1_kalibrasi_produksi/model_export/sihedaf_v3_1.{pt,json}`.** Paket v2 dipertahankan sebagai
  titik rollback (bobot sama, hanya bias berbeda).
- **Hanya v3.1 (dan v2) yang berupa paket siap pakai.** v4–v10 tidak menghasilkan paket. v4–v6 tidak menyimpan bobot. v7–v10
  menyimpan bobot per fold (umumnya 25 file per varian, tanpa bias final) di `cache/vN/model/`. Folder ini tidak ada di git, dan
  `predictor.py` tidak bisa memakainya. Untuk backend dan uji di perangkat, termasuk saat ingin membandingkan dengan model lain, tetap pakai paket v3.1
  sampai ada paket uji tersendiri (versi minor `vN.M_<nama>/`, perlu persetujuan pemilik repo). Model v10 dilatih dengan
  Icentia11k (lisensi CC BY-NC-SA), jadi hanya boleh dipakai untuk akademik/riset.
- Sistem lama (`final.joblib` + `find_peaks`) diganti. Panduan migrasi: [BACKEND.md](v3.1_kalibrasi_produksi/BACKEND.md).
- Model produksi tetap MicroTransformer.
- **Pencarian arsitektur yang inputnya hanya interval RR dihentikan.** Empat arsitektur gagal di pasien AFL yang sama.
  Arsitektur baru hanya layak dicoba jika menambah informasi input (mis. sinyal EKG mentah).
- Semua eksperimen berikutnya memakai protokol v3 dan dibandingkan berpasangan dengan referensi **v3.1**.
- Ensemble MT + BiGRU dan smoothing antar-window **tidak** dipakai sebagai default (v4). Smoothing offline tersedia sebagai
  opsi produk "hemat alarm" jika pemilik produk memilih menukar deteksi AFL dengan lebih sedikit alarm palsu.
- Resep training (v5): window AFL rapat **tidak** dipakai; **bobot kelas lunak (akar) jadi resep bawaan untuk model baru**
  (setara, koreksi bias setengahnya). Model produksi tidak diganti.
- Semua tuas yang inputnya hanya interval RR sudah dicoba (v3–v5) dan berhenti di Macro-F1 ±0,83.
- Input bentuk gelombang EKG 50 Hz (v6) juga tidak memperbaiki: CNN overfit per pasien; late fusion setara (kandidat, perlu
  konfirmasi). **Batas proyek sekarang adalah jumlah pasien AFL (7 di AFDB).** Eksperimen berikutnya (v7) butuh data dengan
  lebih banyak pasien AFL (mis. Chapman–Shaoxing di PhysioNet) — unduhan perlu persetujuan pemilik repo.
- **Batasan produksi yang diketahui (v3.2): detak cepat yang teratur (sinus takikardia, mulai ±115 bpm) dilaporkan sebagai
  AFIB/AFL.** Penyebabnya, AFDB hampir tidak punya ritme normal yang cepat. Peringatan untuk backend ada di BACKEND.md.
  Aturan produk "takikardia teratur, perlu dicek" menunggu keputusan pemilik produk: aturan ini juga mengalihkan 86% AFL yang
  terdeteksi. Data v7 harus memuat sinus takikardia, tidak hanya pasien AFL tambahan.
- Uji stres takikardia wajib untuk setiap model kandidat (AGENTS.md, protokol butir 9).
- **Data eksternal (v8) mengonfirmasi batas interval RR.** Dengan sinus takikardia asli sebagai data latih, alarm palsu ST turun
  42% → 9%, tetapi AFL 2:1 dan SVT ikut disebut Normal. Langkah berikutnya (v9) adalah cabang bentuk gelombang EKG yang
  dilatih dengan AFDB + Chapman (label andal). Label AFL bagian Ningbo di `ecg-arrhythmia` tidak boleh dipakai.
- **Bentuk gelombang (v9) memecahkan alarm palsu takikardia di data klinis eksternal** (Chapman: 42% → 2%), tetapi model
  belum menyeberang ke AFDB (Holter): AFL palsu melonjak dan 08378 hilang. Batas proyek sekarang adalah pergeseran domain.
  Langkah berikutnya: validasi dengan rekaman perangkat SIHEDAF berlabel, lalu training yang tahan domain.
- **Data wearable (v10, Icentia11k) membuat model bentuk gelombang jauh lebih baik di dua domain eksternal** (Icentia: AFL
  terdeteksi untuk pertama kalinya, alarm sinus cepat 83% → 13%; Chapman lebih baik dari v9), tetapi **lebih buruk di AFDB**:
  model menyebut banyak AF "kasar" AFDB sebagai AFL (AUROC AFL turun, kalibrasi tidak menolong). AFL Icentia kebanyakan blok
  bervariasi (tidak teratur), berbeda dari AFL 2:1 di AFDB. Tiga domain publik memberi vonis berlawanan; **langkah berikutnya
  adalah rekaman perangkat SIHEDAF berlabel dokter** dan keputusan produk apakah AFL perlu dibedakan dari AFIB.
- **Enam model tim (v11, permintaan pembimbing) mengonfirmasi batas interval RR.** Dengan resep yang sama, BiGRU terbaik.
  Dengan bobot kelas penuh, BiGRU dan TCN setara v3.1, tetapi tidak lebih baik. Balancing data tidak lebih baik daripada
  bobot kelas. Paket produksi tetap v3.1.
- Augmentasi sinus takikardia sintetis (v7) hanya memindahkan batas alarm palsu ke ±135 bpm, dengan harga sebagian deteksi
  AFL 2:1. Varian 100–130 bpm adalah kandidat yang **menunggu keputusan pemilik produk**. Di 135–160 bpm, interval RR saja
  tidak bisa membedakan sinus takikardia dari AFL 2:1; ini butuh data eksternal dengan bentuk gelombang EKG.
- **Sistem terbaik untuk deteksi AF 2 kelas (episode), v16.2–v16.7: D-1 + aturan episode AF ≥ 30 s (S1), terkonfirmasi di wearable
  kontinu (v16.7: Icentia11k Se 0,94, alarm palsu 0,50/jam).** D-1 = ECGFounder 1-lead
  zero-shot (MIT), AF bila d1 > −1. Lolos atau setara di CPSC 2021, MIT-BIH, LTAFDB, SHDB-AF, afpdb, incartdb. Alarm palsu ±0,8–1,2/jam
  di Holter (v3.1 + 30 s: 1,8–6,0). Saat olahraga 0,13/jam (v3.1: 8,8). Kelemahan: derau berat (SNR ≤ 6 dB: 5–8 alarm palsu/jam,
  perlu gerbang kualitas), alarm palsu pada detak lambat < 50 bpm di sebagian orang (v16.9), Se per window lebih rendah di EKG pendek sesaat, dan tidak membedakan AFIB dari AFL. **Ini bukan paket produksi:** paket 3 kelas tetap v3.1. Mengemas D-1 + aturan sebagai paket uji perangkat
  (versi minor) butuh persetujuan pemilik repo. Data publik Holter yang relevan sudah terpakai, jadi titik kerja berikutnya hanya dari
  rekaman perangkat SIHEDAF berlabel.

## Struktur folder

```
README.md                 dokumentasi untuk manusia
AGENTS.md, CLAUDE.md      aturan untuk agent & tim (CLAUDE.md = penunjuk ke AGENTS.md untuk Claude Code)
requirements.txt          dependensi lingkungan tim
tools/                    run_notebook.py (runner), download_afdb.py (unduh AFDB), report_fonts.py (font laporan)
data/afdb/                AFDB mentah            ← tidak di-commit, unduh dulu
cache/                    hasil ekstraksi & OOF  ← tidak di-commit, dibuat ulang oleh notebook
logs/                     log runner             ← tidak di-commit
v1_lama/                  versi sebelum revisi (arsip, beku)
v2_revisi/                revisi pipeline + model produksi + laporan/
v3_evaluasi_robust/       evaluasi robust + perbandingan dengan v1 + laporan/
v3.1_kalibrasi_produksi/  bias baru + paket produksi aktif + BACKEND.md + laporan/
v3.2_uji_takikardia/      uji stres takikardia, bootstrap bersarang, sensitivitas 04936 (evaluasi saja) + laporan/
v3.3_takikardia_asli/     uji sinus takikardia asli (EKG latihan fisik), aturan produk, abstain, metrik episode + laporan/
v4_smoothing_ensemble/    pasca-proses (smoothing, ensemble) tanpa training + laporan/
v5_training_afl/          dua resep training untuk AFL (window rapat, bobot lunak) + laporan/
v6_cabang_ekg/            cabang sinyal EKG mentah 50 Hz (CNN, dua cabang, late fusion) + laporan/
v7_augmentasi_takikardia/ sinus takikardia sintetis sebagai data latih (100–130, 100–160 bpm) + laporan/
v8_chapman_shaoxing/      data eksternal Chapman–Shaoxing (notebook Kaggle/Colab-first) + laporan/
v9_gelombang_ekg/         cabang bentuk gelombang EKG dengan AFDB + Chapman (Kaggle) + laporan/
v10_icentia_wearable/     data wearable Icentia11k (pindai anotasi, ekstraksi, Kaggle CPU + GPU) + laporan/
v11_model_tim/            enam model tim (satu folder per anggota) + bersama/ + gabungan.ipynb + laporan/
v11.1_dua_kelas/          evaluasi 2 kelas (Normal vs AF/AFL) model yang ada + laporan/
v12_shdb_atrium/          cabang aktivitas atrium + data SHDB-AF (berhenti di gerbang)
v13_ecgfounder/           model fondasi EKG ECGFounder 1-lead (beku: zero-shot, probe, fusi) + laporan/
v14_konteks_menit/        tahap kedua dengan konteks 2–10 menit (dinamika detak) + laporan/
v15_kuantisasi_rr/        pra-registrasi kuantisasi RR + konsistensi pra-QRS (CPSC 2021, PTB-XL) + laporan/
v16_hierarki_ecgfounder/  hierarki: ECGFounder sinus vs AF, lalu RR/fitur atrium AFIB vs AFL (pra-registrasi) + laporan/
v16.1_dua_kelas/          detektor AF 2 kelas dari tahap 1 v16 + fungsi prediksi DetektorAF + laporan/
v16.2_cpsc2021/           uji ambang −1 detektor AF 2 kelas di CPSC 2021 (pra-registrasi) + laporan/
v16.3_replikasi/          replikasi D-1 di CinC 2017, MIT-BIH, LTAFDB (pra-registrasi) + laporan/
v16.4_nsrdb_georgia/      D-1 di NSRDB (Holter sehat) dan Georgia 12-lead (pra-registrasi) + laporan/
v16.5_episode_30s/        aturan episode AF ≥ 30 s, alarm palsu per jam, EDB (pra-registrasi) + laporan/
v16.6_episode_celah/      aturan episode dengan celah; validasi SHDB-AF, afpdb, incartdb (pra-registrasi) + laporan/
v16.7_konfirmasi_wearable/ uji konfirmasi S1 di Icentia11k (wearable) + uji derau nstdb (pra-registrasi) + laporan/
v16.8_gerbang_kualitas/   gerbang kualitas sinyal (SQI templat QRS) untuk S1, gagal (pra-registrasi) + laporan/
v16.9_sinus_bradikardia/  sinus bradikardia dan detak lambat pada S1, evaluasi dari cache (pra-registrasi) + laporan/
v17_fusi_wearable/        fusi ECGFounder + RR dilatih di CinC 2017, uji wearable Icentia (pra-registrasi) + laporan/
```

Setiap folder versi berisi `README.md` (apa, hasil, cara menjalankan) dan `laporan/`
(PDF + `build_report.py` pembuatnya). Notebook dijalankan dengan **folder versinya sebagai
direktori kerja**; data dan cache diakses lewat `../data` dan `../cache`.

## Setup

Disarankan **WSL2 / Linux dengan GPU NVIDIA**. Kaggle dan Google Colab (GPU) juga bisa; langkah khususnya
ada di [AGENTS.md](AGENTS.md) bagian "Lingkungan". CPU saja bisa, tapi lambat.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m ipykernel install --user --name sihedaf-v2     # opsional; runner memakai python3 jika tidak ada
```

Unduh AFDB (±600 MB) ke `data/afdb/`. Skrip ini bisa dilanjutkan jika koneksi terputus:

```bash
python tools/download_afdb.py
```

Hasil di `cache/` tidak ada di git. OOF referensi bisa diunduh dari GitHub Release supaya tidak perlu training ulang,
dan supaya perbandingan memakai OOF yang sama:

| Release | Isi | Ukuran |
|---|---|---|
| [`v11-data`](https://github.com/farhanlhsn/WrapSihedaf/releases/tag/v11-data) | data uji (`dataset_v2_detector.npz`), `folds.npz`, OOF v3.1, v1, kontrol v7 | 53 MB |
| [`v11-oof`](https://github.com/farhanlhsn/WrapSihedaf/releases/tag/v11-oof) | data uji + seluruh `oof_robust/` v3–v7 + OOF 31 setelan enam model v11 (tanpa bobot) | 494 MB |
| [`cache-v3.3-v3.4`](https://github.com/farhanlhsn/WrapSihedaf/releases/tag/cache-v3.3-v3.4) | `cache/v3_3` (window RR latihan fisik, prediksi) dan `cache/v3_4` (OOF, **bobot 25 model fold** resep v3.1, keluaran data eksternal), satu zip per versi | 109 MB |
| [`cache-v12-v17`](https://github.com/farhanlhsn/WrapSihedaf/releases/tag/cache-v12-v17) | `cache/v12` … `cache/v17`, satu zip per versi (`cache_v13.zip`, `cache_v16.2.zip`, …): prediksi, fitur dan skor ECGFounder, ringkasan; tanpa turunan Icentia11k | 1,3 GB |

Cara unduh dan cek sidik jari `v11-oof`: [v11_model_tim/README.md](v11_model_tim/README.md#oof-hasil-v11-dan-v3-github-release-v11-oof).
Untuk `cache-v12-v17` (dan sama untuk `cache-v3.3-v3.4` dengan `cache_v3_4.zip` dan `cache_v3_3_v3_4_sidik_jari.json`), unduh
hanya versi yang dibutuhkan (contoh v16.2), ekstrak ke `cache/`, lalu cek:

```bash
gh release download cache-v12-v17 --repo farhanlhsn/WrapSihedaf --dir cache --pattern 'cache_v16.2.zip' --pattern 'cache_v12_v17_sidik_jari.json'
```

```bash
python -c "import zipfile; zipfile.ZipFile('cache/cache_v16.2.zip').extractall('cache')"
```

```bash
python -c "import hashlib, json, pathlib; s = json.load(open('cache/cache_v12_v17_sidik_jari.json'))['berkas']; ada = {r: v for r, v in s.items() if pathlib.Path('cache', r).exists()}; bad = [r for r, v in ada.items() if hashlib.sha256(pathlib.Path('cache', r).read_bytes()).hexdigest() != v['sha256']]; print(len(ada), 'berkas dicek,', 'cocok semua' if not bad else bad)"
```

Turunan Icentia11k (CC BY-NC-SA: `v13/icentia_50hz.npz`, `v16/icentia_p31.npy`, `v16.7/Icentia/`, `v16.8/sqi/Icentia/`) sengaja
tidak dirilis; evaluasi Icentia di v13, v16, v16.7, v16.8 butuh salinan dari pemilik repo. Begitu juga cache v10 (seluruhnya
turunan Icentia11k). Bobot model selain v3.4 belum ada di release; cache v8 dan v9 ada di dataset Kaggle
`sihedaf-v8-cache`/`sihedaf-v9-cache`.

## Menjalankan

Semua perintah dari root repo dengan venv di atas. Waktu di tabel diukur pada PC desktop dengan GPU NVIDIA 8 GB; di mesin lain
bisa berbeda.

| Tujuan | Perintah | Waktu |
|---|---|---|
| Dataset v1 (lama) | `cd v1_lama && python ekstraksi_hrv.py` | ±20 s |
| Pipeline v2 lengkap (ekstraksi, benchmark, **ekspor model**) | `python tools/run_notebook.py v2_revisi/revisi_sihedaf_v2.ipynb` | ±1 jam |
| Evaluasi robust v3 | `python tools/run_notebook.py v3_evaluasi_robust/evaluasi_robust.ipynb` | ±1 jam 40 mnt (pertama kali), ±5 mnt jika OOF sudah ada |
| Ukur ulang v1 dengan protokol v3 | `cd v3_evaluasi_robust && python eval_versi_lama.py` | ±25 mnt |
| Kalibrasi & paket produksi v3.1 | `python tools/run_notebook.py v3.1_kalibrasi_produksi/kalibrasi_produksi.ipynb` | ±2 mnt |
| Uji stres takikardia v3.2 | `python tools/run_notebook.py v3.2_uji_takikardia/uji_takikardia.ipynb` | ±10 mnt (CPU) pertama kali, lalu dari cache |
| Smoothing & ensemble v4 | `python tools/run_notebook.py v4_smoothing_ensemble/smoothing_ensemble.ipynb` | ±14 mnt (CPU) |
| Resep training v5 | `python tools/run_notebook.py v5_training_afl/training_afl.ipynb` | ±60 mnt (GPU) |
| Cabang EKG v6 | `python tools/run_notebook.py v6_cabang_ekg/cabang_ekg.ipynb` | ±2 jam (GPU) |
| Data eksternal v8 (Kaggle/Colab, GPU) | lihat `v8_chapman_shaoxing/README.md` | ±1,5 jam (Kaggle 2× T4) / ±3 jam (Colab 1× T4) |
| Augmentasi takikardia v7 | `python tools/run_notebook.py v7_augmentasi_takikardia/augmentasi_takikardia.ipynb` | ±45 mnt (GPU) + ±12 mnt evaluasi |
| Data wearable v10 (Kaggle CPU untuk data, lalu GPU) | lihat `v10_icentia_wearable/README.md` | ±35 mnt data + ±5 jam (Kaggle 2× T4) |
| Laporan versi N | `python vN_.../laporan/build_report.py --extract`, lalu `python vN_.../laporan/build_report.py` | ±3 mnt |

Urutan dari nol: unduh AFDB → v2 (membuat `cache/dataset_v2_*.npz`) → v3 (+ `eval_versi_lama.py`) → v3.1 → laporan.
Laporan butuh ReportLab dan font Arial atau Liberation Sans (`apt-get install -y fonts-liberation`);
lihat `tools/report_fonts.py`.

## Memakai model di backend

```python
import sys; sys.path.insert(0, "v2_revisi")
from predictor import ECGPredictor
P = ECGPredictor("v3.1_kalibrasi_produksi/model_export", name="sihedaf_v3_1")
hasil = P.predict_ecg(sinyal_mentah, fs=50)   # atau P.predict_ibi(ibi_ms)  — IBI dalam MILIDETIK
```

Backend menyalin `v2_revisi/{hrv_core.py, sihedaf_models.py, predictor.py}` dan
`v3.1_kalibrasi_produksi/model_export/`. Panduan lengkap, format keluaran, checklist migrasi, dan rollback ada di
[BACKEND.md](v3.1_kalibrasi_produksi/BACKEND.md). `final.joblib` dan `predictor.py` di `v1_lama/` sudah usang;
front-end lamanya membuat recall Normal hanya ±34%.

## Menambah eksperimen baru

Eksperimen baru → `vN_<nama_singkat>/` (berikutnya v18). Tindak lanjut tanpa eksperimen/training baru (kalibrasi, pengemasan,
perbaikan) → versi minor `vN.M_<nama>/`, seperti v3.1. Ikuti checklist "Definition of Done" di [AGENTS.md](AGENTS.md):
notebook ter-eksekusi, kesimpulan, `README.md` versi, laporan PDF di `v4_.../laporan/`, dan baris baru
di tabel "Versi dan hasil" di atas.
