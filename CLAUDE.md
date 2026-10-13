# CLAUDE.md

Aturan proyek ada di AGENTS.md (berlaku untuk semua agent dan anggota tim). Jangan duplikasi isinya di sini.

@AGENTS.md

## Khusus Claude Code

- Berkomunikasi dengan pengguna dalam Bahasa Indonesia. Kode dan nama variabel mengikuti gaya file sekitarnya.
- Cek lingkungan dulu sebelum menjalankan apa pun: WSL/Linux lokal, Colab, atau Kaggle (lihat AGENTS.md
  "Lingkungan"). Jangan berasumsi ada GPU, venv, atau path tertentu.
- Job > 2 menit (training, notebook penuh) dijalankan di background dan dipantau lewat log di `logs/`. Jangan
  polling dengan sleep. Pastikan job bisa dilanjutkan (hasil per langkah disimpan ke `cache/`).
- **Minta persetujuan dulu** sebelum: menjalankan notebook v2 atau v3.1 (keduanya menulis paket model), mengubah
  `*/model_export/` (paket aktif: `v3.1_kalibrasi_produksi/`), menghapus isi `cache/` atau `data/`. Jangan commit/push;
  pemilik repo melakukannya sendiri.
- Saat membuat laporan PDF: hitung angka dari cache (bukan disalin manual), jalankan `build_report.py`, render
  setiap halaman ke PNG lalu **lihat** gambarnya sebelum menyerahkan. Untuk grafik, ikuti palet & aturan di
  AGENTS.md "Gaya laporan".
- Menutup sebuah versi = semua butir "Definition of Done" di AGENTS.md terpenuhi. Laporkan juga yang belum.
