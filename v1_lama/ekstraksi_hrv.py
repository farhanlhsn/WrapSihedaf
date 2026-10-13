"""
AFDB Extraction Pipeline (23 Pasien Bersih - Clinical Gold Standard)
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
import wfdb

class AFDBExtractor:
    # Path disesuaikan dengan struktur repo (v1_lama/ dijalankan sebagai cwd);
    # versi asli: data_folder="./files" dan keluaran ditulis ke data_folder/..
    def __init__(self, data_folder="../data/afdb", out_dir="../cache/v1"):
        self.data_folder = data_folder
        self.out_dir = out_dir
        self.records_file = "RECORDS"
        self.out_tabular = "dataset_hrv_tabular.csv"
        self.out_sequence = "dataset_hrv_sequence_23p.npz"
        
        self.expected_fs = 250
        self.window_sec = 10.0
        
        self.class_stride = {
            0: 5.0,   # Normal: 5s stride
            1: 10.0,  # AFIB: 10s stride
            2: 1.0    # AFL: 1s stride
        }
        
        self.ibi_min = 300.0
        self.ibi_max = 2000.0
        self.max_outliers = 1
        
        self.max_seq_len = 30
        
        self.label_map = {
            "(N": 0,
            "(AFIB": 1,
            "(AFL": 2
        }
        
        self.blacklist = {"00735", "03665"}
        
        self._setup_logger()
        
    def _setup_logger(self):
        self.logger = logging.getLogger("AFDBExtractor")
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            fmt = logging.Formatter("[%(asctime)s] %(message)s", datefmt="%H:%M:%S")
            handler.setFormatter(fmt)
            self.logger.addHandler(handler)
            
    def run_pipeline(self):
        self.logger.info("=" * 60)
        self.logger.info(" AFDB EXTRACTION PIPELINE (23 PASIEN BERSIH)")
        self.logger.info("=" * 60)
        
        records_path = os.path.join(self.data_folder, self.records_file)
        with open(records_path, 'r') as f:
            all_records = [line.strip() for line in f if line.strip()]
            
        valid_records = [r for r in all_records if r not in self.blacklist]
        self.logger.info(f"Ditemukan {len(all_records)} rekaman. Setelah filter blacklist, sisa {len(valid_records)} rekaman.")
        
        all_windows = []
        
        for rec in valid_records:
            self.logger.info(f"── Memproses rekaman: {rec}")
            
            try:
                header = wfdb.rdheader(os.path.join(self.data_folder, rec))
                fs = header.fs
            except Exception as e:
                self.logger.warning(f"   [SKIP] Gagal baca header: {e}")
                continue
                
            path = os.path.join(self.data_folder, rec)
            r_peaks = None
            if rec == "07859":
                if os.path.exists(path + ".qrsc"):
                    r_peaks = wfdb.rdann(path, "qrsc").sample
                else:
                    self.logger.warning("   [SKIP] Rekaman 07859 wajib menggunakan .qrsc, tapi file tidak ditemukan.")
                    continue
            else:
                if os.path.exists(path + ".qrsc"):
                    r_peaks = wfdb.rdann(path, "qrsc").sample
                elif os.path.exists(path + ".qrs"):
                    r_peaks = wfdb.rdann(path, "qrs").sample
                else:
                    self.logger.warning("   [SKIP] Tidak ada file qrs/qrsc.")
                    continue
            
            try:
                atr = wfdb.rdann(path, "atr")
                pos_perubahan = atr.sample
                label_ritme = atr.aux_note
            except Exception as e:
                self.logger.warning(f"   [SKIP] Gagal baca anotasi ritme: {e}")
                continue
                
            if len(r_peaks) < 4:
                continue
                
            waktu_ibi, nilai_ibi, delta_ibi = self.hitung_ibi_delta(r_peaks, fs)
            
            windows = self.proses_windowing(rec, waktu_ibi, nilai_ibi, delta_ibi, r_peaks, pos_perubahan, label_ritme, fs)
            all_windows.extend(windows)
            
            # log stats per rekaman
            counts = {0:0, 1:0, 2:0}
            for w in windows:
                counts[w['label']] += 1
            self.logger.info(f"   → {len(windows)} windows valid (0: {counts[0]}, 1: {counts[1]}, 2: {counts[2]})")
            
        if not all_windows:
            self.logger.error("Tidak ada window yang dihasilkan!")
            return
            
        self.export_tabular(all_windows)
        self.export_sequence(all_windows)
        
        self.logger.info("=" * 60)
        self.logger.info(" RINGKASAN AKHIR")
        self.logger.info("=" * 60)
        self.logger.info(f"Total windows valid: {len(all_windows)}")
        y = np.array([w['label'] for w in all_windows])
        for c in [0, 1, 2]:
            self.logger.info(f"  Kelas {c}: {np.sum(y == c)}")
        self.logger.info("Selesai! ✓")

    def hitung_ibi_delta(self, r_peaks, fs):
        selisih = np.diff(r_peaks)
        ibi = selisih / fs * 1000.0
        waktu_ibi = r_peaks[1:] / fs
        delta_ibi = np.diff(ibi)
        return waktu_ibi.astype(np.float64), ibi, delta_ibi

    def tentukan_label(self, rp, pos, labels):
        idx = np.searchsorted(pos, rp, side='right') - 1
        if idx < 0: return None
        return self.label_map.get(labels[idx], None)

    def proses_windowing(self, rec, waktu_ibi, nilai_ibi, delta_ibi, r_peaks, pos_perubahan, label_ritme, fs):
        if len(waktu_ibi) < 3: return []
        
        waktu_akhir = waktu_ibi[-1]
        windows = []
        
        step = min(self.class_stride.values())
        awal = 0.0
        
        while awal + self.window_sec <= waktu_akhir:
            akhir = awal + self.window_sec
            
            mask = (waktu_ibi >= awal) & (waktu_ibi < akhir)
            idx_ibi = np.where(mask)[0]
            
            if len(idx_ibi) < 3:
                awal += step
                continue
                
            ibi_window = nilai_ibi[idx_ibi]
            
            delta_idx = [j for j in idx_ibi if (j+1) in idx_ibi and j < len(delta_ibi)]
            if not delta_idx:
                awal += step
                continue
            delta_window = delta_ibi[np.array(delta_idx)]
            
            outliers = (ibi_window < self.ibi_min) | (ibi_window > self.ibi_max)
            if np.sum(outliers) > self.max_outliers:
                awal += step
                continue
                
            rp_samples = r_peaks[idx_ibi + 1]
            labels_detak = [self.tentukan_label(rp, pos_perubahan, label_ritme) for rp in rp_samples]
            
            unique_lbl = set(labels_detak)
            if None in unique_lbl or len(unique_lbl) != 1:
                awal += step
                continue
                
            label_kelas = unique_lbl.pop()
            
            windows.append({
                "record": rec,
                "window_start": awal,
                "window_end": akhir,
                "ibi": ibi_window.copy(),
                "delta_ibi": delta_window.copy(),
                "label": label_kelas
            })
            
            awal += self.class_stride.get(label_kelas, step)
            
        return windows

    def export_tabular(self, windows):
        rows = []
        for w in windows:
            ibi = w['ibi']
            delta = w['delta_ibi']
            rows.append({
                "record": w['record'],
                "window_start": round(w['window_start'], 4),
                "window_end": round(w['window_end'], 4),
                "mean_ibi": round(float(np.mean(ibi)), 4),
                "min_ibi": round(float(np.min(ibi)), 4),
                "max_ibi": round(float(np.max(ibi)), 4),
                "range_ibi": round(float(np.max(ibi) - np.min(ibi)), 4),
                "var_delta_ibi": round(float(np.var(delta, ddof=1)), 4) if len(delta) > 1 else 0.0,
                "beat_count": len(ibi),
                "label": w['label']
            })
        
        df = pd.DataFrame(rows)
        os.makedirs(self.out_dir, exist_ok=True)
        out_path = os.path.join(self.out_dir, self.out_tabular)
        df.to_csv(out_path, index=False)
        self.logger.info(f"BRANCH A: Tersimpan di {out_path} ({len(df)} baris)")

    def export_sequence(self, windows):
        n = len(windows)
        X = np.zeros((n, self.max_seq_len, 2), dtype=np.float32)
        mask = np.zeros((n, self.max_seq_len), dtype=np.float32)
        y = np.zeros(n, dtype=np.int64)
        records = []
        
        for i, w in enumerate(windows):
            ibi = w['ibi']
            delta = w['delta_ibi']
            seq_len = min(len(ibi), self.max_seq_len)
            
            X[i, 0, 0] = ibi[0]
            X[i, 0, 1] = 0.0
            
            for j in range(1, seq_len):
                X[i, j, 0] = ibi[j]
                X[i, j, 1] = delta[j-1] if j-1 < len(delta) else 0.0
                
            mask[i, :seq_len] = 1.0
            y[i] = w['label']
            records.append(w['record'])
            
        os.makedirs(self.out_dir, exist_ok=True)
        out_path = os.path.join(self.out_dir, self.out_sequence)
        np.savez_compressed(
            out_path,
            X=X,
            mask=mask,
            y=y,
            records=np.array(records)
        )
        self.logger.info(f"BRANCH B: Tersimpan di {out_path}")


if __name__ == "__main__":
    extractor = AFDBExtractor()
    extractor.run_pipeline()
