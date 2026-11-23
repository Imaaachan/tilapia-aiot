# Tilapia AIoT — Early Warning Kelangsungan Hidup Ikan Nila dari Kualitas Air

Tugas Besar IoT — prediksi risiko kematian ikan tilapia (*Oreochromis niloticus*) dari data sensor kualitas air,
dengan **digital twin kolam** yang dikalibrasi ke dataset IoT Montería (Kolombia, 2024) dan model yang bisa dijalankan di perangkat edge.

## Kenapa digital twin?
Audit terhadap dataset publik *IoT Monitoring of Water Quality and Tilapia* (Mendeley DOI 10.17632/3g2b4sh65m.1, juga di Kaggle) menunjukkan:
- hanya **1.008 dari 4.383 baris (23%)** yang observasi asli; sisanya interpolasi linear;
- kolom suhu/DO/pH/turbiditas utama adalah **nilai bulanan** (91,8% baris konstan);
- `Health Status` identik dengan "suhu bulanan > 28 °C" (praktis = bulan Maret), `Low Oxygen Alert` selalu *Safe*;
- survival hanya memiliki **6 nilai** (satu per bulan).

Karena itu dataset dipakai sebagai **jangkar kalibrasi**, dan data latih/uji dibangkitkan oleh simulator berbasis fisiologi tilapia.

## Struktur repo
```
src/
  simulator.py   # digital twin: lingkungan → manajemen (aerator) → biologi (hazard) → sensor IoT
  features.py    # preprocessing edge + 46 fitur (buffer 72 jam) + label 6/12/24 jam
  modeling.py    # protokol evaluasi bebas leakage (split per kolam + waktu), baseline & model
notebooks/
  01_audit_digital_twin.ipynb   # audit dataset (unduh via kagglehub) + simulasi + validasi
  02_features_models.ipynb      # fitur, label, model, RQ2 (inflasi metrik), figur
scripts/                        # versi skrip dari audit/validasi + builder notebook
edge/                           # (tahap berikut) ekspor model ke C untuk ESP32 + firmware simulasi
dashboard/                      # (tahap berikut) dashboard web statis → GitHub Pages
outputs/                        # hasil (figur, tabel, JSON); dataset sintetis tidak di-commit
```

## Cara menjalankan
```bash
git clone https://github.com/Imaaachan/tilapia-aiot.git
cd tilapia-aiot
pip install -r requirements.txt
jupyter notebook notebooks/01_audit_digital_twin.ipynb   # lalu 02_features_models.ipynb
```
Di Google Colab: buka notebook dari GitHub (File → Open notebook → GitHub), sel pertama akan meng-*clone* repo otomatis.
Data Montería diunduh langsung lewat `kagglehub` (perlu token Kaggle bila diminta). Semua proses memakai *seed* tetap → hasil identik di mana pun.

## Ringkasan hasil (Bagian 2)
Lihat `outputs/table_main.csv`, `outputs/results_rq2.csv`, dan figur di `outputs/`.

## Sumber data & lisensi
Dataset Montería: Baena-Navarro, Carriazo-Regino, Torres-Hoyos (2024), Mendeley Data, CC BY 4.0.

### Hasil utama (split per kolam + waktu, 5 fold; mean ± sd)
| Horizon | Model | PR-AUC | Event recall | Median lead time |
|---|---|---|---|---|
| 6 jam | LightGBM | 0.817 ± 0.072 | 0.729 | 21 jam |
| 12 jam | LightGBM | 0.727 ± 0.031 | 0.795 | 18 jam |
| 24 jam | LightGBM | 0.541 ± 0.113 | 0.745 | 34.5 jam |
| 24 jam | Aturan DO_min24 | 0.252 ± 0.113 | 0.487 | — |

Inflasi PR-AUC LightGBM (H=24 j): split acak 0.855 → per kolam 0.668 → per kolam + waktu 0.541 (+58% jika split acak).
