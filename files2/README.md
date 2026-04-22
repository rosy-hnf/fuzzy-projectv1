# 🧠 Sistem Deteksi Tingkat Stress — Fuzzy Logic Mamdani

Aplikasi Python untuk mendeteksi tingkat stress berdasarkan data **fisiologis** dan **psikologis** menggunakan metode **Logika Fuzzy Mamdani**.

---

## 📋 Fitur

- **6 Variabel Input:**
  - 🫀 Detak Jantung (BPM)
  - 🌡️ Suhu Tubuh (°C)
  - 👣 Aktivitas Fisik (Jumlah Langkah)
  - 😴 Kualitas Tidur (Jam)
  - 💼 Beban Kerja (Skala 1–10)
  - 📱 Screen Time (Jam/hari)

- **28 Aturan Fuzzy** dalam 4 kelompok logika
- **Output:** Skor 1–100 dengan 3 kategori:
  - 🟢 **Rendah** (1–35): Kondisi baik
  - 🟡 **Sedang** (36–65): Perlu waspada
  - 🔴 **Tinggi** (66–100): Segera tangani

---

## 🚀 Cara Menjalankan

### 1. Clone repository ini
```bash
git clone https://github.com/USERNAME/fuzzy-stress-detection.git
cd fuzzy-stress-detection
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Jalankan program
```bash
python fuzzy_stress_detection.py
```

---

## 📦 Struktur Project

```
fuzzy-stress-detection/
├── fuzzy_stress_detection.py   # Program utama
├── requirements.txt            # Library yang dibutuhkan
└── README.md                   # Dokumentasi ini
```

---

## 🔬 Cara Kerja (Pipeline Mamdani)

```
INPUT (6 variabel)
      ↓
[1] FUZZIFIKASI       → Angka → Derajat keanggotaan (0.0–1.0)
      ↓
[2] EVALUASI ATURAN   → 28 aturan IF-THEN
      ↓
[3] AGREGASI          → Gabungkan hasil aturan (operator MAX)
      ↓
[4] DEFUZZIFIKASI     → Metode Centroid → Skor 1–100
      ↓
OUTPUT: Skor + Kategori Stress
```

---

## 📊 Contoh Hasil

| Kondisi | BPM | Tidur | Beban | Skor | Kategori |
|---------|-----|-------|-------|------|----------|
| Ideal   | 65  | 8 jam | 3/10  | 18.5 | 🟢 Rendah |
| Umum    | 82  | 6 jam | 6/10  | 41.3 | 🟡 Sedang |
| Kritis  | 105 | 4.5 jam | 9/10 | 80.6 | 🔴 Tinggi |

---

## 🛠️ Teknologi

- Python 3.8+
- NumPy
- Matplotlib

---

## 📝 Lisensi

Project ini dibuat untuk tujuan edukasi dan pembelajaran Logika Fuzzy.
