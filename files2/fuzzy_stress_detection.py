"""
============================================================
APLIKASI DETEKSI TINGKAT STRESS MENGGUNAKAN LOGIKA FUZZY
Metode: Mamdani
Dibuat untuk: Pemula yang belajar Fuzzy Logic
============================================================

APA ITU LOGIKA FUZZY?
---------------------
Logika fuzzy adalah cara komputer berpikir seperti manusia.
Manusia biasa berkata "suhu AGAK panas" atau "detak jantung SANGAT cepat"
Logika fuzzy meniru cara berpikir seperti itu menggunakan nilai 0.0 sampai 1.0

ALUR KERJA (PIPELINE) FUZZY MAMDANI:
1. FUZZIFIKASI    -> Ubah angka menjadi derajat keanggotaan (0.0 - 1.0)
2. INFERENSI      -> Terapkan aturan IF-THEN (rule evaluation)
3. AGREGASI       -> Gabungkan semua hasil aturan
4. DEFUZZIFIKASI  -> Ubah kembali menjadi angka (skor 1-100)

VARIABEL INPUT (6 variabel):
- Fisiologis: BPM (detak jantung), Suhu Tubuh, Langkah Kaki
- Psikologis: Kualitas Tidur, Beban Kerja, Screen Time

OUTPUT:
- Skor Stress: 1-100
- Kategori: Rendah / Sedang / Tinggi
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# BAGIAN 1: FUNGSI KEANGGOTAAN (MEMBERSHIP FUNCTIONS)
# ============================================================
"""
Fungsi keanggotaan adalah "jembatan" antara nilai nyata dan logika fuzzy.
Ada beberapa bentuk fungsi keanggotaan:
- Trapezoid (trapesium): bentuk seperti trapesium, cocok untuk nilai "normal"
- Triangle (segitiga): bentuk segitiga, cocok untuk nilai tengah
- Shoulder (bahu): trapesium yang melebar ke ujung, cocok untuk nilai ekstrem

Nilai yang dikembalikan adalah DERAJAT KEANGGOTAAN (0.0 - 1.0)
0.0 = sama sekali tidak termasuk kategori ini
1.0 = 100% termasuk kategori ini
0.5 = 50% termasuk kategori ini
"""

def trapezoid(x, a, b, c, d):
    """
    Fungsi keanggotaan bentuk TRAPESIUM
    
    Cara membaca: nilai x mulai masuk di titik 'a', 
    penuh (1.0) antara 'b' dan 'c', lalu turun sampai 'd'
    
    Grafik:
            1.0 |    ____
                |   /    \\
                |  /      \\
            0.0 |_/________\\___
                a  b      c  d
    
    Parameter:
        x : nilai yang akan dihitung derajatnya
        a : titik mulai naik
        b : titik puncak kiri (mulai nilai 1.0)
        c : titik puncak kanan (akhir nilai 1.0)  
        d : titik selesai turun
    """
    if x <= a or x >= d:
        return 0.0
    elif b <= x <= c:
        return 1.0
    elif a < x < b:
        return (x - a) / (b - a)
    else:  # c < x < d
        return (d - x) / (d - c)

def triangle(x, a, b, c):
    """
    Fungsi keanggotaan bentuk SEGITIGA
    
    Cara membaca: nilai x mulai masuk di titik 'a',
    puncak di titik 'b' (nilai 1.0), lalu turun sampai 'c'
    
    Grafik:
            1.0 |    /\\
                |   /  \\
                |  /    \\
            0.0 |_/______\\___
                a    b    c
    
    Parameter:
        x : nilai yang akan dihitung derajatnya
        a : titik mulai naik (kiri)
        b : titik puncak (nilai 1.0)
        c : titik selesai turun (kanan)
    """
    if x <= a or x >= c:
        return 0.0
    elif x == b:
        return 1.0
    elif a < x < b:
        return (x - a) / (b - a)
    else:  # b < x < c
        return (c - x) / (c - b)

def shoulder_left(x, a, b):
    """
    Fungsi keanggotaan BAHU KIRI (untuk nilai sangat rendah)
    Nilai 1.0 untuk semua x <= a, lalu turun sampai b
    
    Grafik:
            1.0 |_____
                |     \\
                |      \\
            0.0 |_______\\___
                a        b
    """
    if x <= a:
        return 1.0
    elif x >= b:
        return 0.0
    else:
        return (b - x) / (b - a)

def shoulder_right(x, a, b):
    """
    Fungsi keanggotaan BAHU KANAN (untuk nilai sangat tinggi)
    Mulai naik dari a sampai b, lalu 1.0 untuk semua x >= b
    
    Grafik:
            1.0 |         _____
                |        /
                |       /
            0.0 |______/________
                a      b
    """
    if x >= b:
        return 1.0
    elif x <= a:
        return 0.0
    else:
        return (x - a) / (b - a)

# ============================================================
# BAGIAN 2: FUZZIFIKASI (FUZZIFICATION)
# ============================================================
"""
FUZZIFIKASI adalah proses mengubah nilai input (angka biasa) 
menjadi nilai fuzzy (derajat keanggotaan 0.0 - 1.0) 
untuk setiap kategori (himpunan fuzzy) yang telah didefinisikan.

Contoh:
- BPM = 85 -> rendah: 0.0, normal: 0.5, tinggi: 0.3, sangat_tinggi: 0.0
- Artinya BPM 85 "setengah normal" dan "sedikit tinggi"
"""

def fuzzify_bpm(bpm):
    """
    Fuzzifikasi DETAK JANTUNG (BPM - Beats Per Minute)
    
    Rentang normal manusia: 60-100 BPM
    - Rendah  : < 60 BPM (bradycardia - terlalu lambat)
    - Normal  : 60-80 BPM (ideal saat istirahat)
    - Tinggi  : 80-100 BPM (mulai stress)
    - Sangat Tinggi: > 100 BPM (tachycardia - stress tinggi)
    
    Mengapa BPM penting untuk deteksi stress?
    Stress mengaktifkan sistem saraf simpatis yang meningkatkan BPM.
    """
    return {
        'rendah':       shoulder_left(bpm, 55, 65),
        'normal':       trapezoid(bpm, 55, 65, 75, 85),
        'tinggi':       triangle(bpm, 75, 90, 105),
        'sangat_tinggi': shoulder_right(bpm, 95, 110)
    }

def fuzzify_suhu(suhu):
    """
    Fuzzifikasi SUHU TUBUH (Celsius)
    
    Suhu tubuh normal manusia: 36.1 - 37.2 °C
    - Rendah  : < 36.0°C (hipotermi ringan)
    - Normal  : 36.0 - 37.0°C (kondisi baik)
    - Tinggi  : 37.0 - 38.0°C (subfebris, mulai stress)
    - Sangat Tinggi: > 38.0°C (demam/hipertermi dari stress)
    
    Mengapa suhu penting?
    Stress dan kecemasan dapat meningkatkan suhu tubuh.
    """
    return {
        'rendah':       shoulder_left(suhu, 35.5, 36.2),
        'normal':       trapezoid(suhu, 35.8, 36.3, 37.0, 37.5),
        'tinggi':       triangle(suhu, 37.0, 37.5, 38.2),
        'sangat_tinggi': shoulder_right(suhu, 37.8, 38.5)
    }

def fuzzify_langkah(langkah):
    """
    Fuzzifikasi LANGKAH KAKI (jumlah langkah per hari)
    
    Rekomendasi WHO: 8000-10000 langkah/hari
    - Sedikit  : < 3000 langkah (sangat sedentary, berisiko stress)
    - Sedang   : 3000-8000 langkah (cukup aktif)
    - Banyak   : > 8000 langkah (aktif, baik untuk mental health)
    
    Mengapa langkah penting?
    Aktivitas fisik yang rendah berkorelasi dengan stress tinggi.
    Olahraga melepas endorfin yang melawan stress.
    """
    return {
        'sedikit': shoulder_left(langkah, 2000, 4000),
        'sedang':  trapezoid(langkah, 3000, 5000, 8000, 10000),
        'banyak':  shoulder_right(langkah, 8000, 12000)
    }

def fuzzify_tidur(tidur):
    """
    Fuzzifikasi KUALITAS TIDUR (jam per malam)
    
    Rekomendasi: 7-9 jam untuk orang dewasa
    - Buruk    : < 5 jam (kurang tidur parah)
    - Cukup    : 5-7 jam (kurang ideal)
    - Baik     : 7-9 jam (ideal)
    - Berlebih : > 9 jam (bisa indikasi depresi)
    
    Mengapa tidur penting?
    Kurang tidur adalah penyebab DAN akibat stress.
    Tidur adalah waktu otak memproses emosi dan stress.
    """
    return {
        'buruk':    shoulder_left(tidur, 4.0, 5.5),
        'cukup':    triangle(tidur, 5.0, 6.0, 7.5),
        'baik':     trapezoid(tidur, 7.0, 7.5, 8.5, 9.5),
        'berlebih': shoulder_right(tidur, 9.0, 10.5)
    }

def fuzzify_beban(beban):
    """
    Fuzzifikasi BEBAN KERJA (skala 1-10, diisi pengguna)
    
    Skala persepsi subjektif:
    - Ringan   : 1-3 (pekerjaan mudah, tidak menekan)
    - Sedang   : 4-6 (wajar, bisa dikelola)
    - Berat    : 7-8 (banyak tekanan)
    - Sangat Berat: 9-10 (overwhelmed, burnout risk)
    
    Ini adalah faktor PSIKOLOGIS utama pemicu stress kerja.
    """
    return {
        'ringan':      shoulder_left(beban, 2.0, 4.0),
        'sedang':      trapezoid(beban, 3.0, 4.5, 6.0, 7.5),
        'berat':       triangle(beban, 6.5, 8.0, 9.5),
        'sangat_berat': shoulder_right(beban, 8.5, 10.0)
    }

def fuzzify_screen(screen):
    """
    Fuzzifikasi SCREEN TIME (jam per hari)
    
    WHO merekomendasikan batasan screen time sehat:
    - Sedikit  : < 2 jam (sangat sehat)
    - Sedang   : 2-6 jam (bisa ditoleransi)
    - Banyak   : 6-10 jam (mulai berbahaya)
    - Berlebihan: > 10 jam (sangat berbahaya)
    
    Mengapa screen time penting?
    Screen time berlebih -> blue light -> gangguan tidur -> stress
    Media sosial -> anxiety -> stress
    """
    return {
        'sedikit':    shoulder_left(screen, 1.5, 3.0),
        'sedang':     trapezoid(screen, 2.0, 4.0, 6.0, 8.0),
        'banyak':     triangle(screen, 6.0, 8.0, 11.0),
        'berlebihan': shoulder_right(screen, 10.0, 14.0)
    }

# ============================================================
# BAGIAN 3: ATURAN FUZZY (FUZZY RULES) - METODE MAMDANI
# ============================================================
"""
ATURAN FUZZY adalah "pengetahuan ahli" yang dikodekan dalam format IF-THEN.
Format: IF (kondisi1) AND/OR (kondisi2) THEN (output)

METODE MAMDANI menggunakan:
- Operator AND = MIN (ambil nilai minimum/terkecil)
- Operator OR  = MAX (ambil nilai maximum/terbesar)
- Implikasi    = MIN (potong output sesuai kekuatan aturan)

Contoh Aturan:
Rule 1: IF bpm TINGGI AND tidur BURUK THEN stress TINGGI
        -> min(0.7, 0.8) = 0.7 (kekuatan aturan = 0.7)
        -> Output "tinggi" dipotong di 0.7

Kami mendefinisikan 4 kelompok logika utama:
1. Aturan berbasis BPM + Suhu (fisiologis kardiovaskular)
2. Aturan berbasis Langkah + Tidur (gaya hidup sehat)
3. Aturan berbasis Beban Kerja + Screen Time (faktor psikologis)
4. Aturan kombinasi semua faktor (holistic assessment)
"""

def apply_rules(f_bpm, f_suhu, f_langkah, f_tidur, f_beban, f_screen):
    """
    Menerapkan semua aturan fuzzy dan mengumpulkan kekuatan tiap aturan.
    
    Return:
        dict dengan 3 kategori output: 'rendah', 'sedang', 'tinggi'
        Setiap kategori berisi LIST kekuatan aturan yang mengaktivasinya
    """
    
    # Inisialisasi: setiap output dimulai dari daftar kosong
    output_rendah = []   # Kumpulan kekuatan aturan untuk stress RENDAH
    output_sedang = []   # Kumpulan kekuatan aturan untuk stress SEDANG
    output_tinggi = []   # Kumpulan kekuatan aturan untuk stress TINGGI
    
    # ----------------------------------------------------------
    # KELOMPOK ATURAN 1: BPM + SUHU (Fisiologis Kardiovaskular)
    # Logika: Kondisi jantung dan suhu mencerminkan respons tubuh terhadap stress
    # ----------------------------------------------------------
    
    # Jika BPM normal DAN suhu normal -> stress rendah (tubuh tenang)
    r1 = min(f_bpm['normal'], f_suhu['normal'])
    output_rendah.append(r1)
    
    # Jika BPM tinggi DAN suhu tinggi -> stress sedang (tubuh mulai bereaksi)
    r2 = min(f_bpm['tinggi'], f_suhu['tinggi'])
    output_sedang.append(r2)
    
    # Jika BPM sangat tinggi DAN suhu tinggi -> stress tinggi
    r3 = min(f_bpm['sangat_tinggi'], f_suhu['tinggi'])
    output_tinggi.append(r3)
    
    # Jika BPM sangat tinggi DAN suhu sangat tinggi -> stress tinggi (parah)
    r4 = min(f_bpm['sangat_tinggi'], f_suhu['sangat_tinggi'])
    output_tinggi.append(r4)
    
    # Jika BPM rendah (rileks) DAN suhu normal -> stress rendah
    r5 = min(f_bpm['rendah'], f_suhu['normal'])
    output_rendah.append(r5)
    
    # Jika BPM tinggi tapi suhu normal -> stress sedang (mungkin habis olahraga)
    r6 = min(f_bpm['tinggi'], f_suhu['normal'])
    output_sedang.append(r6)
    
    # ----------------------------------------------------------
    # KELOMPOK ATURAN 2: LANGKAH + TIDUR (Gaya Hidup Sehat)
    # Logika: Orang yang tidur cukup dan aktif bergerak lebih tahan stress
    # ----------------------------------------------------------
    
    # Jika langkah banyak DAN tidur baik -> stress rendah (gaya hidup sehat)
    r7 = min(f_langkah['banyak'], f_tidur['baik'])
    output_rendah.append(r7)
    
    # Jika langkah sedikit DAN tidur buruk -> stress tinggi (gaya hidup buruk)
    r8 = min(f_langkah['sedikit'], f_tidur['buruk'])
    output_tinggi.append(r8)
    
    # Jika langkah sedang DAN tidur cukup -> stress sedang (rata-rata)
    r9 = min(f_langkah['sedang'], f_tidur['cukup'])
    output_sedang.append(r9)
    
    # Jika langkah banyak DAN tidur cukup -> stress rendah (aktif tapi kurang tidur sedikit)
    r10 = min(f_langkah['banyak'], f_tidur['cukup'])
    output_rendah.append(r10)
    
    # Jika langkah sedikit DAN tidur baik -> stress sedang (tidak gerak tapi tidur oke)
    r11 = min(f_langkah['sedikit'], f_tidur['baik'])
    output_sedang.append(r11)
    
    # Jika tidur berlebih DAN langkah sedikit -> stress sedang (mungkin depresi ringan)
    r12 = min(f_tidur['berlebih'], f_langkah['sedikit'])
    output_sedang.append(r12)
    
    # Jika tidur buruk DAN langkah sedang -> stress sedang
    r13 = min(f_tidur['buruk'], f_langkah['sedang'])
    output_sedang.append(r13)
    
    # ----------------------------------------------------------
    # KELOMPOK ATURAN 3: BEBAN KERJA + SCREEN TIME (Psikologis)
    # Logika: Tekanan mental dan paparan layar berlebih adalah pemicu stress utama
    # ----------------------------------------------------------
    
    # Jika beban ringan DAN screen sedikit -> stress rendah (santai)
    r14 = min(f_beban['ringan'], f_screen['sedikit'])
    output_rendah.append(r14)
    
    # Jika beban sangat berat DAN screen berlebihan -> stress tinggi (burnout)
    r15 = min(f_beban['sangat_berat'], f_screen['berlebihan'])
    output_tinggi.append(r15)
    
    # Jika beban sedang DAN screen sedang -> stress sedang (normal)
    r16 = min(f_beban['sedang'], f_screen['sedang'])
    output_sedang.append(r16)
    
    # Jika beban berat DAN screen banyak -> stress tinggi
    r17 = min(f_beban['berat'], f_screen['banyak'])
    output_tinggi.append(r17)
    
    # Jika beban berat DAN screen sedang -> stress sedang
    r18 = min(f_beban['berat'], f_screen['sedang'])
    output_sedang.append(r18)
    
    # Jika beban sangat berat DAN screen banyak -> stress tinggi
    r19 = min(f_beban['sangat_berat'], f_screen['banyak'])
    output_tinggi.append(r19)
    
    # Jika beban ringan DAN screen berlebihan -> stress sedang (sehat tapi terlalu banyak screen)
    r20 = min(f_beban['ringan'], f_screen['berlebihan'])
    output_sedang.append(r20)
    
    # ----------------------------------------------------------
    # KELOMPOK ATURAN 4: KOMBINASI HOLISTIK
    # Logika: Mempertimbangkan kombinasi faktor fisiologis DAN psikologis bersama
    # ----------------------------------------------------------
    
    # Jika BPM sangat tinggi DAN beban sangat berat -> stress tinggi (parah)
    r21 = min(f_bpm['sangat_tinggi'], f_beban['sangat_berat'])
    output_tinggi.append(r21)
    
    # Jika BPM normal DAN beban ringan DAN tidur baik -> stress rendah (sempurna)
    r22 = min(f_bpm['normal'], f_beban['ringan'], f_tidur['baik'])
    output_rendah.append(r22)
    
    # Jika tidur buruk DAN beban berat -> stress tinggi (kelelahan + tekanan)
    r23 = min(f_tidur['buruk'], f_beban['berat'])
    output_tinggi.append(r23)
    
    # Jika tidur buruk DAN beban sangat berat DAN screen berlebihan -> stress tinggi (triple threat)
    r24 = min(f_tidur['buruk'], f_beban['sangat_berat'], f_screen['berlebihan'])
    output_tinggi.append(r24)
    
    # Jika BPM tinggi DAN tidur buruk -> stress tinggi (jantung kencang + kurang tidur)
    r25 = min(f_bpm['tinggi'], f_tidur['buruk'])
    output_tinggi.append(r25)
    
    # Jika langkah banyak DAN beban ringan DAN screen sedikit -> stress rendah (ideal)
    r26 = min(f_langkah['banyak'], f_beban['ringan'], f_screen['sedikit'])
    output_rendah.append(r26)
    
    # Jika suhu tinggi DAN beban berat -> stress sedang-tinggi
    r27 = min(f_suhu['tinggi'], f_beban['berat'])
    output_tinggi.append(r27)
    
    # Jika semua fisiologis normal DAN beban sedang -> stress rendah-sedang
    r28 = min(f_bpm['normal'], f_suhu['normal'], f_langkah['sedang'])
    output_rendah.append(r28)
    
    return {
        'rendah': output_rendah,
        'sedang': output_sedang,
        'tinggi': output_tinggi
    }

# ============================================================
# BAGIAN 4: AGREGASI (AGGREGATION)
# ============================================================
"""
AGREGASI adalah menggabungkan semua kekuatan aturan untuk setiap kategori output.
Kita mengambil nilai MAKSIMUM dari semua aturan yang mengaktifkan kategori yang sama.

Contoh:
- Stress Tinggi diaktifkan oleh rule3=0.7, rule8=0.6, rule15=0.9
- Hasil agregasi = max(0.7, 0.6, 0.9) = 0.9

Ini membentuk "area" yang akan digunakan untuk defuzzifikasi.
"""

def aggregate(rule_outputs):
    """
    Mengagregasi output dari semua aturan fuzzy.
    
    Parameter:
        rule_outputs: dict dari apply_rules()
    
    Return:
        dict berisi nilai agregasi maksimum untuk tiap kategori
    """
    return {
        'rendah': max(rule_outputs['rendah']) if rule_outputs['rendah'] else 0.0,
        'sedang': max(rule_outputs['sedang']) if rule_outputs['sedang'] else 0.0,
        'tinggi': max(rule_outputs['tinggi']) if rule_outputs['tinggi'] else 0.0
    }

# ============================================================
# BAGIAN 5: DEFUZZIFIKASI (DEFUZZIFICATION) - Metode Centroid
# ============================================================
"""
DEFUZZIFIKASI adalah mengubah nilai fuzzy kembali menjadi angka nyata (crisp value).

METODE CENTROID (Center of Area / Center of Gravity):
- Membuat kurva output gabungan dari semua kategori yang sudah terpotong
- Menghitung titik berat (centroid) dari kurva gabungan tersebut
- Titik berat itulah output akhir (skor stress)

Rumus Centroid:
           Σ(x * μ(x))
z* = ──────────────────
           Σ(μ(x))

Di mana:
- x  = titik pada sumbu output (1-100)
- μ(x) = derajat keanggotaan di titik x (sudah diagregasi)
- z* = nilai crisp hasil defuzzifikasi

Kita mendefinisikan output fuzzy dalam skala 1-100:
- Rendah : 1-40
- Sedang : 30-70
- Tinggi : 60-100
"""

def defuzzify(aggregated):
    """
    Defuzzifikasi menggunakan metode Centroid.
    
    Parameter:
        aggregated: dict dari aggregate() berisi nilai max tiap kategori
    
    Return:
        float: skor stress antara 1-100
    """
    # Rentang output: 1 sampai 100 dengan 1000 titik (presisi tinggi)
    x = np.linspace(1, 100, 1000)
    
    # Definisikan fungsi keanggotaan untuk OUTPUT (skor stress 1-100)
    # Rendah: puncak di tengah kiri (skala 1-40)
    mu_rendah = np.array([trapezoid(xi, 1, 1, 25, 45) for xi in x])
    # Sedang: puncak di tengah (skala 30-70)
    mu_sedang = np.array([triangle(xi, 25, 50, 75) for xi in x])
    # Tinggi: puncak di kanan (skala 60-100)
    mu_tinggi = np.array([trapezoid(xi, 55, 75, 100, 100) for xi in x])
    
    # PEMOTONGAN (Clipping): Potong setiap kurva output sesuai kekuatan agregasi
    # Ini adalah ciri khas Metode Mamdani
    clipped_rendah = np.minimum(mu_rendah, aggregated['rendah'])
    clipped_sedang = np.minimum(mu_sedang, aggregated['sedang'])
    clipped_tinggi = np.minimum(mu_tinggi, aggregated['tinggi'])
    
    # GABUNGKAN: Ambil maximum dari semua kurva yang sudah dipotong
    # Ini membentuk satu kurva gabungan
    combined = np.maximum(np.maximum(clipped_rendah, clipped_sedang), clipped_tinggi)
    
    # CENTROID: Hitung titik berat dari kurva gabungan
    numerator = np.sum(x * combined)    # Σ(x * μ(x))
    denominator = np.sum(combined)       # Σ(μ(x))
    
    if denominator == 0:
        return 50.0  # Nilai default jika tidak ada aturan yang aktif
    
    skor = numerator / denominator
    return float(np.clip(skor, 1, 100))  # Pastikan dalam rentang 1-100

# ============================================================
# BAGIAN 6: FUNGSI UTAMA - SISTEM FUZZY LENGKAP
# ============================================================

def hitung_stress(bpm, suhu, langkah, tidur, beban, screen, verbose=True):
    """
    Fungsi utama yang menjalankan seluruh pipeline fuzzy.
    
    Parameter:
        bpm     : Detak jantung (60-140 BPM)
        suhu    : Suhu tubuh (35.0-39.0 °C)
        langkah : Jumlah langkah per hari (0-20000)
        tidur   : Jam tidur per malam (0-12)
        beban   : Beban kerja skala 1-10
        screen  : Screen time per hari (0-16 jam)
        verbose : Tampilkan detail perhitungan (True/False)
    
    Return:
        tuple: (skor, kategori, detail_fuzzifikasi, detail_agregasi)
    """
    
    if verbose:
        print("\n" + "="*60)
        print("  SISTEM DETEKSI STRESS - FUZZY LOGIC (MAMDANI)")
        print("="*60)
        print("\n📊 INPUT YANG DITERIMA:")
        print(f"   🫀 Detak Jantung  : {bpm} BPM")
        print(f"   🌡️  Suhu Tubuh    : {suhu} °C")
        print(f"   👣 Langkah Kaki  : {langkah:,} langkah")
        print(f"   😴 Jam Tidur     : {tidur} jam")
        print(f"   💼 Beban Kerja   : {beban}/10")
        print(f"   📱 Screen Time   : {screen} jam/hari")
    
    # ---- LANGKAH 1: FUZZIFIKASI ----
    if verbose:
        print("\n" + "-"*60)
        print("LANGKAH 1: FUZZIFIKASI")
        print("(Mengubah angka biasa menjadi derajat keanggotaan 0.0-1.0)")
        print("-"*60)
    
    f_bpm    = fuzzify_bpm(bpm)
    f_suhu   = fuzzify_suhu(suhu)
    f_langkah = fuzzify_langkah(langkah)
    f_tidur  = fuzzify_tidur(tidur)
    f_beban  = fuzzify_beban(beban)
    f_screen = fuzzify_screen(screen)
    
    if verbose:
        print(f"\n  🫀 BPM ({bpm}):")
        for k, v in f_bpm.items():
            bar = "█" * int(v * 20)
            print(f"     {k:15s}: {v:.3f} |{bar}")
        
        print(f"\n  🌡️  Suhu ({suhu}°C):")
        for k, v in f_suhu.items():
            bar = "█" * int(v * 20)
            print(f"     {k:15s}: {v:.3f} |{bar}")
        
        print(f"\n  👣 Langkah ({langkah:,}):")
        for k, v in f_langkah.items():
            bar = "█" * int(v * 20)
            print(f"     {k:15s}: {v:.3f} |{bar}")
        
        print(f"\n  😴 Tidur ({tidur} jam):")
        for k, v in f_tidur.items():
            bar = "█" * int(v * 20)
            print(f"     {k:15s}: {v:.3f} |{bar}")
        
        print(f"\n  💼 Beban ({beban}/10):")
        for k, v in f_beban.items():
            bar = "█" * int(v * 20)
            print(f"     {k:15s}: {v:.3f} |{bar}")
        
        print(f"\n  📱 Screen Time ({screen} jam):")
        for k, v in f_screen.items():
            bar = "█" * int(v * 20)
            print(f"     {k:15s}: {v:.3f} |{bar}")
    
    # ---- LANGKAH 2: EVALUASI ATURAN ----
    if verbose:
        print("\n" + "-"*60)
        print("LANGKAH 2: EVALUASI ATURAN FUZZY (28 Aturan)")
        print("(Menghitung kekuatan setiap aturan IF-THEN)")
        print("-"*60)
    
    rule_outputs = apply_rules(f_bpm, f_suhu, f_langkah, f_tidur, f_beban, f_screen)
    
    # ---- LANGKAH 3: AGREGASI ----
    if verbose:
        print("\n" + "-"*60)
        print("LANGKAH 3: AGREGASI")
        print("(Mengambil nilai MAX dari semua aturan per kategori)")
        print("-"*60)
    
    aggregated = aggregate(rule_outputs)
    
    if verbose:
        print(f"\n  Hasil Agregasi:")
        for cat, val in aggregated.items():
            bar = "█" * int(val * 30)
            emoji = "🟢" if cat=="rendah" else "🟡" if cat=="sedang" else "🔴"
            print(f"  {emoji} Stress {cat:8s}: {val:.4f} |{bar}")
    
    # ---- LANGKAH 4: DEFUZZIFIKASI ----
    if verbose:
        print("\n" + "-"*60)
        print("LANGKAH 4: DEFUZZIFIKASI (Metode Centroid)")
        print("(Mengubah nilai fuzzy kembali menjadi skor 1-100)")
        print("-"*60)
    
    skor = defuzzify(aggregated)
    
    # Tentukan kategori berdasarkan skor
    if skor <= 35:
        kategori = "RENDAH"
        emoji = "🟢"
        pesan = "Anda dalam kondisi baik! Pertahankan gaya hidup sehat."
    elif skor <= 65:
        kategori = "SEDANG"
        emoji = "🟡"
        pesan = "Waspadai tanda-tanda stress. Luangkan waktu untuk relaksasi."
    else:
        kategori = "TINGGI"
        emoji = "🔴"
        pesan = "Tingkat stress tinggi! Segera lakukan manajemen stress."
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"  HASIL AKHIR:")
        print(f"  {emoji} Skor Stress  : {skor:.1f} / 100")
        print(f"  {emoji} Kategori     : {kategori}")
        print(f"  💡 Saran         : {pesan}")
        print(f"{'='*60}")
    
    detail_fuzz = {
        'bpm': f_bpm, 'suhu': f_suhu, 'langkah': f_langkah,
        'tidur': f_tidur, 'beban': f_beban, 'screen': f_screen
    }
    
    return skor, kategori, detail_fuzz, aggregated

# ============================================================
# BAGIAN 7: VISUALISASI
# ============================================================

def plot_membership_functions():
    """
    Menggambar semua fungsi keanggotaan untuk semua variabel input dan output.
    Berguna untuk memahami bagaimana nilai input dipetakan ke fuzzy set.
    """
    fig = plt.figure(figsize=(18, 14))
    fig.patch.set_facecolor('#0f0f1a')
    gs = GridSpec(3, 3, figure=fig, hspace=0.5, wspace=0.4)
    
    titles = ['Detak Jantung (BPM)', 'Suhu Tubuh (°C)', 'Langkah Kaki',
              'Kualitas Tidur (Jam)', 'Beban Kerja (1-10)', 'Screen Time (Jam)',
              'OUTPUT: Skor Stress']
    
    NEON_COLORS = ['#00f5ff', '#ff00e4', '#00ff88', '#ffaa00']
    BG = '#1a1a2e'
    TEXT = '#e0e0ff'
    GRID = '#2a2a4a'
    
    axes_configs = [
        # (x_range, labels, funcs_data)
        (np.linspace(40, 150, 500), ['Rendah', 'Normal', 'Tinggi', 'Sangat Tinggi'],
         lambda x: [shoulder_left(x,55,65), trapezoid(x,55,65,75,85),
                    triangle(x,75,90,105), shoulder_right(x,95,110)]),
        
        (np.linspace(34, 40, 500), ['Rendah', 'Normal', 'Tinggi', 'Sangat Tinggi'],
         lambda x: [shoulder_left(x,35.5,36.2), trapezoid(x,35.8,36.3,37.0,37.5),
                    triangle(x,37.0,37.5,38.2), shoulder_right(x,37.8,38.5)]),
        
        (np.linspace(0, 16000, 500), ['Sedikit', 'Sedang', 'Banyak'],
         lambda x: [shoulder_left(x,2000,4000), trapezoid(x,3000,5000,8000,10000),
                    shoulder_right(x,8000,12000)]),
        
        (np.linspace(0, 12, 500), ['Buruk', 'Cukup', 'Baik', 'Berlebih'],
         lambda x: [shoulder_left(x,4.0,5.5), triangle(x,5.0,6.0,7.5),
                    trapezoid(x,7.0,7.5,8.5,9.5), shoulder_right(x,9.0,10.5)]),
        
        (np.linspace(0, 10, 500), ['Ringan', 'Sedang', 'Berat', 'Sangat Berat'],
         lambda x: [shoulder_left(x,2.0,4.0), trapezoid(x,3.0,4.5,6.0,7.5),
                    triangle(x,6.5,8.0,9.5), shoulder_right(x,8.5,10.0)]),
        
        (np.linspace(0, 16, 500), ['Sedikit', 'Sedang', 'Banyak', 'Berlebihan'],
         lambda x: [shoulder_left(x,1.5,3.0), trapezoid(x,2.0,4.0,6.0,8.0),
                    triangle(x,6.0,8.0,11.0), shoulder_right(x,10.0,14.0)]),
        
        (np.linspace(1, 100, 500), ['Rendah', 'Sedang', 'Tinggi'],
         lambda x: [trapezoid(x,1,1,25,45), triangle(x,25,50,75), trapezoid(x,55,75,100,100)]),
    ]
    
    positions = [(0,0),(0,1),(0,2),(1,0),(1,1),(1,2),(2,1)]
    
    for i, (pos, title, config) in enumerate(zip(positions, titles, axes_configs)):
        ax = fig.add_subplot(gs[pos[0], pos[1]])
        ax.set_facecolor(BG)
        ax.grid(True, color=GRID, linewidth=0.5, alpha=0.7)
        
        x_range, labels, func = config
        num_sets = len(labels)
        colors = NEON_COLORS[:num_sets]
        
        for j, (label, color) in enumerate(zip(labels, colors)):
            y = np.array([func(xi)[j] for xi in x_range])
            ax.plot(x_range, y, color=color, linewidth=2.5, label=label)
            ax.fill_between(x_range, y, alpha=0.15, color=color)
        
        ax.set_title(title, color=TEXT, fontsize=9, fontweight='bold', pad=8)
        ax.set_xlabel('Nilai', color=TEXT, fontsize=7)
        ax.set_ylabel('Derajat Keanggotaan', color=TEXT, fontsize=7)
        ax.tick_params(colors=TEXT, labelsize=7)
        ax.set_ylim(-0.05, 1.15)
        ax.spines['bottom'].set_color(GRID)
        ax.spines['top'].set_color(GRID)
        ax.spines['left'].set_color(GRID)
        ax.spines['right'].set_color(GRID)
        
        legend = ax.legend(fontsize=6, loc='upper right',
                          facecolor='#0f0f1a', edgecolor=GRID,
                          labelcolor=TEXT)
    
    # Kosongkan subplot yang tidak dipakai
    for pos in [(2,0),(2,2)]:
        ax = fig.add_subplot(gs[pos[0], pos[1]])
        ax.set_visible(False)
    
    fig.suptitle('FUNGSI KEANGGOTAAN - SISTEM DETEKSI STRESS FUZZY',
                color=TEXT, fontsize=14, fontweight='bold', y=0.98)
    
    plt.savefig('/mnt/user-data/outputs/membership_functions.png',
                dpi=150, bbox_inches='tight', facecolor='#0f0f1a')
    plt.close()
    print("  ✅ Grafik fungsi keanggotaan disimpan!")

def plot_hasil(skor, kategori, bpm, suhu, langkah, tidur, beban, screen, aggregated):
    """
    Visualisasi hasil analisis stress untuk satu kasus.
    """
    fig = plt.figure(figsize=(16, 10))
    fig.patch.set_facecolor('#0f0f1a')
    
    BG = '#1a1a2e'
    TEXT = '#e0e0ff'
    GRID = '#2a2a4a'
    
    gs = GridSpec(2, 3, figure=fig, hspace=0.5, wspace=0.45)
    
    # ---------- Plot 1: Skor Gauge ----------
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_facecolor(BG)
    ax1.set_xlim(0, 100)
    ax1.set_ylim(0, 1.5)
    
    # Background bar
    ax1.barh(0.5, 100, height=0.4, color='#2a2a4a', left=0, zorder=1)
    
    # Gradient bar (rendah-sedang-tinggi)
    for xi in range(100):
        if xi < 35: c = '#00ff88'
        elif xi < 65: c = '#ffaa00'
        else: c = '#ff3366'
        ax1.barh(0.5, 1, height=0.4, color=c, left=xi, alpha=0.7, zorder=2)
    
    # Pointer
    ax1.axvline(x=skor, color='white', linewidth=3, zorder=5)
    ax1.plot(skor, 0.7, 'v', color='white', markersize=14, zorder=6)
    
    color_cat = '#00ff88' if kategori=='RENDAH' else '#ffaa00' if kategori=='SEDANG' else '#ff3366'
    ax1.text(skor, 1.1, f'{skor:.1f}', color=color_cat,
             fontsize=18, fontweight='bold', ha='center', va='center')
    ax1.text(50, 1.35, f'Kategori: {kategori}', color=color_cat,
             fontsize=11, fontweight='bold', ha='center')
    
    ax1.set_title('SKOR STRESS', color=TEXT, fontsize=10, fontweight='bold')
    ax1.set_xlabel('Skala 1-100', color=TEXT, fontsize=8)
    ax1.set_yticks([])
    ax1.tick_params(colors=TEXT, labelsize=8)
    ax1.spines['bottom'].set_color(GRID)
    ax1.spines['top'].set_color(GRID)
    ax1.spines['left'].set_color(GRID)
    ax1.spines['right'].set_color(GRID)
    
    # ---------- Plot 2: Agregasi Output Fuzzy ----------
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_facecolor(BG)
    ax2.grid(True, color=GRID, linewidth=0.5, alpha=0.7)
    
    x = np.linspace(1, 100, 1000)
    mu_r = np.array([trapezoid(xi,1,1,25,45) for xi in x])
    mu_s = np.array([triangle(xi,25,50,75) for xi in x])
    mu_t = np.array([trapezoid(xi,55,75,100,100) for xi in x])
    
    clip_r = np.minimum(mu_r, aggregated['rendah'])
    clip_s = np.minimum(mu_s, aggregated['sedang'])
    clip_t = np.minimum(mu_t, aggregated['tinggi'])
    combined = np.maximum(np.maximum(clip_r, clip_s), clip_t)
    
    ax2.fill_between(x, clip_r, alpha=0.5, color='#00ff88', label='Rendah')
    ax2.fill_between(x, clip_s, alpha=0.5, color='#ffaa00', label='Sedang')
    ax2.fill_between(x, clip_t, alpha=0.5, color='#ff3366', label='Tinggi')
    ax2.plot(x, combined, color='white', linewidth=1.5, label='Gabungan')
    ax2.axvline(x=skor, color='cyan', linewidth=2, linestyle='--', label=f'Centroid={skor:.1f}')
    
    ax2.set_title('OUTPUT FUZZY (Defuzzifikasi)', color=TEXT, fontsize=9, fontweight='bold')
    ax2.set_xlabel('Skor Stress', color=TEXT, fontsize=8)
    ax2.set_ylabel('Derajat Keanggotaan', color=TEXT, fontsize=8)
    ax2.tick_params(colors=TEXT, labelsize=7)
    ax2.legend(fontsize=7, facecolor='#0f0f1a', edgecolor=GRID, labelcolor=TEXT)
    ax2.spines['bottom'].set_color(GRID)
    ax2.spines['top'].set_color(GRID)
    ax2.spines['left'].set_color(GRID)
    ax2.spines['right'].set_color(GRID)
    
    # ---------- Plot 3: Radar Chart Input ----------
    ax3 = fig.add_subplot(gs[0, 2], polar=True)
    ax3.set_facecolor(BG)
    
    categories = ['BPM\n(norm)', 'Suhu\n(norm)', 'Langkah\n(norm)', 
                  'Tidur\n(norm)', 'Beban\n(norm)', 'Screen\n(norm)']
    N = len(categories)
    
    # Normalisasi input ke 0-1 untuk radar
    vals_norm = [
        (bpm - 40) / (150 - 40),
        (suhu - 34) / (40 - 34),
        1 - min(langkah / 15000, 1),  # Invert: langkah banyak = stress rendah
        1 - min(tidur / 12, 1),        # Invert: tidur banyak = stress rendah
        beban / 10,
        screen / 16
    ]
    vals_norm = [max(0, min(1, v)) for v in vals_norm]
    vals_norm += vals_norm[:1]  # Tutup lingkaran
    
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    
    ax3.plot(angles, vals_norm, 'o-', linewidth=2, color='#00f5ff')
    ax3.fill(angles, vals_norm, alpha=0.3, color='#00f5ff')
    ax3.set_xticks(angles[:-1])
    ax3.set_xticklabels(categories, size=7, color=TEXT)
    ax3.set_ylim(0, 1)
    ax3.grid(color=GRID)
    ax3.set_facecolor(BG)
    ax3.spines['polar'].set_color(GRID)
    ax3.tick_params(colors=TEXT)
    ax3.set_title('Profil Input (Normalized)\n', color=TEXT, fontsize=9, fontweight='bold')
    
    # ---------- Plot 4: Input Summary Bar ----------
    ax4 = fig.add_subplot(gs[1, :])
    ax4.set_facecolor(BG)
    ax4.grid(True, color=GRID, linewidth=0.5, alpha=0.5, axis='x')
    
    input_labels = ['BPM', 'Suhu (°C)', 'Langkah (×100)', 'Tidur (jam)', 'Beban (/10)', 'Screen (jam)']
    input_values = [bpm, suhu, langkah/100, tidur, beban*10, screen]  # Scaled for display
    input_raw = [f"{bpm} BPM", f"{suhu}°C", f"{langkah:,} langkah", f"{tidur} jam", f"{beban}/10", f"{screen} jam"]
    
    bar_colors = ['#00f5ff', '#ff00e4', '#00ff88', '#9966ff', '#ffaa00', '#ff3366']
    
    bars = ax4.barh(input_labels, input_values, color=bar_colors, alpha=0.8, height=0.5)
    
    for bar, val_str in zip(bars, input_raw):
        ax4.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                val_str, va='center', color=TEXT, fontsize=9, fontweight='bold')
    
    ax4.set_title('RINGKASAN INPUT', color=TEXT, fontsize=10, fontweight='bold')
    ax4.tick_params(colors=TEXT, labelsize=9)
    ax4.spines['bottom'].set_color(GRID)
    ax4.spines['top'].set_color(GRID)
    ax4.spines['left'].set_color(GRID)
    ax4.spines['right'].set_color(GRID)
    
    title_color = '#00ff88' if kategori=='RENDAH' else '#ffaa00' if kategori=='SEDANG' else '#ff3366'
    fig.suptitle(f'HASIL ANALISIS STRESS — Skor: {skor:.1f}/100 | Kategori: {kategori}',
                color=title_color, fontsize=14, fontweight='bold', y=0.98)
    
    plt.savefig('/mnt/user-data/outputs/hasil_analisis.png',
                dpi=150, bbox_inches='tight', facecolor='#0f0f1a')
    plt.close()
    print("  ✅ Grafik hasil analisis disimpan!")

# ============================================================
# BAGIAN 8: PROGRAM UTAMA (MAIN)
# ============================================================

def mode_interaktif():
    """
    Mode input interaktif: pengguna memasukkan data sendiri.
    """
    print("\n" + "="*60)
    print("  🧠 DETEKSI STRESS - MODE INTERAKTIF")
    print("="*60)
    print("\nMasukkan data Anda (tekan Enter untuk nilai default):\n")
    
    def input_float(prompt, default, min_val, max_val):
        while True:
            try:
                val = input(f"  {prompt} [{default}]: ").strip()
                if val == "":
                    return default
                val = float(val)
                if min_val <= val <= max_val:
                    return val
                print(f"    ⚠️  Masukkan nilai antara {min_val} - {max_val}")
            except ValueError:
                print("    ⚠️  Masukkan angka yang valid")
    
    bpm     = input_float("🫀 Detak Jantung (BPM, 40-180)", 75, 40, 180)
    suhu    = input_float("🌡️  Suhu Tubuh (°C, 34-40)", 36.6, 34, 40)
    langkah = input_float("👣 Jumlah Langkah Hari Ini (0-20000)", 5000, 0, 20000)
    tidur   = input_float("😴 Jam Tidur Semalam (0-12)", 7, 0, 12)
    beban   = input_float("💼 Beban Kerja (skala 1-10)", 5, 1, 10)
    screen  = input_float("📱 Screen Time Hari Ini (jam, 0-16)", 6, 0, 16)
    
    skor, kategori, detail_fuzz, aggregated = hitung_stress(
        bpm, suhu, langkah, tidur, beban, screen, verbose=True
    )
    
    print("\n📊 Membuat visualisasi...")
    plot_hasil(skor, kategori, bpm, suhu, langkah, tidur, beban, screen, aggregated)
    
    return skor, kategori

def demo_kasus():
    """
    Menjalankan beberapa kasus demo untuk menunjukkan cara kerja sistem.
    """
    kasus = [
        {
            "nama": "KASUS 1: Stress RENDAH (Kondisi Ideal)",
            "data": {"bpm": 65, "suhu": 36.3, "langkah": 10000,
                    "tidur": 8.0, "beban": 3, "screen": 2},
            "penjelasan": "BPM santai, suhu normal, banyak gerak, tidur cukup, kerja ringan, screen sedikit"
        },
        {
            "nama": "KASUS 2: Stress SEDANG (Kondisi Umum)",
            "data": {"bpm": 82, "suhu": 36.8, "langkah": 5000,
                    "tidur": 6.0, "beban": 6, "screen": 7},
            "penjelasan": "BPM agak tinggi, aktivitas cukup, tidur kurang ideal, beban sedang"
        },
        {
            "nama": "KASUS 3: Stress TINGGI (Kondisi Kritis)",
            "data": {"bpm": 105, "suhu": 37.8, "langkah": 1500,
                    "tidur": 4.5, "beban": 9, "screen": 12},
            "penjelasan": "BPM sangat tinggi, kurang gerak, tidur buruk, beban berat, screen berlebihan"
        }
    ]
    
    print("\n" + "="*70)
    print("  🎯 DEMO: 3 KASUS BERBEDA")
    print("="*70)
    
    results = []
    for kasus_data in kasus:
        print(f"\n📌 {kasus_data['nama']}")
        print(f"   💬 {kasus_data['penjelasan']}")
        
        d = kasus_data['data']
        skor, kategori, _, agg = hitung_stress(
            d['bpm'], d['suhu'], d['langkah'],
            d['tidur'], d['beban'], d['screen'],
            verbose=False
        )
        
        emoji = "🟢" if kategori=="RENDAH" else "🟡" if kategori=="SEDANG" else "🔴"
        print(f"   {emoji} Hasil: Skor = {skor:.1f}/100 | Kategori = {kategori}")
        results.append((kasus_data['nama'], skor, kategori))
    
    print("\n" + "="*70)
    print("  📊 RINGKASAN SEMUA KASUS:")
    print("="*70)
    for nama, skor, kat in results:
        emoji = "🟢" if kat=="RENDAH" else "🟡" if kat=="SEDANG" else "🔴"
        bar = "█" * int(skor/5)
        print(f"  {emoji} {nama[-20:]:20s} | Skor: {skor:5.1f} | {bar}")
    
    return results

# ============================================================
# TITIK MASUK PROGRAM
# ============================================================

if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════╗
║   SISTEM DETEKSI STRESS - FUZZY LOGIC MAMDANI           ║
║   Dibuat untuk membantu pemahaman Fuzzy Logic            ║
╚══════════════════════════════════════════════════════════╝

Pilih mode:
  1. Demo otomatis (3 kasus: Rendah, Sedang, Tinggi)
  2. Input data sendiri (interaktif)
  3. Jalankan keduanya + visualisasi fungsi keanggotaan
""")
    
    pilihan = input("Masukkan pilihan (1/2/3) [default: 3]: ").strip()
    if pilihan == "":
        pilihan = "3"
    
    print("\n📊 Membuat grafik fungsi keanggotaan...")
    plot_membership_functions()
    
    if pilihan == "1":
        demo_kasus()
    elif pilihan == "2":
        mode_interaktif()
    else:
        demo_kasus()
        print("\n" + "="*70)
        print("  Sekarang coba dengan data Anda sendiri!")
        print("="*70)
        mode_interaktif()
