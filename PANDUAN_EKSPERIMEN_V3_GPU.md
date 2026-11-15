# Panduan Cepat Training & Evaluasi Eksperimen Dataset v3 di Cloud GPU

Panduan *copy-paste* berurutan untuk melatih **Dataset v3** (`760` train, `172` val, `162` test — sudah bersih dari bug spasi backtick & mark `[Halaman buku N]`) pada **`Qwen/Qwen2.5-3B-Instruct`** dan **`Qwen/Qwen3-4B-Instruct-2507`**, mengevaluasi 40 kasus (`evals/eval_cases.jsonl`), membandingkan kenaikan skor di atas **Baseline v2 (`72,5%`)**, lalu mengunduh hasilnya kembali ke laptop.

---

## Ringkasan Baseline Saat Ini (Sudah Tersimpan di Repo)

Hasil evaluasi Baseline v2 (`Qwen/Qwen2.5-3B-Instruct` + `adapters/qwen-kka-lora/final`) pada 40 kasus tercatat di `exports/eval_baseline_v2_adapter.json`:
- **Pass Rate:** `29 / 40` (**`72,5%`**)
- **Skor Rata-rata (`overall`):** **`0.949`**
  - `tuntas`: `0.975`
  - `akurat`: `0.925` *(4 kasus gagal karena halusinasi istilah fiktif/topik luar KKA)*
  - `proporsional`: `0.845` *(8 kasus gagal karena terlalu panjang & selalu memaksakan pertanyaan balik Sokratik)*
  - `bersih`: `1.000`
  - `fokus`: `1.000`

---

## Langkah 1 — Persiapan Proyek di Cloud GPU (1x di Awal)

Jika Anda mengunggah proyek lewat **ZIP** dari laptop Windows:
1. Kompres folder `c:\project-uta\kka_lab` menjadi `kka_lab.zip` (**kecualikan folder `.venv` dan `__pycache__`**).
2. Unggah `kka_lab.zip` ke JupyterLab Cloud GPU, buka **Terminal**, lalu jalankan:

```bash
unzip -q kka_lab.zip -d ~/kka_lab
cd ~/kka_lab

# Cek GPU Cloud & versi PyTorch
python -c "import torch; print(torch.cuda.get_device_name(0), '| VRAM (GB):', round(torch.cuda.get_device_properties(0).total_memory/1024**3, 1), '| Torch:', torch.__version__, '| bf16:', torch.cuda.is_bf16_supported())"

# Pasang dependensi inti (tanpa menimpa torch bawaan image Cloud GPU)
python -m pip install -r requirements-core.txt

# Verifikasi unit test & keutuhan Dataset v3 (760 train / 172 val / 162 test)
python -m unittest discover -s tests -v
python scripts/evaluate_assistant.py --check-dataset
```

---

## Langkah 2 — Eksperimen A: Latih & Evaluasi Dataset v3 pada `Qwen/Qwen2.5-3B-Instruct` (~8 Menit)

Tahap ini mengisolasi **pengaruh murni Dataset v3 + regularisasi anti-overfitting** pada model dasar yang sama dengan baseline (`Qwen/Qwen2.5-3B-Instruct`).

```bash
# 1. Unduh bobot Qwen/Qwen2.5-3B-Instruct & kunci model_record.json
python scripts/download_model.py \
  --model Qwen/Qwen2.5-3B-Instruct \
  --record model_record.json \
  --force

# 2. Jalankan training QLoRA Dataset v3 (~4-6 menit di L40S / RTX 4090 / RTX 5090)
python training/train_lora.py \
  --data data/lora/lora_train.jsonl \
  --val-data data/lora/lora_val.jsonl \
  --record model_record.json \
  --out adapters/qwen-kka-lora-v3-qwen25 \
  --epochs 4 \
  --eval-strategy epoch \
  --early-stopping-patience 2 \
  --precision bf16 \
  --batch-size 4 \
  --grad-accum 2 \
  --lr 1e-4 \
  --weight-decay 0.05 \
  --max-grad-norm 1.0 \
  --neftune-noise-alpha 5.0

# 3. Evaluasi 40 kasus (evals/eval_cases.jsonl) dengan adaptor v3 Qwen2.5-3B (~3 menit)
python scripts/evaluate_assistant.py \
  --local-adapter adapters/qwen-kka-lora-v3-qwen25/final \
  --save-report exports/eval_v3_qwen25_adapter.json \
  --save-responses exports/eval_v3_qwen25_responses.jsonl
```

---

## Langkah 3 — Eksperimen B: Latih & Evaluasi Dataset v3 pada `Qwen/Qwen3-4B-Instruct-2507` (~10 Menit)

Tahap ini menguji peningkatan kapasitas penalaran & kepatuhan instruksi dengan beralih ke **`Qwen/Qwen3-4B-Instruct-2507`** pada Dataset v3 yang sama.

```bash
# 1. Unduh bobot Qwen/Qwen3-4B-Instruct-2507 & simpan juga salinan model_record
python scripts/download_model.py \
  --model Qwen/Qwen3-4B-Instruct-2507 \
  --record model_record.json \
  --force
cp model_record.json model_record_qwen3_4b.json

# 2. Jalankan training QLoRA Dataset v3 pada Qwen3-4B-Instruct-2507 (~6-8 menit)
python training/train_lora.py \
  --data data/lora/lora_train.jsonl \
  --val-data data/lora/lora_val.jsonl \
  --record model_record.json \
  --out adapters/qwen-kka-lora-v3-qwen3-4b \
  --epochs 4 \
  --eval-strategy epoch \
  --early-stopping-patience 2 \
  --precision bf16 \
  --batch-size 4 \
  --grad-accum 2 \
  --lr 1e-4 \
  --weight-decay 0.05 \
  --max-grad-norm 1.0 \
  --neftune-noise-alpha 5.0

# 3. Evaluasi 40 kasus (evals/eval_cases.jsonl) dengan adaptor v3 Qwen3-4B (~4 menit)
python scripts/evaluate_assistant.py \
  --local-adapter adapters/qwen-kka-lora-v3-qwen3-4b/final \
  --save-report exports/eval_v3_qwen3_4b_adapter.json \
  --save-responses exports/eval_v3_qwen3_4b_responses.jsonl
```

---

## Langkah 4 — Cetak Tabel Perbandingan Ketiga Model (Baseline v2 vs v3-Qwen2.5 vs v3-Qwen3-4B)

Jalankan perintah satu baris ini di Terminal Cloud GPU untuk melihat perbandingan langsung ketiganya:

```bash
python -c "
import json
from pathlib import Path

files = [
    ('Baseline v2 (Qwen2.5-3B)', 'exports/eval_baseline_v2_adapter.json', 'adapters/qwen-kka-lora/evaluation.json'),
    ('Dataset v3 (Qwen2.5-3B)', 'exports/eval_v3_qwen25_adapter.json', 'adapters/qwen-kka-lora-v3-qwen25/evaluation.json'),
    ('Dataset v3 (Qwen3-4B-2507)', 'exports/eval_v3_qwen3_4b_adapter.json', 'adapters/qwen-kka-lora-v3-qwen3-4b/evaluation.json'),
]
print(f'{\"Model / Eksperimen\":<28} | {\"Pass\":<8} | {\"Overall\":<7} | {\"Tuntas\":<6} | {\"Akurat\":<6} | {\"Prop\":<6} | {\"Bersih\":<6} | {\"Fokus\":<6} | {\"ValLoss\":<7}')
print('-'*100)
for label, f_eval, f_loss in files:
    p = Path(f_eval)
    if not p.exists():
        continue
    s = json.loads(p.read_text(encoding='utf-8'))['summary']
    avg = s['averages']
    vloss = '-'
    if Path(f_loss).exists():
        vloss = f\"{json.loads(Path(f_loss).read_text(encoding='utf-8')).get('eval_loss', 0):.4f}\"
    print(f\"{label:<28} | {s['passed']:>2}/{s['total']:<2} ({s['pass_rate']*100:>4.1f}%) | {avg['overall']:<7.3f} | {avg['tuntas']:<6.3f} | {avg['akurat']:<6.3f} | {avg['proporsional']:<6.3f} | {avg['bersih']:<6.3f} | {avg['fokus']:<6.3f} | {vloss:<7}\")
"
```

---

## Langkah 5 — Kompres & Unduh Hasil ke Laptop Sebelum Matikan Sewa GPU

Agar kedua adaptor hasil training (`~60-80 MB` per adaptor) dan seluruh laporan JSON tersimpan permanen di laptop Anda:

```bash
zip -r exports/hasil_eksperimen_v3_lengkap.zip \
  model_record.json \
  model_record_qwen3_4b.json \
  adapters/qwen-kka-lora-v3-qwen25/final \
  adapters/qwen-kka-lora-v3-qwen25/evaluation.json \
  adapters/qwen-kka-lora-v3-qwen25/run_manifest.json \
  adapters/qwen-kka-lora-v3-qwen25/trainer_state.json \
  adapters/qwen-kka-lora-v3-qwen3-4b/final \
  adapters/qwen-kka-lora-v3-qwen3-4b/evaluation.json \
  adapters/qwen-kka-lora-v3-qwen3-4b/run_manifest.json \
  adapters/qwen-kka-lora-v3-qwen3-4b/trainer_state.json \
  exports/eval_*.json \
  exports/eval_*.jsonl
```

1. Di panel kiri JupyterLab Cloud GPU, buka folder `exports/`, klik kanan **`hasil_eksperimen_v3_lengkap.zip`** $\rightarrow$ **Download**.
2. Ekstrak file ZIP tersebut di folder proyek laptop Anda (`c:\project-uta\kka_lab`).
3. Anda bisa langsung memakai adaptor pemenang di laptop maupun di server dengan mengatur `.env`:
   - Jika memilih **Qwen3-4B-Instruct-2507**:
     ```env
     MODEL_ID=Qwen/Qwen3-4B-Instruct-2507
     MODEL_ADAPTER=adapters/qwen-kka-lora-v3-qwen3-4b/final
     ```
   - Jika memilih **Qwen2.5-3B-Instruct**:
     ```env
     MODEL_ID=Qwen/Qwen2.5-3B-Instruct
     MODEL_ADAPTER=adapters/qwen-kka-lora-v3-qwen25/final
     ```
