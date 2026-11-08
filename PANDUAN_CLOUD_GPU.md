# Panduan Lengkap Menjalankan `kka_lab` di Cloud GPU (RTX 5090)

Panduan urut dari awal menyewa instance GPU, memasang Qwen + **Baidu Unlimited-OCR**, melatih LoRA, menyalakan server multi-user untuk siswa, hingga menghubungkannya ke domain & backend TypeScript/Go.

---

## 1. Spesifikasi Sewa Instance yang Dipilih

Saat membuat instance baru di penyedia Cloud GPU:
- **GPU:** RTX 5090 (32 GB VRAM) — *cukup 1 GPU untuk menjalankan Qwen + Unlimited-OCR bersamaan*.
- **Ukuran Instance:** **`8 Core • 16 GB RAM • 200 GB storage`**
- **Image / Template:** **PyTorch + Jupyter** *(berbasis Ubuntu, sudah termasuk driver NVIDIA & CUDA ≥ 12.8)*.
- **SSH Key:** Tambahkan SSH public key laptop Anda (opsional, karena Terminal di web Jupyter juga sudah bisa langsung dipakai).

---

## 2. Memasukkan Proyek ke Cloud GPU

Buka **JupyterLab** dari dashboard penyedia GPU, klik menu **Terminal**, lalu pilih salah satu cara:

### Cara A — Lewat GitHub (Disarankan)
```bash
git clone https://github.com/<username-anda>/kka_lab.git
cd kka_lab
```

### Cara B — Lewat Upload ZIP
1. Di laptop Windows, kompres folder `kka_lab` menjadi `kka_lab.zip` (**JANGAN masukkan folder `.venv`**, karena `.venv` laptop adalah versi Windows).
2. Seret (*drag & drop*) `kka_lab.zip` ke panel kiri JupyterLab.
3. Di Terminal Jupyter, jalankan:
   ```bash
   unzip kka_lab.zip
   cd kka_lab
   ```

---

## 3. Persiapan Awal di Server GPU (Cukup 1 Kali per Instance)

> **PENTING untuk RTX 5090:** **JANGAN** menjalankan perintah `pip install torch ... cu126` yang ada di panduan laptop WSL. Gunakan PyTorch bawaan template Cloud GPU karena RTX 5090 membutuhkan CUDA ≥ 12.8.

Jalankan berurutan di Terminal Jupyter:

```bash
# 1. Pastikan berada di dalam folder kka_lab
cd ~/kka_lab   # sesuaikan jika path berbeda

# 2. Cek GPU & versi CUDA PyTorch (pastikan RTX 5090 terbaca dan CUDA >= 12.8)
python -c "import torch; print(torch.cuda.get_device_name(0), '| Torch:', torch.__version__, '| CUDA:', torch.version.cuda)"

# 3. Pasang pustaka inti, pustaka OCR, dan vLLM (tanpa menimpa torch bawaan)
python -m pip install -r requirements-core.txt -r requirements-ocr.txt vllm

# 4. Salin template .env lalu isi kredensial
cp .env.example .env
nano .env
```

Isi file `.env` seperti ini (simpan di `nano` dengan `Ctrl+O` $\rightarrow$ `Enter` $\rightarrow$ `Ctrl+X`):

```env
ASTRA_DB_API_ENDPOINT=https://xxxx-xxxx.apps.astra.datastax.com
ASTRA_DB_APPLICATION_TOKEN=AstraCS:xxxxxxxx
ASTRA_DB_KEYSPACE=rag
ASTRA_DB_COLLECTION=rag_chatbot

MODEL_ID=Qwen/Qwen2.5-3B-Instruct
# Isi baris di bawah JIKA Anda sudah punya folder hasil training LoRA:
MODEL_ADAPTER=

# Konfigurasi Baidu Unlimited-OCR (jalan lokal di port 8001)
OCR_BACKEND=vllm
OCR_MODEL_ID=baidu/Unlimited-OCR
OCR_BASE_URL=http://127.0.0.1:8001/v1

# Kunci keamanan komunikasi dari Backend TS/Go ke Server AI ini
AI_INTERNAL_SECRET=isi-kunci-rahasia-internal-anda
RAG_ADMIN_SECRET=isi-kunci-khusus-guru-admin

# Auto-learn materi KKA baru dari diskusi ke RAG (ubah ke false saat riset eksperimen terkontrol)
RAG_AUTO_LEARN_ENABLED=true
RAG_AUTO_LEARN_MAX_SIMILARITY=0.82
```

Setelah `.env` tersimpan, unduh bobot kedua model ke disk instance (cukup 1 kali):

```bash
# 5. Unduh model chat Qwen2.5-3B (~6 GB) -> menghasilkan model_record.json
python scripts/download_model.py

# 6. Unduh model Baidu Unlimited-OCR (~7 GB, tidak perlu git clone repo Baidu)
python scripts/download_ocr_model.py
```

> **Cek Astra DB:** Buka [astra.datastax.com](https://astra.datastax.com) dan pastikan database berstatus **Active** (jika **Hibernated**, klik **Resume** dan tunggu 1–2 menit).

---

## 4. (Opsional) Jika Mau Melatih / Fine-Tuning LoRA Dulu

Jika Anda belum memiliki adaptor LoRA atau ingin melatih ulang:

1. Unggah dataset penelitian Anda ke `data/lora_dialogues.jsonl`.
2. Buka notebook **`notebooks/Training_LoRA_CloudGPU.ipynb`** di JupyterLab, lalu jalankan sel 1 sampai 7 secara berurutan.
   *(Atau jalankan langsung via terminal)*:
   ```bash
   python training/train_lora.py \
     --data data/lora_dialogues.jsonl \
     --record model_record.json \
     --out adapters/qwen-kka \
     --precision bf16 --batch-size 4 --grad-accum 2
   ```
3. Setelah selesai, arahkan `.env` ke adaptor hasil training:
   ```env
   MODEL_ADAPTER=adapters/qwen-kka/final
   ```
4. **WAJIB BACKUP:** Unduh file ZIP di folder `exports/` ke laptop Anda (klik kanan di Jupyter $\rightarrow$ **Download**). Jika masa sewa habis dan menyewa GPU baru di kemudian hari, Anda **tidak perlu training ulang** — cukup unggah kembali folder adaptor tersebut.

---

## 5. Menyalakan Layanan AI + Unlimited-OCR untuk Siswa (2 Proses)

Gunakan **`tmux`** supaya server tetap hidup meskipun tab browser Jupyter atau koneksi SSH Anda tertutup.

### Langkah 5A — Nyalakan Engine Unlimited-OCR (Port Lokal `8001`)
```bash
tmux new -s ocr
cd ~/kka_lab
bash scripts/serve_ocr_vllm.sh
```
* Tunggu sampai muncul log `Application startup complete` / `Uvicorn running on http://127.0.0.1:8001`.
* Skrip ini otomatis membatasi pemakaian VRAM OCR sebesar **45% (~14 GB)** sehingga sisa **~18 GB VRAM** aman untuk model Qwen.
* Tekan **`Ctrl+B`**, lepas, lalu tekan **`D`** untuk keluar dari layar `tmux` tanpa mematikan proses OCR.

### Langkah 5B — Nyalakan API Utama `kka_lab` (Port `8000`)
```bash
tmux new -s api
cd ~/kka_lab
uvicorn app.main:app --host 0.0.0.0 --port 8000
```
* Tunggu hingga model Qwen selesai dimuat dan muncul `Uvicorn running on http://0.0.0.0:8000`.
* Tekan **`Ctrl+B`**, lepas, lalu tekan **`D`** untuk melepas layar `tmux`.

### Langkah 5C — Tes Cepat dari Terminal GPU
```bash
curl http://127.0.0.1:8000/api/health
# Output yang diharapkan: {"status":"ok"}
```

*(Tips `tmux`: ketik `tmux ls` untuk melihat proses aktif, `tmux attach -t ocr` untuk melihat log OCR, atau `tmux attach -t api` untuk melihat log FastAPI).*

---

## 6. Menghubungkan Domain (Cloudflare Tunnel) ke Port `8000`

Agar backend TypeScript/Go Anda bisa memanggil server GPU melalui HTTPS (`https://ai.domainanda.com`) tanpa pusing memikirkan IP dinamis atau port NAT:

1. Buka dashboard **Cloudflare Zero Trust** $\rightarrow$ **Networks** $\rightarrow$ **Tunnels** $\rightarrow$ **Create a tunnel**.
2. Salin perintah instalasi Linux (`cloudflared`) dari Cloudflare dan jalankan di Terminal GPU.
3. Pada tab **Public Hostname** di Cloudflare, arahkan:
   - **Subdomain:** `ai` (contoh: `ai.domainanda.com`)
   - **Service Type:** `HTTP`
   - **URL:** `localhost:8000`
   *(Catatan: Port `8001` milik vLLM OCR **tidak perlu** diekspos keluar karena hanya dipanggil secara internal oleh FastAPI).*

---

## 7. Panduan Endpoint untuk Backend TypeScript / Go

Seluruh request dari backend TS/Go ke `https://ai.domainanda.com` wajib menyertakan header:
`X-API-Key: <nilai AI_INTERNAL_SECRET di .env>`

### A. Siswa Upload PDF / Gambar Materi (Privat per Siswa — Opsi 1)
Dokumen hanya bisa dibaca oleh `user_id` pengunggah + materi global kelas:
```bash
curl -X POST "https://ai.domainanda.com/api/rag/documents" \
  -H "X-API-Key: isi-kunci-rahasia-internal-anda" \
  -F "file=@catatan_siswa.pdf" \
  -F "user_id=siswa_101" \
  -F "scope=user" \
  -F "source_title=Catatan Bab Klasifikasi" \
  -F "topic=Klasifikasi"
```
Respons instan (`202 Accepted`):
```json
{
  "job_id": "9f8c1d2e-...",
  "doc_id": "doc_a1b2c3d4e5f60718",
  "status": "queued",
  "scope": "user",
  "source_title": "Catatan Bab Klasifikasi"
}
```

### B. Guru / Admin Upload Materi Kelas Bersama (`scope=global`)
Terbaca oleh seluruh siswa di kelas (wajib menyertakan `X-Admin-Key`):
```bash
curl -X POST "https://ai.domainanda.com/api/rag/documents" \
  -H "X-API-Key: isi-kunci-rahasia-internal-anda" \
  -H "X-Admin-Key: isi-kunci-khusus-guru-admin" \
  -F "file=@modul_resmi_bab3.pdf" \
  -F "user_id=guru_kka" \
  -F "scope=global" \
  -F "source_title=Modul KKA Bab 3 - Evaluasi Model" \
  -F "topic=Evaluasi Model"
```

### C. Cek Status Pemrosesan OCR (`queued` $\rightarrow$ `processing` $\rightarrow$ `completed`)
```bash
curl "https://ai.domainanda.com/api/rag/jobs/9f8c1d2e-...?user_id=siswa_101" \
  -H "X-API-Key: isi-kunci-rahasia-internal-anda"
```

### D. Daftar & Hapus Dokumen Siswa
```bash
# Lihat daftar dokumen yang pernah diunggah siswa
curl "https://ai.domainanda.com/api/users/siswa_101/documents" \
  -H "X-API-Key: isi-kunci-rahasia-internal-anda"

# Hapus dokumen tertentu dari Astra DB & SQLite
curl -X DELETE "https://ai.domainanda.com/api/rag/documents/doc_a1b2c3d4e5f60718?user_id=siswa_101" \
  -H "X-API-Key: isi-kunci-rahasia-internal-anda"

# (Guru/Admin) Pantau catatan KKA yang masuk otomatis dari diskusi siswa
curl "https://ai.domainanda.com/api/rag/auto-learned" \
  -H "X-API-Key: isi-kunci-rahasia-internal-anda" \
  -H "X-Admin-Key: isi-kunci-khusus-guru-admin"

# (Guru/Admin) Hapus catatan otomatis jika ada yang ingin dibatalkan
curl -X DELETE "https://ai.domainanda.com/api/rag/documents/auto_kka_xxxxxxxxxxxxxx" \
  -H "X-API-Key: isi-kunci-rahasia-internal-anda" \
  -H "X-Admin-Key: isi-kunci-khusus-guru-admin"
```

### E. Siswa Mengirim Pertanyaan Chat (Otomatis Membaca RAG Global + Dokumen Privatnya)
```bash
curl -X POST "https://ai.domainanda.com/api/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "siswa_101",
    "question": "Apa perbedaan presisi dan recall dari catatan yang baru saya upload?",
    "mode": "terstruktur",
    "consent_to_training": false
  }'
```

---

## 8. Checklist Wajib Sebelum Menghentikan / Menghapus Instance Sewa

Disk pada Cloud GPU bersifat sementara (*ephemeral*) dan akan terhapus permanen saat instance di-*terminate* (dihapus). Sebelum masa sewa berakhir, pastikan Anda mengunduh:

1. **Adaptor LoRA** (folder `exports/*.zip` atau `adapters/.../final/`) jika Anda baru saja melakukan training.
2. **File `model_record.json` & `ocr_model_record.json`** agar revisi model tetap konsisten di sesi berikutnya.
3. *(Catatan: Seluruh data materi RAG dari PDF/Unlimited-OCR, status job, riwayat chat, dan memori tersimpan aman di cloud **DataStax Astra DB**, sehingga **tidak akan pernah hilang** meskipun instance sewa GPU dimatikan atau dihapus).*

---

## 9. Solusi Masalah Cepat (Troubleshooting)

| Gejala / Pesan Error | Penyebab | Solusi Cepat |
| :--- | :--- | :--- |
| Job berstatus `failed`: `Gagal menghubungi server Unlimited-OCR di http://127.0.0.1:8001` | Proses vLLM OCR belum jalan atau baru loading. | Cek dengan `tmux attach -t ocr`. Tunggu sampai vLLM selesai memuat model. |
| `Astra DB mengembalikan 503 Service Unavailable` | Database Astra DB berstatus *Hibernated*. | Buka dashboard Astra DB, klik **Resume**, tunggu 1–2 menit, lalu ulangi upload/chat. |
| `HTTP 403` saat upload `scope=global` | `RAG_ADMIN_SECRET` belum diisi di `.env` atau header `X-Admin-Key` tidak cocok. | Pastikan `RAG_ADMIN_SECRET` di `.env` sama persis dengan header `X-Admin-Key` yang dikirim backend. |
| `CUDA out of memory` saat menyalakan proses kedua | Sisa VRAM tidak cukup. | Kecilkan porsi memori vLLM: `OCR_GPU_MEMORY_UTILIZATION=0.35 bash scripts/serve_ocr_vllm.sh`. |

