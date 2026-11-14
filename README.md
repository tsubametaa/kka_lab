# Qwen lokal + Astra DB + LoRA

Panduan utama ada pada PDF docs/Tutorial_Qwen_Lokal_AstraDB_LoRA.pdf.
Untuk langkah urut deployment multi-user di server sewaan (RTX 5090 + Unlimited-OCR + Cloudflare Tunnel), baca `PANDUAN_CLOUD_GPU.md`.
Paket ini contoh implementasi untuk pengembangan prototipe, bukan layanan web multiuser.
Kode diperiksa sintaks dan fungsi bantu tanpa GPU/Astra DB milik pengguna.
Belum dijalankan pada RTX 3050 Anda atau sesi Colab Anda.

## Jalur yang dipakai

- Laptop Windows: WSL2 Ubuntu 22.04, Python 3.10, VS Code + WSL.
- Inferensi: Qwen/Qwen2.5-3B-Instruct, NF4 4-bit, satu GPU.
- Bahasa: percakapan bahasa Indonesia dan English; jawaban mengikuti bahasa pertanyaan.
- Retrieval: keyspace `rag`, collection `rag_chatbot`, vectorize yang sudah aktif.
  Kueri diterjemahkan secara lokal ke Indonesia dan English agar materi RAG
  lintas bahasa tetap dapat dicari tanpa mengandalkan vectorizer multilingual.
- Training: Colab GPU, QLoRA (model dasar 4-bit + adaptor LoRA).
- Adaptor dimuat kembali melalui Transformers + PEFT di laptop.

## Struktur Proyek

```text
kka_lab/
├── adapters/                      # Artefak checkpoint & bobot akhir LoRA hasil fine-tuning
│   ├── qwen-kka/                  # Adaptor LoRA v1 (evaluasi, lock file, run manifest)
│   └── qwen-kka-v2/               # Adaptor LoRA v2 (final checkpoint & metadata)
├── app/                           # Modul utama aplikasi (Layered by Feature)
│   ├── core/                      # Fondasi konfigurasi Pydantic & model record pin
│   │   ├── config.py              # Settings global (Astra DB, API keys, model paths)
│   │   └── __init__.py            # Export core helper
│   ├── features/                  # Modul domain independen berbasis fitur
│   │   ├── chat/                  # Engine dialog, prompt pedagogis, GPU lock & router
│   │   ├── dataset/               # Validasi format baris & tokenisasi dataset LoRA
│   │   ├── language/              # Deteksi bahasa, translasi bilingual & locales (id, en)
│   │   ├── memory/                # Penyimpanan sesi (Astra DB / SQLite) & quality gate
│   │   ├── ocr/                   # Render PDF & client Baidu Unlimited-OCR (vLLM)
│   │   ├── persona/               # Sanitasi, guardrail & preferensi persona belajar siswa
│   │   └── rag/                   # Scoped retrieval Astra DB, context builder & auto-learner
│   ├── main.py                    # Entrypoint server REST API (FastAPI)
│   └── __init__.py                # Package descriptor
├── data/                          # Dataset percakapan & database penyimpanan lokal
│   ├── lora/                      # Partisi dataset (train, val, test) & chunk Astra DB
│   ├── demo_dialogues.jsonl       # Contoh dialog buatan untuk pengujian alur
│   └── lora_dialogues.jsonl       # Dataset lengkap skenario dialog
├── docs/                          # Dokumentasi riset, draf skripsi & tutorial PDF
├── notebooks/                     # Jupyter notebooks pelatihan (Colab & Cloud GPU)
│   ├── Training_LoRA_Colab.ipynb     # Notebook QLoRA di Google Colab GPU (T4/A100)
│   └── Training_LoRA_CloudGPU.ipynb  # Notebook QLoRA di Cloud GPU sewaan (RTX 5090)
├── scripts/                       # Skrip diagnostik, setup & dataset generator
│   ├── dataset_builder/           # Modul builder skenario partisi dataset LoRA
│   ├── build_lora_dataset.py      # Generator otomatis dataset LoRA dari skenario
│   ├── check_environment.py       # Pemeriksaan kompatibilitas CUDA & dependensi
│   ├── download_model.py          # Unduh model dasar Qwen2.5-3B-Instruct
│   ├── download_ocr_model.py      # Unduh bobot Baidu Unlimited-OCR
│   └── serve_ocr_vllm.sh          # Runner engine vLLM Unlimited-OCR (port 8001)
├── tests/                         # Unit tests (arsitektur, RAG, memori, OCR, bahasa, dll.)
├── training/                      # Skrip pipeline pelatihan QLoRA (BitsAndBytes 4-bit)
│   └── train_lora.py              # Pelatihan LoRA dengan audit & validasi pedagogis
├── .env.example                   # Template variabel lingkungan (Astra DB, API keys, dll.)
├── ARCHITECTURE.md                # Dokumentasi arsitektur cloud-native & skema data
├── PANDUAN_CLOUD_GPU.md           # Panduan deployment server GPU (FastAPI + vLLM)
├── chat.py                        # Antarmuka CLI percakapan interaktif terminal
├── model_record.json              # Rekam jejak model dasar Hugging Face & commit SHA
├── requirements-core.txt          # Dependensi utama (FastAPI, PyTorch, Transformers, PEFT)
├── requirements-ocr.txt           # Dependensi OCR & manipulasi dokumen (PyMuPDF, vLLM)
└── README.md                      # Dokumentasi utama proyek
```

### Penjelasan Komponen Utama

| Direktori / File | Peran & Deskripsi |
| :--- | :--- |
| **`app/`** | Inti backend dengan arsitektur *Layered by Feature*. Terbagi menjadi `core/` (konfigurasi & record model) dan `features/` (`chat`, `rag`, `memory`, `persona`, `ocr`, `language`, `dataset`) yang modular dan terisolasi. |
| **`adapters/`** | Menyimpan artefak bobot LoRA hasil *fine-tuning* (`final/`), *checkpoint*, metrik evaluasi (`evaluation.json`), serta manifest audit pelatihan (`run_manifest.json`). |
| **`data/`** | Berisi partisi dataset pelatihan (`lora_train.jsonl`, `lora_val.jsonl`, `lora_test.jsonl`), data chunk Astra DB, dan dialog percontohan. |
| **`scripts/`** | Perangkat bantu otomatisasi: generator dataset skenario modular (`build_lora_dataset.py`), pengunduh model (Qwen & OCR), serta runner vLLM. |
| **`training/` & `notebooks/`** | Pipeline QLoRA berbasis BitsAndBytes 4-bit (`train_lora.py`) dan notebook siap pakai untuk Google Colab maupun Cloud GPU sewaan. |
| **`tests/`** | Rangkaian pengujian unit otomatis untuk memverifikasi integritas arsitektur, retrieval RAG, memori percakapan, OCR, dan sanitasi persona. |
| **Dokumentasi Root** | [`ARCHITECTURE.md`](ARCHITECTURE.md) (arsitektur *cloud-native* Astra DB & vLLM) dan [`PANDUAN_CLOUD_GPU.md`](PANDUAN_CLOUD_GPU.md) (panduan langkah deployment multi-user di server sewaan). |

## Pengujian Unit

Jalankan seluruh pengujian unit untuk memverifikasi logika helper dan memori percakapan:

```bash
python -m unittest discover tests
```

## Instalasi di terminal Ubuntu WSL

Ekstrak paket hingga folder aktif berisi requirements-core.txt dan chat.py.
Jangan jalankan perintah Linux ini di PowerShell.

```bash
sudo apt update
sudo apt install -y python3-venv python3-pip unzip
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements-core.txt
python scripts/check_environment.py
cp .env.example .env
```

Isi `.env` dengan endpoint dan token asli. Token tidak boleh masuk Svelte/browser.
PyTorch wheel menyertakan runtime CUDA; langkah ini tidak memerlukan kompilasi CUDA.
Driver NVIDIA Windows harus kompatibel dengan runtime CUDA 12.6.
Jangan pasang driver display NVIDIA Linux di dalam WSL.

## Prinsip pedagogis

Kedua mode tidak boleh memberi jawaban akhir tanpa proses berpikir: jawaban harus
memuat langkah dan alasan singkat serta satu pemantik untuk siswa. Mode `umum`
memberi penjelasan ringkas bertahap dan pemantik penerapan; mode `terstruktur`
meminta upaya siswa lebih dulu dan menambah petunjuk secara bertahap sebelum
memberikan solusi lengkap. Jika siswa buntu, AI tetap memberi bantuan yang cukup.
Perubahan ini membuat mode `umum` bukan lagi kondisi jawaban langsung; protokol
pembanding penelitian perlu menyesuaikan definisinya.

## Jalankan berurutan

```bash
python scripts/download_model.py
python chat.py --mode umum --no-rag
python -m app.features.rag
python chat.py --mode terstruktur --log logs/pilot.jsonl
```

Pada pengujian tanpa RAG cukup kirim sapaan; belum ada materi sumber yang disisipkan.
`/baru` membuat sesi/topik baru dan menghapus riwayat. `/keluar` berhenti.
Default: RAG mengambil sumber baru di setiap pertanyaan. Pertanyaan lanjutan pendek
(kurang dari 6 kata, mis. "contohnya?") digabung dengan topik sebelumnya saat mencari.
Kueri baru diterjemahkan oleh model lokal ke Indonesia dan English sebelum pencarian
Astra; kueri Inggris juga dapat menemukan materi Indonesia dan sebaliknya. Ini
menambah satu generasi lokal dan dua pencarian vector per pertanyaan baru. Skor
similarity bukan probabilitas jawaban benar. Periksa relevansi teks hasilnya.

Opsi tambahan `chat.py`:

- `--rag-per-sesi`: ambil sumber sekali di awal sesi (perilaku versi awal tutorial).
- `--max-prompt-tokens 0`: batas maksimum token prompt (default otomatis mengikuti konteks maksimum arsitektur Qwen2.5-3B, yaitu ~32.224 token tanpa perlu memasang flag tambahan). Sistem secara adaptif memangkas riwayat lama, konteks RAG, hingga meringkas bagian tengah prompt raksasa serta menurunkan budget otomatis jika VRAM tidak mencukupi.
- `--max-new-tokens 512`: panjang maksimum jawaban (default 512).
- `--max-history 4`: jumlah giliran tanya-jawab terakhir yang diingat.
- `--debug`: tampilkan label sumber, skor similarity, query RAG, token, dan waktu.
- `--tanpa-sumber`: sembunyikan baris `📚 Sumber` di bawah jawaban.
- `--persona "..."`: gaya jawaban pilihan siswa (lihat bagian Persona). Di dalam chat:
  `/persona <gaya>` mengatur, `/persona` melihat, `/persona hapus` kembali normal.

Untuk penelitian, gunakan pengaturan RAG dan generasi yang sama pada kedua mode.

## API web lokal dan memori percakapan

Backend HTTP memakai FastAPI. Model dimuat satu kali saat server mulai; frontend
mengirim pertanyaan ke backend, bukan memanggil model atau Astra DB langsung.

```bash
python -m pip install -r requirements-core.txt
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Contoh request `POST http://127.0.0.1:8000/api/chat`:

```json
{
  "user_id": "id-user-dari-aplikasi",
  "question": "Apa perbedaan data latih dan data uji?",
  "mode": "umum",
  "consent_to_training": false
}
```

Kirim kembali `session_id` dari respons untuk melanjutkan sesi. Contoh pemanggilan
dari frontend JavaScript:

```js
const response = await fetch("http://localhost:8000/api/chat", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ user_id, session_id, question, mode: "umum" }),
});
const result = await response.json();
if (!response.ok) throw new Error(result.detail);
session_id = result.session_id;
showMessage(result.answer);
```

SQLite menyimpan riwayat lintas sesi per `user_id`; pertanyaan yang sama setelah
normalisasi teks dan pada mode yang sama bisa memakai jawaban sebelumnya. Pengetahuan
personal yang secara eksplisit dikirim pengguna disimpan terpisah dan dapat diambil
kembali untuk pertanyaan terkait pada chat berikutnya. Contoh menyimpan pengetahuan:

```http
POST /api/users/{user_id}/knowledge
Content-Type: application/json

{"question":"Apa perbedaan data latih dan data uji?","answer":"Data latih dipakai untuk menyesuaikan model, sedangkan data uji untuk evaluasi."}
```

Pengetahuan hanya tersedia untuk `user_id` tersebut, dipakai sebagai konteks yang
belum terverifikasi, dan bukan pelatihan bobot model. Pengguna harus memilih sendiri
informasi yang disimpan. Atur `CHAT_DB_PATH`, `MODEL_RECORD`, `MODEL_ADAPTER`,
`MAX_NEW_TOKENS`, dan `FRONTEND_ORIGINS` melalui environment. Database default
berada di `data/chat_memory.sqlite3`. Endpoint penghapusan riwayat dan pengetahuan:
`DELETE /api/users/{user_id}/memory`.

Penyaringan training hanya memberi status `pending_review` jika pengguna memberi
persetujuan dan pertanyaan/jawaban lolos pemeriksaan kualitas sederhana. Pertanyaan
tanpa topik yang cukup jelas, rujukan samar (mis. “yang itu”), jawaban singkat,
jawaban yang meminta klarifikasi atau menyatakan ketidakpastian, serta jawaban yang
dipakai ulang dari memori dikecualikan. Ragam bahasa santai tidak otomatis ditolak
jika maksudnya cukup jelas. Pemeriksaan ini heuristik dan tidak dapat memastikan
seluruh maksud atau pemahaman; karena itu status pending bukan persetujuan untuk
training. Guru/admin tetap harus meninjau dan menyetujui data secara manual.
Pemeriksaan kualitas mencakup indikator ketidakjelasan dan ketidakpastian bahasa
Indonesia maupun English. Kandidat chat belum memiliki alur peninjauan/ekspor otomatis
ke format LoRA; pelatihan tetap perlu dataset yang disusun dan ditinjau, lalu
dijalankan manual. Contoh English boleh digunakan; buat `group_id` terpisah untuk
setiap skenario dan bahasa, serta pastikan kedua mode memiliki jawaban dalam bahasa
pertanyaan.

Penyimpanan pengetahuan personal di atas membuat informasi dapat dipakai kembali
melalui retrieval tanpa mengubah bobot model. Ini bukan fine-tuning dan tidak membuat
pengetahuan tersedia bagi pengguna lain.

RAG mengambil materi referensi untuk menjawab; RAG bukan proses melatih model.
Memori chat saat ini adalah riwayat privat dan pencarian pertanyaan identik,
bukan koleksi pengetahuan RAG. Untuk deployment multiuser, jangan percaya
`user_id` dari browser: tambahkan autentikasi dan ambil user ID dari token sesi,
batasi akses endpoint penghapusan, gunakan HTTPS, serta tentukan retensi dan
persetujuan penyimpanan data. CORS hanya kontrol browser, bukan autentikasi.

## Persona jawaban siswa

Siswa dapat mengatur gaya jawaban, misalnya tsundere seperti Nino Nakano yang
berkata "Hmph" saat tidak setuju. Persona hanya mengubah nada, sapaan, dan ungkapan
khas; isi materi, sumber `[S1]`, langkah pedagogis, mode, dan aturan bahasa tetap
sama. Teks persona dibungkus tag `<persona_siswa>` di prompt sistem dan diikuti
aturan bahwa persona tidak boleh menimpa aturan lain (pola dari panduan Anthropic
"keep in character" dan "mitigate jailbreaks").

```http
PUT /api/users/{user_id}/persona
Content-Type: application/json

{"persona": "Jawab tsundere seperti Nino Nakano, bilang 'Hmph' kalau tidak setuju"}
```

`GET` membaca dan `DELETE` menghapus persona; `DELETE /api/users/{user_id}/memory`
juga menghapusnya. Maksimal 500 karakter. Persona ditolak (HTTP 422) jika berisi
upaya mengabaikan aturan/prompt sistem, meminta jawaban akhir tanpa proses, atau
gaya kasar/menghina/romantis. Penyaringan ini berbasis daftar pola dan bisa
dilewati dengan parafrase, jadi guru tetap perlu memantau. Model 3B juga bisa tidak
konsisten memerankan persona; ini batas kemampuan model, bukan error.

Untuk penelitian, persona menjadi variabel tambahan: nonaktifkan atau samakan pada
kedua kondisi. Log `chat.py --log` mencatat kolom `persona`.

## Upload materi PDF/gambar (Baidu Unlimited-OCR -> Astra DB)

Repositori `baidu/Unlimited-OCR` **tidak perlu di-clone manual**; kode arsitektur dan bobot
diunduh otomatis dari Hugging Face (`trust_remote_code=True`). Pasang dependensi tambahan:

```bash
python -m pip install -r requirements-ocr.txt
python scripts/download_ocr_model.py
```

Di server GPU besar (mis. RTX 5090 32 GB), jalankan OCR sebagai proses lokal terpisah di
port `8001` agar tidak menahan proses chat Qwen di port `8000`:

```bash
# Terminal 1 (Engine Unlimited-OCR via vLLM, memakai ~45% VRAM):
pip install vllm
bash scripts/serve_ocr_vllm.sh

# Terminal 2 (API utama kka_lab):
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Mode alternatif pada `.env` (`OCR_BACKEND`):
- `vllm` (default): memanggil `OCR_BASE_URL` (`http://127.0.0.1:8001/v1`).
- `transformers`: memuat `baidu/Unlimited-OCR` langsung di proses Python tanpa vLLM.
- `native_pdf`: mengekstrak lapisan teks digital PDF via PyMuPDF di CPU (tanpa GPU OCR).

Ruang lingkup dokumen (Opsi 1):
- `scope="user"` (default siswa): chunk hanya terbaca oleh `user_id` pengunggah ditambah
  materi bawaan/global kelas.
- `scope="global"` (khusus guru/admin): wajib menyertakan header `X-Admin-Key` yang cocok
  dengan `RAG_ADMIN_SECRET` pada `.env`.
- Endpoint: `POST /api/rag/documents` (HTTP 202), `GET /api/rag/jobs/{job_id}`,
  `GET /api/users/{user_id}/documents`, dan `DELETE /api/rag/documents/{doc_id}`.

## Training

Buka `notebooks/Training_LoRA_Colab.ipynb` di Colab dan jalankan urut.
Gunakan dataset percakapan Anda; jangan mengganti namanya agar tampak sebagai data nyata.
`data/demo_dialogues.jsonl` hanya contoh buatan untuk uji alur. Guru belum menelaah isinya.
Setiap skenario memiliki group_id yang sama pada mode umum dan terstruktur.
Pemisahan train/validation dilakukan per group_id; soal uji akhir disimpan di luar file ini.
Label pelatihan hanya menghitung jawaban assistant terakhir, bukan user/system/konteks.
Sebelum training, `validate_rows` menolak target yang tidak memiliki minimal dua
langkah, penjelasan alasan, dan pertanyaan pemantik berpikir. Kriteria ini adalah
gate struktural, bukan penilaian mutu pedagogis otomatis; guru tetap meninjau contoh.

```bash
python training/train_lora.py --data data/lora_dialogues.jsonl --record model_record.json --out adapters/run01
```

### Training di cloud GPU (Jupyter)

Untuk instance GPU sewaan (template PyTorch + Jupyter, mis. RTX 5090/4090/A100), pakai
`notebooks/Training_LoRA_CloudGPU.ipynb`. Jangan unggah `.venv` Windows dan jangan pasang
torch `cu126` dari bagian instalasi laptop: RTX 50xx (Blackwell, sm_120) membutuhkan build
PyTorch CUDA 12.8 atau lebih baru, jadi pakai torch bawaan template. Notebook memeriksa ini.

Flag opsional (default sama dengan konfigurasi Colab: fp16, batch 1, accumulation 8):

```bash
python training/train_lora.py --data data/lora_dialogues.jsonl --record model_record.json \
  --out adapters/run01 --precision bf16 --batch-size 4 --grad-accum 2
```

Effective batch tetap 8. Presisi dan batch tercatat di `run_manifest.json`; samakan antar-run
yang dibandingkan.

Setelah hasil Colab diekstrak hingga ada `adapters/qwen-kka-lora/adapter_config.json`:

```bash
python chat.py --mode terstruktur --adapter adapters/qwen-kka-lora
python chat.py --mode umum --adapter adapters/qwen-kka-lora
```

Kedua kondisi penelitian menggunakan adaptor yang sama, basis model/revisi sama,
RAG sama, serta pengaturan generasi sama. Kini kedua mode sama-sama wajib memberi
langkah dan pemantik; perbedaannya adalah intensitas scaffolding. Perubahan ini
mengubah definisi kondisi `umum` dibanding protokol awal, sehingga instrumen dan
interpretasi hasil penelitian perlu disesuaikan. Audit contoh data agar sesuai
bahasa, mode, dan prinsip pedagogis sebelum training.

## Yang disimpan

- `model_record.json`: ID model dan revisi checkpoint yang persis dipakai.
- `final/`: adaptor, konfigurasi, tokenizer, dan model_record.json.
- `checkpoints/`: pemulihan training (berbeda dari adaptor final).
- `run_manifest.json`: hash data, kelompok split, GPU, parameter, versi Torch.
- `packages.lock.txt`, `evaluation.json`, `trainer_state.json`: catatan eksperimen.

Model 3B ini memakai lisensi qwen-research. Tautan lisensi ada di kartu model resmi.
Jika ingin mengganti ke 1.5B karena memori, buat eksperimen/adaptor baru untuk model itu.
Adaptor 3B tidak kompatibel dengan basis 1.5B atau model/revisi lain.
