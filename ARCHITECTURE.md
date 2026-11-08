# Arsitektur Sistem Asisten Pengajar KKA (`kka_lab`)

Dokumen ini menjelaskan rancangan arsitektur, komponen modular, dan alur kerja hulu ke hilir (*end-to-end*) dari sistem AI Asisten Pengajar KKA (Koding dan Kecerdasan Artifisial). Sistem ini dirancang murni *cloud-native* dengan mengandalkan **DataStax Astra DB** sebagai basis data terpusat (baik untuk vektor RAG maupun data relasional/dokumen sesi, persona, dan antrean job), **LLM lokal (Qwen2.5-3B-Instruct)**, **Baidu Unlimited-OCR**, serta mekanisme **kurasi otomatis materi baru (*guarded auto-learning*)**.

---

## 1. Arsitektur Tingkat Tinggi (High-Level Architecture)

Sistem dirancang dengan pola *Decoupled Microservice* berbasis fitur (*Layered by Feature*). Seluruh data persistensi (vektor RAG, riwayat obrolan, status job OCR, persona siswa) disimpan terpusat di **DataStax Astra DB Cloud**, sehingga server GPU sewaan berstatus *stateless* dan aman dimatikan/dihapus kapan saja tanpa kehilangan data.

```mermaid
flowchart TD
    subgraph Clients ["Lapisan Klien / Pengguna"]
        Siswa["Siswa (Web Browser / Mobile)"]
        Guru["Guru / Admin Kelas"]
    end

    subgraph Gateway ["Gateway & Backend Integrasi"]
        MainBackend["Backend Utama (TypeScript / Go)<br>Manajemen Pengguna, Otentikasi & Sesi"]
        Tunnel["Cloudflare Tunnel / Reverse Proxy<br>HTTPS: ai.domainanda.com"]
    end

    subgraph AIServer ["Server GPU Stateless (RTX 5090 / Cloud GPU)"]
        subgraph FastAPIProcess ["Proses Utama FastAPI (Port 8000)"]
            APIRouter["API Gateway & Auth Guard<br>(X-API-Key & X-Admin-Key)"]
            ChatFeature["Fitur Chat & Pedagogi<br>(Umum & Terstruktur)"]
            RAGFeature["Fitur RAG & Scoped Retrieval<br>(Option 1: User vs Global)"]
            AutoLearn["Guarded Auto-Learner<br>(5-Stage KKA Filter Gate)"]
            QwenEngine["Qwen2.5-3B + LoRA Adapter<br>(GPU Lock & Inference)"]
        end

        subgraph OCRProcess ["Proses Internal OCR (Port 8001)"]
            VLLM["vLLM Serving Engine<br>Baidu Unlimited-OCR<br>(R-SWA, VRAM ~45%)"]
        end
    end

    subgraph AstraCloud ["DataStax Astra DB (Cloud Keyspace: rag)"]
        CollRAG[("Collection: rag_chatbot<br>• Vektor Materi PDF & Unlimited-OCR<br>• Catatan Auto-Learned Global<br>• Server-Side $vectorize")]
        CollJobs[("Collection: ingestion_jobs<br>• Status Antrean OCR & Parsing<br>• Pelacakan Halaman & Chunk")]
        CollChat[("Collection: chat_history<br>• Sesi & Riwayat Obrolan Siswa<br>• Repeated Question Cache")]
        CollUser[("Collection: user_metadata<br>• Persona Gaya Siswa<br>• Memori Pengetahuan Personal")]
    end

    Siswa & Guru --> MainBackend
    MainBackend --> Tunnel
    Tunnel --> APIRouter

    APIRouter --> ChatFeature
    APIRouter --> RAGFeature

    ChatFeature --> QwenEngine
    ChatFeature --> RAGFeature
    ChatFeature <--> CollChat
    ChatFeature <--> CollUser

    RAGFeature -->|"1. Kirim Halaman Gambar"| VLLM
    VLLM -->|"2. Hasil Bersih Markdown"| RAGFeature
    RAGFeature -->|"3. Catat Status Job"| CollJobs
    RAGFeature -->|"4. Chunking & Insert $vectorize"| CollRAG
    RAGFeature <-->|"5. Scoped Vector Search"| CollRAG

    ChatFeature -.->|"Diskusi KKA Baru (Background)"| AutoLearn
    AutoLearn -->|"Kurasi & Netralisasi"| QwenEngine
    AutoLearn -->|"Simpan Ringkasan Global"| CollRAG
```

---

## 2. Alur Kerja Hulu ke Hilir (End-to-End Workflows)

### Alur A: Unggah Dokumen PDF/Materi Siswa via Unlimited-OCR $\rightarrow$ Astra DB

Alur ini memproses materi PDF digital maupun scan/foto menjadi potongan teks terstruktur yang langsung dapat dipanggil saat siswa bertanya.

```mermaid
sequenceDiagram
    autonumber
    participant U as Siswa / Guru (Web)
    participant B as Backend TS/Go
    participant API as FastAPI kka_lab (:8000)
    participant P as PyMuPDF Renderer
    participant OCR as Unlimited-OCR (:8001)
    participant C as Markdown Chunker
    participant Astra as Astra DB (Keyspace: rag)

    U->>B: Unggah PDF Materi Pelajaran
    B->>API: POST /api/rag/documents (Multipart: file, user_id, scope, X-API-Key)
    API->>API: Validasi metadata, ukuran maks (20MB), & tipe file
    API->>Astra: Simpan ke collection 'ingestion_jobs' (status: queued)
    API-->>B: 202 Accepted (job_id, doc_id, status: queued)
    B-->>U: Tampilkan indikator "Sedang Diproses"

    Note over API,Astra: Dijalankan di Background Task
    API->>Astra: Update 'ingestion_jobs' (status: processing)
    API->>P: Render lembar PDF menjadi gambar PNG (150 DPI)
    P-->>API: Daftar RenderedPage (page_0001.png, ...)

    loop Tiap Halaman Dokumen
        API->>OCR: POST /v1/chat/completions (Image Data URL + Prompt)
        OCR-->>API: Teks Markdown Mentah (Tabel, Rumus, & Heading)
        API->>API: Bersihkan token khusus (<｜end▁of▁sentence｜>, codeblock wrapper)
    end

    API->>C: chunk_markdown_pages() berdasarkan hierarki Heading
    Note over C: Jaga keutuhan tabel (|---|) & rumus ($$...$$), target ~300 token/chunk
    C-->>API: Daftar DocumentChunk (text, topic, pages_pdf)

    API->>Astra: collection 'rag_chatbot'.insert_many() ("text" & "$vectorize")
    Note over Astra: Astra DB otomatis menghasilkan vektor embedding server-side
    Astra-->>API: Konfirmasi chunk tersimpan

    API->>Astra: Update 'ingestion_jobs' (status: completed, total_pages, chunks)
    B->>API: GET /api/rag/jobs/{job_id}
    API-->>B: Status: completed
    B-->>U: Notifikasi "Materi Siap Digunakan!"
```

---

### Alur B: Percakapan Pembelajaran dengan Scoped RAG

Alur tanya-jawab yang menerapkan isolasi dokumen (Opsi 1: privat per siswa + materi global kelas) dan prinsip bimbingan bertahap (*pedagogical scaffolding*).

```mermaid
flowchart TD
    Q["Pertanyaan Siswa Masuk<br>POST /api/chat"] --> S1["Deteksi Bahasa (id / en)<br>& Resolusi Topik Percakapan"]
    S1 --> S2["Cek Pertanyaan Berulang & Riwayat<br>di Astra DB (Collection: chat_history)"]
    S2 --> S3["Pencarian Memori Personal Siswa<br>di Astra DB (Collection: user_metadata)"]
    
    S3 --> S4["Terjemahan Kueri Bilingual<br>(ID <--> EN via Qwen Lokal)"]
    S4 --> S5["Retrieval ke Astra DB (rag_chatbot)<br>dengan Filter Scope Terisolasi"]
    
    subgraph ScopeFilter ["Logika Scope Filter (Opsi 1)"]
        F1["metadata.scope: global (Materi Resmi Kelas)"]
        F2["metadata.scope: null (Materi Bawaan Lama)"]
        F3["metadata.scope: user DAN uploaded_by: siswa_id"]
    end
    S5 --- ScopeFilter

    S5 --> S6["Kompilasi Konteks Terurut<br>build_context() maks 900 token"]
    S6 --> S7["Muat Persona Gaya Siswa<br>dari Astra DB (user_metadata)"]
    S7 --> S8["Penyusunan System Prompt & Scaffolding<br>(Mode: Umum / Terstruktur)"]
    S8 --> S9["Penyesuaian Anggaran Token<br>fit_messages_to_budget()"]
    
    S9 --> S10["Inferensi Qwen2.5-3B (+ LoRA Adapter)<br>Dieksekusi dalam GPU Thread Lock"]
    S10 --> S11["Pengecekan Konsistensi Bahasa<br>ensure_response_language()"]
    S11 --> S12["Simpan Pertukaran Chat ke Astra DB<br>(chat_history & memory_hit flag)"]
    S12 --> RES["Kirim Respons JSON ke Siswa<br>(Jawaban, Sumber [S1], Session ID)"]
```

---

### Alur C: Kurasi Otomatis Materi KKA Baru dari Diskusi (Pilihan B)

Mekanisme *guarded auto-learning* yang menyaring dialog siswa secara pasif dan menyerap materi KKA baru yang belum tercakup di Astra DB tanpa campur tangan manual.

```mermaid
flowchart TD
    ChatDone["Sesi Chat Siswa Selesai<br>(Background Task Terpanggil)"] --> G1{"Lapis 1:<br>Anti-Racun & Injeksi"}
    G1 -->|"Ada pola jailbreak, dikte palsu,<br>atau kata kasar"| Reject["Abaikan & Buang"]
    
    G1 -->|"Bersih"| G2{"Lapis 2:<br>Kualitas & Substansi"}
    G2 -->|"Jawaban < 120 char,<br>sapaan, atau AI ragu/tidak yakin"| Reject
    
    G2 -->|"Substansial & Jelas"| G3{"Lapis 3:<br>Domain Kurikulum KKA"}
    G3 -->|"Topik non-KKA<br>(Sejarah, Biologi, Curhat, Game)"| Reject
    
    G3 -->|"Memuat kosakata teknis KKA<br>(Algoritma, Python, ML, Data, dll)"| G4{"Lapis 4:<br>Anti-Duplikat RAG"}
    G4 -->|"Similarity RAG >= 0.82<br>atau Topik sudah ada di Astra DB"| Reject
    
    G4 -->|"Similarity < 0.82<br>(Materi KKA Baru)"| G5["Lapis 5: Kurator & Netralisasi Qwen<br>(Prompt Hakim Buku Ajar KKA)"]
    
    G5 --> Validate{"Validasi Hasil JSON Kurator"}
    Validate -->|"Gagal / Tidak Disetujui"| Reject
    Validate -->|"Disetujui (approved: true)"| Distill["Hasil Ekstraksi Netral:<br>1. Heading '## Judul Sub-Topik'<br>2. Intisari Faktual (2-5 kalimat)<br>3. Bersih dari Persona, Sapaan, & Tanya Balik"]
    
    Distill --> Ingest["Simpan ke Astra DB (rag_chatbot)<br>scope: global<br>source_title: 'Catatan Materi KKA (Diskusi Terkurasi)'"]
    Ingest --> LogDB["Catat di Astra DB (ingestion_jobs / auto_learned)<br>Siap di-audit/dihapus Guru sewaktu-waktu"]
    LogDB --> Ready[("Materi KKA Baru Aktif<br>Bisa Dipakai Seluruh Kelas")]
```

---

## 3. Struktur Direktori dan Tanggung Jawab Modul

Proyek mengadopsi struktur *Layered by Feature*:

```text
kka_lab/
├── app/
│   ├── core/                        # Utilitas inti sistem
│   │   ├── config.py                # Pemuat model_record.json & konfigurasi env
│   │   └── __init__.py
│   ├── features/                    # Domain modular
│   │   ├── chat/                    # Interaksi percakapan
│   │   │   ├── assistant.py         # Inferensi Qwen2.5-3B, kuantisasi NF4, LoRA loader, gpu_lock
│   │   │   ├── prompts.py           # System prompts, budgeting token, penyesuaian konteks
│   │   │   └── router.py            # POST /api/chat
│   │   ├── ocr/                     # Pemrosesan dokumen & OCR
│   │   │   ├── client.py            # Client Baidu Unlimited-OCR (vLLM / Transformers / Native PDF)
│   │   │   ├── exceptions.py        # Exception khusus validasi dokumen & engine OCR
│   │   │   └── pdf_renderer.py      # Render PDF ke PNG beresolusi tinggi (PyMuPDF)
│   │   ├── rag/                     # Retrieval-Augmented Generation (Astra DB)
│   │   │   ├── auto_learner.py      # Saringan 5-lapis kurasi otomatis diskusi KKA (Pilihan B)
│   │   │   ├── chunker.py           # Pemotong Markdown atomik berbasis heading, tabel, & formula
│   │   │   ├── context.py           # Context builder, penyusun sitasi [S1], topic resolver
│   │   │   ├── exceptions.py        # Penanganan error transient Astra DB (503 hibernation hint)
│   │   │   ├── ingestor.py          # Operasi insert_many $vectorize & delete berulang dengan retry
│   │   │   ├── retriever.py         # Multi-query bilingual search & scope filtering
│   │   │   └── router.py            # Endpoints: /api/rag/documents, /jobs, /auto-learned
│   │   ├── memory/                  # Manajemen sesi percakapan & penyimpanan dokumen Astra DB
│   │   │   ├── gate.py              # Quality gate heuristik untuk seleksi kandidat dialog
│   │   │   ├── router.py            # Endpoints personal knowledge (/api/users/{id}/knowledge)
│   │   │   └── store.py             # Store wrapper untuk Astra DB Data API
│   │   ├── persona/                 # Preferensi karakter/gaya bahasa siswa
│   │   │   ├── router.py            # Endpoints CRUD persona siswa
│   │   │   └── validator.py         # Penyaring prompt injection, bypass pedagogis, & kata kasar
│   │   ├── language/                # Dukungan bilingual (Indonesia & English)
│   │   │   ├── detector.py          # Heuristik deteksi bahasa pertanyaan
│   │   │   ├── translator.py        # Penerjemah kueri pencarian & penjaga konsistensi bahasa
│   │   │   └── locales/             # Kamus aturan pedagogis, stop words, & teks prompt (id, en)
│   │   └── dataset/                 # Validasi dataset pelatihan LoRA
│   │       ├── tokenizer.py         # Masking target loss & tokenisasi dialog
│   │       └── validator.py         # Pengecekan syarat struktural minimal 2 langkah & pemantik
│   └── main.py                      # Inisialisasi FastAPI, CORS, lifespan, & registrasi router
├── data/                            # File dataset contoh penelitian
├── notebooks/                       # Notebook Jupyter untuk pelatihan LoRA
│   ├── Training_LoRA_CloudGPU.ipynb # Khusus GPU sewaan (RTX 5090 / CUDA 12.8+, bf16, auto-check)
│   └── Training_LoRA_Colab.ipynb    # Khusus Google Colab gratisan (T4 16GB, Google Drive)
├── scripts/                         # Skrip operasional CLI
│   ├── check_environment.py         # Diagnostik PyTorch, CUDA, VRAM, dan library
│   ├── download_model.py            # Download base model Qwen2.5-3B-Instruct
│   ├── download_ocr_model.py        # Download bobot Baidu Unlimited-OCR ke Hugging Face cache
│   └── serve_ocr_vllm.sh            # Menjalankan server vLLM Unlimited-OCR di port 8001
├── training/                        # Skrip training LoRA
│   └── train_lora.py                # Trainer LoRA / QLoRA berparameter presisi (fp16/bf16, batch)
├── requirements-core.txt            # Dependensi dasar inferensi chat & API
├── requirements-ocr.txt             # Dependensi pengolahan PDF, image, & multipart upload
└── PANDUAN_CLOUD_GPU.md             # Panduan operasional langkah-demi-langkah deployment server
```

---

## 4. Skema Koleksi Astra DB (Cloud Database)

Seluruh entitas sistem disimpan di dalam **DataStax Astra DB** (Keyspace: `rag`). Tidak ada penyimpanan basis data di disk lokal server GPU, sehingga instance bersifat *stateless*.

```mermaid
classDiagram
    class Collection_rag_chatbot {
        string _id
        string text
        string $vectorize
        string source_title
        string topic
        variant pages_pdf
        object metadata
    }
    class Collection_ingestion_jobs {
        string _id
        string job_id
        string doc_id
        string user_id
        string filename
        string source_title
        string topic
        string scope
        string status
        int total_pages
        int chunks_inserted
        string error_message
        string created_at
        string updated_at
    }
    class Collection_chat_history {
        string _id
        string user_id
        string session_id
        string question
        string normalized_question
        string answer
        string mode
        list sources
        string training_status
        string created_at
    }
    class Collection_user_metadata {
        string _id
        string user_id
        string type
        string persona
        string question
        string answer
        string updated_at
    }

    Collection_ingestion_jobs ..> Collection_rag_chatbot : menghasilkan chunk
    Collection_chat_history ..> Collection_rag_chatbot : referensi sumber [S1]
```

### Rincian Koleksi di Astra DB:

1. **`rag_chatbot` (Vektor RAG Collection):**
   * Mengaktifkan server-side `$vectorize`.
   * Berisi pecahan chunk dari materi PDF siswa/guru dan intisari diskusi KKA terkurasi.
   * `metadata`: `{ doc_id, scope: "user"|"global", uploaded_by, chunk_index, filename, origin }`.
2. **`ingestion_jobs` (Dokumen Antrean Job):**
   * Melacak progres pemrosesan OCR latar belakang (`queued` $\rightarrow$ `processing` $\rightarrow$ `completed` / `failed`).
   * Bisa dipantau bersama oleh backend TypeScript/Go tanpa perlu mengakses filesystem GPU.
3. **`chat_history` (Riwayat Sesi Obrolan):**
   * Menyimpan log tanya-jawab, deteksi pertanyaan berulang (*repeated question cache*), dan persetujuan data latih.
4. **`user_metadata` (Persona & Memori Personal):**
   * Menyimpan preferensi persona gaya belajar siswa dan memori personal (`user_knowledge`).

---

## 5. Keamanan, Isolasi, dan Alokasi Sumber Daya GPU

1. **Service-to-Service Security:**
   * Seluruh rute API dilindungi pencocokan timing-safe header `X-API-Key` dengan variabel lingkungan `AI_INTERNAL_SECRET`.
   * Operasi administratif (pengunggahan materi `global`, audit auto-learned, dan penghapusan materi kelas) wajib menyertakan `X-Admin-Key` (`RAG_ADMIN_SECRET`).
2. **Isolasi Data Antar-Siswa (Multi-Tenancy di Astra DB):**
   * Siswa B tidak akan pernah bisa mengakses atau membaca materi privat Siswa A melalui filter `$or` terisolasi di `retriever.py`:
     ```json
     {
       "$or": [
         { "metadata.scope": { "$exists": false } },
         { "metadata.scope": "global" },
         { "metadata.scope": "user", "metadata.uploaded_by": "siswa_101" }
       ]
     }
     ```
   * Penghapusan dokumen siswa diverifikasi kepemilikannya berdasarkan `user_id`.
3. **Pencegahan Concurrency Conflict & OOM di GPU:**
   * **Proses Terpisah:** vLLM Unlimited-OCR dialokasikan maksimal **45% VRAM (~14 GB pada RTX 5090)** di port 8001. Sisa **~18 GB VRAM** dialokasikan khusus untuk Qwen2.5-3B di port 8000.
   * **Thread Lock:** Seluruh generasi teks Qwen dan penerjemahan kueri diatur menggunakan `self.gpu_lock` (`threading.RLock`) pada `LocalAssistant`, mencegah dua thread FastAPI melakukan komputasi serentak yang memicu *CUDA out-of-memory*.
4. **Keuntungan Bebas SQLite di Cloud GPU:**
   * Karena semua data disimpan di Astra DB Cloud, instance GPU sewaan tidak menyimpan status apa pun (*stateless*). Jika Anda mematikan instance GPU untuk menghemat biaya sewa dan menyewa GPU baru di kemudian hari, **semua riwayat chat, materi PDF, dan data persona tetap utuh di Astra DB!**
