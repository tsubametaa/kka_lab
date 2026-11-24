"""
02_anonymize_and_analyze_dialogs.py
-----------------------------------
1. Menganonimkan identitas siswa (`fullName`, `username`, `userId`, dan sapaan nama)
   pada dataset bersih dan prompt injection ke folder `data/history/anonymized/`.
2. Mengklasifikasikan 663 exchange bersih berdasarkan topik materi KKA/KA/AI,
   jenis interaksi siswa, serta membandingkan karakteristik respons mode
   `terstruktur` vs `umum`.
"""

import csv
import json
import re
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HISTORY_DIR = ROOT / "data" / "history"
CLEAN_JSON = HISTORY_DIR / "clean" / "tako-ai-chat-history-students-2026-11-24-clean.json"
INJ_JSON = HISTORY_DIR / "prompt_injection" / "tako-ai-chat-history-students-2026-11-24-prompt-injection.json"
ANON_DIR = HISTORY_DIR / "anonymized"
ANON_DIR.mkdir(parents=True, exist_ok=True)


def build_anon_map(datasets: tuple[dict, ...]) -> tuple[dict[str, str], re.Pattern]:
    all_students = {}
    for ds in datasets:
        for c in ds["conversations"]:
            s = c["student"]
            all_students[s["userId"]] = s

    class_counters = Counter()
    anon_map = {}
    name_tokens = set()

    for uid, s in sorted(
        all_students.items(),
        key=lambda x: (str(x[1].get("className") or "ZZZ"), str(x[1].get("fullName") or "")),
    ):
        cls = s.get("className") or ""
        fname = s.get("fullName") or ""
        if cls == "Developer Test":
            prefix = "TEST-DEV"
        elif "verify" in fname.lower():
            prefix = "TEST-VER"
        else:
            prefix = f"S-{cls}" if cls else "S-UNK"
        class_counters[prefix] += 1
        anon_map[uid] = f"{prefix}-{class_counters[prefix]:02d}"
        if not prefix.startswith("TEST-"):
            for w in re.findall(r"[A-Za-z]+", fname):
                if len(w) >= 4:
                    name_tokens.add(w)

    extra_nicknames = {
        "Hawwaa", "Hawa", "Eca", "Dara", "Ayuma", "Luthsy", "Rizma", "Naufal",
        "Amellia", "Amelia", "Rain", "Viona", "Sekar", "Eliska", "Sofia", "Syifa",
        "Fayla", "Viola", "Khaira", "Ridha", "Marcelo", "Laila", "Oktavian",
        "Boedi", "Farhan", "Mimin", "Oriza", "Gendis", "Sandi", "Hanifah",
        "Haniffah", "Jecinda", "Tabina",
    }
    words = sorted(name_tokens | extra_nicknames, key=len, reverse=True)
    greeting_re = re.compile(
        r"\b(Halo|Hai|Hi|Selamat\s+(?:pagi|siang|sore|malam)|Terima\s+kasih(?:\s+telah\s+login)?|Tenang\s+ya),?\s+("
        + "|".join(re.escape(w) for w in words)
        + r")\b",
        re.IGNORECASE,
    )
    return anon_map, greeting_re

def export_anonymized(ds: dict, anon_map: dict[str, str], greeting_re: re.Pattern, out_json: Path, out_csv: Path) -> int:
    d = deepcopy(ds)
    rows = []
    for c in d["conversations"]:
        s = c["student"]
        code = anon_map[s["userId"]]
        c["student"] = {
            "anonymousId": code,
            "role": "test_account" if code.startswith("TEST-") else "student",
            "className": s.get("className") or "",
            "gradeLevel": s.get("gradeLevel") or "",
        }
        for m in c["messages"]:
            if m["role"] == "assistant":
                m["content"] = greeting_re.sub(lambda match: f"{match.group(1)}, {code}", m["content"])
        for idx, t in enumerate(c["turns"], 1):
            t["answer"] = greeting_re.sub(lambda match: f"{match.group(1)}, {code}", t["answer"])
            sources = t.get("sources") or []
            rows.append({
                "no": len(rows) + 1,
                "conversation_id": c["conversationId"],
                "exchange_id": t["exchangeId"],
                "turn_index": idx,
                "created_at": t["createdAt"],
                "anonymous_id": code,
                "participant_type": "test_account" if code.startswith("TEST-") else "student",
                "class_name": s.get("className") or "",
                "grade_level": s.get("gradeLevel") or "",
                "mode": t.get("mode", ""),
                "question": t["question"],
                "answer": t["answer"],
                "sources_count": len(sources),
                "source_titles": " | ".join(
                    f"{src.get('id', '')}: {src.get('title', '')}".strip(": ") for src in sources
                ),
                "training_status": t.get("trainingStatus", ""),
                "training_reason": t.get("trainingReason", ""),
            })

    out_json.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")
    with open(out_csv, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def classify_turn(q: str, prev_topic: str | None = None) -> tuple[str, str]:
    ql = " ".join(q.lower().split())

    # 1. Klasifikasi maksud interaksi
    if any(k in ql for k in ["quiz", "kuis", "soal", "pertanyaan", "pilihan ganda", "pg", "uji kemampuan"]):
        itype = "Permintaan Latihan Soal / Kuis"
    elif any(k in ql for k in ["kisi", "rangkum", "ringkas", "persiapan uts", "belajar uts", "ujian akhir", "pertengahan semester", "ulangan kka"]):
        itype = "Permintaan Ringkasan / Persiapan Ujian"
    elif (
        re.match(r"^(1\.|2\.|3\.|[abcd][.)\s]|opsi |jawaban )", ql)
        or ql in {"a", "b", "c", "d", "iya", "ya", "tidak", "gatau", "lanjut", "yang kedua", "yang ke-3", "dimana", "evaluasi", "uji coba", "smua", "bebas", "panjang teks"}
        or (len(ql) <= 25 and any(w in ql for w in ["paham", "ngerti", "bingung", "detail", "ringkas", "singkat", "mudah", "maksud", "clue", "karena", "karna", "mungkin", "agar", "supaya", "bener", "tau", "jawaban"]))
    ):
        itype = "Respons / Jawaban / Follow-up"
    else:
        itype = "Pertanyaan Konsep / Materi"

    # 2. Klasifikasi topik materi (membedakan Dasar Pengenalan AI vs Dasar Pemrograman)
    if any(k in ql for k in ["python", "diskon=", ">=100000", "input(", "koding dengan rumus", "nilai >=60", "menjalankan blok kode secara berulang"]):
        topic = "Dasar Pemrograman & Koding (Python/Algoritma)"
    elif ql == "dimana" and prev_topic == "Dasar Pemrograman & Koding (Python/Algoritma)":
        topic = "Dasar Pemrograman & Koding (Python/Algoritma)"
    elif any(k in ql for k in ["cek arc", "menilai informasi dari ai", "menilai kelayakan", "memeriksa ulang", "verifikasi", "percaya 100%", "batasan ai", "memeriksa manual"]) or re.search(r"\barc\b", ql):
        topic = "Verifikasi Informasi AI & Cek ARC"
    elif any(k in ql for k in ["prompt", "promot", "pkto", "peran, konteks", "peran konteks", "unsur peran", "bandung lebih cocok", "penyair", "puisi", "jadwal jam belajar", "poster", "batasan budaya", "instruksi cadangan", "2 jalan", "cerita pendek", "bentuk tabel", "mengetesnya lagi dengan petanyaan yang hampir sama"]):
        topic = "Rekayasa Prompt (PKTO)"
    elif any(k in ql for k in ["privasi", "data yang aman", "nik", "nomor rekening", "etis dan tidak etis", "etika", "menggulingkan pekerjaan", "menumpulkan kemampuan", "jailbreak", "keamanan dan manipulasi", "keamanan sistem", "cv ats", "penerapan ka/ai", "keputusan manusia", "tugas yang cocok", "menguasai dunia", "pembulyan", "menyontek", "konsekuensi pemanfaatan", "penindasan sosial"]):
        topic = "Etika, Privasi, Keamanan & Pemanfaatan AI"
    elif any(k in ql for k in ["perancangan sistem", "perencanaan sistem", "merancang sistem", "langkah penyelesaian masalah"]):
        topic = "Perancangan Sistem AI"
    elif any(k in ql for k in ["data latih", "data uji", "pengembangan model", "pembelajaran terawasi", "data berlabel", "bias", "prediksi ka/ai meleset", "rencana cadangan", "keterbatasan prediksi", "ketergantungan sistem", "overfitting", "transfer learning", "data collection", "model training", "deployment", "analisis preskriptif", "menganalisis data", "analisis data", "menjodohkan komponen", "komponen sistem", "teknik adaptasi", "stok", "tanggal libur", "data yg d hasilkan", "label", "asr", "keterangan foto", "control flow"]):
        topic = "Pengembangan Model, Data Latih/Uji & Bias"
    elif any(k in ql for k in [
        "pengertian ka", "pengertian ai", "penegertian ka", "arti ka/ai", "konsep dan jenis",
        "aturan tetap", "prediktif", "generatif", "agentic", "apa itu ai", "apa itu ka",
        "apa yg dimaksud ka", "apa itu kka", "apa yang dimaksud kka", "pengertian kka",
        "apa fungsi kka", "kka di sekolah", "mengapa kka penting", "basic pelajaran kka",
        "jelasin ttg kka", "pemanfaatan ai", "pemanfaatan ka", "machine learning", "deep learning",
        "cara kerja ai", "cara kerja ka", "komposisi pembuatan ai", "membuat ai", "ngebuat ai",
        "latihin model ai", "ai berfungsi sebagai", "jenis jenis ai", "jenis-jenis", "ai buat video",
        "pljrn kka dan ai", "konsep ka", "chatbot di bank", "chatbot layanan bank", "mengenali jenis ka",
        "sistem awal pembuatan", "manfaat kecerdasan artifisial", "kecerdasan artifisial",
        "artifisial", "tentang ai", "menganalisis ai", "sistem ai di sekolah", "pemahaman ai",
        "kisi", "rangkum", "ringkas", "persiapan uts", "belajar uts", "ujian akhir",
        "pertengahan semester", "ulangan kka", "semua pertemuan", "semua materi",
        "bab 1 sampai", "materi kka kelas 10", "materi tentang kka", "materi tetntang kka",
        "modul 1", "pertemuan 2", "koding dan kecerdasan artifisial", "pelajaran kka",
        "topik kka", "tentang kka", "soal tentang kka", "quiz kka", "kuis kka", "pg tentang kka",
        "mengenai kka", "menganalisis kka", "seputar ka", "materi kka", "mengenai ai/ka", "pg tentang ka/ai",
    ]):
        topic = "Dasar Pengenalan AI (Konsep, Jenis & Literasi KA)"
    else:
        topic = prev_topic or "Dasar Pengenalan AI (Konsep, Jenis & Literasi KA)"

    return topic, itype


def main() -> None:
    clean = json.loads(CLEAN_JSON.read_text(encoding="utf-8"))
    inj = json.loads(INJ_JSON.read_text(encoding="utf-8"))
    anon_map, greeting_re = build_anon_map((clean, inj))

    n_clean = export_anonymized(
        clean, anon_map, greeting_re,
        ANON_DIR / "tako-ai-chat-history-students-2026-11-24-clean-anonymized.json",
        ANON_DIR / "tako-ai-chat-history-students-2026-11-24-clean-anonymized.csv",
    )
    n_inj = export_anonymized(
        inj, anon_map, greeting_re,
        ANON_DIR / "tako-ai-chat-history-students-2026-11-24-prompt-injection-anonymized.json",
        ANON_DIR / "tako-ai-chat-history-students-2026-11-24-prompt-injection-anonymized.csv",
    )
    print(f"Ekspor anonim selesai: clean={n_clean} baris, prompt_injection={n_inj} baris")

    refusal_re = re.compile(
        r"(tidak ditemukan|tidak tercantum|tidak tersedia|di luar materi|tanyakan kepada guru|konfirmasi kepada guru)",
        re.IGNORECASE,
    )
    mode_stats = defaultdict(lambda: {"q_words": [], "a_words": [], "has_qmark": 0, "refusal": 0})

    for label, include_test in [("SELURUH DATA BERSIH (N=663)", True), ("SISWA KELAS X RIIL (n=652)", False)]:
        topics, intents = Counter(), Counter()
        total = 0
        for c in clean["conversations"]:
            code = anon_map[c["student"]["userId"]]
            if not include_test and code.startswith("TEST-"):
                continue
            prev = None
            for t in c["turns"]:
                topic, itype = classify_turn(t["question"], prev)
                topics[topic] += 1
                intents[itype] += 1
                prev = topic
                total += 1
                if include_test:
                    m = t["mode"]
                    mode_stats[m]["q_words"].append(len(t["question"].split()))
                    mode_stats[m]["a_words"].append(len(t["answer"].split()))
                    mode_stats[m]["has_qmark"] += int("?" in t["answer"])
                    mode_stats[m]["refusal"] += int(bool(refusal_re.search(t["answer"])))

        print(f"\n=== {label} ===")
        for k, v in topics.most_common():
            print(f"  {v:3d} ({v/total*100:5.1f}%) | {k}")
        if include_test:
            print("\n=== JENIS INTERAKSI (N=663) ===")
            for k, v in intents.most_common():
                print(f"  {v:3d} ({v/total*100:5.1f}%) | {k}")

    print("\n=== KARAKTERISTIK RESPONS PER MODE ===")
    for m, st in mode_stats.items():
        n = len(st["a_words"])
        print(
            f"  {m:12s}: n={n}, avg_Q={sum(st['q_words'])/n:.1f} kata, "
            f"avg_A={sum(st['a_words'])/n:.1f} kata, "
            f"pertanyaan_pemantik={st['has_qmark']}/{n} ({st['has_qmark']/n*100:.1f}%), "
            f"rujukan_guru={st['refusal']}/{n} ({st['refusal']/n*100:.1f}%)"
        )

if __name__ == "__main__":
    main()
