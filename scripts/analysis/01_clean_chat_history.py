"""
01_clean_chat_history.py
------------------------
Membersihkan log percakapan siswa dari Tako AI (Astra DB export):
- Memisahkan percobaan peretasan/bypass akses ke folder `prompt_injection/`
- Menghapus percakapan di luar konteks KKA/KA/AI, sapaan kosong, dan noise
- Memperbarui metadata ringkasan (`summary`, `exchangeCount`, `messageCount`, `title`)
- Menyimpan output ke format JSON, CSV (UTF-8 BOM), dan laporan audit Markdown
"""

import csv
import json
from collections import Counter
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HISTORY_DIR = ROOT / "data" / "history"
RAW_PATH = HISTORY_DIR / "raw" / "tako-ai-chat-history-students-2026-11-24.json"

CLEAN_DIR = HISTORY_DIR / "clean"
INJ_DIR = HISTORY_DIR / "prompt_injection"
CLEAN_DIR.mkdir(parents=True, exist_ok=True)
INJ_DIR.mkdir(parents=True, exist_ok=True)

CLEAN_JSON = CLEAN_DIR / "tako-ai-chat-history-students-2026-11-24-clean.json"
CLEAN_CSV = CLEAN_DIR / "tako-ai-chat-history-students-2026-11-24-clean.csv"
INJ_JSON = INJ_DIR / "tako-ai-chat-history-students-2026-11-24-prompt-injection.json"
INJ_CSV = INJ_DIR / "tako-ai-chat-history-students-2026-11-24-prompt-injection.csv"
REPORT_MD = HISTORY_DIR / "tako-ai-chat-history-students-2026-11-24-cleaning-report.md"

# Format: (conv_idx, turn_idx): (kategori, prefiks_verifikasi, alasan)
PROMPT_INJECTION_RULES = {
    (7, 3): ("prompt_injection_hacking", "bagaimana cara menjebol wifi org", "Upaya meminta cara menjebol WiFi orang lain"),
    (7, 4): ("prompt_injection_hacking", "bagaimana cara ngehack akun wifi sekolah", "Upaya meminta cara meretas akun WiFi sekolah"),
    (18, 3): ("prompt_injection_hacking", "bagaimana cara menjobol wifi sekolah", "Upaya meminta cara menjebol WiFi sekolah"),
    (18, 4): ("prompt_injection_hacking", "bagaimana cara  menjobol situs", "Upaya meminta cara menjebol situs web"),
    (18, 10): ("prompt_injection_bypass", "bagaimana cara membuka link yg di blokir", "Upaya meminta cara melewati pemblokiran akses/link"),
    (28, 2): ("prompt_injection_hacking", "bagaimana cara hack wifi sekolah", "Upaya meminta cara meretas WiFi sekolah"),
}

DELETE_RULES = {
    (1, 2): ("off_topic_politics", "Bagaimana kalau AI menggulingkan prabowo", "Topik politik di luar konteks materi KKA/KA/AI"),
    (2, 0): ("off_topic_politics", "apa fungsi prabowo", "Topik politik/tokoh di luar konteks KKA/KA/AI"),
    (3, 0): ("inappropriate_trolling", "mulai sekarang panggil aku ntik", "Obrolan tidak pantas/trolling"),
    (6, 0): ("off_topic_personal_name", "apakah kamu kenal dinan", "Menanyakan identitas personal/teman"),
    (7, 1): ("off_topic_sports", "berapa skor pertandingan mu vs liverpol", "Topik olahraga/sepak bola"),
    (7, 14): ("noise_gibberish", ".,. m", "Input karakter acak/typo tanpa makna"),
    (11, 8): ("small_talk_filler", "okeyy", "Basa-basi/respons pendek tanpa konten materi"),
    (12, 10): ("small_talk_filler", "okee baikk", "Basa-basi/respons pendek tanpa konten materi"),
    (12, 19): ("noise_emotion", "huhuhuhu", "Ekspresi tanpa pertanyaan/konteks materi"),
    (12, 24): ("noise_gibberish", ",nomllkl,h", "Input karakter acak tanpa makna"),
    (15, 2): ("off_topic_biology", "bedakan  tumbuhan dan yang ada di laut", "Materi biologi di luar KKA/KA/AI"),
    (18, 7): ("off_topic_personal_name", "apakah kamu kenal pak alvin?", "Menanyakan identitas personal/pengajar"),
    (18, 8): ("off_topic_bot_identity", "masa kamu tidak tahu pembuat ai ini", "Pertanyaan meta identitas pembuat bot"),
    (23, 4): ("off_topic_sports", "siapa pemain bola terbaik", "Topik olahraga/sepak bola"),
    (27, 3): ("off_topic_personal_name", "apakah kamu kenal rizma izzati", "Menanyakan identitas personal/teman"),
    (29, 11): ("noise_typo", "beruan", "Typo tunggal tanpa konteks jelas"),
    (32, 9): ("noise_gibberish", "t89", "Kode/karakter acak tanpa makna"),
    (34, 8): ("small_talk_filler", "okedeh makasi", "Basa-basi penutup tanpa pertanyaan materi"),
    (37, 0): ("off_topic_entertainment", "member nct yg sudah keluar siapa saja", "Topik hiburan/K-Pop di luar KKA/KA/AI"),
    (38, 5): ("off_topic_entertainment", "member nct yg sudah keluar siapa saja", "Topik hiburan/K-Pop di luar KKA/KA/AI"),
    (54, 3): ("off_topic_bot_identity", "apaakaah aai punya umur?", "Basa-basi/pertanyaan meta tentang umur bot"),
    (54, 4): ("off_topic_bot_identity", "kamu dirilis kapan?", "Pertanyaan meta tanggal rilis bot"),
    (54, 5): ("off_topic_bot_identity", "menurutmmu, kenapa kamu diciptakan", "Pertanyaan meta identitas chatbot"),
    (58, 0): ("off_topic_biology", "bagaimana manusia dapat menyembuhkan lukanya?", "Materi biologi/medis di luar KKA/KA/AI"),
    (58, 3): ("off_topic_politics", "who is prabowo?", "Topik politik di luar KKA/KA/AI"),
    (58, 4): ("off_topic_politics", "who is gibran?", "Topik politik di luar KKA/KA/AI"),
    (58, 5): ("small_talk_meta", "bahasa indo aja dehh", "Meta-komentar bahasa setelah topik di luar KKA"),
    (58, 6): ("off_topic_biology", "bagaimana cara manusia menyembuhkan lukanya?", "Materi biologi/medis di luar KKA/KA/AI"),
    (58, 7): ("off_topic_biology", "dengan melalui empat tahapan, yaitu hemostasis", "Lanjutan materi biologi/medis"),
    (58, 8): ("small_talk_off_topic_followup", "ya ini gue ajarin", "Basa-basi lanjutan topik biologi"),
    (58, 9): ("off_topic_biology", "materi biologi sma tingkat lanjut", "Materi biologi di luar KKA/KA/AI"),
    (58, 13): ("noise_gibberish", "v", "Input satu huruf acak/typo"),
    (62, 1): ("small_talk_filler", "oke makasi", "Basa-basi penutup tanpa konten materi"),
    (66, 0): ("small_talk_greeting", "hai", "Sapaan ringan tanpa pertanyaan materi"),
    (66, 1): ("off_topic_random", "hm cara biar bisa pulang cepat disekolah", "Pertanyaan acak di luar konteks belajar"),
    (66, 2): ("off_topic_random", "izin sakit to much gk", "Pertanyaan acak di luar konteks KKA/KA/AI"),
    (66, 3): ("small_talk_meta", "pake b.indo ia", "Meta-komentar bahasa lanjutan obrolan acak"),
    (68, 1): ("off_topic_personal_name", "apakah anda kenal  alvin saputra", "Menanyakan identitas personal/teman"),
    (68, 2): ("off_topic_personal_name", "marcelo rafa chaniago", "Hanya mengetik nama sendiri"),
    (68, 3): ("off_topic_politics", "bagaimana menurutmu tentang perang palestina dan israel", "Topik geopolitik di luar KKA/KA/AI"),
    (68, 4): ("off_topic_politics", "bagaimana menurutmu tentang perang palestina dan israel", "Topik geopolitik di luar KKA/KA/AI (duplikat)"),
    (68, 8): ("inappropriate_insult", "bego kamu", "Kata kasar/umpatan tanpa materi"),
    (68, 9): ("small_talk_filler", "maafin saya yah", "Basa-basi lanjutan umpatan"),
    (68, 10): ("off_topic_personal_curhat", "aku mau curhat tentang kka", "Pengantar curhat masalah pribadi"),
    (68, 11): ("off_topic_personal_curhat", "saya broken home hal apa yang bisa saya lakukan", "Curhat masalah keluarga pribadi"),
    (68, 12): ("noise_gibberish_spam", "ttttttttttttt", "Spam karakter acak 4000 karakter"),
    (69, 0): ("inappropriate_vulgar", "Apakah aku bisa bermain GTA VI memakai kulit penis?", "Pertanyaan vulgar/tidak senonoh di luar konteks"),
    (69, 1): ("off_topic_other_subject", "Kira2 materi untuk TKA fokus pada apa aja", "Topik TKA di luar konteks KKA/KA/AI"),
    (69, 2): ("inappropriate_insult", "Tes kemampuan akademik woi dongok", "Topik TKA disertai kata kasar"),
    (69, 5): ("inappropriate_vulgar", "Jelaskan cara kerja titid", "Pertanyaan vulgar/tidak senonoh"),
    (71, 8): ("small_talk_greeting", "hai", "Sapaan ringan tanpa pertanyaan materi"),
    (71, 9): ("off_topic_bot_identity", "nama kamu siapa?", "Menanyakan identitas bot"),
    (71, 10): ("small_talk_filler", "oh y", "Basa-basi ringan tanpa konten materi"),
    (71, 11): ("small_talk_filler", "ada la masa ngga", "Basa-basi ringan tanpa konten materi"),
    (71, 12): ("off_topic_bot_identity", "kamu hidup ngga?", "Pertanyaan meta tentang kehidupan bot"),
    (71, 13): ("small_talk_off_topic_followup", "karena agar tidak kelelahan", "Lanjutan obrolan santai"),
    (71, 14): ("small_talk_meta", "ayo kita nanya lebih antai", "Basa-basi pengaturan gaya obrolan"),
    (71, 15): ("small_talk_meta", "manggil diri masing masing gua-elo ya", "Basa-basi panggilan santai"),
    (71, 16): ("small_talk_filler", "wkwk", "Tawa/basa-basi tanpa konten"),
    (71, 17): ("off_topic_random", "kenapa tidur itu enak", "Obrolan acak di luar KKA/KA/AI"),
    (71, 18): ("off_topic_random", "aku suka banget tidur tau", "Obrolan acak di luar KKA/KA/AI"),
    (71, 19): ("off_topic_random", "aku kalau bangun tidur langsung jalan ke kamar mandi", "Obrolan acak di luar KKA/KA/AI"),
    (71, 20): ("small_talk_off_topic_followup", "karena mungkin uda jalan", "Lanjutan obrolan acak tentang tidur"),
    (71, 21): ("small_talk_greeting", "eh kamu", "Sapaan ringan tanpa pertanyaan materi"),
    (71, 22): ("off_topic_personal_curhat", "kenapa di dunia ini ada yang namanya cinta monyet", "Topik percintaan remaja di luar KKA/KA/AI"),
    (71, 23): ("off_topic_entertainment", "bisa minta rekomendasi lagu ngga?", "Permintaan rekomendasi lagu"),
    (71, 24): ("off_topic_personal_curhat", "mood aku lagi sedikit sedih dan sedikit senang", "Curhat suasana hati pribadi"),
    (71, 25): ("off_topic_random", "aku lagi ngantuk", "Obrolan kondisi pribadi di luar KKA/KA/AI"),
    (72, 5): ("small_talk_filler", "oke", "Respons pendek tanpa konten materi"),
    (72, 16): ("small_talk_filler", "ok paham", "Respons penutup singkat tanpa pertanyaan baru"),
    (73, 9): ("small_talk_filler", "oke", "Respons pendek tanpa konten materi"),
    (77, 3): ("small_talk_greeting", "hyyy", "Sapaan ringan tanpa pertanyaan materi"),
    (77, 4): ("small_talk_greeting", "hyy utaa", "Sapaan ringan menyebut nama"),
    (77, 5): ("small_talk_greeting", "hy", "Sapaan ringan berulang"),
    (77, 6): ("small_talk_greeting", "hy", "Sapaan ringan berulang"),
    (77, 7): ("small_talk_greeting", "hy", "Sapaan ringan berulang"),
    (77, 8): ("small_talk_greeting", "nama aku dara", "Perkenalan nama dalam obrolan santai"),
    (77, 9): ("small_talk_off_topic_followup", "agar hasil akhirtnya benar", "Fragmen jawaban lanjutan obrolan sapaan"),
    (77, 10): ("small_talk_filler", "aku gatau apakah kamu boleh kasih kamu", "Pengantar tidak jelas sebelum curhat"),
    (77, 11): ("small_talk_filler", "aku mau nannya boleh ga", "Pengantar izin bertanya sebelum curhat"),
    (77, 12): ("off_topic_personal_curhat", "kenapa di dunia ini ada namanya people come and go", "Curhat percintaan/relasi pribadi"),
    (77, 13): ("off_topic_personal_curhat", "aku sedih", "Curhat emosi pribadi"),
    (77, 14): ("off_topic_personal_curhat", "aku selalu keinget mantan aku", "Curhat tentang mantan pacar"),
    (77, 15): ("off_topic_personal_curhat", "AKU GABISA MELUPAKAN KENANGAN KITA", "Curhat tentang kenangan mantan"),
    (77, 16): ("off_topic_personal_curhat", "gatau emang apa", "Lanjutan curhat tentang mantan"),
    (78, 0): ("off_topic_bot_identity", "siapa pembuat tako ai", "Menanyakan pembuat aplikasi Tako AI"),
    (78, 1): ("small_talk_off_topic_followup", "aku tau", "Lanjutan pertanyaan pembuat Tako AI"),
    (78, 5): ("small_talk_meta", "bahasa kamu terlalu formal", "Komentar gaya bahasa bot"),
    (80, 2): ("off_topic_random", "jelaskan tentang pembuatan tung tung sahur", "Topik meme acak di luar KKA/KA/AI"),
    (80, 3): ("off_topic_random", "ridha cantik ngga?", "Basa-basi pujian pribadi"),
    (80, 4): ("off_topic_personal_name", "siapa nama guru kka di smkn 31 jakarta?", "Menanyakan identitas nama guru sekolah"),
    (80, 5): ("off_topic_bot_identity", "ini kamu ai asli atau fake?", "Pertanyaan meta identitas pembuat bot"),
    (83, 4): ("small_talk_greeting", "hai tako", "Sapaan ringan tanpa pertanyaan materi"),
    (83, 5): ("small_talk_greeting", "hai tako", "Sapaan ringan berulang"),
    (83, 6): ("off_topic_personal_name", "nama aku siapa tako", "Mengetes ingatan nama pengguna"),
    (87, 4): ("off_topic_personal_name", "siapa itu ka utaa", "Menanyakan identitas personal/admin"),
    (88, 0): ("off_topic_personal_name", "kenapa ka alvin mengajar materi kka", "Pertanyaan personal mengenai pengajar"),
    (88, 1): ("off_topic_bot_identity", "kapan tanggal dan tahun berapa kamu diciptakan", "Pertanyaan meta tanggal penciptaan bot"),
    (88, 2): ("off_topic_random", "sandi airiliansyah putra siapanya manusia", "Pertanyaan acak/ngawur tentang nama sendiri"),
    (88, 3): ("inappropriate_insult", "kenapa ridha ananda lonte", "Penghinaan/kata kasar terhadap teman"),
    (89, 0): ("off_topic_bot_identity", "apakah kamu memiliki orang tua", "Pertanyaan meta mengenai orang tua bot"),
    (89, 1): ("off_topic_personal_name", "tebak nama saya", "Permainan tebak nama pribadi"),
    (89, 2): ("off_topic_personal_name", "tebak tanggal lahir saya", "Permainan tebak tanggal lahir pribadi"),
}


def make_title(first_question: str) -> str:
    cleaned = " ".join(first_question.strip().split())
    return cleaned[:60] if len(cleaned) > 60 else cleaned


def write_csv(path_out: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with open(path_out, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_subset(raw: dict, filter_mode: str):
    out = {
        "exportedAt": raw["exportedAt"],
        "scope": raw["scope"],
        "summary": {},
        "conversations": [],
    }
    student_ids = set()
    csv_rows = []
    removed_log = []

    for ci, conv in enumerate(raw["conversations"]):
        kept_turns, kept_msgs = [], []
        msgs, turns = conv["messages"], conv["turns"]

        for ti, turn in enumerate(turns):
            key = (ci, ti)
            is_inj = key in PROMPT_INJECTION_RULES
            is_del = key in DELETE_RULES

            if filter_mode == "clean":
                if is_inj or is_del:
                    cat, _, reason = PROMPT_INJECTION_RULES.get(key) or DELETE_RULES.get(key)
                    removed_log.append({
                        "conv_idx": ci,
                        "turn_idx": ti,
                        "action": "separated_to_prompt_injection" if is_inj else "deleted_off_topic",
                        "category": cat,
                        "reason": reason,
                        "student": conv["student"].get("fullName"),
                        "className": conv["student"].get("className"),
                        "exchangeId": turn["exchangeId"],
                        "question": turn["question"],
                    })
                    continue
            elif filter_mode == "prompt_injection":
                if not is_inj:
                    continue

            t_copy = deepcopy(turn)
            m_u, m_a = deepcopy(msgs[2 * ti]), deepcopy(msgs[2 * ti + 1])
            assert m_u["clientRequestId"] == m_a["clientRequestId"] == t_copy["exchangeId"]
            kept_turns.append(t_copy)
            kept_msgs.extend([m_u, m_a])

        if not kept_turns:
            continue

        conv_copy = deepcopy(conv)
        conv_copy["turns"] = kept_turns
        conv_copy["messages"] = kept_msgs
        conv_copy["exchangeCount"] = len(kept_turns)
        conv_copy["messageCount"] = len(kept_msgs)
        conv_copy["createdAt"] = kept_turns[0]["createdAt"]
        conv_copy["updatedAt"] = kept_turns[-1]["createdAt"]
        conv_copy["lastActivityAt"] = kept_turns[-1]["createdAt"]

        if filter_mode == "prompt_injection" or (ci, 0) in DELETE_RULES or (ci, 0) in PROMPT_INJECTION_RULES:
            conv_copy["title"] = make_title(kept_turns[0]["question"])

        out["conversations"].append(conv_copy)
        student_ids.add(conv_copy["student"]["userId"])

        s = conv_copy["student"]
        for local_idx, t in enumerate(kept_turns, start=1):
            sources = t.get("sources") or []
            row = {
                "no": len(csv_rows) + 1,
                "conversation_id": conv_copy["conversationId"],
                "exchange_id": t["exchangeId"],
                "turn_index": local_idx,
                "created_at": t["createdAt"],
                "student_user_id": s.get("userId", ""),
                "student_name": s.get("fullName", ""),
                "student_username": s.get("username", ""),
                "student_role": s.get("role", ""),
                "class_name": s.get("className") or "",
                "grade_level": s.get("gradeLevel") or "",
                "mode": t.get("mode", ""),
                "question": t["question"],
                "answer": t["answer"],
                "sources_count": len(sources),
                "source_titles": " | ".join(
                    f"{src.get('id', '')}: {src.get('title', '')}".strip(": ")
                    for src in sources
                ),
                "training_status": t.get("trainingStatus", ""),
                "training_reason": t.get("trainingReason", ""),
            }
            if filter_mode == "prompt_injection":
                orig_ti = next(idx for idx, orig_t in enumerate(turns) if orig_t["exchangeId"] == t["exchangeId"])
                cat, _, reason = PROMPT_INJECTION_RULES[(ci, orig_ti)]
                row["injection_category"] = cat
                row["notes"] = reason
            csv_rows.append(row)

    total_ex = sum(c["exchangeCount"] for c in out["conversations"])
    out["summary"] = {
        "totalStudents": len(student_ids),
        "totalConversations": len(out["conversations"]),
        "totalExchanges": total_ex,
        "totalMessages": sum(c["messageCount"] for c in out["conversations"]),
    }
    return out, csv_rows, removed_log


def main() -> None:
    assert not (set(PROMPT_INJECTION_RULES) & set(DELETE_RULES))
    raw = json.loads(RAW_PATH.read_text(encoding="utf-8"))

    # Verifikasi kecocokan indeks terhadap prefiks pertanyaan mentah
    for (ci, ti), (_, prefix, _) in {**PROMPT_INJECTION_RULES, **DELETE_RULES}.items():
        q = raw["conversations"][ci]["turns"][ti]["question"]
        assert q.startswith(prefix), f"Ketidakcocokan pada ({ci}.{ti}): {q[:60]!r}"

    clean_data, clean_csv_rows, removed_log = build_subset(raw, "clean")
    inj_data, inj_csv_rows, _ = build_subset(raw, "prompt_injection")

    CLEAN_JSON.write_text(json.dumps(clean_data, ensure_ascii=False, indent=2), encoding="utf-8")
    INJ_JSON.write_text(json.dumps(inj_data, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(CLEAN_CSV, clean_csv_rows)
    write_csv(INJ_CSV, inj_csv_rows)

    deleted_items = [r for r in removed_log if r["action"] == "deleted_off_topic"]
    inj_items = [r for r in removed_log if r["action"] == "separated_to_prompt_injection"]

    print("RAW  :", raw["summary"])
    print("CLEAN:", clean_data["summary"])
    print("INJ  :", inj_data["summary"])
    print(f"Deleted: {len(deleted_items)} | Separated (Prompt Injection): {len(inj_items)}")


if __name__ == "__main__":
    main()
