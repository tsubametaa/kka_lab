"""
04_generate_bab4_charts.py
--------------------------
Menghasilkan grafik resolusi tinggi (300 DPI) untuk Bab IV:
- docs/images/gambar_4_3_distribusi_nilai_pretest.png
- docs/images/gambar_4_4_distribusi_topik_dialog_siswa.png
"""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PRETEST_CSV = ROOT / "docs" / "nilai_pretest.csv"
IMG_DIR = ROOT / "docs" / "images"
IMG_DIR.mkdir(parents=True, exist_ok=True)

def generate_pretest_chart() -> Path:
    with open(PRETEST_CSV, "r", encoding="utf-8-sig") as f:
        rows = list(csv.reader(f, delimiter=";"))
    umum = [float(r[1]) for r in rows[2:] if len(r) > 1 and r[1].strip()]
    ter = [float(r[8]) for r in rows[2:] if len(r) > 8 and r[8].strip()]

    bin_defs = [
        ("< 50", lambda x: x < 50),
        ("50–59", lambda x: 50 <= x < 60),
        ("60–69", lambda x: 60 <= x < 70),
        ("70–79", lambda x: 70 <= x < 80),
        ("80–89", lambda x: 80 <= x < 90),
        ("90–100", lambda x: x >= 90),
    ]
    labels = [b[0] for b in bin_defs]
    u_cnt = [sum(1 for x in umum if fn(x)) for _, fn in bin_defs]
    t_cnt = [sum(1 for x in ter if fn(x)) for _, fn in bin_defs]
    u_pct = [c / len(umum) * 100 for c in u_cnt]
    t_pct = [c / len(ter) * 100 for c in t_cnt]

    fig, ax = plt.subplots(figsize=(11.5, 5.8), dpi=300)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    x = np.arange(len(labels))
    width = 0.37

    bars_u = ax.bar(
        x - width / 2, u_pct, width,
        label=f"Kelompok Umum — X ANM & X DKV (n = {len(umum)}, Rerata {np.mean(umum):.2f})".replace(".", ","),
        color="#94A3B8", edgecolor="#1E293B", linewidth=1.0, zorder=3,
    )
    bars_t = ax.bar(
        x + width / 2, t_pct, width,
        label=f"Kelompok Terstruktur — X MP & X LP (n = {len(ter)}, Rerata {np.mean(ter):.2f})".replace(".", ","),
        color="#10B981", edgecolor="#064E3B", linewidth=1.0, zorder=3,
    )

    ax.yaxis.grid(True, linestyle="--", color="#CBD5E1", alpha=0.75, zorder=0)
    ax.set_axisbelow(True)

    for bar, cnt, pct in zip(bars_u, u_cnt, u_pct):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.8,
            f"{pct:.1f}%\n({cnt})".replace(".", ","),
            ha="center", va="bottom", fontsize=9.5, fontweight="bold", color="#1E293B",
        )
    for bar, cnt, pct in zip(bars_t, t_cnt, t_pct):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.8,
            f"{pct:.1f}%\n({cnt})".replace(".", ","),
            ha="center", va="bottom", fontsize=9.5, fontweight="bold", color="#065F46",
        )

    ax.set_title(
        f"Distribusi Persentase Nilai Pretest Pemahaman KKA (N = {len(umum) + len(ter)} Siswa)",
        fontsize=13.5, fontweight="bold", pad=14, color="#0F172A",
    )
    ax.set_xlabel("Rentang Nilai Pretest (Skala 0–100)", fontsize=11, fontweight="bold", labelpad=8, color="#0F172A")
    ax.set_ylabel("Persentase Siswa (%)", fontsize=11, fontweight="bold", labelpad=8, color="#0F172A")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10.5, fontweight="bold", color="#0F172A")
    ax.set_ylim(0, 44)
    ax.tick_params(axis="y", labelsize=10, colors="#1E293B")
    ax.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="#CBD5E1", framealpha=0.95, fontsize=9.5)

    plt.tight_layout()
    out_path = IMG_DIR / "gambar_4_3_distribusi_nilai_pretest.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path

def generate_dialog_topic_chart() -> Path:
    topics = [
        "Dasar Pengenalan AI\n(Konsep, Jenis, Cara Kerja & Literasi KKA)",
        "Pengembangan Model,\nData Latih/Uji & Bias",
        "Rekayasa Prompt (PKTO)",
        "Etika, Privasi, Keamanan\n& Pemanfaatan AI",
        "Perancangan Sistem AI",
        "Verifikasi Informasi AI & Cek ARC",
        "Dasar Pemrograman & Koding\n(Sintaks Python/Algoritma)",
    ]
    counts = [312, 124, 69, 55, 46, 42, 15]
    total = sum(counts)
    pcts = [c / total * 100 for c in counts]

    fig, ax = plt.subplots(figsize=(11.5, 5.8), dpi=300)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    y = np.arange(len(topics))[::-1]
    colors = ["#10B981"] + ["#94A3B8"] * (len(topics) - 2) + ["#64748B"]
    edges = ["#064E3B"] + ["#1E293B"] * (len(topics) - 1)

    bars = ax.barh(y, pcts, height=0.62, color=colors, edgecolor=edges, linewidth=1.0, zorder=3)
    ax.xaxis.grid(True, linestyle="--", color="#CBD5E1", alpha=0.75, zorder=0)
    ax.set_axisbelow(True)

    for idx, (bar, cnt, pct) in enumerate(zip(bars, counts, pcts)):
        ax.text(
            bar.get_width() + 0.6, bar.get_y() + bar.get_height() / 2,
            f" {pct:.1f}% ({cnt} exchange)".replace(".", ","),
            ha="left", va="center", fontsize=9.8, fontweight="bold",
            color="#065F46" if idx == 0 else "#1E293B",
        )

    ax.set_title(
        f"Distribusi Topik Pertanyaan pada Log Dialog Siswa Bersih (N = {total} Exchange)",
        fontsize=13, fontweight="bold", pad=14, color="#0F172A",
    )
    ax.set_xlabel("Persentase Exchange (%)", fontsize=11, fontweight="bold", labelpad=8, color="#0F172A")
    ax.set_yticks(y)
    ax.set_yticklabels(topics, fontsize=9.8, fontweight="bold", color="#0F172A")
    ax.set_xlim(0, 62)
    ax.tick_params(axis="x", labelsize=10, colors="#1E293B")

    plt.tight_layout()
    out_path = IMG_DIR / "gambar_4_4_distribusi_topik_dialog_siswa.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path

def main() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.edgecolor": "#CBD5E1",
        "axes.linewidth": 1.0,
    })
    p1 = generate_pretest_chart()
    p2 = generate_dialog_topic_chart()
    print("Grafik tersimpan:")
    print(" -", p1)
    print(" -", p2)

if __name__ == "__main__":
    main()
