"""
03_analyze_pretest.py
---------------------
Mengolah data skor pretest siswa (`docs/nilai_pretest.csv`):
- Statistik deskriptif (Mean, SD, Median, Min, Maks, Q1, Q3, IQR)
- Distribusi frekuensi menurut 6 rentang nilai
- Uji normalitas Shapiro-Wilk
- Uji homogenitas varians Levene
- Uji kesetaraan kemampuan awal (Mann-Whitney U & Welch t-test)
"""

import csv
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
PRETEST_CSV = ROOT / "docs" / "nilai_pretest.csv"


def load_pretest(path: Path) -> tuple[list[float], list[float]]:
    umum, terstruktur = [], []
    with open(path, "r", encoding="utf-8-sig") as f:
        rows = list(csv.reader(f, delimiter=";"))
    for r in rows[2:]:
        if len(r) > 1 and r[1].strip():
            umum.append(float(r[1].strip()))
        if len(r) > 8 and r[8].strip():
            terstruktur.append(float(r[8].strip()))
    return umum, terstruktur


def descriptive(values: list[float]) -> dict:
    arr = np.array(values, dtype=float)
    q1, med, q3 = np.percentile(arr, [25, 50, 75])
    return {
        "n": len(arr),
        "mean": float(np.mean(arr)),
        "sd": float(np.std(arr, ddof=1)),
        "median": float(med),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "q1": float(q1),
        "q3": float(q3),
        "iqr": float(q3 - q1),
    }


def frequency_distribution(values: list[float]) -> list[tuple[str, int, float]]:
    bins = [
        ("< 50", lambda x: x < 50),
        ("50-59", lambda x: 50 <= x < 60),
        ("60-69", lambda x: 60 <= x < 70),
        ("70-79", lambda x: 70 <= x < 80),
        ("80-89", lambda x: 80 <= x < 90),
        ("90-100", lambda x: x >= 90),
    ]
    n = len(values)
    return [(label, sum(1 for x in values if fn(x)), sum(1 for x in values if fn(x)) / n * 100) for label, fn in bins]


def main() -> None:
    umum, terstruktur = load_pretest(PRETEST_CSV)
    gabungan = umum + terstruktur

    print("=== STATISTIK DESKRIPTIF NILAI PRETEST ===")
    for name, vals in [("Kelompok Umum", umum), ("Kelompok Terstruktur", terstruktur), ("Gabungan", gabungan)]:
        d = descriptive(vals)
        print(
            f"{name:22s} | n={d['n']:3d} | Mean={d['mean']:.2f} | SD={d['sd']:.2f} | "
            f"Med={d['median']:.2f} | Min={d['min']:.0f} | Max={d['max']:.0f} | "
            f"Q1={d['q1']:.2f} | Q3={d['q3']:.2f} | IQR={d['iqr']:.2f}"
        )

    print("\n=== DISTRIBUSI FREKUENSI ===")
    for name, vals in [("Kelompok Umum", umum), ("Kelompok Terstruktur", terstruktur), ("Total", gabungan)]:
        print(f"\n{name} (n={len(vals)}):")
        for label, cnt, pct in frequency_distribution(vals):
            print(f"  {label:6s}: {cnt:2d} ({pct:5.1f}%)")

    sw_u = stats.shapiro(umum)
    sw_t = stats.shapiro(terstruktur)
    lev = stats.levene(umum, terstruktur)
    mwu = stats.mannwhitneyu(umum, terstruktur, alternative="two-sided")
    welch = stats.ttest_ind(umum, terstruktur, equal_var=False)

    print("\n=== UJI PRASYARAT & KESETARAAN AWAL (alpha = 0.05) ===")
    print(f"Shapiro-Wilk Umum        : W = {sw_u.statistic:.3f}, p = {sw_u.pvalue:.4f}")
    print(f"Shapiro-Wilk Terstruktur : W = {sw_t.statistic:.3f}, p = {sw_t.pvalue:.4f}")
    print(f"Levene Homogenitas       : F = {lev.statistic:.3f}, p = {lev.pvalue:.4f}")
    print(f"Mann-Whitney U           : U = {mwu.statistic:.1f}, p = {mwu.pvalue:.4f}")
    print(f"Welch t-test (pembanding): t = {welch.statistic:.3f}, p = {welch.pvalue:.4f}")


if __name__ == "__main__":
    main()
