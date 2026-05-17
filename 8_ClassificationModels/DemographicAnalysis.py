"""
DemographicAnalysis.py  — patched to add US BLS and Global ILOSTAT
reference markers to the ISCO gender figure (fig_isco_gender).

New CLI arguments:
    --us-bls      Path to us_profession_demographics_enhanced.csv  (optional)
    --global-ilo  Path to global_gender_split_global_per_job.csv   (optional)

All other arguments and outputs are unchanged.
"""

import json
import os
import re
import csv
import argparse
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Constants ─────────────────────────────────────────────────────────────────

ISCO_GROUP_NAMES = {
    0: "Armed Forces",
    1: "Managers",
    2: "Professionals",
    3: "Technicians &\nAssoc. Prof.",
    4: "Clerical\nSupport",
    5: "Service &\nSales",
    6: "Skilled\nAgricultural",
    7: "Craft &\nTrades",
    8: "Plant &\nMachine Op.",
    9: "Elementary\nOccupations",
}

BIN_NAMES     = {0: "Light", 1: "Mid", 2: "Dark"}
LAION_COLOUR  = "#2C7BB6"
SD_COLOUR     = "#D7191C"
LIGHT_COLOUR  = "#FEE08B"
MID_COLOUR    = "#ABDDA4"
DARK_COLOUR   = "#3288BD"

# Reference marker colours
US_BLS_COLOUR    = "#1A7A1A"   # dark green  — solid line
GLOBAL_ILO_COLOUR = "#7030A0"  # purple      — dashed line


# ── ISCO mapping ──────────────────────────────────────────────────────────────

def load_isco_mapping(csv_path):
    mapping = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            profession  = row["job_title_original"].strip().title()
            isco_code   = str(row["isco_code"]).strip().zfill(4)
            major_group = int(isco_code[0])
            mapping[profession] = major_group
    return mapping


# ── Reference data loading ────────────────────────────────────────────────────

def load_reference_isco(us_bls_path, global_ilo_path, isco_mapping):
    """
    Returns (us_isco, global_isco) dicts: {isco_group -> female_pct 0-100}.
    Either path may be None, in which case the corresponding dict is empty.
    """
    us_isco     = {}
    global_isco = {}

    if us_bls_path and os.path.exists(us_bls_path):
        df = pd.read_csv(us_bls_path)
        # column is pct_women (0-100 scale)
        df["isco"] = df["profession"].str.title().map(isco_mapping)
        df = df.dropna(subset=["isco", "pct_women"])
        df["isco"] = df["isco"].astype(int)
        us_isco = (
            df.groupby("isco")["pct_women"].mean().to_dict()
        )
        print(f"  US BLS   : {len(us_isco)} ISCO groups loaded")

    if global_ilo_path and os.path.exists(global_ilo_path):
        df = pd.read_csv(global_ilo_path)
        # column is female_pct (0-1 scale) → convert to 0-100
        df["isco"] = df["job_title_original"].str.title().map(isco_mapping)
        df = df.dropna(subset=["isco", "female_pct"])
        df["isco"] = df["isco"].astype(int)
        global_isco = (
            (df.groupby("isco")["female_pct"].mean() * 100).to_dict()
        )
        print(f"  Global   : {len(global_isco)} ISCO groups loaded")

    return us_isco, global_isco


# ── Data loading ──────────────────────────────────────────────────────────────

def extract_profession(image_path):
    parts  = image_path.replace("\\", "/").split("/")
    folder = parts[0]
    folder = re.sub(r"^A_photo_of_an?_", "", folder, flags=re.IGNORECASE)
    return folder.replace("_", " ").title()


def load_jsonl(path, isco_mapping):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            r["profession"] = extract_profession(r["image"])
            r["isco_group"] = isco_mapping.get(r["profession"], 0)
            records.append(r)
    return records


def to_df(records):
    return pd.DataFrame(records)


# ── Aggregate statistics ──────────────────────────────────────────────────────

def aggregate_gender(records):
    counts = defaultdict(int)
    for r in records:
        counts[r["gender"]] += 1
    total  = sum(counts.values())
    female = counts.get("Female", 0)
    male   = counts.get("Male",   0)
    return {
        "total":        total,
        "female_count": female,
        "male_count":   male,
        "female_pct":   round(100 * female / total, 2) if total else 0,
        "male_pct":     round(100 * male   / total, 2) if total else 0,
    }


def aggregate_skintone(records):
    counts = defaultdict(int)
    for r in records:
        counts[r["bin_label"]] += 1
    total  = sum(counts.values())
    result = {"total": total}
    for label, name in BIN_NAMES.items():
        c = counts.get(label, 0)
        result[f"{name.lower()}_count"] = c
        result[f"{name.lower()}_pct"]   = round(100 * c / total, 2) if total else 0
    return result


# ── Per-profession statistics ─────────────────────────────────────────────────

def build_profession_records(records):
    data = defaultdict(list)
    for r in records:
        data[r["profession"]].append(r)
    return dict(data)


def per_profession_gender(prof_data):
    rows = []
    for profession, recs in sorted(prof_data.items()):
        total      = len(recs)
        female     = sum(1 for r in recs if r["gender"] == "Female")
        male       = total - female
        female_pct = round(100 * female / total, 2) if total else 0
        rows.append({
            "profession":   profession,
            "total":        total,
            "female_count": female,
            "male_count":   male,
            "female_pct":   female_pct,
            "male_pct":     round(100 * male / total, 2) if total else 0,
            "gender_skew":  round(abs(female_pct / 100 - 0.5), 4),
        })
    return rows


def per_profession_skintone(prof_data):
    rows = []
    for profession, recs in sorted(prof_data.items()):
        total  = len(recs)
        counts = defaultdict(int)
        for r in recs:
            counts[r["bin_label"]] += 1
        row = {"profession": profession, "total": total}
        for label, name in BIN_NAMES.items():
            c = counts.get(label, 0)
            row[f"{name.lower()}_count"] = c
            row[f"{name.lower()}_pct"]   = round(100 * c / total, 2) if total else 0
        row["light_skew"] = row["light_pct"] / 100
        rows.append(row)
    return rows


# ── ISCO group statistics ─────────────────────────────────────────────────────

def isco_group_gender(prof_gender_rows, isco_mapping):
    gc = defaultdict(lambda: {"female": 0, "male": 0, "total": 0, "professions": []})
    for row in prof_gender_rows:
        g = isco_mapping.get(row["profession"], 0)
        gc[g]["female"]      += row["female_count"]
        gc[g]["male"]        += row["male_count"]
        gc[g]["total"]       += row["total"]
        gc[g]["professions"].append(row["profession"])
    rows = []
    for group in sorted(gc):
        d = gc[group]; total = d["total"]
        rows.append({
            "isco_group":       group,
            "isco_name":        ISCO_GROUP_NAMES.get(group, "Unknown").replace("\n", " "),
            "profession_count": len(d["professions"]),
            "total_images":     total,
            "female_count":     d["female"],
            "male_count":       d["male"],
            "female_pct":       round(100 * d["female"] / total, 2) if total else 0,
            "male_pct":         round(100 * d["male"]   / total, 2) if total else 0,
        })
    return rows


def isco_group_skintone(prof_skintone_rows, isco_mapping):
    gc = defaultdict(lambda: {"light": 0, "mid": 0, "dark": 0, "total": 0, "professions": []})
    for row in prof_skintone_rows:
        g = isco_mapping.get(row["profession"], 0)
        gc[g]["light"]       += row["light_count"]
        gc[g]["mid"]         += row["mid_count"]
        gc[g]["dark"]        += row["dark_count"]
        gc[g]["total"]       += row["total"]
        gc[g]["professions"].append(row["profession"])
    rows = []
    for group in sorted(gc):
        d = gc[group]; total = d["total"]
        rows.append({
            "isco_group":       group,
            "isco_name":        ISCO_GROUP_NAMES.get(group, "Unknown").replace("\n", " "),
            "profession_count": len(d["professions"]),
            "total_images":     total,
            "light_count":      d["light"],
            "mid_count":        d["mid"],
            "dark_count":       d["dark"],
            "light_pct":        round(100 * d["light"] / total, 2) if total else 0,
            "mid_pct":          round(100 * d["mid"]   / total, 2) if total else 0,
            "dark_pct":         round(100 * d["dark"]  / total, 2) if total else 0,
        })
    return rows


# ── Top skewed professions ────────────────────────────────────────────────────

def top_n_gender_skew(rows, n=10):
    return sorted(rows, key=lambda r: r["gender_skew"], reverse=True)[:n]


def top_n_skintone_skew(rows, n=10):
    return sorted(rows, key=lambda r: r["light_skew"], reverse=True)[:n]


# ── Amplification analysis ────────────────────────────────────────────────────

def amplification_analysis(laion_pgr, sd_pgr, laion_psr, sd_psr):
    lg = {r["profession"]: r for r in laion_pgr}
    sg = {r["profession"]: r for r in sd_pgr}
    lk = {r["profession"]: r for r in laion_psr}
    sk = {r["profession"]: r for r in sd_psr}
    rows = []
    for prof in sorted(set(lg) & set(sg)):
        lf = lg[prof]["female_pct"]
        sf = sg[prof]["female_pct"]
        ll = lk.get(prof, {}).get("light_pct")
        sl = sk.get(prof, {}).get("light_pct")
        rows.append({
            "profession":                 prof,
            "laion_female_pct":           lf,
            "sd_female_pct":              sf,
            "gender_diff_sd_minus_laion": round(sf - lf, 2),
            "laion_direction":            "Female" if lf >= 50 else "Male",
            "sd_direction":               "Female" if sf >= 50 else "Male",
            "direction_agreement":        (lf >= 50) == (sf >= 50),
            "sd_more_extreme_gender":     abs(sf - 50) > abs(lf - 50),
            "laion_light_pct":            ll,
            "sd_light_pct":               sl,
            "light_diff_sd_minus_laion":  round(sl - ll, 2)
                                          if ll is not None and sl is not None else None,
            "sd_more_extreme_skin":       (sl > ll)
                                          if ll is not None and sl is not None else None,
        })
    return rows


# ── CSV helpers ───────────────────────────────────────────────────────────────

def write_csv(path, rows):
    if not rows:
        print(f"  [WARN] No rows for {os.path.basename(path)}")
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"  CSV : {os.path.basename(path)}")


def write_summary(path, laion_records, sd_records,
                  laion_ga, sd_ga, laion_sa, sd_sa, amp):
    n       = len(amp)
    n_agree = sum(1 for r in amp if r["direction_agreement"])
    n_eg    = sum(1 for r in amp if r["sd_more_extreme_gender"])
    n_es    = sum(1 for r in amp if r.get("sd_more_extreme_skin"))
    lines = [
        "=" * 60, "DEMOGRAPHIC ANALYSIS SUMMARY", "=" * 60, "",
        "--- Re-LAION-5B ---",
        f"  Total annotated images : {laion_ga['total']:,}",
        f"  Professions            : {len(set(extract_profession(r['image']) for r in laion_records))}",
        f"  Female                 : {laion_ga['female_count']:,} ({laion_ga['female_pct']}%)",
        f"  Male                   : {laion_ga['male_count']:,} ({laion_ga['male_pct']}%)",
        f"  Light skin             : {laion_sa['light_count']:,} ({laion_sa['light_pct']}%)",
        f"  Mid skin               : {laion_sa['mid_count']:,} ({laion_sa['mid_pct']}%)",
        f"  Dark skin              : {laion_sa['dark_count']:,} ({laion_sa['dark_pct']}%)", "",
        "--- Stable Diffusion v1.5 ---",
        f"  Total annotated images : {sd_ga['total']:,}",
        f"  Professions            : {len(set(extract_profession(r['image']) for r in sd_records))}",
        f"  Female                 : {sd_ga['female_count']:,} ({sd_ga['female_pct']}%)",
        f"  Male                   : {sd_ga['male_count']:,} ({sd_ga['male_pct']}%)",
        f"  Light skin             : {sd_sa['light_count']:,} ({sd_sa['light_pct']}%)",
        f"  Mid skin               : {sd_sa['mid_count']:,} ({sd_sa['mid_pct']}%)",
        f"  Dark skin              : {sd_sa['dark_count']:,} ({sd_sa['dark_pct']}%)", "",
        "--- Amplification Analysis ---",
        f"  Professions in both datasets : {n}",
        f"  Direction agreement (gender) : {n_agree}/{n} ({round(100*n_agree/n,1) if n else 0}%)",
        f"  SD more extreme (gender)     : {n_eg}/{n} professions",
        f"  SD lighter (skin tone)       : {n_es}/{n} professions", "",
        "=" * 60,
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("\n".join(lines))


# ── Figure helpers ────────────────────────────────────────────────────────────

def save_fig(fig, path):
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  PNG : {os.path.basename(path)}")


# ── Figure 1: Aggregate gender ────────────────────────────────────────────────

def fig_aggregate_gender(df_laion, df_sd, out):
    fig, ax = plt.subplots(figsize=(6, 4))
    x, w = np.arange(2), 0.35
    fp = [100*(df_laion["gender"]=="Female").mean(),
          100*(df_sd["gender"]=="Female").mean()]
    mp = [100-p for p in fp]
    b1 = ax.bar(x-w/2, fp, w, label="Female", color="#E8A0BF", edgecolor="white")
    b2 = ax.bar(x+w/2, mp, w, label="Male",   color="#5B9BD5", edgecolor="white")
    ax.axhline(50, color="black", lw=0.8, ls="--", label="Parity (50%)")
    ax.set_xticks(x); ax.set_xticklabels(["Re-LAION-5B", "SDv1.5"], fontsize=11)
    ax.set_ylabel("Proportion (%)", fontsize=11); ax.set_ylim(0, 85)
    ax.set_title("Aggregate Gender Distribution", fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    for b in list(b1)+list(b2):
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.8,
                f"{b.get_height():.1f}%", ha="center", va="bottom", fontsize=9)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    plt.tight_layout()
    save_fig(fig, os.path.join(out, "aggregate_gender.png"))


# ── Figure 2: Aggregate skin tone bins ───────────────────────────────────────

def fig_aggregate_skintone_bins(df_laion, df_sd, out):
    fig, axes = plt.subplots(1, 2, figsize=(8, 4), sharey=True)
    bin_labels  = ["Light (1-3)", "Mid (4-7)", "Dark (8-10)"]
    bin_colours = [LIGHT_COLOUR, MID_COLOUR, DARK_COLOUR]
    for ax, df, title in zip(axes, [df_laion, df_sd], ["Re-LAION-5B", "SDv1.5"]):
        total = len(df)
        pcts  = [100*(df["bin_label"]==v).sum()/total for v in [0, 1, 2]]
        bars  = ax.bar(bin_labels, pcts, color=bin_colours, edgecolor="white", width=0.55)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_ylabel("Proportion (%)" if ax is axes[0] else "", fontsize=10)
        ax.set_ylim(0, 100)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        ax.tick_params(axis="x", labelsize=9)
        for bar, pct in zip(bars, pcts):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.8,
                    f"{pct:.1f}%", ha="center", va="bottom", fontsize=9)
    fig.suptitle("Aggregate Skin Tone Distribution (Three-Bin)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    save_fig(fig, os.path.join(out, "aggregate_skintone_bins.png"))


# ── Figure 3: Aggregate MST 1-10 ─────────────────────────────────────────────

def fig_aggregate_mst(df_laion, df_sd, out):
    fig, ax = plt.subplots(figsize=(11, 5))
    mst = list(range(1, 11))
    x, w = np.arange(10), 0.38
    lp = [100*(df_laion["mst_label"]==m).sum()/len(df_laion) for m in mst]
    sp = [100*(df_sd["mst_label"]==m).sum()/len(df_sd)       for m in mst]
    ax.bar(x-w/2, lp, w, label="Re-LAION-5B", color=LAION_COLOUR, alpha=0.85, edgecolor="white")
    ax.bar(x+w/2, sp, w, label="SDv1.5",      color=SD_COLOUR,    alpha=0.85, edgecolor="white")
    ax.axvspan(-0.5, 2.5, alpha=0.07, color=LIGHT_COLOUR)
    ax.axvspan( 2.5, 6.5, alpha=0.07, color=MID_COLOUR)
    ax.axvspan( 6.5, 9.5, alpha=0.07, color=DARK_COLOUR)
    ymax = ax.get_ylim()[1]
    for xpos, lbl in [(1.0, "Light\n(1–3)"), (4.5, "Mid\n(4–7)"), (8.0, "Dark\n(8–10)")]:
        ax.text(xpos, ymax * 0.96, lbl, ha="center", va="top", fontsize=8, color="grey")
    ax.set_xticks(x); ax.set_xticklabels([f"MST {m}" for m in mst], fontsize=9)
    ax.set_ylabel("Proportion (%)", fontsize=11)
    ax.set_title("Aggregate MST Skin Tone Distribution (MST 1–10)",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    plt.tight_layout()
    save_fig(fig, os.path.join(out, "aggregate_mst_distribution.png"))


# ── Figure 4: ISCO gender  (PATCHED — adds BLS / ILOSTAT reference markers) ──

def fig_isco_gender(df_laion, df_sd, out,
                    us_isco=None, global_isco=None):
    """
    us_isco     : {isco_group (int) -> female_pct (0-100)}  or None
    global_isco : {isco_group (int) -> female_pct (0-100)}  or None
    """
    us_isco     = us_isco     or {}
    global_isco = global_isco or {}

    groups = sorted(df_laion["isco_group"].unique())
    labels = [ISCO_GROUP_NAMES.get(g, str(g)) for g in groups]
    x, w   = np.arange(len(groups)), 0.38

    lf = [100*(df_laion[df_laion["isco_group"]==g]["gender"]=="Female").mean()
          if (df_laion["isco_group"]==g).any() else 0 for g in groups]
    sf = [100*(df_sd[df_sd["isco_group"]==g]["gender"]=="Female").mean()
          if (df_sd["isco_group"]==g).any() else 0 for g in groups]

    fig, ax = plt.subplots(figsize=(13, 5))

    # ── Bars (unchanged) ──────────────────────────────────
    ax.bar(x-w/2, lf, w, label="Re-LAION-5B", color=LAION_COLOUR,
           alpha=0.85, edgecolor="white")
    ax.bar(x+w/2, sf, w, label="SDv1.5",      color=SD_COLOUR,
           alpha=0.85, edgecolor="white")

    # ── Reference markers ─────────────────────────────────
    # Each marker is a horizontal line spanning both bars of the group.
    half_span = w + 0.04   # slightly wider than bar pair

    for i, g in enumerate(groups):
        if g in us_isco:
            y = us_isco[g]
            ax.plot([x[i] - half_span, x[i] + half_span], [y, y],
                    color=US_BLS_COLOUR, linewidth=2.2,
                    solid_capstyle="butt", zorder=4)

        if g in global_isco:
            y = global_isco[g]
            ax.plot([x[i] - half_span, x[i] + half_span], [y, y],
                    color=GLOBAL_ILO_COLOUR, linewidth=2.2,
                    linestyle="--", solid_capstyle="butt", zorder=4)

    # ── Parity line ────────────────────────────────────────
    ax.axhline(50, color="black", lw=0.8, ls="--", alpha=0.6)

    # ── Axes ───────────────────────────────────────────────
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylabel("Female Proportion (%)", fontsize=11)
    ax.set_ylim(0, 80)
    ax.set_title("Female Proportion by ISCO-08 Major Occupational Group",
                 fontsize=12, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # ── Legend ─────────────────────────────────────────────
    legend_handles = [
        mpatches.Patch(color=LAION_COLOUR, alpha=0.85, label="Re-LAION-5B"),
        mpatches.Patch(color=SD_COLOUR,    alpha=0.85, label="SD v1.5"),
    ]
    if us_isco:
        legend_handles.append(
            plt.Line2D([0], [0], color=US_BLS_COLOUR, linewidth=2.2,
                       label="US BLS 2024")
        )
    if global_isco:
        legend_handles.append(
            plt.Line2D([0], [0], color=GLOBAL_ILO_COLOUR, linewidth=2.2,
                       linestyle="--", label="Global ILOSTAT")
        )
    legend_handles.append(
        plt.Line2D([0], [0], color="black", linewidth=0.8,
                   linestyle="--", alpha=0.6, label="Parity (50%)")
    )
    ax.legend(handles=legend_handles, fontsize=9, loc="upper right",
              framealpha=0.9, edgecolor="0.8")

    plt.tight_layout()
    save_fig(fig, os.path.join(out, "isco_gender.png"))


# ── Figure 5: ISCO skin tone bins ─────────────────────────────────────────────

def fig_isco_skintone_bins(df_laion, df_sd, out):
    groups = sorted(df_laion["isco_group"].unique())
    labels = [ISCO_GROUP_NAMES.get(g, str(g)) for g in groups]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for ax, df, title in zip(axes, [df_laion, df_sd], ["Re-LAION-5B", "SDv1.5"]):
        lp, mp, dp = [], [], []
        for g in groups:
            sub = df[df["isco_group"]==g]; n = len(sub)
            lp.append(100*(sub["bin_label"]==0).sum()/n if n else 0)
            mp.append(100*(sub["bin_label"]==1).sum()/n if n else 0)
            dp.append(100*(sub["bin_label"]==2).sum()/n if n else 0)
        x = np.arange(len(groups))
        ax.bar(x, lp, 0.6, label="Light (1-3)", color=LIGHT_COLOUR, edgecolor="white")
        ax.bar(x, mp, 0.6, bottom=lp, label="Mid (4-7)", color=MID_COLOUR, edgecolor="white")
        ax.bar(x, dp, 0.6, bottom=[l+m for l,m in zip(lp,mp)],
               label="Dark (8-10)", color=DARK_COLOUR, edgecolor="white")
        ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=7.5, rotation=15, ha="right")
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_ylabel("Proportion (%)" if ax is axes[0] else "", fontsize=10)
        ax.set_ylim(0, 105)
        ax.legend(fontsize=8, loc="upper right")
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    fig.suptitle("Skin Tone Distribution by ISCO-08 Group",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    save_fig(fig, os.path.join(out, "isco_skintone_bins.png"))


# ── Figure 6: ISCO MST heatmap ────────────────────────────────────────────────

def fig_isco_mst_heatmap(df_laion, df_sd, out):
    groups = sorted(df_laion["isco_group"].unique())
    glabels = [ISCO_GROUP_NAMES.get(g, str(g)).replace("\n", " ") for g in groups]
    mst = list(range(1, 11))

    def make_mat(df):
        mat = np.zeros((len(groups), 10))
        for i, g in enumerate(groups):
            sub = df[df["isco_group"]==g]; n = len(sub)
            if n == 0: continue
            for j, m in enumerate(mst):
                mat[i, j] = 100*(sub["mst_label"]==m).sum()/n
        return mat

    ml = make_mat(df_laion)
    ms = make_mat(df_sd)
    vmax = max(ml.max(), ms.max())

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    for ax, mat, title in zip(axes, [ml, ms], ["Re-LAION-5B", "SDv1.5"]):
        im = ax.imshow(mat, aspect="auto", cmap="YlOrRd", vmin=0, vmax=vmax)
        ax.set_xticks(range(10))
        ax.set_xticklabels([f"MST {m}" for m in mst], fontsize=8, rotation=45, ha="right")
        ax.set_yticks(range(len(glabels))); ax.set_yticklabels(glabels, fontsize=8)
        ax.set_title(title, fontsize=11, fontweight="bold")
        plt.colorbar(im, ax=ax, label="Proportion (%)", shrink=0.8)
        for i in range(len(groups)):
            for j in range(10):
                v = mat[i, j]
                ax.text(j, i, f"{v:.1f}", ha="center", va="center",
                        fontsize=6.5, color="white" if v > 30 else "black")
    fig.suptitle("MST Score Distribution by ISCO-08 Group (%)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    save_fig(fig, os.path.join(out, "isco_mst_heatmap.png"))


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Demographic distribution analysis and plotting.")
    parser.add_argument("--laion",      required=True,  help="Re-LAION-5B annotation JSONL")
    parser.add_argument("--sd",         required=True,  help="SDv1.5 annotation JSONL")
    parser.add_argument("--isco-csv",   required=True,  help="job_list_isco_mapped.csv")
    parser.add_argument("--out",        default="results", help="Output directory")
    parser.add_argument("--top-n",      type=int, default=10,
                        help="Top N skewed professions to report")
    # ── New optional arguments ─────────────────────────────
    parser.add_argument("--us-bls",
                        default=None,
                        help="us_profession_demographics_enhanced.csv (optional)")
    parser.add_argument("--global-ilo",
                        default=None,
                        help="global_gender_split_global_per_job.csv (optional)")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    print("Loading ISCO mapping...")
    isco_mapping = load_isco_mapping(args.isco_csv)
    print(f"  {len(isco_mapping)} professions mapped")

    # ── Load reference data if provided ───────────────────
    print("Loading reference data...")
    us_isco, global_isco = load_reference_isco(
        args.us_bls, args.global_ilo, isco_mapping
    )

    print("Loading annotations...")
    laion_records = load_jsonl(args.laion, isco_mapping)
    sd_records    = load_jsonl(args.sd,    isco_mapping)
    print(f"  Re-LAION-5B : {len(laion_records):,}")
    print(f"  SDv1.5      : {len(sd_records):,}")

    unmapped = (set(r["profession"] for r in laion_records + sd_records)
                - set(isco_mapping))
    if unmapped:
        print(f"\n  [WARN] {len(unmapped)} unmapped professions (assigned group 0):")
        for p in sorted(unmapped):
            print(f"    - {p}")

    df_laion = to_df(laion_records)
    df_sd    = to_df(sd_records)

    laion_prof = build_profession_records(laion_records)
    sd_prof    = build_profession_records(sd_records)

    laion_ga  = aggregate_gender(laion_records)
    sd_ga     = aggregate_gender(sd_records)
    laion_sa  = aggregate_skintone(laion_records)
    sd_sa     = aggregate_skintone(sd_records)

    laion_pgr = per_profession_gender(laion_prof)
    sd_pgr    = per_profession_gender(sd_prof)
    laion_psr = per_profession_skintone(laion_prof)
    sd_psr    = per_profession_skintone(sd_prof)

    laion_ig  = isco_group_gender(laion_pgr,   isco_mapping)
    sd_ig     = isco_group_gender(sd_pgr,       isco_mapping)
    laion_is  = isco_group_skintone(laion_psr,  isco_mapping)
    sd_is     = isco_group_skintone(sd_psr,     isco_mapping)

    amp = amplification_analysis(laion_pgr, sd_pgr, laion_psr, sd_psr)

    print("\nWriting CSVs...")
    write_csv(os.path.join(args.out, "aggregate_gender.csv"),
              [{"dataset": "Re-LAION-5B", **laion_ga},
               {"dataset": "SDv1.5",      **sd_ga}])
    write_csv(os.path.join(args.out, "aggregate_skintone.csv"),
              [{"dataset": "Re-LAION-5B", **laion_sa},
               {"dataset": "SDv1.5",      **sd_sa}])
    write_csv(os.path.join(args.out, "laion_per_profession_gender.csv"),   laion_pgr)
    write_csv(os.path.join(args.out, "sd_per_profession_gender.csv"),      sd_pgr)
    write_csv(os.path.join(args.out, "laion_per_profession_skintone.csv"), laion_psr)
    write_csv(os.path.join(args.out, "sd_per_profession_skintone.csv"),    sd_psr)
    write_csv(os.path.join(args.out, "laion_isco_gender.csv"),   laion_ig)
    write_csv(os.path.join(args.out, "sd_isco_gender.csv"),       sd_ig)
    write_csv(os.path.join(args.out, "laion_isco_skintone.csv"), laion_is)
    write_csv(os.path.join(args.out, "sd_isco_skintone.csv"),     sd_is)
    write_csv(os.path.join(args.out, "top_gender_skew.csv"),
              [{"dataset": "Re-LAION-5B", **r} for r in top_n_gender_skew(laion_pgr, args.top_n)] +
              [{"dataset": "SDv1.5",      **r} for r in top_n_gender_skew(sd_pgr,    args.top_n)])
    write_csv(os.path.join(args.out, "top_skintone_skew.csv"),
              [{"dataset": "Re-LAION-5B", **r} for r in top_n_skintone_skew(laion_psr, args.top_n)] +
              [{"dataset": "SDv1.5",      **r} for r in top_n_skintone_skew(sd_psr,    args.top_n)])
    write_csv(os.path.join(args.out, "amplification_analysis.csv"), amp)
    write_summary(os.path.join(args.out, "summary.txt"),
                  laion_records, sd_records,
                  laion_ga, sd_ga, laion_sa, sd_sa, amp)

    print("\nGenerating figures...")
    fig_aggregate_gender(df_laion, df_sd, args.out)
    fig_aggregate_skintone_bins(df_laion, df_sd, args.out)
    fig_aggregate_mst(df_laion, df_sd, args.out)
    # ── Patched call — passes reference dicts ──────────────
    fig_isco_gender(df_laion, df_sd, args.out,
                    us_isco=us_isco, global_isco=global_isco)
    fig_isco_skintone_bins(df_laion, df_sd, args.out)
    fig_isco_mst_heatmap(df_laion, df_sd, args.out)

    print(f"\nDone. All outputs in: {args.out}/")


if __name__ == "__main__":
    main()