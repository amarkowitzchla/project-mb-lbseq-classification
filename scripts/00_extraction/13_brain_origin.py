#!/usr/bin/env python3
"""
13_brain_origin.py

Indirect brain-cell-of-origin scoring from standard WGS cfDNA BAMs using
the 10 brain-specific methylation marker loci from Lubotzky et al.
JCI Insight 2022 (doi:10.1172/jci.insight.153559).

IMPORTANT LIMITATION
--------------------
Standard WGS (no bisulfite conversion) cannot directly call 5-methylcytosine.
The Lubotzky approach uses bisulfite-converted targeted amplicon sequencing.
This script uses an INDIRECT PROXY: unmethylated CpGs are preferentially
cleaved during apoptotic cell death (DNASE1L3, CAD), so cfDNA fragments from
cell types where a marker is unmethylated tend to END at CpG positions within
that marker window more often than fragments from other cell types.

This is the same principle as the genome-wide CpG-fraction analysis in
11_prebreak_chemo.R, here restricted to the 10 marker loci.

Usage
-----
    python 13_brain_origin.py \
        --bam_dir /path/to/bams \
        --sample_map sample_map.txt \
        --ref /path/to/hg38.fa \
        --out_dir rmds_analysis

Output files (in out_dir)
--------------------------
    brain_origin_per_marker.tsv   — one row per sample × marker
    brain_origin_celltype.tsv     — per-sample mean CpG fraction per cell type
    brain_origin_wide.tsv         — wide format: one row per sample, 4 score columns

Requirements: pysam, pandas
"""

import argparse
import os
import sys
from pathlib import Path

import pysam
import pandas as pd

# ── Marker coordinates: Lubotzky et al. 2022 JCI Insight, Suppl. Table S1 ──
# hg38 amplicon boundaries. Scores: fraction unmethylated in the given cell type.
# Higher score = more DNA from that cell type dying and releasing cfDNA.
#
# Format: (marker_name, cell_type, chrom, amp_start, amp_end)
MARKERS_HG38 = [
    ("AST1",    "astrocyte",       "chr7",   2_656_380,  2_656_519),
    ("WOX",     "astrocyte",       "chr16", 79_158_123, 79_158_241),
    ("PRDM2",   "astrocyte",       "chr1",   3_128_519,  3_128_632),
    ("509",     "neuron",          "chr4",   4_322_982,  4_323_100),
    ("ITF",     "neuron",          "chr16",    310_568,    310_671),
    ("SLC",     "neuron",          "chr17",  79_253_907, 79_254_047),
    ("ZNF238",  "neuron",          "chr1",  244_218_429, 244_218_560),
    ("NMR",     "oligodendrocyte", "chr16",  4_521_272,  4_521_415),
    ("TAF8",    "oligodendrocyte", "chr6",   42_046_310, 42_046_434),
    ("ZFP",     "oligodendrocyte", "chr6",   29_641_035, 29_641_177),
]

# Window around each amplicon centre used when fetching reads.
# ±2 kb gives ~50-150 reads at typical 1-5x WGS depth in this cohort.
DEFAULT_WINDOW = 2000


def detect_chr_prefix(bam: pysam.AlignmentFile) -> bool:
    """Return True if BAM contigs use 'chr' prefix."""
    for ref in bam.references:
        if ref.startswith("chr"):
            return True
    return False


def get_cpg_positions(fasta: pysam.FastaFile, chrom: str,
                      start: int, end: int) -> set:
    """Return 0-based positions where reference has C followed by G."""
    try:
        seq = fasta.fetch(chrom, max(0, start), end + 1).upper()
    except (KeyError, ValueError):
        return set()
    cpg = set()
    for i in range(len(seq) - 1):
        if seq[i] == 'C' and seq[i + 1] == 'G':
            cpg.add(start + i)
    return cpg


def score_marker(bam: pysam.AlignmentFile,
                 fasta: pysam.FastaFile,
                 chrom_bam: str,
                 chrom_ref: str,
                 amp_start: int,
                 amp_end: int,
                 window: int) -> tuple:
    """
    Count cfDNA 5'-end positions landing at CpGs within the window.

    The 5' end of each cfDNA fragment corresponds to a DNA break.
    For read1 (forward): break is at reference_start.
    For read2 (reverse): break is at reference_end.

    Returns (cpg_end_count, total_end_count, cpg_frac).
    """
    center = (amp_start + amp_end) // 2
    w_start = max(0, center - window)
    w_end = center + window

    cpg_pos = get_cpg_positions(fasta, chrom_ref, w_start, w_end)
    if not cpg_pos:
        return 0, 0, float("nan")

    cpg_ends = 0
    total_ends = 0
    seen = set()

    try:
        for read in bam.fetch(chrom_bam, w_start, w_end):
            if (read.is_unmapped
                    or read.is_secondary
                    or read.is_supplementary
                    or not read.is_proper_pair
                    or read.reference_start is None):
                continue
            # One count per read end (not per read pair) to capture both break sites
            uid = (read.query_name, read.is_read1)
            if uid in seen:
                continue
            seen.add(uid)

            # 5' end of THIS read = break point
            if read.is_reverse:
                pos = read.reference_end - 1  # 0-based last mapped base
            else:
                pos = read.reference_start    # 0-based first mapped base

            if w_start <= pos < w_end:
                total_ends += 1
                if pos in cpg_pos:
                    cpg_ends += 1

    except (ValueError, KeyError):
        return 0, 0, float("nan")

    if total_ends == 0:
        return 0, 0, float("nan")
    return cpg_ends, total_ends, cpg_ends / total_ends


def process_sample(bam_path: Path,
                   fasta: pysam.FastaFile,
                   sample_name: str,
                   window: int) -> list:
    rows = []
    with pysam.AlignmentFile(str(bam_path), "rb") as bam:
        use_chr = detect_chr_prefix(bam)
        for name, cell_type, chrom_hg38, amp_start, amp_end in MARKERS_HG38:
            chrom_bam = chrom_hg38 if use_chr else chrom_hg38.lstrip("chr")
            # Reference FASTA might also lack chr prefix
            try:
                fasta.fetch(chrom_hg38, 0, 1)
                chrom_ref = chrom_hg38
            except (KeyError, ValueError):
                chrom_ref = chrom_hg38.lstrip("chr")

            cpg_c, tot_c, frac = score_marker(
                bam, fasta, chrom_bam, chrom_ref, amp_start, amp_end, window)
            rows.append({
                "sample":       sample_name,
                "marker":       name,
                "cell_type":    cell_type,
                "chrom":        chrom_hg38,
                "amp_start":    amp_start,
                "amp_end":      amp_end,
                "window_bp":    window,
                "cpg_end_count": cpg_c,
                "total_end_count": tot_c,
                "cpg_frac":     frac,
            })
    return rows


def main():
    parser = argparse.ArgumentParser(
        description="Brain-cell-of-origin CpG scoring from WGS BAMs.")
    parser.add_argument("--bam_dir",    required=True,
                        help="Directory containing *.bam files")
    parser.add_argument("--sample_map", default=None,
                        help="Optional. If omitted, all BAMs found are "
                             "processed; metadata join happens downstream in R.")
    parser.add_argument("--ref",        required=True,
                        help="Path to hg38 FASTA (must have .fai index)")
    parser.add_argument("--out_dir",    default="rmds_analysis")
    parser.add_argument("--window",     type=int, default=DEFAULT_WINDOW,
                        help=f"±window bp around each amplicon centre "
                             f"(default {DEFAULT_WINDOW})")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)

    # Index BAMs by sample name. BAMs may be nested in per-sample subdirs
    # (e.g. EGAF*/AL890.bam) — walk recursively.
    bam_dir = Path(args.bam_dir)
    bam_index = {}
    for bam_path in sorted(bam_dir.rglob("*.bam")):
        bam_index[bam_path.stem] = bam_path
    print(f"Found {len(bam_index)} BAMs under {bam_dir}", flush=True)

    if args.sample_map:
        meta = pd.read_csv(args.sample_map, sep="\t")
        meta.columns = [c.strip() for c in meta.columns]
        sample_list = [(str(r.get("sample", "")).strip(), r)
                       for _, r in meta.iterrows()]
    else:
        sample_list = [(s, None) for s in sorted(bam_index.keys())]

    fasta = pysam.FastaFile(args.ref)

    all_rows = []
    n_found = 0
    for sname, row in sample_list:
        bam_path = bam_index.get(sname)
        if bam_path is None:
            continue
        n_found += 1
        print(f"[{n_found}] {sname} ...", end=" ", flush=True)
        rows = process_sample(bam_path, fasta, sname, args.window)
        if row is not None:
            meta_cols = {
                "timepoint": row.get("timepoint", ""),
                "subtype":   row.get("SUBTYPE",   ""),
                "FRAC":      row.get("FRAC",      float("nan")),
                "TOTAL_COV": row.get("TOTAL_COV", float("nan")),
                "patient":   row.get("SJID",      ""),
            }
            for r in rows:
                r.update(meta_cols)
        all_rows.extend(rows)
        print("ok", flush=True)

    fasta.close()

    if not all_rows:
        sys.exit("No matching BAMs found. Check --bam_dir and sample names in sample_map.")

    df = pd.DataFrame(all_rows)

    # ── Per-sample, per-cell-type score (mean CpG fraction) ──────────────────
    # Only group by metadata cols that actually exist (sample_map may be absent)
    group_cols = [c for c in ["sample", "cell_type", "timepoint", "subtype",
                              "FRAC", "TOTAL_COV", "patient"]
                  if c in df.columns]
    df_ct = (
        df.dropna(subset=["cpg_frac"])
        .groupby(group_cols)
        .agg(
            n_markers_with_data=("cpg_frac", "count"),
            score=("cpg_frac", "mean"),
            total_reads=("total_end_count", "sum"),
        )
        .reset_index()
    )

    # ── Wide format ───────────────────────────────────────────────────────────
    pivot_idx = [c for c in ["sample", "timepoint", "subtype",
                              "FRAC", "TOTAL_COV", "patient"]
                 if c in df_ct.columns]
    wide = df_ct.pivot_table(
        index=pivot_idx,
        columns="cell_type",
        values="score",
    ).reset_index()
    wide.columns.name = None
    rename_map = {
        "astrocyte":       "score_astrocyte",
        "neuron":          "score_neuron",
        "oligodendrocyte": "score_oligodendrocyte",
    }
    wide = wide.rename(columns={k: v for k, v in rename_map.items()
                                 if k in wide.columns})
    score_cols = [c for c in wide.columns if c.startswith("score_")]
    if score_cols:
        wide["score_brain_total"] = wide[score_cols].mean(axis=1)

    # ── Save ─────────────────────────────────────────────────────────────────
    per_marker_path = out_dir / "brain_origin_per_marker.tsv"
    celltype_path   = out_dir / "brain_origin_celltype.tsv"
    wide_path       = out_dir / "brain_origin_wide.tsv"

    df.to_csv(per_marker_path, sep="\t", index=False)
    df_ct.to_csv(celltype_path, sep="\t", index=False)
    wide.to_csv(wide_path, sep="\t", index=False)

    print(f"\nDone. Processed {n_found} samples.")
    print(f"  {per_marker_path}  ({len(df)} rows)")
    print(f"  {celltype_path}")
    print(f"  {wide_path}")

    # Optional quick summary (only if metadata was joined)
    if "subtype" in wide.columns and "timepoint" in wide.columns:
        mb_sub = wide[wide["subtype"].isin(["SHH", "G3", "G4", "WNT"])]
        if not mb_sub.empty:
            print("\nMean scores by timepoint (MB only):")
            print(mb_sub.groupby("timepoint")[score_cols + ["score_brain_total"]]
                  .mean().round(5).to_string())


if __name__ == "__main__":
    main()
