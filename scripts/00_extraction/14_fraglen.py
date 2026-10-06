#!/usr/bin/env python3
"""
14_fraglen.py

Compute per-sample cfDNA fragment length distributions from BAM files.
Runs on the cluster where BAMs are stored.

For each BAM produces:
  - Fragment length histogram (1 bp bins, 50–500 bp)
  - Summary statistics: mean, median, mode, short fraction (<150 bp),
    nucleosome-free ratio (NFR: <150 bp / 150–250 bp),
    mono-nucleosomal fraction (150–250 bp)

Usage:
    python 14_fraglen.py \
        --bam_dir /path/to/bams \
        --sample_map sample_map.txt \
        --out_dir rmds_analysis \
        [--min_mapq 20] [--min_len 50] [--max_len 500]

Output:
    rmds_analysis/fraglen_histogram.tsv   (sample × length bin matrix)
    rmds_analysis/fraglen_summary.tsv     (per-sample summary stats)
"""

import argparse
import os
from pathlib import Path
from collections import Counter

import pysam
import pandas as pd
import numpy as np


MIN_LEN = 50
MAX_LEN = 500


def process_bam(bam_path: Path, sample: str,
                min_mapq: int, min_len: int, max_len: int) -> dict:
    counts = Counter()
    with pysam.AlignmentFile(str(bam_path), "rb") as bam:
        # until_eof=True iterates all reads without requiring a .bai index
        for read in bam.fetch(until_eof=True):
            if (read.is_unmapped
                    or read.is_secondary
                    or read.is_supplementary
                    or read.is_duplicate
                    or not read.is_proper_pair
                    or not read.is_read1          # count each fragment once
                    or read.mapping_quality < min_mapq):
                continue
            tlen = abs(read.template_length)
            if min_len <= tlen <= max_len:
                counts[tlen] += 1
    return counts


def summarise(counts: Counter, min_len: int, max_len: int) -> dict:
    lengths = np.array(list(counts.keys()))
    freqs   = np.array(list(counts.values()))
    total   = freqs.sum()
    if total == 0:
        return {}

    # Weighted mean and median
    mean_len = np.average(lengths, weights=freqs)

    cumsum = np.cumsum(freqs[np.argsort(lengths)])
    sorted_lens = np.sort(lengths)
    median_len = sorted_lens[np.searchsorted(cumsum, total / 2)]

    mode_len = lengths[np.argmax(freqs)]

    short_n   = sum(v for k, v in counts.items() if k < 150)
    mono_n    = sum(v for k, v in counts.items() if 150 <= k <= 250)
    di_n      = sum(v for k, v in counts.items() if 251 <= k <= 400)
    nfr_ratio = short_n / mono_n if mono_n > 0 else float("nan")

    return {
        "n_fragments": int(total),
        "mean_length": round(mean_len, 2),
        "median_length": int(median_len),
        "mode_length": int(mode_len),
        "short_frac":  round(short_n / total, 5),    # <150 bp
        "mono_frac":   round(mono_n / total, 5),     # 150-250 bp
        "di_frac":     round(di_n   / total, 5),     # 251-400 bp
        "nfr_ratio":   round(nfr_ratio, 5),          # short / mono
    }


def aggregate(per_sample_dir: Path, out_dir: Path) -> None:
    """Combine all per-sample TSVs into the final histogram + summary files."""
    hist_paths = sorted(per_sample_dir.glob("*_hist.tsv"))
    sum_paths  = sorted(per_sample_dir.glob("*_summary.tsv"))
    print(f"Aggregating {len(hist_paths)} hist + {len(sum_paths)} summary files",
          flush=True)

    if hist_paths:
        hist_df = pd.concat([pd.read_csv(p, sep="\t") for p in hist_paths],
                            ignore_index=True)
        hist_df.to_csv(out_dir / "fraglen_histogram.tsv", sep="\t", index=False)
        print(f"  Wrote {out_dir}/fraglen_histogram.tsv  "
              f"({len(hist_df)} samples)", flush=True)

    if sum_paths:
        sum_df = pd.concat([pd.read_csv(p, sep="\t") for p in sum_paths],
                           ignore_index=True)
        sum_df.to_csv(out_dir / "fraglen_summary.tsv", sep="\t", index=False)
        print(f"  Wrote {out_dir}/fraglen_summary.tsv", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bam_dir",    required=True)
    parser.add_argument("--out_dir",    default="rmds_analysis")
    parser.add_argument("--min_mapq",   type=int, default=20)
    parser.add_argument("--min_len",    type=int, default=MIN_LEN)
    parser.add_argument("--max_len",    type=int, default=MAX_LEN)
    parser.add_argument("--chunk_idx",  type=int, default=0,
                        help="0-based index of this SLURM array task")
    parser.add_argument("--n_chunks",   type=int, default=1,
                        help="Total number of SLURM array tasks")
    parser.add_argument("--aggregate",  action="store_true",
                        help="Skip processing; just aggregate per-sample TSVs")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)
    per_sample_dir = out_dir / "per_sample"
    per_sample_dir.mkdir(exist_ok=True)

    if args.aggregate:
        aggregate(per_sample_dir, out_dir)
        return

    # BAMs may be nested in per-sample subdirectories. Walk recursively.
    bam_index = {p.stem: p
                 for p in Path(args.bam_dir).rglob("*.bam")}
    print(f"Found {len(bam_index)} BAMs under {args.bam_dir}", flush=True)

    # Stride partition: chunk_idx gets every n_chunks-th sample.
    # Better load balance than contiguous chunks if BAM sizes vary.
    all_samples = sorted(bam_index.keys())
    sample_names = all_samples[args.chunk_idx::args.n_chunks]
    print(f"Chunk {args.chunk_idx}/{args.n_chunks}: {len(sample_names)} samples",
          flush=True)

    n_done = 0
    n_skipped = 0
    for sname in sample_names:
        bam_path = bam_index.get(sname)
        if bam_path is None:
            continue

        sum_path  = per_sample_dir / f"{sname}_summary.tsv"
        hist_path = per_sample_dir / f"{sname}_hist.tsv"
        if sum_path.exists() and hist_path.exists():
            n_skipped += 1
            continue

        print(f"  {sname} ...", end=" ", flush=True)
        counts = process_bam(bam_path, sname,
                             args.min_mapq, args.min_len, args.max_len)
        print(f"{sum(counts.values())} fragments", flush=True)

        # Write per-sample histogram immediately
        hist_row = {"sample": sname}
        for length in range(args.min_len, args.max_len + 1):
            hist_row[str(length)] = counts.get(length, 0)
        pd.DataFrame([hist_row]).to_csv(hist_path, sep="\t", index=False)

        # Write per-sample summary immediately
        stats = summarise(counts, args.min_len, args.max_len)
        stats["sample"] = sname
        pd.DataFrame([stats]).to_csv(sum_path, sep="\t", index=False)
        n_done += 1

    print(f"Chunk {args.chunk_idx} done: {n_done} processed, "
          f"{n_skipped} skipped (already had outputs)", flush=True)


if __name__ == "__main__":
    main()
