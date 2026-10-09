#!/usr/bin/env python3
"""Compact coordinate-level cfDNA break files for transfer and analysis.

Reads completed *_break_sites.tsv.gz files, collapses exact fragment-coordinate
duplicates with a bounded streaming cache, and writes:
  compact_break_bins_500kb.tsv.gz
  compact_break_motifs.tsv.gz
  compact_break_qc.tsv

The combined outputs are small enough to copy to a workstation. Raw coordinate
files remain on the server for future gene/regulatory annotation.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import multiprocessing as mp
from collections import Counter, OrderedDict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("input_dir")
    p.add_argument("output_dir")
    p.add_argument("--jobs", type=int, default=48)
    p.add_argument("--bin-size", type=int, default=500_000)
    p.add_argument(
        "--dedup-window",
        type=int,
        default=250_000,
        help="Number of recent fragment keys retained for coordinate deduplication",
    )
    return p.parse_args()


def process_one(task: tuple[str, str, int, int]) -> tuple[str, str, str, dict]:
    input_name, output_name, bin_size, dedup_window = task
    src = Path(input_name)
    out_dir = Path(output_name)
    sample = src.name.removesuffix("_break_sites.tsv.gz")
    bin_path = out_dir / "per_sample" / f"{sample}_break_bins.tsv.gz"
    motif_path = out_dir / "per_sample" / f"{sample}_break_motifs.tsv.gz"

    raw_bins: Counter[tuple[str, int]] = Counter()
    dedup_bins: Counter[tuple[str, int]] = Counter()
    raw_motifs: Counter[str] = Counter()
    dedup_motifs: Counter[str] = Counter()
    seen: OrderedDict[tuple[str, int, int, str, int], None] = OrderedDict()
    rows = duplicates = 0

    with gzip.open(src, "rt", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {
            "chrom",
            "cut_start",
            "cut_end",
            "sample",
            "frag_len",
            "strand",
            "motif_5p",
        }
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise RuntimeError(f"Missing required columns in {src}")

        for row in reader:
            rows += 1
            chrom = row["chrom"]
            cut_start = int(row["cut_start"])
            cut_end = int(row["cut_end"])
            frag_len = int(row["frag_len"])
            strand = row["strand"]
            motif = row["motif_5p"]
            bin_start = (cut_start // bin_size) * bin_size
            bin_key = (chrom, bin_start)
            fragment_key = (chrom, cut_start, cut_end, strand, frag_len)

            raw_bins[bin_key] += 1
            raw_motifs[motif] += 1

            if fragment_key in seen:
                duplicates += 1
                # Refresh the key so highly duplicated fragments remain cached.
                seen.move_to_end(fragment_key)
                continue
            seen[fragment_key] = None
            if len(seen) > dedup_window:
                seen.popitem(last=False)

            dedup_bins[bin_key] += 1
            dedup_motifs[motif] += 1

    with gzip.open(bin_path, "wt", newline="", compresslevel=6) as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(
            ["sample", "chrom", "start", "end", "count_raw", "count_deduplicated"]
        )
        for chrom, start in sorted(
            raw_bins, key=lambda x: (x[0].removeprefix("chr").zfill(2), x[1])
        ):
            writer.writerow(
                [
                    sample,
                    chrom,
                    start,
                    start + bin_size,
                    raw_bins[(chrom, start)],
                    dedup_bins[(chrom, start)],
                ]
            )

    raw_total = sum(raw_motifs.values())
    dedup_total = sum(dedup_motifs.values())
    with gzip.open(motif_path, "wt", newline="", compresslevel=6) as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(
            [
                "sample",
                "motif_5p",
                "count_raw",
                "count_deduplicated",
                "fraction_raw",
                "fraction_deduplicated",
            ]
        )
        for motif in sorted(raw_motifs):
            writer.writerow(
                [
                    sample,
                    motif,
                    raw_motifs[motif],
                    dedup_motifs[motif],
                    f"{raw_motifs[motif] / raw_total:.12g}",
                    f"{dedup_motifs[motif] / dedup_total:.12g}",
                ]
            )

    qc = {
        "sample": sample,
        "rows_raw": rows,
        "rows_deduplicated": rows - duplicates,
        "coordinate_duplicates": duplicates,
        "coordinate_duplicate_fraction": duplicates / rows if rows else 0.0,
        "nonempty_bins": len(raw_bins),
    }
    return sample, str(bin_path), str(motif_path), qc


def append_gzip_table(
    output_path: Path, source_paths: list[str], expected_header: list[str]
) -> None:
    with gzip.open(output_path, "wt", newline="", compresslevel=6) as out:
        writer = csv.writer(out, delimiter="\t", lineterminator="\n")
        writer.writerow(expected_header)
        for source in source_paths:
            with gzip.open(source, "rt", newline="") as handle:
                reader = csv.reader(handle, delimiter="\t")
                header = next(reader)
                if header != expected_header:
                    raise RuntimeError(f"Unexpected header in {source}")
                writer.writerows(reader)


def main() -> int:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    if not input_dir.is_dir():
        raise SystemExit(f"ERROR: input directory not found: {input_dir}")
    if args.jobs < 1:
        raise SystemExit("ERROR: --jobs must be positive")
    if args.bin_size < 1 or args.dedup_window < 1:
        raise SystemExit("ERROR: bin size and dedup window must be positive")

    files = sorted(input_dir.glob("*_break_sites.tsv.gz"))
    if not files:
        raise SystemExit(f"ERROR: no completed break-site files found in {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "per_sample").mkdir(exist_ok=True)
    tasks = [
        (str(path), str(output_dir), args.bin_size, args.dedup_window)
        for path in files
    ]
    print(f"Compacting {len(tasks)} samples with {args.jobs} workers...")
    with mp.Pool(processes=min(args.jobs, len(tasks))) as pool:
        results = list(pool.imap_unordered(process_one, tasks))
    results.sort(key=lambda x: x[0])

    bin_header = [
        "sample",
        "chrom",
        "start",
        "end",
        "count_raw",
        "count_deduplicated",
    ]
    motif_header = [
        "sample",
        "motif_5p",
        "count_raw",
        "count_deduplicated",
        "fraction_raw",
        "fraction_deduplicated",
    ]
    append_gzip_table(
        output_dir / f"compact_break_bins_{args.bin_size // 1000}kb.tsv.gz",
        [x[1] for x in results],
        bin_header,
    )
    append_gzip_table(
        output_dir / "compact_break_motifs.tsv.gz",
        [x[2] for x in results],
        motif_header,
    )

    with (output_dir / "compact_break_qc.tsv").open("w", newline="") as handle:
        fields = list(results[0][3])
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(x[3] for x in results)

    print(f"Finished. Combined outputs written to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
