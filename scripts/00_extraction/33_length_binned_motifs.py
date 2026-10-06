#!/usr/bin/env python3
"""Build fixed-fragment-length 5' motif matrices from extended motif tables.

The extended tables already contain counts indexed by motif_5p, pre_break and
frag_len.  Summing over pre_break within prespecified length bins gives a direct
test of motif composition at fixed fragment length, without rereading BAMs.

This script is array-job friendly.  Each task writes one small TSV per sample;
``--aggregate`` combines completed sample files into the analysis matrix.
"""

import argparse
import gzip
from itertools import product
from pathlib import Path

import pandas as pd


BASES = "ACGT"
MOTIFS = ["".join(x) for x in product(BASES, repeat=4)]
BIN_SPECS = (
    ("100_150", 100, 150),
    ("151_180", 151, 180),
    ("181_220", 181, 220),
)
PRIMARY_TP = {"baseline", "mid-chemo", "post-rt"}
MB_SUBTYPES = {"SHH", "G3", "G4", "WNT"}


def primary_samples(sample_map: Path) -> list[str]:
    meta = pd.read_csv(sample_map, sep="\t")
    required = {"sample", "timepoint", "SUBTYPE"}
    missing = required.difference(meta.columns)
    if missing:
        raise ValueError(f"sample map missing columns: {sorted(missing)}")
    keep = meta["timepoint"].isin(PRIMARY_TP) & meta["SUBTYPE"].isin(MB_SUBTYPES)
    return sorted(meta.loc[keep, "sample"].dropna().astype(str).unique())


def bin_index(length: int):
    for idx, (_, lo, hi) in enumerate(BIN_SPECS):
        if lo <= length <= hi:
            return idx
    return None


def process_one(path: Path, sample: str, out_path: Path) -> None:
    motif_index = {m: i for i, m in enumerate(MOTIFS)}
    counts = [[0] * len(MOTIFS) for _ in BIN_SPECS]
    with gzip.open(path, "rt") as handle:
        header = handle.readline().rstrip("\n").split("\t")
        pos = {name: i for i, name in enumerate(header)}
        required = {"count", "motif_5p", "frag_len"}
        if not required.issubset(pos):
            raise ValueError(f"{path} missing {sorted(required.difference(pos))}")
        max_pos = max(pos[x] for x in required)
        for line in handle:
            fields = line.rstrip("\n").split("\t")
            if len(fields) <= max_pos:
                continue
            motif = fields[pos["motif_5p"]].upper()
            mi = motif_index.get(motif)
            if mi is None:
                continue
            try:
                length = int(fields[pos["frag_len"]])
                count = int(fields[pos["count"]])
            except ValueError:
                continue
            bi = bin_index(length)
            if bi is not None and count > 0:
                counts[bi][mi] += count

    rows = []
    for bi, (label, lo, hi) in enumerate(BIN_SPECS):
        row = {"sample": sample, "length_bin": label, "length_lo": lo,
               "length_hi": hi, "n_fragments": sum(counts[bi])}
        row.update(dict(zip(MOTIFS, counts[bi])))
        rows.append(row)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, sep="\t", index=False)


def aggregate(per_sample_dir: Path, out_path: Path) -> None:
    paths = sorted(per_sample_dir.glob("*_length_binned_motifs.tsv"))
    if not paths:
        raise SystemExit(f"No per-sample outputs found under {per_sample_dir}")
    out = pd.concat((pd.read_csv(p, sep="\t") for p in paths), ignore_index=True)
    if out.duplicated(["sample", "length_bin"]).any():
        raise ValueError("duplicate sample/length-bin rows in aggregate")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, sep="\t", index=False)
    print(f"Wrote {out_path}: {out['sample'].nunique()} samples, {len(out)} rows")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input_dir", default="data/motifs_4mer")
    ap.add_argument("--sample_map", default="sample_map.txt")
    ap.add_argument("--out_dir", default="rmds_analysis/length_binned_per_sample")
    ap.add_argument("--aggregate_out", default="rmds_analysis/length_binned_motif_counts.tsv.gz")
    ap.add_argument("--chunk_idx", type=int, default=0)
    ap.add_argument("--n_chunks", type=int, default=1)
    ap.add_argument("--aggregate", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    if args.aggregate:
        aggregate(out_dir, Path(args.aggregate_out))
        return

    samples = primary_samples(Path(args.sample_map))
    selected = samples[args.chunk_idx::args.n_chunks]
    print(f"Chunk {args.chunk_idx}/{args.n_chunks}: {len(selected)} samples")
    done = skipped = missing = 0
    for sample in selected:
        src = Path(args.input_dir) / f"{sample}_4mer_extended_final.txt.gz"
        dst = out_dir / f"{sample}_length_binned_motifs.tsv"
        if dst.exists():
            skipped += 1
            continue
        if not src.exists():
            print(f"MISSING {src}")
            missing += 1
            continue
        print(f"{sample}: {src}", flush=True)
        process_one(src, sample, dst)
        done += 1
    print(f"done={done} skipped={skipped} missing={missing}")


if __name__ == "__main__":
    main()
