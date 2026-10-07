#!/bin/bash
# Per-BAM rMDS computation using FinaleToolkit (Bandaru et al. 2026 method).
#
# What it does:
#   1. 5'-end 4-mer motif freqs + counts per 500-kb bin (FinaleToolkit interval-end-motifs)
#   2. rMDS (normalized Shannon entropy) per bin        (FinaleToolkit interval-mds)
#   3. Right-end 4-mer motif freqs per 500-kb bin       (FinaleToolkit -n flag: reverse-complemented
#                                                        4-mer at each fragment's right end, i.e. the
#                                                        minus-strand 5' end)
#   4. rMDS per bin for the right end
#      (outputs from the -n run keep the _3p suffix)
#   5. Genome-wide aggregated 4-mer motifs + MDS        (sanity check vs existing _corrected.txt)
#   6. Joined per-bin TSV with everything
#
# Usage:
#   ./cfdna_rmds_slurm.sh /path/to/SAMPLE1.bam &
# Or via sbatch:
#   sbatch cfdna_rmds_slurm.sh /path/to/SAMPLE1.bam
#
# Required env vars:
#   REF        : indexed FASTA matching the BAM (e.g. hg38.fa)
#
# Optional env vars:
#   OUTDIR        : output dir                         (default: ./rmds_out)
#   BIN_SIZE      : bin size in bp                     (default: 500000)
#   BINS_BED      : 500-kb autosomal BED               (default: ./bins_500kb_autosomes.bed,
#                                                       auto-generated from $REF.fai if absent)
#   FRAG_MIN      : min fragment length                (default: 50)
#   FRAG_MAX      : max fragment length                (default: 500)
#   MIN_MAPQ      : min mapping quality                (default: 20, matches FinaleToolkit default)
#   CPUS          : FinaleToolkit worker processes     (default: $SLURM_CPUS_PER_TASK or 2)
#   KMER          : k-mer length                       (default: 4)
#
# Dependencies (install once on cluster):
#   pip install --user finaletoolkit
#   samtools (standard) and awk

#SBATCH --job-name=rmds
#SBATCH --output=logs/rmds_%x_%j.out
#SBATCH --error=logs/rmds_%x_%j.err
#SBATCH --time=04:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2

set -euo pipefail

BAM="${1:-}"
if [[ -z "$BAM" ]]; then
    echo "Usage: $0 <bam>" >&2
    exit 1
fi
[[ -f "$BAM" ]] || { echo "ERROR: BAM not found: $BAM" >&2; exit 1; }

: "${REF:?Set REF env var to your indexed FASTA matching the BAM}"
[[ -f "$REF" ]] || { echo "ERROR: REF not found: $REF" >&2; exit 1; }
[[ -f "$REF.fai" ]] || { echo "ERROR: $REF.fai not found - run 'samtools faidx $REF'" >&2; exit 1; }

OUTDIR="${OUTDIR:-./rmds_out}"
BIN_SIZE="${BIN_SIZE:-500000}"
BINS_BED="${BINS_BED:-./bins_$((BIN_SIZE/1000))kb_autosomes.bed}"
FRAG_MIN="${FRAG_MIN:-50}"
FRAG_MAX="${FRAG_MAX:-500}"
MIN_MAPQ="${MIN_MAPQ:-20}"
CPUS="${CPUS:-${SLURM_CPUS_PER_TASK:-2}}"
KMER="${KMER:-4}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

mkdir -p "$OUTDIR" logs

# Auto-generate bins BED if missing
if [[ ! -f "$BINS_BED" ]]; then
    echo "[$(date)] Generating bins BED: $BINS_BED"
    bash "$SCRIPT_DIR/make_bins_bed.sh" "$REF.fai" "$BIN_SIZE" "$BINS_BED"
fi

NAME=$(basename "$BAM" .bam)
TMP="$OUTDIR/$NAME.tmp.$$"
trap 'rm -rf "$TMP"' EXIT
mkdir -p "$TMP"

echo "[$(date)] rMDS pipeline start"
echo "  BAM       = $BAM"
echo "  NAME      = $NAME"
echo "  REF       = $REF"
echo "  BINS_BED  = $BINS_BED"
echo "  BIN_SIZE  = $BIN_SIZE"
echo "  FRAG_LEN  = $FRAG_MIN-$FRAG_MAX"
echo "  MIN_MAPQ  = $MIN_MAPQ"
echo "  KMER      = $KMER"
echo "  CPUS      = $CPUS"
echo "  OUTDIR    = $OUTDIR"

# Common FinaleToolkit args
FT_COMMON=(
    -k "$KMER"
    -min "$FRAG_MIN"
    -max "$FRAG_MAX"
    -q "$MIN_MAPQ"
    -w "$CPUS"
    -v
)

# ---- (1) 5'-end motifs per bin ----
echo "[$(date)] (1/7) interval-end-motifs 5'"
finaletoolkit interval-end-motifs \
    "${FT_COMMON[@]}" \
    -B \
    -o "$OUTDIR/${NAME}_motifs_5p.tsv" \
    "$BAM" "$REF" "$BINS_BED"

# ---- (2) 5' interval rMDS ----
echo "[$(date)] (2/7) interval-mds 5'"
finaletoolkit interval-mds \
    "$OUTDIR/${NAME}_motifs_5p.tsv" \
    "$OUTDIR/${NAME}_rmds_5p.bed"

# ---- (3) right-end (minus-strand 5') motifs per bin ----
echo "[$(date)] (3/7) interval-end-motifs right end (-n)"
finaletoolkit interval-end-motifs \
    "${FT_COMMON[@]}" \
    -B -n \
    -o "$OUTDIR/${NAME}_motifs_3p.tsv" \
    "$BAM" "$REF" "$BINS_BED"

# ---- (4) right-end interval rMDS ----
echo "[$(date)] (4/7) interval-mds right end"
finaletoolkit interval-mds \
    "$OUTDIR/${NAME}_motifs_3p.tsv" \
    "$OUTDIR/${NAME}_rmds_3p.bed"

# (Per-bin fragment counts come from the 'count' column of the interval-end-motifs
# output; no separate samtools step needed.)

# ---- (5) Genome-wide aggregated motifs + global MDS (sanity check) ----
echo "[$(date)] (5/6) genome-wide end-motifs + mds (sanity)"
finaletoolkit end-motifs \
    "${FT_COMMON[@]}" \
    -B \
    -o "$OUTDIR/${NAME}_global_motifs_5p.tsv" \
    "$BAM" "$REF"

finaletoolkit mds "$OUTDIR/${NAME}_global_motifs_5p.tsv" \
    > "$OUTDIR/${NAME}_global_mds_5p.txt"

finaletoolkit end-motifs \
    "${FT_COMMON[@]}" \
    -B -n \
    -o "$OUTDIR/${NAME}_global_motifs_3p.tsv" \
    "$BAM" "$REF"

finaletoolkit mds "$OUTDIR/${NAME}_global_motifs_3p.tsv" \
    > "$OUTDIR/${NAME}_global_mds_3p.txt"

# ---- (6) Build a single joined per-bin TSV ----
# Columns: chrom, start, end, bin_name, rmds_5p, rmds_3p, n_frags
# Sources (note FinaleToolkit's actual output layouts):
#   rmds_5p.bed / rmds_3p.bed : 5 cols -> chrom, start, end, name, mds
#   motifs_5p.tsv : header + 4 metadata cols + count + 256 motif freq cols
#                   (cols: contig, start, stop, name, count, AAAA, ...)
echo "[$(date)] (6/6) joining outputs into ${NAME}_rmds_combined.tsv"

awk -v OFS='\t' '
    FILENAME == ARGV[1] {                                # rmds_3p.bed
        rm3[$1"\t"$2"\t"$3] = $5
        next
    }
    FILENAME == ARGV[2] && FNR > 1 {                     # motifs_5p.tsv (skip header)
        fc[$1"\t"$2"\t"$3] = $5
        next
    }
    FILENAME == ARGV[3] {                                # rmds_5p.bed (master walk)
        k = $1"\t"$2"\t"$3
        name = ($4 == "" ? "." : $4)
        v5 = $5
        v3 = (k in rm3 ? rm3[k] : "NA")
        nf = (k in fc  ? fc[k]  : "0")
        printf "%s\t%s\t%s\t%s\t%s\n", k, name, v5, v3, nf
    }
' "$OUTDIR/${NAME}_rmds_3p.bed" "$OUTDIR/${NAME}_motifs_5p.tsv" "$OUTDIR/${NAME}_rmds_5p.bed" \
  > "$TMP/combined.body"

{
    printf "chrom\tstart\tend\tbin_name\trmds_5p\trmds_3p\tn_frags\n"
    cat "$TMP/combined.body"
} > "$OUTDIR/${NAME}_rmds_combined.tsv"

# ---- summary ----
N_BINS=$(awk 'NR>1' "$OUTDIR/${NAME}_rmds_combined.tsv" | wc -l)
N_BINS_NONZERO=$(awk 'NR>1 && $7>0' "$OUTDIR/${NAME}_rmds_combined.tsv" | wc -l)
N_FRAGS=$(awk 'NR>1 {s+=$7} END {print s+0}' "$OUTDIR/${NAME}_rmds_combined.tsv")
GLOB_MDS_5P=$(cat "$OUTDIR/${NAME}_global_mds_5p.txt" 2>/dev/null | tr -d '\n')
GLOB_MDS_3P=$(cat "$OUTDIR/${NAME}_global_mds_3p.txt" 2>/dev/null | tr -d '\n')

{
    echo "bam	$BAM"
    echo "ref	$REF"
    echo "bins_bed	$BINS_BED"
    echo "bin_size	$BIN_SIZE"
    echo "min_mapq	$MIN_MAPQ"
    echo "frag_len	$FRAG_MIN-$FRAG_MAX"
    echo "kmer	$KMER"
    echo "n_bins	$N_BINS"
    echo "n_bins_with_frags	$N_BINS_NONZERO"
    echo "total_frags_in_bins	$N_FRAGS"
    echo "global_mds_5p	$GLOB_MDS_5P"
    echo "global_mds_3p	$GLOB_MDS_3P"
    echo "ts	$(date -Iseconds)"
} > "$OUTDIR/${NAME}_summary.txt"

echo "[$(date)] rMDS pipeline done: $NAME"
echo "  bins             = $N_BINS"
echo "  bins w/ frags    = $N_BINS_NONZERO"
echo "  global MDS (5')  = $GLOB_MDS_5P"
echo "  global MDS (3')  = $GLOB_MDS_3P"
