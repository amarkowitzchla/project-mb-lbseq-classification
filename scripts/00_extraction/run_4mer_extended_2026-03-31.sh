#!/bin/bash
# =============================================================================
# cfDNA 5-prime End Motif Extractor
# =============================================================================
# This script extracts 5-prime end motif features from a cfDNA BAM file.
# For each properly paired, uniquely mapped, non-supplementary R1 read:
#   (1) 4 bp 5-prime end motif (biological cut site from read sequence)
#   (2) 4 bp pre-break genomic context (from reference FASTA via bedtools)
#   (3) Fragment length (from TLEN)
#   (4) 3 bp cleavage context: -1 (last of pre_break) | 0 (cut) | +1 (motif+1)
# Soft-clipped reads are excluded entirely (cigar ~ /S/).
# CIGAR-based reference length (POSIX-compliant) is used for reverse reads
# to correctly handle insertions and deletions.
# Output columns: fraction | count | motif_5p | pre_break | frag_len | cleavage_ctx | sample
# Output is sorted by descending frequency.
#
# Usage:
#   Single BAM  : bash script.sh sample.bam
#   GNU parallel: ls *.bam | parallel -j 10 bash script.sh {}
#   SLURM array : sbatch array_job.sh /path/to/bams
# =============================================================================

set -eo pipefail

BAM=$1
REF="/scratch/local/resources/hg38.fa"
NAME=$(basename "${BAM}" .bam)

# Temporary files with Process ID for parallel safety
TMP_BED="${NAME}_tmp_$$.bed"
TMP_FASTA="${NAME}_seqs_$$.txt"
TMP_SORTED="${NAME}_tmp_sorted_$$.txt"

# Cleanup all temp files on exit (handles both success and error)
trap 'rm -f "${TMP_BED}" "${TMP_FASTA}" "${TMP_SORTED}"' EXIT

# Input validation
if [[ -z "${BAM}" ]]; then
    echo "ERROR: No BAM file provided."
    echo "Usage: $0 sample.bam"
    exit 1
fi

if [[ ! -f "${BAM}" ]]; then
    echo "ERROR: BAM file not found: ${BAM}"
    exit 1
fi

if [[ ! -f "${REF}" ]]; then
    echo "ERROR: Reference FASTA not found at ${REF}"
    exit 1
fi

if [[ ! -f "${REF}.fai" ]]; then
    echo "ERROR: Reference FASTA index not found. Run: samtools faidx ${REF}"
    exit 1
fi

# -----------------------------------------------------------------------------
# Step 1: Extract 5-prime coordinates and write BED file
# -----------------------------------------------------------------------------
echo "Step 1/3: Extracting 5' coordinates from ${NAME}..."

# -q 60  : Unique mapq
# -f 66  : R1 + properly paired
# -F 2048: Exclude supplementary alignments
samtools view -q 60 -f 66 -F 2048 "${BAM}" | \
awk 'BEGIN {OFS="\t"}

# 100% POSIX-compliant CIGAR parser.
# Iterates character by character to sum reference-consuming operations.
# Reference-consuming ops: M, D, N, =, X
# Non-reference-consuming: I, S, H, P
function ref_len(c,    i, n, len, num, char) {
    len = 0
    num = ""
    n = length(c)
    for (i = 1; i <= n; i++) {
        char = substr(c, i, 1)
        if (char ~ /[0-9]/) {
            num = num char
        } else {
            if (char ~ /[MDN=X]/) len += int(num)
            num = ""
        }
    }
    return len
}

{
    flag  = $2
    chrom = $3
    pos   = $4
    seq   = $10
    tlen  = $9
    cigar = $6

    # Skip unmapped
    if (chrom == "*") next

    is_rev = int(flag / 16) % 2

    # Exclude any read with soft-clipping anywhere in the CIGAR.
    # Soft clips indicate adapter contamination, SV breakpoints, or
    # misalignment — all of which corrupt the true 5-prime end motif.
    if (cigar ~ /S/) next

    # Absolute fragment length from TLEN
    frag_len = (tlen < 0) ? -tlen : tlen
    if (frag_len == 0) next

    if (!is_rev) {
        # Forward R1: 5-prime end is the leftmost mapped position (pos, 1-based)
        # BED is 0-based half-open: pos-5 to pos-1 fetches 4 bases upstream
        if (pos - 5 < 0) next
        print chrom, pos-5, pos-1, seq "|" frag_len, ".", "+"
    } else {
        # Reverse R1: 5-prime end is the rightmost aligned reference base.
        # CIGAR-derived ref length correctly handles indels:
        # - Insertions (I): consume read bases but not reference bases
        # - Deletions (D): consume reference bases but not read bases
        end = pos + ref_len(cigar) - 1
        # Fetch 4 bases downstream of fragment end (bedtools -s will RC for us)
        print chrom, end, end+4, seq "|" frag_len, ".", "-"
    }
}' > "${TMP_BED}"

# -----------------------------------------------------------------------------
# Step 2: Bulk fetch reference sequences via bedtools
# -----------------------------------------------------------------------------
echo "Step 2/3: Bulk fetching reference sequences..."

# -s   : Strand-aware (auto RC for minus strand intervals)
# -tab : Output as two-column tab-separated (name \t sequence)
# -name: Embed BED name field into FASTA header
bedtools getfasta \
    -fi "${REF}" \
    -bed "${TMP_BED}" \
    -s -tab -name \
    > "${TMP_FASTA}"

# -----------------------------------------------------------------------------
# Step 3: Compute motif counts and write final output
# -----------------------------------------------------------------------------
echo "Step 3/3: Finalizing motif counts..."

awk -F'\t' -v n="${NAME}" '
function rc(s,    r, j, b, c) {
    r = ""
    for (j = length(s); j > 0; j--) {
        b = substr(s, j, 1)
        if      (b == "A") c = "T"
        else if (b == "T") c = "A"
        else if (b == "C") c = "G"
        else if (b == "G") c = "C"
        else               c = b
        r = r c
    }
    return r
}
{
    # Header field format from bedtools -name -s:
    # ReadSeq|FragLen::Chrom:Start-End(Strand)
    split($1, meta, "::")
    split(meta[1], fields, "|")
    read_seq  = fields[1]
    frag_len  = fields[2]
    pre_break = toupper($2)

    # Determine strand from bedtools header
    strand = (index($1, "(+)") > 0) ? "+" : "-"

    # Extract 5-prime motif from read sequence
    if (strand == "+") {
        motif_5p = substr(read_seq, 1, 4)
    } else {
        # Reverse R1: biological 5-prime is RC of last 4 bases of reported seq
        motif_5p = rc(substr(read_seq, length(read_seq)-3, 4))
    }

    # Cleavage context: -1 (last base of pre_break) | cut site | +1
    minus1       = substr(pre_break, 4, 1)
    cut_base     = substr(motif_5p,  1, 1)
    plus1        = substr(motif_5p,  2, 1)
    cleavage_ctx = minus1 cut_base plus1

    # Quality filters
    if (motif_5p     ~ /N/ || length(motif_5p)     != 4) next
    if (pre_break    ~ /N/ || length(pre_break)     != 4) next
    if (cleavage_ctx ~ /N/ || length(cleavage_ctx)  != 3) next

    key = motif_5p "\t" pre_break "\t" frag_len "\t" cleavage_ctx
    counts[key]++
    total++
}
END {
    for (k in counts) {
        if (total > 0)
            print counts[k]/total "\t" counts[k] "\t" k "\t" n
    }
}' "${TMP_FASTA}" | sort -k1,1nr > "${TMP_SORTED}"

# Prepend header after sorting so it stays at the top
{ echo -e "fraction\tcount\tmotif_5p\tpre_break\tfrag_len\tcleavage_ctx\tsample"
  cat "${TMP_SORTED}"
} > "${NAME}_4mer_extended_final.txt"

echo "Done. Output: ${NAME}_4mer_extended_final.txt"
