#!/usr/bin/env bash
# =============================================================================
# Locked cfDNA fragment-end extractor (hg38)
# =============================================================================
# Produces, from one coordinate-sorted BAM:
#   1. <sample>_break_sites.tsv.gz
#      One row per retained fragment, preserving genomic cut coordinates,
#      strand, 5' motif, pre-break context, fragment length, and cleavage context.
#   2. <sample>_4mer_corrected.tsv
#      Global 5' 4-mer counts and fractions (256 motifs at most).
#   3. <sample>_4mer_extended_final.tsv.gz
#      Aggregated motif x pre-break x fragment-length x cleavage-context counts.
#   4. <sample>_break_extraction_qc.tsv
#      Input/retained counts and run parameters.
#
# One properly paired R1 is used per fragment. Reverse-strand R1 motifs are
# reverse-complemented into biological 5'-end orientation. Genomic pre-break
# sequence is fetched strand-aware from the reference.
#
# Filters:
#   MAPQ >= 60; properly paired R1; no unmapped, secondary, QC-failed,
#   duplicate, or supplementary alignments; no soft/hard clipping;
#   absolute TLEN between configurable limits (defaults 80-500 bp).
#
# Usage:
#   bash extract_cfdna_breaks_locked.sh sample.bam [reference.fa] [output_dir]
#
# Example:
#   bash extract_cfdna_breaks_locked.sh AL060.bam \
#     /scratch/local/resources/hg38.fa motif_locked
#
# GNU parallel:
#   find /path/to/bams -name '*.bam' -print0 | \
#     parallel -0 -j 10 bash extract_cfdna_breaks_locked.sh {} \
#       /scratch/local/resources/hg38.fa motif_locked
# =============================================================================

set -euo pipefail

if [[ $# -lt 1 || $# -gt 3 ]]; then
    echo "Usage: $0 sample.bam [reference.fa] [output_dir]" >&2
    exit 1
fi

BAM=$1
REF=${2:-/scratch/local/resources/hg38.fa}
OUT_DIR=${3:-.}
MIN_MAPQ=${MIN_MAPQ:-60}
MIN_FRAG_LEN=${MIN_FRAG_LEN:-80}
MAX_FRAG_LEN=${MAX_FRAG_LEN:-500}
SAMTOOLS_THREADS=${SAMTOOLS_THREADS:-1}
NAME=$(basename "${BAM}" .bam)

for tool in samtools bedtools awk sort gzip; do
    if ! command -v "${tool}" >/dev/null 2>&1; then
        echo "ERROR: required program not found: ${tool}" >&2
        exit 1
    fi
done
if [[ ! -f "${BAM}" ]]; then
    echo "ERROR: BAM not found: ${BAM}" >&2
    exit 1
fi
if [[ ! -f "${REF}" ]]; then
    echo "ERROR: reference FASTA not found: ${REF}" >&2
    exit 1
fi
if [[ ! -f "${REF}.fai" ]]; then
    echo "ERROR: FASTA index not found: ${REF}.fai" >&2
    echo "Run: samtools faidx ${REF}" >&2
    exit 1
fi
if ! [[ "${MIN_FRAG_LEN}" =~ ^[0-9]+$ && "${MAX_FRAG_LEN}" =~ ^[0-9]+$ ]] ||
   (( MIN_FRAG_LEN < 1 || MAX_FRAG_LEN < MIN_FRAG_LEN )); then
    echo "ERROR: invalid fragment-length bounds: ${MIN_FRAG_LEN}-${MAX_FRAG_LEN}" >&2
    exit 1
fi

mkdir -p "${OUT_DIR}"
TMP_DIR=$(mktemp -d "${TMPDIR:-/tmp}/${NAME}.breaks.XXXXXX")
trap 'rm -rf "${TMP_DIR}"' EXIT
BED="${TMP_DIR}/prebreak.bed"
BEDSEQ="${TMP_DIR}/prebreak_with_sequence.tsv"
BREAKS="${TMP_DIR}/break_sites.tsv"

OUT_BREAKS="${OUT_DIR}/${NAME}_break_sites.tsv.gz"
OUT_GLOBAL="${OUT_DIR}/${NAME}_4mer_corrected.tsv"
OUT_EXTENDED="${OUT_DIR}/${NAME}_4mer_extended_final.tsv.gz"
OUT_QC="${OUT_DIR}/${NAME}_break_extraction_qc.tsv"

echo "[${NAME}] extracting filtered R1 fragment ends..." >&2

# Required flags: proper pair (0x2) + R1 (0x40) = 66.
# Excluded flags: unmapped (0x4), secondary (0x100), QC fail (0x200),
# duplicate (0x400), supplementary (0x800) = 3844.
samtools view -@ "${SAMTOOLS_THREADS}" -q "${MIN_MAPQ}" -f 66 -F 3844 "${BAM}" |
awk -v min_len="${MIN_FRAG_LEN}" -v max_len="${MAX_FRAG_LEN}" '
BEGIN { OFS = "\t" }

function ref_len(c,    i, n, len, num, ch) {
    len = 0
    num = ""
    n = length(c)
    for (i = 1; i <= n; i++) {
        ch = substr(c, i, 1)
        if (ch ~ /[0-9]/) {
            num = num ch
        } else {
            if (ch ~ /[MDN=X]/) len += int(num)
            num = ""
        }
    }
    return len
}

{
    flag = $2
    chrom = $3
    pos = $4
    mapq = $5
    cigar = $6
    tlen = $9
    seq = toupper($10)

    if (chrom == "*" || cigar == "*" || seq == "*") next
    # Clipped reads do not contain a trustworthy complete biological terminus.
    if (cigar ~ /[SH]/) next

    frag_len = (tlen < 0) ? -tlen : tlen
    if (frag_len < min_len || frag_len > max_len) next

    is_rev = int(flag / 16) % 2
    if (!is_rev) {
        # Forward R1 five-prime base is at BED [pos-1,pos); fetch upstream bases.
        cut0 = pos - 1
        if (cut0 < 4) next
        print chrom, cut0 - 4, cut0, seq "|" frag_len "|" mapq, 0, "+"
    } else {
        # Reverse R1 biological five-prime base is the rightmost aligned base.
        # cut0 is the 0-based boundary immediately after that base.
        cut0 = pos + ref_len(cigar) - 1
        if (cut0 < 1) next
        print chrom, cut0, cut0 + 4, seq "|" frag_len "|" mapq, 0, "-"
    }
}' > "${BED}"

N_FILTERED=$(wc -l < "${BED}" | awk '{print $1}')
if (( N_FILTERED == 0 )); then
    echo "ERROR: no fragments passed filters for ${NAME}" >&2
    exit 1
fi

echo "[${NAME}] fetching strand-aware pre-break sequence for ${N_FILTERED} fragments..." >&2
bedtools getfasta -fi "${REF}" -bed "${BED}" -s -bedOut > "${BEDSEQ}"

echo "[${NAME}] orienting motifs and writing coordinate-level break sites..." >&2
awk -v sample="${NAME}" '
BEGIN {
    FS = OFS = "\t"
    print "chrom", "cut_start", "cut_end", "sample", "frag_len", "strand", \
          "mapq", "motif_5p", "pre_break", "cleavage_ctx"
}
function rc(s,    r, i, b, c) {
    r = ""
    for (i = length(s); i > 0; i--) {
        b = substr(s, i, 1)
        if      (b == "A") c = "T"
        else if (b == "T") c = "A"
        else if (b == "C") c = "G"
        else if (b == "G") c = "C"
        else               c = "N"
        r = r c
    }
    return r
}
{
    chrom = $1
    pre_start = $2
    pre_end = $3
    split($4, meta, "|")
    read_seq = toupper(meta[1])
    frag_len = meta[2]
    mapq = meta[3]
    strand = $6
    pre_break = toupper($7)

    if (strand == "+") {
        motif = substr(read_seq, 1, 4)
        cut_start = pre_end
        cut_end = pre_end + 1
    } else {
        motif = rc(substr(read_seq, length(read_seq) - 3, 4))
        cut_start = pre_start - 1
        cut_end = pre_start
    }
    if (length(motif) != 4 || motif ~ /N/) next
    if (length(pre_break) != 4 || pre_break ~ /N/) next
    cleavage = substr(pre_break, 4, 1) substr(motif, 1, 2)
    if (length(cleavage) != 3 || cleavage ~ /N/) next

    print chrom, cut_start, cut_end, sample, frag_len, strand, mapq, motif, \
          pre_break, cleavage
}' "${BEDSEQ}" > "${BREAKS}"

N_RETAINED=$(awk 'END { print (NR > 0 ? NR - 1 : 0) }' "${BREAKS}")
if (( N_RETAINED == 0 )); then
    echo "ERROR: no fragments remained after sequence-context QC for ${NAME}" >&2
    exit 1
fi

gzip -c "${BREAKS}" > "${OUT_BREAKS}"

echo "[${NAME}] aggregating global and extended motif summaries..." >&2
awk '
BEGIN { FS = OFS = "\t" }
NR == 1 { next }
{ counts[$8]++; total++; sample = $4 }
END {
    for (m in counts) print counts[m] / total, counts[m], m, sample
}' "${BREAKS}" | sort -t $'\t' -k1,1nr > "${TMP_DIR}/global.sorted.tsv"

# Restore the header after numeric sorting.
{
    printf 'fraction\tcount\tmotif_5p\tsample\n'
    cat "${TMP_DIR}/global.sorted.tsv"
} > "${OUT_GLOBAL}"

awk '
BEGIN { FS = OFS = "\t" }
NR == 1 { next }
{
    key = $8 OFS $9 OFS $5 OFS $10
    counts[key]++
    total++
    sample = $4
}
END {
    print "fraction", "count", "motif_5p", "pre_break", "frag_len", \
          "cleavage_ctx", "sample"
    for (k in counts) print counts[k] / total, counts[k], k, sample
}' "${BREAKS}" | gzip -c > "${OUT_EXTENDED}"

# Avoid a second full BAM scan merely to count input alignments. The coordinate
# extractor is intentionally single-pass; filtered and retained counts are the
# relevant reproducibility metrics.
N_INPUT="NA"
{
    printf 'sample\tbam\treference\tmin_mapq\tmin_frag_len\tmax_frag_len\tinput_alignments\tfiltered_fragments\tretained_fragments\n'
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "${NAME}" "${BAM}" "${REF}" "${MIN_MAPQ}" "${MIN_FRAG_LEN}" \
        "${MAX_FRAG_LEN}" "${N_INPUT}" "${N_FILTERED}" "${N_RETAINED}"
} > "${OUT_QC}"

echo "[${NAME}] done" >&2
echo "  ${OUT_BREAKS}" >&2
echo "  ${OUT_GLOBAL}" >&2
echo "  ${OUT_EXTENDED}" >&2
echo "  ${OUT_QC}" >&2
