#!/bin/bash
# Generate a non-overlapping autosomal-bin BED file from a FASTA .fai index.
#
# Usage:
#   ./make_bins_bed.sh REF.fa.fai [BIN_SIZE] [OUT_BED]
#
# Defaults: BIN_SIZE=500000, OUT_BED=bins_500kb_autosomes.bed
#
# Output: 4-column BED (chrom, start, end, name) restricted to autosomes
# (chr1-22 with or without 'chr' prefix). Sex / mito / alt / decoy contigs excluded.

set -euo pipefail

FAI="${1:?Usage: $0 REF.fa.fai [BIN_SIZE] [OUT_BED]}"
BIN_SIZE="${2:-500000}"
OUT="${3:-bins_$((BIN_SIZE/1000))kb_autosomes.bed}"

if [[ ! -f "$FAI" ]]; then
    echo "ERROR: FAI not found: $FAI" >&2
    exit 1
fi

awk -v size="$BIN_SIZE" -v OFS='\t' '
    # Strip optional "chr" prefix; keep only purely numeric (1-22) autosomes.
    {
        c = $1
        if (c ~ /^chr/) cn = substr(c, 4); else cn = c
        if (cn !~ /^[0-9]+$/) next
        if (cn+0 < 1 || cn+0 > 22) next
        n = $2
        for (s = 0; s < n; s += size) {
            e = s + size
            if (e > n) e = n
            printf "%s\t%d\t%d\t%s_%d\n", $1, s, e, $1, s
        }
    }
' "$FAI" > "$OUT"

n=$(wc -l < "$OUT")
echo "Wrote $OUT ($n bins, ${BIN_SIZE}bp)"
