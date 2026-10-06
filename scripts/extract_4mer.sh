#!/bin/bash

# Check if a BAM file path is provided
if [ -z "$1" ]; then
  echo "Usage: $0 <path_to_bam_file>"
  exit 1
fi

BAM_FILE="$1"
BASE_NAME=$(basename "${BAM_FILE%.bam}") # Extract base name without .bam extension

echo "Processing file: ${BASE_NAME}"

# Extract the 5' end motif of each read 1, count unique occurrences, and filter out 'N's
#   -f 0x42 : paired, read 1, properly paired
#   -F 0xB0C: drop unmapped, mate unmapped, secondary, QC-fail and supplementary records
# BAM stores reverse-strand reads reverse-complemented, so for those reads the
# 5' end motif is the reverse complement of the last 4 bases of SEQ, not the first 4.
# Reads soft-clipped at the 5' end are skipped, since their first bases are not at the cut site.
samtools view -q 60 -f 0x42 -F 0xB0C "${BAM_FILE}" | \
awk -F'\t' '
function rc(s,    r, i) {
  r = ""
  for (i = length(s); i > 0; i--) r = r comp[substr(s, i, 1)]
  return r
}
BEGIN { comp["A"]="T"; comp["C"]="G"; comp["G"]="C"; comp["T"]="A"; comp["N"]="N" }
{
  if (int($2 / 16) % 2) {                 # reverse strand
    if ($6 ~ /S$/) next
    print rc(toupper(substr($10, length($10) - 3, 4)))
  } else {
    if ($6 ~ /^[0-9]+S/) next
    print toupper(substr($10, 1, 4))
  }
}' | \
sort | uniq -c | \
awk '{if(length($2)==4 && $2 !~ /[^ACGT]/) print $1,$2}' > "${BASE_NAME}.temp"

# Calculate the total count of all 4-mers
TOTAL_4MERS=$(awk '{s+=$1} END {print s}' "${BASE_NAME}.temp")

# Calculate frequencies and append original counts, 4-mer, and base name
awk -v total="${TOTAL_4MERS}" -v name="${BASE_NAME}" \
'{print $1/total, $0, name}' "${BASE_NAME}.temp" > "${BASE_NAME}_4mer.txt"

# Clean up the temporary file
rm "${BASE_NAME}.temp"

echo "Processing complete. Output saved to ${BASE_NAME}_4mer.txt"
