#!/bin/bash

# Check if a BAM file path is provided
if [ -z "$1" ]; then
  echo "Usage: $0 <path_to_bam_file>"
  exit 1
fi

BAM_FILE="$1"
BASE_NAME=$(basename "${BAM_FILE%.bam}") # Extract base name without .bam extension

echo "Processing file: ${BASE_NAME}"

# Extract 4-mers from BAM, count unique occurrences, and filter out 'N's
samtools view -q 60 -f 0x1 -f 0x40 "${BAM_FILE}" | \
awk -F'\t' '{print substr($10,1,4)}' | \
sort | uniq -c | \
awk '{if($2!~/N/) print $1,$2}' > "${BASE_NAME}.temp"

# Calculate the total count of all 4-mers
TOTAL_4MERS=$(awk '{s+=$1} END {print s}' "${BASE_NAME}.temp")

# Calculate frequencies and append original counts, 4-mer, and base name
awk -v total="${TOTAL_4MERS}" -v name="${BASE_NAME}" \
'{print $1/total, $0, name}' "${BASE_NAME}.temp" > "${BASE_NAME}_4mer.txt"

# Clean up the temporary file
rm "${BASE_NAME}.temp"

echo "Processing complete. Output saved to ${BASE_NAME}_4mer.txt"
