# This script processes 4-mer mutation data, filters samples based on a fraction threshold, 
# and then performs Non-negative Matrix Factorization (NMF) to extract mutational signatures 
# using the MutationalPatterns library.

# Load necessary libraries
library(dplyr)
library(stringr)
library(tidyverse) # Includes tibble, ggplot2, tidyr, readr, purrr, forcats
library(NMF) # For Non-negative Matrix Factorization
library(MutationalPatterns) # For mutational signature analysis

# Optional: Disable scientific notation for better readability of large numbers
options(scipen = 999)

# --- Data Loading and Preprocessing ---

# Initialize a list to store processed 4-mer data from all files
all_4mer_data <- list()

# Define the path to your 4-mer data files
data_path <- "data/4mer/"

# Get a list of all 4-mer files
four_mer_files <- list.files(path = data_path, pattern = "*4mer.txt$", full.names = TRUE, recursive = TRUE)

# Loop through each 4-mer file, read, process, and store the data
for (file_path in four_mer_files) {
  # Extract sample name from the filename
  sample_name <- gsub("_4mer.txt", "", basename(file_path))
  message("Processing sample: ", sample_name) # Use message for user feedback

  # Read the 4-mer data, select relevant columns, and convert V1 to numeric
  current_seg <- read.table(file_path, header = FALSE, sep = " ") %>%
    dplyr::select(-V2) %>% # V2 seems redundant based on your original code
    dplyr::mutate(V1 = as.numeric(as.character(V1))) # Ensure V1 (counts) is numeric

  # Reshape the data from long to wide format (motif by mutation type)
  # The value.var is V1 (counts), V3 is motif, V4 is mutation type
  reshaped_seg <- reshape2::dcast(current_seg, V3 ~ V4, value.var = "V1")

  # Store the reshaped data in the list, naming it by sample_name
  all_4mer_data[[sample_name]] <- reshaped_seg
}

# Combine all reshaped 4-mer data frames into a single data frame
# This assumes that all files have the same V3 (motif) values for joining
# We'll use the first data frame as the base and then left_join others
if (length(all_4mer_data) > 0) {
  # Take the first data frame as the base and rename its V3 column for merging
  seg_all <- all_4mer_data[[1]] %>%
    dplyr::rename(V3 = V3) # Ensure V3 is named consistently for joins

  # Iterate from the second data frame onwards and left_join them
  if (length(all_4mer_data) > 1) {
    for (j in 2:length(all_4mer_data)) {
      seg_all <- dplyr::left_join(seg_all, all_4mer_data[[j]], by = "V3")
    }
  }
} else {
  stop("No 4-mer data files found or processed.")
}

# Set row names to motifs (V3) and remove the V3 column
# The V3 column contains the motifs, which are suitable as row names
rownames(seg_all) <- seg_all$V3
seg_all <- seg_all %>% dplyr::select(-V3)

# Transpose the matrix so samples are rows and motifs are columns for initial processing
# This is often a good intermediate step before filtering samples
seg_all_transposed <- as.data.frame(t(seg_all))

# --- Sample Information Merging and Filtering ---

# Read sample information, selecting relevant columns
# Assuming 'NAME' is the sample identifier, 'FRAC' is a numeric fraction, and 'V12' is another relevant column
sample_info <- readr::read_tsv("data/sample_map.txt", show_col_types = FALSE) %>%
  dplyr::select(NAME, FRAC, V12)

# Prepare the transposed 4-mer data for joining with sample information
# Convert row names (sample IDs) into a column named 'sample'
seg_all_transposed$sample <- rownames(seg_all_transposed)

# Join the 4-mer data with sample information
seg_all_merged <- seg_all_transposed %>%
  dplyr::left_join(sample_info, by = c("sample" = "NAME"))

# Filter samples based on the 'FRAC' column
# Keep samples where 'FRAC' is greater than 0.08 and remove the 'FRAC' column afterwards
filtered_seg_data <- seg_all_merged %>%
  dplyr::filter(FRAC > 0.08) %>%
  dplyr::select(-FRAC) # Remove FRAC as it's used for filtering, not for NMF input directly

# --- Prepare data for NMF ---

# Transpose the filtered data back to "motif x sample" format for NMF
# Set column names to sample names from the 'sample' row
# Remove the 'sample' and 'V12' rows which are now at the top after transposing
nmf_input_matrix <- t(filtered_seg_data)
colnames(nmf_input_matrix) <- nmf_input_matrix["sample", ] # Set column names to sample IDs
nmf_input_matrix <- nmf_input_matrix[!rownames(nmf_input_matrix) %in% c("sample", "V12"), ]

# Convert the matrix to numeric type for NMF
# Ensure it's a true numeric matrix
nmf_input_matrix <- matrix(as.numeric(nmf_input_matrix),
  ncol = ncol(nmf_input_matrix),
  dimnames = list(rownames(nmf_input_matrix), colnames(nmf_input_matrix))
)

# Verify matrix class and dimensions
if (!is.numeric(nmf_input_matrix)) {
  stop("NMF input matrix is not numeric.")
}
message("Dimensions of NMF input matrix (motifs x samples): ", dim(nmf_input_matrix)[1], " x ", dim(nmf_input_matrix)[2])

# Check sum of frequencies per motif
motif_sums <- as.data.frame(apply(nmf_input_matrix, 1, sum)) %>%
  setNames("freq") %>% # Rename column to 'freq'
  arrange(freq)

message("Top 6 motifs by frequency:")
print(tail(motif_sums, 6))

# --- Mutational Signature Extraction (NMF) ---

# Extract mutational signatures using MutationalPatterns::extract_signatures
message("Starting NMF signature extraction...")
ms_nmf_res <- MutationalPatterns::extract_signatures(
  nmf_input_matrix,
  rank = 7,
  nrun = 200,
  single_core = TRUE
)
message("NMF signature extraction complete.")

# --- Visualize and Process Results ---

# Assign meaningful column names to the contribution matrix
# These should correspond to your sample names
contributions <- ms_nmf_res$contribution
colnames(contributions) <- colnames(nmf_input_matrix)

# Plot the relative contribution of signatures per sample
# 'mode = "relative"' shows the proportion of each signature within samples
plot_contribution(contributions, ms_nmf_res$signature, mode = "relative")

# Prepare signatures for visualization or further analysis
# Assign row names (motifs) to the signature matrix for clear identification
rownames(ms_nmf_res$signature) <- rownames(nmf_input_matrix)

# Convert signatures to a long format suitable for `ggplot2`
signatures_df <- as.data.frame(ms_nmf_res$signature) %>%
  tibble::rownames_to_column("motif") # Convert row names to a 'motif' column

# Reshape from wide to long format 
msig_long <- signatures_df %>%
  tidyr::pivot_longer(
    cols = -motif, # Pivot all columns except 'motif'
    names_to = "Signature", # New column for signature names
    values_to = "Contribution" # New column for signature weights/contributions
  )

# Create a new column for motif "end" (e.g., "A-end", "C-end")
# This is used for specific visualization styles in F-profile plots,
# grouping by the first base of the motif.
msig_long$`Motif-end` <- paste0(stringr::str_sub(msig_long$motif, 1, 1), "-end")

message("Script execution complete. Results are in 'ms_nmf_res' and 'msig_long'.")
