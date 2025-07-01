# Medulloblastoma cfDNA Fragmentomics Classifier

This repository contains code for training and evaluating a machine learning meta-classifier to noninvasively classify medulloblastoma subtypes using cfDNA fragmentomic features derived from cerebrospinal fluid (CSF) low-pass whole genome sequencing (LPWGS) data.

## Background

Medulloblastoma comprises four molecular subgroups (WNT, SHH, Group 3, Group 4), each with distinct prognostic implications. Current clinical classification requires tumor tissue, limiting accessibility in some settings. This project explores a cfDNA-based approach using genome-wide features:

- Copy number aberrations (CNA)
- Fragment length profiles
- 4-bp fragment end-motifs
- NMF-derived fragmentation profiles (F-profiles)

## Classifier Overview

We implement a multi-stage classification framework:

- **Binary One-vs-One and One-vs-Rest classifiers** trained using:
  - Logistic Regression
  - Linear SVM
  - Random Forest
  - Gradient Boosting
- **Nested Cross-Validation** for model selection and hyperparameter tuning
- **Ensemble Meta-Classifier** (Logistic Regression) trained on scores from all binary classifiers
- **ROC analysis** across 3-fold stratified outer CV

## Key Results

- Achieved **mean AUC of 0.94** across folds
- Recapitulated known subgroup-specific CNA features (e.g., monosomy 6, i(17q))
- Identified 4 significant F-profiles and 29 discriminatory 4-mer motifs

## Acknowledgments

Original fragmentomics code adapted from [pughlab/fragmentomics.](https://github.com/pughlab/fragmentomics)

This project supports our manuscript:
"Genome-wide cfDNA fragmentation patterns in cerebrospinal fluid reflect medulloblastoma groups"

