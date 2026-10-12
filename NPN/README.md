# Breast Cancer Prognosis & Subtype Classification
A machine-learning research project using the METABRIC breast cancer
cohort to explore consensus-based gene selection, molecular subtype
classification, and survival analysis.

> **Research disclaimer:** This is an exploratory computational study,
> not a clinically validated diagnostic or prognostic tool. The reported
> results have important limitations, including test-set information
> leakage during feature ranking and in-sample survival evaluation.

## Overview

Breast cancer is biologically heterogeneous, and molecular subtypes can
have different characteristics and outcomes. This project investigates
whether combining multiple feature-selection techniques can identify a
compact set of candidate genes associated with molecular subtype
classification, and explores whether adding those genes to clinical
covariates changes survival-model discrimination.

The analysis is based on a notebook workflow using the METABRIC-derived
dataset `METABRIC RNA Mutation.csv`.

### Research objectives

-   Rank 490 candidate gene-related features using multiple
    feature-importance and feature-selection methods.
-   Identify a compact five-gene candidate panel through consensus
    ranking.
-   Evaluate seven-class PAM50 + claudin-low subtype classification.
-   Compare models trained on the full feature set with models trained
    on the five-gene panel.
-   Explore survival patterns using Kaplan--Meier curves and Cox
    proportional-hazards models.
-   Document limitations and methodological improvements needed for more
    reliable evaluation.

## Dataset

The report describes the following cohort:

  Item                                                    Reported value
  ----------------------------------------------------- ----------------
  Patient records                                                  1,904
  Original columns                                                   693
  Candidate gene-related features                                    490
  Clinical variables considered for survival analysis                 21
  Training records                                           1,523 (80%)
  Test records                                                 381 (20%)
  Molecular subtype classes                                            7
  Records retained for survival analysis                           1,903

The target is **`pam50 + claudin-low`**, with the observed classes
`Basal`, `Her2`, `LumA`, `LumB`, `Normal`, `NC`, and `claudin-low`.

The subtype distribution is imbalanced. In particular, the `NC` class
contains only six records, so its reported test-set performance is not a
reliable estimate of generalization to that class.

**Dataset note:** The report refers to the input file as
`METABRIC RNA Mutation.csv`. Confirm the dataset's redistribution terms
and include the data file only if you are permitted to share it.
Otherwise, document how authorized users can obtain it and keep it out
of the repository.

## Methodology

### 1. Preprocessing and split

-   Stratified 80:20 train/test split with random state `42`.
-   Median imputation fitted on the training data and applied to both
    partitions.
-   Standard scaling for methods that require standardized features.
-   Tree-based models trained on imputed, unscaled features.

### 2. Consensus-based feature selection

Ten complementary methods are combined:

1.  Random Forest feature importance
2.  XGBoost feature importance
3.  XGBoost gain
4.  Permutation importance
5.  SHAP importance
6.  LASSO coefficient importance
7.  Elastic Net coefficient importance
8.  Mutual information
9.  ANOVA F-statistic
10. Recursive Feature Elimination (RFE)

Scores are min--max normalized. The consensus score is the arithmetic
mean of normalized scores, while selection frequency counts how often a
feature appears among each method's top 50. The final ranking
prioritizes selection frequency, then consensus score.

### 3. Subtype classification

The report evaluates a seven-class XGBoost classifier on all 490
candidate features. It also presents a separate comparison of RBF-SVM,
LightGBM, and Gaussian Naive Bayes using both the full feature set and
the selected five-gene panel.

Reported metrics include accuracy, precision, recall, and macro-F1.
Macro-F1 is important here because it gives each class equal weight
rather than allowing the most common subtype to dominate the summary.

### 4. Exploratory survival analysis

-   Kaplan--Meier curves compare high- and low-expression groups for
    each selected gene, split at the cohort median.
-   Cox proportional-hazards models compare clinical covariates alone
    against clinical covariates plus the five selected genes.
-   The concordance index (C-index) is used to summarize risk-ranking
    discrimination.

The report describes these survival results as exploratory and notes
that the Cox concordance values are apparent, in-sample estimates.

## Selected candidate genes

The five genes reported by the consensus procedure are:

-   **EGFR**
-   **CDK1**
-   **GATA3**
-   **ERBB2**
-   **ERBB3**

These are candidate features selected by this workflow. Their selection
does **not** establish biological causality, independent prognostic
value, or clinical utility.

## Reported results

### Full-feature XGBoost classification

  Metric           Reported result
  -------------- -----------------
  Accuracy                  0.8346
  Macro-F1                  0.6780
  Test records                 381
  Features                     490

### Separate model comparison

  Feature set         Model                    Accuracy   Macro-F1
  ------------------- ---------------------- ---------- ----------
  Full 490 features   RBF-SVM                    0.7900     0.6756
  Full 490 features   LightGBM                   0.8189     0.6755
  Full 490 features   Gaussian Naive Bayes       0.7113     0.5816
  Five genes          RBF-SVM                    0.6247     0.5429
  Five genes          LightGBM                   0.6798     0.5496
  Five genes          Gaussian Naive Bayes       0.6955     0.5505

The five-gene panel reduces the number of features from 490 to five
(**98.98% fewer features**), but its best reported macro-F1 is lower
than the best full-feature result. The compact panel therefore does not
preserve the full feature set's predictive performance in this
experiment.

### Exploratory Cox concordance

  Model                                Reported C-index
  ---------------------------------- ------------------
  Clinical covariates only                       0.6834
  Clinical covariates + five genes               0.6855
  Difference                                    +0.0021

The difference is small, was calculated in-sample, and does not by
itself demonstrate a statistically significant or clinically meaningful
improvement. The report also states that none of the five gene
coefficients in the augmented Cox model had a p-value below 0.05.

## Key limitations

1.  **Feature-selection leakage:** Test-set permutation-importance and
    SHAP results contribute to feature ranking. If the same test set is
    then used to report performance, the evaluation is no longer an
    unbiased assessment of a fixed pipeline.
2.  **Rare class:** `NC` has only six records overall and one test
    observation. Its class-specific score is highly unstable.
3.  **Exploratory survival estimates:** The reported Cox C-index values
    are computed on the data used to fit the models, which can make
    performance appear optimistic.
4.  **No independent external validation:** The results should not be
    treated as evidence of clinical readiness.
5.  **Compact panel trade-off:** The five-gene models have lower
    macro-F1 than the best reported full-feature model.
6.  **Descriptive Kaplan--Meier plots:** Visual separation alone does
    not establish statistical significance; the report does not provide
    log-rank tests or multiplicity-adjusted survival comparisons.

## Recommended next steps

-   Move all feature selection, normalization, and hyperparameter tuning
    inside training folds using nested cross-validation.
-   Keep a genuinely untouched test set for final evaluation.
-   Recompute the consensus panel without using test-set information.
-   Evaluate a five-gene-only XGBoost model as a separate, explicitly
    reported experiment.
-   Report per-class precision, recall, F1, confusion matrices, and
    suitable uncertainty estimates.
-   Investigate the rare `NC` class and clearly report the limitations
    caused by its small sample size.
-   Evaluate survival models with out-of-sample predictions, confidence
    intervals, proportional-hazards diagnostics, and appropriate
    validation.
-   Seek external-cohort validation before making biological,
    prognostic, or clinical claims.

## Repository structure

The report describes a **notebook-based analysis**. Adjust this example
to match the files you actually publish; do not add filenames that are
not present in your repository.

``` text
.
├── README.md
├── Consensus_Based_Gene_Selection_for_Molecular_Subtype_Classification_and_Exploratory_Survival_Analysis.pdf
├── <analysis-notebook>.ipynb
├── METABRIC RNA Mutation.csv   # only if redistribution is permitted
└── <optional-output-figures-and-tables>/
```

## Reproducibility

The report identifies the notebook and input dataset but does not
provide a complete, verified dependency lockfile or exact installation
commands. To make the repository reproducible:

1.  Publish the analysis notebook and any scripts required to run it.
2.  Add a `requirements.txt` or environment file containing the versions
    actually tested.
3.  Document how to obtain the dataset legally, along with its expected
    filename and required columns.
4.  Record the Python version, random seeds, preprocessing steps, and
    model parameters.
5.  Store generated figures and result tables in clearly named output
    directories, if you intend to publish them.

Do not assume that a fresh environment will reproduce the reported
numbers until the complete workflow has been rerun.

## Streamlit Dashboard
https://vr7ta6xgaqrkwhznaxsmdn.streamlit.app

## How to cite this project

If you use or build on this analysis, cite the project report:

*"Consensus-Based Gene Selection for Molecular Subtype Classification
and Exploratory Survival Analysis in the METABRIC Breast Cancer
Cohort."* Report dated October 10, 2026.



## Disclaimer

This repository is for educational and research purposes only. It is not
intended for diagnosis, treatment selection, or individual patient
prognosis. The selected genes and reported model results require
leakage-free evaluation and independent validation before any clinical
interpretation.
