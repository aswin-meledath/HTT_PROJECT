# A Multi-Model Learning Framework for Cross-Platform Biomarker Identification and Clinical Risk Stratification in Huntington's Disease

## Overview

This project presents a complete machine learning framework for the automated detection and risk stratification of Huntington's Disease (HD) using transcriptomic (gene expression) data.

The framework integrates multiple publicly available Gene Expression Omnibus (GEO) datasets, performs cross-platform harmonization using ComBat batch correction, identifies significant biomarkers through meta-analysis, benchmarks multiple machine learning algorithms, and introduces a novel Huntington's Disease Risk Score (HDRS) for patient risk stratification.

The final framework is externally validated on an independent dataset to evaluate its generalization capability.

---

## Key Features

- Cross-platform transcriptomic data integration
- ComBat batch effect correction
- Fisher's Combined Probability meta-analysis
- 35-gene diagnostic biomarker signature
- Benchmarking of 11 Machine Learning models
- Stacked Ensemble classifier
- SHAP-based explainable AI
- External validation using an independent GEO dataset
- Novel Huntington's Disease Risk Score (HDRS)
- Three-level patient risk stratification (Low / Medium / High)

---

## Datasets

Training Datasets

- GSE3790 (GPL96)
- GSE3790 (GPL97)
- GSE26927

External Validation Dataset

- GSE33000

Source:
NCBI Gene Expression Omnibus (GEO)

---

## Machine Learning Pipeline

Raw GEO Datasets
        │
        ▼
Preprocessing
        │
        ▼
Probe → Gene Mapping
        │
        ▼
Log2 Transformation
        │
        ▼
ComBat Batch Effect Correction
        │
        ▼
Meta-analysis (Fisher's Method)
        │
        ▼
35 Gene Signature Selection
        │
        ▼
Feature Matrix
        │
        ▼
Train 11 ML Models
        │
        ▼
Performance Comparison
        │
        ▼
SHAP Explainability
        │
        ▼
External Validation (GSE33000)
        │
        ▼
HDRS Risk Score
        │
        ▼
Low / Medium / High Risk Stratification

---

## Machine Learning Models Evaluated

- Logistic Regression
- Elastic Net
- Linear Discriminant Analysis
- K-Nearest Neighbors
- Support Vector Machine (RBF)
- Random Forest
- Extra Trees
- CatBoost
- Artificial Neural Network
- TabNet
- Stacked Ensemble

---

## Results

### Best Internal Performance

Model:
Stacked Ensemble

AUC:
0.867

Accuracy:
80%

---

### External Validation

Dataset:
GSE33000

Best AUC:
0.882

The framework demonstrates strong generalization across an independent dataset.

---

## Explainable AI

Model predictions are interpreted using SHAP (SHapley Additive Explanations).

The project provides

- Global feature importance
- Biomarker ranking
- Model interpretation
- Biological validation of important genes

---

## Huntington's Disease Risk Score (HDRS)

This project introduces a weighted risk score using

- Gene expression
- SHAP importance
- Fold change direction

Patients are stratified into

- Low Risk
- Medium Risk
- High Risk

using K-Means clustering.

---

## Project Structure

```
HTT_PROJECT/

│
├── Preprocessing/
├── Training/
├── risk_score/
├── shap/
├── results/
│
├── README.md
├── requirements.txt
└── .gitignore
```

---

## Installation

Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/HTT_PROJECT.git
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

## Output

The framework generates

- Harmonized expression matrix
- Biomarker rankings
- Model leaderboard
- ROC Curves
- SHAP plots
- Confusion matrices
- Risk score distributions
- Heatmaps
- Patient risk stratification

---

## Future Improvements

- Deep Learning based transcriptomic models
- RNA-Seq integration
- Single-cell RNA sequencing support
- Web application for prediction
- Clinical deployment pipeline

---

## Disclaimer

This project is intended for research and educational purposes only.

It is **not** a clinical diagnostic tool and should not be used for medical decision making.

---

## Authors

Aswin M

Department of Computer Science and Engineering

College of Engineering Trivandrum

---

## Acknowledgements

We acknowledge the publicly available datasets provided by the National Center for Biotechnology Information (NCBI) Gene Expression Omnibus (GEO), which made this work possible.
