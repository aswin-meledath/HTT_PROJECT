# Dataset Information

## Overview

This project uses publicly available gene expression datasets from the **NCBI Gene Expression Omnibus (GEO)** for Huntington's Disease analysis.

To keep this repository lightweight and comply with GitHub file size limits, the raw and processed datasets are **not included**.

---

## Datasets Used

### Training Datasets

- GSE3790 (GPL96)
- GSE3790 (GPL97)
- GSE26927

### External Validation Dataset

- GSE33000

Source:

https://www.ncbi.nlm.nih.gov/geo/

---

## Folder Structure

```
dataset_and_matrixfiles/

├── GSE3790-GPL96_series_matrix.txt
├── GSE3790-GPL97_series_matrix.txt
├── GSE26927_series_matrix.txt
├── GSE33000_series_matrix.txt
│
├── label3790-96.txt
├── label3790-97.txt
├── label26927.txt
└── HD_Harmonized_Expression.csv
```

---

## How to Obtain the Data

1. Download the datasets from the NCBI GEO database.
2. Extract the series matrix files.
3. Place all raw datasets into this folder.
4. Run the preprocessing scripts to generate the harmonized expression matrix.

---

## Note

The generated files (such as `HD_Harmonized_Expression.csv`) are created automatically by the preprocessing pipeline and therefore are not included in this repository.

---

## License

The datasets belong to their original authors and are distributed through the NCBI Gene Expression Omnibus (GEO). Please follow the respective dataset licensing and citation requirements when using them.
