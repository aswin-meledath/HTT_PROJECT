import pandas as pd
import numpy as np
import os
from inmoose.pycombat import pycombat_norm

def load_and_harmonize(matrix_path, annotation_path):
    print(f"--- Processing {os.path.basename(matrix_path)} ---")
    
    # 1. Load Expression Matrix (Skip GEO metadata)
    df = pd.read_csv(matrix_path, sep='\t', comment='!', index_col=0)
    
    # 2. Load Annotation with automatic delimiter detection
    anno = pd.read_csv(annotation_path, sep=None, engine='python', on_bad_lines='warn')
    anno.columns = anno.columns.str.strip()
    
    # Identify the Gene Symbol column (handles 'Gene Symbol' or 'SYMBOL')
    # The paper requires this mapping for cross-platform meta-analysis.
    symbol_col = next((c for c in anno.columns if c.upper() in ['GENE SYMBOL', 'SYMBOL']), None)
    
    if 'ID' not in anno.columns or not symbol_col:
        print(f"Headers found in {annotation_path}: {list(anno.columns)}")
        raise ValueError(f"Required columns missing in {annotation_path}.")

    # Rename the found column to 'Gene Symbol' for consistency across all 3 datasets
    anno = anno[['ID', symbol_col]].rename(columns={symbol_col: 'Gene Symbol'})
    anno = anno.dropna(subset=['Gene Symbol']).set_index('ID')

    # 3. Merge Matrix with Annotations
    merged = df.join(anno, how='inner')
    
    # 4. Resolve Multiple Probes
    # Paper: 'In cases where multiple probes mapped to the same gene, the mean expression value was used.'
    gene_level_df = merged.groupby('Gene Symbol').mean()
    
    # 5. Log Transformation
    # Paper: 'Applied log transformation to stabilize the variance.'
    if gene_level_df.max().max() > 50:
        gene_level_df = np.log2(gene_level_df + 1)
        
    return gene_level_df

# --- FILE CONFIGURATION ---
dataset_configs = {
    "GSE3790_GPL96": ("GSE3790-GPL96_series_matrix.txt", "label3790-96.txt"),
    "GSE3790_GPL97": ("GSE3790-GPL97_series_matrix.txt", "label3790-97.txt"),
    "GSE26927": ("GSE26927_series_matrix.txt", "label26927.txt")
}

processed_datasets = {}
for name, (m_path, a_path) in dataset_configs.items():
    processed_datasets[name] = load_and_harmonize(m_path, a_path)

# 6. Meta-Analysis Alignment (Inner Join)
# Ensures uniformity by selecting only genes common to all three cohorts.
combined_df = pd.concat(processed_datasets.values(), axis=1, join='inner')

# 7. Cleaning for ComBat
combined_df = combined_df.dropna()
combined_df = combined_df[combined_df.std(axis=1) > 0.001]

# 8. Batch Effect Correction
# Paper: 'Batch effect correction was performed based on the Combat algorithm.'
batch_labels = []
for name in processed_datasets.keys():
    batch_labels.extend([name] * processed_datasets[name].shape[1])

print(f"Starting ComBat on {combined_df.shape[0]} common genes and {combined_df.shape[1]} samples...")
harmonized_data = pycombat_norm(combined_df, batch_labels)

# 9. Save Final Output
# Input for ML: 'We selected the batch effect-corrected combined GSE expression profile.'
harmonized_data.to_csv("HD_Harmonized_Expression.csv")
print("Process Complete. Harmonized data saved to 'HD_Harmonized_Expression.csv'.")