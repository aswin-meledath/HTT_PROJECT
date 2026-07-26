import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings("ignore")

print("1. Loading Candidate Genes...")
degs_df = pd.read_csv("Candidate_35_DEGs.csv")
candidate_genes = degs_df['Gene'].tolist()

print("2. Preprocessing GSE33000 Matrix and Annotations...")
gse33000_matrix = "GSE33000_series_matrix.txt"
gse33000_anno = "label33000.txt"

# Load the raw expression data and the annotation file
df_new = pd.read_csv(gse33000_matrix, sep='\t', comment='!', index_col=0)
anno_new = pd.read_csv(gse33000_anno, sep='\t', on_bad_lines='warn')
anno_new.columns = anno_new.columns.str.strip()

# Map the 'ORF' column to 'Gene Symbol'
anno_new = anno_new[['ID', 'ORF']].rename(columns={'ORF': 'Gene Symbol'})
anno_new = anno_new.dropna(subset=['Gene Symbol']).set_index('ID')

# Merge, average duplicates, and log2 transform
merged_new = df_new.join(anno_new, how='inner')
gene_level_new = merged_new.groupby('Gene Symbol').mean()

if gene_level_new.max().max() > 50:
    gene_level_new = np.log2(gene_level_new + 1)

print("3. Extracting Labels Directly from Matrix...")
mapping_test = {}
with open(gse33000_matrix, 'r') as f:
    ids = []
    patient_profiles = {}
    
    for line in f:
        if line.startswith('!Sample_geo_accession'):
            ids = [x.strip('"\n') for x in line.split('\t')[1:]]
            for i in ids:
                patient_profiles[i] = ""
                
        # Capture EVERY characteristic line (ch1, ch2, etc.) and glue them together
        elif line.startswith('!Sample_characteristics_'):
            values = [x.strip('"\n').lower() for x in line.split('\t')[1:]]
            if ids:
                for i, val in zip(ids, values):
                    patient_profiles[i] += val + " | "

# Search the glued profiles for our keywords
for i, profile in patient_profiles.items():
    if 'huntington' in profile:
        mapping_test[i] = 1
    elif 'non-demented' in profile or 'control' in profile:
        mapping_test[i] = 0
    # Alzheimer's patients are automatically ignored

print(f"Successfully labeled {len(mapping_test)} pure HD/Control patients.")

print("4. Filtering Genes and Applying Independent Scaling...")
X_test_full = gene_level_new.T
valid_samples = [s for s in X_test_full.index if s in mapping_test]

# Reindex to strictly match our 35 candidate genes (fill missing with 0)
X_test = X_test_full.loc[valid_samples].reindex(columns=candidate_genes).fillna(0)
y_test = pd.Series({s: mapping_test[s] for s in valid_samples}, name='Label')

# INDEPENDENT Z-SCORING: This fixes the "Different Thermometer" error
scaler_test = StandardScaler()
X_test_scaled_array = scaler_test.fit_transform(X_test)
X_test_scaled = pd.DataFrame(X_test_scaled_array, index=X_test.index, columns=X_test.columns)

print("5. Saving Clean Validation Set...")
X_test_scaled.to_csv("GSE33000_X_scaled.csv")
y_test.to_csv("GSE33000_y.csv")
print("Done! Files saved as 'GSE33000_X_scaled.csv' and 'GSE33000_y.csv'.")