import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.stats.multitest import multipletests

# 1. EXTRACT TRUE METADATA MAPPING
# Dynamically parsing the raw files to find the exact HD and Control GSM IDs
mapping = {}
batches = {'GPL96': {'ctrl': [], 'hd': []}, 
           'GPL97': {'ctrl': [], 'hd': []}, 
           'GSE26927': {'ctrl': [], 'hd': []}}

# Parse GPL96 and GPL97
for file, batch in [('GSE3790-GPL96_series_matrix.txt', 'GPL96'), 
                    ('GSE3790-GPL97_series_matrix.txt', 'GPL97')]:
    with open(file, 'r') as f:
        ids, chars = [], []
        for line in f:
            if line.startswith('!Sample_geo_accession'):
                ids = [x.strip('"\n') for x in line.split('\t')[1:]]
            if line.startswith('!Sample_characteristics_ch1'):
                chars = [x.strip('"\n').lower() for x in line.split('\t')[1:]]
        for i, ch in zip(ids, chars):
            if 'control' in ch:
                mapping[i] = 'Control'
                batches[batch]['ctrl'].append(i)
            elif 'hd grade' in ch or 'hd ' in ch or 'huntington' in ch:
                mapping[i] = 'HD'
                batches[batch]['hd'].append(i)

# Parse GSE26927 (Filtering out Alzheimer's/Parkinson's)
with open('GSE26927_series_matrix.txt', 'r') as f:
    ids, titles, diseases = [], [], []
    for line in f:
        if line.startswith('!Sample_geo_accession'):
            ids = [x.strip('"\n') for x in line.split('\t')[1:]]
        if line.startswith('!Sample_title'):
            titles = [x.strip('"\n').lower() for x in line.split('\t')[1:]]
        if line.startswith('!Sample_characteristics_ch1') and 'disease' in line.lower():
            diseases = [x.strip('"\n').lower() for x in line.split('\t')[1:]]
    if ids and titles and diseases:
        for i, t, d in zip(ids, titles, diseases):
            if 'huntington' in d: # Strict filter for HD only
                if 'control' in t:
                    mapping[i] = 'Control'
                    batches['GSE26927']['ctrl'].append(i)
                elif 'disease' in t or 'hd' in t:
                    mapping[i] = 'HD'
                    batches['GSE26927']['hd'].append(i)

# 2. LOAD AND CLEAN HARMONIZED DATA
df = pd.read_csv("HD_Harmonized_Expression.csv", index_col=0)
valid_cols = [c for c in df.columns if c in mapping]
df = df[valid_cols] # Drops the 98 non-HD contaminating samples

print(f"Total pure HD/Control samples: {df.shape[1]}")

# 3. FISHER'S META-ANALYSIS
meta_results = []
for gene in df.index:
    p_values, log2fcs = [], []
    
    for batch, groups in batches.items():
        ctrl_cols = [c for c in groups['ctrl'] if c in df.columns]
        hd_cols = [c for c in groups['hd'] if c in df.columns]
        
        if not ctrl_cols or not hd_cols: continue
            
        ctrl_vals = df.loc[gene, ctrl_cols].astype(float)
        hd_vals = df.loc[gene, hd_cols].astype(float)
        
        # Batch Log2FC and T-Test
        log2fcs.append(hd_vals.mean() - ctrl_vals.mean())
        _, p = stats.ttest_ind(hd_vals, ctrl_vals, nan_policy='omit')
        p_values.append(max(p, 1e-15) if not np.isnan(p) else 1.0)
    
    if p_values:
        # Fisher's combination method to aggregate significance
        chi_sq = -2 * np.sum(np.log(p_values))
        combined_p = stats.chi2.sf(chi_sq, 2 * len(p_values))
        
        meta_results.append({
            'Gene': gene,
            'mean_log2FC': np.mean(log2fcs),
            'combined_p': combined_p
        })

meta_df = pd.DataFrame(meta_results).set_index('Gene')
meta_df['adj_p'] = multipletests(meta_df['combined_p'].fillna(1), method='fdr_bh')[1]

# 4. PAPER THRESHOLDS
lenient = meta_df[(meta_df['adj_p'] < 0.05) & (meta_df['mean_log2FC'].abs() >= 0.85)]
stringent = meta_df[(meta_df['adj_p'] < 0.05) & (meta_df['mean_log2FC'].abs() > 1.0)]

print(f"Lenient DEGs (Paper Target 41): {len(lenient)}")
print(f"Stringent DEGs (Paper Target 13): {len(stringent)}")

# 5. SAVE FOR ML
lenient.to_csv("Candidate_35_DEGs.csv")