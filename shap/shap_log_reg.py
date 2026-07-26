import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import shap
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

# ==========================================
# 1. LOAD DATA & TRAIN MODEL
# ==========================================
print("1. Loading Data and Training Final Logistic Regression Model...")

# A. Load Candidate Genes
degs_df = pd.read_csv("Candidate_35_DEGs.csv")
candidate_genes = degs_df['Gene'].tolist()

# B. Load Original Harmonized Data
harmonized_df = pd.read_csv("HD_Harmonized_Expression.csv", index_col=0)
X_full = harmonized_df.loc[candidate_genes].T

# C. Extract Original Labels
mapping = {}
files = ['GSE3790-GPL96_series_matrix.txt', 'GSE3790-GPL97_series_matrix.txt', 'GSE26927_series_matrix.txt']
for file in files:
    try:
        with open(file, 'r') as f:
            ids, chars = [], []
            for line in f:
                if line.startswith('!Sample_geo_accession'):
                    ids = [x.strip('"\n') for x in line.split('\t')[1:]]
                if line.startswith('!Sample_characteristics_ch1'):
                    chars = [x.strip('"\n').lower() for x in line.split('\t')[1:]]
            for i, ch in zip(ids, chars):
                if 'control' in ch: mapping[i] = 0
                elif 'hd' in ch or 'huntington' in ch: mapping[i] = 1
    except FileNotFoundError:
        continue

valid_samples = [s for s in X_full.index if s in mapping]
X = X_full.loc[valid_samples]
y = np.array([mapping[s] for s in valid_samples])

# D. Scale and Train
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Matching your validation script parameters
lr_model = LogisticRegression(random_state=42, class_weight='balanced', max_iter=1000)
lr_model.fit(X_scaled, y)

# ==========================================
# 2. RUN SHAP EXPLAINER (LinearExplainer)
# ==========================================
print("2. Running SHAP Explainer (LinearExplainer)...")

# LinearExplainer is designed specifically for Logistic Regression/GLM
explainer = shap.LinearExplainer(lr_model, X_scaled)
shap_values = explainer.shap_values(X_scaled)

# Logistic Regression SHAP values are usually 2D (samples, features)
# so we don't need the class [1] indexing that Tree models need.
shap_values_to_plot = shap_values

# ==========================================
# 3. EXTRACT BIOMARKERS
# ==========================================
print("3. Extracting Top Biomarkers...")

# Calculate Mean Absolute SHAP across all samples
shap_sum = np.abs(shap_values_to_plot).mean(axis=0).flatten()

importance_df = pd.DataFrame({
    'Gene': candidate_genes,
    'SHAP_Importance': shap_sum
})

# Sort by importance
importance_df = importance_df.sort_values('SHAP_Importance', ascending=False).reset_index(drop=True)

print("\n--- Logistic Regression: Top 10 Most Critical Biomarkers ---")
importance_df.index += 1 
print(importance_df.head(10))

# Save rankings
importance_df.to_csv("LogisticRegression_Biomarker_Rankings.csv", index=False)

# ==========================================
# 4. GENERATE VISUALIZATION
# ==========================================
print("4. Generating SHAP Visualization...")
plt.figure(figsize=(12, 10))

shap.summary_plot(
    shap_values_to_plot, 
    X_scaled, 
    feature_names=candidate_genes, 
    show=False, 
    max_display=35
)

plt.title("SHAP Value Summary: Logistic Regression Impact on HD Diagnosis", fontsize=14)
plt.tight_layout()
plt.savefig("SHAP_logistic_regression.png", dpi=300, bbox_inches='tight')
plt.close()

print("Visual plot saved successfully as 'SHAP_logistic_regression.png'!")