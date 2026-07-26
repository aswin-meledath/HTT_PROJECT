import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier, StackingClassifier
from sklearn.svm import SVC
from catboost import CatBoostClassifier
from sklearn.neural_network import MLPClassifier

warnings.filterwarnings("ignore")

# ==========================================
# 1. LOAD DATA & TRAIN ENSEMBLE
# ==========================================
print("1. Loading Data and Training Final Stacked Ensemble Model...")

# A. Load Candidate Genes
degs_df = pd.read_csv("Candidate_35_DEGs.csv")
candidate_genes = degs_df['Gene'].tolist()

# B. Load Original Harmonized Data
harmonized_df = pd.read_csv("HD_Harmonized_Expression.csv", index_col=0)
X_full = harmonized_df.loc[candidate_genes].T

# C. Extract Labels
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

# D. Scale Data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# E. Initialize Models
base_models = [
    ('catboost', CatBoostClassifier(verbose=0, random_state=42, auto_class_weights='Balanced')),
    ('extra_trees', ExtraTreesClassifier(n_estimators=100, class_weight='balanced', random_state=42)),
    ('ann', MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, random_state=42)),
    ('svm', SVC(kernel='rbf', probability=True, class_weight='balanced', random_state=42))
]

ensemble_model = StackingClassifier(
    estimators=base_models, 
    final_estimator=LogisticRegression(random_state=42), 
    cv=5
)

# Train the full model
ensemble_model.fit(X_scaled, y)
print("Model training complete.")

# ==========================================
# 2. RUN FAST KERNEL SHAP
# ==========================================
print("2. Running FAST Kernel SHAP (Using 10 K-Means centroids)...")

# SPEED FIX: We summarize the background into 10 points to make Kernel SHAP fast
background_summary = shap.kmeans(X_scaled, 10) 
explainer = shap.KernelExplainer(ensemble_model.predict_proba, background_summary)

# Calculate SHAP values for the whole dataset
# This will now take minutes instead of hours
shap_values = explainer.shap_values(X_scaled)

# Extract Class 1 (Huntington's Disease)
if isinstance(shap_values, list):
    shap_values_to_plot = shap_values[1]
elif len(shap_values.shape) == 3:
    shap_values_to_plot = shap_values[:, :, 1]
else:
    shap_values_to_plot = shap_values

# ==========================================
# 3. EXTRACT BIOMARKERS & DIRECTIONS
# ==========================================
print("3. Extracting Top Biomarkers and Effect Directions...")

# Calculate Mean Absolute SHAP (The Importance Weight)
shap_sum = np.abs(shap_values_to_plot).mean(axis=0).flatten()

# Calculate Direction (Relationship between expression level and risk)
directions = []
for i in range(X_scaled.shape[1]):
    corr = np.corrcoef(X_scaled[:, i], shap_values_to_plot[:, i])[0, 1]
    if corr > 0:
        directions.append("Up-regulated (High Expression = High Risk)")
    else:
        directions.append("Down-regulated (Low Expression = High Risk)")

importance_df = pd.DataFrame({
    'Gene': candidate_genes,
    'SHAP_Importance': shap_sum,
    'Biological_Effect': directions
})

# Sort by importance
importance_df = importance_df.sort_values('SHAP_Importance', ascending=False).reset_index(drop=True)

print("\n--- Stacked Ensemble: Top 10 Critical Biomarkers ---")
importance_df.index += 1
print(importance_df.head(10))

# Save the CSV for your report and your teammate
importance_df.to_csv("StackedEnsemble_Biomarker_Rankings.csv", index=False)

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

plt.title("SHAP Value Summary: Stacked Ensemble Final Analysis", fontsize=14)
plt.tight_layout()
plt.savefig("SHAP_stacked_ensemble.png", dpi=300, bbox_inches='tight')
plt.close()

print("Visual plot saved: 'SHAP_stacked_ensemble.png'!")
print("CSV saved: 'StackedEnsemble_Biomarker_Rankings.csv'!")


'''import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier, StackingClassifier
from sklearn.svm import SVC
from catboost import CatBoostClassifier
from sklearn.neural_network import MLPClassifier

warnings.filterwarnings("ignore")

# ==========================================
# 1. LOAD DATA & TRAIN ENSEMBLE
# ==========================================
print("1. Loading Data and Training Final Stacked Ensemble Model...")

degs_df = pd.read_csv("Candidate_35_DEGs.csv")
candidate_genes = degs_df['Gene'].tolist()

harmonized_df = pd.read_csv("HD_Harmonized_Expression.csv", index_col=0)
X_full = harmonized_df.loc[candidate_genes].T

# Extract Labels (Same logic as your previous script)
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
    except FileNotFoundError: continue

valid_samples = [s for s in X_full.index if s in mapping]
X = X_full.loc[valid_samples]
y = np.array([mapping[s] for s in valid_samples])

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Define the exact base models from your validation code
base_models = [
    ('catboost', CatBoostClassifier(verbose=0, random_state=42, auto_class_weights='Balanced')),
    ('extra_trees', ExtraTreesClassifier(n_estimators=100, class_weight='balanced', random_state=42)),
    ('ann', MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, random_state=42)),
    ('svm', SVC(kernel='rbf', probability=True, class_weight='balanced', random_state=42))
]

ensemble_model = StackingClassifier(
    estimators=base_models, 
    final_estimator=LogisticRegression(random_state=42), 
    cv=5
)
ensemble_model.fit(X_scaled, y)

# ==========================================
# 2. RUN KERNEL SHAP (Slowest part)
# ==========================================
print("2. Running Kernel SHAP (This may take 5-10 minutes)...")

# We use 50 samples as 'background' to represent the typical data distribution
# This makes KernelExplainer finish in a reasonable time.
background = shap.sample(X_scaled, 50) 
explainer = shap.KernelExplainer(ensemble_model.predict_proba, background)

# Calculate SHAP values for the full dataset
shap_values = explainer.shap_values(X_scaled)

# FIX: Kernel SHAP returns [Class 0, Class 1]. We extract Class 1 (Huntington's).
if isinstance(shap_values, list):
    shap_values_to_plot = shap_values[1]
elif len(shap_values.shape) == 3:
    shap_values_to_plot = shap_values[:, :, 1]
else:
    shap_values_to_plot = shap_values

# ==========================================
# 3. EXTRACT BIOMARKERS
# ==========================================
print("3. Extracting Top Biomarkers...")

shap_sum = np.abs(shap_values_to_plot).mean(axis=0).flatten()
importance_df = pd.DataFrame({
    'Gene': candidate_genes,
    'SHAP_Importance': shap_sum
})

importance_df = importance_df.sort_values('SHAP_Importance', ascending=False).reset_index(drop=True)

print("\n--- Stacked Ensemble: Top 10 Most Critical Biomarkers ---")
importance_df.index += 1
print(importance_df.head(10))

importance_df.to_csv("StackedEnsemble_Biomarker_Rankings.csv", index=False)

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

plt.title("SHAP Value Summary: Stacked Ensemble Impact on HD", fontsize=14)
plt.tight_layout()
plt.savefig("SHAP_stacked_ensemble.png", dpi=300, bbox_inches='tight')
plt.close()

print("Visual plot saved successfully as 'SHAP_stacked_ensemble.png'!")
'''