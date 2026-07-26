import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix, accuracy_score, f1_score, recall_score

# Import all models
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, StackingClassifier
from catboost import CatBoostClassifier
from pytorch_tabnet.tab_model import TabNetClassifier

warnings.filterwarnings("ignore")

print("==================================================")
print("1. LOADING DATA AND PREPARING FEATURES")
print("==================================================")

# A. LOAD CANDIDATE GENES (Strictly 35 Genes)
degs_df = pd.read_csv("Candidate_35_DEGs.csv")
candidate_genes = degs_df['Gene'].tolist()

# B. LOAD ORIGINAL HARMONIZED DATA
harmonized_df = pd.read_csv("HD_Harmonized_Expression.csv", index_col=0)
available_genes = [g for g in candidate_genes if g in harmonized_df.index]
X_train_full = harmonized_df.loc[available_genes].T

if len(available_genes) < 35:
    print(f"Note: Only {len(available_genes)} out of 35 genes found in harmonized data.")

# C. EXTRACT ORIGINAL LABELS (TRAINING DATA)
mapping_train = {}
for file in ['GSE3790-GPL96_series_matrix.txt', 'GSE3790-GPL97_series_matrix.txt']:
    with open(file, 'r') as f:
        ids, chars = [], []
        for line in f:
            if line.startswith('!Sample_geo_accession'):
                ids = [x.strip('"\n') for x in line.split('\t')[1:]]
            if line.startswith('!Sample_characteristics_ch1'):
                chars = [x.strip('"\n').lower() for x in line.split('\t')[1:]]
        for i, ch in zip(ids, chars):
            if 'control' in ch: mapping_train[i] = 0
            elif 'hd' in ch or 'huntington' in ch: mapping_train[i] = 1

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
            if 'huntington' in d:
                if 'control' in t: mapping_train[i] = 0
                elif 'disease' in t or 'hd' in t: mapping_train[i] = 1

valid_samples_train = [s for s in X_train_full.index if s in mapping_train]
X_train = X_train_full.loc[valid_samples_train].values
y_train = np.array([mapping_train[s] for s in valid_samples_train])

# D. SCALE THE TRAINING DATA
scaler_train = StandardScaler()
X_train_scaled = scaler_train.fit_transform(X_train)

print("\n2. Loading Independent Validation Set (GSE33000)...")

# E. LOAD THE INDEPENDENTLY SCALED VALIDATION DATA
X_test_scaled_df = pd.read_csv("GSE33000_X_scaled.csv", index_col=0)
X_test_scaled = X_test_scaled_df.values
y_test = pd.read_csv("GSE33000_y.csv", index_col=0)['Label'].values

print(f"Training Data: {X_train_scaled.shape[0]} samples")
print(f"Validation Data (GSE33000): {X_test_scaled.shape[0]} samples\n")

# ==========================================
# 3. FT-TRANSFORMER ARCHITECTURE
# ==========================================
class FeatureTokenizer(nn.Module):
    def __init__(self, num_features, d_token):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_features, d_token))
        self.bias   = nn.Parameter(torch.empty(num_features, d_token))
        nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5))
        nn.init.zeros_(self.bias)
    def forward(self, x):
        return x.unsqueeze(-1) * self.weight.unsqueeze(0) + self.bias.unsqueeze(0)

class TransformerBlock(nn.Module):
    def __init__(self, d_token, n_heads, ffn_factor=4, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_token)
        self.norm2 = nn.LayerNorm(d_token)
        self.attn  = nn.MultiheadAttention(d_token, n_heads, dropout=dropout, batch_first=True)
        self.ffn   = nn.Sequential(
            nn.Linear(d_token, d_token * ffn_factor), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_token * ffn_factor, d_token), nn.Dropout(dropout)
        )
    def forward(self, x):
        attn_out, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x))
        x = x + attn_out
        x = x + self.ffn(self.norm2(x))
        return x

class FTTransformer(nn.Module):
    def __init__(self, num_features, d_token=64, n_heads=8, n_layers=3, ffn_factor=4, dropout=0.1, num_classes=2):
        super().__init__()
        self.tokenizer = FeatureTokenizer(num_features, d_token)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_token))
        self.blocks    = nn.ModuleList([TransformerBlock(d_token, n_heads, ffn_factor, dropout) for _ in range(n_layers)])
        self.head = nn.Sequential(nn.LayerNorm(d_token), nn.Linear(d_token, num_classes))
    def forward(self, x):
        tokens = self.tokenizer(x)
        cls    = self.cls_token.expand(x.size(0), -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
        for block in self.blocks: tokens = block(tokens)
        return self.head(tokens[:, 0, :])

# ==========================================
# 4. INITIALIZE ALL MODELS
# ==========================================
base_models = [
    ('catboost', CatBoostClassifier(verbose=0, random_state=42, auto_class_weights='Balanced')),
    ('extra_trees', ExtraTreesClassifier(n_estimators=100, class_weight='balanced', random_state=42)),
    ('ann', MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, random_state=42)),
    ('svm', SVC(kernel='rbf', probability=True, class_weight='balanced', random_state=42))
]

models = {
    "Logistic Regression (Standard)": LogisticRegression(random_state=42, class_weight='balanced', max_iter=1000),
    "K-Nearest Neighbors (KNN)": KNeighborsClassifier(n_neighbors=5),
    "Support Vector Machine (SVM)": SVC(kernel='rbf', probability=True, class_weight='balanced', random_state=42),
    "Artificial Neural Network (ANN)": MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'),
    "Extra Trees": ExtraTreesClassifier(n_estimators=100, random_state=42, class_weight='balanced'),
    "CatBoost": CatBoostClassifier(verbose=0, random_state=42, auto_class_weights='Balanced'),
    "TabNet": TabNetClassifier(verbose=0, seed=42),
    "Stacked Ensemble": StackingClassifier(estimators=base_models, final_estimator=LogisticRegression(random_state=42), cv=5),
    "Elastic Net GLM": "Custom",
    "FT-Transformer": "Custom"
}

results_list = []

# ==========================================
# 5. TRAIN AND EVALUATE LOOP
# ==========================================
print("\n" + "="*50)
print("3. TRAINING & EVALUATING MODELS ON GSE33000")
print("="*50)

for name, model in models.items():
    print(f"\n--> Processing {name}...")
    
    # --- ELASTIC NET GLM (CUSTOM) ---
    if name == "Elastic Net GLM":
        param_grid = {'C': [0.001, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0], 'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9]}
        base_glm = LogisticRegression(penalty='elasticnet', solver='saga', class_weight='balanced', max_iter=5000, random_state=42)
        grid_search = GridSearchCV(base_glm, param_grid, cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42), scoring='roc_auc', n_jobs=-1, verbose=0)
        grid_search.fit(X_train_scaled, y_train)
        
        best_C, best_l1 = grid_search.best_params_['C'], grid_search.best_params_['l1_ratio']
        glm_model = LogisticRegression(penalty='elasticnet', solver='saga', C=best_C, l1_ratio=best_l1, class_weight='balanced', max_iter=5000, random_state=42)
        glm_model.fit(X_train_scaled, y_train)
        
        # Calculate optimal threshold on training data to prevent leakage
        y_train_proba = glm_model.predict_proba(X_train_scaled)[:, 1]
        best_threshold, best_macro_f1 = 0.5, 0.0
        for thresh in np.arange(0.30, 0.70, 0.01):
            preds = (y_train_proba >= thresh).astype(int)
            macro_f1 = f1_score(y_train, preds, average='macro')
            if macro_f1 > best_macro_f1:
                best_macro_f1, best_threshold = macro_f1, thresh
                
        y_pred_proba = glm_model.predict_proba(X_test_scaled)[:, 1]
        y_pred = (y_pred_proba >= best_threshold).astype(int)
    
    # --- FT-TRANSFORMER (CUSTOM) ---
    elif name == "FT-Transformer":
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        EPOCHS, BATCH, LR = 60, 32, 1e-4
        
        train_dl = DataLoader(TensorDataset(torch.FloatTensor(X_train_scaled), torch.LongTensor(y_train)), batch_size=BATCH, shuffle=True)
        test_dl = DataLoader(TensorDataset(torch.FloatTensor(X_test_scaled.astype(np.float32)), torch.LongTensor(y_test)), batch_size=BATCH)

        ft_model = FTTransformer(num_features=X_train_scaled.shape[1]).to(DEVICE)
        optimizer = torch.optim.AdamW(ft_model.parameters(), lr=LR, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
        criterion = nn.CrossEntropyLoss()

        ft_model.train()
        for epoch in range(EPOCHS):
            for xb, yb in train_dl:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                optimizer.zero_grad()
                criterion(ft_model(xb), yb).backward()
                optimizer.step()
            scheduler.step()

        ft_model.eval()
        all_probs, all_preds = [], []
        with torch.no_grad():
            for xb, _ in test_dl:
                logits = ft_model(xb.to(DEVICE))
                all_probs.extend(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
                all_preds.extend(logits.argmax(dim=1).cpu().numpy())
        
        y_pred_proba = np.array(all_probs)
        y_pred = np.array(all_preds)
    
    # --- TABNET ---
    elif name == "TabNet":
        model.fit(
            X_train=X_train_scaled, y_train=y_train,
            eval_set=[(X_train_scaled, y_train), (X_test_scaled, y_test)],
            eval_name=['train', 'valid'], eval_metric=['auc'],
            max_epochs=100, patience=15, batch_size=32, virtual_batch_size=16
        )
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
        y_pred = model.predict(X_test_scaled)
        
    # --- ALL OTHER CLASSICAL MODELS ---
    else:
        model.fit(X_train_scaled, y_train)
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
        y_pred = model.predict(X_test_scaled)
    
    # --- EVALUATE AND PRINT ---
    test_auc = roc_auc_score(y_test, y_pred_proba)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred) # Calculates recall specifically for the HD class (1)
    
    print(f"\nExternal Validation AUC = {test_auc:.3f}")
    print(f"Recall (Sensitivity) = {recall:.3f}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Control (0)", "HD (1)"]))
    
    results_list.append({
        "Model": name,
        "Validation AUC": test_auc,
        "Accuracy": acc,
        "Macro F1": f1,
        "Recall (HD)": recall
    })

# ==========================================
# 6. GENERATE FINAL LEADERBOARD & VISUALIZATION
# ==========================================
print("\n" + "="*50)
print("FINAL GSE33000 VALIDATION LEADERBOARD (35 GENES)")
print("="*50)

results_df = pd.DataFrame(results_list)
# Sort primarily by AUC, then by Recall if there is a tie
results_df = results_df.sort_values(by=["Validation AUC", "Recall (HD)"], ascending=[False, False]).reset_index(drop=True)

print(results_df.to_string(index=False))
results_df.to_csv("Final_GSE33000_Leaderboard.csv", index=False)
print("\nSaved leaderboard to 'Final_GSE33000_Leaderboard.csv'")

# Generate Bar Chart
plt.figure(figsize=(14, 8))
sns.barplot(x="Validation AUC", y="Model", data=results_df, palette="viridis")
plt.title("Model Performance on Independent Validation Set (GSE33000)", fontsize=16, fontweight='bold')
plt.xlabel("Area Under Curve (AUC)", fontsize=12)
plt.ylabel("")
plt.xlim(0, 1.0)
plt.grid(axis='x', linestyle='--', alpha=0.7)

for index, row in results_df.iterrows():
    # Adding both AUC and Recall to the chart labels for maximum clarity
    label_text = f"AUC: {row['Validation AUC']:.3f} | Recall: {row['Recall (HD)']:.3f}"
    plt.text(row['Validation AUC'] + 0.01, index, label_text, va='center', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig("Final_GSE33000_AUC_Comparison.png", dpi=300)
print("Saved comparison chart to 'Final_GSE33000_AUC_Comparison.png'")