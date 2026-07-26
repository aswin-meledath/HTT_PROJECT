import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
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
print("1. LOADING DATA AND PREPARING FEATURES (RUNS ONCE)")
print("==================================================")

# 1. LOAD CANDIDATE GENES
degs_df = pd.read_csv("Candidate_35_DEGs.csv")
candidate_genes = degs_df['Gene'].tolist()

harmonized_df = pd.read_csv("HD_Harmonized_Expression.csv", index_col=0)

# --- SAFE GENE EXTRACTION ---
available_genes = [g for g in candidate_genes if g in harmonized_df.index]
missing_genes = set(candidate_genes) - set(available_genes)
if missing_genes:
    print(f"Note: Missing genes dropped during harmonization: {missing_genes}")

X_full = harmonized_df.loc[available_genes].T

# 2. EXTRACT LABELS
mapping = {}
for file in ['GSE3790-GPL96_series_matrix.txt', 'GSE3790-GPL97_series_matrix.txt']:
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
                if 'control' in t: mapping[i] = 0
                elif 'disease' in t or 'hd' in t: mapping[i] = 1

valid_samples = [s for s in X_full.index if s in mapping]
X = X_full.loc[valid_samples].values.astype(np.float32)
y = np.array([mapping[s] for s in valid_samples])

# 3. DATA SPLITTING (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

print(f"\nTotal Samples: {X.shape[0]}")
print(f"Training Set (80%): {X_train.shape[0]} samples")
print(f"Testing Set (20%): {X_test.shape[0]} samples\n")

# 4. SCALING
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==========================================
# 2. FT-TRANSFORMER ARCHITECTURE
# ==========================================
class FeatureTokenizer(nn.Module):
    def __init__(self, num_features, d_token):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_features, d_token))
        self.bias   = nn.Parameter(torch.empty(num_features, d_token))
        nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5))
        nn.init.zeros_(self.bias)
    def forward(self, x): return x.unsqueeze(-1) * self.weight.unsqueeze(0) + self.bias.unsqueeze(0)

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
# 3. INITIALIZE ALL MODELS
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

cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
results_list = []

# ==========================================
# 4. TRAIN AND EVALUATE LOOP
# ==========================================
print("\n" + "="*50)
print("3. TRAINING & EVALUATING MODELS (INTERNAL 20% TEST SET)")
print("="*50)

for name, model in models.items():
    print(f"\n--> Processing {name}...")
    cv_mean = "N/A"
    
    # --- ELASTIC NET GLM (CUSTOM TUNING) ---
    if name == "Elastic Net GLM":
        print("  Running GridSearchCV for Hyperparameters...")
        param_grid = {'C': [0.001, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0], 'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9]}
        base_glm = LogisticRegression(penalty='elasticnet', solver='saga', class_weight='balanced', max_iter=5000, random_state=42)
        grid_search = GridSearchCV(base_glm, param_grid, cv=5, scoring='roc_auc', n_jobs=-1, verbose=0)
        grid_search.fit(X_train_scaled, y_train)
        
        best_C, best_l1 = grid_search.best_params_['C'], grid_search.best_params_['l1_ratio']
        glm_model = LogisticRegression(penalty='elasticnet', solver='saga', C=best_C, l1_ratio=best_l1, class_weight='balanced', max_iter=5000, random_state=42)
        
        cv_scores = cross_val_score(glm_model, X_train_scaled, y_train, cv=cv, scoring='roc_auc')
        cv_mean = cv_scores.mean()
        print(f"  10-Fold CV AUC: {cv_mean:.3f}")
        
        glm_model.fit(X_train_scaled, y_train)
        
        # Tuning threshold safely on the training set
        y_train_proba = glm_model.predict_proba(X_train_scaled)[:, 1]
        best_threshold, best_macro_f1 = 0.5, 0.0
        for thresh in np.arange(0.30, 0.70, 0.01):
            preds = (y_train_proba >= thresh).astype(int)
            macro_f1 = f1_score(y_train, preds, average='macro')
            if macro_f1 > best_macro_f1:
                best_macro_f1, best_threshold = macro_f1, thresh
                
        y_pred_proba = glm_model.predict_proba(X_test_scaled)[:, 1]
        y_pred = (y_pred_proba >= best_threshold).astype(int)
        
    # --- FT-TRANSFORMER (CUSTOM PYTORCH) ---
    elif name == "FT-Transformer":
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        EPOCHS, BATCH, LR = 60, 32, 1e-4
        
        print("  Running 10-Fold CV...")
        ft_cv_aucs = []
        for tr_idx, val_idx in cv.split(X_train_scaled, y_train):
            Xtr, Xval = X_train_scaled[tr_idx], X_train_scaled[val_idx]
            ytr, yval = y_train[tr_idx], y_train[val_idx]

            tr_dl = DataLoader(TensorDataset(torch.FloatTensor(Xtr), torch.LongTensor(ytr)), batch_size=BATCH, shuffle=True)
            val_dl = DataLoader(TensorDataset(torch.FloatTensor(Xval), torch.LongTensor(yval)), batch_size=BATCH)

            ft_model_cv = FTTransformer(num_features=Xtr.shape[1]).to(DEVICE)
            optimizer = torch.optim.AdamW(ft_model_cv.parameters(), lr=LR, weight_decay=1e-4)
            criterion = nn.CrossEntropyLoss()
            
            best_auc, patience = 0, 0
            for epoch in range(EPOCHS):
                ft_model_cv.train()
                for xb, yb in tr_dl:
                    xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                    optimizer.zero_grad()
                    criterion(ft_model_cv(xb), yb).backward()
                    optimizer.step()
                
                ft_model_cv.eval()
                val_probs = []
                with torch.no_grad():
                    for xb, _ in val_dl:
                        val_probs.extend(torch.softmax(ft_model_cv(xb.to(DEVICE)), dim=1)[:, 1].cpu().numpy())
                auc = roc_auc_score(yval, val_probs)
                if auc > best_auc:
                    best_auc = auc
                    patience = 0
                else:
                    patience += 1
                    if patience >= 10: break
            ft_cv_aucs.append(best_auc)
        
        cv_mean = np.mean(ft_cv_aucs)
        print(f"  10-Fold CV AUC: {cv_mean:.3f}")
        
        print("  Training Final FT Model...")
        train_dl = DataLoader(TensorDataset(torch.FloatTensor(X_train_scaled), torch.LongTensor(y_train)), batch_size=BATCH, shuffle=True)
        test_dl = DataLoader(TensorDataset(torch.FloatTensor(X_test_scaled), torch.LongTensor(y_test)), batch_size=BATCH)

        ft_model = FTTransformer(num_features=X_train_scaled.shape[1]).to(DEVICE)
        optimizer = torch.optim.AdamW(ft_model.parameters(), lr=LR, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()

        ft_model.train()
        for epoch in range(EPOCHS):
            for xb, yb in train_dl:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                optimizer.zero_grad()
                criterion(ft_model(xb), yb).backward()
                optimizer.step()

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
        print("  TabNet uses internal validation during epochs. Skipping standard CV.")
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
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=cv, scoring='roc_auc')
        cv_mean = cv_scores.mean()
        print(f"  10-Fold CV AUC: {cv_mean:.3f}")
        
        model.fit(X_train_scaled, y_train)
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
        y_pred = model.predict(X_test_scaled)
    
    # --- EVALUATE AND PRINT ---
    test_auc = roc_auc_score(y_test, y_pred_proba)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred) 
    
    print(f"  Internal Test AUC = {test_auc:.3f}")
    
    results_list.append({
        "Model": name,
        "10-Fold CV AUC": cv_mean if isinstance(cv_mean, float) else "N/A",
        "Test AUC (20%)": test_auc,
        "Test Accuracy": acc,
        "Test Macro F1": f1,
        "Test Recall (HD)": recall
    })

# ==========================================
# 5. GENERATE FINAL LEADERBOARD & VISUALIZATION
# ==========================================
print("\n" + "="*50)
print("FINAL INTERNAL TRAINING LEADERBOARD (80/20 SPLIT)")
print("="*50)

results_df = pd.DataFrame(results_list)
results_df = results_df.sort_values(by=["Test AUC (20%)", "Test Recall (HD)"], ascending=[False, False]).reset_index(drop=True)

print(results_df.to_string(index=False))
results_df.to_csv("Final_Training_Leaderboard.csv", index=False)
print("\nSaved leaderboard to 'Final_Training_Leaderboard.csv'")

# Generate Bar Chart
plt.figure(figsize=(14, 8))
sns.barplot(x="Test AUC (20%)", y="Model", data=results_df, palette="magma")
plt.title("Model Performance on Internal 20% Hold-Out Set", fontsize=16, fontweight='bold')
plt.xlabel("Area Under Curve (AUC)", fontsize=12)
plt.ylabel("")
plt.xlim(0, 1.0)
plt.grid(axis='x', linestyle='--', alpha=0.7)

for index, row in results_df.iterrows():
    label_text = f"AUC: {row['Test AUC (20%)']:.3f} | Recall: {row['Test Recall (HD)']:.3f}"
    plt.text(row['Test AUC (20%)'] + 0.01, index, label_text, va='center', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig("Final_Training_AUC_Comparison.png", dpi=300)
print("Saved comparison chart to 'Final_Training_AUC_Comparison.png'")