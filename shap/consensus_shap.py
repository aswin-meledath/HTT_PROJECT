import pandas as pd
import glob
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Define the files (Make sure these match your exact filenames in the folder)
files = {
    "ExtraTrees": "ExtraTrees_Biomarker_Rankings.csv",
    "SVM": "SVM_Biomarker_Rankings.csv", # Based on your first script's name
    "LogReg": "LogisticRegression_Biomarker_Rankings.csv",
    "ElasticNet": "ElasticNet_Biomarker_Rankings.csv",
    "Stacked": "StackedEnsemble_Biomarker_Rankings.csv"
}

# 2. Load all CSVs into a list
all_dfs = []
for model_name, file_path in files.items():
    try:
        df = pd.read_csv(file_path)
        # We only need the Gene and the Importance column
        # Finding the importance column (it might be named 'SHAP_Importance' or 'Score')
        imp_col = [c for c in df.columns if 'Importance' in c or 'Score' in c or 'SHAP' in c][0]
        
        temp_df = df[['Gene', imp_col]].copy()
        temp_df.columns = ['Gene', f'{model_name}_Score']
        temp_df.set_index('Gene', inplace=True)
        all_dfs.append(temp_df)
    except FileNotFoundError:
        print(f"Warning: {file_path} not found. Skipping...")

# 3. Merge all models together
consensus_df = pd.concat(all_dfs, axis=1)

# 4. Calculate Final Average (Ignoring the text columns automatically)
consensus_df['Final_Consensus_Score'] = consensus_df.mean(axis=1)

# 5. Sort by most important
consensus_df = consensus_df.sort_values(by='Final_Consensus_Score', ascending=False)

# 6. Save the FINAL CSV (This is the file for your teammate)
consensus_df.to_csv("FINAL_CONSENSUS_RESULTS.csv")

# 7. Create a Final Chart for your Report
plt.figure(figsize=(10, 8))
top_15 = consensus_df.head(15)
sns.barplot(x=top_15['Final_Consensus_Score'], y=top_15.index, palette='magma')
plt.title("Top 15 Consensus Biomarkers for Huntington's Disease", fontsize=14)
plt.xlabel("Average SHAP Importance (Across 5 Models)")
plt.ylabel("Gene Symbol")
plt.tight_layout()
plt.savefig("FINAL_CONSENSUS_CHART.png", dpi=300)

print("\n--- CONSENSUS COMPLETE ---")
print(f"Total Genes Analyzed: {len(consensus_df)}")
print("Top 10 Overall GENES:")
print(consensus_df['Final_Consensus_Score'].head(10))
print("\nFiles saved: 'FINAL_CONSENSUS_RESULTS.csv' and 'FINAL_CONSENSUS_CHART.png'")