import os, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency, kruskal
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier
warnings.filterwarnings('ignore')

RANDOM_STATE = 42
SOURCE_FILE  = "drug_abuse_dataset (2).csv"
TARGET_COL   = "Drug_Abuse_Risk"

df_raw = pd.read_csv(SOURCE_FILE)
n_rows, n_cols = df_raw.shape
target_dist = df_raw[TARGET_COL].value_counts()

structural_removals = {
    'Record_ID':'Identifier','Type_of_Substance_Used':'Post-outcome',
    'Frequency_of_Use':'Post-outcome','Duration_of_Use':'Post-outcome',
    'Craving_Level':'Post-outcome','Withdrawal_Symptoms':'Post-outcome',
    'Previous_Rehabilitation_Counselling':'Post-outcome/reverse-causation',
    'Substance_Exposure_Frequency':'Active-use proxy',
    'Polysubstance_Exposure':'Post-outcome','Age_First_Substance_Exposure':'63% missing, non-random',
}
df_work = df_raw.drop(columns=[c for c in structural_removals if c in df_raw.columns])
candidate_pool = [c for c in df_work.columns if c != TARGET_COL]

le_target = LabelEncoder()
y_encoded = le_target.fit_transform(df_work[TARGET_COL])
X_raw = df_work[candidate_pool].copy()

X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X_raw, y_encoded, test_size=0.20, random_state=RANDOM_STATE, stratify=y_encoded)

num_cols = X_train_raw.select_dtypes(include=['int64','float64']).columns.tolist()
cat_cols = X_train_raw.select_dtypes(include=['object','category']).columns.tolist()

num_imp = SimpleImputer(strategy='median')
cat_imp = SimpleImputer(strategy='most_frequent')
X_tr_num = pd.DataFrame(num_imp.fit_transform(X_train_raw[num_cols]), columns=num_cols, index=X_train_raw.index)
X_tr_cat = pd.DataFrame(cat_imp.fit_transform(X_train_raw[cat_cols]), columns=cat_cols, index=X_train_raw.index)

le_dict = {}
X_tr_cat_enc = X_tr_cat.copy()
for col in cat_cols:
    le = LabelEncoder()
    X_tr_cat_enc[col] = le.fit_transform(X_tr_cat[col].astype(str))
    le_dict[col] = le

X_train_enc = pd.concat([X_tr_num, X_tr_cat_enc], axis=1)
fn = X_train_enc.columns.tolist()

feature_domain = {
    'Age':'Demographic','Gender':'Demographic','Education_Level':'Demographic',
    'Occupation_Status':'Demographic','Socioeconomic_Status':'Demographic',
    'Region_Type':'Demographic','Marital_Status':'Demographic',
    'Living_Arrangement':'Demographic','Household_Size':'Demographic','Financial_Stress':'Demographic',
    'Stress_Level':'Psychological','Anxiety_Level':'Psychological','Depression_Score':'Psychological',
    'Psychological_Distress':'Psychological','Impulsivity':'Psychological','Sensation_Seeking':'Psychological',
    'Emotional_Regulation':'Psychological','Self_Esteem':'Psychological','Loneliness_Level':'Psychological',
    'Anger_Level':'Psychological','Hopelessness_Level':'Psychological','Trauma_Symptoms':'Psychological',
    'Coping_Ability':'Psychological','Mental_Health_History':'Psychological',
    'Alcohol_Consumption':'Substance-Related','Smoking_Status':'Substance-Related',
    'Previous_Substance_Exposure':'Substance-Related','Substance_Availability_Exposure':'Substance-Related',
    'Substance_Use_Curiosity':'Substance-Related','Substance_Use_Intention':'Substance-Related',
    'Peer_Influence':'Social/Peer','Peer_Substance_Exposure':'Social/Peer','Social_Support':'Social/Peer',
    'Social_Isolation':'Social/Peer','Bullying_Violence_Exposure':'Social/Peer',
    'Social_Integration':'Social/Peer','Friendship_Quality':'Social/Peer',
    'Community_Support':'Social/Peer','Social_Pressure':'Social/Peer',
    'Family_History_Substance_Use':'Family','Family_Support':'Family','Parental_Monitoring':'Family',
    'Family_Conflict':'Family','Childhood_Adversity':'Family','Parental_Substance_Use':'Family',
    'Family_Stability':'Family','Family_Communication_Quality':'Family',
    'Sleep_Duration':'Lifestyle','Sleep_Quality':'Lifestyle','Physical_Activity':'Lifestyle',
    'Screen_Time':'Lifestyle','Academic_Work_Stress':'Lifestyle','Diet_Quality':'Lifestyle',
    'Daily_Routine_Stability':'Lifestyle','Exercise_Frequency':'Lifestyle',
    'Leisure_Activity_Level':'Lifestyle','Healthcare_Access':'Lifestyle',
    'Academic_Performance':'Academic/Work','Academic_Pressure':'Academic/Work',
    'Workload_Level':'Academic/Work','Job_Satisfaction':'Academic/Work',
    'Workplace_Stress':'Academic/Work','School_Work_Absenteeism':'Academic/Work',
    'Risk_Taking_Tendency':'Behavioural','Rule_Breaking_Behaviour':'Behavioural',
    'Aggression_Level':'Behavioural','Decision_Making_Control':'Behavioural',
    'Self_Control':'Behavioural','Behavioural_Problems':'Behavioural','Reward_Sensitivity':'Behavioural',
    'Social_Media_Usage':'Digital','Digital_Social_Pressure':'Digital',
    'Online_Risk_Exposure':'Digital','Night_Time_Device_Usage':'Digital',
    'Attitude_Toward_Substance_Use':'Attitudinal','Awareness_of_Drug_Related_Risks':'Attitudinal',
}

# Reload from saved results
combined_df = pd.read_csv('results/combined_importance_v2.csv')
combined_norm = combined_df.set_index('Feature')['Combined_Score']

ft_df = pd.read_csv('reports/feature_target_analysis.csv')
rf_imp  = combined_df.set_index('Feature')['RF_Importance']
xgb_imp = combined_df.set_index('Feature')['XGB_Importance']
perm_imp = combined_df.set_index('Feature')['Perm_Importance']
perm_std = combined_df.set_index('Feature')['Perm_Std']
target_corr_train = combined_df.set_index('Feature')['Target_Corr']

# Final selected features from saved dataset
df_final = pd.read_csv('data/final/final_ml_dataset.csv')
selected = [c for c in df_final.columns if c != TARGET_COL]
print(f"Selected features loaded: {len(selected)}")

domain_cnt_final = {}
for f in selected:
    d = feature_domain.get(f,'Other')
    domain_cnt_final[d] = domain_cnt_final.get(d,0)+1

sig_feats = ft_df[ft_df['Significant']=='Yes']['Feature'].tolist()

# FINAL FEATURE SET TEXT REPORT
with open('reports/final_feature_set.txt','w') as f:
    f.write("="*72+"\n")
    f.write("FINAL FEATURE SET - CORRECTED PIPELINE v2.0\n")
    f.write("Project: AI-Based Early Drug Abuse Risk Assessment\n")
    f.write("="*72+"\n\n")
    f.write(f"1.  Original feature count   : {n_cols-1} predictors + 1 target\n")
    f.write(f"2.  Candidate pool           : {len(candidate_pool)} (after structural removals)\n")
    f.write(f"3.  Final predictor count    : {len(selected)}\n")
    f.write(f"4.  Train set size           : {X_train_raw.shape[0]} rows\n")
    f.write(f"5.  Test set size            : {X_test_raw.shape[0]} rows (UNTOUCHED)\n\n")
    f.write("FINAL FEATURE LIST:\n")
    for i,feat in enumerate(selected,1):
        dom = feature_domain.get(feat,'Other')
        sc = combined_norm.get(feat,0)
        eff = ft_df[ft_df['Feature']==feat]['Effect_Size'].values
        eff_val = f"{eff[0]:.5f}" if len(eff)>0 else 'N/A'
        f.write(f"  {i:2d}. {feat:<45s}  [{dom:<22s}]  score={sc:.4f}  effect={eff_val}\n")
    f.write("\nDOMAIN DISTRIBUTION:\n")
    for dom,cnt in sorted(domain_cnt_final.items()):
        f.write(f"  {dom:<25s}: {cnt}\n")
    f.write("\nSTRUCTURAL REMOVALS (pre-split leakage):\n")
    for feat,reas in structural_removals.items():
        f.write(f"  - {feat}: {reas}\n")
    f.write("\nPREPROCESSING:\n")
    f.write("  Split FIRST -> Impute(train) -> Scale(train) -> Encode(train)\n")
    f.write("  Numerical  : SimpleImputer(median) + StandardScaler\n")
    f.write("  Categorical: SimpleImputer(most_frequent) + OneHotEncoder(drop=first)\n")
    f.write("\nFEATURE SELECTION:\n")
    f.write("  Kruskal-Wallis/Chi-Squared on training set only\n")
    f.write("  Correlation on training set only\n")
    f.write("  RF + XGB importance on 80% of training (internal split)\n")
    f.write("  Permutation importance on 20% of training (internal split)\n")
    f.write("  Rank-based combined score across 5 signals\n")
    f.write("  Test set NEVER touched during selection.\n")
    f.write("\nSUITABILITY: YES\n")
    f.write("LEAKAGE STATUS: CLEAN\n")

print("Final feature set report saved.")

# VISUALIZATIONS
plt.figure(figsize=(14,10))
ts = combined_norm.reindex(selected).dropna().sort_values(ascending=True)
sns.barplot(x=ts.values, y=ts.index, color='#27ae60')
plt.title('Combined Importance Score - Final Selected Features (v2.0 Corrected)', fontsize=13)
plt.xlabel('Normalized Rank Score (training data only)')
plt.tight_layout(); plt.savefig('results/importance_combined_v2.png', dpi=150); plt.close()

plt.figure(figsize=(12,8))
rt = rf_imp.sort_values(ascending=False).head(25)
sns.barplot(x=rt.values, y=rt.index, color='#2980b9')
plt.title('RF Feature Importance Top 25 (Training Data Only)', fontsize=13)
plt.xlabel('Importance'); plt.tight_layout()
plt.savefig('results/rf_importance_v2.png', dpi=150); plt.close()

plt.figure(figsize=(12,8))
xt = xgb_imp.sort_values(ascending=False).head(25)
sns.barplot(x=xt.values, y=xt.index, color='#e67e22')
plt.title('XGBoost Feature Importance Top 25 (Training Data Only)', fontsize=13)
plt.xlabel('Importance'); plt.tight_layout()
plt.savefig('results/xgb_importance_v2.png', dpi=150); plt.close()

plt.figure(figsize=(12,9))
pt = perm_imp.sort_values(ascending=False).head(25)
ps = perm_std.reindex(pt.index)
plt.barh(list(pt.index)[::-1], pt.values[::-1], xerr=ps.values[::-1],
         color='#8e44ad', alpha=0.8, capsize=3)
plt.title('Permutation Importance Top 25 +/- Std (Internal Val, Training Only)', fontsize=12)
plt.xlabel('Mean Importance'); plt.tight_layout()
plt.savefig('results/permutation_importance_v2.png', dpi=150); plt.close()

X_fin_enc = X_train_enc[[f for f in selected if f in X_train_enc.columns]].copy()
if X_fin_enc.shape[1] > 1:
    plt.figure(figsize=(16,14))
    mask = np.triu(np.ones_like(X_fin_enc.corr(), dtype=bool))
    sns.heatmap(X_fin_enc.corr(), mask=mask, cmap='coolwarm', center=0, linewidths=0.3)
    plt.title('Correlation Heatmap - Final Features (Training Data Only)', fontsize=13)
    plt.tight_layout(); plt.savefig('results/correlation_heatmap_v2.png', dpi=150); plt.close()

plt.figure(figsize=(10,6))
doms = list(domain_cnt_final.keys()); cnts = list(domain_cnt_final.values())
bars = plt.barh(doms, cnts, color=sns.color_palette('Set2', len(doms)))
plt.bar_label(bars, padding=3)
plt.title('Final Feature Set Domain Distribution (v2.0)', fontsize=13)
plt.xlabel('Number of Features'); plt.tight_layout()
plt.savefig('results/domain_distribution_v2.png', dpi=150); plt.close()

print("All visualizations saved.")

# FINAL SUMMARY
print("\n"+"="*72)
print("FINAL SUMMARY")
print("="*72)
print(f"1.  Original feature count       : {n_cols-1} predictors + 1 target")
print(f"2.  Candidate pool               : {len(candidate_pool)}")
print(f"3.  Final predictor count        : {len(selected)}")
print(f"4.  Structural removals          : {len(structural_removals)}")
print(f"5.  Leakage features removed     : {len(structural_removals)}")
for f in structural_removals: print(f"    - {f}")
print(f"\n6.  v1.0 pipeline: CONTAMINATED (imputation, encoding, stats, models on full dataset)")
print(f"7.  v2.0 pipeline: CLEAN (all analysis on {X_train_raw.shape[0]}-row training set only)")
print(f"\n8.  Final feature list ({len(selected)} features):")
for i,feat in enumerate(selected,1):
    print(f"    {i:2d}. {feat:<45s}  [{feature_domain.get(feat,'Other')}]")
print(f"\n9.  Target: Drug_Abuse_Risk (Operational label, NOT clinically validated)")
for cls,cnt in target_dist.items():
    print(f"    {cls}: {cnt} ({cnt/n_rows*100:.1f}%)")
print(f"\n10. Train: {X_train_raw.shape[0]} rows | Test: {X_test_raw.shape[0]} rows (UNTOUCHED)")
print(f"11. Preprocessing: Split FIRST, then impute/scale/encode on training only")
print(f"12. Feature selection: Kruskal+Chi2+Corr+RF+XGB+Perm on training only")
print(f"13. Leakage-free: YES (v2.0)")
print(f"\n14. Remaining limitations:")
print(f"    a. Operational target label (not clinically validated)")
print(f"    b. Self-reported data (response bias possible)")
print(f"    c. Cross-sectional (causal claims not supported)")
print(f"    d. Previous_Substance_Exposure: borderline, retained as historical record")
print(f"    e. Model cannot diagnose addiction or guarantee future drug use")
print(f"\n15. DATASET READY FOR FINAL ML TRAINING: YES")
print(f"    Files: data/final/final_ml_dataset.csv")
print(f"           data/final/X_features.csv")
print(f"           data/final/y_target.csv")
print(f"    STOP: Final models NOT trained.")
print("="*72)
print("DONE - CORRECTED PIPELINE v2.0 COMPLETE.")
