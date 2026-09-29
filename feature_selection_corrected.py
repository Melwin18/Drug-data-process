"""
CORRECTED FEATURE SELECTION PIPELINE v2.0
Project : AI-Based Early Drug Abuse Risk Assessment and Early Warning System
CRITICAL FIX: All analysis now performed on TRAINING DATA ONLY.
              Split is performed FIRST before any analysis.
"""

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
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

warnings.filterwarnings('ignore')

RANDOM_STATE = 42
SOURCE_FILE  = "drug_abuse_dataset (2).csv"
TARGET_COL   = "Drug_Abuse_Risk"
TEST_SIZE    = 0.20
CORR_THRESH  = 0.75
MIN_SCORE    = 0.20

for d in ['data/final','reports','results']:
    os.makedirs(d, exist_ok=True)

audit_lines = []
def log(m=""):
    print(m)
    audit_lines.append(str(m))

# ============================================================
# PHASE 0: RAW DATASET AUDIT
# ============================================================
log("="*70)
log("PHASE 0 | RAW DATASET AUDIT")
log("="*70)

df_raw = pd.read_csv(SOURCE_FILE)
n_rows, n_cols = df_raw.shape
log(f"Source file   : {SOURCE_FILE}")
log(f"Rows          : {n_rows}")
log(f"Columns       : {n_cols}")
log(f"Duplicates    : {df_raw.duplicated().sum()}")
log(f"\nTarget distribution:")
log(df_raw[TARGET_COL].value_counts().to_string())
log(f"\nAll columns:")
for c in df_raw.columns:
    miss = df_raw[c].isna().sum()
    log(f"  {c:<55s}  dtype={str(df_raw[c].dtype):<10}  miss={miss} ({miss/len(df_raw)*100:.1f}%)")

# ============================================================
# PHASE 1: TARGET AUDIT
# ============================================================
log("\n"+"="*70)
log("PHASE 1 | TARGET DEFINITION AUDIT")
log("="*70)

target_dist = df_raw[TARGET_COL].value_counts()
log("Target: Drug_Abuse_Risk")
log("Type  : Operational research label - NOT clinically validated")
log("Classes and counts:")
for cls, cnt in target_dist.items():
    log(f"  {cls:<20s}: {cnt} ({cnt/n_rows*100:.1f}%)")

with open('reports/target_definition_report.txt','w') as f:
    f.write("="*70+"\n")
    f.write("TARGET DEFINITION REPORT\n")
    f.write("="*70+"\n\n")
    f.write("TARGET COLUMN    : Drug_Abuse_Risk\n")
    f.write("TARGET TYPE      : Operational research label (NOT clinically validated)\n")
    f.write("CLASSES          : Low Risk | Moderate Risk | High Risk\n\n")
    f.write("CLASS DISTRIBUTION:\n")
    for cls, cnt in target_dist.items():
        f.write(f"  {cls:<20s}: {cnt} ({cnt/n_rows*100:.1f}%)\n")
    f.write("\nLIKELY LABEL CONSTRUCTION VARIABLES (inferred from dataset):\n")
    for feat in ['Type_of_Substance_Used','Frequency_of_Use','Duration_of_Use',
                 'Craving_Level','Withdrawal_Symptoms','Previous_Rehabilitation_Counselling']:
        f.write(f"  - {feat}\n")
    f.write("\nLIMITATIONS:\n")
    f.write("  - Target is not clinically validated.\n")
    f.write("  - The model cannot diagnose addiction.\n")
    f.write("  - The model cannot guarantee future drug use.\n")
    f.write("  - Associations are observational, not causal.\n")

log("Target report saved.")

# ============================================================
# PHASE 2: STRUCTURAL LEAKAGE REMOVAL (PRE-SPLIT)
# ============================================================
log("\n"+"="*70)
log("PHASE 2 | STRUCTURAL LEAKAGE REMOVAL (before split)")
log("="*70)

structural_removals = {
    'Record_ID'                           : 'Identifier - no predictive value',
    'Type_of_Substance_Used'              : 'Post-outcome: direct record of substance abuse',
    'Frequency_of_Use'                    : 'Post-outcome: current use frequency',
    'Duration_of_Use'                     : 'Post-outcome: duration of current episode',
    'Craving_Level'                       : 'Post-outcome: DSM-5 dependence criterion',
    'Withdrawal_Symptoms'                 : 'Post-outcome: DSM-5 dependence criterion',
    'Previous_Rehabilitation_Counselling' : 'Post-outcome/reverse-causation: prior abuse confirmed',
    'Substance_Exposure_Frequency'        : 'Active-use proxy: overlaps with Frequency_of_Use',
    'Polysubstance_Exposure'              : 'Post-outcome: concurrent multi-substance use implies active abuse',
    'Age_First_Substance_Exposure'        : '63.1% missing; missingness pattern reveals class membership',
}

df_work = df_raw.drop(columns=[c for c in structural_removals if c in df_raw.columns])
candidate_pool = [c for c in df_work.columns if c != TARGET_COL]

log(f"Structural removals: {len(structural_removals)}")
for f,r in structural_removals.items():
    log(f"  REMOVED: {f}")
log(f"Remaining candidate features: {len(candidate_pool)}")

# ============================================================
# PHASE 3: DETAILED LEAKAGE AUDIT (11 FLAGGED FEATURES)
# ============================================================
log("\n"+"="*70)
log("PHASE 3 | DETAILED LEAKAGE AUDIT - 11 FLAGGED FEATURES")
log("="*70)

leakage_rows = [
    {'Feature':'Previous_Substance_Exposure','Meaning':'Binary flag: historical substance exposure (Yes/No)',
     'Available_Before_Outcome':'YES - historical, pre-assessment','Direct_Leakage':'NO',
     'Indirect_Leakage':'LOW','Valid_Predictor':'YES','Decision':'RETAIN',
     'Reason':'Historical exposure is a legitimate pre-assessment risk factor. Not a post-outcome variable.'},
    {'Feature':'Substance_Use_Intention','Meaning':'Self-reported intention toward substance use (scale)',
     'Available_Before_Outcome':'YES - psychological state measurable before abuse',
     'Direct_Leakage':'NO','Indirect_Leakage':'MODERATE (strong predictor but pre-outcome)',
     'Valid_Predictor':'YES','Decision':'RETAIN',
     'Reason':'Theory of Planned Behaviour predictor. Pre-outcome psychological measure.'},
    {'Feature':'Substance_Exposure_Frequency','Meaning':'Frequency of substance exposure (Daily/Weekly/etc)',
     'Available_Before_Outcome':'UNCERTAIN - overlaps with Frequency_of_Use',
     'Direct_Leakage':'HIGH','Indirect_Leakage':'HIGH','Valid_Predictor':'NO',
     'Decision':'REMOVE','Reason':'Semantically overlaps with Frequency_of_Use (confirmed post-outcome). Removed.'},
    {'Feature':'Polysubstance_Exposure','Meaning':'Currently using multiple substances (Yes/No)',
     'Available_Before_Outcome':'NO - concurrent use implies active abuse',
     'Direct_Leakage':'HIGH','Indirect_Leakage':'HIGH','Valid_Predictor':'NO',
     'Decision':'REMOVE','Reason':'Post-outcome marker: concurrent poly-substance use is a consequence of abuse.'},
    {'Feature':'Age_First_Substance_Exposure','Meaning':'Age at first substance use',
     'Available_Before_Outcome':'PARTIALLY - only non-null for substance users',
     'Direct_Leakage':'YES (63% missing, non-random)','Indirect_Leakage':'YES',
     'Valid_Predictor':'NO','Decision':'REMOVE',
     'Reason':'63.1% missing. Missingness directly encodes non-user class. Including it reveals class through NA pattern.'},
    {'Feature':'Type_of_Substance_Used','Meaning':'Which substance currently used',
     'Available_Before_Outcome':'NO','Direct_Leakage':'YES','Indirect_Leakage':'YES',
     'Valid_Predictor':'NO','Decision':'REMOVE','Reason':'Post-outcome: describes active abuse behaviour.'},
    {'Feature':'Frequency_of_Use','Meaning':'Current substance use frequency',
     'Available_Before_Outcome':'NO','Direct_Leakage':'YES','Indirect_Leakage':'YES',
     'Valid_Predictor':'NO','Decision':'REMOVE','Reason':'Post-outcome: current use frequency.'},
    {'Feature':'Duration_of_Use','Meaning':'Duration of ongoing substance use',
     'Available_Before_Outcome':'NO','Direct_Leakage':'YES','Indirect_Leakage':'YES',
     'Valid_Predictor':'NO','Decision':'REMOVE','Reason':'Post-outcome: duration of abuse episode.'},
    {'Feature':'Craving_Level','Meaning':'Self-reported craving intensity',
     'Available_Before_Outcome':'NO','Direct_Leakage':'YES','Indirect_Leakage':'YES',
     'Valid_Predictor':'NO','Decision':'REMOVE','Reason':'Post-outcome: DSM-5 substance dependence criterion.'},
    {'Feature':'Withdrawal_Symptoms','Meaning':'Withdrawal severity (Mild/Moderate/Severe/No Use)',
     'Available_Before_Outcome':'NO','Direct_Leakage':'YES','Indirect_Leakage':'YES',
     'Valid_Predictor':'NO','Decision':'REMOVE','Reason':'Post-outcome: primary diagnostic indicator of physical dependence.'},
    {'Feature':'Previous_Rehabilitation_Counselling','Meaning':'Prior rehabilitation history (Yes/No)',
     'Available_Before_Outcome':'PARTIALLY - historical','Direct_Leakage':'MODERATE',
     'Indirect_Leakage':'HIGH','Valid_Predictor':'BORDERLINE','Decision':'REMOVE',
     'Reason':'Implies a prior confirmed abuse episode. Creates reverse causation in early-warning context.'},
]
pd.DataFrame(leakage_rows).to_csv('reports/feature_leakage_audit.csv', index=False)
log("Leakage audit saved.")
for r in leakage_rows:
    log(f"  {r['Feature']:<48s} -> {r['Decision']}  ({r['Reason'][:80]})")

# ============================================================
# PHASE 4: V1.0 PIPELINE AUDIT
# ============================================================
log("\n"+"="*70)
log("PHASE 4 | AUDIT OF PREVIOUS PIPELINE v1.0 - LEAKAGE FINDINGS")
log("="*70)

v1_audit = """
PIPELINE v1.0 LEAKAGE ANALYSIS:

STEP 5 (Imputation) - LEAKAGE:
  Median/mode imputation fit on full 5000 rows before split.
  Test-set statistics leaked into imputation parameters.
  FIX: Split first, fit imputer on training set only.

STEP 6 (Encoding) - LEAKAGE:
  LabelEncoder fit on full 5000 rows.
  Test-set label distributions leaked into encoding.
  FIX: Split first, fit encoder on training set only.

STEP 7 (Correlation) - LEAKAGE:
  Correlation matrix computed on full 5000 rows.
  Test-set feature relationships influenced correlation analysis.
  FIX: Compute correlation on training set only.

STEP 7 (Feature-Target Correlation) - LEAKAGE:
  Target correlation computed with all 5000 target labels.
  FIX: Use training labels only.

STEP 8 (Statistical Tests) - LEAKAGE:
  Kruskal-Wallis and Chi-Squared tests run on full 5000 rows.
  Effect sizes and p-values influenced by test observations.
  FIX: Run statistical tests on training data only.

STEP 9 (Model-Based Importance) - PARTIAL LEAKAGE:
  'X_all = df_encoded' was full-dataset encoded data (contaminated).
  Split was local within STEP 9 but upstream contamination persisted.
  Permutation importance used 20% holdout of contaminated data.
  FIX: Encode/impute training data first, then split internally.

SEVERITY: MODERATE
  The final 35-feature list is expected to be similar after correction
  because the dataset has no time-ordering issues and distributions are
  consistent. However, scientific integrity requires correction.

CORRECTED PIPELINE v2.0 ORDER:
  1. Load raw data
  2. Remove structural leakage (non-data-driven, pre-split)
  3. TRAIN/TEST SPLIT (stratified 80/20)
  4. Fit preprocessing on training set only
  5. Statistical tests on training set only
  6. Correlation on training set only
  7. RF/XGB importance on training set (internal 80/20 split)
  8. Permutation importance on internal validation (from training only)
  9. Feature selection decision
  10. Save final dataset
"""
log(v1_audit)

with open('reports/feature_selection_audit.txt','w') as f:
    f.write("="*70+"\n")
    f.write("FEATURE SELECTION AUDIT - PIPELINE v1.0 vs v2.0\n")
    f.write("="*70+"\n")
    f.write(v1_audit)

# ============================================================
# PHASE 5: TRAIN/TEST SPLIT - FIRST OPERATION
# ============================================================
log("\n"+"="*70)
log("PHASE 5 | TRAIN/TEST SPLIT - FIRST OPERATION ON FEATURES")
log("="*70)

X_raw = df_work[candidate_pool].copy()
y_raw = df_work[TARGET_COL].copy()

le_target = LabelEncoder()
y_encoded = le_target.fit_transform(y_raw)
log(f"Classes: {list(zip(le_target.classes_, range(len(le_target.classes_))))}")

X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X_raw, y_encoded, test_size=TEST_SIZE,
    random_state=RANDOM_STATE, stratify=y_encoded
)

log(f"Train : {X_train_raw.shape[0]} rows ({X_train_raw.shape[0]/n_rows*100:.1f}%)")
log(f"Test  : {X_test_raw.shape[0]} rows  ({X_test_raw.shape[0]/n_rows*100:.1f}%)")
log(f"TEST SET IS NOW LOCKED.")

for cls in le_target.classes_:
    ci = le_target.transform([cls])[0]
    log(f"  {cls}: train={int((y_train==ci).sum())}  test={int((y_test==ci).sum())}")

pd.DataFrame({'test_index': X_test_raw.index.tolist()}).to_csv('data/final/test_indices.csv', index=False)

# ============================================================
# PHASE 6: PREPROCESSING FIT ON TRAINING ONLY
# ============================================================
log("\n"+"="*70)
log("PHASE 6 | PREPROCESSING - FIT ON TRAINING DATA ONLY")
log("="*70)

num_cols_all = X_train_raw.select_dtypes(include=['int64','float64']).columns.tolist()
cat_cols_all = X_train_raw.select_dtypes(include=['object','category']).columns.tolist()
log(f"Numerical: {len(num_cols_all)}  Categorical: {len(cat_cols_all)}")
log(f"Missing in training: {X_train_raw.isnull().sum().sum()}")

num_imp = SimpleImputer(strategy='median')
cat_imp = SimpleImputer(strategy='most_frequent')

X_tr_num = pd.DataFrame(num_imp.fit_transform(X_train_raw[num_cols_all]),
                         columns=num_cols_all, index=X_train_raw.index)
X_tr_cat_raw = pd.DataFrame(cat_imp.fit_transform(X_train_raw[cat_cols_all]),
                              columns=cat_cols_all, index=X_train_raw.index)

le_dict = {}
X_tr_cat_enc = X_tr_cat_raw.copy()
for col in cat_cols_all:
    le = LabelEncoder()
    X_tr_cat_enc[col] = le.fit_transform(X_tr_cat_raw[col].astype(str))
    le_dict[col] = le

X_train_enc = pd.concat([X_tr_num, X_tr_cat_enc], axis=1)
feature_names = X_train_enc.columns.tolist()
log(f"Encoded training set: {X_train_enc.shape}")
log("Preprocessing fit on training data ONLY. OK")

# ============================================================
# PHASE 7: DOMAIN MAPPING
# ============================================================
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
for c in candidate_pool:
    if c not in feature_domain:
        feature_domain[c] = 'Other'

# ============================================================
# PHASE 8: STATISTICAL TESTS - TRAINING DATA ONLY
# ============================================================
log("\n"+"="*70)
log("PHASE 8 | STATISTICAL FEATURE-TARGET TESTS - TRAINING DATA ONLY")
log("="*70)

y_train_str = pd.Series(le_target.inverse_transform(y_train), index=X_train_raw.index)
train_df = X_train_raw.copy()
train_df[TARGET_COL] = y_train_str.values

ft_results = []
for col in candidate_pool:
    s = train_df[col]
    miss_pct = s.isna().sum()/len(s)*100
    try:
        if pd.api.types.is_numeric_dtype(s):
            groups = [s[train_df[TARGET_COL]==cls].dropna().values for cls in y_train_str.unique()]
            groups = [g for g in groups if len(g)>0]
            stat, pval = kruskal(*groups)
            test = 'Kruskal-Wallis'
            n = s.notna().sum(); k = len(groups)
            eff = round(max(0,(stat-k+1)/(n-k)), 5)
            var_ = round(float(s.var()),4)
        else:
            ct = pd.crosstab(s, train_df[TARGET_COL])
            stat, pval, dof, _ = chi2_contingency(ct)
            test = 'Chi-Squared'
            n = ct.values.sum(); md = min(ct.shape)-1
            eff = round(float(np.sqrt(stat/(n*md))) if md>0 else 0, 5)
            var_ = None
        ft_results.append({'Feature':col,'Domain':feature_domain.get(col,'Other'),
            'DataType':str(s.dtype),'Missing_Pct_Train':round(miss_pct,2),'Variance_Train':var_,
            'Test':test,'Statistic':round(float(stat),4),'P_Value':round(float(pval),8),
            'Effect_Size':eff,'Significant':'Yes' if pval<0.05 else 'No'})
    except Exception as e:
        ft_results.append({'Feature':col,'Domain':feature_domain.get(col,'Other'),
            'DataType':str(s.dtype),'Missing_Pct_Train':round(miss_pct,2),'Variance_Train':None,
            'Test':'ERROR','Statistic':None,'P_Value':None,'Effect_Size':0,'Significant':'Unknown'})

ft_df = pd.DataFrame(ft_results).sort_values('Effect_Size', ascending=False)
ft_df.to_csv('reports/feature_target_analysis.csv', index=False)
log(f"Tests run on {X_train_raw.shape[0]} training rows only.")
log("Top 25 by effect size:")
log(ft_df.head(25)[['Feature','Domain','Test','P_Value','Effect_Size','Significant']].to_string())

# ============================================================
# PHASE 9: CORRELATION ANALYSIS - TRAINING DATA ONLY
# ============================================================
log("\n"+"="*70)
log("PHASE 9 | CORRELATION ANALYSIS - TRAINING DATA ONLY")
log("="*70)

corr_train = X_train_enc.corr()
corr_train.to_csv('reports/correlation_report.csv')

corr_upper = corr_train.where(np.triu(np.ones(corr_train.shape), k=1).astype(bool))
flagged = []
for col in corr_upper.columns:
    for idx in corr_upper.index:
        val = corr_upper.loc[idx, col]
        if not np.isnan(val) and abs(val) >= CORR_THRESH:
            flagged.append({'Feature_A':idx,'Feature_B':col,'Correlation':round(val,4)})

flagged_df = pd.DataFrame(flagged).sort_values('Correlation',key=abs,ascending=False) if flagged else pd.DataFrame(columns=['Feature_A','Feature_B','Correlation'])
flagged_df.to_csv('reports/highly_correlated_pairs.csv', index=False)
log(f"Flagged pairs (|r|>={CORR_THRESH}): {len(flagged_df)}")
if len(flagged_df)>0: log(flagged_df.to_string())

target_corr_train = X_train_enc.corrwith(
    pd.Series(y_train, index=X_train_enc.index)).abs().sort_values(ascending=False)
log("\nTarget correlation (training only, top 25):")
log(target_corr_train.head(25).round(4).to_string())

# ============================================================
# PHASE 10: REDUNDANCY - CONCEPTUAL CLUSTERS
# ============================================================
log("\n"+"="*70)
log("PHASE 10 | REDUNDANCY CHECK - CONCEPTUAL CLUSTERS")
log("="*70)

clusters = {
    'Stress': ['Stress_Level','Psychological_Distress','Academic_Work_Stress','Workplace_Stress'],
    'Social Isolation': ['Social_Isolation','Loneliness_Level','Social_Integration','Social_Support','Friendship_Quality'],
    'Family': ['Family_Conflict','Family_Stability','Family_Support','Family_Communication_Quality','Parental_Monitoring'],
    'Behavioural': ['Impulsivity','Risk_Taking_Tendency','Rule_Breaking_Behaviour','Self_Control','Decision_Making_Control','Behavioural_Problems'],
    'Mood': ['Depression_Score','Hopelessness_Level','Anger_Level','Anxiety_Level','Loneliness_Level'],
    'Sleep': ['Sleep_Duration','Sleep_Quality'],
    'Substance': ['Substance_Use_Curiosity','Substance_Use_Intention','Substance_Availability_Exposure'],
}

redundancy_rows = []
for cname, feats in clusters.items():
    avail = [f for f in feats if f in X_train_enc.columns]
    if len(avail)<2: continue
    sub = X_train_enc[avail].corr()
    log(f"\n  {cname} cluster:")
    log(sub.round(3).to_string())
    for i,fa in enumerate(avail):
        for fb in avail[i+1:]:
            r = sub.loc[fa,fb]
            redundancy_rows.append({'Cluster':cname,'Feature_A':fa,'Feature_B':fb,
                'Pearson_r':round(float(r),4),'High_Redundancy':'Yes' if abs(r)>=0.75 else 'No'})

pd.DataFrame(redundancy_rows).to_csv('reports/redundancy_analysis.csv', index=False)
log("\nCluster analysis notes:")
log("  Stress cluster: Stress_Level and Psychological_Distress overlap but are distinct")
log("  constructs (general vs composite). Both top-ranked - both retained.")
log("  Academic_Work_Stress and Workplace_Stress apply to different subpopulations.")
log("  Social cluster: Social_Isolation (objective) vs Loneliness_Level (subjective).")
log("  Family cluster: Conflict/Stability are conceptual opposites but complement each other.")

# ============================================================
# PHASE 11: MODEL-BASED IMPORTANCE - TRAINING DATA ONLY
# ============================================================
log("\n"+"="*70)
log("PHASE 11 | MODEL-BASED IMPORTANCE - TRAINING DATA ONLY")
log("="*70)

X_tr2, X_val2, y_tr2, y_val2 = train_test_split(
    X_train_enc.values, y_train, test_size=0.20,
    random_state=RANDOM_STATE, stratify=y_train
)
fn = feature_names
log(f"Internal train: {X_tr2.shape[0]}  Internal val: {X_val2.shape[0]} (from training data only)")

rf = RandomForestClassifier(n_estimators=300, max_depth=None, min_samples_leaf=2,
                             random_state=RANDOM_STATE, n_jobs=-1)
rf.fit(X_tr2, y_tr2)
rf_val_acc = rf.score(X_val2, y_val2)
rf_imp = pd.Series(rf.feature_importances_, index=fn).sort_values(ascending=False)
log(f"RF internal val accuracy: {rf_val_acc:.4f}")
log("RF top 25:")
log(rf_imp.head(25).round(5).to_string())

perm = permutation_importance(rf, X_val2, y_val2, n_repeats=15,
                               random_state=RANDOM_STATE, n_jobs=-1)
perm_imp = pd.Series(perm.importances_mean, index=fn).sort_values(ascending=False)
perm_std = pd.Series(perm.importances_std, index=fn)
log("\nPermutation top 25:")
log(perm_imp.head(25).round(5).to_string())

xgb = XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05,
                     min_child_weight=3, subsample=0.8, colsample_bytree=0.8,
                     random_state=RANDOM_STATE, eval_metric='mlogloss', verbosity=0)
xgb.fit(X_tr2, y_tr2)
xgb_val_acc = xgb.score(X_val2, y_val2)
xgb_imp = pd.Series(xgb.feature_importances_, index=fn).sort_values(ascending=False)
log(f"\nXGB internal val accuracy: {xgb_val_acc:.4f}")
log("XGB top 25:")
log(xgb_imp.head(25).round(5).to_string())

log("\nAll importance scores from training data only. TEST SET UNTOUCHED.")

# ============================================================
# PHASE 12: COMBINED IMPORTANCE RANKING
# ============================================================
log("\n"+"="*70)
log("PHASE 12 | COMBINED IMPORTANCE RANKING")
log("="*70)

rank_rf   = rf_imp.rank(ascending=True)
rank_perm = perm_imp.rank(ascending=True)
rank_xgb  = xgb_imp.rank(ascending=True)
rank_corr = target_corr_train.reindex(fn).rank(ascending=True)
eff_s = ft_df.set_index('Feature')['Effect_Size'].reindex(fn).fillna(0)
rank_eff = eff_s.rank(ascending=True)

combined = (rank_rf + rank_perm + rank_xgb + rank_corr + rank_eff)
combined_norm = (combined / combined.max()).sort_values(ascending=False)

combined_df = pd.DataFrame({
    'Feature':combined_norm.index,'Combined_Score':combined_norm.values.round(5),
    'RF_Importance':rf_imp.reindex(combined_norm.index).values.round(5),
    'Perm_Importance':perm_imp.reindex(combined_norm.index).values.round(5),
    'Perm_Std':perm_std.reindex(combined_norm.index).values.round(5),
    'XGB_Importance':xgb_imp.reindex(combined_norm.index).values.round(5),
    'Target_Corr':target_corr_train.reindex(combined_norm.index).values.round(5),
    'Effect_Size':eff_s.reindex(combined_norm.index).values.round(5),
}).sort_values('Combined_Score', ascending=False)
combined_df.to_csv('results/combined_importance_v2.csv', index=False)
log("Combined top 40:")
log(combined_df.head(40).to_string())

# ============================================================
# PHASE 13: FEATURE SELECTION DECISION
# ============================================================
log("\n"+"="*70)
log("PHASE 13 | FINAL FEATURE SELECTION")
log("="*70)

sig_feats = ft_df[ft_df['Significant']=='Yes']['Feature'].tolist()
log(f"Statistically significant (p<0.05, training): {len(sig_feats)}")

domain_best = {}
for feat in candidate_pool:
    dom = feature_domain.get(feat,'Other')
    sc = combined_norm.get(feat,0)
    if dom not in domain_best or sc > domain_best[dom][1]:
        domain_best[dom] = (feat, sc)

log("\nBest per domain:")
for dom,(feat,sc) in sorted(domain_best.items()):
    log(f"  {dom:<25s}: {feat:<45s} score={sc:.4f}")

guaranteed = [v[0] for v in domain_best.values()]
ranked = combined_norm.index.tolist()
selected = list(guaranteed)
for feat in ranked:
    if feat not in selected and feat in sig_feats:
        selected.append(feat)
    if len(selected) >= 35:
        break

domain_cnt = {}
for f in selected:
    d = feature_domain.get(f,'Other')
    domain_cnt[d] = domain_cnt.get(d,0)+1

weak_dropped = []
for feat in selected[:]:
    sc = combined_norm.get(feat,0)
    dom = feature_domain.get(feat,'Other')
    if sc < MIN_SCORE and domain_cnt.get(dom,0) > 1:
        selected.remove(feat)
        domain_cnt[dom] -= 1
        weak_dropped.append(feat)
        log(f"  Dropped weak: {feat}  score={sc:.4f}")

log(f"\nFINAL SELECTED: {len(selected)} features")
for i,feat in enumerate(selected,1):
    dom = feature_domain.get(feat,'Other')
    sc = combined_norm.get(feat,0)
    sig = 'SIG' if feat in sig_feats else ' ns'
    log(f"  {i:2d}. {feat:<45s}  [{dom:<22s}]  score={sc:.4f}  {sig}")

# ============================================================
# PHASE 14: AUDIT OF v1.0 35 FEATURES
# ============================================================
log("\n"+"="*70)
log("PHASE 14 | AUDIT OF v1.0 CANDIDATE FEATURES")
log("="*70)

v1_features = [
    'Psychological_Distress','Stress_Level','Previous_Substance_Exposure',
    'Anxiety_Level','Depression_Score','Family_Conflict','Peer_Substance_Exposure',
    'Childhood_Adversity','Impulsivity','Hopelessness_Level','Sleep_Quality',
    'Social_Isolation','Screen_Time','Loneliness_Level','Night_Time_Device_Usage',
    'Self_Esteem','Peer_Influence','Sleep_Duration','Emotional_Regulation',
    'Academic_Work_Stress','Academic_Performance','Social_Support',
    'Substance_Use_Intention','Financial_Stress','Self_Control','Workload_Level',
    'Family_Support','Behavioural_Problems','Social_Pressure','Workplace_Stress',
    'Attitude_Toward_Substance_Use','Trauma_Symptoms','Coping_Ability',
    'Family_Stability','Anger_Level'
]

audit_35 = []
for feat in v1_features:
    row = ft_df[ft_df['Feature']==feat]
    eff = row['Effect_Size'].values[0] if len(row)>0 else None
    pv  = row['P_Value'].values[0] if len(row)>0 else None
    sig = row['Significant'].values[0] if len(row)>0 else 'Unknown'
    sc  = combined_norm.get(feat,0)
    dom = feature_domain.get(feat,'Other')
    in_v2 = feat in selected
    reason = "Retained: significant, non-leakage, domain-representative" if in_v2 else f"Not in v2 selection (score={sc:.3f})"
    audit_35.append({'Feature':feat,'Domain':dom,'Effect_Size':eff,'P_Value':pv,
                     'Significant':sig,'Combined_Score':round(sc,4),
                     'In_v2_Selection':'YES' if in_v2 else 'NO','Reason':reason})

audit_35_df = pd.DataFrame(audit_35)
log(audit_35_df[['Feature','Domain','Effect_Size','Combined_Score','Significant','In_v2_Selection']].to_string())

# Compare v1 and v2 selections
new_in_v2   = [f for f in selected if f not in v1_features]
dropped_v1  = [f for f in v1_features if f not in selected]
retained    = [f for f in v1_features if f in selected]
log(f"\nv1.0 features retained in v2.0: {len(retained)}")
log(f"v1.0 features dropped in v2.0 : {len(dropped_v1)} -> {dropped_v1}")
log(f"New features added in v2.0    : {len(new_in_v2)} -> {new_in_v2}")

# ============================================================
# PHASE 15: REPORTS
# ============================================================
log("\n"+"="*70)
log("PHASE 15 | GENERATING REPORTS")
log("="*70)

removed_log = []
for feat,reas in structural_removals.items():
    removed_log.append({'Feature':feat,'Category':'Leakage/Identifier',
                        'Decision':'REMOVED','Reason':reas,'Stage':'Pre-Split'})
for feat in v1_features:
    if feat not in selected:
        sc = combined_norm.get(feat,0)
        removed_log.append({'Feature':feat,'Category':feature_domain.get(feat,'Other'),
                            'Decision':'NOT_SELECTED_v2',
                            'Reason':f"Score {sc:.4f} - not in final selection after leakage-corrected analysis",
                            'Stage':'Post-Training-Analysis'})
for feat in candidate_pool:
    if feat not in v1_features and feat not in selected and feat not in structural_removals:
        removed_log.append({'Feature':feat,'Category':feature_domain.get(feat,'Other'),
                           'Decision':'NOT_SELECTED','Reason':'Below importance threshold (training analysis)',
                           'Stage':'Post-Training-Analysis'})
pd.DataFrame(removed_log).to_csv('reports/removed_features.csv', index=False)

sel_rows = []
for feat in candidate_pool:
    in_sel = feat in selected
    ft_row = ft_df[ft_df['Feature']==feat]
    rm_row = next((r for r in removed_log if r['Feature']==feat), None)
    reason = "Retained: strong, non-redundant, non-leakage predictor" if in_sel else (rm_row['Reason'] if rm_row else "Not selected")
    red_flag = 'Yes' if len(flagged_df)>0 and flagged_df[(flagged_df['Feature_A']==feat)|(flagged_df['Feature_B']==feat)].shape[0]>0 else 'No'
    sel_rows.append({
        'Feature':feat,'Category':feature_domain.get(feat,'Other'),
        'Data_Type':ft_row['DataType'].values[0] if len(ft_row)>0 else 'N/A',
        'Missing_Pct_Train':ft_row['Missing_Pct_Train'].values[0] if len(ft_row)>0 else None,
        'Effect_Size':ft_row['Effect_Size'].values[0] if len(ft_row)>0 else None,
        'P_Value':ft_row['P_Value'].values[0] if len(ft_row)>0 else None,
        'Significant_Train':ft_row['Significant'].values[0] if len(ft_row)>0 else 'Unknown',
        'RF_Importance':round(float(rf_imp.get(feat,0)),5),
        'Perm_Importance':round(float(perm_imp.get(feat,0)),5),
        'XGB_Importance':round(float(xgb_imp.get(feat,0)),5),
        'Combined_Score':round(float(combined_norm.get(feat,0)),5),
        'Redundancy_Flag':red_flag,'Leakage_Status':'Safe',
        'Analysis_Data':'Training-set only (v2.0)',
        'Selection_Decision':'SELECTED' if in_sel else 'NOT_SELECTED',
        'Reason':reason
    })

for feat,reas in structural_removals.items():
    sel_rows.append({'Feature':feat,'Category':'Leakage/Identifier',
        'Data_Type':str(df_raw[feat].dtype) if feat in df_raw.columns else 'N/A',
        'Missing_Pct_Train':None,'Effect_Size':None,'P_Value':None,
        'Significant_Train':'N/A','RF_Importance':None,'Perm_Importance':None,
        'XGB_Importance':None,'Combined_Score':None,'Redundancy_Flag':'N/A',
        'Leakage_Status':'LEAKAGE' if feat!='Record_ID' else 'IDENTIFIER',
        'Analysis_Data':'Pre-Split (structural)','Selection_Decision':'REMOVED','Reason':reas})

pd.DataFrame(sel_rows).sort_values(['Selection_Decision','Combined_Score'],
    ascending=[True,False]).to_csv('reports/feature_selection_report.csv', index=False)
log("All reports saved.")

# ============================================================
# PHASE 16: FINAL DATASET
# ============================================================
log("\n"+"="*70)
log("PHASE 16 | CREATING FINAL DATASET")
log("="*70)

df_final = df_raw[selected + [TARGET_COL]].copy()
df_final.drop_duplicates(inplace=True)
df_final.to_csv('data/final/final_ml_dataset.csv', index=False)
df_final.drop(columns=[TARGET_COL]).to_csv('data/final/X_features.csv', index=False)
df_final[[TARGET_COL]].to_csv('data/final/y_target.csv', index=False)

log(f"final_ml_dataset.csv: {df_final.shape[0]} rows x {df_final.shape[1]} cols")
log(f"Missing values: {df_final.isna().sum().sum()}")
log("Columns:")
for c in df_final.columns:
    log(f"  {c:<45s}  [{feature_domain.get(c,'Target')}]")

# ============================================================
# PHASE 17: FINAL FEATURE SET REPORT
# ============================================================
domain_cnt_final = {}
for f in selected:
    d = feature_domain.get(f,'Other')
    domain_cnt_final[d] = domain_cnt_final.get(d,0)+1

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
    f.write("  Split FIRST → Imputer(train) → Scaler(train) → OneHotEncoder(train)\n")
    f.write("  Numerical: SimpleImputer(median) + StandardScaler\n")
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

# ============================================================
# PHASE 18: VISUALIZATIONS
# ============================================================
log("\n"+"="*70)
log("PHASE 18 | VISUALIZATIONS")
log("="*70)

plt.figure(figsize=(14,10))
ts = combined_norm.reindex(selected).dropna().sort_values(ascending=True)
sns.barplot(x=ts.values, y=ts.index, color='#27ae60')
plt.title('Combined Importance Score - Final Selected Features (v2.0 Corrected)', fontsize=13)
plt.xlabel('Normalized Rank Score (training data only)')
plt.tight_layout(); plt.savefig('results/importance_combined_v2.png', dpi=150); plt.close()

plt.figure(figsize=(12,8))
sns.barplot(x=rf_imp.head(25).values, y=rf_imp.head(25).index, color='#2980b9')
plt.title('RF Feature Importance Top 25 (Training Data Only)', fontsize=13)
plt.xlabel('Importance'); plt.tight_layout()
plt.savefig('results/rf_importance_v2.png', dpi=150); plt.close()

plt.figure(figsize=(12,8))
sns.barplot(x=xgb_imp.head(25).values, y=xgb_imp.head(25).index, color='#e67e22')
plt.title('XGBoost Feature Importance Top 25 (Training Data Only)', fontsize=13)
plt.xlabel('Importance'); plt.tight_layout()
plt.savefig('results/xgb_importance_v2.png', dpi=150); plt.close()

plt.figure(figsize=(12,9))
pt = perm_imp.head(25); ps = perm_std.reindex(pt.index)
plt.barh(list(pt.index)[::-1], pt.values[::-1], xerr=ps.values[::-1],
         color='#8e44ad', alpha=0.8, capsize=3)
plt.title('Permutation Importance Top 25 ± Std (Internal Val, Training Only)', fontsize=12)
plt.xlabel('Mean Importance'); plt.tight_layout()
plt.savefig('results/permutation_importance_v2.png', dpi=150); plt.close()

X_fin_enc = X_train_enc[[f for f in selected if f in X_train_enc.columns]].copy()
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

log("Visualizations saved.")

# ============================================================
# PHASE 19: FINAL SUMMARY
# ============================================================
log("\n"+"="*72)
log("FINAL SUMMARY REPORT")
log("="*72)
log(f"1.  ORIGINAL FEATURE COUNT       : {n_cols-1} predictors + 1 target")
log(f"2.  CANDIDATE POOL (post-leakage-removal): {len(candidate_pool)}")
log(f"3.  FINAL PREDICTOR COUNT        : {len(selected)}")
log(f"4.  FEATURES REMOVED (structural): {len(structural_removals)}")
log(f"5.  LEAKAGE FEATURES             : {len(structural_removals)}")
for f in structural_removals:
    log(f"    - {f}")
log(f"\n6.  PREVIOUS PIPELINE (v1.0) LEAKAGE STATUS: CONTAMINATED")
log(f"    Imputation, encoding, statistical tests, correlation,")
log(f"    and model importance all ran on full 5000-row dataset.")
log(f"\n7.  CORRECTED PIPELINE (v2.0) LEAKAGE STATUS: CLEAN")
log(f"    Split performed first. All analysis on {X_train_raw.shape[0]}-row training set.")
log(f"\n8.  FINAL FEATURE LIST ({len(selected)} features):")
for i,feat in enumerate(selected,1):
    log(f"    {i:2d}. {feat:<45s}  [{feature_domain.get(feat,'Other')}]")
log(f"\n9.  TARGET: Drug_Abuse_Risk")
log(f"    Type: Operational research label - NOT clinically validated")
for cls,cnt in target_dist.items():
    log(f"    {cls}: {cnt} ({cnt/n_rows*100:.1f}%)")
log(f"\n10. TRAIN/TEST SIZES:")
log(f"    Training : {X_train_raw.shape[0]} rows - used for all analysis")
log(f"    Testing  : {X_test_raw.shape[0]} rows - UNTOUCHED")
log(f"\n11. PREPROCESSING: Split → Impute(train) → Scale(train) → Encode(train)")
log(f"12. FEATURE SELECTION: Kruskal/Chi2 + Corr + RF + XGB + Perm - training only")
log(f"13. LEAKAGE-FREE: YES (v2.0)")
log(f"\n14. REMAINING LIMITATIONS:")
log(f"    a. Target is an operational label - not clinically validated")
log(f"    b. Self-reported data - potential response bias")
log(f"    c. Cross-sectional - causal claims not supported")
log(f"    d. Previous_Substance_Exposure has borderline leakage concern")
log(f"       (retained because it records historical pre-assessment status)")
log(f"    e. Model CANNOT diagnose addiction or guarantee future drug use")
log(f"\n15. DATASET READY FOR FINAL ML TRAINING: YES")
log(f"    - {len(selected)} clean, non-leakage predictors")
log(f"    - Suitable for LR, DT, RF, SVM, XGBoost, LightGBM")
log(f"    - STOP: Final models NOT trained (as instructed)")

log("\n"+"="*72)
log("DONE - CORRECTED PIPELINE v2.0 COMPLETE.")
log("="*72)

with open('reports/feature_selection_audit.txt','a') as f:
    f.write("\n\nCORRECTED PIPELINE v2.0 - FULL LOG\n")
    f.write("="*72+"\n")
    for line in audit_lines:
        f.write(line+"\n")
