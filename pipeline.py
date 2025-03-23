import pandas as pd
from catboost import CatBoostClassifier
import matplotlib.pyplot as plt

# Load data
input_file_path = "/Users/omrijan/Downloads/data_scientist_home_assignment.csv"
df = pd.read_csv(input_file_path)

# Step 1: Convert EffectiveDate to week number (1–52)
df['EffectiveDate'] = pd.to_datetime(df['EffectiveDate'])
df['WeekNum'] = df['EffectiveDate'].dt.isocalendar().week

# Step 2: Create unique datapoint ID
df['DatapointID'] = df['MemberID'].astype(str) + "_W" + df['WeekNum'].astype(str)

# Step 3: Create EffectiveOutcome (True if contacted or left message)
df['EffectiveOutcome'] = df['OutReachOutcome'].isin(['contacted', 'left message'])
# TODO: explain why you binarized this way

# Step 3.5: Add BMI feature
df['BMI'] = df['Weight'] / (df['Height'] ** 2)

# Step 4: One-hot encode categorical variables (excluding OutReachOutcome)
categorical_cols = ['Gender', 'DEXAScanResult']
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# Define feature columns (exclude label and known identifiers)
excluded_cols = ['MemberID', 'EffectiveDate', 'WeekNum', 'DatapointID', 'OutReachOutcome', 'EffectiveOutcome', 'ChurnIn30Days']
feature_cols = [col for col in df.columns if col not in excluded_cols]

# Step 5: Temporal train-test split (80-20 based on week number)
train_weeks = sorted(df['WeekNum'].unique())[:int(0.8 * 52)]  # Weeks 1 to 41
test_weeks = sorted(df['WeekNum'].unique())[int(0.8 * 52):]   # Weeks 42 to 52
# TODO: remove pot. leaks by member id (memebers in test that are within 4 weeks of train datapoint)

train_df = df[df['WeekNum'].isin(train_weeks)].copy()
test_df = df[df['WeekNum'].isin(test_weeks)].copy()

print("Preprocessing complete.")
print(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")

########################################################################################################################

# Train compliance model only on people assigned treatment (Z = 1)
df_z1 = train_df[train_df['OutReach'] == 1]
X_comp = df_z1[feature_cols]
y_comp = df_z1['EffectiveOutcome'].astype(int)

# Train CatBoost (no scaling needed)
compliance_model = CatBoostClassifier(
    iterations=300,
    learning_rate=0.05,
    depth=4,
    random_seed=42
)
compliance_model.fit(X_comp, y_comp)

# Predict compliance probabilities for full train/test sets
train_df['ComplianceProb'] = compliance_model.predict_proba(train_df[feature_cols])[:, 1]
test_df['ComplianceProb'] = compliance_model.predict_proba(test_df[feature_cols])[:, 1]
#TODO: come back to this and add regularization and maybe early stopping or checking overfit somehow.
# Best thing here is temporal based CV

print("✅ Compliance model updated to CatBoost.")

########################################################################################################################
# Modelling

# Train treated model (T=1)
treated_df = train_df[train_df['OutReach'] == 1]
X_treated = treated_df[feature_cols]
y_treated = treated_df['ChurnIn30Days']

model_treated = CatBoostClassifier(
    iterations=300,
    learning_rate=0.05,
    depth=4,
    random_seed=42
)
model_treated.fit(X_treated, y_treated)

# Train control model (T=0)
control_df = train_df[train_df['OutReach'] == 0]
X_control = control_df[feature_cols]
y_control = control_df['ChurnIn30Days']

model_control = CatBoostClassifier(
    iterations=300,
    learning_rate=0.05,
    depth=4,
    random_seed=42
)
model_control.fit(X_control, y_control)

# Predict potential outcomes for train set
train_df['PredChurn_Treated'] = model_treated.predict_proba(train_df[feature_cols])[:, 1]
train_df['PredChurn_Control'] = model_control.predict_proba(train_df[feature_cols])[:, 1]
train_df['CATE'] = train_df['PredChurn_Treated'] - train_df['PredChurn_Control']
train_df['ranking_score'] = -train_df['CATE'] * train_df['ComplianceProb']
train_df = train_df.sort_values('ranking_score', ascending=True)

# Predict potential outcomes for test set
test_df['PredChurn_Treated'] = model_treated.predict_proba(test_df[feature_cols])[:, 1]
test_df['PredChurn_Control'] = model_control.predict_proba(test_df[feature_cols])[:, 1]
test_df['CATE'] = test_df['PredChurn_Treated'] - test_df['PredChurn_Control']
test_df['ranking_score'] = - test_df['CATE'] * test_df['ComplianceProb']
test_df = test_df.sort_values('ranking_score', ascending=True)

print("✅ T-learner complete: predicted outcomes, CATE, and ranking score added.")

#######################################################################################################################
# Evaluation

def build_qini_table_v2(df, score_col='ranking_score', treatment_col='OutReach',
                        outcome_col='ChurnIn30Days', n_bins=100, max_bin=None):
    df = df.copy()

    # 1. Sort by model score (descending)
    df = df.sort_values(by=score_col, ascending=False).reset_index(drop=True)

    # 2. Assign percentiles by rank
    df['percentile'] = pd.qcut(df.index, q=n_bins, labels=False)

    # 3. Compute total uplift over entire dataset (used for baseline)
    treated = df[df[treatment_col] == 1]
    control = df[df[treatment_col] == 0]

    rate_treated = treated[outcome_col].mean()
    rate_control = control[outcome_col].mean()
    total_treated = len(treated)

    uplift_total = (rate_control - rate_treated) * total_treated

    # 4. For each bin, compute number of treated/control + churns
    bin_stats = df.groupby('percentile').apply(lambda g: pd.Series({
        'n_total': len(g),
        'n_treated': g[treatment_col].sum(),
        'n_control': (1 - g[treatment_col]).sum(),
        'churns_treated': g.loc[g[treatment_col] == 1, outcome_col].sum(),
        'churns_control': g.loc[g[treatment_col] == 0, outcome_col].sum()
    })).reset_index()

    # 5. Compute cumulative sums
    bin_stats['cum_treated'] = bin_stats['n_treated'].cumsum()
    bin_stats['cum_control'] = bin_stats['n_control'].cumsum()
    bin_stats['cum_churns_treated'] = bin_stats['churns_treated'].cumsum()
    bin_stats['cum_churns_control'] = bin_stats['churns_control'].cumsum()
    bin_stats['cum_total'] = bin_stats['n_total'].cumsum()

    # 6. Compute cumulative rates
    bin_stats['cum_rate_treated'] = bin_stats['cum_churns_treated'] / bin_stats['cum_treated'].replace(0, 1)
    bin_stats['cum_rate_control'] = bin_stats['cum_churns_control'] / bin_stats['cum_control'].replace(0, 1)

    # 7. Compute cumulative model-based uplift (more stable)
    bin_stats['cum_uplift'] = (bin_stats['cum_rate_control'] - bin_stats['cum_rate_treated']) * bin_stats['cum_treated']

    # 8. Compute cumulative random baseline uplift (linear curve)
    bin_stats['percent_population'] = 100 * bin_stats['cum_total'] / bin_stats['cum_total'].iloc[-1]
    bin_stats['cum_random_uplift'] = bin_stats['percent_population'] / 100 * uplift_total

    # Limit to bins 0 to max_bin if provided
    if max_bin is not None:
        bin_stats = bin_stats[bin_stats['percentile'] <= max_bin].copy()

    return bin_stats


qini_df = build_qini_table_v2(test_df, n_bins=500, max_bin=150)

# Function to plot the Qini table and save the figure (without showing it)
def plot_qini_table(qini_df, output_file='qini_plot.png'):
    plt.figure(figsize=(8, 6))
    plt.plot(qini_df['percent_population'], qini_df['cum_uplift'], label='Model Qini')
    plt.plot(qini_df['percent_population'], qini_df['cum_random_uplift'], linestyle='--', label='Random Baseline')
    plt.xlabel('Percent Population')
    plt.ylabel('Cumulative Uplift')
    plt.title('Qini Curve')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()

plot_qini_table(qini_df, output_file="/Users/omrijan/Downloads/qini.png")
print("Qini plot saved.")
