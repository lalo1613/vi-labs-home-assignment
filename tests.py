from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

input_file_path = "/Users/omrijan/Downloads/data_scientist_home_assignment.csv"
pps_save_path = "/Users/omrijan/Downloads/pps_kde_graph.png"
pca_save_path = "/Users/omrijan/Downloads/pca1_kde_graph.png"

# Load data
df = pd.read_csv(input_file_path)

####
# monoticity test
####
# df["OutReach"].mean()
# df.groupby("ChurnIn30Days")["OutReach"].value_counts(normalize=True)
# df.groupby("OutReach")["ChurnIn30Days"].value_counts(normalize=True)
print(df.groupby("OutReachOutcome", dropna=False)["ChurnIn30Days"].value_counts(normalize=True))

####
# PPS test
####
# One-hot encode categorical variables (excluding OutReachOutcome)
categorical_cols = ['Gender', 'DEXAScanResult']
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# Drop columns that can't be used as features
excluded_cols = ['MemberID', 'EffectiveDate', 'OutReachOutcome', 'OutReach', 'ChurnIn30Days']  # 'DatapointID' isn't in the original CSV, just in script 1
feature_cols = [col for col in df.columns if col not in excluded_cols]

# Drop rows with missing values in selected features
df_features = df[feature_cols].dropna()

# Standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_features)

# Build propensity scores model
ps_model = LogisticRegression().fit(X_scaled, df['OutReach'])
df['pscore'] = ps_model.predict_proba(X_scaled)[:, 1]

# Plot density of propensity scores by treatment group
sns.kdeplot(data=df[df['OutReach'] == 1], x='pscore', label='Treated', fill=True, alpha=0.5)
sns.kdeplot(data=df[df['OutReach'] == 0], x='pscore', label='Untreated', fill=True, alpha=0.5)
plt.title("Propensity Score Overlap")
plt.xlabel("Estimated P(T=1 | X)")
plt.legend()
plt.tight_layout()
plt.savefig(pps_save_path, dpi=300)
plt.close()

