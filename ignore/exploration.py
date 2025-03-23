import pandas as pd

input_file_path = "/Users/omrijan/Downloads/data_scientist_home_assignment.csv"
df = pd.read_csv(input_file_path)

dtypes = df.dtypes
df.isna().mean()
cat_cols = dtypes[dtypes == object].index.tolist()
numeric_cols = dtypes[dtypes != object].index.tolist()

for cat_col in cat_cols:
    print(cat_col + f" unique vals: {df[[cat_col]].nunique().values[0]}")
    print(cat_col + " count dist:")
    print(df[[cat_col]].value_counts().describe())

for num_col in numeric_cols:
    print(num_col + " dist:")
    print(df[[num_col]].describe())


"""
Thoughts so far:
- Data is 100 points per day for the same day of the week for a year = 5200 datapoints.
- No missing vals in my data, but it's been pointed out that I should have code that can deal with it. 
- Could simulate missing in some way. But that'd assume the mechanism (and randomness/lack-thereof). Best to explain this point thoroughly either way.
- I guess best approach would be one where we rank likelihood of churn, that way should work for any N value.
- Members do repeat, and there is a chance that there's leakage between datapoints (and potentially datasets) if we don't cover that.
- Another cool idea is estimating also the std. dev. with CatBoost, that way I can for instance rank pot. callers by LB CI they'll stay (we can check expectation here).
    This would NOT maximize expected kept customers, but might be a minmax on potential churns
- Another cool idea is to look into how valuable a call might be. They've already looked forward to say who churns so we can learn from this, for what set of Xs does a call work
- Worth thinking about a feature that cumulates calls in the past N days, and/or learning whether call values diminish
- Not sure how to use the "operator has called" feature given that the resulting behavior from the model saying high churn potential is a call... maybe this is overkill but we do like showing off
- Other than that, EDA and a nice pipeline will probably do it. Leaving it for tomorrow
"""

"""
Plan:
- Tests re approach (conclusion is CF w. IV)

- translate date to days week #
- since forward-look is 30 days, train-test should avoid same member within 30 days
- do we have to split by time if no member is in both sets? I thought only if concept shift, but really dist of clients changing is also enough reason to cause bias

"""

df["OutReach"].mean()
df.groupby("ChurnIn30Days")["OutReach"].value_counts(normalize=True)
df.groupby("OutReach")["ChurnIn30Days"].value_counts(normalize=True)
df.groupby("OutReachOutcome", dropna=False)["ChurnIn30Days"].value_counts(normalize=True)


########################################################################################################################

# Install required packages first (in your terminal or notebook)
# pip install econml scikit-learn pandas numpy

import pandas as pd
import numpy as np
from econml.iv.forest import CausalForestIV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# ========== STEP 1: Load and define data ==========

# Replace this with your actual dataset
input_file_path = "/Users/omrijan/Downloads/data_scientist_home_assignment.csv"
df = pd.read_csv(input_file_path)

# ========== STEP 2: Define causal variables ==========

# Binary treatment: 1 if member was reached (answered or left message), else 0
df['T'] = df['OutReachOutcome'].isin(['contacted', 'left message']).astype(int)

# Binary instrument: 1 if outreach attempted, else 0
df['Z'] = df['OutReach']

# Outcome variable
df['Y'] = df['ChurnIn30Days']

# Feature columns (excluding outcome, treatment, instrument)
features = [
    'Age', 'Gender', 'Height', 'Weight', 'DEXAScanResult',
    'AppUsage', 'GymVisitsLast2W', 'GymVisitsLast6W', 'GymVisitsLast12W'
]

X = df[features]
Y = df['Y'].values
T = df['T'].values
Z = df['Z'].values

# ========== STEP 3: Preprocess features ==========

numeric_features = ['Age', 'Height', 'Weight', 'AppUsage', 'GymVisitsLast2W', 'GymVisitsLast6W', 'GymVisitsLast12W']
categorical_features = ['Gender', 'DEXAScanResult']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', SimpleImputer(strategy='mean'), numeric_features),
        ('cat', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(drop='first'))
        ]), categorical_features)
    ]
)

X_transformed = preprocessor.fit_transform(X)

# Optional: split for testing
X_train, X_test, Y_train, Y_test, T_train, T_test, Z_train, Z_test = train_test_split(
    X_transformed, Y, T, Z, test_size=0.2, random_state=42
)

# ========== STEP 4: Fit causal forest IV ==========

model = CausalForestIV(
    n_estimators=500,
    min_samples_leaf=20,
    max_depth=10,
    verbose=1
)

model.fit(Y_train, T_train, Z_train, X=X_train)

# ========== STEP 5: Estimate treatment effects (CATE) ==========

cate_preds = model.effect(X_test)
cate_intervals = model.effect_interval(X_test)

# ========== STEP 6: Select top-N clients to call ==========

N = 20  # top N members to call
top_n_idx = np.argsort(-cate_preds)[:N]  # highest uplift
top_n_df = pd.DataFrame({
    'CATE Estimate': cate_preds[top_n_idx],
    '95% CI Lower': cate_intervals[0][top_n_idx],
    '95% CI Upper': cate_intervals[1][top_n_idx]
})

print("Top N clients to call (based on uplift):")
print(top_n_df)

# Estimate total expected reduction in churn:
expected_saved_clients = top_n_df['CATE Estimate'].sum()
print(f"\nExpected clients retained by calling these {N}: {expected_saved_clients:.2f}")
