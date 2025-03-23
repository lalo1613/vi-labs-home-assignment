import pandas as pd

input_file_path = "/Users/omrijan/Downloads/data_scientist_home_assignment.csv"
df = pd.read_csv(input_file_path)


from scipy.stats import chi2_contingency

# Construct a contingency table from the binary variables
contingency_table = pd.crosstab(df['OutReach'], df['ChurnIn30Days'])
print("Contingency Table:")
print(contingency_table)

# Perform the chi-square test for independence
chi2, p_value, dof, expected = chi2_contingency(contingency_table)

print("\nChi-Square Test for Independence")
print("Chi2 statistic:", chi2)
print("Degrees of Freedom:", dof)
print("p-value:", p_value)

########################################################################################################################
import statsmodels.formula.api as smf
from scipy import stats

# For clarity, rename columns to valid Python identifiers.
df_reg = df.copy()
df_reg = df_reg.rename(columns={
    'AppUsage (hrs)': 'AppUsage',
    'GymVisitsLast2W': 'GymVisits2W',
    'GymVisitsLast6W': 'GymVisits6W',
    'GymVisitsLast12W': 'GymVisits12W'
})

# Define the logistic regression formula.
formula = 'ChurnIn30Days ~ OutReach + Age + AppUsage + GymVisits2W + GymVisits6W + GymVisits12W'

# Fit the logistic regression model.
logit_model = smf.logit(formula, data=df_reg).fit()
print("\nLogistic Regression Summary (including Wald tests):")
print(logit_model.summary())

########################################################################################################################

df['EffectiveOutcome'] = df['OutReachOutcome'].apply(lambda x: 1 if x in ['contacted', 'left message'] else 0)

# Build a contingency table for EffectiveOutcome vs. ChurnIn30Days
contingency_table_outcome = pd.crosstab(df['EffectiveOutcome'], df['ChurnIn30Days'])
print("Contingency Table for Effective Outcome vs. Churn:")
print(contingency_table_outcome)

# Run chi-square test
chi2, p_value, dof, expected = chi2_contingency(contingency_table_outcome)
print("\nChi-Square Test for Effective Outcome vs. Churn")
print("Chi2 statistic:", chi2)
print("Degrees of Freedom:", dof)
print("p-value:", p_value)

########################################################################################################################

df_reg = df.copy()
df_reg = df_reg.rename(columns={
    'AppUsage (hrs)': 'AppUsage',
    'GymVisitsLast2W': 'GymVisits2W',
    'GymVisitsLast6W': 'GymVisits6W',
    'GymVisitsLast12W': 'GymVisits12W'
})

# Assume EffectiveOutcome is binary (1 if successful, 0 otherwise) as defined above.
# Include Outreach (whether attempted) and EffectiveOutcome as separate variables.
formula = 'ChurnIn30Days ~ OutReach + EffectiveOutcome + Age + AppUsage + GymVisits2W + GymVisits6W + GymVisits12W'
logit_model = smf.logit(formula, data=df_reg).fit()
print("\nLogistic Regression Summary (including EffectiveOutcome):")
print(logit_model.summary())

########################################################################################################################

model1 = smf.logit('ChurnIn30Days ~ OutReach + Age + AppUsage + GymVisits2W + GymVisits6W + GymVisits12W', data=df_reg).fit()
print("Model without EffectiveOutcome:")
print(model1.summary())

# Model 2: Including EffectiveOutcome
model2 = smf.logit('ChurnIn30Days ~ OutReach + EffectiveOutcome + Age + AppUsage + GymVisits2W + GymVisits6W + GymVisits12W', data=df_reg).fit()
print("\nModel with EffectiveOutcome:")
print(model2.summary())

# Likelihood Ratio Test comparing Model 2 and Model 1
lr_stat = 2 * (model2.llf - model1.llf)
df_diff = model2.df_model - model1.df_model  # degrees of freedom difference
p_value = stats.chi2.sf(lr_stat, df_diff)

print("\nLikelihood Ratio Test")
print("LR statistic:", lr_stat)
print("Degrees of freedom:", df_diff)
print("p-value:", p_value)

########################################################################################################################


import statsmodels.api as sm
from sklearn.neighbors import NearestNeighbors
import numpy as np
from statsmodels.stats.proportion import proportions_ztest

# Define covariates for propensity score estimation.
covariates = ['Age', 'AppUsage', 'GymVisitsLast2W', 'GymVisitsLast6W', 'GymVisitsLast12W']

# Ensure there are no missing values in the covariates and treatment/outcome.
df_ps = df.dropna(subset=covariates + ['EffectiveOutcome', 'ChurnIn30Days'])
df[covariates + ['EffectiveOutcome', 'ChurnIn30Days']].isna().mean()

# Create the design matrix for propensity score estimation and add a constant.
X_ps = df_ps[covariates]
X_ps = sm.add_constant(X_ps)
y_ps = df_ps['EffectiveOutcome']  # Treatment: Effective outcome (1) vs. ineffective (0)

# Fit logistic regression to estimate propensity scores.
ps_model = sm.Logit(y_ps, X_ps).fit(disp=False)
df_ps['propensity_score'] = ps_model.predict(X_ps)

# Split data into treated (effective outcome) and control groups.
treated = df_ps[df_ps['EffectiveOutcome'] == 1].copy()
control = df_ps[df_ps['EffectiveOutcome'] == 0].copy()

# Perform nearest neighbor matching on the propensity score.
nn = NearestNeighbors(n_neighbors=1)
nn.fit(control[['propensity_score']])
distances, indices = nn.kneighbors(treated[['propensity_score']])

# Get matched control indices
treated['matched_index'] = indices.flatten()
matched_controls = control.iloc[treated['matched_index']].copy()
matched_controls.index = treated.index  # Align indices for easier comparison

# Compare churn rates in the matched sample.
treated_churns = treated['ChurnIn30Days'].sum()
control_churns = matched_controls['ChurnIn30Days'].sum()
n_treated = len(treated)
n_control = len(matched_controls)

# Two-proportion z-test
count = np.array([treated_churns, control_churns])
nobs = np.array([n_treated, n_control])
z_stat, pval = proportions_ztest(count, nobs)

print("\nTwo-Proportion z-test after Propensity Score Matching on EffectiveOutcome")
print("Treated (Effective Outcome) churns: {} out of {}".format(treated_churns, n_treated))
print("Control (Ineffective Outcome) churns: {} out of {}".format(control_churns, n_control))
print("z-statistic:", z_stat)
print("p-value:", pval)

########################################################################################################################

import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.metrics import roc_auc_score

# Logistic regression: predict OutReach using the same X covariates
formula_iv_check = 'OutReach ~ Age + AppUsage + GymVisits2W + GymVisits6W + GymVisits12W'
iv_logit_model = smf.logit(formula_iv_check, data=df_reg).fit()

print("\nLogistic Regression Summary (OutReach ~ X):")
print(iv_logit_model.summary())

# Get predicted probabilities
df_reg['Z_hat'] = iv_logit_model.predict()

# AUC score: how well X predicts Z (OutReach)
auc = roc_auc_score(df_reg['OutReach'], df_reg['Z_hat'])
print(f"\nAUC for predicting OutReach from X: {auc:.3f}")

# McFadden's pseudo-R²
pseudo_r2 = 1 - iv_logit_model.llf / iv_logit_model.llnull
print(f"Pseudo-R² for OutReach ~ X: {pseudo_r2:.3f}")
