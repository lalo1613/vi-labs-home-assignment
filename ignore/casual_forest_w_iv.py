import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator
from sklearn.tree import DecisionTreeRegressor
from sklearn.utils import resample
from joblib import Parallel, delayed

# ===== HonestTreeIV and CausalForestIV =====

class HonestTreeIV:
    def __init__(self, max_depth=5, min_samples_leaf=10, random_state=None):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state

    def fit(self, X, Y, T, Z):
        n = X.shape[0]
        idx_split = np.random.choice([0, 1], size=n)
        X_split, X_est = X[idx_split == 0], X[idx_split == 1]
        Y_est, T_est, Z_est = Y[idx_split == 1], T[idx_split == 1], Z[idx_split == 1]

        self.tree = DecisionTreeRegressor(
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.random_state
        )
        self.tree.fit(X_split, T[idx_split == 0])

        self.leaf_ids_ = self.tree.apply(X_est)
        self.leaf_effects_ = {}

        for leaf in np.unique(self.leaf_ids_):
            idx = self.leaf_ids_ == leaf
            if np.sum(idx) < self.min_samples_leaf:
                continue

            y1 = Y_est[idx & (Z_est == 1)]
            y0 = Y_est[idx & (Z_est == 0)]
            t1 = T_est[idx & (Z_est == 1)]
            t0 = T_est[idx & (Z_est == 0)]

            if len(t1) == 0 or len(t0) == 0:
                self.leaf_effects_[leaf] = 0.0
                continue

            delta_y = y1.mean() - y0.mean()
            delta_t = t1.mean() - t0.mean()
            effect = delta_y / delta_t if np.abs(delta_t) > 1e-6 else 0.0
            self.leaf_effects_[leaf] = effect

    def predict(self, X):
        leaves = self.tree.apply(X)
        return np.array([self.leaf_effects_.get(leaf, 0.0) for leaf in leaves])


class CausalForestIV(BaseEstimator):
    def __init__(self, n_estimators=100, max_depth=5, min_samples_leaf=10, bootstrap=True, n_jobs=1, random_state=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.bootstrap = bootstrap
        self.n_jobs = n_jobs
        self.random_state = random_state

    def fit(self, Y, T, Z, X):
        rng = np.random.RandomState(self.random_state)

        def build_tree(i):
            indices = np.arange(X.shape[0])
            if self.bootstrap:
                indices = resample(indices, random_state=rng.randint(1e6))
            tree = HonestTreeIV(
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                random_state=rng.randint(1e6)
            )
            tree.fit(X[indices], Y[indices], T[indices], Z[indices])
            return tree

        self.trees_ = Parallel(n_jobs=self.n_jobs)(
            delayed(build_tree)(i) for i in range(self.n_estimators)
        )
        return self

    def effect(self, X):
        preds = np.array([tree.predict(X) for tree in self.trees_])
        return preds.mean(axis=0)

    def effect_interval(self, X, alpha=0.05):
        preds = np.array([tree.predict(X) for tree in self.trees_])
        lower = np.percentile(preds, 100 * (alpha / 2), axis=0)
        upper = np.percentile(preds, 100 * (1 - alpha / 2), axis=0)
        return lower, upper

# ===== Simulate Data =====

input_file_path = "/Users/omrijan/Downloads/data_scientist_home_assignment.csv"
df = pd.read_csv(input_file_path)

# Define T, Z, Y
df['T'] = df['OutReachOutcome'].isin(['contacted', 'left message']).astype(int)
df['Z'] = df['OutReach']
df['Y'] = df['ChurnIn30Days']

# ===== Features and Preprocessing =====

feature_cols = [
    'Age', 'Gender', 'Height', 'Weight', 'DEXAScanResult',
    'AppUsage', 'GymVisitsLast2W', 'GymVisitsLast6W', 'GymVisitsLast12W'
]

X = df[feature_cols]
Y = df['Y'].values
T = df['T'].values
Z = df['Z'].values

numeric_features = ['Age', 'Height', 'Weight', 'AppUsage', 'GymVisitsLast2W', 'GymVisitsLast6W', 'GymVisitsLast12W']
categorical_features = ['Gender', 'DEXAScanResult']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', SimpleImputer(strategy='mean'), numeric_features),
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(drop='first'))
        ]), categorical_features)
    ]
)

X_transformed = preprocessor.fit_transform(X)

# ===== Train/Test Split =====

X_train, X_test, Y_train, Y_test, T_train, T_test, Z_train, Z_test = train_test_split(
    X_transformed, Y, T, Z, test_size=0.2, random_state=42
)

# ===== Fit Causal Forest IV =====

forest = CausalForestIV(
    n_estimators=100,
    max_depth=6,
    min_samples_leaf=20,
    bootstrap=True,
    n_jobs=-1,
    random_state=42
)
forest.fit(Y=Y_train, T=T_train, Z=Z_train, X=X_train)

# ===== Estimate CATE =====

tau_hat = forest.effect(X_test)
tau_lower, tau_upper = forest.effect_interval(X_test)

# ===== Output Results =====

summary_df = pd.DataFrame({
    'Effect Estimate': tau_hat,
    '95% CI Lower': tau_lower,
    '95% CI Upper': tau_upper,
    'Actual Churn': Y_test,
    'Treatment Received': T_test,
    'Assigned Treatment': Z_test
})

summary_df['Effect Estimate'].describe()
summary_df = summary_df.sort_values('Effect Estimate')
print(summary_df.head(10))

