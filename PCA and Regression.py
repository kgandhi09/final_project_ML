# This was run using the Kaggle notebook to easily access the data and make importing/reading in the data more feasible
# as it is a large dataset
import pandas as pd
import lightgbm as lgb
import numpy as np
from sklearn.preprocessing import scale
from sklearn import model_selection
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Reading in stock prices csv
df = pd.read_csv('../input/jpx-tokyo-stock-exchange-prediction/supplemental_files/stock_prices.csv')
# link for gbm regression help: https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMRegressor.html
# also learned about gbm regression while looking at methods to use in final project for MA 4635
features = ['SecuritiesCode', 'Open', 'High', 'Low', 'Close', 'Volume', 'AdjustmentFactor', 'ExpectedDividend',
            'SupervisionFlag']
target = ['Target']

# creating light gbm model
feature_df = df.copy()
leaves = 15
for i in range(1, leaves):
    model = lgb.LGBMRegressor(n_estimators=200, num_leaves=2 ** i, min_child_samples=1, random_state=4342)
    i += 1
    print(model.fit(feature_df[features], feature_df[target]))
    print(model.score(feature_df[features], feature_df[target]))

# # Principal Component Analysis (PCA) & Linear Regression
pca_df = df.copy()
pca_df.corr()

features = pca_df[['SecuritiesCode', 'Open', 'High', 'Low', 'Close', 'Volume', 'AdjustmentFactor', 'ExpectedDividend',
                   'SupervisionFlag']]
target = pca_df[['Target']]
print(features.head())
print(target.head())

df = df.copy()
df = df[["Open", "High", "Low", "Close", "Volume", "Target"]]
print(df.head())
new_df = df.dropna()
new_df = df[0:200]
normalized_df = (new_df - new_df.mean()) / new_df.std()
print(normalized_df)
norm_df = normalized_df.dropna(axis=0)
print(norm_df)

# help for regression PCA https://www.statology.org/principal-components-regression-in-python/
X = norm_df[["Open", "High", "Low", "Close", "Volume"]]
y = norm_df[["Target"]]
norm_df = norm_df
pca = PCA()
X_reduced = pca.fit_transform(scale(X))

# Cross validation
cv = RepeatedKFold(n_splits=8, n_repeats=3, random_state=1)
regr = LinearRegression()
mse = []

# MSE w/ only intercept
score = -1 * model_selection.cross_val_score(regr,
                                             np.ones((len(X_reduced), 1)), y, cv=cv,
                                             scoring='neg_mean_squared_error').mean()
mse.append(score)

# MSE w/ Cross Validation
for i in np.arange(1, 6):
    score = -1 * model_selection.cross_val_score(regr,
                                                 X_reduced[:, :i], y, cv=cv, scoring='neg_mean_squared_error').mean()
    mse.append(score)

# Plot Cross Validation Results
plt.plot(mse)
plt.xlabel('# of Principal Components')
plt.ylabel('MSE')
plt.title('Target')

# Training and Testing Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Scaling Data
X_red_train = pca.fit_transform(scale(X_train))
X_red_test = pca.transform(scale(X_test))[:, :1]

# Training
regr = LinearRegression()
regr.fit(X_red_train[:, :1], y_train)

# RMSE Calculation
pred = regr.predict(X_red_test)
np.sqrt(mean_squared_error(y_test, pred))