import pandas as pd
import numpy as np
from sklearn.feature_selection import RFE
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn import svm
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data"

column_names = [
    "CRIM",
    "ZN",
    "INDUS",
    "CHAS",
    "NOX",
    "RM",
    "AGE",
    "DIS",
    "RAD",
    "TAX",
    "PTRATIO",
    "B",
    "LSTAT",
    "MEDV",
]

# MSE = error margin, the lower the better
# R2 = accuracy, the closer to one, the better

boston = pd.read_csv(url, header=None, delim_whitespace=True, names=column_names)

# Shuffle the rows of the DataFrame
shuffled_boston = boston.sample(frac=1, random_state=42)
shuffled_boston.reset_index(drop=True, inplace=True)

# Unshuffled
# X = boston.drop(columns=["MEDV"]).values
#Y = boston["MEDV"].values

# Shuffled
X = shuffled_boston.drop(columns=["MEDV"]).values
Y = shuffled_boston["MEDV"].values

cv = 10

print("\nlinear regression")
lin = LinearRegression()
scores = cross_val_score(lin, X, Y, cv=cv)
print("mean R2: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
predicted = cross_val_predict(lin, X, Y, cv=cv)
print("MSE: %0.2f" % mean_squared_error(Y, predicted))

print("\nridge regression")
ridge = Ridge(alpha=1.0)
scores = cross_val_score(ridge, X, Y, cv=cv)
print("mean R2: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
predicted = cross_val_predict(ridge, X, Y, cv=cv)
print("MSE: %0.2f" % mean_squared_error(Y, predicted))

print("\nlasso regression")
lasso = Lasso(alpha=0.1)
scores = cross_val_score(lasso, X, Y, cv=cv)
print("mean R2: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
predicted = cross_val_predict(lasso, X, Y, cv=cv)
print("MSE: %0.2f" % mean_squared_error(Y, predicted))

print("\ndecision tree regression")
tree = DecisionTreeRegressor(random_state=0)
scores = cross_val_score(tree, X, Y, cv=cv)
print("mean R2: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
predicted = cross_val_predict(tree, X, Y, cv=cv)
print("MSE: %0.2f" % mean_squared_error(Y, predicted))

print("\nrandom forest regression")
forest = RandomForestRegressor(
    n_estimators=50, max_depth=None, min_samples_split=2, random_state=0
)
scores = cross_val_score(forest, X, Y, cv=cv)
print("mean R2: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
predicted = cross_val_predict(forest, X, Y, cv=cv)
print("MSE: %0.2f" % mean_squared_error(Y, predicted))

print("\nlinear support vector machine")
svm_lin = svm.SVR(epsilon=0.2, kernel="linear", C=1)
scores = cross_val_score(svm_lin, X, Y, cv=cv)
print("mean R2: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
predicted = cross_val_predict(svm_lin, X, Y, cv=cv)
print("MSE: %0.2f" % mean_squared_error(Y, predicted))

print("\nsupport vector machine rbf")
clf = svm.SVR(epsilon=0.2, kernel="rbf", C=1.0)
scores = cross_val_score(clf, X, Y, cv=cv)
print("mean R2: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
predicted = cross_val_predict(clf, X, Y, cv=cv)
print("MSE: %0.2f" % mean_squared_error(Y, predicted))

print("\nknn")
knn = KNeighborsRegressor()
scores = cross_val_score(knn, X, Y, cv=cv)
print("mean R2: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
predicted = cross_val_predict(knn, X, Y, cv=cv)
print("MSE: %0.2f" % mean_squared_error(Y, predicted))  

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
print("Lmao")

best_features=4
rfe_lin = RFE(lin,n_features_to_select=best_features).fit(X,Y)

supported_features=rfe_lin.get_support(indices=True)
for i in range(0, 4):
    z=supported_features[i]
    print(i + 1, column_names[z])

print("\nfeature selection on linear regression")
mask = np.array(rfe_lin.support_)
scores = cross_val_score(lin, X[:, mask], Y, cv=cv)
print("R2: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
predicted = cross_val_predict(lin, X[:, mask], Y, cv=cv)
print("MSE: %0.2f" % mean_squared_error(Y, predicted))

print("feature selection ridge regression")
rfe_ridge = RFE(ridge, n_features_to_select=best_features).fit(X, Y)
mask = np.array(rfe_ridge.support_)
scores = cross_val_score(ridge, X[:, mask], Y, cv=cv)
print("R2: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
predicted = cross_val_predict(ridge, X[:, mask], Y, cv=cv)
print("MSE: %0.2f" % mean_squared_error(Y, predicted))

print("feature selection on lasso regression")
rfe_lasso = RFE(lasso, n_features_to_select=best_features).fit(X, Y)
mask = np.array(rfe_lasso.support_)
scores = cross_val_score(lasso, X[:, mask], Y, cv=cv)
print("R2: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
predicted = cross_val_predict(lasso, X[:, mask], Y, cv=cv)
print("MSE: %0.2f" % mean_squared_error(Y, predicted))

print("feature selection on decision tree")
rfe_tree = RFE(tree, n_features_to_select=best_features).fit(X, Y)
mask = np.array(rfe_tree.support_)
scores = cross_val_score(tree, X[:, mask], Y, cv=cv)
print("R2: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
predicted = cross_val_predict(tree, X[:, mask], Y, cv=cv)
print("MSE: %0.2f" % mean_squared_error(Y, predicted))

print("feature selection on random forest")
rfe_forest = RFE(forest, n_features_to_select=best_features).fit(X, Y)
mask = np.array(rfe_forest.support_)
scores = cross_val_score(forest, X[:, mask], Y, cv=cv)
print("R2: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
predicted = cross_val_predict(forest, X[:, mask], Y, cv=cv)
print("MSE: %0.2f" % mean_squared_error(Y, predicted))

print("feature selection on linear support vector machine")
rfe_svm = RFE(svm_lin, n_features_to_select=best_features).fit(X, Y)
mask = np.array(rfe_svm.support_)
scores = cross_val_score(svm_lin, X[:, mask], Y, cv=cv)
print("R2: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
predicted = cross_val_predict(svm_lin, X, Y, cv=cv)
print("MSE: %0.2f" % mean_squared_error(Y, predicted))
