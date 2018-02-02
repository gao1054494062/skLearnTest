import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import datasets, linear_model

# read_csv里面的参数是csv在你电脑上的路径，此处csv文件放在notebook运行目录下面的CCPP目录里
data = pd.read_excel('E:\python\PythonPro\CCPP\Folds5x2_pp.xlsx')

X = data[['AT', 'V', 'AP', 'RH']]
y = data[['PE']]

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

from sklearn.linear_model import Ridge
ridge = Ridge(alpha = 1)
ridge.fit(X_train, y_train)

print(ridge.coef_)
print(ridge.intercept_)

from sklearn.linear_model import RidgeCV
ridgecv = RidgeCV(alphas=[0.01, 0.1, 0.5, 1, 3, 5, 7, 10, 20, 100])
ridgecv.fit(X_train, y_train)
print(ridgecv.alpha_)
print("ridgecv coef_ = ", ridgecv.coef_)
print("ridgecv intercept_ = ",ridgecv.intercept_)


from sklearn.linear_model import LassoCV
lassoCV = LassoCV(alphas=[0.01, 0.1, 0.5, 1, 3, 5, 7, 10, 20, 100])
lassoCV.fit(X_train, y_train)
print(lassoCV.alpha_)
print("lassoCV coef_ = ", lassoCV.coef_)
print("lassoCV intercept_ = ",lassoCV.intercept_)
