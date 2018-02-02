from sklearn import svm
from sklearn.svm import SVC
from sklearn.datasets import make_moons, make_circles, make_classification

# X = [[0, 0], [1, 1]]
# y = [0, 1]

X, y = make_circles(noise=0.2, factor=0.5, random_state=1);

clf = SVC()
clf.fit(X, y) 
SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False)

clf.predict([[2., 2.]])

# 获得支持向量
print(clf.support_vectors_)

# 获得支持向量的索引get indices of support vectors
print(clf.support_)

# 为每一个类别获得支持向量的数量
print(clf.n_support_)
