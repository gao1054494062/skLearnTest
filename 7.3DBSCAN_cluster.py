import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import metrics

X1, y1=datasets.make_circles(n_samples=5000, factor=.6, noise=.05)
X2, y2 = datasets.make_blobs(n_samples=1000, n_features=2, centers=[[1.2,1.2]], cluster_std=[[.1]],random_state=9)

X = np.concatenate((X1, X2))
plt.subplot(2,3,1)
plt.scatter(X[:, 0], X[:, 1], marker='o')

from sklearn.cluster import KMeans
y_pred = KMeans(n_clusters=3, random_state=9).fit_predict(X)
plt.subplot(2,3,2)
plt.scatter(X[:, 0], X[:, 1], c=y_pred)
plt.text(.99, .01, ('score: %.2f' % (metrics.calinski_harabaz_score(X, y_pred))),transform=plt.gca().transAxes, size=10,horizontalalignment='right')

from sklearn.cluster import DBSCAN
y_pred = DBSCAN().fit_predict(X)
plt.subplot(2,3,3)
plt.scatter(X[:, 0], X[:, 1], c=y_pred)

y_pred = DBSCAN(eps = 0.1).fit_predict(X)
plt.subplot(2,3,4)
plt.scatter(X[:, 0], X[:, 1], c=y_pred)
plt.text(.99, .01, ('score: %.2f' % (metrics.calinski_harabaz_score(X, y_pred))),transform=plt.gca().transAxes, size=10,horizontalalignment='right')

y_pred = DBSCAN(eps = 0.1, min_samples = 10).fit_predict(X)
plt.subplot(2,3,5)
plt.scatter(X[:, 0], X[:, 1], c=y_pred)
plt.text(.99, .01, ('score: %.2f' % (metrics.calinski_harabaz_score(X, y_pred))),transform=plt.gca().transAxes, size=10,horizontalalignment='right')

y_pred = DBSCAN(eps = 0.05, min_samples = 5).fit_predict(X)
plt.subplot(2,3,6)
plt.scatter(X[:, 0], X[:, 1], c=y_pred)
plt.text(.99, .01, ('score: %.2f' % (metrics.calinski_harabaz_score(X, y_pred))),transform=plt.gca().transAxes, size=10,horizontalalignment='right')
plt.show()