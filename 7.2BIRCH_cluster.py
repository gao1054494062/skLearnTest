import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_blobs
from sklearn.cluster import Birch
from sklearn import metrics

# X为样本特征，Y为样本簇类别， 共1000个样本，每个样本2个特征，共4个簇，簇中心在[-1,-1], [0,0],[1,1], [2,2]
X, y = make_blobs(n_samples=1000, n_features=2, centers=[[-1,-1], [0,0], [1,1], [2,2]], cluster_std=[0.4, 0.3, 0.4, 0.3], 
                  random_state =9)
plt.subplot(2,3,1)
plt.scatter(X[:, 0], X[:, 1], marker='o')
#plt.show()

y_pred = Birch(n_clusters = 3).fit_predict(X)
print("Calinski-Harabasz Score", metrics.calinski_harabaz_score(X, y_pred))
plt.subplot(2,3,2)
plt.scatter(X[:, 0], X[:, 1], c=y_pred)
plt.text(.99, .01, ('score: %.2f' % (metrics.calinski_harabaz_score(X, y_pred))),transform=plt.gca().transAxes, size=10,horizontalalignment='right')

y_pred = Birch(n_clusters = 4, threshold = 0.3).fit_predict(X)
print("Calinski-Harabasz Score", metrics.calinski_harabaz_score(X, y_pred))
plt.subplot(2,3,3)
plt.scatter(X[:, 0], X[:, 1], c=y_pred)
plt.text(.99, .01, ('score: %.2f' % (metrics.calinski_harabaz_score(X, y_pred))),transform=plt.gca().transAxes, size=10,horizontalalignment='right')

y_pred = Birch(n_clusters = 4, threshold = 0.1).fit_predict(X)
print("Calinski-Harabasz Score", metrics.calinski_harabaz_score(X, y_pred))
plt.subplot(2,3,4)
plt.scatter(X[:, 0], X[:, 1], c=y_pred)
plt.text(.99, .01, ('score: %.2f' % (metrics.calinski_harabaz_score(X, y_pred))),transform=plt.gca().transAxes, size=10,horizontalalignment='right')

y_pred = Birch(n_clusters = 4, threshold = 0.3, branching_factor = 20).fit_predict(X)
print("Calinski-Harabasz Score", metrics.calinski_harabaz_score(X, y_pred))
plt.subplot(2,3,5)
plt.scatter(X[:, 0], X[:, 1], c=y_pred)
plt.text(.99, .01, ('score: %.2f' % (metrics.calinski_harabaz_score(X, y_pred))),transform=plt.gca().transAxes, size=10,horizontalalignment='right')

y_pred = Birch(n_clusters = None).fit_predict(X)
print("Calinski-Harabasz Score", metrics.calinski_harabaz_score(X, y_pred))
plt.subplot(2,3,6)
plt.scatter(X[:, 0], X[:, 1], c=y_pred)
plt.text(.99, .01, ('score: %.2f' % (metrics.calinski_harabaz_score(X, y_pred))),transform=plt.gca().transAxes, size=10,horizontalalignment='right')
plt.show()