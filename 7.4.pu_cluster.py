import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.cluster import SpectralClustering
from sklearn import metrics

X, y = datasets.make_blobs(n_samples=500, n_features=6, centers=5, cluster_std=[0.4, 0.3, 0.4, 0.3, 0.4], random_state=11)
y_pred = SpectralClustering().fit_predict(X)
print("Calinski-Harabasz Score", metrics.calinski_harabaz_score(X, y_pred))

count=1
for index, gamma in enumerate((0.01,0.1,1,1.5, 10)):
    for index, k in enumerate((3,4,5,6,10)):
        y_pred = SpectralClustering(n_clusters=k, gamma=gamma).fit_predict(X)
        plt.subplot(5,5,count)
        plt.scatter(X[:, 0], X[:, 1], c=y_pred)
        print("Calinski-Harabasz Score with gamma=", gamma, "n_clusters=", k,"score:", metrics.calinski_harabaz_score(X, y_pred))
        count += 1
plt.show()
y_pred = SpectralClustering(gamma=0.1).fit_predict(X)
print("Calinski-Harabasz Score", metrics.calinski_harabaz_score(X, y_pred))