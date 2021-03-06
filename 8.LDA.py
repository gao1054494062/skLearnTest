﻿import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.datasets.samples_generator import make_classification
X, y = make_classification(n_samples=1000, n_features=5, n_redundant=0, n_classes=5, n_informative=4,
                           n_clusters_per_class=1,class_sep =0.5, random_state =10)
fig = plt.figure()
ax = Axes3D(fig, rect=[0, 0, 1, 1], elev=30, azim=20)
plt.subplot(2,2,1)
plt.scatter(X[:, 0], X[:, 1], X[:, 2],marker='o',c=y)

from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca.fit(X)
print( pca.explained_variance_ratio_ )
print( pca.explained_variance_ )

X_new = pca.transform(X)
plt.subplot(2,2,2)
plt.scatter(X_new[:, 0], X_new[:, 1],marker='o',c=y)

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
lda = LinearDiscriminantAnalysis(n_components=2)
lda.fit(X,y)
X_new = lda.transform(X)
plt.subplot(2,2,3)
plt.scatter(X_new[:, 0], X_new[:, 1],marker='o',c=y)

lda = LinearDiscriminantAnalysis(n_components=3)
lda.fit(X,y)
X_new = lda.transform(X)
plt.subplot(2,2,4)
plt.scatter(X_new[:, 0],X_new[:, 1], X_new[:, 2],marker='o',c=y)

plt.show()