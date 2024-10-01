from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()
print(cancer.DESCR)
print(len(cancer.data[cancer.target==1]))

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler() #instantiate
# compute the mean and standard which will be used in the next command
scaler.fit(cancer.data)
X_scaled=scaler.transform(cancer.data)
# we check the minimum and maximum of the scaled features which we expect to be 0 and 1
print("after scaling minimum", X_scaled.min(axis=0))

from sklearn.decomposition import PCA
pca=PCA(n_components=3)
pca.fit(X_scaled)
X_pca=pca.transform(X_scaled)
print("shape of X_pca", X_pca.shape) # let's check the shape of X_pca array

ex_variance=np.var(X_pca,axis=0)
ex_variance_ratio = ex_variance/np.sum(ex_variance)
print(ex_variance_ratio)

#Create the two principal components.
#Xax=X_pca[:,0]
#Yax=X_pca[:,1]

#Combine the PCA components with the third:
Xax=X_pca[:,0]+X_pca[:,2]
Yax=X_pca[:,1]+X_pca[:,2]

labels=cancer.target
cdict={0:'red',1:'green'}
labl={0:'Malignant',1:'Benign'}
marker={0:'*',1:'o'}
alpha={0:.3, 1:.5, 2:.5}
fig,ax=plt.subplots(figsize=(7,5))
fig.patch.set_facecolor('white')
for l in np.unique(labels):
 ix=np.where(labels==l)
 ax.scatter(Xax[ix],Yax[ix],c=cdict[l],s=40,label=labl[l],marker=marker[l],alpha=alpha[l])
plt.xlabel("First Principal Component",fontsize=14)
plt.ylabel("Second Principal Component",fontsize=14)
plt.legend()
plt.show()

plt.matshow(pca.components_,cmap='viridis')
plt.yticks([0,1,2],['1st Comp','2nd Comp','3rd Comp'],fontsize=10)
plt.colorbar()
plt.xticks(range(len(cancer.feature_names)),cancer.feature_names,rotation=65,ha='left')
plt.tight_layout()
plt.show()

