import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
X,y=make_blobs(n_samples=300,n_features=2,centers=4,cluster_std=1.0,random_state=42) #using synthetic dataset coz, karne do bhai🙏
sse=[]

for k in range(1,11):
    km=KMeans(n_clusters=k,random_state=42)
    km.fit(X)

    sse.append(km.inertia_)

plt.plot(range(1,11),sse,'-o')
plt.xlabel("Number of clusters")
plt.ylabel("SSE")

plt.title("Elbow Method")
plt.show()
optimal_k=4
km=KMeans(n_clusters=optimal_k,random_state=42)
y_pred=km.fit_predict(X)

plt.scatter(X[:,0],X[:,1],c=y_pred,cmap='viridis')
plt.scatter(km.cluster_centers_[:,0],km.cluster_centers_[:,1],s=200,c='red',label='Centroids')
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")

plt.title(f"K-Means Clustering with {optimal_k} clusters")
plt.legend()
plt.show()
for i in range(optimal_k):
    print(f"Cluster {i}: {np.sum(y_pred==i)} points")
