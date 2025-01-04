import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
class KMeansClustering:
    def __init__(self,k=3,iterations=1000):
        self.k=k
        self.iterations=iterations
        self.centroids=[]
        self.clusters=[[] for _ in range(self.k)]

    def predict(self,X):
        self.X=X
        self.m,self.n=X.shape
        randomindices=np.random.choice(self.m,size=self.k,replace=False)
        self.centroids=X[randomindices]
        
        for i in range(self.iterations):
            y=[]
            for data in X:
                #distances=euclidiandist(data,self.centroids)
                distances= np.sqrt(np.sum((self.centroids-data)**2,axis=1))
                clusternum=np.argmin(distances)
                y.append(clusternum)

            y=np.array(y)

            clusterindices=[]
            for i in range(self.k):
                clusterindices.append(np.argwhere(y==i).flatten())

            clustercentres=[]

            for i,indices in enumerate(clusterindices):  #i will fetch the indices and indices will fed the list at that particular i
                if len(indices)==0:
                    clustercentres.append(np.array(self.centroids[i]))

                else:
                    clustercentres.append(np.mean(X[indices],axis=0))


            if np.max(np.abs(self.centroids-np.array(clustercentres)))<0.0001:
                break

            else:
                self.centroids=np.array(clustercentres)

        self.clusterindices=clusterindices
        return y

    def sse(self,X,labels):
        sse=0
        for i in range(self.k):
            clusterpoints=X[labels==i]
            if len(clusterpoints) > 0:
                sse += np.sum((clusterpoints - self.centroids[i]) ** 2)
        return sse

    def elbow(self,X,maxk=10):
        ssevalues=[]
        for k in range(1,maxk+1):
            model=kmeans(k=k)
            labels=model.predict(X)
            sse=model.sse(X,labels)
            ssevalues.append(sse)

        plt.figure(figsize=(10, 6))
        plt.plot(range(1, maxk + 1), ssevalues, marker='o')
        plt.title('SSE vs. Number of Clusters (k)')
        plt.xlabel('Number of Clusters (k)')
        plt.ylabel('Sum of Squared Errors (SSE)')
        plt.xticks(range(1, maxk + 1))
        plt.grid()
        plt.show()
