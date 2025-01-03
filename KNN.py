import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
class KNN:
    def __init__(self,k=5):
        self.k=k

    def fit(self,X_train,y_train):
        self.X_train=np.array(X_train)
        self.y_train=np.array(y_train)

    def predict(self,X_test):
        X_test=np.array(X_test)
        predictions=[]
        for X in X_test:
            distances=[np.sqrt(np.sum((X_train - X)**2)) for X_train in self.X_train]
            kindices=np.argsort(distances)[:self.k]
            knearestlabels=self.y_train[kindices]
            labelcount={}
            for label in knearestlabels:
                if label in labelcount:
                    labelcount[label]+=1

                else:
                    labelcount[label]=1

            mostcommonlabel=max(labelcount,key=labelcount.get)
            predictions.append(mostcommonlabel)

        return np.array(predictions)
                
    def accuracy(self,y_pred,y_test):
        y_pred=np.array(y_pred)
        y_test=np.array(y_test)
        return np.mean(y_test==y_pred)

    def insights(self,X_test,y_test,maxk):
        X_test=np.array(X_test)
        y_test=np.array(y_test)
        accuracies=[]
        for k in range(1,maxk+1):
            self.k=k
            predictions=self.predict(X_test)
            accuracy=self.accuracy(predictions,y_test)
            accuracies.append(accuracy)

        plt.figure(figsize=(8,6))
        plt.plot(range(1,maxk+1),accuracies, marker='o')
        plt.title('Accuracy vs K Values')
        plt.xlabel('K Values')
        plt.ylabel('Accuracy')
        plt.grid()
        plt.show()