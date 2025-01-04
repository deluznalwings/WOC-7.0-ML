import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class LinearRegression:
    def __init__(self,decay=0,L=0.001, iterations=1001,momentum=0,lamdal1=0,lamdal2=0):   
        self.momentum=momentum
        self.lamdal1=lamdal1
        self.lamdal2=lamdal2
        self.decay=decay
        self.iterations=iterations
        self.L=L
        self.w=None
        self.b=None
        self.wmomentum=None
        self.bmomentum=None
        self.costhistory=[]
        self.r2history=[]
        self.msehistory=[]

    def normalize(self,X):
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)
        self.m=X.shape[0]
        self.n=X.shape[1]
        Xmean= np.mean(X,axis=0)
        Xstd= np.std(X,axis=0)
        return (X-Xmean)/Xstd

    def initialize(self):
        self.w = np.zeros(self.n)
        self.b = 0
        self.wmomentum=np.zeros(self.n)
        self.bmomentum= 0
        
    def forward(self,X):
        return np.dot(X,self.w)+self.b
        
    def cost(self,y_pred,y_true):
        cost=np.sum((y_pred-y_true)**2)/(2*self.m)
        if self.lamdal1>0:
            cost+= (self.lamdal1 / self.m) * np.sum(np.abs(self.w))
        if self.lamdal2>0:
            cost+= (self.lamdal2 / (2 * self.m)) * np.sum(self.w ** 2)

        return cost

    def gradient(self,X,y_pred,y_true):
        dw=1/(self.m)*(np.dot(X.T,(y_pred-y_true)))
        db=np.sum((y_pred - y_true),axis=0) / self.m
        if self.lamdal2 >0:
            dw += (self.lamdal2 / self.m) * self.w  
        if self.lamdal1 >0:
            dw += (self.lamdal1 / self.m) * np.sign(self.w) 
        
        return dw,db

    def r2score(self,y_pred,y_true):
        m=y_true.shape[0]
        totalvariance = np.sum((y_true - np.mean(y_true)) ** 2)
        explainedvariance = np.sum((y_true - y_pred) ** 2)
        return 1-(explainedvariance / totalvariance)

    def MSE(self,y_pred,y_true):
        return np.mean((y_pred - y_true) ** 2)

    def fit(self,X,y):
        X=self.normalize(X)
        
        self.initialize()
        for i in range(self.iterations):
            y_pred=self.forward(X)
            cost=self.cost(y_pred,y)
            dw,db=self.gradient(X,y_pred,y)
            L=self.L/(1 + self.decay * i)
            wupdates=self.momentum*self.wmomentum-L*dw
            bupdates=self.momentum*self.bmomentum-L*db
            self.wmomentum=wupdates
            self.bmomentum=bupdates
            self.w+=wupdates
            self.b+=bupdates
            r2=self.r2score(y_pred,y)
            mse = self.MSE(y_pred,y)
            self.costhistory.append(cost)
            self.r2history.append(r2)
            self.msehistory.append(mse)
            if (i%100)==0:
                print(f"iterations:{i},cost:{cost:.3f},r2:{r2:.3f},MSE:{mse:.3f}")

        print(f"Final weights:{self.w}, Final biases:{self.b}")

    def predict(self,X):
        X = (X - self.mean) / self.std
        return self.forward(X)

    def insights(self):
        plt.figure(figsize=(12,5))
        plt.subplot(1,3,1)
        plt.plot(range(self.iterations), self.costhistory, label='Cost')
        plt.xlabel('Iterations')
        plt.ylabel('Cost')
        plt.title('Cost over Iterations')

        plt.subplot(1,3,2)
        plt.plot(range(self.iterations), self.r2history, label='R2 Score', color='orange')
        plt.xlabel('Iterations')
        plt.ylabel('R2 Score')
        plt.title('R2 Score over Iterations')

        plt.subplot(1,3,3)
        plt.plot(range(self.iterations), self.msehistory, label='MSE History', color='orange')
        plt.xlabel('Iterations')
        plt.ylabel('MSE')
        plt.title('MSE over Iterations')

    def evaluation(self,X_test,y_test):
        X_test = (X_test - self.mean) / self.std
        y_pred = self.forward(X_test)
        cost = self.cost(y_pred, y_test)
        r2 = self.r2score(y_pred, y_test)
        mse = self.MSE(y_pred, y_test)
        print(f"Evaluation Results: Cost={cost:.3f}, R2 Score={r2:.3f}, MSE={mse:.3f}")
