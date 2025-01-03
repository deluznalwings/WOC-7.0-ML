import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import combinations_with_replacement

class PolynomialRegression:
    def __init__(self,degree=2,interaction=True,iterations=1000,decay=0,L=0.001, momentum=0.8,lamdal1=5e-4,lamdal2=5e-4):
        self.degree=degree
        self.iterations=iterations
        self.momentum=momentum
        self.interaction=interaction
        self.lamdal1=lamdal1
        self.lamdal2=lamdal2
        self.decay=decay
        self.L=L
        self.w=None
        self.b=None
        self.wmomentum=None
        self.bmomentum=None
        self.polyfeatures=None
        self.costhistory=[]
        self.r2history=[]

    def normalize(self,X):
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)
        return (X-self.mean)/self.std

    def polynomial(self,X):
        if self.polyfeatures is None:
            self.m,self.n=X.shape
            polyfeatures=[]
            for d in range(1,self.degree+1):
                for combo in combinations_with_replacement(range(self.n),d):
                    polyfeatures.append(combo)
            self.polyfeatures=polyfeatures
        polymatrix=[]
        for sample in X:
            row=[]
            for combo in self.polyfeatures:
                term=1
                for idx in combo:
                    term*=sample[idx]
                row.append(term)
            polymatrix.append(row)
        return np.array(polymatrix)
        

    def initialize(self,npoly):
        self.w = np.zeros(npoly)
        self.b = 0
        self.wmomentum=np.zeros(npoly)
        self.bmomentum= 0
        
    def forward(self,X):
        return np.dot(X,self.w)+self.b
        
    def cost(self,y_pred,y_true,runtimeissue=False):
        if runtimeissue:
            cost=np.sum((y_pred-y_true)**2)/(2*(self.m)*(self.m))
        else:   
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
        residualvariance = np.sum((y_true - y_pred) ** 2)
        return 1-(residualvariance / totalvariance)

    def fit(self,X,y,toprint=True):
        X=self.normalize(X)
        Xpoly=self.polynomial(X)
        self.m=X.shape[0]
        self.n=X.shape[1]
        self.initialize(Xpoly.shape[1])
        for i in range(self.iterations):
            y_pred=self.forward(Xpoly)
            cost=self.cost(y_pred,y)
            dw,db=self.gradient(Xpoly,y_pred,y)
            L=self.L/(1 + self.decay * i)
            wupdates=self.momentum*self.wmomentum-L*dw
            bupdates=self.momentum*self.bmomentum-L*db
            self.wmomentum=wupdates
            self.bmomentum=bupdates
            self.w+=wupdates
            self.b+=bupdates
            r2=self.r2score(y_pred,y)
            self.costhistory.append(cost)
            self.r2history.append(r2)
            if toprint and (i%100)==0:
                print(f"iterations:{i},cost:{cost:.7f},r2:{r2:.7f}")
        if toprint:
            print(f"Final weights:{self.w}, Final biases:{self.b}")

    def predict(self,X):
        X = (X - self.mean) / self.std
        Xpoly=self.polynomial(X)

    def BIC(self,X,y,mindegree,maxdegree):
        bicscore=[]
        weightsperdegree={}
        biasesperdegree={}
        for degree in range(mindegree,maxdegree+1):
            self.degree=degree
            self.polyfeatures = None
            Xpoly=self.polynomial(X)
            self.fit(X,y,toprint=False)
            ypred=self.predict(X)
            RSS=np.sum((ypred-y)**2)
            m=X.shape[0]
            k=Xpoly.shape[1]
            bic=m*np.log(RSS/m)+k*np.log(m)
            bicscore.append(bic)
            weightsperdegree[degree]=self.w.copy()
            biasesperdegree[degree]=self.b

        plt.plot(figsize=(8,6))
        plt.plot(range(mindegree,maxdegree+1),bicscore,marker='o')
        plt.title('BIC vs Polynomial Degree')
        plt.xlabel('Polynomial Degree')
        plt.ylabel('BIC')
        plt.show()
        for degree in range(mindegree,maxdegree+1):
            print(f"Degree {degree}: Weights: {weightsperdegree[degree]}, Bias: {biasesperdegree[degree]}")
        

    def insights(self):
        plt.figure(figsize=(12,5))
        plt.subplot(1,2,1)
        plt.plot(range(self.iterations), self.costhistory, label='Cost')
        plt.xlabel('Iterations')
        plt.ylabel('Cost')
        plt.title('Cost over Iterations')

        plt.subplot(1,2,2)
        plt.plot(range(self.iterations), self.r2history, label='R2 Score', color='orange')
        plt.xlabel('Iterations')
        plt.ylabel('R2 Score')
        plt.title('R2 Score over Iterations')

    def evaluation(self, X_test, y_test):
        X_test = (X_test - self.mean) / self.std
        Xpoly_test = self.polynomial(X_test)
        y_pred = self.forward(Xpoly_test)
        r2 = self.r2score(y_pred, y_test)
        cost = self.cost(y_pred, y_test)
        print(f"Evaluation Results:")
        print(f"R2 Score: {r2:.3f}")
        print(f"Cost: {cost:.3f}")