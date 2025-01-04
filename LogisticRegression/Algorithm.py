import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
class LogisticRegression:
    def __init__(self, decay=0, L=0.001, iterations=1000, momentum=0, lamdal1=0, lamdal2=0):   
        self.momentum = momentum
        self.lamdal1 = lamdal1
        self.lamdal2 = lamdal2
        self.iterations = iterations
        self.decay = decay
        self.L = L
        self.w = None
        self.b = None
        self.wmomentum = None
        self.bmomentum = None
        self.costhistory = []
        self.f1history = []

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def normalize(self, X):
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)
        self.m = X.shape[0]
        self.n = X.shape[1]
        return (X - self.mean) / self.std

    def initialize(self):
        self.w = np.zeros(self.n)
        self.b = 0
        self.wmomentum = np.zeros(self.n)
        self.bmomentum = 0
        
    def forward(self, X):
        return self.sigmoid(np.dot(X, self.w) + self.b)
        
    def cost(self, y_pred, y_true):
        epsilon = 1e-7  # To prevent log(0)
        cost = -np.sum(y_true * np.log(y_pred + epsilon) + (1 - y_true) * np.log(1 - y_pred + epsilon)) / self.m
        if self.lamdal1 > 0:
            cost += (self.lamdal1 / self.m) * np.sum(np.abs(self.w))
        if self.lamdal2 > 0:
            cost += (self.lamdal2 / (2 * self.m)) * np.sum(self.w ** 2)
        return cost

    def gradient(self, X, y_pred, y_true):
        dw = 1 / self.m * (np.dot(X.T, (y_pred - y_true)))
        db = np.sum((y_pred - y_true), axis=0) / self.m
        if self.lamdal2 > 0:
            dw += (self.lamdal2 / self.m) * self.w  
        if self.lamdal1 > 0:
            dw += (self.lamdal1 / self.m) * np.sign(self.w) 
        
        return dw, db

    def fit(self, X, y):
        X = self.normalize(X)
        self.initialize()
        for i in range(self.iterations):
            y_pred = self.forward(X)
            cost = self.cost(y_pred, y)
            dw, db = self.gradient(X, y_pred, y)
            L = self.L / (1 + self.decay * i)
            wupdates = self.momentum * self.wmomentum - L * dw
            bupdates = self.momentum * self.bmomentum - L * db
            self.wmomentum = wupdates
            self.bmomentum = bupdates
            self.w += wupdates
            self.b += bupdates
            binary_pred = (y_pred >= 0.5).astype(int)
            accuracy = np.mean(binary_pred == y)
            f1 = f1_score(y, binary_pred, average='macro')
            self.costhistory.append(cost)
            self.f1history.append(f1)
            if (i % 100) == 0:
                print(f"iterations: {i}, cost: {cost:.3f}, f1: {f1:.3f}, accuracy: {accuracy:.3f}")

        print(f"Final weights: {self.w}, Final biases: {self.b}")

    def evaluation(self, X, y):
        X = (X - self.mean) / self.std
        y_pred = self.forward(X)
        binary_pred = (y_pred >= 0.5).astype(int)
        
        cost = self.cost(y_pred, y)
        accuracy = np.mean(binary_pred == y)
        f1 = f1_score(y, binary_pred, average='macro')
        
        
        print(f"Evaluation Metrics:")
        print(f"Cost: {cost:.3f}")
        print(f"Accuracy: {accuracy:.3f}")
        print(f"F1 Score: {f1:.3f}")
        return {"cost": cost, "accuracy": accuracy, "f1": f1}

    def predictprobability(self, X):
        X = (X - self.mean) / self.std
        return self.forward(X)

    def predict(self, X):
        probabilities = self.predictprobability(X)
        return (probabilities >= 0.5).astype(int)

    def insights(self):
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(range(self.iterations), self.costhistory, label='Cost')
        plt.xlabel('Iterations')
        plt.ylabel('Cost')
        plt.title('Cost over Iterations')

        plt.subplot(1, 2, 2)
        plt.plot(range(self.iterations), self.f1history, label='F1 Score', color='orange')
        plt.xlabel('Iterations')
        plt.ylabel('F1 Score')
        plt.title('F1 Score over Iterations')
