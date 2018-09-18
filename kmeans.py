# -*- coding: utf-8 -*-
"""
Created on Sun Apr 22 01:02:06 2018

@author: Gunari Bharath
"""

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from matplotlib import pyplot as plt

file = "bc.txt"
data = np.loadtxt(file, delimiter = ",")
data = data[:,1:10]
ks = [2,3,4,5,6,7,8]

def GetCentroids():
    new_centroids = []
    for i in range(k):
        sum1 = []
        for j in range(len(indecis)):
            if i == indecis[j]:
                sum1.append(data[j])
        new_centroids.append(np.mean(sum1, axis = 0))  
    return new_centroids  

func = []      
for k in ks:
    centroids = np.random.randint(0,10,(k,9))
    diff = 2
    while(diff != 0):
        distances = cdist(data,centroids)
        indecis = np.argmin(distances, axis = 1)
        newcentroids = GetCentroids()
        diff = np.absolute(np.sum(np.subtract(newcentroids,centroids)))
        centroids = newcentroids 
    centroids = np.round(centroids)
    distances_2 = np.square(cdist(data,centroids))
    sum2 = 0
    for i in range(k):
        sum1 = 0
        for j in range(len(indecis)):
            if i == indecis[j]:
                sum1 += distances_2[j][i]
        sum2 += sum1
    print("The value of the potential function for k ", k , " is " , sum2)
    func.append(sum2)


fig,ax=plt.subplots()
plt.title("Potential Function vs k")
plt.xlabel("values k ")
plt.ylabel("Potential functions ")
ax.plot(ks,func, '*-',label="K Means Classifier")
ax.legend()
plt.show()
