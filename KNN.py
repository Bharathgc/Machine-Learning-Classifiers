# -*- coding: utf-8 -*-
import time
import scipy
from operator import itemgetter
from sklearn.datasets import fetch_mldata
import numpy as np
from matplotlib import pyplot as plt
from mnist import MNIST

#custom_data_home = "C:\\Users\\USER\\Desktop\\ASU\\SML\\assignment2"
#mnist = fetch_mldata('MNIST original', data_home=custom_data_home)
start_time = time.time()
mndata = MNIST("C:\\Users\\USER\\Desktop\\ASU\\SML\\assignment2\\data")
traindata, trainlabel = mndata.load_training()
testdata, testlabel = mndata.load_testing()
traindata =  traindata[:6000]
trainlabel = trainlabel[:6000]
testdata = testdata[-1000:]
testlabel = testlabel[-1000:]


def CalculateEuclidianDistance(X,Y):
    return scipy.spatial.distance.euclidean(X,Y)

def CalucalteDataDistances():
    DistanceRow = {}
    DistanceMatrix = {}
    for i in range(len(testdata)):
        for j in range(len(traindata)):
            DistanceRow[j] = CalculateEuclidianDistance(testdata[i],traindata[j])
            
        DistanceRow = dict(sorted(DistanceRow.items(), key=itemgetter(1)))
        DistanceMatrix[i] = DistanceRow
        DistanceRow = {}
    return DistanceMatrix

def Calculateneighbors(DistanceMatrix , k):
    AssginedClassDictionary = {}
    classes = []
    for i in range(len(testdata)):
        leng = list(DistanceMatrix[i].keys())
        leng = leng[:k]
        for j in leng:
            classes.append(trainlabel[j])   
        AssginedClassDictionary[i] = max(classes, key = classes.count)
        classes = []
    return AssginedClassDictionary  
     
def GetAccuracy(AssignedClasses):
    out = []
    PredictedTestLabels = list(AssignedClasses.values())
    for i in range(len(testlabel)):
        if(int(PredictedTestLabels[i]) == int(testlabel[i])):
            out.append(i)
    Accuracy = len(out)/len(testlabel)
    return (Accuracy)

if __name__ == '__main__':
    kneigbhors = [ 1, 9, 19, 29, 39, 49, 59, 69, 79, 89, 99]
    DistanceMatrix = CalucalteDataDistances()
    AssignedClasses = {}
    Accuracies = []
    for k in kneigbhors:
        AssignedClasses = Calculateneighbors(DistanceMatrix , k)
        Accuracy = GetAccuracy(AssignedClasses)
        print("Accuracy for ",k, " Neighbors is - ", Accuracy*100)
        Accuracies.append(1.0 - Accuracy)
        AssignedClasses= []
    
    fig,ax=plt.subplots()
    plt.title("Error curve")
    plt.xlabel("No of Neigbhors K ")
    plt.ylabel("Error Rate ")
    ax.plot(kneigbhors,Accuracies, '*-',label="KNN Classifier")
    ax.legend()
    plt.show()
    stop_time = time.time()
    print("Time taken for the execution", (stop_time - start_time)/60, "mins")
