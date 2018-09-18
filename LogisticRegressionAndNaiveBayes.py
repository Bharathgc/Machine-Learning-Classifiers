
# coding: utf-8

# In[13]:


import pandas as pd 
import numpy as np 
import math
from matplotlib import pyplot as plt
df=pd.read_csv('data.txt',sep=",",header=None)
train=df.sample(frac=1,random_state=200)
train.columns = ['feature1','feature2','feature3','feature4','lable']
lb=train['lable']
maintraindata = train.iloc[:int(2/3*len(train))]
testdata = train.iloc[int(2/3*len(train)):]
columns=list(train)
columns.remove('lable')
labelCol = list(testdata['lable'])

def calculateProbability(x, mean, stdev):
    exponent = math.exp(-(math.pow(x-mean,2)/(2*math.pow(stdev,2))))
    return (1 / (math.sqrt(2*math.pi) * stdev)) * exponent

def GaussianNaiveBayes(traindata):
    trainzerodata = traindata[traindata['lable'] == 0]
    trainonedata = traindata[traindata['lable'] == 1]
    Pofone = len(trainonedata)/(len(traindata))
    pofzero = len(trainzerodata)/(len(traindata))
    trainzerodatamean = []
    for i in range(len(columns)):
        Tempmean = trainzerodata[columns[i]].mean()
        trainzerodatamean.append(Tempmean)
    trainzerodataSD = []
    for i in range(len(columns)):
        TempSD = trainzerodata[columns[i]].std()
        trainzerodataSD.append(TempSD)
    trainonedatamean = []
    for i in range(len(columns)):
        Tempmean = trainonedata[columns[i]].mean()
        trainonedatamean.append(Tempmean)
    trainonedataSD = []
    for i in range(len(columns)):
        TempSD = trainonedata[columns[i]].std()
        trainonedataSD.append(TempSD)
    probability = 1
    probabilitylistzero = []
    for i in range(len(testdata)):
        probability = 1
        for j in range(len(columns)):
            Currprobability = calculateProbability(testdata.iloc[i][j], trainzerodatamean[j], trainzerodataSD[j])
            probability = probability * Currprobability
        probabilitylistzero.append(probability)
    probabilitylistone = []
    for i in range(len(testdata)):
        probability = 1
        for j in range(len(columns)):
            Currprobability = calculateProbability(testdata.iloc[i][j], trainonedatamean[j], trainonedataSD[j])
            probability = probability * Currprobability
        probabilitylistone.append(probability)
    finalprobabilityone = []
    for i in range(len(testdata)):
        numerator = probabilitylistone[i] * Pofone
        denominator = (probabilitylistzero[i] * pofzero) + numerator
        finalprobabilityone.append(numerator/denominator)
    finalprobabilityzero = []
    for i in range(len(testdata)):
        numerator = probabilitylistzero[i] * pofzero
        denominator = (probabilitylistone[i] * Pofone) + numerator
        finalprobabilityzero.append(numerator/denominator)
    finalprobabilit = []
    for i in  range(len(testdata)):
        if(finalprobabilityone[i] > finalprobabilityzero[i]):
            finalprobabilit.append("1")
        else:
            finalprobabilit.append("0")
    out = [int(labelCol[i])==int(finalprobabilit[i]) for i in range(len(labelCol))]
    return (np.mean(out)*100),trainonedatamean,trainonedataSD

def CalculateAccuracy(testdata,weights):
    w = np.dot(testdata,np.transpose(weights))
    yhat = 1.0 / (1 + np.exp(-w))
    return yhat
    
def LogisticRegression(traindata,weights):
    TraindataLable = traindata['lable']
    traindata['feature0'] = 1
    traindata = traindata[['feature0','feature1','feature2','feature3','feature4']]
    LearningRate = 0.5
    count = 0
    for i in range(1000):
        w = np.dot(traindata,np.transpose(weights))
        yhat = 1.0 / (1 + np.exp(-w))
        TraindataLable = np.array(TraindataLable)
        NewWeights=(np.dot(np.transpose(TraindataLable-yhat),traindata))/len(traindata)
        weights += LearningRate * NewWeights

    return weights,yhat

def generatormodel(mean,var):
    data=np.random.normal(mean,np.sqrt(var),(400,4))
    return data

if __name__ == '__main__':
    fractions = [0.01, 0.02, 0.05, 0.1, 0.625, 1]
    traindatalength = len(maintraindata)-700
    finalaverages = []
    finalaccuracies = []
    for fraction in fractions:
        print("For fraction ",fraction)
        traindataSubsetlength = np.random.randint(traindatalength,size = 5)
        for Subsetlength in traindataSubsetlength:
            averages = []
            accuracies = []
            traindataSubset = maintraindata[Subsetlength:int(Subsetlength+len(maintraindata)*fraction)]
            #gaussian naive bayes
            if(fraction == 1):
                tempavreage , mean, variance = GaussianNaiveBayes(maintraindata)
                averages.append(tempavreage)
            else:
                tempavreage , mean, variance = GaussianNaiveBayes(traindataSubset)
                averages.append(tempavreage)
            
            #logistic regression
            accuracies = []
            testdata_dummy = testdata
            testdata_dummy['feature0'] = 1
            testdata_dummy = testdata_dummy[['feature0','feature1','feature2','feature3','feature4']]
            weights = np.random.rand(5)*0.01
            if(fraction == 1):
                weights,yhat = LogisticRegression(maintraindata,weights)
                yhat = CalculateAccuracy(testdata_dummy,weights) 
            else:
                weights,yhat = LogisticRegression(traindataSubset,weights)
                yhat = CalculateAccuracy(testdata_dummy,weights)
            
            Accuracy = [int(labelCol[i])==int(np.round(yhat[i])) for i in range(len(testdata))]
            accuracies.append(np.mean(Accuracy)*100)
            
            #generative model
            data = generatormodel(mean,variance)
            trainonedatamean = np.sum(data,0)/len(data)
            temp = data - trainonedatamean
            trainonedataSD = np.sum(np.power(temp,2),0)/(len(data)-1)
            print("mean and variance of train data",mean,variance)
            print("mean and variance of generated data",trainonedataSD,trainonedatamean)
            
                
        print("Gaussian naive bayes",np.array(averages).mean())
        finalaverages.append(np.array(averages).mean())
        print("Logistic regression",np.array(accuracies).mean())
        finalaccuracies.append(np.array(accuracies).mean())
        
        
    fig,ax = plt.subplots()
    plt.title("Learning curve")
    plt.xlabel("Size Ratio")
    plt.ylabel(" Results")
    ax.plot(fractions,finalaverages,'*-',label = " Gaussian Naive Bayes")
    ax.plot(fractions,finalaccuracies,'o-',label = " Logistic Regression")
    ax.legend()
    plt.show()
    

