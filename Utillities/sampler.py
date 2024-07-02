import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

#currently this will create sets that may have overlap, need to find a way to ensure that there is no overlap

def randomSample(numOfSamples:int, seed:int, file:str) -> pd.DataFrame:
    with open(file,"r") as myfile:
        rawData = pd.read_csv(myfile)
        filterData = rawData.query("status == True", inplace=False)
        return filterData.sample(numOfSamples, random_state= seed)
    

def sciRandomSample(file:str) -> pd.DataFrame:
    with open(file,"r") as myfile:
        rawData = pd.read_csv(myfile)
        filterData = rawData.query("status == True", inplace=False)
        train,test = train_test_split(filterData,train_size=10000, test_size= 2000)
        return train,test
    
trainingData, testData = sciRandomSample("f(Validated)trainingSet.csv")

trainingData.to_csv("trainingSet(10000).csv",sep=",")

testData, validationData = train_test_split(testData, train_size=1600,test_size=400)

testData.to_csv("testSet(1600).csv", sep = ",")
validationData.to_csv("validationSet(400).csv", sep= ",")