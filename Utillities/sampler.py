import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

#currently this will create sets that may have overlap, need to find a way to ensure that there is no overlap

def randomSample(numOfSamples:int, seed:int, file:str) -> pd.DataFrame:
    with open(file,"r") as myfile:
        rawData = pd.read_csv(myfile)
        filterData = rawData.query("status == True", inplace=False)
        return filterData.sample(numOfSamples, random_state= seed)
    

def sciRandomSample(testPercentage:float,file:str) -> pd.DataFrame:
    with open(file,"r") as myfile:
        rawData = pd.read_csv(myfile)
        filterData = rawData.query("status == True", inplace=False)
        train,test = train_test_split(filterData,train_size=5000, test_size= 1000)
        return train,test
    
trainingData, testData = sciRandomSample(0.0388, "f(Validated)trainingSet.csv")

trainingData.to_csv("trainingSet(5000).csv",sep=",")

testData, validationData = train_test_split(testData, train_size=800,test_size=200)

testData.to_csv("testSet(1000).csv", sep = ",")
validationData.to_csv("validationSet(200).csv", sep= ",")