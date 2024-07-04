import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


stopWords = set(stopwords.words('english'))

def preProcess(data:pd.DataFrame):
    stopRemoved = removeStop(data)
    print(replaceNum(stopRemoved))
    
def removeStop(data:pd.DataFrame) -> list[str]:
    """remove stop words"""
    for i in data['text']:
        wordTokens = word_tokenize(i)        
        filteredSentence = [w for w in wordTokens if not w.lower() in stopWords]        
        return filteredSentence
    
def replaceNum(data:list[str]) -> list[str]:
    """replace numbers with [num] and angles with [ang]"""
    for i in range(len(data)):
        if any(chr.isdigit() for chr in data[i]):
            if (float(data[i]) // 10) <= 1:
                data[i]='[NUM]'
            elif (float(data[i])//10) > 1:
                data[i]='[ANG]'
    return data

preProcess(pd.read_csv('validationSet(400).csv'))