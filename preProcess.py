import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import glob


def preProcess(data:pd.DataFrame,type= "None"):
    if type == "All":
        stopRemoved = removeStop(data)
        return replaceNum(stopRemoved)
    if type == "Stop":
        return removeStop(data)
    if type == "Token":
        return replaceNum(data)
    if type == "None":
        return data

def readTEXT_to_LIST(input_file):
    with open(input_file, "r", encoding="utf-8") as infile:
        data = []
        for line in infile:
            data.append(line)
    return data

def get_cleaned_stopwords():
    # from https://github.com/igorbrigadir/stopwords
    stopword_files = glob.glob("stopwords/en/*.txt")
    num_str = {'one','two','three','four','five','six','seven','eight','nine'}

    all_stopwords_list = set()

    for file_path in stopword_files:
        all_stopwords_list |= set(readTEXT_to_LIST(file_path))

    cleaned_list_for_mat = {wrd.replace("\n", "").strip() for wrd in all_stopwords_list} - {wrd for wrd in all_stopwords_list if wrd.isdigit()} - num_str
    
    return cleaned_list_for_mat

def removeStop(data:pd.DataFrame) -> pd.DataFrame:
    """remove stop words"""
    stopWords= get_cleaned_stopwords()
    stopWords= get_cleaned_stopwords()
    for i in data['text']:
        wordTokens = word_tokenize(i)        
        filteredSentence = [w for w in wordTokens if not w.lower() in stopWords]
        out = "".join(filteredSentence)        
        data['text'].replace(i,out,inplace=True)
    return data
    
def replaceNum(data:pd.DataFrame) -> pd.DataFrame:
def replaceNum(data:pd.DataFrame) -> pd.DataFrame:
    """replace numbers with [num] and angles with [ang]"""
    for sent in data['text']:
        editedSent = list(sent)
        for i in range(len(editedSent)):
            if any(chr.isdigit() for chr in editedSent[i]):
                if (float(editedSent[i]) // 10) <= 1:
                    editedSent[i]='[NUM]'
                elif (float(data[i])//10) > 1:
                    editedSent[i]='[ANG]'
        out = "".join(editedSent)
        data['text'].replace(sent,out,inplace=True)
    return data

