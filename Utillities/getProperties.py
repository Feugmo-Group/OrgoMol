import pandas as pd
from tqdm import tqdm

#have property arg because in the future may wanna use this to get different properties
#right now getting homo-luma gap
#weird thing with pandas writing an extra column! Need to fix! Fixed as of July 2024

def getProperties(dataSet:str, property = None) -> None:
    with open(dataSet, "r", newline='') as myfile:
        data = pd.read_csv(myfile)
        homoLumoGap = []
        for row in tqdm(data['fileName']):
            rowRaw, *_ = row.partition(".")
            with open(f'{rowRaw}.xyz') as xyz:
                haValue = xyz.readlines()[1].split()[9]
                evValue = haValue * 27.211386245981
                homoLumoGap.append(evValue)
        
        data.insert(3,"homoLumoGap",homoLumoGap)
        data.to_csv(f'f{dataSet}', index=False)      
        
def convert(dataSet:str) -> None:
    with open(dataSet, "r", newline='') as myfile:
        data = pd.read_csv(myfile)
        propList = data["homoLumoGap"].to_list()
        for i in range(len(propList)):
            propList[i] = propList[i] * 27.211386245981
        data.drop(columns='homoLumoGap', axis='columns',inplace=True)
        data.insert(4,"homoLumoGap",propList)
        data.to_csv(f'{dataSet}', index=False)      


convert("validationSet(400).csv")
convert("testSet(1600).csv")
convert("trainingSet(10000).csv")
