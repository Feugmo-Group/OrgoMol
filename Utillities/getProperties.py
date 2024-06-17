import pandas as pd
from tqdm import tqdm

#have property arg because in the future may wanna use this to get different properties
#right now getting homo-luma gap
#weird thing with pandas writing an extra column! Need to fix!

def getProperties(dataSet:str, property = None):
    with open(dataSet, "r", newline='') as myfile:
        data = pd.read_csv(myfile)
        homoLumoGap = []
        for row in tqdm(data['fileName']):
            rowRaw, *_ = row.partition(".")
            with open(f'{rowRaw}.xyz') as xyz:
                homoLumoGap.append(xyz.readlines()[1].split()[9])
        
        data.insert(3,"homoLumoGap",homoLumoGap)
        data.to_csv(f'f{dataSet}')      
        
getProperties("(Validated)trainingSet.csv")