import gcutil
import numpy as np
from scipy.spatial.distance import cdist
from p_tqdm import p_map
import csv
import pandas as pd

#goal is to convert zmat back to xyz
#then check whether the new xyz files are the same as the old xyz files
#what metric to use
#keep track of which ones pass the test and which ones do not
#maybe write this to a csv file??

#as of now when I am doing the RMSD I am assuming that both matrices have the same labels for each entry ie poisiton 1,1 corresponds to the same atom for both which may be eggregious
#also assuming matrixes are same shape

#No point in doing rMSD and I might delete the implementation if I check it and find out it's wrong because a better metric that is permutation variant is the distance matrix for which
#i already have a function

#Current way of comparing. Compute distance matrix for both xyz files. Then using Frobenius Norm to see if they are the same or not
#can also use correlation coefficent or compare entry wise

#SHOWING POSITIVE RESULTS!! Matrixes have same Frobenius norm. Now for future datasets, not this one because this one the conversion conserves the permutation, but for future need to allign
#the permutation before computing norm



def getArgs(zmatInput:str,xyzInput:str) -> None:
    with open(zmatInput, "r"), open(xyzInput,"r"):
        zmatAtomNames, rconnect, rlist, aconnect, alist, dconnect, dlist = gcutil.readzmat(zmatInput)
        xyzarr, xyzAtomNames = gcutil.readxyz(xyzInput)
        zarr =  gcutil.write_xyz(zmatAtomNames,rconnect,rlist,aconnect,alist,dconnect,dlist)
        return zarr, xyzarr
            
def matrixInner(mat1,mat2)-> int:
    mat2T = np.transpose(mat2)
    return np.trace(np.matmul(mat2T,mat1))

def zmatToXyz(zmatInput:str) -> None:
    with open(zmatInput, "r"):
        atomnames, rconnect, rlist, aconnect, alist, dconnect, dlist =  gcutil.readzmat(zmatInput)
        gcutil.write_xyz(atomnames, rconnect, rlist, aconnect, alist, dconnect, dlist, f'{zmatInput}Conv.xyz')

def rMSD(mat1:np.ndarray,mat2:np.ndarray,numOfAtoms:int) -> float:
    mat1Reduced = []
    mat2Reduced = []
    for n in np.nditer(mat1): 
        mat1Reduced.append(n)
    for x in np.nditer(mat2):
        mat2Reduced.append(x)
    array1, array2 = np.array(mat1Reduced), np.array(mat2Reduced)
    rMSD = np.sqrt((sum((array1-array2)**2))*(1/numOfAtoms))
    return rMSD

def distanceMatrix(arg1:np.ndarray, arg2:np.ndarray) -> np.ndarray:
    return cdist(arg1,arg2)
    
def compare(file1:str,file2:str) -> bool:      
    zarr,xyzarr =  getArgs(file1,file2)
    zDMatrix = distanceMatrix(zarr,zarr)
    xDMatrix = distanceMatrix(xyzarr,xyzarr)
    zNorm = np.linalg.norm(zDMatrix)
    xNorm = np.linalg.norm(xDMatrix)
    if (zNorm-xNorm)**2 < 0.01:
      return True
    else:
        return False
    
    
#now just need a way to compare 128,744 files quickly! and store the results somewhere
#current validation function is not really paralllelizable, not sure how much of an issue this is considering that you would not run this calculation many times

def validation(dataSet:str) -> None:
      with open(dataSet, "r", newline = '') as myfile:
            data = pd.read_csv(myfile, delimiter=',', header= None, skip_blank_lines=True, names=['fileName','text'])
            status = []
            for row in data['fileName']:
                rawRef, *_ = row.partition(".")
                status.append(str(compare(f'{rawRef}.zmat',f'{rawRef}.xyz')))
                
            data.insert(2,"status",status)
            data.to_csv(dataSet, sep = ",", )
            
validation("testSet.csv")