import gcutil
import numpy as np

#goal is to convert zmat back to xyz
#then check whether the new xyz files are the same as the old xyz files
#what metric to use
#keep track of which ones pass the test and which ones do not
#maybe write this to a csv file??

#as of now when I am doing the RMSD I am assuming that both matrices have the same labels for each entry ie poisiton 1,1 corresponds to the same atom for both which may be eggregious
#also assuming matrixes are same shape

def getArgs(zmatInput:str,xyzInput:str) -> None:
    with open(zmatInput, "r"), open(xyzInput,"r"):
        zmatAtomNames, rconnect, rlist, aconnect, alist, dconnect, dlist = gcutil.readzmat(zmatInput)
        xyzarr, xyzAtomNames = gcutil.readxyz(xyzInput)
        zarr =  gcutil.write_xyz(zmatAtomNames,rconnect,rlist,aconnect,alist,dconnect,dlist)
        return zarr, xyzarr
            
def inner(mat1,mat2)-> int:
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
      
zarr,xyzarr =  getArgs("dsgdb9nsd_000001.zmat", "dsgdb9nsd_000001.xyz") 
print(rMSD(zarr,xyzarr,5))    
  
    

       