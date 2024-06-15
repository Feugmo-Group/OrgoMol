import gcutil
import numpy as np

#goal is to convert zmat back to xyz
#then check whether the new xyz files are the same as the old xyz files
#what metric to use
#keep track of which ones pass the test and which ones do not
#maybe write this to a csv file??

def getArgs(zmatInput:str,xyzInput:str) -> None:
    with open(zmatInput, "r"), open(xyzInput,"r"):
        zmatAtomNames, rconnect, rlist, aconnect, alist, dconnect, dlist = gcutil.readzmat(zmatInput)
        xyzarr, xyzAtomNames = gcutil.readxyz(xyzInput)
        zarr =  gcutil.write_xyz(zmatAtomNames,rconnect,rlist,aconnect,alist,dconnect,dlist)
        return zarr, xyzarr,zmatAtomNames,xyzAtomNames
            
def inner(mat1,mat2)-> int:
    mat2T = np.transpose(mat2)
    return np.trace(np.matmul(mat2T,mat1))

def zmatToXyz(zmatInput:str) -> None:
    with open(zmatInput, "r"):
        atomnames, rconnect, rlist, aconnect, alist, dconnect, dlist =  gcutil.readzmat(zmatInput)
        gcutil.write_xyz(atomnames, rconnect, rlist, aconnect, alist, dconnect, dlist, f'{zmatInput}Conv.xyz')

def rMSD(mat1:np.ndarray,mat2:np.ndarray,numOfAtoms) -> float:
    return True

print(getArgs("dsgdb9nsd_000001.zmat","dsgdb9nsd_000001.xyz"))
    

       