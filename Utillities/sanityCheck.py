import gcutil

#goal is to convert zmat back to xyz
#then check whether the new xyz files are the same as the old xyz files
#keep track of which ones pass the test and which ones do not
#maybe write this to a csv file??

def compare(zmatInput:str,xyzInput:str) -> None:
    with open(zmatInput, "r") as infile, open(xyzInput,"r") as outfile:
        zmatAtomNames, rconnect, rlist, aconnect, alist, dconnect, dlist = gcutil.readzmat(input)
        xyzarr, xyzAtomNames = gcutil.readxyz(xyzInput)
        print(zmatAtomNames, xyzAtomNames)
        print(xyzarr)
        
compare("dsgdb9nsd_000001.zmat", "dsgdb9nsd_000001.xyz")
        
        