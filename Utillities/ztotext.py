import gcutil
from p_tqdm import p_umap


def xyzToZmat(xyzfilename:str) -> None:
    """ Converts .xyz file to a .zmat file"""

    with open(xyzfilename,"r"):
        outName = xyzfilename.split(".")[0] + ".zmat"
        xyzarr, atomnames = gcutil.readxyz(xyzfilename)
        distmat = gcutil.distance_matrix(xyzarr)
        return gcutil.write_zmat(xyzarr, distmat, atomnames,outName)


def zmatToText(zmatfilename) -> None:
    outName = zmatfilename.split(".")[0] + ".txt"
    with open(zmatfilename, "r") as myfile, open(outName,"w") as f:
        atomNum = 1
        for line in myfile.readlines(): #this could maybe be rewritten with enumerate??! save you the counter variable don't really know how that helps
            splitLine = line.split()
            if len(splitLine) == 1:
                typeFirst(splitLine, f, atomNum)
            elif len(splitLine) == 3:
                typeSecond(splitLine,f,atomNum)
            elif len(splitLine) == 5:
                typeThird(splitLine,f, atomNum)
            elif len(splitLine) == 7:
                typeFourth(splitLine,f, atomNum)
            atomNum += 1

def typeFirst(line, outFile, atomNum):
    outFile.write(f"Atom number {atomNum} is {line[0]}. ")

def typeSecond(line,outFile, atomNum):
    outFile.write(f"Atom number {atomNum} is {line[0]}. It is bonded to atom number {line[1]} and the internuclear distance between them is {line[2]}. ")

def typeThird(line,outFile, atomNum): 
    outFile.write(f"Atom number {atomNum} is {line[0]}. It is bonded to atom number {line[1]} and the internuclear distance between them is {line[2]}. The bond angle between atom numbers {atomNum}, {line[3]} and {line[1]} is {line[4]}. ")

def typeFourth(line,outFile, atomNum):
    outFile.write(f"Atom number {atomNum} is {line[0]}. It is bonded to atom number {line[1]} and the internuclear distance between them is {line[2]}. The bond angle between {atomNum}, {line[3]} and {line[1]} is {line[4]}. The dihederal angle between {atomNum}, {line[5]}, {line[3]} and {line[1]} is {line[6]}. ")

if __name__ == '__main__':
    with open("QM9Index.txt", "r") as myfile:
        index = myfile.read().splitlines()
        p_umap(zmatToText,index)


