
def makeList(numOfFiles):
    
    numberOfFiles = range(1,numOfFiles)
    with open("QM9Index.txt","w") as g:

        for i in numberOfFiles:
            if 10>i>=0:
                g.write(f'dsgdb9nsd_00000{i}.xyz\n')
            elif 100>i>=10:
                g.write(f'dsgdb9nsd_0000{i}.xyz\n')
            elif 1000>i>-100:
                g.write(f'dsgdb9nsd_000{i}.xyz\n')
            elif 10000>i>1000:
                g.write(f'dsgdb9nsd_00{i}.xyz\n')
            elif 100000 > i >= 10000:
                g.write(f'dsgdb9nsd_0{i}.xyz\n')
            elif i > 100000:
                g.write(f'dsgdb9nsd_{i}.xyz\n')
        

def molToIndex(file):
    with open(file, "r") as g, open("QM9Index.txt","w") as f:
        for line in g:
            f.write(f'{line.strip()}.zmat\n')

molToIndex("molecules_all")