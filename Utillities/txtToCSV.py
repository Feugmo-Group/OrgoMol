import os
import csv

def filesToCSV(directory:str, output:str) -> None:
    with open(output, "w", newline='') as outfile:
        for file in os.listdir(directory):
            f = os.path.join(directory, file) 
            if os.path.isfile(f):
                with open(file, "r") as infile:
                    outWriter = csv.writer(outfile, delimiter=",")
                    outWriter.writerow([file, infile.readline()])

filesToCSV(os.getcwd(), "trainingSet.csv")
              
        