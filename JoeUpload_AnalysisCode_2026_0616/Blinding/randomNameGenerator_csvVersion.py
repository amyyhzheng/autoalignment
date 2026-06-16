import os
from os import path
# import xlsxwriter
import csv
import shutil
import random
from distutils.dir_util import copy_tree

def getFilePath():
    path = input("What is the path of folder containing ONLY the files to be blinded: ")
    while (os.path.isdir(path) == False):
        path = input("Path does not exist. Enter a valid path of folder containing ONLY the files to be blinded: ")
    return path


def rename(original, new, path):
    newname = path  + new + os.path.splitext(original)[1]
    oldname = path + original
    os.rename(oldname, newname)
    return newname


path = getFilePath()
# shuffled = os.listdir(path)
all_files = os.listdir(path)
shuffled = [f for f in all_files if f != "blinded" and f != "randomNameGenerator.py"]
random.shuffle(shuffled)

# Get user preferences
prefix = input("What prefix do you want to add to the blinded names (e.g. If desired file names are SOM_Image#, input SOM_Image). If none, press Enter.")
withinFolders = input("Do you want the blinded files to be sorted into separate folders (not applicable to folders)? Y/N ")


# Copy all contents in directory to a new folder 'blinded'
shutil.copytree(path, path + '/blinded')

# Make excel
# excelDir = path + '/DO_NOT_OPEN_blindingInfo.xlsx'
# workbook = xlsxwriter.Workbook(excelDir)
# worksheet = workbook.add_worksheet()
# worksheet.write('A1', 'Unblinded Name')
# worksheet.write('B1', 'Blinded Name')

# Make CSV
blindedCsv = path + "\DO_NOT_OPEN_blindingInfo.csv"

fields = ['Unblinded Name', 'Blinded Name']

with open(blindedCsv, 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(fields)

    # Blinding
    blinded_path = path + '/blinded/'

    for i in range(len(shuffled)):
        filepath = rename(shuffled[i], str(prefix) + str(i), blinded_path)

        # Place files (not folders) in respective individual folders
        if withinFolders.lower() == "y" and os.path.isdir(filepath) == False:
            os.mkdir(blinded_path + str(prefix) + str(i))
            
            shutil.move(filepath, blinded_path + str(prefix) + str(i))

        # Add to excel
    #     worksheet.write('A' + str(i+2), str(shuffled[i]))
    #     worksheet.write('B' + str(i+2), str(prefix) + str(i))
    # workbook.close()

        # Add to csv
        rows = [[str(shuffled[i]) ,str(prefix)+str(i)]]
        csvwriter.writerows(rows)



print("Completed blinding! The blinded files are in the folder named 'blinded'. The excel sheet 'DO_NOT_OPEN_blindinginfo' should only be opened after analysis. Good luck!")
