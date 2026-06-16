# Goal: Export xyz coordinates for every point along a path. 
# Need: .traces file of traced branch. Can be .traces file for one branch or an entire branch arbor. 
# Code obtained from: https://forum.image.sc/t/exporting-x-y-z-coordinates-for-every-point-along-a-path/4732/4
# Modified and finalized by Phoebe - 03/05/22

#@ File (label="Select a directory to save .csv file", style="directory") csv_dir
#@File (label="Select the .traces file", style="file") traces_file

#@LogService log
#@UIService ui

import os 
import csv
from sc.fiji.snt import Tree
from org.scijava.log import LogLevel

os.chdir(str(csv_dir))
filename_withExtension = os.path.basename(str(traces_file)) # demo.traces
filename = os.path.splitext(filename_withExtension)[0] # demo
originalname = filename_xyz = filename + "_xyzCoordinates" # demo_xyzCoordinates

count = 1
while os.path.exists(filename_xyz + ".csv") == True:
	print(filename_xyz + ".csv already exists.")
	filename_xyz = originalname + "(" + str(count) + ")"
	count +=1

# load traces file: since file may contain one or more rooted structures
# (trees), we'll use a convenience method that handles multiple trees in
# a single file (or directory). For details:
# https://javadoc.scijava.org/Fiji/sc/fiji/snt/Tree.html
trees = Tree.listFromFile(str(traces_file))
if not trees:
    ui.showDialog("Could not retrieve coordinates. Not a valid file?")
else:
	data = []
	for tree in trees:
		log.log(LogLevel.INFO, 'Processing %s...' % tree.getLabel())
		
		for node in tree.getNodes():
			log.log(LogLevel.INFO, node)
			print(node.onPath, node.x, node.y, node.z)
			data.append([node.onPath, node.x, node.y, node.z])
	ui.getDefaultUI().getConsolePane().show()
	
# save coordinates in CSV file
with open(filename_xyz + '.csv', 'wb') as csvfile:
	csvwriter = csv.writer(csvfile)
	header = ['path', 'x', 'y', 'z']
	csvwriter.writerow(header)
	csvwriter.writerows(data)
        	
# To show the raw coordinates in console.		
    
print("Saved coordinates in " + filename_xyz + ".csv")