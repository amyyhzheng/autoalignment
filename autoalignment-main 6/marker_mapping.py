''' README
There are three sections to this entire alignment code: 1.) computational 2.) clustering 3.) mapping
1.) Computational: This part gets the raw branch and marker information. The raw data is then processed (i.e. normalized, converted from pixels to um).
    We then break a branch into segments based on the fiducials, and scale each segment differently.
    The goal of this section is so that we'll have a standardized way (scaled distances) to compare markers across imaging sessions.
2.) Clustering: Given the scaled distances for markers across imaging sessions, we then group them into different clusters.
    Clustering is done separately for spine and shaft synapses. The points in each cluster should be within 2um of each other.
    Since starting the kmeans clustering algorithm with different timepoints as the initial centroid seed could yield different clustering results, we determined the best cluster by averaging the standard deviation of all the clusters and finding the cluster with the closest standard deviation to the average stdev. (Yes, this is kind of arbirtrary since if there is only ONE true cluster, the average is influenced by all the false clusters)
    The goal is of this section is to group and assign marker IDs to each marker so that they match across session.
3.) Mapping: Given the grouping information, assign IDs and map each marker back to their raw coordinates.
    The synapse markers are mapped back to its original coordinate (i.e. where we placed them initially)
    The nothing markers are calculated using the average distance for each cluster.
    The goal is to generate a .csv file that contains markerID and its xyz information for all synapse and nothing markers across all imaging sessions.
'''


'''
Docstring comments are Amy's 6/25/24
Line comments are Phoebe's unless otherwise labelled

'''
#################### USER CAN CHANGE THE FOLLOWING TO MATCH THEIR NEEDS ####################
inhibitoryshaft = "InhibitoryShaft" # This should match what is in the objectJ tool bar (not case sensitive)
inhibitoryspine = "spinewithInhsynapse" # This should match what is in the objectJ tool bar (not case sensitive)
scaling_factor = [0.25, 0.25, 1] # This is the scaling factor from pixels to um (i.e. x and y are 0.25um per pixel while z step-size is 1 um per pixel)
num_channels = 4 # This is the number of channels in the .tif file (this will affect the z scaling)
ojj_tifName = "Image" # In the CombinedResults.csv for markers, the entries in the "ojj File Name" column should contain the word "Image" (i.e. SOM022_b2_Image1 or SOM022_b2_Image2). Change to something else if it doesn't contain the word Image#. For example, change to "timepoint" if it reads "SOM022_b2_timepoint1"

""" Comment out one of the SNT_branchName or make your own """
SNT_branchName = "b%s" # This should match the SNT traced branch name (i.e. b1 )
# SNT_branchName = "Path (%s)" # This should match the SNT traced branch name (i.e. Path (1) )



####################################################################  CODE ####################################################################
""" IMPORTING PACKAGES AND MODULES """
# For handling directory and getting user input
import os, sys
import tkinter as tk
from tkinter import Y, simpledialog, filedialog
from tkinter.filedialog import askopenfilename

# For handling .csv files, data, and computation
import csv, math
import pandas as pd
import numpy as np
import napari 
import tifffile as tif

# For plotting
import matplotlib.pyplot as plt 
from collections import defaultdict

from visualization import process_and_add_splines
from scipy.interpolate import splprep, splev

""""""""""""""""""""""""""""""""""""""""""
"""      SECTION #1: COMPUTATIONAL     """
"""  Goal: Scale the marker distances  """
""""""""""""""""""""""""""""""""""""""""""
#################### FUNCTIONS ####################
# Primarily for the computational section, though clustering and mapping sections also use some of these functions)

# Function that gets user inputs

print('running')
def getUserInput():
    xyz_fileNames = {}
    tk.Tk().withdraw()
    '''
    IMPORTANT! -- GUI code is commented out here and replaced with paths 
    for ease of debugging - Uncomment this and comment out pathname code 
    for different paths. 
    '''
    # number_of_timepoints = simpledialog.askinteger(title="Timepoints", prompt="Enter the number of imaging sessions (i.e. Enter 6 for six imaging sessions). Note that Image1 should equal Timepoint 1.")
    # animal_ID = simpledialog.askstring(title="Animal ID", prompt="Enter animal ID? (i.e. SOM022)")
    # branch_ID = simpledialog.askstring(title="Branch ID", prompt="What branch do you want to analyze? (i.e. Enter 2 for branch/path ID 2.)")
    # marker_filename = askopenfilename(title="Select the CombinedResults.csv for markers on branch " + str(branch_ID))
    # fiducial_fileName = askopenfilename(title="Select the CombinedResults.csv for the fiducials on branch " + str(branch_ID))

    number_of_timepoints = 6
    animal_ID = 'SOM022'
    branch_ID = '2' #Not really checked

    '''
    
    FOR KENDYLL JOE EDIT FILE PATHS HERE
    
    '''    

    #Commented out below is the objectj filenames. 
    # marker_filename = '/Volumes/nedividata/Amy/files_for_amy_fromJoe/example_analysis_SOM022/PunctaScoring/b2/SynapseMarkers/beforeAlignment/CombinedResults.csv'
    # fiducial_fileName = '/Volumes/nedividata/Amy/files_for_amy_fromJoe/example_analysis_SOM022/PunctaScoring/b2/Fiducials/CombinedResults.csv'

    '''
    Napari version input all of your markers as a list here
    (Separate each file path with a comma) 
    '''
    marker_filenames = ['/Volumes/nedividata/Amy/files_for_amy_fromJoe/example_analysis_SOM022/PunctaScoring/b2/SynapseMarkers/beforeAlignment/CombinedResults.csv',
                        '/Volumes/nedividata/Amy/files_for_amy_fromJoe/example_analysis_SOM022/PunctaScoring/b2/SynapseMarkers/beforeAlignment/CombinedResults.csv'
                       ] #Example list. Input a file from each timepoint(note that landmarks can be included in this and should have type = "Landmark"
    # fiducial_fileName = '/Volumes/nedividata/Amy/files_for_amy_fromJoe/example_analysis_SOM022/PunctaScoring/b2/Fiducials/CombinedResults.csv'

    xyz_fileNames['Timepoint 1'] = '/Volumes/nedividata/Amy/files_for_amy_fromJoe/example_analysis_SOM022/SNT_Tracing/Image1/SOM022Image1FullTrace_withbrancheslabeled_xyzCoordinates.csv'
    xyz_fileNames['Timepoint 2'] = '/Volumes/nedividata/Amy/files_for_amy_fromJoe/example_analysis_SOM022/SNT_Tracing/Image2/SOM022_Image2_fulltrace_withbrancheslabeled_xyzCoordinates.csv'
    xyz_fileNames['Timepoint 3'] = '/Volumes/nedividata/Amy/files_for_amy_fromJoe/example_analysis_SOM022/SNT_Tracing/Image3/SOM022Image3_fulltrace_withbrancheslabeled_xyzCoordinates.csv'
    xyz_fileNames['Timepoint 4'] = '/Volumes/nedividata/Amy/files_for_amy_fromJoe/example_analysis_SOM022/SNT_Tracing/Image4/SOM022Image4_fulltrace_withbrancheslabeled_xyzCoordinates.csv'
    xyz_fileNames['Timepoint 5'] = '/Volumes/nedividata/Amy/files_for_amy_fromJoe/example_analysis_SOM022/SNT_Tracing/Image5/SOM022Image5_fulltrace_withbrancheslabeled_xyzCoordinates.csv'
    xyz_fileNames['Timepoint 6'] = '/Volumes/nedividata/Amy/files_for_amy_fromJoe/example_analysis_SOM022/SNT_Tracing/Image6/SOM022Image6fulltrace_withbrancheslabeled_xyzCoordinates.csv'
    # for timepoint in range(number_of_timepoints):
    #     xyz_fileNames["Timepoint " + str(timepoint+1)] = askopenfilename(title="Select the .csv file with the branch(es) xyz coordinates for time point " + str(timepoint+1))
    # for timepoint in range(number_of_timepoints):
    #     xyz_fileNames["Timepoint " + str(timepoint+1)] = askopenfilename(title="Select the .csv file with the branch(es) xyz coordinates for time point " + str(timepoint+1))
    export_csv_directory = filedialog.askdirectory(title="Choose the folder to save the exported .csv ")
    return number_of_timepoints, branch_ID, xyz_fileNames, marker_filenames, export_csv_directory, fiducial_fileName, animal_ID





# Function that takes in the branch coordinate .csv files and returns the xyz coordinates for the specified branch
def getBranchCoordinates(xyz_fileNames, branch_ID):
    '''
    Input:
    xyz_fileNames(dictionary): Dictionary mapping from 'Timepoint _': file_path for timepoint
                            Represents branch coordinates (branch coordinates are each point along branch)
    branch_ID(int)

    Output:
    raw_branch_coordinates(list): A list of lists of tuples containing(x, y, z) 
                                such that each list has the branch coordinates for a separate 
                                timepoint. 


    '''
    raw_branch_coordinates = []
    for timepoint in range(len(xyz_fileNames)):
        timepoint_list = []
        file_path = xyz_fileNames['Timepoint ' + str(timepoint+1)]
        file = pd.read_csv(file_path)
        branch = file.loc[file['path'] == SNT_branchName%branch_ID] 

        # for i in range(len(branch)):
        #     x = branch["x"].iloc[i]
        #     y = branch["y"].iloc[i]
        #     z = branch["z"].iloc[i]
        #     timepoint_list.append(tuple([x, y, z]))
            # Fit spline
        x, y, z = branch["x"].values, branch["y"].values, branch["z"].values
        tck, u = splprep([x, y, z], s=0)
        u_new = np.linspace(0, 1, 100)
        x_new, y_new, z_new = splev(u_new, tck)
        timepoint_list = list(zip(x_new, y_new, z_new))
        '''
        Changed to include all of
        '''

        # Store results
        # timepoint_list.append(formatted)


    #         # Check if there are enough points for spline (we need at least 2)
    # if points.shape[0] > 1:
    #     tck, u = splprep(points.T, s=0)  # Compute B-spline representation
    #     u_new = np.linspace(0, 1, num=10)  # Interpolation with 10 points
    #     x_new, y_new, z_new = splev(u_new, tck)  # Evaluate spline

    #     # Stack in z-x-y order
    #     interpolated_points = np.stack([z_new, x_new, y_new], axis=1)
    # else:
    #     interpolated_points = points  # If only one point, keep it as is

    # timepoint_list.append(interpolated_points)

        raw_branch_coordinates.append(timepoint_list)
        print("Successfully saved " + str(len(timepoint_list)) + " branch coordinates in Timepoint " + str(timepoint+1))
    return raw_branch_coordinates


# Function that reads the markers CombinedResults.csv and returns the marker type and marker coordinates, where Z is imageJ Z and not objectJ Z, across all imaging sessions.
# def getMarkersInfo(marker_fileName, number_of_timepoints):
#     '''
#     Input: 
#         marker_fileName(str): file path to CombinedResults for markers,
#         number_of_timepoints(int): Number of timepoints being analyzed
#     Output: 
#         raw_markers(list): List of lists where each list contains all markers from 
#         each timepoint. Markers have format (markertype, (x, y, z))
#         All markers with label 'Landmark' are excluded. All z coordinates 
#         transformed from objectJ to imageJ coordinates. 
    
#     '''
#     # raw_markers = []
#     # file = pd.read_csv(marker_fileName)

#     # for timepoint in range(number_of_timepoints):
#     #     image = file.loc[file['ojj File Name'].str.contains("_" + ojj_tifName + str(timepoint+1), na=False)] # na=False in case there are rows with no/missing values
#     #     timepoint_list = []
#     #     for i in range(len(image)):
#     #         markerType = image["Final S1"].iloc[i]
#     #         x = image["xpos S1"].iloc[i]
#     #         y = image["ypos S1"].iloc[i]
#     #         objectJ_z = image["zpos S1"].iloc[i]
#     #         imageJ_z = transformZ_ObjectJtoImageJ(objectJ_z) # Converts from objectJ to imageJ Z information
#     #         markerInfo = (markerType, (int(x), int(y), imageJ_z))
#     #         timepoint_list.append(tuple(markerInfo))

#     #     new_markerList_noLandmarks = [item for item in timepoint_list if item[0] != 'Landmark']
#     #     raw_markers.append(new_markerList_noLandmarks)
#     #     print("Successfully parsed the marker type and coordinates of " + str(len(new_markerList_noLandmarks)) + " synapse markers (excluding landmarks) for Timepoint/Image " + str(timepoint+1))
#     raw_markers = []
#     file = pd.read_csv(marker_fileName)

#     for timepoint in range(number_of_timepoints):
#         # Napari Points typically don't encode filenames directly; adjust logic if your pipeline adds them
#         image = file[file['name'].str.contains("_" + ojj_tifName + str(timepoint+1), na=False)] if 'name' in file.columns else file

#         timepoint_list = []
#         for i in range(len(image)):
#             if 'markerType' in image.columns:
#                 markerType = image["markerType"].iloc[i]
#             elif 'label' in image.columns:
#                 markerType = image["label"].iloc[i]
#             else:
#                 raise ValueError("No column found for marker type. Expected 'markerType' or 'label'.")

#             x = image["axis-0"].iloc[i]
#             y = image["axis-1"].iloc[i]
#             z = image["axis-2"].iloc[i] if 'axis-2' in image.columns else 0
#             imageJ_z = transformZ_ObjectJtoImageJ(z)
#             markerInfo = (markerType, (int(x), int(y), imageJ_z))
#             timepoint_list.append(tuple(markerInfo))

#         new_markerList_noLandmarks = [item for item in timepoint_list if item[0].lower() != 'landmark']
#         raw_markers.append(new_markerList_noLandmarks)
#         print("Parsed " + str(len(new_markerList_noLandmarks)) + " markers (excluding landmarks) for Timepoint/Image " + str(timepoint+1))
    
#     return raw_markers
#     # return raw_markers

def getMarkersInfo(marker_file_list):
    """
    Input: 
        marker_file_list (list): List of file paths to Napari-style CSVs, one per timepoint.
    Output: 
        raw_markers (list): List of lists of (markertype, (x, y, z)) for synapse markers.
        raw_fiducials (list): List of lists of (markertype, (x, y, z)) for fiducials (type == 'landmark').
    """
    raw_markers = []
    raw_fiducials = []

    for timepoint, marker_fileName in enumerate(marker_file_list):
        file = pd.read_csv(marker_fileName)

        timepoint_markers = []
        timepoint_fiducials = []

        for i in range(len(file)):
            if 'type' in file.columns:
                markerType = file["type"].iloc[i]
            elif 'label' in file.columns:
                markerType = file["label"].iloc[i]
            else:
                raise ValueError("No column found for marker type. Expected 'type' or 'label'.")

            z = file["axis-0"].iloc[i]
            x = file["axis-1"].iloc[i]
            y = file["axis-2"].iloc[i] if 'axis-2' in file.columns else 0
            imageJ_z = transformZ_ObjectJtoImageJ(z)

            markerInfo = (markerType, (x, y, imageJ_z))

            if markerType.lower() == "landmark":
                timepoint_fiducials.append(markerInfo)
            else:
                timepoint_markers.append(markerInfo)

        raw_markers.append(timepoint_markers)
        raw_fiducials.append(timepoint_fiducials)

        print(f"Timepoint {timepoint+1}: {len(timepoint_markers)} markers, {len(timepoint_fiducials)} fiducials parsed.")

    return raw_markers, raw_fiducials


# Function that reads the CombinedResults.csv for fiducials and returns the fiducial coordinates across all imaging sessions in imageJ Z (not objectJ Z)
def getFiducialInfo(fiducial_fileName, number_of_timepoints):
    '''
    Input: 
        fiducial_fileName(str): file name for the fiducials 
        number_of_timepoints(int): int for number of time points

    Output:
        raw_fiducials(list): contains all fiducials of all timepoints
    '''
    # raw_fiducials = []
    # file = pd.read_csv(fiducial_fileName)
    # fiducial_number = file.Marker.count()

    # for i in range(number_of_timepoints):
    #     coordinates = []
    #     for j in range(fiducial_number):
    #         x = file['xpos S' + str(i+1)].iloc[j]
    #         y = file['ypos S' + str(i+1)].iloc[j]
    #         objectJ_z = file['zpos S' + str(i+1)].iloc[j]
    #         imageJ_z = transformZ_ObjectJtoImageJ(objectJ_z)
    #         fiducial_info = (x, y, imageJ_z)
    #         coordinates.append(tuple(fiducial_info))
    #     raw_fiducials.append(coordinates)
    #     print("Successfully saved " + str(len(coordinates)) + " fiducial coordinates for Timepoint " + str(i+1))
    # return raw_fiducials

    raw_fiducials = []
    file = pd.read_csv(fiducial_fileName)

    if 'timepoint' not in file.columns:
        raise ValueError("Expected a 'timepoint' column in the Napari fiducial file.")

    for i in range(number_of_timepoints):
        coords = []
        tp_df = file[file['timepoint'] == i]
        for _, row in tp_df.iterrows():
            x = row['axis-0']
            y = row['axis-1']
            z = row['axis-2'] if 'axis-2' in row else 0
            imageJ_z = transformZ_ObjectJtoImageJ(z)
            coords.append((x, y, imageJ_z))
        raw_fiducials.append(coords)
        print(f"Successfully saved {len(coords)} fiducial coordinates for Timepoint {i+1}")
    
    return raw_fiducials

# Functions that converts between objectJ to imageJ Z information
def transformZ_ObjectJtoImageJ(objectJ_z):
    imageJ_z = (int(objectJ_z) - 1)/num_channels
    return imageJ_z
def transformZ_ImageJtoObjectJ(imageJ_Z):
    objectJ_z = (imageJ_Z*num_channels+1)
    return objectJ_z

# Functions that scales and normalizes the raw branch and marker coordinates to its real coordinates
def get_xyzs(coordinates):
    # input: (1, 2, 3), (5, 6, 7), (2, 3, 4) and returns (1, 5, 2), (2, 6, 3), (3, 7, 4)
    return [point[0] for point in coordinates], [point[1] for point in coordinates], [point[2] for point in coordinates]

# Function that normalizes and scales the coordinates
def normalize_and_scale(branch, markers, fiducials, scale):
    min_branch = min(branch) # Center graph at beginning
    return [(x - min_branch) * scale for x in branch], [(x - min_branch) * scale for x in markers],  [(x - min_branch) * scale for x in fiducials]

# Function that calculates the closest distance between a marker and branch point
def getMin(distances_to_branch):
    min_distance = min(distances_to_branch)
    min_index = distances_to_branch.index(min_distance)
    return min_index

# Function that maps the fiducial or branch coordinate to the closest branch point
def getBranchIndexes(coordinates_to_map_to_branch, i):
    '''
    Input:
        coordinates_to_map_to_branch(list): list of lists of coordinates branches to 
        map i.e. branch points as (x, y, z) i.e. normalized_branch_coordinates
        format
        i: Timepoint of examination 

    Output:
        indexes_per_timepoint(list): Outputs a list where each the element at
        each index corresponds to the branch point index closes to the point
        in coordinates_to_map_to_branch that is the closest
    
    Computes all distances to every single branch point and
    outputs the point with minimum distance. Note that this only considers
    their distance in xy and this is the distance only to branch points. 

    '''
    indexes_per_timepoint = []
    for coordinate in coordinates_to_map_to_branch[i]:
        #Amy: We loop through coordinates within that time point
        distances_toBranch = []
        for branch_point in normalized_branch_coordinates[i]:
            distance = dist_between_two_points_in_xy(coordinate, branch_point)
            distances_toBranch.append(distance)
        min_dist_index = getMin(distances_toBranch)
        indexes_per_timepoint.append(min_dist_index)
    return indexes_per_timepoint

# Function that calculates the euclidean distance in xy only between two points 
def dist_between_two_points_in_xy(start, end):

    distance = math.sqrt((end[0]-start[0])**2 + (end[1]-start[1])**2)
    return distance

# Function that calculates the distance along branch given two points
def dist_along_branch(branch, start, end):
    '''
    Input: 
        branch(list): Contains all coords of points along input branch
        start(int): index of start point on branch
        end(int): index of end point on branch

    Output:
        distances(float): sum of distances along traced branch
    '''
    distances = 0
    for i in range(start, end):
        distance = dist_between_two_points_in_xy(branch[i], branch[i+1])
        distances += distance
    return distances

'''
Commented out because feels unnecessary/redundant
'''
# Function that gets the segment scaling factor
# def get_scale_factor(distances):
#     try:
#         return distances[0]/distances[1]
#     except ZeroDivisionError: # if Point N is on the same spot as Point N+1, return 0
#         print("Zero division error: the distance between two points are 0 either because different indexes have the same xy coordinates\nStopping code. Please fix fiducials or branch tracing.")
#         sys.exit("\nZERO DIVISION ERROR: the distance between two points are 0 either because different indexes have the same xy coordinates\nStopping code. Please fix fiducials or branch tracing.")

# Given list of branch indexes, return list of branch coordinates
def index_to_branch(normalized_branch_coordinates, indexes):
    branch_points_AllTimepoint = []
    for i in range(len(normalized_branch_coordinates)):
        branch_points = []
        for index in indexes[i]:
            branch_points.append(normalized_branch_coordinates[i][index])
        branch_points_AllTimepoint.append(branch_points)
    return branch_points_AllTimepoint
'''

amy ~COME BACK HERE

'''
# Function that gets a particular segment of branch
def getSegment(marker_index, segments):
    '''
    Input:
        marker_index: 
        segments: 
    Output: segments[j], j
        a tuple containing the segment timepoint list i.e. 
        just a list of the segments for a timepoint and the index within that list
        where the marker_index is 

    '''
    # TODO: what are the constraints of segments? Is it sorted?
    # Do we guarantee that the end of a segment is the start of the next segment?
    # Does the start of the first segment always start from 0? Is marker_index non-negative?
    for j in range(len(segments)):
        end = segments[j][1]
        if j == len(segments)-1:
            end = segments[j][1]+1
        if marker_index in range(segments[j][0], end):
            return segments[j], j
    raise Exception("Marker index is not in any segment?")

# Function that adds up the distsances of each segment to get the cumulative starting distance for each segment
def getCumulativeDistance(segmentDist_allTimepoints):
    cumulativeDist = []
    for timepoint in segmentDist_allTimepoints:
        cumulative_distance = 0
        cumulative_distance_list = [0]
        for scaled_seg_length in timepoint:
            cumulative_distance += scaled_seg_length
            cumulative_distance_list.append(cumulative_distance)
        cumulativeDist.append(cumulative_distance_list)
    return cumulativeDist

# Save to .csv for export
def csv_row_data(i, point):
    image = "Image" + str(i+1)
    markerType = point[1][0]
    markerX, markerY = point[1][1][0], point[1][1][1]
    branchX, branchY = point[0][0], point[0][1]
    markerZ = transformZ_ImageJtoObjectJ(point[1][1][2])
    branchZ = transformZ_ImageJtoObjectJ(point[0][2])
    return [image, markerType, markerX, markerY, markerZ, branchX, branchY, branchZ]

# Save fiducial plots
def savePlot_xy():
    
    fiducial_to_BranchPoint = index_to_branch(normalized_branch_coordinates, closest_branchIndex_forFiducial_AllTimepoints)
    colors = ['r', 'g', 'b', 'c', 'm', 'y']

    # Get maximum X and Y
    x_list = []
    y_list = []
    for i in range(len(normalized_branch_coordinates)):
        x_list.extend(list(point[0] for point in normalized_branch_coordinates[i]))
        y_list.extend(list(point[1] for point in normalized_branch_coordinates[i]))
    max_x = max(x_list)
    max_y = max(y_list)

    for i in range(len(normalized_branch_coordinates)):
        fig = plt.figure() 
        ax = fig.add_subplot(111)

        # For normalized branch
        branch_x = []
        branch_y = []
        for point in normalized_branch_coordinates[i]:
            branch_x.append(point[0])
            branch_y.append(point[1])
    
        # For normalized fiducial that we placed
        fiducial_x = []
        fiducial_y = []
        for point in normalized_fiducial_coordinates[i]:
            fiducial_x.append(point[0])
            fiducial_y.append(point[1])

        # For normalized fiducial mapped to branch
        fiducial_x_onBranch = []
        fiducial_y_onBranch = []
        for point in fiducial_to_BranchPoint[i]:
            fiducial_x_onBranch.append(point[0])
            fiducial_y_onBranch.append(point[1])

        ax.scatter(fiducial_x, fiducial_y, c="black", marker='o', label='placed fiducials')
        ax.scatter(fiducial_x_onBranch, fiducial_y_onBranch, c='red', marker='X', label='mapped fiducials')
        ax.plot(branch_x, branch_y, c=colors[i])

        ax.set_title('Timepoint ' + str(i+1))
        ax.set_xlabel('X (um)')
        ax.set_ylabel('Y (um)') 

        ax.set_xlim(-2, max_x + 5)
        ax.set_ylim(-2, max_y + 5)
        
        # Shrink current axis by 20%
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.75, box.height])
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5)) # Put a legend to the right of the current axis
        
        newpath = export_csv_directory + '/fiducialPlots'
        if not os.path.exists(newpath):
            os.makedirs(newpath)
        plt.savefig(export_csv_directory + '/fiducialPlots/' + animal_ID + '_b' + branch_ID + '_timepoint' + str(i+1) + '_fiducialPlot.png') 
    return "Exported fiducial plots to " + export_csv_directory + '/fiducialPlots'

#################### CODE ####################

# Calls functions to get branch coordinates and markers information
number_of_timepoints, branch_ID, xyz_fileNames, marker_fileNames, export_csv_directory, fiducial_fileName, animal_ID = getUserInput()
raw_branch_coordinates = getBranchCoordinates(xyz_fileNames, branch_ID) # Uses imageJ Z
raw_markers = getMarkersInfo(marker_fileNames) # Uses imageJ Z
#Would be nice to add some sort of confirmation with timepoints and number of filenames here


raw_fiducials_coordinates = getFiducialInfo(fiducial_fileName, number_of_timepoints) # Uses imageJ Z
# print("raw branch coordinates =", raw_branch_coordinates)
# print("raw markers =", raw_markers)
# print("raw fiducials coordinates =", raw_fiducials_coordinates)


# normalizing and scaling marker coordinates
'''AMY - just move this to the raw_markers function? Why do we need to loop through array twice?'''
raw_marker_coordinates = []
raw_marker_type = []

for timepoint in raw_markers:
    coordinates = [markerInfo[1] for markerInfo in timepoint]
    marker_type = [markerInfo[0] for markerInfo in timepoint]
    raw_marker_coordinates.append(coordinates)
    raw_marker_type.append(marker_type)

normalized_branch_coordinates = []
normalized_marker_coordinates = []
normalized_fiducial_coordinates = []


#Amy: Loop through each time point's raw branch coordinates
for i in range(len(raw_branch_coordinates)):
    #Amy: Separate out to be in each timepoint 
    branch = raw_branch_coordinates[i]
    marker = raw_marker_coordinates[i]
    fiducial = raw_fiducials_coordinates[i]

    #Amy: Output a list for each of x, y. z coordinates of the branch at this timepoint
    xs_branch, ys_branch, zs_branch = get_xyzs(branch)
    xs_marker, ys_marker, zs_marker = get_xyzs(marker)
    xs_fiducial, ys_fiducial, zs_fiducial = get_xyzs(fiducial)

    #Amy: Normalize each coordinate with respect to the branch and in the correct scale(um)
    normal_x_branch, normal_x_marker, normal_x_fiducial = normalize_and_scale(xs_branch, xs_marker, xs_fiducial, scaling_factor[0])
    normal_y_branch, normal_y_marker, normal_y_fiducial= normalize_and_scale(ys_branch, ys_marker, ys_fiducial, scaling_factor[1])
    normal_z_branch, normal_z_marker, normal_z_fiducial= normalize_and_scale(zs_branch, zs_marker, zs_fiducial, scaling_factor[2])

    #Amy: Reconstruct branch/marker/fiducial to contian the x,y,z coordinates in each element
    normal_branch = [(normal_x_branch[i], normal_y_branch[i], normal_z_branch[i]) for i in range(len(normal_x_branch))]
    normal_marker = [(normal_x_marker[i], normal_y_marker[i], normal_z_marker[i]) for i in range(len(normal_x_marker))]
    normal_fiducial = [(normal_x_fiducial[i], normal_y_fiducial[i], normal_z_fiducial[i]) for i in range(len(normal_x_fiducial))]

    #Amy: normalized_branch_coordinates ends up being of the same form as raw_branch_coordinates 
    #Amy: It is a list of lists where each list has the points for a timepoint
    normalized_branch_coordinates.append(normal_branch)
    normalized_marker_coordinates.append(normal_marker)
    normalized_fiducial_coordinates.append(normal_fiducial)
print("Successfully normalized and scaled all branch, marker, and fiducial coordinates for Branch " + str(branch_ID) + " across " + str(len(normalized_branch_coordinates)) + " timepoints.")
# print("normalized branch coordinates =", normalized_branch_coordinates)
# print("normalized marker coordinates =", normalized_marker_coordinates)
# print("normalized fiducial coordinates =", normalized_fiducial_coordinates)


# Save all printed information in a log
stdoutOrigin=sys.stdout 
sys.stdout = open(export_csv_directory + "/" + animal_ID + "_b" + branch_ID + "_computationalsection_Log.txt", "w")

# Get the index and coordinates of branch point with shortest distance to marker and/or fiducial
closest_branchIndex_forMarker_AllTimepoints = []
closest_branchIndex_forFiducial_AllTimepoints = []

for i in range(len(normalized_marker_coordinates)):
    #Amy:Loop through each timepoint of marker coordinates and make list of shortest distaances to branch coordinates
    closest_branchIndex_forMarker = getBranchIndexes(normalized_marker_coordinates, i)
    closest_branchIndex_forFiducial = getBranchIndexes(normalized_fiducial_coordinates, i)
    
    checked_closest_branchIndex_forFiducial = [1 if item == 0 else item for item in closest_branchIndex_forFiducial]

    closest_branchIndex_forMarker_AllTimepoints.append(closest_branchIndex_forMarker)
    closest_branchIndex_forFiducial_AllTimepoints.append(checked_closest_branchIndex_forFiducial)
print("closest branch Index for Markers =", closest_branchIndex_forMarker_AllTimepoints, "\n")
print("closest branch Index for Fiducials =", closest_branchIndex_forFiducial_AllTimepoints, "\n")

# Get branch length
raw_branch_distance_allTimepoints = []
for i in range(len(raw_branch_coordinates)):
    end = len(raw_branch_coordinates[i])-1
    raw_branch_distance_allTimepoints.append(dist_along_branch(raw_branch_coordinates[i], 0, end))
print("raw branch distance for all timepoints =", raw_branch_distance_allTimepoints, "\n")
#Amy: raw_branch_distance_allTimepoints contains list of full branch lengths for each timepoint

normalized_branch_distance_allTimepoints = []
for i in range(len(normalized_branch_coordinates)):
    end = len(normalized_branch_coordinates[i])-1
    normalized_branch_distance_allTimepoints.append(dist_along_branch(normalized_branch_coordinates[i], 0, end))
print("normalized branch distance for all timepoints =", normalized_branch_distance_allTimepoints, "\n")


######## Scaling Calculations
# Combine the start, end, and fiducial indexes and sort them in ascending order. Modify duplicates
combined_startend_indexes = []
for i in range(len(normalized_branch_coordinates)):
    start_end = [0, len(normalized_branch_coordinates[i])-1]
    start_end.extend(closest_branchIndex_forFiducial_AllTimepoints[i])
    start_end.sort()

    # Check that the first fiducial is not indexed to the same point as the start of the branch
    if start_end[0] == start_end[1]:
        start_end[1] += 1
    
    # Check that the last fiducial is not indexed to the same point as the end of the branch
    if start_end[-1] == start_end[-2]:
        start_end[-2] -= 1

    combined_startend_indexes.append(start_end)

    # Check if there are duplicate indexes
    if len(start_end) != len(set(start_end)):
        plot_msg = savePlot_xy()
        print("combined_startend_indexes =", combined_startend_indexes, "\n")
        print("ERROR: Some of the fiducials in Image " + str(i+1) + " are mapped to the same index. Check fiducialPlots to see whether to fix fiducial placement or retrace branch.")
        sys.exit(plot_msg + "\nERROR: Some of the fiducials in Image " + str(i+1) + " are mapped to the same index. Check fiducialPlots to see whether to fix fiducial placement or retrace branch.")

print("combined_startend_indexes =", combined_startend_indexes, "\n")

segment_distance_allTimepoints = []
segments_AllTimepoints = []
for i in range(len(closest_branchIndex_forMarker_AllTimepoints)):
    segment_dist = []
    segment = []
    for j in range(len(combined_startend_indexes[i])-1):
        lower = combined_startend_indexes[i][j]
        upper = combined_startend_indexes[i][j+1]
        if lower == upper:
            print("Potential error: segment start " + str(lower) + " and end " + str(upper) + " are the same point")
        segment_dist.append(dist_along_branch(normalized_branch_coordinates[i], lower, upper))
        segment.append((lower, upper))
    segment_distance_allTimepoints.append(segment_dist) #Amy: contains the distance between fiducial branch points all timepoints
    segments_AllTimepoints.append(segment) #Amy: contains a list of pairs (lower, upper) of all the fiducial indices
print("segments_AllTimepoints =", segments_AllTimepoints, "\n")
print("segment_distance_allTimepoints =", segment_distance_allTimepoints, "\n")

# Get maximum distance for each segment
'''
max_dist_perSegment construction -> Constructing a list of max distances for
each segment between the fiducials over the different timepoints 
'''
max_dist_perSegment = []
grouped_segment_distances = list(zip(*segment_distance_allTimepoints))
for each_segment in grouped_segment_distances:
    max_seg_dist = max(each_segment)
    max_dist_perSegment.append(max_seg_dist)
print("max_dist_perSegment =", max_dist_perSegment, "\n")

'''
scale_factor_allTimepoints construction ->  

A list of lists of scale factors where we divide max distance of segment
across timepoints by segment at timepoint. Each sublist represents
scale factors at timepoint 

'''

print(segment_distance_allTimepoints)
scale_factor_allTimepoints = []

for segments_in_timepoint in segment_distance_allTimepoints:
    scale_factor = []
    print(f"Processing timepoint with segments: {segments_in_timepoint}")
    maxDist_and_segmentDist = [*zip(max_dist_perSegment, segments_in_timepoint)]
    for segment_distances in maxDist_and_segmentDist:
        if segment_distances[1] ==0:
            scale_factor_current = 0.01
            # The issue in b2 was that one of the branches started at 
            # a point that was basically 0 because it was the first one. 
        else:
            scale_factor_current = segment_distances[0]/segment_distances[1]
            # print(f"scale_factor_current: {scale_factor_current}")
        scale_factor_allTimepoints.append(scale_factor)
        # except ZeroDivisionError:
        #     print(f"segment_distances: {segment_distances}")
        #     print("ERROR: There is a problem with the scale factor calculation")
        #     sys.exit("ZERO DIVISION ERROR...STOPPING CODE")
        scale_factor.append(scale_factor_current)

print("scale_factor_allTimepoints =", scale_factor_allTimepoints, "\n")

'''
segment_length_scaled_allTimepoints construction ->

Create a list of lists where each list is a timepoint. Scale by scalefactors
from previous computation. (Essentially scale so that every single segment
is the max length) -> Amy: why is this necessary did we not just create timepoints# 
duplicates of the same list with the maximum of each one 
'''            #Boolean Stating g + (f from goal to start) - (h to start) is not as good as our best
segment_length_scaled_allTimepoints = []
for i in range(len(segment_distance_allTimepoints)):
    scaled_segment = []
    for eachDistance_and_Scale in (zip(segment_distance_allTimepoints[i], scale_factor_allTimepoints[i])):
        scaled_segment.append(eachDistance_and_Scale[0]*eachDistance_and_Scale[1])
    segment_length_scaled_allTimepoints.append(scaled_segment)


'''
cumulative_distance_unScaled_allTimepoints ->
cumulative_distance_allTimepoints ->

Each sublist is a timepoint. Components of list are the segments 
added together at each segment.

'''


cumulative_distance_unScaled_allTimepoints = getCumulativeDistance(segment_distance_allTimepoints)
cumulative_distance_allTimepoints = getCumulativeDistance(segment_length_scaled_allTimepoints)
print("cumulative_distance_unScaled_allTimepoints =", cumulative_distance_unScaled_allTimepoints, "\n")
print("cumulative_distance_allTimepoints (scaled) =", cumulative_distance_allTimepoints, "\n")


'''

'''
# mapping_dictionary = []
marker_info = []
final_marker_distance = []
rawDist_map_scaledDist_allTimepoints = []
newpath = export_csv_directory + '/markerInfo'
if not os.path.exists(newpath):
    os.makedirs(newpath)
for i in range(len(closest_branchIndex_forMarker_AllTimepoints)):
    marker_timepoint = []
    marker_dist = []
    rawDist_map_scaledDist = []
    for marker_index in closest_branchIndex_forMarker_AllTimepoints[i]:
        '''
        Loop through each branch index that is closest to each marker
        Find the segment(from fiducials) that contains this branch index
        Calculate distance from beginning of segment and scale by relevant scaling factor
        Add to cumulative distance up to respective segment 
        Put all necessary values in dictionary for this marker


        final_marker_distance -> A list of all cumulative distances from above calculations
        '''

        segment, segment_index = getSegment(marker_index, segments_AllTimepoints[i])
        distance = dist_along_branch(normalized_branch_coordinates[i], segment[0], marker_index)
        scaled_distance = distance*scale_factor_allTimepoints[i][segment_index]
        cumulative_distance = scaled_distance + cumulative_distance_allTimepoints[i][segment_index]
        marker_dictionary = {"marker_index": marker_index, "segment": segment, "distance": distance, "scaled_distance": scaled_distance, "final_distance": cumulative_distance}
        marker_timepoint.append(marker_dictionary)
        marker_dist.append(cumulative_distance)
        rawDist_map_scaledDist.append((distance, cumulative_distance))
        # Export marker info for each timepoint
        with open(export_csv_directory + '/markerInfo/' + str(animal_ID) + '_b' + str(branch_ID) + '_markerInfo_timepoint' + str(i+1) + '.csv', 'w', newline='') as csvfile:
            fieldnames = ['marker_index', 'segment', 'distance', 'scaled_distance', 'final_distance']
            writer = csv.DictWriter(csvfile, fieldnames = fieldnames)
            writer.writeheader()
            writer.writerows(marker_timepoint)
    rawDist_map_scaledDist_allTimepoints.append(rawDist_map_scaledDist)
    marker_info.append(marker_timepoint)
    final_marker_distance.append(marker_dist)
    # mapping_dictionary.append()
print("marker_info =", marker_info, "\n")
# print("final_marker_distance =", final_marker_distance, "\n")
# print("marker type =", raw_marker_type, "\n")
# print("rawDist map to scaledDist all timepoints =", rawDist_map_scaledDist_allTimepoints, "\n")


#################### EXPORTS ####################

# Conditions
print("Below are the conditions used in the computation: ")
print("The pixel to um conversion for [x, y, z] is " + str(scaling_factor))
print("Mapping of markers and fiducials to branch are done in xy")
print("Distance calculation is done in xy")

sys.stdout.close()
sys.stdout=stdoutOrigin

print("Exported detailed calculation and condition info to " + export_csv_directory + "/" + animal_ID + "_b" + branch_ID + "_log.txt")

plot_msg = savePlot_xy()
print(plot_msg)
# plt.show()
print("FINISHED THE COMPUTATIONAL SECTION, yay! \n\nSTARTING THE CLUSTERING SECTION.")



""""""""""""""""""""""""""""""""""""
"""   SECTION #2: CLUSTERING     """
""""""""""""""""""""""""""""""""""""
def separateShaftfromSpine():
    '''
    Separate shaft spines from spines based on labels
    Output:
        shaft_allTimepoints -> list of cumulative distances for shaft spines
        spine_allTimepoints -> list of cumulative distances for regular spines
    '''
    spine_allTimepoints = []
    shaft_allTimepoints = []
    for i in range(len(raw_marker_type)):
        zipped_list = list(zip(raw_marker_type[i], final_marker_distance[i]))
        shaft_points = [point[1] for point in zipped_list if point[0].lower() == inhibitoryshaft.lower()]
        spine_points = [point[1] for point in zipped_list if point[0].lower() == inhibitoryspine.lower()]

        if len(shaft_points) != len(set(shaft_points)): # Check no duplicate distances, or error during clustering step
            raise Exception("Error: There are duplicate distances for shafts (i.e. multiple shaft synapses are mapped to the same branch index)")
        else:
            shaft_allTimepoints.append(shaft_points)
        
        if len(spine_points) != len(set(spine_points)): # Check no duplicate distances, or error during clustering step
            raise Exception("Error: There are duplicate distances for spines (i.e. multiple shaft synapses are mapped to the same branch index)")
        else:
            spine_allTimepoints.append(spine_points)      
    return shaft_allTimepoints, spine_allTimepoints

def find_closest_centroid_index(point, centroids):
    '''
    Closest centroid is just subtraction  from point

    Output: Output the index within the distances_to_centroid list(created in this function)
    with the minimum distance. The order of the distances to centroid list is by 
    original input centroid
    '''
    distances_to_centroid = [abs(centroid - point) for centroid in centroids]
    return distances_to_centroid.index(min(distances_to_centroid))

def k_means(centroids, distance_data):
    '''
    Input: 
    
        centroids: Initial centroids(Used when looped through all
        the different possibilities for initial centroids) 
        distance_data: List of lists format of cumulative data

    Output:
        current_centroids(list): new centroids after everything converges
        cluster_map(dictionary): new centroids dictionary of form in 
                                closest_map 
    
    '''
    def reassign_centroids(cluster_map):
        # return a list of centroids in the same order
        '''
        Input:cluster_map(dictionary) The structure of this dictionary is in create_closest_map
        
        Important Variables
            idx_to_centroid(dictionary): Keys are the centroid indices
                                        Values are the average cumulative distances of 
                                        markers that have been mapped to this point
        Output: new centroids(list): Average cumulative distances of 
                                    markers mapped to each centroid

        Algorithm
        ---------
            We make an idx_to_centroid dictionary that has
            keys: cluster_idx
            values: average cumulative distance for each marker mapped to said
                    cluster_idx.
            
            We then output each of the average cumulative distances 
            in list format [basically just a list of the values]
        
        '''
        idx_to_centroid = {}
        for cluster_idx, tuple_list in cluster_map.items():
            idx_to_centroid[cluster_idx] = round(sum([tuple[1] for tuple in tuple_list])/len(tuple_list), 2)

        #Amy: Could change to just output idx_to_centroid.values or just directly output in the for loop/make list there. 
        
        new_centroids = []
        for i in range(len(cluster_map)):
            new_centroids.append(idx_to_centroid[i])
        return new_centroids

    def create_closest_map(current_centroids):
        '''
        Input: Current centroids(list)
        Output: cluster_to_data dictionary explained in below comment
        '''
        cluster_to_data = {}
        for time_point in range(len(distance_data)):
            '''
            We loop through each timepoint
            time_point_list is the distance data for each timepoint
            We loop through each point in time_point_list and find
            the closest centroid index to it
            '''
            time_point_list = distance_data[time_point]
            for i in range(len(time_point_list)):
                #Amy: Find the closest centroid within the current centroids to the specific marker
                closest_idx = find_closest_centroid_index(time_point_list[i], current_centroids)
                '''
                Construction of cluster_to_data:
                Create a dictionary where keys are centroids that are closest to current
                timepoint element. Values are lists of tuples containing

                0: closest_idx(closest centroid index), 
                1: cumulative distance value for current marker, 
                2: index of current marker in final_marker_distance

                Each tuple essentially holds information for each marker
                that has been mapped to this centroid
                
                '''
                if closest_idx not in cluster_to_data:
                    cluster_to_data[closest_idx] = []
                cluster_to_data[closest_idx].append((closest_idx, time_point_list[i], time_point, final_marker_distance[time_point].index(time_point_list[i])))
        return cluster_to_data

    # loop until nothing changes
    current_centroids = centroids
    for i in range(10):
        cluster_to_data = create_closest_map(current_centroids)
        new_centroids = reassign_centroids(cluster_to_data)
        if new_centroids == current_centroids:
            break
        current_centroids = new_centroids
    return current_centroids, cluster_to_data

def split_if_error(centroids, data_map):
    """
    Returns a new list of centroids if there is any error. 
    Error: points from the same timepoint in the same cluster OR points in a cluster that have a distance >2um
    The new list will contain one more than the original list. 
    """
    def get_error_tuples():
        # Tuples are (clusterId, timepoint)
        cluster_timepoint_set = set()
        error_tuples = []
        for cluster_idx, tuple_list in data_map.items():
            for _, _, tp, _ in tuple_list:
                tuple = (cluster_idx, tp)
                if tuple in cluster_timepoint_set:
                    error_tuples.append(tuple)
                else:
                    cluster_timepoint_set.add(tuple)
        return error_tuples

    def find_worst_tuple(error_tuples):
        """
        Given points from the same timepoint within the same cluster. 
        Worst tuple is one that is furthest from centroid (from ALL repeated points)
        Return the cluster_id and the position
        """
        worst_dist = -1
        worst_position = -1
        worst_cluster = -1
        for cluster_idx, tp in error_tuples:
            tuple_list = data_map[cluster_idx]
            same_tuples = [item for item in tuple_list if item[0] == cluster_idx and item[2] == tp]

            for _, position, _, _ in same_tuples:
                distance = abs(centroids[cluster_idx] - position)
                if distance > worst_dist:
                    worst_position = position
                    worst_dist = distance
                    worst_cluster = cluster_idx
        return worst_cluster, worst_position

    def check_withinDist(centroids):
        for i in range(len(data_map)):
            tuple_list = data_map[i]
            distances = [distance for _, distance, _, _ in tuple_list]
            dist_range = max(distances) - min(distances)
            average = sum(distances)/len(distances)

            if dist_range > 2:
                points = [abs(average - point) for point in distances]
                furthest_point_idx = points.index(max(points))
                return compute_new_centroids(tuple_list[0][0], tuple_list[furthest_point_idx][1])
        return centroids

    def compute_new_centroids(worst_cluster, worst_position):
        """
        All clusters are left untouched except for the cluster that will be split off. 
        The worst position will become a new centroid, and the original cluster's centroid will shift because one point is removed.
        """
        new_centroids = [worst_position]
        for i in range(len(centroids)):
            if i != worst_position:
                new_centroids.append(centroids[i])
                continue
            points_in_this_cluster = len(data_map[i])
            new_centroid = (centroids[i] * points_in_this_cluster - worst_position) / (points_in_this_cluster - 1)
            new_centroids.append(round(new_centroid, 2))
        return new_centroids

    error_tuples = get_error_tuples()
    if not error_tuples:
        centroids = check_withinDist(centroids)
        return centroids
    cluster_to_split, position_to_break = find_worst_tuple(error_tuples)
    return compute_new_centroids(cluster_to_split, position_to_break)

def calculateClusters(distance_data, filename):
    '''
    Input:
        distance_data(list): contains the cumulative distances in the 
        timepoint list of list form. This is calculated separately for 
        spines and shaft spines.
    Output:
        cluster_mapping(prettyprint):

    
    '''
    def getClustersGivenInitialSeeds(centroids):
        '''
        Input: 
            centroids -> We start originally with an individual timepoint cumulative. 
        
        Output:
            (centroids, data_map, stdev) -> 
            

        '''
        def get_stdev(data_map):
            '''
            data_map: (Result of k_means)

            '''
            stdev_allClusters = []
            for i in range(len(data_map)):
                tuple_list = data_map[i]
                distances = [distance for _, distance, _, _ in tuple_list]
                stdev_allClusters.append(np.std(distances)) # get standard deviation of a cluster
            average_stdev = sum(stdev_allClusters)/len(stdev_allClusters) # calculate the average stdev across all clusters
            return average_stdev

        for i in range(20):
            new_centroids, data_map = k_means(centroids, distance_data)
            splitted_centroids = split_if_error(new_centroids, data_map)
            if splitted_centroids == centroids:
                break
            centroids = splitted_centroids
        stdev = get_stdev(data_map)
        return centroids, data_map, stdev

    stdev_diffSeeding = []
    stdev_index = []
    print("Trying different clusterings with initial centroids from each of the timepoints.")
    for i in range(len(distance_data)):
        if not distance_data[i]:
            pass
        else:
            centroids, data_map, stdev = getClustersGivenInitialSeeds(distance_data[i])
            stdev_index.append(i)
            stdev_diffSeeding.append(stdev)
    stdev_list = [abs(stdev - sum(stdev_diffSeeding)/len(stdev_diffSeeding)) for stdev in stdev_diffSeeding]
    best_seed = stdev_index[stdev_list.index(min(stdev_list))]
    print("The best timepoint to set as initial cluster is Timepoint", best_seed + 1)
    centroids, data_map, stdev = getClustersGivenInitialSeeds(distance_data[best_seed])
    clusters_mapping = pretty_print(centroids, data_map, filename)
    return clusters_mapping

# PRINTING AND SAVING AS .CSV
def pretty_print(centroids, data_map, filename):
    '''
    Input
    -----
        centroids: 
        data_map: 
        filename: 

    Output 
    -----
    clusters_mapping: List of tuples in which each
        element is of the format (timepoint, point_idx, distance)


    Called
    ------
    calculateClusters Calls this to output cluster_mapping


    Function
    ------

    '''
    sorted_cluster_indices = np.argsort(centroids)
    grouping_list = []
    clusters_mapping = []
    # print("Here's the different clusters (timepoint, point_idx, distance): ")
    for i in range(len(sorted_cluster_indices)):
        cluster_id = sorted_cluster_indices[i]
        tuple_list = data_map[cluster_id]
        pretty_objects = [(tp, point_idx, position) for _, position, tp, point_idx in tuple_list]
        for j in range(number_of_timepoints):
            if j not in [tp for tp, _, _ in pretty_objects]:
                '''
                For all of timepoints within a certain group, we add an element that is
                NA if it does not exist
                within that timepoint i.e. "Nothing" marker
                '''
                pretty_objects.append((j, 'NA', 'NA'))
                pretty_objects.sort()
        # print(pretty_objects)
        clusters_mapping.append(pretty_objects)
        grouping_list.append([position for _, _, position in pretty_objects])

    # Save as CSV
    fieldname = [list("Group" + str(i+1) for i in range(len(data_map)))]
    newpath = export_csv_directory + '/autoAlignment'
    if not os.path.exists(newpath):
        os.makedirs(newpath)
    with open(export_csv_directory+ '/autoAlignment/' + str(animal_ID) + '_b' + str(branch_ID) +'_' + filename + '_grouping.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(fieldname)
        writer.writerows(list(zip(*grouping_list)))
    print("Exported the " + filename + " grouping to "+ export_csv_directory + '/autoAlignment/' + str(animal_ID) + '_b' + str(branch_ID) +'_' + filename + '_grouping.csv')
    return clusters_mapping

shaft_distances, spine_distances = separateShaftfromSpine()
print(f"spine_distance PRINT {spine_distances}")

if not any(shaft_distances) == False:
    shaft_grouping_list = calculateClusters(shaft_distances, "inhibitoryshaft")

if not any(spine_distances) == False:
    spine_grouping_list = calculateClusters(spine_distances, "inhibitoryspine")

# print("shaft_grouping_list =", shaft_grouping_list, "\n")
# print("spine_grouping_list =", spine_grouping_list, "\n")
print("FINISHED THE CLUSTERING SECTION, yay! \n\nSTARTING THE MAPPING SECTION.")






""""""""""""""""""""""""""""""""""""
"""   SECTION #3: MAPPING     """
""""""""""""""""""""""""""""""""""""
def average_clusterDistance(cluster):
    synapseDistances = [point[2] for point in cluster if point[2] != 'NA']
    average_clustDist = sum(synapseDistances)/len(synapseDistances)
    return average_clustDist

def estimateCoordinates(point, average_clusterDist):
    cumulative_distances = cumulative_distance_allTimepoints[0] # just use the first one because all the same
    for i in range(0, len(cumulative_distances)-1):
        if average_clusterDist <= cumulative_distances[i+1]:
            unscaledDist = (average_clusterDist - cumulative_distance_allTimepoints[point[0]][i])/scale_factor_allTimepoints[point[0]][i] + cumulative_distance_unScaled_allTimepoints[point[0]][i]
            raw_branchPoint = findCoordinate_givenDistance(point[0], unscaledDist)
            return raw_branchPoint
    raise Exception("Error: The calculated distance for point is out of bounds.")
    

def findCoordinate_givenDistance(imagenum, unscaledDist):
    for i in range(len(normalized_branch_coordinates[imagenum])):
        distance = dist_along_branch(normalized_branch_coordinates[imagenum], 0, i)
        if distance >= unscaledDist - 1e-8:
            return raw_branch_coordinates[imagenum][i]
    raise Exception("The unscaled distance is longer than the branch")

def avg_Translation(cluster):
    '''
    Input
    -------
    cluster: Called from cluster_csv 
            Actual format can be found as [One of the clusters from 
            the clusterlist outputted by calculateClusters]
        Note that this function is meant to only be used on the spine synapses 
    Output
    -------
    (xTranslation, yTranslation): Goes through all spinePoints which
        exist and calculates the distance from the branch point it was mapped to
        (separately with x and y)
        Finally calculates the mean of these translations 

    '''
    spinePoints = [point for point in cluster if point[1] != 'NA']
    xtrans_list = []
    ytrans_list = []
    for point in spinePoints:
        rawXY = [raw_marker_coordinates[point[0]][point[1]][0], raw_marker_coordinates[point[0]][point[1]][1]]
        mapped_index = closest_branchIndex_forMarker_AllTimepoints[point[0]][point[1]]
        mappedXY = [raw_branch_coordinates[point[0]][mapped_index][0], raw_branch_coordinates[point[0]][mapped_index][1]]
        xtrans_list.append(rawXY[0] - mappedXY[0])
        ytrans_list.append(rawXY[1] - mappedXY[1])
    avg_Xtranslation = sum(xtrans_list)/len(xtrans_list)
    avg_Ytranslation = sum(ytrans_list)/len(ytrans_list)
    return round(avg_Xtranslation), round(avg_Ytranslation)

def cluster_csv(clusterList, markerType, emptyMarker, markerStart, translation):
    '''
    Input
    ------
    clusterList: 
    markerType: String for name of MarkerType - For objectj must match 
    emptyMarker: Same for the empty marker type
    markerStart: So that you can increase index when moving from shaft to spine
    translation 

    Output
    ------
    csv_list which is a list of 
    list of list of strings containing the image, the marker, the type, and the coordinates

    '''
    csv_list = []
    for cluster in clusterList:
        if translation == True:
            avg_Xtranslation, avg_Ytranslation = avg_Translation(cluster)
        average_clusterDist = average_clusterDistance(cluster)
        for point in cluster:
            if point[1] == "NA":
                new_markerType = emptyMarker
                coordinates = list(estimateCoordinates(point, average_clusterDist))
                if translation == True:
                    coordinates[0] = coordinates[0] + avg_Xtranslation
                    coordinates[1] = coordinates[1] + avg_Ytranslation
            else:
                new_markerType = markerType
                coordinates = raw_marker_coordinates[point[0]][point[1]]

            '''
                Commented out by Amy so that I don't have to see the 
                Marker string with every label. This probably messes with the 
                original objectJ Macros but for napari 
                this works correctly
                This is a ~temporary~ solution
            '''
            #csv_list.append(["Image"+str(point[0]+1), "Marker"+str(markerStart), new_markerType, coordinates[0], coordinates[1], transformZ_ImageJtoObjectJ(coordinates[2])])
            csv_list.append(["Image"+str(point[0]+1), markerStart, new_markerType, coordinates[0], coordinates[1], transformZ_ImageJtoObjectJ(coordinates[2])])
        markerStart += 1
    return csv_list


'''
Napari types implementation Starts Here
- Amy

'''
def napari_view(csv_list, markerType, emptyMarker, viewer_mapping): 
    '''
    Input
    -----
    csv_list as described in the cluster_csv function
    markerType: type of marker (e.g. "InhibitoryShaft" or "SpinewithInhSynapse")
    emptyMarker: type for missing markers (e.g. "Nothing" or "NudeSpine")
    viewer_mapping: dictionary mapping timepoint to viewer instance
    '''
    # Define color mapping for different marker types
    color_map = {
        'InhibitoryShaft': 'red',
        'Nothing': 'gray',
        'SpinewithInhSynapse': 'blue', 
        'NudeSpine': 'lightblue'
    }

    timepoint_grouping = defaultdict(list)
    
    # Group points by timepoint
    for element in csv_list:
        timepoint = int(element[0][-1])
        timepoint_grouping[timepoint].append(element[0:6])

    # Add each timepoint's points as a separate layer
    for timepoint, elements in timepoint_grouping.items():
        points = np.array([[transformZ_ObjectJtoImageJ(el[5]), el[4], el[3]] for el in elements])
        features = {
            'label': [el[1] for el in elements],
            'type': [el[2] for el in elements]
        }
        text = {
            "string": [el[1] for el in elements],
            'size': 10,
            'color': 'white',
        }
        
        # Get colors based on marker types
        face_colors = [color_map[el[2]] for el in elements]
        
        viewer = viewer_mapping.get(timepoint)
        viewer.add_points(
            points,
            features=features,
            size=5,
            edge_width=0.1,
            edge_width_is_relative=True,
            edge_color='white',
            face_color=face_colors,  # Use the color list instead of face_color_cycle
            text=text,
            name=f'Timepoint: {timepoint} Type: {markerType}'
        )

def load_fourchannel_image(image_path, image_name, viewer):
    '''
    Input:
    ------
    input the image path and the name/timepoint of the image_name and the viewer the image is added to
    
    '''
    print(f'Loading Image {image_name}')
    image = tif.imread(image_path)

    ch1 = image[:, 0, :, :]
    ch2 = image[:, 1, :, :]
    ch3 = image[:, 2, :, :] 
    ch4 = image[:, 3, :, :]

    viewer.add_image(ch1, name = f'Gephyrin {image_name}', colormap = 'green', blending = 'additive')
    viewer.add_image(ch2, name = f'RFP {image_name}', colormap = 'cyan', blending = 'additive')
    viewer.add_image(ch3, name = f'Cell Fill {image_name}', colormap = 'gray', blending = 'additive')
    viewer.add_image(ch4, name = f'Bassoon {image_name}', colormap ='orange', blending = 'additive')

def testResults(actual_csv_path, predicted_csv_list):
    '''
    Compare manually checked results with algorithm predictions and visualize in napari
    '''
    # Read actual results
    df_actual = pd.read_csv(actual_csv_path)
    
    # Convert predicted results to dataframe
    predicted_df = pd.DataFrame(predicted_csv_list, 
                              columns=['image', 'markerID', 'markerType', 'x', 'y', 'z'])
    
    # Define color mapping with lowercase keys
    color_map = {
        'inhibitoryshaft': 'red',
        'nothing': 'gray',
        'spinewithinhibitorysynapse': 'blue',
        'spinewithinhsynapse': 'blue',  # Alternative spelling
        'nudespine': 'lightblue',
        'landmark': 'white'  # Added in case landmarks are present
    }

    # Helper function to get color safely
    def get_color(marker_type):
        return color_map.get(str(marker_type).lower(), 'white')  # Default to white if type not found
    
    # Lists to store points and their colors
    correct_points = []
    incorrect_pred_points = []
    missed_actual_points = []
    
    # Lists to store labels
    correct_labels = []
    incorrect_pred_labels = []
    missed_actual_labels = []
    
    # Lists to store detailed information about errors
    incorrect_predictions_info = []
    missed_actuals_info = []
    
    # Lists to store actual points data
    actual_all_points = []
    actual_all_labels = []
    actual_all_colors = []
    
    # Lists to store all predicted points data
    predicted_all_points = []
    predicted_all_labels = []
    predicted_all_colors = []
    
    viewers = {}  # Dictionary to hold viewers for each image

    for image in range(1, 7):
        image_col = f'S{image}'
        final_col = f'Final S{image}'
        xpos_col = f'xpos S{image}'
        ypos_col = f'ypos S{image}'
        zpos_col = f'zpos S{image}'
        
        image_name = f'Image{image}'
        
        # Create a new viewer for each image
        viewer = napari.Viewer(ndisplay=3)
        viewers[image] = viewer
        
        # Get actual markers
        actual_coords = list(zip(df_actual[zpos_col], 
                               df_actual[xpos_col],
                               df_actual[ypos_col]))
        actual_types = df_actual[final_col].tolist()
        actual_groups = df_actual['Marker'].tolist()  # Get actual group numbers
        
        # Collect actual points data (skipping NaN values)
        valid_rows = df_actual[~pd.isna(df_actual[final_col])]
        coords = list(zip(valid_rows[zpos_col], 
                         valid_rows[xpos_col],
                         valid_rows[ypos_col]))
        types = valid_rows[final_col].tolist()
        groups = valid_rows['Marker'].tolist()
        
        actual_all_points.extend(coords)
        actual_all_labels.extend([f"A{group}" for group in groups])
        actual_all_colors.extend([get_color(type_) for type_ in types])
        
        # Get predicted markers
        pred_image = predicted_df[predicted_df['image'] == image_name]
        pred_coords = list(zip(pred_image['z'],
                             pred_image['x'], 
                             pred_image['y']))
        pred_types = pred_image['markerType'].tolist()
        pred_groups = pred_image['markerID'].tolist()  # Get predicted group numbers
        
        # Track which points have been matched
        actual_matched = [False] * len(actual_coords)
        pred_matched = [False] * len(pred_coords)
        
        # Find matches
        for i, (actual_coord, actual_type) in enumerate(zip(actual_coords, actual_types)):
            # Skip NaN values in actual_type
            if pd.isna(actual_type):
                continue
                
            for j, (pred_coord, pred_type) in enumerate(zip(pred_coords, pred_types)):
                if not pred_matched[j] and str(actual_type).lower() == str(pred_type).lower():
                    # Calculate distance
                    dist = np.sqrt(sum((a - b) ** 2 for a, b in zip(actual_coord, pred_coord)))
                    if dist <= 5.0:  # 2μm threshold
                        correct_points.append(pred_coord)
                        correct_labels.append(f"P{pred_groups[j]}/A{actual_groups[i]}")
                        actual_matched[i] = True
                        pred_matched[j] = True
                        break
        
        # Add unmatched predictions to incorrect list
        for j, (pred_coord, matched) in enumerate(zip(pred_coords, pred_matched)):
            if not matched:
                incorrect_pred_points.append(pred_coord)
                incorrect_pred_labels.append(f"P{pred_groups[j]}")
                incorrect_predictions_info.append({
                    'timepoint': image,
                    'coordinates': pred_coord,
                    'type': pred_types[j],
                    'group': pred_groups[j]
                })
        
        # Add unmatched actual points to missed list
        for i, (actual_coord, matched) in enumerate(zip(actual_coords, actual_matched)):
            if not matched and not pd.isna(actual_types[i]):
                missed_actual_points.append(actual_coord)
                missed_actual_labels.append(f"A{actual_groups[i]}")
                missed_actuals_info.append({
                    'timepoint': image,
                    'coordinates': actual_coord,
                    'type': actual_types[i],
                    'group': actual_groups[i]
                })
        
        # Collect predicted points data
        pred_coords = list(zip(pred_image['z'],
                             pred_image['x'], 
                             pred_image['y']))
        pred_types = pred_image['markerType'].tolist()
        pred_groups = pred_image['markerID'].tolist()
        
        # Add all predicted points data
        predicted_all_points.extend(pred_coords)
        predicted_all_labels.extend([f"P{group}" for group in pred_groups])
        predicted_all_colors.extend([get_color(type_) for type_ in pred_types])
        
        # Add actual points layer to the viewer
        if actual_all_points:
            viewer.add_points(
                actual_all_points,
                size=10,
                face_color=actual_all_colors,
                name='Actual Points',
                text={
                    'string': actual_all_labels,
                    'size': 10,
                    'color': 'white'
                }
            )
        
        # Add predicted points layer to the viewer
        if predicted_all_points:
            viewer.add_points(
                predicted_all_points,
                size=10,
                face_color=predicted_all_colors,
                name='Predicted Points',
                text={
                    'string': predicted_all_labels,
                    'size': 10,
                    'color': 'white'
                }
            )
        
        # Add comparison layers
        if correct_points:
            viewer.add_points(
                correct_points, 
                size=10, 
                face_color='green',
                name='Correct Predictions',
                text={
                    'string': correct_labels,
                    'size': 10,
                    'color': 'white'
                }
            )
        
        if incorrect_pred_points:
            viewer.add_points(
                incorrect_pred_points, 
                size=10, 
                face_color='red',
                name='Incorrect Predictions',
                text={
                    'string': incorrect_pred_labels,
                    'size': 10,
                    'color': 'white'
                }
            )
        
        if missed_actual_points:
            viewer.add_points(
                missed_actual_points, 
                size=10, 
                face_color='orange',
                name='Missed Actual Points',
                text={
                    'string': missed_actual_labels,
                    'size': 10,
                    'color': 'white'
                }
            )
    
    # Print metrics
    total_actual = len(correct_points) + len(missed_actual_points)
    total_predicted = len(correct_points) + len(incorrect_pred_points)
    correct_predictions = len(correct_points)
    
    precision = correct_predictions / total_predicted if total_predicted > 0 else 0
    recall = correct_predictions / total_actual if total_actual > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    print(f"\nAlignment Results:")
    print(f"Total actual markers: {total_actual}")
    print(f"Total predicted markers: {total_predicted}")
    print(f"Correctly predicted markers: {correct_predictions}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1:.2f}")
    
    print("\nIncorrect Predictions:")
    for info in incorrect_predictions_info:
        print(f"Timepoint {info['timepoint']}: {info['type']} Group {info['group']} at coordinates {info['coordinates']}")
    
    print("\nMissed Actual Points:")
    for info in missed_actuals_info:
        print(f"Timepoint {info['timepoint']}: {info['type']} Group {info['group']} at coordinates {info['coordinates']}")
    
    return viewers  # Return the viewers for further use if needed

''' 
Napari exports are grouped with Phoebe's CSV Exports below
in each try/except This should all be put in a main function because this is dumb
Also make the viewers more readable i.e. loop and path loop. 
'''
# viewer1 = napari.Viewer()
# viewer2 = napari.Viewer()
# viewer3 = napari.Viewer()
# viewer4 = napari.Viewer()
# viewer5 = napari.Viewer()
# viewer6 = napari.Viewer()
# viewer_mapping = {1:viewer1, 
#                   2: viewer2, 
#                   3: viewer3, 
#                   4: viewer4, 
#                   5: viewer5, 
#                   6: viewer6}
# # load_fourchannel_image(r"Z:\Amy\files_for_amy_fromJoe\example_analysis_SOM022\Automated_Puncta_Detection\Image1\With_normch4-SOM022_Image 1_MotionCorrected.tif", 'image1', viewer1)
# # load_fourchannel_image(r"Z:\Amy\files_for_amy_fromJoe\example_analysis_SOM022\Automated_Puncta_Detection\Image2\With_normch4-SOM022_Image 2_MotionCorrected.tif", 'image2', viewer2)
# # load_fourchannel_image(r"Z:\Amy\files_for_amy_fromJoe\example_analysis_SOM022\Automated_Puncta_Detection\Image3\With_normch4-SOM022_Image 3_MotionCorrected.tif", 'image3', viewer3)
# # load_fourchannel_image(r"Z:\Amy\files_for_amy_fromJoe\example_analysis_SOM022\Automated_Puncta_Detection\Image4\With_normch4-SOM022_Image 4_MotionCorrected.tif", 'image4', viewer4)
# # load_fourchannel_image(r"Z:\Amy\files_for_amy_fromJoe\example_analysis_SOM022\Automated_Puncta_Detection\Image5\With_normch4-SOM022_Image 5_MotionCorrected.tif", 'image5', viewer5)
# # load_fourchannel_image(r"Z:\Amy\files_for_amy_fromJoe\example_analysis_SOM022\Automated_Puncta_Detection\Image6\With_normch4-SOM022_Image 6_MotionCorrected.tif", 'image6', viewer6)

try:
    shaft_grouping_list
except NameError:
    print("There are no shaft synapses for this branch.")
    num_shaft_markers = 0
    shaft_clusters_csv = []
else:
    shaft_clusters_csv= cluster_csv(shaft_grouping_list, "InhibitoryShaft", "Nothing", 1, False)
    num_shaft_markers = len(shaft_grouping_list)
 
try:
    spine_grouping_list
except NameError:
    print("There are no spine synapses for this branch.")
    num_spine_markers = 0
    spine_clusters_csv = []
else: 
    spine_clusters_csv = cluster_csv(spine_grouping_list, "SpinewithInhSynapse", "NudeSpine", len(shaft_grouping_list)+1, True)
    print(spine_clusters_csv)
    num_spine_markers = len(spine_grouping_list)

# napari.run()
num_shafts_and_spines = num_shaft_markers + num_spine_markers
landmark_csv = []
for i in range(len(raw_fiducials_coordinates)):
    num_landmarks = 1
    for point in raw_fiducials_coordinates[i]:
        landmark_csv.append(["Image"+str(i+1), "Marker"+str(num_shafts_and_spines+num_landmarks), "Landmark", point[0], point[1], transformZ_ImageJtoObjectJ(point[2])])
        num_landmarks += 1

export_cluster_csv = shaft_clusters_csv + spine_clusters_csv + landmark_csv
testing_csv = shaft_clusters_csv + spine_clusters_csv 
#Running TESTING HERE
print('started testing')
viewers = testResults('/Volumes/nedividata/Amy/files_for_amy_fromJoe/example_analysis_SOM022/PunctaScoring/b2/SynapseMarkers/Aligned_afterManualCheck/CombinedResults.csv', testing_csv)
for iteration in zip(viewers.values(), xyz_fileNames.values()):
    process_and_add_splines(iteration[1], iteration[0] )
    image = tif.imread('/Volumes/nedividata/Amy/files_for_amy_fromJoe/example_analysis_SOM022/Automated_Puncta_Detection/Image1/SOM022_Image 1_MotionCorrected.tif')
    ch1 = image[:, 0, :, :] 
    iteration[0].add_image(ch1, scale = [4, 1, 1])
napari.run()
# print(metrics)
print('finished testing')

# napari_view(shaft_clusters_csv, "InhibitoryShaft", "Nothing", viewer_mapping)
# napari_view(spine_clusters_csv,  "SpinewithInhSynapse", "NudeSpine", viewer_mapping)

np.savetxt(export_csv_directory+ '/autoAlignment/' + str(animal_ID) + '_b' + str(branch_ID) +'_alignmentMapping.csv', export_cluster_csv, header="image, markerID, markerType, marker_X, marker_Y, marker_Z", delimiter=',', fmt='%s', comments='')
print("Exported the alignment mapping to " + export_csv_directory+ '/autoAlignment/' + str(animal_ID) + '_b' + str(branch_ID) +'_alignmentMapping.csv')
print("FINISHED THE MAPPING SECTION, yay! \n\nFINISHED RUNNING.")

# Group by timepoint
csv_by_timepoint = defaultdict(list)

for row in export_cluster_csv:
    image_label = row[0]  # "Image1", "Image2", etc.
    timepoint = image_label.replace("Image", "")
    markerID = row[1]
    markerType = row[2]
    x, y, z = row[3], row[4], row[5]

    # Reformat to Napari-friendly column order: z, y, x, label, type
    napari_row = [z, y, x, markerID, markerType]
    csv_by_timepoint[timepoint].append(napari_row)

# Write each timepoint to its own CSV file
napari_header = ["z", "y", "x", "label", "type"]
autoalign_path = os.path.join(export_csv_directory, 'autoAlignment')
os.makedirs(autoalign_path, exist_ok=True)

for tp, rows in csv_by_timepoint.items():
    file_path = os.path.join(autoalign_path, f"{animal_ID}_b{branch_ID}_timepoint{tp}_napari.csv")
    with open(file_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(napari_header)
        writer.writerows(rows)
    print(f"Exported Napari-compatible CSV for timepoint {tp} to {file_path}")
 
