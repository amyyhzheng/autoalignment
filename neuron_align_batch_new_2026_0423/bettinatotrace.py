import matplotlib.pyplot as plt
from io_utils import read_markers_csv_list, read_branch_csv


marker_csv_path = '/Volumes/nedividata/Joe/2p_data/SOM/ThirdRound/SOM026_DOB081520_RV/Analysis/Analysis_withAmyCode/PunctaScoring/branch3/SOM026_Image1_branch3.csv'
raw_markers, _ = read_markers_csv_list(marker_csv_path)
# Flatten if only one timepoint
marker_coords = [coord for label, coord in raw_markers[0]]

branch_csv_path = '/Volumes/nedividata/Joe/2p_data/SOM/ThirdRound/SOM026_DOB081520_RV/Analysis/Analysis_withAmyCode/SNTTrace/Image1/SOM026_Image1_Trace_xyzCoordinates.csv'
branch_name = 'branch3'  
branch_df = read_branch_csv(branch_csv_path, branch_name)
branch_coords = list(zip(branch_df['x'], branch_df['y'], branch_df['z']))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
bx, by, bz = zip(*branch_coords)
mx, my, mz = zip(*marker_coords)
ax.plot(bx, by, bz, '-k', label='Branch')
ax.scatter(mx, my, mz, c='r', label='Markers')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend()
plt.show()