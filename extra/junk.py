import json
import pyvista as pv
import numpy as np

# path to your JSON
json_path = "/Users/elise/elisedonszelmann-lund/Masters_Utils/Pig_Data/pig2/Registration/CT_segmentations/intra1_seg/hinges/L1_L2_hinge.json"

# load JSON
with open(json_path, 'r') as f:
    data = json.load(f)

# extract points
control_points = data['markups'][0]['controlPoints']
points = {p['label']: np.array(p['position']) for p in control_points}

point1 = points['HingeCenter']
point2 = points['DiscNormEnd']
point3 = points['HingeNormEnd']


# direction = endpoint - startpoint
dir_line_disc = point2 - point1
dir_line_hinge = point3 - point1

dir_line_disc_unit = dir_line_disc / np.linalg.norm(dir_line_disc)
dir_line_hinge_unit = dir_line_hinge / np.linalg.norm(dir_line_hinge)

print("\nDirection vector (disc):", dir_line_disc)
print("Unit direction vector (disc):", dir_line_disc_unit)

print("\nDirection vector (hinge):", dir_line_hinge)
print("Unit direction vector (hinge):", dir_line_hinge_unit)


# # create PyVista lines
# line1 = pv.Line(point1, point2)  # HingeCenter --> DiscNormEnd
# line2 = pv.Line(point1, point3)  # HingeCenter --> HingeNormEnd

# # create a plotter
# plotter = pv.Plotter()
# plotter.add_mesh(line1, color='red', line_width=3)
# plotter.add_mesh(line2, color='blue', line_width=3)

# # show the points as spheres
# plotter.add_mesh(pv.PolyData(point1), color='yellow', point_size=15, render_points_as_spheres=True)
# plotter.add_mesh(pv.PolyData(point2), color='green', point_size=15, render_points_as_spheres=True)
# plotter.add_mesh(pv.PolyData(point3), color='purple', point_size=15, render_points_as_spheres=True)

# # add labels
# plotter.add_point_labels([point1, point2, point3], ['HingeCenter', 'DiscNormEnd', 'HingeNormEnd'], font_size=12)

# plotter.show()
