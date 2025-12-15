import json
import numpy as np

def save_json_points(centroid, disc_normal, ml_axis, save_path, scale=10.0, volume_node_id="vtkMRMLScalarVolumeNode1"):

    
    points = [
        centroid.tolist(),
        (centroid + disc_normal * scale).tolist(),
        (centroid + ml_axis * scale).tolist()
    ]

    point_names = [
        "HingeCenter",
        "DiscNormEnd",
        "HingeNormEnd"
    ]
    
    control_points = []
    for i, (pt,name) in enumerate(zip(points, point_names), start=1):
        control_points.append({
            "id": str(i),
            "label": name,
            "description": "",
            "associatedNodeID": volume_node_id,
            "position": pt,
            "orientation": [-1.0, -0.0, -0.0, -0.0, -1.0, -0.0, 0.0, 0.0, 1.0],
            "selected": True,
            "locked": False,
            "visibility": True,
            "positionStatus": "defined"
        })
    
    markups_json = {
        "@schema": "https://raw.githubusercontent.com/slicer/slicer/master/Modules/Loadable/Markups/Resources/Schema/markups-schema-v1.0.3.json#",
        "markups": [
            {
                "type": "Fiducial",
                "coordinateSystem": "LPS",
                "coordinateUnits": "mm",
                "locked": False,
                "fixedNumberOfControlPoints": False,
                "labelFormat": "%N-%d",
                "lastUsedControlPointNumber": len(points),
                "controlPoints": control_points,
                "measurements": [],
                "display": {
                    "visibility": True,
                    "opacity": 1.0,
                    "color": [0.4, 1.0, 1.0],
                    "selectedColor": [0.333, 0.0, 0.0],
                    "activeColor": [0.4, 1.0, 0.0],
                    "pointLabelsVisibility": True,
                    "textScale": 2.9,
                    "glyphType": "Sphere3D",
                    "glyphScale": 1.7,
                    "glyphSize": 3.0,
                    "useGlyphScale": True
                }
            }
        ]
    }

    # for i, pt in enumerate(points, start=1):
    #     print(f"Point {i}: {pt}")
    
    with open(save_path, "w") as f:
        json.dump(markups_json, f, indent=4)
    
    print(f"Saved hinge points JSON to {save_path}")
