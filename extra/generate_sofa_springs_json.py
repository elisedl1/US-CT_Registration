import json
import numpy as np
from pathlib import Path
import pyvista as pv

def visualize_point_pairs(mesh1_points, mesh2_points, pairs_i, pairs_j, subsample=50, title="Spring Pairs"):
    """Visualize point pairs with lines connecting them."""
    if len(pairs_i) > subsample:
        idx = np.random.choice(len(pairs_i), subsample, replace=False)
        pairs_i = pairs_i[idx]
        pairs_j = pairs_j[idx]

    plotter = pv.Plotter()
    plotter.add_points(mesh1_points, color="lightgray", point_size=3)
    plotter.add_points(mesh2_points, color="lightblue", point_size=3)
    plotter.add_points(pairs_i, color="red", point_size=6, render_points_as_spheres=True)
    plotter.add_points(pairs_j, color="green", point_size=6, render_points_as_spheres=True)

    for p, q in zip(pairs_i, pairs_j):
        plotter.add_mesh(pv.Line(p, q), color="yellow", line_width=2)

    plotter.add_title(title)
    plotter.show()
    plotter.close()
    del plotter

def generate_sofa_springs_json_with_viz(vertebra_pairs_file, facet_pairs_file, output_json_file,
                                        subsample_for_viz=50,
                                        stiffness_body=8000, damping_body=500,
                                        stiffness_facet=4000, damping_facet=300,
                                        visualize=True):
    """
    Converts point pairs JSONs into a SOFA-compatible spring JSON and optionally visualizes them.
    """

    # Load JSONs
    with open(vertebra_pairs_file, "r") as f:
        vertebra_pairs = json.load(f)

    with open(facet_pairs_file, "r") as f:
        facet_pairs = json.load(f)

    sofa_springs = {"springs": {}, "fixed_points_positions": {}, "fixed_points_indices": {}}

    for idx, (pair_name, vertebra_list) in enumerate(vertebra_pairs.items()):
        sofa_pair_key = f"v{idx}v{idx+1}"

        # Body springs
        body_springs = []
        mesh1_points = []
        mesh2_points = []
        for pair in vertebra_list:
            body_springs.append({
                "i": pair["i"],
                "j": pair["j"],
                "d0": pair["d0"],
                "stiffness": stiffness_body,
                "damping": damping_body
            })
            mesh1_points.append(pair["i"])
            mesh2_points.append(pair["j"])

        mesh1_points = np.array(mesh1_points)
        mesh2_points = np.array(mesh2_points)

        # Facet springs (duplicated for left/right)
        facet_list = facet_pairs.get(pair_name, [])
        facet_left = []
        facet_right = []
        facet_i_points = []
        facet_j_points = []

        for pair in facet_list:
            facet_left.append({
                "i": pair["i"],
                "j": pair["j"],
                "d0": pair.get("d0", 1.0),
                "stiffness": stiffness_facet,
                "damping": damping_facet
            })
            facet_right.append({
                "i": pair["i"],
                "j": pair["j"],
                "d0": pair.get("d0", 1.0),
                "stiffness": stiffness_facet,
                "damping": damping_facet
            })
            facet_i_points.append(pair["i"])
            facet_j_points.append(pair["j"])

        sofa_springs["springs"][sofa_pair_key] = {
            "body": body_springs,
            "facet_left": facet_left,
            "facet_right": facet_right
        }

        # Visualize if requested
        if visualize:
            if len(mesh1_points) > 0:
                visualize_point_pairs(mesh1_points, mesh2_points, mesh1_points, mesh2_points,
                                      subsample=subsample_for_viz, title=f"{sofa_pair_key} Body Springs")
            if len(facet_i_points) > 0:
                visualize_point_pairs(np.array(facet_i_points), np.array(facet_j_points),
                                      np.array(facet_i_points), np.array(facet_j_points),
                                      subsample=subsample_for_viz, title=f"{sofa_pair_key} Facet Springs")

    # Fixed points (first and last vertebra)
    first_verts = vertebra_pairs[list(vertebra_pairs.keys())[0]]
    last_verts = vertebra_pairs[list(vertebra_pairs.keys())[-1]]

    sofa_springs["fixed_points_positions"]["v1"] = [p["i"] for p in first_verts[:5]]
    sofa_springs["fixed_points_positions"][f"v{len(vertebra_pairs)+1}"] = [p["j"] for p in last_verts[:5]]

    sofa_springs["fixed_points_indices"]["v1"] = list(range(len(sofa_springs["fixed_points_positions"]["v1"])))
    sofa_springs["fixed_points_indices"][f"v{len(vertebra_pairs)+1}"] = list(range(len(sofa_springs["fixed_points_positions"][f"v{len(vertebra_pairs)+1}"])))

    # Save to JSON
    Path(output_json_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_json_file, "w") as f:
        json.dump(sofa_springs, f, indent=2)

    print(f"SOFA spring JSON saved to {output_json_file}")


# Example usage:
generate_sofa_springs_json_with_viz("/Users/elise/elisedonszelmann-lund/Masters_Utils/Pig_Data/pig2/Registration/CT_segmentations/cropped/original/body_point_pairs.json", "/Users/elise/elisedonszelmann-lund/Masters_Utils/Pig_Data/pig2/Registration/CT_segmentations/cropped/original/facet_point_pairs.json", "sofa_springs.json")


"""

generate_sofa_springs_json(
    vertebra_pairs_file="vertebra_pairs.json",
    facet_pairs_file="facet_pairs.json",
    output_json_file="sofa_springs.json"
)

"""