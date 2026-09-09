import igl
import numpy as np
from meshplot import plot, subplot, interact
from icecream import ic
import os
from scipy.sparse.linalg import spsolve


# Averaged angle distorsion on all faces to check conformality
# (conformal will preserve angle)
def angle_distorsion_frac(v1, v2, f):
    # v1: inpute point positions
    # v2: data points to be compared with
    input_angles = igl.internal_angles(v1, f)
    out_angles = igl.internal_angles(v2, f)
    angle_distorsions = np.max(np.abs(input_angles - out_angles) / input_angles, axis=1)
    return angle_distorsions


# maxiaml angle distorsion over each face
def find_angle_distorsion(v1, v2, f):
    # v1: inpute point positions
    # v2: data points to be compared with
    input_angles = igl.internal_angles(v1, f)
    out_angles = igl.internal_angles(v2, f)
    angle_distorsions = np.max(np.abs(input_angles - out_angles), axis=1)
    return angle_distorsions


# Mean cross edge ratios between two meshes
def edge_length_distorsion(mesh1, mesh2):
    old_cross_edge_ratios = np.array(mesh1.cross_edge_ratios())
    new_cross_edge_ratios = np.array(mesh2.cross_edge_ratios())
    mean_edge_distor = np.average(
        np.abs(old_cross_edge_ratios / new_cross_edge_ratios - 1)
    )
    return mean_edge_distor


root_folder = os.getcwd()
adillo_v, adillo_f = igl.read_triangle_mesh(
    os.path.join(root_folder, "../data", "armadillo_orient.off")
)
k = igl.gaussian_curvature(adillo_v, adillo_f)

# get Gauss Map by computing vertex normals
n = igl.per_vertex_normals(adillo_v, adillo_f)
# compute cotmatrix
l = igl.cotmatrix(adillo_v, adillo_f)
# CMCF
v = np.copy(n)
# store results
v_diffs = []
mean_angle_dists = []
sum_angle_dists = []
n_iters = 10000
# n_iters = 10 # for test
for i in range(1, n_iters + 1):
    v_old = np.copy(v)
    m = igl.massmatrix(v, adillo_f, igl.MASSMATRIX_TYPE_BARYCENTRIC)
    s = m - 0.001 * l
    b = m.dot(v)
    v = spsolve(s, b)

    # mean zero by translation
    v = v - np.mean(v, axis=0, keepdims=True)
    scaling_factor = 1 / np.linalg.norm(v, axis=1, keepdims=True)
    v = scaling_factor * v

    # check conformality
    out_angles = igl.internal_angles(v, adillo_f)
    if np.any(np.isnan(out_angles)):
        print("!!! i={}, exists zero angle".format(i))
    # vertices difference
    step_diff = np.sum(np.linalg.norm(v - v_old, axis=1))

    # print results
    angle_distorsions = find_angle_distorsion(v, adillo_v, adillo_f)
    mean_angle_distorsion = np.average(angle_distorsions)
    sum_angle_distorsion = np.sum(angle_distorsions)
    v_diffs.append(step_diff)
    mean_angle_dists.append(mean_angle_distorsion)
    sum_angle_dists.append(sum_angle_distorsion)

    if i % 100 == 0:
        print(
            "i={}, vertices difference = {:.4f}, mean angle distorsion = {:.4f}".format(
                i, step_diff, mean_angle_distorsion
            )
        )

np.save("vertices_diff", v_diffs)
np.save("mean_angle_distors", mean_angle_dists)
np.save("sum_angle_distors", sum_angle_dists)
np.save("final_vertices", v)
