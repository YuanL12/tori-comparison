import numpy as np
import torch
import shapecomp as sc
from tqdm import tqdm
import matplotlib.pyplot as plt
from run_flat_torus import (
    construct_planar_tori_domain_and_codomain_meshes,
    run_flat_torus_optimization,
    construct_planar_tori_domain_and_codomain_meshes_subdivided_twice,
)
import itertools


def non_identity_initial_map(x: np.ndarray) -> np.ndarray:
    """Apply the non-identity initial map to the 2D points.
    @param:
        x: np.ndarray, the points to be mapped
    @return:
        np.ndarray, the mapped points
    """
    return np.array([1 - (1 - x[0]) ** 2, 1 - (1 - x[1]) ** 2])


flat_torus_vectors = np.array(
    [
        [0.0, 1.0, 0.0],
        [1.0, 10.0, 0.0],
        [1.0, 0.1, 0.0],
        [-1.0, 10.0, 0.0],
        [-1.0, 0.1, 0.0],
    ]
)

shared_inds_sparser = [
    [0, 1, 2, 3],
    [4, 8],
    [6, 7],
    [11, 20],
    [12, 19],
    [15, 18],
    [16, 17],
]
shared_inds_denser = [
    [0, 1, 2, 3],
    [4, 8],
    [6, 7],
    [11, 20],
    [12, 19],
    [15, 18],
    [16, 17],
    [45, 64],
    [46, 63],
    [47, 62],
    [48, 61],
    [53, 60],
    [54, 59],
    [55, 58],
    [56, 57],
]


optimized_energy = np.zeros((10, 10))
target_energy = np.zeros((10, 10))

n_iterations = 100000

for i, j in tqdm(itertools.product(range(10), range(10))):
    matrix_idx = (i, j)
    flat_torus_vector_index_i = i // 2  # which torus for domain (0..4)
    flat_torus_vector_index_j = j // 2  # which torus for codomain (0..4)
    resubdivide_domain = i % 2  # 0 = coarse domain, 1 = fine domain
    # if flat_torus_vector_index_i != flat_torus_vector_index_j:
    #     continue  # only compute the diagonal entries
    print(matrix_idx)

    domain_v1 = np.array([1.0, 0.0, 0.0])
    domain_v2 = flat_torus_vectors[flat_torus_vector_index_i]
    codomain_v1 = np.array([1.0, 0.0, 0.0])
    codomain_v2 = flat_torus_vectors[flat_torus_vector_index_j]

    if resubdivide_domain:
        # fine domain mesh (subdivided twice)
        (
            cutted_domain_vertices,
            cutted_domain_faces,
            L1,
            L2,
            Teichmuller_distance,
            hyperbolic_dist,
            V_codomain,
            F_codomain,
        ) = construct_planar_tori_domain_and_codomain_meshes_subdivided_twice(
            domain_v1,
            domain_v2,
            codomain_v1,
            codomain_v2,
        )
    else:
        # coarse domain mesh
        (
            cutted_domain_vertices,
            cutted_domain_faces,
            L1,
            L2,
            Teichmuller_distance,
            hyperbolic_dist,
            V_codomain,
            F_codomain,
        ) = construct_planar_tori_domain_and_codomain_meshes(
            domain_v1,
            domain_v2,
            codomain_v1,
            codomain_v2,
        )

    # run the optimization
    (
        image_vertices_uv_list_approx_max,
        area_change_record_list_approx_max,
        shape_stretch_record_list_approx_max,
        _,
        best_energy,
    ) = run_flat_torus_optimization(
        cutted_domain_vertices,
        cutted_domain_faces,
        shared_inds_denser if resubdivide_domain else shared_inds_sparser,
        L1,
        L2,
        non_identity_initial_map,
        n_iterations=n_iterations,
        approx_max_type="logsumexp",
        reduce_lr_on_plateau=True,
        energy_plateau_patience=10000,
        reduce_lr_patience=5000,
        reduce_lr_min_lr=1e-5,
    )

    #  store the target and optimized energy
    optimized_energy[matrix_idx] = best_energy
    target_energy[matrix_idx] = (
        abs(np.sqrt(domain_v2[1]) - np.sqrt(codomain_v2[1])) + Teichmuller_distance
    )

np.save("flat_torus_optimized_energy.npy", optimized_energy)
np.save("flat_torus_target_energy.npy", target_energy)
