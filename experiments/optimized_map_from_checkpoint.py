"""
Save the optimized map from a checkpoint.
Usage:
First spcify the checkpoint path and mesh context path in the code.
Then run the script.
  python optimized_map_from_checkpoint.py

"""

from __future__ import annotations
import numpy as np
import torch
from tori_comparison.data import load_meshes_and_construct_planar_locators
from utils import bary_coordinates_to_euc_point
from mesh_path import MESH_PATH


def generate_torus_mesh_with_face_colors(
    n_theta: int, n_phi: int, r: float, R: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate a torus point cloud with normals.

    Parameters
    ----------
    n_theta : int
        The number of points in the theta direction.
    n_phi : int
        The number of points in the phi direction.
    r : float
        The radius of the torus.
    R : float
        The radius of the tube.

    Returns
    -------
    V : numpy.ndarray of shape (n_theta * n_phi, 3)
        The vertices of the torus.
    normals : numpy.ndarray of shape (n_theta * n_phi, 3)
        The normals of the torus.
    F : numpy.ndarray of shape (n_theta * n_phi * 2, 3)
        The faces of the torus.
    """

    theta = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
    phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    theta, phi = np.meshgrid(theta, phi)
    # r will auto boardcast to the shape of theta and phi
    x = (R + r * np.cos(phi)) * np.cos(theta)
    y = (R + r * np.cos(phi)) * np.sin(theta)
    z = r * np.sin(phi)

    m, n = theta.shape

    def matrix_index_to_linear_index(i: int, j: int) -> int:
        return i * n + j

    # vertices of shape (n_theta * n_phi, 3)
    V = np.concatenate([x.reshape(-1, 1), y.reshape(-1, 1), z.reshape(-1, 1)], axis=1)

    # Add the faces by connecting the vertices along each band (phi)
    F = []
    face_colors = []
    for i in range(m):
        for j in range(n):
            # four index of the square
            four_inds = [
                (i, j),
                (i, (j + 1) % n),
                ((i + 1) % m, (j + 1) % n),
                ((i + 1) % m, j),
            ]
            # print(four_inds)
            four_inds = [matrix_index_to_linear_index(i, j) for i, j in four_inds]
            F.append([four_inds[0], four_inds[1], four_inds[2]])
            F.append([four_inds[0], four_inds[2], four_inds[3]])
            # assgin face color by theta
            face_colors.append(theta[i, j])
            face_colors.append(theta[i, j])
    F = np.array(F)
    face_colors = np.array(face_colors)

    return V, F, face_colors


def save_optimized_map_from_checkpoint(
    ckpt_path,
    mesh_context_path,
    obj_path1,
    obj_path2,
    domain_mesh_with_face_vals_out_path,
    wrapped_domain_mesh_with_face_vals_out_path,
    energy_objective: str = "area_shape",
):
    # PyTorch 2.6+ defaults weights_only=True; this checkpoint has numpy + dicts.
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location="cpu")

    try:
        mesh_context = torch.load(
            mesh_context_path, map_location="cpu", weights_only=False
        )
    except TypeError:
        mesh_context = torch.load(mesh_context_path, map_location="cpu")

    arrays = mesh_context.get("arrays", mesh_context)
    resume_state = ckpt.get("resume_state", {})

    area_change_per_face = np.asarray(
        resume_state["area_change_per_face"], dtype=np.float64
    )
    shape_stretch_per_face = np.asarray(
        resume_state["shape_stretch_per_face"], dtype=np.float64
    )
    dirichlet_per_face = None
    if energy_objective == "dirichlet":
        if "dirichlet_per_face" not in resume_state:
            raise KeyError(
                "resume_state is missing 'dirichlet_per_face' for energy_objective='dirichlet'"
            )
        dirichlet_per_face = np.asarray(
            resume_state["dirichlet_per_face"], dtype=np.float64
        )
    distinct_direction = mesh_context.get("distinct_direction")
    domain_tutte_embedding_type = mesh_context.get("domain_tutte_embedding_type")
    codomain_tutte_embedding_type = mesh_context.get("codomain_tutte_embedding_type")
    codomain_swap_generators = bool(mesh_context.get("codomain_swap_generators", False))
    if domain_tutte_embedding_type is None or codomain_tutte_embedding_type is None:
        # print(
        #     "mesh_context.pt is missing domain_tutte_embedding_type or codomain_tutte_embedding_type. \n"
        #     "Using default Tutte embedding type: Uniform. \n"
        #     "However, you should find the Tutte embedding type from the log. "
        # )
        domain_tutte_embedding_type = "Uniform"
        codomain_tutte_embedding_type = "Uniform"

    # load meshes
    arrays_ckpt = arrays
    img_f_uvs = np.asarray(ckpt["img_f_uvs"], dtype=float)
    domain_vertices = np.asarray(arrays_ckpt["domain_vertices"], dtype=np.float64)
    domain_faces = np.asarray(arrays_ckpt["domain_faces"], dtype=np.int64)
    codomain_vertices = np.asarray(arrays_ckpt["codomain_vertices"], dtype=np.float64)
    codomain_faces = np.asarray(arrays_ckpt["codomain_faces"], dtype=np.int64)

    # Load meshes and construct planar locators
    planar1, planar2, _ = load_meshes_and_construct_planar_locators(
        obj_path1=str(obj_path1),
        obj_path2=str(obj_path2),
        normalize_area_to_one=True,
        domain_Tutte_embedding_type=domain_tutte_embedding_type,
        codomain_Tutte_embedding_type=codomain_tutte_embedding_type,
        cut_on_shortest_generator=True,
        distinct_direction=distinct_direction,
        codomain_swap_generators=codomain_swap_generators,
    )

    # Compute energy per face
    energy_per_face = shape_stretch_per_face + area_change_per_face
    energy_per_face = np.asarray(energy_per_face, dtype=np.float64).ravel()

    # save domain mesh to npz
    if energy_objective == "area_shape":
        np.savez_compressed(
            domain_mesh_with_face_vals_out_path,
            vertices=np.asarray(domain_vertices, dtype=np.float64),
            faces=np.asarray(domain_faces, dtype=np.int64),
            face_colors=np.asarray(energy_per_face, dtype=np.float64).ravel(),
        )
    elif energy_objective == "dirichlet":
        np.savez_compressed(
            domain_mesh_with_face_vals_out_path,
            vertices=np.asarray(domain_vertices, dtype=np.float64),
            faces=np.asarray(domain_faces, dtype=np.int64),
            face_colors=np.asarray(dirichlet_per_face, dtype=np.float64).ravel(),
        )
    else:
        raise ValueError(
            f"Unknown energy_objective {energy_objective!r}; expected 'area_shape' or 'dirichlet'."
        )
    # ================================================================

    # Visualize the image mesh f(T_1) on T_2 in 3D.
    # We keep the same face connectivity as the (cut) domain mesh, but lift the
    # optimized UVs (`img_f_uvs`) onto the codomain surface using `planar2`.

    # 1) Build a representative map to dedupe seam-identified vertices.
    ident_map = planar1.get_identification_map()

    nV = int(img_f_uvs.shape[0])
    rep = np.arange(nV, dtype=np.int64)
    for k, vs in ident_map.items():
        k = int(k)
        for v in vs:
            rep[int(v)] = k

    # 2) Unitize UVs and locate each unique rep on the cut codomain mesh.
    uv_unit = np.asarray(img_f_uvs, dtype=np.float64) % 1.0
    unique_rep = np.unique(rep)
    uv_unique = uv_unit[unique_rep]

    # Each entry is (face_idx, bary_coords, ...). We only use the first two.
    locator_res = planar2.find_bary_coords_parallel(uv_unique)

    V_unique = np.zeros((uv_unique.shape[0], 3), dtype=np.float64)
    for i, pr in enumerate(locator_res):
        face_idx = int(pr[0])
        bary = np.asarray(pr[1], dtype=np.float64).reshape(3)
        V_unique[i] = bary_coordinates_to_euc_point(
            face_idx,
            bary,
            codomain_faces,
            codomain_vertices,
        )

    rep_to_pos = {int(r): V_unique[i] for i, r in enumerate(unique_rep)}

    # 3) Expand back to all vertices (copy seam duplicates).
    V_img = np.zeros((nV, 3), dtype=np.float64)
    for i in range(nV):
        V_img[i] = rep_to_pos[int(rep[i])]

    # save domain mesh to npz
    if energy_objective == "area_shape":
        np.savez_compressed(
            wrapped_domain_mesh_with_face_vals_out_path,
            vertices=np.asarray(V_img, dtype=np.float64),
            faces=np.asarray(domain_faces, dtype=np.int64),
            face_colors=np.asarray(energy_per_face, dtype=np.float64).ravel(),
        )
    elif energy_objective == "dirichlet":
        np.savez_compressed(
            wrapped_domain_mesh_with_face_vals_out_path,
            vertices=np.asarray(V_img, dtype=np.float64),
            faces=np.asarray(domain_faces, dtype=np.int64),
            face_colors=np.asarray(dirichlet_per_face, dtype=np.float64).ravel(),
        )
    else:
        raise ValueError(
            f"Unknown energy_objective {energy_objective!r}; expected 'area_shape' or 'dirichlet'."
        )


if __name__ == "__main__":
    energy_objective = "area_shape"
    # rot_tori = ["r_1_d", "r_2_d", "r_5_d", "r_8_d", "r_9_d"]
    # n_rot_tori = len(rot_tori)
    dataset = [
        "ring_0",
        "ring_1",
        "nut_circle_0",
        "nut_circle_1",
        "roller_0",
        "roller_1",
        "sphere_0",
        "sphere_1",
    ]
    n_dataset = len(dataset)
    for i in range(n_dataset):
        for j in range(n_dataset):
            if i == j:
                continue
            ckpt_path = f"checkpoints_lrc_05_10_2026/{dataset[i]}_vs_{dataset[j]}_lrc/checkpoint.pt"
            mesh_context_path = f"checkpoints_lrc_05_10_2026/{dataset[i]}_vs_{dataset[j]}_lrc/mesh_context.pt"
            obj_path1 = MESH_PATH[dataset[i]]
            obj_path2 = MESH_PATH[dataset[j]]
            domain_mesh_with_face_vals_out_path = (
                f"plot_data/thingi10k/domain_{dataset[i]}_vs_{dataset[j]}.npz"
            )
            wrapped_domain_mesh_with_face_vals_out_path = (
                f"plot_data/thingi10k/wrapped_domain_{dataset[i]}_vs_{dataset[j]}.npz"
            )
            save_optimized_map_from_checkpoint(
                ckpt_path,
                mesh_context_path,
                obj_path1,
                obj_path2,
                domain_mesh_with_face_vals_out_path,
                wrapped_domain_mesh_with_face_vals_out_path,
                energy_objective=energy_objective,
            )

    # energy_objective = "area_shape"
    # ckpt_path = "checkpoints_thingi10k/ring_0_vs_roller_1_lrc/checkpoint.pt"
    # mesh_context_path = "checkpoints_thingi10k/ring_0_vs_roller_1_lrc/mesh_context.pt"
    # obj_path1 = MESH_PATH["ring_0"]
    # obj_path2 = MESH_PATH["roller_1"]
    # domain_mesh_with_face_vals_out_path = (
    #     "plot_data/thingi10k/domain_ring_0_vs_roller_1_lrc.npz"
    # )
    # wrapped_domain_mesh_with_face_vals_out_path = (
    #     "plot_data/thingi10k/wrapped_domain_ring_0_vs_roller_1_lrc.npz"
    # )
    # save_optimized_map_from_checkpoint(
    #     ckpt_path,
    #     mesh_context_path,
    #     obj_path1,
    #     obj_path2,
    #     domain_mesh_with_face_vals_out_path,
    #     wrapped_domain_mesh_with_face_vals_out_path,
    #     energy_objective=energy_objective,
    # )
