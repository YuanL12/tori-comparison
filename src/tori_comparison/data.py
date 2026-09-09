from typing import Dict, Tuple, Optional, Sequence

import shapecomp as sc  # pyright: ignore[reportMissingImports]
import numpy as np
import torch


def load_mesh(obj_path: str, normalize_area_to_one: bool) -> sc.Mesh:
    return sc.load_mesh(obj_path, normalize=normalize_area_to_one)


def count_boundary_sides(uvs: np.ndarray, tol: float = 1e-6) -> dict[str, int]:
    """Count cut-mesh vertices on each unit-square boundary side."""
    x = uvs[:, 0]
    y = uvs[:, 1]
    return {
        "bottom": int(np.sum(np.isclose(y, 0.0, atol=tol))),
        "top": int(np.sum(np.isclose(y, 1.0, atol=tol))),
        "left": int(np.sum(np.isclose(x, 0.0, atol=tol))),
        "right": int(np.sum(np.isclose(x, 1.0, atol=tol))),
    }


def load_mesh_and_construct_planar_locator(
    *,
    obj_path: str,
    normalize_area_to_one: bool,
    Tutte_embedding_type: str,
    cut_on_shortest_generator: bool,
    distinct_direction: Optional[Sequence[float]] = None,
    swap_generators: bool = False,
) -> tuple[sc.PlanarLocator, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load one mesh and construct its planar locator + derived arrays.

    Returns:
      planar: sc.PlanarLocator
      faces: faces array for this side (subdivided faces if not cutting)
      vertices: vertices array for this side (subdivided vertices if not cutting)
      cutted_vertices: cut mesh vertices
      cutted_faces: cut mesh faces
    """
    mesh = load_mesh(obj_path, normalize_area_to_one=normalize_area_to_one)
    if cut_on_shortest_generator:
        planar = sc.PlanarLocator()
        planar.constructPlanarLocatorReebGraphOriented(
            mesh,
            Tutte_embedding_type=Tutte_embedding_type,
            distinctDirection=distinct_direction,
            swap_generators=swap_generators,
        )
        faces = mesh.get_faces()
        vertices = mesh.get_vertices()
    else:
        planar = sc.PlanarLocator(mesh, Tutte_embedding_type=Tutte_embedding_type)
        faces = planar.get_subdivided_mesh_faces()
        vertices = planar.get_subdivided_mesh_vertices()

    cutted_vertices = planar.get_cutted_mesh_vertices()
    cutted_faces = planar.get_cutted_mesh_faces()
    return planar, faces, vertices, cutted_vertices, cutted_faces


def load_meshes_and_construct_planar_locators(
    obj_path1: str,
    obj_path2: str,
    normalize_area_to_one: bool,
    Tutte_embedding_type: str = "Uniform",
    domain_Tutte_embedding_type: str | None = None,
    codomain_Tutte_embedding_type: str | None = None,
    cut_on_shortest_generator: bool = False,
    distinct_direction: Optional[Sequence[float]] = None,
    codomain_swap_generators: bool = False,
) -> Tuple[sc.PlanarLocator, sc.PlanarLocator, Dict[str, np.ndarray]]:
    """
    Load both meshes and construct planar locators.

    By default, uses the same `Tutte_embedding_type` for both domain and codomain.
    If `domain_Tutte_embedding_type` or `codomain_Tutte_embedding_type` are provided,
    they override the embedding type per side.
    """
    domain_type = domain_Tutte_embedding_type or Tutte_embedding_type
    codomain_type = codomain_Tutte_embedding_type or Tutte_embedding_type

    (
        planar1,
        domain_faces,
        domain_vertices,
        cutted_mesh_vertices_1,
        cutted_mesh_faces_1,
    ) = load_mesh_and_construct_planar_locator(
        obj_path=obj_path1,
        normalize_area_to_one=normalize_area_to_one,
        Tutte_embedding_type=domain_type,
        cut_on_shortest_generator=cut_on_shortest_generator,
        distinct_direction=distinct_direction,
    )
    (
        planar2,
        codomain_faces,
        codomain_vertices,
        cutted_mesh_vertices_2,
        cutted_mesh_faces_2,
    ) = load_mesh_and_construct_planar_locator(
        obj_path=obj_path2,
        normalize_area_to_one=normalize_area_to_one,
        Tutte_embedding_type=codomain_type,
        cut_on_shortest_generator=cut_on_shortest_generator,
        distinct_direction=distinct_direction,
        swap_generators=codomain_swap_generators,
    )

    arrays = {
        "domain_faces": domain_faces,
        "domain_vertices": domain_vertices,
        "cutted_mesh_vertices_1": cutted_mesh_vertices_1,
        "cutted_mesh_faces_1": cutted_mesh_faces_1,
        "codomain_faces": codomain_faces,
        "codomain_vertices": codomain_vertices,
        "cutted_mesh_vertices_2": cutted_mesh_vertices_2,
        "cutted_mesh_faces_2": cutted_mesh_faces_2,
    }
    return planar1, planar2, arrays


def numpy_to_torch_tensors(arrays: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
    return {
        "domain_faces": torch.tensor(arrays["domain_faces"], dtype=torch.int32),
        "codomain_faces": torch.tensor(arrays["codomain_faces"], dtype=torch.int32),
        "domain_vertices": torch.tensor(arrays["domain_vertices"], dtype=torch.float64),
        "codomain_vertices": torch.tensor(
            arrays["codomain_vertices"], dtype=torch.float64
        ),
        "cutted_domain_vertices": torch.tensor(
            arrays["cutted_mesh_vertices_1"], dtype=torch.float64
        ),
        "cutted_codomain_vertices": torch.tensor(
            arrays["cutted_mesh_vertices_2"], dtype=torch.float64
        ),
        "cutted_codomain_faces": torch.tensor(
            arrays["cutted_mesh_faces_2"], dtype=torch.int32
        ),
        "cutted_domain_faces": torch.tensor(
            arrays["cutted_mesh_faces_1"], dtype=torch.int32
        ),
    }


def get_uvs_and_shared(
    planar1: sc.PlanarLocator, planar2: sc.PlanarLocator
) -> Tuple[np.ndarray, np.ndarray, list[list[int]]]:
    """
    Get the uv positions of the domain and codomain meshes and the shared indices
    Args:
        planar1: sc.PlanarLocator
        planar2: sc.PlanarLocator
    Returns:
        domain_uvs: np.ndarray, shape=(nV_1, 2)
        codomain_uvs: np.ndarray, shape=(nV_2, 2)
        shared_inds: list of list
            each list contains the indices identified on the boundary of the cutted mesh
    """
    domain_uvs = planar1.get_uv_positions()
    codomain_uvs = planar2.get_uv_positions()

    shared_inds = []
    for k, val in planar1.get_identification_map().items():
        shared_inds.append([k, *val])

    return domain_uvs, codomain_uvs, shared_inds
