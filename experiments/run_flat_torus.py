from typing import Callable, List, Literal, Optional, Tuple

import shapecomp as sc
import numpy as np
import torch
from tqdm import tqdm

from tori_comparison.energy import (
    compute_energy_torch,
    compute_flip_faces_count,
    compute_flip_loss,
)
from tori_comparison.optim import (
    ReduceLROnPlateauEMA,
    create_reduce_lr_on_plateau_scheduler,
)
from tori_comparison.par_euc import consturct_euclidean_PARs_list_vectorized
from tori_comparison.stop_conditions import LossPlateauDetector


def hyperbolic_distance(z1, z2):
    """
    Compute hyperbolic distance between two points in the upper half-plane model.

    Parameters:
        z1, z2 : numpy arrays of shape (2,)
            Each represents a point (x, y) with y > 0.

    Returns:
        float : hyperbolic distance
    """
    if len(z1) == 2:
        x1, y1 = z1
        x2, y2 = z2
    else:
        x1, y1, _ = z1
        x2, y2, _ = z2

    # Compute squared Euclidean difference
    num = (x1 - x2) ** 2 + (y1 - y2) ** 2
    den = 2 * y1 * y2

    # arcosh formula
    return np.arccosh(1 + num / den)


def construct_planar_tori_domain_and_codomain_meshes(
    domain_v1: np.ndarray,
    domain_v2: np.ndarray,
    codomain_v1: np.ndarray,
    codomain_v2: np.ndarray,
) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float, np.ndarray, np.ndarray
]:
    """
    Construct the domain and codomain meshes for the flat torus.
    @param:
        domain_v1, domain_v2: np.ndarray, the spanning vectors of the domain parallelogram
        codomain_v1, codomain_v2: np.ndarray, the spanning vectors of the codomain parallelogram
    @return:
        V_domain, F_domain: np.ndarray, the vertices and faces of the domain mesh
        V_codomain, F_codomain: np.ndarray, the vertices and faces of the codomain mesh
    """
    # specify the domain parallelogram btex-fmt
    origin = [0.0, 0.0, 0.0]
    v_1, v_2 = domain_v1, domain_v2
    v_3 = np.array(origin) + np.array(v_1) + np.array(v_2)

    # linear map from unit square to domain parallelogram
    L1 = np.array([v_1[:2], v_2[:2]]).T

    # define the domain mesh
    V_domain = np.array([origin, v_1, v_2, v_3])
    F_domain = np.array([[0, 1, 2], [1, 3, 2]])
    domain_mesh = sc.load_mesh(V_domain, F_domain, normalize=False)

    # subdivide the domain mesh twice to get a simplicial complex
    bary_divided_domain_mesh = sc.subdivide_mesh(domain_mesh)
    re_bary_divided_domain_mesh = sc.subdivide_mesh(bary_divided_domain_mesh)
    V_subdivided = re_bary_divided_domain_mesh.get_vertices()
    F_subdivided = re_bary_divided_domain_mesh.get_faces()

    # specify the codomain parallelogram
    origin = [0.0, 0.0, 0.0]
    v_1, v_2 = codomain_v1, codomain_v2
    v_3 = np.array(origin) + np.array(v_1) + np.array(v_2)

    # linear map from unit square to codomain parallelogram
    L2 = np.array([v_1[:2], v_2[:2]]).T

    # define the codomain mesh
    V_codomain = np.array([origin, v_1, v_2, v_3])
    F_codomain = np.array([[0, 1, 2], [1, 3, 2]])
    # codomain_mesh = sc.load_mesh(V_codomain, F_codomain)

    # compute the Teichmuller distance
    Teichumller_map = L2 @ np.linalg.inv(L1)
    # Compute the singular values of M
    singular_values = np.linalg.svd(Teichumller_map, compute_uv=False)
    # the Teichmuller distance is the log of the ratio of the singular values
    Teichmuller_distance = 1 / 2 * np.log(singular_values[0] / singular_values[1])

    hyperbolic_dist = hyperbolic_distance(V_domain[2], V_codomain[2])
    return (
        V_subdivided,
        F_subdivided,
        L1,
        L2,
        Teichmuller_distance,
        hyperbolic_dist,
        V_codomain,
        F_codomain,
    )


def construct_planar_tori_domain_and_codomain_meshes_subdivided_twice(
    domain_v1: np.ndarray,
    domain_v2: np.ndarray,
    codomain_v1: np.ndarray,
    codomain_v2: np.ndarray,
) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float, np.ndarray, np.ndarray
]:
    """
    Construct the domain and codomain meshes for the flat torus.
    @param:
        domain_v1, domain_v2: np.ndarray, the spanning vectors of the domain parallelogram
        codomain_v1, codomain_v2: np.ndarray, the spanning vectors of the codomain parallelogram
    @return:
        V_domain, F_domain: np.ndarray, the vertices and faces of the domain mesh
        V_codomain, F_codomain: np.ndarray, the vertices and faces of the codomain mesh
    """
    # specify the domain parallelogram btex-fmt
    origin = [0.0, 0.0, 0.0]
    v_1, v_2 = domain_v1, domain_v2
    v_3 = np.array(origin) + np.array(v_1) + np.array(v_2)

    # linear map from unit square to domain parallelogram
    L1 = np.array([v_1[:2], v_2[:2]]).T

    # define the domain mesh
    V_domain = np.array([origin, v_1, v_2, v_3])
    F_domain = np.array([[0, 1, 2], [1, 3, 2]])
    domain_mesh = sc.load_mesh(V_domain, F_domain, normalize=False)

    # subdivide the domain mesh twice to get a simplicial complex
    bary_divided_domain_mesh = sc.subdivide_mesh(domain_mesh)
    re_bary_divided_domain_mesh = sc.subdivide_mesh(
        sc.subdivide_mesh(bary_divided_domain_mesh)
    )
    V_subdivided = re_bary_divided_domain_mesh.get_vertices()
    F_subdivided = re_bary_divided_domain_mesh.get_faces()

    # specify the codomain parallelogram
    origin = [0.0, 0.0, 0.0]
    v_1, v_2 = codomain_v1, codomain_v2
    v_3 = np.array(origin) + np.array(v_1) + np.array(v_2)

    # linear map from unit square to codomain parallelogram
    L2 = np.array([v_1[:2], v_2[:2]]).T

    # define the codomain mesh
    V_codomain = np.array([origin, v_1, v_2, v_3])
    F_codomain = np.array([[0, 1, 2], [1, 3, 2]])
    # codomain_mesh = sc.load_mesh(V_codomain, F_codomain)

    # compute the Teichmuller distance
    Teichumller_map = L2 @ np.linalg.inv(L1)
    # Compute the singular values of M
    singular_values = np.linalg.svd(Teichumller_map, compute_uv=False)
    # the Teichmuller distance is the log of the ratio of the singular values
    Teichmuller_distance = 1 / 2 * np.log(singular_values[0] / singular_values[1])

    hyperbolic_dist = hyperbolic_distance(V_domain[2], V_codomain[2])
    return (
        V_subdivided,
        F_subdivided,
        L1,
        L2,
        Teichmuller_distance,
        hyperbolic_dist,
        V_codomain,
        F_codomain,
    )


def find_equivalent_boundary_points_on_unit_square(points: np.ndarray):
    """
    Find the equivalent boundary points on the fundamental domain of the torus by checking if close to 0 and 1.
    Note: the points has to be on the unit square.
    @param:
        points: np.ndarray, the points on the fundamental domain of the torus
    @return:
        index_groups: list of lists, each list contains the indices of the equivalent boundary points
    """
    # Tolerance for boundary detection
    eps = 1e-8

    def is_on_boundary(p):
        return np.any(np.isclose(p, 0, atol=eps)) or np.any(np.isclose(p, 1, atol=eps))

    def canonical(p):
        # Map all points to the "origin" [0, 0] via modulo 1
        return tuple((p[0] % 1, p[1] % 1))

    index_groups = {}
    for idx, p in enumerate(points):
        if is_on_boundary(p):
            # For points on boundary, compute canonical key
            key = canonical(p)
            index_groups.setdefault(key, []).append(idx)

    # Convert to list of lists
    return list(index_groups.values())


class GradientHook:
    def __init__(self, shared_inds: list[list[int]]) -> None:
        self.shared_inds = (
            shared_inds  # each list is a group of vertices that are identified
        )

    def __call__(self, grad: torch.Tensor) -> torch.Tensor:
        """
        Set the gradients of boundary vertices to be the same.
        """
        grad = grad.clone()  # Avoid modifying the original gradient
        for inds in self.shared_inds:
            shared_grad = grad[inds].mean(dim=0)
            grad[inds] = shared_grad  # Set shared gradients
        return grad


def run_flat_torus_optimization(
    cutted_domain_vertices: np.ndarray,
    cutted_domain_faces: np.ndarray,
    shared_inds: list[list[int]],
    L1: np.ndarray,
    L2: np.ndarray,
    non_identity_initial_map: Callable,
    approx_max_type: str = "none",
    n_iterations: int = 2000,
    learning_rate: float = 1e-3,
    *,
    reduce_lr_on_plateau: bool = True,
    lr_scheduler: Literal["ReduceLROnPlateau", "ReduceLROnPlateauEMA"] = (
        "ReduceLROnPlateauEMA"
    ),
    reduce_lr_factor: float = 0.5,
    reduce_lr_patience: int = 1000,
    reduce_lr_min_lr: float = 1e-7,
    reduce_lr_tol: float = 1e-4,
    reduce_lr_tol_mode: Literal["rel", "abs"] = "rel",
    reduce_lr_ema: float = 0.44,
    energy_plateau_patience: Optional[int] = 2000,
    energy_plateau_rel_tol: float = 1e-5,
) -> Tuple[List[np.ndarray], List[float], List[float], List[float], float]:
    """
    Run the flat torus optimization.

    @param:
        cutted_domain_vertices: np.ndarray, the vertices of the cutted domain mesh
        cutted_domain_faces: np.ndarray, the faces of the cutted domain mesh
        shared_inds: list[list[int]], the indices of the shared vertices
        L1: np.ndarray, the linear map from the unit square to the domain parallelogram
        L2: np.ndarray, the linear map from the unit square to the codomain parallelogram
        non_identity_initial_map: Callable, the non-identity initial map
        approx_max_type: str, the type of approximate max of log(R), "none", "logsumexp", "pnorm", "Q_to_2"
    @return:
        image_vertices_uv_list: List[np.ndarray], the image of the domain uv coordinates
        area_loss_for_record_list: List[float], the area loss for each iteration
        shape_stretch_loss_for_record_list: List[float], the shape stretch loss for each iteration

    Learning rate uses ``ReduceLROnPlateau`` (or an EMA-smoothed variant) on the
    geometric energy ``term1 + term2`` (excluding flip penalty), matching ``run.py``.
    Optional early stop: when ``energy_plateau_patience`` is not ``None``, stop if
    energy has not relatively improved for that many steps (see ``LossPlateauDetector``).
    """
    # compute the image of uv of the domain mesh
    cutted_domain_vertices = cutted_domain_vertices[:, :2]  # remove the third column
    cutted_domain_vertices_uv = (
        cutted_domain_vertices @ np.linalg.inv(L1).T
    )  # map to F_1

    # apply the non-identity map to map to F_2(second unit square)
    image_vertices_uv = np.array(
        [non_identity_initial_map(row) for row in cutted_domain_vertices_uv]
    )

    # we will optimize image of the domain uv coordinates
    img_f_uvs = torch.tensor(image_vertices_uv, dtype=torch.float64, requires_grad=True)

    # set the gradients of shared vertices on boundary to be the same
    img_f_uvs.register_hook(GradientHook(shared_inds))

    # define the optimizer
    optimizer = torch.optim.Adam([img_f_uvs], lr=learning_rate)

    lr_scheduler_obj: (
        torch.optim.lr_scheduler.ReduceLROnPlateau | ReduceLROnPlateauEMA | None
    ) = None
    if reduce_lr_on_plateau:
        if lr_scheduler == "ReduceLROnPlateau":
            lr_scheduler_obj = create_reduce_lr_on_plateau_scheduler(
                optimizer,
                factor=reduce_lr_factor,
                patience=reduce_lr_patience,
                min_lr=reduce_lr_min_lr,
                tol=reduce_lr_tol,
                tol_mode=reduce_lr_tol_mode,
            )
        elif lr_scheduler == "ReduceLROnPlateauEMA":
            lr_scheduler_obj = ReduceLROnPlateauEMA(
                optimizer,
                ema_weight=reduce_lr_ema,
                factor=reduce_lr_factor,
                patience=reduce_lr_patience,
                min_lr=reduce_lr_min_lr,
                tol=reduce_lr_tol,
                tol_mode=reduce_lr_tol_mode,
            )
        else:
            raise RuntimeError(f"Unknown lr_scheduler: {lr_scheduler!r}")

    plateau_detector: LossPlateauDetector | None = None
    if energy_plateau_patience is not None:
        plateau_detector = LossPlateauDetector(
            patience=int(energy_plateau_patience),
            rel_tol=float(energy_plateau_rel_tol),
        )

    # convert numpy array to torch tensor
    L2_tensor = torch.tensor(L2, dtype=torch.float64, requires_grad=False)
    cutted_domain_vertices_tensor = torch.tensor(
        cutted_domain_vertices, dtype=torch.float64, requires_grad=False
    )

    # record the image of uv coordinates to track the optimization
    image_vertices_uv_list = [img_f_uvs.clone().detach().numpy()]

    area_loss_for_record_list = []
    shape_stretch_loss_for_record_list = []
    loss_list = []
    best_energy = float("inf")
    # start optimization
    for i in tqdm(range(n_iterations)):
        optimizer.zero_grad()

        # compute the PARs
        Ps, As, Rs, Ds = consturct_euclidean_PARs_list_vectorized(
            cutted_domain_vertices=cutted_domain_vertices_tensor,
            cutted_domain_faces=cutted_domain_faces,
            image_of_cutted_domain_vertices=img_f_uvs @ L2_tensor.T,
        )

        # compute the energy for record
        area_loss_for_record, shape_stretch_loss_for_record = compute_energy_torch(
            Ps, As, Rs, D=Ds
        )
        area_loss_for_record_list.append(area_loss_for_record.item())
        shape_stretch_loss_for_record_list.append(shape_stretch_loss_for_record.item())
        best_energy = min(
            best_energy,
            area_loss_for_record.item() + shape_stretch_loss_for_record.item(),
        )

        # compute the loss for optimization
        term1, term2 = compute_energy_torch(
            Ps, As, Rs, D=Ds, smooth_max_type=approx_max_type
        )
        flip_loss = compute_flip_loss(img_f_uvs, cutted_domain_faces)
        loss = term1 + term2 + flip_loss
        loss_list.append(loss.item())

        # Geometric energy (no flip penalty): same role as ``energy_rec`` in ``run.py``.
        # energy_for_scheduler = area_loss_for_record + shape_stretch_loss_for_record
        # energy_for_scheduler = float(energy_for_scheduler.item())
        energy_for_scheduler = float(term1.item() + term2.item())

        # record the image of uv coordinates
        image_vertices_uv_list.append(img_f_uvs.clone().detach().numpy())

        # compute the gradient and update the uv coordinates
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            flip_count_post = int(
                compute_flip_faces_count(img_f_uvs, cutted_domain_faces)
            )

        if lr_scheduler_obj is not None:
            lr_scheduler_obj.step(energy_for_scheduler)

        if plateau_detector is not None and plateau_detector.update(
            energy_for_scheduler, flip_count_post
        ):
            print(
                f"Flat torus optimization stopped early: loss plateau at step {i} "
                f"(best energy = {best_energy:.6f})"
            )
            break

    return (
        image_vertices_uv_list,
        area_loss_for_record_list,
        shape_stretch_loss_for_record_list,
        loss_list,
        best_energy,
    )
