from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
DEFAULT_MESH = REPO_ROOT / "test" / "input" / "torus" / "torus.obj"

sys.path.insert(0, str(REPO_ROOT / "experiments"))
from tori_comparison.cuda_locator import (
    build_uniform_grid_for_locator,
    load_locator,
    locate_uv_grid_tensors,
)


def locate_uv_cpu(
    points: np.ndarray, tri_uv: np.ndarray, eps: float
) -> tuple[np.ndarray, np.ndarray]:
    face = np.full(points.shape[0], -1, dtype=np.int64)
    bary = np.zeros((points.shape[0], 3), dtype=points.dtype)

    for i, (px, py) in enumerate(points):
        for f, ((ax, ay), (bx, by), (cx, cy)) in enumerate(tri_uv):
            denom = (bx - ax) * (cy - ay) - (cx - ax) * (by - ay)
            if abs(float(denom)) <= 1e-30:
                continue
            l0 = ((bx - px) * (cy - py) - (cx - px) * (by - py)) / denom
            l1 = ((cx - px) * (ay - py) - (ax - px) * (cy - py)) / denom
            l2 = 1 - l0 - l1
            if l0 >= -eps and l1 >= -eps and l2 >= -eps:
                face[i] = f
                bary[i] = (l0, l1, l2)
                break

    return face, bary


def check_result(
    name: str,
    label: str,
    points_np: np.ndarray,
    tri_uv_np: np.ndarray,
    expected_face: np.ndarray,
    expected_bary: np.ndarray,
    eps: float,
    atol: float,
    got_face_t: torch.Tensor,
    got_bary_t: torch.Tensor,
) -> None:
    torch.cuda.synchronize()
    got_face = got_face_t.cpu().numpy()
    got_bary = got_bary_t.cpu().numpy()
    np.testing.assert_array_equal(
        got_face, expected_face, err_msg=f"{name} {label} face mismatch"
    )

    inside = expected_face >= 0
    np.testing.assert_allclose(
        got_bary[inside],
        expected_bary[inside],
        atol=atol,
        rtol=0,
        err_msg=f"{name} {label} bary mismatch",
    )
    if np.any(got_bary[inside] < -eps):
        raise AssertionError(f"{name} {label} negative barycentric coordinate")
    np.testing.assert_allclose(
        got_bary[inside].sum(axis=1),
        1,
        atol=atol,
        rtol=0,
        err_msg=f"{name} {label} bary sum mismatch",
    )
    recon = (tri_uv_np[expected_face[inside]] * got_bary[inside, :, None]).sum(axis=1)
    np.testing.assert_allclose(
        recon,
        points_np[inside],
        atol=atol,
        rtol=0,
        err_msg=f"{name} {label} reconstructed UV mismatch",
    )


def assert_cuda_matches(
    locator,
    name: str,
    points_np: np.ndarray,
    tri_uv_np: np.ndarray,
    expected_face: np.ndarray,
    expected_bary: np.ndarray,
    eps: float,
    grid_res: int,
) -> None:
    for dtype, atol in ((torch.float32, 2e-5), (torch.float64, 1e-10)):
        points = torch.as_tensor(points_np, dtype=dtype, device="cuda")
        tri_uv = torch.as_tensor(tri_uv_np, dtype=dtype, device="cuda")

        got_face_t, got_bary_t = locator.locate_uv(points, tri_uv, eps)
        check_result(
            name,
            f"brute {dtype}",
            points_np,
            tri_uv_np,
            expected_face,
            expected_bary,
            eps,
            atol,
            got_face_t,
            got_bary_t,
        )

        cell_starts, cell_tris = build_uniform_grid_for_locator(tri_uv, grid_res)
        got_face_t, got_bary_t = locate_uv_grid_tensors(
            points, tri_uv, cell_starts, cell_tris, grid_res, eps
        )
        check_result(
            name,
            f"grid {dtype}",
            points_np,
            tri_uv_np,
            expected_face,
            expected_bary,
            eps,
            atol,
            got_face_t,
            got_bary_t,
        )

    print(f"[ok] {name}: {points_np.shape[0]} points, {tri_uv_np.shape[0]} triangles")


def synthetic_case(locator, eps: float, grid_res: int) -> None:
    tri_uv = np.array(
        [
            [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
            [[1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
        ],
        dtype=np.float64,
    )
    points = np.array(
        [
            [0.2, 0.2],
            [0.8, 0.8],
            [0.75, 0.1],
            [0.5, 0.5],
            [-0.1, 0.2],
            [1.1, 0.2],
        ],
        dtype=np.float64,
    )
    face, bary = locate_uv_cpu(points, tri_uv, eps)
    assert_cuda_matches(locator, "synthetic", points, tri_uv, face, bary, eps, grid_res)


def import_shapecomp():
    sys.path.insert(0, str(REPO_ROOT / "build"))
    import shapecomp as sc  # pyright: ignore[reportMissingImports]

    return gp


def planar_locator_case(locator, mesh_path: Path, max_points: int, eps: float, grid_res: int) -> None:
    gp = import_shapecomp()
    mesh = sc.load_mesh(str(mesh_path))
    planar = sc.PlanarLocator(mesh)
    uv = np.asarray(planar.get_uv_positions(), dtype=np.float64)
    faces = np.asarray(planar.get_cutted_mesh_faces(), dtype=np.int64)
    tri_uv = uv[faces]
    points = tri_uv.mean(axis=1)[:max_points]

    cpu = planar.find_bary_coords_parallel(points)
    expected_face = np.array([int(row[0]) for row in cpu], dtype=np.int64)
    expected_bary = np.array([np.asarray(row[1], dtype=np.float64) for row in cpu])
    assert_cuda_matches(
        locator,
        "PlanarLocator centroids",
        points,
        tri_uv,
        expected_face,
        expected_bary,
        eps,
        grid_res,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate the CUDA UV locator.")
    parser.add_argument(
        "--mesh",
        type=Path,
        default=DEFAULT_MESH,
        help="Optional torus mesh for PlanarLocator check.",
    )
    parser.add_argument("--max-points", type=int, default=256)
    parser.add_argument("--eps", type=float, default=1e-10)
    parser.add_argument("--grid-res", type=int, default=64)
    parser.add_argument("--skip-planar", action="store_true")
    parser.add_argument("--verbose-build", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit(
            "CUDA is not available; run python -m py_compile for syntax-only validation."
        )

    locator = load_locator(verbose=args.verbose_build)
    synthetic_case(locator, args.eps, args.grid_res)

    if not args.skip_planar and args.mesh.exists():
        planar_locator_case(locator, args.mesh, args.max_points, args.eps, args.grid_res)


if __name__ == "__main__":
    main()
