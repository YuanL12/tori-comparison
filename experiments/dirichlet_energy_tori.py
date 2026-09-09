import argparse
from typing import Any

import numpy as np
from scipy.integrate import solve_bvp, simpson


def torus_area(R: float, r: float) -> float:
    return 4 * np.pi**2 * R * r


def normalize_torus_area_one(R: float, r: float) -> tuple[float, float]:
    scale = 1 / np.sqrt(torus_area(R, r))
    return scale * R, scale * r


def harmonic_torus_map(
    R1: float,
    r1: float,
    R2: float,
    r2: float,
    n_mesh: int = 400,
    n_eval: int = 4000,
    tol: float = 1e-6,
    normalize_area_to_one: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, Any]:
    """
    Harmonic map between tori of revolution under ansatz

        f(u,v) = (phi(u), v)

    Source metric:
        g1 = r1^2 du^2 + (R1 + r1 cos u)^2 dv^2

    Target metric:
        g2 = r2^2 dphi^2 + (R2 + r2 cos phi)^2 dv^2

    If normalize_area_to_one is True, each torus is independently rescaled so
    that its total area 4*pi^2*R*r is one before solving/evaluating the energy.

    Returns:
        u_eval, phi, phi_prime, Dirichlet energy
    """
    if R1 <= 0 or r1 <= 0 or R2 <= 0 or r2 <= 0:
        raise ValueError("All radii must be positive")
    if r1 >= R1 or r2 >= R2:
        raise ValueError("Expected ring tori with R > r")

    if normalize_area_to_one:
        R1, r1 = normalize_torus_area_one(R1, r1)
        R2, r2 = normalize_torus_area_one(R2, r2)

    two_pi = 2 * np.pi

    def ode(u: np.ndarray, y: np.ndarray) -> np.ndarray:
        phi = y[0]
        p = y[1]  # phi'

        A = R1 + r1 * np.cos(u)
        B = R2 + r2 * np.cos(phi)

        # ODE:
        # (A phi')' + (r1^2/r2) * B sin(phi) / A = 0
        #
        # Expanded:
        # phi'' = (r1 sin u / A) phi'
        #          - (r1^2/r2) * B sin(phi) / A^2

        phi_prime = p
        p_prime = (r1 * np.sin(u) / A) * p - (r1**2 / r2) * B * np.sin(phi) / A**2

        return np.vstack((phi_prime, p_prime))

    def bc(ya: np.ndarray, yb: np.ndarray) -> np.ndarray:
        # Degree-one periodic lift:
        # phi(2pi) = phi(0) + 2pi
        # phi'(2pi) = phi'(0)
        return np.array(
            [
                yb[0] - ya[0] - two_pi,
                yb[1] - ya[1],
            ]
        )

    # Initial mesh and initial guess
    u = np.linspace(0, two_pi, n_mesh)
    y_guess = np.vstack((u, np.ones_like(u)))

    sol = solve_bvp(ode, bc, u, y_guess, tol=tol, max_nodes=20000)

    if not sol.success:
        raise RuntimeError(sol.message)

    # Evaluate solution finely
    u_eval = np.linspace(0, two_pi, n_eval)
    phi, phi_prime = sol.sol(u_eval)

    A = R1 + r1 * np.cos(u_eval)
    B = R2 + r2 * np.cos(phi)

    # Singular values:
    sigma_u = r2 * np.abs(phi_prime) / r1
    sigma_v = B / A

    # dA1 = r1 * A du dv
    # E_D = ∫∫ (sigma_u^2 + sigma_v^2) dA1
    integrand = (sigma_u**2 + sigma_v**2) * r1 * A

    E = 2 * np.pi * simpson(integrand, x=u_eval)

    return u_eval, phi, phi_prime, E, sol


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute rotational harmonic Dirichlet energy between two tori."
    )
    parser.add_argument("--R1", type=float, default=1.0)
    parser.add_argument("--r1", type=float, default=0.1)
    parser.add_argument("--R2", type=float, default=1.0)
    parser.add_argument("--r2", type=float, default=0.2)
    parser.add_argument("--n-mesh", type=int, default=400)
    parser.add_argument("--n-eval", type=int, default=4000)
    parser.add_argument("--tol", type=float, default=1e-6)
    parser.add_argument(
        "--normalize-area-to-one",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="independently rescale both tori to area one before computing energy (default: true)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    u, phi, phi_prime, E, sol = harmonic_torus_map(
        args.R1,
        args.r1,
        args.R2,
        args.r2,
        n_mesh=args.n_mesh,
        n_eval=args.n_eval,
        tol=args.tol,
        normalize_area_to_one=args.normalize_area_to_one,
    )

    print("Dirichlet energy =", E)
