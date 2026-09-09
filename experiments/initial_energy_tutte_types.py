"""
Compute the two energy terms (area + shape-stretch) for the *initial* maps produced by
different Tutte embedding types, without running any optimization.

Default meshes: r_8 vs r_9 (see experiments/run_all_pairs.sh).

Run (recommended from experiments/ so tori_comparison imports shapecomp):
  python initial_energy_tutte_types.py

Optional:
  python initial_energy_tutte_types.py --edge_length_type geodesic
  python initial_energy_tutte_types.py --cut_on_shortest_generator true --seed 111
  python initial_energy_tutte_types.py --obj_path1 data/synthetic/... --obj_path2 data/synthetic/...
"""

from __future__ import annotations

import argparse

import torch

from tori_comparison.initial_Tutte import (
    TUTTE_TYPES,
    select_tutte_type_minimal_initial_energy,
    sample_unit_direction,
)


def _parse_bool(s: str) -> bool:
    s = s.strip().lower()
    if s in {"1", "true", "t", "yes", "y"}:
        return True
    if s in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected bool, got: {s!r}")


def main() -> None:
    # ------------------------------------------------------------
    # Parse arguments
    # ------------------------------------------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--obj_path1",
        type=str,
        default="data/synthetic/torus_ratio_0.8_n_theta_20_n_phi_10.obj",
        help="Domain mesh path (default: r_8).",
    )
    parser.add_argument(
        "--obj_path2",
        type=str,
        default="data/synthetic/torus_ratio_0.9_n_theta_20_n_phi_10.obj",
        help="Codomain mesh path (default: r_9).",
    )
    parser.add_argument(
        "--edge_length_type",
        type=str,
        default="geodesic",
        choices=["geodesic", "euclidean", "flip_geodesic"],
    )
    parser.add_argument("--normalize_area_to_one", type=_parse_bool, default=True)
    parser.add_argument("--cut_on_shortest_generator", type=_parse_bool, default=True)
    parser.add_argument(
        "--codomain_swap_generators", type=_parse_bool, default=False
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=111,
        help="Used only when cut_on_shortest_generator=true (Reeb direction).",
    )
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    device = str(args.device)
    torch.set_default_dtype(torch.float64)

    distinct_direction = None
    if bool(args.cut_on_shortest_generator):
        distinct_direction = sample_unit_direction(int(args.seed))

    codomain_swap_generators = bool(args.codomain_swap_generators)

    # ------------------------------------------------------------
    # Print settings
    # ------------------------------------------------------------
    print("Meshes:")
    print("  obj_path1:", args.obj_path1)
    print("  obj_path2:", args.obj_path2)
    print("Settings:")
    print("  edge_length_type:", args.edge_length_type)
    print("  normalize_area_to_one:", bool(args.normalize_area_to_one))
    print("  cut_on_shortest_generator:", bool(args.cut_on_shortest_generator))
    print("  codomain_swap_generators:", codomain_swap_generators)
    if distinct_direction is not None:
        print("  distinct_direction:", distinct_direction)
    print()

    # ------------------------------------------------------------
    # Select the best pair of Tutte types
    # ------------------------------------------------------------
    best_domain, best_codomain, energy_by_pair = (
        select_tutte_type_minimal_initial_energy(
            obj_path1=args.obj_path1,
            obj_path2=args.obj_path2,
            normalize_area_to_one=bool(args.normalize_area_to_one),
            edge_length_type=args.edge_length_type,
            cut_on_shortest_generator=bool(args.cut_on_shortest_generator),
            distinct_direction=distinct_direction,
            codomain_swap_generators=codomain_swap_generators,
            device=device,
        )
    )
    print(f"Best pair: domain={best_domain}, codomain={best_codomain}")
    print()

    header = f"{'domain':<10} | {'codomain':<10} | {'energy':>12}"
    print(header)
    print("-" * len(header))
    for td in TUTTE_TYPES:
        for tc in TUTTE_TYPES:
            key = f"{td}_{tc}"
            e = energy_by_pair[key]
            print(f"{td:<10} | {tc:<10} | {e:12.6f}")


if __name__ == "__main__":
    main()
