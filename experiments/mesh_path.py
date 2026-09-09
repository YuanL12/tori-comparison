from pathlib import Path

# Resolve paths next to this file (experiments/) so synthetic OBJs work regardless of cwd.
_EXP = Path(__file__).resolve().parent


def _synthetic(name: str) -> str:
    return str(_EXP / "data" / "synthetic" / name)


MESH_PATH = {
    "ACE_coffee_mug": "data/real/ACE_Coffee_Mug_Model.obj",
    "Cole_mug": "data/real/Cole_Hardware_Mug_Classic_Blue.obj",
    "room_essential": "data/real/Room_Essentials_Mug_White_Yellow.obj",
    "teapot": "data/real/Threshold_Porcelain_Teapot_White.obj",
    "L1_17": "data/real/L1_17.obj",
    "L3_15": "data/real/L3_15.obj",
    "L5_9": "data/real/L5_9.obj",
    "threshold_mug_remesh": "data/real/remeshed/Threshold_Coffee_Mug_remesh.obj",
    "ace_mug_remesh": "data/real/remeshed/ACE_Coffee_Mug_Model_remesh.obj",
    "cole_mug_remesh": "data/real/remeshed/Cole_Hardware_Mug_Classic_Blue_remesh.obj",
    "room_essential_remesh": "data/real/remeshed/Room_Essentials_Mug_White_Yellow_remesh.obj",
    "r_1": "data/synthetic/torus_ratio_0.1_n_theta_20_n_phi_10.obj",
    "r_1_5050": "data/synthetic/torus_ratio_0.1_n_theta_50_n_phi_50.obj",
    "r_1_sub": "data/synthetic/torus_ratio_0.1_n_theta_20_n_phi_10_subdivided.obj",
    "r_1_d": "data/synthetic/torus_ratio_0.1_n_theta_20_n_phi_30.obj",
    "r_2": "data/synthetic/torus_ratio_0.2_n_theta_20_n_phi_10.obj",
    "r_2_5050": "data/synthetic/torus_ratio_0.2_n_theta_50_n_phi_50.obj",
    "r_2_sub": "data/synthetic/torus_ratio_0.2_n_theta_20_n_phi_10_subdivided.obj",
    "r_2_d": "data/synthetic/torus_ratio_0.2_n_theta_20_n_phi_30.obj",
    "r_5": "data/synthetic/torus_ratio_0.5_n_theta_20_n_phi_10.obj",
    "r_5_sub": "data/synthetic/torus_ratio_0.5_n_theta_20_n_phi_10_subdivided.obj",
    "r_5_d": "data/synthetic/torus_ratio_0.5_n_theta_20_n_phi_30.obj",
    "r_8": "data/synthetic/torus_ratio_0.8_n_theta_20_n_phi_10.obj",
    "r_8_sub": "data/synthetic/torus_ratio_0.8_n_theta_20_n_phi_10_subdivided.obj",
    "r_8_d": "data/synthetic/torus_ratio_0.8_n_theta_20_n_phi_30.obj",
    "r_9": "data/synthetic/torus_ratio_0.9_n_theta_20_n_phi_10.obj",
    "r_9_sub": "data/synthetic/torus_ratio_0.9_n_theta_20_n_phi_10_subdivided.obj",
    "r_9_d": "data/synthetic/torus_ratio_0.9_n_theta_20_n_phi_30.obj",
    "r_9_5050": "data/synthetic/torus_ratio_0.9_n_theta_50_n_phi_50.obj",
    "Gorilla_m_395635": "data/real/pelvis_subsampled/Gorilla-beringei-beringei-m-NMNH-395636-pelvis-L-000091238.obj",
    "Gorilla_m_397351": "data/real/pelvis_subsampled/Gorilla-beringei-beringei-m-NMNH-397351-pelvis-L-000090881.obj",
    "Hoolock_f_83416": "data/real/pelvis_subsampled/Hoolock-hoolock-f-AMNH-83416-pelvis-L-000168063.obj",
    "Hoolock_f_83423": "data/real/pelvis_subsampled/Hoolock-hoolock-f-AMNH-83423-pelvis-L-000168067.obj",
    "Pan_f_15296": "data/real/pelvis_subsampled/Pan-paniscus-f-RMCA-15296-pelvis-L-000164217.obj",
    "Pan_m_27696": "data/real/pelvis_subsampled/Pan-paniscus-m-RMCA-27696-pelvis-L-000093793.obj",
    "Papio_f_384235": "data/real/pelvis_subsampled/Papio-anubis-neumanni-f-NMNH-384235-pelvis-L-000090811.obj",
    "Papio_m_384229": "data/real/pelvis_subsampled/Papio-anubis-neumanni-m-NMNH-384229-pelvis-L-000089800.obj",
    "Pongo_f_588109": "data/real/pelvis_subsampled/Pongo-abelii-f-NMNH-588109-pelvis-L-000093958.obj",
    "Pongo_f_145302": "data/real/pelvis_subsampled/Pongo-pygmaeus-f-NMNH-145302-pelvis-L-000095230.obj",
    "Petauroides": "data/real/pelvis_subsampled/SAMAM7326_Petauroides_volans_Pelvis_m_Polyga_GM.obj",
    # ["ring_0"]="data/real/thingi10k/ring_754638_nV_606.obj"
    # ["ring_1"]="data/real/thingi10k/ring_370885_nV_389.obj"
    # ["nut_0"]="data/real/thingi10k_subsampled/nut_41275_nV_500.obj"
    # ["nut_1"]="data/real/thingi10k_subsampled/nut_370993_nV_500.obj"
    # ["gear_0"]="data/real/thingi10k/gear_100077_nV_264.obj"
    # ["gear_1"]="data/real/thingi10k_subsampled/gear_641140_nV_500.obj"
    # ["gear_2"]="data/real/thingi10k_subsampled/gear_1312974_nV_500.obj"
    # ["gear_3"]="data/real/thingi10k_subsampled/gear_1146170_nV_500.obj"
    # ["roller_0"]="data/real/thingi10k_subsampled/roller_1207667_nV_500.obj"
    # ["pendant_0"]="data/real/thingi10k_subsampled/pendant_428605_nV_500.obj"
    # ["pendant_1"]="data/real/thingi10k_subsampled/pendant_428607_nV_500.obj"
    # used in the paper
    "sphere_0": "data/real/thingi10k_subsampled/sphere_1396892_nV_600.obj",
    "sphere_1": "data/real/thingi10k_subsampled/sphere_1396893_nV_600.obj",
    "ring_0": "data/real/thingi10k_subsampled/ring_76715_nV_600.obj",  # 76715
    "ring_1": "data/real/thingi10k_subsampled/ring_76716_nV_600.obj",  # 76716
    "nut_circle_0": "data/real/thingi10k_subsampled/nut_circle_200966_nV_600.obj",  # 200966
    "nut_circle_1": "data/real/thingi10k_subsampled/nut_circle_200967_nV_600.obj",  # 200967
    "roller_0": "data/real/thingi10k_subsampled/roller_1207667_nV_600.obj",
    "roller_1": "data/real/thingi10k_subsampled/roller_1207669_nV_600.obj",
    "nut_star_0": "data/real/thingi10k_subsampled/nut_star_200962_nV_600.obj",  # 200962
    "nut_star_1": "data/real/thingi10k_subsampled/nut_star_200963_nV_600.obj",  # 200963
    # not used in the paper
    "nut_hexagon_0": "data/real/thingi10k_subsampled/nut_hexagon_200961_nV_600.obj",  # 200961
    "nut_hexagon_1": "data/real/thingi10k_subsampled/nut_hexagon_200969_nV_600.obj",  # 200969
    "gear_0": "data/real/thingi10k_subsampled/gear_669969_nV_600.obj",
    "gear_1": "data/real/thingi10k_subsampled/gear_669970_nV_600.obj",
    "trumpet_0": "data/real/thingi10k_subsampled/trumpet_250397_nV_600.obj",
    "trumpet_1": "data/real/thingi10k_subsampled/trumpet_250398_nV_600.obj",
    "toy_home_0": "data/real/thingi10k/toy_home_100679_nV_248.obj",
    # Annulus + 8t / 16t gears, r_in in {1,2}, outer radius 6 (generate_annulus_gear_meshes.py)
    "hausdorff_annulus_r1": _synthetic("annulus_rin1_Rout6.obj"),
    "hausdorff_annulus_r2": _synthetic("annulus_rin2_Rout6.obj"),
    "hausdorff_gear8_r1": _synthetic("gear_rin1_Rtip6_8t.obj"),
    "hausdorff_gear8_r2": _synthetic("gear_rin2_Rtip6_8t.obj"),
    "hausdorff_gear16_r1": _synthetic("gear_rin1_Rtip6_16t.obj"),
    "hausdorff_gear16_r2": _synthetic("gear_rin2_Rtip6_16t.obj"),
    # 16t, r_in=1: R_valley = 2 (web 1) vs R_valley = 4 (web 3) from inner bore
    "hausdorff_gear16_r1_Rval2": _synthetic("gear_rin1_Rtip6_16t_Rval2.obj"),
    "hausdorff_gear16_r1_Rval4": _synthetic("gear_rin1_Rtip6_16t_Rval4.obj"),
    # 32t, r_in=1: R_valley = 2 (web 1) vs R_valley = 4 (web 3) from inner bore
    "hausdorff_gear32_r1_Rval2": _synthetic("gear_rin1_Rtip6_32t_Rval2.obj"),
    "hausdorff_gear32_r1_Rval4": _synthetic("gear_rin1_Rtip6_32t_Rval4.obj"),
    "hausdorff_annulus_r1_iso": _synthetic("annulus_rin1_Rout6_iso.obj"),
    "hausdorff_annulus_r2_iso": _synthetic("annulus_rin2_Rout6_iso.obj"),
    "hausdorff_gear8_r1_iso": _synthetic("gear_rin1_Rtip6_8t_iso.obj"),
    "hausdorff_gear8_r2_iso": _synthetic("gear_rin2_Rtip6_8t_iso.obj"),
    "hausdorff_gear16_r1_iso": _synthetic("gear_rin1_Rtip6_16t_iso.obj"),
    "hausdorff_gear16_r2_iso": _synthetic("gear_rin2_Rtip6_16t_iso.obj"),
}
