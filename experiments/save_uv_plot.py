import shapecomp as sc
import numpy as np
import matplotlib.pyplot as plt
from utils import plotly_draw_3D_mesh, plot_2D_mesh, save_obj
from torus_data_path import (
    simple_torus_path,
    a_path,
    a2_path,
    ACE_coffee_mug_path,
    Cole_mug_path,
    room_essential_path,
    teapot_path,
    b2_path,
    b_path,
)

# cole_mug_mesh = sc.load_mesh(Cole_mug_path)
# V, F = cole_mug_mesh.get_vertices(), cole_mug_mesh.get_faces()
# print(V.shape, F.shape)

# room_essential_mesh = sc.load_mesh(room_essential_path)
# V, F = room_essential_mesh.get_vertices(), room_essential_mesh.get_faces()
# print(V.shape, F.shape)

# ACE_coffee_mug_mesh = sc.load_mesh(ACE_coffee_mug_path)
# V, F = ACE_coffee_mug_mesh.get_vertices(), ACE_coffee_mug_mesh.get_faces()
# print(V.shape, F.shape)

mesh = sc.load_mesh(ACE_coffee_mug_path)
V, F = mesh.get_vertices(), mesh.get_faces()
planar3 = sc.PlanarLocator()
planar3.constructPlanarLocatorReebGraph(mesh)

uv_positions_reeb = planar3.get_uv_positions()
cutted_faces_reeb = planar3.get_cutted_mesh_faces()
fig, ax = plot_2D_mesh(
    uv_positions_reeb, cutted_faces_reeb, fill_color="grey", edge_color="grey"
)
ax.set_aspect("equal", adjustable="box")
fig.savefig("ACE_mug_uv.pdf", bbox_inches="tight")
