import numpy as np
import plotly.graph_objects as go
from utils import plotly_draw_3D_mesh, plotly_draw_2D_mesh
import shapecomp as sc
from torus_data_path import (
    b_path,
    b2_path,
    b2_subdivide_each_face_path,
    a2_path,
    c2_path,
    b3030_path,
    b4040_path,
    ir_phi_2010_path,
    ir_theta_2010_path,
    ACE_coffee_mug_path,
    Cole_mug_path,
    room_essential_path,
)

obj_path = Cole_mug_path
mesh = sc.load_mesh(obj_path, normalize=True)
V, F = mesh.get_vertices(), mesh.get_faces()
print(V.shape, F.shape)
fig = plotly_draw_3D_mesh(
    V,
    F,
    show_now=False,
    show_edges=True,
    mesh_color="grey",
    mesh_color_opacity=0.5,
    edge_color_opacity=0.5,
)


planar = sc.PlanarLocator()
planar.constructPlanarLocatorReebGraph(mesh)
two_generators_circle_reeb = planar.get_generator_paths_3d()
two_generators_circle_reeb = [
    np.concatenate([gen, gen[0:1]]) for gen in two_generators_circle_reeb
]

fig.add_trace(
    go.Scatter3d(
        x=two_generators_circle_reeb[0][:, 0],
        y=two_generators_circle_reeb[0][:, 1],
        z=two_generators_circle_reeb[0][:, 2],
        mode="lines",
        line=dict(color="blue", width=10),
        showlegend=True,
        name="Reeb loop 1",
    )
)

fig.add_trace(
    go.Scatter3d(
        x=two_generators_circle_reeb[1][:, 0],
        y=two_generators_circle_reeb[1][:, 1],
        z=two_generators_circle_reeb[1][:, 2],
        mode="lines",
        line=dict(color="red", width=10),
        showlegend=True,
        name="Reeb loop 2",
    )
)


def save_plotly_to_pdf(fig: go.Figure, path: str = "test.pdf") -> None:
    # hide background
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        scene=dict(
            domain=dict(x=[0, 1], y=[0, 1]),
            bgcolor="rgba(0,0,0,0)",
            xaxis=dict(
                visible=False,
                showbackground=False,
                zeroline=False,
            ),
            yaxis=dict(
                visible=False,
                showbackground=False,
                zeroline=False,
            ),
            zaxis=dict(
                visible=False,
                showbackground=False,
                zeroline=False,
            ),
        ),
        showlegend=False,
        margin=dict(l=0, r=0, t=0, b=0),
    )
    # set camera
    camera = dict(
        eye=dict(x=1.5, y=1.5, z=1.5),
        center=dict(x=0, y=0, z=0),  # where the camera looks
        up=dict(x=0, y=0, z=1),  # "up" direction
    )
    fig.update_layout(scene_camera=camera)
    fig.write_image(
        path,
        format="pdf",
        width=2000,
        height=2000,
        scale=1,  # keep scale = 1 for vector output
    )


save_plotly_to_pdf(fig, path="Cole_coffee_mug_with_reeb_loops.pdf")
