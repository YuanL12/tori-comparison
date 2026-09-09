from typing import Callable
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# import cv2
import pymeshlab

import trimesh
from scipy.spatial import cKDTree


def get_mesh_points(mesh, n_samples=None, sample=True):
    """
    If sample=True: area-sample points on mesh surface.
    If sample=False: use all mesh vertices.
    """
    if sample:
        if n_samples is None:
            raise ValueError("n_samples must be provided when sample=True.")
        pts, _ = trimesh.sample.sample_surface(mesh, n_samples)
        return pts
    else:
        return np.asarray(mesh.vertices)


def symmetric_hausdorff_mesh(
    mesh_path_1,
    mesh_path_2,
    n_samples=100_000,
    sample=True,
    normalize_area_to_one=True,
):
    mesh1 = trimesh.load(mesh_path_1, force="mesh")
    mesh2 = trimesh.load(mesh_path_2, force="mesh")

    # normalize the mesh to unit size
    if normalize_area_to_one:
        mesh1 = mesh1.copy()
        mesh1.apply_scale(1.0 / np.sqrt(mesh1.area))
        mesh2 = mesh2.copy()
        mesh2.apply_scale(1.0 / np.sqrt(mesh2.area))

    P = get_mesh_points(mesh1, n_samples=n_samples, sample=sample)
    Q = get_mesh_points(mesh2, n_samples=n_samples, sample=sample)

    tree_Q = cKDTree(Q)
    tree_P = cKDTree(P)

    d_P_to_Q, _ = tree_Q.query(P, k=1)
    d_Q_to_P, _ = tree_P.query(Q, k=1)

    h_PQ = d_P_to_Q.max()
    h_QP = d_Q_to_P.max()

    return max(h_PQ, h_QP), h_PQ, h_QP


def downsample_remesh(
    *,
    V: np.ndarray | None = None,
    F: np.ndarray | None = None,
    obj_path: str | None = None,
    target_num_faces: int = 4000,
    isotropic_remesh: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    ms = pymeshlab.MeshSet()
    if obj_path is not None:
        ms.load_new_mesh(obj_path)
    elif V is not None and F is not None:
        m = pymeshlab.Mesh(vertex_matrix=V, face_matrix=F)
        ms.add_mesh(m, "my_mesh")
    else:
        raise ValueError("Either obj_path or V and F must be provided")

    print(
        "Input\nnF =",
        ms.current_mesh().face_number(),
        "nV =",
        ms.current_mesh().vertex_number(),
    )
    # Step 1: Isotropic remeshing (will create zigzag on the rim)
    if isotropic_remesh:
        ms.meshing_isotropic_explicit_remeshing(
            iterations=10,
            targetlen=pymeshlab.PercentageValue(1.0),  # or AbsoluteValue(0.5)
            adaptive=True,
        )

    # Step 2: Decimate to reduce mesh size
    ms.meshing_decimation_quadric_edge_collapse(
        targetfacenum=target_num_faces,  # Target number of faces
        preservenormal=True,  # Keep surface normals
        preservetopology=True,  # Keep mesh closed (no new boundaries)
        planarquadric=True,  # Better for flat regions
        preserveboundary=True,  # only really helps if you have actual boundaries/open rims in the mesh
        boundaryweight=20.0,  # increase boundary importance
    )

    print(
        "Output\nnF =",
        ms.current_mesh().face_number(),
        "nV =",
        ms.current_mesh().vertex_number(),
    )

    # ms.save_current_mesh(output_path)
    return ms.current_mesh().vertex_matrix(), ms.current_mesh().face_matrix()


def cos_angles(V, F, *, eps: float = 1e-12, return_degenerate_count: bool = False):
    """Return cosines of the 3 interior angles for each triangular face.

    Parameters
    - V: (nV, 3) float array of vertex positions
    - F: (nF, 3) int array of triangle indices
    - eps: denom threshold for degenerate angles
    - return_degenerate_count: if True, also return how many angle-denoms <= eps

    Returns
    - cos: (nF, 3) float array: [cos(angle at v0), cos(angle at v1), cos(angle at v2)]
    - degenerate_count (optional): int, total count over all faces/vertices
    """
    V = np.asarray(V)
    F = np.asarray(F)

    A = V[F[:, 0]]
    B = V[F[:, 1]]
    C = V[F[:, 2]]

    # Angle at A between (B-A) and (C-A)
    u0 = B - A
    v0 = C - A
    # Angle at B between (A-B) and (C-B)
    u1 = A - B
    v1 = C - B
    # Angle at C between (A-C) and (B-C)
    u2 = A - C
    v2 = B - C

    def _cos_and_degen(u, v):
        uv = np.einsum("ij,ij->i", u, v)
        nu = np.linalg.norm(u, axis=1)
        nv = np.linalg.norm(v, axis=1)
        denom = nu * nv
        good = denom > eps
        out = np.ones_like(uv, dtype=float)
        out[good] = uv[good] / denom[good]
        return out, (~good).sum()

    c0, d0 = _cos_and_degen(u0, v0)
    c1, d1 = _cos_and_degen(u1, v1)
    c2, d2 = _cos_and_degen(u2, v2)

    cos = np.stack((c0, c1, c2), axis=1)
    degenerate_count = int(d0 + d1 + d2)

    if return_degenerate_count:
        return cos, degenerate_count
    return cos


def face_angles(
    V,
    F,
    *,
    degrees: bool = False,
    eps: float = 1e-12,
    return_degenerate_count: bool = False,
):
    """Return the 3 interior angles (radians by default) for each triangular face."""
    if return_degenerate_count:
        cos, degen = cos_angles(V, F, eps=eps, return_degenerate_count=True)
        ang = np.arccos(cos)
        if degrees:
            ang = np.degrees(ang)
        return ang, degen

    ang = np.arccos(cos_angles(V, F, eps=eps))
    if degrees:
        ang = np.degrees(ang)
    return ang


def compute_isotropic_energy(V: np.ndarray, F: np.ndarray) -> float:
    """Compute the isotropic energy of a mesh. 1 is the best, 0 is the worst.
    It measureing the ratio of the area of the triangle to the sum of the squared edge lengths.

    Parameters
    ----------
    V : numpy.ndarray of shape (n_vertices, 3) or (n_vertices, 2)
        The vertices of the mesh.
    F : numpy.ndarray of shape (n_faces, 3)
        The faces of the mesh.

    Returns
    -------
    isotropic_energy : float
        The isotropic energy of the mesh.
    """
    total = 0.0
    for tri in F:
        a = V[tri[0]]
        b = V[tri[1]]
        c = V[tri[2]]

        # Edge vectors
        e0 = b - a
        e1 = c - b
        e2 = a - c

        # Sum squared edge lengths
        l2 = np.dot(e0, e0) + np.dot(e1, e1) + np.dot(e2, e2)

        # Area
        area = 0.5 * np.linalg.norm(np.cross(e0, e1))

        # Mean-ratio
        mr = (4.0 * np.sqrt(3.0) * area) / (l2 + 1e-20)

        # Energy: push mr toward 1 (equilateral)
        total += (1.0 - mr) ** 2

    return total / len(F)


def compute_edge_lengths_and_angles(
    V: np.ndarray, F: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute the edge lengths, angles, and degrees of vertices.
    Can be used to test the mesh quality.

    Parameters
    ----------
    V : numpy.ndarray of shape (n_vertices, 3) or (n_vertices, 2)
        The vertices of the mesh.
    F : numpy.ndarray of shape (n_faces, 3)
        The faces of the mesh.

    Returns
    -------
    edge_lengths : numpy.ndarray of shape (n_edges, 3)
        The edge lengths of the mesh.
    angles : numpy.ndarray of shape (n_edges, 3)
        The angles of the mesh.
    degrees : numpy.ndarray of shape (n_vertices, 1)
        The degrees of the vertices.
    """
    edge_lengths = []
    angles = []
    degrees = []  # degree of each vertex
    for f in F:
        e1, e2, e3 = V[f[1]] - V[f[0]], V[f[2]] - V[f[0]], V[f[2]] - V[f[1]]
        edge_lengths.append(np.linalg.norm(e1))
        edge_lengths.append(np.linalg.norm(e2))
        edge_lengths.append(np.linalg.norm(e3))
        angles.append(
            np.arccos(np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2)))
        )
        angles.append(
            np.arccos(np.dot(e2, e3) / (np.linalg.norm(e2) * np.linalg.norm(e3)))
        )
        angles.append(
            np.arccos(np.dot(e3, e1) / (np.linalg.norm(e3) * np.linalg.norm(e1)))
        )
    edge_lengths = np.array(edge_lengths)
    angles = np.array(angles)

    # compute the degree of each vertex
    for i in range(len(V)):
        degree = 0
        for f in F:
            if i in f:
                degree += 1
        degrees.append(degree)
    degrees = np.array(degrees)
    return edge_lengths, angles, degrees


def generate_torus_mesh_with_normals(
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
    F = np.array(F)

    # compute the outward normals on each vertex
    normals = np.zeros_like(V)
    normals[:, 0] = (np.cos(phi) * np.cos(theta)).reshape(-1)
    normals[:, 1] = (np.cos(phi) * np.sin(theta)).reshape(-1)
    normals[:, 2] = (np.sin(phi)).reshape(-1)
    return V, normals, F


# generate a torus point cloud from an even function
# func: a function that is positive and even (symmetric around 0)
def generate_torus_mesh_with_variable_r(
    n_theta: int,
    n_phi: int,
    r: float,
    R: float,
    func: Callable,
    tile_along_theta: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a torus point cloud with variable radius.

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
    func : Callable
        A function that is positive and even (symmetric around 0).
    tile_along_theta : bool, optional
        If True, tile the radius along the theta direction.

    Returns
    -------
    V : numpy.ndarray of shape (n_theta * n_phi, 3)
        The vertices of the torus.
    F : numpy.ndarray of shape (n_theta * n_phi * 2, 3)
        The faces of the torus.
    """
    theta = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
    phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)

    theta, phi = np.meshgrid(theta, phi)
    # expand the rs to match to the shape of theta and phi
    # First reshape rs to (n_phi, 1), then tile along the second dimension
    if tile_along_theta:
        rs = np.linspace(-1, 1, n_phi)
        rs = r * func(rs)
        rs = np.tile(rs[:, np.newaxis], (1, n_theta))
    else:
        rs = np.linspace(-1, 1, n_theta)
        rs = r * func(rs)
        rs = np.tile(rs[np.newaxis, :], (n_phi, 1))

    # print(rs.shape, theta.shape, phi.shape)
    x = (R + rs * np.cos(phi)) * np.cos(theta)
    y = (R + rs * np.cos(phi)) * np.sin(theta)
    z = rs * np.sin(phi)

    m, n = theta.shape

    def matrix_index_to_linear_index(i: int, j: int) -> int:
        return i * n + j

    # vertices of shape (n_theta * n_phi, 3)
    V = np.concatenate([x.reshape(-1, 1), y.reshape(-1, 1), z.reshape(-1, 1)], axis=1)

    # Add the faces by connecting the vertices along each band (phi)
    F = []
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
    F = np.array(F)

    return V, F


def compute_surface_area(vertices: np.ndarray, faces: np.ndarray) -> float:
    """Compute the surface area of the mesh.

    Parameters
    ----------
    vertices : numpy.ndarray of shape (n, 3) or (n, 2)
        The vertices of the mesh.
    faces : numpy.ndarray of shape (m, 3)
        The faces of the mesh.

    Returns
    -------
    surface_area : float
        The surface area of the mesh.
    """
    surface_area = 0
    for f in faces:
        v0, v1, v2 = vertices[f]
        surface_area += np.linalg.norm(np.cross(v1 - v0, v2 - v0)) / 2
    return surface_area


# draw a planar mesh
def plotly_draw_2D_mesh(vertices, faces, title="2D Mesh", fig=None, show_now=False):
    # draw the mesh in 2D
    fig = go.Figure()
    # draw the vertices
    fig.add_trace(go.Scatter(x=vertices[:, 0], y=vertices[:, 1], mode="markers"))
    # draw the faces
    for face in faces:
        for i in range(len(face)):
            edge = [face[i], face[(i + 1) % len(face)]]
            fig.add_trace(
                go.Scatter(
                    x=vertices[edge, 0],
                    y=vertices[edge, 1],
                    mode="lines",
                    line=dict(color="red", width=1),
                )
            )

    # set the layout with equal aspect ratio for x and y axes
    fig.update_layout(
        title=title,
        # xaxis=dict(title='X', scaleanchor='y', scaleratio=1),
        yaxis=dict(title="Y"),
        showlegend=False,
    )

    if show_now:
        fig.show()
    return fig


# def create_uv_pos_video(uv_pos_res, faces, save_path):
#     # Parameters for the video
#     output_filename = save_path
#     fps = 15  # Frames per second
#     width, height = 1200, 1200  # Increased resolution for larger canvas

#     # Initialize the video writer
#     fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Codec for MP4
#     video_writer = cv2.VideoWriter(output_filename, fourcc, fps, (width, height))

#     # Create a reusable figure and axis
#     fig, ax = plt.subplots(figsize=(width / 100, height / 100), dpi=100)

#     # find the max and min on x and y axes of the uv_pos_res respectively
#     max_x = max([np.max(uv_pos_res[i][:, 0]) for i in range(len(uv_pos_res))])
#     min_x = min([np.min(uv_pos_res[i][:, 0]) for i in range(len(uv_pos_res))])
#     x_diff = max_x - min_x
#     max_y = max([np.max(uv_pos_res[i][:, 1]) for i in range(len(uv_pos_res))])
#     min_y = min([np.min(uv_pos_res[i][:, 1]) for i in range(len(uv_pos_res))])
#     y_diff = max_y - min_y

#     # Generate frames
#     for i, uv_positions in enumerate(uv_pos_res):
#         # Clear the axis to prepare for the new frame
#         ax.clear()

#         # Plot the mesh
#         plot_2D_mesh(
#             uv_positions,
#             faces,
#             vertex_labels=None,
#             fig=fig,
#             ax=ax,
#             show_vertex_label=False,
#         )
#         ax.set_aspect("equal", "box")
#         # Add the unit square with dashed black boundary
#         unit_square = np.array(
#             [[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]
#         )  # Coordinates of the square (closed loop)
#         ax.plot(
#             unit_square[:, 0],
#             unit_square[:, 1],
#             linestyle="--",
#             color="black",
#             zorder=2,
#         )
#         ax.set_title(f"Frame: {i}", fontsize=16, pad=20)
#         # set the x and y limits
#         ax.set_xlim(min_x - x_diff * 0.1, max_x + x_diff * 0.1)
#         ax.set_ylim(min_y - y_diff * 0.1, max_y + y_diff * 0.1)

#         # plt.tight_layout()

#         # Save the plot to a NumPy array
#         fig.canvas.draw()
#         frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
#         frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))

#         # Write the frame to the video
#         video_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

#     # Release the video writer and close the figure
#     video_writer.release()
#     plt.close(fig)

#     print(f"Video saved as {output_filename}")


def plotly_draw_3D_mesh(
    vertices,
    faces,
    fig=None,
    show_edges=True,
    show_now=False,
    mesh_color="blue",
    edge_color="black",
    mesh_color_opacity=0.3,
    edge_color_width=1,
    edge_color_opacity=0.2,
):
    """
    Draw a 3D mesh using Plotly.

    Parameters
    ----------
    vertices : numpy.ndarray of shape (n, 3)
        The vertices of the mesh.
    faces : numpy.ndarray of shape (m, 3)
        The faces of the mesh.
    fig : plotly.graph_objects.Figure, optional
        The figure to add the mesh to. If None, a new figure is created.
    show_now : bool, optional
        If True, the figure is shown immediately.
    """

    mesh_3d_plot = go.Mesh3d(
        x=vertices[:, 0],
        y=vertices[:, 1],
        z=vertices[:, 2],
        i=faces[:, 0],
        j=faces[:, 1],
        k=faces[:, 2],
        name="Mesh",
        legendgroup="Mesh",
        showlegend=True,
        color=mesh_color,
        opacity=mesh_color_opacity,
        # showscale=True
    )

    if not fig:
        # Define the mesh (cube)
        fig = go.Figure(data=[mesh_3d_plot])
    else:
        fig.add_trace(mesh_3d_plot)

    # Create the edges
    # Define the edges as pairs of vertex indices
    edges = set()
    for f in faces:
        for i in range(3):
            edge = [f[i], f[(i + 1) % 3]]
            edges.add(tuple(sorted(edge)))
    edge_x = []
    edge_y = []
    edge_z = []

    if show_edges:
        for edge in edges:
            for point in edge:
                edge_x.append(vertices[point][0])
                edge_y.append(vertices[point][1])
                edge_z.append(vertices[point][2])
            # Add None to prevent incorrect connections
            edge_x.append(None)
            edge_y.append(None)
            edge_z.append(None)

        edges_trace = go.Scatter3d(
            x=edge_x,
            y=edge_y,
            z=edge_z,
            mode="lines",
            line=dict(color=edge_color, width=edge_color_width),
            opacity=edge_color_opacity,
            name="Edges",
            legendgroup="Edges",
            showlegend=True,
        )
        fig.add_trace(edges_trace)

    # Find the min and max values for each axis
    x_range = [vertices[:, 0].min(), vertices[:, 0].max()]
    y_range = [vertices[:, 1].min(), vertices[:, 1].max()]
    z_range = [vertices[:, 2].min(), vertices[:, 2].max()]

    # Compute the max range to enforce equal scaling
    max_range = max(
        x_range[1] - x_range[0], y_range[1] - y_range[0], z_range[1] - z_range[0]
    )
    center_x = sum(x_range) / 2
    center_y = sum(y_range) / 2
    center_z = sum(z_range) / 2

    equal_x_range = [center_x - max_range / 2, center_x + max_range / 2]
    equal_y_range = [center_y - max_range / 2, center_y + max_range / 2]
    equal_z_range = [center_z - max_range / 2, center_z + max_range / 2]

    # Set equal axis scaling
    fig.update_layout(
        scene=dict(
            xaxis=dict(range=equal_x_range),
            yaxis=dict(range=equal_y_range),
            zaxis=dict(range=equal_z_range),
            aspectmode="cube",
        )
    )
    if show_now:
        # Show the figure
        fig.show()
    return fig


def add_path_on_plotly_mesh(
    fig, path_points, figshow_now=False, show_markers=True, path_color="red"
):
    # Add the path as a line
    fig.add_trace(
        go.Scatter3d(
            x=path_points[:, 0],
            y=path_points[:, 1],
            z=path_points[:, 2],
            mode="lines+markers",
            line=dict(color=path_color, width=5),
            marker=dict(size=1, color="black", symbol="circle"),
            name="Path",
        )
    )
    if figshow_now:
        fig.show()
    return fig


def bary_coordinates_to_euc_point(
    face_idx: int,
    bary_coords: np.ndarray,
    faces: np.ndarray,
    vertex_postions: np.ndarray,
) -> np.ndarray:
    """
    Compute the Eucledian point from the barycentric coordinates.
    Parameters
    ----------
    face_idx : int
        The index of the face.
    bary_coords : numpy.ndarray of shape (3,)
        The barycentric coordinates.
    faces : numpy.ndarray of shape (m, 3)
        The faces of the mesh
    vertex_postions : numpy.ndarray of shape (n, 3) or (n, 2)
        The vertex positions of the mesh either 2D or 3D.
    """
    if vertex_postions.shape[1] == 2:
        p_position = np.zeros(2)
    else:
        p_position = np.zeros(3)
    for i in range(3):
        p_position += bary_coords[i] * vertex_postions[faces[face_idx][i]]
    return p_position


# Function to plot 2D mesh with vertex labels
def plot_2D_mesh(
    vertices,
    faces,
    vertex_labels: dict[str, str] | None = None,
    fig=None,
    ax=None,
    edge_color="blue",
    fill_polygon=True,
    fill_color="blue",
    show_vertex_label=False,
    fill_alpha=0.2,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot a 2D mesh with vertex labels.
    Parameters
    ----------
    vertices : numpy.ndarray of shape (n, 2) or (n, 3)
        The vertices of the mesh. The third dimension is optional.
    faces : numpy.ndarray of shape (m, 3)
        The faces of the mesh.
    vertex_labels : dict, optional
        The vertex labels.
    fig : matplotlib.figure.Figure, optional
        The figure to plot the mesh on.
    """

    if fig is None or ax is None:
        fig, ax = plt.subplots()

    # plot traingles
    for face in faces:
        polygon = [vertices[idx] for idx in face]
        polygon = np.array(polygon)
        ax.fill(
            polygon[:, 0],
            polygon[:, 1],
            edgecolor=edge_color,
            fill=fill_polygon,
            color=fill_color,
            alpha=fill_alpha,
            zorder=1,
        )  # Cyan fill with opacity

    # Annotate each vertex with its label
    if show_vertex_label:
        if vertex_labels is not None:
            for idx in range(len(vertices)):
                x, y = vertices[idx]
                ax.text(
                    x,
                    y,
                    vertex_labels[str(idx)] + "/" + str(idx),
                    fontsize=10,
                    ha="center",
                    color="black",
                    zorder=3,
                )
        else:
            for idx in range(len(vertices)):
                x, y = vertices[idx]
                ax.text(
                    x,
                    y,
                    str(idx),
                    fontsize=10,
                    ha="center",
                    color="black",
                    zorder=3,
                )

    # plt.show()
    return fig, ax


def read_uv_obj(filename):
    with open(filename, "r") as f:
        lines = f.readlines()
    vertices = []
    faces = []
    vertex_label = {}
    for line in lines:
        if line[0] == "v":
            # example line is: vt 0.333333 -0.333333
            # so we split the line by space and take the
            # last two elements and convert them to float
            vertices.append([float(i) for i in line.split()[1:]])

        elif line[0] == "f":
            # example line is: f 3/-1 13/0 1/-1
            # we only need the index in front of the /
            # so we split the line by space and then by /
            # and then take the first element of the split
            # and convert it to int
            for i in line.split()[1:]:
                after_cut_index = i.split("/")[0]
                ref_index = i.split("/")[1]
                if after_cut_index not in vertex_label:
                    if ref_index == "-1":
                        vertex_label[after_cut_index] = after_cut_index
                    else:
                        vertex_label[after_cut_index] = ref_index

            faces.append([int(i.split("/")[0]) for i in line.split()[1:]])

    return np.array(vertices), np.array(faces), vertex_label


# read obj file, note that the faces are 1-indexed
def read_obj(filename: str) -> tuple[np.ndarray, np.ndarray]:
    with open(filename, "r") as f:
        lines = f.readlines()
    vertices = []
    faces = []
    for line in lines:
        if line[0] == "v":
            # example line is: v 0.1 -0.2 0.3
            # so we split the line by space and take the
            # last two elements and convert them to float
            vertices.append([float(i) for i in line.split()[1:]])

        elif line[0] == "f":
            # example line is: f 3 2 1
            faces.append([int(i) for i in line.split()[1:]])

    return np.array(vertices), np.array(faces) - 1  # 1-based to 0-based for obj file


def save_obj(vertices: np.ndarray, faces: np.ndarray, filename: str) -> None:
    # if the min of faces is 0, then add 1 to all the faces
    if np.min(faces) == 0:
        faces = faces + 1  # 0-based to 1-based for obj file

    # save the vertices and faces to a .obj file
    with open(filename, "w") as file:
        for v in vertices:  # use :.6f to format the vertices
            file.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for f in faces:
            file.write(f"f {f[0]} {f[1]} {f[2]}\n")
