import matplotlib.pyplot as plt
import numpy as np


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

    # plt.show()
    return fig, ax
