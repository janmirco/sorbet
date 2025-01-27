"""Utilities regarding post-processing and visualization"""

import logging
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pyvista as pv


def get_connectivity(elements: npt.NDArray[np.int64], num_nodes_per_element: np.int64) -> npt.NDArray[np.int64]:
    """Get vector containing the connectivity information for PyVista.UnstructuredGrid"""

    return np.hstack(
        [
            np.full(
                (elements.shape[0], 1),
                num_nodes_per_element,
            ),
            elements,
        ],
    ).flatten()


def get_cell_type_array(num_elements: int, cell_type: str = "HEXAHEDRON") -> npt.NDArray[np.int64]:
    """
    Set array containing VTK cell type number

    See: https://vtk.org/doc/nightly/html/vtkCellType_8h_source.html
    """

    if cell_type != "HEXAHEDRON":
        raise ValueError("Used cell type is not implemented!")

    return 12 * np.ones(num_elements, dtype=np.int64)


def get_mesh(num_elements: np.int64, num_nodes_per_element: np.int64, nodes: npt.NDArray[np.float64], elements: npt.NDArray[np.int64]):
    """"""

    connectivity = get_connectivity(elements, num_nodes_per_element)
    cell_types = get_cell_type_array(num_elements)

    return pv.UnstructuredGrid(connectivity, cell_types, nodes)


def plot_mesh(num_elements: np.int64, num_nodes_per_element: np.int64, nodes: npt.NDArray[np.float64], elements: npt.NDArray[np.int64]) -> None:
    """"""

    mesh = get_mesh(num_elements, num_nodes_per_element, nodes, elements)
    p = pv.Plotter()
    p.add_mesh(mesh, show_edges=True)
    p.show()


def plot_deformed_mesh(num_elements: np.int64, num_nodes_per_element: np.int64, nodes: npt.NDArray[np.float64], elements: npt.NDArray[np.int64], displacement: npt.NDArray[np.float64]) -> None:
    """"""

    original_mesh = get_mesh(num_elements, num_nodes_per_element, nodes, elements)

    connectivity = get_connectivity(elements, num_nodes_per_element)
    cell_types = get_cell_type_array(num_elements)
    deformed_nodes = np.copy(nodes) + displacement
    deformed_mesh = pv.UnstructuredGrid(connectivity, cell_types, deformed_nodes)

    p = pv.Plotter()
    p.add_mesh(original_mesh, color="white", opacity=0.5)
    p.add_mesh(deformed_mesh, show_edges=True)
    p.show()


def color_plot_nodal_values(
    num_elements: np.int64,
    num_nodes_per_element: np.int64,
    nodes: npt.NDArray[np.float64],
    elements: npt.NDArray[np.int64],
    displacement: npt.NDArray[np.float64],
    nodal_values: npt.NDArray[np.float64],
    nodal_values_name: str,
    cmap: str = "jet",
    opacity: float = 1.0,
    show_edges: bool = False,
    deformed: bool = False,
) -> None:
    """
    Options for cmap: https://matplotlib.org/stable/users/explain/colors/colormaps.html
    """

    if deformed:
        nodes += displacement

    mesh = get_mesh(num_elements, num_nodes_per_element, nodes, elements)
    mesh[nodal_values_name] = nodal_values

    p = pv.Plotter()
    p.add_mesh(mesh, scalars=nodal_values_name, cmap=cmap, opacity=opacity, show_edges=show_edges)
    p.show()


def create_output_dir(dir_name: str = "output") -> Path:
    """"""

    current_dir = Path.cwd()
    output_dir = current_dir / Path(dir_name)
    if not output_dir.exists():
        output_dir.mkdir()
        logging.info(f"Created output directory! See: {output_dir}")
    return output_dir


def save_nodal_values(nodal_values: npt.NDArray[np.float64], file_name: str) -> None:
    """"""

    output_dir = create_output_dir()

    nodal_values_txt = output_dir / Path(f"{file_name}.txt")
    np.savetxt(nodal_values_txt, nodal_values, delimiter=" ")

    nodal_values_npy = output_dir / Path(f"{file_name}.npy")
    np.save(nodal_values_npy, nodal_values)
