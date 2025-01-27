"""Utilities regarding sparsity of the global system of equations"""

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt


def plot_sparsity_pattern(nodes: npt.NDArray[np.float64], elements: npt.NDArray[np.int64]) -> None:
    """Plot sparsity pattern of global stiffness matrix"""

    num_nodes = nodes.shape[0]
    num_space_dims = nodes.shape[1]

    num_dof_per_node = num_space_dims
    num_dof = num_nodes * num_dof_per_node

    k = np.zeros((num_dof, num_dof))
    for element in elements:
        for i in element:
            i_start = i * 3
            i_end = i_start + 3
            for j in element:
                j_start = j * 3
                j_end = j_start + 3
                k[i_start:i_end, j_start:j_end] += 1.0

    plt.spy(k, markersize=1)
    plt.title("Sparsity pattern of the global stiffness matrix")
    plt.show()
