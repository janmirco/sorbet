import numpy as np

import sorbet


def main() -> None:
    nodes, elements = sorbet.mesh.create_cube(num_elements_thickness=7, show_geometry=False, show_mesh=False)
    material_parameters = {"E": 2.1e5, "nu": 0.3}
    K = sorbet.fem.assemble_global_stiffness_matrix(nodes, elements, material_parameters)

    num_nodes = nodes.shape[0]
    f = np.zeros(3 * num_nodes)  # Initialize force vector with zeros

    # Find nodes on the bottom face (z=0) and top face (z=max)
    bottom_nodes = np.where(np.isclose(nodes[:, 2], 0))[0]
    top_nodes = np.where(np.isclose(nodes[:, 2], nodes[:, 2].max()))[0]

    # Define displacement boundary conditions
    bcs = []
    for node in bottom_nodes:
        bcs.extend([(node, 0, 0.0), (node, 1, 0.0), (node, 2, 0.0)])  # Fix bottom face

    for node in top_nodes:
        bcs.append((node, 2, 0.1))  # Prescribe z-displacement of top face

    # Apply boundary conditions
    for node, dof, value in bcs:
        dof_index = 3 * node + dof
        K[dof_index, :] = 0
        K[dof_index, dof_index] = 1
        f[dof_index] = value

    # Solve the system
    u = np.linalg.solve(K, f)
    u = u.reshape(-1, 3)
