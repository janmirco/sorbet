import numpy as np

import sorbet


def main() -> None:
    nodes, elements = sorbet.mesh.create_cube(num_elements_thickness=7)
    material_parameters = {"E": 2.1e5, "nu": 0.3}
    K = sorbet.fem.assemble_global_stiffness_matrix(nodes, elements, material_parameters)

    num_nodes = nodes.shape[0]
    f = np.zeros(3 * num_nodes)  # initialize force vector with zeros

    # Find nodes at all six faces
    face_x_min = np.where(np.isclose(nodes[:, 0], 0))[0]
    face_x_max = np.where(np.isclose(nodes[:, 0], nodes[:, 0].max()))[0]
    face_y_min = np.where(np.isclose(nodes[:, 1], 0))[0]
    face_z_min = np.where(np.isclose(nodes[:, 2], 0))[0]

    # Define displacement boundary conditions
    bcs = []
    for node in face_x_min:
        bcs.extend([(node, 0, 0.0)])
    for node in face_x_max:
        bcs.append((node, 0, 0.5))
    for node in face_y_min:
        bcs.extend([(node, 1, 0.0)])
    for node in face_z_min:
        bcs.extend([(node, 2, 0.0)])

    # Apply boundary conditions
    for node, dof, value in bcs:
        dof_index = 3 * node + dof
        K[dof_index, :] = 0
        K[dof_index, dof_index] = 1
        f[dof_index] = value

    # Solve the system
    u = np.linalg.solve(K, f)
    displacement = u.reshape(-1, 3)
    sorbet.post_processing.save_nodal_values(displacement, "displacement")
    num_elements = elements.shape[0]
    num_elements_per_node = elements[0, :].shape[0]
    sorbet.post_processing.plot_deformed_mesh(num_elements, num_elements_per_node, nodes, elements, displacement)
    sorbet.post_processing.color_plot_nodal_values(num_elements, num_elements_per_node, nodes, elements, displacement, displacement[:, 0], "Displacement X", deformed=True)
