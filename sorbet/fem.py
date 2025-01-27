import numpy as np
from numpy.typing import NDArray


def linear_shape_functions(xi: float, eta: float, zeta: float) -> NDArray[np.float64]:
    N = np.zeros(8)
    N[0] = 0.125 * (1.0 - xi) * (1.0 - eta) * (1.0 - zeta)
    N[1] = 0.125 * (1.0 + xi) * (1.0 - eta) * (1.0 - zeta)
    N[2] = 0.125 * (1.0 + xi) * (1.0 + eta) * (1.0 - zeta)
    N[3] = 0.125 * (1.0 - xi) * (1.0 + eta) * (1.0 - zeta)
    N[4] = 0.125 * (1.0 - xi) * (1.0 - eta) * (1.0 + zeta)
    N[5] = 0.125 * (1.0 + xi) * (1.0 - eta) * (1.0 + zeta)
    N[6] = 0.125 * (1.0 + xi) * (1.0 + eta) * (1.0 + zeta)
    N[7] = 0.125 * (1.0 - xi) * (1.0 + eta) * (1.0 + zeta)
    return N


def linear_shape_function_derivatives(xi: float, eta: float, zeta: float) -> NDArray[np.float64]:
    dN = np.zeros((8, 3))

    dN[0, 0] = -0.125 * (eta - 1.0) * (zeta - 1.0)
    dN[0, 1] = -0.125 * (xi - 1.0) * (zeta - 1.0)
    dN[0, 2] = -0.125 * (xi - 1.0) * (eta - 1.0)

    dN[1, 0] = 0.125 * (eta - 1.0) * (zeta - 1.0)
    dN[1, 1] = 0.125 * (xi + 1.0) * (zeta - 1.0)
    dN[1, 2] = 0.125 * (xi + 1.0) * (eta - 1.0)

    dN[2, 0] = -0.125 * (eta + 1.0) * (zeta - 1.0)
    dN[2, 1] = -0.125 * (xi + 1.0) * (zeta - 1.0)
    dN[2, 2] = -0.125 * (xi + 1.0) * (eta + 1.0)

    dN[3, 0] = 0.125 * (eta + 1.0) * (zeta - 1.0)
    dN[3, 1] = 0.125 * (xi - 1.0) * (zeta - 1.0)
    dN[3, 2] = 0.125 * (xi - 1.0) * (eta + 1.0)

    dN[4, 0] = 0.125 * (eta - 1.0) * (zeta + 1.0)
    dN[4, 1] = 0.125 * (xi - 1.0) * (zeta + 1.0)
    dN[4, 2] = 0.125 * (xi - 1.0) * (eta - 1.0)

    dN[5, 0] = -0.125 * (eta - 1.0) * (zeta + 1.0)
    dN[5, 1] = -0.125 * (xi + 1.0) * (zeta + 1.0)
    dN[5, 2] = -0.125 * (xi + 1.0) * (eta - 1.0)

    dN[6, 0] = 0.125 * (eta + 1.0) * (zeta + 1.0)
    dN[6, 1] = 0.125 * (xi + 1.0) * (zeta + 1.0)
    dN[6, 2] = 0.125 * (xi + 1.0) * (eta + 1.0)

    dN[7, 0] = -0.125 * (eta + 1.0) * (zeta + 1.0)
    dN[7, 1] = -0.125 * (xi - 1.0) * (zeta + 1.0)
    dN[7, 2] = -0.125 * (xi - 1.0) * (eta + 1.0)

    return dN


def B_operator(dN, J):
    dN_phys = dN @ np.linalg.inv(J)
    B = np.zeros((6, 24))
    for i in range(8):
        index_start = 3 * i
        index_end = 3 * (i + 1)
        B[0, index_start:index_end] = dN_phys[i, 0], 0.0, 0.0
        B[1, index_start:index_end] = 0.0, dN_phys[i, 1], 0.0
        B[2, index_start:index_end] = 0.0, 0.0, dN_phys[i, 2]
        B[3, index_start:index_end] = dN_phys[i, 1], dN_phys[i, 0], 0.0
        B[4, index_start:index_end] = dN_phys[i, 2], 0.0, dN_phys[i, 0]
        B[5, index_start:index_end] = 0.0, dN_phys[i, 2], dN_phys[i, 1]
    return B


def gauss_quadrature(num_points: int = 8) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    match num_points:
        case 1:
            points = np.zeros((1, 3))
            weights = np.array([8.0])
            return points, weights
        case 8:
            x, _ = np.polynomial.legendre.leggauss(2)
            points = np.array(
                [
                    [x[0], x[0], x[0]],
                    [x[1], x[0], x[0]],
                    [x[1], x[1], x[0]],
                    [x[0], x[1], x[0]],
                    [x[0], x[0], x[1]],
                    [x[1], x[0], x[1]],
                    [x[1], x[1], x[1]],
                    [x[0], x[1], x[1]],
                ]
            )
            weights = np.ones(num_points)
            return points, weights
        case _:
            raise NotImplementedError(f"Currently only supporting one or eight Gauss points. num_points = {num_points}")


def linear_elastic_material_tangent(E: float, nu: float) -> NDArray[np.float64]:
    lmb = (E * nu) / ((1.0 + nu) * (1.0 - 2.0 * nu))  # Lamé's first parameter
    mu = E / (2.0 * (1.0 + nu))  # Lamé's second parameter (shear modulus)
    C = np.zeros((6, 6))
    C[0, :] = lmb + 2.0 * mu, lmb, lmb, 0.0, 0.0, 0.0
    C[1, :] = lmb, lmb + 2.0 * mu, lmb, 0.0, 0.0, 0.0
    C[2, :] = lmb, lmb, lmb + 2.0 * mu, 0.0, 0.0, 0.0
    C[3, :] = 0.0, 0.0, 0.0, mu, 0.0, 0.0
    C[4, :] = 0.0, 0.0, 0.0, 0.0, mu, 0.0
    C[5, :] = 0.0, 0.0, 0.0, 0.0, 0.0, mu
    return C


def element_stiffness_matrix(element_nodes, material_parameters):
    K_e = np.zeros((24, 24))
    points, weights = gauss_quadrature(num_points=8)
    C = linear_elastic_material_tangent(E=material_parameters["E"], nu=material_parameters["nu"])
    for (xi, eta, zeta), weight in zip(points, weights):
        dN = linear_shape_function_derivatives(xi, eta, zeta)
        J = element_nodes.T @ dN
        B = B_operator(dN, J)
        K_e += B.T @ C @ B * np.linalg.det(J) * weight
    return K_e


def assemble_global_stiffness_matrix(nodes, elements, material_parameters):
    num_nodes = nodes.shape[0]
    K = np.zeros((3 * num_nodes, 3 * num_nodes))
    for element in elements:
        K_e = element_stiffness_matrix(nodes[element], material_parameters)
        for i, node_i in enumerate(element):
            for j, node_j in enumerate(element):
                i_global = 3 * node_i
                j_global = 3 * node_j
                K[i_global : i_global + 3, j_global : j_global + 3] += K_e[3 * i : 3 * i + 3, 3 * j : 3 * j + 3]
    return K
