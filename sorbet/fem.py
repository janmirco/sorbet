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


def gauss_quadrature(num_points: int = 8) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    match num_points:
        case 8:
            x, y = np.polynomial.legendre.leggauss(2)
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
            raise NotImplementedError(f"Currently only supporting eight Gauss points. num_points = {num_points}")


def element_stiffness_matrix():
    K_e = np.zeros((24, 24))
    points, weights = gauss_quadrature()

    for (xi, eta, zeta), weight in zip(points, weights):
        N = linear_shape_functions(xi, eta, zeta)
        print(N)
        # Calculate B matrix (strain-displacement matrix)
        # Calculate Jacobian
        # Calculate constitutive matrix D
        # K_e += B.T * D * B * det(J) * w_i * w_j

    print()
    raise SystemExit()
    print(K_e)
    return K_e
