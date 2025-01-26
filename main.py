import jax
import jax.numpy as jnp
import numpy as np

import sorbet


def linear_shape_functions_jax(xi: float, eta: float, zeta: float) -> jnp.ndarray:
    return jnp.array(
        [
            0.125 * (1.0 - xi) * (1.0 - eta) * (1.0 - zeta),
            0.125 * (1.0 + xi) * (1.0 - eta) * (1.0 - zeta),
            0.125 * (1.0 + xi) * (1.0 + eta) * (1.0 - zeta),
            0.125 * (1.0 - xi) * (1.0 + eta) * (1.0 - zeta),
            0.125 * (1.0 - xi) * (1.0 - eta) * (1.0 + zeta),
            0.125 * (1.0 + xi) * (1.0 - eta) * (1.0 + zeta),
            0.125 * (1.0 + xi) * (1.0 + eta) * (1.0 + zeta),
            0.125 * (1.0 - xi) * (1.0 + eta) * (1.0 + zeta),
        ]
    )


@jax.jit
def linear_shape_function_derivatives_jax(xi: float, eta: float, zeta: float) -> jnp.ndarray:
    return jnp.array(jax.jacobian(linear_shape_functions_jax, argnums=[0, 1, 2])(xi, eta, zeta)).T


def main() -> None:
    nodes, elements = sorbet.mesh.create_cube(num_elements_thickness=7, show_geometry=False, show_mesh=False)
    # print(nodes)
    # print(f"{nodes.shape = }")
    # print(elements)
    # print(f"{elements.shape = }")

    # print("\nJP:")
    # print(sorbet.fem.linear_shape_functions(0.0, 0.0, 0.0))
    # print(sorbet.fem.linear_shape_function_derivatives(0.0, 0.0, 0.0))
    #
    # print("\nJAX:")
    # print(linear_shape_functions_jax(0.0, 0.0, 0.0))
    # print(linear_shape_function_derivatives_jax(0.0, 0.0, 0.0))

    for i in range(10_000):
        xi, eta, zeta = 2.0 * np.random.rand(3) - 1.0
        jp_1 = sorbet.fem.linear_shape_functions(xi, eta, zeta)
        jp_2 = sorbet.fem.linear_shape_function_derivatives(xi, eta, zeta)
        jax_1 = linear_shape_functions_jax(xi, eta, zeta)
        jax_2 = linear_shape_function_derivatives_jax(xi, eta, zeta)

        if not np.allclose(jp_1, jax_1):
            print(i)
            print(f"{jp_1 = }")
            print(f"{jax_1 = }")
            raise ValueError("shape functions wrong")

        if not np.allclose(jp_2, jax_2):
            print(i)
            print(f"{jp_2 = }")
            print(f"{jax_2 = }")
            raise ValueError("shape function derivatives wrong")
