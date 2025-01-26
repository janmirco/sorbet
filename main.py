import sorbet


def main() -> None:
    nodes, elements = sorbet.mesh.create_cube(num_elements_thickness=7, show_geometry=False, show_mesh=False)
    print(nodes)
    print(f"{nodes.shape = }")
    print(elements)
    print(f"{elements.shape = }")
    print(sorbet.fem.linear_shape_functions(0.0, 0.0, 0.0))
    print(sorbet.fem.linear_shape_function_derivatives(0.0, 0.0, 0.0))
