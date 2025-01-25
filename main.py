"""Sorbet"""

import sorbet


def main() -> None:
    nodes, elements = sorbet.mesh.create_cube(num_elements_thickness=7, show_geometry=False, show_mesh=False)
    print(nodes)
    print(f"{nodes.shape = }")
    print(elements)
    print(f"{elements.shape = }")
