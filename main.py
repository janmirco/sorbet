"""Sorbet"""

import sorbet.mesh as m


def main() -> None:
    mesh = m.create_cube(num_elements_thickness=7, show_geometry=True, show_mesh=True)
    nodes = mesh.points
    elements = mesh.cells_dict["hexahedron"]

    print(nodes[0])
    print(elements[0])
