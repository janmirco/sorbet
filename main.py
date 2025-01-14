"""Sorbet"""

from sorbet.mesh import create_notched_specimen


def main() -> None:
    mesh = create_notched_specimen(show_mesh=True)
    nodes = mesh.points
    elements = mesh.cells_dict["hexahedron"]

    print(nodes[0])
    print(elements[0])
