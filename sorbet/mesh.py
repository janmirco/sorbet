"""Wrapper for easy handling of Gmsh's Python API"""

import logging
from pathlib import Path

import gmsh
import numpy as np
from numpy.typing import NDArray

import sorbet


class GmshManager:
    """Context manager for Gmsh initialization, meshing, finalization, and more"""

    def __init__(self, model_name: str = "Sorbet mesh", debug_mode: bool = False):
        self.model_name = model_name
        self.debug_mode = debug_mode

    def __enter__(self):
        gmsh.initialize()
        gmsh.model.add(self.model_name)
        gmsh.option.set_number("General.Terminal", self.debug_mode)
        gmsh.option.set_number("General.Tooltips", self.debug_mode)
        return self

    def __exit__(self, *_):
        gmsh.finalize()

    def create_mesh(
        self,
        dimension: int = 3,
        mesh_size: float | bool = False,
        recombine_all: bool = True,
        quasi_structured: bool = False,
        element_order: int = 1,
        smoothing: int = 100,
        transfinite_automatic: bool = False,
        mesh_file_name: str = "mesh.msh",
    ) -> None:
        """Mesh created geometry with sane defaults"""

        output_dir = sorbet.paths.setup()
        self.mesh_file = output_dir / Path(mesh_file_name)
        if self.mesh_file.exists():
            logging.info(f"Mesh file with chosen name already exists and is not overwritten: {self.mesh_file}")
            gmsh.open(self.mesh_file.as_posix())

        else:
            if mesh_size:
                gmsh.option.set_number("Mesh.MeshSizeFromPoints", False)
                gmsh.option.set_number("Mesh.MeshSizeMin", mesh_size)
                gmsh.option.set_number("Mesh.MeshSizeMax", mesh_size)
            gmsh.option.set_number("Mesh.RecombineAll", recombine_all)
            if quasi_structured:
                gmsh.option.set_number("Mesh.Algorithm", 11)  # quasi-structured
            gmsh.option.set_number("Mesh.ElementOrder", element_order)
            gmsh.option.set_number("Mesh.Smoothing", smoothing)
            if transfinite_automatic:
                gmsh.model.mesh.set_transfinite_automatic()
            gmsh.model.mesh.generate(dim=dimension)

            # save mesh
            gmsh.write(self.mesh_file.as_posix())

        # store mesh information
        self.nodes = gmsh.model.mesh.get_nodes(dim=dimension, tag=-1, includeBoundary=True, returnParametricCoord=False)[1].reshape(-1, 3)
        element_types, _, element_node_tags_list = gmsh.model.mesh.get_elements(dim=dimension, tag=-1)
        if (element_types.shape[0] > 1) or (element_types[0] != 5):
            raise NotImplementedError(f"Currently only supporting hexahedral elements. Mesh needs to be changed. element_types = {element_types}")
        self.elements = element_node_tags_list[0].reshape(-1, 8) - 1

    def show_geometry(
        self,
        points: bool = True,
        lines: bool = True,
        surfaces: bool = True,
        point_numbers: bool = False,
        line_numbers: bool = False,
        surface_numbers: bool = False,
    ) -> None:
        """Open GUI to show created geometry"""

        gmsh.option.set_number("Geometry.Points", points)
        gmsh.option.set_number("Geometry.Lines", lines)
        gmsh.option.set_number("Geometry.Surfaces", surfaces)
        gmsh.option.set_number("Geometry.PointNumbers", point_numbers)
        gmsh.option.set_number("Geometry.LineNumbers", line_numbers)
        gmsh.option.set_number("Geometry.SurfaceNumbers", surface_numbers)
        gmsh.fltk.run()

    def show_mesh(
        self,
        node_numbers: bool = False,
        element_numbers: bool = False,
        element_surfaces: bool = True,
    ) -> None:
        """Open GUI to show created mesh"""

        gmsh.open(self.mesh_file.as_posix())
        gmsh.option.set_number("Geometry.Points", False)
        gmsh.option.set_number("Geometry.Lines", False)
        gmsh.option.set_number("Geometry.Surfaces", False)
        gmsh.option.set_number("Mesh.SurfaceFaces", element_surfaces)
        gmsh.option.set_number("Mesh.PointNumbers", node_numbers)
        gmsh.option.set_number("Mesh.SurfaceNumbers", element_numbers)
        gmsh.fltk.run()


def create_cube(
    geometry_width: float = 1.0,
    geometry_height: float = 1.0,
    geometry_thickness: float = 1.0,
    mesh_size_plane: float = 0.2,
    num_elements_thickness: int = 3,
    show_geometry: bool = False,
    show_mesh: bool = False,
) -> tuple[NDArray[np.float64], NDArray[np.uint64]]:
    section = "create_cube"
    sorbet.log.start(section)
    with GmshManager() as gm:
        # create geometry
        x, y, z = 0.0, 0.0, 0.0  # position of bottom left point of rectangle
        rec = gmsh.model.occ.add_rectangle(x, y, z, geometry_width, geometry_height)
        gmsh.model.occ.extrude([(2, rec)], dx=0.0, dy=0.0, dz=geometry_thickness, numElements=[num_elements_thickness], recombine=True)
        gmsh.model.occ.synchronize()  # needs to be called before any use of functions outside of the OCC kernel
        if show_geometry:
            gm.show_geometry()

        # create mesh
        gm.create_mesh(dimension=3, mesh_size=mesh_size_plane, transfinite_automatic=True)
        if show_mesh:
            gm.show_mesh()
    sorbet.log.end(section)
    return gm.nodes, gm.elements


def create_notched_specimen(
    geometry_width: float = 8.0,
    geometry_height: float = 3.0,
    geometry_thickness: float = 0.5,
    mesh_size_plane: float = 0.2,
    num_elements_thickness: int = 3,
    show_geometry: bool = False,
    show_mesh: bool = False,
) -> tuple[NDArray[np.float64], NDArray[np.uint64]]:
    section = "create_notched_specimen"
    sorbet.log.start(section)
    with GmshManager() as gm:
        # create geometry
        x, y, z = 0.0, 0.0, 0.0  # position of bottom left point of rectangle
        radius = 0.5
        rec = gmsh.model.occ.add_rectangle(x, y, z, geometry_width, geometry_height)
        cyl1 = gmsh.model.occ.add_cylinder(x + geometry_width / 2.0, y, z - 0.5, 0.0, 0.0, 1.0, radius)
        cyl2 = gmsh.model.occ.add_cylinder(x + geometry_width / 2.0, y + geometry_height, z - 0.5, 0.0, 0.0, 1.0, radius)
        plane = gmsh.model.occ.cut([(2, rec)], [(3, cyl1), (3, cyl2)])[0][0][1]
        gmsh.model.occ.extrude([(2, plane)], dx=0.0, dy=0.0, dz=geometry_thickness, numElements=[num_elements_thickness], recombine=True)
        gmsh.model.occ.synchronize()  # needs to be called before any use of functions outside of the OCC kernel
        if show_geometry:
            gm.show_geometry()

        # create mesh
        gm.create_mesh(dimension=3, mesh_size=mesh_size_plane, quasi_structured=True)
        if show_mesh:
            gm.show_mesh()
    sorbet.log.end(section)
    return gm.nodes, gm.elements
