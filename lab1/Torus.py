import gmsh
import math
import os
import sys

gmsh.initialize()


gmsh.model.occ.addTorus(1,-3,0, 10, 5)


gmsh.model.occ.synchronize()
gmsh.option.setNumber('Mesh.MeshSizeFromCurvature', 100)
gmsh.model.mesh.generate(3)
gmsh.write('Torus.msh')

# Launch the GUI to see the results:
if '-nopopup' not in sys.argv:
    gmsh.fltk.run()

gmsh.finalize()
