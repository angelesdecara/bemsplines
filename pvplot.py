import numpy as np
import pyvista as pv
from vedo import *
from Geometry import Triangulation, faceNormal, vertexNormal_new

def computeNormals(vtx, idx):    
    nrml = np.zeros(vtx.shape, np.float32)

    # compute normal per triangle
    triN = np.cross(vtx[idx[:,1]] - vtx[idx[:,0]], vtx[idx[:,2]] - vtx[idx[:,0]])

    # sum normals at vtx
    nrml[idx[:,0]] += triN[:]
    nrml[idx[:,1]] += triN[:]
    nrml[idx[:,2]] += triN[:]

    # compute norms
    nrmlNorm = np.sqrt(nrml[:,0]*nrml[:,0]+nrml[:,1]*nrml[:,1]+nrml[:,2]*nrml[:,2])
    
    return nrml/nrmlNorm.reshape(-1,1) 


data=np.load("data/nodes_vertices_measures.npz")
faces0 = data['torso_faces'].T
pvfaces = np.insert(faces0, 0 , 3, axis = 1).astype(int)
verts =  data['torso_vertices'].T

#pvtorso = pv.PolyData(verts, np.hstack(pvfaces))
#vtorso = Mesh([verts, faces0.astype(int)])
mesh = pv.PolyData(verts, np.hstack(pvfaces))

plt=pv.Plotter()
c = mesh.cell_centers()
ifaces=faces0.astype(int)
cn = computeNormals(verts, ifaces-1)
mesh.plot()
#show(mesh,pv.plot_arrows(mesh.points[0],cn[0]))
#mesh.compute_normals(consistent_normals=False)
#mesh.plot_normals(mag=0.1, faces=True, show_edges=True)

#vtorso.compute_normals(points=False,cells=True)
## compute_normals is crashing





