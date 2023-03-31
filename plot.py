
import argparse
import json
import logging

import scipy.io
import numpy as np

from mesh import Mesh
from bem import BEM
from laplace_interpolation import Interpolate
from spline_parameters import spline_parameters
from spline import SPLINE
from validation_errors import ERRORS
import matplotlib.pyplot as plt
import pyvista as pv
#from vedo import *
import vedo
from Geometry import Triangulation, faceNormal

torsopotsControl=scipy.io.loadmat("/home/angeles/Downloads/UtahExp/Interventions/Control/Torso/rsm15may02-ts-0003.mat")
tpotentials=torsopotsControl['ts']['potvals'][0,0]


readtorsomesh=scipy.io.loadmat("/home/angeles/Downloads/UtahExp/Meshes/tank_192.mat")
torso_vertices=readtorsomesh['tank']['pts'][0,0].T
torso_faces=readtorsomesh['tank']['fac'][0,0].T 

#
closedtorsomesh=scipy.io.loadmat("/home/angeles/Downloads/UtahExp/Meshes/tank_771_closed.mat")
closed_torso_vertices=closedtorsomesh['torso']['node'][0,0].T
closed_torso_faces=closedtorsomesh['torso']['face'][0,0].T 

torso_mesh = vedo.Mesh([closed_torso_vertices, closed_torso_faces - 1],alpha=0.4)
torso_mesh.backcolor('violet').linecolor('tomato').linewidth(2)
labs = torso_mesh.labels('id').c('black')

torso_pts = vedo.Points(closed_torso_vertices)
torso_pts.cmap("viridis")#, tpotentials[:,0])

torso_center = np.zeros(closed_torso_faces.shape)
for i, t in enumerate(closed_torso_faces-1):
    torso_center[i]= np.sum(closed_torso_vertices[t,:],axis=0)



face_centers = torso_mesh.cell_centers()
torsoTr = Triangulation(Points=closed_torso_vertices, ConnectivityList = closed_torso_faces - 1)
fn = faceNormal(torsoTr)

print(fn+torso_center)
origin_mesh=np.zeros(fn.shape)
print(origin_mesh)
torso_arrows = vedo.Arrows(face_centers, face_centers + fn*100)#, res=100, thickness=10)#(face_centers, fn)

plt = vedo.show(torso_mesh, torso_arrows, torso_pts,  labs, __doc__, viewup='z', axes=1).close()


readcagemesh=scipy.io.loadmat("/home/angeles/Downloads/UtahExp/Meshes/cage.mat")
heart_vertices = readcagemesh['cage']['node'][0,0].T
heart_faces = readcagemesh['cage']['face'][0,0].T


pvtorsofaces = np.array((len(torso_faces),torso_faces.shape[1]+1))
pvtorsofaces = np.insert(torso_faces,0,3,axis=1)

hpots=scipy.io.loadmat("/home/angeles/Downloads/UtahExp/Interventions/Control/Cage/rsm15may02-cage-0003.mat")['ts']['potvals'][0,0]
spline_pots=np.load("9mar_spline_bem_t10h10.npz.npy")


torso = vedo.Mesh([torso_vertices,torso_faces-1]).cmap("jet",tpotentials[:,0],on='points')
vedo.show(torso).close()
heart=vedo.Mesh([heart_vertices,heart_faces-1])
vedo.show(heart).close()

#plot some signals to see if it's one heart cycle
fig = plt.figure()
signal1 = fig.add_subplot(411)
signal1.set_title("spline inferred signal 90")
signal1.plot(spline_pots[90,:])

signal2 = fig.add_subplot(412)
signal2.set_title("observed signal 90")
signal2.plot(tpotentials[90,:])

signal3 = fig.add_subplot(413)
signal3.set_title("spline inferred signal 110")
signal3.plot(spline_pots[110,:])

signal4 = fig.add_subplot(414)
signal4.set_title("observed signal 110")
signal4.plot(tpotentials[110,:])

plt.show()

