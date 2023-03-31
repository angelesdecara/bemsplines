import numpy as np
import os
import scipy.special
import itertools
import scipy.io

readtorsomesh=scipy.io.loadmat("/home/angeles/Downloads/UtahExp/Meshes/tank_771_closed.mat")
torso_vertices=readtorsomesh['torso']['node'][0,0].T
torso_faces=readtorsomesh['torso']['face'][0,0].T


readcagemesh=scipy.io.loadmat("/home/angeles/Downloads/UtahExp/Meshes/cage.mat")
heart_vertices = readcagemesh['cage']['node'][0,0].T
heart_faces = readcagemesh['cage']['face'][0,0].T


torsopotsControl=scipy.io.loadmat("/home/angeles/Downloads/UtahExp/Interventions/Control/Torso/rsm15may02-ts-0003.mat")
tpotentials=torsopotsControl['ts']['potvals'][0,0]

np.savez("UtahTankCage.npz",torso_faces=torso_faces,torso_vertices=torso_vertices,heart_faces=heart_faces,heart_vertices=heart_vertices,torso_measures=tpotentials)

