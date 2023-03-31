import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from validation_errors import ERRORS
from bem import BEM
from laplace_interpolation import Interpolate
from mesh import Mesh

scirun_fwd = np.load("data/tmutah.npz")['ForwardMatrix']
print(scirun_fwd.shape)
data = np.load("data/nodes_vertices_measures.npz")
mesh_torso_ = Mesh(data["torso_faces"].T.astype(int), data["torso_vertices"].T, np.array([0, 0.2]))
mesh_heart = Mesh(data["heart_faces"].T.astype(int), data["heart_vertices"].T, np.array([0.2, 0.5]))
tpotentials = data["torso_measures"]
bem = BEM([mesh_torso_, mesh_heart], tpotentials)
transfer_matrix = bem.calc_transfer_matrix()
print(transfer_matrix.shape)

fig=plt.figure()
obs=fig.add_subplot(121,projection='3d')
x=np.arange(transfer_matrix.shape[1])
y=np.arange(transfer_matrix.shape[0])
x, y = np.meshgrid(x,y)
z=transfer_matrix
plt.title("A's bem")
obs.plot_surface(x, y, z)

obs=fig.add_subplot(122, projection='3d')
x=np.arange(scirun_fwd.shape[1])
y=np.arange(scirun_fwd.shape[0])
x, y = np.meshgrid(x,y)
z= scirun_fwd
plt.title("SciRun forward matrix")
obs.plot_surface(x, y, z)

plt.show()
