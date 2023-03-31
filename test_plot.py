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


hpots=scipy.io.loadmat("../../Interventions/Control/Cage/rsm15may02-cage-0003.mat")
heartpotentials=hpots['ts']['potvals'][0,0]
#errors=ERRORS(heartpotentials,otro)

fig=plt.figure()
obs=fig.add_subplot(221,projection='3d')
x=np.arange(transfer_matrix.shape[1])
y=np.arange(transfer_matrix.shape[0])
x, y = np.meshgrid(x,y)
z=transfer_matrix

obs.plot_surface(x,y,z)

obs=fig.add_subplot(222, projection='3d')
x=np.arange(scirun_fwd.shape[1])
y=np.arange(scirun_fwd.shape[0])
x, y = np.meshgrid(x,y)
z= scirun_fwd
obs.plot_surface(x, y, z)


dim1error = np.empty(transfer_matrix.shape[1])
dim2error = np.empty(transfer_matrix.shape[1])
for t in np.arange(transfer_matrix.shape[1]):
    errors = ERRORS(transfer_matrix[t,:], scirun_fwd[t,:])
    dim1error[t] = errors.calculate_correlation()
    dim2error[t] = errors.calculate_rmse()

obs=fig.add_subplot(223)
obs.plot(dim1error)
obs=fig.add_subplot(224)
obs.plot(dim2error)
plt.show()

