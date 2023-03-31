import numpy as np
import scipy as sp
import scipy.special
import scipy.io

class Interpolate:
    def __init__(self, closed_mesh:str, measured_mesh:str, potentials):
        self.closed_mesh = closed_mesh
        self.measured_mesh = measured_mesh
        self.potentials = potentials

        
    def neighbour_distance_matrix(self, nodes, faces) -> sp.sparse.coo_matrix:
        n, _ = nodes.shape
        m, _ = faces.shape

        rows = faces.flatten(order="F")  # f1, f2, f3
        cols = np.roll(rows, shift=m)  # f3, f1, f2
        data = np.linalg.norm(nodes[rows, :] - nodes[cols, :], ord=2, axis=1)
        # data = np.reciprocal(data) if reciprocal else data

        S = sp.sparse.coo_matrix((data, (rows, cols)), shape=(n, n), dtype=np.float32)
        H = (S + S.T) / 2.0

        return H

    def laplace_operator(self, nodes, faces) -> sp.sparse.coo_matrix:
        H = self.neighbour_distance_matrix(nodes, faces)  # sparse
        b = 1.0 / H.sum(axis=1)  # ndarray
        # C = neighbour_distance_matrix(nodes, faces, reciprocal=True) # sparse
        np.reciprocal(H.data, out=H.data)
        # c = C.sum(axis=1).A1 # ndarray
        c = H.sum(axis=1).A1  # ndarray
        # L = C.multiply(np.broadcast_to(b, shape=(b.size, b.size))) # sparse
        L = H.multiply(np.broadcast_to(b, shape=(b.size, b.size)))  # sparse
        L.setdiag(-c * b.A1, k=0)  # sparse
        L.data *= 4.0
        return L


    def laplace_interpolation(
            self, 
            nodes: np.ndarray,
            faces: np.ndarray,
            potentials: np.ndarray,
            bad_channels: np.ndarray,
            operator: np.ndarray = None,
            in_place: bool = False,
    ) -> np.ndarray:
        """
        See Oostendorp TF, van Oosterom A, Huiskamp G. Interpolation on a triangulated 3D surface. Journal of Computational Physics. 1989 Feb 1;80(2):331–43.
        """

        L = (self.laplace_operator(nodes, faces)).tocsc() if operator is None else operator

        channels = np.arange(L.shape[0], dtype=np.int32)
        good_channels = np.delete(channels, bad_channels)

        L11 = L[np.ix_(bad_channels, bad_channels)]
        L12 = L[np.ix_(bad_channels, good_channels)]
        L21 = L[np.ix_(good_channels, bad_channels)]
        L22 = L[np.ix_(good_channels, good_channels)]
        
        f2 = np.delete(potentials, bad_channels, axis=0)
        Y = -sp.sparse.vstack((L12, L22)).dot(f2)
        A = sp.sparse.vstack((L11, L21))
        f1, _, _, _ = sp.linalg.lstsq(A.toarray(), Y)

        if not in_place:
            interpolated = potentials.copy()
            interpolated[bad_channels] = f1
        else:
            measured[bad_channels] = f1
            interpolated = potentials

        return interpolated


    def discrete_laplace_operator(self, nodes: np.ndarray, faces: np.ndarray) -> sp.sparse.coo_matrix:
        H = self.neighbour_distance_matrix(nodes, faces)  # sparse
        H.data.fill(1)
        H.setdiag(-H.sum(axis=1), k=0)
        return H

    def bad_channels(self):
        closed_torso = scipy.io.loadmat(self.closed_mesh)
        closed_torso_faces = closed_torso['torso']['face'][0,0].T
        closed_torso_vertices = closed_torso['torso']['node'][0,0].T

        measured_torso = scipy.io.loadmat(self.measured_mesh)
        measured_torso_faces = measured_torso['tank']['fac'][0,0].T
        measured_torso_vertices = measured_torso['tank']['pts'][0,0].T

        good_measured_indices = []
        good_closed_indices = []
        for i in np.arange(len(closed_torso_vertices)):
            j = np.argwhere((measured_torso_vertices[:,0] == closed_torso_vertices[i,0])&
                            (measured_torso_vertices[:,1] == closed_torso_vertices[i,1])&
                            (measured_torso_vertices[:,2] == closed_torso_vertices[i,2]))
            if (len(j) > 0):
                good_measured_indices.append([j.ravel()])
                good_closed_indices.append([i])

        good_measured_indices = np.array(good_measured_indices).flatten()
        good_closed_indices = np.array(good_closed_indices).flatten()

        return good_measured_indices, good_closed_indices, closed_torso_vertices, closed_torso_faces

    def calculate_interpolation(self):#closed_mesh:str, measured_mesh:str, potentials: float):
        """
        Function from nodes, faces and measured potentials that builds indices without measurements
        + a potentials array of the shape corresponding to mesh
        and returns interpolated potentials
        """
        good_measured_indices, good_closed_indices, closed_torso_vertices, closed_torso_faces = self.bad_channels()
        closed_potentials=np.empty((len(closed_torso_vertices),self.potentials.shape[1]))
        closed_potentials[:]=np.nan
        
        for idx,el in enumerate(good_closed_indices):
            closed_potentials[el] = self.potentials[good_measured_indices[idx]]
           
        for t in np.arange(closed_potentials.shape[1]):
            bad_indices=[]
            for it,valtp in enumerate(closed_potentials[:,t]):
                if np.isnan(valtp):
                    bad_indices.append(it)
            interpola = self.laplace_interpolation(closed_torso_vertices,closed_torso_faces-1,closed_potentials[:,t],bad_indices)
            closed_potentials[:,t]=interpola

        return closed_potentials

    def calc_interpolation(self):
        return self.calculate_interpolation()
