from dataclasses import dataclass

import numpy as np


@dataclass
class Mesh:
    triplets: np.ndarray
    points: np.ndarray
    sigma: np.ndarray


# routine to determine surface normals at mesh nodes
def mesh_node_normals(mesh):
    vertices, faces = mesh.points, mesh.triplets

    # Create a zeroed array with the same type and shape as our vertices i.e., per
    # vertex normal
    norm = np.zeros(vertices.shape, dtype=vertices.dtype)
    dots = np.zeros(faces.shape, dtype=float)
    # Create an indexed view into the vertex array using the array of three indices
    # for triangles
    tris = vertices[faces]
    # Calculate the normal for all the triangles, by taking the cross product of
    # the vectors v1-v0, and v2-v0 in each triangle
    n = np.cross(tris[::, 1] - tris[::, 0], tris[::, 2] - tris[::, 0])
    dots[:, 0] = angle_between(tris[::, 1] - tris[::, 0], tris[::, 2] - tris[::, 0])
    dots[:, 1] = angle_between(tris[::, 0] - tris[::, 1], tris[::, 2] - tris[::, 1])
    dots[:, 2] = 1.0 - dots[:, 0] - dots[:, 1]
    # n is now an array of normals per triangle. The length of each normal is
    # dependent the vertices,
    # we need to normalize these, so that our next step weights each normal equally.
    n = n / np.linalg.norm(n)  # Normalize
    # now we have a normalized array of normals, one per triangle, i.e., per
    # triangle normals.
    # But instead of one per triangle (i.e., flat shading), we add to each vertex
    # in that triangle,
    # the triangles' normal. Multiple triangles would then contribute to every
    # vertex, so we need to normalize again afterwards.
    # The cool part, we can actually add the normals through an indexed view of
    # our (zeroed) per vertex normal array
    #       - This bit did not seem to work, hence the for loop - PRJ.
    for i in range(len(faces)):
        norm[faces[i, 0]] += n[i] * dots[i, 0]
        norm[faces[i, 1]] += n[i] * dots[i, 1]
        norm[faces[i, 2]] += n[i] * dots[i, 2]
    norm = norm / np.linalg.norm(norm)
    return norm


def angle_between(a, b):
    cosang = np.einsum("ij,ij->i", a, b) / np.linalg.norm(a) / np.linalg.norm(b)
    return np.arccos(cosang) / np.pi
