
import numpy as np

class Triangulation:

    def __init__(self,Points,ConnectivityList):
        self.Points = Points                        # P puntos (vertices)
        self.ConnectivityList = ConnectivityList      # T triangulos (caras m x n)
        #T triangulos (caras m x n) , P puntos (vertices)

    
def faceNormal(tr):

    #tr: Objeto de tipo triangulation

    p = tr.Points
    t = tr.ConnectivityList   #(punto_index1,punto_index2,punto_index3)(Cara_index)
    j=0
    v12 = np.zeros(t.shape)
    v13 = np.zeros(t.shape)
    f_norm = np.zeros(v12.shape)
    for i in t:
        v12 = p[i[1]] - p[i[0]]                                                        #Vector 1 que define al triangulo
        v13 = p[i[2]] - p[i[0]]                                                        #Vector 2 que define al triangulo
        v_norm = np.cross(v12, v13)                                                    #Producto vectorial --> vector perpendicular a la cara en el Punto 1
        #########################
        #v_norm = v_norm - p[i[0]]                  #Desplazar vector del punto P1 al 0,0 --> El codigo original NO lo hace pero habría que mirar si es interesante                          
        #########################
        distance = np.linalg.norm(v_norm)                                       #Normalizar vector
        f_norm[j] = v_norm /distance       
        j +=1      

    return f_norm


def vertexNormal_new(tr):
    
    #Matriz de caras: [cara1,cara2,cara3...] --> cara = [p1,p2,p3] --> p=[x,y,z]
    ns = np.zeros((tr.Points.shape))
    fN = faceNormal(tr)
    t = tr.ConnectivityList
    p = tr.Points
    #fN = tr.ConnectivityList

    for f in range(0,fN.shape[0]):
        for v in (tr.ConnectivityList[f]):
            ns[v,:] += fN[f,:]                                        #ns[:,v] += fN[f,:]

    #normalizacion de los vectores --> Sustituye a mapslices(normalize,ns,dims=2)´
    for  i in range(0,ns.shape[0]):
        module = np.linalg.norm(ns[i,:])                                       #Normalizar vector
        ns[i,:] = ns[i,:] / module

    return ns

