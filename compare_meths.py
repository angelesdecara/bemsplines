import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import matplotlib.cm as cm
import numpy as np
import scipy.io as sci

def representateHeart(ax, color_faces ,heart_nodes,heart_faces):
    polygons = []
    for i in range(heart_faces.shape[0]):
        face = heart_faces[i]
        polygon = Poly3DCollection([heart_nodes[face]], alpha=.75, facecolor= color_faces[i] ,linewidths=2)
        polygons.append(polygon)
        ax.add_collection3d(polygon)

    # Establecemos los límites del gráfico en función de los valores mínimos y máximos de las coordenadas x, y, y z de los vértices
    ax.set_xlim3d(np.min(heart_nodes[:, 0]), np.max(heart_nodes[:, 0]))
    ax.set_ylim3d(np.min(heart_nodes[:, 1]), np.max(heart_nodes[:, 1]))
    ax.set_zlim3d(np.min(heart_nodes[:, 2]), np.max(heart_nodes[:, 2]))
    

def interpolate_nodes2faces(colorNodes,nodes, faces ,MAX_VALUE):
    colored_faces = []
    for face in faces:
        face_color = (colorNodes[face[0]] + colorNodes[face[1]] + colorNodes[face[2]]) / (3*MAX_VALUE)
        colored_faces.append(face_color)
    return  cm.jet(colored_faces)


def plot_meth_3D(m_vm, m_vm_est1 , m_vm_est2 , m_vm_est3 ,heart_nodes, heart_faces, time):
    #3D represntation
    fig = plt.figure()
    ax1 = fig.add_subplot(2, 2, 1, projection='3d')
    ax2 = fig.add_subplot(2, 2, 2, projection='3d')
    ax3 = fig.add_subplot(2, 2, 3, projection='3d')
    ax4 = fig.add_subplot(2, 2, 4, projection='3d')
    
    MAX_COLOR = np.max(m_vm) + 1
    MIN_COLOR = np.max(m_vm) - 1

    # Extraer las coordenadas x, y, z de los vértices
    color_faces = interpolate_nodes2faces(m_vm[:,time],heart_nodes, heart_faces ,MAX_COLOR)
    representateHeart(ax1, color_faces ,heart_nodes,heart_faces)
    ax1.set_title('ORIGINAL' )

    color_faces = interpolate_nodes2faces(m_vm_est1[:,time],heart_nodes, heart_faces ,MAX_COLOR)
    representateHeart(ax2, color_faces ,heart_nodes,heart_faces)
    ax2.set_title('ROJO' )

    color_faces = interpolate_nodes2faces(m_vm_est2[:,time],heart_nodes, heart_faces ,MAX_COLOR)
    representateHeart(ax3, color_faces ,heart_nodes,heart_faces)
    ax3.set_title('SPLINES' )

    color_faces = interpolate_nodes2faces(m_vm_est3[:,time],heart_nodes, heart_faces ,MAX_COLOR)
    representateHeart(ax4, color_faces ,heart_nodes,heart_faces)
    ax4.set_title('MFS' )

    sm = plt.cm.ScalarMappable(cmap='jet')
    sm.set_array([MAX_COLOR , (MAX_COLOR - MIN_COLOR)/2 , MIN_COLOR])
    fig.colorbar(sm)
    
    plt.show()

def plot_RMSE(RF1, RC1,MRF1,MRC1, RF2, RC2,MRF2,MRC2, RF3, RC3,MRF3,MRC3):
    fig = plt.figure()
    ax1 = fig.add_subplot(2, 3, 1)
    ax2 = fig.add_subplot(2, 3, 2)
    ax3 = fig.add_subplot(2, 3, 3)
    ax4 = fig.add_subplot(2, 3, 4)
    ax5 = fig.add_subplot(2, 3, 5)
    ax6 = fig.add_subplot(2, 3, 6)

    ax1.set_title('RMSE x tiempo ROJO, medio: ' + str(MRF1))
    ax1.set_xlabel('Tiempo')
    ax1.set_ylabel('RMSE')
    ax1.plot(RF1)

    ax2.set_title('RMSE BEM + SPLINES, medio: ' + str(MRF2))
    ax2.set_xlabel('Tiempo')
    ax2.set_ylabel('RMSE')
    ax2.plot(RF2)

    ax3.set_title('RMSE MFS, medio: ' + str(MRF3))
    ax3.set_xlabel('Tiempo')
    ax3.set_ylabel('RMSE')
    ax3.plot(RF3)

    ax4.set_title('RMSE ROJO, medio: ' + str(MRC1))
    ax4.set_xlabel('NODOS')
    ax4.set_ylabel('RMSE')
    ax4.plot(RC1)

    ax5.set_title('RMSE BEM + SPLINES, medio: ' + str(MRC2))
    ax5.set_xlabel('NODOS')
    ax5.set_ylabel('RMSE')
    ax5.plot(RC2)

    ax6.set_title('RMSE MFS, medio: ' + str(MRC3))
    ax6.set_xlabel('NODOS')
    ax6.set_ylabel('RMSE')
    ax6.plot(RC3)

    fig.tight_layout()
    plt.show()

def plot_SMAPE(RF1, RC1,MRF1,MRC1, RF2, RC2,MRF2,MRC2, RF3, RC3,MRF3,MRC3):
    fig = plt.figure()
    ax1 = fig.add_subplot(2, 3, 1)
    ax2 = fig.add_subplot(2, 3, 2)
    ax3 = fig.add_subplot(2, 3, 3)
    ax4 = fig.add_subplot(2, 3, 4)
    ax5 = fig.add_subplot(2, 3, 5)
    ax6 = fig.add_subplot(2, 3, 6)

    ax1.set_title('SMAPE ROJO, medio: ' + str(MRF1))
    ax1.set_xlabel('Tiempo')
    ax1.set_ylabel('RMSE')
    ax1.plot(RF1)

    ax2.set_title('SMAPE BEM + SPLINES, medio: ' + str(MRF2))
    ax2.set_xlabel('Tiempo')
    ax2.set_ylabel('RMSE')
    ax2.plot(RF2)

    ax3.set_title('SMAPE MFS, medio: ' + str(MRF3))
    ax3.set_xlabel('Tiempo')
    ax3.set_ylabel('RMSE')
    ax3.plot(RF3)

    ax4.set_title('SMAPE ROJO, medio: ' + str(MRC1))
    ax4.set_xlabel('NODOS')
    ax4.set_ylabel('RMSE')
    ax4.plot(RC1)

    ax5.set_title('SMAPE BEM + SPLINES, medio: ' + str(MRC2))
    ax5.set_xlabel('NODOS')
    ax5.set_ylabel('RMSE')
    ax5.plot(RC2)

    ax6.set_title('SMAPE MFS , medio: ' + str(MRC3))
    ax6.set_xlabel('NODOS')
    ax6.set_ylabel('RMSE')
    ax6.plot(RC3)

    fig.tight_layout()
    plt.show()

def plot_XCORR(RF1, RC1,MRF1,MRC1, RF2, RC2,MRF2,MRC2, RF3, RC3,MRF3,MRC3,TIME):
    fig = plt.figure()
    ax1 = fig.add_subplot(2, 3, 1)
    ax2 = fig.add_subplot(2, 3, 2)
    ax3 = fig.add_subplot(2, 3, 3)
    ax4 = fig.add_subplot(2, 3, 4)
    ax5 = fig.add_subplot(2, 3, 5)
    ax6 = fig.add_subplot(2, 3, 6)

    ax1.set_title('XCORR ROJO, medio: ' + str(MRF1))
    ax1.set_xlabel('Tiempo')
    ax1.set_ylabel('RMSE')
    ax1.plot(RF1[TIME])

    ax2.set_title('XCORR BEM + SPLINES , medio: ' + str(MRF2))
    ax2.set_xlabel('Tiempo')
    ax2.set_ylabel('RMSE')
    ax2.plot(RF2[TIME])

    ax3.set_title('XCORR MFS ,medio: ' + str(MRF3))
    ax3.set_xlabel('Tiempo')
    ax3.set_ylabel('RMSE')
    ax3.plot(RF3[TIME])

    ax4.set_title('XCORR ROJO ,medio: ' + str(MRC1))
    ax4.set_xlabel('NODOS')
    ax4.set_ylabel('RMSE')
    ax4.plot(RC1[TIME])

    ax5.set_title('XCORR BEM + SPLINES ,medio: ' + str(MRC2))
    ax5.set_xlabel('NODOS')
    ax5.set_ylabel('RMSE')
    ax5.plot(RC2[TIME])

    ax6.set_title('XCORR MFS ,medio: ' + str(MRC3))
    ax6.set_xlabel('NODOS')
    ax6.set_ylabel('RMSE')
    ax6.plot(RC3[TIME])

    fig.tight_layout()
    plt.show()

def calculate_RMSE(m_vm ,m_vm_est):

    RMSE = []
    for i in range(0,m_vm.shape[1]):
        RMSE.append( np.sqrt(np.sum(np.power(m_vm[:,i] - m_vm_est[:,i],2) / m_vm.shape[0])))

    RMSE_temp = np.array(RMSE)
    RMSE_mean_temp = np.sum(RMSE_temp) / RMSE_temp.shape[0]

    RMSE = []
    for i in range(0,m_vm.shape[0]):
        RMSE.append( np.sqrt(np.sum(np.power(m_vm[i] - m_vm_est[i],2) / m_vm.shape[1])))

    RMSE_nodes = np.array(RMSE)
    RMSE_mean_nodes = np.sum(RMSE_nodes) / RMSE_nodes.shape[0]

    return RMSE_temp, RMSE_mean_temp , RMSE_nodes, RMSE_mean_nodes


def calculate_CORR(m_vm ,m_vm_est):

    CORR = []
    for i in range(0,m_vm.shape[1]):
        CORR.append(  np.correlate(m_vm[:,i], m_vm_est[:,i], mode='full'))

    CORR_temp = np.array(CORR)
    CORR_mean_temp = np.sum(CORR_temp) / CORR_temp.shape[0]

    CORR = []
    for i in range(0,m_vm.shape[0]):
        CORR.append(np.correlate(m_vm[i], m_vm_est[i], mode='full'))

    CORR_nodes = np.array(CORR)
    CORR_mean_nodes = np.sum(CORR_nodes) / CORR_nodes.shape[0]

    return CORR_temp, CORR_mean_temp , CORR_nodes, CORR_mean_nodes
    

def smape_formula(actual, predicted) -> float:
    smape = 100 * np.mean(np.abs(predicted - actual) / (np.abs(predicted) + np.abs(actual)/2 ))
    return smape

def notZero(v ,MINIMO):
    v[v == 0] = MINIMO
    return v

def calculate_SMAPE(m_vm, m_vm_est):
    
    #Evitar 0's
    MIN = 0.00000000000000000000000000000000000000000000000000000000000000000000000000000000000001
    m_vm = notZero(m_vm ,MIN)
    m_vm_est = notZero(m_vm_est,MIN)
    ############

    SMAPE = []
    for i in range(0,m_vm.shape[1]):
        SMAPE.append(smape_formula(m_vm[:,i] ,m_vm_est[:,i] ))

    SMAPE_temp = np.array(SMAPE)
    SMAPE_mean_temp = np.sum(SMAPE_temp) / SMAPE_temp.shape[0]
    
    SMAPE_nodes = []
    for i in range(0,m_vm.shape[0]):
        SMAPE_nodes.append( smape_formula(m_vm[i] ,m_vm_est[i] ))

    SMAPE_nodes = np.array(SMAPE_nodes)
    SMAPE_mean_nodes = np.sum(SMAPE_nodes) / SMAPE_nodes.shape[0]

    return SMAPE_temp, SMAPE_mean_temp , SMAPE_nodes, SMAPE_mean_nodes


def compare_meth():

    #Para almacenar los resultados obtenidos utilizad la siguiente sintaxis y guardadlo todo en la carpeta results"
    #                       np.save('results/rojo_INV.npy', matriz_estimada)

    #LOAD ORIGINAL AND MESH
        #Potencial
    ORIGINAL = sci.loadmat('../../Interventions/Control/Cage/rsm15may02-cage-0003.mat')['ts']['potvals'][0,0]
        #3D
    readcagemesh = sci.loadmat("../../Meshes/cage.mat")
    heart_nodes= readcagemesh['cage']['node'][0,0]
    heart_faces = readcagemesh['cage']['face'][0,0]
    heart_faces = heart_faces -1                                                    #julia/matlab - python offset
    heart_nodes = heart_nodes.T
    heart_faces = heart_faces.T

    #LOAD RESULTS
    ROJO = np.ones(ORIGINAL.shape) * 8000   #PlaceHolder
    SPLINE = np.ones(ORIGINAL.shape) *8000  #PlaceHolder
    MFS = np.ones(ORIGINAL.shape) * 8000    #Placeholder

    #DESCOMENTAR!!!!!!!!!!!!!!!!!!!!!!!
    #ROJO =  np.load("rojo_inv.npy")   #'results/rojo_inv.npy')
    SPLINE = np.load('spline_bem_t.1h.5.npz.npy')
    #MFS = np.load('results/mfs_INV.npy')

    #####3D Representation 
    FIXED_TIME = 200 #Muestra temporal que se quiere analizar (solo para la representacion 3D y correlacion)
    plot_meth_3D(ORIGINAL, ROJO , SPLINE , MFS , heart_nodes, heart_faces, FIXED_TIME)

    #####RMSE
    RMSE_temp_ROJO, RMSE_mean_temp_ROJO , RMSE_nodes_ROJO, RMSE_mean_nodes_ROJO = calculate_RMSE(ORIGINAL,ROJO)
    RMSE_temp_SPLINE, RMSE_mean_temp_SPLINE , RMSE_nodes_SPLINE, RMSE_mean_nodes_SPLINE = calculate_RMSE(ORIGINAL,SPLINE)
    RMSE_temp_MFS, RMSE_mean_temp_MFS , RMSE_nodes_MFS, RMSE_mean_nodes_MFS = calculate_RMSE(ORIGINAL,MFS)

    plot_RMSE(RMSE_temp_ROJO, RMSE_nodes_ROJO, RMSE_mean_temp_ROJO ,RMSE_mean_nodes_ROJO,
     RMSE_temp_SPLINE, RMSE_nodes_SPLINE, RMSE_mean_temp_SPLINE,  RMSE_mean_nodes_SPLINE,
      RMSE_temp_MFS, RMSE_nodes_MFS,RMSE_mean_temp_MFS ,RMSE_mean_nodes_MFS
      )

    #####SMAPE
    SMAPE_temp_ROJO, SMAPE_mean_temp_ROJO, SMAPE_nodes_ROJO, SMAPE_mean_nodes_ROJO = calculate_SMAPE(ORIGINAL, ROJO)
    SMAPE_temp_SPLINE, SMAPE_mean_temp_SPLINE, SMAPE_nodes_SPLINE, SMAPE_mean_nodes_SPLINE = calculate_SMAPE(ORIGINAL, SPLINE)
    SMAPE_temp_MFS, SMAPE_mean_temp_MFS, SMAPE_nodes_MFS, SMAPE_mean_nodes_MFS = calculate_SMAPE(ORIGINAL, MFS)

    plot_SMAPE(SMAPE_temp_ROJO, SMAPE_nodes_ROJO,SMAPE_mean_temp_ROJO,SMAPE_mean_nodes_ROJO,
     SMAPE_temp_SPLINE, SMAPE_nodes_SPLINE,SMAPE_mean_temp_SPLINE,SMAPE_mean_nodes_SPLINE,
      SMAPE_temp_MFS, SMAPE_nodes_MFS , SMAPE_mean_temp_MFS, SMAPE_mean_nodes_MFS)

    #####CORRELATION

    CORR_temp_ROJO, CORR_mean_temp_ROJO, CORR_nodes_ROJO, CORR_mean_nodes_ROJO = calculate_CORR(ORIGINAL, ROJO)
    CORR_temp_SPLINE, CORR_mean_temp_SPLINE, CORR_nodes_SPLINE, CORR_mean_nodes_SPLINE = calculate_CORR(ORIGINAL, SPLINE)
    CORR_temp_MFS, CORR_mean_temp_MFS, CORR_nodes_MFS, CORR_mean_nodes_MFS = calculate_CORR(ORIGINAL, MFS)

    plot_XCORR(CORR_temp_ROJO, CORR_nodes_ROJO,CORR_mean_temp_ROJO,CORR_mean_nodes_ROJO,
     CORR_temp_SPLINE, CORR_nodes_SPLINE,CORR_mean_temp_SPLINE,CORR_mean_nodes_SPLINE,
      CORR_temp_MFS, CORR_nodes_MFS , CORR_mean_temp_MFS, CORR_mean_nodes_MFS , FIXED_TIME)

compare_meth()
