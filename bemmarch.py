"""
Author: SpikaTech
Date: 02/02/2023
Description: This file contains the BEM class which is used to solve the inverse problem
"""
import itertools

import numpy as np


class BEM:
    def __init__(self, mesh, tpotentials) -> None:
        self.mesh = mesh
        self.tpotentials = tpotentials

    def _bem_ej_matrix(self, surf1=None, surf2=None, mode="lin"):
        if not isinstance(surf1, np.ndarray):
            surf1 = np.asarray([surf1])

        rowlen = 0
        for i, r in enumerate(surf1):
            rowlen += len(self.mesh[surf1[i]].points)

        rowstart = 0
        row = np.zeros(rowlen)  # array with surface numbers, so goes from 0 to total number of surfaces
        rows = []  # as surfaces have diff number of nodes

        for p in np.arange(len(surf1)):
            rows.append(rowstart + np.arange(len(self.mesh[surf1[p]].points)))
            row[rowstart + np.arange(len(rows[p]))] = surf1[p]
            rowstart = rows[p][-1] + 1

        if not isinstance(surf2, np.ndarray):
            surf2 = np.array([surf2])

        collen = 0
        for i, c in enumerate(surf2):
            if "lin" == mode:
                collen += len(self.mesh[surf2[i]].points)
            if "const" == mode:
                collen += len(self.mesh[surf2[i]].triplets)

        colstart = 0
        cols = []
        col = np.zeros(collen)
        for p in np.arange(len(surf2)):
            if "lin" == mode:
                cols.append(colstart + np.arange(len(self.mesh[surf2[p]].points)))
            else:
                if "const" == mode:
                    cols.append(colstart + np.arange(len(self.mesh[surf2[p]].triplets)))
            col[colstart + np.arange(len(cols[p]))] = surf2[p]
            colstart = cols[p][-1] + 1

        EJ = np.zeros((rowstart, colstart))
        for p in np.arange(len(surf1)):
            for q in np.arange(len(surf2)):
                print("\nCalculating current density to potential transfer surface %d and %d" % (surf1[p], surf2[q]))
                mat = self._ej_calc_matrix(surf1[p], surf2[q], mode)
                for i, r in enumerate(rows[p]):
                    for j, c in enumerate(cols[q]):
                        EJ[r, c] = mat[i, j]
                # EJ[rows[p],cols[q]] = mat

        return EJ, row, col

    def _bem_ee_matrix(self, surf1=None, surf2=None):
        """
        This function computes the transfer matrix for potentials in the boundary element formulation
        Each potential can be written as a sum of contributions of normal current densities and potentials
        at each surface. This function computes the transfer between potentials
        For instance the boundary element is formulated as
         EE*[p1 p2]' + EJ*[j1 j2]' = 0
        In this equation this function computes the EE matrix. It also computes the auto angles if they are
        present.

        Args:
            surf1 (np.ndarray): surface potential at the row space (denoted in surface numbers)
            surf2 (np.ndarray): surface potential at the column space (denoted in surface numbers)
        Returns:
            EE (np.ndarray): submatrix of boundary element method
            row (np.ndarray): vector containing the surface numbers of each row
        """

        ## surf1 and surf2 in matlab in bemPP are arrays, and for two surfaces would be [0,1] ([1 2] in matlab)
        rowlen = 0

        for p in surf1.reshape(-1):  # this would transpose the vector, which I think in python doesnt matter as it's 1d
            rowlen = rowlen + self.mesh[p].points.shape[0]

        row = np.zeros(rowlen)
        rows = []
        rowstart = 0

        for p in np.arange(len(surf1)):
            rows.append(rowstart + np.arange(len(self.mesh[surf1[p]].points)))
            row[(rowstart + np.arange(len(rows[p]))).astype(int)] = surf1[p]
            rowstart = rows[p][-1] + 1

        collen = 0
        for p in surf2.reshape(-1):
            collen = collen + self.mesh[p].points.shape[0]

        col = np.zeros((collen))
        cols = []  # np.zeros((len(surf2),model[surf2[p]].node.shape[1])) #cell(1,len(surf2))
        colstart = 0
        for p in np.arange(len(surf2)):
            cols.append(colstart + np.arange(len(self.mesh[surf2[p]].points)))
            col[(colstart + np.arange(len(cols[p]))).astype(int)] = surf2[p]
            colstart = cols[p][-1] + 1

        EE = np.zeros((int(rowstart), int(colstart)))

        for p in range(len(surf1)):
            for q in range(len(surf2)):
                print("\nCalculating potential transfer surface %d and %d" % (surf1[p], surf2[q]))
                mat = self._ee_calc_matrix(surf1[p], surf2[q])
                i = 0
                j = 0
                for r in rows[p]:
                    for c in cols[q]:
                        EE[r, c] = mat[i, j]
                        j = j + 1
                    j = 0
                    i = i + 1

        return EE, row, col

    def _bem_matrix_pp(self):
        """
        This function computes the transfer matrix between the inner and
        the most outer surface. It is assumed that surface 1 is the outer
        most surface and surface N the inner most. At the outermost surface
        a Neumann condition is assumed that no currents leave the model.

        Returns:
            T (np.ndarray): transfer matrix
        """

        hsurf = len(self.mesh)  # 2 in matlab, it's the innermost surface
        bsurf = np.arange(hsurf)  # 1 to hsurf-1, flat array of indices of outermost to inside surfaces

        # The EE Matrix computed the potential to potential matrix.
        # The boundary element method is formulated as

        # EE*u + EJ*j = source

        # Here u is the potential at the surfaces and j is the current density normal to the surfac.
        # and source are dipolar sources in the interior of the mode space, being ignored in this formulation

        test = 1
        for p in np.arange(len(self.mesh)):
            if hasattr(self.mesh[p], "cal"):
                test *= test
            else:
                test = 0

        EE, row, _ = self._bem_ee_matrix(np.arange(hsurf), np.arange(hsurf))

        Gbh, _, _ = self._bem_ej_matrix(bsurf[0], hsurf - 1)
        Ghh, _, _ = self._bem_ej_matrix(hsurf - 1, hsurf - 1)

        if test == 1:
            # constuct a deflation vector
            # based on the information of which
            # nodes sum to zero.
            eig = np.ones((EE.shape[2 - 1], 1))
            p = np.zeros((1, EE.shape[2 - 1]))
            k = 0
            for q in np.arange(1, len(self.mesh) + 1).reshape(-1):
                p[self.mesh[q].cal + k] = 1
                k = k + self.mesh[q].points.shape[2 - 1]
            p = p / np.nnz(p)
            EE = EE + eig * p
        else: # scirun is based on pp2 which has this line commented
           EE = EE + 1 / (EE.shape[2 - 1]) * np.ones(EE.shape)

        # Get the proper indices for column and row numbers

        b = np.where(row != hsurf - 1)  ## changed =hsurf, as hsurf=len(model)
        h = np.where(row == hsurf - 1)
        # b = find(row != hsurf)
        # h = find(row == hsurf)

        b = list(itertools.chain(*b))
        h = list(itertools.chain(*h))

        Pbb = np.zeros((len(b), len(b)))
        Phh = np.zeros((len(h), len(h)))
        Pbh = np.zeros((len(b), len(h)))
        Phb = np.zeros((len(h), len(b)))

        for i, b1 in enumerate(b):
            for j, b2 in enumerate(b):
                Pbb[i, j] = EE[b1, b2]
            for k, h1 in enumerate(h):
                Pbh[i, k] = EE[b1, h1]
                Phb[k, i] = EE[h1, b1]

        for i, h1 in enumerate(h):
            for j, h2 in enumerate(h):
                Phh[i, j] = EE[h1, h2]

        iGhh = np.linalg.inv(Ghh)
        # Formula as used by Barr et al.

        # The transfer function from innersurface to outer surfaces (forward problem)
        # T = np.linalg.inv(Pbb - Gbh * iGhh * Phb) * (Gbh * iGhh * Phh - Pbh)
        T = np.linalg.inv(Pbb - Gbh @ iGhh @ Phb) @ (Gbh @ iGhh @ Phh - Pbh)
        return T

    def _bem_radon(self, TRI=None, POS=None, OBPOS=None):
        """
        Radon integration over plane triangle, for mono layer
        (linear distributed source strenghth)
        When the distance between the OBPOS and POS is smaller than
        eps this singularity is analytically handled by bem_sing.

        J. Radon (1948), Zur Mechanischen Kubatur,
        # Monat. Mathematik, 52, pp 286-300

        Args:
            TRI (_type_, optional): Indexes of triangles. Defaults to None.
            POS (_type_, optional): Positions [x,y,z] of triangles. Defaults to None.
            OBPOS (_type_, optional): Position of observation point [x,y,z]. Defaults to None.

        Returns:
            W (_type_): Weight, linearly distributed over triangle. W(10,3) is weight of triangle 10, vertex 1
        """

        eps = 1e-12
        # initial weights
        s15 = np.sqrt(15)
        w1 = 9 / 40
        w2 = (155 + s15) / 1200
        w3 = w2
        w4 = w2
        w5 = (155 - s15) / 1200
        w6 = w5
        w7 = w6
        s = (1 - s15) / 7
        r = (1 + s15) / 7
        # how many, allocate output
        nrTRI = len(TRI)
        nrPOS = len(POS)
        W = np.zeros((nrTRI, 3))
        # move all positions to OBPOS as origin
        POS = POS - np.ones((nrPOS, 1)) * OBPOS
        # find how many POS are near OBPOS
        I = np.argwhere(sum(np.transpose(POS) ** 2) < eps)
        Ising = np.empty(0, dtype=int)
        if len(I) != 0:
            Ising = np.empty(0, dtype=int)
            for p in np.arange(len(I)):
                # is a location where TRI==I(p)
                tx = np.argwhere(TRI.T - 1 == I[p])[:,1]
                Ising=np.concatenate((Ising,tx))

        # corners, center and area of each triangle
        P1 = POS[TRI[:, 0] - 1, :]
        P2 = POS[TRI[:, 1] - 1, :]
        P3 = POS[TRI[:, 2] - 1, :]
        C = (P1 + P2 + P3) / 3
        N = np.cross(P2 - P1, P3 - P1)
        A = 0.5 * np.transpose(np.sqrt(sum(np.transpose(N) ** 2)))
        # point of summation (positions)
        q1 = C
        q2 = s * P1 + (1 - s) * C
        q3 = s * P2 + (1 - s) * C
        q4 = s * P3 + (1 - s) * C
        q5 = r * P1 + (1 - r) * C
        q6 = r * P2 + (1 - r) * C
        q7 = r * P3 + (1 - r) * C
        # norm of the positions
        nq1 = np.transpose(np.sqrt(sum(np.transpose(q1) ** 2)))
        nq2 = np.transpose(np.sqrt(sum(np.transpose(q2) ** 2)))
        nq3 = np.transpose(np.sqrt(sum(np.transpose(q3) ** 2)))
        nq4 = np.transpose(np.sqrt(sum(np.transpose(q4) ** 2)))
        nq5 = np.transpose(np.sqrt(sum(np.transpose(q5) ** 2)))
        nq6 = np.transpose(np.sqrt(sum(np.transpose(q6) ** 2)))
        nq7 = np.transpose(np.sqrt(sum(np.transpose(q7) ** 2)))
        # weight factors for linear distribution of strengths
        a1 = 2 / 3
        b1 = 1 / 3
        a2 = 1 - (2 * s + 1) / 3
        b2 = (1 - s) / 3
        a3 = (s + 2) / 3
        b3 = (1 - s) / 3
        a4 = (s + 2) / 3
        b4 = (2 * s + 1) / 3
        a5 = 1 - (2 * r + 1) / 3
        b5 = (1 - r) / 3
        a6 = (r + 2) / 3
        b6 = (1 - r) / 3
        a7 = (r + 2) / 3
        b7 = (2 * r + 1) / 3
        # calculated different weights
        W[:, 0] = np.multiply(
            A,
            (
                (1 - a1) * w1 / nq1
                + (1 - a2) * w2 / nq2
                + (1 - a3) * w3 / nq3
                + (1 - a4) * w4 / nq4
                + (1 - a5) * w5 / nq5
                + (1 - a6) * w6 / nq6
                + (1 - a7) * w7 / nq7
            ),
        )
        W[:, 1] = np.multiply(
            A,
            (
                (a1 - b1) * w1 / nq1
                + (a2 - b2) * w2 / nq2
                + (a3 - b3) * w3 / nq3
                + (a4 - b4) * w4 / nq4
                + (a5 - b5) * w5 / nq5
                + (a6 - b6) * w6 / nq6
                + (a7 - b7) * w7 / nq7
            ),
        )
        W[:, 2] = np.multiply(
            A,
            (
                b1 * w1 / nq1
                + b2 * w2 / nq2
                + b3 * w3 / nq3
                + b4 * w4 / nq4
                + b5 * w5 / nq5
                + b6 * w6 / nq6
                + b7 * w7 / nq7
            ),
        )
        # do singular triangles!
        for i in np.arange(len(Ising)):
            I = Ising[i]
            W[I, :] = self._bem_sing(POS[TRI[I, :] - 1, :])

        return W

    def _bem_sing(self, TRIPOS=None):
        """
        W(J) is the contribution at vertex 1 from unit strength
        at vertex J, J = 1,2,3l

        Args:
            TRIPOS (_type_, optional): Description. Defaults to None.

        Returns:
            W
        """

        eps = 1e-12
        # find point of singularity and arrange tripos
        ISIN = np.argwhere(sum(np.transpose(TRIPOS) ** 2) < eps)
        if len(ISIN) == 0:
            raise Exception("Did not find singularity!")
            return W

        # this function expects the pos(tri(singular,:),:)
        temp = np.array([[0, 1, 2], [1, 2, 0], [2, 0, 1]])
        ARRANGE = temp[ISIN.ravel(), :].ravel()
        # Divide vertices in RA .. RC
        # The singular node is called A, its cyclic neighbours B and C
        RA = TRIPOS[ARRANGE[0], :]
        RB = TRIPOS[ARRANGE[1], :]
        RC = TRIPOS[ARRANGE[2], :]
        # Find projection of vertex A (observation point) on the line
        # running from B to C
        RL, AP = self._laline(RA, RB, RC)

        # find length of vectors BC,BP,CP,AB,AC
        BC = np.linalg.norm(RC - RB, 2)
        BP = np.abs(RL) * BC
        CP = np.abs(1 - RL) * BC
        AB = np.linalg.norm(RB - RA, 2)
        AC = np.linalg.norm(RC - RA, 2)
        # set up basic weights of the rectangular triangle APB
        # WAPB(J) is contribution at vertex A (== observation position!)
        # from unit strength in vertex J, J = A,B,C
        WAPB = np.zeros(3)
        if np.abs(RL) > eps:
            a = AP
            b = BP
            c = AB
            log_term = np.log((b + c) / a)
            WAPB[0] = a / 2 * log_term
            w = 1 - RL
            WAPB[1] = a * ((a - c) * (-1 + w) + b * w * log_term) / (2 * b)
            w = RL
            WAPB[2] = a * w * (a - c + b * log_term) / (2 * b)
        else:
            WAPB = np.array([0, 0, 0])

        # set up basic weights of the rectangular triangle APB
        # WAPC(J) is contribution at vertex A (== observation position!)
        # from unit strength in vertex J, J = A,B,C
        WAPC = np.zeros(3)
        if np.abs(RL - 1) > eps:
            a = AP
            b = CP
            c = AC
            log_term = np.log((b + c) / a)
            WAPC[0] = a / 2 * log_term
            w = 1 - RL
            WAPC[1] = a * w * (a - c + b * log_term) / (2 * b)
            w = RL
            WAPC[2] = a * ((a - c) * (-1 + w) + b * w * log_term) / (2 * b)
        else:
            WAPC = np.array([0, 0, 0])

        # Compute total weights taking into account the position P on BC
        if RL < 0:
            WAPB = -WAPB

        if RL > 1:
            WAPC = -WAPC

        W = WAPB + WAPC
        Wtemp=np.zeros(W.shape)
        for i,e in enumerate(ARRANGE):
            Wtemp[e] = W[i]

        W = Wtemp
        return W

    def _ee_calc_matrix(self, surf1=None, surf2=None):
        Pts = self.mesh[surf1].points
        Pos = self.mesh[surf2].points
        Tri = self.mesh[surf2].triplets
        NumPts = len(Pts)
        NumPos = len(Pos)
        NumTri = len(Tri)
        if hasattr(self.mesh[surf2], "facmaskp"):
            facmaskp = self.mesh[surf2].facemaskp
            Tri = Tri[facmaskp, :]

        In = np.ones((NumTri))
        GeoData=np.zeros((NumPts,NumTri,3))
        for p in np.arange(NumPts):
            # Define all triangles that are no autoangles
            if surf1 == surf2:
                Sel =np.argwhere((Tri[:,0]-1 != p)&(Tri[:,1]-1 != p)&(Tri[:,2]-1 != p)).ravel()
            else:
                Sel = np.arange(NumTri)


            # Define vectors for position p
            # returns a matrix of 3 x ntriangles (or traspose)
            ym = Pts[p,:][:,None] * np.ones(NumTri)
            # this ym is an array of npoints * 3 matrix
            y1 = Pos[Tri[:,0]-1,:] - ym.T  
            y2 = Pos[Tri[:,1]-1,:] - ym.T 
            y3 = Pos[Tri[:,2]-1,:] - ym.T 
            
            epsilon = 1e-12
            gamma = np.zeros((NumTri,3))
            # Speeding up the computation by splitting the formulas
            y21 = y2 - y1
            y32 = y3 - y2
            y13 = y1 - y3

            Ny1 = np.sqrt(np.sum(y1 ** 2,axis=1))
            Ny2 = np.sqrt(np.sum(y2 ** 2,axis=1))
            Ny3 = np.sqrt(np.sum(y3 ** 2,axis=1))
            Ny21 = np.sqrt(np.sum((y21) ** 2,axis=1))
            Ny32 = np.sqrt(np.sum((y32) ** 2,axis=1))
            Ny13 = np.sqrt(np.sum((y13) ** 2,axis=1))

            # element wise multiplication

            NomGamma = Ny1*Ny21 + np.sum(y1*y21,axis=1) 
            DenomGamma = Ny2*Ny21 + np.sum(y2*y21,axis=1)
            W = np.argwhere(np.logical_and(np.logical_and((np.abs(DenomGamma - NomGamma) > epsilon),(DenomGamma != 0)),(NomGamma != 0)))

            gamma[W,0] =-np.ones((1,W.shape[2-1])) / Ny21[W] * np.log(NomGamma[W] / DenomGamma[W])
            NomGamma = Ny2*Ny32 + np.sum(y2*y32,axis=1)

            DenomGamma = Ny3*Ny32 + np.sum(y3*y32,axis=1)
            W = np.argwhere(np.logical_and(np.logical_and((np.abs(DenomGamma - NomGamma) > epsilon),(DenomGamma != 0)),(NomGamma != 0)))

            gamma[W,1] = -np.ones((1,W.shape[2-1])) / Ny32[W]*np.log(NomGamma[W] / DenomGamma[W])

            NomGamma = Ny3*Ny13 + np.sum(y3*y13,axis=1)
            DenomGamma = Ny1*Ny13 + np.sum(y1*y13,axis=1)
            W = np.argwhere(np.logical_and(np.logical_and((np.abs(DenomGamma - NomGamma) > epsilon),(DenomGamma != 0)),(NomGamma != 0)))
            gamma[W,2] = - np.ones((1,W.shape[2-1])) / Ny13[W]*np.log(NomGamma[W] / DenomGamma[W])

            d = np.sum(np.multiply(y1,np.cross(y2,y3)),axis=1)
            N = np.cross(y21,- y13)
            A2 = np.sum(np.multiply(N,N),axis=1)

                
            OmegaVec = (np.transpose(np.ones((1,3)))*(gamma[:,2] - gamma[:,0]))* y1.T + np.transpose(np.ones((1,3)))*(gamma[:,0] - gamma[:,1]) * y2.T + np.transpose(np.ones((1,3))) * (gamma[:,1] - gamma[:,2]) *y3.T
            # In order to avoid problems with the arctan used in de Muncks paper
            # the result is tested. A problem is that his formula under certain
            # circumstances leads to unexpected changes of signs. Hence to avoid
            # this, the denominator is checked and 2*pi is added if necessary.
            # The problem without the two pi results in the following situation in
            # which division of the triangle into three pieces results into
            # an opposite sign compared to the sperical angle of the total
            # triangle. These cases are rare but existing.
            Nn=Ny1*Ny2*Ny3+Ny1*np.sum(y2*y3,axis=1)+Ny2*np.sum(y1*y3,axis=1)+Ny3*np.sum(y2*y1,axis=1)

            Omega = np.zeros(NumTri)
            Vz = np.argwhere(Nn[Sel] == 0)
            Vp = np.argwhere(Nn[Sel] > 0)
            Vn = np.argwhere(Nn[Sel] < 0)

            if Vp.shape[0] > 0:
                Omega[Sel[Vp]] = 2 * np.arctan(d[Sel[Vp]] / Nn[Sel[Vp]])
            if Vn.size > 0:
                Omega[Sel[Vn]] = 2 * np.arctan(d[Sel[Vn]] / Nn[Sel[Vn]]) + 2 * np.pi
            if Vz.size > 0:
                Omega[Sel[Vz]] = np.pi * np.sign(d[Sel[Vz]])
            zn1 = np.sum(np.multiply(np.cross(y2,y3),N),axis=1)
            zn2 = np.sum(np.multiply(np.cross(y3,y1),N),axis=1)
            zn3 = np.sum(np.multiply(np.cross(y1,y2),N),axis=1)
            # Compute spherical angles
            GeoData[p,Sel,0] =(In[Sel]/A2[Sel]) * ((zn1[Sel]*Omega[Sel]) + d[Sel] * np.sum(y32[Sel,:]*OmegaVec[:,Sel].T,axis=1))
            GeoData[p,Sel,1] =(In[Sel]/A2[Sel]) * ((zn2[Sel]*Omega[Sel]) + d[Sel] * np.sum(y13[Sel,:]*OmegaVec[:,Sel].T,axis=1))
            GeoData[p,Sel,2] =(In[Sel]/A2[Sel]) * ((zn3[Sel]*Omega[Sel]) + d[Sel] * np.sum(y21[Sel,:]*OmegaVec[:,Sel].T,axis=1))

        EE = np.zeros((NumPts,NumPos))
        
        C = (1 / (4 * np.pi)) * (self.mesh[surf2].sigma[0] - self.mesh[surf2].sigma[1])
        for q in np.arange(NumPos): # changed from 1 to NumPos+1
            V = np.argwhere(Tri[:,0]-1 == q).ravel()
            for r in np.arange(len(V)):
                EE[:,q] = EE[:,q] - C * GeoData[:,V[r],0]
            V = np.argwhere(Tri[:,1]-1 == q).ravel()
            for r in np.arange(len(V)):
                EE[:,q] = EE[:,q] - C * GeoData[:,V[r],1]
            V = np.argwhere(Tri[:,2]-1 == q).ravel()
            for r in np.arange(len(V)):
                EE[:,q] = EE[:,q] - C * GeoData[:,V[r],2]

    
        if surf1 == surf2:
            print('\nCalculating diagonal elements' % ())
            # added a correction for the layers outside this one.
            # It is assumed that the outer most layer has a conductivity
            # of zero.
            for p in range(NumPts):
                EE[p,p] = - sum(EE[p,:]) + self.mesh[surf2].sigma[0]
    
        print('\nCompleted computation submatrix\n')


        return EE

    
    def _ej_calc_matrix(self, surf1=None, surf2=None, mode="lin"):
        """
        This function deals with the analytical solutions of the various integrals in the stiffness matrix
        The function is concerned with the integrals of the linear interpolations functions over the triangles
        and takes care of the solid spherical angle. As in most cases not all values are needed the computation
        is split up in integrals from one surface to another one.
        The computational scheme follows the analytical formulas derived by the de Munck 1992 (IEEE Trans Biomed Engng, 39-9, pp 986-90)
        The program is based on SBEM_SSOURCE.m (c) Stinstra 1997.
        All comments have been translated from Dutch into English

        Args:
            surf1 (np.ndarray): surface potential at the row space (denoted in surface numbers). Defaults to None.
            surf2 (np.ndarray): surface potential at the column space (denoted in surface numbers). Defaults to None.
            mode (str, optional): Defaults to "lin".

        Returns:
            EJ (np.ndarray)
        """

        Pts = self.mesh[surf1].points
        Pos = self.mesh[surf2].points
        Tri = self.mesh[surf2].triplets
        NumPts = len(Pts)
        NumPos = len(Pos)
        NumTri = len(Tri)
        if mode == "lin":
            if hasattr(self.mesh[surf2], "facmaskj"):
                facmaskj = self.mesh[surf2].facemaskj
                Tri = Tri[facmaskj, :]
            EJ = np.zeros((NumPts, NumPos))
            for k in np.arange(NumPts):
                W = self._bem_radon(Tri, Pos, Pts[k, :])
                for l in np.arange(NumTri):
                    EJ[k, Tri[l, :] - 1] = EJ[k, Tri[l, :] - 1] + (1 / (4 * np.pi)) * W[l, :]


        # What is const mode? Interpolation in the triangles. Why is it not used? TODO: check this
        if mode == "const":
            for p in np.arange(NumPts):
                if surf1 == surf2:
                    Sel = np.where(
                        np.logical_and(
                            np.logical_and((Tri[:, 0] - 1 != p), (Tri[:, 1] - 1 != p)),
                            (Tri[:, 2] - 1 != p),
                        )
                    )
                else:
                    Sel = np.arange(1, NumTri + 1)
                # Define vectors for position p
                ym = np.array([Pts[p, :]]).T * np.ones(NumTri)
                y1 = Pos[Tri[:, 0] - 1, :] - ym
                y2 = Pos[Tri[:, 1] - 1, :] - ym
                y3 = Pos[Tri[:, 2] - 1, :] - ym

                epsilon = 1e-12
                gamma = np.zeros((3, NumTri))
                NomGamma = np.sqrt(np.multiply(sum(np.multiply(y1, y1)), sum(np.multiply((y2 - y1), (y2 - y1))))) + sum(
                    np.multiply(y1, (y2 - y1))
                )
                DenomGamma = np.sqrt(
                    np.multiply(sum(np.multiply(y2, y2)), sum(np.multiply((y2 - y1), (y2 - y1))))
                ) + sum(np.multiply(y2, (y2 - y1)))
                W = np.where(
                    np.logical_and(
                        np.logical_and((np.abs(DenomGamma - NomGamma) > epsilon), (DenomGamma != 0)),
                        (NomGamma != 0),
                    )
                )
                gamma[1, W] = np.multiply(
                    -np.ones((1, W.shape[2 - 1]))
                    / np.sqrt(sum(np.multiply((y2[:, W] - y1[:, W]), (y2[:, W] - y1[:, W])))),
                    np.log(NomGamma[W] / DenomGamma[W]),
                )
                NomGamma = np.sqrt(np.multiply(sum(np.multiply(y2, y2)), sum(np.multiply((y3 - y2), (y3 - y2))))) + sum(
                    np.multiply(y2, (y3 - y2))
                )
                DenomGamma = np.sqrt(
                    np.multiply(sum(np.multiply(y3, y3)), sum(np.multiply((y3 - y2), (y3 - y2))))
                ) + sum(np.multiply(y3, (y3 - y2)))
                W = np.where(
                    np.logical_and(
                        np.logical_and((np.abs(DenomGamma - NomGamma) > epsilon), (DenomGamma != 0)),
                        (NomGamma != 0),
                    )
                )
                gamma[2, W] = np.multiply(
                    -np.ones((1, W.shape[2 - 1]))
                    / np.sqrt(sum(np.multiply((y3[:, W] - y2[:, W]), (y3[:, W] - y2[:, W])))),
                    np.log(NomGamma[W] / DenomGamma[W]),
                )
                NomGamma = np.sqrt(np.multiply(sum(np.multiply(y3, y3)), sum(np.multiply((y1 - y3), (y1 - y3))))) + sum(
                    np.multiply(y3, (y1 - y3))
                )
                DenomGamma = np.sqrt(
                    np.multiply(sum(np.multiply(y1, y1)), sum(np.multiply((y1 - y3), (y1 - y3))))
                ) + sum(np.multiply(y1, (y1 - y3)))
                W = np.where(
                    np.logical_and(
                        np.logical_and((np.abs(DenomGamma - NomGamma) > epsilon), (DenomGamma != 0)),
                        (NomGamma != 0),
                    )
                )
                gamma[3, W] = np.multiply(
                    -np.ones((1, W.shape[2 - 1]))
                    / np.sqrt(sum(np.multiply((y1[:, W] - y3[:, W]), (y1[:, W] - y3[:, W])))),
                    np.log(NomGamma[W] / DenomGamma[W]),
                )
                d = sum(np.multiply(y1, np.cross(y2, y3)))
                N = np.cross((y2 - y1), (y3 - y1))

                # For which OmegaVec is calculated? TODO: check this
                OmegaVec = (
                    np.multiply(np.transpose(np.array([1, 1, 1])) * (gamma[3, :] - gamma[1, :]), y1)
                    + np.multiply(np.transpose(np.array([1, 1, 1])) * (gamma[1, :] - gamma[2, :]), y2)
                    + np.multiply(np.transpose(np.array([1, 1, 1])) * (gamma[2, :] - gamma[3, :]), y3)
                )
                # In order to avoid problems with the arctan used in de Muncks paper
                # the result is tested. A problem is that his formula under certain
                # circumstances leads to unexpected changes of signs. Hence to avoid
                # this, the denominator is checked and 2*pi is added if necessary.
                # The problem without the two pi results in the following situation in
                # which division of the triangle into three pieces results into
                # an opposite sign compared to the sperical angle of the total
                # triangle. These cases are rare but existing.
                Nn = (
                    np.multiply(
                        np.multiply(np.sqrt(sum(np.multiply(y1, y1))), np.sqrt(sum(np.multiply(y2, y2)))),
                        np.sqrt(sum(np.multiply(y3, y3))),
                    )
                    + np.multiply(np.sqrt(sum(np.multiply(y1, y1))), sum(np.multiply(y2, y3)))
                    + np.multiply(np.sqrt(sum(np.multiply(y3, y3))), sum(np.multiply(y1, y2)))
                    + np.multiply(np.sqrt(sum(np.multiply(y2, y2))), sum(np.multiply(y3, y1)))
                )
                Omega = np.zeros((1, NumTri))
                Vz = np.where(Nn[Sel] == 0)
                Vp = np.where(Nn[Sel] > 0)
                Vn = np.where(Nn[Sel] < 0)
                if Vp.shape[1 - 1] > 0:
                    Omega[Sel[Vp]] = 2 * np.arctan(d[Sel[Vp]] / Nn[Sel[Vp]])
                if Vn.shape[1 - 1] > 0:
                    Omega[Sel[Vn]] = 2 * np.arctan(d[Sel[Vn]] / Nn[Sel[Vn]]) + 2 * np.pi
                if Vz.shape[1 - 1] > 0:
                    Omega[Sel[Vz]] = np.pi * np.sign(d[Sel[Vz]])

                # For which zn1, zn2, zn3 are calculated? TODO: check this
                zn1 = sum(np.multiply(np.cross(y2, y3), N))
                zn2 = sum(np.multiply(np.cross(y3, y1), N))
                zn3 = sum(np.multiply(np.cross(y1, y2), N))
                # What is n? TODO: check this
                C = np.multiply((np.transpose(np.array([1, 1, 1])) * sum(np.multiply(y1, n))), n)
                Temp = (-1 / (4 * np.pi)) * (
                    np.multiply(sum(np.multiply(np.cross(y1, y2), n)), gamma[1, :])
                    + np.multiply(sum(np.multiply(np.cross(y2, y3), n)), gamma[2, :])
                    + np.multiply(sum(np.multiply(np.cross(y3, y1), n)), gamma[3, :])
                    - np.multiply(sum(np.multiply(n, C)), Omega)
                )
                EJ[p, :] = Temp

        return EJ

    def _laline(self, ra=None, rb=None, rc=None):
        """Find projection P of vertex A (observation point) on the line
        running from B to C

        Args:
            ra (_type_, optional): _description_. Defaults to None.
            rb (_type_, optional): _description_. Defaults to None.
            rc (_type_, optional): _description_. Defaults to None.

        Returns:
            rl (_type_): factor of vector BC, position p = rl*(rc-rb)+rb
            ap (_type_): distance from A to P
            _type_: _description_
        """

        # difference vectors
        rba = ra - rb
        rbc = rc - rb
        # normalize rbc
        nrbc = rbc / np.linalg.norm(rbc)
        # inproduct and correct for normalization
        rl = np.inner(rba, nrbc)  # rba * np.transpose(nrbc)
        rl = rl / np.linalg.norm(rbc, 2)
        # point of projection
        p = rl * (rbc) + rb
        # distance
        ap = np.linalg.norm(ra - p, 2)
        ##### end laline #####
        return rl, ap

    def calc_transfer_matrix(self):
        return self._bem_matrix_pp()
