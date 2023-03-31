import argparse
import json
import logging
import numpy as np
import scipy
import scipy.special
import scipy.sparse as sp
from scipy import interpolate

from mesh import Mesh
from bem import BEM
from spline_parameters import spline_parameters

log = logging.getLogger(__name__)

class SPLINE:
    def __init__(self, parameters, transfer_matrix) -> None:
        #self.number_of_knots = number_of_knots
        #self.interpolation_density = interpolation_density
        #self.minimum_derivatives_cost_reduction = minimum_derivatives_cost_reduction
        #self.minimum_overall_cost_reduction = minimum_overall_cost_reduction
        #self.projection_interpolation_density = projection_interpolation_density
        #self.torso_potentials = torso_potentials
        self.parameters = parameters
        print("params =",self.parameters)
        self.transfer_matrix = transfer_matrix

    def initialize_curve_params_from_time_series(self, Data, number_of_knots):
        """
        Initialise temporal series of potentials at chest on knots adding derivatives on edges
        Data = Time series of potentials on chest
        number_of_knots = knots to obtain manifold and cubic piecewise
        """
        dims_Data = np.shape(Data)
        N, T = Data.shape
    
        knotinds = np.floor(np.linspace(1,T, number_of_knots)).astype(int)-1
        Knots = Data[:,knotinds] # data sampled at these times
    
        FirstKnotDeriv = (Knots[:,1]-Knots[:,0])/(knotinds[1]-knotinds[0])
        LastKnotDeriv = (Knots[:,-1]-Knots[:,-2])/(knotinds[-1]-knotinds[-2])

        CurveParams = np.concatenate((FirstKnotDeriv[:,None],Knots,LastKnotDeriv[:,None]),axis=1)
    
        return CurveParams


    def minimize_distance_to_curve(
            self,
            InitialCurveParameters,
            DataPoints,
            InterpolationDensity,
            mincostreduction = 1e-6,
            mode = None,
            PeriodicDims = None):
        """
        """
        DEFAULTmincostreduction = 1e-06
        justderivsflag = 0
        SizeCurveParameters = InitialCurveParameters.shape
        ProdSizeCurveParameters = np.prod(SizeCurveParameters)
        PeriodicDims = []
        PeriodicityFlag = np.zeros((np.asarray(SizeCurveParameters).size - 1,1))

        if (mode == 'JustDerivatives'):
            justderivsflag = 1
            if (mode == 'Periodic'):
                # Unless this is followed by a numeric array of dimension
                # indices that should be periodic, assume they are all
                # periodic
                if (PeriodicDims):
                    if isinstance(PeriodicDims,np.ndarray):
                        PeriodicityFlag[PeriodicDims] = 1
                        if (np.asarray(PeriodicityFlag).size > np.asarray(SizeCurveParameters).size - 1):
                            print('WARNING: Dimensions specified as being periodic exceed input dimensions.\n' % ())
                        else:
                            PeriodicityFlag = np.ones((PeriodicityFlag.shape,PeriodicityFlag.shape))

        # Minimize over CurveParameters -> S

        def wolfeparams(object):
            pass
        
        setattr(wolfeparams,'alphamax' , 1)
    
        phi = lambda S = None: (
            self.total_curve_projections(
                np.reshape(S, tuple(SizeCurveParameters), order="F"),
                DataPoints,
                InterpolationDensity,
                PeriodicDims))

        TensorEdgeDimensions = SizeCurveParameters[1:]
        SplineMatrix = []
        for i in np.arange(np.asarray(TensorEdgeDimensions).size):
            if (PeriodicityFlag[i] == 0):
                SplineMatrix.append(np.transpose(self.spline_matrix(TensorEdgeDimensions[i] - 2,1,InterpolationDensity)))

            else:
                SplineMatrix.append(np.transpose(self.spline_matrix(TensorEdgeDimensions[i] - 2,1,InterpolationDensity,'Periodic')))
                
        dphidx = lambda S = None: self.curve_projections_descent_direction(
            np.reshape(S, tuple(SizeCurveParameters), order="F"),
            DataPoints,
            InterpolationDensity,
            SplineMatrix,
            justderivsflag,
            PeriodicDims,
            PeriodicityFlag)

        CurveParameters,dfdx,alpha = self.steepest_descent(
            phi, dphidx, InitialCurveParameters, mincostreduction, wolfeparams.alphamax)
        CurveParameters = np.reshape(CurveParameters[-1], tuple(SizeCurveParameters), order="F")
        Cost = phi(CurveParameters)
        return CurveParameters, Cost

    def total_curve_projections(self, CurveParameters = None,DataPoints = None,InterpolationDensity = None,Periodicity = None):
        MAXELEMSPAIRWISEDISTS = 5000000.0
        # 1. Interpolate the spline
        if (sum(Periodicity) > 0):
            ISet = self.interpolate_curve(CurveParameters,InterpolationDensity,'Periodic',Periodicity)
        else:
            ISet = self.interpolate_curve(CurveParameters,InterpolationDensity)

        SizeISet = ISet.shape
        ISet = ISet.reshape(SizeISet[0],np.prod(SizeISet[1:]))
        # 3. Calculate distances squared between ISet (row indices) and data (column indices)
        max_elem = int(np.sqrt(MAXELEMSPAIRWISEDISTS))
        P2P = self.pairwise_distance([ISet, DataPoints], max_elem) \
            if (ISet.shape[1] * DataPoints.shape[1] > MAXELEMSPAIRWISEDISTS) \
               else self.pairwise_distance([ISet, DataPoints])
        # 4. Calculate total curve projection distances squared, divided by number of projected data points
        # 3 lines could be faster than 1
        Cost = np.empty(P2P.shape[1])
        np.amin(P2P, axis=0, out=Cost)
        Cost = (1 / np.asarray(DataPoints).size) * np.sum(Cost)

        """
        if (ISet.shape[2-1] * DataPoints.shape[2-1] > MAXELEMSPAIRWISEDISTS):
            P2P = self.pairwise_distance([ISet,DataPoints],int(np.floor(np.sqrt(MAXELEMSPAIRWISEDISTS))))
        else:
            P2P = self.pairwise_distance([ISet,DataPoints])
            # 4. Calculate total curve projection distances squared, divided by number of projected data points
        Cost = (1 / np.asarray(DataPoints).size) * sum(np.amin(P2P,axis=0))
        """
        return Cost


    def curve_projections_descent_direction(
            self,
            CurveParameters = None,
            DataPoints = None,
            InterpolationDensity = None,
            SplineMatrix = None,
            justderivsflag = None,
            PeriodicDims = None,
            PeriodicityFlag = None):

        SizeCurveParams = np.array(CurveParameters.shape)
        Ts = SizeCurveParams[1:]-2
        TotalCurvePoints = np.multiply(InterpolationDensity,(Ts - 1)) + Ts
        M = SizeCurveParams[0]
        """
        
        M, *Ts = CurveParameters.shape
        TotalCurvePoints = np.multiply(InterpolationDensity, (np.array(Ts) - 1)) + np.array(Ts)

        """
        # Project DataPoints to the curves, returning the ProjectedPoints and the
        # (linear) indices on the curves to which they project, ProjectedIndices
        if (np.asarray(PeriodicDims).size > 0):
            ProjectedPoints,ProjectedIndices = self.project_data_points_to_curve(CurveParameters,DataPoints,InterpolationDensity,'Periodic',PeriodicDims)
        else:
            ProjectedPoints,ProjectedIndices = self.project_data_points_to_curve(CurveParameters,DataPoints,InterpolationDensity)
    
        # Compute a vector of desired changes, pointing from DataPoints to ProjectedPoints
        DataPointsChange = ProjectedPoints - DataPoints

        # For every point on the curves, calculate the mean of the desired changes
        TargetChange = np.zeros((M,np.prod(TotalCurvePoints)))
        for i in np.arange(np.prod(TotalCurvePoints)):
            if (sum(ProjectedIndices == i) > 0):
                TargetChange[:,i] = np.mean(DataPointsChange[:,ProjectedIndices == i],axis=1) # matlab mean on 2 dimension
    
        templist = [M,TotalCurvePoints]
        tl_flat = []
        for i, el in enumerate(templist):
            try:
                iterator = iter(el)
            except TypeError:
                tl_flat.append(el)
            else:
                for j in el:
                    tl_flat.append(j)

        TargetChange = np.reshape(TargetChange, tl_flat, order="F")

        # Find the least squares change in spline parameters for the mean desired changes
        for i in np.arange(np.asarray(Ts).size):
            # is it periodic? if so, modify this so that we're not solving a
            # rank-deficient system
            if (PeriodicityFlag[i] == 1):
                TargetChange = self.tensor_right_matrix_solve(TargetChange,i,SplineMatrix[i],np.arange(1,SplineMatrix[i].shape[1-1] - 2+1))
            else:
                TargetChange = self.tensor_right_matrix_solve(TargetChange,i,SplineMatrix[i])

    
        if (justderivsflag == 1):
            """
            # NOTE: I'm not sure that zero-ing the search directions that we want
            # to remain unchanged is equivalent to solving for a least squares
            # TargetChange such that they are zero... will have to look into it.
            """
            indexstring = 'TargetChange[:'
            for i in np.arange(np.asarray(Ts).size):
                indexstring = '%s,1:-1' % indexstring
                indexstring = '%s]=0' % indexstring
                """
                ends up being sth like indexstring=TargetChange(:,2:end-1,2:end-1,2:end-1)=0
                """
            exec(indexstring)

        return TargetChange

    def spline_matrix(self, NumberOfKnotPoints,DimensionOfSpace,InterpolationDensity):
        """
        function S = splineMatrix(NumberOfKnotPoints,DimensionOfSpace,InterpolationDensity)
        Author: Burak Erem and adaptations by SpikaTech
        Description:
        Given a matrix X of [DerivAtFirstKnot,KnotPoints,DerivAtLastKnot],
        this function produces a matrix S that multiplies a vectorized X:
        y = S*vec(X)
        Y=reshape(y,DimensionOfSpace,T) is a matrix of points on the interpolated
        spline curve where
        T=InterpolationDensity*(NumberOfKnotPoints-1)+NumberOfKnotPoints
        is the number of interpolated points (i.e. with InterpolationDensity
        number of points between each knot point)
        """
        N = NumberOfKnotPoints + 2
        S = sp.lil_matrix((InterpolationDensity*(NumberOfKnotPoints-1)+NumberOfKnotPoints,N), dtype=np.float32)

        """
        conditions in bc_type are first derivative at first and last point
        """
        temp = np.zeros(NumberOfKnotPoints)
        ss = interpolate.CubicSpline(np.arange(1,NumberOfKnotPoints+1),temp,bc_type=((1,1),(1,0)))
        S[:,0] = ss(np.linspace(1,NumberOfKnotPoints,InterpolationDensity*(NumberOfKnotPoints-1)+NumberOfKnotPoints))
        ss = interpolate.CubicSpline(np.arange(1,NumberOfKnotPoints+1),temp,bc_type=((1,0),(1,1)))
        S[:,N-1] = ss(np.linspace(1,NumberOfKnotPoints,InterpolationDensity*(NumberOfKnotPoints-1)+NumberOfKnotPoints))
        for i in range(1,NumberOfKnotPoints+1):
            temp = np.zeros(NumberOfKnotPoints)
            temp[i-1] = 1
            ss = interpolate.CubicSpline(np.arange(1,NumberOfKnotPoints+1),temp,bc_type=((1,0),(1,0)))
            S[:,i] = ss(np.linspace(1,NumberOfKnotPoints,InterpolationDensity*(NumberOfKnotPoints-1)+NumberOfKnotPoints))

        S = sp.kron(S, sp.eye(DimensionOfSpace));

        return S    

    def line_search(self, phi, dphidx, x, p, alphamax):
        """
        line search algorithm from Numerical Optimization alg 3.2
        by Nocedal & Wright
        auth: Burak Erem, SciRun
        output:
        alphafinal chosen alpha
        input:
        phi, dphidx: function and gradient
        p: search direction
        alphamax: upperbound
        """
        phialpha = lambda a: phi(x+a*p)
        dphidalpha = lambda a: dphidx(x+a*p).T @ p
        #phialpha = lambda a: phi(x.ravel()[:,None]+a*p)
        #dphidalpha = lambda a: dphidx(x.ravel()[:,None]+a*p).T @ p
    
        # "Exact" line search    
        eps = np.finfo(float).eps
        alphas = np.linspace(eps, alphamax, 10)

        costs = np.zeros(alphas.shape)    
        for i in range(0,alphas.size):
            costs[i] = phialpha(alphas[i])

        ind = np.argmin(costs, axis=0)
        alphafinal = alphas[ind]

        return alphafinal


    def steepest_descent(self, phi, dphidx, xinit, mingradnorm, alphamax = 10):
        """
        SciRun algorithm by Burak Erem
        Input:
        phi = function, dphidx = gradient of function, xinit = initialisation of step sequence
        mingradnorm = smallest 2-norm of dphidx before stopping
        --
        output:
        x = matrix whose colums are gradients at x
        dfdx = matrix whose columns are the gradients at x
        alpha = array of the step size parameters used
        """

        #set up
        k = 0
        x = []
        x.append(xinit)
        f = phi(x[k])
        dfdx = []
        dfdx.append(dphidx(x[k]))
        p = []
        p.append(-dfdx[k])

        alpha = []
        costs = []
        costs.append(phi(x[k]))
        cost_reduction = np.inf

        while (cost_reduction > mingradnorm):
            # Line Search (satisfies strong Wolfe conditions)
            alpha.append(self.line_search(phi, dphidx, x[k], p[k], alphamax))
    
            # Set next sequence step x_(k+1)
            x.append(x[k] + alpha[k] * p[k])

            # Evaluate gradient(f_(k+1))
            dfdx.append(dphidx(x[k + 1]))

            # Set search direction p_(k+1)
            p.append(-dfdx[k + 1])
            costs.append(phi(x[k + 1]))
            cost_reduction = costs[k] - costs[k + 1]

            # Report progress on the current iteration
            print('k={0}    objfun={1}      costreduction={2}'.format(k, costs[k + 1], cost_reduction))

            # Increment k
            k = k + 1

        return x, dfdx, alpha

    def pairwise_distance(self, input, maxcolsize = []):
        """
        This function calculates pairwise distances (2norm squared) between points
        in X and Y or, if called with one argument, between points in X only.
        D = PairwiseDistance(A, [B])
        Input:
        A - matrix with points at columns
        optional B - matrix with points at columns
        Output:
        D = matrix D_ij distance 2norm squared of diff between i point in A and j in B if provided
         or btween i and j in A if B is not provided
        """

        nargin = len(input)
        if not maxcolsize:
            if nargin==1:
                # Calculates the pairwise distances squared between columns of a matrix X
                # dist(xi,xj) = (xi - xj)^2 = ||xi||^2 + ||xj||^2 - 2*xi'*xj
                # //2norm squared
                D = self.distSingle(input[0])

            if nargin==2:
                # Calculates the pairwise distances between points in matrix X and Y
                # dist(xi,yj) = (xi - yj)^2 = ||xi||^2 + ||yj||^2 - 2*xi'*yj
                # 2norm squared
                D = self.distDouble(input[0],input[1])
        else:
            #Block column wise computations
            N1=input[0].shape[1]
            N2=input[1].shape[1]

            blocks1=np.arange(0,N1,maxcolsize)
            if np.max(blocks1) != N1:
                blocks1 = np.append(blocks1, N1)

            blockcount1 = blocks1.size
            blocks2=np.arange(0,N2,maxcolsize)
            if np.max(blocks2) != N2:
                blocks2 = np.append(blocks2, N2)
            blockcount2 = blocks2.size
            D = np.zeros((N1,N2))

            for i in range(blockcount1-1):
                blockrange1=np.arange(blocks1[i],blocks1[i+1])
                for j in range(0,blockcount2-1):
                    blockrange2=np.arange(blocks2[j],blocks2[j+1])
                
                    D[np.ix_(blockrange1,blockrange2)] = self.distDouble(input[0][:,blockrange1],input[1][:,blockrange2])

        return D

    def distSingle(self, singleMatrix):
        K = singleMatrix.T@singleMatrix;
        vD = np.diag(K);
        N = vD.size;
        A = np.tile(vD.reshape(-1,1),(1,N))
        D = -2*K + A + A.T
        return D

    def distDouble(self, firstMatrix, secondMatrix):
        Dy = np.tile(np.sum(secondMatrix**2,axis=0),(firstMatrix.shape[1],1))
        Dx = np.tile(np.sum(firstMatrix**2,axis=0).reshape(-1,1),(1,secondMatrix.shape[1]))
        Dxy = firstMatrix.T@secondMatrix
        D = Dy + Dx - 2*Dxy
        return D


    def project_data_points_to_curve(
            self,
            CurveParameters = None,
            DataPoints = None,
            InterpolationDensity = None,
            mode = None,
            PeriodicDims = None): 
        # Author: Burak Erem in matlab scirun
    
        MAXELEMSPAIRWISEDISTS = 5000000.0
        SizeCurveParameters = CurveParameters.shape
        PeriodicityFlag = np.zeros((np.asarray(SizeCurveParameters).size - 1,1))
        if (mode == 'Periodic'):
            # Unless this is followed by a numeric array of dimension
            # indices that should be periodic, assume they are all
            # periodic
            if (PeriodicDims): #if array of dims provided
                #if (not ischar(varargin[i + 1]) ):
                PeriodicityFlag = np.append(PeriodicityFlag,np.ones(np.asarray(PeriodicDims).size))
                if (np.asarray(PeriodicityFlag).size > np.asarray(SizeCurveParameters).size - 1):
                    print('WARNING: Dimensions specified as being periodic exceed input dimensions.\n' % ())
                else:
                    PeriodicityFlag = np.ones((PeriodicityFlag.shape,PeriodicityFlag.shape))
    
        # 1. Interpolate the spline
        if (np.sum(PeriodicityFlag) > 0):
            ISet = self.interpolate_curve(CurveParameters,InterpolationDensity,'Periodic',np.array(np.where(PeriodicityFlag)))
        else:
            ISet = self.interpolate_curve(CurveParameters,InterpolationDensity)
    
        # 2. Reshape the tensor into a matrix with points in the columns
        SizeISet = ISet.shape
        ISet = np.reshape(ISet,(SizeISet[0],np.prod(SizeISet[1:])))

        # 3. Calculate distances squared between ISet (row indices) and data (column indices)
        if (ISet.shape[1] * DataPoints.shape[1] > MAXELEMSPAIRWISEDISTS):
            P2P = self.pairwise_distance([ISet,DataPoints],int(np.floor(np.sqrt(MAXELEMSPAIRWISEDISTS))))
        else:
            P2P = self.pairwise_distance([ISet,DataPoints])
    
            # Find the index of the minimum of each column
            # we need to find where P2P min in each col is
            # (in matlab achieved with [~,projindex]=min(p2p) gives the indices where min occurs
        ProjectedIndices = np.argmin(P2P,axis=0)
        
        # The projected points are the points that correspond to these minima
        ProjectedPoints = ISet[:,ProjectedIndices]
        return ProjectedPoints,ProjectedIndices


    def tensor_right_matrix_solve(self, intensor , rightIndex , rightMatrix , varargin = []):
        SUBSETFLAG = 0
        blockmat = intensor.tolist()
        matsize = len(blockmat[0])
        cellsize = intensor.shape
        blockvecmat = np.squeeze(blockmat)

        vecmat = blockvecmat

        if (np.asarray(varargin).size > 0):
            SUBSETFLAG = 1
            subsetrows = varargin[0]
            M,N = rightMatrix.shape
            rightMatrix = rightMatrix.tocsr()[subsetrows,:]#.todense()

            multvecmat = np.zeros((vecmat.shape[1-1],M))
            #multvecmat[:,subsetrows] = vecmat / rightMatrix
            tempsol = np.linalg.lstsq(rightMatrix.todense().T, vecmat.T,rcond=None)
            multvecmat[:,subsetrows] = tempsol[0].T

        else:
            #beware matlab A/B means solve x=A/B, A=xB
            #multvecmat = vecmat / rightMatrix.todense()
            multvecmat = np.linalg.lstsq(rightMatrix.todense().T, vecmat.T,rcond=None)
            multvecmat = multvecmat[0].T
        
        REblockvecmat = multvecmat
    
        n = np.arange(0,np.asarray(cellsize).size)
        m = np.array([0,rightIndex])
        multcellsize = n[n!=m].size
        """
        beware with reshape
        """
        outtensor = REblockvecmat
        """
        here swap columns
        """

        multinds = np.arange(len(cellsize)) #(1,np.asarray(cellsize).size+1)
        multinds[1:rightIndex] = np.arange(2,rightIndex + 1+1)
        #multinds[np.arange[2,rightIndex+1]] = np.arange(3,rightIndex + 1+1)
        multinds[rightIndex + 1] = 1
        #outtensor = permute(outtensor,multinds)
        outtensor = np.transpose(outtensor,multinds)
        return outtensor

    def tik_inv(self, u, s, vt, b, l):
        s = s.reshape(-1,1)
        fil_fac = (s/(s*s+l*l)).flatten()
        inv = np.dot(np.dot(vt.T,np.dot(np.diag(fil_fac),u.T)),b)
        return inv

    def l_curve(self, lamb,u,sigma,vt,rhs,amat,alpha):
        # lambda cannot be 0
        avec = self.tik_inv(u,sigma,vt,rhs,lamb)
        rho = np.linalg.norm(np.dot(amat,avec)-rhs)
        eta = np.linalg.norm(avec)
        rho2 = rho**2
        eta2 = eta**2
        eta_dash = -4.0 / lamb * np.sum(lamb**2*sigma**2/(sigma**2+lamb**2)**3*alpha**2)
        kappa = 2.0 * eta2 * rho2 * (lamb**2 *eta_dash * rho2
                                     + 2.0 * eta2 * rho2 * lamb+lamb**4*eta2*eta_dash) \
                                     /(eta_dash*(lamb**2*eta2**2+rho2**2)**(1.5))
        return kappa

    def burak(self, lambdas,u,sigma,vt,rhs, transfer_matrix):
        #avec = tik_inv(u,sigma,vt,rhs,lambdas)
        rho = np.zeros(len(lambdas))
        eta = np.zeros(len(lambdas))
        print("Shape u, sigma, vt",u.shape,sigma.shape,vt.shape)
        print("rhs shape",rhs.shape)
        print("transfer matrix shape =",transfer_matrix.shape)
        for l in np.arange(len(lambdas)):
            avec = self.tik_inv(u,sigma,vt,rhs,lambdas[l])
            rho[l] = np.linalg.norm(np.dot(transfer_matrix,avec)-rhs)
            eta[l] = np.linalg.norm(avec)

        rholog=np.log10(rho)
        etalog=np.log10(eta)
        Trho=2*len(lambdas)
        #etalog=spline(rholog,etalog,linspace(min(rholog),max(rholog),Trho));
        tck = interpolate.splrep(rholog, etalog)
        etalog = interpolate.splev(np.linspace(np.min(rholog),np.max(rholog),Trho),tck)
    
        etalog=self.low_pass_moving_average(np.atleast_2d(etalog),10)
        tck = interpolate.splrep(rholog,lambdas)
        lambdaspline = interpolate.splev(np.linspace(np.min(rholog),np.max(rholog),Trho),tck)
    

        detalog=np.diff(etalog,2);
        signchanges=np.diff(np.sign(detalog)).flatten()
    
        lastchangeind=np.argwhere(signchanges<0)[-1] # find(signchanges<0,1,'last');

        lambdaret = lambdaspline[lastchangeind];
        return lambdaret

    def low_pass_moving_average(self, M, win):
        Low = M
        if win < 2:
            return Low
        trans = 0
        dims_M = np.shape(M)
        nsig = dims_M[0]
        nt = dims_M[1]
        if nt == 1:
            trasp = 1
            M = np.transpose(M)
            dims_M = np.shape(M)
            nsig = dims_M[0]
            nt = dims_M[1]
            
        winb = (np.floor(win/2)).astype(int)
        wine = win-winb
        LEAD = np.outer(M[:,0],np.ones(winb))
        TRAIL = np.outer(M[:,0],np.ones(wine))
        dims_LEAD = np.shape(LEAD)
        dims_TRAIL = np.shape(TRAIL)
        rows = dims_M[0]
        columns = dims_LEAD[1]+dims_M[1]+dims_TRAIL[1]
        storage = np.zeros((rows,columns))
        storage[:,0:dims_LEAD[1]] = LEAD
        storage[:,dims_LEAD[1]:dims_LEAD[1]+dims_M[1]] = M
        storage[:,dims_LEAD[1]+dims_M[1]:dims_LEAD[1]+dims_M[1]*dims_TRAIL[1]] = TRAIL
        M = storage
        X = np.cumsum(np.concatenate((np.zeros((nsig,1)),M),axis=1),axis=1)
        LOW = X[:,win:win+nt]-X[:,0:nt]
        LOW = LOW/win
        if trans == 1:
            LOW = np.transpose(LOW)

        return LOW

        
    def regularise_transfer_matrix(self, InitialTransferMatrix,TorsoPotentials):
        """
        Inputs requires are the transfer matrix, aka forward matrix,
        and potentials at chest
        Note that torso input dimensions include time instances
        """
        # Load transfer matrix and svd
        Tmat = InitialTransferMatrix

        # can be rewritten as Tmat = u D v.T (svd)
        u, D, vt = np.linalg.svd(Tmat, full_matrices = False)

        """
        inv(Tmat) = inv(v.T) inv(D) inv(u)
        and with tikhonov we divide all values in diagonal of D with lambdas
        """
        lambdas = 10**np.linspace(-2,2,100)

        """
        #find optimal lambda with l-curve method
        alpha = np.dot(np.transpose(u),TorsoPotentials)    
        lc = [l_curve(l,u,D,vt,TorsoPotentials,Tmat,alpha) for l in lambdas]
        optimal_lambda = lambdas[np.argmax(np.array(lc))]
        Inv_Matrix = tik_inv(u,D,vt,TorsoPotentials,optimal_lambda)
        """    

        # reg= Inv_Matrix @ TorsoPotentials
        
        # with burak
        optimal_lambda = self.burak(lambdas, u, D, vt, TorsoPotentials, Tmat)
        Inv_Matrix = self.tik_inv(u, D, vt, TorsoPotentials, optimal_lambda)
    
        return Inv_Matrix, optimal_lambda, lambdas

    def interpolate_curve(
            self,
            CurveParameters,
            InterpolationDensity,
            mode = None,
            PeriodicDims = None):
        dims_CurveParameters = np.shape(CurveParameters)
        prod_dims_CurveParameters = np.prod(dims_CurveParameters,axis=0)
        numel_dims_dims = 0
        for i in dims_CurveParameters:
            numel_dims_dims+=1
        PeriodicityFlag = np.zeros((numel_dims_dims-1,1))
        numel_varg = 0
        if (mode):
            numel_varg +=1
        if (PeriodicDims):
            numel_varg +=1

        if numel_varg > 0:
            p = 'Periodic'
            for ii in range(0,numel_varg,1):
                if mode == p:
                    # unless this is followed by a numeric array of dimension
                    # indices that should be periodic,
                    # assume they are all periodic
                    if numel_varg > ii:
                        if PeriodicDims:#type(varargin[ii+1]) != type(p):
                            PeriodicityFlag[varargin[ii+1]] = 1
                            dims_PeriodicityFlag = np.shape(PeriodicityFlag)
                            numel_PeriodicityFlag = dims_PeriodicityFlag[0]*dims_PeriodicityFlag[1]
                            if numel_PeriodicityFlag > (numel_dims - 1):
                                print('WARNING: Dimensions specified as being periodic \
                                exceed input dimensions. \n')
                    else:
                        PeriodicityFlag = np.ones((dims_PeriodicityFlag[0],dims_PeriodicityFlag[1]))

        # Form 1-D spline interpolation matrices of appropriate sizes
        TensorEdgeDimensions = dims_CurveParameters[1:]
        dims_TensorEdgeDimensions = np.shape(TensorEdgeDimensions)

        #lets calculate numel_Tensor through len of len
        numel_Tensor = sum(1 for i in dims_TensorEdgeDimensions)
    
        SplineMatrix = [None]*numel_Tensor
        for jj in range(numel_Tensor):
            if PeriodicityFlag[jj] == 0: # if not periodic
                SplineMatrix[jj] = np.transpose(self.spline_matrix(TensorEdgeDimensions[jj]-2,1,InterpolationDensity))
            else: # if periodic
                SplineMatrix[jj] = np.transpose(self.spline_matrix(TensorEdgeDimensions[jj]-2,1,InterpolationDensity,'Periodic'))
    
        # intialize interpolated curves as curve parameters
        InterpolatedCurve = CurveParameters
    
        # Interpolate spline curves using tensor-matrix "right" multiplication, for
        # all the "right-hand" sides (i.e. tensor indices, not including the first one)
        for kk in range(0,(numel_dims_dims-1),1):
            InterpolatedCurve = self.tensor_right_matrix_multiply(InterpolatedCurve, kk, SplineMatrix[kk])
    
        return InterpolatedCurve

    def tensor_right_matrix_multiply(self, intensor = None, rightIndex = None, rightMatrix = None): 
        blockmat = intensor.tolist()
        matsize = len(blockmat[0])
        cellsize = intensor.shape 
        blockvecmat = np.squeeze(blockmat)
        vecmat = blockvecmat
        multvecmat = vecmat @ rightMatrix

        REblockvecmat = multvecmat
        n = np.arange(0,np.asarray(cellsize).size)
        m = np.array([0,rightIndex])
        multcellsize = n[n!=m].size

        outtensor = REblockvecmat

        multinds = np.arange(len(cellsize))
        multinds[1:rightIndex] = np.arange(2,rightIndex + 1+1)
        multinds[rightIndex + 1] = 1

        outtensor = np.transpose(outtensor,multinds)
        return outtensor

    
    def initialise_run_spline_inverse(self):
        closedtpotentials = self.parameters.torso_potentials
        #heartpotentials = data["heart_measures"]
        number_of_knots = self.parameters.number_of_knots
        InterpolationDensity = self.parameters.interpolation_density
        minderivcostreduction = self.parameters.minimum_derivatives_cost_reduction
        minoverallcostreduction = self.parameters.minimum_overall_cost_reduction
        ProjectionInterpolationDensity = self.parameters.projection_interpolation_density
        transfer_matrix = self.transfer_matrix

        print("describing curve parameters", closedtpotentials)
        CurveParams = self.initialize_curve_params_from_time_series(closedtpotentials, number_of_knots)
        CurveParams,_ = self.minimize_distance_to_curve(
            CurveParams, closedtpotentials, InterpolationDensity, minderivcostreduction,'JustDerivatives')
        CurveParams,_ = self.minimize_distance_to_curve(
            CurveParams, closedtpotentials, InterpolationDensity, minoverallcostreduction)
        torsoPotentials_Manifold,  torsoPotentials_timeWarp  = self.project_data_points_to_curve(
            CurveParams, closedtpotentials, ProjectionInterpolationDensity)
        # we need here A = transfer matrix from BEM
        CurveParams_heart,_,_ = self.regularise_transfer_matrix(transfer_matrix, CurveParams)

        heartmat = {"curveparams_heart": CurveParams_heart}
        scipy.io.savemat("hmat.mat",heartmat)

        heartPotentialsManifold = self.interpolate_curve(CurveParams_heart, ProjectionInterpolationDensity)
        heartPotentialsReconstructed = heartPotentialsManifold[:, torsoPotentials_timeWarp]

        return heartPotentialsReconstructed
        
    def spline_inverse(self):
        return self.initialise_run_spline_inverse()
