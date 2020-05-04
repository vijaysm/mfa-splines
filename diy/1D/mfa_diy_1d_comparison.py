# coding: utf-8
import sys, getopt, math
import numpy as np
import scipy

import diy
from matplotlib import pyplot as plt
import mpl_toolkits.mplot3d.axes3d as axes3d
from matplotlib import cm
import pandas as pd
from mpi4py import MPI

from scipy import linalg, matrix
from scipy.interpolate import LSQUnivariateSpline, UnivariateSpline, interp1d, Rbf
from scipy.optimize import minimize

from makruth_solver import basis, getControlPoints, decode, Error, L2LinfErrors
from makruth_solver import knotInsert, knotRefine, deCasteljau, pieceBezierDer22

plt.style.use(['seaborn-whitegrid'])
params = {"ytick.color" : "b",
          "xtick.color" : "b",
          "axes.labelcolor" : "b",
          "axes.edgecolor" : "b"}
plt.rcParams.update(params)

# --- set problem input parameters here ---
nSubDomains    = 4
degree         = 3
nControlPoints = 46 #(3*degree + 1) #minimum number of control points
useDecodedResidual = True
overlapData    = 50
overlapCP      = 0
problem        = 0
scale          = 1
showplot       = True
nASMIterations = 5
# Look at ovRBFPower param below if using useDecodedResidual = True
#
# ------------------------------------------
# Solver parameters
solverscheme   = 'SLSQP' # [SLSQP, COBYLA]
useAdditiveSchwartz = True
useDerivativeConstraints = 1
enforceBounds = False
disableAdaptivity = True
# 
#                            0        1         2        3     4       5          6           7
subdomainSolverSchemes = ['LCLSQ', 'SLSQP', 'L-BFGS-B', 'CG', 'lm', 'krylov', 'broyden2', 'anderson']
subdomainSolver = subdomainSolverSchemes[2]

maxAbsErr       = 1e-5
maxRelErr       = 1e-8

solverMaxIter   = 20
globalTolerance = 1e-12

# AdaptiveStrategy = 'extend'
AdaptiveStrategy = 'reset'

# Initialize DIY
w = diy.mpi.MPIComm()           # world
commW = MPI.COMM_WORLD
nprocs = commW.size
rank = commW.rank

if rank == 0: print('Argument List:', str(sys.argv))

##########################
# Parse command line overrides if any
##
argv=sys.argv[1:]
def usage():
  print(sys.argv[0], '-p <problem> -n <nsubdomains> -d <degree> -c <controlpoints> -o <overlapData> -a <nASMIterations>')
  sys.exit(2)
try:
  opts, args = getopt.getopt(argv,"hp:n:d:c:o:a:",["problem=","nsubdomains=","degree=","controlpoints=","overlap=","nasm="])
except getopt.GetoptError:
  usage()

for opt, arg in opts:
  if opt == '-h':
      usage()
  elif opt in ("-n", "--nsubdomains"):
      nSubDomains = int(arg)
  elif opt in ("-d", "--degree"):
      degree = int(arg)
  elif opt in ("-c", "--controlpoints"):
      nControlPoints = int(arg)
  elif opt in ("-o", "--overlap"):
      overlapData = int(arg)
  elif opt in ("-p", "--problem"):
      problem = int(arg)
  elif opt in ("-a", "--nasm"):
      nASMIterations = int(arg)


# Ax = b 
# A = A1 + A2
# (A1x1 + A2 x2) = b1 + b2
# (A1 (A2) x1 + A2 (A1) x2) = b1 + b2

##----------------------
## Problematic settings
# problem        = 0
# nSubDomains    = 4
# degree         = 3
# nControlPoints = (3*degree + 1) #minimum number of control points
# subdomainSolver = subdomainSolverSchemes[0]
##----------------------

if problem == 0:
    Dmin           = -4.
    Dmax           = 4.
    nPoints        = 1025
    x = np.linspace(Dmin, Dmax, nPoints)
    scale          = 100
    # y = scale * (np.sinc(x+1))
    # y = scale * (np.sinc(x+1) + np.sinc(2*x) + np.sinc(x-1))
    y = scale * (np.sinc(x) + np.sinc(2*x-1) + np.sinc(3*x+1.5))
    # y = scale * np.sin(math.pi * x/4)
elif problem == 1:
    y = np.fromfile("data/s3d.raw", dtype=np.float64) #
    print('Real data shape: ', y.shape)
    nPoints = y.shape[0]
    Dmin           = 0
    Dmax           = 1.
    x = np.linspace(Dmin, Dmax, nPoints)
elif problem == 2:
    Y = np.fromfile("data/nek5000.raw", dtype=np.float64) #
    Y = Y.reshape(200,200)
    y = Y[100,:] # Y[:,150] # Y[110,:]
    Dmin           = 0
    Dmax           = 1.
    nPoints = y.shape[0]
    x = np.linspace(Dmin, Dmax, nPoints)
else:
    '''
    Y = np.fromfile("data/FLDSC_1_1800_3600.dat", dtype=np.float32).reshape(3600, 1800) #

    def plot3D(fig, Z, x=None, y=None):
        if x is None:
            x = np.arange(Z.shape[0])
        if y is None:
            y = np.arange(Z.shape[1])
        X, Y = np.meshgrid(x, y)
        print("Plot shapes: [x, y, z] = ", x.shape, y.shape, Z.shape, X.shape, Y.shape)
        ax = fig.gca(projection='3d')
        surf = ax.plot_surface(X, Y, Z.T, cmap=cm.coolwarm,
                        linewidth=0, antialiased=True)
        fig.colorbar(surf)

        plt.show()

    fig = plt.figure()
    plot3D(fig, Y)

    print (Y.shape)
    sys.exit(2)
    '''

    DJI = pd.read_csv("data/DJI.csv")
    y = DJI['Close']
    Dmin           = 0
    Dmax           = 100.
    nPoints = y.shape[0]
    x = np.linspace(Dmin, Dmax, nPoints)

    
# if nPoints % nSubDomains > 0:
#     print ( "[ERROR]: The total number of points do not divide equally with subdomains" )
#     sys.exit(1)

ymin = y.min()
ymax = y.max()
yRange = ymax-ymin
yRange = 1
if showplot:
    mpl_fig = plt.figure()
    plt.plot(x, y, 'r-', ms=2)
# plt.plot(x, y[:,150], 'g-', ms=2)
# plt.plot(x, y[110,:], 'b-', ms=2)


#------------------------------------

### Print parameter details ###
if rank == 0:
    print('\n==================')
    print('Parameter details')
    print('==================\n')
    print('problem = ', problem, '[0 = sinc, 1 = S3D, 2 = Nek5000]')
    print('Total number of input points: ', nPoints)
    print('nSubDomains = ', nSubDomains)
    print('degree = ', degree)
    print('nControlPoints = ', nControlPoints)
    print('nASMIterations = ', nASMIterations)
    print('overlapData = ', overlapData)
    print('overlapCP = ', overlapCP)
    print('useAdditiveSchwartz = ', useAdditiveSchwartz)
    print('useDerivativeConstraints = ', useDerivativeConstraints)
    print('useDecodedResidual = ', useDecodedResidual)
    print('enforceBounds = ', enforceBounds)
    print('maxAbsErr = ', maxAbsErr)
    print('maxRelErr = ', maxRelErr)
    print('solverMaxIter = ', solverMaxIter)
    print('globalTolerance = ', globalTolerance)
    print('AdaptiveStrategy = ', AdaptiveStrategy)
    print('solverscheme = ', solverscheme)
    print('subdomainSolver = ', subdomainSolver)
    print('\n=================\n')

#------------------------------------

EPS    = 1e-14
basis  = lambda u,p,T: ((T[:-1]<=u) * (u<=T[1:])).astype(np.float) if p==0 else ((u - T[:-p]) /(T[p:]  -T[:-p]+EPS))[:-1] * basis(u,p-1,T)[:-1] + ((T[p+1:] - u)/(T[p+1:]-T[1:-p]+EPS))     * basis(u,p-1,T)[1:]

#--------------------------------------
# Let do the recursive iterations
# The variable `additiveSchwartz` controls whether we use
# Additive vs Multiplicative Schwartz scheme for DD resolution
####

class InputControlBlock:

    def __init__(self, nControlPoints, coreb, xb, xl, yl):
        self.nControlPoints = nControlPoints
        self.nControlPointSpans = nControlPoints - 1
        self.nInternalKnotSpans = self.nControlPointSpans - degree + 1
        self.nPointsPerSubD = xl.shape[0] #int(nPoints / nSubDomains) + overlapData
        self.xbounds = xb
        # self.corebounds = coreb
        self.corebounds = [coreb.min[0]-xb.min[0], -1+coreb.max[0]-xb.max[0]]
        self.xl = xl
        self.yl = yl
        self.pAdaptive = []
        self.WAdaptive = []
        self.knotsAdaptive = []
        self.leftconstraint = np.zeros(overlapCP+1)
        self.leftconstraintKnots = np.zeros(overlapCP+1)
        self.rightconstraint = np.zeros(overlapCP+1)
        self.rightconstraintKnots = np.zeros(overlapCP+1)
        # Allocate for the constraints
        self.interface_constraints_obj = dict()
        self.interface_constraints_obj['P']=[[],[],[]]
        self.interface_constraints_obj['W']=[[],[],[]]
        self.interface_constraints_obj['T']=[[],[],[]]
        self.decodedAdaptive = np.zeros(xl.shape)
        self.decodedAdaptiveOld = np.zeros(xl.shape)

        ## Convergence related metrics and checks
        self.outerIteration = 0
        self.outerIterationConverged = False
        self.decodederrors = np.zeros(2) # L2 and Linf
        self.errorMetricsL2 = np.zeros((nASMIterations), dtype='float64')
        self.errorMetricsLinf = np.zeros((nASMIterations), dtype='float64')

    def show(self, cp):
        # print("Rank: %d, Subdomain %d: Bounds = [%d, %d], Core = [%d, %d]" % (w.rank, cp.gid(), self.xbounds.min[0], self.xbounds.max[0]+1, self.corebounds.min[0], self.corebounds.max[0]))
        print("Rank: %d, Subdomain %d: Bounds = [%d, %d], Core = [%d, %d]" % (w.rank, cp.gid(), self.xbounds.min[0], self.xbounds.max[0]+1, self.corebounds[0], self.corebounds[1]))
        #cp.enqueue(diy.BlockID(1, 0), "abc")

    def plot(self, cp):
#         print(w.rank, cp.gid(), self.core)
        self.decodedAdaptive = decode(self.pAdaptive, self.WAdaptive, self.xl, 
                          self.knotsAdaptive,# * (Dmax - Dmin) + Dmin,
                          degree)
        coeffs_x = getControlPoints(self.knotsAdaptive, degree) #* (Dmax - Dmin) + Dmin
#         print ('Coeffs-x original: ', coeffs_x)
        plt.plot(self.xl, self.decodedAdaptive, linestyle='--', lw=2, color=['r','g','b','y','c'][cp.gid()%5], label="Decoded-%d"%(cp.gid()+1))
        plt.plot(coeffs_x, self.pAdaptive, marker='o', linestyle='', color=['r','g','b','y','c'][cp.gid()%5], label="Control-%d"%(cp.gid()+1))

    def plot_error(self, cp):
#         print(w.rank, cp.gid(), self.core)
        error = self.decodedAdaptive - self.yl
        plt.plot(self.xl, error, linestyle='--', color=['r','g','b','y','c'][cp.gid()%5], lw=2, label="Subdomain(%d) Error"%(cp.gid()+1))


    def plot_with_cp(self, cp, cploc, ctrlpts, lgndtitle, indx):
#         print(w.rank, cp.gid(), self.core)
        pMK = decode(ctrlpts, self.WAdaptive, self.xl, 
                          self.knotsAdaptive * (Dmax - Dmin) + Dmin, degree)
        plt.plot(self.xl, pMK, linestyle='--', color=['g','b','y','c'][indx%5], lw=3, label=lgndtitle)

            
    def plot_with_cp_and_knots(self, cp, cploc, knots, ctrlpts, weights, lgndtitle, indx):
#         print(w.rank, cp.gid(), self.core)
        print('Plot: shapes = ', ctrlpts.shape[0], cploc.shape[0], knots.shape[0], degree)
        pMK = decode(ctrlpts, weights, self.xl, knots, degree)
        plt.plot(self.xl, pMK, linestyle='--', color=['g','b','y','c'][indx], lw=3, label=lgndtitle)


    def print_error_metrics(self, cp):
        # print('Size: ', commW.size, ' rank = ', commW.rank, ' Metrics: ', self.errorMetricsL2[:])
        print('Rank:', commW.rank, ' SDom:', cp.gid(), ' L2 Error table: ', self.errorMetricsL2)

    def send(self, cp):
        link = cp.link()
        for i in range(len(link)):
            target = link.target(i)
            o = np.zeros(overlapCP+1)
            if target.gid > cp.gid(): # target is to the right of current subdomain
                if len(self.pAdaptive):
                    if useDecodedResidual:
                        # print('Subdomain: ', cp.gid()+1, self.decodedAdaptive)
#                         o = self.decodedAdaptive[-1:-2-overlapData:-1]
                        # o = self.decodedAdaptive[-1-overlapData:]
                        if overlapData > 0:
                            o = self.decodedAdaptive[self.corebounds[1]-overlapData:self.corebounds[1]+1]
                        else:
                            o = np.array([self.decodedAdaptive[-1]])
                    else:
                        o = np.array([self.pAdaptive.shape[0], self.knotsAdaptive.shape[0], self.pAdaptive[:], self.knotsAdaptive[:]])
                        # o = self.pAdaptive[-1-overlapCP:]
                # print("%d sending to %d: %s" % (cp.gid()+1, target.gid+1, o))
                print("%d sending to %d" % (cp.gid()+1, target.gid+1))
            else: # target is to the left of current subdomain
                if len(self.pAdaptive):
                    if useDecodedResidual:
                        # print('Subdomain: ', cp.gid()+1, self.decodedAdaptive)
                        o = self.decodedAdaptive[0:overlapData+1]
                        # if self.corebounds[0] == 0:
                            # o = self.decodedAdaptive[0:overlapData+1]
                            # o = self.decodedAdaptive[self.corebounds[0]-overlapData:overlapData+1]
                    else:
                        o = np.array([self.pAdaptive.shape[0], self.knotsAdaptive.shape[0], self.pAdaptive[:], self.knotsAdaptive[:]])
                        # o = self.pAdaptive[0:overlapCP+1]
                # print("%d sending to %d: %s" % (cp.gid()+1, target.gid+1, o))
                print("%d sending to %d" % (cp.gid()+1, target.gid+1))
            cp.enqueue(target, o) 

    def recv(self, cp):
        link = cp.link()
        for i in range(len(link)):
            tgid = link.target(i).gid
            o = np.array(cp.dequeue(tgid))
            if tgid > cp.gid(): # target is to the right of current subdomain; receive constraint for right end point
                if useDecodedResidual:
                    self.rightconstraint = np.array(o[:]) if len(o) else np.zeros(overlapData+1)
                else:
                    nc = int(o[0]) if len(o) > 1 else 0
                    nk = int(o[1]) if len(o) > 1 else 0
                    self.rightconstraint = np.array(o[2]) if nc else np.zeros(1)
                    self.rightconstraintKnots = np.array(o[3]) if nk else np.zeros(1)
            else:
                if useDecodedResidual:
                    self.leftconstraint = np.array(o[:]) if len(o) else np.zeros(overlapData+1)
                else:
                    nc = int(o[0]) if len(o) > 1 else 0
                    nk = int(o[1]) if len(o) > 1 else 0
                    self.leftconstraint = np.array(o[2]) if nc else np.zeros(1)
                    self.leftconstraintKnots = np.array(o[3]) if nk else np.zeros(1)
            # print("%d received from %d: %s" % (cp.gid()+1, tgid+1, o))
            print("%d received from %d" % (cp.gid()+1, tgid+1))


    def WeightedConstrainedLSQ_PT(self, idom, Nall, Wall, ysloc, U, t, degree, nSubDomains, constraintsAll=None, useDerivatives=0, solver='SLSQP'):

        if constraintsAll is not None:
            constraints = np.array(constraintsAll['P'])
            knotsAll = np.array(constraintsAll['T'])
            weightsAll = np.array(constraintsAll['W'])
            # print ('Constraints for Subdom = ', idom, ' is = ', constraints)
        else:
            print ('Constraints are all null. Solving unconstrained.')

        def ComputeL2Error0(P, N, W, ysl, U, t, degree):
            E = np.sum(Error(P, W, ysl, U, t, degree)**2)/len(P)
            return math.sqrt(E)

        def ComputeL2Error(P, N, W, ysl, U, t, degree):
            RN = (N*W)/(np.sum(N*W, axis=1)[:,np.newaxis])
            E = (RN.dot(P) - ysl)
            return math.sqrt(np.sum(E**2)/len(E))

        #print('shapes idom, Nall, Wall, ysloc: ', idom, Nall.shape, Wall.shape, ysloc.shape)

        # Solve unconstrained::
        lenNRows = Nall.shape[0]
        if idom == 1:
            print('Left-most subdomain')
            indices = range(lenNRows-1-overlapCP, lenNRows)
            print('Indices:', indices, Wall.shape)
            N = np.delete(Nall, np.s_[indices], axis=0)
            M = np.array(Nall[-1:-2-overlapCP:-1, :])
            W = Wall[:lenNRows-1-overlapCP]
            ysl = ysloc[:-1-overlapCP]
    #         T = np.array([constraints[0][-1-overlap]])
            T = np.array([0.5*(constraints[1][-1-overlapCP] + constraints[2][overlapCP])])
        elif idom == nSubDomains:
            print('Right-most subdomain')
            indices = range(0,overlapCP+1,1)
            print('Indices:', indices)
            N = np.delete(Nall, np.s_[indices], axis=0)
            M = np.array(Nall[0:overlapCP+1, :])
            W = Wall[overlapCP+1:]
            ysl = ysloc[overlapCP+1:]
    #         T = np.array([constraints[2][overlap]])
            T = np.array([0.5*(constraints[1][overlapCP] + constraints[0][-1-overlapCP])])
        else:
            print('Middle subdomain')
            indices1 = range(0,overlapCP+1,1)
            indices2 = range(lenNRows-1-overlapCP, lenNRows)
            print('Indices:', indices1, indices2)
            N = np.delete(np.delete(Nall, np.s_[indices2], axis=0), np.s_[indices1], axis=0)
            print('N shapes:', Nall.shape, N.shape)
            M = np.array([Nall[0:overlapCP+1, :].T, Nall[-1:-2-overlapCP:-1, :].T])[:,:,0]
            W = Wall[overlapCP+1:lenNRows-1-overlapCP]
            ysl = ysloc[overlapCP+1:-1-overlapCP]
    #         T = np.array([constraints[0][-1-overlap], constraints[2][overlap]])
            T = 0.5*np.array([(constraints[1][overlapCP] + constraints[0][-1-overlapCP]), (constraints[1][-1-overlapCP] + constraints[2][overlapCP])])

        W = Wall[:]

        # Solve the unconstrained solution directly
        RN = (Nall*Wall)/(np.sum(Nall*Wall, axis=1)[:,np.newaxis])
        LHS = np.matmul(RN.T,RN)
        RHS = np.matmul(RN.T, ysloc)
        UnconstrainedLSQSol = linalg.lstsq(LHS, RHS)[0] # This is P(Uc)

        # LM = inv(M*inv(NT*W*N)*MT) * (M*inv(NT*W*N)*NT*W*S - T)
    #     NTWN = N.T * (W.T * N)
        NW = N * W

    #     print('shapes N, W, N*W, ysl: ', N.shape, W.shape, (N*W).shape, ysl.shape)
        NTWN = np.matmul(N.T, NW)
        LUF, LUP = scipy.linalg.lu_factor(NTWN)
    #     srhs = np.matmul((N*W), ysl)
        srhs = NW.T @ ysl

    #     print('LU shapes - NTWN, srhs, T, M: ', NTWN.shape, srhs.shape, T.shape, M.shape)
        LMConstraintsA = scipy.linalg.lu_solve((LUF,LUP), srhs)
        LMConstraints = np.matmul(M, LMConstraintsA) - T
        
    #     print('shapes LMConstraints, M, LUF', LMConstraints.shape, M.shape, NTWN.shape)
        Alhs = np.matmul(M, scipy.linalg.lu_solve((LUF,LUP), M.T))
    #     print('shapes Alhs', Alhs.shape, Alhs)
        
        # Equation 9.76 - Piccolo and Tiller
        ALU, ALUP = scipy.linalg.lu_factor(Alhs, overwrite_a=False)
        A = scipy.linalg.lu_solve((ALU, ALUP), LMConstraints.T)
    #     print('shapes A', A.shape, A)

        P2 = M.T @ A
        #print('shapes P2', LMConstraintsA.shape, NTWN.shape, P2.shape)
        P = LMConstraintsA - scipy.linalg.lu_solve((LUF,LUP), P2)
        
        #print('shapes P', P.shape, P, P-UnconstrainedLSQSol)

    #     return UnconstrainedLSQSol
        return P
    #     return 0.5*(P+UnconstrainedLSQSol)

    def lsqFit(self, N, W, y, U, t, degree):
        RN = (N*W)/(np.sum(N*W, axis=1)[:,np.newaxis])
        LHS = np.matmul(RN.T,RN)
        RHS = np.matmul(RN.T, y)
        return linalg.lstsq(LHS, RHS)[0]

    def lsqFitWithCons(self, N, W, ysl, U, t, degree, constraints=[], continuity=0):
        def l2(P, W, ysl, U, t, degree):
            return np.sum(Error(P, W, ysl, U, t, degree)**2)

        res = minimize(l2, np.ones_like(W), method='SLSQP', args=(W, ysl, U, t, degree), 
                    constraints=constraints,
                        options={'disp': True})
        return res.x

    def LSQFit_Constrained(self, idom, N, W, ysl, U, t, degree, nSubDomains, constraintsAll=None, useDerivatives=0, solver='SLSQP'):

        RN = (N*W)/(np.sum(N*W, axis=1)[:,np.newaxis])

        constraints = None
        if constraintsAll is not None:
            constraints = np.array(constraintsAll['P'])
            knotsAll = np.array(constraintsAll['T'])
            weightsAll = np.array(constraintsAll['W'])
            # print ('Constraints for Subdom = ', idom, ' is = ', constraints)
        else:
            print ('Constraints are all null. Solving unconstrained.')

        if useDerivatives > 0 and constraints is not None and len(constraints) > 0:

            bzD = pieceBezierDer22(constraints[1], weightsAll[1], U, knotsAll[1], degree)
            # bzDD = pieceBezierDDer22(bzD, W, U, knotsD, degree-1)
            # print ("Subdomain Derivatives (1,2) : ", bzD, bzDD)
            if idom < nSubDomains:
                bzDp = pieceBezierDer22(constraints[2], weightsAll[2], U, knotsAll[2], degree)
                # bzDDp = pieceBezierDDer22(bzDp, W, U, knotsD, degree-1)
                # print ("Left Derivatives (1,2) : ", bzDp, bzDDp )
                # print ('Right derivative error offset: ', ( bzD[-1] - bzDp[0] ) )
            if idom > 1:
                bzDm = pieceBezierDer22(constraints[0], weightsAll[0], U, knotsAll[0], degree)
                # bzDDm = pieceBezierDDer22(bzDm, W, U, knotsD, degree-1)
                # print ("Right Derivatives (1,2) : ", bzDm, bzDDm )
                # print ('Left derivative error offset: ', ( bzD[0] - bzDm[-1] ) )

        def ComputeL2Error0(P, N, W, ysl, U, t, degree):
            E = np.sum(Error(P, W, ysl, U, t, degree)**2)/len(P)
            return math.sqrt(E)

        def ComputeL2Error(P, N, W, ysl, U, t, degree):  # checkpoint1
    #         E = P - constraints[1]

    #         RN = (N*W)/(np.sum(N*W, axis=1)[:,np.newaxis])
            E = (RN.dot(P) - ysl)/yRange
    #         return math.sqrt(np.sum(E**2)/len(E))
        
    #         LHS = np.matmul(RN.T, RN)
    #         RHS = np.matmul(RN.T, ysl)
    #         E = LHS @ P - RHS
    #         return linalg.lstsq(LHS, RHS)[0]
            errorres = math.sqrt(np.sum(E**2)/len(E))
    #         print('Sol: ', P, ', Error residual: ', E, ', Norm = ', errorres)
            return errorres


    #     def print_iterate(P, state):

    #         print('Iteration %d: max error = %f' % (self.globalIterationNum, state.maxcv))
    #         self.globalIterationNum += 1
    #         return False

        cons = []
        if solver is 'SLSQP':
            if constraints is not None and len(constraints) > 0:
                if idom > 1:
                    if useDerivatives >= 0:
    #                     cons.append( {'type': 'eq', 'fun' : lambda x: np.array([ ( x[overlap] - (constraints[0][-1-overlap]) ) ])} )
                        print('Left delx: ', (x[overlapCP]-constraints[0][-1-overlapCP]))
                        cons.append( {'type': 'eq', 'fun' : lambda x: np.array([ ( x[overlapCP] - (constraints[1][overlapCP] + constraints[0][-1-overlapCP])/2 ) ])} )
                        if useDerivatives > 0:
                            cons.append( {'type': 'eq', 'fun' : lambda x: np.array([ ( pieceBezierDer22(x, W, U, t, degree)[overlapCP] - ( bzD[overlapCP] - bzDm[-1-overlapCP] )/2 ) ])} )
                            # cons.append( {'type': 'eq', 'fun' : lambda x: np.array([ ( pieceBezierDer22(x, W, U, t, degree)[0] - ( bzDm[-1]  ) ) ])} )
                            # cons.append( {'type': 'eq', 'fun' : lambda x: np.array([ ( (x[1] - x[0])/(knotsAll[idom-1][degree+1]-knotsAll[idom-1][0]) - ( constraints[idom-2][-1] - constraints[idom-2][-2] )/(knotsAll[idom-2][-degree-2] - knotsAll[idom-2][-1]) ) ])} )
                            if useDerivatives > 1:
                                cons.append( {'type': 'eq', 'fun' : lambda x: np.array([ ( (x[2+overlapCP] - x[1+overlap])/(knotsAll[1][degree+2+overlapCP]-knotsAll[1][1+overlapCP]) - (x[1+overlapCP] - x[overlapCP])/(knotsAll[1][degree+1+overlapCP]-knotsAll[1][overlapCP]) - ( (constraints[0][-3-overlapCP] - constraints[0][-2-overlapCP])/(knotsAll[0][-3-overlapCP]-knotsAll[0][-degree-2-overlapCP]) - (constraints[0][-2-overlapCP] - constraints[0][-1-overlapCP])/(knotsAll[0][-2-overlapCP]-knotsAll[0][-degree-overlapCP-3])  ) ) ])} )
                            
                            # cons.append( {'type': 'eq', 'fun' : lambda x: np.array([ ( x[1] - x[0] - 0.5*( constraints[idom-1][1] - constraints[idom-1][0] + constraints[idom-2][-1] - constraints[idom-2][-2] ) ) ])} )

                        # print 'Left delx: ', (knotsAll[idom-1][degree+1]-knotsAll[idom-1][0]), ' and ', (knotsAll[idom-1][degree+2]-knotsAll[idom-1][1])
                if idom < nSubDomains:
                    if useDerivatives >= 0:
    #                     cons.append( {'type': 'eq', 'fun' : lambda x: np.array([ ( x[-1-overlap] - (constraints[2][overlap]) ) ])} )
                        print('Right delx: ', (x[-1-overlapCP]-constraints[2][overlapCP]))
                        cons.append( {'type': 'eq', 'fun' : lambda x: np.array([ ( x[-1-overlapCP] - (constraints[1][-1-overlapCP] + constraints[2][overlapCP])/2 ) ])} )
                        if useDerivatives > 0:
                            cons.append( {'type': 'eq', 'fun' : lambda x: np.array([ ( pieceBezierDer22(x, W, U, t, degree)[-1-overlapCP] - (bzD[-1-overlapCP] - bzDp[overlapCP])/2 ) ])} )
                            # cons.append( {'type': 'eq', 'fun' : lambda x: np.array([ ( pieceBezierDer22(x, W, U, t, degree)[-1] - (bzDp[0]) ) ])} )
                            # cons.append( {'type': 'eq', 'fun' : lambda x: np.array([ ( (x[-1] - x[-2])/(knotsAll[idom-1][-degree-2] - knotsAll[idom-1][-1]) - ( constraints[idom][1] - constraints[idom][0] )/(knotsAll[idom][degree+1] - knotsAll[idom][0]) ) ])} )
                            if useDerivatives > 1:
                                cons.append( {'type': 'eq', 'fun' : lambda x: np.array([ ( (x[-3-overlapCP] - x[-2-overlapCP])/(knotsAll[1][-2-overlapCP] - knotsAll[1][-degree-3-overlapCP]) - (x[-2-overlapCP] - x[-1-overlapCP])/(knotsAll[1][-1-overlapCP] - knotsAll[1][-degree-2-overlapCP]) + ( (constraints[2][1+overlapCP] - constraints[2][overlapCP])/(knotsAll[2][degree+1+overlap] - knotsAll[2][overlapCP]) - (constraints[2][2+overlapCP] - constraints[2][1+overlapCP])/(knotsAll[2][degree+2+overlapCP] - knotsAll[2][1+overlapCP]) ) ) ])} )
                            # cons.append( {'type': 'eq', 'fun' : lambda x: np.array([ ( x[-1] - x[-2] - 0.5*( constraints[idom-1][-1] - constraints[idom-1][-2] + constraints[idom][1] - constraints[idom][0] ) ) ])} )
                        
                        # print 'Right delx: ', (knotsAll[idom-1][-2] - knotsAll[idom-1][-degree-3]), ' and ', (knotsAll[idom-1][-1] - knotsAll[idom-1][-degree-2])

                initSol = constraints[1][:] if len(constraints[1]) else np.ones_like(W)

    #             print len(initSol), len(W), len(ysl), len(U), len(t)
    #             E = np.sum(Error(initSol, W, ysl, U, t, degree)**2)
    #             print "unit error = ", E
                if enforceBounds:
                    # bnds = np.tensordot(np.ones(initSol.shape[0]), np.array([ysl.min(), ysl.max()]), axes=0)
                    bnds = np.tensordot(np.ones(initSol.shape[0]), np.array([initSol.min(), initSol.max()]), axes=0)
                    # bnds = None
                else:
                    bnds = None

                res = minimize(ComputeL2Error, x0=initSol, method='SLSQP', args=(N, W, ysl, U, t, degree),
                            constraints=cons, #callback=print_iterate,
                            bounds=bnds,
                            options={'disp': True, 'ftol': 1e-10, 'iprint': 1, 'maxiter': 1000})
            else:

    #             initSol = np.ones_like(W)
                initSol = lsqFit(N, W, ysl, U, t, degree)
                if enforceBounds:
                    bnds = np.tensordot(np.ones(initSol.shape[0]), np.array([ysl.min(), ysl.max()]), axes=0)
                else:
                    bnds = None

                print('Initial solution from LSQFit: ', initSol)
                res = minimize(ComputeL2Error, x0=initSol, method=solver, # Nelder-Mead, SLSQP, CG, L-BFGS-B
                            args=(N, W, ysl, U, t, degree),
                            bounds=bnds,
                            options={'disp': True, 'ftol': 1e-10, 'iprint': 1, 'maxiter': 1000})

                print('Final solution from LSQFit: ', res.x)
        else:
            if constraints is not None and len(constraints) > 0:
                if idom > 1:
                    print (idom, ': Left constraint ', (constraints[idom-1][overlap] + constraints[idom-2][-1-overlap])/2 )
                    cons.append( {'type': 'ineq', 'fun' : lambda x: np.array([ ( x[overlap] - (constraints[overlap][idom-1] + constraints[-1-overlap][idom-2])/2 ) ])} )
                if idom < nSubDomains:
                    print (idom, ': Right constraint ', (constraints[idom-1][-1-overlap] + constraints[idom][overlap])/2)
                    cons.append( {'type': 'ineq', 'fun' : lambda x: np.array([ ( x[-1-overlap] - (constraints[-1-overlap][idom-1] + constraints[overlap][idom])/2 ) ])} )

            res = minimize(ComputeL2Error, initSol, method='COBYLA', args=(N, W, ysl, U, t, degree),
                        constraints=cons, #x0=constraints,
                        options={'disp': False, 'tol': 1e-6, 'catol': 1e-2})

        print ('[%d] : %s' % (idom, res.message))
        return res.x


    def NonlinearOptimize(self, idom, N, W, ysl, U, t, degree, nSubDomains, constraintsAll=None, useDerivatives=0, solver='L-BFGS-B'):
        import autograd.numpy as np
        from autograd import elementwise_grad as egrad

        globalIterationNum = 0
        constraints = None
        RN = (N*W)/(np.sum(N*W, axis=1)[:,np.newaxis])
        if constraintsAll is not None:
            constraints = np.array(constraintsAll['P'])
            knotsAll = np.array(constraintsAll['T'])
            weightsAll = np.array(constraintsAll['W'])
            # print ('Constraints for Subdom = ', idom, ' is = ', constraints)

            # Update the local decoded data
            if len(constraints) and useDecodedResidual:
                decodedPrevIterate = RN.dot(constraints[1])
    #             decodedPrevIterate = decode(constraints[1], weightsAll[1], U,  t * (Dmax - Dmin) + Dmin, degree)
            
            decodedconstraint = RN.dot(constraints[1])
            residual_decodedcons = (decodedconstraint - ysl)#/yRange
            #print('actual error in input decoded data: ', (residual_decodedcons))

        else:
            print ('Constraints are all null. Solving unconstrained.')

        if useDerivatives > 0 and not useDecodedResidual and constraints is not None and len(constraints) > 0:

            bzD = np.array(pieceBezierDer22(constraints[1], weightsAll[1], U, knotsAll[1], degree))
            if idom < nSubDomains:
                bzDp = np.array(pieceBezierDer22(constraints[2], weightsAll[2], U, knotsAll[2], degree))
            if idom > 1:
                bzDm = np.array(pieceBezierDer22(constraints[0], weightsAll[0], U, knotsAll[0], degree))
            
            if useDerivatives > 1:
                bzDD = np.array(pieceBezierDer22(constraints[1], weightsAll[1], U, knotsAll[1], degree-1))
                if idom < nSubDomains:
                    bzDDp = np.array(pieceBezierDer22(constraints[2], weightsAll[2], U, knotsAll[2], degree-1))
                if idom > 1:
                    bzDDm = np.array(pieceBezierDer22(constraints[0], weightsAll[0], U, knotsAll[0], degree-1))
                

        def ComputeL2Error(P):
            E = (RN.dot(P) - ysl)
            return math.sqrt(np.sum(E**2)/len(E))

        def residual(Pin, verbose=False, vverbose=False):  # checkpoint2
            
            from autograd.numpy import linalg as LA
            decoded_data = RN.dot(Pin) # RN @ Pin #
    #         decoded_data = decode(Pin, W, U,  t * (Dmax - Dmin) + Dmin, degree)
            residual_decoded = (decoded_data[self.corebounds[0]:self.corebounds[1]] - ysl[self.corebounds[0]:self.corebounds[1]])/yRange # decoded_data[0:overlapData+1]
            residual_decoded_nrm = np.sqrt(np.sum(residual_decoded**2)/len(residual_decoded))
            # residual_decoded_nrm = LA.norm(residual_decoded, ord=2)
    #         print('actual decoded data: ', Pin, residual_decoded, residual_decoded_nrm)

            if useDecodedResidual:
                bc_penalty = 1e5
                ovRBFPower = 2.0
                overlapWeight = np.ones(overlapData+1)/( np.power(range(1, overlapData+2), ovRBFPower) )
                overlapWeightSum = np.sum(overlapWeight)
            else:
                bc_penalty = 1e12
            residual_constrained_nrm = 0
            nBndOverlap = 0
            # vverbose = True
            if constraints is not None and len(constraints) > 0:
                if idom > 1: # left constraint
                    if useDecodedResidual:
                        lconstraints = np.copy(constraints[0][:])
                        # lconstraints = np.flip(constraints[0][:])
                        # nBndOverlap += len(lconstraints)
                        nBndOverlap += 1
                        if vverbose:
                            # print('Left decoded delx: ', (decoded_data[0:overlapData+1]), (lconstraints))
                            print('Left decoded delx: ', (decoded_data[overlapData-1]), decodedPrevIterate[overlapData-1], (lconstraints[-1])) 
                        # residual_constrained_nrm += np.sum( (decoded_data[0:overlapData+1] - (lconstraints))**2 )
                        
                        # residual_constrained_nrm += np.sum( np.dot( (decoded_data[0:overlapData+1] - 
                        #                                     0.5 * ( decodedPrevIterate[0:overlapData+1] + lconstraints ) )**2, overlapWeight) ) / overlapWeightSum
                        residual_constrained_nrm += (decoded_data[overlapData] - 
                                                            0.5 * ( decodedPrevIterate[overlapData] + lconstraints[-1] ) )**2
                    else:
                        if useDerivatives >= 0:
                            # print('Left delx: ', (x[overlap]-constraints[0][-1-overlap]))
                            residual_constrained_nrm += bc_penalty * np.power( Pin[0] - (constraints[1][0] + constraints[0][-1])/2, 2.0 )
                            if useDerivatives > 0:
                                residual_constrained_nrm += np.power(bc_penalty, 0.5) * np.power( pieceBezierDer22(Pin, W, U, t, degree)[0] - ( bzD[0] + bzDm[-1] )/2, 2.0  )
                                # residual_constrained_nrm += np.power(bc_penalty, 0.5) * 0.5 * np.power( (knotsAll[1][1]-knotsAll[1][degree+2]) * ( pieceBezierDer22(Pin, W, U, t, degree)[0] - ( bzD[0] + bzDm[-1] )/2), 1.0 )
                                if useDerivatives > 1:
                                    residual_constrained_nrm += np.power(bc_penalty, 0.125) * \
                                                                np.power( (Pin[2] - Pin[1]) - (Pin[1] - Pin[0]) + 
                                                                #np.power( (Pin[2] - Pin[1])/(knotsAll[1][degree+2]-knotsAll[1][1]) - (Pin[1] - Pin[0])/(knotsAll[1][degree+1]-knotsAll[1][0]) + 
                                                                        ( (constraints[0][-3] - constraints[0][-2]) - #/(knotsAll[0][-3]-knotsAll[0][-degree-2]) - 
                                                                            (constraints[0][-2] - constraints[0][-1]) ), #/(knotsAll[0][-2]-knotsAll[0][-degree-3])  ), 
                                                                        2.0 )

                if idom < nSubDomains: # right constraint
                    if useDecodedResidual:
                        # rconstraints = np.copy(constraints[2][:])
                        rconstraints = np.flip(constraints[2][:])
                        # nBndOverlap += len(rconstraints)
                        nBndOverlap += 1
                        if vverbose:
                            # print('Right decoded delx: ', decoded_data[-1-overlapData:], rconstraints)
                            print('Right decoded delx: ', decoded_data[-1-overlapData], decodedPrevIterate[-1-overlapData], rconstraints[0])
                        # residual_constrained_nrm += np.sum( (decoded_data[-1-overlapData:] - (rconstraints))**2 )

                        # residual_constrained_nrm += np.sum( np.dot( (decoded_data[-1-overlapData:] - 
                        #                                     0.5 * ( decodedPrevIterate[-1-overlapData:] + rconstraints ) )**2, overlapWeight) ) / overlapWeightSum
                        residual_constrained_nrm += np.sum( (decoded_data[-1-overlapData] - 
                                                            0.5 * ( decodedPrevIterate[-1-overlapData] + rconstraints[0] ) )**2)
                    else:
                        if useDerivatives >= 0:
        #                         print('Right delx: ', (x[-1-overlap]-constraints[2][overlap]))
                            residual_constrained_nrm += bc_penalty * np.power( Pin[-1] - (constraints[1][-1] + constraints[2][0])/2, 2.0 )
                            if useDerivatives > 0:
                                residual_constrained_nrm += np.power(bc_penalty, 0.5) * np.power( pieceBezierDer22(Pin, W, U, t, degree)[-1] - (bzD[-1] + bzDp[0])/2, 2.0 )
                                # residual_decoded[-1] += np.power(bc_penalty, 0.5) * 0.5 * np.power( (knotsAll[1][-degree-3] - knotsAll[1][-2]) * ( pieceBezierDer22(Pin, W, U, t, degree)[-1] - (bzD[-1] + bzDp[0])/2 ), 1.0)
                                if useDerivatives > 1:
                                    residual_constrained_nrm += np.power(bc_penalty, 0.125) * \
                                                                np.power( (Pin[-3] - Pin[-2]) - (Pin[-2] - Pin[-1]) + 
                                                                #np.power( (Pin[-3] - Pin[-2])/(knotsAll[1][-2] - knotsAll[1][-degree-3]) - (Pin[-2] - Pin[-1])/(knotsAll[1][-1] - knotsAll[1][-degree-2]) + 
                                                                        ( (constraints[2][1] - constraints[2][0]) - #/(knotsAll[2][degree+1] - knotsAll[2][0]) - 
                                                                            (constraints[2][2] - constraints[2][1]) ), #/(knotsAll[2][degree+2] - knotsAll[2][1]) ),
                                                                        2.0 )

            if useDecodedResidual:
                residual_constrained_nrm = np.sqrt(residual_constrained_nrm/(nBndOverlap+1))
            else:
                residual_constrained_nrm = np.sqrt(residual_constrained_nrm/(useDerivatives+1))

            residual_nrm = residual_decoded_nrm + residual_constrained_nrm if disableAdaptivity else 0.0

            if verbose:
                print('NLConstrained residual norm: total = ', residual_nrm, 'decoded = ', residual_decoded_nrm, 'boundary-constraints = ', residual_constrained_nrm)

            return residual_nrm

        def residual_vec(Pin, verbose=False, vverbose=False):  # checkpoint2
            # RN : Operatror to project from Control point space to Decoded
            # RN.T : Reverse direction
            from autograd.numpy import linalg as LA
            decoded_data = RN.dot(Pin) # RN @ Pin #
            # residual_decoded_full = np.power( (decoded_data - ysl)/yRange, 2.0 )
            residual_decoded_full = np.power( (decoded_data - ysl)/yRange, 1.0 )

            # RN.T * RN * P = RN.T * Y
            # RN.T * (RN * P - Y)
            if not useDecodedResidual:
                residual_decoded = RN.T @ residual_decoded_full
            else:
                residual_decoded = residual_decoded_full[:]

            if useDecodedResidual:
                bc_penalty = 1e0
                ovRBFPower = 0.0
                overlapWeight = np.ones(overlapData+1)/( np.power(range(1, overlapData+2), ovRBFPower) )
                overlapWeightSum = np.sum(overlapWeight)
            else:
                bc_penalty = 1e5
            residual_constrained_nrm = 0
            if constraints is not None and len(constraints) > 0:
                if idom > 1: # left constraint
                    if useDecodedResidual:
                        lconstraints = np.copy(constraints[0][:])
                        # lconstraints = np.flip(constraints[0][:])
                        if vverbose or True:
                            print('Left decoded delx: ', (decoded_data[0:overlapData+1]), (lconstraints))
                        # residual_decoded += np.sum( (decoded_data[0:overlapData+1] - (lconstraints))**2 )
                        residual_decoded[0:overlapData+1] += bc_penalty * np.multiply( (decoded_data[0:overlapData+1] - 0.5*(decodedPrevIterate[0:overlapData+1] + lconstraints)), overlapWeight) / overlapWeightSum
                    else:
                        if useDerivatives >= 0:
                            residual_decoded[0] += bc_penalty * ( Pin[0] - (constraints[1][0] + constraints[0][-1])/2 )
                            if useDerivatives > 0:
                                residual_decoded[0] += np.power(bc_penalty, 0.5) * 0.5 * (knotsAll[1][1]-knotsAll[1][degree+2]) * np.power( pieceBezierDer22(Pin, W, U, t, degree)[0] - ( bzD[0] + bzDm[-1] )/2, 1.0  )
                                if useDerivatives > 1:
                                    residual_decoded[0] += bc_penalty * np.power( (Pin[2+overlapCP] - Pin[1+overlapCP])/(knotsAll[1][degree+2]-knotsAll[1][1]) - (Pin[1+overlapCP] - Pin[overlapCP])/(knotsAll[1][degree+1+overlapCP]-knotsAll[1][overlapCP]) - ( (constraints[0][-3-overlapCP] - constraints[0][-2-overlapCP])/(knotsAll[0][-3-overlapCP]-knotsAll[0][-degree-2-overlapCP]) - (constraints[0][-2-overlapCP] - constraints[0][-1-overlapCP])/(knotsAll[0][-2-overlapCP]-knotsAll[0][-degree-overlapCP-3])  ), 1.0 )

                if idom < nSubDomains: # right constraint
                    if useDecodedResidual:
                        rconstraints = np.copy(constraints[2][:])
                        # rconstraints = np.flip(constraints[2][:])
                        if vverbose:
                            print('Right decoded delx: ', decoded_data[-1-2*overlapData:-overlapData], rconstraints)
                        # residual_decoded += np.sum( (decoded_data[-1-overlapData:] - (rconstraints))**2 )
                        residual_decoded[-1-2*overlapData:-overlapData] += bc_penalty * np.multiply( (decoded_data[-1-2*overlapData:-overlapData] - 0.5*(decodedPrevIterate[-1-2*overlapData:-overlapData] + rconstraints)), overlapWeight) / overlapWeightSum
                    else:
                        if useDerivatives >= 0:
                            residual_decoded[-1] += bc_penalty * ( Pin[-1] - (constraints[1][-1] + constraints[2][0])/2 )
                            if useDerivatives > 0:
                                residual_decoded[-1] += np.power(bc_penalty, 0.5) * 0.5 * (knotsAll[1][-degree-3] - knotsAll[1][-2]) * np.power( pieceBezierDer22(Pin, W, U, t, degree)[-1] - (bzD[-1] + bzDp[0])/2, 1.0 )
                                if useDerivatives > 1:
                                    residual_decoded[-1] += bc_penalty * np.power( (Pin[-3-overlapCP] - Pin[-2-overlapCP])/(knotsAll[1][-2] - knotsAll[1][-degree-3]) - (Pin[-2-overlapCP] - Pin[-1-overlapCP])/(knotsAll[1][-1-overlapCP] - knotsAll[1][-degree-2-overlapCP]) + ( (constraints[2][1+overlapCP] - constraints[2][overlapCP])/(knotsAll[2][degree+1+overlapCP] - knotsAll[2][overlapCP]) - (constraints[2][2+overlapCP] - constraints[2][1+overlapCP])/(knotsAll[2][degree+2+overlapCP] - knotsAll[2][1+overlapCP]) ), 1.0 )

            if verbose:
                print('NLConstrained residual vector norm: ', np.sqrt(np.sum(np.abs(residual_decoded))/residual_decoded.shape[0]))

            if useDecodedResidual:
                residual_decoded_t = RN.T @ residual_decoded
                residual_decoded = residual_decoded_t[:] 

            return residual_decoded

        # Use automatic-differentiation to compute the Jacobian value for minimizer
        def jacobian(Pin):
            jacobian = egrad(residual)(Pin)
            return jacobian

        def jacobian_vec(Pin):
            jacobian = egrad(residual_vec)(Pin)
            return jacobian

        def print_iterate(P, res=None):
            if res is None:
                res = residual(P, verbose=True)
            else:
                print('NLConstrained residual vector norm: ', np.sqrt(np.sum(np.abs(res))/res.shape[0]), ' Boundary: ', res[0], res[-1])
            return False

        initSol = constraints[1][:] if constraints is not None else np.ones_like(W)
        # initSol = np.ones_like(W)*0

        if enforceBounds:
            bnds = np.tensordot(np.ones(initSol.shape[0]), np.array([initSol.min(), initSol.max()]), axes=0)
            # bnds = None
        else:
            bnds = None

        if solver in ['Nelder-Mead', 'Powell', 'Newton-CG', 'TNC', 'trust-ncg', 'trust-krylov', 'SLSQP', 'L-BFGS-B', 'CG']:
            minimizer_options={ 'disp': False, 
                                'ftol': maxRelErr, 
                                'gtol': globalTolerance, 
                                'maxiter': solverMaxIter
                            }
            # jacobian_const = egrad(residual)(initSol)
            res = minimize(residual, x0=initSol, method=solver, #'SLSQP', #'L-BFGS-B', #'TNC', 'CG', 'Newton-CG'
                        bounds=bnds,
                        #jac=jacobian,
                        callback=print_iterate,
                        tol=globalTolerance, 
                        options=minimizer_options)
        else:
            optimizer_options={ 'disp': False, 
                                'ftol': globalTolerance, 
                                'maxiter': solverMaxIter
                            }
            # jacobian_const = egrad(residual)(initSol)
            res = scipy.optimize.root(residual_vec, x0=initSol, 
                        method=solver, #'krylov', 'lm'
                        # jac=jacobian_vec,
                        callback=print_iterate,
                        tol=globalTolerance, 
                        options=optimizer_options)

        print ('[%d] : %s' % (idom, res.message))
        return res.x



    def adaptive(self, iSubDom, interface_constraints_obj, u, xl, yl, strategy='reset', r=1, MAX_ERR=1e-2, MAX_ITER=5, split_all=False):
        splitIndeces = []
        r = min(r,degree) #multiplicity can not be larger than degree
        nPointsPerSubD = xl.shape[0]

        T = interface_constraints_obj['T'][1]
        if len(interface_constraints_obj['P']):
            P = interface_constraints_obj['P'][1]
            W = interface_constraints_obj['W'][1]

        if len(P) == 0:
            W = np.ones(len(T) - 1 - degree)
            N = basis(u[np.newaxis,:],degree,T[:,np.newaxis]).T
            P = self.lsqFit(N, W, yl, u, T, degree)
    #         P = LSQFit_Constrained(iSubDom, N, W, yl, u, T, degree, nSubDomains, 
    #                        None, #interface_constraints_obj, 
    #                        useDerivativeConstraints, 'SLSQP')
    #         P = NonlinearOptimize(iSubDom, N, W, yl, u, T, degree, nSubDomains, 
    #                                None, #interface_constraints_obj, 
    #                                useDerivativeConstraints, subdomainSolver)


            RN = (N*W)/(np.sum(N*W, axis=1)[:,np.newaxis])
            decodedconstraint = RN @ P # RN.dot(P)
            residual_decodedcons = (decodedconstraint - yl)#/yRange
            MAX_ITER = 0
            # print(iSubDom, ' -- actual error in input decoded data after LSQFit: ', P, (residual_decodedcons))

        for iteration in range(MAX_ITER):
            E = Error(P, W, yl, u, T, degree)[self.corebounds[0]:self.corebounds[1]]/yRange

            # L2Err = np.linalg.norm(E, ord=2)
            L2Err = np.sqrt(np.sum(E**2)/len(E))

            print (" -- Adaptive iteration: ", iteration, ': Error = ', L2Err)
            if disableAdaptivity:
                Tnew = np.copy(T)
            else:
                Tnew,splitIndeces = knotRefine(P, W, T, u, degree, yl, E, r, MAX_ERR=MAX_ERR, find_all=split_all)
                if ((len(T)==len(Tnew)) or len(T)-degree-1 > nPointsPerSubD) and not (iteration == 0) :
                    break
            
            if strategy == 'extend' and ~split_all:   #only use when coupled with a solver
                k = splitIndeces[0]
                u = Tnew[k+1]
                P, W = deCasteljau(P, W, T, u, k, r)
            elif strategy == 'reset':
                Tnew = np.sort(Tnew)
                W = np.ones(len(Tnew) - 1 - degree)
                N = basis(u[np.newaxis,:],degree,Tnew[:,np.newaxis]).T


    #             P = lsqFit(N, W, yl, u, Tnew, degree)
    #             return P, W, Tnew

    #             P = lsqFitWithCons(N, W, yl, u, Tnew, degree)

    #             if len(interface_constraints_obj['P'][iSubDom - 1]) > 0 and len(interface_constraints_obj['P'][iSubDom]) > 0 and len(interface_constraints_obj['P'][iSubDom - 2]) > 0:
                if len(interface_constraints_obj['P'][1]) > 0:
                    # Interpolate or project the data to new Knot locations: From (P, T) to (Pnew, Tnew)
                    coeffs_x = getControlPoints(T, degree) #* (Dmax - Dmin) + Dmin
                    coeffs_xn = getControlPoints(Tnew, degree) #* (Dmax - Dmin) + Dmin
                    
                    PnewFn = Rbf(coeffs_x, P, function='cubic')
                    Pnew = PnewFn(coeffs_xn)

    #                 PnewFn = interp1d(coeffs_x, P, kind='quintic') #, kind='cubic')
    #                 Pnew = PnewFn(coeffs_xn)
                    # print 'coeffs_x = ', [coeffs_x, coeffs_xn, P, Pnew]
                    interface_constraints_obj['P'][1]=Pnew[:]
                    interface_constraints_obj['W'][1]=W[:]
                    interface_constraints_obj['T'][1]=Tnew[:]
                    # if iSubDom < nSubDomains:
                    #     print ('Constraints for left-right Subdom = ', iSubDom, ' is = ', [Pnew, interface_constraints_obj['P'][iSubDom]])
                    # else:
                    #     print ('Constraints for right-left Subdom = ', iSubDom, ' is = ', [interface_constraints_obj['P'][iSubDom-2], Pnew] )

                    print('Solving the boundary-constrained LSQ problem')
                    if subdomainSolver == 'LCLSQ':
                        P = self.WeightedConstrainedLSQ_PT(iSubDom, N, W, yl, u, Tnew, degree, nSubDomains, 
                                            interface_constraints_obj, 
                                            useDerivativeConstraints)
                    elif subdomainSolver == 'SLSQP':
                        P = self.LSQFit_Constrained(iSubDom, N, W, yl, u, Tnew, degree, nSubDomains, 
                                            interface_constraints_obj, 
                                            useDerivativeConstraints, 'SLSQP')
                    else: # subdomainSolver == 'NonlinearSolver'
                        P = self.NonlinearOptimize(iSubDom, N, W, yl, u, Tnew, degree, nSubDomains, 
                                            interface_constraints_obj, 
                                            useDerivativeConstraints, subdomainSolver)

                else:
                    print('Solving the unconstrained LSQ problem')
                    P = self.lsqFit(N, W, yl, u, Tnew, degree)

            else:
                print ("Not Implemented!!")
            
            T = Tnew
            
        return P, W, T


    def solve_adaptive(self, cp):

        if self.outerIterationConverged:
            print(cp.gid()+1, ' subdomain has already converged to its solution. Skipping solve ...')
            return

        ## Subdomain ID: iSubDom = cp.gid()+1
#         domStart = (cp.gid()) * 1.0 / nSubDomains
#         domEnd   = (cp.gid()+1) * 1.0 / nSubDomains
        domStart = self.xl[0]#/(Dmax-Dmin)
        domEnd   = self.xl[-1]#/(Dmax-Dmin)

        U   = np.linspace(domStart, domEnd, self.nPointsPerSubD)

        newSolve = False
        if len(self.pAdaptive) == 0:
            newSolve = True

        if newSolve:

            inc = (domEnd - domStart) / self.nInternalKnotSpans
            knots  = np.linspace(domStart + inc, domEnd - inc, self.nInternalKnotSpans - 1)
            knots  = np.concatenate(([domStart] * (degree+1), knots, [domEnd] * (degree+1)))

            popt = []
            W = []

        else:

            knots = self.knotsAdaptive[:]
            popt = self.pAdaptive[:]
            W = self.WAdaptive[:]
            nControlPoints = len(popt)

            self.nControlPointSpans = self.nControlPoints - 1
            self.nInternalKnotSpans = self.nControlPointSpans - degree + 1

        print ("\nSubdomain -- ", cp.gid()+1, "starting adaptive solver ...")

        # Let do the recursive iterations
        # Use the previous MAK solver solution as initial guess; Could do something clever later
        self.interface_constraints_obj['P'][1] = popt[:]
        self.interface_constraints_obj['T'][1] = knots[:]
        self.interface_constraints_obj['W'][1] = W[:]
        
        # if there is overlapping data, project to the interface and insert the control point positions
        # as additional DoFs
        leftconstraint_projected = np.copy(self.leftconstraint)
        rightconstraint_projected = np.copy(self.rightconstraint)
        leftconstraint_projected_knots = np.copy(self.leftconstraintKnots)
        rightconstraint_projected_knots = np.copy(self.rightconstraintKnots)
        if overlapCP > 0:
            if len(self.leftconstraint) > 0:
                leftconstraint_projected, leftconstraint_projected_knots = self.interpolate(self.leftconstraint, self.leftconstraintKnots, knots)
            if len(self.rightconstraint) > 0:
                rightconstraint_projected, rightconstraint_projected_knots = self.interpolate(self.rightconstraint, self.rightconstraintKnots, knots)

        self.interface_constraints_obj['P'][0] = leftconstraint_projected[:]
        self.interface_constraints_obj['T'][0] = leftconstraint_projected_knots[:]
        self.interface_constraints_obj['P'][2] = rightconstraint_projected[:]
        self.interface_constraints_obj['T'][2] = rightconstraint_projected_knots[:]

        if disableAdaptivity:
            nmaxAdaptIter = 1
        else:
            nmaxAdaptIter = 3

        xSD = U * (Dmax - Dmin) + Dmin

        # Invoke the adaptive fitting routine for this subdomain
        self.pAdaptive, self.WAdaptive, self.knotsAdaptive = self.adaptive(cp.gid()+1, self.interface_constraints_obj, U, 
                                                                            #self.xSD, self.ySD, 
                                                                            self.xl, self.yl,
                                                                            #MAX_ERR=maxAdaptErr,
                                                                            MAX_ERR=maxAbsErr,
                                                                            split_all=True, 
                                                                            strategy=AdaptiveStrategy, 
                                                                            r=1, MAX_ITER=nmaxAdaptIter)

        # NAdaptive = basis(U[np.newaxis,:],degree,knotsAdaptive[:,np.newaxis]).T
        # E = Error(pAdaptive, WAdaptive, ySD, U, knotsAdaptive, degree)
        # print ("Sum of squared error:", np.sum(E**2))
        # print ("Normalized max error:", np.abs(E).max()/yRange)
        
        # Update the local decoded data
        self.decodedAdaptiveOld = np.copy(self.decodedAdaptive)
        self.decodedAdaptive = decode(self.pAdaptive, self.WAdaptive, self.xl, self.knotsAdaptive, degree)
        
#        print('Local decoded data for Subdomain: ', cp.gid(), self.xl.shape, self.decodedAdaptive.shape, self.xl, self.decodedAdaptive)

        # errorMAK = L2LinfErrors(self.pAdaptive, self.WAdaptive, self.yl, U, self.knotsAdaptive, degree)
        E = (self.decodedAdaptive[self.corebounds[0]:self.corebounds[1]] - self.yl[self.corebounds[0]:self.corebounds[1]])/yRange
        LinfErr = np.linalg.norm(E, ord=np.inf)
        L2Err = np.sqrt(np.sum(E**2)/len(E))

        self.decodederrors[0] = L2Err
        self.decodederrors[1] = LinfErr

        print ("Subdomain -- ", cp.gid()+1, ": L2 error: ", L2Err, ", Linf error: ", LinfErr)

#     return PAdaptDomain, WAdaptDomain, KnotAdaptDomains


    def check_convergence(self, cp):

        if len(self.decodedAdaptiveOld):

            # Let us compute the relative change in the solution between current and previous iteration
            iterateChangeVec = (self.decodedAdaptive - self.decodedAdaptiveOld)
            errorMetricsSubDomL2 = np.linalg.norm(iterateChangeVec, ord=2) / np.linalg.norm(self.pAdaptive, ord=2)
            errorMetricsSubDomLinf = np.linalg.norm(iterateChangeVec, ord=np.inf) / np.linalg.norm(self.pAdaptive, ord=np.inf)

            self.errorMetricsL2[self.outerIteration] = self.decodederrors[0]
            self.errorMetricsLinf[self.outerIteration] = self.decodederrors[1]

            if errorMetricsSubDomLinf < 1e-13 and np.abs(self.errorMetricsLinf[self.outerIteration]-self.errorMetricsLinf[self.outerIteration-1]) < 1e-10:
                print('Subdomain ', cp.gid()+1, ' has converged to its final solution with error = ', errorMetricsSubDomLinf)
                self.outerIterationConverged = True
        
        self.outerIteration += 1


#########
from cycler import cycler

# Routine to recursively add a block and associated data to it
def add_input_control_block(gid, core, bounds, domain, link):
    print("Subdomain %d: " % gid, core, bounds, domain, link)
    minb = bounds.min
    maxb = bounds.max
    xlocal = x[minb[0]:maxb[0]+1]
    ylocal = y[minb[0]:maxb[0]+1]
    # print("Subdomain %d: " % gid, xlocal.shape, ylocal.shape, core, bounds, domain)
    mc.add(gid, InputControlBlock(nControlPoints,core,bounds,xlocal,ylocal), link)

# TODO: If working in parallel with MPI or DIY, do a global reduce here

# print "Initial condition data: ", interface_constraints
errors = np.zeros([10,1]) # Store L2, Linf errors as function of iteration

# Let us initialize DIY and setup the problem
share_face = [True]
wrap = [False]
ghosts = [overlapData]

# Initialize DIY
mc = diy.Master(w)         # master
domain_control = diy.DiscreteBounds([0], [len(x)-1])

d_control = diy.DiscreteDecomposer(1, domain_control, nSubDomains, share_face, wrap, ghosts)
a_control = diy.ContiguousAssigner(w.size, nSubDomains)

d_control.decompose(w.rank, a_control, add_input_control_block)

mc.foreach(InputControlBlock.show)

convergedASMIterates = False

sys.stdout.flush()
commW.Barrier()

#########
import timeit
start_time = timeit.default_timer()
for iterIdx in range(nASMIterations):

    print ("\n---- Starting ASM Iteration: %d with %s inner solver ----" % (iterIdx, subdomainSolver))
    
    # Now let us perform send-receive to get the data on the interface boundaries from 
    # adjacent nearest-neighbor subdomains
    mc.foreach(InputControlBlock.send)
    mc.exchange(False)
    mc.foreach(InputControlBlock.recv)

    if iterIdx > 1: 
        disableAdaptivity = True

    if iterIdx > 0: print("")
    mc.foreach(InputControlBlock.solve_adaptive)
    
    if disableAdaptivity:
        print("")
        mc.foreach(InputControlBlock.check_convergence)

        # convergedASMIterates = np.all(iterationResult)

    if showplot:
        # Let us plot the initial data
        # if nSubDomains > 5:
        #     plt.rc('axes', prop_cycle=(cycler('color', ['r', 'g', 'b', 'y', 'c']) + cycler('linestyle', ['-','--',':','-.',''])))
        
        plt.figure()
        plt.plot(x, y, 'b-', ms=5, label='Input')
        mc.foreach(InputControlBlock.plot)
        plt.legend()

        plt.figure()
        mc.foreach(InputControlBlock.plot_error)
        plt.legend()

        plt.draw()

    print('')

    # if convergedASMIterates: break

sys.stdout.flush()

elapsed = timeit.default_timer() - start_time

commW.Barrier()
if rank == 0: print('Total computational time for solve = ', elapsed)

np.set_printoptions(formatter={'float': '{: 5.6e}'.format})
mc.foreach(InputControlBlock.print_error_metrics)

if showplot: plt.show()
