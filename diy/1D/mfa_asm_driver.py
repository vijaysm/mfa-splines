# coding: utf-8

# TODO:
# Do not pin the internal subdomain boundary points
# Encode and decode correctly. Same solution always as single subdomain case
# Then start doing overlap of data/control points (?) and do iterative solution
#
import timeit

import splipy as sp

from cycler import cycler
import sys
import getopt
import math
import numpy as np
import scipy

import diy
from matplotlib import pyplot as plt
import mpl_toolkits.mplot3d.axes3d as axes3d
from matplotlib import cm
import pandas as pd
from mpi4py import MPI

from scipy import linalg

from makruth_solver import getControlPoints, Error, L2LinfErrors
from makruth_solver import knotInsert, knotRefine, deCasteljau, pieceBezierDer22

plt.style.use(['seaborn-whitegrid'])
params = {"ytick.color": "b",
          "xtick.color": "b",
          "axes.labelcolor": "b",
          "axes.edgecolor": "b"}
plt.rcParams.update(params)

# --- set problem input parameters here ---
nSubDomains = 2
degree = 2
nControlPoints = 12  # (3*degree + 1) #minimum number of control points
overlapData = 0
overlapCP = 0
problem = 0
scale = 1
showplot = False
nASMIterations = 2
augmentSpanSpace = 0
extrapolate = False
useAitken = True
nWynnEWork = 10
#
# ------------------------------------------
# Solver parameters
solverscheme = 'SLSQP'  # [SLSQP, COBYLA]
useAdditiveSchwartz = True
useDerivativeConstraints = 0
enforceBounds = False
disableAdaptivity = True
#
#                            0        1         2        3     4       5          6           7         8
subdomainSolverSchemes = ['LCLSQ', 'SLSQP', 'L-BFGS-B', 'CG', 'lm', 'krylov', 'broyden2', 'anderson', 'custom']
subdomainSolver = subdomainSolverSchemes[2]

maxAbsErr = 1e-2
maxRelErr = 1e-8

solverMaxIter = 2
globalTolerance = 1e-12
fullyPinned = False

# AdaptiveStrategy = 'extend'
AdaptiveStrategy = 'reset'

# Initialize DIY
commW = MPI.COMM_WORLD
nprocs = commW.size
rank = commW.rank

if rank == 0:
    print('Argument List:', str(sys.argv))

##########################
# Parse command line overrides if any
##
argv = sys.argv[1:]


def usage():
    print(
        sys.argv[0],
        '-p <problem> -n <nsubdomains> -d <degree> -c <controlpoints> -o <overlapData> -a <nASMIterations> -g <ExtraKnotSpans>')
    sys.exit(2)


try:
    opts, args = getopt.getopt(
        argv, "hn:d:c:o:p:a:g:i:s",
        ["nsubdomains=", "degree=", "controlpoints=", "overlap=", "problem=", "nasm=", "aug=", "plot", "accel", "wynn"])
except getopt.GetoptError:
    usage()

print(opts)
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
    elif opt in ("-g", "--aug"):
        augmentSpanSpace = int(arg)
    elif opt in ("-i", "--plot"):
        showplot = True
    elif opt in ("-s", "--accel"):
        extrapolate = True
    elif opt in ("--wynn"):
        useAitken = False


# ----------------------
# Problematic settings
# problem        = 0
# nSubDomains    = 4
# degree         = 3
# nControlPoints = (3*degree + 1) #minimum number of control points
# subdomainSolver = subdomainSolverSchemes[0]
# ----------------------

if problem == 0:
    Dmin = -4.
    Dmax = 4.
    nPoints = 10001
    x = np.linspace(Dmin, Dmax, nPoints)
    scale = 100
    # y = scale * (np.sinc(x-1)+np.sinc(x+1))
    # y = scale * (np.sinc(x+1) + np.sinc(2*x) + np.sinc(x-1))
    y = scale * (np.sinc(x) + np.sinc(2*x-1) + np.sinc(3*x+1.5))
    # y = np.zeros(x.shape)
    # y[x <= 0] = 1
    # y[x > 0] = -1
    # y = scale * np.sin(math.pi * x/4)
elif problem == 1:
    y = np.fromfile("data/s3d.raw", dtype=np.float64)
    print('Real data shape: ', y.shape)
    nPoints = y.shape[0]
    Dmin = 0
    Dmax = 1.
    x = np.linspace(Dmin, Dmax, nPoints)
elif problem == 2:
    Y = np.fromfile("data/nek5000.raw", dtype=np.float64)
    Y = Y.reshape(200, 200)
    y = Y[100, :]  # Y[:,150] # Y[110,:]
    Dmin = 0
    Dmax = 1.
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
    Dmin = 0
    Dmax = 100.
    nPoints = y.shape[0]
    x = np.linspace(Dmin, Dmax, nPoints)


xmin = x.min()
xmax = x.max()
ymin = y.min()
ymax = y.max()
yRange = ymax-ymin
yRange = 1

if showplot:
    mpl_fig = plt.figure()
    plt.plot(x, y, 'r-', ms=2)

# ------------------------------------

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
    print('enforceBounds = ', enforceBounds)
    print('maxAbsErr = ', maxAbsErr)
    print('maxRelErr = ', maxRelErr)
    print('solverMaxIter = ', solverMaxIter)
    print('globalTolerance = ', globalTolerance)
    print('AdaptiveStrategy = ', AdaptiveStrategy)
    print('solverscheme = ', solverscheme)
    print('subdomainSolver = ', subdomainSolver)
    print('\n=================\n')

# --------------------------------------
# Let do the recursive iterations
# The variable `additiveSchwartz` controls whether we use
# Additive vs Multiplicative Schwartz scheme for DD resolution
####

isConverged = np.zeros(nSubDomains, dtype='int32')
L2err = np.zeros(nSubDomains)


class InputControlBlock:

    def __init__(self, nControlPoints, coreb, xb, xl, yl):
        self.nControlPoints = nControlPoints
        self.nControlPointSpans = nControlPoints - 1
        self.nInternalKnotSpans = self.nControlPointSpans - degree + 1
        assert(self.nInternalKnotSpans > 1)
        self.nPointsPerSubD = xl.shape[0]  # int(nPoints / nSubDomains) + overlapData
        self.xbounds = xb
        # self.corebounds = coreb
        # self.corebounds = [coreb.min[0], coreb.max[0]]
        self.corebounds = [coreb.min[0]-xb.min[0], -1+coreb.max[0]-xb.max[0]]
        self.xl = xl
        self.yl = yl
        self.B = None  # Basis function object
        self.pAdaptive = []
        self.WAdaptive = []
        self.knotsAdaptive = []
        self.leftconstraint = np.zeros(overlapCP+1)
        self.leftconstraintKnots = np.zeros(overlapCP+1)
        self.rightconstraint = np.zeros(overlapCP+1)
        self.rightconstraintKnots = np.zeros(overlapCP+1)
        # Allocate for the constraints
        self.interface_constraints_obj = dict()
        self.interface_constraints_obj['P'] = [[], [], []]
        self.interface_constraints_obj['W'] = [[], [], []]
        self.interface_constraints_obj['T'] = [[], [], []]
        self.decodedAdaptive = []
        self.decodedAdaptiveOld = []

        self.RN = []

        self.leftclamped = self.rightclamped = False

        # Convergence related metrics and checks
        self.outerIteration = 0
        self.outerIterationConverged = False
        self.decodederrors = np.zeros(2)  # L2 and Linf
        self.errorMetricsL2 = np.zeros((nASMIterations), dtype='float64')
        self.errorMetricsLinf = np.zeros((nASMIterations), dtype='float64')

        self.pAdaptiveHistory = []

    def show(self, cp):
        # print("Rank: %d, Subdomain %d: Bounds = [%d, %d], Core = [%d, %d]" % (w.rank, cp.gid(), self.xbounds.min[0], self.xbounds.max[0]+1, self.corebounds.min[0], self.corebounds.max[0]))
        print(
            "Rank: %d, Subdomain %d: Bounds = [%d, %d], Core = [%d, %d]" %
            (w.rank, cp.gid(),
             self.xbounds.min[0],
             self.xbounds.max[0] + 1, self.corebounds[0],
             self.corebounds[1]))
        # cp.enqueue(diy.BlockID(1, 0), "abc")

    # Greville points
    def getLocalControlPoints(self, knots, k):

        return np.array(sp.BSplineBasis(order=k+1, knots=knots).greville())

    def plot(self, cp):
        #         print(w.rank, cp.gid(), self.core)
        # self.decodedAdaptive = self.decode(self.pAdaptive)
        coeffs_x = self.getLocalControlPoints(self.knotsAdaptive, degree)  # * (Dmax - Dmin) + Dmin
        # print(coeffs_x.shape, self.pAdaptive.shape, coeffs_x)
        plt.plot(self.xl, self.decodedAdaptive, linestyle='--', lw=2,
                 color=['r', 'g', 'b', 'y', 'c'][cp.gid() % 5], label="Decoded-%d" % (cp.gid()+1))
        plt.plot(coeffs_x, self.pAdaptive, marker='o', linestyle='', color=[
                 'r', 'g', 'b', 'y', 'c'][cp.gid() % 5], label="Control-%d" % (cp.gid()+1))

    def plot_error(self, cp):
        error = self.yl - self.decodedAdaptive
        plt.plot(self.xl[self.corebounds[0]: self.corebounds[1]+1],
                 error[self.corebounds[0]: self.corebounds[1]+1],
                 # plt.plot(self.xl,
                 #          error,
                 linestyle='--', color=['r', 'g', 'b', 'y', 'c'][cp.gid() % 5],
                 lw=2, label="Subdomain(%d) Error" % (cp.gid() + 1))

    def plot_with_cp(self, cp, cploc, ctrlpts, lgndtitle, indx):
        pMK = self.decode(ctrlpts)
        plt.plot(self.xl, pMK, linestyle='--', color=['g', 'b', 'y', 'c'][indx % 5], lw=3, label=lgndtitle)

    def plot_with_cp_and_knots(self, cp, cploc, knots, ctrlpts, weights, lgndtitle, indx):
        # print('Plot: shapes = ', ctrlpts.shape[0], cploc.shape[0], knots.shape[0], degree)
        pMK = self.decode(ctrlpts)
        plt.plot(self.xl, pMK, linestyle='--', color=['g', 'b', 'y', 'c'][indx], lw=3, label=lgndtitle)

    def print_error_metrics(self, cp):
        print('Rank:', commW.rank, ' SDom:', cp.gid(), ' L2 Error table: ',
              self.errorMetricsL2[np.nonzero(self.errorMetricsL2)])

    def send(self, cp):
        link = cp.link()
        for i in range(len(link)):
            target = link.target(i)
            o = np.zeros(overlapCP+1)
            if target.gid > cp.gid():  # target is to the right of current subdomain
                if len(self.pAdaptive):
                    o = np.array(
                        [self.pAdaptive.shape[0],
                            self.knotsAdaptive.shape[0],
                            self.pAdaptive[:],
                            self.knotsAdaptive[:]])
                # print("%d sending to %d: %s" % (cp.gid()+1, target.gid+1, o))
                # print("%d sending to %d" % (cp.gid()+1, target.gid+1))
            else:  # target is to the left of current subdomain
                if len(self.pAdaptive):
                    o = np.array(
                        [self.pAdaptive.shape[0],
                            self.knotsAdaptive.shape[0],
                            self.pAdaptive[:],
                            self.knotsAdaptive[:]])
                # print("%d sending to %d: %s" % (cp.gid()+1, target.gid+1, o))
                # print("%d sending to %d" % (cp.gid()+1, target.gid+1))
            cp.enqueue(target, o)

    def recv(self, cp):
        link = cp.link()
        for i in range(len(link)):
            tgid = link.target(i).gid
            o = np.array(cp.dequeue(tgid))
            if tgid > cp.gid():  # target is to the right of current subdomain; receive constraint for right end point
                nc = int(o[0]) if len(o) > 1 else 0
                nk = int(o[1]) if len(o) > 1 else 0
                self.rightconstraint = np.array(o[2]) if nc else np.zeros(1)
                self.rightconstraintKnots = np.array(o[3]) if nk else np.zeros(1)
            else:
                nc = int(o[0]) if len(o) > 1 else 0
                nk = int(o[1]) if len(o) > 1 else 0
                self.leftconstraint = np.array(o[2]) if nc else np.zeros(1)
                self.leftconstraintKnots = np.array(o[3]) if nk else np.zeros(1)
            # print("%d received from %d: %s" % (cp.gid()+1, tgid+1, o))
            # print("%d received from %d" % (cp.gid()+1, tgid+1))

    def lsqFit(self, yl):
        print('lsqfit: ', self.RN.shape, yl.shape)
        # print('lsqfit: ', t, np.sum(N*W, axis=1)[:, np.newaxis].T)

        useLSQSolver = True
        if useLSQSolver:
            return linalg.lstsq(self.RN, yl)[0]
        else:
            LHS = np.matmul(self.RN.T, RN)
            RHS = np.matmul(self.RN.T, yl)
            # print('LSQFIT: ', LHS.shape, RHS.shape)
            lu, piv = scipy.linalg.lu_factor(LHS)
            return scipy.linalg.lu_solve((lu, piv), RHS)
            # return linalg.lstsq(LHS, RHS)[0]

    # Code for performing Wynn-Epsilon acceleration
    # This is a scalar version that can be applied to each DoF

    def VectorWynnEpsilon(self, sn, r):
        """Perform Wynn Epsilon Convergence Algorithm"""
        r = int(r)
        n = 2 * r + 1
        e = np.zeros(shape=(n + 1, n + 1))

        for i in range(1, n + 1):
            e[i, 1] = sn[i - 1]

        for i in range(3, n + 2):
            for j in range(3, i + 1):
                if abs(e[i - 1, j - 2] - e[i - 2, j - 2]) > 1e-10:
                    e[i - 1, j - 1] = e[i - 2, j - 3] + 1 / (e[i - 1, j - 2] - e[i - 2, j - 2])
                else:
                    e[i - 1, j - 1] = e[i - 2, j - 3]  # + 1 / (e[i - 1, j - 2] - e[i - 2, j - 2])

        er = e[:, 1:n + 1:2]
        return er

    def WynnEpsilon(self, sn, r):
        """Perform Wynn Epsilon Convergence Algorithm"""
        """https://github.com/pjlohr/WynnEpsilon/blob/master/wynnpi.py"""
        r = int(r)
        n = 2 * r + 1
        e = np.zeros(shape=(n + 1, n + 1))

        for i in range(1, n + 1):
            e[i, 1] = sn[i - 1]

        for i in range(2, n + 1):
            for j in range(2, i):
                e[i, j] = e[i - 1, j - 2] + 1.0 / (e[i, j - 1] - e[i - 1, j - 1])

        er = e[:, 1:n + 1:2]
        return er

    def VectorAitken(self, sn):
        v1 = sn[:, 1] - sn[:, 0]
        v1mag = (np.linalg.norm(v1, ord=2))
        v2 = sn[:, 0] - 2 * sn[:, 1] + sn[:, 2]
        v2mag = (np.linalg.norm(v2, ord=2))
        return sn[:, 0] - (v1mag/v2mag) * v2

    def Aitken(self, sn):
        # return sn[2]
        # return sn[0] - (sn[2] - sn[1])**2/(sn[2] - 2*sn[1] + sn[0])
        if abs(sn[2] - 2*sn[1] + sn[0]) > 1e-8:
            return sn[0] - (sn[2] - sn[1])**2/(sn[2] - 2*sn[1] + sn[0])
        else:
            return sn[2]

    def extrapolate_guess(self, cp, iterationNumber):
        self.pAdaptiveHistory[:, :-1] = self.pAdaptiveHistory[:, 1:]
        self.pAdaptiveHistory[:, -1] = self.pAdaptive[:]

        vAcc = []
        if not useAitken:
            if iterationNumber > 3:  # For Wynn-E[silon
                vAcc = np.zeros(self.pAdaptive.shape)
                for dofIndex in range(len(self.pAdaptive)):
                    expVal = self.WynnEpsilon(
                        self.pAdaptiveHistory[dofIndex, :],
                        math.floor((self.pAdaptiveHistory.shape[1] - 1) / 2))
                    vAcc[dofIndex] = expVal[-1, -1]
                print('Performing scalar Wynn-Epsilon algorithm: Error is ',
                      np.linalg.norm(self.pAdaptive - vAcc), (self.pAdaptive - vAcc))
                self.pAdaptive = vAcc[:]

        else:
            if iterationNumber > 3:  # For Aitken acceleration
                vAcc = self.VectorAitken(self.pAdaptiveHistory)
                # vAcc = np.zeros(self.pAdaptive.shape)
                # for dofIndex in range(len(self.pAdaptive)):
                #     vAcc[dofIndex] = self.Aitken(self.pAdaptiveHistory[dofIndex, :])
                print('Performing Aitken Acceleration algorithm: Error is ', np.linalg.norm(self.pAdaptive - vAcc))
                self.pAdaptive = vAcc[:]

        # Update the controil point vector
        self.update_error_metrics(cp)

    def NonlinearOptimize(
            self, idom, N, W, ysl, U, t, degree, nSubDomains, constraintsAll=None):

        globalIterationNum = 0
        constraints = None
        RN = (N*W)/(np.sum(N*W, axis=1)[:, np.newaxis])
        if constraintsAll is not None:
            constraints = np.array(constraintsAll['P'])
            knotsAll = np.array(constraintsAll['T'])
            weightsAll = np.array(constraintsAll['W'])
            # print ('Constraints for Subdom = ', idom, ' is = ', constraints)

            decodedconstraint = RN.dot(constraints[1])
            residual_decodedcons = (decodedconstraint - ysl)  # /yRange
            # print('actual error in input decoded data: ', (residual_decodedcons))

        else:
            print('Constraints are all null. Solving unconstrained.')

        def ComputeL2Error(P):
            E = (RN.dot(P) - ysl)
            return math.sqrt(np.sum(E**2)/len(E))

        def residual_operator_Ab(Pin, verbose=False, vverbose=False):  # checkpoint3

            Aoper = np.matmul(RN.T, RN)
            Brhs = RN.T @ ysl
            # print('Input P = ', Pin, Aoper.shape, Brhs.shape)

            oddDegree = (degree % 2)
            oddDegreeImpose = True

            # num_constraints = (degree)/2 if degree is even
            # num_constraints = (degree+1)/2 if degree is odd
            nconstraints = augmentSpanSpace + (int(degree/2.0) if not oddDegree else int((degree+1)/2.0))
            # if oddDegree and not oddDegreeImpose:
            #     nconstraints -= 1
            # nconstraints = degree-1
            print('nconstraints: ', nconstraints)

            residual_constrained_nrm = 0
            nBndOverlap = 0
            if constraints is not None and len(constraints) > 0:
                if idom > 1:  # left constraint

                    loffset = -2*augmentSpanSpace if oddDegree else -2*augmentSpanSpace

                    if oddDegree and not oddDegreeImpose:
                        print('left: ', nconstraints, -degree+nconstraints+loffset,
                              Pin[nconstraints], constraints[0][-degree+nconstraints+loffset])
                        constraintVal = 0.5 * (Pin[nconstraints] - constraints[0][-degree+nconstraints+loffset])
                        Brhs -= constraintVal * Aoper[:, nconstraints]

                    for ic in range(nconstraints):
                        Brhs[ic] = 0.5 * (Pin[ic] + constraints[0][-degree+ic+loffset])
                        # Brhs[ic] = constraints[0][-degree+ic]
                        Aoper[ic, :] = 0.0
                        Aoper[ic, ic] = 1.0

                if idom < nSubDomains:  # right constraint

                    loffset = 2*augmentSpanSpace if oddDegree else 2*augmentSpanSpace

                    if oddDegree and not oddDegreeImpose:
                        print('right: ', -nconstraints-1, degree-1-nconstraints+loffset,
                              Pin[-nconstraints-1], constraints[2]
                              [degree-1-nconstraints+loffset])
                        constraintVal = -0.5 * (Pin[-nconstraints-1] - constraints[2]
                                                [degree-1-nconstraints+loffset])
                        Brhs -= constraintVal * Aoper[:, -nconstraints-1]

                    for ic in range(nconstraints):
                        Brhs[-ic-1] = 0.5 * (Pin[-ic-1] + constraints[2][degree-1-ic+loffset])
                        # Brhs[-ic-1] = constraints[2][degree-1-ic]
                        Aoper[-ic-1, :] = 0.0
                        Aoper[-ic-1, -ic-1] = 1.0

            # print(Aoper, Brhs)
            return [Aoper, Brhs]

        # Use previous iterate as initial solution
        initSol = constraints[1][:] if constraints is not None else np.ones_like(W)
        # initSol = np.ones_like(W)*0

        if enforceBounds:
            bnds = np.tensordot(np.ones(initSol.shape[0]), np.array([initSol.min(), initSol.max()]), axes=0)
            # bnds = None
        else:
            bnds = None

        [Aoper, Brhs] = residual_operator_Ab(constraints[1][:], False, False)

        lu, piv = scipy.linalg.lu_factor(Aoper)
        # print(lu, piv)
        initSol = scipy.linalg.lu_solve((lu, piv), Brhs)

        residual_nrm_vec = Brhs - Aoper @ initSol
        residual_nrm = np.linalg.norm(residual_nrm_vec, ord=2)

        return initSol

    def compute_basis(self, u, degree, T, W):
        self.B = sp.BSplineBasis(order=degree+1, knots=T)
        N = np.array(self.B.evaluate(u))

        RN = (N*W)/(np.sum(N*W, axis=1)[:, np.newaxis])

        return N, RN

    def decode(self, P):
        return (self.RN @ P)   # self.RN.dot(P)

    def adaptive(
            self, iSubDom, interface_constraints_obj, strategy='reset', r=1, MAX_ERR=1e-2, MAX_ITER=5,
            split_all=True):

        splitIndeces = []
        r = min(r, degree)  # multiplicity can not be larger than degree

        # Subdomain ID: iSubDom = cp.gid()+1
        u = np.linspace(self.xl[0], self.xl[-1], self.nPointsPerSubD)

        T = interface_constraints_obj['T'][1]
        if len(interface_constraints_obj['P']):
            P = interface_constraints_obj['P'][1]
            W = interface_constraints_obj['W'][1]

        if np.linalg.norm(P, 1) == 0:
            W = np.ones(len(T) - 1 - degree)
            print(np.min(u), np.max(u), np.min(T), np.max(T))
            N, self.RN = self.compute_basis(u, degree, T, W)
            # print('After basis: ', N.shape, self.RN.shape)

            # k = splinelab.augknt(knots, p)  # add endpoint repeats as appropriate for spline order p
            # B = bspline.Bspline(T, degree)       # create spline basis of order p on knots k
            # N = B.collmat(u)

            print('Size of U: ', u.shape, x.shape)
            P = self.lsqFit(self.yl)

            # print('Control points: ', P)
            decodedconstraint = self.decode(P)
            residual_decodedcons = (decodedconstraint - self.yl)  # /yRange
            MAX_ITER = 0

        for iteration in range(MAX_ITER):
            E = (self.decode(P) - self.yl)[self.corebounds[0]:self.corebounds[1]]/yRange

            L2Err = np.sqrt(np.sum(E**2)/len(E))

            if rank == 0 and not disableAdaptivity:
                print(" -- Adaptive iteration: ", iteration, ': Error = ', L2Err)
            if disableAdaptivity:
                Tnew = np.copy(T)
            else:
                Tnew, splitIndeces = knotRefine(P, W, T, u, degree, self.yl, E, r, MAX_ERR=MAX_ERR, find_all=split_all)
                if ((len(T) == len(Tnew)) or len(T)-degree-1 > self.nPointsPerSubD) and not (iteration == 0):
                    break

            if strategy == 'extend' and ~split_all:  # only use when coupled with a solver
                k = splitIndeces[0]
                u = Tnew[k+1]
                P, W = deCasteljau(P, W, T, u, k, r)
            elif strategy == 'reset':
                Tnew = np.sort(Tnew)
                W = np.ones(len(Tnew) - 1 - degree)

                N, self.RN = self.compute_basis(u, degree, Tnew, W)

                if len(interface_constraints_obj['P'][1]) > 0:

                    Pnew = P[:]

                    interface_constraints_obj['P'][1] = Pnew[:]
                    interface_constraints_obj['W'][1] = W[:]
                    interface_constraints_obj['T'][1] = Tnew[:]

                    # print('Solving the boundary-constrained LSQ problem')
                    P = self.NonlinearOptimize(iSubDom, N, W, self.yl, u, Tnew, degree, nSubDomains,
                                               interface_constraints_obj)

                else:
                    print('Solving the unconstrained LSQ problem')
                    P = self.lsqFit(yl)

            else:
                print("Not Implemented!!")

            T = Tnew

        return P, W, T

    def initialize_data(self, cp):

        # Subdomain ID: iSubDom = cp.gid()+1
        domStart = self.xl[self.corebounds[0]]  # /(Dmax-Dmin)
        domEnd = self.xl[self.corebounds[1]]  # /(Dmax-Dmin)
        # domStart = self.xl[0]  # /(Dmax-Dmin)
        # domEnd = self.xl[-1]  # /(Dmax-Dmin)

        # U = np.linspace(domStart, domEnd, self.nPointsPerSubD)

        inc = (domEnd - domStart) / self.nInternalKnotSpans
        # print('data: ', inc, self.nInternalKnotSpans)
        self.knotsAdaptive = np.linspace(domStart + inc, domEnd - inc, self.nInternalKnotSpans - 1)
        if nSubDomains > 1 and not fullyPinned:
            if cp.gid() == 0:
                # knots = np.concatenate(([domStart] * (degree+1), knots, [domEnd], [domEnd+inc]))
                self.knotsAdaptive = np.concatenate(([domStart] * (degree+1), self.knotsAdaptive, [domEnd]))
                self.leftclamped = True
            elif cp.gid() == nSubDomains-1:
                # knots = np.concatenate(([domStart-inc], [domStart], knots, [domEnd] * (degree+1)))
                self.knotsAdaptive = np.concatenate(([domStart], self.knotsAdaptive, [domEnd] * (degree+1)))
                self.rightclamped = True
            else:
                # knots = np.concatenate(([domStart-inc], [domStart], knots, [domEnd], [domEnd+inc]))
                self.knotsAdaptive = np.concatenate(([domStart], self.knotsAdaptive, [domEnd]))
                self.leftclamped = self.rightclamped = False
        else:
            self.knotsAdaptive = np.concatenate(([domStart] * (degree+1), self.knotsAdaptive, [domEnd] * (degree+1)))
            self.leftclamped = self.rightclamped = True

        # print("Subdomain: ", cp.gid(), self.corebounds, domStart, domEnd, " knots = ", self.knotsAdaptive)

        self.pAdaptive = np.zeros(self.nControlPoints)
        self.WAdaptive = np.ones(self.nControlPoints)

    def augment_spans(self, cp):

        # print("Subdomain -- ", cp.gid()+1, ": before Shapes: ",
        #       self.pAdaptive.shape, self.WAdaptive.shape, self.knotsAdaptive)
        if not self.leftclamped:  # Pad knot spans from the left of subdomain
            if degree % 2 == 0:
                self.knotsAdaptive = np.concatenate(
                    (self.leftconstraintKnots[-degree-1-augmentSpanSpace:-1], self.knotsAdaptive))
            else:
                self.knotsAdaptive = np.concatenate(
                    (self.leftconstraintKnots[-degree-1-augmentSpanSpace:-1], self.knotsAdaptive))

        if not self.rightclamped:  # Pad knot spans from the right of subdomain
            if degree % 2 == 0:
                self.knotsAdaptive = np.concatenate(
                    (self.knotsAdaptive, self.rightconstraintKnots[1:degree+1+augmentSpanSpace]))
            else:
                self.knotsAdaptive = np.concatenate(
                    (self.knotsAdaptive, self.rightconstraintKnots[1:degree+1+augmentSpanSpace]))

        # print("Subdomain -- ", cp.gid()+1, ": after Shapes: ",
        #       self.pAdaptive.shape, self.WAdaptive.shape, self.knotsAdaptive)

    def augment_inputdata(self, cp):

        verbose = False
        indices = np.where(np.logical_and(x >= self.knotsAdaptive[degree], x <= self.knotsAdaptive[-degree-1]))

        if verbose:
            print("Subdomain -- {0}: before augment -- left = {1}, right = {2}, shape = {3}".format(cp.gid()+1,
                                                                                                    self.xl[0], self.xl[-1], self.xl.shape))

        self.xl = x[indices]
        self.yl = y[indices]
        self.nPointsPerSubD = self.xl.shape[0]  # int(nPoints / nSubDomains) + overlapData

        h = (xmax-xmin)/nSubDomains
        # Store the core indices before augment
        cindices = np.array(np.where(np.logical_and(self.xl >= xmin + cp.gid()
                                                    * h-1e-8, self.xl <= xmin + (cp.gid()+1)*h+1e-8)))
        # print('cindices: ', cindices, self.xl,
        #       cp.gid(), h, xmin + cp.gid()*h-1e-8, xmin + (cp.gid()+1)*h+1e-8)
        self.corebounds = [cindices[0][0], cindices[0][-1]]

        # print('Corebounds:', cindices[0][0], cindices[-1][-1])

        if verbose:
            print("Subdomain -- {0}: after augment -- left = {1}, right = {2}, shape = {3}".format(cp.gid()+1,
                                                                                                   x[indices[0]], x[indices[-1]], self.xl.shape))

        # print("Subdomain -- {0}: cindices -- {1} {2}, original x bounds = {3} {4}".format(cp.gid()+1,
        #                                                                                   self.xl[self.corebounds[0]], self.xl[self.corebounds[1]], self.xl[0], self.xl[-1]))
        self.decodedAdaptive = np.zeros(self.yl.shape)
        self.decodedAdaptiveOld = np.zeros(self.yl.shape)

    def update_error_metrics(self, cp):
        # errorMAK = L2LinfErrors(self.pAdaptive, self.WAdaptive, self.yl, U, self.knotsAdaptive, degree)
        E = (self.yl[self.corebounds[0]: self.corebounds[1]] - self.decodedAdaptive
             [self.corebounds[0]: self.corebounds[1]]) / yRange
        L2Err = np.sqrt(np.sum(E**2)/len(E))
        LinfErr = np.linalg.norm(E, ord=np.inf)

        self.decodederrors[0] = L2Err
        self.decodederrors[1] = LinfErr

        print("Subdomain -- {0}, L2 error: {1}, Linf error = {2}".format(cp.gid()+1, L2Err, LinfErr))

    def solve_adaptive(self, cp):

        global isConverged
        if isConverged[cp.gid()] == 1:
            print(cp.gid()+1, ' subdomain has already converged to its solution. Skipping solve ...')
            return

        newSolve = False
        if len(self.pAdaptive) == 0:
            newSolve = True

        knots = self.knotsAdaptive[:]
        popt = self.pAdaptive[:]
        W = self.WAdaptive[:]
        nControlPoints = len(popt)

        self.nControlPointSpans = self.nControlPoints - 1
        self.nInternalKnotSpans = self.nControlPointSpans - degree + 1

        if rank == 0 and not disableAdaptivity:
            print("\nSubdomain -- ", cp.gid()+1, "starting adaptive solver ...")

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
                leftconstraint_projected, leftconstraint_projected_knots = self.interpolate(
                    self.leftconstraint, self.leftconstraintKnots, knots)
            if len(self.rightconstraint) > 0:
                rightconstraint_projected, rightconstraint_projected_knots = self.interpolate(
                    self.rightconstraint, self.rightconstraintKnots, knots)

        self.interface_constraints_obj['P'][0] = leftconstraint_projected[:]
        self.interface_constraints_obj['T'][0] = leftconstraint_projected_knots[:]
        self.interface_constraints_obj['P'][2] = rightconstraint_projected[:]
        self.interface_constraints_obj['T'][2] = rightconstraint_projected_knots[:]

        if disableAdaptivity:
            nmaxAdaptIter = 1
        else:
            nmaxAdaptIter = 3

        # xSD = U * (Dmax - Dmin) + Dmin

        # Invoke the adaptive fitting routine for this subdomain
        self.pAdaptive, self.WAdaptive, self.knotsAdaptive = self.adaptive(cp.gid()+1, self.interface_constraints_obj,
                                                                           # self.xSD, self.ySD,
                                                                           # MAX_ERR=maxAdaptErr,
                                                                           MAX_ERR=maxAbsErr,
                                                                           split_all=True,
                                                                           strategy=AdaptiveStrategy,
                                                                           r=1, MAX_ITER=nmaxAdaptIter)

        # NAdaptive = basis(U[np.newaxis,:],degree,knotsAdaptive[:,np.newaxis]).T
        # NAdaptive, self.RN = self.compute_basis(U, degree, self.knotsAdaptive, self.WAdaptive)

        # E = Error(pAdaptive, WAdaptive, ySD, U, knotsAdaptive, degree)
        # print ("Sum of squared error:", np.sum(E**2))
        # print ("Normalized max error:", np.abs(E).max()/yRange)

        if len(self.pAdaptiveHistory) == 0:
            if useAitken:
                self.pAdaptiveHistory = np.zeros((len(self.pAdaptive), 3))
            else:
                self.pAdaptiveHistory = np.zeros((len(self.pAdaptive), nWynnEWork))

        # Update the local decoded data
        self.decodedAdaptiveOld = np.copy(self.decodedAdaptive)
        self.decodedAdaptive = self.decode(self.pAdaptive)

#        print('Local decoded data for Subdomain: ', cp.gid(), self.xl.shape, self.decodedAdaptive.shape, self.xl, self.decodedAdaptive)

        self.update_error_metrics(cp)

#     return PAdaptDomain, WAdaptDomain, KnotAdaptDomains

    def check_convergence(self, cp):

        global isConverged, L2err
        if len(self.decodedAdaptiveOld):

            # Let us compute the relative change in the solution between current and previous iteration
            iterateChangeVec = (self.decodedAdaptive - self.decodedAdaptiveOld)
            errorMetricsSubDomL2 = np.linalg.norm(iterateChangeVec, ord=2) / yRange
            errorMetricsSubDomLinf = np.linalg.norm(iterateChangeVec, ord=np.inf) / yRange

            self.errorMetricsL2[self.outerIteration] = self.decodederrors[0]
            self.errorMetricsLinf[self.outerIteration] = self.decodederrors[1]

            # print(cp.gid()+1, ' Convergence check: ', errorMetricsSubDomL2, errorMetricsSubDomLinf,
            #                 np.abs(self.errorMetricsLinf[self.outerIteration]-self.errorMetricsLinf[self.outerIteration-1]),
            #                 errorMetricsSubDomLinf < 1e-8 and np.abs(self.errorMetricsL2[self.outerIteration]-self.errorMetricsL2[self.outerIteration-1]) < 1e-10)

            L2err[cp.gid()] = self.decodederrors[0]
            # if errorMetricsSubDomL2 < self.decodederrors[0] and self.errorMetricsLinf[self.outerIteration] < 5e-2 * self.decodederrors[0]:
            if errorMetricsSubDomL2 < 5e-2 * self.decodederrors[0] and errorMetricsSubDomLinf < 5e-2 * self.decodederrors[1]:
                # if errorMetricsSubDomLinf < 1e-10 and np.abs(
                #         self.errorMetricsL2[self.outerIteration] - self.errorMetricsL2[self.outerIteration - 1]) < 1e-12:
                print(
                    'Subdomain ', cp.gid() + 1, ' has converged to its final solution with change in error = ',
                    errorMetricsSubDomLinf)
                isConverged[cp.gid()] = 1

        self.outerIteration += 1


#########

# Routine to recursively add a block and associated data to it
def add_input_control_block(gid, core, bounds, domain, link):
    print("Subdomain %d: " % gid, core, bounds, domain, link)
    minb = bounds.min
    maxb = bounds.max
    xlocal = x[minb[0]:maxb[0]+1]
    ylocal = y[minb[0]:maxb[0]+1]
    # print("Subdomain %d: " % gid, xlocal.shape, ylocal.shape, core, bounds, domain)
    mc.add(gid, InputControlBlock(nControlPoints, core, bounds, xlocal, ylocal), link)

# TODO: If working in parallel with MPI or DIY, do a global reduce here


# print "Initial condition data: ", interface_constraints
errors = np.zeros([nASMIterations+1, 1])  # Store L2, Linf errors as function of iteration

# Let us initialize DIY and setup the problem
share_face = [True]
wrap = [False]
ghosts = [overlapData]

# Initialize DIY
w = diy.mpi.MPIComm()           # world
mc = diy.Master(w)         # master
domain_control = diy.DiscreteBounds([0], [len(x)-1])

d_control = diy.DiscreteDecomposer(1, domain_control, nSubDomains, share_face, wrap, ghosts)
a_control = diy.ContiguousAssigner(nprocs, nSubDomains)

d_control.decompose(rank, a_control, add_input_control_block)

mc.foreach(InputControlBlock.show)

sys.stdout.flush()
commW.Barrier()

#########
start_time = timeit.default_timer()

# Before starting the solve, let us exchange the initial conditions
# including the knot vector locations that need to be used for creating
# padded knot vectors in each subdomain
mc.foreach(InputControlBlock.initialize_data)

mc.foreach(InputControlBlock.send)
mc.exchange(False)
mc.foreach(InputControlBlock.recv)

if not fullyPinned:
    mc.foreach(InputControlBlock.augment_spans)
    mc.foreach(InputControlBlock.augment_inputdata)

for iterIdx in range(nASMIterations):

    if rank == 0:
        print("\n---- Starting %d ASM Iteration  ----" % (iterIdx))

    if iterIdx > 1:
        disableAdaptivity = True

    if iterIdx > 0 and rank == 0:
        print("")
    mc.foreach(InputControlBlock.solve_adaptive)

    if disableAdaptivity:
        # if rank == 0: print("")
        mc.foreach(InputControlBlock.check_convergence)

    isASMConverged = commW.allreduce(np.sum(isConverged), op=MPI.SUM)

    if showplot:
        # Let us plot the initial data
        # if nSubDomains > 5:
        #     plt.rc('axes', prop_cycle=(cycler('color', ['r', 'g', 'b', 'y', 'c']) + cycler('linestyle', ['-','--',':','-.',''])))

        if True or isASMConverged or iterIdx == 0 or iterIdx == nASMIterations-1:
            plt.figure()
            plt.plot(x, y, 'b-', ms=5, label='Input')
            mc.foreach(InputControlBlock.plot)
            plt.legend()

            plt.figure()
            mc.foreach(InputControlBlock.plot_error)
            plt.legend()

            plt.draw()

    # if rank == 0: print('')

    # commW.Barrier()
    sys.stdout.flush()

    if isASMConverged == nSubDomains:
        if rank == 0:
            print("\n\nASM solver converged after %d iterations\n\n" % (iterIdx))
        break
    else:

        if extrapolate:
            mc.foreach(lambda icb, cp: InputControlBlock.extrapolate_guess(icb, cp, iterIdx))

        # Now let us perform send-receive to get the data on the interface boundaries from
        # adjacent nearest-neighbor subdomains
        mc.foreach(InputControlBlock.send)
        mc.exchange(False)
        mc.foreach(InputControlBlock.recv)


elapsed = timeit.default_timer() - start_time

if rank == 0:
    print('')

sys.stdout.flush()
if rank == 0:
    print('Total computational time for solve = ', elapsed)

avgL2err = commW.allreduce(np.sum(L2err[np.nonzero(L2err)]**2), op=MPI.SUM)
avgL2err = np.sqrt(avgL2err/nSubDomains)
maxL2err = commW.allreduce(np.max(np.abs(L2err[np.nonzero(L2err)])), op=MPI.MAX)
minL2err = commW.allreduce(np.min(np.abs(L2err[np.nonzero(L2err)])), op=MPI.MIN)

if rank == 0:
    print("\nError metrics: L2 average = %6.12e, L2 maxima = %6.12e, L2 minima = %6.12e\n" % (avgL2err, maxL2err, minL2err))
    print('')

# np.set_printoptions(formatter={'float': '{: 5.12e}'.format})
# mc.foreach(InputControlBlock.print_error_metrics)

if showplot:
    plt.show()
