# coding: utf-8

# TODO:
# Do not pin the internal subdomain boundary points
# Encode and decode correctly. Same solution always as single subdomain case
# Then start doing overlap of data/control points (?) and do iterative solution
#
import timeit
from numba import jit

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

from scipy import linalg, matrix
from scipy.interpolate import LSQUnivariateSpline, UnivariateSpline, interp1d, Rbf
from scipy.optimize import minimize

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
nControlPoints = 10  # (3*degree + 1) #minimum number of control points
useDecodedResidual = False
overlapData = 0
overlapCP = 0
problem = 0
scale = 1
showplot = True
nASMIterations = 2
# Look at ovRBFPower param below if using useDecodedResidual = True
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

solverMaxIter = 20
globalTolerance = 1e-12

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
    print(sys.argv[0], '-p <problem> -n <nsubdomains> -d <degree> -c <controlpoints> -o <overlapData> -a <nASMIterations>')
    sys.exit(2)


try:
    opts, args = getopt.getopt(argv, "hp:n:d:c:o:a:", [
                               "problem=", "nsubdomains=", "degree=", "controlpoints=", "overlap=", "nasm="])
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
    nPoints = 1001
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

# ------------------------------------

EPS = 1e-14


@jit(nopython=True)  # Set "nopython" mode for best performance, equivalent to @njit
def basis2(u, p, T):
    if p == 0:
        return 1.0 if ((T[:-1] <= u) * (u <= T[1:])) else 0.0
    else:
        return ((u - T[:-p]) / (T[p:] - T[:-p]+EPS))[:-1] * basis(u, p-1, T)[:-1] + ((T[p+1:] - u)/(T[p+1:]-T[1:-p]+EPS)) * basis(u, p-1, T)[1:]


def dbasis(u, p, T): return ((T[:-1] <= u) * (u <= T[1:])).astype(np.float) if p == 0 else (
    p / (T[p:] - T[:-p]+EPS))[:-1] * dbasis(u, p-1, T)[:-1] + (p/(T[p+1:]-T[1:-p]+EPS)) * dbasis(u, p-1, T)[1:]


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
        self.decodedAdaptive = np.zeros(yl.shape)
        self.decodedAdaptiveOld = np.zeros(yl.shape)

        self.RN = []

        self.leftclamped = self.rightclamped = False

        # Convergence related metrics and checks
        self.outerIteration = 0
        self.outerIterationConverged = False
        self.decodederrors = np.zeros(2)  # L2 and Linf
        self.errorMetricsL2 = np.zeros((nASMIterations), dtype='float64')
        self.errorMetricsLinf = np.zeros((nASMIterations), dtype='float64')

    def show(self, cp):
        # print("Rank: %d, Subdomain %d: Bounds = [%d, %d], Core = [%d, %d]" % (w.rank, cp.gid(), self.xbounds.min[0], self.xbounds.max[0]+1, self.corebounds.min[0], self.corebounds.max[0]))
        print(
            "Rank: %d, Subdomain %d: Bounds = [%d, %d], Core = [%d, %d]" %
            (w.rank, cp.gid(),
             self.xbounds.min[0],
             self.xbounds.max[0] + 1, self.corebounds[0],
             self.corebounds[1]))
        # cp.enqueue(diy.BlockID(1, 0), "abc")

    def getLocalControlPoints2(self, knots, k):

        nCtrlPts = len(knots) - k - 1
        # nCtrlPts = len(knots)
        # if self.leftclamped:
        #     nCtrlPts -= k + 1
        # if self.rightclamped and not self.leftclamped:
        #     nCtrlPts -= k + 1
        cx = np.zeros(nCtrlPts)
        for i in range(nCtrlPts):
            tsum = 0
            for j in range(1, k+1) if self.leftclamped else range(1, k+1):
                tsum += knots[i + j]
            cx[i] = float(tsum) / (k)
        return cx

    # Greville points
    def getLocalControlPoints(self, knots, k):

        return np.array(sp.BSplineBasis(order=k+1, knots=knots).greville())
        # return self.getLocalControlPoints2(knots, k)
        if self.leftclamped and self.rightclamped:
            return self.getLocalControlPoints2(knots, k)

        # nCtrlPts = len(knots) - 1 - k
        nCtrlPts = len(knots)
        if self.leftclamped:
            nCtrlPts -= k + 1
        if self.rightclamped and not self.leftclamped:
            nCtrlPts -= k + 1
        cx = np.zeros(nCtrlPts)

        offset = 0
        cx[0] = knots[0]
        if self.leftclamped:
            offset += k + 2
        else:
            offset += 1

        if self.rightclamped:
            cx[1:-1] = knots[offset:-k-2]
            cx[-1] = knots[-1]
        else:
            cx[1:] = knots[offset:]

        return cx

    # coeffs_x = getLocalControlPoints(t, degree)  # * (Dmax - Dmin) + Dmin
    # print('Control point locations: ', coeffs_x, t)

    def plot(self, cp):
        #         print(w.rank, cp.gid(), self.core)
        # self.decodedAdaptive = self.decode(self.pAdaptive)
        coeffs_x = self.getLocalControlPoints(self.knotsAdaptive, degree)  # * (Dmax - Dmin) + Dmin
        print(coeffs_x.shape, self.pAdaptive.shape, coeffs_x)
        plt.plot(self.xl, self.decodedAdaptive, linestyle='--', lw=2,
                 color=['r', 'g', 'b', 'y', 'c'][cp.gid() % 5], label="Decoded-%d" % (cp.gid()+1))
        plt.plot(coeffs_x, self.pAdaptive, marker='o', linestyle='', color=[
                 'r', 'g', 'b', 'y', 'c'][cp.gid() % 5], label="Control-%d" % (cp.gid()+1))

    def plot_error(self, cp):
        error = self.decodedAdaptive - self.yl
        plt.plot(self.xl, error, linestyle='--', color=['r', 'g', 'b', 'y', 'c']
                 [cp.gid() % 5], lw=2, label="Subdomain(%d) Error" % (cp.gid()+1))

    def plot_with_cp(self, cp, cploc, ctrlpts, lgndtitle, indx):
        pMK = self.decode(ctrlpts)
        plt.plot(self.xl, pMK, linestyle='--', color=['g', 'b', 'y', 'c'][indx % 5], lw=3, label=lgndtitle)

    def plot_with_cp_and_knots(self, cp, cploc, knots, ctrlpts, weights, lgndtitle, indx):
        print('Plot: shapes = ', ctrlpts.shape[0], cploc.shape[0], knots.shape[0], degree)
        pMK = self.decode(ctrlpts)
        plt.plot(self.xl, pMK, linestyle='--', color=['g', 'b', 'y', 'c'][indx], lw=3, label=lgndtitle)

    def print_error_metrics(self, cp):
        print('Rank:', commW.rank, ' SDom:', cp.gid(), ' L2 Error table: ', self.errorMetricsL2)

    def send(self, cp):
        link = cp.link()
        for i in range(len(link)):
            target = link.target(i)
            o = np.zeros(overlapCP+1)
            if target.gid > cp.gid():  # target is to the right of current subdomain
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
                        o = np.array(
                            [self.pAdaptive.shape[0],
                             self.knotsAdaptive.shape[0],
                             self.pAdaptive[:],
                             self.knotsAdaptive[:]])
                        # o = self.pAdaptive[-1-overlapCP:]
                # print("%d sending to %d: %s" % (cp.gid()+1, target.gid+1, o))
                # print("%d sending to %d" % (cp.gid()+1, target.gid+1))
            else:  # target is to the left of current subdomain
                if len(self.pAdaptive):
                    if useDecodedResidual:
                        # print('Subdomain: ', cp.gid()+1, self.decodedAdaptive)
                        o = self.decodedAdaptive[0:overlapData+1]
                        # if self.corebounds[0] == 0:
                        # o = self.decodedAdaptive[0:overlapData+1]
                        # o = self.decodedAdaptive[self.corebounds[0]-overlapData:overlapData+1]
                    else:
                        o = np.array(
                            [self.pAdaptive.shape[0],
                             self.knotsAdaptive.shape[0],
                             self.pAdaptive[:],
                             self.knotsAdaptive[:]])
                        # o = self.pAdaptive[0:overlapCP+1]
                # print("%d sending to %d: %s" % (cp.gid()+1, target.gid+1, o))
                # print("%d sending to %d" % (cp.gid()+1, target.gid+1))
            cp.enqueue(target, o)

    def recv(self, cp):
        link = cp.link()
        for i in range(len(link)):
            tgid = link.target(i).gid
            o = np.array(cp.dequeue(tgid))
            if tgid > cp.gid():  # target is to the right of current subdomain; receive constraint for right end point
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
            # print("%d received from %d" % (cp.gid()+1, tgid+1))

    def WeightedConstrainedLSQ_PT(
            self, idom, Nall, Wall, ysloc, U, t, degree, nSubDomains, constraintsAll=None, useDerivatives=0,
            solver='SLSQP'):

        if constraintsAll is not None:
            constraints = np.array(constraintsAll['P'])
            knotsAll = np.array(constraintsAll['T'])
            weightsAll = np.array(constraintsAll['W'])
            # print('Constraints for Subdom = ', idom, ' is = ', constraints)
        else:
            print('Constraints are all null. Solving unconstrained.')

        def ComputeL2Error0(P, N, W, ysl, U, t, degree):
            E = np.sum(Error(P, W, ysl, U, t, degree)**2)/len(P)
            return math.sqrt(E)

        def ComputeL2Error(P, N, W, ysl, U, t, degree):
            # RN = (N*W)/(np.sum(N*W, axis=1)[:, np.newaxis])
            E = (self.RN.dot(P) - ysl)
            return math.sqrt(np.sum(E**2)/len(E))

        # print('shapes idom, Nall, Wall, ysloc: ', idom, Nall.shape, Wall.shape, ysloc.shape)
        if useDerivatives > 0 and not useDecodedResidual and constraints is not None and len(constraints) > 0:

            # bzD = np.array(pieceBezierDer22(constraints[1], weightsAll[1], U, knotsAll[1], degree))
            # Bd = bspline.Bspline(knotsAll[1], degree)
            # BdD = Bd.collmat(knotsAll[1][degree:-degree], deriv_order=1)
            Bd = sp.BSplineBasis(order=degree, knots=knotsAll[1])
            BdD = Bdp.evaluate(knotsAll[1][degree:-degree], d=1)
            if idom < nSubDomains:
                # bzDp = np.array(pieceBezierDer22(constraints[2], weightsAll[2], U, knotsAll[2], degree))
                # Bdp = bspline.Bspline(knotsAll[2], degree)
                # BdDp = Bdp.collmat(knotsAll[2][degree:-degree], deriv_order=1)
                Bdp = sp.BSplineBasis(order=degree, knots=knotsAll[2])
                BdDp = Bdp.evaluate(knotsAll[2][degree:-degree], d=1)
                print('2. Just internal knots:', knotsAll[2][degree:-degree])
            if idom > 1:
                # bzDm = np.array(pieceBezierDer22(constraints[0], weightsAll[0], U, knotsAll[0], degree))
                # Bdm = bspline.Bspline(knotsAll[0], degree)
                # BdDm = Bdm.collmat(knotsAll[0][degree:-degree], deriv_order=1)
                Bdm = sp.BSplineBasis(order=degree, knots=knotsAll[0])
                Bdm = Bdm.evaluate(knotsAll[0][degree:-degree], d=1)
                print('0. Just internal knots:', knotsAll[0][degree:-degree])

            if useDerivatives > 1:
                # bzDD = np.array(pieceBezierDer22(bzD, weightsAll[1], U, knotsAll[1][1:-2], degree-1))
                BdDD = Bd.collmat(knotsAll[1][degree+1:-degree-1], deriv_order=2)
                if idom < nSubDomains:
                    # bzDDp = np.array(pieceBezierDer22(bzDp, weightsAll[2], U, knotsAll[2][1:-2], degree-1))
                    BdDDp = Bdp.collmat(knotsAll[2][degree+1:-degree-1], deriv_order=2)
                if idom > 1:
                    # bzDDm = np.array(pieceBezierDer22(bzDm, weightsAll[0], U, knotsAll[0][1:-2], degree-1))
                    BdDDm = Bdm.collmat(knotsAll[0][degree+1:-degree-1], deriv_order=2)

        # Solve unconstrained::
        lenNRows = Nall.shape[0]
        if idom == 1:
            print('Left-most subdomain: right constraints')
            indices = range(lenNRows-1-degree, lenNRows-degree)
            print('Indices:', indices, Wall.shape)
            N = np.delete(Nall, np.s_[indices], axis=0)
            M = np.array(Nall[-1:-2-useDerivatives:-1, :])
            W = Wall[:lenNRows-1]
            ysl = ysloc[:-1]
            T = np.zeros((1+useDerivatives))
            T[0] = 0.5*(constraints[1][-1] + constraints[2][0])
            if useDerivatives > 0:
                M[1, :] += BdD[-1]
                T[1] += np.dot(BdDp[0], constraints[2])
                if useDerivatives > 1:
                    M[2, :] += BdDD[-1]
                    T[2] += np.dot(BdDDp[0], constraints[2])
        elif idom == nSubDomains:
            print('Right-most subdomain: left constraints')
            indices = range(degree, degree+1, 1)
            print('Indices:', indices)
            N = np.delete(Nall, np.s_[indices], axis=0)
            M = np.array(Nall[degree:degree1+useDerivatives, :])
            W = Wall[degree:]
            ysl = ysloc[degree:]
            T = np.zeros((1+useDerivatives))
            T[0] = 0.5*(constraints[1][0] + constraints[0][-1])
            if useDerivatives > 0:
                M[1, :] += BdD[0]
                # print('BdDm[-1]*constraints[0] = ', BdDm[-1].shape, constraints[0].shape)
                T[1] += np.dot(BdDm[-1], constraints[0])
                if useDerivatives > 1:
                    M[2, :] += BdDD[0]
                    T[2] += np.dot(BdDDm[-1], constraints[0])
        else:
            print('Middle subdomain')
            indices1 = range(degree, degree+1, 1)
            indices2 = range(lenNRows-1-degree, lenNRows-degree)
            print('Indices:', indices1, indices2)
            N = np.delete(np.delete(Nall, np.s_[indices2], axis=0), np.s_[indices1], axis=0)
            # M = np.array([Nall[0:1+useDerivatives, :].T, Nall[-1:-2-useDerivatives:-1, :].T])[:, :, 0]
            M = np.zeros((2*(useDerivatives+1), Nall.shape[1]))
            M[0:1+useDerivatives] = Nall[0:1+useDerivatives, :]
            M[1+useDerivatives:2+2*useDerivatives] = Nall[-1:-2-useDerivatives:-1, :]
            W = Wall[1:lenNRows-1]
            ysl = ysloc[1:-1]
            T = np.zeros(2*(1+useDerivatives))
            # left constraints
            T[0] = 0.5*(constraints[1][0] + constraints[0][-1])
            if useDerivatives > 0:
                M[1, :] += BdD[0]
                T[1] += np.dot(BdDm[-1], constraints[0])
                if useDerivatives > 1:
                    M[2, :] += BdDD[0]
                    T[2] = np.dot(BdDDm[-1], constraints[0])
            # right constraints
            T[useDerivatives+1] = 0.5*(constraints[1][-1] + constraints[2][0])
            if useDerivatives > 0:
                M[useDerivatives+2, :] += BdD[-1]
                T[useDerivatives+2] += np.dot(BdDp[0], constraints[2])
                if useDerivatives > 1:
                    M[useDerivatives+3, :] += BdDD[-1]
                    T[useDerivatives+3] += np.dot(BdDDp[0], constraints[2])

        W = Wall[:]

        # Solve the unconstrained solution directly
        RN = (Nall*Wall)/(np.sum(Nall*Wall, axis=1)[:, np.newaxis])
        LHS = np.matmul(RN.T, RN)
        RHS = np.matmul(RN.T, ysloc)
        UnconstrainedLSQSol = linalg.lstsq(LHS, RHS)[0]  # This is P(Uc)

        # LM = inv(M*inv(NT*W*N)*MT) * (M*inv(NT*W*N)*NT*W*S - T)
    #     NTWN = N.T * (W.T * N)
        NW = N * W

        NTWN = np.matmul(N.T, NW)

        # compute the LU factor for N.T * W * N
        LUF, LUP = scipy.linalg.lu_factor(NTWN)

        # srhs is N.T * W * S in (9.75)
        srhs = NW.T @ ysl

        # LMConstraintsA is inv(N.T * W * N) * srhs in (9.75)
        LMConstraintsA = scipy.linalg.lu_solve((LUF, LUP), srhs)
        # LMConstraints is the entire RHS of Equation (9.75)
        # LMConstraints = M * LMConstraintsA - T
        LMConstraints = M @ LMConstraintsA - T

        Alhs = np.matmul(M, scipy.linalg.lu_solve((LUF, LUP), M.T))

        # Equation 9.76 - Piccolo and Tiller
        ALU, ALUP = scipy.linalg.lu_factor(Alhs, overwrite_a=False)
        A = scipy.linalg.lu_solve((ALU, ALUP), LMConstraints.T)

        P2 = M.T @ A
        # print('shapes P2', LMConstraintsA.shape, NTWN.shape, P2.shape)
        # print('Before applying constraints: ', P2)
        P = LMConstraintsA - scipy.linalg.lu_solve((LUF, LUP), P2)
        # print('After applying constraints: ', P)

        print('shapes P', P.shape, P, P-UnconstrainedLSQSol)

    #     return UnconstrainedLSQSol
        return P
    #     return 0.5*(P+UnconstrainedLSQSol)

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

    def lsqFitWithCons(W, ysl, U, t, degree, constraints=[], continuity=0):
        def l2(P, W, ysl, U, t, degree):
            return np.sum(Error(P, W, ysl, U, t, degree)**2)

        res = minimize(l2, np.ones_like(W), method='SLSQP', args=(W, ysl, U, t, degree),
                       constraints=constraints,
                       options={'disp': True})
        return res.x

    def LSQFit_Constrained(
            self, idom, W, ysl, U, t, degree, nSubDomains, constraintsAll=None, useDerivatives=0, solver='SLSQP'):

        # RN = (N*W)/(np.sum(N*W, axis=1)[:, np.newaxis])

        constraints = None
        if constraintsAll is not None:
            constraints = np.array(constraintsAll['P'])
            knotsAll = np.array(constraintsAll['T'])
            weightsAll = np.array(constraintsAll['W'])
            # print ('Constraints for Subdom = ', idom, ' is = ', constraints)
        else:
            print('Constraints are all null. Solving unconstrained.')

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
            E = (self.RN.dot(P) - ysl)/yRange
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
                        cons.append({'type': 'eq', 'fun': lambda x: np.array(
                            [(x[overlapCP] - (constraints[1][overlapCP] + constraints[0][-1-overlapCP])/2)])})
                        if useDerivatives > 0:
                            cons.append({'type': 'eq', 'fun': lambda x: np.array(
                                [(pieceBezierDer22(x, W, U, t, degree)[overlapCP] - (bzD[overlapCP] - bzDm[-1-overlapCP])/2)])})
                            # cons.append( {'type': 'eq', 'fun' : lambda x: np.array([ ( pieceBezierDer22(x, W, U, t, degree)[0] - ( bzDm[-1]  ) ) ])} )
                            # cons.append( {'type': 'eq', 'fun' : lambda x: np.array([ ( (x[1] - x[0])/(knotsAll[idom-1][degree+1]-knotsAll[idom-1][0]) - ( constraints[idom-2][-1] - constraints[idom-2][-2] )/(knotsAll[idom-2][-degree-2] - knotsAll[idom-2][-1]) ) ])} )
                            if useDerivatives > 1:
                                cons.append(
                                    {'type': 'eq', 'fun': lambda x: np.array(
                                        [((x[2 + overlapCP] - x[1 + overlap]) /
                                          (knotsAll[1][degree + 2 + overlapCP] - knotsAll[1][1 + overlapCP]) -
                                          (x[1 + overlapCP] - x[overlapCP]) /
                                          (knotsAll[1][degree + 1 + overlapCP] - knotsAll[1][overlapCP]) -
                                          ((constraints[0][-3 - overlapCP] - constraints[0][-2 - overlapCP]) /
                                            (knotsAll[0][-3 - overlapCP] - knotsAll[0][-degree - 2 - overlapCP]) -
                                            (constraints[0][-2 - overlapCP] - constraints[0][-1 - overlapCP]) /
                                            (knotsAll[0][-2 - overlapCP] - knotsAll[0][-degree - overlapCP - 3])))])})

                            # cons.append( {'type': 'eq', 'fun' : lambda x: np.array([ ( x[1] - x[0] - 0.5*( constraints[idom-1][1] - constraints[idom-1][0] + constraints[idom-2][-1] - constraints[idom-2][-2] ) ) ])} )

                        # print 'Left delx: ', (knotsAll[idom-1][degree+1]-knotsAll[idom-1][0]), ' and ', (knotsAll[idom-1][degree+2]-knotsAll[idom-1][1])
                if idom < nSubDomains:
                    if useDerivatives >= 0:
                        #                     cons.append( {'type': 'eq', 'fun' : lambda x: np.array([ ( x[-1-overlap] - (constraints[2][overlap]) ) ])} )
                        print('Right delx: ', (x[-1-overlapCP]-constraints[2][overlapCP]))
                        cons.append({'type': 'eq', 'fun': lambda x: np.array(
                            [(x[-1-overlapCP] - (constraints[1][-1-overlapCP] + constraints[2][overlapCP])/2)])})
                        if useDerivatives > 0:
                            cons.append({'type': 'eq', 'fun': lambda x: np.array(
                                [(pieceBezierDer22(x, W, U, t, degree)[-1-overlapCP] - (bzD[-1-overlapCP] - bzDp[overlapCP])/2)])})
                            # cons.append( {'type': 'eq', 'fun' : lambda x: np.array([ ( pieceBezierDer22(x, W, U, t, degree)[-1] - (bzDp[0]) ) ])} )
                            # cons.append( {'type': 'eq', 'fun' : lambda x: np.array([ ( (x[-1] - x[-2])/(knotsAll[idom-1][-degree-2] - knotsAll[idom-1][-1]) - ( constraints[idom][1] - constraints[idom][0] )/(knotsAll[idom][degree+1] - knotsAll[idom][0]) ) ])} )
                            if useDerivatives > 1:
                                cons.append(
                                    {'type': 'eq', 'fun': lambda x: np.array(
                                        [((x[-3 - overlapCP] - x[-2 - overlapCP]) /
                                          (knotsAll[1][-2 - overlapCP] - knotsAll[1][-degree - 3 - overlapCP]) -
                                          (x[-2 - overlapCP] - x[-1 - overlapCP]) /
                                          (knotsAll[1][-1 - overlapCP] - knotsAll[1][-degree - 2 - overlapCP]) +
                                          ((constraints[2][1 + overlapCP] - constraints[2][overlapCP]) /
                                            (knotsAll[2][degree + 1 + overlap] - knotsAll[2][overlapCP]) -
                                            (constraints[2][2 + overlapCP] - constraints[2][1 + overlapCP]) /
                                            (knotsAll[2][degree + 2 + overlapCP] - knotsAll[2][1 + overlapCP])))])})
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
                               constraints=cons,  # callback=print_iterate,
                               bounds=bnds,
                               options={'disp': True, 'ftol': 1e-10, 'iprint': 1, 'maxiter': 1000})
            else:

                #             initSol = np.ones_like(W)
                initSol = lsqFit(ysl)
                if enforceBounds:
                    bnds = np.tensordot(np.ones(initSol.shape[0]), np.array([ysl.min(), ysl.max()]), axes=0)
                else:
                    bnds = None

                print('Initial solution from LSQFit: ', initSol)
                res = minimize(ComputeL2Error, x0=initSol, method=solver,  # Nelder-Mead, SLSQP, CG, L-BFGS-B
                               args=(N, W, ysl, U, t, degree),
                               bounds=bnds,
                               options={'disp': True, 'ftol': 1e-10, 'iprint': 1, 'maxiter': 1000})

                print('Final solution from LSQFit: ', res.x)
        else:
            if constraints is not None and len(constraints) > 0:
                if idom > 1:
                    print(idom, ': Left constraint ', (constraints[idom-1]
                                                       [overlap] + constraints[idom-2][-1-overlap])/2)
                    cons.append({'type': 'ineq', 'fun': lambda x: np.array(
                        [(x[overlap] - (constraints[overlap][idom-1] + constraints[-1-overlap][idom-2])/2)])})
                if idom < nSubDomains:
                    print(idom, ': Right constraint ', (constraints[idom-1][-1-overlap] + constraints[idom][overlap])/2)
                    cons.append({'type': 'ineq', 'fun': lambda x: np.array(
                        [(x[-1-overlap] - (constraints[-1-overlap][idom-1] + constraints[overlap][idom])/2)])})

            res = minimize(ComputeL2Error, initSol, method='COBYLA', args=(N, W, ysl, U, t, degree),
                           constraints=cons,  # x0=constraints,
                           options={'disp': False, 'tol': 1e-6, 'catol': 1e-2})

        print('[%d] : %s' % (idom, res.message))
        return res.x

    def NonlinearOptimize(
            self, idom, N, W, ysl, U, t, degree, nSubDomains, constraintsAll=None, useDerivatives=0, solver='L-BFGS-B'):
        import autograd.numpy as np
        from autograd import elementwise_grad as egrad

        globalIterationNum = 0
        constraints = None
        RN = (N*W)/(np.sum(N*W, axis=1)[:, np.newaxis])
        if constraintsAll is not None:
            constraints = np.array(constraintsAll['P'])
            knotsAll = np.array(constraintsAll['T'])
            weightsAll = np.array(constraintsAll['W'])
            # print ('Constraints for Subdom = ', idom, ' is = ', constraints)

            # Update the local decoded data
            if len(constraints) and useDecodedResidual:
                decodedPrevIterate = RN.dot(constraints[1])
    #             decodedPrevIterate = decode(constraints[1])

            decodedconstraint = RN.dot(constraints[1])
            residual_decodedcons = (decodedconstraint - ysl)  # /yRange
            # print('actual error in input decoded data: ', (residual_decodedcons))

        else:
            print('Constraints are all null. Solving unconstrained.')

        if useDerivatives > 0 and not useDecodedResidual and constraints is not None and len(constraints) > 0:

            bzD = np.array(pieceBezierDer22(constraints[1], weightsAll[1], U, knotsAll[1], degree))
            if idom < nSubDomains:
                bzDp = np.array(pieceBezierDer22(constraints[2], weightsAll[2], U, knotsAll[2], degree))
            if idom > 1:
                bzDm = np.array(pieceBezierDer22(constraints[0], weightsAll[0], U, knotsAll[0], degree))

            if useDerivatives > 1:
                bzDD = np.array(pieceBezierDer22(bzD, weightsAll[1], U, knotsAll[1][1:-2], degree-1))
                if idom < nSubDomains:
                    bzDDp = np.array(pieceBezierDer22(bzDp, weightsAll[2], U, knotsAll[2][1:-2], degree-1))
                if idom > 1:
                    bzDDm = np.array(pieceBezierDer22(bzDm, weightsAll[0], U, knotsAll[0][1:-2], degree-1))

        def ComputeL2Error(P):
            E = (RN.dot(P) - ysl)
            return math.sqrt(np.sum(E**2)/len(E))

        def residual(Pin, verbose=False, vverbose=False):  # checkpoint2

            from autograd.numpy import linalg as LA
            decoded_data = RN.dot(Pin)  # RN @ Pin #
    #         decoded_data = decode(Pin)
            residual_decoded = (decoded_data[self.corebounds[0]: self.corebounds[1]] -
                                ysl[self.corebounds[0]: self.corebounds[1]]) / yRange  # decoded_data[0:overlapData+1]
            residual_decoded_nrm = np.sqrt(np.sum(residual_decoded**2)/len(residual_decoded))
            # residual_decoded_nrm = LA.norm(residual_decoded, ord=2)
    #         print('actual decoded data: ', Pin, residual_decoded, residual_decoded_nrm)

            if useDecodedResidual:
                bc_penalty = 1e5
                ovRBFPower = 2.0
                overlapWeight = np.ones(overlapData+1)/(np.power(range(1, overlapData+2), ovRBFPower))
                overlapWeightSum = np.sum(overlapWeight)
            else:
                bc_penalty = 1e12
            residual_constrained_nrm = 0
            nBndOverlap = 0
            # vverbose = True
            if constraints is not None and len(constraints) > 0:
                if idom > 1:  # left constraint
                    if useDecodedResidual:
                        lconstraints = np.copy(constraints[0][:])
                        # lconstraints = np.flip(constraints[0][:])
                        # nBndOverlap += len(lconstraints)
                        nBndOverlap += 1
                        if vverbose:
                            # print('Left decoded delx: ', (decoded_data[0:overlapData+1]), (lconstraints))
                            print(
                                'Left decoded delx: ', (decoded_data[overlapData - 1]),
                                decodedPrevIterate[overlapData - 1],
                                (lconstraints[-1]))
                        # residual_constrained_nrm += np.sum( (decoded_data[0:overlapData+1] - (lconstraints))**2 )

                        # residual_constrained_nrm += np.sum( np.dot( (decoded_data[0:overlapData+1] -
                        #                                     0.5 * ( decodedPrevIterate[0:overlapData+1] + lconstraints ) )**2, overlapWeight) ) / overlapWeightSum
                        residual_constrained_nrm += (decoded_data[overlapData] -
                                                     0.5 * (decodedPrevIterate[overlapData] + lconstraints[-1]))**2
                    else:
                        if useDerivatives >= 0:
                            # print('Left delx: ', (x[overlap]-constraints[0][-1-overlap]))
                            residual_constrained_nrm += bc_penalty * np.power(
                                Pin[0] - (constraints[1][0] + constraints[0][-1])/2, 2.0)
                            if useDerivatives > 0:
                                residual_constrained_nrm += np.power(bc_penalty, 0.5) * np.power(
                                    pieceBezierDer22(Pin, W, U, t, degree)[0] - (bzD[0] + bzDm[-1])/2, 2.0)
                                # residual_constrained_nrm += np.power(bc_penalty, 0.5) * 0.5 * np.power( (knotsAll[1][1]-knotsAll[1][degree+2]) * ( pieceBezierDer22(Pin, W, U, t, degree)[0] - ( bzD[0] + bzDm[-1] )/2), 1.0 )
                                if useDerivatives > 1:
                                    residual_constrained_nrm += np.power(bc_penalty, 0.125) * \
                                        np.power((Pin[2] - Pin[1]) - (Pin[1] - Pin[0]) +
                                                 # np.power( (Pin[2] - Pin[1])/(knotsAll[1][degree+2]-knotsAll[1][1]) - (Pin[1] - Pin[0])/(knotsAll[1][degree+1]-knotsAll[1][0]) +
                                                 ((constraints[0][-3] - constraints[0][-2]) -  # /(knotsAll[0][-3]-knotsAll[0][-degree-2]) -
                                                  (constraints[0][-2] - constraints[0][-1])),  # /(knotsAll[0][-2]-knotsAll[0][-degree-3])  ),
                                                 2.0)

                if idom < nSubDomains:  # right constraint
                    if useDecodedResidual:
                        # rconstraints = np.copy(constraints[2][:])
                        rconstraints = np.flip(constraints[2][:])
                        # nBndOverlap += len(rconstraints)
                        nBndOverlap += 1
                        if vverbose:
                            # print('Right decoded delx: ', decoded_data[-1-overlapData:], rconstraints)
                            print(
                                'Right decoded delx: ', decoded_data[-1 - overlapData],
                                decodedPrevIterate[-1 - overlapData],
                                rconstraints[0])
                        # residual_constrained_nrm += np.sum( (decoded_data[-1-overlapData:] - (rconstraints))**2 )

                        # residual_constrained_nrm += np.sum( np.dot( (decoded_data[-1-overlapData:] -
                        #                                     0.5 * ( decodedPrevIterate[-1-overlapData:] + rconstraints ) )**2, overlapWeight) ) / overlapWeightSum
                        residual_constrained_nrm += np.sum(
                            (decoded_data[-1 - overlapData] - 0.5 *
                             (decodedPrevIterate[-1 - overlapData] + rconstraints[0])) ** 2)
                    else:
                        if useDerivatives >= 0:
                            #                         print('Right delx: ', (x[-1-overlap]-constraints[2][overlap]))
                            residual_constrained_nrm += bc_penalty * np.power(
                                Pin[-1] - (constraints[1][-1] + constraints[2][0])/2, 2.0)
                            if useDerivatives > 0:
                                residual_constrained_nrm += np.power(bc_penalty, 0.5) * np.power(
                                    pieceBezierDer22(Pin, W, U, t, degree)[-1] - (bzD[-1] + bzDp[0])/2, 2.0)
                                # residual_decoded[-1] += np.power(bc_penalty, 0.5) * 0.5 * np.power( (knotsAll[1][-degree-3] - knotsAll[1][-2]) * ( pieceBezierDer22(Pin, W, U, t, degree)[-1] - (bzD[-1] + bzDp[0])/2 ), 1.0)
                                if useDerivatives > 1:
                                    residual_constrained_nrm += np.power(bc_penalty, 0.125) * \
                                        np.power((Pin[-3] - Pin[-2]) - (Pin[-2] - Pin[-1]) +
                                                 # np.power( (Pin[-3] - Pin[-2])/(knotsAll[1][-2] - knotsAll[1][-degree-3]) - (Pin[-2] - Pin[-1])/(knotsAll[1][-1] - knotsAll[1][-degree-2]) +
                                                 ((constraints[2][1] - constraints[2][0]) -  # /(knotsAll[2][degree+1] - knotsAll[2][0]) -
                                                  (constraints[2][2] - constraints[2][1])),  # /(knotsAll[2][degree+2] - knotsAll[2][1]) ),
                                                 2.0)

            if useDecodedResidual:
                residual_constrained_nrm = np.sqrt(residual_constrained_nrm/(nBndOverlap+1))
            else:
                residual_constrained_nrm = np.sqrt(residual_constrained_nrm/(useDerivatives+1))

            residual_nrm = residual_decoded_nrm + residual_constrained_nrm if disableAdaptivity else 0.0

            if verbose:
                print('NLConstrained residual norm: total = ', residual_nrm, 'decoded = ',
                      residual_decoded_nrm, 'boundary-constraints = ', residual_constrained_nrm)

            return residual_nrm

        RNTRN = np.matmul(RN.T, RN)
        QDec = RN.T @ ysl
        # RNTRN = np.copy(RN)
        # QDec = np.copy(ysl)

        def residual_operator_Ab(Pin, verbose=False, vverbose=False):  # checkpoint3

            # Aoper = np.zeros((Pin.shape[0], Pin.shape[0]))
            Aoper = np.copy(RNTRN)
            Brhs = np.copy(QDec)
            print('Input P = ', Pin, Aoper.shape, Brhs.shape)

            # print('Aoper and Brhs: ', Aoper.shape, Brhs.shape)
            # 6 - constraints: 3 - left, 3- right
            # 3 - dofs in the middle

            # 9x9 square matrix = A
            # A {X} = F,

            # Res = F - A * X

            # X = X0 (LSQ), Res0 = F - A X0

            # X = [Xi, Xc]

            # +   -   +
            # 1   2   3

            # 3x3

            #           A
            #       decoded      constraint
            # 1: [ 0: decoded,  [1.0  0  0]    ]
            # 2: [ xx: D[2,:] + [-1.0 2.0 -1.0]    ]
            # 3: [ 0: decoded,  [0   0  1.0]   ]

            #           F
            # 1:  [   constraint_value{1}   ]
            # 2:  [    decoded: NT B [2]    ]
            # 3:  [   constraint_value{3}   ]

            # N P = B

            # NT B = <control point space>

            # NT N P = NT B

            # NT N = D [ 3x3 ]

            # if useDerivatives > 0:
            #     cbzD = np.array(pieceBezierDer22(Pin, W, U, knotsAll[1], degree))
            #     if useDerivatives > 1:
            #         cbzDD = np.array(pieceBezierDer22(cbzD, W, U, knotsAll[1][1:-2], degree-1))

            # num_constraints = (degree)/2 if degree is even
            # num_constraints = (degree+1)/2 if degree is odd
            nconstraints = int(degree/2.0) if (degree % 2 == 0) else int((degree+1)/2.0)
            print('nconstraints: ', nconstraints)

            residual_constrained_nrm = 0
            nBndOverlap = 0
            # vverbose = True
            if constraints is not None and len(constraints) > 0:
                if idom > 1:  # left constraint
                    if useDecodedResidual:
                        lconstraints = np.copy(constraints[0][:])
                        nBndOverlap += 1
                        if vverbose:
                            print(
                                'Left decoded delx: ', (decoded_data[overlapData - 1]),
                                decodedPrevIterate[overlapData - 1],
                                (lconstraints[-1]))
                        Aoper[0, :] = 0.0
                        Aoper[0, 0] = 1.0
                        Brhs[0] = 0.5 * (decodedPrevIterate[overlapData] + lconstraints[-1])
                    else:
                        if useDerivatives >= 0:

                            for ic in range(nconstraints):
                                Brhs[ic] = 0.5 * (Pin[ic] + constraints[0][-degree+ic])
                                # Brhs[ic] = constraints[0][-degree+ic]
                                Aoper[ic, :] = 0.0
                                Aoper[ic, ic] = 1.0

                            # for ic in range(nconstraints):
                            #     constraintVal = constraints[0][-degree+ic]
                            #     print('Constraint: ', ic, constraintVal,
                            #           Brhs[ic], constraintVal * Aoper[:, ic])
                            #     Brhs -= constraintVal * Aoper[:, ic]

                            if useDerivatives > 0 and False:
                                print('Left: ', t[degree+1]-t[1], t[degree+1], t[1],
                                      knotsAll[0][-1] - knotsAll[0][-2 - degree])

                                # Aoper[1, :] = 0.0
                                # Brhs[1] = 0.0
                                Aoper[1, 0] -= degree/(t[degree+1]-t[1])
                                Aoper[1, 1] += degree/(t[degree+1]-t[1])
                                Brhs[1] += 0.5 * bzD[0] - 0.5 * bzDm[-1]
                                # Brhs[1] += bzDm[-1]

                                print("Left oper: ", Aoper[0, :], Aoper[1, :])

                                # Brhs[1] -= (bzDm[-1] - cbzD[0])
                                # Brhs[1] -= 0.5 * bzD[0] + 0.5 * (constraints[0][-1] - constraints[0][-2]) * (
                                #    degree / (t[degree+1]-t[1]))  # (knotsAll[0][-1] - knotsAll[0][-2 - degree]))
                                if useDerivatives > 1 and True:
                                    Aoper[2, 0] += 0.50 * degree*(degree-1)/(t[degree+1]-t[1])**2
                                    Aoper[2, 1] -= 0.75 * degree*(degree-1)/(t[degree+1]-t[1])**2
                                    Aoper[2, 2] += 0.25 * degree*(degree-1)/(t[degree+1]-t[1])**2
                                    Brhs[2] += 0.5*(bzDDm[-1]+bzDD[0])
                                    # Brhs[2] += (bzDDm[-1] - cbzDD[0])
                                    # Brhs[2] -= 0.5 * bzDD[0] + 0.5 * (0.5 * constraints[0][-1] - 0.75 *
                                    #                                   constraints[0][-2] + 0.25 * constraints[0][-3]) * (degree /
                                    #                                                                                      (t[-1]-t[-2-degree])) ** 2

                if idom < nSubDomains:  # right constraint
                    if useDecodedResidual:
                        rconstraints = np.flip(constraints[2][:])
                        nBndOverlap += 1
                        if vverbose:
                            print(
                                'Right decoded delx: ', decoded_data[-1 - overlapData],
                                decodedPrevIterate[-1 - overlapData],
                                rconstraints[0])
                        Aoper[-1, :] = 0.0
                        Aoper[-1, -1] = 1.0
                        Brhs[-1] = 0.5 * (decodedPrevIterate[-1 - overlapData] + rconstraints[0])
                    else:
                        if useDerivatives >= 0:

                            for ic in range(nconstraints):
                                Brhs[-ic-1] = 0.5 * (Pin[-ic-1] + constraints[2][degree-1-ic])
                                # Brhs[-ic-1] = constraints[2][degree-1-ic]
                                Aoper[-ic-1, :] = 0.0
                                Aoper[-ic-1, -ic-1] = 1.0

                            # for ic in range(nconstraints):
                            #     # constraintVal = 0.5 * (Pin[-ic-1] + constraints[2][degree-1-ic])
                            #     constraintVal = constraints[2][degree-1-ic]
                            #     print('Constraint: ', ic, constraintVal,
                            #           Brhs[ic], constraintVal * Aoper[:, -ic-1])
                            #     Brhs -= constraintVal * Aoper[:, -ic-1]

                            if useDerivatives > 0 and False:
                                print('Right: ', t[-1]-t[-2-degree], t[-1], t[-2-degree],
                                      (knotsAll[2][degree + 1] - knotsAll[2][1]))

                                # Aoper[-2, :] = 0.0
                                # Brhs[-2] = 0.0
                                Aoper[-2, -1] -= degree/(t[-1]-t[-2-degree])
                                Aoper[-2, -2] += degree/(t[-1]-t[-2-degree])
                                Brhs[-2] += 0.5 * (bzD[-1] - bzDp[0])
                                # Brhs[-2] += bzDp[0]

                                print("Right oper: ", Aoper[-1, :], Aoper[-2, :])

                                # Brhs[-2] += (bzDp[0] - cbzD[-1])

                                # Brhs[-1] += 0.5 * bzD[-1] + 0.5 * (constraints[2][1] - constraints[2][0]) * (
                                #     degree / (t[-1]-t[-2-degree]))  # (knotsAll[2][degree + 1] - knotsAll[2][1]))
                                if useDerivatives > 1 and True:
                                    Aoper[-3, -1] += 0.50 * degree*(degree-1)/(t[degree+1]-t[1])**2
                                    Aoper[-3, -2] -= 0.75 * degree*(degree-1)/(t[degree+1]-t[1])**2
                                    Aoper[-3, -3] += 0.25 * degree*(degree-1)/(t[degree+1]-t[1])**2
                                    Brhs[-3] -= 0.5*(bzDDp[0]+bzDD[-1])
                                    # Brhs[-3] += (bzDDp[0] - cbzDD[-1])
                                    # Brhs[-3] -= 0.5 * bzDD[-1] + 0.5 * (0.5 * constraints[2][0] - 0.75 *
                                    #                                     constraints[2][1] + 0.25 * constraints[2][2]) * (degree /
                                    #                                                                                      (t[degree + 1] - t[1])) ** 2

            # print(Aoper, Brhs)
            return [Aoper, Brhs]

        def residual_operator(Pin, verbose=False, vverbose=False):  # checkpoint3
            [Aoper, Brhs] = residual_operator_Ab(Pin, verbose, vverbose)
            residual_nrm_vec = Brhs - Aoper @ Pin

            # from autograd.numpy import linalg as LA
            # residual_nrm = LA.norm(residual_nrm_vec, ord=2)

            return residual_nrm_vec

        def residual_operator_norm(Pin, verbose=False, vverbose=False):  # checkpoint3
            [Aoper, Brhs] = residual_operator_Ab(Pin, verbose, vverbose)
            residual_nrm_vec = Brhs - Aoper @ Pin

            from autograd.numpy import linalg as LA
            residual_nrm = LA.norm(residual_nrm_vec, ord=2)

            return residual_nrm

        def residual_vec(Pin, verbose=False, vverbose=False):  # checkpoint2
            # RN : Operatror to project from Control point space to Decoded
            # RN.T : Reverse direction
            from autograd.numpy import linalg as LA
            decoded_data = RN.dot(Pin)  # RN @ Pin #
            # residual_decoded_full = np.power( (decoded_data - ysl)/yRange, 2.0 )
            residual_decoded_full = np.power((decoded_data - ysl)/yRange, 1.0)

            # RN.T * RN * P = RN.T * Y
            # RN.T * (RN * P - Y)
            if not useDecodedResidual:
                residual_decoded = RN.T @ residual_decoded_full
            else:
                residual_decoded = residual_decoded_full[:]

            if useDecodedResidual:
                bc_penalty = 1e0
                ovRBFPower = 0.0
                overlapWeight = np.ones(overlapData+1)/(np.power(range(1, overlapData+2), ovRBFPower))
                overlapWeightSum = np.sum(overlapWeight)
            else:
                bc_penalty = 1e5
            residual_constrained_nrm = 0
            if constraints is not None and len(constraints) > 0:
                if idom > 1:  # left constraint
                    if useDecodedResidual:
                        lconstraints = np.copy(constraints[0][:])
                        # lconstraints = np.flip(constraints[0][:])
                        if vverbose:
                            print('Left decoded delx: ', (decoded_data[0:overlapData+1]), (lconstraints))
                        # residual_decoded += np.sum( (decoded_data[0:overlapData+1] - (lconstraints))**2 )
                        residual_decoded[0: overlapData + 1] += bc_penalty * np.multiply((
                            decoded_data[0: overlapData + 1] - 0.5 *
                            (decodedPrevIterate[0: overlapData + 1] + lconstraints)),
                            overlapWeight) / overlapWeightSum
                    else:
                        if useDerivatives >= 0:
                            residual_decoded[0] += bc_penalty * (Pin[0] - (constraints[1][0] + constraints[0][-1])/2)
                            if useDerivatives > 0:
                                residual_decoded[0] += np.power(bc_penalty, 0.5) * 0.5 * (knotsAll[1][1]-knotsAll[1][degree+2]) * np.power(
                                    pieceBezierDer22(Pin, W, U, t, degree)[0] - (bzD[0] + bzDm[-1])/2, 1.0)
                                if useDerivatives > 1:
                                    residual_decoded[0] += bc_penalty * np.power(
                                        (Pin[2 + overlapCP] - Pin[1 + overlapCP]) /
                                        (knotsAll[1][degree + 2] - knotsAll[1][1]) -
                                        (Pin[1 + overlapCP] - Pin[overlapCP]) /
                                        (knotsAll[1][degree + 1 + overlapCP] - knotsAll[1][overlapCP]) -
                                        ((constraints[0][-3 - overlapCP] - constraints[0][-2 - overlapCP]) /
                                         (knotsAll[0][-3 - overlapCP] - knotsAll[0][-degree - 2 - overlapCP]) -
                                         (constraints[0][-2 - overlapCP] - constraints[0][-1 - overlapCP]) /
                                         (knotsAll[0][-2 - overlapCP] - knotsAll[0][-degree - overlapCP - 3])),
                                        1.0)

                if idom < nSubDomains:  # right constraint
                    if useDecodedResidual:
                        rconstraints = np.copy(constraints[2][:])
                        # rconstraints = np.flip(constraints[2][:])
                        if vverbose:
                            print('Right decoded delx: ', decoded_data[-1-2*overlapData:-overlapData], rconstraints)
                        # residual_decoded += np.sum( (decoded_data[-1-overlapData:] - (rconstraints))**2 )
                        residual_decoded[-1 - 2 * overlapData: -overlapData] += bc_penalty * np.multiply((
                            decoded_data[-1 - 2 * overlapData: -overlapData] - 0.5 *
                            (decodedPrevIterate[-1 - 2 * overlapData: -overlapData] + rconstraints)),
                            overlapWeight) / overlapWeightSum
                    else:
                        if useDerivatives >= 0:
                            residual_decoded[-1] += bc_penalty * (Pin[-1] - (constraints[1][-1] + constraints[2][0])/2)
                            if useDerivatives > 0:
                                residual_decoded[-1] += np.power(bc_penalty, 0.5) * 0.5 * (knotsAll[1][-degree-3] - knotsAll[1][-2]) * np.power(
                                    pieceBezierDer22(Pin, W, U, t, degree)[-1] - (bzD[-1] + bzDp[0])/2, 1.0)
                                if useDerivatives > 1:
                                    residual_decoded[-1] += bc_penalty * np.power((Pin[-3 - overlapCP] -
                                                                                   Pin[-2 - overlapCP]) / (knotsAll[1][-2] - knotsAll[1][-degree - 3]) -
                                                                                  (Pin[-2 - overlapCP] - Pin[-1 - overlapCP]) /
                                                                                  (knotsAll[1][-1 - overlapCP] - knotsAll[1][-degree - 2 - overlapCP]) +
                                                                                  ((constraints[2][1 + overlapCP] - constraints[2][overlapCP]) /
                                                                                   (knotsAll[2][degree + 1 + overlapCP] - knotsAll[2][overlapCP]) -
                                                                                   (constraints[2][2 + overlapCP] - constraints[2][1 + overlapCP]) /
                                                                                   (knotsAll[2][degree + 2 + overlapCP] - knotsAll[2][1 + overlapCP])),
                                                                                  1.0)

            if verbose:
                print('NLConstrained residual vector norm: ', np.sqrt(
                    np.sum(np.abs(residual_decoded))/residual_decoded.shape[0]))

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
                return False
            else:
                print('NLConstrained residual vector norm: ', np.sqrt(
                    np.sum(np.abs(res))/res.shape[0]), ' Boundary: ', res[0], res[-1])
                return True if (np.abs(res[0]) < 1e-12 and np.abs(res[-1]) < 1e-12) else False

        # Use previous iterate as initial solution
        initSol = constraints[1][:] if constraints is not None else np.ones_like(W)
        # initSol = np.ones_like(W)*0

        if enforceBounds:
            bnds = np.tensordot(np.ones(initSol.shape[0]), np.array([initSol.min(), initSol.max()]), axes=0)
            # bnds = None
        else:
            bnds = None

        [Aoper, Brhs] = residual_operator_Ab(constraints[1][:], False, False)

        # print(Aoper)
        lu, piv = scipy.linalg.lu_factor(Aoper)
        # print(lu, piv)
        initSol = scipy.linalg.lu_solve((lu, piv), Brhs)

        return initSol

        if solver in ['Nelder-Mead', 'Powell', 'Newton-CG', 'TNC', 'trust-ncg', 'trust-krylov', 'SLSQP', 'L-BFGS-B', 'CG']:
            minimizer_options = {'disp': False,
                                 'ftol': maxRelErr,
                                 'gtol': globalTolerance,
                                 'maxiter': solverMaxIter
                                 }
            # jacobian_const = egrad(residual)(initSol)
            res = minimize(residual, x0=initSol, method=solver,  # 'SLSQP', #'L-BFGS-B', #'TNC', 'CG', 'Newton-CG'
                           bounds=bnds,
                           # jac=jacobian,
                           callback=print_iterate,
                           tol=globalTolerance,
                           options=minimizer_options)
            print('[%d] : %s' % (idom, res.message))
            return res.x

        elif solver in ['krylov', 'lm']:
            optimizer_options = {'disp': False,
                                 # 'ftol': 1e-5,
                                 'fatol': globalTolerance,
                                 'maxiter': solverMaxIter,
                                 'jac_options': {'method': 'cgs'}  # {‘lgmres’, ‘gmres’, ‘bicgstab’, ‘cgs’, ‘minres’}
                                 }
            # jacobian_const = egrad(residual)(initSol)
            res = scipy.optimize.root(residual_vec, x0=initSol,
                                      method=solver,  # 'krylov', 'lm'
                                      # jac=jacobian_vec,
                                      callback=print_iterate,
                                      tol=globalTolerance,
                                      options=optimizer_options)
            print('[%d] : %s' % (idom, res.message))
            return res.x

        else:

            [Aoper, Brhs] = residual_operator_Ab(constraints[1][:], False, False)

            lu, piv = scipy.linalg.lu_factor(Aoper)
            # print(lu, piv)
            initSol = scipy.linalg.lu_solve((lu, piv), Brhs)
            # initSol = scipy.sparse.linalg.gmres(Aoper, Brhs, x0=initSol, tol=1e-10)[0]
            # print(initSol, Brhs.shape)
            print_iterate(initSol, (Brhs - Aoper @ initSol))

            return initSol

    # def basis(self, u, degree, T):

    #     # print('Basis: knots = ', T)
    #     # print('Basis: evaluation = ', u)

    #     # B = bspline.Bspline(T, degree)       # create spline basis of order p on knots T
    #     # basisOp = B.collmat(u)           # collocation matrix for function value at sites "u"

    #     self.B = sp.BSplineBasis(order=degree+1, knots=T)
    #     basisOp = self.B.evaluate(u)

    #     # print('Basis: at x=0: ', B(0.))
    #     # print('Basis', basisOp)
    #     # print('Knot spans', B.knot_spans())
    #     # print('Greville', B.greville())
    #     return basisOp

    def compute_basis1(self, u, degree, T, W):
        import bspline
        self.B = bspline.Bspline(T, degree)       # create spline basis of order p on knots T

        # N = basis(u[np.newaxis, :], degree, T[:, np.newaxis]).T

        N = self.B.collmat(u)           # collocation matrix for function value at sites "u"

        if self.leftclamped:
            N[0, :] = 0.0
            N[0, 0] = 1.0

        if self.rightclamped:
            N[-1, :] = 0.0
            N[-1, -1] = 1.0

        RN = (N*W)/(np.sum(N*W, axis=1)[:, np.newaxis])

        return N, RN

    def compute_basis2(self, u, degree, T, W):
        self.B = sp.BSplineBasis(order=degree+1, knots=T)
        N = np.array(self.B.evaluate(u))

        RN = (N*W)/(np.sum(N*W, axis=1)[:, np.newaxis])

        # print(N)

        return N, RN

    def compute_basis(self, u, degree, T, W):
        # return self.compute_basis1(u, degree, T, W)
        return self.compute_basis2(u, degree, T, W)

    def decode(self, P):
        return (self.RN @ P)   # self.RN.dot(P)

    def adaptive(
            self, iSubDom, interface_constraints_obj, u, xl, yl, strategy='reset', r=1, MAX_ERR=1e-2, MAX_ITER=5,
            split_all=True):
        splitIndeces = []
        r = min(r, degree)  # multiplicity can not be larger than degree
        nPointsPerSubD = xl.shape[0]

        T = interface_constraints_obj['T'][1]
        if len(interface_constraints_obj['P']):
            P = interface_constraints_obj['P'][1]
            W = interface_constraints_obj['W'][1]

        if np.linalg.norm(P, 1) == 0:
            W = np.ones(len(T) - 1 - degree)

            # print('Before basis: ', u.shape, degree, T.shape, W.shape)
            N, self.RN = self.compute_basis(u, degree, T, W)
            # print('After basis: ', N.shape, self.RN.shape)

            # k = splinelab.augknt(knots, p)  # add endpoint repeats as appropriate for spline order p
            # B = bspline.Bspline(T, degree)       # create spline basis of order p on knots k
            # N = B.collmat(u)

            P = self.lsqFit(yl)
    #         P = LSQFit_Constrained(iSubDom, N, W, yl, u, T, degree, nSubDomains,
    #                        None, #interface_constraints_obj,
    #                        useDerivativeConstraints, 'SLSQP')
    #         P = NonlinearOptimize(iSubDom, N, W, yl, u, T, degree, nSubDomains,
    #                                None, #interface_constraints_obj,
    #                                useDerivativeConstraints, subdomainSolver)

            decodedconstraint = self.decode(P)
            residual_decodedcons = (decodedconstraint - yl)  # /yRange
            MAX_ITER = 0
            # print(iSubDom, ' -- actual error in input decoded data after LSQFit: ', P, (residual_decodedcons))

        for iteration in range(MAX_ITER):
            E = (self.decode(P) - yl)[self.corebounds[0]:self.corebounds[1]]/yRange

            # L2Err = np.linalg.norm(E, ord=2)
            L2Err = np.sqrt(np.sum(E**2)/len(E))

            print(" -- Adaptive iteration: ", iteration, ': Error = ', L2Err)
            if disableAdaptivity:
                Tnew = np.copy(T)
            else:
                Tnew, splitIndeces = knotRefine(P, W, T, u, degree, yl, E, r, MAX_ERR=MAX_ERR, find_all=split_all)
                if ((len(T) == len(Tnew)) or len(T)-degree-1 > nPointsPerSubD) and not (iteration == 0):
                    break

            if strategy == 'extend' and ~split_all:  # only use when coupled with a solver
                k = splitIndeces[0]
                u = Tnew[k+1]
                P, W = deCasteljau(P, W, T, u, k, r)
            elif strategy == 'reset':
                Tnew = np.sort(Tnew)
                W = np.ones(len(Tnew) - 1 - degree)

                # N = basis(u[np.newaxis, :], degree, Tnew[:, np.newaxis]).T
                # print('Before basis: ', u.shape, degree, Tnew.shape, W.shape)
                N, self.RN = self.compute_basis(u, degree, Tnew, W)
                # print('After basis: ', N.shape, self.RN.shape)

    #             P = lsqFit(yl)
    #             return P, W, Tnew

    #             P = lsqFitWithCons(N, W, yl, u, Tnew, degree)

    #             if len(interface_constraints_obj['P'][iSubDom - 1]) > 0 and len(interface_constraints_obj['P'][iSubDom]) > 0 and len(interface_constraints_obj['P'][iSubDom - 2]) > 0:
                if len(interface_constraints_obj['P'][1]) > 0:

                    # # Interpolate or project the data to new Knot locations: From (P, T) to (Pnew, Tnew)
                    # coeffs_x = getControlPoints(T, degree)  # * (Dmax - Dmin) + Dmin
                    # coeffs_xn = getControlPoints(Tnew, degree)  # * (Dmax - Dmin) + Dmin

                    # PnewFn = Rbf(coeffs_x, P, function='cubic')
                    # Pnew = PnewFn(coeffs_xn)

                    Pnew = P[:]

    #                 PnewFn = interp1d(coeffs_x, P, kind='quintic') #, kind='cubic')
    #                 Pnew = PnewFn(coeffs_xn)
                    # print 'coeffs_x = ', [coeffs_x, coeffs_xn, P, Pnew]
                    interface_constraints_obj['P'][1] = Pnew[:]
                    interface_constraints_obj['W'][1] = W[:]
                    interface_constraints_obj['T'][1] = Tnew[:]
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
                    else:  # subdomainSolver == 'NonlinearSolver'
                        P = self.NonlinearOptimize(iSubDom, N, W, yl, u, Tnew, degree, nSubDomains,
                                                   interface_constraints_obj,
                                                   useDerivativeConstraints, subdomainSolver)

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

        U = np.linspace(domStart, domEnd, self.nPointsPerSubD)

        inc = (domEnd - domStart) / self.nInternalKnotSpans
        print('data: ', inc, self.nInternalKnotSpans)
        self.knotsAdaptive = np.linspace(domStart + inc, domEnd - inc, self.nInternalKnotSpans - 1)
        if nSubDomains > 1:
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

        print("Subdomain -- ", cp.gid()+1, ": before Shapes: ",
              self.pAdaptive.shape, self.WAdaptive.shape, self.knotsAdaptive)
        if not self.leftclamped:  # Pad knot spans from the left of subdomain
            self.knotsAdaptive = np.concatenate((self.leftconstraintKnots[-degree-1:-1], self.knotsAdaptive))

        if not self.rightclamped:  # Pad knot spans from the right of subdomain
            self.knotsAdaptive = np.concatenate((self.knotsAdaptive, self.rightconstraintKnots[1:degree+1]))

        print("Subdomain -- ", cp.gid()+1, ": after Shapes: ",
              self.pAdaptive.shape, self.WAdaptive.shape, self.knotsAdaptive)

    def solve_adaptive(self, cp):

        global isConverged
        if isConverged[cp.gid()] == 1:
            print(cp.gid()+1, ' subdomain has already converged to its solution. Skipping solve ...')
            return

        # Subdomain ID: iSubDom = cp.gid()+1
        U = np.linspace(self.xl[0], self.xl[-1], self.nPointsPerSubD)

        newSolve = False
        if len(self.pAdaptive) == 0:
            newSolve = True

        knots = self.knotsAdaptive[:]
        popt = self.pAdaptive[:]
        W = self.WAdaptive[:]
        nControlPoints = len(popt)

        self.nControlPointSpans = self.nControlPoints - 1
        self.nInternalKnotSpans = self.nControlPointSpans - degree + 1

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
        self.pAdaptive, self.WAdaptive, self.knotsAdaptive = self.adaptive(cp.gid()+1, self.interface_constraints_obj, U,
                                                                           # self.xSD, self.ySD,
                                                                           self.xl, self.yl,
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

        # Update the local decoded data
        self.decodedAdaptiveOld = np.copy(self.decodedAdaptive)
        self.decodedAdaptive = self.decode(self.pAdaptive)

#        print('Local decoded data for Subdomain: ', cp.gid(), self.xl.shape, self.decodedAdaptive.shape, self.xl, self.decodedAdaptive)

        # errorMAK = L2LinfErrors(self.pAdaptive, self.WAdaptive, self.yl, U, self.knotsAdaptive, degree)
        E = (self.decodedAdaptive[self.corebounds[0]: self.corebounds[1]] - self.yl
             [self.corebounds[0]: self.corebounds[1]]) / yRange
        LinfErr = np.linalg.norm(E, ord=np.inf)
        L2Err = np.sqrt(np.sum(E**2)/len(E))

        self.decodederrors[0] = L2Err
        self.decodederrors[1] = LinfErr

        print("Subdomain -- ", cp.gid()+1, ": L2 error: ", L2Err, ", Linf error: ", LinfErr)

#     return PAdaptDomain, WAdaptDomain, KnotAdaptDomains

    def check_convergence(self, cp):

        global isConverged, L2err
        if len(self.decodedAdaptiveOld):

            # Let us compute the relative change in the solution between current and previous iteration
            iterateChangeVec = (self.decodedAdaptive - self.decodedAdaptiveOld)
            errorMetricsSubDomL2 = np.linalg.norm(iterateChangeVec, ord=2) / np.linalg.norm(self.pAdaptive, ord=2)
            errorMetricsSubDomLinf = np.linalg.norm(
                iterateChangeVec, ord=np.inf) / np.linalg.norm(self.pAdaptive, ord=np.inf)

            self.errorMetricsL2[self.outerIteration] = self.decodederrors[0]
            self.errorMetricsLinf[self.outerIteration] = self.decodederrors[1]

            # print(cp.gid()+1, ' Convergence check: ', errorMetricsSubDomL2, errorMetricsSubDomLinf,
            #                 np.abs(self.errorMetricsLinf[self.outerIteration]-self.errorMetricsLinf[self.outerIteration-1]),
            #                 errorMetricsSubDomLinf < 1e-8 and np.abs(self.errorMetricsL2[self.outerIteration]-self.errorMetricsL2[self.outerIteration-1]) < 1e-10)

            L2err[cp.gid()] = self.decodederrors[0]
            if errorMetricsSubDomLinf < 1e-10 and np.abs(
                    self.errorMetricsL2[self.outerIteration] - self.errorMetricsL2[self.outerIteration - 1]) < 1e-12:
                print('Subdomain ', cp.gid()+1, ' has converged to its final solution with error = ', errorMetricsSubDomLinf)
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

mc.foreach(InputControlBlock.augment_spans)

for iterIdx in range(nASMIterations):

    if rank == 0:
        print("\n---- Starting ASM Iteration: %d with %s inner solver ----" % (iterIdx, subdomainSolver))

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

np.set_printoptions(formatter={'float': '{: 5.12e}'.format})
mc.foreach(InputControlBlock.print_error_metrics)

if showplot:
    plt.show()
