# coding: utf-8
# get_ipython().magic(u'matplotlib notebook')
# %matplotlib notebook
import sys
import getopt
import math
import timeit

import splipy as sp

# Autograd AD impots
from autograd import elementwise_grad as egrad
import autograd.numpy as np
from autograd.numpy import linalg as LA

# SciPY imports
import scipy
from scipy.linalg import svd
from scipy.optimize import minimize, linprog, root, anderson, newton_krylov  # , BroydenFirst, KrylovJacobian
from scipy import linalg, matrix
from scipy.ndimage import zoom

# MPI imports
from mpi4py import MPI
import diy

# Plotting imports
from matplotlib import pyplot as plt
from matplotlib import cm
from pyevtk.hl import gridToVTK

plt.style.use(['seaborn-whitegrid'])
params = {"ytick.color": "b",
          "xtick.color": "b",
          "axes.labelcolor": "b",
          "axes.edgecolor": "b"}
plt.rcParams.update(params)

# --- set problem input parameters here ---
problem = 1
dimension = 2
degree = 3
nSubDomains = np.array([2] * dimension, dtype=np.uint32)
nSubDomains = [3, 3]
nSubDomainsX = nSubDomains[0]
nSubDomainsY = nSubDomains[1] if dimension > 1 else 1
nSubDomainsZ = nSubDomains[2] if dimension > 2 else 1

verbose = False
showplot = False if dimension > 1 else True
useVTKOutput = True if dimension > 1 else False

augmentSpanSpace = 0
useDiagonalBlocks = False

relEPS = 5e-2
fullyPinned = False
useAdditiveSchwartz = True
enforceBounds = True
alwaysSolveConstrained = False

# ------------------------------------------
# Solver parameters

#                      0      1       2         3          4       5       6      7       8
solverMethods = ['L-BFGS-B', 'CG', 'SLSQP', 'Newton-CG', 'TNC', 'krylov', 'lm', 'trf', 'anderson', 'hybr']
solverScheme = solverMethods[1]
solverMaxIter = 20
nASMIterations = 10
maxAbsErr = 1e-6
maxRelErr = 1e-12

# Solver acceleration
extrapolate = False
useAitken = False
nWynnEWork = 3

##################
# Initialize
Dmin = Dmax = 0
##################

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
        '-p <problem> -n <nsubdomains> -x <nsubdomains_x> -y <nsubdomains_y> -d <degree> -c <controlpoints> -o <overlapData> -a <nASMIterations>')
    sys.exit(2)


try:
    opts, args = getopt.getopt(argv, "hi:p:n:x:y:z:d:c:a:g:s:",
                               ["dimension=", "problem=", "nsubdomains=", "nx=", "ny=", "nz=", "degree=",
                                "controlpoints=", "nasm=", "disableadaptivity", "aug=", "accel", "wynn"])
except getopt.GetoptError:
    usage()

nControlPointsInputIn = 16
# nSubDomainsX = nSubDomainsY = nSubDomainsZ = 1
nPoints = np.array([1, 1, 1], dtype=np.uint32)
Dmin = np.array(3, dtype=np.float)
Dmax = np.array(3, dtype=np.float)

for opt, arg in opts:
    if opt == '-h':
        usage()
    elif opt in ("-i", "--dimension"):
        dimension = int(arg)
    elif opt in ("-n", "--nsubdomains"):
        nSubDomainsX = int(arg)
        nSubDomainsY = int(arg) if dimension > 1 else 1
        nSubDomainsZ = int(arg) if dimension > 2 else 1
    elif opt in ("-x", "--nx"):
        nSubDomainsX = int(arg)
    elif opt in ("-y", "--ny"):
        nSubDomainsY = int(arg)
    elif opt in ("-z", "--nz"):
        nSubDomainsZ = int(arg)
    elif opt in ("-d", "--degree"):
        degree = int(arg)
    elif opt in ("-c", "--controlpoints"):
        nControlPointsInputIn = int(arg)
    elif opt in ("-p", "--problem"):
        problem = int(arg)
    elif opt in ("-a", "--nasm"):
        nASMIterations = int(arg)
    elif opt in ("--disableadaptivity"):
        disableAdaptivity = True
    elif opt in ("-g", "--aug"):
        augmentSpanSpace = int(arg)
    elif opt in ("-s", "--accel"):
        extrapolate = True
    elif opt in ("--wynn"):
        useAitken = False

# nControlPointsInput = nControlPointsInputIn * np.ones((dimension, 1), dtype=np.uint32)
nSubDomains = [nSubDomainsX, nSubDomainsY, nSubDomainsZ]
# -------------------------------------

nTotalSubDomains = nSubDomainsX * nSubDomainsY * nSubDomainsZ
isConverged = np.zeros(nTotalSubDomains, dtype='int32')
L2err = np.zeros(nTotalSubDomains)

# def read_problem_parameters():
xcoord = ycoord = zcoord = None
solution = None
if dimension == 1:
    print('Setting up problem for 1-D')
elif dimension == 2:
    print('Setting up problem for 2-D')
    if problem == 1:
        nPoints[0] = 1025
        nPoints[1] = 1025
        scale = 1
        Dmin = [-4., -4.]
        Dmax = [4., 4.]

        xcoord = np.linspace(Dmin[0], Dmax[0], nPoints[0])
        ycoord = np.linspace(Dmin[1], Dmax[1], nPoints[1])
        X, Y = np.meshgrid(xcoord, ycoord)

        # solution = 100 * (1+0.25*np.tanh((X**2+Y**2)/16)) * (np.sinc(np.sqrt((X-2) ** 2 + (Y+2)**2)) +
        #                                               np.sinc(np.sqrt((X+2)**2 + (Y-2)**2)) -
        #                                               2 * (1-np.tanh((X)**2 + (Y)**2)) +
        #                                               np.exp(-((X-2)**2/2)-((Y-2)**2/2))
        #                                               #   + np.sign(X+Y)
        #                                               )

        # noise = np.random.uniform(0, 0.005, X.shape)
        # solution = solution * (1 + noise)

        # solution = scale * X * Y
        solution = scale * (np.sinc(np.sqrt(X**2 + Y**2)) + np.sinc(2*((X-2)**2 + (Y+2)**2)))
        # solution = scale * (np.sinc((X+1)**2 + (Y-1)**2) + np.sinc(((X-1)**2 + (Y+1)**2)))
        # solution = X**2 * (DmaxX - Y)**2 + X**2 * Y**2 + 64 * (np.sinc(np.sqrt((X-2) ** 2 + (Y+2)**2)))
        # solution = (3-abs(X))**3 * (1-abs(Y))**5 + (1-abs(X))**3 * (3-abs(Y))**5
        # solution = X * Y
        # solution = scale * (np.sinc(X) * np.sinc(Y))
        # (3*degree + 1) #minimum number of control points
        del X, Y

    elif problem == 2:
        nPoints[0] = 501
        nPoints[1] = 501
        scale = 1
        shiftX = 0.25*0
        shiftY = -0.25*0
        Dmin = [0., 0.]
        Dmax = [math.pi, math.pi]

        xcoord = np.linspace(Dmin[0], Dmax[0], nPoints[0])
        ycoord = np.linspace(Dmin[1], Dmax[1], nPoints[1])
        X, Y = np.meshgrid(xcoord+shiftX, ycoord+shiftY)
        # solution = scale * np.sinc(X) * np.sinc(Y)
        solution = scale * np.sin(Y)
        # solution = scale * X * Y
        # (3*degree + 1) #minimum number of control points
        del X, Y

    elif problem == 3:
        z = np.fromfile("data/nek5000.raw", dtype=np.float64).reshape(200, 200)
        print("Nek5000 shape:", z.shape)
        nPoints[0] = z.shape[0]
        nPoints[1] = z.shape[1]
        Dmin = [0, 0]
        Dmax = [100., 100.]

        xcoord = np.linspace(Dmin[0], Dmax[0], nPoints[0])
        ycoord = np.linspace(Dmin[1], Dmax[1], nPoints[1])
        # (3*degree + 1) #minimum number of control points
        if nControlPointsInputIn == 0:
            nControlPointsInputIn = 20

    elif problem == 4:

        binFactor = 4.0
        z = np.fromfile("data/s3d_2D.raw", dtype=np.float64).reshape(540, 704)
        # z = z[:540,:540]
        # z = zoom(z, 1./binFactor, order=4)
        nPoints[0] = z.shape[0]
        nPoints[1] = z.shape[1]
        Dmin = [0, 0]
        Dmax = nPoints
        print("S3D shape:", z.shape)
        xcoord = np.linspace(Dmin[0], Dmax[0], nPoints[0])
        ycoord = np.linspace(Dmin[1], Dmax[1], nPoints[1])

    elif problem == 5:
        z = np.fromfile("data/FLDSC_1_1800_3600.dat", dtype=np.float32).reshape(1800, 3600).T
        nPointsX = z.shape[0]
        nPointsY = z.shape[1]
        DminX = DminY = 0
        DmaxX = 1.0*nPointsX
        DmaxY = 1.0*nPointsY
        print("CESM data shape: ", z.shape)
        xcoord = np.linspace(Dmin[0], Dmax[0], nPoints[0])
        ycoord = np.linspace(Dmin[1], Dmax[1], nPoints[1])

    elif problem == 6:
        # A grid of c-values
        nPointsX = 501
        nPointsY = 501
        scale = 1.0
        shiftX = 0.25
        shiftY = 0.5
        DminX = -2
        DminY = -1.5
        DmaxX = 1
        DmaxY = 1.5

        x = np.linspace(DminX, DmaxX, nPoints[0])
        y = np.linspace(DminY, DmaxY, nPoints[1])
        X, Y = np.meshgrid(x+shiftX, y+shiftY)

        N_max = 255
        some_threshold = 50.0

        # from PIL import Image
        # image = Image.new("RGB", (nPointsX, nPointsY))
        mandelbrot_set = np.zeros((nPoints[0], nPoints[1]))
        for yi in range(nPointsY):
            zy = yi * (DmaxY - DminY) / (nPointsY - 1) + DminY
            y[yi] = zy
            for xi in range(nPoints[0]):
                zx = xi * (DmaxX - DminX) / (nPointsX - 1) + DminX
                x[xi] = zx
                z = zx + zy * 1j
                c = z
                for i in range(N_max):
                    if abs(z) > 2.0:
                        break
                    z = z * z + c
                # image.putpixel((xi, yi), (i % 4 * 64, i % 8 * 32, i % 16 * 16))
                # RGB = (R*65536)+(G*256)+B
                mandelbrot_set[xi, yi] = (i % 4 * 64) * 65536 + (i % 8 * 32) * 256 + (i % 16 * 16)

        # image.show()

        z = mandelbrot_set.T / 1e5

        plt.imshow(z, extent=[DminX, DmaxX, DminY, DmaxY])
        plt.show()

        if nControlPointsInputIn == 0:
            nControlPointsInputIn = 50

    else:
        print('Not a valid problem')
        exit(1)


nControlPointsInput = np.array([nControlPointsInputIn] * dimension, dtype=np.uint32)
nControlPointsInput = np.array([30, 25], dtype=np.uint32)

# if nPointsX % nSubDomainsX > 0 or nPointsY % nSubDomainsY > 0:
#     print ( "[ERROR]: The total number of points do not divide equally with subdomains" )
#     sys.exit(1)

xyzMin = np.array([xcoord.min(), 0, 0])
xyzMax = np.array([xcoord.max(), 0, 0])
if dimension > 1:
    xyzMin[1] = ycoord.min()
    xyzMax[1] = ycoord.max()
    if dimension > 2:
        xyzMin[2] = zcoord.min()
        xyzMax[2] = zcoord.max()

solutionBounds = [solution.min(), solution.max()]
solutionRange = solution.max() - solution.min()


def plot_solution(solVector):
    if dimension == 1:
        x = xcoord[:]
        gridToVTK("./structured", xcoord, np.ones(x.shape[0]), np.ones(x.shape[0]),
                  pointData={"solution": solVector.reshape(1, x.shape[0])})
    elif dimension == 2:
        x, y = np.meshgrid(xcoord, ycoord)
        gridToVTK("./structured", x.reshape(1, x.shape[0], x.shape[1]), y.reshape(1, y.shape[0], y.shape[1]), np.ones(
            x.reshape(1, x.shape[0], x.shape[1]).shape), pointData={"solution": solVector.T.reshape(1, x.shape[1], x.shape[0])})
    elif dimension == 3:
        x, y, z = np.meshgrid(xcoord, ycoord, zcoord)
        gridToVTK(
            "./structured", x.reshape(x.shape[0],
                                      x.shape[1],
                                      x.shape[2]),
            y.reshape(y.shape[0],
                      y.shape[1],
                      y.shape[2]),
            z.reshape(x.shape[0],
                      x.shape[1],
                      x.shape[2]),
            pointData={"solution": solVector.reshape(x.shape[0],
                                                     x.shape[1],
                                                     x.shape[2])})
    else:
        print("No visualization output available for dimension > 2")


# Store the reference solution
plot_solution(solution)

### Print parameter details ###
if rank == 0:
    print('\n==================')
    print('Parameter details')
    print('==================\n')
    print('dimension = ', dimension)
    print('problem = ', problem, '[1 = sinc, 2 = sine, 3 = Nek5000, 4 = S3D, 5 = CESM]')
    print('Total number of input points: ', nPoints[0]*nPoints[1]*nPoints[2])
    print('nSubDomains = ', nSubDomainsX * nSubDomainsY)
    print('degree = ', degree)
    print('nControlPoints = ', nControlPointsInput)
    print('nASMIterations = ', nASMIterations)
    print('augmentSpanSpace = ', augmentSpanSpace)
    print('useAdditiveSchwartz = ', useAdditiveSchwartz)
    print('enforceBounds = ', enforceBounds)
    print('maxAbsErr = ', maxAbsErr)
    print('maxRelErr = ', maxRelErr)
    print('solverMaxIter = ', solverMaxIter)
    print('solverscheme = ', solverScheme)
    print('\n=================\n')

# ------------------------------------

sys.stdout.flush()

# Let us create a parallel VTK file
# @profile


def WritePVTKFile(iteration):
    pvtkfile = open("pstructured-mfa-%d.pvts" % (iteration), "w")

    pvtkfile.write('<?xml version="1.0"?>\n')
    pvtkfile.write('<VTKFile type="PStructuredGrid" version="1.0" byte_order="LittleEndian" header_type="UInt64">\n')
    pvtkfile.write('<PStructuredGrid WholeExtent="%f %f %f %f 0 0" GhostLevel="0">\n' %
                   (xyzMin[0], xyzMax[0], xyzMax[0], xyzMax[1]))
    pvtkfile.write('\n')
    pvtkfile.write('    <PCellData>\n')
    pvtkfile.write('      <PDataArray type="Float64" Name="solution"/>\n')
    pvtkfile.write('      <PDataArray type="Float64" Name="error"/>\n')
    pvtkfile.write('    </PCellData>\n')
    pvtkfile.write('    <PPoints>\n')
    pvtkfile.write('      <PDataArray type="Float64" Name="points" NumberOfComponents="3"/>\n')
    pvtkfile.write('    </PPoints>\n')

    isubd = 0
    # dx = (xyzMax[0]-xyzMin[0])/nSubDomainsX
    # dy = (xyzMax[1]-xyzMin[1])/nSubDomainsY
    dxyz = (xyzMax-xyzMin)/nSubDomains
    xoff = xyzMin[0]
    xx = dxyz[0]
    for ix in range(nSubDomainsX):
        yoff = xyzMin[1]
        yy = dxyz[1]
        for iy in range(nSubDomainsY):
            pvtkfile.write(
                '    <Piece Extent="%f %f %f %f 0 0" Source="structured-%d-%d.vts"/>\n' %
                (xoff, xx, yoff, yy, isubd, iteration))
            isubd += 1
            yoff = yy
            yy += dxyz[1]
        xoff = xx
        xx += dxyz[0]
    pvtkfile.write('\n')
    pvtkfile.write('</PStructuredGrid>\n')
    pvtkfile.write('</VTKFile>\n')

    pvtkfile.close()

# ------------------------------------


EPS = 1e-32
GTOL = 1e-2


def compute_decode_operators(W, iNuvw):
    RN = {'x': [], 'y': [], 'z': []}
    if dimension == 1:
        RN['x'] = iNuvw['x'] * np.sum(W, axis=0)

    elif dimension == 2:
        RN['x'] = iNuvw['x'] * np.sum(W, axis=1)
        RN['x'] /= np.sum(RN['x'], axis=1)[:, np.newaxis]
        RN['y'] = iNuvw['y'] * np.sum(W, axis=0)
        RN['y'] /= np.sum(RN['y'], axis=1)[:, np.newaxis]
        # print('Decode error res: ', RNx.shape, RNy.shape)
        # decoded = np.matmul(np.matmul(RNx, P), RNy.T)

    else:
        error('No implementation')

    return RN


def decode_2D(P, W, iNu, iNv):
    decoded = []
    method = 2
    if method == 1:
        Nu = iNu[..., np.newaxis]
        Nv = iNv[:, np.newaxis]
        NN = []
        for ui in range(Nu.shape[0]):
            for vi in range(Nv.shape[0]):
                NN.append(Nu[ui]*Nv[vi])
        NN = np.array(NN)

        decoded = np.tensordot(NN, P * W) / np.tensordot(NN, W)
        del NN
        decoded = decoded.reshape((Nu.shape[0], Nv.shape[0]))
    else:

        RNx = iNu * np.sum(W, axis=1)
        RNx /= np.sum(RNx, axis=1)[:, np.newaxis]
        RNy = iNv * np.sum(W, axis=0)
        RNy /= np.sum(RNy, axis=1)[:, np.newaxis]
        # print('Decode error res: ', RNx.shape, RNy.shape)
        decoded = np.matmul(np.matmul(RNx, P), RNy.T)
        # print('Decode error res: ', decoded.shape, decoded2.shape, np.max(np.abs(decoded.reshape((Nu.shape[0], Nv.shape[0])) - decoded2)))
        # print('Decode error res: ', z.shape, decoded.shape)

    return decoded


def decode_1D(P, W, Nu):
    RN = Nu * np.sum(W, axis=0)
    return (RN @ P)   # self.RN.dot(P)


def decode(P, W, iNuvw):
    if dimension == 1:
        return decode_1D(P, W, iNuvw['x'])
    elif dimension == 2:
        return decode_2D(P, W, iNuvw['x'], iNuvw['y'])
    else:
        error('Not implemented and invalid dimension')


def lsqFit(Nu, Nv, W, z, use_cho=True):
    RNx = Nu * np.sum(W, axis=1)
    RNx /= np.sum(RNx, axis=1)[:, np.newaxis]
    RNy = Nv * np.sum(W, axis=0)
    RNy /= np.sum(RNy, axis=1)[:, np.newaxis]
    if use_cho:
        X = linalg.cho_solve(linalg.cho_factor(np.matmul(RNx.T, RNx)), RNx.T)
        Y = linalg.cho_solve(linalg.cho_factor(np.matmul(RNy.T, RNy)), RNy.T)
        zY = np.matmul(z, Y.T)
        return np.matmul(X, zY)
    else:
        NTNxInv = np.linalg.inv(np.matmul(RNx.T, RNx))
        NTNyInv = np.linalg.inv(np.matmul(RNy.T, RNy))
        NxTQNy = np.matmul(RNx.T, np.matmul(z, RNy))
        return np.matmul(NTNxInv, np.matmul(NxTQNy, NTNyInv))

# Let do the recursive iterations
# The variable `additiveSchwartz` controls whether we use
# Additive vs Multiplicative Schwartz scheme for DD resolution
####


class InputControlBlock:

    def __init__(self, bid, nCPi, coreb, xb, pl, xl, yl=None, zl=None):
        self.nControlPoints = nCPi[:]
        self.nControlPointSpans = self.nControlPoints - 1
        self.nInternalKnotSpans = self.nControlPoints - degree
        self.xbounds = xb
        if dimension == 1:
            self.corebounds = [coreb.min[0]-xb.min[0], -1+coreb.max[0]-xb.max[0],
                               coreb.min[1]-xb.min[1], -1+coreb.max[1]-xb.max[1]]
            self.nPointsPerSubD = [len(xl)]  # int(nPointsX / nSubDomainsX)
            self.xyzCoordLocal = {'x': xl[:]}
            self.solutionLocal = pl[:]
            self.Dmini = np.array([min(xl)])
            self.Dmaxi = np.array([max(xl)])
            self.basisFunction = {'x': None}  # Basis function object in x-dir
            self.knotsAdaptive = {'x': []}
            self.isClamped = {'left': False, 'right': False}

        elif dimension == 2:
            self.corebounds = [coreb.min[0]-xb.min[0], -1+coreb.max[0]-xb.max[0],
                               coreb.min[1]-xb.min[1], -1+coreb.max[1]-xb.max[1]]
            self.nPointsPerSubD = [len(xl), len(yl)]  # int(nPointsX / nSubDomainsX)
            self.xyzCoordLocal = {'x': xl[:],
                                  'y': yl[:]}
            self.solutionLocal = pl[:, :]
            self.Dmini = np.array([min(xl), min(yl)])
            self.Dmaxi = np.array([max(xl), max(yl)])
            self.basisFunction = {'x': None, 'y': None}  # Basis function object in x-dir and y-dir
            self.knotsAdaptive = {
                'x': [],
                'y': []}
            self.isClamped = {'left': False, 'right': False, 'top': False, 'bottom': False}

        self.controlPointData = np.zeros(self.nControlPoints)
        self.weightsData = np.ones(self.nControlPoints)
        self.controlPointBounds = np.array(solutionBounds, copy=True)

        self.solutionDecoded = np.zeros(pl.shape)
        self.solutionDecodedOld = np.zeros(pl.shape)

        self.UVW = {'x': [], 'y': [], 'z': []}
        self.NUVW = {'x': [], 'y': [], 'z': []}

        # The constraints have a pre-determined ordering
        # 1-Dimension = 0: left, 1: right
        # 2-Dimension = 0: left, 1: right, 2: top, 3: bottom, 4: top-left, 5: top-right, 6: bottom-left, 7: bottom-right
        self.boundaryConstraints = {}
        self.boundaryConstraintKnots = {}
        self.ghostKnots = {}

        self.figHnd = None
        self.figHndErr = None
        self.figSuffix = ""

        # Convergence related metrics and checks
        self.outerIteration = 0
        self.globalIterationNum = 0
        self.adaptiveIterationNum = 0
        self.globalTolerance = 1e-13

        self.outerIterationConverged = False
        self.decodederrors = np.zeros(2)  # L2 and Linf
        self.errorMetricsL2 = np.zeros((nASMIterations), dtype='float64')
        self.errorMetricsLinf = np.zeros((nASMIterations), dtype='float64')

        self.solutionLocalHistory = []

    def show(self, cp):

        if dimension == 1:
            print(
                "Rank: %d, Subdomain %d: Bounds = [%d - %d, %d - %d]" %
                (commWorld.rank, cp.gid(),
                 self.xbounds.min[0],
                 self.xbounds.max[0]))
        elif dimension == 2:
            print(
                "Rank: %d, Subdomain %d: Bounds = [%d - %d, %d - %d]" %
                (commWorld.rank, cp.gid(),
                 self.xbounds.min[0],
                 self.xbounds.max[0],
                 self.xbounds.min[1],
                 self.xbounds.max[1]))
        else:
            error("No implementation")

    def compute_basis_1D(self, degree, knotVectors):
        self.basisFunction['x'] = sp.BSplineBasis(order=degree+1, knots=knotVectors['x'])
        # print("TU = ", knotVectors['x'], self.UVW['x'][0], self.UVW['x'][-1], self.basisFunctio['x'].greville())
        self.NUVW['x'] = np.array(self.basisFunction['x'].evaluate(self.UVW['x']))

    def compute_basis_2D(self, degree, knotVectors):
        self.basisFunction['x'] = sp.BSplineBasis(order=degree+1, knots=knotVectors['x'])
        self.basisFunction['y'] = sp.BSplineBasis(order=degree+1, knots=knotVectors['y'])
        # print("TU = ", knotVectors['x'], self.UVW['x'][0], self.UVW['x'][-1], self.basisFunction['x'].greville())
        # print("TV = ", knotVectors['y'], self.UVW['y'][0], self.UVW['y'][-1], self.basisFunction['y'].greville())
        self.NUVW['x'] = np.array(self.basisFunction['x'].evaluate(self.UVW['x']))
        self.NUVW['y'] = np.array(self.basisFunction['y'].evaluate(self.UVW['y']))

    def compute_basis(self, degree, knotVectors):
        if dimension == 1:
            self.compute_basis_1D(degree, knotVectors)
        elif dimension == 2:
            self.compute_basis_2D(degree, knotVectors)
        else:
            error('Invalid dimension')

    def output_vtk(self, cp):

        if useVTKOutput:

            self.pMK = decode(self.controlPointData, self.weightsData, self.NUVW)
            errorDecoded = (self.solutionLocal - self.pMK) / solutionRange

            Xi, Yi = np.meshgrid(self.xyzCoordLocal['x'], self.xyzCoordLocal['y'])

            Xi = Xi.reshape(1, Xi.shape[0], Xi.shape[1])
            Yi = Yi.reshape(1, Yi.shape[0], Yi.shape[1])
            Zi = np.ones(Xi.shape)
            PmK = self.pMK.T.reshape(1, self.pMK.shape[1], self.pMK.shape[0])
            errorDecoded = errorDecoded.T.reshape(1, errorDecoded.shape[1], errorDecoded.shape[0])
            gridToVTK("./structured-%s" % (self.figSuffix), Xi, Yi, Zi,
                      pointData={"solution": PmK, "error": errorDecoded})

    def set_fig_handles(self, cp, fig=None, figerr=None, suffix=""):
        self.figHnd = fig
        self.figHndErr = figerr
        self.figSuffix = suffix

    def print_solution(self, cp):
        # self.pMK = decode(self.solutionLocal, self.weightsData, self.Nu, self.Nv)
        #         errorDecoded = self.solutionLocal - self.pMK

        print("Domain: ", cp.gid()+1, "Exact = ", self.solutionLocal)
        print("Domain: ", cp.gid()+1, "Exact - Decoded = ", np.abs(self.solutionLocal - self.pMK))

    def send_diy(self, cp):
        verbose = False
        link = cp.link()
        for i in range(len(link)):
            target = link.target(i)
            if len(self.controlPointData):
                dir = link.direction(i)
                if dir[0] == 0 and dir[1] == 0:
                    continue

                # ONLY consider coupling through faces and not through verties
                # This means either dir[0] or dir[1] has to be "0" for subdomain coupling to be active
                # Hence we only consider 4 neighbor cases, instead of 8.
                if dir[0] == 0:  # target is coupled in Y-direction
                    if dir[1] > 0:  # target block is above current subdomain
                        if verbose:
                            print("%d sending to %d" % (cp.gid(), target.gid), ' Top: ',
                                  self.controlPointData[:, -1:-2-degree-augmentSpanSpace:-1].shape)

                        # cp.enqueue(target, self.controlPointData[:, -1-degree-augmentSpanSpace:])
                        cp.enqueue(target, self.controlPointData)
                        cp.enqueue(target, self.knotsAdaptive['x'][:])
                        cp.enqueue(target, self.knotsAdaptive['y'][-1:-2-degree-augmentSpanSpace:-1])

                    else:  # target block is below current subdomain
                        if verbose:
                            print("%d sending to %d" % (cp.gid(), target.gid), ' Bottom: ',
                                  self.controlPointData[:, 0:1+degree+augmentSpanSpace].shape)

                        # cp.enqueue(target, self.controlPointData[:, 0:1+degree+augmentSpanSpace])
                        cp.enqueue(target, self.controlPointData)
                        cp.enqueue(target, self.knotsAdaptive['x'][:])
                        cp.enqueue(target, self.knotsAdaptive['y'][0:1+degree+augmentSpanSpace])

                elif dir[1] == 0:  # target is coupled in X-direction
                    if dir[0] > 0:  # target block is to the right of current subdomain
                        if verbose:
                            print("%d sending to %d" % (cp.gid(), target.gid), 'Left: ',
                                  self.controlPointData[-1:-2-degree-augmentSpanSpace:-1, :].shape)

                        # cp.enqueue(target, self.controlPointData[-1-degree-augmentSpanSpace:, :])
                        cp.enqueue(target, self.controlPointData)
                        cp.enqueue(target, self.knotsAdaptive['y'][:])
                        cp.enqueue(target, self.knotsAdaptive['x'][-1:-2-degree-augmentSpanSpace:-1])

                    else:  # target block is to the left of current subdomain
                        if verbose:
                            print("%d sending to %d" % (cp.gid(), target.gid), 'Right: ',
                                  self.controlPointData[degree+augmentSpanSpace::-1, :].shape)

                        # cp.enqueue(target, self.controlPointData[0:1+degree+augmentSpanSpace, :])
                        cp.enqueue(target, self.controlPointData)
                        cp.enqueue(target, self.knotsAdaptive['y'][:])
                        cp.enqueue(target, self.knotsAdaptive['x'][0:(degree+augmentSpanSpace+1)])

                else:

                    verbose = True
                    if dir[0] > 0 and dir[1] > 0:  # target block is diagonally top right to current subdomain

                        cp.enqueue(target, self.controlPointData)
                        # cp.enqueue(target, self.controlPointData[-1-degree-augmentSpanSpace:, -1-degree-augmentSpanSpace:])
                        if verbose:
                            print("%d sending to %d" % (cp.gid(), target.gid), ' Diagonal = right-top: ',
                                  self.controlPointData[-1:-2-degree-augmentSpanSpace:-1, :1+degree+augmentSpanSpace])
                    if dir[0] < 0 and dir[1] > 0:  # target block is diagonally top left to current subdomain
                        cp.enqueue(target, self.controlPointData)
                        # cp.enqueue(target, self.controlPointData[: 1 + degree + augmentSpanSpace, -1:-2-degree-augmentSpanSpace:-1])
                        if verbose:
                            print("%d sending to %d" % (cp.gid(), target.gid), ' Diagonal = left-top: ',
                                  self.controlPointData[-1:-2-degree-augmentSpanSpace:-1, :1+degree+augmentSpanSpace])

                    if dir[0] < 0 and dir[1] < 0:  # target block is diagonally left bottom  current subdomain
                        cp.enqueue(target, self.controlPointData)
                        # cp.enqueue(target, self.controlPointData[-1-degree-augmentSpanSpace:, :1+degree+augmentSpanSpace])

                        if verbose:
                            print("%d sending to %d" % (cp.gid(), target.gid), ' Diagonal = left-bottom: ',
                                  self.controlPointData[:1+degree+augmentSpanSpace,  -1-degree-augmentSpanSpace:])
                    if dir[0] > 0 and dir[1] < 0:  # target block is diagonally right bottom of current subdomain
                        cp.enqueue(target, self.controlPointData)
                        # cp.enqueue(target, self.controlPointData[:1+degree+augmentSpanSpace, :1+degree+augmentSpanSpace])
                        if verbose:
                            print("%d sending to %d" % (cp.gid(), target.gid), ' Diagonal = right-bottom: ',
                                  self.controlPointData[:1+degree+augmentSpanSpace,  -1 - degree - augmentSpanSpace:])
                    verbose = False

        return

    def recv_diy(self, cp):
        verbose = False
        link = cp.link()
        for i in range(len(link)):
            tgid = link.target(i).gid
            dir = link.direction(i)
            # print("%d received from %d: %s from direction %s, with sizes %d+%d" % (cp.gid(), tgid, o, dir, pl, tl))

            # ONLY consider coupling through faces and not through verties
            # This means either dir[0] or dir[1] has to be "0" for subdomain coupling to be active
            # Hence we only consider 4 neighbor cases, instead of 8.
            if dir[0] == 0 and dir[1] == 0:
                continue

            if dir[0] == 0:  # target is coupled in Y-direction
                if dir[1] > 0:  # target block is above current subdomain
                    self.boundaryConstraints['top'] = cp.dequeue(tgid)
                    self.boundaryConstraintKnots['top'] = cp.dequeue(tgid)
                    self.ghostKnots['top'] = cp.dequeue(tgid)
                    if verbose:
                        print("Top: %d received from %d: from direction %s" %
                              (cp.gid(), tgid, dir), self.topconstraint.shape, self.topconstraintKnots.shape)
                else:  # target block is below current subdomain
                    self.boundaryConstraints['bottom'] = cp.dequeue(tgid)
                    self.boundaryConstraintKnots['bottom'] = cp.dequeue(tgid)
                    self.ghostKnots['bottom'] = cp.dequeue(tgid)
                    if verbose:
                        print("Bottom: %d received from %d: from direction %s" %
                              (cp.gid(), tgid, dir), self.bottomconstraint.shape, self.bottomconstraintKnots.shape)

            elif dir[1] == 0:  # target is coupled in X-direction
                if dir[0] < 0:  # target block is to the left of current subdomain
                    # print('Right: ', np.array(o[2:pl+2]).reshape(useDerivativeConstraints+1,pll).T)

                    self.boundaryConstraints['left'] = cp.dequeue(tgid)
                    self.boundaryConstraintKnots['left'] = cp.dequeue(tgid)
                    self.ghostKnots['left'] = cp.dequeue(tgid)
                    if verbose:
                        print("Left: %d received from %d: from direction %s" %
                              (cp.gid(), tgid, dir), self.leftconstraint.shape, self.leftconstraintKnots.shape)

                else:  # target block is to right of current subdomain

                    self.boundaryConstraints['right'] = cp.dequeue(tgid)
                    self.boundaryConstraintKnots['right'] = cp.dequeue(tgid)
                    self.ghostKnots['right'] = cp.dequeue(tgid)
                    if verbose:
                        print("Right: %d received from %d: from direction %s" %
                              (cp.gid(), tgid, dir), self.rightconstraint.shape, self.rightconstraintKnots.shape)
            else:

                # 2-Dimension = 0: left, 1: right, 2: top, 3: bottom, 4: top-left, 5: top-right, 6: bottom-left, 7: bottom-right
                verbose = True
                if dir[0] > 0 and dir[1] > 0:  # sender block is diagonally right top to  current subdomain
                    self.boundaryConstraints['top-right'] = cp.dequeue(tgid)
                    if verbose:
                        print("Top-right: %d received from %d: from direction %s" %
                              (cp.gid(), tgid, dir), self.boundaryConstraints['top-right'].shape)
                if dir[0] > 0 and dir[1] < 0:  # sender block is diagonally left top to current subdomain
                    self.boundaryConstraints['bottom-right'] = cp.dequeue(tgid)
                    if verbose:
                        print("Bottom-right: %d received from %d: from direction %s" %
                              (cp.gid(), tgid, dir), self.boundaryConstraints['bottom-right'].shape)
                if dir[0] < 0 and dir[1] < 0:  # sender block is diagonally left bottom  current subdomain
                    self.boundaryConstraints['bottom-left'] = cp.dequeue(tgid)
                    if verbose:
                        print("Bottom-left: %d received from %d: from direction %s" %
                              (cp.gid(), tgid, dir), self.boundaryConstraints['bottom-left'].shape)
                if dir[0] < 0 and dir[1] > 0:  # sender block is diagonally left to current subdomain

                    self.boundaryConstraints['top-left'] = cp.dequeue(tgid)
                    if verbose:
                        print("Top-left: %d received from %d: from direction %s" %
                              (cp.gid(), tgid, dir), self.boundaryConstraints['top-left'].shape)
                verbose = False

        return

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
        # for i in range(3, n + 2):
        #     for j in range(3, i + 1):
        #         e[i - 1, j - 1] = e[i - 2, j - 3] + 1 / (e[i - 1, j - 2] - e[i - 2, j - 2])

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
        plen = self.controlPointData.shape[0]*self.controlPointData.shape[1]
        self.solutionLocalHistory[:, 1:] = self.solutionLocalHistory[:, :-1]
        self.solutionLocalHistory[:, 0] = np.copy(self.controlPointData).reshape(plen)

        vAcc = []
        if not useAitken:
            if iterationNumber > nWynnEWork:  # For Wynn-E[silon
                vAcc = np.zeros(plen)
                for dofIndex in range(plen):
                    expVal = self.WynnEpsilon(
                        self.solutionLocalHistory[dofIndex, :],
                        math.floor((nWynnEWork - 1) / 2))
                    vAcc[dofIndex] = expVal[-1, -1]
                print('Performing scalar Wynn-Epsilon algorithm: Error is ',
                      np.linalg.norm(self.controlPointData.reshape(plen) - vAcc))  # , (self.solutionLocal - vAcc))
                self.controlPointData = vAcc[:].reshape(self.controlPointData.shape[0],
                                                        self.controlPointData.shape[1])

        else:
            if iterationNumber > 3:  # For Aitken acceleration
                vAcc = self.VectorAitken(self.solutionLocalHistory).reshape(self.controlPointData.shape[0],
                                                                            self.controlPointData.shape[1])
                # vAcc = np.zeros(self.controlPointData.shape)
                # for dofIndex in range(len(self.controlPointData)):
                #     vAcc[dofIndex] = self.Aitken(self.solutionLocalHistory[dofIndex, :])
                print('Performing Aitken Acceleration algorithm: Error is ',
                      np.linalg.norm(self.controlPointData - vAcc))
                self.controlPointData = vAcc[:]

    def initialize_data(self, cp):

        # Subdomain ID: iSubDom = cp.gid()+1
        # self.nControlPointSpans = self.nControlPoints - 1
        # self.nInternalKnotSpans = self.nControlPoints - degree

        inc = (self.Dmaxi - self.Dmini) / self.nInternalKnotSpans
        # print ("self.nInternalKnotSpans = ", self.nInternalKnotSpans, " inc = ", inc)

        # # Generate the knots in X and Y direction
        # tu = np.linspace(self.Dmini[0] + inc[0], self.Dmaxi[0] - inc[0], self.nInternalKnotSpans[0] - 1)
        # tv = np.linspace(self.Dmini[1] + inc[1], self.Dmaxi[1] - inc[1], self.nInternalKnotSpans[1] - 1)
        # self.knotsAdaptive['x'] = np.concatenate(([self.Dmini[0]] * (degree+1), tu, [self.Dmaxi[0]] * (degree+1)))
        # self.knotsAdaptive['y'] = np.concatenate(([self.Dmini[1]] * (degree+1), tv, [self.Dmaxi[1]] * (degree+1)))

        # self.UVW.x = np.linspace(self.Dmini[0], self.Dmaxi[0], self.nPointsPerSubDX)  # self.nPointsPerSubDX, nPointsX
        # self.UVW.y = np.linspace(self.Dmini[1], self.Dmaxi[1], self.nPointsPerSubDY)  # self.nPointsPerSubDY, nPointsY

        tu = np.linspace(self.Dmini[0] + inc[0], self.Dmaxi[0] - inc[0], self.nInternalKnotSpans[0] - 1)
        if dimension > 1:
            tv = np.linspace(self.Dmini[1] + inc[1], self.Dmaxi[1] - inc[1], self.nInternalKnotSpans[1] - 1)

        if nTotalSubDomains > 1 and not fullyPinned:

            print("Subdomain: ", cp.gid(), " X: ", self.Dmini[0], self.Dmaxi[0])
            if abs(self.Dmaxi[0] - xyzMax[0]) < 1e-12 and abs(self.Dmini[0] - xyzMin[0]) < 1e-12:
                self.knotsAdaptive['x'] = np.concatenate(
                    ([self.Dmini[0]] * (degree + 1),
                     tu, [self.Dmaxi[0]] * (degree + 1)))
                self.isClamped['left'] = self.isClamped['right'] = True
            else:
                if abs(self.Dmaxi[0] - xyzMax[0]) < 1e-12:
                    self.knotsAdaptive['x'] = np.concatenate(([self.Dmini[0]] * (1), tu, [self.Dmaxi[0]] * (degree+1)))
                    self.isClamped['right'] = True

                else:
                    if abs(self.Dmini[0] - xyzMin[0]) < 1e-12:
                        self.knotsAdaptive['x'] = np.concatenate(
                            ([self.Dmini[0]] * (degree+1), tu, [self.Dmaxi[0]] * (1)))
                        self.isClamped['left'] = True

                    else:
                        self.knotsAdaptive['x'] = np.concatenate(
                            ([self.Dmini[0]] * (1),
                                tu, [self.Dmaxi[0]] * (1)))
                        self.isClamped['left'] = self.isClamped['right'] = False

            if dimension == 1:
                print(
                    "Subdomain: ", cp.gid(),
                    " clamped ? ", self.isClamped['left'], self.isClamped['right'])

            if dimension > 1:
                print("Subdomain: ", cp.gid(), " Y: ", self.Dmini[1], self.Dmaxi[1])
                if abs(self.Dmaxi[1] - xyzMax[1]) < 1e-12 and abs(self.Dmini[1] - xyzMin[1]) < 1e-12:
                    print("Subdomain: ", cp.gid(), " checking top and bottom Y: ",
                          self.Dmaxi[1], xyzMax[1], abs(self.Dmaxi[1] - xyzMax[1]))
                    self.knotsAdaptive['y'] = np.concatenate(
                        ([self.Dmini[1]] * (degree + 1),
                         tv, [self.Dmaxi[1]] * (degree + 1)))
                    self.isClamped['top'] = self.isClamped['bottom'] = True
                else:

                    if abs(self.Dmaxi[1] - xyzMax[1]) < 1e-12:
                        print("Subdomain: ", cp.gid(), " checking top Y: ",
                              self.Dmaxi[1], xyzMax[1], abs(self.Dmaxi[1] - xyzMax[1]))
                        self.knotsAdaptive['y'] = np.concatenate(
                            ([self.Dmini[1]] * (1), tv, [self.Dmaxi[1]] * (degree+1)))
                        self.isClamped['top'] = True

                    else:

                        print("Subdomain: ", cp.gid(), " checking bottom Y: ",
                              self.Dmini[1], xyzMin[1], abs(self.Dmini[1] - xyzMin[1]))
                        if abs(self.Dmini[1] - xyzMin[1]) < 1e-12:
                            self.knotsAdaptive['y'] = np.concatenate(
                                ([self.Dmini[1]] * (degree+1), tv, [self.Dmaxi[1]] * (1)))
                            self.isClamped['bottom'] = True

                        else:
                            self.knotsAdaptive['y'] = np.concatenate(
                                ([self.Dmini[1]] * (1),
                                    tv, [self.Dmaxi[1]] * (1)))

                            self.isClamped['top'] = self.isClamped['bottom'] = False

                print(
                    "Subdomain: ", cp.gid(),
                    " clamped ? ", self.isClamped['left'],
                    self.isClamped['right'],
                    self.isClamped['top'],
                    self.isClamped['bottom'])

        else:
            self.knotsAdaptive['x'] = np.concatenate(([self.Dmini[0]] * (degree+1), tu, [self.Dmaxi[0]] * (degree+1)))
            self.isClamped['left'] = self.isClamped['right'] = True

            if dimension > 1:
                self.knotsAdaptive['y'] = np.concatenate(
                    ([self.Dmini[1]] * (degree + 1),
                     tv, [self.Dmaxi[1]] * (degree + 1)))
                self.isClamped['top'] = self.isClamped['bottom'] = True

        # self.UVW.x = np.linspace(self.Dmini[0], self.Dmaxi[0], self.nPointsPerSubDX)  # self.nPointsPerSubDX, nPointsX
        # self.UVW.y = np.linspace(self.Dmini[1], self.Dmaxi[1], self.nPointsPerSubDY)  # self.nPointsPerSubDY, nPointsY
        self.UVW['x'] = np.linspace(self.xyzCoordLocal['x'][0], self.xyzCoordLocal['x'][-1], self.nPointsPerSubD[0])
        if dimension > 1:
            self.UVW['y'] = np.linspace(self.xyzCoordLocal['y'][0], self.xyzCoordLocal['y'][-1], self.nPointsPerSubD[1])

    def augment_spans(self, cp):

        if verbose:
            if dimension == 1:
                print("Subdomain -- ", cp.gid()+1, ": before Shapes: ",
                      self.solutionLocal.shape, self.weightsData.shape, self.knotsAdaptive['x'])
            else:
                print("Subdomain -- ", cp.gid()+1, ": before Shapes: ", self.solutionLocal.shape,
                      self.weightsData.shape, self.knotsAdaptive['x'], self.knotsAdaptive['y'])

        if not self.isClamped['left']:  # Pad knot spans from the left of subdomain
            print("\tSubdomain -- ", cp.gid()+1, ": left ghost: ", self.ghostKnots['left'])
            self.knotsAdaptive['x'] = np.concatenate(
                (self.ghostKnots['left'][-1:0:-1], self.knotsAdaptive['x']))

        if not self.isClamped['right']:  # Pad knot spans from the right of subdomain
            print("\tSubdomain -- ", cp.gid()+1, ": right ghost: ", self.ghostKnots['right'])
            self.knotsAdaptive['x'] = np.concatenate(
                (self.knotsAdaptive['x'], self.ghostKnots['right'][1:]))

        if dimension > 1:
            if not self.isClamped['top']:  # Pad knot spans from the left of subdomain
                print("\tSubdomain -- ", cp.gid()+1, ": top ghost: ", self.ghostKnots['top'])
                self.knotsAdaptive['y'] = np.concatenate(
                    (self.knotsAdaptive['y'], self.ghostKnots['top'][1:]))

            if not self.isClamped['bottom']:  # Pad knot spans from the right of subdomain
                print("\tSubdomain -- ", cp.gid()+1, ": bottom ghost: ", self.ghostKnots['bottom'])
                self.knotsAdaptive['y'] = np.concatenate(
                    (self.ghostKnots['bottom'][-1:0:-1], self.knotsAdaptive['y']))

        if verbose:
            if dimension == 1:
                print("Subdomain -- ", cp.gid()+1, ": after Shapes: ",
                      self.solutionLocal.shape, self.weightsData.shape, self.knotsAdaptive['x'])
            else:
                print("Subdomain -- ", cp.gid()+1, ": after Shapes: ", self.solutionLocal.shape,
                      self.weightsData.shape, self.knotsAdaptive['x'], self.knotsAdaptive['y'])

    # def augment_inputdata(self, cp):

    #     verbose = False
    #     indicesX = np.where(np.logical_and(
    #         x >= self.knotsAdaptive['x'][degree]-1e-10, x <= self.knotsAdaptive['x'][-degree-1]+1e-10))
    #     print(indicesX)
    #     indicesY = np.where(np.logical_and(
    #         y >= self.knotsAdaptive['y'][degree]-1e-10, y <= self.knotsAdaptive['y'][-degree-1]+1e-10))
    #     print(indicesY)
    #     lboundX = indicesX[0][0]-1 if indicesX[0][0] > 0 else 0
    #     uboundX = indicesX[0][-1]+1 if indicesX[0][-1] < len(x) else indicesX[0][-1]
    #     lboundY = indicesY[0][0]-1 if indicesY[0][0] > 0 else 0
    #     uboundY = indicesY[0][-1]+1 if indicesY[0][-1] < len(y) else indicesY[0][-1]

    #     print(lboundX, uboundX, lboundY, uboundY)
    #     if verbose:
    #         print("Subdomain -- {0}: before augment -- left = {1}, right = {2}, shape = {3}".format(cp.gid()+1,
    #                                                                                                 self.xl[0], self.xl[-1], self.xl.shape))

    #     self.xl = x[lboundX:uboundX]  # x[indicesX]
    #     self.yl = y[lboundY:uboundY]  # y[indicesY]
    #     print(z.shape, indicesX[0].shape, indicesY[0].shape)
    #     self.solutionLocal = solution[lboundX:uboundX, lboundY:uboundY]  # y[lbound:ubound]
    #     self.nPointsPerSubDX = self.xl.shape[0]  # int(nPoints / nSubDomains) + overlapData
    #     self.nPointsPerSubDY = self.yl.shape[0]  # int(nPoints / nSubDomains) + overlapData

    #     hx = (xyzMax[0]-xyzMin[0])/nSubDomainsX
    #     hy = (xyzMax[1]-xyzMin[1])/nSubDomainsY
    #     # Store the core indices before augment

    #     postol = 1e-10
    #     cindicesX = np.array(np.where(np.logical_and(
    #         self.xl >= self.xbounds.min[0]-postol, self.xl <= self.xbounds.max[0]+postol)))
    #     cindicesY = np.array(np.where(np.logical_and(
    #         self.yl >= self.xbounds.min[1]-postol, self.yl <= self.xbounds.max[1]+postol)))

    #     print("self.corebounds = ", self.xbounds, self.corebounds)
    #     # cindices = np.array(
    #     #     np.where(
    #     #         np.logical_and(
    #     #             np.logical_and(
    #     #                 self.xl >= xmin + cp.gid() * hx - 1e-10, self.xl <= xmin + (cp.gid() + 1) * hx + 1e-10),
    #     #             np.logical_and(
    #     #                 self.yl >= ymin + cp.gid() * hy - 1e-10, self.yl <= ymin + (cp.gid() + 1) * hy + 1e-10))))
    #     # print('cindices: ', cindices, self.xl, self.yl,
    #     #       cp.gid(), hx, hy, xmin + cp.gid()*hx-1e-8, xmin + (cp.gid()+1)*hx+1e-8)
    #     print(
    #         'cindicesX: ', cindicesX, self.xl[0],
    #         self.xl[-1],
    #         x[self.xbounds.min[0]],
    #         x[self.xbounds.max[0]])
    #     print('cindicesY: ', cindicesY, self.yl[0], self.yl[-1], y[self.xbounds.min[1]], y[self.xbounds.max[1]])
    #     self.corebounds = [[cindicesX[0][0], cindicesX[0][-1]], [cindicesY[0][0], cindicesY[0][-1]]]

    #     # print('Corebounds:', cindices[0][0], cindices[-1][-1])

    #     if verbose:
    #         print("Subdomain -- {0}: after augment -- left = {1}, right = {2}, shape = {3}".format(cp.gid()+1,
    #                                                                                                x[indices[0]], x[indices[-1]], self.xl.shape))

    #     # print("Subdomain -- {0}: cindices -- {1} {2}, original x bounds = {3} {4}".format(cp.gid()+1,
    #     #                                                                                   self.xl[self.corebounds[0]], self.xl[self.corebounds[1]], self.xl[0], self.xl[-1]))
    #     self.solutionDecoded = np.zeros(self.yl.shape)
    #     self.solutionDecodedOld = np.zeros(self.yl.shape)

    def LSQFit_NonlinearOptimize(self, idom, degree, constraints=None):

        solution = []

        # Initialize relevant data
        if constraints is not None:
            initSol = np.copy(constraints).reshape(constraints.shape[0]*constraints.shape[1], order='C')
            # initSol = np.ones_like(self.weightsData)

        else:
            print('Constraints are all null. Solving unconstrained.')
            initSol = np.ones_like(self.weightsData)

        # Compute hte linear operators needed to compute decoded residuals
        # self.Nu = basis(self.UVW.x[np.newaxis,:],degree,Tunew[:,np.newaxis]).T
        # self.Nv = basis(self.UVW.y[np.newaxis,:],degree,Tvnew[:,np.newaxis]).T
        decodeOpXYZ = compute_decode_operators(self.weightsData, self.NUVW)

        # Compute the residual as sum of two components
        # 1. The decoded error evaluated at P
        # 2. A penalized domain boundary constraint component
        # @jit(nopython=True, parallel=False)
        def residual2D(Pin, verbose=False):

            P = np.array(Pin, copy=True).reshape(self.weightsData.shape, order='C')
            # P = Pin
            # alpha = 1
            # beta = 1
            bc_penalty = 1e5
            diag_penalty = 1e2
            if constraints is not None:
                decoded_penalty = 1
            else:
                decoded_penalty = 1
            # bc_penalty = 100
            # decoded_penalty = 0
            decoded_residual_norm = 0

            # Residuals are in the decoded space - so direct way to constrain the boundary data
            # decoded = decode(P, W, self.Nu, self.Nv)
            decoded = decode(P, self.weightsData, self.NUVW)
            # decoded = np.matmul(np.matmul(decodeOpXYZ['x'], P), decodeOpXYZ['y'].T)
            # decoded = np.matmul(np.matmul(decodeOpXYZ['y'], P), decodeOpXYZ['x'].T).T
            # decoded = np.matmul(np.matmul(decodeOpY, P), decodeOpX.T)
            residual_decoded = (self.solutionLocal - decoded)/solutionRange
            residual_vec_decoded = residual_decoded.reshape(residual_decoded.shape[0]*residual_decoded.shape[1])
            decoded_residual_norm = np.sqrt(np.sum(residual_vec_decoded**2)/len(residual_vec_decoded))

            # return decoded_residual_norm

#             res1 = np.dot(res_dec, self.Nv)
#             residual = np.dot(self.Nu.T, res1).reshape(self.Nu.shape[1]*self.Nv.shape[1])

            ltn = rtn = tpn = btn = 0
            constrained_residual_norm = 0
            oddDegree = (degree % 2)
            nconstraints = augmentSpanSpace + (int(degree/2.0) if not oddDegree else int((degree+1)/2.0))
            # Odd degree: [nconstraints-1, -nconstraints]
            # print('nconstraints = ', nconstraints)
            # cIndex = (int(degree/2.0) if not oddDegree else int((degree+1)/2.0))
            if constraints is not None and len(constraints) > 0:
                if 'left' in self.boundaryConstraints:

                    # degree 2: 1
                    loffset = -2*augmentSpanSpace if oddDegree else -2*augmentSpanSpace

                    # Compute the residual for left interface condition
                    # ltn = np.sum((P[nconstraints - 1, :] - 0.5 *
                    #               (constraints[nconstraints - 1, :] + self.boundaryConstraints['left']
                    #                [-nconstraints, :])) ** 2)
                    ltn = np.sum((P[nconstraints - 1, :] - 0.5 *
                                  (constraints[nconstraints - 1, :] + self.boundaryConstraints['left']
                                   [-nconstraints, :])) ** 2)

                    constrained_residual_norm += (ltn/len(P[0, :]))

                if 'right' in self.boundaryConstraints:

                    loffset = 2*augmentSpanSpace if oddDegree else 2*augmentSpanSpace

                    # Compute the residual for right interface condition
                    rtn = np.sum((P[-nconstraints, :] - 0.5 *
                                  (constraints[-nconstraints, :] + self.boundaryConstraints['right']
                                   [nconstraints - 1, :])) ** 2)

                    constrained_residual_norm += (rtn/len(P[-1, :]))

                if 'top' in self.boundaryConstraints:

                    loffset = -2*augmentSpanSpace if oddDegree else -2*augmentSpanSpace

                    # Compute the residual for top interface condition
                    tpn = np.sum((P[:, -nconstraints] - 0.5 *
                                  (constraints[:, -nconstraints] + self.boundaryConstraints['top']
                                   [:, nconstraints - 1])) ** 2)

                    constrained_residual_norm += (tpn / len(P[:, -1]))

                if 'bottom' in self.boundaryConstraints:

                    loffset = 2*augmentSpanSpace if oddDegree else 2*augmentSpanSpace

                    # Compute the residual for bottom interface condition
                    btn = np.sum(
                        (P[:, nconstraints - 1] - 0.5 *
                         (constraints[:, nconstraints - 1] + self.boundaryConstraints['bottom']
                          [:, -nconstraints])) ** 2)

                    constrained_residual_norm += (btn / len(P[:, 0]))

                if useDiagonalBlocks:
                    nDiagonalConstraints = 0
                    diagonal_boundary_residual_norm = 0
                    # 2-Dimension = 0: left, 1: right, 2: top, 3: bottom, 4: top-left, 5: top-right, 6: bottom-left, 7: bottom-right
                    topleftBndErr = toprightBndErr = bottomleftBndErr = bottomrightBndErr = []

                    if 'top-left' in self.boundaryConstraints:

                        # Compute the residual for top left interface condition
                        # topleftBndErr = (
                        #     P[-nconstraints, nconstraints-1] - 0.5 *
                        #     (constraints[-nconstraints, nconstraints-1] + self.boundaryConstraints['top-left']
                        #      [nconstraints-1, -nconstraints]))
                        topleftBndErr = (
                            P[nconstraints-1, -nconstraints] - 0.25 *
                            (constraints[nconstraints-1, -nconstraints] + self.boundaryConstraints['top-left']
                             [-nconstraints, nconstraints-1] + self.boundaryConstraints['top'][-nconstraints, nconstraints-1]
                             + self.boundaryConstraints['left'][nconstraints-1, -nconstraints]
                             ))
                        diagonal_boundary_residual_norm += np.sum(topleftBndErr**2)
                        nDiagonalConstraints += 1

                    if 'top-right' in self.boundaryConstraints:

                        # Compute the residual for top right interface condition
                        toprightBndErr = (
                            P[-nconstraints, -nconstraints] - 0.25 *
                            (constraints[-nconstraints, -nconstraints] +
                             self.boundaryConstraints['top-right'][nconstraints-1, nconstraints-1] +
                             self.boundaryConstraints['top'][-nconstraints, nconstraints-1] +
                             self.boundaryConstraints['right'][nconstraints-1, -nconstraints]
                             ))
                        diagonal_boundary_residual_norm += np.sum(toprightBndErr**2)
                        nDiagonalConstraints += 1

                    if 'bottom-left' in self.boundaryConstraints:

                        # Compute the residual for bottom left interface condition
                        bottomleftBndErr = (
                            P[nconstraints-1, nconstraints-1] - 0.25 *
                            (constraints[nconstraints-1, nconstraints-1] + self.boundaryConstraints['bottom-left']
                             [-nconstraints, -nconstraints] + self.boundaryConstraints['bottom']
                             [nconstraints-1, -nconstraints] + self.boundaryConstraints['left']
                             [-nconstraints, nconstraints-1]
                             ))
                        diagonal_boundary_residual_norm += np.sum(bottomleftBndErr**2)
                        nDiagonalConstraints += 1

                    if 'bottom-right' in self.boundaryConstraints:

                        # Compute the residual for bottom right interface condition
                        bottomrightBndErr = (
                            P[-nconstraints, nconstraints-1] - 0.25 *
                            (constraints[-nconstraints, nconstraints-1] +
                             self.boundaryConstraints['bottom-right'][nconstraints-1, -nconstraints] +
                             self.boundaryConstraints['bottom'][-nconstraints, nconstraints-1] +
                             self.boundaryConstraints['right'][nconstraints-1, nconstraints-1]
                             ))
                        diagonal_boundary_residual_norm += np.sum(bottomrightBndErr**2)
                        nDiagonalConstraints += 1

                    if nDiagonalConstraints > 0:
                        # diagonal_boundary_residual_norm = diagonal_boundary_residual_norm / nDiagonalConstraints / degree**2
                        diagonal_boundary_residual_norm = diagonal_boundary_residual_norm / nDiagonalConstraints

            # compute the net residual norm that includes the decoded error in the current subdomain and the penalized
            # constraint error of solution on the subdomain interfaces
            net_residual_norm = decoded_penalty * decoded_residual_norm + bc_penalty * np.sqrt(constrained_residual_norm) + (
                diag_penalty * np.sqrt(diagonal_boundary_residual_norm) if useDiagonalBlocks else 0)

            if verbose and True:
                print('Residual = ', net_residual_norm, ' and decoded = ', decoded_residual_norm, ', constraint = ',
                      constrained_residual_norm, ', diagonal = ', diagonal_boundary_residual_norm if useDiagonalBlocks else 0)
                print('Constraint errors = ', ltn, rtn, tpn, btn, constrained_residual_norm)
                if useDiagonalBlocks:
                    print('Constraint diagonal errors = ', topleftBndErr, toprightBndErr, bottomleftBndErr,
                          bottomrightBndErr, diagonal_boundary_residual_norm)

            return net_residual_norm

        # Set a function handle to the appropriate residual evaluator
        residualFunction = None
        if dimension == 1:
            residualFunction = residual1D
        else:
            residualFunction = residual2D

        def print_iterate(P, res=None):
            if res is None:
                res = residualFunction(P, verbose=True)
                # print('NLConstrained residual vector norm: ', np.linalg.norm(res, ord=2))
                self.globalIterationNum += 1
                return False
            else:
                rW = res.reshape(W.shape)
                # np.sqrt(np.sum(np.abs(res))/P.shape[0])
                # print('NLConstrained residual vector norm: ', np.linalg.norm(res, ord=2))
                # print('NLConstrained: bottom terms: ', np.linalg.norm(res.reshape(W.shape)[:, 0], ord=2))
                # print('NLConstrained: top terms: ', np.linalg.norm(res.reshape(W.shape)[:, -1-overlapData], ord=2))
                # print('NLConstrained: right terms: ', np.linalg.norm(res.reshape(W.shape)[-1-overlapData, :], ord=2))
                # print('NLConstrained: left terms: ', np.linalg.norm(res.reshape(W.shape)[overlapData, :], ord=2))
                print('NLConstrained [left, right, top, bottom] terms: ', np.linalg.norm(rW[overlapData, :], ord=2), np.linalg.norm(
                    rW[-1-overlapData, :], ord=2), np.linalg.norm(rW[:, -1-overlapData], ord=2), np.linalg.norm(rW[:, 0], ord=2))
                if(np.linalg.norm(rW[overlapData, :], ord=2) + np.linalg.norm(rW[-1 - overlapData, :], ord=2) +
                        np.linalg.norm(rW[:, -1 - overlapData], ord=2) + np.linalg.norm(rW[:, 0], ord=2) < 1e-12):
                    return True
                else:
                    return False

        # Use automatic-differentiation to compute the Jacobian value for minimizer
        def jacobian(P):
            #             if jacobian_const is None:
            #                 jacobian_const = egrad(residual)(P)

            jacobian = egrad(residualFunction)(P)
#             jacobian = jacobian_const
            return jacobian

        # if constraintsAll is not None:
        #    jacobian_const = egrad(residual)(initSol)

        # Now invoke the solver and compute the constrained solution
        if constraints is None and not alwaysSolveConstrained:
            solution = lsqFit(self.NUVW['x'], self.NUVW['y'], self.weightsData, self.solutionLocal)
            # solution = solution.reshape(W.shape)
        else:

            print('Initial calculation')
            print_iterate(initSol)

            if enforceBounds:
                bnds = np.tensordot(np.ones(self.controlPointData.shape[0]*self.controlPointData.shape[1]),
                                    self.controlPointBounds, axes=0)
            else:
                bnds = None
            print('Using optimization solver = ', solverScheme)
            # Solver options: https://docs.scipy.org/doc/scipy-0.13.0/reference/generated/scipy.optimize.show_options.html
            if solverScheme == 'L-BFGS-B':
                res = minimize(residualFunction, x0=initSol, method=solverScheme,  # 'SLSQP', #'L-BFGS-B', #'TNC',
                               bounds=bnds,
                               jac=jacobian,
                               callback=print_iterate,
                               tol=self.globalTolerance,
                               options={'disp': False, 'ftol': maxRelErr, 'gtol': self.globalTolerance, 'maxiter': solverMaxIter})
            elif solverScheme == 'CG':
                res = minimize(residualFunction, x0=initSol, method=solverScheme,  # Unbounded - can blow up
                               jac=jacobian,
                               bounds=bnds,
                               callback=print_iterate,
                               tol=self.globalTolerance,
                               options={'disp': False, 'ftol': self.globalTolerance, 'norm': 2, 'maxiter': solverMaxIter})
            elif solverScheme == 'SLSQP':
                res = minimize(residualFunction, x0=initSol, method=solverScheme,  # 'SLSQP', #'L-BFGS-B', #'TNC',
                               bounds=bnds,
                               callback=print_iterate,
                               tol=self.globalTolerance,
                               options={'disp': False, 'ftol': maxRelErr, 'maxiter': solverMaxIter})
            elif solverScheme == 'Newton-CG' or solverScheme == 'TNC':
                res = minimize(residualFunction, x0=initSol, method=solverScheme,  # 'SLSQP', #'L-BFGS-B', #'TNC',
                               jac=jacobian,
                               bounds=bnds,
                               # jac=egrad(residual)(initSol),
                               callback=print_iterate,
                               tol=self.globalTolerance,
                               options={'disp': False, 'eps': self.globalTolerance, 'maxiter': solverMaxIter})
            elif solverScheme == 'trust-krylov' or solverScheme == 'trust-ncg':
                res = minimize(residualFunction, x0=initSol, method=solverScheme,  # 'SLSQP', #'L-BFGS-B', #'TNC',
                               jac=jacobian,
                               hess=hessian,
                               bounds=bnds,
                               callback=print_iterate,
                               tol=self.globalTolerance,
                               options={'inexact': True})
            else:
                error('No implementation available')

            print('[%d] : %s' % (idom, res.message))
            solution = np.copy(res.x).reshape(self.weightsData.shape)

            # solution = res.reshape(W.shape)

        return solution

    def print_error_metrics(self, cp):
        # print('Size: ', commW.size, ' rank = ', commW.rank, ' Metrics: ', self.errorMetricsL2[:])
        # print('Rank:', commW.rank, ' SDom:', cp.gid(), ' L2 Error table: ', self.errorMetricsL2)
        print(
            'Rank:', commW.rank, ' SDom:', cp.gid(),
            ' Error: ', self.errorMetricsL2[self.outerIteration - 1],
            ', Convergence: ',
            [self.errorMetricsL2[1: self.outerIteration] - self.errorMetricsL2[0: self.outerIteration - 1]])

        # L2NormVector = MPI.gather(self.errorMetricsL2[self.outerIteration - 1], root=0)

    def check_convergence(self, cp, iterationNum):

        global isConverged, L2err
        if len(self.solutionDecodedOld):

            # Let us compute the relative change in the solution between current and previous iteration
            iterateChangeVec = (self.solutionDecoded - self.solutionDecodedOld)
            errorMetricsSubDomL2 = np.linalg.norm(iterateChangeVec.reshape(
                iterateChangeVec.shape[0] * iterateChangeVec.shape[1]),
                ord=2) / np.linalg.norm(self.solutionLocal, ord=2)
            errorMetricsSubDomLinf = np.linalg.norm(iterateChangeVec.reshape(
                iterateChangeVec.shape[0] * iterateChangeVec.shape[1]),
                ord=np.inf) / np.linalg.norm(self.solutionLocal, ord=np.inf)

            self.errorMetricsL2[self.outerIteration] = self.decodederrors[0]
            self.errorMetricsLinf[self.outerIteration] = self.decodederrors[1]

            print(cp.gid()+1, ' Convergence check: ', errorMetricsSubDomL2, errorMetricsSubDomLinf,
                  np.abs(self.errorMetricsLinf[self.outerIteration]-self.errorMetricsLinf[self.outerIteration-1]),
                  errorMetricsSubDomLinf < 1e-8 and np.abs(self.errorMetricsL2[self.outerIteration]-self.errorMetricsL2[self.outerIteration-1]) < 1e-10)

            L2err[cp.gid()] = self.decodederrors[0]
            if errorMetricsSubDomLinf < 1e-12 and np.abs(
                    self.errorMetricsL2[self.outerIteration] - self.errorMetricsL2[self.outerIteration - 1]) < 1e-12:
                print('Subdomain ', cp.gid()+1, ' has converged to its final solution with error = ', errorMetricsSubDomLinf)
                isConverged[cp.gid()] = 1

        # self.outerIteration = iterationNum+1
        self.outerIteration += 1

        # isASMConverged = commW.allreduce(self.outerIterationConverged, op=MPI.LAND)

    def solve_adaptive(self, cp):

        global isConverged
        if isConverged[cp.gid()] == 1:
            print(cp.gid()+1, ' subdomain has already converged to its solution. Skipping solve ...')
            return

        # Subdomain ID: iSubDom = cp.gid()+1
        newSolve = False
        if (np.sum(np.abs(self.controlPointData)) < 1e-14 and len(self.controlPointData) > 0) or len(self.controlPointData) == 0:
            newSolve = True

        if not newSolve:

            self.nControlPointSpans = self.nControlPoints - 1
            self.nInternalKnotSpans = self.nControlPoints - degree

        print("Subdomain -- ", cp.gid()+1)

        # Let do the recursive iterations
        # Use the previous MAK solver solution as initial guess; Could do something clever later

        # Invoke the adaptive fitting routine for this subdomain
        iSubDom = cp.gid()+1

        self.solutionDecodedOld = np.copy(self.solutionDecoded)
        # self.globalTolerance = 1e-3 * 1e-3**self.adaptiveIterationNum

        self.compute_basis(degree, self.knotsAdaptive)

        if ((np.sum(np.abs(self.controlPointData)) < 1e-14 and len(self.controlPointData) > 0) or len(self.controlPointData) == 0) and self.outerIteration == 0:
            print(iSubDom, " - Applying the unconstrained solver.")
            constraints = None

        else:
            print(iSubDom, " - Applying the constrained solver.")
            constraints = np.copy(self.controlPointData)

        #  Invoke the local subdomain solver
        self.controlPointData = self.LSQFit_NonlinearOptimize(iSubDom, degree, constraints)

        if constraints is None:  # We just solved the initial LSQ problem.
            # Store the maximum bounds to respect so that we remain monotone
            self.controlPointBounds = np.array([np.min(self.controlPointData), np.max(self.controlPointData)])

        # Update the local decoded data
        self.solutionDecoded = decode(self.controlPointData, self.weightsData, self.NUVW)
        # decodedError = np.abs(np.array(self.solutionLocal - self.solutionDecoded)) / solutionRange

        if len(self.solutionLocalHistory) == 0:
            if useAitken:
                self.solutionLocalHistory = np.zeros((self.controlPointData.shape[0]*self.controlPointData.shape[1], 3))
            else:
                self.solutionLocalHistory = np.zeros(
                    (self.controlPointData.shape[0]*self.controlPointData.shape[1], nWynnEWork))

        # E = (self.solutionDecoded[self.corebounds[0]:self.corebounds[1]] - self.controlPointData[self.corebounds[0]:self.corebounds[1]])/solutionRange
        decodedError = (
            self.solutionLocal[self.corebounds[0]: self.corebounds[1],
                               self.corebounds[2]: self.corebounds[3]] - self.solutionDecoded
            [self.corebounds[0]: self.corebounds[1],
             self.corebounds[2]: self.corebounds[3]]) / solutionRange
        decodedError = (decodedError.reshape(decodedError.shape[0]*decodedError.shape[1]))
        LinfErr = np.linalg.norm(decodedError, ord=np.inf)
        L2Err = np.sqrt(np.sum(decodedError**2)/len(decodedError))

        self.decodederrors[0] = L2Err
        self.decodederrors[1] = LinfErr

        print("Subdomain -- ", iSubDom, ": L2 error: ", L2Err, ", Linf error: ", LinfErr)


#########
# Initialize DIY
commWorld = diy.mpi.MPIComm()           # world
mc2 = diy.Master(commWorld)         # master
domain_control = diy.DiscreteBounds(np.zeros((dimension, 1), dtype=np.uint32), nPoints-1)

# Routine to recursively add a block and associated data to it

print('')


def add_input_control_block2(gid, core, bounds, domain, link):
    print("Subdomain %d: " % gid, core, bounds, domain)
    minb = bounds.min
    maxb = bounds.max

    xlocal = xcoord[minb[0]:maxb[0]+1]
    ylocal = ycoord[minb[1]:maxb[1]+1]
    sollocal = solution[minb[0]:maxb[0]+1, minb[1]:maxb[1]+1]

    # print("Subdomain %d: " % gid, minb[0], minb[1], maxb[0], maxb[1], z.shape, zlocal.shape)
    mc2.add(gid, InputControlBlock(gid, nControlPointsInput, core, bounds, sollocal, xlocal, ylocal), link)


# TODO: If working in parallel with MPI or DIY, do a global reduce here
errors = np.zeros([nASMIterations+1, 2])  # Store L2, Linf errors as function of iteration

# Let us initialize DIY and setup the problem
share_face = np.ones((dimension, 1)) > 0
wrap = np.ones((dimension, 1)) < 0
ghosts = np.zeros((dimension, 1), dtype=np.uint32)

d_control = diy.DiscreteDecomposer(2, domain_control, nTotalSubDomains, share_face, wrap, ghosts, nSubDomains)
a_control = diy.ContiguousAssigner(nprocs, nTotalSubDomains)

d_control.decompose(rank, a_control, add_input_control_block2)

mc2.foreach(InputControlBlock.show)

sys.stdout.flush()
commW.Barrier()

if rank == 0:
    print("\n---- Starting Global Iterative Loop ----")

mc2.foreach(InputControlBlock.show)

#########


def send_receive_all():
    useDIY_SR = True

    if useDIY_SR:
        mc2.foreach(InputControlBlock.send_diy)
        mc2.exchange(False)
        mc2.foreach(InputControlBlock.recv_diy)
    else:
        mc2.foreach(InputControlBlock.send)
        mc2.exchange(False)
        mc2.foreach(InputControlBlock.recv)

    return


# Before starting the solve, let us exchange the initial conditions
# including the knot vector locations that need to be used for creating
# padded knot vectors in each subdomain
mc2.foreach(InputControlBlock.initialize_data)

# Send and receive initial condition data as needed
send_receive_all()

if not fullyPinned:
    mc2.foreach(InputControlBlock.augment_spans)
    if augmentSpanSpace > 0:
        mc2.foreach(InputControlBlock.augment_inputdata)

del xcoord, ycoord, solution

start_time = timeit.default_timer()
for iterIdx in range(nASMIterations):

    if rank == 0:
        print("\n---- Starting Iteration: %d ----" % iterIdx)

    if iterIdx > 1:
        disableAdaptivity = True
        constrainInterfaces = True
    # else:
        # disableAdaptivity = False
        # constrainInterfaces = False

    if iterIdx > 0 and rank == 0:
        print("")

    # run our local subdomain solver
    mc2.foreach(InputControlBlock.solve_adaptive)

    # check if we have locally converged within criteria
    mc2.foreach(lambda icb, cp: InputControlBlock.check_convergence(icb, cp, iterIdx))

    isASMConverged = commW.allreduce(np.sum(isConverged), op=MPI.SUM)

    # commW.Barrier()
    sys.stdout.flush()

    if useVTKOutput:

        mc2.foreach(lambda icb, cp: InputControlBlock.set_fig_handles(
            icb, cp, None, None, "%d-%d" % (cp.gid(), iterIdx)))
        mc2.foreach(InputControlBlock.output_vtk)

        if rank == 0:
            WritePVTKFile(iterIdx)

    if isASMConverged == nTotalSubDomains:
        if rank == 0:
            print("\n\nASM solver converged after %d iterations\n\n" % (iterIdx+1))
        break

    else:
        if extrapolate:
            mc2.foreach(lambda icb, cp: InputControlBlock.extrapolate_guess(icb, cp, iterIdx))

        # Now let us perform send-receive to get the data on the interface boundaries from
        # adjacent nearest-neighbor subdomains
        send_receive_all()


# mc2.foreach(InputControlBlock.print_solution)

elapsed = timeit.default_timer() - start_time
sys.stdout.flush()
if rank == 0:
    print('\nTotal computational time for solve = ', elapsed, '\n')

avgL2err = commW.allreduce(np.sum(L2err[np.nonzero(L2err)]**2), op=MPI.SUM)
avgL2err = np.sqrt(avgL2err/nTotalSubDomains)
maxL2err = commW.allreduce(np.max(np.abs(L2err[np.nonzero(L2err)])), op=MPI.MAX)
minL2err = commW.allreduce(np.min(np.abs(L2err[np.nonzero(L2err)])), op=MPI.MIN)

# np.set_printoptions(formatter={'float': '{: 5.12e}'.format})
# mc.foreach(InputControlBlock.print_error_metrics)
if rank == 0:
    print("\nError metrics: L2 average = %6.12e, L2 maxima = %6.12e, L2 minima = %6.12e\n" % (avgL2err, maxL2err, minL2err))
    print('')

commW.Barrier()

np.set_printoptions(formatter={'float': '{: 5.12e}'.format})
mc2.foreach(InputControlBlock.print_error_metrics)

if showplot:
    plt.show()

# ---------------- END MAIN FUNCTION -----------------
