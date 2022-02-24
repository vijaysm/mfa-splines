# coding: utf-8
# get_ipython().magic(u'matplotlib notebook')
# %matplotlib notebook
import sys
import math
import argparse
import timeit

# import numpy as np
from functools import reduce

# from numpy import dtype

import numpy as npo
import pandas as pd

# Autograd AD impots
from autograd import elementwise_grad as egrad
import autograd.numpy as np

# from pymoab import core, types
# from pymoab.scd import ScdInterface
# from pymoab.hcoord import HomCoord

# SciPY imports
import scipy
from scipy.optimize import minimize
from scipy import linalg

# MPI imports
from mpi4py import MPI
import diy

from numba import jit

# Plotting imports
from matplotlib import pyplot as plt
from matplotlib import cm

# from pyevtk.hl import gridToVTK

from ProblemSolver1D import ProblemSolver1D
from ProblemSolver2D import ProblemSolver2D
from ProblemSolver3D import ProblemSolver3D

plt.style.use(["seaborn-whitegrid"])
params = {
    "ytick.color": "b",
    "xtick.color": "b",
    "axes.labelcolor": "b",
    "axes.edgecolor": "b",
}
plt.rcParams.update(params)

directions = ["x", "y", "z"]
# --- set problem input parameters here ---
problem = 1
dimension = 3
degree = 3
nSubDomains = np.array([1] * dimension, dtype=np.uint32)
nSubDomains = [2, 2, 2]
nSubDomainsX = nSubDomains[0]
nSubDomainsY = nSubDomains[1] if dimension > 1 else 1
nSubDomainsZ = nSubDomains[2] if dimension > 2 else 1

closedFormFunctional = True
nControlPointsInputIn = 50
debugProblem = False
verbose = False
showplot = False
useVTKOutput = True
useMOABMesh = False

augmentSpanSpace = 0
useDiagonalBlocks = True

relEPS = 5e-5
fullyPinned = False
useAdditiveSchwartz = True
enforceBounds = False
alwaysSolveConstrained = False

if fullyPinned:
    useDiagonalBlocks = False

# ------------------------------------------
# Solver parameters

#                      0      1       2         3          4         5
solverMethods = ["L-BFGS-B", "CG", "SLSQP", "COBYLA", "Newton-CG", "TNC"]
solverScheme = solverMethods[0]
solverMaxIter = 0
nASMIterations = 5
maxAbsErr = 1e-6
maxRelErr = 1e-12

# Solver acceleration
extrapolate = False
useAitken = True
nWynnEWork = 3

##################
# Initialize
nProblemInputPoints = 0
solutionRange = 1
##################

# Initialize DIY
# commWorld = diy.mpi.MPIComm()           # world
commWorld = MPI.COMM_WORLD
masterControl = diy.Master(commWorld)  # master
nprocs = commWorld.size
rank = commWorld.rank

if rank == 0:
    print("Argument List:", str(sys.argv))

##########################
# Parse command line overrides if any
##
argv = sys.argv[1:]


def usage():
    print(
        sys.argv[0],
        "-p <problem> -n <nsubdomains> -x <nsubdomains_x> -y <nsubdomains_y> -d <degree> -c <controlpoints> -o <overlapData> -a <nASMIterations> -g <augmentSpanSpace> -i <input_points>",
    )
    sys.exit(2)


# nSubDomainsX = nSubDomainsY = nSubDomainsZ = 1
Dmin = np.array(dimension, dtype=np.float32)
Dmax = np.array(dimension, dtype=np.float32)

######################################################################
# Get the argumnts from command-line to override options
parser = argparse.ArgumentParser(prog="UnclampedSolver")
parser.add_argument(
    "-v",
    "--verbose",
    help="increase output verbosity",
    default=False,
    action="store_true",
)
parser.add_argument(
    "--dimension", help="specify problem spatial dimension", type=int, default=dimension
)
parser.add_argument(
    "--input_points", help="specify problem input points", type=int, default=nProblemInputPoints
)
parser.add_argument(
    "--problem", help="specify problem identifier", type=int, default=problem
)
parser.add_argument(
    "-n",
    "--nsubdomains",
    help="specify number of subdomains in all directions",
    type=int,
    default=1,
)
parser.add_argument(
    "-x",
    "--nx",
    help="specify number of subdomains in X-direction",
    type=int,
    default=nSubDomainsX,
)
parser.add_argument(
    "-y",
    "--ny",
    help="specify number of subdomains in Y-direction",
    type=int,
    default=nSubDomainsY,
)
parser.add_argument(
    "-z",
    "--nz",
    help="specify number of subdomains in Z-direction",
    type=int,
    default=nSubDomainsZ,
)
parser.add_argument(
    "--degree", help="specify BSpline basis degree", type=int, default=degree
)
parser.add_argument(
    "--controlpoints",
    help="specify number of control points/subdomain in each direction",
    type=int,
    default=nControlPointsInputIn,
)
parser.add_argument(
    "-g",
    "--aug",
    help="specify number of overlap regions to exchange between subdomains",
    type=int,
    default=augmentSpanSpace,
)
parser.add_argument(
    "-a",
    "--nasm",
    help="specify number of outer iterations",
    type=int,
    default=nASMIterations,
)
parser.add_argument(
    "--solverIter",
    help="specify maximum number of subdomain solver iterations",
    type=int,
    default=solverMaxIter,
)
parser.add_argument(
    "--accel",
    help="specify whether to accelerate the outer iteration convergence",
    action="store_true",
)
parser.add_argument(
    "--wynn",
    help="specify whether Wynn-Epsilon algorithm (vs Aitken acceleration) should be used",
    action="store_true",
)

# Process the arguments
args = parser.parse_args()

# Retrieve and store it in our local variables
verbose = args.verbose
if args.dimension != dimension:
    dimension = args.dimension
if args.problem != problem:
    problem = args.problem

nProblemInputPoints = args.input_points
if args.nsubdomains > 0:
    nSubDomainsX = args.nsubdomains
    nSubDomainsY = args.nsubdomains if dimension > 1 else 1
    nSubDomainsZ = args.nsubdomains if dimension > 2 else 1
if args.nx != nSubDomainsX:
    nSubDomainsX = args.nx
if args.ny != nSubDomainsY:
    nSubDomainsY = args.ny
if args.nz != nSubDomainsZ:
    nSubDomainsZ = args.nz

if args.degree != degree:
    degree = args.degree
if args.controlpoints != nControlPointsInputIn:
    nControlPointsInputIn = args.controlpoints
if args.aug != augmentSpanSpace:
    augmentSpanSpace = args.aug
if args.nasm != nASMIterations:
    nASMIterations = args.nasm
if args.solverIter != solverMaxIter:
    solverMaxIter = args.solverIter
extrapolate = args.accel
useAitken = not args.wynn
######################################################################

nSubDomainsY = 1 if dimension < 2 else nSubDomainsY
nSubDomainsZ = 1 if dimension < 3 else nSubDomainsZ
# showplot = False if dimension > 1 else True
# nControlPointsInput = nControlPointsInputIn * np.ones((dimension, 1), dtype=np.uint32)
nSubDomains = [nSubDomainsX, nSubDomainsY, nSubDomainsZ]
# -------------------------------------

nPoints = np.array([1] * dimension, dtype=np.uint32)
nTotalSubDomains = nSubDomainsX * nSubDomainsY * nSubDomainsZ
isConverged = np.zeros(nTotalSubDomains, dtype="int32")
L2err = np.zeros(nTotalSubDomains)

# globalExtentDict = np.zeros(nTotalSubDomains*2*dimension, dtype='int32')
# globalExtentDict[cp.gid()*4:cp.gid()*4+4] = extents
localExtents = {}


def interpolate_inputdata(solprofile, Xi, newX, Yi=None, newY=None, Zi=None, newZ=None):

    from scipy.interpolate import interp1d
    from scipy.interpolate import RectBivariateSpline
    from scipy.interpolate import RegularGridInterpolator

    # InterpCp = interp1d(coeffs_x, self.pAdaptive, kind=interpOrder)
    if dimension == 1:
        interp_oneD = interp1d(Xi, solprofile, kind="cubic")

        return interp_oneD(newX)

    elif dimension == 2:
        interp_spline = RectBivariateSpline(Xi, Yi, solprofile)

        return interp_spline(newX, newY)

    else:
        interp_multiD = RegularGridInterpolator((Xi, Yi, Zi), solprofile)

        return interp_multiD((newX, newY, newZ))


# def read_problem_parameters():
coordinates = {"x": None, "y": None, "z": None}
solution = None

if dimension == 1:
    if rank == 0:
        print("Setting up problem for 1-D")

    if problem == 1:
        Dmin = [-4.0]
        Dmax = [4.0]
        nPoints[0] = 10001
        coordinates["x"] = np.linspace(Dmin[0], Dmax[0], nPoints[0])
        scale = 100
        # solution = lambda x: scale * (
        #     np.sinc(coordinates["x"] - 1) + np.sinc(coordinates["x"] + 1)
        # )
        # solution = lambda x: scale * (np.sinc(x+1) + np.sinc(2*x) + np.sinc(x-1))
        solution = lambda x: scale * (
            np.sinc(x)
            + np.sinc(2 * x - 1)
            + np.sinc(3 * x + 1.5)
        )
        # solution = lambda x: np.zeros(x.shape)
        # solution[coordinates["x"] <= 0] = 1
        # solution[coordinates["x"] > 0] = -1
        # solution = lambda x: scale * np.sin(math.pi * x/4)
        solutionRange = scale
    elif problem == 2:
        solution = np.fromfile("data/s3d.raw", dtype=np.float64)
        print("Real data shape: ", solution.shape)
        Dmin = [0.0]
        Dmax = [1.0]
        nPoints[0] = solution.shape[0]
        coordinates["x"] = np.linspace(Dmin[0], Dmax[0], nPoints[0])
        relEPS = 5e-8
        closedFormFunctional = False
    elif problem == 3:
        Y = np.fromfile("data/nek5000.raw", dtype=np.float64)
        Y = Y.reshape(200, 200)
        solution = Y[100, :]  # Y[:,150] # Y[110,:]
        Dmin = [0.0]
        Dmax = [1.0]
        nPoints[0] = solution.shape[0]
        coordinates["x"] = np.linspace(Dmin[0], Dmax[0], nPoints[0])
        closedFormFunctional = False
    else:
        """
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
        """

        DJI = pd.read_csv("data/DJI.csv")
        solution = np.array(DJI["Close"])
        Dmin = [0]
        Dmax = [100.0]
        coordinates["x"] = np.linspace(Dmin[0], Dmax[0], solution.shape[0])
        closedFormFunctional = False

        nPoints[0] = solution.shape[0]

elif dimension == 2:
    print("Setting up problem for 2-D")

    if problem == 0:
        nPoints[0] = 1049
        nPoints[1] = 2049
        scale = 1
        Dmin = [0.0, 0.0]
        Dmax = [1.0, 1.0]

        coordinates["x"] = np.linspace(Dmin[0], Dmax[0], nPoints[0])
        coordinates["y"] = np.linspace(Dmin[1], Dmax[1], nPoints[1])
        solution = 10 * np.ones((nPoints[0], nPoints[1]))
        solution[:, 0] = 0
        solution[:, -1] = 0
        solution[0, :] = 0
        solution[-1, :] = 0

        debugProblem = True

    elif problem == 1:
        nPoints[0] = nProblemInputPoints if nProblemInputPoints else 9001
        nPoints[1] = nProblemInputPoints if nProblemInputPoints else 9001
        scale = 100
        Dmin = [-4.0, -4.0]
        Dmax = [4.0, 4.0]
        solutionRange = scale

        coordinates["x"] = np.linspace(Dmin[0], Dmax[0], nPoints[0])
        coordinates["y"] = np.linspace(Dmin[1], Dmax[1], nPoints[1])
        # X, Y = np.meshgrid(coordinates["x"], coordinates["y"])

        # solution = 100 * (1+0.25*np.tanh((X**2+Y**2)/16)) * (np.sinc(np.sqrt((X-2) ** 2 + (Y+2)**2)) +
        #                                               np.sinc(np.sqrt((X+2)**2 + (Y-2)**2)) -
        #                                               2 * (1-np.tanh((X)**2 + (Y)**2)) +
        #                                               np.exp(-((X-2)**2/2)-((Y-2)**2/2))
        #                                               #   + np.sign(X+Y)
        #                                               )

        # noise = np.random.uniform(0, 0.005, X.shape)
        # solution = solution * (1 + noise)

        # solution = lambda x, y: scale * x * y
        solution = lambda x, y: (
            scale
            * (
                np.sinc(np.sqrt(x**2 + y**2))
                + np.sinc(2 * ((x - 2) ** 2 + (y + 2) ** 2))
            )
        )

        # solution = lambda x, y: scale * (np.sinc(x) + np.sinc(2 *
        #                     x-1) + np.sinc(3*x+1.5)).T
        # solution = lambda x, y: ((4-x)*(4-y)).T

        # Test against 1-d solution. Matches correctly
        # solution = lambda x, y: scale * (np.sinc(x-1)+np.sinc(x+1)).T

        # solution = lambda x, y: scale * (np.sinc((X+1)**2 + (Y-1)**2) + np.sinc(((X-1)**2 + (Y+1)**2)))
        # solution = lambda x, y: X**2 * (DmaxX - Y)**2 + X**2 * Y**2 + 64 * (np.sinc(np.sqrt((X-2) ** 2 + (Y+2)**2)))
        # solution = lambda x, y: (3-abs(X))**3 * (1-abs(Y))**5 + (1-abs(X))**3 * (3-abs(Y))**5
        # solution = lambda x, y: (Dmax[0]+X)**5 * (Dmax[1]-Y)**5
        # solution = lambda x, y: solution / np.linalg.norm(solution)
        # solution = lambda x, y: scale * (np.sinc(X) * np.sinc(Y))
        # solution = solution.T
        # (3*degree + 1) #minimum number of control points
        # del X, Y

    elif problem == 2:
        nPoints[0] = 501
        nPoints[1] = 501
        scale = 1
        shiftX = 0.25 * 0
        shiftY = -0.25 * 0
        Dmin = [0.0, 0.0]
        Dmin = [-math.pi, -math.pi]
        Dmax = [math.pi, math.pi]

        coordinates["x"] = np.linspace(Dmin[0], Dmax[0], nPoints[0])
        coordinates["y"] = np.linspace(Dmin[1], Dmax[1], nPoints[1])
        # X, Y = np.meshgrid(coordinates["x"] + shiftX, coordinates["y"] + shiftY)
        # solution = scale * np.sinc(X) * np.sinc(Y)
        solution = lambda x, y: scale * np.cos(x + 1) * np.cos(y) + scale * np.sin(x - 1) * np.sin(y)
        # solution = scale * X * Y
        # (3*degree + 1) #minimum number of control points
        # del X, Y

    elif problem == 3:
        nPoints[0] = 1000
        nPoints[1] = 1000
        solution = np.fromfile("data/nek5000.raw", dtype=np.float64).reshape(200, 200)

        Dmin = [0, 0]
        Dmax = [100.0, 100.0]
        coordinates["x"] = np.linspace(Dmin[0], Dmax[0], nPoints[0])
        coordinates["y"] = np.linspace(Dmin[1], Dmax[1], nPoints[1])

        if nPoints[0] != solution.shape[0] or nPoints[1] != solution.shape[1]:
            solution = interpolate_inputdata(
                solprofile=np.copy(solution),
                Xi=np.linspace(Dmin[0], Dmax[0], solution.shape[0]),
                newX=coordinates["x"],
                Yi=np.linspace(Dmin[1], Dmax[1], solution.shape[1]),
                newY=coordinates["y"],
            )
        print("Nek5000 shape:", solution.shape)
        closedFormFunctional = False

        # (3*degree + 1) #minimum number of control points
        if nControlPointsInputIn == 0:
            nControlPointsInputIn = 20

    elif problem == 4:

        nPoints[0] = 5400
        nPoints[1] = 7040
        binFactor = 4.0
        solution = np.fromfile("data/s3d_2D.raw", dtype=np.float64).reshape(540, 704)

        Dmin = [0, 0]
        Dmax = nPoints
        coordinates["x"] = np.linspace(Dmin[0], Dmax[0], nPoints[0])
        coordinates["y"] = np.linspace(Dmin[1], Dmax[1], nPoints[1])

        if nPoints[0] != solution.shape[0] or nPoints[1] != solution.shape[1]:
            solution = interpolate_inputdata(
                solprofile=np.copy(solution),
                Xi=np.linspace(Dmin[0], Dmax[0], solution.shape[0]),
                newX=coordinates["x"],
                Yi=np.linspace(Dmin[1], Dmax[1], solution.shape[1]),
                newY=coordinates["y"],
            )
        # z = z[:540,:540]
        # z = zoom(z, 1./binFactor, order=4)
        print("S3D shape:", solution.shape)
        closedFormFunctional = False

    elif problem == 5:
        nPoints[0] = 1800
        nPoints[1] = 3600
        solution = np.fromfile("data/FLDSC_1_1800_3600.dat", dtype=np.float32).reshape(
            1800, 3600
        )
        Dmin = [0, 0]
        Dmax = nPoints
        coordinates["x"] = np.linspace(Dmin[0], Dmax[0], nPoints[0])
        coordinates["y"] = np.linspace(Dmin[1], Dmax[1], nPoints[1])

        if nPoints[0] != solution.shape[0] or nPoints[1] != solution.shape[1]:
            solution = interpolate_inputdata(
                solprofile=np.copy(solution),
                Xi=np.linspace(Dmin[0], Dmax[0], solution.shape[0]),
                newX=coordinates["x"],
                Yi=np.linspace(Dmin[1], Dmax[1], solution.shape[1]),
                newY=coordinates["y"],
            )
        print("CESM data shape: ", solution.shape)
        closedFormFunctional = False

    elif problem == 6:
        # A grid of c-values
        nPoints[0] = 6001
        nPoints[1] = 6001
        # nPoints[0] = 2501
        # nPoints[1] = 2501
        scale = 1.0
        shiftX = 0.25
        shiftY = 0.5
        Dmin = [-2, -1.5]
        Dmax = [1, 1.5]

        N_max = 255
        some_threshold = 50.0

        @jit
        def mandelbrot(c, maxiter):
            z = c
            for n in range(maxiter):
                if abs(z) > 2:
                    return (n % 4 * 64) * 65536 + (n % 8 * 32) * 256 + (n % 16 * 16)
                    # return n
                z = z * z + c
            return 0

        @jit
        def mandelbrot_set(xmin, xmax, ymin, ymax, width, height, maxiter):
            r1 = npo.linspace(xmin, xmax, width, dtype=npo.float)
            r2 = npo.linspace(ymin, ymax, height, dtype=npo.float)
            n3 = npo.empty((width, height))
            for i in range(width):
                for j in range(height):
                    n3[i, j] = mandelbrot(r1[i] + 1j * r2[j], maxiter)
            return (r1, r2, n3)

        coordinates["x"], coordinates["y"], solution = mandelbrot_set(
            Dmin[0], Dmax[0], Dmin[1], Dmax[1], nPoints[0], nPoints[1], N_max
        )
        solution /= 1e4

        # coordinates["x"] = np.linspace(Dmin[0], Dmax[0], nPoints[0])
        # coordinates["y"] = np.linspace(Dmin[1], Dmax[1], nPoints[1])
        # # X, Y = np.meshgrid(coordinates["x"]+shiftX, coordinates["y"]+shiftY)

        # # from PIL import Image
        # # image = Image.new("RGB", (nPoints[0], nPoints[1]))
        # mandelbrot_set = np.zeros((nPoints[0], nPoints[1]))
        # for yi in range(nPoints[1]):
        #     zy = yi * (Dmax[1] - Dmin[1]) / (nPoints[1] - 1) + Dmin[1]
        #     coordinates["y"][yi] = zy
        #     for xi in range(nPoints[0]):
        #         zx = xi * (Dmax[0] - Dmin[0]) / (nPoints[0] - 1) + Dmin[0]
        #         coordinates["x"][xi] = zx
        #         z = zx + zy * 1j
        #         c = z
        #         for i in range(N_max):
        #             if abs(z) > 2.0:
        #                 break
        #             z = z * z + c
        #         # image.putpixel((xi, yi), (i % 4 * 64, i % 8 * 32, i % 16 * 16))
        #         # RGB = (R*65536)+(G*256)+B
        #         mandelbrot_set[xi, yi] = (
        #             i % 4 * 64) * 65536 + (i % 8 * 32) * 256 + (i % 16 * 16)

        # # image.show()

        # solution = mandelbrot_set / 1e5

        # plt.imshow(solution, extent=[Dmin[0], Dmax[0], Dmin[1], Dmax[1]])
        # plt.show()

        closedFormFunctional = False
        if nControlPointsInputIn == 0:
            nControlPointsInputIn = 50

    else:
        print("Not a valid problem")
        exit(1)

elif dimension == 3:
    print("Setting up problem for 3-D")

    if problem == 1:
        nPoints[0] = 101
        nPoints[1] = 101
        nPoints[2] = 101
        scale = 100
        Dmin = [-4.0, -4.0, -4.0]
        Dmax = [4.0, 4.0, 4.0]

        coordinates["x"] = np.linspace(Dmin[0], Dmax[0], nPoints[0])
        coordinates["y"] = np.linspace(Dmin[1], Dmax[1], nPoints[1])
        coordinates["z"] = np.linspace(Dmin[2], Dmax[2], nPoints[2])
        # X, Y, Z = np.meshgrid(
        #     coordinates["x"], coordinates["y"], coordinates["z"], indexing="ij"
        # )

        # solution = 100 * (1+0.25*np.tanh((X**2+Y**2)/16)) * (np.sinc(np.sqrt((X-2) ** 2 + (Y+2)**2)) +
        #                                               np.sinc(np.sqrt((X+2)**2 + (Y-2)**2)) -
        #                                               2 * (1-np.tanh((X)**2 + (Y)**2)) +
        #                                               np.exp(-((X-2)**2/2)-((Y-2)**2/2))
        #                                               #   + np.sign(X+Y)
        #                                               )

        # noise = np.random.uniform(0, 0.005, X.shape)
        # solution = solution * (1 + noise)

        # solution = (Z)

        # solution = scale * X * Y * Z
        # solution = scale * (np.sinc(np.sqrt(X**2 + Y**2 + Z**2)) +
        #                     np.sinc(2*((X-2)**2 + (Y+2)**2 + Z**2)))

        solution = lambda x, y, z: scale * (
            np.sinc(np.sqrt(x**2 + y**2 + z**2))
            + np.sinc(2 * (x - 2) ** 2 + (y + 2) ** 2 + (z - 2) ** 2)
        )

        # solution = scale * (np.sinc(X) + np.sinc(2 *
        #                     X-1) + np.sinc(3*X+1.5)).T
        # solution = ((4-X)*(4-Y)).T

        # Test against 1-d solution. Matches correctly
        # solution = scale * (np.sinc(X-1)+np.sinc(X+1)).T

        # solution = scale * (np.sinc((X+1)**2 + (Y-1)**2) + np.sinc(((X-1)**2 + (Y+1)**2)))
        # solution = X**2 * (DmaxX - Y)**2 + X**2 * Y**2 + 64 * (np.sinc(np.sqrt((X-2) ** 2 + (Y+2)**2)))
        # solution = (3-abs(X))**3 * (1-abs(Y))**5 + (1-abs(X))**3 * (3-abs(Y))**5
        # solution = (Dmax[0]+X)**5 * (Dmax[1]-Y)**5
        # solution = solution / np.linalg.norm(solution)
        # solution = scale * (np.sinc(X) * np.sinc(Y))
        # solution = solution.T
        # (3*degree + 1) #minimum number of control points
        # del X, Y, Z

nControlPointsInput = np.array([nControlPointsInputIn] * dimension, dtype=np.uint32)
# if dimension == 2:
#     nControlPointsInput = np.array([10, 10], dtype=np.uint32)

# if nPointsX % nSubDomainsX > 0 or nPointsY % nSubDomainsY > 0:
#     print ( "[ERROR]: The total number of points do not divide equally with subdomains" )
#     sys.exit(1)

xyzMin = np.array([coordinates["x"].min(), 0, 0])
xyzMax = np.array([coordinates["x"].max(), 0, 0])
if dimension > 1:
    xyzMin[1] = coordinates["y"].min()
    xyzMax[1] = coordinates["y"].max()
    if dimension > 2:
        xyzMin[2] = coordinates["z"].min()
        xyzMax[2] = coordinates["z"].max()

solutionShape = None

def plot_solution(solVector):
    from uvw import RectilinearGrid, DataArray

    if dimension == 1:
        mpl_fig = plt.figure()
        plt.plot(coordinates["x"], solVector, "r-", ms=2)
        mpl_fig.show()
    elif dimension == 2:
        with RectilinearGrid(
            "./structured.vtr", (coordinates["x"], coordinates["y"])
        ) as rect:
            rect.addPointData(DataArray(solVector, range(2), "solution"))
    elif dimension == 3:
        with RectilinearGrid(
            "./structured.vtr", (coordinates["x"], coordinates["y"], coordinates["z"])
        ) as rect:
            rect.addPointData(DataArray(solVector, range(3), "solution"))
    else:
        print("No visualization output available for dimension > 2")


# Store the reference solution
if useVTKOutput:
    if dimension > 1:
        print('Writing out reference output solution')
        if closedFormFunctional:
            if dimension == 2:
                X, Y = np.meshgrid(
                    coordinates["x"], coordinates["y"], indexing="ij"
                )
                solVis = solution(X, Y)
            else:
                X, Y, Z = np.meshgrid(
                    coordinates["x"], coordinates["y"], coordinates["z"], indexing="ij"
                )
                solVis = solution(X, Y, Z)
            print('Redundant solution shape: ', solVis.shape)
            plot_solution(solVis)
            del solVis
        else:
            plot_solution(solution)

### Print parameter details ###
if rank == 0:
    print("\n==================")
    print("Parameter details")
    print("==================\n")
    print("dimension = ", dimension)
    print("problem = ", problem, "[1 = sinc, 2 = sine, 3 = Nek5000, 4 = S3D, 5 = CESM]")
    print("Input points = ", nPoints, "; Total = ", np.prod(nPoints))
    print("nSubDomains = ", nSubDomains, "; Total = ", np.prod(nSubDomains) )
    print("degree = ", degree)
    print("nControlPoints = ", nControlPointsInput)
    print("nASMIterations = ", nASMIterations)
    print("augmentSpanSpace = ", augmentSpanSpace)
    print("useAdditiveSchwartz = ", useAdditiveSchwartz)
    print("enforceBounds = ", enforceBounds)
    print("maxAbsErr = ", maxAbsErr)
    print("maxRelErr = ", maxRelErr)
    print("solverMaxIter = ", solverMaxIter)
    print("solverscheme = ", solverScheme)
    print("\n=================\n")

# ------------------------------------
sys.stdout.flush()


def WritePVTKFile(iteration):
    # print(globalExtentDict)
    pvtkfile = open("pstructured-mfa-%d.pvtr" % (iteration), "w")

    pvtkfile.write('<?xml version="1.0"?>\n')
    pvtkfile.write(
        '<VTKFile type="PRectilinearGrid" version="1.0" byte_order="LittleEndian" header_type="UInt64">\n'
    )
    if dimension == 2:
        pvtkfile.write(
            '<PRectilinearGrid WholeExtent="%d %d %d %d %d %d" GhostLevel="0">\n'
            % (0, solutionShape[0] - 1, 0, solutionShape[1] - 1, 0, 0)
        )
    else:
        pvtkfile.write(
            '<PRectilinearGrid WholeExtent="%d %d %d %d %d %d" GhostLevel="0">\n'
            % (
                0,
                solutionShape[0] - 1,
                0,
                solutionShape[1] - 1,
                0,
                solutionShape[2] - 1,
            )
        )
    pvtkfile.write("\n")
    pvtkfile.write("    <PPointData>\n")
    pvtkfile.write('      <PDataArray type="Float64" Name="solution"/>\n')
    pvtkfile.write('      <PDataArray type="Float64" Name="error"/>\n')
    pvtkfile.write("    </PPointData>\n")
    pvtkfile.write("    <PCellData></PCellData>\n")
    pvtkfile.write("    <PCoordinates>\n")
    pvtkfile.write(
        '      <PDataArray type="Float64" Name="x_coordinates" NumberOfComponents="1"/>\n'
    )
    pvtkfile.write(
        '      <PDataArray type="Float64" Name="y_coordinates" NumberOfComponents="1"/>\n'
    )
    pvtkfile.write(
        '      <PDataArray type="Float64" Name="z_coordinates" NumberOfComponents="1"/>\n'
    )
    pvtkfile.write("    </PCoordinates>\n\n")

    isubd = 0
    for iz in range(nSubDomainsZ):
        for iy in range(nSubDomainsY):
            # xoff = 0
            # ncx = 0
            for ix in range(nSubDomainsX):
                if dimension == 2:
                    pvtkfile.write(
                        '    <Piece Extent="%d %d %d %d 0 0" Source="structured-%d-%d.vtr"/>\n'
                        % (
                            0,
                            globalExtentDict[isubd][1] - globalExtentDict[isubd][0],
                            0,
                            globalExtentDict[isubd][3] - globalExtentDict[isubd][2],
                            isubd,
                            iteration,
                        )
                    )
                else:
                    pvtkfile.write(
                        '    <Piece Extent="%d %d %d %d %d %d" Source="structured-%d-%d.vtr"/>\n'
                        % (
                            0,
                            globalExtentDict[isubd][1] - globalExtentDict[isubd][0],
                            0,
                            globalExtentDict[isubd][3] - globalExtentDict[isubd][2],
                            0,
                            globalExtentDict[isubd][5] - globalExtentDict[isubd][4],
                            isubd,
                            iteration,
                        )
                    )
                isubd += 1
                # xoff += globalExtentDict[isubd][0]
                # ncx += nControlPointsInput[0]
            # yoff += globalExtentDict[isubd][1]
            # ncy += nControlPointsInput[1]
    pvtkfile.write("\n")
    pvtkfile.write("</PRectilinearGrid>\n")
    pvtkfile.write("</VTKFile>\n")

    pvtkfile.close()


# Write control point data


def WritePVTKControlFile(iteration):
    nconstraints = int(degree / 2.0) if not degree % 2 else int((degree + 1) / 2.0)
    # nconstraints=1
    print("Nconstraints = ", nconstraints)
    pvtkfile = open("pstructured-control-mfa-%d.pvtr" % (iteration), "w")

    pvtkfile.write('<?xml version="1.0"?>\n')
    pvtkfile.write(
        '<VTKFile type="PRectilinearGrid" version="1.0" byte_order="LittleEndian" header_type="UInt64">\n'
    )
    if dimension == 2:
        pvtkfile.write(
            '<PRectilinearGrid WholeExtent="%d %d %d %d 0 0" GhostLevel="%d">\n'
            % (
                0,
                nSubDomainsX * nControlPointsInput[0] - (nSubDomainsX - 1),
                0,
                nSubDomainsY * nControlPointsInput[1] - (nSubDomainsY - 1),
                0 * (nconstraints + augmentSpanSpace),
            )
        )
    else:
        pvtkfile.write(
            '<PRectilinearGrid WholeExtent="%d %d %d %d %d %d" GhostLevel="%d">\n'
            % (
                0,
                nSubDomainsX * nControlPointsInput[0] - (nSubDomainsX - 1),
                0,
                nSubDomainsY * nControlPointsInput[1] - (nSubDomainsY - 1),
                0,
                nSubDomainsZ * nControlPointsInput[2] - (nSubDomainsZ - 1),
                0 * (nconstraints + augmentSpanSpace),
            )
        )
    pvtkfile.write("\n")
    pvtkfile.write("    <PPointData>\n")
    pvtkfile.write('      <PDataArray type="Float64" Name="controlpoints"/>\n')
    pvtkfile.write("    </PPointData>\n")
    pvtkfile.write("    <PCellData></PCellData>\n")
    pvtkfile.write("    <PCoordinates>\n")
    pvtkfile.write(
        '      <PDataArray type="Float64" Name="x_coordinates" NumberOfComponents="1"/>\n'
    )
    pvtkfile.write(
        '      <PDataArray type="Float64" Name="y_coordinates" NumberOfComponents="1"/>\n'
    )
    pvtkfile.write(
        '      <PDataArray type="Float64" Name="z_coordinates" NumberOfComponents="1"/>\n'
    )
    pvtkfile.write("    </PCoordinates>\n\n")

    isubd = 0
    # dx = (xyzMax[0]-xyzMin[0])/nSubDomainsX
    # dy = (xyzMax[1]-xyzMin[1])/nSubDomainsY
    dxyz = (xyzMax - xyzMin) / nSubDomains

    zoff = xyzMin[2] if dimension == 3 else 0
    ncz = 0
    for iz in range(nSubDomainsZ):
        yoff = xyzMin[1]
        ncy = 0
        for iy in range(nSubDomainsY):
            xoff = xyzMin[0]
            ncx = 0
            for ix in range(nSubDomainsX):
                if dimension == 3:
                    pvtkfile.write(
                        '    <Piece Extent="%d %d %d %d %d %d" Source="structuredcp-%d-%d.vtr"/>\n'
                        % (
                            ncx,
                            ncx + nControlPointsInput[0] - 1,
                            ncy,
                            ncy + nControlPointsInput[1] - 1,
                            ncz,
                            ncz + nControlPointsInput[2] - 1,
                            isubd,
                            iteration,
                        )
                    )
                else:
                    pvtkfile.write(
                        '    <Piece Extent="%d %d %d %d 0 0" Source="structuredcp-%d-%d.vtr"/>\n'
                        % (
                            ncx,
                            ncx + nControlPointsInput[0] - 1,
                            ncy,
                            ncy + nControlPointsInput[1] - 1,
                            isubd,
                            iteration,
                        )
                    )
                isubd += 1
                xoff += dxyz[0]
                ncx += nControlPointsInput[0]
            yoff += dxyz[1]
            ncy += nControlPointsInput[1]

        zoff += dxyz[2] if dimension == 3 else 0
        ncz += nControlPointsInput[2] if dimension == 3 else 0

    pvtkfile.write("\n")
    pvtkfile.write("</PRectilinearGrid>\n")
    pvtkfile.write("</VTKFile>\n")

    pvtkfile.close()


def flattenList(t):
    return [item for sublist in t for item in sublist]


def flattenDict(d):
    return reduce(
        lambda new_d, kv: isinstance(kv[1], dict)
        and {**new_d, **flatten(kv[1], kv[0])}
        or {**new_d, kv[0]: kv[1]},
        d.items(),
        {},
    )


def flattenListDict(t):
    ndict = {}
    for item in t:
        ndict.update(item)
    return ndict


# Let do the recursive iterations
# The variable `additiveSchwartz` controls whether we use
# Additive vs Multiplicative Schwartz scheme for DD resolution
####
class InputControlBlock:
    def __init__(self, bid, nCPi, coreb, xb, pl, xl, yl=None, zl=None):
        self.nControlPoints = np.copy(nCPi)

        if useMOABMesh:
            self.mbInterface = core.Core()
            self.scdGrid = ScdInterface(self.mbInterface)

        if dimension == 1:
            self.problemInterface = ProblemSolver1D(
                self, coreb, xb, xl, degree, augmentSpanSpace, useDiagonalBlocks
            )
        elif dimension == 2:
            self.problemInterface = ProblemSolver2D(
                self, coreb, xb, xl, yl, degree, augmentSpanSpace, useDiagonalBlocks
            )
        elif dimension == 3:
            self.problemInterface = ProblemSolver3D(
                self, coreb, xb, xl, yl, zl, degree, augmentSpanSpace, useDiagonalBlocks
            )
        else:
            error("Not implemented")

        self.refSolutionLocal = np.copy(pl)
        self.controlPointData = np.zeros(self.nControlPoints)
        self.weightsData = np.ones(self.nControlPoints)
        self.controlPointBounds = np.array([pl.min(), pl.max()], copy=True)

        self.solutionDecoded = np.zeros(pl.shape)
        self.solutionDecodedOld = np.zeros(pl.shape)

        self.UVW = {"x": [], "y": [], "z": []}
        self.NUVW = {"x": [], "y": [], "z": []}

        # The constraints have a pre-determined ordering
        # 1-Dimension = 0: left, 1: right
        # 2-Dimension = 0: left, 1: right, 2: top, 3: bottom, 4: top-left, 5: top-right, 6: bottom-left, 7: bottom-right
        self.boundaryConstraints = {}
        self.ghostKnots = {}

        self.figHnd = None
        self.figErrHnd = None
        self.figSuffix = ""

        # Convergence related metrics and checks
        self.outerIteration = 0
        self.globalIterationNum = 0
        self.adaptiveIterationNum = 0
        self.globalTolerance = 1e-13

        self.outerIterationConverged = False
        self.decodederrors = np.zeros(2)  # L2 and Linf
        self.errorMetricsL2 = np.zeros((nASMIterations), dtype="float64")
        self.errorMetricsLinf = np.zeros((nASMIterations), dtype="float64")

        self.solutionLocalHistory = []

    def show(self, cp):
        print(
            "Rank: %d, Subdomain %d:" % (commWorld.rank, cp.gid()),
            " Bounds = ",
            self.xbounds,
        )

    def update_bounds(self, cp):
        localExtents[cp.gid()] = self.problemInterface.update_bounds()

    def compute_basis(self, constraints=None):
        # Call the appropriate basis computation function
        self.problemInterface.compute_basis(constraints)

        if useMOABMesh:
            coords = []
            verts = []

            if dimension == 1:

                verts = self.mbInterface.get_entities_by_type(0, types.MBVERTEX)
                if len(verts) == 0:
                    # Now let us generate a MOAB SCD box
                    xc = self.basisFunction["x"].greville()
                    for xi in xc:
                        coords += [xi, 0.0, 0.0]
                    scdbox = self.scdGrid.construct_box(
                        HomCoord([0, 0, 0, 0]), HomCoord([len(xc) - 1, 0, 0, 0]), coords
                    )

            elif dimension == 2:

                verts = self.mbInterface.get_entities_by_type(0, types.MBVERTEX)
                if len(verts) == 0:
                    # Now let us generate a MOAB SCD box
                    xc = self.basisFunction["x"].greville()
                    yc = self.basisFunction["y"].greville()
                    for yi in yc:
                        for xi in xc:
                            coords += [xi, yi, 0.0]
                    scdbox = self.scdGrid.construct_box(
                        HomCoord([0, 0, 0, 0]),
                        HomCoord([len(xc) - 1, len(yc) - 1, 0, 0]),
                        coords,
                    )

            elif dimension == 3:

                verts = self.mbInterface.get_entities_by_type(0, types.MBVERTEX)
                if len(verts) == 0:
                    # Now let us generate a MOAB SCD box
                    xc = self.basisFunction["x"].greville()
                    yc = self.basisFunction["y"].greville()
                    zc = self.basisFunction["y"].greville()
                    for zi in zc:
                        for yi in yc:
                            for xi in xc:
                                coords += [xi, yi, zi]
                    scdbox = self.scdGrid.construct_box(
                        HomCoord([0, 0, 0, 0]),
                        HomCoord([len(xc) - 1, len(yc) - 1, len(zc) - 1, 0]),
                        coords,
                    )

            else:
                error("Invalid dimension")

            print("MOAB structured mesh now has", len(verts), "vertices")

    # ------------------------------------

    def compute_decode_operators(self):
        RN = {"x": [], "y": [], "z": []}
        # compute the decoded operators now
        self.problemInterface.compute_decode_operators(RN)
        return RN

    def decode(self, P, RN):
        return self.problemInterface.decode(P, RN)

    def lsqFit(self):
        return self.problemInterface.lsqFit()

    def output_solution(self, cp):

        if dimension == 1:

            # axHnd = self.figHnd.gca()
            self.pMK = self.decode(self.controlPointData, self.decodeOpXYZ)

            xl = self.xyzCoordLocal["x"].reshape(self.xyzCoordLocal["x"].shape[0], 1)

            plt.subplot(211)
            # Plot the control point solution
            coeffs_x = self.basisFunction["x"].greville()
            plt.plot(
                xl,
                self.pMK,
                linestyle="--",
                lw=2,
                color=["r", "g", "b", "y", "c"][cp.gid() % 5],
                label="Decoded-%d" % (cp.gid() + 1),
            )
            plt.plot(
                coeffs_x,
                self.controlPointData,
                marker="x",
                linestyle="",
                color=["r", "g", "b", "y", "c"][cp.gid() % 5],
                label="Control-%d" % (cp.gid() + 1),
            )

            # Plot the error
            errorDecoded = self.refSolutionLocal.reshape(
                self.refSolutionLocal.shape[0], 1
            ) - self.pMK.reshape(
                self.pMK.shape[0], 1
            )

            plt.subplot(212)
            plt.plot(
                xl,
                errorDecoded,
                # plt.plot(self.xyzCoordLocal['x'],
                #          error,
                linestyle="--",
                color=["r", "g", "b", "y", "c"][cp.gid() % 5],
                lw=2,
                label="Subdomain(%d) Error" % (cp.gid() + 1),
            )

        else:
            if useVTKOutput:

                self.output_vtk(cp)

            if False and cp.gid() == 0:
                self.PlotControlPoints()

    def output_vtk(self, cp):

        from uvw import RectilinearGrid, DataArray
        from uvw.parallel import PRectilinearGrid

        assert useVTKOutput

        self.pMK = self.decode(self.controlPointData, self.decodeOpXYZ)
        errorDecoded = (self.refSolutionLocal - self.pMK) / solutionRange

        locX = []
        locY = []
        locZ = []
        if augmentSpanSpace > 0:
            locX = self.xyzCoordLocal["x"][
                self.corebounds[0][0] : self.corebounds[0][1]
            ]
            locY = self.xyzCoordLocal["y"][
                self.corebounds[1][0] : self.corebounds[1][1]
            ]
            if dimension > 2:
                locZ = self.xyzCoordLocal["z"][
                    self.corebounds[2][0] : self.corebounds[2][1]
                ]

            if dimension == 2:
                coreData = np.ascontiguousarray(
                    self.pMK[
                        self.corebounds[0][0] : self.corebounds[0][1],
                        self.corebounds[1][0] : self.corebounds[1][1],
                    ]
                )
                errorDecoded = np.ascontiguousarray(
                    errorDecoded[
                        self.corebounds[0][0] : self.corebounds[0][1],
                        self.corebounds[1][0] : self.corebounds[1][1],
                    ]
                )

                proc = np.ones((locX.size - 1, locY.size - 1)) * commWorld.Get_rank()
            else:
                coreData = np.ascontiguousarray(
                    self.pMK[
                        self.corebounds[0][0] : self.corebounds[0][1],
                        self.corebounds[1][0] : self.corebounds[1][1],
                        self.corebounds[2][0] : self.corebounds[2][1],
                    ]
                )
                errorDecoded = np.ascontiguousarray(
                    errorDecoded[
                        self.corebounds[0][0] : self.corebounds[0][1],
                        self.corebounds[1][0] : self.corebounds[1][1],
                        self.corebounds[2][0] : self.corebounds[2][1],
                    ]
                )
                proc = (
                    np.ones((locX.size - 1, locY.size - 1, locZ.size - 1))
                    * commWorld.Get_rank()
                )
        else:

            locX = self.xyzCoordLocal["x"]
            locY = self.xyzCoordLocal["y"]
            if dimension > 2:
                locZ = self.xyzCoordLocal["z"]
                proc = (
                    np.ones((locX.size - 1, locY.size - 1, locZ.size - 1))
                    * commWorld.Get_rank()
                )
            else:
                proc = np.ones((locX.size - 1, locY.size - 1)) * commWorld.Get_rank()
            coreData = self.pMK

        if dimension == 2:
            with RectilinearGrid(
                "./structured-%s.vtr" % (self.figSuffix), (locX, locY)
            ) as rect:
                rect.addPointData(DataArray(coreData, range(dimension), "solution"))
                rect.addPointData(DataArray(errorDecoded, range(dimension), "error"))
                rect.addCellData(DataArray(proc, range(dimension), "process"))
        else:
            with RectilinearGrid(
                "./structured-%s.vtr" % (self.figSuffix), (locX, locY, locZ)
            ) as rect:
                rect.addPointData(DataArray(coreData, range(dimension), "solution"))
                rect.addPointData(DataArray(errorDecoded, range(dimension), "error"))
                rect.addCellData(DataArray(proc, range(dimension), "process"))

        cpx = np.array(self.basisFunction["x"].greville())
        cpy = np.array(self.basisFunction["y"].greville())
        if dimension > 2:
            cpz = np.array(self.basisFunction["z"].greville())
            with RectilinearGrid(
                "./structuredcp-%s.vtr" % (self.figSuffix), (cpx, cpy, cpz)
            ) as rectc:
                rectc.addPointData(
                    DataArray(self.controlPointData, range(dimension), "controlpoints")
                )
        else:
            with RectilinearGrid(
                "./structuredcp-%s.vtr" % (self.figSuffix), (cpx, cpy)
            ) as rectc:
                rectc.addPointData(
                    DataArray(self.controlPointData, range(dimension), "controlpoints")
                )

    def PlotControlPoints(self):
        import pyvista as pv
        import numpy as np

        # Create the spatial reference
        grid = pv.UniformGrid()

        cpx = self.basisFunction["x"].greville()
        cpy = self.basisFunction["y"].greville()
        # cpz = np.ones(len(cpx))
        Xi, Yi = np.meshgrid(cpx, cpy)

        print(Xi.shape, Yi.shape, self.controlPointData.shape)

        points = np.c_[
            Xi.reshape(-1),
            Yi.reshape(-1),
            np.ones(self.controlPointData.T.reshape(-1).shape),
        ]

        grid = pv.StructuredGrid()
        grid.points = points
        # set the dimensions
        grid.dimensions = [len(cpx), len(cpy), 1]

        grid["values"] = self.controlPointData.T.reshape(-1)
        # grid.point_array(grid, "ControlPoints") = self.controlPointData.T.reshape(-1)

        # grid = pv.StructuredGrid(Xi, Yi, self.controlPointData)

        # corners = np.stack((Xi, Yi, Zi))
        # # corners = np.stack((cpx, cpy, cpz))
        # corners = corners.transpose()
        # dims = np.asarray((len(cpx), len(cpy), len(cpz)))+1
        # grid = pv.ExplicitStructuredGrid(dims, corners)
        # grid.compute_connectivity()

        # Now plot the grid!
        grid.plot(show_edges=True, show_grid=False, cpos="xy")

        # print some information about the grid to screen
        # print(grid)

        pv.save_meshio("pyvista-out-%s.vtk" % (self.figSuffix), grid)

    def set_fig_handles(self, cp, fig=None, figerr=None, suffix=""):
        self.figHnd = fig
        self.figErrHnd = figerr
        self.figSuffix = suffix

    def print_solution(self, cp):
        # self.pMK = decode(self.refSolutionLocal, self.weightsData, self.Nu, self.Nv)
        #         errorDecoded = self.refSolutionLocal - self.pMK

        print("Domain: ", cp.gid() + 1, "Exact = ", self.refSolutionLocal)
        print(
            "Domain: ",
            cp.gid() + 1,
            "Exact - Decoded = ",
            np.abs(self.refSolutionLocal - self.pMK),
        )

    def send_diy(self, cp):
        self.problemInterface.send_diy(self, cp)
        return

    def recv_diy(self, cp):
        self.problemInterface.recv_diy(self, cp)
        return

    # def send_diy_12d(self, cp):

    #     oddDegree = (degree % 2)
    #     nconstraints = augmentSpanSpace + \
    #         (int(degree/2.0) if not oddDegree else int((degree+1)/2.0))
    #     loffset = degree + 2*augmentSpanSpace
    #     link = cp.link()
    #     for i in range(len(link)):
    #         target = link.target(i)
    #         if len(self.controlPointData):
    #             dir = link.direction(i)
    #             if dir[0] == 0 and dir[1] == 0 and dir[2] == 0:
    #                 continue

    #             # ONLY consider coupling through faces and not through verties
    #             # This means either dir[0] or dir[1] has to be "0" for subdomain coupling to be active
    #             # Hence we only consider 4 neighbor cases, instead of 8.
    #             if dir[0] == 0:  # target is coupled in Y-direction
    #                 if dir[1] > 0:  # target block is above current subdomain
    #                     if verbose:
    #                         print("%d sending to %d" % (cp.gid(), target.gid), ' Top: ',
    #                               self.controlPointData[:, -1:-2-degree-augmentSpanSpace:-1].shape)

    #                     if dimension == 1: cp.enqueue(target, self.controlPointData[-loffset:])
    #                     elif dimension == 2: cp.enqueue(target, self.controlPointData[:, -loffset:])
    #                     # cp.enqueue(target, self.controlPointData)
    #                     cp.enqueue(
    #                         target, self.knotsAdaptive['y'][-1:-2-degree-augmentSpanSpace:-1])

    #                 else:  # target block is below current subdomain
    #                     if verbose:
    #                         print("%d sending to %d" % (cp.gid(), target.gid), ' Bottom: ',
    #                               self.controlPointData[:, 0:1+degree+augmentSpanSpace].shape)

    #                     if dimension == 1: cp.enqueue(target, self.controlPointData[:loffset])
    #                     elif dimension == 2: cp.enqueue(target, self.controlPointData[:, :loffset])
    #                     # cp.enqueue(target, self.controlPointData)
    #                     cp.enqueue(
    #                         target, self.knotsAdaptive['y'][0:1+degree+augmentSpanSpace])

    #             # target is coupled in X-direction
    #             elif dimension == 1 or dir[1] == 0:
    #                 if dir[0] > 0:  # target block is to the right of current subdomain
    #                     if verbose:
    #                         print("%d sending to %d" % (cp.gid(), target.gid), 'Left: ',
    #                               self.controlPointData[-1:-2-degree-augmentSpanSpace:-1, :].shape)

    #                     if dimension == 1: cp.enqueue(target, self.controlPointData[-loffset:])
    #                     elif dimension == 2: cp.enqueue(target, self.controlPointData[-loffset:, :])
    #                     # cp.enqueue(target, self.controlPointData)
    #                     cp.enqueue(
    #                         target, self.knotsAdaptive['x'][-1:-2-degree-augmentSpanSpace:-1])

    #                 else:  # target block is to the left of current subdomain
    #                     if verbose:
    #                         print("%d sending to %d" % (cp.gid(), target.gid), 'Right: ',
    #                               self.controlPointData[degree+augmentSpanSpace::-1, :].shape)

    #                     if dimension == 1: cp.enqueue(target, self.controlPointData[:loffset])
    #                     elif dimension == 2: cp.enqueue(target, self.controlPointData[:loffset, :])
    #                     # cp.enqueue(target, self.controlPointData)
    #                     cp.enqueue(target, self.knotsAdaptive['x'][0:(
    #                         degree+augmentSpanSpace+1)])

    #             else:

    #                 if useDiagonalBlocks and dimension > 1:
    #                     # target block is diagonally top right to current subdomain
    #                     if dir[0] > 0 and dir[1] > 0:

    #                         cp.enqueue(target, self.controlPointData[-loffset:,-loffset:])
    #                         # cp.enqueue(target, self.controlPointData[-1-degree-augmentSpanSpace:, -1-degree-augmentSpanSpace:])
    #                         if verbose:
    #                             print(
    #                                 "%d sending to %d" % (cp.gid(),
    #                                                       target.gid),
    #                                 ' Diagonal = right-top: ', self.controlPointData
    #                                 [-1: -2 - degree - augmentSpanSpace: -1, : 1 + degree +
    #                                  augmentSpanSpace])
    #                     # target block is diagonally top left to current subdomain
    #                     if dir[0] < 0 and dir[1] > 0:
    #                         cp.enqueue(target, self.controlPointData[:loffset:,-loffset:])
    #                         # cp.enqueue(target, self.controlPointData[: 1 + degree + augmentSpanSpace, -1:-2-degree-augmentSpanSpace:-1])
    #                         if verbose:
    #                             print(
    #                                 "%d sending to %d" % (cp.gid(),
    #                                                       target.gid),
    #                                 ' Diagonal = left-top: ', self.controlPointData
    #                                 [-1: -2 - degree - augmentSpanSpace: -1, : 1 + degree +
    #                                  augmentSpanSpace])

    #                     # target block is diagonally left bottom  current subdomain
    #                     if dir[0] < 0 and dir[1] < 0:
    #                         cp.enqueue(target, self.controlPointData[:loffset:,:loffset])
    #                         # cp.enqueue(target, self.controlPointData[-1-degree-augmentSpanSpace:, :1+degree+augmentSpanSpace])

    #                         if verbose:
    #                             print(
    #                                 "%d sending to %d" % (cp.gid(),
    #                                                       target.gid),
    #                                 ' Diagonal = left-bottom: ', self.controlPointData
    #                                 [: 1 + degree + augmentSpanSpace, -1 - degree - augmentSpanSpace:])
    #                     # target block is diagonally right bottom of current subdomain
    #                     if dir[0] > 0 and dir[1] < 0:
    #                         cp.enqueue(target, self.controlPointData[-loffset:,:loffset])
    #                         # cp.enqueue(target, self.controlPointData[:1+degree+augmentSpanSpace, :1+degree+augmentSpanSpace])
    #                         if verbose:
    #                             print(
    #                                 "%d sending to %d" % (cp.gid(),
    #                                                       target.gid),
    #                                 ' Diagonal = right-bottom: ', self.controlPointData
    #                                 [: 1 + degree + augmentSpanSpace, -1 - degree - augmentSpanSpace:])

    #             if dimension == 2 and self.basisFunction['x'] is not None:
    #                 cp.enqueue(target, self.basisFunction['x'].greville())
    #                 cp.enqueue(target, self.basisFunction['y'].greville())

    #     return

    # def recv_diy_12d(self, cp):

    #     # oddDegree = (degree % 2)
    #     # nconstraints = augmentSpanSpace + \
    #     #     (int(degree/2.0) if not oddDegree else int((degree+1)/2.0))

    #     link = cp.link()
    #     for i in range(len(link)):
    #         tgid = link.target(i).gid
    #         dir = link.direction(i)
    #         # print("%d received from %d: %s from direction %s, with sizes %d+%d" % (cp.gid(), tgid, o, dir, pl, tl))

    #         # ONLY consider coupling through faces and not through verties
    #         # This means either dir[0] or dir[1] has to be "0" for subdomain coupling to be active
    #         # Hence we only consider 4 neighbor cases, instead of 8.
    #         if dir[0] == 0 and dir[1] == 0:
    #             continue

    #         if dir[0] == 0:  # target is coupled in Y-direction
    #             if dir[1] > 0:  # target block is above current subdomain
    #                 self.boundaryConstraints['top'] = cp.dequeue(tgid)
    #                 self.ghostKnots['top'] = cp.dequeue(tgid)
    #                 if verbose:
    #                     print("Top: %d received from %d: from direction %s" % (cp.gid(), tgid,
    #                           dir), self.topconstraint.shape, self.topconstraintKnots.shape)

    #                 # if oddDegree:
    #                 #     self.controlPointData[:,-(degree-nconstraints):] = self.boundaryConstraints['top'][:, :degree-nconstraints]
    #                 # else:
    #                 #     self.controlPointData[:,-(degree-nconstraints)+1:] = self.boundaryConstraints['top'][:, :degree-nconstraints]
    #                 if dimension == 2 and self.basisFunction['x'] is not None:
    #                     self.greville['topx'] = cp.dequeue(tgid)
    #                     self.greville['topy'] = cp.dequeue(tgid)

    #             else:  # target block is below current subdomain
    #                 self.boundaryConstraints['bottom'] = cp.dequeue(tgid)
    #                 self.ghostKnots['bottom'] = cp.dequeue(tgid)
    #                 if verbose:
    #                     print("Bottom: %d received from %d: from direction %s" % (
    #                         cp.gid(), tgid, dir), self.bottomconstraint.shape, self.bottomconstraintKnots.shape)

    #                 if dimension == 2 and self.basisFunction['x'] is not None:
    #                     self.greville['bottomx'] = cp.dequeue(tgid)
    #                     self.greville['bottomy'] = cp.dequeue(tgid)

    #                 # if oddDegree:
    #                 #     self.controlPointData[:,:(degree-nconstraints)] = self.boundaryConstraints['bottom'][:, -(degree-nconstraints):]
    #                 # else:
    #                 #     self.controlPointData[:,:(degree-nconstraints)] = self.boundaryConstraints['bottom'][:, -(degree-nconstraints)+1:]

    #         # target is coupled in X-direction
    #         elif dimension == 1 or dir[1] == 0:
    #             if dir[0] < 0:  # target block is to the left of current subdomain
    #                 # print('Right: ', np.array(o[2:pl+2]).reshape(useDerivativeConstraints+1,pll).T)

    #                 self.boundaryConstraints['left'] = cp.dequeue(tgid)
    #                 self.ghostKnots['left'] = cp.dequeue(tgid)
    #                 if verbose:
    #                     print("Left: %d received from %d: from direction %s" % (cp.gid(), tgid,
    #                           dir), self.leftconstraint.shape, self.leftconstraintKnots.shape)

    #                 if dimension == 2 and self.basisFunction['x'] is not None:
    #                     self.greville['leftx'] = cp.dequeue(tgid)
    #                     self.greville['lefty'] = cp.dequeue(tgid)
    #                 # if oddDegree:
    #                 #     self.controlPointData[:(degree-nconstraints),:] = self.boundaryConstraints['left'][-(degree-nconstraints):,:]
    #                 # else:
    #                 #     self.controlPointData[:(degree-nconstraints),] = self.boundaryConstraints['left'][-(degree-nconstraints)+1:,:]

    #             else:  # target block is to right of current subdomain

    #                 self.boundaryConstraints['right'] = cp.dequeue(tgid)
    #                 self.ghostKnots['right'] = cp.dequeue(tgid)
    #                 if verbose:
    #                     print("Right: %d received from %d: from direction %s" % (cp.gid(), tgid,
    #                           dir), self.rightconstraint.shape, self.rightconstraintKnots.shape)

    #                 if dimension == 2 and self.basisFunction['x'] is not None:
    #                     self.greville['rightx'] = cp.dequeue(tgid)
    #                     self.greville['righty'] = cp.dequeue(tgid)
    #                 # if oddDegree:
    #                 #     self.controlPointData[-(degree-nconstraints):,:] = self.boundaryConstraints['right'][:(degree-nconstraints):,:]
    #                 # else:
    #                 #     self.controlPointData[-(degree-nconstraints):,:] = self.boundaryConstraints['right'][:(degree-nconstraints):,:]

    #         else:

    #             if useDiagonalBlocks:
    #                 # 2-Dimension = 0: left, 1: right, 2: top, 3: bottom, 4: top-left, 5: top-right, 6: bottom-left, 7: bottom-right
    #                 # sender block is diagonally right top to  current subdomain
    #                 if dir[0] > 0 and dir[1] > 0:
    #                     self.boundaryConstraints['top-right'] = cp.dequeue(
    #                         tgid)
    #                     if verbose:
    #                         print("Top-right: %d received from %d: from direction %s" %
    #                               (cp.gid(), tgid, dir), self.boundaryConstraints['top-right'].shape)

    #                     if dimension == 2 and self.basisFunction['x'] is not None:
    #                         self.greville['toprightx'] = cp.dequeue(tgid)
    #                         self.greville['toprighty'] = cp.dequeue(tgid)
    #                 # sender block is diagonally left top to current subdomain
    #                 if dir[0] > 0 and dir[1] < 0:
    #                     self.boundaryConstraints['bottom-right'] = cp.dequeue(
    #                         tgid)
    #                     if verbose:
    #                         print("Bottom-right: %d received from %d: from direction %s" %
    #                               (cp.gid(), tgid, dir), self.boundaryConstraints['bottom-right'].shape)

    #                     if dimension == 2 and self.basisFunction['x'] is not None:
    #                         self.greville['bottomrightx'] = cp.dequeue(tgid)
    #                         self.greville['bottomrighty'] = cp.dequeue(tgid)
    #                 # sender block is diagonally left bottom  current subdomain
    #                 if dir[0] < 0 and dir[1] < 0:
    #                     self.boundaryConstraints['bottom-left'] = cp.dequeue(
    #                         tgid)
    #                     if verbose:
    #                         print("Bottom-left: %d received from %d: from direction %s" %
    #                               (cp.gid(), tgid, dir), self.boundaryConstraints['bottom-left'].shape)

    #                     if dimension == 2 and self.basisFunction['x'] is not None:
    #                         self.greville['bottomleftx'] = cp.dequeue(tgid)
    #                         self.greville['bottomlefty'] = cp.dequeue(tgid)
    #                 # sender block is diagonally left to current subdomain
    #                 if dir[0] < 0 and dir[1] > 0:

    #                     self.boundaryConstraints['top-left'] = cp.dequeue(tgid)
    #                     if verbose:
    #                         print("Top-left: %d received from %d: from direction %s" %
    #                               (cp.gid(), tgid, dir), self.boundaryConstraints['top-left'].shape)

    #                     if dimension == 2 and self.basisFunction['x'] is not None:
    #                         self.greville['topleftx'] = cp.dequeue(tgid)
    #                         self.greville['toplefty'] = cp.dequeue(tgid)
    #     return

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
                    e[i - 1, j - 1] = e[i - 2, j - 3] + 1 / (
                        e[i - 1, j - 2] - e[i - 2, j - 2]
                    )
                else:
                    # + 1 / (e[i - 1, j - 2] - e[i - 2, j - 2])
                    e[i - 1, j - 1] = e[i - 2, j - 3]

        er = e[:, 1 : n + 1 : 2]
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

        er = e[:, 1 : n + 1 : 2]
        return er

    def VectorAitken(self, sn):
        v1 = sn[:, 1] - sn[:, 0]
        v1mag = np.linalg.norm(v1, ord=2)
        v2 = sn[:, 0] - 2 * sn[:, 1] + sn[:, 2]
        v2mag = np.linalg.norm(v2, ord=2)
        return sn[:, 0] - (v1mag / v2mag) * v2

    def Aitken(self, sn):
        # return sn[2]
        # return sn[0] - (sn[2] - sn[1])**2/(sn[2] - 2*sn[1] + sn[0])
        if abs(sn[2] - 2 * sn[1] + sn[0]) > 1e-8:
            return sn[0] - (sn[2] - sn[1]) ** 2 / (sn[2] - 2 * sn[1] + sn[0])
        else:
            return sn[2]

    def extrapolate_guess(self, cp, iterationNumber):
        plen = self.controlPointData.shape[0] * self.controlPointData.shape[1]
        self.solutionLocalHistory[:, 1:] = self.solutionLocalHistory[:, :-1]
        self.solutionLocalHistory[:, 0] = np.copy(self.controlPointData).reshape(plen)

        vAcc = []
        if not useAitken:
            if iterationNumber > nWynnEWork:  # For Wynn-E[silon
                vAcc = np.zeros(plen)
                for dofIndex in range(plen):
                    expVal = self.WynnEpsilon(
                        self.solutionLocalHistory[dofIndex, :],
                        math.floor((nWynnEWork - 1) / 2),
                    )
                    vAcc[dofIndex] = expVal[-1, -1]
                print(
                    "Performing scalar Wynn-Epsilon algorithm: Error is ",
                    np.linalg.norm(self.controlPointData.reshape(plen) - vAcc),
                )  # , (self.refSolutionLocal - vAcc))
                self.controlPointData = vAcc[:].reshape(
                    self.controlPointData.shape[0], self.controlPointData.shape[1]
                )

        else:
            if iterationNumber > 3:  # For Aitken acceleration
                vAcc = self.VectorAitken(self.solutionLocalHistory).reshape(
                    self.controlPointData.shape[0], self.controlPointData.shape[1]
                )
                # vAcc = np.zeros(self.controlPointData.shape)
                # for dofIndex in range(len(self.controlPointData)):
                #     vAcc[dofIndex] = self.Aitken(self.solutionLocalHistory[dofIndex, :])
                print(
                    "Performing Aitken Acceleration algorithm: Error is ",
                    np.linalg.norm(self.controlPointData - vAcc),
                )
                self.controlPointData = vAcc[:]

    def initialize_data(self, cp):

        # Subdomain ID: iSubDom = cp.gid()+1

        inc = (self.Dmaxi - self.Dmini) / (self.nControlPoints - degree)
        # print ("self.nInternalKnotSpans = ", self.nInternalKnotSpans, " inc = ", inc)

        # # Generate the knots in X and Y direction
        # tu = np.linspace(self.Dmini[0] + inc[0], self.Dmaxi[0] - inc[0], self.nInternalKnotSpans[0] - 1)
        # tv = np.linspace(self.Dmini[1] + inc[1], self.Dmaxi[1] - inc[1], self.nInternalKnotSpans[1] - 1)
        # self.knotsAdaptive['x'] = np.concatenate(([self.Dmini[0]] * (degree+1), tu, [self.Dmaxi[0]] * (degree+1)))
        # self.knotsAdaptive['y'] = np.concatenate(([self.Dmini[1]] * (degree+1), tv, [self.Dmaxi[1]] * (degree+1)))

        # self.UVW.x = np.linspace(self.Dmini[0], self.Dmaxi[0], self.nPointsPerSubDX)  # self.nPointsPerSubDX, nPointsX
        # self.UVW.y = np.linspace(self.Dmini[1], self.Dmaxi[1], self.nPointsPerSubDY)  # self.nPointsPerSubDY, nPointsY

        tu = np.linspace(
            self.Dmini[0] + inc[0],
            self.Dmaxi[0] - inc[0],
            self.nControlPoints[0] - degree - 1,
        )
        if dimension > 1:
            tv = np.linspace(
                self.Dmini[1] + inc[1],
                self.Dmaxi[1] - inc[1],
                self.nControlPoints[1] - degree - 1,
            )
            if dimension > 2:
                tw = np.linspace(
                    self.Dmini[2] + inc[2],
                    self.Dmaxi[2] - inc[2],
                    self.nControlPoints[2] - degree - 1,
                )

        if nTotalSubDomains > 1 and not fullyPinned:

            if verbose:
                print("Subdomain: ", cp.gid(), " X: ", self.Dmini[0], self.Dmaxi[0])
            if (
                abs(self.Dmaxi[0] - xyzMax[0]) < 1e-12
                and abs(self.Dmini[0] - xyzMin[0]) < 1e-12
            ):
                self.knotsAdaptive["x"] = np.concatenate(
                    ([self.Dmini[0]] * (degree + 1), tu, [self.Dmaxi[0]] * (degree + 1))
                )
                self.isClamped["left"] = self.isClamped["right"] = True
            else:
                if abs(self.Dmaxi[0] - xyzMax[0]) < 1e-12:
                    self.knotsAdaptive["x"] = np.concatenate(
                        ([self.Dmini[0]] * (1), tu, [self.Dmaxi[0]] * (degree + 1))
                    )
                    self.isClamped["right"] = True

                else:
                    if abs(self.Dmini[0] - xyzMin[0]) < 1e-12:
                        self.knotsAdaptive["x"] = np.concatenate(
                            ([self.Dmini[0]] * (degree + 1), tu, [self.Dmaxi[0]] * (1))
                        )
                        self.isClamped["left"] = True

                    else:
                        self.knotsAdaptive["x"] = np.concatenate(
                            ([self.Dmini[0]] * (1), tu, [self.Dmaxi[0]] * (1))
                        )
                        self.isClamped["left"] = self.isClamped["right"] = False

            if dimension == 1:
                if verbose:
                    print(
                        "Subdomain: ",
                        cp.gid(),
                        " clamped ? ",
                        self.isClamped["left"],
                        self.isClamped["right"],
                    )

            if dimension > 1:
                if verbose:
                    print("Subdomain: ", cp.gid(), " Y: ", self.Dmini[1], self.Dmaxi[1])
                if (
                    abs(self.Dmaxi[1] - xyzMax[1]) < 1e-12
                    and abs(self.Dmini[1] - xyzMin[1]) < 1e-12
                ):
                    print(
                        "Subdomain: ",
                        cp.gid(),
                        " checking top and bottom Y: ",
                        self.Dmaxi[1],
                        xyzMax[1],
                        abs(self.Dmaxi[1] - xyzMax[1]),
                    )
                    self.knotsAdaptive["y"] = np.concatenate(
                        (
                            [self.Dmini[1]] * (degree + 1),
                            tv,
                            [self.Dmaxi[1]] * (degree + 1),
                        )
                    )
                    self.isClamped["top"] = self.isClamped["bottom"] = True
                else:

                    if abs(self.Dmaxi[1] - xyzMax[1]) < 1e-12:
                        print(
                            "Subdomain: ",
                            cp.gid(),
                            " checking top Y: ",
                            self.Dmaxi[1],
                            xyzMax[1],
                            abs(self.Dmaxi[1] - xyzMax[1]),
                        )
                        self.knotsAdaptive["y"] = np.concatenate(
                            ([self.Dmini[1]] * (1), tv, [self.Dmaxi[1]] * (degree + 1))
                        )
                        self.isClamped["top"] = True

                    else:

                        if verbose:
                            print(
                                "Subdomain: ",
                                cp.gid(),
                                " checking bottom Y: ",
                                self.Dmini[1],
                                xyzMin[1],
                                abs(self.Dmini[1] - xyzMin[1]),
                            )
                        if abs(self.Dmini[1] - xyzMin[1]) < 1e-12:
                            self.knotsAdaptive["y"] = np.concatenate(
                                (
                                    [self.Dmini[1]] * (degree + 1),
                                    tv,
                                    [self.Dmaxi[1]] * (1),
                                )
                            )
                            self.isClamped["bottom"] = True

                        else:
                            self.knotsAdaptive["y"] = np.concatenate(
                                ([self.Dmini[1]] * (1), tv, [self.Dmaxi[1]] * (1))
                            )

                            self.isClamped["top"] = self.isClamped["bottom"] = False

            if dimension > 2:
                if verbose:
                    print("Subdomain: ", cp.gid(), " Z: ", self.Dmini[2], self.Dmaxi[2])
                if (
                    abs(self.Dmaxi[2] - xyzMax[2]) < 1e-12
                    and abs(self.Dmini[2] - xyzMin[2]) < 1e-12
                ):
                    print(
                        "Subdomain: ",
                        cp.gid(),
                        " checking top and bottom Y: ",
                        self.Dmaxi[2],
                        xyzMax[2],
                        abs(self.Dmaxi[2] - xyzMax[2]),
                    )
                    self.knotsAdaptive["z"] = np.concatenate(
                        (
                            [self.Dmini[2]] * (degree + 1),
                            tw,
                            [self.Dmaxi[2]] * (degree + 1),
                        )
                    )
                    self.isClamped["up"] = self.isClamped["down"] = True
                else:

                    if abs(self.Dmaxi[2] - xyzMax[2]) < 1e-12:
                        print(
                            "Subdomain: ",
                            cp.gid(),
                            " checking top Z: ",
                            self.Dmaxi[2],
                            xyzMax[2],
                            abs(self.Dmaxi[2] - xyzMax[2]),
                        )
                        self.knotsAdaptive["z"] = np.concatenate(
                            ([self.Dmini[2]] * (1), tw, [self.Dmaxi[2]] * (degree + 1))
                        )
                        self.isClamped["up"] = True

                    else:

                        if verbose:
                            print(
                                "Subdomain: ",
                                cp.gid(),
                                " checking bottom Z: ",
                                self.Dmini[2],
                                xyzMin[2],
                                abs(self.Dmini[2] - xyzMin[2]),
                            )
                        if abs(self.Dmini[2] - xyzMin[2]) < 1e-12:
                            self.knotsAdaptive["z"] = np.concatenate(
                                (
                                    [self.Dmini[2]] * (degree + 1),
                                    tw,
                                    [self.Dmaxi[2]] * (1),
                                )
                            )
                            self.isClamped["down"] = True

                        else:
                            self.knotsAdaptive["z"] = np.concatenate(
                                ([self.Dmini[2]] * (1), tw, [self.Dmaxi[2]] * (1))
                            )

                            self.isClamped["up"] = self.isClamped["down"] = False

                if verbose:
                    print(
                        "Subdomain: ",
                        cp.gid(),
                        " clamped ? ",
                        self.isClamped["left"],
                        self.isClamped["right"],
                        self.isClamped["top"],
                        self.isClamped["bottom"],
                    )
                    if dimension == 3:
                        print(
                            "Subdomain: ",
                            cp.gid(),
                            self.isClamped["up"],
                            self.isClamped["down"],
                        )

        else:
            self.knotsAdaptive["x"] = np.concatenate(
                ([self.Dmini[0]] * (degree + 1), tu, [self.Dmaxi[0]] * (degree + 1))
            )
            self.isClamped["left"] = self.isClamped["right"] = True

            if dimension > 1:
                self.knotsAdaptive["y"] = np.concatenate(
                    ([self.Dmini[1]] * (degree + 1), tv, [self.Dmaxi[1]] * (degree + 1))
                )
                self.isClamped["top"] = self.isClamped["bottom"] = True

            if dimension > 2:
                self.knotsAdaptive["z"] = np.concatenate(
                    ([self.Dmini[2]] * (degree + 1), tw, [self.Dmaxi[2]] * (degree + 1))
                )
                self.isClamped["up"] = self.isClamped["down"] = True

        # self.UVW['x'] = np.linspace(
        #     self.xyzCoordLocal['x'][0], self.xyzCoordLocal['x'][-1], self.nPointsPerSubD[0])
        self.UVW["x"] = self.xyzCoordLocal["x"]
        if dimension > 1:
            # self.UVW['y'] = np.linspace(
            #     self.xyzCoordLocal['y'][0], self.xyzCoordLocal['y'][-1], self.nPointsPerSubD[1])
            self.UVW["y"] = self.xyzCoordLocal["y"]
        if dimension > 2:
            # self.UVW['y'] = np.linspace(
            #     self.xyzCoordLocal['y'][0], self.xyzCoordLocal['y'][-1], self.nPointsPerSubD[1])
            self.UVW["z"] = self.xyzCoordLocal["z"]

        if debugProblem:
            if not self.isClamped["left"]:
                self.refSolutionLocal[:, 0] = cp.gid() - 1
            if not self.isClamped["right"]:
                self.refSolutionLocal[:, -1] = cp.gid() + 1
            if not self.isClamped["top"]:
                self.refSolutionLocal[-1, :] = cp.gid() + nSubDomainsX
            if not self.isClamped["bottom"]:
                self.refSolutionLocal[0, :] = cp.gid() - nSubDomainsX

    def augment_spans(self, cp):

        if fullyPinned:
            return

        if verbose:
            print(
                "augment_spans:",
                cp.gid(),
                "Number of control points = ",
                self.nControlPoints,
            )
            if dimension == 1:
                print(
                    "Subdomain -- ",
                    cp.gid() + 1,
                    ": before Shapes: ",
                    self.refSolutionLocal.shape,
                    self.weightsData.shape,
                    self.knotsAdaptive["x"],
                )
            else:
                print(
                    "Subdomain -- ",
                    cp.gid() + 1,
                    ": before Shapes: ",
                    self.refSolutionLocal.shape,
                    self.weightsData.shape,
                    self.knotsAdaptive["x"],
                    self.knotsAdaptive["y"],
                )

        if not self.isClamped["left"]:  # Pad knot spans from the left of subdomain
            if verbose:
                print(
                    "\tSubdomain -- ",
                    cp.gid() + 1,
                    ": left ghost: ",
                    self.ghostKnots["left"],
                )
            self.knotsAdaptive["x"] = np.concatenate(
                (self.ghostKnots["left"][-1:0:-1], self.knotsAdaptive["x"])
            )

        if not self.isClamped["right"]:  # Pad knot spans from the right of subdomain
            if verbose:
                print(
                    "\tSubdomain -- ",
                    cp.gid() + 1,
                    ": right ghost: ",
                    self.ghostKnots["right"],
                )
            self.knotsAdaptive["x"] = np.concatenate(
                (self.knotsAdaptive["x"], self.ghostKnots["right"][1:])
            )

        if dimension > 1:
            # Pad knot spans from the left of subdomain
            if not self.isClamped["top"]:
                if verbose:
                    print(
                        "\tSubdomain -- ",
                        cp.gid() + 1,
                        ": top ghost: ",
                        self.ghostKnots["top"],
                    )
                self.knotsAdaptive["y"] = np.concatenate(
                    (self.knotsAdaptive["y"], self.ghostKnots["top"][1:])
                )

            # Pad knot spans from the right of subdomain
            if not self.isClamped["bottom"]:
                if verbose:
                    print(
                        "\tSubdomain -- ",
                        cp.gid() + 1,
                        ": bottom ghost: ",
                        self.ghostKnots["bottom"],
                    )
                self.knotsAdaptive["y"] = np.concatenate(
                    (self.ghostKnots["bottom"][-1:0:-1], self.knotsAdaptive["y"])
                )

        if dimension > 2:
            # Pad knot spans from the left of subdomain
            if not self.isClamped["up"]:
                if verbose or True:
                    print(
                        "\tSubdomain -- ",
                        cp.gid() + 1,
                        ": up ghost: ",
                        self.ghostKnots["up"],
                    )
                self.knotsAdaptive["z"] = np.concatenate(
                    (self.knotsAdaptive["z"], self.ghostKnots["up"][1:])
                )

            # Pad knot spans from the right of subdomain
            if not self.isClamped["down"]:
                if verbose or True:
                    print(
                        "\tSubdomain -- ",
                        cp.gid() + 1,
                        ": down ghost: ",
                        self.ghostKnots["down"],
                    )
                self.knotsAdaptive["z"] = np.concatenate(
                    (self.ghostKnots["down"][-1:0:-1], self.knotsAdaptive["z"])
                )

        if verbose:
            if dimension == 1:
                print(
                    "Subdomain -- ",
                    cp.gid() + 1,
                    ": after Shapes: ",
                    self.refSolutionLocal.shape,
                    self.weightsData.shape,
                    self.knotsAdaptive["x"],
                )
            else:
                print(
                    "Subdomain -- ",
                    cp.gid() + 1,
                    ": after Shapes: ",
                    self.refSolutionLocal.shape,
                    self.weightsData.shape,
                    self.knotsAdaptive["x"],
                    self.knotsAdaptive["y"],
                )

            print(
                "augment_spans:",
                cp.gid(),
                "Number of control points = ",
                self.nControlPoints,
            )

    def augment_inputdata(self, cp):

        if fullyPinned:
            return

        postol = 1e-10
        if verbose:
            print(
                "augment_inputdata:",
                cp.gid(),
                "Number of control points = ",
                self.nControlPoints,
            )

        if verbose:
            if dimension == 1:
                print(
                    "Subdomain -- {0}: before augment -- bounds = {1}, {2}, shape = {3}, knots = {4}, {5}".format(
                        cp.gid() + 1,
                        self.xyzCoordLocal["x"][0],
                        self.xyzCoordLocal["x"][-1],
                        self.xyzCoordLocal["x"].shape,
                        self.knotsAdaptive["x"][degree],
                        self.knotsAdaptive["x"][-degree],
                    )
                )
            else:
                print(
                    "Subdomain -- {0}: before augment -- bounds = {1}, {2}, {3}, {4}, shape = {5}, {6}, knots = {7}, {8}, {9}, {10}".format(
                        cp.gid() + 1,
                        self.xyzCoordLocal["x"][0],
                        self.xyzCoordLocal["x"][-1],
                        self.xyzCoordLocal["y"][0],
                        self.xyzCoordLocal["y"][-1],
                        self.xyzCoordLocal["x"].shape,
                        self.xyzCoordLocal["y"].shape,
                        self.knotsAdaptive["x"][degree],
                        self.knotsAdaptive["x"][-degree],
                        self.knotsAdaptive["y"][degree],
                        self.knotsAdaptive["y"][-degree],
                    )
                )

        # print('Knots: ', self.knotsAdaptive['x'], self.knotsAdaptive['y'])
        # locX = sp.BSplineBasis(order=degree+1, knots=self.knotsAdaptive['x']).greville()
        # xCP = [locX[0], locX[-1]]

        lboundXYZ = np.zeros(dimension, dtype=int)
        uboundXYZ = np.zeros(dimension, dtype=int)
        for idir in range(dimension):
            cDirection = directions[idir]

            print("Checking direction ", cDirection)

            xyzCP = [
                self.knotsAdaptive[cDirection][degree],
                self.knotsAdaptive[cDirection][-degree - 1],
            ]

            indicesXYZ = np.where(
                np.logical_and(
                    coordinates[cDirection] >= xyzCP[0] - postol,
                    coordinates[cDirection] <= xyzCP[1] + postol,
                )
            )

            lboundXYZ[idir] = indicesXYZ[0][0]
            # indicesX[0][-1] < len(coordinates[cDirection])
            uboundXYZ[idir] = min(indicesXYZ[0][-1] + 1, len(coordinates[cDirection]))
            if (
                (idir == 0 and self.isClamped["left"] and self.isClamped["right"])
                or (idir == 1 and self.isClamped["top"] and self.isClamped["bottom"])
                or (idir == 2 and self.isClamped["up"] and self.isClamped["down"])
            ):
                self.xyzCoordLocal[cDirection] = coordinates[cDirection][:]
            elif (
                (idir == 0 and self.isClamped["left"])
                or (idir == 1 and self.isClamped["bottom"])
                or (idir == 2 and self.isClamped["down"])
            ):
                self.xyzCoordLocal[cDirection] = coordinates[cDirection][
                    : uboundXYZ[idir]
                ]
            elif (
                (idir == 0 and self.isClamped["right"])
                or (idir == 1 and self.isClamped["top"])
                or (idir == 2 and self.isClamped["up"])
            ):
                self.xyzCoordLocal[cDirection] = coordinates[cDirection][
                    lboundXYZ[idir] :
                ]
            else:
                self.xyzCoordLocal[cDirection] = coordinates[cDirection][
                    lboundXYZ[idir] : uboundXYZ[idir]
                ]
            print(
                "%s bounds: " % (cDirection),
                self.xyzCoordLocal[cDirection][0],
                self.xyzCoordLocal[cDirection][-1],
                xyzCP,
            )


        # Store the core indices before augment
        cindicesX = np.array(
            np.where(
                np.logical_and(
                    self.xyzCoordLocal["x"]
                    >= coordinates["x"][self.xbounds[0]] - postol,
                    self.xyzCoordLocal["x"]
                    <= coordinates["x"][self.xbounds[1]] + postol,
                )
            )
        )
        if dimension > 1:
            cindicesY = np.array(
                np.where(
                    np.logical_and(
                        self.xyzCoordLocal["y"]
                        >= coordinates["y"][self.xbounds[2]] - postol,
                        self.xyzCoordLocal["y"]
                        <= coordinates["y"][self.xbounds[3]] + postol,
                    )
                )
            )
            if dimension > 2:
                cindicesZ = np.array(
                    np.where(
                        np.logical_and(
                            self.xyzCoordLocal["z"]
                            >= coordinates["z"][self.xbounds[4]] - postol,
                            self.xyzCoordLocal["z"]
                            <= coordinates["z"][self.xbounds[5]] + postol,
                        )
                    )
                )
                self.corebounds = [
                    [
                        cindicesX[0][0],
                        len(self.xyzCoordLocal["x"])
                        if self.isClamped["right"]
                        else cindicesX[0][-1] + 1,
                    ],
                    [
                        cindicesY[0][0],
                        len(self.xyzCoordLocal["y"])
                        if self.isClamped["top"]
                        else cindicesY[0][-1] + 1,
                    ],
                    [
                        cindicesZ[0][0],
                        len(self.xyzCoordLocal["z"])
                        if self.isClamped["up"]
                        else cindicesZ[0][-1] + 1,
                    ],
                ]
            else:
                self.corebounds = [
                    [
                        cindicesX[0][0],
                        len(self.xyzCoordLocal["x"])
                        if self.isClamped["right"]
                        else cindicesX[0][-1] + 1,
                    ],
                    [
                        cindicesY[0][0],
                        len(self.xyzCoordLocal["y"])
                        if self.isClamped["top"]
                        else cindicesY[0][-1] + 1,
                    ],
                ]

        else:
            self.corebounds = [
                [
                    cindicesX[0][0],
                    len(self.xyzCoordLocal["x"])
                    if self.isClamped["right"]
                    else cindicesX[0][-1] + 1,
                ]
            ]

        if verbose:
            print("self.corebounds = ", self.xbounds, self.corebounds)

        # self.UVW['x'] = self.xyzCoordLocal['x'][self.corebounds[0][0]:self.corebounds[0][1]] / (self.xyzCoordLocal['x'][self.corebounds[0][1]] - self.xyzCoordLocal['x'][self.corebounds[0][0]])
        # if dimension > 1:
        #     self.UVW['y'] = self.xyzCoordLocal['y'][self.corebounds[1][0]:self.corebounds[1][1]] / (self.xyzCoordLocal['y'][self.corebounds[1][1]] - self.xyzCoordLocal['y'][self.corebounds[1][0]])

        if dimension == 1:
            if closedFormFunctional:
                self.refSolutionLocal = solution(self.xyzCoordLocal["x"])
            else:
                self.refSolutionLocal = solution[lboundXYZ[0] : uboundXYZ[0]]
            self.refSolutionLocal = self.refSolutionLocal.reshape(
                (len(self.refSolutionLocal), 1)
            )
        elif dimension == 2:
            if closedFormFunctional:
                X, Y = np.meshgrid(
                        self.xyzCoordLocal["x"], self.xyzCoordLocal["y"], indexing="ij"
                    )
                self.refSolutionLocal = solution(X, Y)
            else:
                self.refSolutionLocal = solution[
                    lboundXYZ[0] : uboundXYZ[0], lboundXYZ[1] : uboundXYZ[1]
            ]
        else:
            if closedFormFunctional:
                X, Y, Z = np.meshgrid(
                        self.xyzCoordLocal["x"], self.xyzCoordLocal["y"], self.xyzCoordLocal["z"], indexing="ij"
                    )
                self.refSolutionLocal = solution(X, Y, Z)
            else:
                self.refSolutionLocal = solution[
                    lboundXYZ[0] : uboundXYZ[0],
                    lboundXYZ[1] : uboundXYZ[1],
                    lboundXYZ[2] : uboundXYZ[2],
                ]

        for idir in range(dimension):
            cDirection = directions[idir]
            self.UVW[cDirection] = self.xyzCoordLocal[cDirection]

        if verbose:
            if dimension == 1:
                print(
                    "Subdomain -- {0}: after augment -- bounds = {1}, {2}, shape = {3}, knots = {4}, {5}".format(
                        cp.gid() + 1,
                        self.xyzCoordLocal["x"][0],
                        self.xyzCoordLocal["x"][-1],
                        self.xyzCoordLocal["x"].shape,
                        self.knotsAdaptive["x"][degree],
                        self.knotsAdaptive["x"][-degree],
                    )
                )
            else:
                print(
                    "Subdomain -- {0}: after augment -- bounds = {1}, {2}, {3}, {4}, shape = {5}, {6}, knots = {7}, {8}, {9}, {10}".format(
                        cp.gid() + 1,
                        self.xyzCoordLocal["x"][0],
                        self.xyzCoordLocal["x"][-1],
                        self.xyzCoordLocal["y"][0],
                        self.xyzCoordLocal["y"][-1],
                        self.xyzCoordLocal["x"].shape,
                        self.xyzCoordLocal["y"].shape,
                        self.knotsAdaptive["x"][degree],
                        self.knotsAdaptive["x"][-degree],
                        self.knotsAdaptive["y"][degree],
                        self.knotsAdaptive["y"][-degree],
                    )
                )

        # self.nControlPoints += augmentSpanSpace
        self.nControlPoints[0] += (
            augmentSpanSpace if not self.isClamped["left"] else 0
        ) + (augmentSpanSpace if not self.isClamped["right"] else 0)
        if dimension > 1:
            self.nControlPoints[1] += (
                augmentSpanSpace if not self.isClamped["top"] else 0
            ) + (augmentSpanSpace if not self.isClamped["bottom"] else 0)
            if dimension > 2:
                self.nControlPoints[2] += (
                    augmentSpanSpace if not self.isClamped["up"] else 0
                ) + (augmentSpanSpace if not self.isClamped["down"] else 0)

        self.controlPointData = np.zeros(self.nControlPoints)
        self.weightsData = np.ones(self.nControlPoints)
        self.solutionDecoded = np.zeros(self.refSolutionLocal.shape)
        self.solutionDecodedOld = np.zeros(self.refSolutionLocal.shape)

        print(
            "augment_inputdata:",
            cp.gid(),
            "Number of control points = ",
            self.nControlPoints,
        )

    def LSQFit_NonlinearOptimize(self, idom, degree, constraints=None):

        solution = []

        # Initialize relevant data
        if constraints is not None:
            initSol = np.copy(constraints)

        else:
            print("Constraints are all null. Solving unconstrained.")
            initSol = np.ones_like(self.controlPointData)

        # Compute hte linear operators needed to compute decoded residuals
        # self.Nu = basis(self.UVW.x[np.newaxis,:],degree,Tunew[:,np.newaxis]).T
        # self.Nv = basis(self.UVW.y[np.newaxis,:],degree,Tvnew[:,np.newaxis]).T
        # self.decodeOpXYZ = compute_decode_operators(self.NUVW)

        initialDecodedError = 0.0

        # Compute the residual as sum of two components
        # 1. The decoded error evaluated at P
        # 2. A penalized domain boundary constraint component

        def residual2DRev(Pin, printVerbose=False):

            bc_penalty = 1e2
            bc_norm = 0.0
            P = np.array(Pin.reshape(self.controlPointData.shape), copy=True)

            oddDegree = degree % 2
            nconstraints = augmentSpanSpace + (
                int(degree / 2.0) if not oddDegree else int((degree + 1) / 2.0)
            )

            # Residuals are in the decoded space - so direct way to constrain the boundary data
            decoded = self.decode(P, self.decodeOpXYZ)
            residual_decoded = (self.refSolutionLocal - decoded) / solutionRange

            residual_encoded = np.matmul(
                self.decodeOpXYZ["x"].T,
                np.matmul(residual_decoded, self.decodeOpXYZ["y"]),
            )

            if not self.isClamped["left"]:
                bc_norm += np.sum(residual_encoded[:nconstraints, :] ** 2) / len(
                    residual_encoded[0, :]
                )
            if not self.isClamped["right"]:
                bc_norm += np.sum(residual_encoded[-nconstraints - 1 :, :] ** 2) / len(
                    residual_encoded[0, :]
                )
            if not self.isClamped["bottom"]:
                bc_norm += np.sum(residual_encoded[:, :nconstraints] ** 2) / len(
                    residual_encoded[:, 0]
                )
            if not self.isClamped["top"]:
                bc_norm += np.sum(residual_encoded[:, -nconstraints - 1 :] ** 2) / len(
                    residual_encoded[:, 0]
                )

            residual_vec_encoded = residual_encoded.reshape(-1)
            net_residual_norm = (
                np.sum(residual_vec_encoded**2) / len(residual_vec_encoded)
            ) + bc_penalty * np.sqrt(bc_norm / 4)
            if type(Pin) is not np.numpy_boxes.ArrayBox:
                print("Residual = ", net_residual_norm)

            return net_residual_norm

        def residual2D(Pin):

            P = np.array(Pin.reshape(self.controlPointData.shape), copy=True)

            decoded_residual_norm = 0
            bc_penalty = 0

            # Residuals are in the decoded space - so direct way to constrain the boundary data
            decoded = self.decode(P, self.decodeOpXYZ)
            residual_decoded = (self.refSolutionLocal - decoded) / solutionRange

            if augmentSpanSpace > 0:
                # residual_decoded[self.corebounds[0][0], :] *= bc_penalty
                # residual_decoded[self.corebounds[0][1]-1, :] *= bc_penalty
                # residual_decoded[:, self.corebounds[1][0]] *= bc_penalty
                # residual_decoded[:, self.corebounds[1][1]-1] *= bc_penalty
                residual_decoded = residual_decoded[
                    self.corebounds[0][0] : self.corebounds[0][1],
                    self.corebounds[1][0] : self.corebounds[1][1],
                ]
                # residual_decoded[0, :] *= bc_penalty
                # residual_decoded[-1, :] *= bc_penalty
                # residual_decoded[:, 0] *= bc_penalty
                # residual_decoded[:, -1] *= bc_penalty

            residual_vec_decoded = residual_decoded.reshape(-1)
            decoded_residual_norm_a = np.sqrt(
                np.sum(residual_vec_decoded**2) / len(residual_vec_decoded)
            )

            tvector = residual_decoded[0, :].reshape(-1)
            tvector = np.concatenate(
                (tvector, residual_decoded[-1, :].reshape(-1)), axis=0
            )
            tvector = np.concatenate((tvector, residual_decoded[:, 0].T), axis=0)
            residual_decoded = np.concatenate(
                (tvector, residual_decoded[:, -1].T), axis=0
            )
            decoded_residual_norm_b = np.sqrt(
                np.sum(residual_decoded**2) / len(residual_decoded)
            )

            # if type(residual_decoded) is np.numpy_boxes.ArrayBox:
            #     decoded_residual_norm = np.linalg.norm(residual_decoded._value, ord=2)
            # else:
            #     decoded_residual_norm = np.linalg.norm(residual_decoded, ord=2)

            # return decoded_residual_norm

            #             res1 = np.dot(res_dec, self.Nv)
            #             residual = np.dot(self.Nu.T, res1).reshape(self.Nu.shape[1]*self.Nv.shape[1])

            # compute the net residual norm that includes the decoded error in the current subdomain and the penalized
            # constraint error of solution on the subdomain interfaces
            net_residual_norm = decoded_residual_norm_a + bc_penalty * (
                decoded_residual_norm_b
            )

            # bc_penalty = 1e7
            # BCPenalty = np.ones(residual_decoded.shape)

            # # left: BCPenalty[:nconstraints, :]
            # if not self.isClamped['left']: BCPenalty[:self.corebounds[0][0], :] = bc_penalty
            # # right: BCPenalty[-nconstraints:, :]
            # if not self.isClamped['right']:
            #     BCPenalty[self.corebounds[0][1]-1:, :] = bc_penalty
            # # top: BCPenalty[:, -nconstraints:]
            # if not self.isClamped['top']:
            #     BCPenalty[:, self.corebounds[1][1]-1:] = bc_penalty
            # # bottom: BCPenalty[:, : nconstraints]
            # if not self.isClamped['bottom']:
            #     BCPenalty[:, : self.corebounds[1][0]] = bc_penalty

            # penalized_residual = np.multiply(BCPenalty, residual_decoded).reshape(-1)
            # decoded_residual_norm = np.sqrt(np.sum(penalized_residual**2)/len(penalized_residual))
            # net_residual_norm = decoded_residual_norm

            if type(Pin) is not np.numpy_boxes.ArrayBox:
                # print('Residual = ', net_residual_norm, ' and decoded = ', decoded_residual_norm, ', constraint = ',
                #       constrained_residual_norm, ', diagonal = ', diagonal_boundary_residual_norm if useDiagonalBlocks else 0)
                # print('Constraint errors = ', ltn, rtn,
                #       tpn, btn, constrained_residual_norm)
                print("Residual = ", net_residual_norm)
                # if useDiagonalBlocks:
                #     print('Constraint diagonal errors = ', topleftBndErr, toprightBndErr, bottomleftBndErr,
                #           bottomrightBndErr, diagonal_boundary_residual_norm)

            return net_residual_norm

        def residual_general(Pin, printVerbose=False):

            P = np.array(Pin.reshape(self.controlPointData.shape), copy=True)

            # Residuals are in the decoded space - so direct way to constrain the boundary data
            decoded = self.decode(P, self.decodeOpXYZ)
            residual_decoded = (self.refSolutionLocal - decoded) / solutionRange

            # residual_encoded = np.matmul(self.decodeOpXYZ['x'].T, np.matmul(
            #     self.decodeOpXYZ['x'].T, np.matmul(residual_decoded, self.decodeOpXYZ['z'])))

            net_residual_vec = residual_decoded.reshape(-1)
            net_residual_norm = np.sqrt(
                np.sum(net_residual_vec**2) / len(net_residual_vec)
            )

            if type(Pin) is not np.numpy_boxes.ArrayBox:
                print("Residual = ", net_residual_norm)

            return net_residual_norm

        # Set a function handle to the appropriate residual evaluator
        # residualFunction = None
        # if dimension == 1:
        #     residualFunction = residual1D
        # elif dimension == 2:
        #     residualFunction = residual2DRev  # residual2DRev
        # else:
        #     residualFunction = residual3D

        residualFunction = residual_general

        def print_iterate(P):
            res = residualFunction(P, printVerbose=True)
            # print('NLConstrained residual vector norm: ', np.linalg.norm(res, ord=2))
            self.globalIterationNum += 1
            return False

        # Use automatic-differentiation to compute the Jacobian value for minimizer
        def jacobian(P):
            #             if jacobian_const is None:
            #                 jacobian_const = egrad(residual)(P)

            # Create a gradient function to pass to the minimizer
            jacobian = egrad(residualFunction)(P, printVerbose=False)
            #             jacobian = jacobian_const
            return jacobian

        # Now invoke the solver and compute the constrained solution
        if constraints is None and not alwaysSolveConstrained:
            solution = self.lsqFit()
            print(
                "LSQFIT solution: min = ", np.min(solution), "max = ", np.max(solution)
            )
        else:

            if constraints is None and alwaysSolveConstrained:
                initSol = self.lsqFit()

            oddDegree = degree % 2
            # alpha = 0.5 if dimension == 2 or oddDegree else 0.0
            alpha = 0.5
            beta = 0.0
            localAssemblyWeights = np.zeros(initSol.shape)
            localBCAssembly = np.zeros(initSol.shape)
            if dimension == 1:
                freeBounds = [0, len(localBCAssembly[:, 0])]
            elif dimension == 2:
                freeBounds = [
                    0,
                    len(localBCAssembly[:, 0]),
                    0,
                    len(localBCAssembly[0, :]),
                ]
            else:
                freeBounds = [
                    0,
                    len(localBCAssembly[:, 0, :]),
                    0,
                    len(localBCAssembly[0, :, :]),
                    0,
                    len(localBCAssembly[:, :, 0]),
                ]
            print("Initial calculation")
            # Lets update our initial solution with constraints
            if constraints is not None and len(constraints) > 0:

                initSol = self.problemInterface.initialize_solution(
                    self, initSol, degree, augmentSpanSpace, fullyPinned
                )

                # if dimension < 3:
                #     initSol = self.problemInterface.initialize_solution(self, initSol, degree, augmentSpanSpace, fullyPinned)
                # else: #dimension == 3:
                #     if fullyPinned:
                #         # First update hte control point vector with constraints for supporting points
                #         if 'left' in self.boundaryConstraints:
                #             initSol[:, :, 0] += self.boundaryConstraints['left'][:, :, -1]
                #             localAssemblyWeights[0, :, :] += 1.0
                #         if 'right' in self.boundaryConstraints:
                #             initSol[:, :, -1] += self.boundaryConstraints['right'][:, :, 0]
                #             localAssemblyWeights[-1, :, :] += 1.0
                #         if 'top' in self.boundaryConstraints:
                #             initSol[:, -1, :] += self.boundaryConstraints['top'][:, 0, :]
                #             localAssemblyWeights[:, -1, :] += 1.0
                #         if 'bottom' in self.boundaryConstraints:
                #             initSol[:, 0, :] += self.boundaryConstraints['bottom'][:, -1, :]
                #             localAssemblyWeights[:, 0, :] += 1.0
                #         if 'up' in self.boundaryConstraints:
                #             initSol[:, :, -1] += self.boundaryConstraints['up'][:, :, 0]
                #             localAssemblyWeights[:, :, -1] += 1.0
                #         if 'down' in self.boundaryConstraints:
                #             initSol[:, :, 0] += self.boundaryConstraints['down'][:, :, -1]
                #             localAssemblyWeights[:, :, 0] += 1.0

                #         if 'top-left' in self.boundaryConstraints:
                #             initSol[0, -1, :] += self.boundaryConstraints['top-left'][-1, 0, :]
                #             localAssemblyWeights[0, -1, :] += 1.0
                #         if 'bottom-right' in self.boundaryConstraints:
                #             initSol[-1, 0, :] = self.boundaryConstraints['bottom-right'][0, -1, :]
                #             localAssemblyWeights[-1, 0, :] += 1.0
                #         if 'bottom-left' in self.boundaryConstraints:
                #             initSol[0, 0, :] = self.boundaryConstraints['bottom-left'][-1, -1, :]
                #             localAssemblyWeights[0, 0, :] += 1.0
                #         if 'top-right' in self.boundaryConstraints:
                #             initSol[-1, -1, :] = self.boundaryConstraints['top-right'][0, 0, :]
                #             localAssemblyWeights[-1, -1, :] += 1.0

                #     else:
                #         nconstraints = augmentSpanSpace + \
                #             (int(degree/2.0) if not oddDegree else int((degree+1)/2.0))
                #         loffset = 2*augmentSpanSpace
                #         print('Nconstraints = ', nconstraints,
                #             'loffset = ', loffset)

                #         if dimension == 3:
                #             initSol = np.moveaxis(initSol, 0, -1)
                #             initSol = np.moveaxis(initSol, 0, 1)

                #         freeBounds[0] = 0 if self.isClamped['left'] else (
                #             nconstraints-1 if oddDegree else nconstraints)
                #         freeBounds[1] = len(localBCAssembly[:, 0]) if self.isClamped['right'] else len(
                #             localBCAssembly[:, 0]) - (nconstraints - 1 if oddDegree else nconstraints)
                #         if dimension > 1:

                #             freeBounds[2] = 0 if self.isClamped['bottom'] else(
                #                 nconstraints - 1 if oddDegree else nconstraints)
                #             freeBounds[3] = len(localBCAssembly[0, :]) if self.isClamped['top'] else len(
                #                 localBCAssembly[0, :])-(nconstraints-1 if oddDegree else nconstraints)

                #             if dimension > 2:
                #                 freeBounds[4] = 0 if self.isClamped['down'] else(
                #                     nconstraints - 1 if oddDegree else nconstraints)
                #                 freeBounds[5] = len(localBCAssembly[:, :, 0]) if self.isClamped['up'] else len(
                #                     localBCAssembly[:, :, 0])-(nconstraints-1 if oddDegree else nconstraints)

                #         # First update hte control point vector with constraints for supporting points
                #         if dimension == 1:

                #             if 'left' in self.boundaryConstraints:
                #                 if oddDegree:
                #                     if nconstraints > 1:
                #                         initSol[: nconstraints -
                #                                 1] = beta * initSol[: nconstraints -1] + (1-beta) * self.boundaryConstraints['left'][-degree-loffset: -nconstraints]
                #                     initSol[nconstraints-1] = alpha * initSol[nconstraints-1] + (
                #                         1-alpha) * self.boundaryConstraints['left'][-nconstraints]
                #                 else:
                #                     initSol[: nconstraints] = beta * initSol[: nconstraints] + (1-beta) * self.boundaryConstraints['left'][-degree -
                #                                                                             loffset: -nconstraints]

                #             if 'right' in self.boundaryConstraints:
                #                 if oddDegree:
                #                     if nconstraints > 1:
                #                         initSol[-nconstraints +
                #                                 1:] = beta * initSol[-nconstraints +
                #                                 1:] + (1-beta) * self.boundaryConstraints['right'][nconstraints: degree+loffset]
                #                     initSol[-nconstraints] = alpha * initSol[-nconstraints] + (
                #                         1-alpha) * self.boundaryConstraints['right'][nconstraints-1]
                #                 else:
                #                     initSol[-nconstraints:] = beta * initSol[-nconstraints:] + (1-beta) * self.boundaryConstraints['right'][nconstraints: degree+loffset]

                #         # END: if dimension == 1
                #         elif dimension == 2:

                #             if 'left' in self.boundaryConstraints:
                #                 if oddDegree:
                #                     localBCAssembly[nconstraints-1, freeBounds[2]:freeBounds[3]
                #                                     ] += self.boundaryConstraints['left'][-nconstraints, freeBounds[2]:freeBounds[3]]
                #                     localAssemblyWeights[nconstraints-1,
                #                                         freeBounds[2]:freeBounds[3]] += 1.0

                #                     if nconstraints > 1:
                #                         initSol[:nconstraints-1, freeBounds[2]:freeBounds[3]
                #                                 ] = self.boundaryConstraints['left'][-degree - loffset:-nconstraints, freeBounds[2]:freeBounds[3]]
                #                         localAssemblyWeights[:nconstraints-1,
                #                                             freeBounds[2]:freeBounds[3]] += 1.0
                #                 else:
                #                     localAssemblyWeights[:nconstraints,
                #                                         freeBounds[2]:freeBounds[3]] += 1.0
                #                     initSol[:nconstraints, freeBounds[2]:freeBounds[3]
                #                             ] = self.boundaryConstraints['left'][-degree - loffset:-nconstraints, freeBounds[2]:freeBounds[3]]

                #             if 'right' in self.boundaryConstraints:
                #                 if oddDegree:
                #                     localBCAssembly[-nconstraints, freeBounds[2]:freeBounds[3]
                #                                     ] += self.boundaryConstraints['right'][nconstraints-1, freeBounds[2]:freeBounds[3]]
                #                     localAssemblyWeights[-nconstraints,
                #                                         freeBounds[2]:freeBounds[3]] += 1.0

                #                     if nconstraints > 1:
                #                         initSol[-nconstraints + 1:, freeBounds[2]:freeBounds[3]
                #                                 ] = self.boundaryConstraints['right'][nconstraints: degree + loffset, freeBounds[2]:freeBounds[3]]
                #                         localAssemblyWeights[-nconstraints + 1:,
                #                                             freeBounds[2]:freeBounds[3]] += 1.0
                #                 else:
                #                     localAssemblyWeights[-nconstraints:,
                #                                         freeBounds[2]:freeBounds[3]] += 1.0
                #                     initSol[-nconstraints:, freeBounds[2]: freeBounds[3]] = self.boundaryConstraints['right'][
                #                         nconstraints: degree + loffset, freeBounds[2]: freeBounds[3]]

                #             if 'top' in self.boundaryConstraints:
                #                 if oddDegree:
                #                     localBCAssembly[freeBounds[0]: freeBounds[1],
                #                                     -nconstraints] += self.boundaryConstraints['top'][
                #                         freeBounds[0]: freeBounds[1],
                #                         nconstraints - 1]
                #                     localAssemblyWeights[freeBounds[0]
                #                         :freeBounds[1], -nconstraints] += 1.0

                #                     if nconstraints > 1:
                #                         initSol[freeBounds[0]: freeBounds[1],
                #                                 -nconstraints + 1:] = self.boundaryConstraints['top'][
                #                             freeBounds[0]: freeBounds[1],
                #                             nconstraints: loffset + degree]
                #                         localAssemblyWeights[freeBounds[0]
                #                             :freeBounds[1], -nconstraints+1:] += 1.0
                #                 else:
                #                     initSol[freeBounds[0]:freeBounds[1], -nconstraints:] = self.boundaryConstraints['top'][freeBounds[0]                                                                                                                   :freeBounds[1], nconstraints:loffset+degree]
                #                     localAssemblyWeights[freeBounds[0]
                #                         :freeBounds[1], -nconstraints:] += 1.0

                #             if 'bottom' in self.boundaryConstraints:
                #                 if oddDegree:
                #                     localBCAssembly[freeBounds[0]:freeBounds[1],
                #                                     nconstraints-1] += self.boundaryConstraints['bottom'][freeBounds[0]:freeBounds[1], -nconstraints]
                #                     localAssemblyWeights[freeBounds[0]
                #                         :freeBounds[1], nconstraints-1] += 1.0

                #                     if nconstraints > 1:
                #                         initSol[freeBounds[0]: freeBounds[1],
                #                                 : nconstraints - 1] = self.boundaryConstraints['bottom'][
                #                             freeBounds[0]: freeBounds[1],
                #                             -degree - loffset: -nconstraints]
                #                         localAssemblyWeights[freeBounds[0]: freeBounds[1],
                #                                             : nconstraints - 1] += 1.0
                #                 else:
                #                     initSol[freeBounds[0]: freeBounds[1],
                #                             : nconstraints] = self.boundaryConstraints['bottom'][
                #                         freeBounds[0]: freeBounds[1],
                #                         -degree - loffset: -nconstraints]
                #                     localAssemblyWeights[freeBounds[0]
                #                         :freeBounds[1], : nconstraints] += 1.0

                #             # if 'top-left' in self.boundaryConstraints:
                #             #     localAssemblyWeights[: nconstraints -1, -nconstraints + 1:] += 1.0
                #             # if 'bottom-right' in self.boundaryConstraints:
                #             #     localAssemblyWeights[-nconstraints +1:, : nconstraints - 1] += 1.0
                #             # if 'bottom-left' in self.boundaryConstraints:
                #             #     localAssemblyWeights[: nconstraints - 1, : nconstraints - 1] += 1.0
                #             # if 'top-right' in self.boundaryConstraints:
                #             #     localAssemblyWeights[-nconstraints +1:, -nconstraints + 1:] += 1.0

                #             if 'top-left' in self.boundaryConstraints:
                #                 if oddDegree:
                #                     localBCAssembly[nconstraints-1, -
                #                                     nconstraints] += self.boundaryConstraints['top-left'][-nconstraints, nconstraints-1]
                #                     localAssemblyWeights[nconstraints -
                #                                         1, -nconstraints] += 1.0

                #                     if nconstraints > 1:
                #                         assert(freeBounds[0] == nconstraints - 1)
                #                         initSol[: nconstraints -
                #                                 1, -nconstraints + 1:] = 0
                #                         localAssemblyWeights[: nconstraints -
                #                                             1, -nconstraints + 1:] += 1.0
                #                         localBCAssembly[: nconstraints - 1, -nconstraints + 1:] += self.boundaryConstraints['top-left'][-degree -
                #                                                                                                                         loffset: -nconstraints, nconstraints: degree + loffset]
                #                 else:
                #                     initSol[: nconstraints, -nconstraints:] = self.boundaryConstraints['top-left'][-degree -
                #                                                                                                 loffset: -nconstraints, nconstraints: degree + loffset]
                #                     localAssemblyWeights[: nconstraints, -
                #                                         nconstraints:] += 1.0

                #             if 'bottom-right' in self.boundaryConstraints:
                #                 if oddDegree:
                #                     localBCAssembly[-nconstraints,
                #                                     nconstraints-1] += self.boundaryConstraints['bottom-right'][nconstraints-1, -nconstraints]
                #                     localAssemblyWeights[-nconstraints,
                #                                         nconstraints-1] += 1.0

                #                     if nconstraints > 1:
                #                         assert(freeBounds[2] == nconstraints - 1)
                #                         initSol[-nconstraints + 1:, : nconstraints - 1] = self.boundaryConstraints['bottom-right'][
                #                             nconstraints: degree + loffset, -degree - loffset: -nconstraints]
                #                         localAssemblyWeights[-nconstraints +
                #                                             1:, : nconstraints - 1] += 1.0
                #                 else:
                #                     initSol[-nconstraints:, : nconstraints] = self.boundaryConstraints['bottom-right'][
                #                         nconstraints: degree + loffset, -degree - loffset: -nconstraints]
                #                     localAssemblyWeights[-nconstraints:,
                #                                         : nconstraints] += 1.0

                #             if 'bottom-left' in self.boundaryConstraints:
                #                 if oddDegree:
                #                     localBCAssembly[nconstraints-1,
                #                                     nconstraints-1] += self.boundaryConstraints['bottom-left'][-nconstraints, -nconstraints]
                #                     localAssemblyWeights[nconstraints -
                #                                         1, nconstraints-1] += 1.0

                #                     if nconstraints > 1:
                #                         assert(freeBounds[0] == nconstraints - 1)
                #                         initSol[: nconstraints - 1, : nconstraints - 1] = self.boundaryConstraints['bottom-left'][-degree -
                #                                                                                                                 loffset: -nconstraints, -degree - loffset: -nconstraints]
                #                         localAssemblyWeights[: nconstraints -
                #                                             1, : nconstraints - 1] += 1.0
                #                 else:
                #                     initSol[: nconstraints, : nconstraints] = self.boundaryConstraints['bottom-left'][-degree -
                #                                                                                                     loffset: -nconstraints, -degree - loffset: -nconstraints]
                #                     localAssemblyWeights[: nconstraints,
                #                                         : nconstraints] += 1.0

                #             if 'top-right' in self.boundaryConstraints:
                #                 if oddDegree:
                #                     localBCAssembly[-nconstraints, -nconstraints] += self.boundaryConstraints['top-right'][
                #                         nconstraints - 1, nconstraints - 1]
                #                     localAssemblyWeights[-nconstraints, -
                #                                         nconstraints] += 1.0

                #                     if nconstraints > 1:
                #                         initSol[-nconstraints + 1:, -nconstraints + 1:] = self.boundaryConstraints['top-right'][
                #                             nconstraints: degree + loffset, nconstraints: degree + loffset]
                #                         localAssemblyWeights[-nconstraints +
                #                                             1:, -nconstraints + 1:] += 1.0
                #                 else:
                #                     initSol[-nconstraints:, -nconstraints:] = self.boundaryConstraints['top-right'][
                #                         nconstraints: degree + loffset, nconstraints: degree + loffset]
                #                     localAssemblyWeights[-nconstraints:, -
                #                                         nconstraints:] += 1.0

                #         # END: if dimension == 2
                #         else:

                #             if 'left' in self.boundaryConstraints:
                #                 if oddDegree:
                #                     localBCAssembly[nconstraints-1, freeBounds[2]:freeBounds[3],
                #                                     freeBounds[4]:freeBounds[5]] += self.boundaryConstraints['left'][-nconstraints, freeBounds[2]:freeBounds[3], freeBounds[4]:freeBounds[5]]
                #                     localAssemblyWeights[nconstraints-1,
                #                                         freeBounds[2]:freeBounds[3], freeBounds[4]:freeBounds[5]] += 1.0

                #                     if nconstraints > 1:
                #                         initSol[:nconstraints-1, freeBounds[2]:freeBounds[3], freeBounds[4]:freeBounds[5]
                #                                 ] = self.boundaryConstraints['left'][-degree - loffset:-nconstraints, freeBounds[2]:freeBounds[3], freeBounds[4]:freeBounds[5]]
                #                         localAssemblyWeights[:nconstraints-1,
                #                                             freeBounds[2]:freeBounds[3],freeBounds[4]:freeBounds[5]] += 1.0
                #                 else:
                #                     localAssemblyWeights[:nconstraints,
                #                                         freeBounds[2]:freeBounds[3], freeBounds[4]:freeBounds[5]] += 1.0
                #                     initSol[:nconstraints, freeBounds[2]:freeBounds[3], freeBounds[4]:freeBounds[5]
                #                             ] = self.boundaryConstraints['left'][-degree - loffset:-nconstraints, freeBounds[2]:freeBounds[3], freeBounds[4]:freeBounds[5]]

                #             if 'right' in self.boundaryConstraints:
                #                 if oddDegree:
                #                     localBCAssembly[-nconstraints, freeBounds[2]:freeBounds[3], freeBounds[4]:freeBounds[5]
                #                                     ] += self.boundaryConstraints['right'][nconstraints-1, freeBounds[2]:freeBounds[3], freeBounds[4]:freeBounds[5]]
                #                     localAssemblyWeights[-nconstraints,
                #                                         freeBounds[2]:freeBounds[3], freeBounds[4]:freeBounds[5]] += 1.0

                #                     if nconstraints > 1:
                #                         initSol[-nconstraints + 1:, freeBounds[2]:freeBounds[3], freeBounds[4]:freeBounds[5]
                #                                 ] = self.boundaryConstraints['right'][nconstraints: degree + loffset, freeBounds[2]:freeBounds[3], freeBounds[4]:freeBounds[5]]
                #                         localAssemblyWeights[-nconstraints + 1:,
                #                                             freeBounds[2]:freeBounds[3], freeBounds[4]:freeBounds[5]] += 1.0
                #                 else:
                #                     localAssemblyWeights[-nconstraints:,
                #                                         freeBounds[2]:freeBounds[3], freeBounds[4]:freeBounds[5]] += 1.0
                #                     initSol[-nconstraints:, freeBounds[2]: freeBounds[3], freeBounds[4]:freeBounds[5]] = self.boundaryConstraints['right'][
                #                         nconstraints: degree + loffset, freeBounds[2]: freeBounds[3], freeBounds[4]:freeBounds[5]]

                #             if 'top' in self.boundaryConstraints:
                #                 if oddDegree:
                #                     localBCAssembly[freeBounds[0]: freeBounds[1],
                #                                     -nconstraints, freeBounds[4]:freeBounds[5]] += self.boundaryConstraints['top'][
                #                         freeBounds[0]: freeBounds[1],
                #                         nconstraints - 1, freeBounds[4]:freeBounds[5]]
                #                     localAssemblyWeights[freeBounds[0]
                #                         :freeBounds[1], -nconstraints, freeBounds[4]:freeBounds[5]] += 1.0

                #                     if nconstraints > 1:
                #                         initSol[freeBounds[0]: freeBounds[1],
                #                                 -nconstraints + 1:, freeBounds[4]:freeBounds[5]] = self.boundaryConstraints['top'][
                #                             freeBounds[0]: freeBounds[1],
                #                             nconstraints: loffset + degree, freeBounds[4]:freeBounds[5]]
                #                         localAssemblyWeights[freeBounds[0]
                #                             :freeBounds[1], -nconstraints+1:, freeBounds[4]:freeBounds[5]] += 1.0
                #                 else:
                #                     initSol[freeBounds[0]:freeBounds[1], -nconstraints:, freeBounds[4]:freeBounds[5]] = self.boundaryConstraints['top'][freeBounds[0]                                                                                                                   :freeBounds[1], nconstraints:loffset+degree, freeBounds[4]:freeBounds[5]]
                #                     localAssemblyWeights[freeBounds[0]
                #                         :freeBounds[1], -nconstraints:, freeBounds[4]:freeBounds[5]] += 1.0

                #             if 'bottom' in self.boundaryConstraints:
                #                 if oddDegree:
                #                     localBCAssembly[freeBounds[0]:freeBounds[1],
                #                                     nconstraints-1, freeBounds[4]:freeBounds[5]] += self.boundaryConstraints['bottom'][freeBounds[0]:freeBounds[1], -nconstraints, freeBounds[4]:freeBounds[5]]
                #                     localAssemblyWeights[freeBounds[0]
                #                         :freeBounds[1], nconstraints-1, freeBounds[4]:freeBounds[5]] += 1.0

                #                     if nconstraints > 1:
                #                         initSol[freeBounds[0]: freeBounds[1],
                #                                 : nconstraints - 1, freeBounds[4]:freeBounds[5]] = self.boundaryConstraints['bottom'][
                #                             freeBounds[0]: freeBounds[1],
                #                             -degree - loffset: -nconstraints, freeBounds[4]:freeBounds[5]]
                #                         localAssemblyWeights[freeBounds[0]: freeBounds[1],
                #                                             : nconstraints - 1, freeBounds[4]:freeBounds[5]] += 1.0
                #                 else:
                #                     initSol[freeBounds[0]: freeBounds[1],
                #                             : nconstraints, freeBounds[4]:freeBounds[5]] = self.boundaryConstraints['bottom'][
                #                         freeBounds[0]: freeBounds[1],
                #                         -degree - loffset: -nconstraints, freeBounds[4]:freeBounds[5]]
                #                     localAssemblyWeights[freeBounds[0]
                #                         :freeBounds[1], : nconstraints, freeBounds[4]:freeBounds[5]] += 1.0

                #             if 'down' in self.boundaryConstraints:
                #                 if oddDegree:
                #                     localBCAssembly[freeBounds[0]: freeBounds[1], freeBounds[2]: freeBounds[3],
                #                                     nconstraints - 1] += self.boundaryConstraints['down'][
                #                         freeBounds[0]: freeBounds[1], freeBounds[2]: freeBounds[3],
                #                         -nconstraints]
                #                     localAssemblyWeights[freeBounds[0]
                #                         :freeBounds[1], freeBounds[2]: freeBounds[3], nconstraints - 1] += 1.0

                #                     if nconstraints > 1:
                #                         initSol[freeBounds[0]: freeBounds[1], freeBounds[2]: freeBounds[3],
                #                                 : nconstraints - 1] = self.boundaryConstraints['down'][
                #                             freeBounds[0]: freeBounds[1], freeBounds[2]: freeBounds[3],
                #                             -degree - loffset: -nconstraints]
                #                         localAssemblyWeights[freeBounds[0]
                #                             :freeBounds[1], freeBounds[2]: freeBounds[3], : nconstraints - 1] += 1.0
                #                 else:
                #                     initSol[freeBounds[0]:freeBounds[1], freeBounds[2]: freeBounds[3], : nconstraints] = self.boundaryConstraints['down'][freeBounds[0]                                                                                                                   :freeBounds[1], freeBounds[2]: freeBounds[3], -degree - loffset: -nconstraints]
                #                     localAssemblyWeights[freeBounds[0]
                #                         :freeBounds[1], freeBounds[2]: freeBounds[3], : nconstraints] += 1.0

                #             if 'up' in self.boundaryConstraints:
                #                 if oddDegree:
                #                     localBCAssembly[freeBounds[0]:freeBounds[1], freeBounds[2]: freeBounds[3],
                #                                     -nconstraints] += self.boundaryConstraints['up'][freeBounds[0]:freeBounds[1], freeBounds[2]: freeBounds[3], nconstraints-1]
                #                     localAssemblyWeights[freeBounds[0]
                #                         :freeBounds[1], freeBounds[2]: freeBounds[3], -nconstraints] += 1.0

                #                     if nconstraints > 1:
                #                         initSol[freeBounds[0]: freeBounds[1], freeBounds[2]: freeBounds[3],-nconstraints + 1:
                #                                 ] = self.boundaryConstraints['up'][
                #                             freeBounds[0]: freeBounds[1], freeBounds[2]: freeBounds[3],
                #                             nconstraints: loffset + degree]
                #                         localAssemblyWeights[freeBounds[0]: freeBounds[1], freeBounds[2]: freeBounds[3],
                #                                             -nconstraints + 1:] += 1.0
                #                 else:
                #                     initSol[freeBounds[0]: freeBounds[1], freeBounds[2]: freeBounds[3],
                #                         -nconstraints:] = self.boundaryConstraints['up'][
                #                         freeBounds[0]: freeBounds[1], freeBounds[2]: freeBounds[3],
                #                         nconstraints:loffset+degree]
                #                     localAssemblyWeights[freeBounds[0]
                #                         :freeBounds[1], freeBounds[2]: freeBounds[3], -nconstraints:] += 1.0

                #     # All internal DoFs should get a weight = 1.0
                #     if dimension == 1:
                #         localAssemblyWeights[freeBounds[0]: freeBounds[1]] += 1.0
                #     elif dimension == 2:
                #         localAssemblyWeights[freeBounds[0]: freeBounds[1],
                #                             freeBounds[2]: freeBounds[3]] += 1.0
                #     else:
                #         localAssemblyWeights[freeBounds[0]: freeBounds[1],
                #                             freeBounds[2]: freeBounds[3],
                #                             freeBounds[4]: freeBounds[5]] += 1.0

                #     if verbose:
                #         if 'left' in self.boundaryConstraints:
                #             print(
                #                 'Error left: ', np.abs(
                #                     self.boundaryConstraints['left']
                #                     [-degree - loffset: -nconstraints, :] -
                #                     initSol[: nconstraints - 1, :]).T)
                #         if 'right' in self.boundaryConstraints:
                #             print(
                #                 'Error right: ', np.abs(
                #                     self.boundaryConstraints['right']
                #                     [nconstraints: degree + loffset, :] -
                #                     initSol[-nconstraints + 1:, :]).T)
                #         if 'top' in self.boundaryConstraints:
                #             print(
                #                 'Error top: ', np.abs(
                #                     self.boundaryConstraints['top'][:, nconstraints:loffset+degree] -
                #                     initSol[:, -nconstraints+1:]))
                #         if 'bottom' in self.boundaryConstraints:
                #             print(
                #                 'Error bottom: ', np.abs(
                #                     self.boundaryConstraints['bottom']
                #                     [:, -degree - loffset: -nconstraints] -
                #                     initSol[:, : nconstraints - 1]))

                #     if dimension > 1:
                #         initSol = np.divide(
                #             initSol + localBCAssembly, localAssemblyWeights)
                #         if dimension == 3:
                #             initSol = np.moveaxis(initSol, -1, 0)
                #             initSol = np.moveaxis(initSol, 1, 2)
            ########################### END ########################

            initialDecodedError = residualFunction(initSol, printVerbose=True)

            if solverMaxIter > 0:
                if enforceBounds:
                    if dimension == 1:
                        nshape = self.controlPointData.shape[0]
                        bnds = np.tensordot(
                            np.ones(nshape), self.controlPointBounds, axes=0
                        )

                        oddDegree = degree % 2
                        nconstraints = augmentSpanSpace + (
                            int(degree / 2.0)
                            if not oddDegree
                            else int((degree + 1) / 2.0)
                        )

                        if not self.isClamped["left"]:
                            for i in range(nconstraints + 1):
                                bnds[i][:] = initSol[i]
                        if not self.isClamped["right"]:
                            for i in range(nconstraints):
                                bnds[-i - 1][:] = initSol[-i - 1]
                    elif dimension == 2:
                        nshape = (
                            self.controlPointData.shape[0]
                            * self.controlPointData.shape[1]
                        )
                        bnds = np.tensordot(
                            np.ones(nshape), self.controlPointBounds, axes=0
                        )
                    else:
                        nshape = (
                            self.controlPointData.shape[0]
                            * self.controlPointData.shape[1]
                            * self.controlPointData.shape[2]
                        )
                        bnds = np.tensordot(
                            np.ones(nshape), self.controlPointBounds, axes=0
                        )
                else:
                    bnds = None

                print("Using optimization solver = ", solverScheme)
                # Solver options: https://docs.scipy.org/doc/scipy-0.13.0/reference/generated/scipy.optimize.show_options.html
                if solverScheme == "L-BFGS-B":
                    res = minimize(
                        residualFunction,
                        x0=initSol,
                        method=solverScheme,  # 'SLSQP', #'L-BFGS-B', #'TNC',
                        bounds=bnds,
                        jac=jacobian,
                        callback=print_iterate,
                        tol=self.globalTolerance,
                        options={
                            "disp": False,
                            "ftol": maxRelErr,
                            "gtol": self.globalTolerance,
                            "maxiter": solverMaxIter,
                        },
                    )
                elif solverScheme == "CG":
                    res = minimize(
                        residualFunction,
                        x0=initSol,
                        method=solverScheme,  # Unbounded - can blow up
                        jac=jacobian,
                        bounds=bnds,
                        callback=print_iterate,
                        tol=self.globalTolerance,
                        options={"disp": False, "norm": 2, "maxiter": solverMaxIter},
                    )
                elif solverScheme == "SLSQP" or solverScheme == "COBYLA":
                    res = minimize(
                        residualFunction,
                        x0=initSol,
                        method=solverScheme,  # 'SLSQP', #'L-BFGS-B', #'TNC',
                        bounds=bnds,
                        jac=jacobian,
                        callback=print_iterate,
                        tol=self.globalTolerance,
                        options={
                            "disp": False,
                            "ftol": maxRelErr,
                            "maxiter": solverMaxIter,
                        },
                    )
                elif solverScheme == "Newton-CG" or solverScheme == "TNC":
                    res = minimize(
                        residualFunction,
                        x0=initSol,
                        method=solverScheme,  # 'SLSQP', #'L-BFGS-B', #'TNC',
                        jac=jacobian,
                        bounds=bnds,
                        # jac=egrad(residual)(initSol),
                        callback=print_iterate,
                        tol=self.globalTolerance,
                        options={
                            "disp": False,
                            "eps": self.globalTolerance,
                            "maxiter": solverMaxIter,
                        },
                    )
                else:
                    error("No implementation available")

                print("[%d] : %s" % (idom, res.message))
                solution = np.copy(res.x).reshape(self.controlPointData.shape)

            else:

                # from scipy.sparse.linalg import LinearOperator

                # A = LinearOperator(self.controlPointData.shape, matvec=residual2DRev)

                solution = np.copy(initSol)

        return solution

    def print_error_metrics(self, cp):
        # print('Size: ', commW.size, ' rank = ', commW.rank, ' Metrics: ', self.errorMetricsL2[:])
        # print('Rank:', commW.rank, ' SDom:', cp.gid(), ' L2 Error table: ', self.errorMetricsL2)
        print(
            "Rank:",
            commWorld.rank,
            " SDom:",
            cp.gid(),
            " Error: ",
            self.errorMetricsL2[self.outerIteration - 1],
            ", Convergence: ",
            np.abs(
                [
                    self.errorMetricsL2[1 : self.outerIteration]
                    - self.errorMetricsL2[0 : self.outerIteration - 1]
                ]
            ),
        )

        # L2NormVector = MPI.gather(self.errorMetricsL2[self.outerIteration - 1], root=0)

    def check_convergence(self, cp, iterationNum):

        global isConverged, L2err
        if len(self.solutionDecodedOld):

            # Let us compute the relative change in the solution between current and previous iteration
            iterateChangeVec = (
                self.solutionDecoded - self.solutionDecodedOld
            ).flatten()
            errorMetricsSubDomL2 = np.linalg.norm(
                iterateChangeVec, ord=2
            ) / np.linalg.norm(self.refSolutionLocal.reshape(-1), ord=2)
            errorMetricsSubDomLinf = np.linalg.norm(
                iterateChangeVec, ord=np.inf
            ) / np.linalg.norm(self.refSolutionLocal.reshape(-1), ord=np.inf)

            self.errorMetricsL2[self.outerIteration] = self.decodederrors[0]
            self.errorMetricsLinf[self.outerIteration] = self.decodederrors[1]

            print(
                cp.gid() + 1,
                " Convergence check: ",
                errorMetricsSubDomL2,
                errorMetricsSubDomLinf,
                np.abs(
                    self.errorMetricsLinf[self.outerIteration]
                    - self.errorMetricsLinf[self.outerIteration - 1]
                ),
                errorMetricsSubDomLinf < 1e-8
                and np.abs(
                    self.errorMetricsL2[self.outerIteration]
                    - self.errorMetricsL2[self.outerIteration - 1]
                )
                < 1e-10,
            )

            L2err[cp.gid()] = self.decodederrors[0]
            if (
                errorMetricsSubDomLinf < 1e-12
                and np.abs(
                    self.errorMetricsL2[self.outerIteration]
                    - self.errorMetricsL2[self.outerIteration - 1]
                )
                < 1e-12
            ):
                print(
                    "Subdomain ",
                    cp.gid() + 1,
                    " has converged to its final solution with error = ",
                    errorMetricsSubDomLinf,
                )
                isConverged[cp.gid()] = 1

        # self.outerIteration = iterationNum+1
        self.outerIteration += 1

        # isASMConverged = commWorld.allreduce(self.outerIterationConverged, op=MPI.LAND)

    def subdomain_solve(self, cp):

        global isConverged
        if isConverged[cp.gid()] == 1:
            print(
                cp.gid() + 1,
                " subdomain has already converged to its solution. Skipping solve ...",
            )
            return

        # Subdomain ID: iSubDom = cp.gid()+1
        newSolve = False
        if (
            np.sum(np.abs(self.controlPointData)) < 1e-14
            and len(self.controlPointData) > 0
        ) or len(self.controlPointData) == 0:
            newSolve = True

            # Compute the basis functions now that we are ready to solve the problem
            self.compute_basis()

            self.decodeOpXYZ = self.compute_decode_operators()

        print("Subdomain -- ", cp.gid() + 1)

        # Let do the recursive iterations
        # Use the previous MAK solver solution as initial guess; Could do something clever later

        # Invoke the adaptive fitting routine for this subdomain
        iSubDom = cp.gid() + 1

        self.solutionDecodedOld = np.copy(self.solutionDecoded)
        # self.globalTolerance = 1e-3 * 1e-3**self.adaptiveIterationNum

        if (
            (
                np.sum(np.abs(self.controlPointData)) < 1e-14
                and len(self.controlPointData) > 0
            )
            or len(self.controlPointData) == 0
        ) and self.outerIteration == 0:
            print(iSubDom, " - Applying the unconstrained solver.")
            constraints = None

        else:
            print(iSubDom, " - Applying the constrained solver.")
            constraints = np.copy(self.controlPointData)

        #  Invoke the local subdomain solver
        self.controlPointData = self.LSQFit_NonlinearOptimize(
            iSubDom, degree, constraints
        )

        if constraints is None:  # We just solved the initial LSQ problem.
            # Store the maximum bounds to respect so that we remain monotone
            self.controlPointBounds = np.array(
                [np.min(self.controlPointData), np.max(self.controlPointData)]
            )

        # Update the local decoded data
        self.solutionDecoded = self.decode(self.controlPointData, self.decodeOpXYZ)
        # decodedError = np.abs(np.array(self.refSolutionLocal - self.solutionDecoded)) / solutionRange

        if len(self.solutionLocalHistory) == 0 and extrapolate:
            if useAitken:
                self.solutionLocalHistory = np.zeros(
                    (self.controlPointData.shape[0] * self.controlPointData.shape[1], 3)
                )
            else:
                self.solutionLocalHistory = np.zeros(
                    (
                        self.controlPointData.shape[0] * self.controlPointData.shape[1],
                        nWynnEWork,
                    )
                )

        # E = (self.solutionDecoded[self.corebounds[0]:self.corebounds[1]] - self.controlPointData[self.corebounds[0]:self.corebounds[1]])/solutionRange
        if dimension == 1:
            decodedError = (
                self.refSolutionLocal[self.corebounds[0][0] : self.corebounds[0][1]]
                - self.solutionDecoded[self.corebounds[0][0] : self.corebounds[0][1]]
            ) / solutionRange
        elif dimension == 2:
            decodedErrorT = (
                self.refSolutionLocal - self.solutionDecoded
            ) / solutionRange
            decodedError = decodedErrorT[
                self.corebounds[0][0] : self.corebounds[0][1],
                self.corebounds[1][0] : self.corebounds[1][1],
            ].reshape(-1)
        elif dimension == 3:
            print("Shapes: ", self.refSolutionLocal.shape, self.solutionDecoded.shape)
            decodedErrorT = (
                self.refSolutionLocal - self.solutionDecoded
            ) / solutionRange
            decodedError = decodedErrorT[
                self.corebounds[0][0] : self.corebounds[0][1],
                self.corebounds[1][0] : self.corebounds[1][1],
                self.corebounds[2][0] : self.corebounds[2][1],
            ].reshape(-1)
        LinfErr = np.linalg.norm(decodedError, ord=np.inf)
        L2Err = np.sqrt(np.sum(decodedError**2) / len(decodedError))

        self.decodederrors[0] = L2Err
        self.decodederrors[1] = LinfErr

        print(
            "Subdomain -- ", iSubDom, ": L2 error: ", L2Err, ", Linf error: ", LinfErr
        )


#########
domain_control = diy.DiscreteBounds(
    np.zeros((dimension, 1), dtype=np.uint32), nPoints - 1
)

# Routine to recursively add a block and associated data to it

print("")

locSolutionShape = None

def add_input_control_block2(gid, core, bounds, domain, link):
    print("Subdomain %d: " % gid, core, bounds, domain)
    minb = bounds.min
    maxb = bounds.max

    xlocal = coordinates["x"][minb[0] : maxb[0] + 1]
    if dimension > 1:
        ylocal = coordinates["y"][minb[1] : maxb[1] + 1]
        if dimension > 2:
            zlocal = coordinates["z"][minb[2] : maxb[2] + 1]
            if closedFormFunctional:
                X, Y, Z = np.meshgrid(xlocal, ylocal, zlocal, indexing="ij")
                sollocal = solution(X, Y, Z)
                del X, Y, Z
            else:
                sollocal = solution[
                    minb[0] : maxb[0] + 1, minb[1] : maxb[1] + 1, minb[2] : maxb[2] + 1
                ]

        else:
            zlocal = None
            if closedFormFunctional:
                X, Y = np.meshgrid(xlocal, ylocal, indexing="ij")
                sollocal = solution(X, Y)
                del X, Y
            else:
                sollocal = solution[minb[0] : maxb[0] + 1, minb[1] : maxb[1] + 1]

    else:
        ylocal = None
        zlocal = None
        if closedFormFunctional:
            sollocal = solution(xlocal)
        else:
            sollocal = solution[minb[0] : maxb[0] + 1]
        sollocal = sollocal.reshape((len(sollocal), 1))


    # print("Subdomain %d: " % gid, minb[0], minb[1], maxb[0], maxb[1], z.shape, zlocal.shape)
    masterControl.add(
        gid,
        InputControlBlock(
            gid, nControlPointsInput, core, bounds, sollocal, xlocal, ylocal, zlocal
        ),
        link,
    )

if dimension == 1:
    solutionShape = [coordinates["x"].shape[0]]
elif dimension == 2:
    solutionShape = [coordinates["x"].shape[0], coordinates["y"].shape[0]]
else:
    solutionShape = [coordinates["x"].shape[0], coordinates["y"].shape[0], coordinates["z"].shape[0]]

# TODO: If working in parallel with MPI or DIY, do a global reduce here
# Store L2, Linf errors as function of iteration
errors = np.zeros([nASMIterations + 1, 2])

# Let us initialize DIY and setup the problem
share_face = np.ones((dimension, 1)) > 0
wrap = np.ones((dimension, 1)) < 0
ghosts = np.zeros((dimension, 1), dtype=np.uint32)

discreteDec = diy.DiscreteDecomposer(
    dimension, domain_control, nTotalSubDomains, share_face, wrap, ghosts, nSubDomains
)

contigAssigner = diy.ContiguousAssigner(nprocs, nTotalSubDomains)

discreteDec.decompose(rank, contigAssigner, add_input_control_block2)

if verbose:
    masterControl.foreach(InputControlBlock.show)

#########

def send_receive_all():

    masterControl.foreach(InputControlBlock.send_diy)

    # if dimension < 3:
    #     masterControl.foreach(InputControlBlock.send_diy)
    # else:
    #     masterControl.foreach(InputControlBlock.send_diy_3d)

    masterControl.exchange(False)

    masterControl.foreach(InputControlBlock.recv_diy)

    # if dimension < 3:
    #     masterControl.foreach(InputControlBlock.recv_diy)
    # else:
    #     masterControl.foreach(InputControlBlock.recv_diy_3d)

    return


sys.stdout.flush()
commWorld.Barrier()
start_time = timeit.default_timer()

if rank == 0:
    print("\n---- Starting Global Iterative Loop ----")

# Before starting the solve, let us exchange the initial conditions
# including the knot vector locations that need to be used for creating
# padded knot vectors in each subdomain
masterControl.foreach(InputControlBlock.initialize_data)

# Send and receive initial condition data as needed
send_receive_all()

if not fullyPinned:
    masterControl.foreach(InputControlBlock.augment_spans)
    if augmentSpanSpace > 0:
        masterControl.foreach(InputControlBlock.augment_inputdata)

if useVTKOutput or showplot:
    masterControl.foreach(InputControlBlock.update_bounds)
    # globalExtentDict = np.array(commWorld.gather(localExtents, root=0)[0])
    print(rank, " - localExtents = ", localExtents)
    globalExtentDict = commWorld.gather(flattenDict(localExtents), root=0)
    if rank == 0:
        if nprocs == 1:
            globalExtentDict = globalExtentDict[0]
        else:
            globalExtentDict = flattenListDict(globalExtentDict)
        print("Global extents consolidated  = ", globalExtentDict)

del coordinates
if not closedFormFunctional:
    del solution

elapsed = timeit.default_timer() - start_time
sys.stdout.flush()
if rank == 0:
    print("\nTotal setup time for solver = ", elapsed, "\n")

commWorld.Barrier()
start_time = timeit.default_timer()

for iterIdx in range(nASMIterations):

    if rank == 0:
        print("\n---- Starting Iteration: %d ----" % iterIdx)

    if iterIdx > 0 and rank == 0:
        print("")

    # run our local subdomain solver
    masterControl.foreach(InputControlBlock.subdomain_solve)

    # check if we have locally converged within criteria
    masterControl.foreach(
        lambda icb, cp: InputControlBlock.check_convergence(icb, cp, iterIdx)
    )

    isASMConverged = commWorld.allreduce(np.sum(isConverged), op=MPI.SUM)

    # comm.Barrier()
    sys.stdout.flush()

    if useVTKOutput or showplot:

        if dimension == 1:

            figHnd = plt.figure()
            figErrHnd = None
            # figErrHnd = plt.figure()
            # plt.plot(coordinates["x"], solution, 'b-', ms=5, label='Input')

            masterControl.foreach(
                lambda icb, cp: InputControlBlock.set_fig_handles(
                    icb, cp, figHnd, figErrHnd, "%d-%d" % (cp.gid(), iterIdx)
                )
            )
            masterControl.foreach(InputControlBlock.output_solution)

            # plt.legend()
            plt.draw()
            figHnd.show()

        else:

            if useVTKOutput:
                masterControl.foreach(
                    lambda icb, cp: InputControlBlock.set_fig_handles(
                        icb, cp, None, None, "%d-%d" % (cp.gid(), iterIdx)
                    )
                )
                # masterControl.foreach(InputControlBlock.output_solution)
                masterControl.foreach(InputControlBlock.output_vtk)

                if rank == 0:
                    WritePVTKFile(iterIdx)
                    WritePVTKControlFile(iterIdx)

    if isASMConverged == nTotalSubDomains:
        if rank == 0:
            print("\n\nASM solver converged after %d iterations\n\n" % (iterIdx + 1))
        break

    else:
        if extrapolate:
            masterControl.foreach(
                lambda icb, cp: InputControlBlock.extrapolate_guess(icb, cp, iterIdx)
            )

        # Now let us perform send-receive to get the data on the interface boundaries from
        # adjacent nearest-neighbor subdomains
        send_receive_all()


# masterControl.foreach(InputControlBlock.print_solution)

elapsed = timeit.default_timer() - start_time
sys.stdout.flush()
if rank == 0:
    print("\nTotal computational time for solve = ", elapsed, "\n")

avgL2err = commWorld.allreduce(np.sum(L2err[np.nonzero(L2err)] ** 2), op=MPI.SUM)
avgL2err = np.sqrt(avgL2err / nTotalSubDomains)
maxL2err = commWorld.allreduce(np.max(np.abs(L2err[np.nonzero(L2err)])), op=MPI.MAX)
minL2err = commWorld.allreduce(np.min(np.abs(L2err[np.nonzero(L2err)])), op=MPI.MIN)

# np.set_printoptions(formatter={'float': '{: 5.12e}'.format})
# mc.foreach(InputControlBlock.print_error_metrics)
if rank == 0:
    print(
        "\nError metrics: L2 average = %6.12e, L2 maxima = %6.12e, L2 minima = %6.12e\n"
        % (avgL2err, maxL2err, minL2err)
    )
    print("")

commWorld.Barrier()

np.set_printoptions(formatter={"float": "{: 5.12e}".format})
masterControl.foreach(InputControlBlock.print_error_metrics)

if showplot:
    plt.show()

# ---------------- END MAIN FUNCTION -----------------
