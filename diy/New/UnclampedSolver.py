# coding: utf-8
# get_ipython().magic(u'matplotlib notebook')
# %matplotlib notebook
import sys
import getopt
import math
import timeit
import autograd
# import numpy as np
from functools import reduce

import splipy as sp

# Autograd AD impots
from autograd import elementwise_grad as egrad
import autograd.numpy as np

from pymoab import core, types
from pymoab.scd import ScdInterface
from pymoab.hcoord import HomCoord

# SciPY imports
import scipy
from scipy.linalg import svd
# , BroydenFirst, KrylovJacobian
from scipy.optimize import minimize, linprog, root, anderson, newton_krylov
from scipy import linalg, matrix
from scipy.ndimage import zoom

# MPI imports
from mpi4py import MPI
import diy

# Plotting imports
from matplotlib import pyplot as plt
from matplotlib import cm
# from pyevtk.hl import gridToVTK

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
nSubDomains = np.array([3] * dimension, dtype=np.uint32)
nSubDomains = [3, 3]
nSubDomainsX = nSubDomains[0]
nSubDomainsY = nSubDomains[1] if dimension > 1 else 1
nSubDomainsZ = nSubDomains[2] if dimension > 2 else 1

debugProblem = False
verbose = False
useVTKOutput = True
useMOABMesh = False

augmentSpanSpace = 2
useDiagonalBlocks = True

relEPS = 5e-5
fullyPinned = False
useAdditiveSchwartz = True
enforceBounds = False
alwaysSolveConstrained = False

# ------------------------------------------
# Solver parameters

#                      0      1       2         3          4       5       6      7       8
solverMethods = ['L-BFGS-B', 'CG', 'SLSQP', 'Newton-CG',
                 'TNC', 'krylov', 'lm', 'trf']
solverScheme = solverMethods[1]
solverMaxIter = 2
nASMIterations = 10
maxAbsErr = 1e-6
maxRelErr = 1e-12

# Solver acceleration
extrapolate = False
useAitken = True
nWynnEWork = 3

##################
# Initialize
Dmin = Dmax = 0
##################

# Initialize DIY
# commWorld = diy.mpi.MPIComm()           # world
commWorld = MPI.COMM_WORLD
masterControl = diy.Master(commWorld)         # master
nprocs = commWorld.size
rank = commWorld.rank

if rank == 0:
    print('Argument List:', str(sys.argv))

##########################
# Parse command line overrides if any
##
argv = sys.argv[1:]


def usage():
    print(
        sys.argv[0],
        '-p <problem> -n <nsubdomains> -x <nsubdomains_x> -y <nsubdomains_y> -d <degree> -c <controlpoints> -o <overlapData> -a <nASMIterations> -g <augmentSpanSpace>')
    sys.exit(2)


try:
    opts, args = getopt.getopt(argv, "hi:p:n:x:y:z:d:c:a:g:s:",
                               ["dimension=", "problem=", "nsubdomains=", "nx=", "ny=", "nz=", "degree=",
                                "controlpoints=", "nasm=", "aug=", "accel", "wynn"])
except getopt.GetoptError:
    usage()

nControlPointsInputIn = 15
# nSubDomainsX = nSubDomainsY = nSubDomainsZ = 1
Dmin = np.array(3, dtype=np.float32)
Dmax = np.array(3, dtype=np.float32)

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
    elif opt in ("-g", "--aug"):
        augmentSpanSpace = int(arg)
    elif opt in ("-s", "--accel"):
        extrapolate = True
    elif opt in ("--wynn"):
        useAitken = False

nSubDomainsY = 1 if dimension < 2 else nSubDomainsY
nSubDomainsZ = 1 if dimension < 3 else nSubDomainsZ
showplot = False if dimension > 1 else True
# nControlPointsInput = nControlPointsInputIn * np.ones((dimension, 1), dtype=np.uint32)
nSubDomains = [nSubDomainsX, nSubDomainsY, nSubDomainsZ]
# -------------------------------------

nPoints = np.array([1] * dimension, dtype=np.uint32)
nTotalSubDomains = nSubDomainsX * nSubDomainsY * nSubDomainsZ
isConverged = np.zeros(nTotalSubDomains, dtype='int32')
L2err = np.zeros(nTotalSubDomains)

# globalExtentDict = np.zeros(nTotalSubDomains*2*dimension, dtype='int32')
# globalExtentDict[cp.gid()*4:cp.gid()*4+4] = extents
localExtents = {}

# def read_problem_parameters():
xcoord = ycoord = zcoord = None
solution = None
if dimension == 1:
    print('Setting up problem for 1-D')

    if problem == 1:
        Dmin = [-4.]
        Dmax = [4.]
        xcoord = np.linspace(Dmin[0], Dmax[0], 10001)
        scale = 100
        # solution = scale * (np.sinc(xcoord-1)+np.sinc(xcoord+1))
        # solution = scale * (np.sinc(xcoord+1) + np.sinc(2*xcoord) + np.sinc(xcoord-1))
        solution = scale * (np.sinc(xcoord) + np.sinc(2 *
                            xcoord-1) + np.sinc(3*xcoord+1.5))
        # solution = np.zeros(xcoord.shape)
        # solution[xcoord <= 0] = 1
        # solution[xcoord > 0] = -1
        # solution = scale * np.sin(math.pi * xcoord/4)
    elif problem == 2:
        solution = np.fromfile("input/1d/s3d.raw", dtype=np.float64)
        print('Real data shape: ', solution.shape)
        Dmin = [0.]
        Dmax = [1.]
        xcoord = np.linspace(Dmin[0], Dmax[0], solution.shape[0])
        relEPS = 5e-8
    elif problem == 3:
        Y = np.fromfile("input/2d/nek5000.raw", dtype=np.float64)
        Y = Y.reshape(200, 200)
        solution = Y[100, :]  # Y[:,150] # Y[110,:]
        Dmin = [0.]
        Dmax = [1.]
        xcoord = np.linspace(Dmin, Dmax, nPoints)
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

        DJI = pd.read_csv("input/1d/DJI.csv")
        solution = DJI['Close']
        Dmin = [0]
        Dmax = [100.]
        xcoord = np.linspace(Dmin[0], Dmax[0], nPoints)

    nPoints[0] = solution.shape[0]

elif dimension == 2:
    print('Setting up problem for 2-D')

    if problem == 0:
        nPoints[0] = 1025
        nPoints[1] = 2049
        scale = 1
        Dmin = [0., 0.]
        Dmax = [1., 1.]

        xcoord = np.linspace(Dmin[0], Dmax[0], nPoints[0])
        ycoord = np.linspace(Dmin[1], Dmax[1], nPoints[1])
        solution = 10*np.ones((nPoints[0], nPoints[1]))
        solution[:, 0] = 0
        solution[:, -1] = 0
        solution[0, :] = 0
        solution[-1, :] = 0

        debugProblem = True

    elif problem == 1:
        nPoints[0] = 2048
        nPoints[1] = 1025
        scale = 100
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
        solution = scale * (np.sinc(np.sqrt(X**2 + Y**2)) +
                            np.sinc(2*((X-2)**2 + (Y+2)**2))).T

        # solution = scale * (np.sinc((X+1)**2 + (Y-1)**2) + np.sinc(((X-1)**2 + (Y+1)**2)))
        # solution = X**2 * (DmaxX - Y)**2 + X**2 * Y**2 + 64 * (np.sinc(np.sqrt((X-2) ** 2 + (Y+2)**2)))
        # solution = (3-abs(X))**3 * (1-abs(Y))**5 + (1-abs(X))**3 * (3-abs(Y))**5
        # solution = (Dmax[0]+X)**5 * (Dmax[1]-Y)**5
        # solution = solution / np.linalg.norm(solution)
        # solution = scale * (np.sinc(X) * np.sinc(Y))
        # solution = solution.T
        # (3*degree + 1) #minimum number of control points
        del X, Y

    elif problem == 2:
        nPoints[0] = 501
        nPoints[1] = 501
        scale = 1
        shiftX = 0.25*0
        shiftY = -0.25*0
        Dmin = [0., 0.]
        Dmin = [-math.pi, -math.pi]
        Dmax = [math.pi, math.pi]

        xcoord = np.linspace(Dmin[0], Dmax[0], nPoints[0])
        ycoord = np.linspace(Dmin[1], Dmax[1], nPoints[1])
        X, Y = np.meshgrid(xcoord+shiftX, ycoord+shiftY)
        # solution = scale * np.sinc(X) * np.sinc(Y)
        solution = scale * np.cos(X+1) * np.cos(Y) + \
            scale * np.sin(X-1) * np.sin(Y)
        # solution = scale * X * Y
        # (3*degree + 1) #minimum number of control points
        del X, Y

    elif problem == 3:
        solution = np.fromfile(
            "input/2d/nek5000.raw", dtype=np.float64).reshape(200, 200)
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
        solution = np.fromfile(
            "input/2d/s3d_2D.raw", dtype=np.float64).reshape(540, 704)
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
        solution = np.fromfile("input/2d/FLDSC_1_1800_3600.dat",
                               dtype=np.float32).reshape(1800, 3600)
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
                mandelbrot_set[xi, yi] = (
                    i % 4 * 64) * 65536 + (i % 8 * 32) * 256 + (i % 16 * 16)

        # image.show()

        z = mandelbrot_set.T / 1e5

        plt.imshow(z, extent=[DminX, DmaxX, DminY, DmaxY])
        plt.show()

        if nControlPointsInputIn == 0:
            nControlPointsInputIn = 50

    else:
        print('Not a valid problem')
        exit(1)


nControlPointsInput = np.array(
    [nControlPointsInputIn] * dimension, dtype=np.uint32)
# if dimension == 2:
#     nControlPointsInput = np.array([12, 15], dtype=np.uint32)

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
solutionShape = solution.shape


def plot_solution(solVector):
    from uvw import RectilinearGrid, DataArray
    if dimension == 1:
        mpl_fig = plt.figure()
        plt.plot(xcoord, solVector, 'r-', ms=2)
        mpl_fig.show()
    elif dimension == 2:
        # x, y = np.meshgrid(xcoord, ycoord)
        # x = x.reshape(1, x.shape[1], x.shape[0])
        # y = x.reshape(1, y.shape[1], y.shape[0])
        # gridToVTK("./structured", xcoord, ycoord, np.ones(1),
        #           pointData={"solution": solVector.reshape(1, xcoord.shape[0], ycoord.shape[0])})
        with RectilinearGrid("./structured.vtr", (xcoord, ycoord)) as rect:
            rect.addPointData(DataArray(solVector, range(2), 'solution'))
    elif dimension == 3:
        with RectilinearGrid("./structured.vtr", (xcoord, ycoord, zcoord)) as rect:
            rect.addPointData(DataArray(solVector, range(3), 'solution'))
        # x, y, z = np.meshgrid(xcoord, ycoord, zcoord)
        # gridToVTK(
        #     "./structured", x.reshape(x.shape[0],
        #                               x.shape[1],
        #                               x.shape[2]),
        #     y.reshape(y.shape[0],
        #               y.shape[1],
        #               y.shape[2]),
        #     z.reshape(x.shape[0],
        #               x.shape[1],
        #               x.shape[2]),
        #     pointData={"solution": solVector.reshape(x.shape[0],
        #                                              x.shape[1],
        #                                              x.shape[2])})
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
    print('problem = ', problem,
          '[1 = sinc, 2 = sine, 3 = Nek5000, 4 = S3D, 5 = CESM]')
    print('Total number of input points: ', np.prod(nPoints))
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


def flattenList(t):
    return [item for sublist in t for item in sublist]


def flattenDict(d):
    return(reduce(
        lambda new_d, kv:
        isinstance(kv[1], dict) and
        {**new_d, **flatten(kv[1], kv[0])} or
            {**new_d, kv[0]: kv[1]},
        d.items(),
        {}
    ))


def flattenListDict(t):
    ndict = {}
    for item in t:
        ndict.update(item)
    return ndict


def WritePVTKFile(iteration):
    # print(globalExtentDict)
    pvtkfile = open("pstructured-mfa-%d.pvtr" % (iteration), "w")

    pvtkfile.write('<?xml version="1.0"?>\n')
    pvtkfile.write(
        '<VTKFile type="PRectilinearGrid" version="1.0" byte_order="LittleEndian" header_type="UInt64">\n')
    pvtkfile.write('<PRectilinearGrid WholeExtent="%d %d %d %d 0 0" GhostLevel="0">\n' %
                   (0, solutionShape[0]-1, 0, solutionShape[1]-1))
    pvtkfile.write('\n')
    pvtkfile.write('    <PPointData>\n')
    pvtkfile.write('      <PDataArray type="Float64" Name="solution"/>\n')
    pvtkfile.write('      <PDataArray type="Float64" Name="error"/>\n')
    pvtkfile.write('    </PPointData>\n')
    pvtkfile.write('    <PCellData></PCellData>\n')
    pvtkfile.write('    <PCoordinates>\n')
    pvtkfile.write(
        '      <PDataArray type="Float64" Name="x_coordinates" NumberOfComponents="1"/>\n')
    pvtkfile.write(
        '      <PDataArray type="Float64" Name="y_coordinates" NumberOfComponents="1"/>\n')
    pvtkfile.write(
        '      <PDataArray type="Float64" Name="z_coordinates" NumberOfComponents="1"/>\n')
    pvtkfile.write('    </PCoordinates>\n\n')

    isubd = 0
    # dx = (xyzMax[0]-xyzMin[0])/nSubDomainsX
    # dy = (xyzMax[1]-xyzMin[1])/nSubDomainsY
    # dxyz = [solutionShape[0]/nSubDomains[0], solutionShape[1]/nSubDomains[1]]
    # print("Dxyz = ", dxyz)
    # yoff = 0
    # ncy = 0
    for iy in range(nSubDomainsY):
        # xoff = 0
        # ncx = 0
        for ix in range(nSubDomainsX):
            # pvtkfile.write(
            #     '    <Piece Extent="0.0 0.0 %f %f %f %f" Source="structured-%d-%d.vts"/>\n' %
            #     (max(xoff-dxyz[0], xyzMin[0]), min(xyzMax[0], xoff+dxyz[0]), max(yoff-dxyz[1], xyzMin[1]), min(xyzMax[1], yoff+dxyz[1]), isubd, iteration))

            pvtkfile.write(
                '    <Piece Extent="%d %d %d %d 0 0" Source="structured-%d-%d.vtr"/>\n' %
                (0,
                 globalExtentDict[isubd][1]-globalExtentDict[isubd][0]-1,
                 0,
                 globalExtentDict[isubd][3]-globalExtentDict[isubd][2]-1,
                 isubd, iteration))
            isubd += 1
            # xoff += globalExtentDict[isubd][0]
            # ncx += nControlPointsInput[0]
        # yoff += globalExtentDict[isubd][1]
        # ncy += nControlPointsInput[1]
    pvtkfile.write('\n')
    pvtkfile.write('</PRectilinearGrid>\n')
    pvtkfile.write('</VTKFile>\n')

    pvtkfile.close()

# Write control point data


def WritePVTKControlFile(iteration):
    nconstraints = (int(degree/2.0) if not degree %
                    2 else int((degree+1)/2.0))
    # nconstraints=1
    print('Nconstraints = ', nconstraints)
    pvtkfile = open("pstructured-control-mfa-%d.pvtr" % (iteration), "w")

    pvtkfile.write('<?xml version="1.0"?>\n')
    pvtkfile.write(
        '<VTKFile type="PRectilinearGrid" version="1.0" byte_order="LittleEndian" header_type="UInt64">\n')
    pvtkfile.write('<PRectilinearGrid WholeExtent="%d %d %d %d 0 0" GhostLevel="%d">\n' % (
        0, nSubDomainsX*nControlPointsInput[0]-1, 0, nSubDomainsY*nControlPointsInput[1]-1, nconstraints+augmentSpanSpace))
    pvtkfile.write('\n')
    pvtkfile.write('    <PCellData>\n')
    pvtkfile.write('      <PDataArray type="Float64" Name="controlpoints"/>\n')
    pvtkfile.write('    </PCellData>\n')
    pvtkfile.write('    <PPoints>\n')
    pvtkfile.write(
        '      <PDataArray type="Float64" Name="points" NumberOfComponents="3"/>\n')
    pvtkfile.write('    </PPoints>\n')

    isubd = 0
    # dx = (xyzMax[0]-xyzMin[0])/nSubDomainsX
    # dy = (xyzMax[1]-xyzMin[1])/nSubDomainsY
    dxyz = (xyzMax-xyzMin)/nSubDomains
    yoff = xyzMin[1]
    ncy = 0
    for iy in range(nSubDomainsY):
        xoff = xyzMin[0]
        ncx = 0
        for ix in range(nSubDomainsX):
            pvtkfile.write(
                '    <Piece Extent="0 0 %d %d %d %d" Source="structuredcp-%d-%d.vtr"/>\n' %
                (ncy, ncy+nControlPointsInput[1]-1, ncx, ncx+nControlPointsInput[0]-1, isubd, iteration))
            isubd += 1
            xoff += dxyz[0]
            ncx += nControlPointsInput[0]
        yoff += dxyz[1]
        ncy += nControlPointsInput[1]
    pvtkfile.write('\n')
    pvtkfile.write('</PRectilinearGrid>\n')
    pvtkfile.write('</VTKFile>\n')

    pvtkfile.close()


# ------------------------------------

EPS = 1e-32
GTOL = 1e-2


def compute_decode_operators(iNuvw):
    RN = {'x': [], 'y': [], 'z': []}
    if dimension == 1:
        W = np.ones(iNuvw['x'].shape[1])
        RN['x'] = (iNuvw['x']*W)/(np.sum(iNuvw['x']*W, axis=1)[:, np.newaxis])

    elif dimension == 2:
        W = np.ones((iNuvw['x'].shape[1], iNuvw['y'].shape[1]))
        RN['x'] = iNuvw['x'] * np.sum(W, axis=1)
        # assert( np.min(np.sum(RN['x']*np.sum(W, axis=1), axis=1)) > 1e-8)
        RN['x'] /= np.sum(RN['x']*np.sum(W, axis=1), axis=1)[:, np.newaxis]
        RN['y'] = iNuvw['y'] * np.sum(W, axis=0)
        # assert( np.min(np.sum(RN['x']*np.sum(W, axis=1), axis=1)) > 1e-8)
        RN['y'] /= np.sum(RN['y']*np.sum(W, axis=0), axis=1)[:, np.newaxis]
        # print('Decode error res: ', RNx.shape, RNy.shape)
        # decoded = np.matmul(np.matmul(RNx, P), RNy.T)

    else:
        error('No implementation')

    return RN


def decode_2D(P, RN):
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

        # RNx = iNu * np.sum(W, axis=1)
        # RNx /= np.sum(RNx, axis=1)[:, np.newaxis]
        # RNy = iNv * np.sum(W, axis=0)
        # RNy /= np.sum(RNy, axis=1)[:, np.newaxis]
        # print('Decode error res: ', RNx.shape, RNy.shape)

        decoded = np.matmul(np.matmul(RN['x'], P), RN['y'].T)
        # print('Decode error res: ', decoded.shape, decoded2.shape, np.max(np.abs(decoded.reshape((Nu.shape[0], Nv.shape[0])) - decoded2)))
        # print('Decode error res: ', z.shape, decoded.shape)

    return decoded


def decode_1D(P, RN):
    # RN = Nu * np.sum(W, axis=0)
    # RN = (Nu*W)/(np.sum(Nu*W, axis=1)[:, np.newaxis])
    return (RN['x'] @ P)   # self.RN.dot(P)


def decode(P, RN):
    if dimension == 1:
        return decode_1D(P, RN)
    elif dimension == 2:
        return decode_2D(P, RN)
    else:
        error('Not implemented and invalid dimension')


def lsqFit(RNx, RNy, z):
    if dimension == 1:
        # RN = (Nu*W)/(np.sum(Nu*W, axis=1)[:, np.newaxis])
        z = z.reshape(z.shape[0], 1)
        return linalg.lstsq(RNx, z)[0]
    elif dimension == 2:
        use_cho = False
        # RNx = Nu * np.sum(W, axis=1)
        # RNx /= np.sum(RNx, axis=1)[:, np.newaxis]
        # RNy = Nv * np.sum(W, axis=0)
        # RNy /= np.sum(RNy, axis=1)[:, np.newaxis]
        if use_cho:
            X = linalg.cho_solve(linalg.cho_factor(
                np.matmul(RNx.T, RNx)), RNx.T)
            Y = linalg.cho_solve(linalg.cho_factor(
                np.matmul(RNy.T, RNy)), RNy.T)
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
        if useMOABMesh:
            self.mbInterface = core.Core()
            self.scdGrid = ScdInterface(self.mbInterface)

        if dimension == 1:
            self.corebounds = [[coreb.min[0] -
                               xb.min[0], -1+coreb.max[0]-xb.max[0]]]
            self.xyzCoordLocal = {'x': xl[:]}
            self.Dmini = np.array([min(xl)])
            self.Dmaxi = np.array([max(xl)])
            self.basisFunction = {'x': None}  # Basis function object in x-dir
            self.decodeOpXYZ = {'x': None}
            self.knotsAdaptive = {'x': []}
            self.isClamped = {'left': False, 'right': False}

        elif dimension == 2:
            self.corebounds = [[coreb.min[0]-xb.min[0], -1+coreb.max[0]-xb.max[0]],
                               [coreb.min[1]-xb.min[1], -1+coreb.max[1]-xb.max[1]]]
            # int(nPointsX / nSubDomainsX)
            self.xyzCoordLocal = {'x': xl[:],
                                  'y': yl[:]}
            self.Dmini = np.array([min(xl), min(yl)])
            self.Dmaxi = np.array([max(xl), max(yl)])
            # Basis function object in x-dir and y-dir
            self.basisFunction = {'x': None, 'y': None}
            self.decodeOpXYZ = {'x': None, 'y': None}
            self.knotsAdaptive = {
                'x': [],
                'y': []}
            self.isClamped = {'left': False, 'right': False,
                              'top': False, 'bottom': False}

        else:

            error('Not implemented')

        self.refSolutionLocal = np.copy(pl)
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
        self.figErrHnd = None
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
                "Rank: %d, Subdomain %d: Bounds = [%d - %d]" %
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

    def update_bounds(self, cp):

        if dimension == 1:
            extents = [self.corebounds[0][0], self.corebounds[0][1]]
        elif dimension == 2:
            extents = [
                self.corebounds[0][0],
                self.corebounds[0][1],
                self.corebounds[1][0],
                self.corebounds[1][1]]
        else:
            extents = [
                self.corebounds[0][0],
                self.corebounds[0][1],
                self.corebounds[1][0],
                self.corebounds[1][1], self.corebounds[2][0], self.corebounds[2][1]]

        localExtents[cp.gid()] = extents

    def compute_basis_1D(self, degree, knotVectors):
        self.basisFunction['x'] = sp.BSplineBasis(
            order=degree+1, knots=knotVectors['x'])
        print("Number of basis functions = ",
              self.basisFunction['x'].num_functions())
        # print("TU = ", knotVectors['x'], self.UVW['x'][0], self.UVW['x'][-1], self.basisFunction['x'].greville())
        self.NUVW['x'] = np.array(
            self.basisFunction['x'].evaluate(self.UVW['x']))

    def compute_basis_2D(self, degree, knotVectors):
        self.basisFunction['x'] = sp.BSplineBasis(
            order=degree+1, knots=knotVectors['x'])
        self.basisFunction['y'] = sp.BSplineBasis(
            order=degree+1, knots=knotVectors['y'])
        # print("TU = ", knotVectors['x'], self.UVW['x'][0], self.UVW['x'][-1], self.basisFunction['x'].greville())
        # print("TV = ", knotVectors['y'], self.UVW['y'][0], self.UVW['y'][-1], self.basisFunction['y'].greville())
        self.NUVW['x'] = np.array(
            self.basisFunction['x'].evaluate(self.UVW['x']))
        self.NUVW['y'] = np.array(
            self.basisFunction['y'].evaluate(self.UVW['y']))

    def compute_basis(self, degree, knotVectors):
        coords = []
        verts = []
        if dimension == 1:
            self.compute_basis_1D(degree, knotVectors)

            if useMOABMesh:
                verts = self.mbInterface.get_entities_by_type(0, types.MBVERTEX)
                if len(verts) == 0:
                    # Now let us generate a MOAB SCD box
                    xc = self.basisFunction['x'].greville()
                    for xi in xc:
                        coords += [xi, 0.0, 0.0]
                    scdbox = self.scdGrid.construct_box(
                        HomCoord([0, 0, 0, 0]), HomCoord([len(xc) - 1, 0, 0, 0]), coords)

        elif dimension == 2:
            self.compute_basis_2D(degree, knotVectors)

            if useMOABMesh:
                verts = self.mbInterface.get_entities_by_type(0, types.MBVERTEX)
                if len(verts) == 0:
                    # Now let us generate a MOAB SCD box
                    xc = self.basisFunction['x'].greville()
                    yc = self.basisFunction['y'].greville()
                    for yi in yc:
                        for xi in xc:
                            coords += [xi, yi, 0.0]
                    scdbox = self.scdGrid.construct_box(
                        HomCoord([0, 0, 0, 0]),
                        HomCoord([len(xc) - 1, len(yc) - 1, 0, 0]),
                        coords)

        else:
            error('Invalid dimension')

        if useMOABMesh:
            print("MOAB structured mesh now has", len(verts), "vertices")

    def output_solution(self, cp):

        if dimension == 1:

            # axHnd = self.figHnd.gca()
            self.pMK = decode(self.controlPointData, self.decodeOpXYZ)

            xl = self.xyzCoordLocal['x'].reshape(
                self.xyzCoordLocal['x'].shape[0], 1)

            plt.subplot(211)
            # Plot the control point solution
            coeffs_x = self.basisFunction['x'].greville()
            plt.plot(xl, self.pMK, linestyle='--', lw=2,
                     color=['r', 'g', 'b', 'y', 'c'][cp.gid() % 5], label="Decoded-%d" % (cp.gid()+1))
            plt.plot(coeffs_x, self.controlPointData, marker='o', linestyle='', color=[
                'r', 'g', 'b', 'y', 'c'][cp.gid() % 5], label="Control-%d" % (cp.gid()+1))

            # Plot the error
            errorDecoded = (
                self.refSolutionLocal.reshape(self.refSolutionLocal.shape[0],
                                              1) - self.pMK.reshape(self.pMK.shape[0],
                                                                    1))  # / solutionRange
            print('Error shape: ', errorDecoded.shape,
                  self.xyzCoordLocal['x'].shape)
            plt.subplot(212)
            plt.plot(xl, errorDecoded,
                     # plt.plot(self.xyzCoordLocal['x'],
                     #          error,
                     linestyle='--', color=['r', 'g', 'b', 'y', 'c'][cp.gid() % 5],
                     lw=2, label="Subdomain(%d) Error" % (cp.gid() + 1))

        else:
            if useVTKOutput:

                self.pMK = decode(self.controlPointData,
                                  self.decodeOpXYZ)
                errorDecoded = (self.refSolutionLocal -
                                self.pMK) / solutionRange

                locX = []
                locY = []
                if augmentSpanSpace > 0:
                    locX = self.xyzCoordLocal['x'][self.corebounds[0][0]: self.corebounds[0][1]]
                    locY = self.xyzCoordLocal['y'][self.corebounds[1][0]: self.corebounds[1][1]]

                    coreData = np.ascontiguousarray(
                        self.pMK
                        [self.corebounds[0][0]: self.corebounds[0][1],
                         self.corebounds[1][0]: self.corebounds[1][1]])
                    errorDecoded = np.ascontiguousarray(
                        errorDecoded
                        [self.corebounds[0][0]: self.corebounds[0][1],
                         self.corebounds[1][0]: self.corebounds[1][1]])

                else:

                    locX = self.xyzCoordLocal['x']
                    locY = self.xyzCoordLocal['y']
                    coreData = self.pMK

                print(cp.gid(), ' Min, Max - ', np.min(locX), np.max(locX), np.min(locY), np.max(locY))
                # Xi, Yi = np.meshgrid(locX, locY)
                # Xi = Xi.reshape(1, Xi.shape[0], Xi.shape[1])
                # Yi = Yi.reshape(1, Yi.shape[0], Yi.shape[1])
                # Zi = np.ones(Xi.shape)
                # PmK = coreData.T.reshape(
                #     1, coreData.shape[1], coreData.shape[0])
                # errorDecoded = errorDecoded.T.reshape(
                #     1, errorDecoded.shape[1], errorDecoded.shape[0])
                # gridToVTK("./structured-%s" % (self.figSuffix), Xi, Yi, Zi,
                #           pointData={"solution": PmK, "error": errorDecoded})
                # del Xi, Yi, Zi
                gridToVTK("./structured-%s" % (self.figSuffix), locX, locY, np.ones(1),
                          pointData={"solution": coreData.reshape(1, locX.shape[0], locY.shape[0]),
                                     "error": errorDecoded.reshape(1, locX.shape[0], locY.shape[0])
                                     }
                          )

                # replace the whole extent
                # pvtkfile.write('<PRectilinearGrid WholeExtent="%d %d %d %d 0 0" GhostLevel="0">\n' %
                #                (0, solutionShape[0]-1, 0, solutionShape[1]-1))

                cpx = np.array(self.basisFunction['x'].greville())
                cpy = np.array(self.basisFunction['y'].greville())

                # Xi, Yi = np.meshgrid(cpx, cpy)
                # print('Structured-CP-', self.figSuffix, np.min(cpx),
                #       np.max(cpx), np.min(cpy), np.max(cpy))

                # Xi = Xi.reshape(1, Xi.shape[0], Xi.shape[1])
                # Yi = Yi.reshape(1, Yi.shape[0], Yi.shape[1])
                # Zi = np.ones(Xi.shape)
                # gridToVTK("./structuredcp-%s" % (self.figSuffix), Xi, Yi, Zi,
                #           pointData={"controlpoints": self.controlPointData.T.reshape(
                #               1, self.controlPointData.shape[1], self.controlPointData.shape[0])})
                gridToVTK("./structuredcp-%s" % (self.figSuffix), cpx, cpy, np.ones(1),
                          pointData={"controlpoints": self.controlPointData.reshape(1, cpx.shape[0], cpy.shape[0])})
                # gridToVTK("./structured", xcoord, ycoord, np.ones(1), pointData={"solution": solVector.reshape(1, xcoord.shape[0], ycoord.shape[0])})

                if False and cp.gid() == 0:
                    self.PlotControlPoints()

    def output_vtk(self, cp):

        from uvw import RectilinearGrid, DataArray
        from uvw.parallel import PRectilinearGrid

        assert(useVTKOutput)

        self.pMK = decode(self.controlPointData,
                          self.decodeOpXYZ)
        errorDecoded = (self.refSolutionLocal -
                        self.pMK) / solutionRange

        locX = []
        locY = []
        if augmentSpanSpace > 0:
            locX = self.xyzCoordLocal['x'][self.corebounds[0][0]: self.corebounds[0][1]]
            locY = self.xyzCoordLocal['y'][self.corebounds[1][0]: self.corebounds[1][1]]

            coreData = np.ascontiguousarray(
                self.pMK
                [self.corebounds[0][0]: self.corebounds[0][1],
                    self.corebounds[1][0]: self.corebounds[1][1]])
            errorDecoded = np.ascontiguousarray(
                errorDecoded
                [self.corebounds[0][0]: self.corebounds[0][1],
                    self.corebounds[1][0]: self.corebounds[1][1]])

        else:

            locX = self.xyzCoordLocal['x']
            locY = self.xyzCoordLocal['y']
            coreData = self.pMK

        # Indicating rank info with a cell array
        proc = np.ones((locX.size-1, locY.size-1)) * commWorld.Get_rank()
        # print("Writing using uvw VTK writer to uwv-structured-%s.pvtr" % (self.figSuffix))
        # with PRectilinearGrid("./structured-%s.vtr" % (self.figSuffix), (locX, locY), [self.xbounds.min[0], self.xbounds.min[1]]) as rect:
        with RectilinearGrid("./structured-%s.vtr" % (self.figSuffix), (locX, locY)) as rect:
            rect.addPointData(DataArray(coreData, range(2), 'solution'))
            rect.addPointData(DataArray(errorDecoded, range(2), 'error'))
            rect.addCellData(DataArray(proc, range(2), 'process'))

        # replace the whole extent
        # pvtkfile.write('<PRectilinearGrid WholeExtent="%d %d %d %d 0 0" GhostLevel="0">\n' %
        #                (0, solutionShape[0]-1, 0, solutionShape[1]-1))

        # cpx = np.array(self.basisFunction['x'].greville())
        # cpy = np.array(self.basisFunction['y'].greville())

        # gridToVTK("./structuredcp-%s" % (self.figSuffix), cpx, cpy, np.ones(1),
        #           pointData={"controlpoints": self.controlPointData.reshape(1, cpx.shape[0], cpy.shape[0])})

    def PlotControlPoints(self):
        import pyvista as pv
        import numpy as np

        # Create the spatial reference
        grid = pv.UniformGrid()

        cpx = self.basisFunction['x'].greville()
        cpy = self.basisFunction['y'].greville()
        # cpz = np.ones(len(cpx))
        Xi, Yi = np.meshgrid(cpx, cpy)

        print(Xi.shape, Yi.shape, self.controlPointData.shape)

        points = np.c_[Xi.reshape(-1), Yi.reshape(-1),
                       np.ones(self.controlPointData.T.reshape(-1).shape)]

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

        print("Domain: ", cp.gid()+1, "Exact = ", self.refSolutionLocal)
        print("Domain: ", cp.gid()+1, "Exact - Decoded = ",
              np.abs(self.refSolutionLocal - self.pMK))

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
                        cp.enqueue(
                            target, self.knotsAdaptive['y'][-1:-2-degree-augmentSpanSpace:-1])

                    else:  # target block is below current subdomain
                        if verbose:
                            print("%d sending to %d" % (cp.gid(), target.gid), ' Bottom: ',
                                  self.controlPointData[:, 0:1+degree+augmentSpanSpace].shape)

                        # cp.enqueue(target, self.controlPointData[:, 0:1+degree+augmentSpanSpace])
                        cp.enqueue(target, self.controlPointData)
                        cp.enqueue(target, self.knotsAdaptive['x'][:])
                        cp.enqueue(
                            target, self.knotsAdaptive['y'][0:1+degree+augmentSpanSpace])

                # target is coupled in X-direction
                elif dimension == 1 or dir[1] == 0:
                    if dir[0] > 0:  # target block is to the right of current subdomain
                        if verbose:
                            print("%d sending to %d" % (cp.gid(), target.gid), 'Left: ',
                                  self.controlPointData[-1:-2-degree-augmentSpanSpace:-1, :].shape)

                        # cp.enqueue(target, self.controlPointData[-1-degree-augmentSpanSpace:, :])
                        cp.enqueue(target, self.controlPointData)
                        if dimension > 1:
                            cp.enqueue(target, self.knotsAdaptive['y'][:])
                        cp.enqueue(
                            target, self.knotsAdaptive['x'][-1:-2-degree-augmentSpanSpace:-1])

                    else:  # target block is to the left of current subdomain
                        if verbose:
                            print("%d sending to %d" % (cp.gid(), target.gid), 'Right: ',
                                  self.controlPointData[degree+augmentSpanSpace::-1, :].shape)

                        # cp.enqueue(target, self.controlPointData[0:1+degree+augmentSpanSpace, :])
                        cp.enqueue(target, self.controlPointData)
                        if dimension > 1:
                            cp.enqueue(target, self.knotsAdaptive['y'][:])
                        cp.enqueue(target, self.knotsAdaptive['x'][0:(
                            degree+augmentSpanSpace+1)])

                else:

                    if useDiagonalBlocks:
                        # target block is diagonally top right to current subdomain
                        if dir[0] > 0 and dir[1] > 0:

                            cp.enqueue(target, self.controlPointData)
                            # cp.enqueue(target, self.controlPointData[-1-degree-augmentSpanSpace:, -1-degree-augmentSpanSpace:])
                            if verbose:
                                print("%d sending to %d" % (cp.gid(), target.gid), ' Diagonal = right-top: ',
                                      self.controlPointData[-1:-2-degree-augmentSpanSpace:-1, :1+degree+augmentSpanSpace])
                        # target block is diagonally top left to current subdomain
                        if dir[0] < 0 and dir[1] > 0:
                            cp.enqueue(target, self.controlPointData)
                            # cp.enqueue(target, self.controlPointData[: 1 + degree + augmentSpanSpace, -1:-2-degree-augmentSpanSpace:-1])
                            if verbose:
                                print("%d sending to %d" % (cp.gid(), target.gid), ' Diagonal = left-top: ',
                                      self.controlPointData[-1:-2-degree-augmentSpanSpace:-1, :1+degree+augmentSpanSpace])

                        # target block is diagonally left bottom  current subdomain
                        if dir[0] < 0 and dir[1] < 0:
                            cp.enqueue(target, self.controlPointData)
                            # cp.enqueue(target, self.controlPointData[-1-degree-augmentSpanSpace:, :1+degree+augmentSpanSpace])

                            if verbose:
                                print("%d sending to %d" % (cp.gid(), target.gid), ' Diagonal = left-bottom: ',
                                      self.controlPointData[:1+degree+augmentSpanSpace,  -1-degree-augmentSpanSpace:])
                        # target block is diagonally right bottom of current subdomain
                        if dir[0] > 0 and dir[1] < 0:
                            cp.enqueue(target, self.controlPointData)
                            # cp.enqueue(target, self.controlPointData[:1+degree+augmentSpanSpace, :1+degree+augmentSpanSpace])
                            if verbose:
                                print("%d sending to %d" % (cp.gid(), target.gid), ' Diagonal = right-bottom: ',
                                      self.controlPointData[:1+degree+augmentSpanSpace,  -1 - degree - augmentSpanSpace:])

        return

    def recv_diy(self, cp):

        # oddDegree = (degree % 2)
        # nconstraints = augmentSpanSpace + \
        #     (int(degree/2.0) if not oddDegree else int((degree+1)/2.0))

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

                    # if oddDegree:
                    #     self.controlPointData[:,-(degree-nconstraints):] = self.boundaryConstraints['top'][:, :degree-nconstraints]
                    # else:
                    #     self.controlPointData[:,-(degree-nconstraints)+1:] = self.boundaryConstraints['top'][:, :degree-nconstraints]

                else:  # target block is below current subdomain
                    self.boundaryConstraints['bottom'] = cp.dequeue(tgid)
                    self.boundaryConstraintKnots['bottom'] = cp.dequeue(tgid)
                    self.ghostKnots['bottom'] = cp.dequeue(tgid)
                    if verbose:
                        print("Bottom: %d received from %d: from direction %s" %
                              (cp.gid(), tgid, dir), self.bottomconstraint.shape, self.bottomconstraintKnots.shape)

                    # if oddDegree:
                    #     self.controlPointData[:,:(degree-nconstraints)] = self.boundaryConstraints['bottom'][:, -(degree-nconstraints):]
                    # else:
                    #     self.controlPointData[:,:(degree-nconstraints)] = self.boundaryConstraints['bottom'][:, -(degree-nconstraints)+1:]

            # target is coupled in X-direction
            elif dimension == 1 or dir[1] == 0:
                if dir[0] < 0:  # target block is to the left of current subdomain
                    # print('Right: ', np.array(o[2:pl+2]).reshape(useDerivativeConstraints+1,pll).T)

                    self.boundaryConstraints['left'] = cp.dequeue(tgid)
                    if dimension > 1:
                        self.boundaryConstraintKnots['left'] = cp.dequeue(tgid)
                    self.ghostKnots['left'] = cp.dequeue(tgid)
                    if verbose:
                        print("Left: %d received from %d: from direction %s" %
                              (cp.gid(), tgid, dir), self.leftconstraint.shape, self.leftconstraintKnots.shape)

                    # if oddDegree:
                    #     self.controlPointData[:(degree-nconstraints),:] = self.boundaryConstraints['left'][-(degree-nconstraints):,:]
                    # else:
                    #     self.controlPointData[:(degree-nconstraints),] = self.boundaryConstraints['left'][-(degree-nconstraints)+1:,:]

                else:  # target block is to right of current subdomain

                    self.boundaryConstraints['right'] = cp.dequeue(tgid)
                    if dimension > 1:
                        self.boundaryConstraintKnots['right'] = cp.dequeue(
                            tgid)
                    self.ghostKnots['right'] = cp.dequeue(tgid)
                    if verbose:
                        print("Right: %d received from %d: from direction %s" %
                              (cp.gid(), tgid, dir), self.rightconstraint.shape, self.rightconstraintKnots.shape)

                    # if oddDegree:
                    #     self.controlPointData[-(degree-nconstraints):,:] = self.boundaryConstraints['right'][:(degree-nconstraints):,:]
                    # else:
                    #     self.controlPointData[-(degree-nconstraints):,:] = self.boundaryConstraints['right'][:(degree-nconstraints):,:]

            else:

                if useDiagonalBlocks:
                    # 2-Dimension = 0: left, 1: right, 2: top, 3: bottom, 4: top-left, 5: top-right, 6: bottom-left, 7: bottom-right
                    # sender block is diagonally right top to  current subdomain
                    if dir[0] > 0 and dir[1] > 0:
                        self.boundaryConstraints['top-right'] = cp.dequeue(
                            tgid)
                        if verbose:
                            print("Top-right: %d received from %d: from direction %s" %
                                  (cp.gid(), tgid, dir), self.boundaryConstraints['top-right'].shape)
                    # sender block is diagonally left top to current subdomain
                    if dir[0] > 0 and dir[1] < 0:
                        self.boundaryConstraints['bottom-right'] = cp.dequeue(
                            tgid)
                        if verbose:
                            print("Bottom-right: %d received from %d: from direction %s" %
                                  (cp.gid(), tgid, dir), self.boundaryConstraints['bottom-right'].shape)
                    # sender block is diagonally left bottom  current subdomain
                    if dir[0] < 0 and dir[1] < 0:
                        self.boundaryConstraints['bottom-left'] = cp.dequeue(
                            tgid)
                        if verbose:
                            print("Bottom-left: %d received from %d: from direction %s" %
                                  (cp.gid(), tgid, dir), self.boundaryConstraints['bottom-left'].shape)
                    # sender block is diagonally left to current subdomain
                    if dir[0] < 0 and dir[1] > 0:

                        self.boundaryConstraints['top-left'] = cp.dequeue(tgid)
                        if verbose:
                            print("Top-left: %d received from %d: from direction %s" %
                                  (cp.gid(), tgid, dir), self.boundaryConstraints['top-left'].shape)

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
                    e[i - 1, j - 1] = e[i - 2, j - 3] + 1 / \
                        (e[i - 1, j - 2] - e[i - 2, j - 2])
                else:
                    # + 1 / (e[i - 1, j - 2] - e[i - 2, j - 2])
                    e[i - 1, j - 1] = e[i - 2, j - 3]

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
                e[i, j] = e[i - 1, j - 2] + 1.0 / \
                    (e[i, j - 1] - e[i - 1, j - 1])
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
        self.solutionLocalHistory[:, 0] = np.copy(
            self.controlPointData).reshape(plen)

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
                      np.linalg.norm(self.controlPointData.reshape(plen) - vAcc))  # , (self.refSolutionLocal - vAcc))
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

        tu = np.linspace(
            self.Dmini[0] + inc[0], self.Dmaxi[0] - inc[0], self.nInternalKnotSpans[0] - 1)
        if dimension > 1:
            tv = np.linspace(
                self.Dmini[1] + inc[1], self.Dmaxi[1] - inc[1], self.nInternalKnotSpans[1] - 1)

        if nTotalSubDomains > 1 and not fullyPinned:

            print("Subdomain: ", cp.gid(), " X: ",
                  self.Dmini[0], self.Dmaxi[0])
            if abs(self.Dmaxi[0] - xyzMax[0]) < 1e-12 and abs(self.Dmini[0] - xyzMin[0]) < 1e-12:
                self.knotsAdaptive['x'] = np.concatenate(
                    ([self.Dmini[0]] * (degree + 1),
                     tu, [self.Dmaxi[0]] * (degree + 1)))
                self.isClamped['left'] = self.isClamped['right'] = True
            else:
                if abs(self.Dmaxi[0] - xyzMax[0]) < 1e-12:
                    self.knotsAdaptive['x'] = np.concatenate(
                        ([self.Dmini[0]] * (1), tu, [self.Dmaxi[0]] * (degree+1)))
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
                print("Subdomain: ", cp.gid(), " Y: ",
                      self.Dmini[1], self.Dmaxi[1])
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
            self.knotsAdaptive['x'] = np.concatenate(
                ([self.Dmini[0]] * (degree+1), tu, [self.Dmaxi[0]] * (degree+1)))
            self.isClamped['left'] = self.isClamped['right'] = True

            if dimension > 1:
                self.knotsAdaptive['y'] = np.concatenate(
                    ([self.Dmini[1]] * (degree + 1),
                     tv, [self.Dmaxi[1]] * (degree + 1)))
                self.isClamped['top'] = self.isClamped['bottom'] = True

        # self.UVW['x'] = np.linspace(
        #     self.xyzCoordLocal['x'][0], self.xyzCoordLocal['x'][-1], self.nPointsPerSubD[0])
        self.UVW['x'] = self.xyzCoordLocal['x']
        if dimension > 1:
            # self.UVW['y'] = np.linspace(
            #     self.xyzCoordLocal['y'][0], self.xyzCoordLocal['y'][-1], self.nPointsPerSubD[1])
            self.UVW['y'] = self.xyzCoordLocal['y']

        if debugProblem:
            if not self.isClamped['left']:
                self.refSolutionLocal[:, 0] = cp.gid() - 1
            if not self.isClamped['right']:
                self.refSolutionLocal[:, -1] = cp.gid() + 1
            if not self.isClamped['top']:
                self.refSolutionLocal[-1, :] = cp.gid() + nSubDomainsX
            if not self.isClamped['bottom']:
                self.refSolutionLocal[0, :] = cp.gid() - nSubDomainsX

    def augment_spans(self, cp):

        if verbose:
            if dimension == 1:
                print("Subdomain -- ", cp.gid()+1, ": before Shapes: ",
                      self.refSolutionLocal.shape, self.weightsData.shape, self.knotsAdaptive['x'])
            else:
                print("Subdomain -- ", cp.gid()+1, ": before Shapes: ", self.refSolutionLocal.shape,
                      self.weightsData.shape, self.knotsAdaptive['x'], self.knotsAdaptive['y'])

        if not self.isClamped['left']:  # Pad knot spans from the left of subdomain
            print("\tSubdomain -- ", cp.gid()+1,
                  ": left ghost: ", self.ghostKnots['left'])
            self.knotsAdaptive['x'] = np.concatenate(
                (self.ghostKnots['left'][-1:0:-1], self.knotsAdaptive['x']))

        if not self.isClamped['right']:  # Pad knot spans from the right of subdomain
            print("\tSubdomain -- ", cp.gid()+1,
                  ": right ghost: ", self.ghostKnots['right'])
            self.knotsAdaptive['x'] = np.concatenate(
                (self.knotsAdaptive['x'], self.ghostKnots['right'][1:]))

        if dimension > 1:
            # Pad knot spans from the left of subdomain
            if not self.isClamped['top']:
                print("\tSubdomain -- ", cp.gid()+1,
                      ": top ghost: ", self.ghostKnots['top'])
                self.knotsAdaptive['y'] = np.concatenate(
                    (self.knotsAdaptive['y'], self.ghostKnots['top'][1:]))

            # Pad knot spans from the right of subdomain
            if not self.isClamped['bottom']:
                print("\tSubdomain -- ", cp.gid()+1,
                      ": bottom ghost: ", self.ghostKnots['bottom'])
                self.knotsAdaptive['y'] = np.concatenate(
                    (self.ghostKnots['bottom'][-1:0:-1], self.knotsAdaptive['y']))

        if verbose:
            if dimension == 1:
                print("Subdomain -- ", cp.gid()+1, ": after Shapes: ",
                      self.refSolutionLocal.shape, self.weightsData.shape, self.knotsAdaptive['x'])
            else:
                print("Subdomain -- ", cp.gid()+1, ": after Shapes: ", self.refSolutionLocal.shape,
                      self.weightsData.shape, self.knotsAdaptive['x'], self.knotsAdaptive['y'])

    def augment_inputdata(self, cp):

        verbose = True
        postol = 1e-10

        if verbose:
            if dimension == 1:
                print(
                    "Subdomain -- {0}: before augment -- bounds = {1}, {2}, shape = {3}, knots = {4}, {5}".format(
                        cp.gid() + 1, self.xyzCoordLocal['x'][0],
                        self.xyzCoordLocal['x'][-1],
                        self.xyzCoordLocal['x'].shape, self.knotsAdaptive['x'][degree],
                        self.knotsAdaptive['x'][-degree]))
            else:
                print(
                    "Subdomain -- {0}: before augment -- bounds = {1}, {2}, {3}, {4}, shape = {5}, {6}, knots = {7}, {8}, {9}, {10}".
                    format(
                        cp.gid() + 1, self.xyzCoordLocal['x'][0],
                        self.xyzCoordLocal['x'][-1],
                        self.xyzCoordLocal['y'][0],
                        self.xyzCoordLocal['y'][-1],
                        self.xyzCoordLocal['x'].shape, self.xyzCoordLocal['y'].shape, self.knotsAdaptive['x']
                        [degree],
                        self.knotsAdaptive['x'][-degree],
                        self.knotsAdaptive['y'][degree],
                        self.knotsAdaptive['y'][-degree]))

        # oddDegree = (degree % 2)
        # nconstraints = 2*augmentSpanSpace + (int(degree/2.0) if not oddDegree else int((degree+1)/2.0))
        # nconstraints = degree
        # print('nconstraints = ', nconstraints)

        # print('Knots: ', self.knotsAdaptive['x'], self.knotsAdaptive['y'])
        # locX = sp.BSplineBasis(order=degree+1, knots=self.knotsAdaptive['x']).greville()
        # xCP = [locX[0], locX[-1]]
        xCP = [self.knotsAdaptive['x'][degree],
               self.knotsAdaptive['x'][-degree-1]]
        indicesX = np.where(np.logical_and(
            xcoord >= xCP[0]-postol, xcoord <= xCP[1]+postol))
        # print(indicesX)
        lboundX = indicesX[0][0]
        # indicesX[0][-1] < len(xcoord)
        uboundX = min(indicesX[0][-1]+1, len(xcoord))
        if self.isClamped['left'] and self.isClamped['right']:
            self.xyzCoordLocal['x'] = xcoord[:]
        elif self.isClamped['left']:
            self.xyzCoordLocal['x'] = xcoord[:uboundX]
        elif self.isClamped['right']:
            self.xyzCoordLocal['x'] = xcoord[lboundX:]
        else:
            self.xyzCoordLocal['x'] = xcoord[lboundX:uboundX]
        print("X bounds: ", self.xyzCoordLocal['x']
              [0], self.xyzCoordLocal['x'][-1], xCP)

        if dimension > 1:
            # locY = sp.BSplineBasis(order=degree+1, knots=self.knotsAdaptive['y']).greville()
            # yCP = [locY[0], locY[-1]]
            yCP = [self.knotsAdaptive['y'][degree],
                   self.knotsAdaptive['y'][-degree-1]]
            indicesY = np.where(np.logical_and(
                ycoord >= yCP[0]-postol, ycoord <= yCP[1]+postol))
            # print(indicesY)
            lboundY = indicesY[0][0]
            uboundY = min(indicesY[0][-1]+1, len(ycoord))
            if self.isClamped['top'] and self.isClamped['bottom']:
                self.xyzCoordLocal['y'] = ycoord[:]
            elif self.isClamped['bottom']:
                self.xyzCoordLocal['y'] = ycoord[:uboundY]
            elif self.isClamped['top']:
                self.xyzCoordLocal['y'] = ycoord[lboundY:]
            else:
                self.xyzCoordLocal['y'] = ycoord[lboundY:uboundY]

            print("Y bounds: ",
                  self.xyzCoordLocal['y'][0], self.xyzCoordLocal['y'][-1], yCP)

        # int(nPoints / nSubDomains) + overlapData
        if dimension == 1:
            self.refSolutionLocal = solution[lboundX:uboundX]
        else:
            self.refSolutionLocal = solution[lboundX:uboundX, lboundY:uboundY]
            # int(nPoints / nSubDomains) + overlapData

        # Store the core indices before augment
        cindicesX = np.array(
            np.where(
                np.logical_and(
                    self.xyzCoordLocal['x'] >= xcoord[self.xbounds.min[0]] - postol, self.xyzCoordLocal['x'] <=
                    xcoord[self.xbounds.max[0]] + postol)))
        if dimension > 1:
            cindicesY = np.array(
                np.where(
                    np.logical_and(
                        self.xyzCoordLocal['y'] >= ycoord[self.xbounds.min[1]] - postol, self.xyzCoordLocal['y'] <=
                        ycoord[self.xbounds.max[1]] + postol)))
            self.corebounds = [
                [cindicesX[0][0], len(self.xyzCoordLocal['x']) if self.isClamped['right'] else cindicesX[0][-1]+1],
                [cindicesY[0][0], len(self.xyzCoordLocal['y']) if self.isClamped['top'] else cindicesY[0][-1]+1]]

        else:
            self.corebounds = [[cindicesX[0][0], len(
                self.xyzCoordLocal['x']) if self.isClamped['right'] else cindicesX[0][-1]+1]]

        print("self.corebounds = ", self.xbounds, self.corebounds)

        # self.UVW['x'] = self.xyzCoordLocal['x'][self.corebounds[0][0]:self.corebounds[0][1]] / (self.xyzCoordLocal['x'][self.corebounds[0][1]] - self.xyzCoordLocal['x'][self.corebounds[0][0]])
        # if dimension > 1:
        #     self.UVW['y'] = self.xyzCoordLocal['y'][self.corebounds[1][0]:self.corebounds[1][1]] / (self.xyzCoordLocal['y'][self.corebounds[1][1]] - self.xyzCoordLocal['y'][self.corebounds[1][0]])

        # / (self.xyzCoordLocal['x'][-1] - self.xyzCoordLocal['x'][0])
        self.UVW['x'] = self.xyzCoordLocal['x']
        if dimension > 1:
            # / (self.xyzCoordLocal['y'][-1] - self.xyzCoordLocal['y'][0])
            self.UVW['y'] = self.xyzCoordLocal['y']

        if verbose:
            if dimension == 1:
                print(
                    "Subdomain -- {0}: after augment -- bounds = {1}, {2}, shape = {3}, knots = {4}, {5}".format(
                        cp.gid() + 1, self.xyzCoordLocal['x'][0],
                        self.xyzCoordLocal['x'][-1],
                        self.xyzCoordLocal['x'].shape, self.knotsAdaptive['x'][degree],
                        self.knotsAdaptive['x'][-degree]))
            else:
                print(
                    "Subdomain -- {0}: after augment -- bounds = {1}, {2}, {3}, {4}, shape = {5}, {6}, knots = {7}, {8}, {9}, {10}".
                    format(
                        cp.gid() + 1, self.xyzCoordLocal['x'][0],
                        self.xyzCoordLocal['x'][-1],
                        self.xyzCoordLocal['y'][0],
                        self.xyzCoordLocal['y'][-1],
                        self.xyzCoordLocal['x'].shape, self.xyzCoordLocal['y'].shape, self.knotsAdaptive['x']
                        [degree],
                        self.knotsAdaptive['x'][-degree],
                        self.knotsAdaptive['y'][degree],
                        self.knotsAdaptive['y'][-degree]))

        # self.nControlPoints += augmentSpanSpace
        self.nControlPoints[0] += (augmentSpanSpace if not self.isClamped['left']
                                   else 0) + (augmentSpanSpace if not self.isClamped['right'] else 0)
        if dimension > 1:
            self.nControlPoints[1] += (augmentSpanSpace if not self.isClamped['top']
                                       else 0) + (augmentSpanSpace if not self.isClamped['bottom'] else 0)
        self.nControlPointSpans = self.nControlPoints - 1
        self.nInternalKnotSpans = self.nControlPoints - degree
        self.controlPointData = np.zeros(self.nControlPoints)
        self.weightsData = np.ones(self.nControlPoints)
        self.solutionDecoded = np.zeros(self.refSolutionLocal.shape)
        self.solutionDecodedOld = np.zeros(self.refSolutionLocal.shape)

    def LSQFit_NonlinearOptimize(self, idom, degree, constraints=None):

        solution = []

        # Initialize relevant data
        if constraints is not None:
            initSol = np.copy(constraints)
            # initSol = np.ones_like(self.weightsData)

        else:
            print('Constraints are all null. Solving unconstrained.')
            initSol = np.ones_like(self.weightsData)

        # Compute hte linear operators needed to compute decoded residuals
        # self.Nu = basis(self.UVW.x[np.newaxis,:],degree,Tunew[:,np.newaxis]).T
        # self.Nv = basis(self.UVW.y[np.newaxis,:],degree,Tvnew[:,np.newaxis]).T
        # self.decodeOpXYZ = compute_decode_operators(self.NUVW)

        initialDecodedError = 0.0

        def residual_operator_1D(Pin, verbose=False, vverbose=False):  # checkpoint3

            RN = (self.NUVW['x'] * self.weightsData) / \
                (np.sum(self.NUVW['x']*self.weightsData,
                 axis=1)[:, np.newaxis])
            Aoper = np.matmul(RN.T, RN)
            Brhs = RN.T @ self.refSolutionLocal
            # print('Input P = ', Pin, Aoper.shape, Brhs.shape)

            oddDegree = (degree % 2)
            oddDegreeImpose = True

            # num_constraints = (degree)/2 if degree is even
            # num_constraints = (degree+1)/2 if degree is odd
            nconstraints = augmentSpanSpace + \
                (int(degree/2.0) if not oddDegree else int((degree+1)/2.0))
            if oddDegree and not oddDegreeImpose:
                nconstraints -= 1
            # nconstraints = degree-1
            # print('nconstraints: ', nconstraints)

            residual_constrained_nrm = 0
            nBndOverlap = 0
            if constraints is not None and len(constraints) > 0:
                if idom > 1:  # left constraint

                    loffset = -2*augmentSpanSpace if oddDegree else -2*augmentSpanSpace

                    if oddDegree and not oddDegreeImpose:
                        # print('left: ', nconstraints, -degree+nconstraints+loffset,
                        #       Pin[nconstraints], self.boundaryConstraints['left'][-degree+nconstraints+loffset])
                        constraintVal = 0.5 * \
                            (Pin[nconstraints] - self.boundaryConstraints['left']
                             [-degree+nconstraints+loffset])
                        Brhs -= constraintVal * Aoper[:, nconstraints]

                    for ic in range(nconstraints):
                        Brhs[ic] = 0.5 * \
                            (Pin[ic] + self.boundaryConstraints['left']
                             [-degree+ic+loffset])
                        # Brhs[ic] = constraints[0][-degree+ic]
                        Aoper[ic, :] = 0.0
                        Aoper[ic, ic] = 1.0

                if idom < nSubDomains[0]:  # right constraint

                    loffset = 2*augmentSpanSpace if oddDegree else 2*augmentSpanSpace

                    if oddDegree and not oddDegreeImpose:
                        # print('right: ', -nconstraints-1, degree-1-nconstraints+loffset,
                        #       Pin[-nconstraints-1], self.boundaryConstraints['right']
                        #       [degree-1-nconstraints+loffset])
                        constraintVal = -0.5 * (Pin[-nconstraints-1] - self.boundaryConstraints['right']
                                                [degree-1-nconstraints+loffset])
                        Brhs -= constraintVal * Aoper[:, -nconstraints-1]

                    for ic in range(nconstraints):
                        Brhs[-ic-1] = 0.5 * \
                            (Pin[-ic-1] + self.boundaryConstraints['right']
                             [degree-1-ic+loffset])
                        # Brhs[-ic-1] = constraints[2][degree-1-ic]
                        Aoper[-ic-1, :] = 0.0
                        Aoper[-ic-1, -ic-1] = 1.0

            # print(Aoper, Brhs)
            return [Aoper, Brhs]

        def residual1D(Pin, verbose=False):

            # Use previous iterate as initial solution
            # initSol = constraints[1][:] if constraints is not None else np.ones_like(
            #     W)
            # initSol = np.ones_like(W)*0

            if False:
                [Aoper, Brhs] = residual_operator_1D(
                    self.controlPointData, False, False)
                # if type(Pin) is np.numpy_boxes.ArrayBox:
                #     [Aoper, Brhs] = residual_operator_1D(Pin._value[:], False, False)
                # else:
                #     [Aoper, Brhs] = residual_operator_1D(Pin, False, False)

                lu, piv = scipy.linalg.lu_factor(Aoper)
                # print(lu, piv)
                initSol = scipy.linalg.lu_solve((lu, piv), Brhs)

                # residual_nrm_vec = Brhs - Aoper @ Pin
                # residual_nrm = np.linalg.norm(residual_nrm_vec, ord=2)
                residual_nrm = np.linalg.norm(initSol-Pin, ord=2)
            else:

                # New scheme like 2-D
                decoded = decode(Pin, self.decodeOpXYZ)
                residual_decoded = (
                    self.refSolutionLocal - decoded)/solutionRange
                decoded_residual_norm = np.sqrt(
                    np.sum(residual_decoded**2)/len(residual_decoded))

                residual_nrm = (decoded_residual_norm-initialDecodedError)

            if type(Pin) is not np.numpy_boxes.ArrayBox:
                print('Residual 1D: ', decoded_residual_norm, residual_nrm)

            return residual_nrm * 0

        # Compute the residual as sum of two components
        # 1. The decoded error evaluated at P
        # 2. A penalized domain boundary constraint component

        def residual2D(Pin, verbose=False):

            P = np.array(Pin.reshape(self.controlPointData.shape), copy=True)

            decoded_residual_norm = 0

            # Residuals are in the decoded space - so direct way to constrain the boundary data
            decoded = decode(P, self.decodeOpXYZ)
            residual_decoded = (self.refSolutionLocal - decoded)/solutionRange

            if augmentSpanSpace > 0:
                residual_decoded = residual_decoded[self.corebounds[0][0]
                    : self.corebounds[0][1], self.corebounds[1][0]: self.corebounds[1][1]]

            residual_vec_decoded = residual_decoded.reshape(-1)
            decoded_residual_norm = np.sqrt(np.sum(residual_vec_decoded**2)/len(residual_vec_decoded))

            # if type(residual_decoded) is np.numpy_boxes.ArrayBox:
            #     decoded_residual_norm = np.linalg.norm(residual_decoded._value, ord=2)
            # else:
            #     decoded_residual_norm = np.linalg.norm(residual_decoded, ord=2)

            # return decoded_residual_norm

#             res1 = np.dot(res_dec, self.Nv)
#             residual = np.dot(self.Nu.T, res1).reshape(self.Nu.shape[1]*self.Nv.shape[1])

            # compute the net residual norm that includes the decoded error in the current subdomain and the penalized
            # constraint error of solution on the subdomain interfaces
            net_residual_norm = (decoded_residual_norm-initialDecodedError)

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

            if verbose and True:
                # print('Residual = ', net_residual_norm, ' and decoded = ', decoded_residual_norm, ', constraint = ',
                #       constrained_residual_norm, ', diagonal = ', diagonal_boundary_residual_norm if useDiagonalBlocks else 0)
                # print('Constraint errors = ', ltn, rtn,
                #       tpn, btn, constrained_residual_norm)
                print('Residual = ', decoded_residual_norm)
                # if useDiagonalBlocks:
                #     print('Constraint diagonal errors = ', topleftBndErr, toprightBndErr, bottomleftBndErr,
                #           bottomrightBndErr, diagonal_boundary_residual_norm)

            return net_residual_norm

        # Set a function handle to the appropriate residual evaluator
        residualFunction = None
        if dimension == 1:
            residualFunction = residual1D
        else:
            residualFunction = residual2D

        # Create a gradient function to pass to the minimizer
        jacobianFunction = egrad(residualFunction)
        # jacobianFunction =  jit(grad(residualFunction))

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

            jacobian = jacobianFunction(P)
#             jacobian = jacobian_const
            return jacobian

        # if constraintsAll is not None:
        #    jacobian_const = egrad(residual)(initSol)

        # Now invoke the solver and compute the constrained solution
        if constraints is None and not alwaysSolveConstrained:
            solution = lsqFit(
                self.decodeOpXYZ['x'], self.decodeOpXYZ['y'], self.refSolutionLocal)
            # solution = solution.reshape(W.shape)
        else:

            if constraints is None and alwaysSolveConstrained:
                initSol = lsqFit(
                    self.decodeOpXYZ['x'], self.decodeOpXYZ['y'], self.refSolutionLocal)

            localAssemblyWeights = np.ones(initSol.shape)
            localBCAssembly = np.zeros(initSol.shape)
            print('Initial calculation')
            # Lets update our initial solution with constraints
            if constraints is not None and len(constraints) > 0 and True:
                oddDegree = (degree % 2)
                # alpha = 0.5 if dimension == 2 or oddDegree else 0.0
                alpha = 0.5
                nconstraints = augmentSpanSpace + \
                    (int(degree/2.0)+1 if not oddDegree else int((degree+1)/2.0))
                loffset = 2*augmentSpanSpace
                print('Nconstraints = ', nconstraints, 'loffset = ', loffset)

                if dimension == 2 and nconstraints > 1:
                    if 'left' in self.boundaryConstraints:
                        initSol[:nconstraints-1, :] = 0
                        localAssemblyWeights[:nconstraints-1, :] = 1.0
                    if 'right' in self.boundaryConstraints:
                        initSol[-nconstraints + 1:, :] = 0
                        localAssemblyWeights[-nconstraints+1:, :] = 1.0
                    if 'top' in self.boundaryConstraints:
                        initSol[:, -nconstraints+1:] = 0
                        localAssemblyWeights[:, -nconstraints+1:] = 1.0
                    if 'bottom' in self.boundaryConstraints:
                        initSol[:, : nconstraints - 1] = 0
                        localAssemblyWeights[:, : nconstraints - 1] = 1.0
                    if 'top-left' in self.boundaryConstraints:
                        initSol[: nconstraints - 1, -nconstraints + 1:] = 0
                        localAssemblyWeights[: nconstraints - 1, -nconstraints + 1:] = 1.0
                    if 'bottom-left' in self.boundaryConstraints:
                        initSol[: nconstraints - 1, : nconstraints - 1] = 0
                        localAssemblyWeights[: nconstraints - 1, : nconstraints - 1] = 1.0
                    if 'top-right' in self.boundaryConstraints:
                        initSol[-nconstraints + 1:, -nconstraints + 1:] = 0
                        localAssemblyWeights[-nconstraints + 1:, -nconstraints + 1:] = 1.0
                    if 'bottom-right' in self.boundaryConstraints:
                        initSol[-nconstraints + 1:, : nconstraints - 1] = 0
                        localAssemblyWeights[-nconstraints + 1:, : nconstraints - 1] = 1.0

                # First update hte control point vector with constraints for supporting points
                if 'left' in self.boundaryConstraints:
                    if dimension == 1:
                        if nconstraints > 1:
                            initSol[: nconstraints -
                                    1] = self.boundaryConstraints['left'][-degree-loffset: -nconstraints]
                        initSol[nconstraints-1] = alpha * initSol[nconstraints-1] + (
                            1-alpha) * self.boundaryConstraints['left'][-nconstraints]

                    else:
                        localBCAssembly[nconstraints-1, :] += self.boundaryConstraints['left'][-nconstraints, :]
                        localAssemblyWeights[nconstraints-1, :] += 1.0

                        if nconstraints > 1:
                            localBCAssembly[:nconstraints-1, :] = self.boundaryConstraints['left'][-degree -
                                                                                                   loffset:-nconstraints, :]

                if 'right' in self.boundaryConstraints:
                    if dimension == 1:
                        if nconstraints > 1:
                            initSol[-nconstraints +
                                    1:] = self.boundaryConstraints['right'][nconstraints: degree+loffset]
                        initSol[-nconstraints] = alpha * initSol[-nconstraints] + (
                            1-alpha) * self.boundaryConstraints['right'][nconstraints-1]

                    else:
                        localBCAssembly[-nconstraints, :] += self.boundaryConstraints['right'][nconstraints-1, :]
                        localAssemblyWeights[-nconstraints, :] += 1.0

                        if nconstraints > 1:
                            localBCAssembly[-nconstraints + 1:,
                                            :] = self.boundaryConstraints['right'][nconstraints: degree + loffset, :]

                if 'top' in self.boundaryConstraints:
                    localBCAssembly[:, -nconstraints] += self.boundaryConstraints['top'][:, nconstraints-1]
                    localAssemblyWeights[:, -nconstraints] += 1.0

                    if nconstraints > 1:
                        localBCAssembly[:, -nconstraints+1:] = self.boundaryConstraints['top'][:,
                                                                                               nconstraints:loffset+degree]

                if 'bottom' in self.boundaryConstraints:
                    localBCAssembly[:, nconstraints-1] += self.boundaryConstraints['bottom'][:, -nconstraints]
                    localAssemblyWeights[:, nconstraints-1] += 1.0

                    if nconstraints > 1:
                        localBCAssembly[:, : nconstraints - 1] = self.boundaryConstraints['bottom'][:, -degree -
                                                                                                    loffset: -nconstraints]

                if 'top-left' in self.boundaryConstraints:
                    localBCAssembly[nconstraints-1, -nconstraints] += self.boundaryConstraints['top-left'][-nconstraints, nconstraints-1]
                    localAssemblyWeights[nconstraints-1, -nconstraints] += 1.0

                    if nconstraints > 1:
                        localBCAssembly[: nconstraints - 1, -nconstraints + 1:] = self.boundaryConstraints['top-left'][-degree -
                                                                                                                       loffset: -nconstraints, nconstraints: degree + loffset]

                if 'bottom-left' in self.boundaryConstraints:
                    localBCAssembly[nconstraints-1,
                                    nconstraints-1] += self.boundaryConstraints['bottom-left'][-nconstraints, -nconstraints]
                    localAssemblyWeights[nconstraints-1, nconstraints-1] += 1.0

                    if nconstraints > 1:
                        localBCAssembly[: nconstraints - 1, : nconstraints - 1] = self.boundaryConstraints['bottom-left'][-degree -
                                                                                                                          loffset: -nconstraints, -degree - loffset: -nconstraints]

                if 'top-right' in self.boundaryConstraints:
                    localBCAssembly[-nconstraints, -nconstraints] += self.boundaryConstraints['top-right'][nconstraints-1, nconstraints-1]
                    localAssemblyWeights[-nconstraints, -nconstraints] += 1.0

                    if nconstraints > 1:
                        localBCAssembly[-nconstraints + 1:, -nconstraints + 1:] = self.boundaryConstraints['top-right'][
                            nconstraints: degree + loffset, nconstraints: degree + loffset]

                if 'bottom-right' in self.boundaryConstraints:
                    localBCAssembly[-nconstraints,
                                    nconstraints-1] += self.boundaryConstraints['bottom-right'][nconstraints-1, -nconstraints]
                    localAssemblyWeights[-nconstraints, nconstraints-1] += 1.0

                    if nconstraints > 1:
                        localBCAssembly[-nconstraints + 1:, : nconstraints - 1] = self.boundaryConstraints['bottom-right'][nconstraints:
                                                                                                                           degree + loffset, -degree - loffset: -nconstraints]


                if dimension > 1:
                    initSol = np.divide(initSol + localBCAssembly, localAssemblyWeights)
                    # initSol = (initSol + np.divide(localBCAssembly, localAssemblyWeights))

            if idom in [5, 4, 2, 1]:
                print('debug me here')

            if 'left' in self.boundaryConstraints:
                print(
                    'Error left: ', np.abs(
                        self.boundaryConstraints['left'][-degree - loffset: -nconstraints, :] -
                        initSol[: nconstraints - 1, :]).T)
            if 'right' in self.boundaryConstraints:
                print(
                    'Error right: ', np.abs(
                        self.boundaryConstraints['right'][nconstraints: degree + loffset, :] -
                        initSol[-nconstraints + 1:, :]).T)
            if 'top' in self.boundaryConstraints:
                print(
                    'Error top: ', np.abs(
                        self.boundaryConstraints['top'][:, nconstraints:loffset+degree] -
                        initSol[:, -nconstraints+1:]))
            if 'bottom' in self.boundaryConstraints:
                print(
                    'Error bottom: ', np.abs(
                        self.boundaryConstraints['bottom'][:, -degree - loffset: -nconstraints] -
                        initSol[:, : nconstraints - 1]))

            initialDecodedError = residualFunction(initSol, verbose=True)

            if enforceBounds:
                if dimension == 1:
                    nshape = self.controlPointData.shape[0]
                elif dimension == 2:
                    nshape = self.controlPointData.shape[0] * \
                        self.controlPointData.shape[1]
                else:
                    nshape = self.controlPointData.shape[0] * \
                        self.controlPointData.shape[1] * \
                        self.controlPointData.shape[2]
                bnds = np.tensordot(np.ones(nshape),
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
            solution = np.copy(res.x).reshape(self.controlPointData.shape)

            # solution = np.copy(initSol)

        return solution

    def print_error_metrics(self, cp):
        # print('Size: ', commW.size, ' rank = ', commW.rank, ' Metrics: ', self.errorMetricsL2[:])
        # print('Rank:', commW.rank, ' SDom:', cp.gid(), ' L2 Error table: ', self.errorMetricsL2)
        print(
            'Rank:', commWorld.rank, ' SDom:', cp.gid(),
            ' Error: ', self.errorMetricsL2[self.outerIteration - 1],
            ', Convergence: ',
            np.abs([self.errorMetricsL2[1: self.outerIteration] - self.errorMetricsL2[0: self.outerIteration - 1]]))

        # L2NormVector = MPI.gather(self.errorMetricsL2[self.outerIteration - 1], root=0)

    def check_convergence(self, cp, iterationNum):

        global isConverged, L2err
        if len(self.solutionDecodedOld):

            # Let us compute the relative change in the solution between current and previous iteration
            iterateChangeVec = (self.solutionDecoded -
                                self.solutionDecodedOld).flatten()
            errorMetricsSubDomL2 = np.linalg.norm(
                iterateChangeVec, ord=2) / np.linalg.norm(self.refSolutionLocal, ord=2)
            errorMetricsSubDomLinf = np.linalg.norm(
                iterateChangeVec, ord=np.inf) / np.linalg.norm(self.refSolutionLocal, ord=np.inf)

            self.errorMetricsL2[self.outerIteration] = self.decodederrors[0]
            self.errorMetricsLinf[self.outerIteration] = self.decodederrors[1]

            print(cp.gid()+1, ' Convergence check: ', errorMetricsSubDomL2, errorMetricsSubDomLinf,
                  np.abs(self.errorMetricsLinf[self.outerIteration] -
                         self.errorMetricsLinf[self.outerIteration-1]),
                  errorMetricsSubDomLinf < 1e-8 and np.abs(self.errorMetricsL2[self.outerIteration]-self.errorMetricsL2[self.outerIteration-1]) < 1e-10)

            L2err[cp.gid()] = self.decodederrors[0]
            if errorMetricsSubDomLinf < 1e-12 and np.abs(
                    self.errorMetricsL2[self.outerIteration] - self.errorMetricsL2[self.outerIteration - 1]) < 1e-12:
                print('Subdomain ', cp.gid(
                )+1, ' has converged to its final solution with error = ', errorMetricsSubDomLinf)
                isConverged[cp.gid()] = 1

        # self.outerIteration = iterationNum+1
        self.outerIteration += 1

        # isASMConverged = commWorld.allreduce(self.outerIterationConverged, op=MPI.LAND)

    def subdomain_solve(self, cp):

        global isConverged
        if isConverged[cp.gid()] == 1:
            print(
                cp.gid()+1, ' subdomain has already converged to its solution. Skipping solve ...')
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
        self.decodeOpXYZ = compute_decode_operators(self.NUVW)

        if ((np.sum(np.abs(self.controlPointData)) < 1e-14 and len(self.controlPointData) > 0) or len(self.controlPointData) == 0) and self.outerIteration == 0:
            print(iSubDom, " - Applying the unconstrained solver.")
            constraints = None

        else:
            print(iSubDom, " - Applying the constrained solver.")
            constraints = np.copy(self.controlPointData)

        #  Invoke the local subdomain solver
        self.controlPointData = self.LSQFit_NonlinearOptimize(
            iSubDom, degree, constraints)

        if constraints is None:  # We just solved the initial LSQ problem.
            # Store the maximum bounds to respect so that we remain monotone
            self.controlPointBounds = np.array(
                [np.min(self.controlPointData), np.max(self.controlPointData)])

        # Update the local decoded data
        self.solutionDecoded = decode(self.controlPointData, self.decodeOpXYZ)
        # decodedError = np.abs(np.array(self.refSolutionLocal - self.solutionDecoded)) / solutionRange

        if len(self.solutionLocalHistory) == 0 and extrapolate:
            if useAitken:
                self.solutionLocalHistory = np.zeros(
                    (self.controlPointData.shape[0]*self.controlPointData.shape[1], 3))
            else:
                self.solutionLocalHistory = np.zeros(
                    (self.controlPointData.shape[0]*self.controlPointData.shape[1], nWynnEWork))

        # E = (self.solutionDecoded[self.corebounds[0]:self.corebounds[1]] - self.controlPointData[self.corebounds[0]:self.corebounds[1]])/solutionRange
        if dimension == 1:
            decodedError = (
                self.refSolutionLocal[self.corebounds[0][0]: self.corebounds[0][1]] - self.solutionDecoded
                [self.corebounds[0][0]: self.corebounds[0][1]]) / solutionRange
        elif dimension == 2:
            decodedError = (
                self.refSolutionLocal[self.corebounds[0][0]: self.corebounds[0][1],
                                      self.corebounds[1][0]: self.corebounds[1][1]] - self.solutionDecoded
                [self.corebounds[0][0]: self.corebounds[0][1],
                 self.corebounds[1][0]: self.corebounds[1][1]]) / solutionRange
            decodedError = (decodedError.reshape(
                decodedError.shape[0]*decodedError.shape[1]))
        LinfErr = np.linalg.norm(decodedError, ord=np.inf)
        L2Err = np.sqrt(np.sum(decodedError**2)/len(decodedError))

        self.decodederrors[0] = L2Err
        self.decodederrors[1] = LinfErr

        print("Subdomain -- ", iSubDom, ": L2 error: ",
              L2Err, ", Linf error: ", LinfErr)


#########
domain_control = diy.DiscreteBounds(
    np.zeros((dimension, 1), dtype=np.uint32), nPoints-1)

# Routine to recursively add a block and associated data to it

print('')


def add_input_control_block2(gid, core, bounds, domain, link):
    print("Subdomain %d: " % gid, core, bounds, domain)
    minb = bounds.min
    maxb = bounds.max

    # if dimension == 1:
    #     globalExtentDict[gid] = [minb[0], maxb[0]]
    # elif dimension == 2:
    #     globalExtentDict[gid] = [minb[0], maxb[0], minb[1], maxb[1]]
    # else:
    #     globalExtentDict[gid] = [minb[0], maxb[0],
    #                              minb[1], maxb[1], minb[2], maxb[2]]

    xlocal = xcoord[minb[0]:maxb[0]+1]
    if dimension > 1:
        ylocal = ycoord[minb[1]:maxb[1]+1]
        if dimension > 2:
            zlocal = zcoord[minb[2]:maxb[2]+1]
            sollocal = solution[minb[0]:maxb[0]+1,
                                minb[1]:maxb[1]+1, minb[2]:maxb[2]+1]
        else:
            zlocal = None
            sollocal = solution[minb[0]:maxb[0]+1, minb[1]:maxb[1]+1]

    else:
        ylocal = None
        zlocal = None
        sollocal = solution[minb[0]:maxb[0]+1]
        sollocal = sollocal.reshape((len(sollocal), 1))

    # print("Subdomain %d: " % gid, minb[0], minb[1], maxb[0], maxb[1], z.shape, zlocal.shape)
    masterControl.add(gid, InputControlBlock(
        gid, nControlPointsInput, core, bounds, sollocal, xlocal, ylocal, zlocal), link)


# TODO: If working in parallel with MPI or DIY, do a global reduce here
# Store L2, Linf errors as function of iteration
errors = np.zeros([nASMIterations+1, 2])

# Let us initialize DIY and setup the problem
share_face = np.ones((dimension, 1)) > 0
wrap = np.ones((dimension, 1)) < 0
ghosts = np.zeros((dimension, 1), dtype=np.uint32)

discreteDec = diy.DiscreteDecomposer(
    dimension, domain_control, nTotalSubDomains, share_face, wrap, ghosts, nSubDomains)

contigAssigner = diy.ContiguousAssigner(nprocs, nTotalSubDomains)

discreteDec.decompose(rank, contigAssigner, add_input_control_block2)

masterControl.foreach(InputControlBlock.show)

sys.stdout.flush()
commWorld.Barrier()

if rank == 0:
    print("\n---- Starting Global Iterative Loop ----")

#########
start_time = timeit.default_timer()


def send_receive_all():

    masterControl.foreach(InputControlBlock.send_diy)
    masterControl.exchange(False)
    masterControl.foreach(InputControlBlock.recv_diy)

    return


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

del xcoord, ycoord, solution

for iterIdx in range(nASMIterations):

    if rank == 0:
        print("\n---- Starting Iteration: %d ----" % iterIdx)

    if iterIdx > 0 and rank == 0:
        print("")

    # run our local subdomain solver
    masterControl.foreach(InputControlBlock.subdomain_solve)

    # check if we have locally converged within criteria
    masterControl.foreach(
        lambda icb, cp: InputControlBlock.check_convergence(icb, cp, iterIdx))

    isASMConverged = commWorld.allreduce(np.sum(isConverged), op=MPI.SUM)

    # commW.Barrier()
    sys.stdout.flush()

    if useVTKOutput or showplot:

        if dimension == 1:

            figHnd = plt.figure()
            figErrHnd = None
            # figErrHnd = plt.figure()
            # plt.plot(xcoord, solution, 'b-', ms=5, label='Input')

            masterControl.foreach(lambda icb, cp: InputControlBlock.set_fig_handles(
                icb, cp, figHnd, figErrHnd, "%d-%d" % (cp.gid(), iterIdx)))
            masterControl.foreach(InputControlBlock.output_solution)

            # plt.legend()
            plt.draw()
            figHnd.show()

        else:

            masterControl.foreach(lambda icb, cp: InputControlBlock.set_fig_handles(
                icb, cp, None, None, "%d-%d" % (cp.gid(), iterIdx)))
            # masterControl.foreach(InputControlBlock.output_solution)
            masterControl.foreach(InputControlBlock.output_vtk)

            if rank == 0:
                WritePVTKFile(iterIdx)
                WritePVTKControlFile(iterIdx)

    if isASMConverged == nTotalSubDomains:
        if rank == 0:
            print("\n\nASM solver converged after %d iterations\n\n" % (iterIdx+1))
        break

    else:
        if extrapolate:
            masterControl.foreach(
                lambda icb, cp: InputControlBlock.extrapolate_guess(icb, cp, iterIdx))

        # Now let us perform send-receive to get the data on the interface boundaries from
        # adjacent nearest-neighbor subdomains
        send_receive_all()


# masterControl.foreach(InputControlBlock.print_solution)

elapsed = timeit.default_timer() - start_time
sys.stdout.flush()
if rank == 0:
    print('\nTotal computational time for solve = ', elapsed, '\n')

avgL2err = commWorld.allreduce(np.sum(L2err[np.nonzero(L2err)]**2), op=MPI.SUM)
avgL2err = np.sqrt(avgL2err/nTotalSubDomains)
maxL2err = commWorld.allreduce(
    np.max(np.abs(L2err[np.nonzero(L2err)])), op=MPI.MAX)
minL2err = commWorld.allreduce(
    np.min(np.abs(L2err[np.nonzero(L2err)])), op=MPI.MIN)

# np.set_printoptions(formatter={'float': '{: 5.12e}'.format})
# mc.foreach(InputControlBlock.print_error_metrics)
if rank == 0:
    print("\nError metrics: L2 average = %6.12e, L2 maxima = %6.12e, L2 minima = %6.12e\n" % (
        avgL2err, maxL2err, minL2err))
    print('')

commWorld.Barrier()

np.set_printoptions(formatter={'float': '{: 5.12e}'.format})
masterControl.foreach(InputControlBlock.print_error_metrics)

if showplot:
    plt.show()

# ---------------- END MAIN FUNCTION -----------------
