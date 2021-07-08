
# coding: utf-8

# In[2]:


# coding: utf-8
# get_ipython().magic(u'matplotlib notebook')
# %matplotlib notebook
import sys
import getopt
import math
import timeit
# import numpy as np

import splipy as sp

# from numba import jit

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
import mpl_toolkits.mplot3d.axes3d as axes3d
from matplotlib import pyplot as plt
from matplotlib import cm
from scipy.interpolate import Rbf
from pyevtk.hl import gridToVTK

# Profiling imports
# import tracemalloc

plt.style.use(['seaborn-whitegrid'])
# plt.style.use(['ggplot'])
# plt.style.use(['classic'])
params = {"ytick.color": "b",
          "xtick.color": "b",
          "axes.labelcolor": "b",
          "axes.edgecolor": "b"}
plt.rcParams.update(params)

# --- set problem input parameters here ---
nSubDomainsX = 2
nSubDomainsY = 2
degree = 3
problem = 1
verbose = False
showplot = False
useVTKOutput = True

augmentSpanSpace = 0
relEPS = 5e-2
fullyPinned = False

Dmin = Dmax = 0

useDecodedConstraints = False

# ------------------------------------------
# Solver parameters
useAdditiveSchwartz = True
useDerivativeConstraints = 0

#                      0      1       2         3          4       5       6      7       8
solverMethods = ['L-BFGS-B', 'CG', 'SLSQP', 'Newton-CG', 'TNC', 'krylov', 'lm', 'trf', 'anderson', 'hybr']
solverScheme = solverMethods[1]
solverMaxIter = 20
nASMIterations = 10

projectData = False
enforceBounds = False
alwaysSolveConstrained = False
constrainInterfaces = True

overlapData = 0
useDeCastelJau = True
disableAdaptivity = True
variableResolution = False

maxAbsErr = 1e-4
maxRelErr = 1e-10
maxAdaptIter = 3
# AdaptiveStrategy = 'extend'
AdaptiveStrategy = 'reset'
interpOrder = 'cubic'

extrapolate = False
useAitken = False
nWynnEWork = 3

# override
# extrapolate = True
# useAitken = True
# ------------------------------------------

# tracemalloc.start()

# @profile


def plot3D(fig, Z, x=None, y=None):
    if x is None:
        x = np.arange(Z.shape[0])
    if y is None:
        y = np.arange(Z.shape[1])
    X, Y = np.meshgrid(x, y)
    # print("Plot shapes: [x, y, z] = ", x.shape, y.shape, Z.shape, X.shape, Y.shape)

    if showplot:
        ax = fig.gca(projection='3d')
        surf = ax.plot_surface(X, Y, Z.T, cmap=cm.coolwarm,
                               linewidth=0, antialiased=True)
        fig.colorbar(surf)

    if useVTKOutput:
        X = X.reshape(1, X.shape[0], X.shape[1])
        Y = Y.reshape(1, Y.shape[0], Y.shape[1])
        Zi = np.ones(X.shape)
        Z = Z.T.reshape(1, Z.shape[1], Z.shape[0])
        # print(X.shape, Y.shape, Zi.shape, Z.shape)
        gridToVTK("./structured", X, Y, Zi, pointData={"solution": Z})

    # plt.show()


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
    opts, args = getopt.getopt(argv, "hp:n:x:y:d:c:o:p:a:g:s:",
                               ["problem=", "nsubdomains=", "nx=", "ny=", "degree=", "controlpoints=", "overlap=",
                                "problem=", "nasm=", "disableadaptivity", "aug=", "accel", "wynn"])
except getopt.GetoptError:
    usage()

nPointsX = nPointsY = 0
DminX = DminY = DmaxX = DmaxY = 0.0
nControlPointsInput = []
for opt, arg in opts:
    if opt == '-h':
        usage()
    elif opt in ("-n", "--nsubdomains"):
        nSubDomainsX = int(arg)
        nSubDomainsY = int(arg)
    elif opt in ("-x", "--nx"):
        nSubDomainsX = int(arg)
    elif opt in ("-y", "--ny"):
        nSubDomainsY = int(arg)
    elif opt in ("-d", "--degree"):
        degree = int(arg)
    elif opt in ("-c", "--controlpoints"):
        nControlPointsInput = np.array([int(arg), int(arg)])
    elif opt in ("-o", "--overlap"):
        overlapData = int(arg)
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

# -------------------------------------

nSubDomains = nSubDomainsX * nSubDomainsY
isConverged = np.zeros(nSubDomains, dtype='int32')
L2err = np.zeros(nSubDomains)

# def read_problem_parameters():
x = y = z = None
if problem == 1:
    nPointsX = 1025
    nPointsY = 1025
    scale = 1
    shiftX = 0.0
    shiftY = 0.0
    DminX = DminY = -4.
    DmaxX = DmaxY = 4.

    x = np.linspace(DminX, DmaxX, nPointsX)
    y = np.linspace(DminY, DmaxY, nPointsY)
    X, Y = np.meshgrid(x+shiftX, y+shiftY)

    # z = 100 * (1+0.25*np.tanh((X**2+Y**2)/16)) * (np.sinc(np.sqrt((X-2) ** 2 + (Y+2)**2)) +
    #                                               np.sinc(np.sqrt((X+2)**2 + (Y-2)**2)) -
    #                                               2 * (1-np.tanh((X)**2 + (Y)**2)) +
    #                                               np.exp(-((X-2)**2/2)-((Y-2)**2/2))
    #                                               #   + np.sign(X+Y)
    #                                               )

    # noise = np.random.uniform(0, 0.005, X.shape)
    # z = z * (1 + noise)

    # z = scale * (np.sinc(np.sqrt(X**2 + Y**2)) + np.sinc(2*((X-2)**2 + (Y+2)**2)))
    # z = scale * (np.sinc((X+1)**2 + (Y-1)**2) + np.sinc(((X-1)**2 + (Y+1)**2)))
    # z = X**2 * (DmaxX - Y)**2 + X**2 * Y**2 + 64 * (np.sinc(np.sqrt((X-2) ** 2 + (Y+2)**2)))
    # z = X**3 * Y**3
    z = scale * (np.sinc(X) * np.sinc(Y))
    z = z.T
    # (3*degree + 1) #minimum number of control points
    if len(nControlPointsInput) == 0:
        nControlPointsInput = 16*np.array([1, 1])
    del X, Y

elif problem == 2:
    nPointsX = 501
    nPointsY = 501
    scale = 1.0
    shiftX = 0.25*0
    shiftY = -0.25*0
    DminX = DminY = 0
    DmaxX = DmaxY = math.pi

    x = np.linspace(DminX, DmaxX, nPointsX)
    y = np.linspace(DminY, DmaxY, nPointsY)
    X, Y = np.meshgrid(x+shiftX, y+shiftY)
    z = scale * np.sin(X) * np.sin(Y)
    z = z.T
    # z = scale * np.sin(Y)
    # z = scale * X
    # (3*degree + 1) #minimum number of control points
    if len(nControlPointsInput) == 0:
        nControlPointsInput = 4*np.array([1, 1])
    del X, Y

elif problem == 3:
    z = np.fromfile("data/nek5000.raw", dtype=np.float64).reshape(200, 200)
    print("Nek5000 shape:", z.shape)
    nPointsX = z.shape[0]
    nPointsY = z.shape[1]
    DminX = DminY = 0
    DmaxX = DmaxY = 100.

    x = np.linspace(DminX, DmaxX, nPointsX)
    y = np.linspace(DminY, DmaxY, nPointsY)
    # (3*degree + 1) #minimum number of control points
    if len(nControlPointsInput) == 0:
        nControlPointsInput = 20*np.array([1, 1])

elif problem == 4:

    binFactor = 4.0
    z = np.fromfile("data/s3d_2D.raw", dtype=np.float64).reshape(540, 704)
    # z = z[:540,:540]
    # z = zoom(z, 1./binFactor, order=4)
    nPointsX = z.shape[0]
    nPointsY = z.shape[1]
    DminX = DminY = 0
    DmaxX = 1.0*nPointsX
    DmaxY = 1.0*nPointsY
    print("S3D shape:", z.shape)
    x = np.linspace(DminX, DmaxX, nPointsX)
    y = np.linspace(DminY, DmaxY, nPointsY)
    # (3*degree + 1) #minimum number of control points
    if len(nControlPointsInput) == 0:
        nControlPointsInput = 25*np.array([1, 1])

elif problem == 5:
    z = np.fromfile("data/FLDSC_1_1800_3600.dat", dtype=np.float32).reshape(1800, 3600).T
    nPointsX = z.shape[0]
    nPointsY = z.shape[1]
    DminX = DminY = 0
    DmaxX = 1.0*nPointsX
    DmaxY = 1.0*nPointsY
    print("CESM data shape: ", z.shape)
    x = np.linspace(DminX, DmaxX, nPointsX)
    y = np.linspace(DminY, DmaxY, nPointsY)

    # (3*degree + 1) #minimum number of control points
    if len(nControlPointsInput) == 0:
        nControlPointsInput = 25*np.array([1, 1])

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

    x = np.linspace(DminX, DmaxX, nPointsX)
    y = np.linspace(DminY, DmaxY, nPointsY)
    X, Y = np.meshgrid(x+shiftX, y+shiftY)

    N_max = 255
    some_threshold = 50.0

    # from PIL import Image
    # image = Image.new("RGB", (nPointsX, nPointsY))
    mandelbrot_set = np.zeros((nPointsX, nPointsY))
    for yi in range(nPointsY):
        zy = yi * (DmaxY - DminY) / (nPointsY - 1) + DminY
        y[yi] = zy
        for xi in range(nPointsX):
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

    if len(nControlPointsInput) == 0:
        nControlPointsInput = 50*np.array([1, 1])

else:
    print('Not a valid problem')
    exit(1)

# if nPointsX % nSubDomainsX > 0 or nPointsY % nSubDomainsY > 0:
#     print ( "[ERROR]: The total number of points do not divide equally with subdomains" )
#     sys.exit(1)

xmin = x.min()
xmax = x.max()
ymin = y.min()
ymax = y.max()
zmin = z.min()
zmax = z.max()
zRange = zmax-zmin
fig = None
if showplot:
    fig = plt.figure()
plot3D(fig, z, x, y)

### Print parameter details ###
if rank == 0:
    print('\n==================')
    print('Parameter details')
    print('==================\n')
    print('problem = ', problem, '[1 = sinc, 2 = sine, 3 = Nek5000, 4 = S3D, 5 = CESM]')
    print('Total number of input points: ', nPointsX*nPointsY)
    print('nSubDomains = ', nSubDomainsX * nSubDomainsY)
    print('degree = ', degree)
    print('nControlPoints = ', nControlPointsInput)
    print('nASMIterations = ', nASMIterations)
    print('overlapData = ', overlapData)
    print('augmentSpanSpace = ', augmentSpanSpace)
    print('useAdditiveSchwartz = ', useAdditiveSchwartz)
    print('useDerivativeConstraints = ', useDerivativeConstraints)
    print('enforceBounds = ', enforceBounds)
    print('maxAbsErr = ', maxAbsErr)
    print('maxRelErr = ', maxRelErr)
    print('solverMaxIter = ', solverMaxIter)
    print('AdaptiveStrategy = ', AdaptiveStrategy)
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
    pvtkfile.write('<PStructuredGrid WholeExtent="%f %f %f %f 0 0" GhostLevel="0">\n' % (xmin, xmax, ymin, ymax))
    pvtkfile.write('\n')
    pvtkfile.write('    <PCellData>\n')
    pvtkfile.write('      <PDataArray type="Float64" Name="solution"/>\n')
    pvtkfile.write('      <PDataArray type="Float64" Name="error"/>\n')
    pvtkfile.write('    </PCellData>\n')
    pvtkfile.write('    <PPoints>\n')
    pvtkfile.write('      <PDataArray type="Float64" Name="points" NumberOfComponents="3"/>\n')
    pvtkfile.write('    </PPoints>\n')

    isubd = 0
    dx = (xmax-xmin)/nSubDomainsX
    dy = (ymax-ymin)/nSubDomainsY
    xoff = xmin
    xx = dx
    for ix in range(nSubDomainsX):
        yoff = ymin
        yy = dy
        for iy in range(nSubDomainsY):
            pvtkfile.write(
                '    <Piece Extent="%f %f %f %f 0 0" Source="structured-%d-%d.vts"/>\n' %
                (xoff, xx, yoff, yy, isubd, iteration))
            isubd += 1
            yoff = yy
            yy += dy
        xoff = xx
        xx += dx
    pvtkfile.write('\n')
    pvtkfile.write('</PStructuredGrid>\n')
    pvtkfile.write('</VTKFile>\n')

    pvtkfile.close()

# ------------------------------------


EPS = 1e-32
GTOL = 1e-2


# def basis(u, p, T): return ((T[:-1] <= u) * (u <= T[1:])).astype(np.float) if p == 0 else ((u - T[:-p]) / (
# T[p:] - T[:-p]+EPS))[:-1] * basis(u, p-1, T)[:-1] + ((T[p+1:] - u)/(T[p+1:]-T[1:-p]+EPS)) * basis(u, p-1, T)[1:]


# @jit(nopython=True, parallel=False)
def getControlPoints(knots, k):
    nCtrlPts = len(knots) - 1 - k
    cx = np.zeros(nCtrlPts)
    for i in range(nCtrlPts):
        tsum = 0
        for j in range(1, k + 1):
            tsum += knots[i + j]
        cx[i] = float(tsum) / k
    return cx

# @profile


# @jit(nopython=True, parallel=False)
def get_decode_operator(W, Nu, Nv):
    RNx = Nu * np.sum(W, axis=0)
    RNx /= np.sum(RNx, axis=1)[:, np.newaxis]
    RNy = Nv * np.sum(W, axis=1)
    RNy /= np.sum(RNy, axis=1)[:, np.newaxis]
    # print('Decode error res: ', RNx.shape, RNy.shape)
    # decoded = np.matmul(np.matmul(RNx, P), RNy.T)

    return RNx, RNy

# @profile


def decode(P, W, iNu, iNv):
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

        RNx = iNu * np.sum(W, axis=0)
        RNx /= np.sum(RNx, axis=1)[:, np.newaxis]
        RNy = iNv * np.sum(W, axis=1)
        RNy /= np.sum(RNy, axis=1)[:, np.newaxis]
        # print('Decode error res: ', RNx.shape, RNy.shape)
        decoded = np.matmul(np.matmul(RNx, P), RNy.T)
        # print('Decode error res: ', decoded.shape, decoded2.shape, np.max(np.abs(decoded.reshape((Nu.shape[0], Nv.shape[0])) - decoded2)))
        # print('Decode error res: ', z.shape, decoded.shape)

    return decoded


def Error(P, W, z, degree, Nu, Nv):
    Ploc = decode(P, W, Nu, Nv)
    return (Ploc - z)


def RMSE(P, W, z, degree, Nu, Nv):
    E = Error(P, W, z, degree, Nu, Nv)
    return math.sqrt(np.sum(E**2)/len(E))


def NMaxError(P, W, z, degree, Nu, Nv):
    E = Error(P, W, z, degree, Nu, Nv)
    return np.abs(E).max()/zRange


def NMSE(P, W, z, degree, Nu, Nv):
    return (Error(P, W, z, degree, Nu, Nv)**2).mean()/zRange


def toHomogeneous(P, W):
    return np.hstack(((P*W)[:, np.newaxis], W[:, np.newaxis]))


def fromHomogeneous(PW):
    P = PW[:, 0]
    W = PW[:, 1]
    return P/W, W


def getSplits(T, U):
    t = T[degree:-degree]
    toSplit = [((t[:-1] <= u) * (u <= t[1:])).astype(np.float) for u in U]
    toSplit = np.array(toSplit).ravel()
    splitSize = t.shape[0]-1
    splits = []
    for i in range(0, toSplit.shape[0], splitSize):
        splits.append(toSplit[i:i+splitSize])
    toSplit = np.unique(np.array([np.unique(np.where(split))[0] for split in splits]))
    toSplit += degree
    return toSplit


def splitT(T, toSplit, nPoints):
    Tnew = []
    for ti, tval in enumerate(T):
        TnewVal = tval.copy()
        Tnew.append(TnewVal)
        if ti in toSplit:
            inc = (float(T[ti+1]-tval)/2.)
            if ((TnewVal+inc)*nPoints) - (TnewVal*nPoints) < 2:
                print("Not enough input points to split", TnewVal, T[ti+1])
                toSplit.remove(ti)
                continue
            Tnew.append(TnewVal+inc)
    return np.array(Tnew)


def knotInsert(TU, TV, us, splitUs=True, r=1):
    toSplitU = set(getSplits(TU, us[..., 0]))
    toSplitV = set(getSplits(TV, us[..., 1]))
    return splitT(TU, toSplitU, nPointsX), splitT(TV, toSplitV, nPointsY), toSplitU, toSplitV


def knotRefine(P, W, TU, TV, Nu, Nv, U, V, zl, r=1, find_all=True, MAX_ERR=1e-2, reuseE=None):
    if reuseE is None:
        NSE = np.abs(Error(P, W, zl, degree, Nu, Nv))/zRange
    else:
        NSE = np.abs(reuseE)
    # NMSE = NSE.mean()
    Ev = np.copy(NSE).reshape(NSE.shape[0]*NSE.shape[1])
    NMSE = np.sqrt(np.sum(Ev**2)/len(Ev))

    if(NMSE <= MAX_ERR):
        return TU, TV, [], [], NSE, NMSE
    if find_all:
        rows, cols = np.where(NSE >= MAX_ERR)
        us = np.vstack((U[rows], V[cols])).T
    else:
        maxRow, maxCol = np.unravel_index(np.argmax(NSE), NSE.shape)
        NSE[maxRow, maxCol] = 0
        us = np.array([[U[maxRow], V[maxCol]]])
    TUnew, TVnew, Usplits, Vsplits = knotInsert(TU, TV, us, splitUs=find_all, r=r)
    return TUnew, TVnew, Usplits, Vsplits, NSE, NMSE


def insert_knot_u(P, W, T, u, k, degree, r=1, s=0):
    # Algorithm A5.3
    NP = np.array(P.shape)
    Q = np.zeros((NP[0]+r, NP[1], 2))
    # Initialize a local array of length p + 1
    R = np.zeros((degree+1, 2))
    PW = toHomogeneous(P, W)

    # Save the alphas
    alpha = [[0.0 for _ in range(r + 1)] for _ in range(degree - s)]
    for j in range(1, r + 1):
        L = k - degree + j
        for i in range(0, degree - j - s + 1):
            alpha[i][j] = (u - T[L + i]) / (T[i + k + 1] - T[L + i])

    # Update control points
    for row in range(0, NP[1]):
        for i in range(0, k - degree + 1):
            Q[i][row] = PW[i][row]
        for i in range(k - s, NP[0]):
            Q[i + r][row] = PW[i][row]
        # Load auxiliary control points
        for i in range(0, degree - s + 1):
            R[i] = (PW[k - degree + i][row]).copy()
        # Insert the knot r times
        for j in range(1, r + 1):
            L = k - degree + j
            for i in range(0, degree - j - s + 1):
                R[i][:] = [alpha[i][j] * elem2 + (1.0 - alpha[i][j]) * elem1
                           for elem1, elem2 in zip(R[i], R[i + 1])]
            Q[L][row] = R[0].copy()
            Q[k + r - j - s][row] = R[degree - j - s].copy()
        # Load the remaining control points
        L = k - degree + r
        for i in range(L + 1, k - s):
            Q[i][row] = R[i - L].copy()

    return Q


def insert_knot_v(P, W, T, u, k, degree, r=1, s=0):
    # Algorithm A5.3
    NP = np.array(P.shape)
    Q = np.zeros((NP[0], NP[1]+r, 2))
    # Initialize a local array of length p + 1
    R = np.zeros((degree+1, 2))
    PW = toHomogeneous(P, W)

    # Save the alphas
    alpha = [[0.0 for _ in range(r + 1)] for _ in range(degree - s)]
    for j in range(1, r + 1):
        L = k - degree + j
        for i in range(0, degree - j - s + 1):
            alpha[i][j] = (u - T[L + i]) / (T[i + k + 1] - T[L + i])

    # Update control points
    for col in range(0, NP[0]):
        for i in range(0, k - degree + 1):
            Q[col][i] = PW[col][i]
        for i in range(k - s, NP[1]):
            Q[col][i + r] = PW[col][i]
        # Load auxiliary control points
        for i in range(0, degree - s + 1):
            R[i] = (PW[col][k - degree + i]).copy()
        # Insert the knot r times
        for j in range(1, r + 1):
            L = k - degree + j
            for i in range(0, degree - j - s + 1):
                R[i][:] = [alpha[i][j] * elem2 + (1.0 - alpha[i][j]) * elem1
                           for elem1, elem2 in zip(R[i], R[i + 1])]
            Q[col][L] = R[0].copy()
            Q[col][k + r - j - s] = R[degree - j - s].copy()
        # Load the remaining control points
        L = k - degree + r
        for i in range(L + 1, k - s):
            Q[col][i] = R[i - L].copy()

    return Q


def deCasteljau2D(P, W, TU, TV, u, k, r=1):
    Qu = insert_knot_u(P, W, TU, u[0], k[0], degree, r)
    P, W = fromHomogeneous(Qu)
    Q = insert_knot_v(P, W, TV, u[1], k[1], degree, r)
    P, W = fromHomogeneous(Q)
    return P, W


def deCasteljau1D(P, W, T, u, k, r=1):
    NP = len(P)
    Qnew = np.zeros((NP+r, P.ndim+1))
    Rw = np.zeros((degree+1, P.ndim+1))
    PW = toHomogeneous(P, W)

    mp = NP+degree+1
    nq = len(Qnew)

    Qnew[:k-degree+1] = PW[:k-degree+1]
    Qnew[k+r:NP+1+r] = PW[k:NP+1]
    Rw[:degree+1] = PW[k-degree:k+1]

    for j in range(1, r+1):
        L = k-degree+j
        for i in range(degree-j+1):
            alpha = (u-T[L+i])/(T[i+k+1]-T[L+i])
            Rw[i] = alpha*Rw[i+1] + (1.0-alpha)*Rw[i]

        Qnew[L] = Rw[0]
        Qnew[k+r-j] = Rw[degree-j]
        Qnew[L+1:k] = Rw[1:k-L]

    P, W = fromHomogeneous(Qnew)
    return P, W


def L2LinfErrors(P, W, z, degree, Nu, Nv):

    E = Error(P, W, z, degree, Nu, Nv)
    LinfErr = np.abs(E).max()/zRange
    L2Err = math.sqrt(np.sum(E**2)/len(E))
    return [L2Err, LinfErr]


def lsqFit(Nu, Nv, W, z, use_cho=True):
    RNx = Nu * np.sum(W, axis=1)
    RNx /= np.sum(RNx, axis=1)[:, np.newaxis]
    RNy = Nv * np.sum(W, axis=0)
    RNy /= np.sum(RNy, axis=1)[:, np.newaxis]
    if use_cho:
        X = linalg.cho_solve(linalg.cho_factor(np.matmul(RNx.T, RNx)), RNx.T)
        Y = linalg.cho_solve(linalg.cho_factor(np.matmul(RNy.T, RNy)), RNy.T)
        zY = np.matmul(z, Y.T)
        return np.matmul(X, zY), []
    else:
        NTNxInv = np.linalg.inv(np.matmul(RNx.T, RNx))
        NTNyInv = np.linalg.inv(np.matmul(RNy.T, RNy))
        NxTQNy = np.matmul(RNx.T, np.matmul(z, RNy))
        return np.matmul(NTNxInv, np.matmul(NxTQNy, NTNyInv)), []
#             NTNxInv = np.linalg.inv(np.matmul(RNx.T,RNx))
#             NTNyInv = np.linalg.inv(np.matmul(RNy.T,RNy))
#             NxTQNy = np.matmul(RNx.T, np.matmul(z, RNy))
#             return np.matmul(NTNxInv, np.matmul(NxTQNy, NTNyInv)), []

# Let do the recursive iterations
# The variable `additiveSchwartz` controls whether we use
# Additive vs Multiplicative Schwartz scheme for DD resolution
####


class InputControlBlock:

    def __init__(self, bid, nCPi, coreb, xb, xl, yl, zl):
        if projectData and variableResolution:
            nCP = nCPi * (bid+1)
        else:
            nCP = nCPi
        self.nControlPoints = nCP[:]
        self.nControlPointSpans = self.nControlPoints - 1
        self.nInternalKnotSpans = self.nControlPoints - degree
        self.nPointsPerSubDX = len(xl)  # int(nPointsX / nSubDomainsX)
        self.nPointsPerSubDY = len(yl)  # int(nPointsY / nSubDomainsY)
        self.xbounds = xb
        self.corebounds = [coreb.min[0]-xb.min[0], -1+coreb.max[0]-xb.max[0],
                           coreb.min[1]-xb.min[1], -1+coreb.max[1]-xb.max[1]]
        self.xl = xl
        self.yl = yl
        self.zl = zl
        self.Dmini = np.array([min(xl), min(yl)])
        self.Dmaxi = np.array([max(xl), max(yl)])
        self.Bx = None  # Basis function object in x-dir
        self.By = None  # Basis function object in y-dir
        self.pAdaptive = np.zeros(self.nControlPoints)
        self.WAdaptive = np.ones(self.nControlPoints)
        self.knotsAdaptiveU = np.zeros(self.nControlPoints[0]+degree+1)
        self.knotsAdaptiveV = np.zeros(self.nControlPoints[1]+degree+1)
        self.knotsAll = []
        self.decodedAdaptive = np.zeros(zl.shape)
        self.decodedAdaptiveOld = np.zeros(zl.shape)

        self.U = []
        self.V = []
        self.Nu = []
        self.Nv = []
        self.leftconstraint = []
        self.rightconstraint = []
        self.topconstraint = []
        self.bottomconstraint = []
        self.topleftconstraint = []
        self.toprightconstraint = []
        self.bottomleftconstraint = []
        self.bottomrightconstraint = []
        self.leftconstraintKnots = []
        self.ghostleftknots = []
        self.rightconstraintKnots = []
        self.ghostrightknots = []
        self.topconstraintKnots = []
        self.ghosttopknots = []
        self.bottomconstraintKnots = []
        self.ghostbottomknots = []
        self.figHnd = None
        self.figHndErr = None
        self.figSuffix = ""
        self.globalIterationNum = 0
        self.adaptiveIterationNum = 0
        self.globalTolerance = 1e-13

        self.leftclamped = self.rightclamped = False
        self.topclamped = self.bottomclamped = False

        # Convergence related metrics and checks
        self.outerIteration = 0
        self.outerIterationConverged = False
        self.decodederrors = np.zeros(2)  # L2 and Linf
        self.errorMetricsL2 = np.zeros((nASMIterations), dtype='float64')
        self.errorMetricsLinf = np.zeros((nASMIterations), dtype='float64')

        self.pAdaptiveHistory = []

    def show(self, cp):

        print(
            "Rank: %d, Subdomain %d: Bounds = [%d - %d, %d - %d]" %
            (commWorld.rank, cp.gid(),
             self.xbounds.min[0],
             self.xbounds.max[0],
             self.xbounds.min[1],
             self.xbounds.max[1]))

    def compute_basis(self, degree, Tu, Tv):
        self.Bx = sp.BSplineBasis(order=degree+1, knots=Tu)
        self.By = sp.BSplineBasis(order=degree+1, knots=Tv)
        # print("TU = ", Tu, self.U[0], self.U[-1], self.Bx.greville())
        # print("TV = ", Tv, self.V[0], self.V[-1], self.By.greville())
        self.Nu = np.array(self.Bx.evaluate(self.U))
        self.Nv = np.array(self.By.evaluate(self.V))

        # return self.Nu, self.Nv

    def plot_control(self, cp):

        self.pMK = decode(self.pAdaptive, self.WAdaptive, self.Nu, self.Nv)

        Xi, Yi = np.meshgrid(self.xl, self.yl)

        axHnd = self.figHnd.gca(projection='3d')

        mycolors = cm.Spectral(self.pMK/zmax)
        # mycolors = cm.Spectral(self.pMK)
#         surf = axHnd.plot_surface(Xi, Yi, self.zl, cmap=cm.coolwarm, antialiased=True, alpha=0.75, label='Input')

        surf = axHnd.plot_surface(Xi, Yi, self.pMK.T, antialiased=False, alpha=0.95, cmap=cm.Spectral, label='Decoded',
                                  vmin=zmin, vmax=zmax,
                                  # facecolors=mycolors,
                                  linewidth=0.1, edgecolors='k')

        # surf._facecolors2d=surf._facecolors3d
        # surf._edgecolors2d=surf._edgecolors3d

        if cp.gid() == 0:
            # #             self.figHnd.colorbar(surf)
            # cbar_ax = self.figHnd.add_axes([min(self.xl), max(self.xl), 0.05, 0.7])
            self.figHnd.subplots_adjust(right=0.8)
            cbar_ax = self.figHnd.add_axes([0.85, 0.15, 0.05, 0.7])

            self.figHnd.colorbar(surf, cax=cbar_ax)

        if len(self.figSuffix):
            self.figHnd.savefig("decoded-data-%s.png" % (self.figSuffix))   # save the figure to file
        else:
            self.figHnd.savefig("decoded-data.png")   # save the figure to file

    def plot_error(self, cp):

        # self.pMK = decode(self.pAdaptive, self.WAdaptive, self.Nu, self.Nv)
        errorDecoded = np.abs(self.zl - self.pMK)/zRange

        Xi, Yi = np.meshgrid(self.xl, self.yl)

        axHnd = self.figHndErr.gca(projection='3d')
        surf = axHnd.plot_surface(Xi, Yi, np.log10(errorDecoded.T), cmap=cm.Spectral,
                                  label='Error', antialiased=False, alpha=0.95, linewidth=0.1, edgecolors='k')
        surf._facecolors2d = surf._facecolors3d
        surf._edgecolors2d = surf._edgecolors3d

        # self.figHndErr.colorbar(surf)
        # axHnd.legend()

        if len(self.figSuffix):
            self.figHndErr.savefig("error-data-%s.png" % (self.figSuffix))   # save the figure to file
        else:
            self.figHndErr.savefig("error-data.png")   # save the figure to file

    def output_vtk(self, cp):

        if useVTKOutput:

            self.pMK = decode(self.pAdaptive, self.WAdaptive, self.Nu, self.Nv)
            errorDecoded = np.abs(self.zl - self.pMK) / zRange

            Xi, Yi = np.meshgrid(self.xl, self.yl)

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
        #         self.pMK = decode(self.pAdaptive, self.WAdaptive, self.Nu, self.Nv)
        #         errorDecoded = self.zl - self.pMK

        print("Domain: ", cp.gid()+1, "Exact = ", self.zl)
        print("Domain: ", cp.gid()+1, "Exact - Decoded = ", np.abs(self.zl - self.pMK))

    def send_diy(self, cp):
        verbose = False
        link = cp.link()
        for i in range(len(link)):
            target = link.target(i)
            if len(self.pAdaptive):
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
                                  self.pAdaptive[:, -1:-2-degree-augmentSpanSpace:-1].shape)

                        cp.enqueue(target, self.pAdaptive[:, -1:-2-degree-augmentSpanSpace:-1])
                        cp.enqueue(target, self.knotsAdaptiveU[:])
                        cp.enqueue(target, self.knotsAdaptiveV[-1:-2-degree-augmentSpanSpace:-1])

                    else:  # target block is below current subdomain
                        if verbose:
                            print("%d sending to %d" % (cp.gid(), target.gid), ' Bottom: ',
                                  self.pAdaptive[:, 0:1+degree+augmentSpanSpace].shape)

                        cp.enqueue(target, self.pAdaptive[:, 0:1+degree+augmentSpanSpace])
                        cp.enqueue(target, self.knotsAdaptiveU[:])
                        cp.enqueue(target, self.knotsAdaptiveV[0:1+degree+augmentSpanSpace])

                elif dir[1] == 0:  # target is coupled in X-direction
                    if dir[0] > 0:  # target block is to the right of current subdomain
                        if verbose:
                            print("%d sending to %d" % (cp.gid(), target.gid), 'Left: ',
                                  self.pAdaptive[-1:-2-degree-augmentSpanSpace:-1, :].shape)

                        cp.enqueue(target, self.pAdaptive[-1:-2-degree-augmentSpanSpace:-1, :])
                        cp.enqueue(target, self.knotsAdaptiveV[:])
                        cp.enqueue(target, self.knotsAdaptiveU[-1:-2-degree-augmentSpanSpace:-1])

                    else:  # target block is to the left of current subdomain
                        if verbose:
                            print("%d sending to %d" % (cp.gid(), target.gid), 'Right: ',
                                  self.pAdaptive[degree+augmentSpanSpace::-1, :].shape)

                        cp.enqueue(target, self.pAdaptive[degree+augmentSpanSpace::-1, :])
                        cp.enqueue(target, self.knotsAdaptiveV[:])
                        cp.enqueue(target, self.knotsAdaptiveU[0:(degree+augmentSpanSpace+1)])

                else:

                    verbose = True
                    if dir[0] > 0 and dir[1] > 0:  # target block is diagonally right to  current subdomain
                        cp.enqueue(target, self.pAdaptive[:, -1:-2-degree-augmentSpanSpace:-1])
                        if verbose:
                            print("%d sending to %d" % (cp.gid(), target.gid), ' Diagonal = right-top: ',
                                  self.pAdaptive[:, -1:-2-degree-augmentSpanSpace:-1].shape)
                    if dir[0] < 0 and dir[1] > 0:  # target block is diagonally left to current subdomain
                        cp.enqueue(target, self.pAdaptive[:, 0:1+degree+augmentSpanSpace])
                        if verbose:
                            print("%d sending to %d" % (cp.gid(), target.gid), ' Diagonal = left-top: ',
                                  self.pAdaptive[:, 0:1+degree+augmentSpanSpace].shape)

                    if dir[0] < 0 and dir[1] < 0:  # target block is diagonally left bottom  current subdomain
                        cp.enqueue(target, self.pAdaptive[:, -1:-2-degree-augmentSpanSpace:-1])
                        if verbose:
                            print("%d sending to %d" % (cp.gid(), target.gid), ' Diagonal = left-bottom: ',
                                  self.pAdaptive[:, -1:-2-degree-augmentSpanSpace:-1].shape)
                    if dir[0] > 0 and dir[1] < 0:  # target block is diagonally left to current subdomain
                        cp.enqueue(target, self.pAdaptive[:, 0:1+degree+augmentSpanSpace])
                        if verbose:
                            print("%d sending to %d" % (cp.gid(), target.gid), ' Diagonal = right-bottom: ',
                                  self.pAdaptive[:, 0:1+degree+augmentSpanSpace].shape)
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
                    self.topconstraint = cp.dequeue(tgid)
                    self.topconstraintKnots = cp.dequeue(tgid)
                    self.ghosttopknots = cp.dequeue(tgid)
                    if verbose:
                        print("Top: %d received from %d: from direction %s" %
                              (cp.gid(), tgid, dir), self.topconstraint.shape, self.topconstraintKnots.shape)
                else:  # target block is below current subdomain
                    self.bottomconstraint = cp.dequeue(tgid)
                    self.bottomconstraintKnots = cp.dequeue(tgid)
                    self.ghostbottomknots = cp.dequeue(tgid)
                    if verbose:
                        print("Bottom: %d received from %d: from direction %s" %
                              (cp.gid(), tgid, dir), self.bottomconstraint.shape, self.bottomconstraintKnots.shape)

            elif dir[1] == 0:  # target is coupled in X-direction
                if dir[0] < 0:  # target block is to the left of current subdomain
                    # print('Right: ', np.array(o[2:pl+2]).reshape(useDerivativeConstraints+1,pll).T)

                    self.leftconstraint = cp.dequeue(tgid).T
                    self.leftconstraintKnots = cp.dequeue(tgid)
                    self.ghostleftknots = cp.dequeue(tgid)
                    if verbose:
                        print("Left: %d received from %d: from direction %s" %
                              (cp.gid(), tgid, dir), self.leftconstraint.shape, self.leftconstraintKnots.shape)

                else:  # target block is to right of current subdomain

                    self.rightconstraint = cp.dequeue(tgid).T
                    self.rightconstraintKnots = cp.dequeue(tgid)
                    self.ghostrightknots = cp.dequeue(tgid)
                    if verbose:
                        print("Right: %d received from %d: from direction %s" %
                              (cp.gid(), tgid, dir), self.rightconstraint.shape, self.rightconstraintKnots.shape)
            else:

                verbose = True
                if dir[0] < 0 and dir[1] < 0:  # sender block is diagonally right top to  current subdomain
                    self.toprightconstraint = cp.dequeue(tgid)
                    if verbose:
                        print("Top-right: %d received from %d: from direction %s" %
                              (cp.gid(), tgid, dir), self.toprightconstraint.shape)
                if dir[0] > 0 and dir[1] < 0:  # sender block is diagonally left top to current subdomain
                    self.topleftconstraint = cp.dequeue(tgid)
                    if verbose:
                        print("Top-left: %d received from %d: from direction %s" %
                              (cp.gid(), tgid, dir), self.topleftconstraint.shape)
                if dir[0] > 0 and dir[1] > 0:  # sender block is diagonally left bottom  current subdomain
                    self.bottomleftconstraint = cp.dequeue(tgid)
                    if verbose:
                        print("Bottom-left: %d received from %d: from direction %s" %
                              (cp.gid(), tgid, dir), self.bottomleftconstraint.shape)
                if dir[0] < 0 and dir[1] > 0:  # sender block is diagonally left to current subdomain
                    self.bottomrightconstraint = cp.dequeue(tgid)
                    if verbose:
                        print("Bottom-right: %d received from %d: from direction %s" %
                              (cp.gid(), tgid, dir), self.bottomrightconstraint.shape)

        return

    def send(self, cp):
        verbose = False
        debug = False
        link = cp.link()
        for i in range(len(link)):
            target = link.target(i)
            o = np.zeros(1)
            if len(self.pAdaptive):
                dir = link.direction(i)
                if dir[0] == 0 and dir[1] == 0:
                    continue

                # ONLY consider coupling through faces and not through verties
                # This means either dir[0] or dir[1] has to be "0" for subdomain coupling to be active
                # Hence we only consider 4 neighbor cases, instead of 8.
                if dir[0] == 0:  # target is coupled in Y-direction
                    if dir[1] > 0:  # target block is above current subdomain
                        pl = (degree+augmentSpanSpace+1)*len(self.pAdaptive[:, -1])
                        if pl == 0:
                            continue
                        o = np.zeros(2+pl+self.knotsAdaptiveU.shape[0]+(degree+augmentSpanSpace+1))

                        o[0] = pl
                        o[1] = self.knotsAdaptiveU.shape[0]

                        if verbose:
                            print("%d sending to %d" % (cp.gid(), target.gid), ' Top: ',
                                  self.pAdaptive[:, -1:-2-degree-augmentSpanSpace:-1])
                        o[2:pl+2] = self.pAdaptive[:, -1:-2-degree-augmentSpanSpace:-1].T.reshape(pl)
                        o[pl+2:pl+2+int(o[1])] = self.knotsAdaptiveU[:]
                        o[pl+2+int(o[1]):] = self.knotsAdaptiveV[-1:-2-degree-augmentSpanSpace:-1]

                    else:  # target block is below current subdomain
                        pl = (degree+augmentSpanSpace+1)*len(self.pAdaptive[:, 0])
                        if pl == 0:
                            continue
                        o = np.zeros(2+pl+self.knotsAdaptiveU.shape[0]+(degree+augmentSpanSpace+1))

                        o[0] = pl
                        o[1] = self.knotsAdaptiveU.shape[0]

                        if verbose:
                            print("%d sending to %d" % (cp.gid(), target.gid), ' Bottom: ',
                                  self.pAdaptive[:, 0:1+degree+augmentSpanSpace])
                        o[2:pl+2] = self.pAdaptive[:, 0:1+degree+augmentSpanSpace].T.reshape(pl)
                        o[pl+2:pl+2+int(o[1])] = self.knotsAdaptiveU[:]
                        o[pl+2+int(o[1]):] = self.knotsAdaptiveV[0:(degree+augmentSpanSpace+1)]

                # else:  # target is coupled in Y-direction
                #     if dir[1] > 0:  # target block is above current subdomain
                #         pl = (degree+augmentSpanSpace+1)*len(self.pAdaptive[:, -1])
                #         if pl == 0:
                #             continue
                #         o = np.zeros(2+pl+self.knotsAdaptiveU.shape[0]+(degree+augmentSpanSpace+1))

                #         o[0] = pl
                #         o[1] = self.knotsAdaptiveU.shape[0]

                #         if verbose:
                #             print("%d sending to %d" % (cp.gid(), target.gid), ' Top: ',
                #                   self.pAdaptive[:, -1:-2-degree-augmentSpanSpace:-1])
                #         o[2:pl+2] = self.pAdaptive[:, -1:-2-degree-augmentSpanSpace:-1].T.reshape(pl)
                #         o[pl+2:pl+2+int(o[1])] = self.knotsAdaptiveU[:]
                #         o[pl+2+int(o[1]):] = self.knotsAdaptiveV[-1:-2-degree-augmentSpanSpace:-1]

                #     else:  # target block is below current subdomain
                #         pl = (degree+augmentSpanSpace+1)*len(self.pAdaptive[:, 0])
                #         if pl == 0:
                #             continue
                #         o = np.zeros(2+pl+self.knotsAdaptiveU.shape[0]+(degree+augmentSpanSpace+1))

                #         o[0] = pl
                #         o[1] = self.knotsAdaptiveU.shape[0]

                #         if verbose:
                #             print("%d sending to %d" % (cp.gid(), target.gid), ' Bottom: ',
                #                   self.pAdaptive[:, 0:1+degree+augmentSpanSpace])
                #         o[2:pl+2] = self.pAdaptive[:, 0:1+degree+augmentSpanSpace].T.reshape(pl)
                #         o[pl+2:pl+2+int(o[1])] = self.knotsAdaptiveU[:]
                #         o[pl+2+int(o[1]):] = self.knotsAdaptiveV[0:(degree+augmentSpanSpace+1)]

                if dir[1] == 0:  # target is coupled in X-direction
                    if dir[0] > 0:  # target block is to the right of current subdomain
                        pl = (degree+augmentSpanSpace+1)*len(self.pAdaptive[0, :])
                        if debug:
                            pl = self.pAdaptive.shape[0]*self.pAdaptive.shape[1]
                        if pl == 0:
                            continue
                        o = np.zeros(2+pl+self.knotsAdaptiveV.shape[0]+(degree+augmentSpanSpace+1))

                        o[0] = pl
                        o[1] = self.knotsAdaptiveV.shape[0]

                        if verbose:
                            print("%d sending to %d" % (cp.gid(), target.gid), 'Left: ',
                                  self.pAdaptive[0:1+degree+augmentSpanSpace:, :])

                        if debug:
                            o[2:pl+2] = self.pAdaptive.reshape(pl)
                        else:
                            # o[2:pl+2] = self.pAdaptive[0:1+degree+augmentSpanSpace:, :].reshape(pl)
                            o[2:pl+2] = self.pAdaptive[-1:-2-degree-augmentSpanSpace:-1, :].reshape(pl)
                        o[pl+2:pl+2+int(o[1])] = self.knotsAdaptiveV[:]
                        o[pl+2+int(o[1]):] = self.knotsAdaptiveU[-1:-2-degree-augmentSpanSpace:-1]

                    else:  # target block is to the left of current subdomain
                        pl = (degree+augmentSpanSpace+1)*len(self.pAdaptive[-1, :])
                        if debug:
                            pl = self.pAdaptive.shape[0]*self.pAdaptive.shape[1]
                        if pl == 0:
                            continue
                        o = np.zeros(2+pl+self.knotsAdaptiveV.shape[0]+(degree+augmentSpanSpace+1))

                        o[0] = pl
                        o[1] = self.knotsAdaptiveV.shape[0]

                        if verbose:
                            print("%d sending to %d" % (cp.gid(), target.gid), 'Right: ',
                                  self.pAdaptive[-1:-2-degree-augmentSpanSpace:-1, :])

                        if debug:
                            o[2:pl+2] = self.pAdaptive.reshape(pl)
                        else:
                            # o[2:pl+2] = self.pAdaptive[-1:-2-degree-augmentSpanSpace:-1, :].reshape(pl)
                            # o[2:pl+2] = self.pAdaptive[0:1+degree+augmentSpanSpace:, :].reshape(pl)
                            o[2:pl+2] = self.pAdaptive[degree+augmentSpanSpace::-1, :].reshape(pl)
                        o[pl+2:pl+2+int(o[1])] = self.knotsAdaptiveV[:]
                        o[pl+2+int(o[1]):] = self.knotsAdaptiveU[0:(degree+augmentSpanSpace+1)]

            if len(o) > 1 and verbose:
                print("%d sending to %d: %s to direction %s" % (cp.gid(), target.gid, o, dir))
            cp.enqueue(target, o)

    def recv(self, cp):
        verbose = False
        debug = False
        link = cp.link()
        for i in range(len(link)):
            tgid = link.target(i).gid
            o = cp.dequeue(tgid)
            if len(o) == 1:
                continue

            dir = link.direction(i)
            pl = int(o[0])
            pll = int(pl/(degree+augmentSpanSpace+1))
            tl = int(o[1])
            # print("%d received from %d: %s from direction %s, with sizes %d+%d" % (cp.gid(), tgid, o, dir, pl, tl))

            # ONLY consider coupling through faces and not through verties
            # This means either dir[0] or dir[1] has to be "0" for subdomain coupling to be active
            # Hence we only consider 4 neighbor cases, instead of 8.
            if dir[0] == 0 and dir[1] == 0:
                continue

            if dir[0] == 0:  # target is coupled in Y-direction
                if dir[1] > 0:  # target block is above current subdomain
                    self.topconstraint = np.array(
                        o[2:pl+2]).reshape(degree+augmentSpanSpace+1, pll).T if pl > 0 else []
                    self.topconstraintKnots = np.array(o[pl+2:pl+2+tl]) if tl > 0 else []
                    self.ghosttopknots = np.array(o[pl+2+tl:]) if tl > 0 else []
                    if verbose:
                        print("Top: %d received from %d: from direction %s, with sizes %d+%d" %
                              (cp.gid(), tgid, dir, pl, tl), self.topconstraint, self.topconstraintKnots)
                else:  # target block is below current subdomain
                    self.bottomconstraint = np.array(
                        o[2:pl+2]).reshape(degree+augmentSpanSpace+1, pll).T if pl > 0 else []
                    self.bottomconstraintKnots = np.array(o[pl+2:pl+2+tl]) if tl > 0 else []
                    self.ghostbottomknots = np.array(o[pl+2+tl:]) if tl > 0 else []
                    if verbose:
                        print("Bottom: %d received from %d: from direction %s, with sizes %d+%d" %
                              (cp.gid(), tgid, dir, pl, tl), self.bottomconstraint, self.bottomconstraintKnots)

            if dir[1] == 0:  # target is coupled in X-direction
                if dir[0] < 0:  # target block is to the left of current subdomain
                    # print('Right: ', np.array(o[2:pl+2]).reshape(useDerivativeConstraints+1,pll).T)
                    if debug:
                        self.leftconstraint = np.array(
                            o[2:pl+2]).reshape(16, int(pl/16)).T if pl > 0 else []
                    else:
                        self.leftconstraint = np.array(
                            o[2:pl+2]).reshape(degree+augmentSpanSpace+1, pll).T if pl > 0 else []
                    self.leftconstraintKnots = np.array(o[pl+2:pl+2+tl]) if tl > 0 else []
                    self.ghostleftknots = np.array(o[pl+2+tl:]) if tl > 0 else []
                    if verbose:
                        print("Left: %d received from %d: from direction %s, with sizes %d+%d" %
                              (cp.gid(), tgid, dir, pl, tl), self.leftconstraint, self.leftconstraintKnots)
                        # print("Left: %d received from %d: from direction %s, with sizes %d+%d" % (cp.gid(), tgid, dir, pl, tl), self.leftconstraint, - self.pAdaptive[:,0])

                else:  # target block is to right of current subdomain
                    # print('Left: ', np.array(o[2:pl+2]).reshape(useDerivativeConstraints+1,pll).T)
                    if debug:
                        self.leftconstraint = np.array(
                            o[2:pl+2]).reshape(16, int(pl/16)).T if pl > 0 else []
                    else:
                        # self.rightconstraint = np.array(
                        #     o[2:pl+2]).reshape(degree+augmentSpanSpace+1, pll).T if pl > 0 else []
                        self.rightconstraint = np.array(
                            o[2:pl+2]).reshape(degree+augmentSpanSpace+1, pll).T if pl > 0 else []
                    self.rightconstraintKnots = np.array(o[pl+2:pl+2+tl]) if tl > 0 else []
                    self.ghostrightknots = np.array(o[pl+2+tl:]) if tl > 0 else []
                    if verbose:
                        print("Right: %d received from %d: from direction %s, with sizes %d+%d" %
                              (cp.gid(), tgid, dir, pl, tl), self.rightconstraint, self.rightconstraintKnots)
                    # print("Right: %d received from %d: from direction %s, with sizes %d+%d" % (cp.gid(), tgid, dir, pl, tl), self.rightconstraint, - self.pAdaptive[:,-1])

    # @profile

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
        plen = self.pAdaptive.shape[0]*self.pAdaptive.shape[1]
        self.pAdaptiveHistory[:, 1:] = self.pAdaptiveHistory[:, :-1]
        self.pAdaptiveHistory[:, 0] = np.copy(self.pAdaptive).reshape(plen)

        vAcc = []
        if not useAitken:
            if iterationNumber > nWynnEWork:  # For Wynn-E[silon
                vAcc = np.zeros(plen)
                for dofIndex in range(plen):
                    expVal = self.WynnEpsilon(
                        self.pAdaptiveHistory[dofIndex, :],
                        math.floor((nWynnEWork - 1) / 2))
                    vAcc[dofIndex] = expVal[-1, -1]
                print('Performing scalar Wynn-Epsilon algorithm: Error is ',
                      np.linalg.norm(self.pAdaptive.reshape(plen) - vAcc))  # , (self.pAdaptive - vAcc))
                self.pAdaptive = vAcc[:].reshape(self.pAdaptive.shape[0],
                                                 self.pAdaptive.shape[1])

        else:
            if iterationNumber > 3:  # For Aitken acceleration
                vAcc = self.VectorAitken(self.pAdaptiveHistory).reshape(self.pAdaptive.shape[0],
                                                                        self.pAdaptive.shape[1])
                # vAcc = np.zeros(self.pAdaptive.shape)
                # for dofIndex in range(len(self.pAdaptive)):
                #     vAcc[dofIndex] = self.Aitken(self.pAdaptiveHistory[dofIndex, :])
                print('Performing Aitken Acceleration algorithm: Error is ', np.linalg.norm(self.pAdaptive - vAcc))
                self.pAdaptive = vAcc[:]

    def initialize_data(self, cp):

        # Subdomain ID: iSubDom = cp.gid()+1
        self.nControlPointSpans = self.nControlPoints - 1
        self.nInternalKnotSpans = self.nControlPoints - degree  # + 1

        inc = (self.Dmaxi - self.Dmini) / self.nInternalKnotSpans
        # print ("self.nInternalKnotSpans = ", self.nInternalKnotSpans, " inc = ", inc)

        # # Generate the knots in X and Y direction
        # tu = np.linspace(self.Dmini[0] + inc[0], self.Dmaxi[0] - inc[0], self.nInternalKnotSpans[0] - 1)
        # tv = np.linspace(self.Dmini[1] + inc[1], self.Dmaxi[1] - inc[1], self.nInternalKnotSpans[1] - 1)
        # self.knotsAdaptiveU = np.concatenate(([self.Dmini[0]] * (degree+1), tu, [self.Dmaxi[0]] * (degree+1)))
        # self.knotsAdaptiveV = np.concatenate(([self.Dmini[1]] * (degree+1), tv, [self.Dmaxi[1]] * (degree+1)))

        # self.U = np.linspace(self.Dmini[0], self.Dmaxi[0], self.nPointsPerSubDX)  # self.nPointsPerSubDX, nPointsX
        # self.V = np.linspace(self.Dmini[1], self.Dmaxi[1], self.nPointsPerSubDY)  # self.nPointsPerSubDY, nPointsY

        tu = np.linspace(self.Dmini[0] + inc[0], self.Dmaxi[0] - inc[0], self.nInternalKnotSpans[0] - 1)
        tv = np.linspace(self.Dmini[1] + inc[1], self.Dmaxi[1] - inc[1], self.nInternalKnotSpans[1] - 1)
        if nSubDomains > 1 and not fullyPinned:

            print("Subdomain: ", cp.gid(), " dmax_X: ", self.Dmini[0], self.Dmaxi[0], xmin, xmax)
            if abs(self.Dmaxi[0] - xmax) < 1e-12 and abs(self.Dmini[0] - xmin) < 1e-12:
                self.knotsAdaptiveU = np.concatenate(([self.Dmini[0]] * (degree+1), tu, [self.Dmaxi[0]] * (degree+1)))
                self.leftclamped = self.rightclamped = True
            else:
                if abs(self.Dmaxi[0] - xmax) < 1e-12:
                    self.knotsAdaptiveU = np.concatenate(([self.Dmini[0]] * (1), tu, [self.Dmaxi[0]] * (degree+1)))
                    self.rightclamped = True

                else:
                    if abs(self.Dmini[0] - xmin) < 1e-12:
                        self.knotsAdaptiveU = np.concatenate(([self.Dmini[0]] * (degree+1), tu, [self.Dmaxi[0]] * (1)))
                        self.leftclamped = True

                    else:
                        self.knotsAdaptiveU = np.concatenate(
                            ([self.Dmini[0]] * (1),
                                tu, [self.Dmaxi[0]] * (1)))
                        self.leftclamped = self.rightclamped = False

            print("Subdomain: ", cp.gid(), " dmax_Y: ", self.Dmini[1], self.Dmaxi[1], ymin, ymax)
            if abs(self.Dmaxi[1] - ymax) < 1e-12 and abs(self.Dmini[1] - ymin) < 1e-12:
                print("Subdomain: ", cp.gid(), " checking top and bottom Y: ",
                      self.Dmaxi[1], ymax, abs(self.Dmaxi[1] - ymax))
                self.knotsAdaptiveV = np.concatenate(([self.Dmini[1]] * (degree+1), tv, [self.Dmaxi[1]] * (degree+1)))
                self.topclamped = self.bottomclamped = True
            else:

                if abs(self.Dmaxi[1] - ymax) < 1e-12:
                    print("Subdomain: ", cp.gid(), " checking top Y: ", self.Dmaxi[1], ymax, abs(self.Dmaxi[1] - ymax))
                    self.knotsAdaptiveV = np.concatenate(([self.Dmini[1]] * (1), tv, [self.Dmaxi[1]] * (degree+1)))
                    self.topclamped = True

                else:

                    print("Subdomain: ", cp.gid(), " checking bottom Y: ",
                          self.Dmini[1], ymin, abs(self.Dmini[1] - ymin))
                    if abs(self.Dmini[1] - ymin) < 1e-12:
                        self.knotsAdaptiveV = np.concatenate(([self.Dmini[1]] * (degree+1), tv, [self.Dmaxi[1]] * (1)))
                        self.bottomclamped = True

                    else:
                        self.knotsAdaptiveV = np.concatenate(
                            ([self.Dmini[1]] * (1),
                                tv, [self.Dmaxi[1]] * (1)))

                        self.topclamped = self.bottomclamped = False

            print("Subdomain: ", cp.gid(), " clamped ? ", self.leftclamped,
                  self.rightclamped, self.topclamped, self.bottomclamped)

        else:
            self.knotsAdaptiveU = np.concatenate(([self.Dmini[0]] * (degree+1), tu, [self.Dmaxi[0]] * (degree+1)))
            self.knotsAdaptiveV = np.concatenate(([self.Dmini[1]] * (degree+1), tv, [self.Dmaxi[1]] * (degree+1)))
            self.leftclamped = self.rightclamped = self.topclamped = self.bottomclamped = False

        # self.U = np.linspace(self.Dmini[0], self.Dmaxi[0], self.nPointsPerSubDX)  # self.nPointsPerSubDX, nPointsX
        # self.V = np.linspace(self.Dmini[1], self.Dmaxi[1], self.nPointsPerSubDY)  # self.nPointsPerSubDY, nPointsY
        self.U = np.linspace(self.xl[0], self.xl[-1], self.nPointsPerSubDX)
        self.V = np.linspace(self.yl[0], self.yl[-1], self.nPointsPerSubDY)

        self.pAdaptive = np.zeros(self.nControlPoints)
        self.WAdaptive = np.ones(self.nControlPoints)

    def augment_spans(self, cp):

        print("Subdomain -- ", cp.gid()+1, ": before Shapes: ",
              self.pAdaptive.shape, self.WAdaptive.shape, self.knotsAdaptiveU, self.knotsAdaptiveV)
        if not self.leftclamped:  # Pad knot spans from the left of subdomain
            print("\tSubdomain -- ", cp.gid()+1, ": left ghost: ", self.ghostleftknots)
            self.knotsAdaptiveU = np.concatenate(
                (self.ghostleftknots[-1:0:-1], self.knotsAdaptiveU))

        if not self.rightclamped:  # Pad knot spans from the right of subdomain
            print("\tSubdomain -- ", cp.gid()+1, ": right ghost: ", self.ghostrightknots)
            self.knotsAdaptiveU = np.concatenate(
                (self.knotsAdaptiveU, self.ghostrightknots[1:]))

        if not self.topclamped:  # Pad knot spans from the left of subdomain
            print("\tSubdomain -- ", cp.gid()+1, ": top ghost: ", self.ghosttopknots)
            self.knotsAdaptiveV = np.concatenate(
                (self.knotsAdaptiveV, self.ghosttopknots[1:]))

        if not self.bottomclamped:  # Pad knot spans from the right of subdomain
            print("\tSubdomain -- ", cp.gid()+1, ": bottom ghost: ", self.ghostbottomknots)
            self.knotsAdaptiveV = np.concatenate(
                (self.ghostbottomknots[-1:0:-1], self.knotsAdaptiveV))

        print("Subdomain -- ", cp.gid()+1, ": after Shapes: ",
              self.pAdaptive.shape, self.WAdaptive.shape, self.knotsAdaptiveU, self.knotsAdaptiveV)

    def augment_inputdata(self, cp):

        verbose = False
        indicesX = np.where(np.logical_and(
            x >= self.knotsAdaptiveU[degree]-1e-10, x <= self.knotsAdaptiveU[-degree-1]+1e-10))
        print(indicesX)
        indicesY = np.where(np.logical_and(
            y >= self.knotsAdaptiveV[degree]-1e-10, y <= self.knotsAdaptiveV[-degree-1]+1e-10))
        print(indicesY)
        lboundX = indicesX[0][0]-1 if indicesX[0][0] > 0 else 0
        uboundX = indicesX[0][-1]+1 if indicesX[0][-1] < len(x) else indicesX[0][-1]
        lboundY = indicesY[0][0]-1 if indicesY[0][0] > 0 else 0
        uboundY = indicesY[0][-1]+1 if indicesY[0][-1] < len(y) else indicesY[0][-1]

        print(lboundX, uboundX, lboundY, uboundY)
        if verbose:
            print("Subdomain -- {0}: before augment -- left = {1}, right = {2}, shape = {3}".format(cp.gid()+1,
                                                                                                    self.xl[0], self.xl[-1], self.xl.shape))

        self.xl = x[lboundX:uboundX]  # x[indicesX]
        self.yl = y[lboundY:uboundY]  # y[indicesY]
        print(z.shape, indicesX[0].shape, indicesY[0].shape)
        self.zl = z[lboundX:uboundX, lboundY:uboundY]  # y[lbound:ubound]
        self.nPointsPerSubDX = self.xl.shape[0]  # int(nPoints / nSubDomains) + overlapData
        self.nPointsPerSubDY = self.yl.shape[0]  # int(nPoints / nSubDomains) + overlapData

        hx = (xmax-xmin)/nSubDomainsX
        hy = (ymax-ymin)/nSubDomainsY
        # Store the core indices before augment

        postol = 1e-10
        cindicesX = np.array(np.where(np.logical_and(
            self.xl >= self.xbounds.min[0]-postol, self.xl <= self.xbounds.max[0]+postol)))
        cindicesY = np.array(np.where(np.logical_and(
            self.yl >= self.xbounds.min[1]-postol, self.yl <= self.xbounds.max[1]+postol)))

        print("self.corebounds = ", self.xbounds, self.corebounds)
        # cindices = np.array(
        #     np.where(
        #         np.logical_and(
        #             np.logical_and(
        #                 self.xl >= xmin + cp.gid() * hx - 1e-10, self.xl <= xmin + (cp.gid() + 1) * hx + 1e-10),
        #             np.logical_and(
        #                 self.yl >= ymin + cp.gid() * hy - 1e-10, self.yl <= ymin + (cp.gid() + 1) * hy + 1e-10))))
        # print('cindices: ', cindices, self.xl, self.yl,
        #       cp.gid(), hx, hy, xmin + cp.gid()*hx-1e-8, xmin + (cp.gid()+1)*hx+1e-8)
        print(
            'cindicesX: ', cindicesX, self.xl[0],
            self.xl[-1],
            x[self.xbounds.min[0]],
            x[self.xbounds.max[0]])
        print('cindicesY: ', cindicesY, self.yl[0], self.yl[-1], y[self.xbounds.min[1]], y[self.xbounds.max[1]])
        self.corebounds = [[cindicesX[0][0], cindicesX[0][-1]], [cindicesY[0][0], cindicesY[0][-1]]]

        # print('Corebounds:', cindices[0][0], cindices[-1][-1])

        if verbose:
            print("Subdomain -- {0}: after augment -- left = {1}, right = {2}, shape = {3}".format(cp.gid()+1,
                                                                                                   x[indices[0]], x[indices[-1]], self.xl.shape))

        # print("Subdomain -- {0}: cindices -- {1} {2}, original x bounds = {3} {4}".format(cp.gid()+1,
        #                                                                                   self.xl[self.corebounds[0]], self.xl[self.corebounds[1]], self.xl[0], self.xl[-1]))
        self.decodedAdaptive = np.zeros(self.yl.shape)
        self.decodedAdaptiveOld = np.zeros(self.yl.shape)

    def LSQFit_NonlinearOptimize(self, idom, W, degree, constraintsAll=None):

        constraints = None
        jacobian_const = None
        solution = []

        # Initialize relevant data
        if constraintsAll is not None:
            constraints = constraintsAll['P']  # [:,:].reshape(W.shape)
            # knotsAllU = constraintsAll['Tu']
            # knotsAllV = constraintsAll['Tv']
            # weightsAll = constraintsAll['W']
            initSol = constraints[:, :]

        else:
            print('Constraints are all null. Solving unconstrained.')
            initSol = np.ones_like(W)

        # interface_constraints_obj['left'] = self.leftconstraint
        # interface_constraints_obj['leftknots'] = self.leftconstraintKnots
        # interface_constraints_obj['right'] = self.rightconstraint
        # interface_constraints_obj['rightknots'] = self.rightconstraintKnots
        # interface_constraints_obj['top'] = self.topconstraint
        # interface_constraints_obj['topknots'] = self.topconstraintKnots
        # interface_constraints_obj['bottom'] = self.bottomconstraint
        # interface_constraints_obj['bottomknots'] = self.bottomconstraintKnots

        # Compute hte linear operators needed to compute decoded residuals
        # self.Nu = basis(self.U[np.newaxis,:],degree,Tunew[:,np.newaxis]).T
        # self.Nv = basis(self.V[np.newaxis,:],degree,Tvnew[:,np.newaxis]).T
        decodeOpX, decodeOpY = get_decode_operator(W, self.Nu, self.Nv)

        if constraints is not None:
            decodedPrevIterate = np.matmul(np.matmul(decodeOpX, constraints), decodeOpY.T)

        def residual_operator_Ab(Pin, verbose=False):  # checkpoint3

            P = Pin.reshape(W.shape)

            # decodedValues = np.matmul(np.matmul(decodeOpX, P), decodeOpY.T)

            # Error = Z - D
            # D = (X * P) * Y^T
            # X^T Z Y = (X^T * X) * P * (Y^T * Y)
            Aoper = np.matmul(np.matmul(decodeOpX.T, decodeOpX), np.matmul(
                decodeOpY.T, decodeOpY))  # np.matmul(RN.T, RN)
            Brhs = np.matmul(np.matmul(decodeOpX.T, self.zl), decodeOpY)  # RN.T @ ysl
            # print('Input P = ', Pin, Aoper.shape, Brhs.shape)

            residual_nrm_vec = Brhs - Aoper @ P
            residual_nrm = np.linalg.norm(residual_nrm_vec, ord=2)

            print("Error Norm vector: ", residual_nrm)

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

        # Compute the residual as sum of two components
        # 1. The decoded error evaluated at P
        # 2. A penalized domain boundary constraint component
        # @jit(nopython=True, parallel=False)
        def residual(Pin, verbose=False):

            P = Pin.reshape(W.shape)
            alpha = 1
            bc_penalty = 1e7
            decoded_penalty = 1
            # bc_penalty = 1
            # decoded_penalty = 0
            useDiagonalBlocks = True

            # Residuals are in the decoded space - so direct way to constrain the boundary data
            # Residuals are in the decoded space - so direct way to constrain the boundary data
            # decoded = decode(P, W, self.Nu, self.Nv)
            # decoded = np.matmul(np.matmul(decodeOpX, P), decodeOpY.T)
            decoded = np.matmul(np.matmul(decodeOpY, P), decodeOpX.T)
            residual_decoded = np.abs(decoded - self.zl)/zRange
            residual_vec_decoded = residual_decoded.reshape(residual_decoded.shape[0]*residual_decoded.shape[1])
            decoded_residual_norm = np.sqrt(np.sum(residual_vec_decoded**2)/len(residual_vec_decoded))

#             res1 = np.dot(res_dec, self.Nv)
#             residual = np.dot(self.Nu.T, res1).reshape(self.Nu.shape[1]*self.Nv.shape[1])

            ltn = rtn = tpn = btn = 0
            constrained_residual_norm = 0
            if constraints is not None and len(constraints) > 0 and constrainInterfaces:
                if len(self.leftconstraint):

                    # Compute the residual for left interface condition
                    # constrained_residual_norm += np.sum( ( P[0,:] - (leftdata[:]) )**2 ) / len(leftdata[:,0])
                    # ltn = np.sqrt(
                    #     (np.sum((P[0, :] - 0.5 * (constraints[0, :] + left[:, degree-1])) ** 2) / len(P[0, :])))
                    ltn = np.sqrt(
                        np.sum(
                            (alpha * P[0, :] + (1 - alpha) * constraints[0, :] - self.leftconstraint[:, degree - 1]) ** 2) /
                        len(P[0, :]))
                    constrained_residual_norm += ltn
                    # print(idom, ': Left constrained norm: ', ltn)
                    # print('Left Shapes knots: ', constrained_residual_norm, len(left[:]), len(
                    #     knotsAllV), self.yl.shape, left, (constraints[-1, :]), P[0, :], P[-1, :])

                if len(self.rightconstraint):

                    # Compute the residual for right interface condition
                    # constrained_residual_norm += np.sum( ( P[-1,:] - (rightdata[:]) )**2 ) / len(rightdata[:,0])
                    # rtn = np.sqrt(
                    #     (np.sum((P[-1, :] - 0.5 * (constraints[-1, :] + right[:, degree-1]))**2) / len(right[:, -1])))
                    rtn = np.sqrt(
                        np.sum(
                            (alpha * P[-1, :] + (1 - alpha) * constraints[-1, :] - self.rightconstraint[:, 1]) ** 2) /
                        len(P[-1, :]))
                    # rtn = np.sqrt(
                    #     (np.sum((0.5 * (P[-1, :] - right[:, degree-1]))**2) / len(P[-1, :])))
                    constrained_residual_norm += rtn
                    # print(idom, ': Right constrained norm: ', rtn)
                    # print('Right Shapes knots: ', constrained_residual_norm, len(right[:]), len(
                    #     knotsAllV), self.yl.shape, right, (constraints[0, :]), P[0, :], P[-1, :])

                if len(self.topconstraint):

                    # Compute the residual for top interface condition
                    # constrained_residual_norm += np.sum( ( P[:,-1] - (topdata[:]) )**2 ) / len(topdata[:,0])
                    # tpn = np.sqrt(
                    #     (np.sum((P[:, -1] - 0.5 * (constraints[:, -1] + top[:, degree-1])) ** 2) / len(top[:, 0])))
                    tpn = np.sqrt(
                        np.sum(
                            (alpha * P[:, -1] + (1 - alpha) * constraints[:, -1] - self.topconstraint[:, degree - 1]) ** 2) /
                        len(P[:, -1]))
                    # tpn = np.sqrt(
                    #     (np.sum((0.5 * (P[:, -1] + top[:, degree-1])) ** 2) / len(P[:, -1])))
                    constrained_residual_norm += tpn
                    # print(idom, ': Top constrained norm: ', tpn)
                    # print('Top: ', constrained_residual_norm, P[:, -1],
                    #       constraints[:, -1], P[:, 0], constraints[:, 0], top[:])

                if len(self.bottomconstraint):

                    # Compute the residual for bottom interface condition
                    # constrained_residual_norm += np.sum( ( P[:,0] - (bottomdata[:]) )**2 ) / len(bottomdata[:,0])
                    # btn = np.sqrt(
                    #     np.sum((P[:, 0] - 0.5 * (constraints[:, 0] + bottom[:, degree-1]))**2) / len(bottom[:, 0]))
                    btn = np.sqrt(
                        np.sum(
                            (alpha * P[:, 0] + (1 - alpha) * constraints[:, 0] - self.bottomconstraint[:, degree - 1]) ** 2) /
                        len(P[:, 0]))
                    # btn = np.sqrt(
                    #     np.sum((0.5 * (P[:, 0] + bottom[:, degree-1]))**2) / len(P[:, 0]))
                    constrained_residual_norm += btn
                    # print(idom, ': Bottom constrained norm: ', btn)
                    # print(
                    #     'Bottom: ', constrained_residual_norm, P[:, -1],
                    #     constraints[:, -1],
                    #     P[:, 0],
                    #     constraints[:, 0],
                    #     bottom[:])

                if useDiagonalBlocks:

                    if len(self.topleftconstraint):

                        # Compute the residual for top left interface condition
                        bnderr = np.sqrt(np.sum(
                            (alpha * P[: degree, : degree] + (1 - alpha) *
                             constraints[: degree, : degree] - self.topleftconstraint
                             [: degree, : degree].T) ** 2) / len(P[0, :]))
                        constrained_residual_norm += bnderr

                    if len(self.toprightconstraint):

                        # Compute the residual for top left interface condition
                        bnderr = np.sqrt(np.sum(
                            (alpha * P[-degree:, : degree] + (1 - alpha) *
                             constraints[-degree:, : degree] - self.toprightconstraint
                             [: degree, : degree][-1:: -1, :]) ** 2) / len(P[0, :]))
                        constrained_residual_norm += bnderr

                    if len(self.bottomleftconstraint):

                        # Compute the residual for top left interface condition
                        bnderr = np.sqrt(np.sum(
                            (alpha * P[: degree, -degree:] + (1 - alpha) *
                             constraints[: degree, -degree:] - self.bottomleftconstraint
                             [: degree, : degree].T) ** 2) / len(P[0, :]))
                        constrained_residual_norm += bnderr

                    if len(self.bottomrightconstraint):

                        # Compute the residual for top left interface condition
                        bnderr = np.sqrt(np.sum(
                            (alpha * P[-degree:, -degree:] + (1 - alpha) * constraints
                             [-degree:, -degree:] - self.bottomrightconstraint[: degree, : degree]
                             [-1:: -1, :]) ** 2) / len(P[0, :]))
                        constrained_residual_norm += bnderr

            # compute the net residual norm that includes the decoded error in the current subdomain and the penalized
            # constraint error of solution on the subdomain interfaces
            net_residual_norm = decoded_penalty * decoded_residual_norm + bc_penalty * constrained_residual_norm

            if verbose:
                print('Residual = ', net_residual_norm, ' and res_dec = ', decoded_residual_norm,
                      ' and constraint = ', constrained_residual_norm)
                # print('Constraint errors = ', ltn, rtn, tpn, btn, constrained_residual_norm)

            return net_residual_norm

        # Compute the residual as sum of two components
        # 1. The decoded error evaluated at P
        # 2. A penalized domain boundary constraint component
        # @jit(nopython=True, parallel=False)
        def residual_vec(Pin, verbose=False):

            P = Pin.reshape(W.shape)
            alpha = 1
            bc_penalty = 1e0

            # Residuals are in the decoded space - so direct way to constrain the boundary data
            # decoded = decode(P, W, self.Nu, self.Nv)
            decoded = np.matmul(np.matmul(decodeOpX, P), decodeOpY.T)
            residual_decoded = np.abs(decoded - self.zl)/zRange

            def decode1D(dir, P):

                W = np.ones(P.shape)
                if dir == 'x':  # x-direction
                    # dec = (self.Nu*W)/(np.sum(self.Nu*W, axis=1)[:, np.newaxis]) @ P
                    dec = decodeOpX @ P
                else:  # y-direction
                    # dec = (self.Nv*W)/(np.sum(self.Nv*W, axis=1)[:, np.newaxis]) @ P
                    dec = decodeOpY @ P

                return dec

            constrained_residual_norm = 0
            if constraints is not None and len(constraints) > 0 and constrainInterfaces:
                if len(left) > 1:

                    # Compute the residual for left interface condition
                    # constrained_residual_norm += np.sum( ( P[0,:] - (leftdata[:]) )**2 ) / len(leftdata[:,0])
                    # ltn = np.sqrt(
                    #     (np.sum((P[0, :] - 0.5 * (constraints[0, :] + left[:, degree-1])) ** 2) / len(P[0, :])))
                    # ltn = np.sqrt(
                    #     np.sum((alpha * P[0, :] + (1 - alpha) * constraints[0, :] - left[:, degree - 1]) ** 2) / len(P[0, :]))
                    # print('Left shapes: ', residual_decoded[0, :].shape, P[0,
                    #                                                        :].shape, constraints[0, :].shape, left[:, degree - 1].shape)

                    constraintSol = 0.5 * (alpha * P[0, :] + (1 - alpha)
                                           * constraints[0, :] + left[:, degree - 1])
                    leftConstraintRes = (decode1D('y', constraintSol) - self.zl[0, :])
                    residual_decoded[0, :] -= bc_penalty * leftConstraintRes

                    # print(idom, ': Left constrained norm: ', ltn)
                    # print('Left Shapes knots: ', constrained_residual_norm, len(left[:]), len(
                    #     knotsAllV), self.yl.shape, left, (constraints[-1, :]), P[0, :], P[-1, :])

                if len(right) > 1:

                    # Compute the residual for right interface condition
                    # constrained_residual_norm += np.sum( ( P[-1,:] - (rightdata[:]) )**2 ) / len(rightdata[:,0])
                    # rtn = np.sqrt(
                    #     (np.sum((P[-1, :] - 0.5 * (constraints[-1, :] + right[:, degree-1]))**2) / len(right[:, -1])))
                    # rtn = np.sqrt(
                    #     np.sum((alpha * P[-1, :] + (1 - alpha) * constraints[-1, :] - right[:, degree - 1]) ** 2) /
                    #     len(P[-1, :]))

                    constraintSol = 0.5 * (alpha * P[-1, :] + (1 - alpha)
                                           * constraints[-1, :] + right[:, 1].T)
                    rightConstraintRes = (decode1D('y', constraintSol) - self.zl[-1, :])
                    residual_decoded[-1, :] -= bc_penalty * rightConstraintRes

                    # print(idom, ': Right constrained norm: ', rtn)
                    # print('Right Shapes knots: ', constrained_residual_norm, len(right[:]), len(
                    #     knotsAllV), self.yl.shape, right, (constraints[0, :]), P[0, :], P[-1, :])

                if len(top) > 1:

                    # Compute the residual for top interface condition
                    # constrained_residual_norm += np.sum( ( P[:,-1] - (topdata[:]) )**2 ) / len(topdata[:,0])
                    # tpn = np.sqrt(
                    #     (np.sum((P[:, -1] - 0.5 * (constraints[:, -1] + top[:, degree-1])) ** 2) / len(top[:, 0])))
                    # tpn = np.sqrt(
                    #     np.sum((alpha * P[:, -1] + (1 - alpha) * constraints[:, -1] - top[:, degree - 1]) ** 2) /
                    #     len(P[:, -1]))

                    constraintSol = 0.5 * (alpha * P[:, -1] + (1 - alpha)
                                           * constraints[:, -1] + top[:, degree - 1])
                    topConstraintRes = (decode1D('x', constraintSol) - self.zl[:, 0])
                    residual_decoded[:, -1] -= bc_penalty * topConstraintRes

                    # print(idom, ': Top constrained norm: ', tpn)
                    # print('Top: ', constrained_residual_norm, P[:, -1],
                    #       constraints[:, -1], P[:, 0], constraints[:, 0], top[:])

                if len(bottom) > 1:

                    # Compute the residual for bottom interface condition
                    # constrained_residual_norm += np.sum( ( P[:,0] - (bottomdata[:]) )**2 ) / len(bottomdata[:,0])
                    # btn = np.sqrt(
                    #     np.sum((P[:, 0] - 0.5 * (constraints[:, 0] + bottom[:, degree-1]))**2) / len(bottom[:, 0]))
                    # btn = np.sqrt(
                    #     np.sum((alpha * P[:, 0] + (1 - alpha) * constraints[:, 0] - bottom[:, degree - 1]) ** 2) /
                    #     len(P[:, 0]))

                    constraintSol = 0.5 * (alpha * P[:, 0] + (1 - alpha)
                                           * constraints[:, 0] + bottom[:, degree - 1])
                    dec = decode1D('x', constraintSol)
                    bottomConstraintRes = (dec - self.zl[:, -1])
                    # print('Bottom: ',  np.linalg.norm(dec - self.zl[:, -1]),
                    #       np.linalg.norm(dec - self.zl[:, 0]),
                    #       #np.linalg.norm(dec - self.zl[-1, :]),
                    #       #np.linalg.norm(dec - self.zl[0, :])
                    #       )
                    residual_decoded[:, 0] -= bc_penalty * bottomConstraintRes

                    # print(idom, ': Bottom constrained norm: ', btn)
                    # print(
                    #     'Bottom: ', constrained_residual_norm, P[:, -1],
                    #     constraints[:, -1],
                    #     P[:, 0],
                    #     constraints[:, 0],
                    #     bottom[:])

            # decoded = np.matmul(np.matmul(decodeOpX, P), decodeOpY.T)
            # residual_decoded_cpts = np.matmul(np.matmul(decodeOpY.T, residual_decoded), decodeOpX)
            residual_decoded_cpts = np.matmul(np.matmul(decodeOpX.T, residual_decoded), decodeOpY)
            residual_vec_encoded = residual_decoded_cpts.reshape(
                residual_decoded_cpts.shape[0]*residual_decoded_cpts.shape[1], )

            if verbose:
                print('Residual = ', net_residual_norm, ' and res_dec = ', decoded_residual_norm,
                      ' and constraint = ', constrained_residual_norm)
                # print('Constraint errors = ', ltn, rtn, tpn, btn, constrained_residual_norm)

            return residual_vec_encoded

        def print_iterate(P, res=None):
            if res is None:
                res = residual(P, verbose=True)
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

            jacobian = egrad(residual)(P)
#             jacobian = jacobian_const
            return jacobian

        # Use automatic-differentiation to compute the Jacobian value for minimizer
        def hessian(P):
            return egrad(hessian)(P)

        # if constraintsAll is not None:
        #    jacobian_const = egrad(residual)(initSol)

        # Now invoke the solver and compute the constrained solution
        if constraints is None and not alwaysSolveConstrained:
            solution, _ = lsqFit(self.Nu, self.Nv, W, self.zl)
            solution = solution.reshape(W.shape)
        else:

            print('Initial calculation')
            print_iterate(initSol)
            if enforceBounds:
                bnds = np.tensordot(np.ones(initSol.shape[0]*initSol.shape[1]),
                                    np.array([self.zl.min(), self.zl.max()]), axes=0)
            else:
                bnds = None
            print('Using optimization solver = ', solverScheme)
            # Solver options: https://docs.scipy.org/doc/scipy-0.13.0/reference/generated/scipy.optimize.show_options.html
            if solverScheme == 'L-BFGS-B':
                res = minimize(residual, x0=initSol, method=solverScheme,  # 'SLSQP', #'L-BFGS-B', #'TNC',
                               bounds=bnds,
                               jac=jacobian,
                               callback=print_iterate,
                               tol=self.globalTolerance,
                               options={'disp': False, 'ftol': maxRelErr, 'gtol': self.globalTolerance, 'maxiter': solverMaxIter})
            elif solverScheme == 'CG':
                res = minimize(residual, x0=initSol, method=solverScheme,  # Unbounded - can blow up
                               jac=jacobian,
                               bounds=bnds,
                               callback=print_iterate,
                               tol=self.globalTolerance,
                               options={'disp': False, 'ftol': self.globalTolerance, 'norm': 2, 'maxiter': solverMaxIter})
            elif solverScheme == 'SLSQP':
                res = minimize(residual, x0=initSol, method=solverScheme,  # 'SLSQP', #'L-BFGS-B', #'TNC',
                               bounds=bnds,
                               callback=print_iterate,
                               tol=self.globalTolerance,
                               options={'disp': False, 'ftol': maxRelErr, 'maxiter': solverMaxIter})
            elif solverScheme == 'Newton-CG' or solverScheme == 'TNC':
                res = minimize(residual, x0=initSol, method=solverScheme,  # 'SLSQP', #'L-BFGS-B', #'TNC',
                               jac=jacobian,
                               bounds=bnds,
                               # jac=egrad(residual)(initSol),
                               callback=print_iterate,
                               tol=self.globalTolerance,
                               options={'disp': False, 'eps': self.globalTolerance, 'maxiter': solverMaxIter})
            elif solverScheme == 'trust-krylov' or solverScheme == 'trust-ncg':
                res = minimize(residual, x0=initSol, method=solverScheme,  # 'SLSQP', #'L-BFGS-B', #'TNC',
                               jac=jacobian,
                               hess=hessian,
                               bounds=bnds,
                               callback=print_iterate,
                               tol=self.globalTolerance,
                               options={'inexact': True})
            elif solverScheme == 'trf' or solverScheme == 'lm' or solverScheme == 'dogbox':
                res = scipy.optimize.least_squares(residual_vec, x0=initSol.reshape(initSol.shape[0]*initSol.shape[1]), method=solverScheme,  # {‘trf’, ‘dogbox’, ‘lm’},
                                                   #    jac=jacobian,
                                                   #    hess=hessian,
                                                   #    bounds=bnds,
                                                   #    callback=print_iterate,
                                                   gtol=self.globalTolerance,
                                                   verbose=2,
                                                   tr_solver='lsmr',  # {None, ‘exact’, ‘lsmr’},
                                                   )
            else:
                optimizer_options = {'disp': False,
                                     #  'ftol': 1e-5,
                                     #  'ftol': self.globalTolerance,
                                     'maxiter': solverMaxIter,
                                     # {‘lgmres’, ‘gmres’, ‘bicgstab’, ‘cgs’, ‘minres’}
                                     'jac_options': {'method': 'lgmres'}
                                     }
                # jacobian_const = egrad(residual)(initSol)
                res = scipy.optimize.root(residual_vec, x0=initSol.reshape(initSol.shape[0]*initSol.shape[1]),
                                          method=solverScheme,  # 'krylov', 'lm'
                                          # jac=jacobian_vec,
                                          jac=False,
                                          callback=print_iterate,
                                          tol=self.globalTolerance,
                                          options=optimizer_options)

            print('[%d] : %s' % (idom, res.message))
            solution = res.x.reshape(W.shape)

            # solution = res.reshape(W.shape)

        return solution

    def interpolate_knots(self, knots, tnew):

        r = 1

        # print('Original interpolation shapes: ', Pnew.shape, knots, tnew)
        # For all entries that are missing in self.knotsAdaptive, call castleDeJau
        # and recompute control points one by one
        for knot in tnew:
            # knotInd = np.searchsorted(knots, knot)
            found = False
            # let us do a linear search
            for k in knots:
                if abs(k - knot) < 1e-10:
                    found = True
                    break

            if not found:
                knotInd = np.searchsorted(knots, knot)
                knots = np.insert(knots, knotInd, knot)

        return knots

    def interpolate_private(self, P, knots, tnew):

        r = 1
        Pnew = np.copy(P)
        W = np.ones(Pnew.shape)

        # print('Original interpolation shapes: ', Pnew.shape, knots, tnew)
        # For all entries that are missing in self.knotsAdaptive, call castleDeJau
        # and recompute control points one by one
        for knot in tnew:
            # knotInd = np.searchsorted(knots, knot)
            found = False
            # let us do a linear search
            for k in knots:
                if abs(k - knot) < 1e-10:
                    found = True
                    break

            if not found:
                knotInd = np.searchsorted(knots, knot)
                # if Pnew.shape[0] == knotInd:
                # print('Knot index is same as length of control points')

                # print('New interpolation shapes: ', Pnew.shape, knots.shape, ' before inserting ', knot, ' at ', knotInd)
                Pnew, W = deCasteljau1D(Pnew, W, knots, knot, knotInd-1, r)
                knots = np.insert(knots, knotInd, knot)

        # cplocCtrPt = getControlPoints(knots, degree) * (Dmax - Dmin) + Dmin
        # coeffs_xy = getControlPoints(knots, degree)

        # return Pnew, knots, W
        return Pnew, knots

    def interpolate(self, P, knots, tnew):

        r = 1
        knotsnew = self.interpolate_knots(knots, tnew)
        # print (knotsnew.shape[0]-degree-1, useDerivativeConstraints+1)
        Pnew = np.zeros([knotsnew.shape[0]-degree-1, useDerivativeConstraints+1])
        for col in range(useDerivativeConstraints+1):
            Pnew[:, col], knotsnew = self.interpolate_private(P[:, col], knots, tnew)

        return Pnew, knotsnew

    def interpolate_spline(self, pold, coeffs_xy, ncoeffs_xy):

        # InterpCp = interp1d(coeffs_x, self.pAdaptive, kind=interpOrder)
        InterpCp = Rbf(coeffs_xy, pold, function=interpOrder)

        Pnew = InterpCp(ncoeffs_xy)

        return Pnew

    def flatten(self, x):
        import collections
        if isinstance(x, collections.Iterable):
            return [a for i in x for a in self.flatten(i)]
        else:
            return [x]

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
        if len(self.decodedAdaptiveOld):

            # Let us compute the relative change in the solution between current and previous iteration
            iterateChangeVec = (self.decodedAdaptive - self.decodedAdaptiveOld)
            errorMetricsSubDomL2 = np.linalg.norm(iterateChangeVec.reshape(
                iterateChangeVec.shape[0] * iterateChangeVec.shape[1]),
                ord=2) / np.linalg.norm(self.pAdaptive, ord=2)
            errorMetricsSubDomLinf = np.linalg.norm(iterateChangeVec.reshape(
                iterateChangeVec.shape[0] * iterateChangeVec.shape[1]),
                ord=np.inf) / np.linalg.norm(self.pAdaptive, ord=np.inf)

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

    # @profile
    def adaptive(self, iSubDom, xl, yl, zl, strategy='extend', weighted=False,
                 r=1, MAX_ERR=1e-3, MAX_ITER=5,
                 split_all=True,
                 decodedError=None):

        from scipy.interpolate import Rbf

        k = []
        r = min(r, degree)  # multiplicity can not be larger than degree
        reuseE = (decodedError) if decodedError is not None else None

        splitIndeces = []
        r = min(r, degree)  # multiplicity can not be larger than degree

        Tu = np.copy(self.knotsAdaptiveU)
        Tv = np.copy(self.knotsAdaptiveV)
        P = np.copy(self.pAdaptive)
        W = np.copy(self.WAdaptive)

        globalIterationNum = 0

        # self.Nu = basis(self.U[np.newaxis, :], degree, Tu[:, np.newaxis]).T
        # self.Nv = basis(self.V[np.newaxis, :], degree, Tv[:, np.newaxis]).T
        self.compute_basis(degree, Tu, Tv)

        if ((np.sum(np.abs(P)) < 1e-14 and len(P) > 0) or len(P) == 0) and self.outerIteration == 0:
            print(iSubDom, " - Applying the unconstrained solver.")
            P = self.LSQFit_NonlinearOptimize(iSubDom, W, degree, None)
#             P,_ = lsqFit(self.Nu, self.Nv, W, self.zl)
            decodedError = decode(P, W, self.Nu, self.Nv)  # - self.zl
            MAX_ITER = 0
        else:
            if disableAdaptivity:
                MAX_ITER = 1

        iteration = 0
        interface_constraints_obj = dict()
        for iteration in range(MAX_ITER):
            fev = 0

            if disableAdaptivity:

                interface_constraints_obj['P'] = P
                interface_constraints_obj['W'] = W
                interface_constraints_obj['Tu'] = Tu
                interface_constraints_obj['Tv'] = Tv

                print(iSubDom, " - Applying the constrained solver.")
                P = self.LSQFit_NonlinearOptimize(iSubDom, W, degree, interface_constraints_obj)

            else:
                self.adaptiveIterationNum += 1
                Tunew, Tvnew, Usplits, Vsplits, E, maxE = knotRefine(P, W, Tu, Tv, self.Nu, self.Nv, self.U, self.V, self.zl, r,
                                                                     # reuseE=reuseE,
                                                                     MAX_ERR=MAX_ERR,
                                                                     find_all=split_all)

                # maxE = LA.norm(E.reshape(E.shape[0]*E.shape[1]),np.inf)
                print("Adaptive iteration ", iteration+1, ", at ", maxE, " maxError")

                if (len(Tu) == len(Tunew) and len(Tv) == len(Tvnew)) or \
                        len(Tunew)-degree-1 > self.nPointsPerSubDX or \
                        len(Tvnew)-degree-1 > self.nPointsPerSubDY:
                    print("Nothing more to refine: E = ", E)
                    reuseE = np.copy(E)
                    Tunew = np.copy(Tu)
                    Tvnew = np.copy(Tv)
                    # break

                if len(Usplits) == 0 or len(Vsplits) == 0:
                    if (maxE > MAX_ERR):
                        print("Max error hit: E = ", E)
                        reuseE = np.copy(E)
                        continue

                if(maxE <= MAX_ERR):
                    k = [-1, -1]
                    print("Adaptive done in %d iterations at %e maxError" % (iteration+1, maxE))
                    break

                nControlPointsPrev = self.nControlPoints
                self.nControlPoints = np.array([Tunew.shape[0]-1-degree, Tvnew.shape[0]-1-degree])
                self.nControlPointSpans = self.nControlPoints - 1
                self.nInternalKnotSpans = self.nControlPoints - degree

                print('-- Spatial Adaptivity: Before = ', nControlPointsPrev, ', After = ', self.nControlPoints)

                if strategy == 'extend' and not split_all:  # only use when coupled with a solver
                    k = [Usplits.pop(), Vsplits.pop()]  # if reuseE is None else [-1,-1]
                    u = np.array([Tunew[k[0]+1], Tvnew[k[1]+1]])
                    print(iSubDom, " - Applying the deCasteljau solver")
                    P, W = deCasteljau2D(P, W, Tu, Tv, u, k, r)

                elif strategy == 'reset':

                    Tunew = np.sort(Tunew)
                    Tvnew = np.sort(Tvnew)

                    W = np.ones(self.nControlPoints)

                    # Need to compute a projection from [P, Tu, Tv] to [Pnew, Tunew, Tvnew]
                    coeffs_x1 = getControlPoints(Tu, degree)
                    coeffs_y1 = getControlPoints(Tv, degree)
                    Xi, Yi = np.meshgrid(coeffs_x1, coeffs_y1)

                    coeffs_x2 = getControlPoints(Tunew, degree)
                    coeffs_y2 = getControlPoints(Tvnew, degree)
                    Xi2, Yi2 = np.meshgrid(coeffs_x2, coeffs_y2)

                    rbfi = Rbf(Yi, Xi, P, function='cubic')  # radial basis function interpolator instance
                    Pnew = rbfi(Yi2, Xi2).reshape(W.shape)   # interpolated values

                    interface_constraints_obj['P'] = Pnew[:, :]
                    interface_constraints_obj['W'] = W[:, :]
                    interface_constraints_obj['Tu'] = Tunew[:]
                    interface_constraints_obj['Tv'] = Tvnew[:]

                    # self.Nu = basis(self.U[np.newaxis, :], degree, Tunew[:, np.newaxis]).T
                    # self.Nv = basis(self.V[np.newaxis, :], degree, Tvnew[:, np.newaxis]).T
                    self.compute_basis(degree, Tunew, Tvnew)
                    print('shapes: ', W.shape, self.Nu.shape, self.Nv.shape)

                    print(iSubDom, " - Applying the adaptive constrained solver.")
                    P = self.LSQFit_NonlinearOptimize(iSubDom, W, degree, interface_constraints_obj)

            if not disableAdaptivity:
                Tu = np.copy(Tunew)
                Tv = np.copy(Tvnew)

            decoded = decode(P, W, self.Nu, self.Nv)
            decodedError = np.abs(np.array(decoded-self.zl)) / zRange

            reuseE = (decodedError.reshape(decodedError.shape[0]*decodedError.shape[1]))
            print("\tDecoded error: ", np.sqrt(np.sum(reuseE**2)/len(reuseE)))

            iteration += 1

        return P, W, Tu, Tv, np.array([k]), decodedError

    # @profile
    def solve_adaptive(self, cp):

        global isConverged
        if isConverged[cp.gid()] == 1:
            print(cp.gid()+1, ' subdomain has already converged to its solution. Skipping solve ...')
            return

        # Subdomain ID: iSubDom = cp.gid()+1
        newSolve = False
        if (np.sum(np.abs(self.pAdaptive)) < 1e-14 and len(self.pAdaptive) > 0) or len(self.pAdaptive) == 0:
            newSolve = True

        if not newSolve:

            self.nControlPointSpans = self.nControlPoints - 1
            self.nInternalKnotSpans = self.nControlPoints - degree

        print("Subdomain -- ", cp.gid()+1)

        # Let do the recursive iterations
        # Use the previous MAK solver solution as initial guess; Could do something clever later

        # Invoke the adaptive fitting routine for this subdomain
        adaptiveErr = None
        # self.globalTolerance = 1e-3 * 1e-3**self.adaptiveIterationNum
        self.pAdaptive, self.WAdaptive, self.knotsAdaptiveU, self.knotsAdaptiveV, kv, adaptiveErr = self.adaptive(cp.gid()+1,
                                                                                                                  self.xl, self.yl, self.zl,
                                                                                                                  strategy=AdaptiveStrategy, weighted=False,
                                                                                                                  r=1, MAX_ERR=maxAbsErr, MAX_ITER=maxAdaptIter,  # MAX_ERR=maxAdaptErr,
                                                                                                                  split_all=True,
                                                                                                                  decodedError=adaptiveErr)

        if len(self.pAdaptiveHistory) == 0:
            if useAitken:
                self.pAdaptiveHistory = np.zeros((self.pAdaptive.shape[0]*self.pAdaptive.shape[1], 3))
            else:
                self.pAdaptiveHistory = np.zeros((self.pAdaptive.shape[0]*self.pAdaptive.shape[1], nWynnEWork))

        # Update the local decoded data
        self.decodedAdaptiveOld = np.copy(self.decodedAdaptive)
        self.decodedAdaptive = decode(self.pAdaptive, self.WAdaptive, self.Nu, self.Nv)

        # E = (self.decodedAdaptive[self.corebounds[0]:self.corebounds[1]] - self.zl[self.corebounds[0]:self.corebounds[1]])/zRange
        E = (self.decodedAdaptive[self.corebounds[0]: self.corebounds[1],
                                  self.corebounds[2]: self.corebounds[3]] - self.zl
             [self.corebounds[0]: self.corebounds[1],
              self.corebounds[2]: self.corebounds[3]]) / zRange
        E = (E.reshape(E.shape[0]*E.shape[1]))
        LinfErr = np.linalg.norm(E, ord=np.inf)
        L2Err = np.sqrt(np.sum(E**2)/len(E))

        self.decodederrors[0] = L2Err
        self.decodederrors[1] = LinfErr

        print("Subdomain -- ", cp.gid()+1, ": L2 error: ", L2Err, ", Linf error: ", LinfErr)

#         print("adaptiveErr: ", self.pAdaptive.shape, self.WAdaptive.shape, self.zl.shape, self.Nu.shape, self.Nv.shape)
        # errorMAK = L2LinfErrors(self.pAdaptive, self.WAdaptive, self.zl, degree, self.Nu, self.Nv)


#########

# @profile
# def execute_asm_loop():
# read_problem_parameters()

# Initialize DIY
commWorld = diy.mpi.MPIComm()           # world
mc2 = diy.Master(commWorld)         # master
domain_control = diy.DiscreteBounds([0, 0], [len(x)-1, len(y)-1])

# Routine to recursively add a block and associated data to it


def add_input_control_block2(gid, core, bounds, domain, link):
    print("Subdomain %d: " % gid, core, bounds, domain)
    minb = bounds.min
    maxb = bounds.max

    xlocal = x[minb[0]:maxb[0]+1]
    ylocal = y[minb[1]:maxb[1]+1]
    zlocal = z[minb[0]:maxb[0]+1, minb[1]:maxb[1]+1]

    # print("Subdomain %d: " % gid, minb[0], minb[1], maxb[0], maxb[1], z.shape, zlocal.shape)
    mc2.add(gid, InputControlBlock(gid, nControlPointsInput, core, bounds, xlocal, ylocal, zlocal), link)


# TODO: If working in parallel with MPI or DIY, do a global reduce here
errors = np.zeros([nASMIterations+1, 2])  # Store L2, Linf errors as function of iteration

# Let us initialize DIY and setup the problem
share_face = [True, True]
wrap = [False, False]
# ghosts = [useDerivativeConstraints,useDerivativeConstraints]
ghosts = [overlapData, overlapData]
divisions = [nSubDomainsX, nSubDomainsY]

d_control = diy.DiscreteDecomposer(2, domain_control, nSubDomainsX*nSubDomainsY, share_face, wrap, ghosts, divisions)
a_control2 = diy.ContiguousAssigner(nprocs, nSubDomainsX*nSubDomainsY)

d_control.decompose(rank, a_control2, add_input_control_block2)

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

send_receive_all()

if not fullyPinned:
    mc2.foreach(InputControlBlock.augment_spans)
    if augmentSpanSpace > 0:
        mc2.foreach(InputControlBlock.augment_inputdata)

del x, y, z

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

    if showplot:

        figHnd = plt.figure()
        figHndErr = plt.figure()

        mc2.foreach(lambda icb, cp: InputControlBlock.set_fig_handles(
            icb, cp, figHnd, figHndErr, "%d-%d" % (cp.gid(), iterIdx)))

        # Now let us draw the data from each subdomain
        mc2.foreach(InputControlBlock.plot_control)
        mc2.foreach(InputControlBlock.plot_error)

        # figHnd.savefig("decoded-data-%d-%d.png"%(iterIdx))   # save the figure to file
        # figHndErr.savefig("error-data-%d-%d.png"%(iterIdx))   # save the figure to file

    # commW.Barrier()
    sys.stdout.flush()

    if useVTKOutput:

        mc2.foreach(lambda icb, cp: InputControlBlock.set_fig_handles(
            icb, cp, None, None, "%d-%d" % (cp.gid(), iterIdx)))
        mc2.foreach(InputControlBlock.output_vtk)

        if rank == 0:
            WritePVTKFile(iterIdx)

    if isASMConverged == nSubDomains:
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
avgL2err = np.sqrt(avgL2err/nSubDomains)
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


### If we want to trace memory footprint ###

def display_top(snapshot, key_type='lineno', limit=3):
    import os
    import linecache
    snapshot = snapshot.filter_traces((
        tracemalloc.Filter(False, "<frozen importlib._bootstrap>"),
        tracemalloc.Filter(False, "<unknown>"),
    ))
    top_stats = snapshot.statistics(key_type)

    print("Top %s lines" % limit)
    for index, stat in enumerate(top_stats[:limit], 1):
        frame = stat.traceback[0]
        # replace "/path/to/module/file.py" with "module/file.py"
        filename = os.sep.join(frame.filename.split(os.sep)[-2:])
        print("#%s: %s:%s: %.1f KiB"
              % (index, filename, frame.lineno, stat.size / 1024))
        line = linecache.getline(frame.filename, frame.lineno).strip()
        if line:
            print('    %s' % line)

    other = top_stats[limit:]
    if other:
        size = sum(stat.size for stat in other)
        print("%s other: %.1f KiB" % (len(other), size / 1024))
    total = sum(stat.size for stat in top_stats)
    print("Total allocated size: %.1f KiB" % (total / 1024))

# snapshot = tracemalloc.take_snapshot()
# display_top(snapshot)

# ---------------- END MAIN FUNCTION -----------------

# if __name__ == '__main__':
# execute_asm_loop()

# %%
