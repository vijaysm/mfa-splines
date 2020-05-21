
# coding: utf-8

# In[2]:


# coding: utf-8
# get_ipython().magic(u'matplotlib notebook')
# %matplotlib notebook

import sys, getopt, math
import autograd.numpy as np
# import numpy as np
import scipy
from mpi4py import MPI
import diy

import mpl_toolkits.mplot3d.axes3d as axes3d
from matplotlib import pyplot as plt
from matplotlib import cm
from scipy.interpolate import Rbf
from pyevtk.hl import gridToVTK 

plt.style.use(['seaborn-whitegrid'])
# plt.style.use(['ggplot'])
# plt.style.use(['classic'])
params = {"ytick.color" : "b",
          "xtick.color" : "b",
          "axes.labelcolor" : "b",
          "axes.edgecolor" : "b"}
plt.rcParams.update(params)
useVTKOutput = True

# --- set problem input parameters here ---
nSubDomainsX   = 2
nSubDomainsY   = 2
degree         = 3
problem        = 0
verbose        = False
showplot       = False

# ------------------------------------------
# Solver parameters
useAdditiveSchwartz = True
useDerivativeConstraints = 0

# L-BFGS-B: 0m33.795s, TNC: 0m20.516s (wrong), CG: 0m52.079s (large errors, unbounded), SLSQP: 2m40.323s (dissipative), Newton-CG: 8 mins and no progress, trust-krylov: hessian required
# L-BFGS-B: 1m7.170s, TNC: 0m35.488s
#                      0        1     2       3         4              5
solverMethods  = ['L-BFGS-B', 'TNC', 'CG', 'SLSQP', 'Newton-CG', 'trust-krylov', 'krylov']
solverScheme   = solverMethods[0]
solverMaxIter  = 25
nASMIterations = 4

projectData    = True
enforceBounds  = False
alwaysSolveConstrained = False
constrainInterfaces = True

useDeCastelJau = True
useDecodedConstraints = False
disableAdaptivity = False
variableResolution = False

maxAbsErr      = 1e-4
maxRelErr      = 1e-10
maxAdaptIter   = 5
# AdaptiveStrategy = 'extend'
AdaptiveStrategy = 'reset'
# ------------------------------------------

def plot3D(fig, Z, x=None, y=None):
    if x is None:
        x = np.arange(Z.shape[0])
    if y is None:
        y = np.arange(Z.shape[1])
    X, Y = np.meshgrid(x, y)
    print("Plot shapes: [x, y, z] = ", x.shape, y.shape, Z.shape, X.shape, Y.shape)

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
        print(X.shape, Y.shape, Zi.shape, Z.shape)
        gridToVTK("./structured", X, Y, Zi, pointData = {"solution" : Z})

    # plt.show()


# Initialize DIY
commW = MPI.COMM_WORLD
nprocs = commW.size
rank = commW.rank
isASMConverged = 0

if rank == 0: print('Argument List:', str(sys.argv))

##########################
# Parse command line overrides if any
##
argv=sys.argv[1:]
def usage():
  print(sys.argv[0], '-p <problem> -n <nsubdomains> -nx <nsubdomains_x> -ny <nsubdomains_y> -d <degree> -c <controlpoints> -o <overlapData> -a <nASMIterations>')
  sys.exit(2)
try:
  opts, args = getopt.getopt(argv,"hp:n:d:c:o:a:",["problem=","nsubdomains=","degree=","controlpoints=","overlap=","nasm="])
except getopt.GetoptError:
  usage()

for opt, arg in opts:
  if opt == '-h':
      usage()
  elif opt in ("-n", "--nsubdomains"):
      nSubDomainsX = int(arg)
      nSubDomainsY = int(arg)
  elif opt in ("-nx", "--nsubdomainsx"):
      nSubDomainsX = int(arg)
  elif opt in ("-ny", "--nsubdomainsy"):
      nSubDomainsY = int(arg)
  elif opt in ("-d", "--degree"):
      degree = int(arg)
  elif opt in ("-c", "--controlpoints"):
      nControlPointsInput = int(arg)
  elif opt in ("-o", "--overlap"):
      overlapData = int(arg)
  elif opt in ("-p", "--problem"):
      problem = int(arg)
  elif opt in ("-a", "--nasm"):
      nASMIterations = int(arg)

# -------------------------------------

from scipy.ndimage import zoom

if problem == 1:
    nPointsX = 101
    nPointsY = 101
    scale    = 1
    shiftX   = 0.5
    shiftY   = 0.5
    Dmin     = -4.
    Dmax     = 4.

    x = np.linspace(Dmin, Dmax, nPointsX)
    y = np.linspace(Dmin, Dmax, nPointsY)
    X, Y = np.meshgrid(x+shiftX, y+shiftY)
    # z = scale * ( np.sinc(np.sqrt(X**2 + Y**2)) + np.sinc(2*((X-2)**2 + (Y+2)**2)) )
    z = scale * ( np.sinc(2*((X-1)**2 + (Y+1)**2)) )
    # z = X**2 + Y**2
    z = z.T
    nControlPointsInput = 4*np.array([6,6]) #(3*degree + 1) #minimum number of control points

elif problem == 2:
    nPointsX = 101
    nPointsY = 101
    scale    = 1.0
    shiftX   = 0.25
    shiftY   = 0.5
    Dmin     = 0
    Dmax     = math.pi

    x = np.linspace(Dmin, Dmax, nPointsX)
    y = np.linspace(Dmin, Dmax, nPointsY)
    X, Y = np.meshgrid(x+shiftX, y+shiftY)
    z = scale * np.sin(X) * np.sin(Y)
    z = z.T
    # z = scale * np.sin(Y)
    # z = scale * X
    nControlPointsInput = 4*np.array([1,1]) #(3*degree + 1) #minimum number of control points

elif problem == 3:
    z = np.fromfile("data/nek5000.raw", dtype=np.float64).reshape(200,200)
    print ("Nek5000 shape:", z.shape)
    nPointsX = z.shape[0]
    nPointsY = z.shape[1]
    Dmin     = 0.
    Dmax     = 100.
    x = np.linspace(Dmin, Dmax, nPointsX)
    y = np.linspace(Dmin, Dmax, nPointsY)
    nControlPointsInput = 25*np.array([1,1]) #(3*degree + 1) #minimum number of control points

elif problem == 4:

    binFactor = 4.0
    z = np.fromfile("data/s3d_2D.raw", dtype=np.float64).reshape(540,704)
    #z = z[:540,:540]
    #z = zoom(z, 1./binFactor, order=4)
    print ("S3D shape:", z.shape)
    nPointsX = z.shape[0]
    nPointsY = z.shape[1]
    Dmin     = 0.
    Dmax     = 100.
    x = np.linspace(Dmin, Dmax, nPointsX)
    y = np.linspace(Dmin, Dmax, nPointsY)
    nControlPointsInput = 20*np.array([1,1]) #(3*degree + 1) #minimum number of control points

else:
    z = np.fromfile("data/FLDSC_1_1800_3600.dat", dtype=np.float32).reshape(3600, 1800) #
    nPointsX = z.shape[0]
    nPointsY = z.shape[1]
    Dmin     = 0.
    Dmax     = 100.
    x = np.linspace(Dmin, Dmax, nPointsX)
    y = np.linspace(Dmin, Dmax, nPointsY)

    nControlPointsInput = 25*np.array([1,1]) #(3*degree + 1) #minimum number of control points

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
if showplot: fig = plt.figure()
plot3D(fig, z, x, y)

# Let us create a parallel VTK file
def WritePVTKFile(iteration):
    pvtkfile = open("pstructured-mfa-%d.pvts"%(iteration),"w") 

    pvtkfile.write('<?xml version="1.0"?>\n')
    pvtkfile.write('<VTKFile type="PStructuredGrid" version="1.0" byte_order="LittleEndian" header_type="UInt64">\n')
    pvtkfile.write('<PStructuredGrid WholeExtent="%f %f %f %f 0 0" GhostLevel="0">\n'%(xmin, xmax, ymin, ymax))
    pvtkfile.write('\n')
    pvtkfile.write('    <PCellData>\n')
    pvtkfile.write('      <PDataArray type="Float64" Name="solution"/>\n')
    pvtkfile.write('      <PDataArray type="Float64" Name="error"/>\n')
    pvtkfile.write('    </PCellData>\n')
    pvtkfile.write('    <PPoints>\n')
    pvtkfile.write('      <PDataArray type="Float64" Name="points" NumberOfComponents="3"/>\n')
    pvtkfile.write('    </PPoints>\n')

    isubd=0
    dx=(xmax-xmin)/nSubDomainsX
    dy=(ymax-ymin)/nSubDomainsY
    xoff=xmin
    xx=dx
    for ix in range(nSubDomainsX):
	    yoff=ymin
	    yy=dy
	    for iy in range(nSubDomainsY):
		    pvtkfile.write('    <Piece Extent="%f %f %f %f 0 0" Source="structured-%d-%d.vts"/>\n'%(xoff, xx, yoff, yy, isubd, iteration))
		    isubd += 1
		    yoff = yy
		    yy += dy
	    xoff = xx
	    xx += dx
    pvtkfile.write('\n')
    pvtkfile.write('</PStructuredGrid>\n')
    pvtkfile.write('</VTKFile>\n')

    pvtkfile.close()


#------------------------------------

### Print parameter details ###
if rank == 0:
    print('\n==================')
    print('Parameter details')
    print('==================\n')
    print('problem = ', problem, '[0 = sinc, 1 = sine, 2 = Nek5000, 3 = S3D, 4 = CESM]')
    print('Total number of input points: ', nPointsX*nPointsY)
    print('nSubDomains = ', nSubDomainsX * nSubDomainsY)
    print('degree = ', degree)
    print('nControlPoints = ', nControlPointsInput)
    print('nASMIterations = ', nASMIterations)
    # print('overlapData = ', overlapData)
    print('useAdditiveSchwartz = ', useAdditiveSchwartz)
    print('useDerivativeConstraints = ', useDerivativeConstraints)
    print('enforceBounds = ', enforceBounds)
    print('maxAbsErr = ', maxAbsErr)
    print('maxRelErr = ', maxRelErr)
    print('solverMaxIter = ', solverMaxIter)
    print('AdaptiveStrategy = ', AdaptiveStrategy)
    print('solverscheme = ', solverScheme)
    print('\n=================\n')

#------------------------------------


# In[3]:


from autograd import elementwise_grad as egrad
from scipy import linalg, matrix
from scipy.optimize import minimize, linprog
from scipy.linalg import svd

from autograd.numpy import linalg as LA

EPS    = 1e-32
GTOL   = 1e-2
basis  = lambda u,p,T: ((T[:-1]<=u) * (u<=T[1:])).astype(np.float) if p==0 else ((u - T[:-p]) /(T[p:]  -T[:-p]+EPS))[:-1] * basis(u,p-1,T)[:-1] + ((T[p+1:] - u)/(T[p+1:]-T[1:-p]+EPS)) * basis(u,p-1,T)[1:]


def getControlPoints(knots, k):
    nCtrlPts = len(knots) - 1 - k
    cx       = np.zeros(nCtrlPts)
    for i in range(nCtrlPts):
        tsum = 0
        for j in range(1, k + 1):
            tsum += knots[i + j]
        cx[i] = float(tsum) / k
    return cx

def get_decode_operator(W, Nu, Nv):
    Nu = Nu[...,np.newaxis]
    Nv = Nv[:,np.newaxis]
    NN = []
    for ui in range(Nu.shape[0]):
        for vi in range(Nv.shape[0]):
          NN.append(Nu[ui]*Nv[vi])  
    NN = np.array(NN)

    # RN = (NN*W)/(np.sum(NN*W, axis=1)[:,np.newaxis])
    # RN = np.tensordot(NN, W) / np.tensordot(NN, W)
    RN = (NN * W) / np.tensordot(NN, W)

    print('RN.shape = ', RN.shape)

    return RN, [Nu.shape[0], Nv.shape[0]]
    # decoded = np.tensordot(NN, P * W) / np.tensordot(NN, W)
    # return decoded.reshape((Nu.shape[0], Nv.shape[0]))

def decode(P, W, Nu, Nv):
    Nu = Nu[...,np.newaxis]
    Nv = Nv[:,np.newaxis]
    NN = []
    for ui in range(Nu.shape[0]):
        for vi in range(Nv.shape[0]):
          NN.append(Nu[ui]*Nv[vi])  
    NN = np.array(NN)

    decoded = np.tensordot(NN, P * W) / np.tensordot(NN, W)
    return decoded.reshape((Nu.shape[0], Nv.shape[0]))
#     RNx = N[0] * np.sum(W, axis=0)
#     RNx /= np.sum(RNx, axis=1)[:,np.newaxis]
#     RNy = N[1] * np.sum(W, axis=1)
#     RNy /= np.sum(RNy, axis=1)[:,np.newaxis]
#     return np.matmul(np.matmul(RNx, P), RNy.T)

def Error(P, W, z, degree, Nu, Nv):
    Ploc = decode(P, W, Nu, Nv)
    # print('Error shapes: ', Ploc.shape, z.shape)
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
    return np.hstack( ( (P*W)[:,np.newaxis], W[:,np.newaxis]) )

def fromHomogeneous(PW):
    P = PW[:,0]
    W = PW[:,1]
    return P/W, W


def getSplits(T, U):
    t = T[degree:-degree]
    toSplit = [((t[:-1]<=u) * (u<=t[1:])).astype(np.float) for u in U]
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
    for ti,tval in enumerate(T):
        TnewVal = tval.copy()
        Tnew.append(TnewVal);
        if ti in toSplit:
            inc = (float(T[ti+1]-tval)/2.)
            if ((TnewVal+inc)*nPoints) - (TnewVal*nPoints) < 2:
                print ("Not enough input points to split", TnewVal, T[ti+1])
                toSplit.remove(ti)
                continue
            Tnew.append(TnewVal+inc)
    return np.array(Tnew)

def knotInsert(TU, TV, us, splitUs=True, r=1):
    toSplitU = set(getSplits(TU, us[...,0]))
    toSplitV = set(getSplits(TV, us[...,1]))
    return splitT(TU, toSplitU, nPointsX), splitT(TV, toSplitV, nPointsY), toSplitU, toSplitV
               
def knotRefine(P, W, TU, TV, Nu, Nv, U, V, zl, r=1, find_all=True, MAX_ERR = 1e-2, reuseE=None):
    if reuseE is None:
        NSE = np.abs(Error(P, W, zl, degree, Nu, Nv))/zRange
    else:
        NSE = np.abs(reuseE)
    # NMSE = NSE.mean()
    NMSE = LA.norm(NSE, 2)
    if(NMSE<=MAX_ERR):
        return TU, TV, [], [], NSE, NMSE
    if find_all:
        rows,cols = np.where( NSE>=MAX_ERR )
        us = np.vstack( (U[rows] , V[cols]) ).T
    else:
        maxRow,maxCol = np.unravel_index(np.argmax(NSE), NSE.shape)
        NSE[maxRow,maxCol] = 0
        us = np.array([[U[maxRow],V[maxCol]]])
    TUnew,TVnew,Usplits,Vsplits=knotInsert(TU, TV, us, splitUs=find_all, r=r)
    return TUnew,TVnew,Usplits,Vsplits,NSE,NMSE

def insert_knot_u(P, W, T, u, k, degree, r=1, s=0):
    # Algorithm A5.3
    NP = np.array(P.shape)
    Q = np.zeros( (NP[0]+r, NP[1], 2) )
    # Initialize a local array of length p + 1
    R = np.zeros( (degree+1, 2) )
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
    Q = np.zeros( (NP[0], NP[1]+r, 2) )
    # Initialize a local array of length p + 1
    R = np.zeros( (degree+1, 2) )
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
    P,W= fromHomogeneous(Qu)
    Q  = insert_knot_v(P, W, TV, u[1], k[1], degree, r)
    P,W = fromHomogeneous(Q)
    return P,W


def deCasteljau1D(P, W, T, u, k, r=1):
    NP = len(P)
    Qnew = np.zeros( (NP+r, P.ndim+1) )
    Rw = np.zeros( (degree+1, P.ndim+1) )
    PW = toHomogeneous(P, W)

    mp = NP+degree+1
    nq = len(Qnew)

    Qnew[:k-degree+1] = PW[:k-degree+1]
    Qnew[k+r:NP+1+r] = PW[k:NP+1]
    Rw[:degree+1] = PW[k-degree:k+1]

    for j in range(1,r+1):
        L = k-degree+j
        for i in range(degree-j+1):
            alpha = (u-T[L+i])/(T[i+k+1]-T[L+i])
            Rw[i] = alpha*Rw[i+1] + (1.0-alpha)*Rw[i]

        Qnew[L] = Rw[0]
        Qnew[k+r-j] = Rw[degree-j]
        Qnew[L+1:k] = Rw[1:k-L]

    P,W = fromHomogeneous(Qnew)
    return P,W

def L2LinfErrors(P, W, z, degree, Nu, Nv):

    E = Error(P, W, z, degree, Nu, Nv)
    LinfErr = np.abs(E).max()/zRange
    L2Err = math.sqrt(np.sum(E**2)/len(E))
    return [L2Err, LinfErr]

def lsqFit(Nu, Nv, W, z, use_cho=True):
    RNx = Nu * np.sum(W, axis=1)
    RNx /= np.sum(RNx, axis=1)[:,np.newaxis]
    RNy = Nv * np.sum(W, axis=0)
    RNy /= np.sum(RNy, axis=1)[:,np.newaxis]
    if use_cho:
        X = linalg.cho_solve(linalg.cho_factor(np.matmul(RNx.T,RNx)), RNx.T)
        Y = linalg.cho_solve(linalg.cho_factor(np.matmul(RNy.T,RNy)), RNy.T)
        zY = np.matmul(z, Y.T)
        return np.matmul(X, zY), []
    else:
        NTNxInv = np.linalg.inv(np.matmul(RNx.T,RNx))
        NTNyInv = np.linalg.inv(np.matmul(RNy.T,RNy))
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

    def __init__(self, bid, nCPi, xb, xl, yl, zl):
        if projectData and variableResolution:
            nCP = nCPi * (bid+1)
        else:
            nCP = nCPi
        self.nControlPoints = nCP[:]
        self.nControlPointSpans = self.nControlPoints - 1
        self.nInternalKnotSpans = self.nControlPoints - degree
        self.nPointsPerSubDX = len(xl)#int(nPointsX / nSubDomainsX)
        self.nPointsPerSubDY = len(yl)#int(nPointsY / nSubDomainsY)
        self.xbounds = xb
        self.xl = xl
        self.yl = yl
        self.zl = zl
        #self.corebounds = [coreb.min[0]-xb.min[0], -1+coreb.max[0]-xb.max[0]]
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
        self.Dmini = []
        self.Dmaxi = []
        self.leftconstraint = np.zeros(1)
        self.rightconstraint = np.zeros(1)
        self.topconstraint = np.zeros(1)
        self.bottomconstraint = np.zeros(1)
        self.leftconstraintKnots = np.zeros(1)
        self.rightconstraintKnots = np.zeros(1)
        self.topconstraintKnots = np.zeros(1)
        self.bottomconstraintKnots = np.zeros(1)
        self.figHnd = None
        self.figHndErr = None
        self.figSuffix = ""
        self.globalIterationNum = 0
        self.adaptiveIterationNum = 0
        self.globalTolerance = 1e-13
        
        ## Convergence related metrics and checks
        self.outerIteration = 0
        self.outerIterationConverged = False
        self.decodederrors = np.zeros(2) # L2 and Linf
        self.errorMetricsL2 = np.zeros((nASMIterations), dtype='float64')
        self.errorMetricsLinf = np.zeros((nASMIterations), dtype='float64')


    def show(self, cp):
        self.Dmini = np.array([min(self.xl), min(self.yl)])
        self.Dmaxi = np.array([max(self.xl), max(self.yl)])

        print("Rank: %d, Subdomain %d: Bounds = [%d - %d, %d - %d]" % (commWorld.rank, cp.gid(), self.xbounds.min[0], self.xbounds.max[0], self.xbounds.min[1], self.xbounds.max[1]))

    def plot_control(self, cp):

        self.pMK = decode(self.pAdaptive, self.WAdaptive, self.Nu, self.Nv)

        Xi, Yi = np.meshgrid(self.xl, self.yl)

        axHnd = self.figHnd.gca(projection='3d')

        mycolors = cm.Spectral(self.pMK/zmax)
        # mycolors = cm.Spectral(self.pMK)
#         surf = axHnd.plot_surface(Xi, Yi, self.zl, cmap=cm.coolwarm, antialiased=True, alpha=0.75, label='Input')

        surf = axHnd.plot_surface(Xi, Yi, self.pMK.T, antialiased=False, alpha=0.95, cmap=cm.Spectral, label='Decoded', 
            vmin=zmin, vmax=zmax,  
            #facecolors=mycolors, 
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
            self.figHnd.savefig("decoded-data-%s.png"%(self.figSuffix))   # save the figure to file
        else:
            self.figHnd.savefig("decoded-data.png")   # save the figure to file


    def plot_error(self, cp):

        # self.pMK = decode(self.pAdaptive, self.WAdaptive, self.Nu, self.Nv)
        errorDecoded = np.abs(self.zl - self.pMK)

        Xi, Yi = np.meshgrid(self.xl, self.yl)

        axHnd = self.figHndErr.gca(projection='3d')
        surf = axHnd.plot_surface(Xi, Yi, np.log10(errorDecoded.T), cmap=cm.Spectral, label='Error', antialiased=False, alpha=0.95, linewidth=0.1, edgecolors='k')
        surf._facecolors2d=surf._facecolors3d
        surf._edgecolors2d=surf._edgecolors3d
        
        # if cp.gid() == 0:
        self.figHndErr.colorbar(surf)
        axHnd.legend()
        
        if len(self.figSuffix):
            self.figHndErr.savefig("error-data-%s.png"%(self.figSuffix))   # save the figure to file
        else:
            self.figHndErr.savefig("error-data.png")   # save the figure to file


    def output_vtk(self, cp):

        if useVTKOutput:

            self.pMK = decode(self.pAdaptive, self.WAdaptive, self.Nu, self.Nv)
            errorDecoded = np.abs(self.zl - self.pMK)

            Xi, Yi = np.meshgrid(self.xl, self.yl)

            Xi = Xi.reshape(1, Xi.shape[0], Xi.shape[1])
            Yi = Yi.reshape(1, Yi.shape[0], Yi.shape[1])
            Zi = np.ones(Xi.shape)
            PmK = self.pMK.T.reshape(1, self.pMK.shape[1], self.pMK.shape[0])
            errorDecoded = errorDecoded.T.reshape(1, errorDecoded.shape[1], errorDecoded.shape[0])
            gridToVTK("./structured-%s"%(self.figSuffix), Xi, Yi, Zi, pointData = {"solution" : PmK, "error": errorDecoded})


    def set_fig_handles(self, cp, fig=None, figerr=None, suffix=""):
        self.figHnd = fig
        self.figHndErr = figerr
        self.figSuffix = suffix
        
    def print_solution(self, cp):
#         self.pMK = decode(self.pAdaptive, self.WAdaptive, self.Nu, self.Nv)
#         errorDecoded = self.zl - self.pMK

        print("Domain: ", cp.gid()+1, "Exact = ", self.zl)
        print("Domain: ", cp.gid()+1, "Exact - Decoded = ", np.abs(self.zl - self.pMK))

    def send(self, cp):
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
                if dir[0] == 0: # target is coupled in Y-direction
                    if dir[1] > 0: # target block is above current subdomain
                        pl = (useDerivativeConstraints+1)*len(self.pAdaptive[:, -1])
                        if pl == 0: continue
                        o = np.zeros(2+pl+self.knotsAdaptiveU.shape[0])

                        o[0] = pl
                        o[1] = self.knotsAdaptiveU.shape[0]

                        if verbose: print('Top: ', self.pAdaptive[:, -1:-2-useDerivativeConstraints:-1].T)
                        o[2:pl+2] = self.pAdaptive[:, -1:-2-useDerivativeConstraints:-1].T.reshape(pl)
                        o[pl+2:] = self.knotsAdaptiveU[:]
                    else: # target block is below current subdomain
                        pl = (useDerivativeConstraints+1)*len(self.pAdaptive[:, 0])
                        if pl == 0: continue
                        o = np.zeros(2+pl+self.knotsAdaptiveU.shape[0])
                        o[0] = pl
                        o[1] = self.knotsAdaptiveU.shape[0]

                        if verbose: print('Bottom: ', self.pAdaptive[:, 0:useDerivativeConstraints+1].T)
                        o[2:pl+2] = self.pAdaptive[:, 0:useDerivativeConstraints+1].T.reshape(pl)
                        o[pl+2:] = self.knotsAdaptiveU[:]
                    
                if dir[1] == 0: # target is coupled in X-direction
                    if dir[0] > 0: # target block is to the right of current subdomain
                        pl = (useDerivativeConstraints+1)*len(self.pAdaptive[-1, :])
                        if pl == 0: continue
                        o = np.zeros(2+pl+self.knotsAdaptiveV.shape[0])
                        o[0] = pl
                        o[1] = self.knotsAdaptiveV.shape[0]

                        if verbose: print('Left: ', self.pAdaptive[-1:-2-useDerivativeConstraints:-1, :])
                        o[2:pl+2] = self.pAdaptive[-1:-2-useDerivativeConstraints:-1, :].reshape(pl)
                        o[pl+2:] = self.knotsAdaptiveV[:]

                    else: # target block is to the left of current subdomain
                        pl = (useDerivativeConstraints+1)*len(self.pAdaptive[0, :])
                        if pl == 0: continue
                        o = np.zeros(2+pl+self.knotsAdaptiveV.shape[0])
                        o[0] = pl
                        o[1] = self.knotsAdaptiveV.shape[0]

                        if verbose: print('Right: ', self.pAdaptive[0:useDerivativeConstraints+1, :])
                        o[2:pl+2] = self.pAdaptive[0:useDerivativeConstraints+1, :].reshape(pl)
                        o[pl+2:] = self.knotsAdaptiveV[:]

            if len(o) > 1 and verbose:
                print("%d sending to %d: %s to direction %s" % (cp.gid(), target.gid, o, dir))
            cp.enqueue(target, o)

    def recv(self, cp):
        link = cp.link()
        for i in range(len(link)):
            tgid = link.target(i).gid
            o = cp.dequeue(tgid)
            if len(o) == 1:
                continue

            pl = int(o[0])
            pll = int(pl/(useDerivativeConstraints+1))
            tl = int(o[1])
            dir = link.direction(i)
            # print("%d received from %d: %s from direction %s, with sizes %d+%d" % (cp.gid(), tgid, o, dir, pl, tl))

            # ONLY consider coupling through faces and not through verties
            # This means either dir[0] or dir[1] has to be "0" for subdomain coupling to be active
            # Hence we only consider 4 neighbor cases, instead of 8.
            if dir[0] == 0 and dir[1] == 0:
                continue
            if dir[0] == 0: # target is coupled in Y-direction
                if dir[1] > 0: # target block is above current subdomain
                    self.topconstraint = np.array(o[2:pl+2]).reshape(useDerivativeConstraints+1,pll).T if pl > 0 else []
                    self.topconstraintKnots = np.array(o[pl+2:]) if tl > 0 else []
                    # print("Top: %d received from %d: from direction %s, with sizes %d+%d" % (cp.gid(), tgid, dir, pl, tl), self.topconstraint, self.topconstraintKnots)
                else: # target block is below current subdomain
                    self.bottomconstraint = np.array(o[2:pl+2]).reshape(useDerivativeConstraints+1,pll).T if pl > 0 else []
                    self.bottomconstraintKnots = np.array(o[pl+2:]) if tl > 0 else []
                    # print("Bottom: %d received from %d: from direction %s, with sizes %d+%d" % (cp.gid(), tgid, dir, pl, tl), self.bottomconstraint, self.bottomconstraintKnots)

            if dir[1] == 0: # target is coupled in X-direction
                if dir[0] < 0: # target block is to the left of current subdomain
                    # print('Right: ', np.array(o[2:pl+2]).reshape(useDerivativeConstraints+1,pll).T)
                    self.leftconstraint = np.array(o[2:pl+2]).reshape(useDerivativeConstraints+1,pll).T if pl > 0 else []
                    self.leftconstraintKnots = np.array(o[pl+2:]) if tl > 0 else []
                    # print("Left: %d received from %d: from direction %s, with sizes %d+%d" % (cp.gid(), tgid, dir, pl, tl), self.leftconstraint, self.leftconstraintKnots)
                    # print("Left: %d received from %d: from direction %s, with sizes %d+%d" % (cp.gid(), tgid, dir, pl, tl), self.leftconstraint, - self.pAdaptive[:,0])
                else: # target block is to right of current subdomain
                    # print('Left: ', np.array(o[2:pl+2]).reshape(useDerivativeConstraints+1,pll).T)
                    self.rightconstraint = np.array(o[2:pl+2]).reshape(useDerivativeConstraints+1,pll).T if pl > 0 else []
                    self.rightconstraintKnots = np.array(o[pl+2:]) if tl > 0 else []
                    # print("Right: %d received from %d: from direction %s, with sizes %d+%d" % (cp.gid(), tgid, dir, pl, tl), self.rightconstraint, self.rightconstraintKnots)
                    # print("Right: %d received from %d: from direction %s, with sizes %d+%d" % (cp.gid(), tgid, dir, pl, tl), self.rightconstraint, - self.pAdaptive[:,-1])



    def LSQFit_NonlinearOptimize(self, idom, W, degree, constraintsAll=None):

        from scipy.optimize import root, anderson, newton_krylov#, BroydenFirst, KrylovJacobian

        constraints = None
        jacobian_const = None
        solution = []

        # Initialize relevant data
        if constraintsAll is not None:
            constraints = constraintsAll['P']#[:,:].reshape(W.shape)
            knotsAllU = constraintsAll['Tu']
            knotsAllV = constraintsAll['Tv']
            weightsAll = constraintsAll['W']
            left = constraintsAll['left']
            leftknots = constraintsAll['leftknots']
            right = constraintsAll['right']
            rightknots = constraintsAll['rightknots']
            top = constraintsAll['top']
            topknots = constraintsAll['topknots']
            bottom = constraintsAll['bottom']
            bottomknots = constraintsAll['bottomknots']
            initSol = constraints[:,:]

        else:
            print ('Constraints are all null. Solving unconstrained.')
            initSol = np.ones_like(W)

        # Compute the residual as sum of two components
        # 1. The decoded error evaluated at P
        # 2. A penalized domain boundary constraint component
        def residual(Pin, verbose=False):
            bc_penalty = 1e7
            norm_type = 2
            
            P = Pin.reshape(W.shape)

            # Residuals are in the decoded space - so direct way to constrain the boundary data
            if constrainInterfaces and False:
                decoded_residual_norm = 0
            else:
                decoded = decode(P, W, self.Nu, self.Nv)
                residual_decoded = np.abs( decoded - self.zl )/zRange
                residual_vec_decoded = residual_decoded.reshape(residual_decoded.shape[0]*residual_decoded.shape[1])
                decoded_residual_norm = LA.norm(residual_vec_decoded, norm_type)

#             res1 = np.dot(res_dec, self.Nv)
#             residual = np.dot(self.Nu.T, res1).reshape(self.Nu.shape[1]*self.Nv.shape[1])

            def decode1D(P, W, x, t):
                return np.array([(np.sum(basis(x[u],degree,t) *P*W)/(np.sum(basis(x[u],degree,t)*W))) for u,_ in enumerate(x)])
            
            # cons.append( {'type': 'eq', 'fun' : lambda x: np.array([ ( pieceBezierDer22(x, W, U, t, degree)[-1] - (bzD[-1] + bzDp[0])/2 ) ])} )

            def pieceBezierDer22(P, W, U, T, degree):
                bezierP = range(0, len(P)-degree, degree)
                Q = []

                for ps in bezierP:
                    pp = P[ps:ps+degree+1]
                    qq = np.asarray([degree*(pp[i+1]-pp[i])/(T[degree+ps+i+1]-T[ps+i]) for i in range(len(pp)-1)])
                    Q.extend(qq)

                return Q

            interpOrder = 'cubic'
            constrained_residual_norm = 0
            constrained_residual_norm2 = 0
            if constraints is not None and len(constraints) > 0 and constrainInterfaces:
                if len(left) > 1:
                    if useDecodedConstraints:
                        leftdata = decode1D(P[0, :], np.ones(P[0, :].shape), self.yl, knotsAllV)
                        constrained_residual_norm += ( np.sum( ( leftdata[:] - left[:] )**2 ) / len(leftdata) )
                    else:
                        ctrlpts_pos = getControlPoints(knotsAllV, degree)
                        ctrlpts_ppos = getControlPoints(leftknots, degree)
                        leftdata = np.zeros((ctrlpts_pos.shape[0], useDerivativeConstraints+1))
                        for col in range(useDerivativeConstraints+1):
                            rbf = Rbf(ctrlpts_ppos, left[:, col], function=interpOrder)
                            leftdata[:, col] = np.array(rbf(ctrlpts_pos))

                        # Compute the residual for left interface condition
                        # constraintRes += np.sum( ( P[0,:] - (leftdata[:]) )**2 ) / len(leftdata)
                        constrained_residual_norm += ( np.sum( ( P[0, :] - 0.5 * (constraints[0, :] + leftdata[:, 0]) )**2 ) / len(leftdata[:,0]) )
                        # print('Left Shapes knots: ', constrained_residual_norm, len(left[:]), len(knotsAllV), self.yl.shape, leftdata, (constraints[0,:] ), P[0,:], P[-1,:])

                        # Let us evaluate higher order derivative constraints as well
                        if useDerivativeConstraints > 0:
                            constrained_residual_norm2 += ( np.sum( ( (P[1, :] - P[0, :]) - 0.5 * ( (constraints[1, :] - constraints[0, :]) + (leftdata[:, 0] - leftdata[:, 1]) ) )**2 ) / len(leftdata[:,0]) )
                            # constrained_residual_norm2 += ( np.sum( ( 0.5 * (P[1, :] - P[0, :] + constraints[1, :] - constraints[0, :]) - (leftdata[:, 0] - leftdata[:, 1]) )**2 ) / len(leftdata[:,0]) )
                            # constrained_residual_norm2 += ( np.sum( ( P[1, :] - P[0, :] - (0.5 * constraints[0, :] + 0.5 * leftdata[:, 0] - leftdata[:, 1]) )**2 ) / len(leftdata[:,0]) )

                if len(right) > 1:
                    if useDecodedConstraints:
                        rightdata = decode1D(P[-1, :], np.ones(P[-1, :].shape), self.yl, knotsAllV)
                        constrained_residual_norm += ( np.sum( ( rightdata[:] - right[:] )**2 ) / len(rightdata) )
                    else:
                        ctrlpts_pos = getControlPoints(knotsAllV, degree)
                        ctrlpts_ppos = getControlPoints(rightknots, degree)
                        # Let us get the local interpolator baseed on control point locations
                        rightdata = np.zeros((ctrlpts_pos.shape[0], useDerivativeConstraints+1))
                        for col in range(useDerivativeConstraints+1):
                            rbf = Rbf(ctrlpts_ppos, right[:, col], function=interpOrder)
                            rightdata[:, col] = np.array(rbf(ctrlpts_pos))

                        # Compute the residual for right interface condition
                        # constraintRes += np.sum( ( P[-1,:] - (rightdata[:]) )**2 ) / len(rightdata)
                        constrained_residual_norm += ( np.sum( ( P[-1, :] - 0.5 * (constraints[-1, :] + rightdata[:, 0]) )**2 ) / len(rightdata[:,0]) )
                        # print('Right Shapes knots: ', constrained_residual_norm, len(right[:]), len(knotsAllV), self.yl.shape, rightdata, (constraints[-1,:]), P[0,:], P[-1,:])

                        # Let us evaluate higher order derivative constraints as well
                        if useDerivativeConstraints > 0:
                            # First derivative
                            constrained_residual_norm2 += ( np.sum( ( (P[-2, :] - P[-1, :]) - 0.5 * ((constraints[-2, :] - constraints[-1, :]) + (rightdata[:, 0] - rightdata[:, 1]) ) )**2 ) / len(rightdata) )
                            # constrained_residual_norm2 += ( np.sum( ( 0.5 * (P[-2, :] - P[-1, :] + constraints[-2, :] - constraints[-1, :]) - (rightdata[:, 0] - rightdata[:, 1]) )**2 ) / len(rightdata[:,0]) )
                            # constrained_residual_norm2 += ( np.sum( ( (P[-2, :] - P[-1, :]) - ( (0.5 * constraints[-1, :] + 0.5 * rightdata[:, 0] - rightdata[:, 1] ) ) )**2 ) / len(rightdata[:,0]) )

                if len(top) > 1:
                    if useDecodedConstraints:
                        topdata = decode1D(P[:, -1], np.ones(P[:, -1].shape), self.xl, knotsAllU)
                        constrained_residual_norm += ( np.sum( ( topdata[:] - top[:] )**2 ) / len(topdata) )
                    else:
                        ctrlpts_pos = getControlPoints(knotsAllU, degree)
                        ctrlpts_ppos = getControlPoints(topknots, degree)
                        topdata = np.zeros((ctrlpts_pos.shape[0], useDerivativeConstraints+1))
                        for col in range(useDerivativeConstraints+1):
                            rbf = Rbf(ctrlpts_ppos, top[:, col], function=interpOrder)
                            topdata[:, col] = np.array(rbf(ctrlpts_pos))

                        # Compute the residual for top interface condition
                        # constraintRes += np.sum( ( P[:,-1] - (topdata[:]) )**2 ) / len(topdata)
                        constrained_residual_norm += ( np.sum( ( P[:, -1] - 0.5 * (constraints[:, -1] + topdata[:, 0]) )**2 ) / len(topdata[:,0]) ) 
                        # print('Top: ', constrained_residual_norm, P[:, -1], constraints[:, -1], P[:, 0], constraints[:, 0], topdata[:])

                        # Let us evaluate higher order derivative constraints as well
                        if useDerivativeConstraints > 0:
                            # First derivative
                            constrained_residual_norm2 += ( np.sum( ( (P[:, -2] - P[:, -1]) - 0.5 * ( (constraints[:, -2] - constraints[:, -1]) + (topdata[:, 0] - topdata[:, 1])) )**2 ) / len(topdata) )
                            # constrained_residual_norm2 += ( np.sum( ( 0.5 * (P[:, -2] - P[:, -1] + constraints[:, -2] - constraints[:, -1]) - (topdata[:, 0] - topdata[:, 1]) )**2 ) / len(topdata[:,0]) )
                            # constrained_residual_norm2 += ( np.sum( ( (P[:, -2] - P[:, -1]) - ( 0.5 * constraints[:, -1] + 0.5 * topdata[:, 0] - topdata[:, 1]) )**2 ) / len(topdata[:,0]) )

                if len(bottom) > 1:
                    if useDecodedConstraints:
                        bottomdata = decode1D(P[:, 0], np.ones(P[:, 0].shape), self.xl, knotsAllU)
                        constrained_residual_norm += ( np.sum( ( bottomdata[:] - bottom[:] )**2 ) / len(bottomdata) )
                    else:
                        ctrlpts_pos = getControlPoints(knotsAllU, degree)
                        ctrlpts_ppos = getControlPoints(bottomknots, degree)
                        bottomdata = np.zeros((ctrlpts_pos.shape[0], useDerivativeConstraints+1))
                        for col in range(useDerivativeConstraints+1):
                            rbf = Rbf(ctrlpts_ppos, bottom[:, col], function=interpOrder)
                            bottomdata[:, col] = np.array(rbf(ctrlpts_pos))

                        # Compute the residual for bottom interface condition
                        # constraintRes += np.sum( ( P[:,0] - (bottomdata[:]) )**2 ) / len(bottomdata)
                        constrained_residual_norm += ( np.sum( ( P[:, 0] - 0.5 * (constraints[:, 0] + bottomdata[:, 0]) )**2 ) / len(bottomdata[:,0]) ) 
                        # print('Bottom: ', constrained_residual_norm, P[:, -1], constraints[:, -1], P[:, 0], constraints[:, 0], bottomdata[:])

                        # Let us evaluate higher order derivative constraints as well
                        if useDerivativeConstraints > 0:
                            constrained_residual_norm2 += ( np.sum( ( (P[:, 1] - P[:, 0]) - 0.5 * ((constraints[:, 1] - constraints[:, 0]) + (bottomdata[:, 0] - bottomdata[:, 1])) )**2 ) / len(bottomdata) ) 
                            # constrained_residual_norm2 += ( np.sum( ( 0.5 * (P[:, 1] - P[:, 0] + constraints[:, 1] - constraints[:, 0]) - (bottomdata[:, 0] - bottomdata[:, 1]) )**2 ) / len(bottomdata[:,0]) )
                            # constrained_residual_norm2 += ( np.sum( ( (P[:, 1] - P[:, 0]) - ( 0.5 * constraints[:, 0] + 0.5 * bottomdata[:, 0] - bottomdata[:, 1]) )**2 ) / len(bottomdata[:,0]) ) 

            # compute the net residual norm that includes the decoded error in the current subdomain and the penalized
            # constraint error of solution on the subdomain interfaces
            net_residual_norm = decoded_residual_norm + (bc_penalty * np.sqrt(constrained_residual_norm) + np.power(bc_penalty, 1.0) * np.sqrt(constrained_residual_norm2) )

            if verbose:
                print('Residual = ', net_residual_norm, ' and res_dec = ', decoded_residual_norm, ' and constraint = ', np.sqrt(constrained_residual_norm), ' der = ', np.sqrt(constrained_residual_norm2))

            return net_residual_norm


        # Compute the residual as sum of two components
        # 1. The decoded error evaluated at P
        # 2. A penalized domain boundary constraint component
        def residual_vec(Pin, verbose=False):
            bc_penalty = 1e7
            norm_type = 2
            
            P = Pin.reshape(W.shape)

            # Residuals are in the decoded space - so direct way to constrain the boundary data
            decodeOperator, decodeShape = get_decode_operator(W, self.Nu, self.Nv)
            decoded = decodeOperator.dot(P)
            decoded = decoded.reshape(decodeShape)
            residual_decoded = np.abs( decoded - self.zl )/zRange

#             res1 = np.dot(res_dec, self.Nv)
#             residual = np.dot(self.Nu.T, res1).reshape(self.Nu.shape[1]*self.Nv.shape[1])

            def decode1D(P, W, x, t):
                return np.array([(np.sum(basis(x[u],degree,t) *P*W)/(np.sum(basis(x[u],degree,t)*W))) for u,_ in enumerate(x)])
            
            # cons.append( {'type': 'eq', 'fun' : lambda x: np.array([ ( pieceBezierDer22(x, W, U, t, degree)[-1] - (bzD[-1] + bzDp[0])/2 ) ])} )

            def pieceBezierDer22(P, W, U, T, degree):
                bezierP = range(0, len(P)-degree, degree)
                Q = []

                for ps in bezierP:
                    pp = P[ps:ps+degree+1]
                    qq = np.asarray([degree*(pp[i+1]-pp[i])/(T[degree+ps+i+1]-T[ps+i]) for i in range(len(pp)-1)])
                    Q.extend(qq)

                return Q

            interpOrder = 'cubic'
            constrained_residual_norm = 0
            constrained_residual_norm2 = 0
            if constraints is not None and len(constraints) > 0 and constrainInterfaces:
                if len(left) > 1:
                    indx=0
                    if useDecodedConstraints:
                        leftdata = decode1D(P[0, :], np.ones(P[0, :].shape), self.yl, knotsAllV)
                        residual_decoded[indx, :] += ( leftdata[:] - left[:] )
                    else:
                        ctrlpts_pos = getControlPoints(knotsAllV, degree)
                        ctrlpts_ppos = getControlPoints(leftknots, degree)
                        leftdata = np.zeros((ctrlpts_pos.shape[0], useDerivativeConstraints+1))
                        for col in range(useDerivativeConstraints+1):
                            rbf = Rbf(ctrlpts_ppos, left[:, col], function=interpOrder)
                            leftdata[:, col] = np.array(rbf(ctrlpts_pos))

                        # Compute the residual for left interface condition
                        # constraintRes += np.sum( ( P[0,:] - (leftdata[:]) )**2 ) / len(leftdata)
                        # constrained_residual_norm += ( np.sum( ( P[indx, :] - 0.5 * (constraints[indx, :] + leftdata[:, 0]) )**2 ) / len(leftdata[:,0]) )
                        constraintSol = ( P[indx, :] - 0.5 * (constraints[indx, :] + leftdata[:, 0]) )
                        leftConstraintRes = decode1D(constraintSol, np.ones(constraintSol.shape), self.yl, knotsAllV)
                        residual_decoded[indx, :] += bc_penalty * leftConstraintRes
                        # print('Left Shapes knots: ', constrained_residual_norm, len(left[:]), len(knotsAllV), self.yl.shape, leftdata, (constraints[0,:] ), P[0,:], P[-1,:])

                        # Let us evaluate higher order derivative constraints as well
                        if useDerivativeConstraints > 0:
                            residual_decoded[indx, :] += np.power(bc_penalty, 0.5) * ( (P[1, :] - P[0, :]) - 0.5 * ( (constraints[1, :] - constraints[0, :]) + (leftdata[:, 0] - leftdata[:, 1]) ) )
                            # constrained_residual_norm2 += ( np.sum( ( 0.5 * (P[1, :] - P[0, :] + constraints[1, :] - constraints[0, :]) - (leftdata[:, 0] - leftdata[:, 1]) )**2 ) / len(leftdata[:,0]) )
                            # constrained_residual_norm2 += ( np.sum( ( P[1, :] - P[0, :] - (0.5 * constraints[0, :] + 0.5 * leftdata[:, 0] - leftdata[:, 1]) )**2 ) / len(leftdata[:,0]) )

                if len(right) > 1:
                    indx=-1
                    if useDecodedConstraints:
                        rightdata = decode1D(P[0, :], np.ones(P[0, :].shape), self.yl, knotsAllV)
                        residual_decoded[indx, :] += ( rightdata[:] - right[:] )
                    else:
                        ctrlpts_pos = getControlPoints(knotsAllV, degree)
                        ctrlpts_ppos = getControlPoints(rightknots, degree)
                        # Let us get the local interpolator baseed on control point locations
                        rightdata = np.zeros((ctrlpts_pos.shape[0], useDerivativeConstraints+1))
                        for col in range(useDerivativeConstraints+1):
                            rbf = Rbf(ctrlpts_ppos, right[:, col], function=interpOrder)
                            rightdata[:, col] = np.array(rbf(ctrlpts_pos))

                        # Compute the residual for right interface condition
                        # constraintRes += np.sum( ( P[-1,:] - (rightdata[:]) )**2 ) / len(rightdata)
                        constraintSol = ( P[indx, :] - 0.5 * (constraints[indx, :] + rightdata[:, 0]) )
                        rightConstraintRes = decode1D(constraintSol, np.ones(constraintSol.shape), self.yl, knotsAllV)
                        residual_decoded[indx, :] += bc_penalty * rightConstraintRes
                        # print('Right Shapes knots: ', constrained_residual_norm, len(right[:]), len(knotsAllV), self.yl.shape, rightdata, (constraints[-1,:]), P[0,:], P[-1,:])

                        # Let us evaluate higher order derivative constraints as well
                        if useDerivativeConstraints > 0:
                            # First derivative
                            residual_decoded[indx, :] += np.power(bc_penalty, 0.5) * ( (P[-2, :] - P[-1, :]) - 0.5 * ( (constraints[-2, :] - constraints[-1, :]) + (rightdata[:, 0] - rightdata[:, 1]) ) )
                            # constrained_residual_norm2 += ( np.sum( ( 0.5 * (P[-2, :] - P[-1, :] + constraints[-2, :] - constraints[-1, :]) - (rightdata[:, 0] - rightdata[:, 1]) )**2 ) / len(rightdata[:,0]) )
                            # constrained_residual_norm2 += ( np.sum( ( (P[-2, :] - P[-1, :]) - ( (0.5 * constraints[-1, :] + 0.5 * rightdata[:, 0] - rightdata[:, 1] ) ) )**2 ) / len(rightdata[:,0]) )

                if len(top) > 1:
                    indx=-1
                    if useDecodedConstraints:
                        topdata = decode1D(P[:, -1], np.ones(P[:, -1].shape), self.xl, knotsAllU)
                        residual_decoded[:, indx] += ( topdata[:] - top[:] )
                    else:
                        ctrlpts_pos = getControlPoints(knotsAllU, degree)
                        ctrlpts_ppos = getControlPoints(topknots, degree)
                        topdata = np.zeros((ctrlpts_pos.shape[0], useDerivativeConstraints+1))
                        for col in range(useDerivativeConstraints+1):
                            rbf = Rbf(ctrlpts_ppos, top[:, col], function=interpOrder)
                            topdata[:, col] = np.array(rbf(ctrlpts_pos))

                        # Compute the residual for top interface condition
                        # constraintRes += np.sum( ( P[:,-1] - (topdata[:]) )**2 ) / len(topdata)
                        constraintSol = ( P[:, indx] - 0.5 * (constraints[:, indx] + topdata[:, 0]) )
                        topConstraintRes = decode1D(constraintSol, np.ones(constraintSol.shape), self.xl, knotsAllU)
                        residual_decoded[:, indx] += bc_penalty * topConstraintRes
                        # print('Top: ', constrained_residual_norm, P[:, -1], constraints[:, -1], P[:, 0], constraints[:, 0], topdata[:])

                        # Let us evaluate higher order derivative constraints as well
                        if useDerivativeConstraints > 0:
                            # First derivative
                            residual_decoded[:, indx] += np.power(bc_penalty, 0.5) * ( (P[:, -2] - P[:, -1]) - 0.5 * ( (constraints[:, -2] - constraints[:, -1]) + (topdata[:, 0] - topdata[:, 1]) ) )
                            # constrained_residual_norm2 += ( np.sum( ( 0.5 * (P[:, -2] - P[:, -1] + constraints[:, -2] - constraints[:, -1]) - (topdata[:, 0] - topdata[:, 1]) )**2 ) / len(topdata[:,0]) )
                            # constrained_residual_norm2 += ( np.sum( ( (P[:, -2] - P[:, -1]) - ( 0.5 * constraints[:, -1] + 0.5 * topdata[:, 0] - topdata[:, 1]) )**2 ) / len(topdata[:,0]) )

                if len(bottom) > 1:
                    indx=0
                    if useDecodedConstraints:
                        bottomdata = decode1D(P[:, 0], np.ones(P[:, 0].shape), self.xl, knotsAllU)
                        residual_decoded[:, indx] += ( bottomdata[:] - bottom[:] )
                    else:
                        ctrlpts_pos = getControlPoints(knotsAllU, degree)
                        ctrlpts_ppos = getControlPoints(bottomknots, degree)
                        bottomdata = np.zeros((ctrlpts_pos.shape[0], useDerivativeConstraints+1))
                        for col in range(useDerivativeConstraints+1):
                            rbf = Rbf(ctrlpts_ppos, bottom[:, col], function=interpOrder)
                            bottomdata[:, col] = np.array(rbf(ctrlpts_pos))

                        # Compute the residual for bottom interface condition
                        # constraintRes += np.sum( ( P[:,0] - (bottomdata[:]) )**2 ) / len(bottomdata)
                        constraintSol = ( P[:, indx] - 0.5 * (constraints[:, indx] + bottomdata[:, 0]) )
                        bottomConstraintRes = decode1D(constraintSol, np.ones(constraintSol.shape), self.xl, knotsAllU)
                        residual_decoded[:, indx] += bc_penalty * bottomConstraintRes
                        # print('Bottom: ', constrained_residual_norm, P[:, -1], constraints[:, -1], P[:, 0], constraints[:, 0], bottomdata[:])

                        # Let us evaluate higher order derivative constraints as well
                        if useDerivativeConstraints > 0:
                            residual_decoded[:, indx] += np.power(bc_penalty, 0.5) * ( (P[:, 1] - P[:, 0]) - 0.5 * ((constraints[:, 1] - constraints[:, 0]) + (bottomdata[:, 0] - bottomdata[:, 1]) ) ) 
                            # constrained_residual_norm2 += ( np.sum( ( 0.5 * (P[:, 1] - P[:, 0] + constraints[:, 1] - constraints[:, 0]) - (bottomdata[:, 0] - bottomdata[:, 1]) )**2 ) / len(bottomdata[:,0]) )
                            # constrained_residual_norm2 += ( np.sum( ( (P[:, 1] - P[:, 0]) - ( 0.5 * constraints[:, 0] + 0.5 * bottomdata[:, 0] - bottomdata[:, 1]) )**2 ) / len(bottomdata[:,0]) ) 

            # compute the net residual norm that includes the decoded error in the current subdomain and the penalized
            # constraint error of solution on the subdomain interfaces
            # net_residual_norm = decoded_residual_norm + (bc_penalty * np.sqrt(constrained_residual_norm) + np.power(bc_penalty, 1.0) * np.sqrt(constrained_residual_norm2) )

            # if verbose:
            #     print('Residual = ', net_residual_norm, ' and res_dec = ', decoded_residual_norm, ' and constraint = ', np.sqrt(constrained_residual_norm), ' der = ', np.sqrt(constrained_residual_norm2))

            print('Decodeoperator: ', decodeOperator.shape, decodeOperatorT.shape, 'residual_decoded: ', residual_decoded.shape)
            residual_vec_ctrlpts = decodeOperator.T.dot(residual_decoded)
            residual_vec_decoded = residual_vec_ctrlpts.reshape(residual_vec_ctrlpts.shape[0]*residual_vec_ctrlpts.shape[1])
            return residual_vec_decoded

        def print_iterate(P):

            res = residual(P, verbose=True)
            self.globalIterationNum += 1

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

        #if constraintsAll is not None:
        #    jacobian_const = egrad(residual)(initSol)

        # Now invoke the solver and compute the constrained solution
        if constraints is None and not alwaysSolveConstrained:
            solution,_ = lsqFit(self.Nu, self.Nv, W, self.zl)
            solution = solution.reshape(W.shape)
        else:

            print('Initial calculation')
            print_iterate(initSol)
            if enforceBounds:
                bnds = np.tensordot(np.ones(initSol.shape[0]*initSol.shape[1]), np.array([self.zl.min(), self.zl.max()]), axes=0)
            else:
                bnds = None
            print('Using optimization solver = ', solverScheme)
            # Solver options: https://docs.scipy.org/doc/scipy-0.13.0/reference/generated/scipy.optimize.show_options.html
            if solverScheme == 'L-BFGS-B' or solverScheme == 'TNC':
                res = minimize(residual, x0=initSol, method=solverScheme, #'SLSQP', #'L-BFGS-B', #'TNC', 
                            bounds=bnds,
                            jac=jacobian,
                            callback=print_iterate,
                            tol=self.globalTolerance, 
                            options={'disp': False, 'ftol': maxRelErr, 'gtol': self.globalTolerance, 'maxiter': solverMaxIter})
            elif solverScheme == 'CG':
                res = minimize(residual, x0=initSol, method=solverScheme, # Unbounded - can blow up
                            jac=jacobian,
                            callback=print_iterate,
                            tol=self.globalTolerance, 
                            options={'disp': False, 'gtol': self.globalTolerance, 'norm': 2, 'maxiter': solverMaxIter})
            elif solverScheme == 'SLSQP':
                res = minimize(residual, x0=initSol, method=solverScheme, #'SLSQP', #'L-BFGS-B', #'TNC', 
                            bounds=bnds,
                            callback=print_iterate,
                            tol=self.globalTolerance, 
                            options={'disp': False, 'ftol': maxRelErr, 'maxiter': solverMaxIter})
            elif solverScheme == 'Newton-CG':
                res = minimize(residual, x0=initSol, method=solverScheme, #'SLSQP', #'L-BFGS-B', #'TNC', 
                        jac=jacobian,
                        # jac=egrad(residual)(initSol),
                        callback=print_iterate,
                        tol=self.globalTolerance, 
                        options={'disp': False, 'eps': self.globalTolerance, 'maxiter': solverMaxIter})
            elif solverScheme == 'trust-krylov':
                res = minimize(residual, x0=initSol, method=solverScheme, #'SLSQP', #'L-BFGS-B', #'TNC', 
                        jac=jacobian,
                        hess=hessian,
                        callback=print_iterate,
                        tol=self.globalTolerance, 
                        options={'inexact': True})
            else:
                optimizer_options={ 'disp': False, 
                                # 'ftol': 1e-5, 
                                'fatol': self.globalTolerance,
                                'maxiter': solverMaxIter,
                                'jac_options': { 'method': 'lgmres'} # {‘lgmres’, ‘gmres’, ‘bicgstab’, ‘cgs’, ‘minres’} 
                            }
                # jacobian_const = egrad(residual)(initSol)
                res = scipy.optimize.root(residual_vec, x0=initSol, 
                            method=solverScheme, #'krylov', 'lm'
                            # jac=jacobian_vec,
                            callback=print_iterate,
                            tol=self.globalTolerance, 
                            options=optimizer_options)
            print ('[%d] : %s' % (idom, res.message))
            solution = res.x.reshape(W.shape)

        return solution

    def interpolate_knots(self, knots, tnew):

        r = 1

        # print('Original interpolation shapes: ', Pnew.shape, knots, tnew)
        # For all entries that are missing in self.knotsAdaptive, call castleDeJau 
        # and recompute control points one by one
        for knot in tnew:
            #knotInd = np.searchsorted(knots, knot)
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
            #knotInd = np.searchsorted(knots, knot)
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
                Pnew,W = deCasteljau1D(Pnew, W, knots, knot, knotInd-1, r)
                knots = np.insert(knots, knotInd, knot)
        
        #cplocCtrPt = getControlPoints(knots, degree) * (Dmax - Dmin) + Dmin
        #coeffs_xy = getControlPoints(knots, degree)

        #return Pnew, knots, W
        return Pnew, knots
    

    def interpolate(self, P, knots, tnew):

        r = 1
        knotsnew = self.interpolate_knots(knots, tnew)
        # print (knotsnew.shape[0]-degree-1, useDerivativeConstraints+1)
        Pnew = np.zeros([knotsnew.shape[0]-degree-1, useDerivativeConstraints+1])
        for col in range(useDerivativeConstraints+1):
            Pnew[:,col], knotsnew = self.interpolate_private(P[:,col], knots, tnew)

        return Pnew, knotsnew

    def interpolate_spline(self, pold, coeffs_xy, ncoeffs_xy):
       
        interpOrder = 'linear' # 'linear', 'cubic', 'quintic'
        #InterpCp = interp1d(coeffs_x, self.pAdaptive, kind=interpOrder)
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
        print('Rank:', commW.rank, ' SDom:', cp.gid(), ' L2 Error table: ', self.errorMetricsL2)


    def check_convergence(self, cp):

        if len(self.decodedAdaptiveOld):

            # Let us compute the relative change in the solution between current and previous iteration
            iterateChangeVec = (self.decodedAdaptive - self.decodedAdaptiveOld)
            errorMetricsSubDomL2 = np.linalg.norm(iterateChangeVec, ord=2) / np.linalg.norm(self.pAdaptive, ord=2)
            errorMetricsSubDomLinf = np.linalg.norm(iterateChangeVec, ord=np.inf) / np.linalg.norm(self.pAdaptive, ord=np.inf)

            self.errorMetricsL2[self.outerIteration] = self.decodederrors[0]
            self.errorMetricsLinf[self.outerIteration] = self.decodederrors[1]

            # print(cp.gid()+1, ' Convergence check: ', errorMetricsSubDomL2, errorMetricsSubDomLinf, 
            #                 np.abs(self.errorMetricsLinf[self.outerIteration]-self.errorMetricsLinf[self.outerIteration-1]), 
            #                 errorMetricsSubDomLinf < 1e-8 and np.abs(self.errorMetricsL2[self.outerIteration]-self.errorMetricsL2[self.outerIteration-1]) < 1e-10)

            if errorMetricsSubDomLinf < 1e-8 and np.abs(self.errorMetricsL2[self.outerIteration]-self.errorMetricsL2[self.outerIteration-1]) < 1e-10:
                print('Subdomain ', cp.gid()+1, ' has converged to its final solution with error = ', errorMetricsSubDomLinf)
                self.outerIterationConverged = True
        
        self.outerIteration += 1

        isASMConverged = commW.allreduce(self.outerIterationConverged, op=MPI.MIN)

    def adaptive(self, iSubDom, xl, yl, zl, strategy='extend', weighted=False, 
                 r=1, MAX_ERR=1e-3, MAX_ITER=5, 
                 split_all=True, 
                 decodedError=None):

        from scipy.interpolate import Rbf

        k = []
        r = min(r,degree) #multiplicity can not be larger than degree
        reuseE = (decodedError) if decodedError is not None else None

        splitIndeces = []
        r = min(r,degree) #multiplicity can not be larger than degree

        Tu = np.copy(self.knotsAdaptiveU)
        Tv = np.copy(self.knotsAdaptiveV)
        P = np.copy(self.pAdaptive)
        W = np.copy(self.WAdaptive)
        
        globalIterationNum = 0

        self.Nu = basis(self.U[np.newaxis,:],degree,Tu[:,np.newaxis]).T
        self.Nv = basis(self.V[np.newaxis,:],degree,Tv[:,np.newaxis]).T

        if (np.sum(P) == 0 and len(P) > 0) or len(P) == 0:
            #W = np.ones(self.nControlPoints)
            print(iSubDom, " - Applying the unconstrained solver.")
            P = self.LSQFit_NonlinearOptimize(iSubDom, W, degree, None)
#             P,_ = lsqFit(self.Nu, self.Nv, W, self.zl)
            decodedError = decode(P, W, self.Nu, self.Nv) #- self.zl
            MAX_ITER = 0
        else:
            if disableAdaptivity:
                MAX_ITER = 1

        interpOrder = 'cubic'
        iteration = 0
        for iteration in range(MAX_ITER):
            fev = 0

            if projectData:

                if not useDeCastelJau:
                    # Need to compute a projection from [P, Tu, Tv] to [Pnew, Tunew, Tvnew]
                    coeffs_x1 = getControlPoints(Tu, degree)
                    coeffs_y1 = getControlPoints(Tv, degree)

                    if iSubDom == 1:
                        if nSubDomainsY > 1:
                            coeffs_x2 = np.linspace(np.min(coeffs_x1), np.max(coeffs_x1), len(coeffs_x1)*2)
                            coeffs_y2 = np.copy(coeffs_y1)
                        else:
                            coeffs_y2 = np.linspace(np.min(coeffs_y1), np.max(coeffs_y1), len(coeffs_y1)*2)
                            coeffs_x2 = np.copy(coeffs_x1)
        #                 coeffs_y2 = coeffs_y1[::2]
                    else:
                        if nSubDomainsY > 1:
                            coeffs_x2 = np.copy(coeffs_x1)
        #                     coeffs_x1 = coeffs_x2[::2]
                            coeffs_x1 = np.linspace(np.min(coeffs_x2), np.max(coeffs_x2), len(coeffs_x2)/2)
                            coeffs_y2 = np.copy(coeffs_y1)
                        else:
                            coeffs_y2 = np.copy(coeffs_y1)
        #                     coeffs_x1 = coeffs_x2[::2]
                            coeffs_y1 = np.linspace(np.min(coeffs_y2), np.max(coeffs_y2), len(coeffs_y2)/2)
                            coeffs_x2 = np.copy(coeffs_x1)

                def decode1D(P, W, x, t):
                    return np.array([(np.sum(basis(x[u],degree,t) *P*W)/(np.sum(basis(x[u],degree,t)*W))) for u,_ in enumerate(x)])

                # Create a projection for also the solution vector along subdomain boundaries
                leftconstraint_projected = np.copy(self.leftconstraint)
                leftconstraint_projected_knots = np.copy(Tv)
                if len(self.leftconstraint) > 1:
                    if useDeCastelJau:
                        #print('NURBSInterp Left proj: ', self.leftconstraint.shape, Tu.shape, Tv.shape, self.leftconstraintKnots.shape)
                        if useDecodedConstraints:
                            leftconstraint_projected_cp, leftconstraint_projected_knots = self.interpolate(self.leftconstraint, self.leftconstraintKnots, Tv[:])
                            leftconstraint_projected = decode1D(leftconstraint_projected_cp, np.ones(leftconstraint_projected_cp.shape), self.yl, leftconstraint_projected_knots)
                        else:
                            leftconstraint_projected, leftconstraint_projected_knots = self.interpolate(self.leftconstraint, self.leftconstraintKnots, Tv[:])
                        # print('NURBSInterp Left proj: ', self.leftconstraint.shape, Tv.shape, self.leftconstraintKnots.shape, leftconstraint_projected.shape)
                    else:
#                     rbfL = Rbf(coeffs_y2, self.leftconstraint, function=interpOrder)
#                     leftconstraint_projected = rbfL(coeffs_y1)
                        # print('SplineInterp Left proj: ', coeffs_y2.shape, self.leftconstraint.shape)
                        leftconstraint_projected = self.interpolate_spline(self.leftconstraint, coeffs_y2, coeffs_y1)

                rightconstraint_projected = np.copy(self.rightconstraint)
                rightconstraint_projected_knots = np.copy(Tv)
                if len(self.rightconstraint) > 1:
                    if useDeCastelJau:
                        #print('NURBSInterp Right proj: ', self.rightconstraint.shape, Tu.shape, Tv.shape, self.rightconstraintKnots.shape)
                        if useDecodedConstraints:
                            rightconstraint_projected_cp, rightconstraint_projected_knots = self.interpolate(self.rightconstraint, self.rightconstraintKnots, Tv[:])
                            rightconstraint_projected = decode1D(rightconstraint_projected_cp, np.ones(rightconstraint_projected_cp.shape), self.yl, rightconstraint_projected_knots)
                        else:
                            # print('NURBSInterp Right proj: ', self.rightconstraint.shape, Tv.shape, self.rightconstraintKnots.shape)
                            rightconstraint_projected, rightconstraint_projected_knots = self.interpolate(self.rightconstraint, self.rightconstraintKnots, Tv[:])

                    else:
#                     rbfR = Rbf(coeffs_y1, self.rightconstraint, function=interpOrder)
#                     rightconstraint_projected = rbfR(coeffs_y2)
                        # print('SplineInterp Right proj: ', coeffs_y1.shape, self.rightconstraint.shape)
                        rightconstraint_projected = self.interpolate_spline(self.rightconstraint, coeffs_y1, coeffs_y2)

                topconstraint_projected = np.copy(self.topconstraint)
                topconstraint_projected_knots = np.copy(Tu)
                if len(self.topconstraint) > 1:
                    if useDeCastelJau:
                        if useDecodedConstraints:
                            topconstraint_projected_cp, topconstraint_projected_knots = self.interpolate(self.topconstraint, self.topconstraintKnots, Tu[:])
                            topconstraint_projected = decode1D(topconstraint_projected_cp, np.ones(topconstraint_projected_cp.shape), self.xl, topconstraint_projected_knots)
                        else:
                            topconstraint_projected, topconstraint_projected_knots = self.interpolate(self.topconstraint, self.topconstraintKnots, Tu[:])

                    else:
                        topconstraint_projected = self.interpolate_spline(self.topconstraint, coeffs_x2, coeffs_x1)
#                     rbfT = Rbf(coeffs_x2, self.topconstraint, function=interpOrder)
#                     topconstraint_projected = rbfT(coeffs_x1)

                bottomconstraint_projected = np.copy(self.bottomconstraint)
                bottomconstraint_projected_knots = np.copy(Tu)
                if len(self.bottomconstraint) > 1:
                    if useDeCastelJau:
                        if useDecodedConstraints:
                            bottomconstraint_projected_cp, bottomconstraint_projected_knots = self.interpolate(self.bottomconstraint, self.bottomconstraintKnots, Tu[:])
                            bottomconstraint_projected = decode1D(bottomconstraint_projected_cp, np.ones(bottomconstraint_projected_cp.shape), self.xl, bottomconstraint_projected_knots)
                        else:
                            bottomconstraint_projected, bottomconstraint_projected_knots = self.interpolate(self.bottomconstraint, self.bottomconstraintKnots, Tu[:])
                    else:
                        bottomconstraint_projected = self.interpolate_spline(self.bottomconstraint, coeffs_x1, coeffs_x2)
#                     rbfB = Rbf(coeffs_x1, self.bottomconstraint, function=interpOrder)
#                     bottomconstraint_projected = rbfB(coeffs_x2)

            else:
                leftconstraint_projected = self.leftconstraint
                rightconstraint_projected = self.rightconstraint
                topconstraint_projected = self.topconstraint
                bottomconstraint_projected = self.bottomconstraint

            interface_constraints_obj = dict()
            interface_constraints_obj['left'] = leftconstraint_projected
            interface_constraints_obj['leftknots'] = leftconstraint_projected_knots
            interface_constraints_obj['right'] = rightconstraint_projected
            interface_constraints_obj['rightknots'] = rightconstraint_projected_knots
            interface_constraints_obj['top'] = topconstraint_projected
            interface_constraints_obj['topknots'] = topconstraint_projected_knots
            interface_constraints_obj['bottom'] = bottomconstraint_projected
            interface_constraints_obj['bottomknots'] = bottomconstraint_projected_knots

            if disableAdaptivity:

                interface_constraints_obj['P']=P
                interface_constraints_obj['W']=W
                interface_constraints_obj['Tu']=Tu
                interface_constraints_obj['Tv']=Tv

                print(iSubDom, " - Applying the constrained solver.")
                P = self.LSQFit_NonlinearOptimize(iSubDom, W, degree, interface_constraints_obj)

            else:
                Tunew,Tvnew,Usplits,Vsplits,E,maxE = knotRefine(P, W, Tu, Tv, self.Nu, self.Nv, self.U, self.V, self.zl, r, 
                                                                #reuseE=reuseE, 
                                                                MAX_ERR=MAX_ERR, 
                                                                find_all=split_all)
            
                # maxE = LA.norm(E.reshape(E.shape[0]*E.shape[1]),np.inf)
                print("Adaptive iteration ", iteration+1, ", at ", maxE, " maxError")

                if (len(Tu)==len(Tunew) and len(Tv)==len(Tvnew)) or \
                    len(Tunew)-degree-1 > self.nPointsPerSubDX or \
                    len(Tvnew)-degree-1 > self.nPointsPerSubDY:
                    print ("Nothing more to refine: E = ", E)
                    reuseE = E
                    Tunew = np.copy(Tu)
                    Tvnew = np.copy(Tv)
                    #break

                if len(Usplits)==0 or len(Vsplits)==0:
                    if (maxE>MAX_ERR):
                        print ("Max error hit: E = ", E)
                        reuseE = E
                        continue

                if(maxE<=MAX_ERR):
                    k = [-1, -1]
                    print ("Adaptive done in %d iterations at %e maxError"%(iteration+1,maxE))
                    break

                nControlPointsPrev = self.nControlPoints
                self.nControlPoints = np.array([Tunew.shape[0]-1-degree, Tvnew.shape[0]-1-degree])
                self.nControlPointSpans = self.nControlPoints - 1
                self.nInternalKnotSpans = self.nControlPoints - degree

                print('-- Spatial Adaptivity: Before = ', nControlPointsPrev, ', After = ', self.nControlPoints)

                if strategy == 'extend' and not split_all:   #only use when coupled with a solver
                    k = [Usplits.pop(),Vsplits.pop()] #if reuseE is None else [-1,-1]
                    u = np.array([Tunew[k[0]+1],Tvnew[k[1]+1]])
                    print(iSubDom, " - Applying the deCasteljau solver")
                    P,W = deCasteljau2D(P, W, Tu, Tv, u, k, r)

                elif strategy == 'reset':

                    Tunew = np.sort(Tunew)
                    Tvnew = np.sort(Tvnew)

                    self.Nu = basis(self.U[np.newaxis,:],degree,Tunew[:,np.newaxis]).T
                    self.Nv = basis(self.V[np.newaxis,:],degree,Tvnew[:,np.newaxis]).T

                    W = np.ones( self.nControlPoints )

                    # Need to compute a projection from [P, Tu, Tv] to [Pnew, Tunew, Tvnew]
                    coeffs_x1 = getControlPoints(Tu, degree)
                    coeffs_y1 = getControlPoints(Tv, degree)
                    Xi, Yi = np.meshgrid(coeffs_x1, coeffs_y1)

                    coeffs_x2 = getControlPoints(Tunew, degree)
                    coeffs_y2 = getControlPoints(Tvnew, degree)
                    Xi2, Yi2 = np.meshgrid(coeffs_x2, coeffs_y2)

                    rbfi = Rbf(Yi, Xi, P, function='cubic')  # radial basis function interpolator instance
                    Pnew = rbfi(Yi2, Xi2).reshape(W.shape)   # interpolated values

                    interface_constraints_obj['P']=Pnew[:,:]
                    interface_constraints_obj['W']=W[:,:]
                    interface_constraints_obj['Tu']=Tunew[:]
                    interface_constraints_obj['Tv']=Tvnew[:]

                    print(iSubDom, " - Applying the adaptive constrained solver.")
                    P = self.LSQFit_NonlinearOptimize(iSubDom, W, degree, interface_constraints_obj)


#             if len(self.leftconstraint) > 1:
#                 print('Errors-left: ', P[0,:]-leftconstraint_projected[:])
#             if len(self.rightconstraint) > 1:
#                 print('Errors-right: ', P[-1,:]-rightconstraint_projected[:])
#             if len(self.topconstraint) > 1:
#                 print('Errors-top: ', P[:,-1]-topconstraint_projected[:])
#             if len(self.bottomconstraint) > 1:
#                 print('Errors-bottom: ', P[:,0]-bottomconstraint_projected[:])

            if not disableAdaptivity:
                Tu = np.copy(Tunew)
                Tv = np.copy(Tvnew)

            self.Nu = basis(self.U[np.newaxis,:],degree,Tu[:,np.newaxis]).T
            self.Nv = basis(self.V[np.newaxis,:],degree,Tv[:,np.newaxis]).T

            decoded = decode(P, W, self.Nu, self.Nv)
            decodedError = np.abs(np.array(decoded-self.zl)) / zRange

            reuseE = (decodedError.reshape(decodedError.shape[0]*decodedError.shape[1]))
            print("\tDecoded error: ", LA.norm(reuseE, 2))

            iteration += 1

        return P, W, Tu, Tv, np.array([k]), decodedError

    def solve_adaptive(self, cp):

        if self.outerIterationConverged:
            print(cp.gid()+1, ' subdomain has already converged to its solution. Skipping solve ...')
            return

        ## Subdomain ID: iSubDom = cp.gid()+1
        newSolve = False
        if (np.sum(self.pAdaptive) == 0 and len(self.pAdaptive) > 0) or len(self.pAdaptive) == 0:
            newSolve = True

        if not newSolve:

            self.nControlPointSpans = self.nControlPoints - 1
            self.nInternalKnotSpans = self.nControlPoints - degree

        inc = (self.Dmaxi - self.Dmini) / self.nInternalKnotSpans
        # print ("self.nInternalKnotSpans = ", self.nInternalKnotSpans, " inc = ", inc)
        if (len(self.knotsAdaptiveU) == 0) or (len(self.knotsAdaptiveU) > 0 and np.sum(self.knotsAdaptiveU) == 0):
            tu   = np.linspace(self.Dmini[0] + inc[0], self.Dmaxi[0] - inc[0], self.nInternalKnotSpans[0] - 1)
            tv   = np.linspace(self.Dmini[1] + inc[1], self.Dmaxi[1] - inc[1], self.nInternalKnotSpans[1] - 1)
            self.knotsAdaptiveU = np.concatenate(([self.Dmini[0]] * (degree+1), tu, [self.Dmaxi[0]] * (degree+1)))
            self.knotsAdaptiveV = np.concatenate(([self.Dmini[1]] * (degree+1), tv, [self.Dmaxi[1]] * (degree+1)))

        self.U    = np.linspace(self.Dmini[0], self.Dmaxi[0], self.nPointsPerSubDX)#self.nPointsPerSubDX, nPointsX
        self.V    = np.linspace(self.Dmini[1], self.Dmaxi[1], self.nPointsPerSubDY)#self.nPointsPerSubDY, nPointsY

        print ("Subdomain -- ", cp.gid()+1)

        # Let do the recursive iterations
        # Use the previous MAK solver solution as initial guess; Could do something clever later

        # Invoke the adaptive fitting routine for this subdomain
        adaptiveErr = None
        # self.globalTolerance = 1e-3 * 1e-3**self.adaptiveIterationNum
        self.pAdaptive, self.WAdaptive, self.knotsAdaptiveU, self.knotsAdaptiveV, kv, adaptiveErr = self.adaptive(cp.gid()+1,
                                       self.xl, self.yl, self.zl,
                                       strategy=AdaptiveStrategy, weighted=False,
                                       r=1, MAX_ERR=maxAbsErr, MAX_ITER=maxAdaptIter, #MAX_ERR=maxAdaptErr,
                                       split_all=True,
                                       decodedError=adaptiveErr)
        self.adaptiveIterationNum += 1

        # Update the local decoded data
        self.decodedAdaptiveOld = np.copy(self.decodedAdaptive)
        self.decodedAdaptive = decode(self.pAdaptive, self.WAdaptive, self.Nu, self.Nv)

        # E = (self.decodedAdaptive[self.corebounds[0]:self.corebounds[1]] - self.yl[self.corebounds[0]:self.corebounds[1]])/zRange
        E = (self.decodedAdaptive - self.yl)/zRange
        E = (E.reshape(E.shape[0]*E.shape[1]))
        LinfErr = np.linalg.norm(E, ord=np.inf)
        L2Err = np.sqrt(np.sum(E**2)/len(E))

        self.decodederrors[0] = L2Err
        self.decodederrors[1] = LinfErr

        print ("Subdomain -- ", cp.gid()+1, ": L2 error: ", L2Err, ", Linf error: ", LinfErr)
        
#         print("adaptiveErr: ", self.pAdaptive.shape, self.WAdaptive.shape, self.zl.shape, self.Nu.shape, self.Nv.shape)
        # errorMAK = L2LinfErrors(self.pAdaptive, self.WAdaptive, self.zl, degree, self.Nu, self.Nv)


#########

# Initialize DIY
commWorld = diy.mpi.MPIComm()           # world
mc2 = diy.Master(commWorld)         # master
domain_control = diy.DiscreteBounds([0,0], [len(x)-1,len(y)-1])

# Routine to recursively add a block and associated data to it
def add_input_control_block2(gid, core, bounds, domain, link):
    print("Subdomain %d: " % gid, core, bounds, domain)
    minb = bounds.min
    maxb = bounds.max
    
    xlocal = x[minb[0]:maxb[0]+1]
    ylocal = y[minb[1]:maxb[1]+1]
    zlocal = z[minb[0]:maxb[0]+1,minb[1]:maxb[1]+1]
    # zlocal = z[minb[0]:maxb[0]+1,minb[1]:maxb[1]+1]

    # print("Subdomain %d: " % gid, minb[0], minb[1], maxb[0], maxb[1], z.shape, zlocal.shape)

    mc2.add(gid, InputControlBlock(gid,nControlPointsInput,core,xlocal,ylocal,zlocal), link)

# TODO: If working in parallel with MPI or DIY, do a global reduce here

errors = np.zeros([nASMIterations,2]) # Store L2, Linf errors as function of iteration

# Let us initialize DIY and setup the problem
share_face = [True,True]
wrap = [False,False]
# ghosts = [useDerivativeConstraints,useDerivativeConstraints]
ghosts = [0,0]
divisions = [nSubDomainsX,nSubDomainsY]

d_control = diy.DiscreteDecomposer(2, domain_control, nSubDomainsX*nSubDomainsY, share_face, wrap, ghosts, divisions)
a_control2 = diy.ContiguousAssigner(nprocs, nSubDomainsX*nSubDomainsY)

d_control.decompose(rank, a_control2, add_input_control_block2)

del x, y, z

mc2.foreach(InputControlBlock.show)

#########
import timeit
start_time = timeit.default_timer()
for iterIdx in range(nASMIterations):

    if rank == 0: print ("\n---- Starting Iteration: %d ----" % iterIdx)
    
    # Now let us perform send-receive to get the data on the interface boundaries from 
    # adjacent nearest-neighbor subdomains
    mc2.foreach(InputControlBlock.send)
    mc2.exchange(False)
    mc2.foreach(InputControlBlock.recv)

    if iterIdx > 1:
        disableAdaptivity = True
        constrainInterfaces = True
    # else:
    #     disableAdaptivity = False
    #     constrainInterfaces = False

    # disableAdaptivity = True
    # constrainInterfaces = True

    if iterIdx > 0 and rank == 0: print("")
    mc2.foreach(InputControlBlock.solve_adaptive)
    
    if disableAdaptivity:
        # if rank == 0: print("")
        mc2.foreach(InputControlBlock.check_convergence)

    if showplot:

        figHnd = plt.figure()
        figHndErr = plt.figure()

        mc2.foreach(lambda icb, cp: InputControlBlock.set_fig_handles(icb, cp, figHnd, figHndErr, "%d-%d"%(cp.gid(),iterIdx)))

        # Now let us draw the data from each subdomain
        mc2.foreach(InputControlBlock.plot_control)
        mc2.foreach(InputControlBlock.plot_error)

        #figHnd.savefig("decoded-data-%d-%d.png"%(iterIdx))   # save the figure to file
        #figHndErr.savefig("error-data-%d-%d.png"%(iterIdx))   # save the figure to file

    commW.Barrier()
    sys.stdout.flush()

    if useVTKOutput:

        mc2.foreach(lambda icb, cp: InputControlBlock.set_fig_handles(icb, cp, None, None, "%d-%d"%(cp.gid(),iterIdx)))
        mc2.foreach(InputControlBlock.output_vtk)

        WritePVTKFile(iterIdx)

    if isASMConverged == 1:
        if rank == 0: print( "\n\nASM solver converged after %d iterations\n\n" % (iterIdx) )
        break

# mc2.foreach(InputControlBlock.print_solution)

elapsed = timeit.default_timer() - start_time

sys.stdout.flush()
if rank == 0:
    print('\nTotal computational time for solve = ', elapsed, '\n')

# np.set_printoptions(formatter={'float': '{: 5.12e}'.format})
# mc.foreach(InputControlBlock.print_error_metrics)
if rank == 0: print('')

np.set_printoptions(formatter={'float': '{: 5.12e}'.format})
mc2.foreach(InputControlBlock.print_error_metrics)

if showplot: plt.show()

