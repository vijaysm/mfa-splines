    # coding: utf-8

# In[1]:
# get_ipython().magic(u'matplotlib notebook')
import sys, math
import numpy as np
import scipy
import diy
import matplotlib.pyplot as plt
# import plotly.tools as tls
#from IPython.display import clear_output

plt.style.use(['seaborn-whitegrid'])
# plt.style.use(['classic'])
params = {"ytick.color" : "w",
          "xtick.color" : "w",
          "axes.labelcolor" : "w",
          "axes.edgecolor" : "w"}
plt.rcParams.update(params)

# --- set problem input parameters here ---
Dmin           = -4.
Dmax           = 4.
nPoints        = 21
nSubDomains    = 1
degree         = 3
sincFunc       = False
useAdditiveSchwartz = True
useDerivativeConstraints = 1

solverscheme   = 'SLSQP' # [SLSQP, COBYLA]
nControlPoints = (4*degree + 1) #minimum number of control points
nmaxiter       = 4
scale          = 1
maxAbsErr      = 1e-3
maxRelErr      = 1e-5
# AdaptiveStrategy = 'extend'
AdaptiveStrategy = 'reset'
# ------------------------------------------

if sincFunc:
    x = np.linspace(Dmin, Dmax, nPoints)
    y = scale * np.sinc(x)
    # y = scale * np.sin(math.pi * x/4)
else:
    y = np.fromfile("s3d.raw", dtype=np.float64) #
    nPoints = y.shape[0]
    x = np.linspace(Dmin, Dmax, nPoints)

if nPoints % nSubDomains > 0:
    print ( "[ERROR]: The total number of points do not divide equally with subdomains" )
    sys.exit(1)

mpl_fig = plt.figure()
plt.plot(x, y)

# Initialize DIY
w = diy.mpi.MPIComm()           # world
m = diy.Master(w)               # master
#domain = diy.DiscreteBounds([min(x),min(y)], [max(x),max(y)])
#d = diy.DiscreteDecomposer(2, domain, nSubDomains)
#a = diy.ContiguousAssigner(w.size, nblocks)

domain = diy.DiscreteBounds([0], [len(x)-1])

d = diy.DiscreteDecomposer(1, domain, nSubDomains)
a = diy.ContiguousAssigner(w.size, nSubDomains)



# In[2]:

from scipy import linalg, matrix
from scipy.interpolate import LSQUnivariateSpline, UnivariateSpline, interp1d
from scipy.optimize import minimize, linprog
from scipy.linalg import svd

EPS    = 1e-14
basis  = lambda u,p,T: ((T[:-1]<=u) * (u<=T[1:])).astype(np.float) if p==0 else                     ((u - T[:-p]) /(T[p:]  -T[:-p]+EPS))[:-1] * basis(u,p-1,T)[:-1] +                     ((T[p+1:] - u)/(T[p+1:]-T[1:-p]+EPS))     * basis(u,p-1,T)[1:]

def getControlPoints(knots, k):
    nCtrlPts = len(knots) - 1 - k
    cx       = np.zeros(nCtrlPts)
    for i in range(nCtrlPts):
        tsum = 0
        for j in range(1, k + 1):
            tsum += knots[i + j]
        cx[i] = float(tsum) / k
    return cx

def decode(P, W, x, t, degree):
    return np.array([(np.sum(basis(x[u],degree,t) *P*W)/(np.sum(basis(x[u],degree,t)*W))) for u,_ in enumerate(x)])

def decode_derivative(P, W, x, t, degree):
    return np.array([(np.sum(basis(x[u],degree,t) *P*W)/(np.sum(basis(x[u],degree,t)*W))) for u,_ in enumerate(x)])

def Error(P, W, y, x, t, degree):
    return decode(P, W, x, t, degree) - y

def nullspace(A, eps=1e-15):
    u, s, vh = scipy.linalg.svd(A)
    null_mask = (s <= eps)
    null_space = scipy.compress(null_mask, vh, axis=0)
    return scipy.transpose(null_space)

def null_space(A, rcond=None):
    u, s, vh = svd(A, full_matrices=True)
    # print 'EigValues: ', s
    M, N = u.shape[0], vh.shape[1]
    if rcond is None:
        rcond = np.finfo(s.dtype).eps * max(M, N)
    tol = np.amax(s) * rcond
    num = np.sum(s > tol, dtype=int)
    Q = vh[num:,:].T.conj()
    return Q

def pieceBezierDer22(P, W, U, T, degree):
    bezierP = range(0, len(P)-degree, degree)
    Q = []

    for ps in bezierP:
        pp = P[ps:ps+degree+1]
        qq = np.asarray([degree*(pp[i+1]-pp[i])/(T[degree+ps+i+1]-T[ps+i]) for i in range(len(pp)-1)])
        Q.extend(qq)

    return Q

def pieceBezierDDer22(P, W, U, T, degree):
    bezierP = range(0, len(P)-degree, degree)
    Q = []
    
    for ps in bezierP:
        pp = P[ps:ps+degree+1]
        qq = np.asarray([degree*(pp[i+1]-pp[i])/(T[degree+ps+i+1]-T[ps+i]) for i in range(len(pp)-1)])
        Q.extend(qq)

    return Q

def pointBezierDer(pip, pi, W, x, degree):
    Q = [degree*(pip-pi)]
    T = np.concatenate(([0]*degree, [1]*degree))
    x = np.linspace(0,1,x.shape[0])
    return decode(Q, W[:-1], x, T, degree-1)

def RMSE(P, W, y, U, T, degree):
    E = Error(P, W, y, U, T, degree)
    return sqrt(np.sum(E**2))

def NMaxError(P, W, y, U, T, degree):
    yRange = y.max()-y.min()
    
    E = Error(P, W, y, U, T, degree)
    return np.abs(E).max()/yRange

def L2LinfErrors(P, W, y, U, T, degree):
    yRange = y.max()-y.min()

    E = Error(P, W, y, U, T, degree)
    LinfErr = np.abs(E).max()/yRange
    L2Err = math.sqrt(np.sum(E**2))
    return [L2Err, LinfErr]

# # **Adaptive fitting per subdomain**

# In[4]:


import collections

def rint(x):
    return int(round(x))

def toHomogeneous(P, W):
    return np.hstack( ( (P*W)[:,np.newaxis], W[:,np.newaxis]) )

def fromHomogeneous(PW):
    P = PW[:,0]
    W = PW[:,1]
    return P/W, W

def knotInsert(T, xu, us, splitUs=True, r=1):
    if not isinstance(us, collections.Iterable):
        us = [us]
    Tnew = []
    # lu=min(range(len(xu)), key=lambda i: abs(xu[i]-us))
    # lu=min(enumerate(xu), key=lambda x: abs(x[1]-us))
    # lu = (np.abs(xu - us)).argmin()

    # print xu
    # print T
    toSplit = us
    if splitUs:
        t = T[degree:-degree]
        toSplit = np.unique(np.concatenate([ np.where(((t[:-1]<=u) * (u<=t[1:])).astype(np.float))[0] for u in us]).ravel())
        toSplit += degree
        for ti,tval in enumerate(T):
            Tnew.append(tval)
            if np.isin(ti, toSplit):
                lu = (np.abs(xu - tval)).argmin()
                # If there exists a physical point between the knot span, let us add
                # to the adaptive knot location list
                if xu[lu]-T[ti] >= 0 and xu[lu+1]-T[ti+1] <= 0:
                    for i in range(r):
                        Tnew.append(tval+((T[ti+1]-tval)/2.))
    else:
        Tnew = T.tolist()
        for u in us:
            for i in range(r):
                Tnew.append(u)
        Tnew = np.sort(Tnew)
    return np.array(Tnew), toSplit
               
def knotRefine(P, W, T, u, yl, E, r=1, find_all=True, MAX_ERR = 1e-2):
#     print len(P),len(W),len(u),len(yl),len(T)
    yRange = yl.max()-yl.min()
    if(E.max()<=MAX_ERR*yRange):
        return T, []
    us = u[np.where( E >=(MAX_ERR*yRange) )[0] if find_all else np.argmax(E)]

    return knotInsert(T, u, us, r=r)

def deCasteljau(P, W, T, u, k, r=1):
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


def lsqFit(N, W, y, U, t, degree, constraints=None, continuity=0):
    if constraints is None or len(constraints)==0:
        RN = (N*W)/(np.sum(N*W, axis=1)[:,np.newaxis])
        LHS = np.matmul(RN.T,RN)
        RHS = np.matmul(RN.T, y)
        return linalg.lstsq(LHS, RHS)[0]

def lsqFitWithCons(N, W, ysl, U, t, degree, constraints=[], continuity=0):
    def l2(P, W, ysl, U, t, degree):
        return np.sum(Error(P, W, ysl, U, t, degree)**2)
#     print len(W), len(ysl),len(U),len(t)
#     print 'Checking data: ', l2(np.ones_like(W),W,ysl,U,t,degree)
    res = minimize(l2, np.ones_like(W), method='SLSQP', args=(W, ysl, U, t, degree), 
                   constraints=constraints,
                    options={'disp': True})
    return res.x

def adaptive(iSubDom, interface_constraints_obj, u, xl, yl, strategy='reset', r=1, MAX_ERR=1e-2, MAX_ITER=5, split_all=True):
    splitIndeces = []
    r = min(r,degree) #multiplicity can not be larger than degree
    nPointsPerSubD = nPoints / nSubDomains

    T = interface_constraints_obj['T'][iSubDom - 1]
    if len(interface_constraints_obj['P']):
        P = interface_constraints_obj['P'][iSubDom-1]
        W = interface_constraints_obj['W'][iSubDom - 1]

    if len(P) == 0:
        W = np.ones(len(T) - 1 - degree)
        N = basis(u[np.newaxis,:],degree,T[:,np.newaxis]).T
        P = lsqFit(N, W, yl, u, T, degree)
        print ('Unconstrained: ', P)

    for iteration in range(MAX_ITER):
        print (" -- Adaptive iteration: ", iteration)
        E = np.abs(Error(P, W, yl, u, T, degree))
        Tnew,splitIndeces = knotRefine(P, W, T, u, yl, E, r, MAX_ERR=MAX_ERR, find_all=split_all)
        if (len(T)==len(Tnew)) or len(T)-degree-1 > nPointsPerSubD:
            break
        
        if strategy == 'extend' and ~split_all:   #only use when coupled with a solver
            k = splitIndeces[0]
            u = Tnew[k+1]
            P,W = deCasteljau(P, W, T, u, k, r)
        elif strategy == 'reset':
            # print ('Knots-OLD: ', T)
            Tnew = np.sort(Tnew)
            # print ('Knots-NEW: ', Tnew)
            # print ('Knots-NEW: ', splitIndeces)
            W = np.ones(len(Tnew) - 1 - degree)
            N = basis(u[np.newaxis,:],degree,Tnew[:,np.newaxis]).T
            # P = lsqFit(N, W, yl, u, Tnew, degree)
#             P = lsqFitWithCons(N, W, yl, u, Tnew, degree)
#             if len(interface_constraints_obj['P'][iSubDom - 1]) > 0 and len(interface_constraints_obj['P'][iSubDom]) > 0 and len(interface_constraints_obj['P'][iSubDom - 2]) > 0:
            if len(interface_constraints_obj['P'][iSubDom - 1]) > 0:
                # Interpolate or project the data to new Knot locations: From (P, T) to (Pnew, Tnew)
                coeffs_x = getControlPoints(T, degree)
                coeffs_xn = getControlPoints(Tnew, degree)
                PnewFn = interp1d(coeffs_x ,P, kind='cubic') #, kind='cubic')
                Pnew = PnewFn(coeffs_xn)
                # print 'coeffs_x = ', [coeffs_x, coeffs_xn, P, Pnew]
                interface_constraints_obj['P'][iSubDom-1]=Pnew[:]
                # interface_constraints_obj['P'][iSubDom-1]=0*W[:]
                interface_constraints_obj['W'][iSubDom-1]=W[:]
                interface_constraints_obj['T'][iSubDom-1]=Tnew[:]
                # if iSubDom < nSubDomains:
                #     print ('Constraints for left-right Subdom = ', iSubDom, ' is = ', [Pnew, interface_constraints_obj['P'][iSubDom]])
                # else:
                #     print ('Constraints for right-left Subdom = ', iSubDom, ' is = ', [interface_constraints_obj['P'][iSubDom-2], Pnew] )

                P = LSQFit_Constrained(iSubDom, N, W, yl, u, Tnew, degree, nSubDomains, interface_constraints_obj, useDerivativeConstraints, 'SLSQP')
            else:
                P = lsqFit(N, W, yl, u, Tnew, degree)

        else:
            print ("Not Implemented!!")
        
        T = Tnew
        
    return P, W, T, splitIndeces


# In[5]:

def compute_adaptive_fit_for_subdomains(nSubDomains, degree, nControlPoints, x, y, showplot=False, maxAdaptErr=maxAbsErr, PAdaptDomain=[], WAdaptDomain=[], KnotAdaptDomains=[]):

    import matplotlib.pyplot as plt
    import numpy as np
    from cycler import cycler

    nPoints = len(x)
    Dmin = min(x)
    Dmax = max(x)
    nControlPointSpans = nControlPoints - 1
    nInternalKnotSpans = nControlPointSpans - degree + 1
    nPointsPerSubD = int(nPoints / nSubDomains)

    # The variable `additiveSchwartz` controls whether we use
    # Additive vs Multiplicative Schwartz scheme for DD resolution
    if showplot:
        plt.figure()
        if nSubDomains > 5:
            plt.rc('axes', prop_cycle=(cycler('color', ['r', 'g', 'b', 'y', 'c']) + cycler('linestyle', ['-','--',':','-.',''])))
        plt.plot(x, y, 'b-', ms=5, label='Input')

    newSolve = False
    if len(PAdaptDomain[0]) == 0:
        newSolve = True

    interface_constraints_obj = dict()
    interface_constraints_obj['P'] = PAdaptDomain
    interface_constraints_obj['W'] = WAdaptDomain
    interface_constraints_obj['T'] = KnotAdaptDomains

    for iSubDom in range(1,nSubDomains+1):

        domStart = (iSubDom-1) * 1.0 / nSubDomains
        domEnd   = iSubDom * 1.0 / nSubDomains
        U   = np.linspace(domStart, domEnd, nPointsPerSubD)
        if newSolve:

            inc = (domEnd - domStart) / nInternalKnotSpans
            t   = np.linspace(domStart + inc, domEnd - inc, nInternalKnotSpans - 1)
            # dSpan = range( (iSubDom-1) * nPoints / nSubDomains, iSubDom * nPoints / nSubDomains )

        else:

            knots = KnotAdaptDomains[iSubDom-1]
            popt = PAdaptDomain[iSubDom-1]
            W = WAdaptDomain[iSubDom-1]
            nControlPoints = len(popt)

            nControlPointSpans = nControlPoints - 1
            nInternalKnotSpans = nControlPointSpans - degree + 1

            inc = (domEnd - domStart) / nInternalKnotSpans

        print ("Subdomain -- ", iSubDom)
        # print [U, x[dSpan]]

        # dSpan = np.linspace( (iSubDom-1) * nPointsPerSubD, iSubDom * nPointsPerSubD, nPointsPerSubD, dtype=int )
        dSpan = range( (iSubDom-1) * nPointsPerSubD, iSubDom * nPointsPerSubD )

        Dmini = Dmin + (Dmax - Dmin)*domStart
        Dmaxi = Dmin + (Dmax - Dmin)*domEnd

        xSD = U * (Dmax - Dmin) + Dmin
        # xSD = np.array(x[dSpan])
        ySD = np.array(y[dSpan])

        if newSolve:
            #spl = LSQUnivariateSpline(U, ySD, t, k=degree)
            
            # get the control points
            #knots    = spl.get_knots()
            #knots    = np.concatenate(([knots[0]] * degree, knots, [knots[-1]] * degree))
            knots  = np.linspace(domStart + inc, domEnd - inc, nInternalKnotSpans - 1)
            knots  = np.concatenate(([domStart] * (degree+1), knots, [domEnd] * (degree+1)))

            popt = [] # interface_constraints[iSubDom-1]
            W = [] # np.ones_like(popt)

        interface_constraints_obj['P'][iSubDom-1] = popt[:]
        interface_constraints_obj['T'][iSubDom-1] = knots[:]
        interface_constraints_obj['W'][iSubDom-1] = W[:]

        # Invoke the adaptive fitting routine for this subdomain
        pAdaptive, WAdaptive, knotsAdaptive,_ = adaptive(iSubDom, interface_constraints_obj, U, xSD, ySD, MAX_ERR=maxAdaptErr, split_all=True, strategy=AdaptiveStrategy)#, r=2, MAX_ITER=5)
        # NAdaptive = basis(U[np.newaxis,:],degree,knotsAdaptive[:,np.newaxis]).T
        # E = Error(pAdaptive, WAdaptive, ySD, U, knotsAdaptive, degree)
        # print ("Sum of squared error:", np.sum(E**2))
        # print ("Normalized max error:", np.abs(E).max()/yRange)

        if showplot:
            pMK = decode(pAdaptive, WAdaptive, xSD, knotsAdaptive * (Dmax - Dmin) + Dmin, degree)
            plt.plot(xSD, pMK, 'g--', lw=3, label='Decoded')

            # print 'xsd = ', xSD[0], xSD[-1]

            coeffs_x = getControlPoints(knotsAdaptive, degree) * (Dmax - Dmin) + Dmin
            if nSubDomains < 5:
                plt.plot(coeffs_x, pAdaptive, marker='o', linestyle='--', color=['r','g','b','y','c'][iSubDom-1], label='Control')
            else:
                plt.plot(coeffs_x, pAdaptive, marker='o', label='Control')

        errorMAK = L2LinfErrors(pAdaptive, WAdaptive, ySD, U, knotsAdaptive, degree)
        print ("Subdomain: ", iSubDom, " -- L2 error: ", errorMAK[0], ", Linf error: ", errorMAK[1])

        PAdaptDomain[iSubDom-1] = pAdaptive.copy()
        WAdaptDomain[iSubDom-1] = WAdaptive.copy()
        KnotAdaptDomains[iSubDom-1] = knotsAdaptive.copy()

    if showplot:
        plt.legend()
        plt.draw()

    return PAdaptDomain, WAdaptDomain, KnotAdaptDomains


def LSQFit_Constrained(idom, N, W, ysl, U, t, degree, nSubDomains, constraintsAll=None, useDerivatives=0, solver='SLSQP'):
    constraints = None
    if constraintsAll is not None:
        constraints = constraintsAll['P']
        knotsAll = constraintsAll['T']
        weightsAll = constraintsAll['W']
        # print 'Constraints for Subdom = ', idom, ' is = ', constraints

    if useDerivatives > 0 and constraints is not None and len(constraints) > 0:

        bzD = pieceBezierDer22(constraints[idom-1], weightsAll[idom-1], U, knotsAll[idom-1], degree)
        # bzDD = pieceBezierDDer22(bzD, W, U, knotsD, degree-1)
        # print ("Subdomain Derivatives (1,2) : ", bzD, bzDD)
        if idom < nSubDomains:
            bzDp = pieceBezierDer22(constraints[idom], weightsAll[idom], U, knotsAll[idom], degree)
            # bzDDp = pieceBezierDDer22(bzDp, W, U, knotsD, degree-1)
            # print ("Left Derivatives (1,2) : ", bzDp, bzDDp )
            # print ('Right derivative error offset: ', ( bzD[-1] - bzDp[0] ) )
        if idom > 1:
            bzDm = pieceBezierDer22(constraints[idom-2], weightsAll[idom-2], U, knotsAll[idom-2], degree)
            # bzDDm = pieceBezierDDer22(bzDm, W, U, knotsD, degree-1)
            # print ("Right Derivatives (1,2) : ", bzDm, bzDDm )
            # print ('Left derivative error offset: ', ( bzD[0] - bzDm[-1] ) )

    def ComputeL2ErrorO(P, N, W, ysl, U, t, degree):
        E = np.sum(Error(P, W, ysl, U, t, degree)**2)
        return E


    def ComputeL2Error(P, N, W, ysl, U, t, degree):
        RN = (N*W)/(np.sum(N*W, axis=1)[:,np.newaxis])
        E = (RN.dot(P) - ysl)
        return np.sum(E**2)

    cons = []
    if solver is 'SLSQP':
        if constraints is not None and len(constraints) > 0:
            if idom > 1:
                if useDerivatives >= 0:
                    cons.append( {'type': 'eq', 'fun' : lambda x: np.array([ ( x[0] - (constraints[idom-1][0] + constraints[idom-2][-1])/2 ) ])} )
                    if useDerivatives > 0:
                        cons.append( {'type': 'eq', 'fun' : lambda x: np.array([ ( pieceBezierDer22(x, W, U, t, degree)[0] - ( bzD[0] + bzDm[-1] )/2 ) ])} )
                        # cons.append( {'type': 'eq', 'fun' : lambda x: np.array([ ( pieceBezierDer22(x, W, U, t, degree)[0] - ( bzDm[-1]  ) ) ])} )
                        # cons.append( {'type': 'eq', 'fun' : lambda x: np.array([ ( (x[1] - x[0])/(knotsAll[idom-1][degree+1]-knotsAll[idom-1][0]) - ( constraints[idom-2][-1] - constraints[idom-2][-2] )/(knotsAll[idom-2][-degree-2] - knotsAll[idom-2][-1]) ) ])} )
                        if useDerivatives > 1:
                            cons.append( {'type': 'eq', 'fun' : lambda x: np.array([ ( (x[2] - x[1])/(knotsAll[idom-1][degree+2]-knotsAll[idom-1][1]) - (x[1] - x[0])/(knotsAll[idom-1][degree+1]-knotsAll[idom-1][0]) - ( (constraints[idom-2][-3] - constraints[idom-2][-2])/(knotsAll[idom-2][-3]-knotsAll[idom-2][-degree-2]) - (constraints[idom-2][-2] - constraints[idom-2][-1])/(knotsAll[idom-2][-2]-knotsAll[idom-2][-degree-3])  ) ) ])} )
                        
                        # cons.append( {'type': 'eq', 'fun' : lambda x: np.array([ ( x[1] - x[0] - 0.5*( constraints[idom-1][1] - constraints[idom-1][0] + constraints[idom-2][-1] - constraints[idom-2][-2] ) ) ])} )

                    # print 'Left delx: ', (knotsAll[idom-1][degree+1]-knotsAll[idom-1][0]), ' and ', (knotsAll[idom-1][degree+2]-knotsAll[idom-1][1])
            if idom < nSubDomains:
                if useDerivatives >= 0:
                    cons.append( {'type': 'eq', 'fun' : lambda x: np.array([ ( x[-1] - (constraints[idom-1][-1] + constraints[idom][0])/2 ) ])} )
                    if useDerivatives > 0:
                        cons.append( {'type': 'eq', 'fun' : lambda x: np.array([ ( pieceBezierDer22(x, W, U, t, degree)[-1] - (bzD[-1] + bzDp[0])/2 ) ])} )
                        # cons.append( {'type': 'eq', 'fun' : lambda x: np.array([ ( pieceBezierDer22(x, W, U, t, degree)[-1] - (bzDp[0]) ) ])} )
                        # cons.append( {'type': 'eq', 'fun' : lambda x: np.array([ ( (x[-1] - x[-2])/(knotsAll[idom-1][-degree-2] - knotsAll[idom-1][-1]) - ( constraints[idom][1] - constraints[idom][0] )/(knotsAll[idom][degree+1] - knotsAll[idom][0]) ) ])} )
                        if useDerivatives > 1:
                            cons.append( {'type': 'eq', 'fun' : lambda x: np.array([ ( (x[-3] - x[-2])/(knotsAll[idom-1][-2] - knotsAll[idom-1][-degree-3]) - (x[-2] - x[-1])/(knotsAll[idom-1][-1] - knotsAll[idom-1][-degree-2]) + ( (constraints[idom][1] - constraints[idom][0])/(knotsAll[idom][degree+1] - knotsAll[idom][0]) - (constraints[idom][2] - constraints[idom][1])/(knotsAll[idom][degree+2] - knotsAll[idom][1]) ) ) ])} )
                        # cons.append( {'type': 'eq', 'fun' : lambda x: np.array([ ( x[-1] - x[-2] - 0.5*( constraints[idom-1][-1] - constraints[idom-1][-2] + constraints[idom][1] - constraints[idom][0] ) ) ])} )
                    
                    # print 'Right delx: ', (knotsAll[idom-1][-2] - knotsAll[idom-1][-degree-3]), ' and ', (knotsAll[idom-1][-1] - knotsAll[idom-1][-degree-2])

            initSol = constraints[idom-1][:]
            # initSol = np.ones_like(W)

#             print len(initSol), len(W), len(ysl), len(U), len(t)
#             E = np.sum(Error(initSol, W, ysl, U, t, degree)**2)
#             print "unit error = ", E
            res = minimize(ComputeL2Error, x0=initSol, method='SLSQP', args=(N, W, ysl, U, t, degree),
                           constraints=cons, #x0=initSol,
                           options={'disp': False, 'ftol': 1e-10})
        else:

            res = minimize(ComputeL2Error, np.ones_like(W), method='SLSQP', args=(N, W, ysl, U, t, degree),
                           options={'disp': False, 'ftol': 1e-10})

    else:
        if constraints is not None and len(constraints) > 0:
            if idom > 1:
                print (idom, ': Left constraint ', (constraints[idom-1][0] + constraints[idom-2][-1])/2 )
                cons.append( {'type': 'ineq', 'fun' : lambda x: np.array([ ( x[0] - (constraints[0][idom-1] + constraints[-1][idom-2])/2 ) ])} )
            if idom < nSubDomains:
                print (idom, ': Right constraint ', (constraints[idom-1][-1] + constraints[idom][0])/2)
                cons.append( {'type': 'ineq', 'fun' : lambda x: np.array([ ( x[-1] - (constraints[-1][idom-1] + constraints[0][idom])/2 ) ])} )

        res = minimize(ComputeL2Error, initSol, method='COBYLA', args=(N, W, ysl, U, t, degree),
                       constraints=cons, #x0=constraints,
                       options={'disp': False, 'tol': 1e-6, 'catol': 1e-2})

    print ('[%d] : %s' % (idom, res.message))
    return res.x


def compute_mak95_fit_with_constraints(nSubDomains, degree, nControlPoints, x, y, 
                                        interface_constraints_obj, additiveSchwartz=True, useDerivatives=0,
                                        showplot=False, nosolveplot=False, 
                                        solver='SLSQP'):

    import matplotlib.pyplot as plt
    import numpy as np
    from cycler import cycler
    
    interface_constraints = interface_constraints_obj['P']

    Dmin = min(x)
    Dmax = max(x)
    nPointsPerSubD = nPoints / nSubDomains

    # The variable `additiveSchwartz` controls whether we use
    # Additive vs Multiplicative Schwartz scheme for DD resolution
    if showplot:
        plt.figure()
        if nSubDomains > 5:
            plt.rc('axes', prop_cycle=(cycler('color', ['r', 'g', 'b', 'y', 'c']) + cycler('linestyle', ['-','--',':','-.',''])))
        plt.plot(x, y, 'b-', ms=5, label='Input')

    if interface_constraints is not None and len(interface_constraints) > 0:
        POptDomain = interface_constraints[:]
    else:
        POptDomain = np.zeros([nControlPoints,nSubDomains])

    for iSubDom in range(1,nSubDomains+1):

        if interface_constraints is not None and len(interface_constraints) > 0:
          popt = interface_constraints[iSubDom-1]
          nControlPoints = len(popt)

        nControlPointSpans = nControlPoints - 1
        nInternalKnotSpans = nControlPointSpans - degree + 1
        # print 'Using new subdom point data: ', iSubDom, ' ', nControlPoints, nPoints

        domStart = (iSubDom-1) * 1.0 / nSubDomains
        domEnd   = iSubDom * 1.0 / nSubDomains

        inc = (domEnd - domStart) / nInternalKnotSpans
        U   = np.linspace(domStart, domEnd, nPointsPerSubD)
        dSpan = range( (iSubDom-1) * nPoints / nSubDomains, iSubDom * nPoints / nSubDomains )

        Dmini = Dmin + (Dmax - Dmin)*domStart
        Dmaxi = Dmin + (Dmax - Dmin)*domEnd
    
        ySD = np.asarray(y[dSpan])

        # Lets compute the control points and weights with Ma & Kruth method
        knots    = interface_constraints_obj['T'][iSubDom-1]
        W = interface_constraints_obj['W'][iSubDom-1]
        if not nosolveplot:
            N = basis(U[np.newaxis,:],degree,knots[:,np.newaxis]).T
            popt = LSQFit_Constrained(iSubDom, N, W, ySD, U, knots, degree, nSubDomains, interface_constraints_obj, useDerivatives, solver)

        if showplot:
            pMK = decode(popt, W, U, knots, degree)
            xlocal = U * (Dmax - Dmin) + Dmin
            plt.plot(xlocal, pMK, 'g--', lw=3, label='Decoded')

            coeffs_x = getControlPoints(knots, degree) * (Dmax - Dmin) + Dmin
            if nSubDomains < 5:
                plt.plot(coeffs_x, popt, marker='o', linestyle='--', color=['r','g','b','y','c'][iSubDom-1], label='Control')
            else:
                plt.plot(coeffs_x, popt, marker='o', label='Control')

        errorMAK = L2LinfErrors(popt, W, ySD, U, knots, degree)
        print ("Subdomain: %d -- L2 error: %f, Linf error: %f" % (iSubDom, errorMAK[0], errorMAK[1]) )

        POptDomain[iSubDom-1] = popt.copy()

        #
        # For multiplicative Schwartz scheme, update the solution of the current subdomain
        # in the constraints vector so that the next subdomain gets the newer information
        # Note: Remember that the additiveSchwartz is equivalent to solving the decomposed
        #       problem with a block-Jacobi scheme, while the multiplicative is more like
        #       block-Gauss-Seidel.
        #
        # So ASM scales without dependencies (with slower convergence), but MSM converges
        # faster with the constraint that some sort of coloring procedure may be needed
        # to identify independent regions
        #
        if not additiveSchwartz:
            interface_constraints[iSubDom-1] = popt.copy()

    if showplot:
        plt.legend()
        plt.draw()

    return POptDomain

##############################################################################################

# # Let us try adaptivity in the entire domain: 1 sub-D
# U   = np.linspace(0.0, 1.0, nPoints)
# inc = 1.0/nPoints
# t   = np.linspace(inc, 1.0 - inc, nControlPoints - degree - 1)
# spl = LSQUnivariateSpline(x, y, t, k=degree)
# P        = spl.get_coeffs()
# W = np.ones_like(P)
# knots    = spl.get_knots()
# knots    = np.concatenate(([knots[0]] * degree, knots, [knots[-1]] * degree))
# # print len(newconstraints[:,0])
# print len(W)
# print len(t)
# print len(knots)
# pad = adaptive(1, None, None, knots, U, x, y, MAX_ERR=1e-2, split_all=True)


# # Now implement the same for all subdmomains with just one pass
# adaptiveP, adaptiveW, adaptiveT = compute_adaptive_fit_for_subdomains(nSubDomains, degree, nControlPoints, x, y, showplot=True)
# print adaptiveP


# In[6]:


# Let do the recursive iterations
# Use the previous MAK solver solution as initial guess; Could do something clever later
interface_constraints_obj = dict()
interface_constraints = []

interface_constraints_obj['P']=[[] for i in range(nSubDomains)]
interface_constraints_obj['W']=[[] for i in range(nSubDomains)]
interface_constraints_obj['T']=[[] for i in range(nSubDomains)]

# interface_constraints = interface_constraints_obj['P']

# print "Initial condition data: ", interface_constraints
errors = np.zeros([nmaxiter,2]) # Store L2, Linf errors as function of iteration
adaptiveP = []
adaptiveW = []
adaptiveT = []

for iterIdx in range(nmaxiter):

    print ("\n---- Starting Iteration: %d ----" % iterIdx)
    adaptiveP, adaptiveW, adaptiveT = compute_adaptive_fit_for_subdomains(nSubDomains, degree, nControlPoints, x, y, showplot=True,
                                                                          maxAdaptErr=max(1e-2/2**iterIdx,maxAbsErr),
                                                                          PAdaptDomain=interface_constraints_obj['P'],
                                                                          WAdaptDomain=interface_constraints_obj['W'],
                                                                          KnotAdaptDomains=interface_constraints_obj['T']
                                                                          )
    interface_constraints_obj['P']=adaptiveP[:]
    interface_constraints_obj['W']=adaptiveW[:]
    interface_constraints_obj['T']=adaptiveT[:]

    # Call our MFA functional fitting routine for all the subdomains
    # for icons in range(3):
    #     newconstraints = compute_mak95_fit_with_constraints(nSubDomains, degree, nControlPoints, x, y,
    #                                                         interface_constraints_obj,
    #                                                         additiveSchwartz=useAdditiveSchwartz,
    #                                                         useDerivatives=useDerivativeConstraints,
    #                                                         showplot=True, nosolveplot=False,
    #                                                         solver=solverscheme)
    #     interface_constraints_obj['P'] = newconstraints[:]

    # REMOVE
    # newconstraints = adaptiveP[:]
    #
    # if len(interface_constraints) == 0:
    #      interface_constraints = adaptiveP[:]

    # Let's compute the delta in the iterate
    # iteratedelta = (np.array(newconstraints)-np.array(interface_constraints))
    
    # interface_constraints_obj['P'] = newconstraints[:] # - 0.5 * iteratedelta
    # interface_constraints = newconstraints[:]

    # Compute the error and convergence
#     return np.array([(np.sum(iteratedelta[u]**2)) for u,_ in enumerate(iteratedelta)])

    # errors[iterIdx, 0] = math.sqrt(np.sum(np.array([(np.sum(iteratedelta[u]**2)) for u,_ in enumerate(iteratedelta)])))
    # errors[iterIdx, 1] = np.max(np.array([(np.sum(iteratedelta[u]**2)) for u,_ in enumerate(iteratedelta)]))
    # if errors[iterIdx, 1] < 1e-16:
    #   errors[iterIdx, 0] = 1e-24
    #   errors[iterIdx, 1] = 1e-24

    # print "\nIteration: ", iterIdx, " -- L2 error: ", errors[iterIdx, 0], ", Linf error: ", errors[iterIdx, 1]
    # if iterIdx == 0:
    #     print "\tUnconstrained solution: ", newconstraints
    # else:
    #     print "\tIterate delta: ", iteratedelta
    # print "------------------------\n"

    # Check for convergence
    # if errors[iterIdx, 0] < maxRelErr:
    #     break

newconstraints = interface_constraints_obj
print ("\tConverged solution: ", newconstraints)
# print newconstraints[0].shape, newconstraints[1].shape, newconstraints[2].shape, newconstraints[3].shape

# print "Error convergence: ", errors
# if iterIdx > 1:
#     plt.figure()
#     xx = range(iterIdx+1)
#     plt.plot(xx, np.log10(errors[xx,0]), 'b--', ms=5, label='L_2 error')
#     plt.plot(xx, np.log10(errors[xx,1]), 'r--', lw=3, label='Linf error')
#     plt.legend()
#     plt.draw()

plt.show()
