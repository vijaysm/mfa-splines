
import math
import numpy as np
import scipy

from scipy import linalg, matrix
from scipy.interpolate import LSQUnivariateSpline, UnivariateSpline
from scipy.optimize import minimize, linprog
from scipy.linalg import svd
from quadprog import solve_qp

EPS    = 1e-14
GTOL   = 1e-2
basis  = lambda u,p,T: ((T[:-1]<=u) * (u<=T[1:])).astype(np.float) if p==0 else \
                    ((u - T[:-p]) /(T[p:]  -T[:-p]+EPS))[:-1] * basis(u,p-1,T)[:-1] + \
                    ((T[p+1:] - u)/(T[p+1:]-T[1:-p]+EPS))     * basis(u,p-1,T)[1:]


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

def Error(P, W, y, x, t, degree):
    return decode(P, W, x, t, degree) - y

def lsqFit(N, W, y):
    RN = (N*W)/(np.sum(N*W, axis=1)[:,np.newaxis])
    LHS = np.matmul(RN.T,RN)
    RHS = np.matmul(RN.T, y)
    return linalg.lstsq(LHS, RHS)[0]

def nullspace(A, eps=1e-15):
    u, s, vh = scipy.linalg.svd(A)
    null_mask = (s <= eps)
    null_space = scipy.compress(null_mask, vh, axis=0)
    return scipy.transpose(null_space)

def null_space(A, rcond=None):
    u, s, vh = svd(A, full_matrices=True)
    print ('EigValues: ', s)
    M, N = u.shape[0], vh.shape[1]
    if rcond is None:
        rcond = np.finfo(s.dtype).eps * max(M, N)
    tol = np.amax(s) * rcond
    num = np.sum(s > tol, dtype=int)
    Q = vh[num:,:].T.conj()
    return Q

def quadprog_solve_qp(y, N): # , P, q, G=None, h=None, A=None, b=None
    Nt  = N.T
    NtN  = np.matmul(Nt,N)
    NtNi = linalg.inv(NtN)
    NtQN = np.matmul(np.matmul(Nt, np.diag(y)), N)
    NtQ2N= np.matmul(np.matmul(Nt, np.diag(y**2)), N)
    
    M = NtQ2N - np.matmul(np.matmul(NtQN,NtNi),NtQN)

#     print M
    W = null_space(M)
#     print 'W = ', W

    return W

    # G = None
    # h = None
    A = None
    b = None
    P = np.matmul(M.T, M)
    q = np.zeros(len(M))
    A = M
    b = np.zeros(len(M))
    G = -M.T #np.identity(len(M))
    h = np.zeros(len(M))


    qp_G = .5 * (P + P.T)   # make sure P is symmetric
    qp_a = -q
    if A is not None:
        qp_C = -np.vstack([A, G]).T
        qp_b = -np.hstack([b, h])
        meq = A.shape[0]
    else:  # no equality constraint
        qp_C = -G.T
        qp_b = -h
        meq = 0
    
    return solve_qp(qp_G, qp_a, qp_C, qp_b, meq)[0]

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
    L2Err = math.sqrt(np.sum(E**2)/len(E))
    return [L2Err, LinfErr]


import collections

def rint(x):
    return int(round(x))

def toHomogeneous(P, W):
    return np.hstack( ( (P*W)[:,np.newaxis], W[:,np.newaxis]) )

def fromHomogeneous(PW):
    P = PW[:,0]
    W = PW[:,1]
    return P/W, W

def knotInsert(T, degree, xu, us, splitUs=True, r=1):
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

def knotRefine(P, W, T, u, degree, yl, E, r=1, find_all=True, MAX_ERR = 1e-2):
    # yRange = yl.max()-yl.min()
    # NSE = (E/yRange)**2
    NSE = np.abs(E)/np.linalg.norm(E, ord=np.inf)
    if(NSE.mean()<=MAX_ERR):
        return T, []
    us = u[np.where( NSE >=(MAX_ERR) )[0] if find_all else np.argmax(NSE)]

    return knotInsert(T, degree, u, us, r=r)

def deCasteljau(P, W, T, u, degree, k, r=1):
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



def MaKruth95(y, N):
    Nt  = N.T
    NtN  = np.matmul(Nt,N)
    NtNi = linalg.inv(NtN)
    NtQN = np.matmul(np.matmul(Nt, np.diag(y)), N)
    NtQ2N= np.matmul(np.matmul(Nt, np.diag(y**2)), N)
    
    M = NtQ2N - np.matmul(np.matmul(NtQN,NtNi),NtQN)
    Mcond = np.linalg.cond(M)
    w, v = linalg.eigh(M)
    
    if np.all(v[:,0] > 0):
        return v[:,0], Mcond
    elif np.all(v[:,0] < 0):
        return -v[:,0], Mcond
    elif False:
        print ('Using custom minimizer')
        minwgts = ComputeWeightsMinimized(y,N)
        print ('Optimized weights = ', minwgts)
        return minwgts, Mcond
    else:
        w0 = v[:,0].copy()
        for i in range(1,v.shape[1]):
            V = v[:,:i+1]
            #V[abs(V)<1e-12] = 0
            # A = abs(V)
            A = V[:,:].copy()
            c = A.shape[0]
            ub = -np.ones(c)
            
            res = linprog(np.ones(A.shape[1]), A_ub=A, b_ub=ub, options={"maxiter": 100, 'disp': False}, method='simplex')
            if res.success:
                W = np.matmul(V, res.x)
                if np.all(W < 0):
                    W = -W
                if np.all(W > 0):
                    print ("Successful linear programming solve from", i+1, "eigenvectors")
                    return W, Mcond
    print ("Failed to find a solution to the linear programming problem combining eigenvectors")
    return np.ones(M.shape[0]), Mcond


def compute_mak95_fit(nSubDomains, degree, nControlPoints, x, y):
    import matplotlib.pyplot as plt
    import numpy as np

    nPoints = len(x)
    Dmin = min(x)
    Dmax = max(x)
    nControlPointSpans = nControlPoints - 1
    nInternalKnotSpans = nControlPointSpans - degree + 1
    print ("nInternalKnotSpans: ", nInternalKnotSpans)

    plt.figure()
    plt.plot(x, y, 'b-', ms=5, label='Input')
    POptDomain = np.zeros([nInternalKnotSpans+3,nSubDomains])

    for iSubDom in range(1,nSubDomains+1):

        domStart = (iSubDom-1) * 1.0 / nSubDomains
        domEnd   = iSubDom * 1.0 / nSubDomains

        inc = (domEnd - domStart) / nInternalKnotSpans
        t   = np.linspace(domStart + inc, domEnd - inc, nInternalKnotSpans - 1)
        U   = np.linspace(domStart, domEnd, nPoints/nSubDomains)

        print ("iSubdomain: ", iSubDom, " Total: ", nPoints, " NSubdomain: ", nSubDomains)
        Dmini = Dmin + (Dmax - Dmin)*domStart
        Dmaxi = Dmin + (Dmax - Dmin)*domEnd
        print ("Subdomain span: ", Dmini, Dmaxi)

        pSpan = int(len(y) / nSubDomains)
        dSpan = range( (iSubDom-1) * pSpan, iSubDom * pSpan )
        ySD = y[dSpan]

        spl = LSQUnivariateSpline(U, ySD, t, k=degree)

        # get the control points
        knots    = spl.get_knots()
        knots    = np.concatenate(([knots[0]] * degree, knots, [knots[-1]] * degree))
        P        = spl.get_coeffs()
        coeffs_x = getControlPoints(knots, degree) * (Dmax - Dmin) + Dmin

        # Lets compute the control points and weights with Ma & Kruth method
        N = basis(U[np.newaxis,:],degree,knots[:,np.newaxis]).T
        W = np.ones_like(P)
#         popt = lsqFit(N, W, ySD)
    #     errorMAK = L2LinfErrors(popt, W, ySD, U, knots, degree)
    #     print errorMAK[1]

        W, Mcond = MaKruth95(ySD, N)
        popt = lsqFit(N, W, ySD)
        print ("Ma&Kruth Weights:", W)
    #     print "M matrix condition number:", Mcond
    #     errorMAK = L2LinfErrors(popt, W, ySD, U, knots, degree)
    #     print errorMAK[1]

        pMK = decode(popt, W, U, knots, degree)
        plt.plot(x[dSpan], pMK, 'g--', lw=3, label='Decoded')
        plt.plot(coeffs_x, popt, marker='o', linestyle='--', color=['r','g','b','y','c'][iSubDom-1], label='Control')

        # print "Knots: ", knots
    #     print "Normalized knots:", 2 * (knots) / (Dmaxi - Dmini)
    #     print "Control point x:", coeffs_x
    #     print "Control point y:", pMK
    #     print "Subdomain: ", iSubDom, " -- Sum of squared residuals of the spline approximation", spl.get_residual()
        errorMAK = L2LinfErrors(popt, W, ySD, U, knots, degree)
        print ("Subdomain: ", iSubDom, " -- L2 error: ", errorMAK[0], ", Linf error: ", errorMAK[1])
        print ("-----------\n")
        
        # Store the data for subdomain
        POptDomain[:,iSubDom-1] = popt.copy()

    plt.legend()
    plt.draw()
    
    return POptDomain


def ComputeWeightsMinimized(y, N):
    def ComputeObjective(P, MtM):
        xv = P.T * MtM * P
        return np.sum(xv)

    Nt  = N.T
    NtN  = np.matmul(Nt,N)
    NtNi = linalg.inv(NtN)
    NtQN = np.matmul(np.matmul(Nt, np.diag(y)), N)
    NtQ2N= np.matmul(np.matmul(Nt, np.diag(y**2)), N)
    
    M = NtQ2N - np.matmul(np.matmul(NtQN,NtNi),NtQN)
    MtM = np.matmul(M.T, M)

    solver='SLSQP1'
    bnds = np.zeros([len(M), 2])
    bnds[:,0] = 1e-5
    bnds[:,1] = 100

    cons = []
    if solver is 'SLSQP':

        cons.append( {'type': 'eq', 'fun' : lambda x: len(M)**2 - np.sum(x) } )
        res = minimize(ComputeObjective, np.ones(len(M)), method='SLSQP', args=(MtM),
                       constraints=cons, bounds=bnds, #x0=constraints,
                       options={'disp': True, 'ftol': 1e-6})
        resvec = res.x

    elif solver is 'COBYLA':
        cons.append( {'type': 'ineq', 'fun' : lambda x: len(M)**2 - np.sum(x) } )
        res = minimize(ComputeObjective, np.ones(len(M)), method='COBYLA', args=(MtM),
                       constraints=cons, bounds=bnds, #x0=constraints,
                       options={'disp': True, 'tol': 1e-6, 'catol': 1e-2})
        resvec = res.x
    elif solver is 'BFGS':
        res = minimize(ComputeObjective, np.ones(len(M)), method='BFGS', args=(MtM),
                       bounds=bnds, #x0=constraints,
                       options={'disp': True, 'gtol': 1e-6})
        resvec = res.x
    else:
        resvec = quadprog_solve_qp(y, N)


    return resvec



def LSQFit_Constrained(idom, N, W, y, U, t, degree, nSubDomains, constraints=None, solver='SLSQP'):
    def ComputeL2Error(P, N, W, y, U, t, degree):
        return np.sum(Error(P, W, y, U, t, degree)**2)

#         ErrVec = np.array([(np.sum(basis(x[u],degree,t) *P*W)/(np.sum(basis(x[u],degree,t)*W))) for u,_ in enumerate(x)]) - y
#         print 'basis: ', N
#         print 'control: ', P
#         print 'weights: ', W
#         print 'numerator: ', N*P*W
#         print 'denominator: ', N*W
#         ErrVec = (N*P*W)/(N*W) - y
#         return np.sum(ErrVec**2)

    cons = []
    if solver is 'SLSQP':
        if constraints is not None:
            if idom > 1:
                cons.append( {'type': 'eq', 'fun' : lambda x: np.array([ ( x[0] - (constraints[0][idom-1] + constraints[-1][idom-2])/2 ) ])} )
            if idom < nSubDomains:
                cons.append( {'type': 'eq', 'fun' : lambda x: np.array([ ( x[-1] - (constraints[-1][idom-1] + constraints[0][idom])/2 ) ])} )

        res = minimize(ComputeL2Error, np.ones_like(W), method='SLSQP', args=(N, W, y, U, t, degree),
                       constraints=cons, #x0=constraints,
                       options={'disp': False, 'ftol': 1e-6})

    else:
        if constraints is not None:
            if idom > 1:
                print (idom, ': Left constraint ', constraints[0][idom-1], constraints[-1][idom-2])
                cons.append( {'type': 'ineq', 'fun' : lambda x: np.array([ ( x[0] - (constraints[0][idom-1] + constraints[-1][idom-2])/2 ) ])} )
            if idom < nSubDomains:
                print (idom, ': Right constraint ', constraints[-1][idom-1] + constraints[0][idom])
                cons.append( {'type': 'ineq', 'fun' : lambda x: np.array([ ( x[-1] - (constraints[-1][idom-1] + constraints[0][idom])/2 ) ])} )

        res = minimize(ComputeL2Error, np.ones_like(W), method='COBYLA', args=(N, W, y, U, t, degree),
                       constraints=cons, #x0=constraints,
                       options={'disp': True, 'tol': 1e-6, 'catol': 1e-2})

    return res.x



def compute_mak95_fit_with_constraints(nSubDomains, degree, nControlPoints, x, y, interface_constraints, additiveSchwartz=True, showplot=False, nosolveplot=False, solver='SLSQP'):

    import matplotlib.pyplot as plt
    import numpy as np

    nPoints = len(x)
    Dmin = min(x)
    Dmax = max(x)
    nControlPointSpans = nControlPoints - 1
    nInternalKnotSpans = nControlPointSpans - degree + 1
    # print "nInternalKnotSpans: ", nInternalKnotSpans

    # The variable `additiveSchwartz` controls whether we use
    # Additive vs Multiplicative Schwartz scheme for DD resolution

    if showplot:
        plt.figure()
        plt.plot(x, y, 'b-', ms=5, label='Input')

    POptDomain = interface_constraints[:,:].copy()

    for iSubDom in range(1,nSubDomains+1):

        domStart = (iSubDom-1) * 1.0 / nSubDomains
        domEnd   = iSubDom * 1.0 / nSubDomains

        inc = (domEnd - domStart) / nInternalKnotSpans
        t   = np.linspace(domStart + inc, domEnd - inc, nInternalKnotSpans - 1)
        U   = np.linspace(domStart, domEnd, nPoints/nSubDomains)

        Dmini = Dmin + (Dmax - Dmin)*domStart
        Dmaxi = Dmin + (Dmax - Dmin)*domEnd

        pSpan = int(len(y) / nSubDomains)
        dSpan = range( (iSubDom-1) * pSpan, iSubDom * pSpan )
        # dSpan = range( (iSubDom-1) * len(y) / nSubDomains, iSubDom * len(y) / nSubDomains )
        ySD = y[dSpan]

        spl = LSQUnivariateSpline(U, ySD, t, k=degree)

        # get the control points
        knots    = spl.get_knots()
        knots    = np.concatenate(([knots[0]] * degree, knots, [knots[-1]] * degree))
        P        = spl.get_coeffs()

        # Lets compute the control points and weights with Ma & Kruth method
        W = np.ones_like(P)
        if not nosolveplot:
            N = basis(U[np.newaxis,:],degree,knots[:,np.newaxis]).T
#             popt = lsqFit(N, W, ySD)
            popt = LSQFit_Constrained(iSubDom, N, W, ySD, U, knots, degree, nSubDomains, interface_constraints, solver)
#            popt = LSQFit_Constrained(iSubDom, N, W, ySD, U, knots, degree, nSubDomains)
        else:
            popt = interface_constraints[:,iSubDom-1]

#         W, Mcond = MaKruth95_Constrained(iSubDom, ySD, N, interface_constraints)
#         popt = LSQFit_Constrained(iSubDom, N, W, ySD, U, knots, degree, nSubDomains, interface_constraints)

        if showplot:
            pMK = decode(popt, W, U, knots, degree)
            plt.plot(x[dSpan], pMK, 'g--', lw=3, label='Decoded')
            coeffs_x = getControlPoints(knots, degree) * (Dmax - Dmin) + Dmin
            plt.plot(coeffs_x, popt, marker='o', linestyle='--', color=['r','g','b','y','c'][iSubDom-1], label='Control')

        if not nosolveplot:
            errorMAK = L2LinfErrors(popt, W, ySD, U, knots, degree)
            print ("Subdomain: ", iSubDom, " -- L2 error: ", errorMAK[0], ", Linf error: ", errorMAK[1])

        POptDomain[:,iSubDom-1] = popt.copy()

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
            interface_constraints[:,iSubDom-1] = popt.copy()

    if showplot:
        plt.legend()
        plt.draw()

    return POptDomain, t, W

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


def mymain():
# --- set problem input parameters here ---
    Dmin           = -4.
    Dmax           = 4.
    nPoints        = 480
    scale          = 1
    sincFunc       = True
# ------------------------------------------

    if sincFunc:
    	x = np.linspace(Dmin, Dmax, nPoints)
    #     y = scale * np.sinc(x+1)
    	y = scale * np.sin(math.pi * x/4)
    else:
    	y = np.fromfile("s3d.raw", dtype=np.float64) #
    	nPoints = y.shape[0]
    	x = np.linspace(Dmin, Dmax, nPoints)

