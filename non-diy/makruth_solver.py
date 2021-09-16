#!/usr/bin/env python
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
    print('EigValues: ', s)
    M, N = u.shape[0], vh.shape[1]
    if rcond is None:
        rcond = np.finfo(s.dtype).eps * max(M, N)
    tol = np.amax(s) * rcond
    num = np.sum(s > tol, dtype=int)
    Q = vh[num:,:].T.conj()
    return Q

def pieceBezierDer22(P, W, U, T, degree):
    bezierP = list(range(0, len(P)-degree, degree))
    Q = []
    
    for ps in bezierP:
        pp = P[ps:ps+degree+1]
        qq = np.asarray([degree*(pp[i+1]-pp[i])/(T[degree+ps+i+1]-T[ps+i]) for i in range(len(pp)-1)])
        Q.extend(qq)

    return Q

def pieceBezierDDer22(P, W, U, T, degree):
    bezierP = list(range(0, len(P)-degree, degree))
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


def LSQFit_Constrained(idom, N, W, y, U, t, spl, degree, nSubDomains, constraints=None, useDerivatives=False, solver='SLSQP'):
    
    if useDerivatives and len(constraints) > 0:

        splD     = spl.derivative()
        knotsD   = splD.get_knots()
        knotsD   = np.concatenate(([knotsD[0]] * (degree-1), knotsD, [knotsD[-1]] * (degree-1) ))
        PD       = splD.get_coeffs()

        bzD = pieceBezierDer22(constraints[:,idom-1], W, U, t, degree)
        # bzDD = pieceBezierDDer22(bzD, W, U, knotsD, degree-1)
        # print "Subdomain Derivatives (1,2) : ", bzD, bzDD
        if idom < nSubDomains:
            bzDp = pieceBezierDer22(constraints[:,idom], W, U, t, degree)
            # bzDDp = pieceBezierDDer22(bzDp, W, U, knotsD, degree-1)
            # print "Left Derivatives (1,2) : ", bzDp, bzDDp
            # print 'Right derivative error offset: ', ( bzD[-1] - bzDp[0] )
        if idom > 1:
            bzDm = pieceBezierDer22(constraints[:,idom-2], W, U, t, degree)
            # bzDDm = pieceBezierDDer22(bzDm, W, U, knotsD, degree-1)
            # print "Right Derivatives (1,2) : ", bzDm, bzDDm
            # print 'Left derivative error offset: ', ( bzD[0] - bzDm[-1] )

    def ComputeL2Error(P, N, W, y, U, t, degree):
        return np.sum(Error(P, W, y, U, t, degree)**2)

    cons = []
    if solver is 'SLSQP':
        if constraints is not None and len(constraints) > 0:
            if idom > 1:
                # if bzD is not None:
                # print 'Left derivative: ', bzD[0]
                cons.append( {'type': 'eq', 'fun' : lambda x: np.array([ ( x[0] - (constraints[0,idom-1] + constraints[-1,idom-2])/2 ) ])} )
                if useDerivatives:
                    # cons.append( {'type': 'eq', 'fun' : lambda x: np.array([ ( pieceBezierDer22(x, W, U, t, degree)[0] - ( bzD[0] + bzDm[-1] )/2 ) ])} )
                    # cons.append( {'type': 'eq', 'fun' : lambda x: np.array([ ( pieceBezierDer22(x, W, U, t, degree)[0] - ( bzDm[-1]  ) ) ])} )
                    # cons.append( {'type': 'eq', 'fun' : lambda x: np.array([ ( x[1] - x[0] - ( constraints[-1,idom-2] - constraints[-2,idom-2] ) ) ])} )
                    cons.append( {'type': 'eq', 'fun' : lambda x: np.array([ ( x[1] - x[0] - 0.5*( constraints[1,idom-1] - constraints[0,idom-1] + constraints[-1,idom-2] - constraints[-2,idom-2] ) ) ])} )
                    # cons.append( {'type': 'eq', 'fun' : lambda x: np.array([ ( (x[2] - x[1])/(t[degree+1]-t[0]) - (x[1] - x[0])/(t[degree+2]-t[1]) - ( (constraints[-3,idom-2] - constraints[-2,idom-2])/(t[degree+1]-t[0]) - (constraints[-2,idom-2] - constraints[-1,idom-2])/(t[degree+1]-t[0])  ) ) ])} )
                    
                    # cons.append( {'type': 'eq', 'fun' : lambda x: np.array([ ( pieceBezierDDer22(pieceBezierDer22(x, W, U, t, degree), W, U, t, degree)[0] - ( bzDDm[-1]  ) ) ])} )
                    # cons.append( {'type': 'eq', 'fun' : lambda x: np.array([ ( pieceBezierDer22(x, W, U, t, degree)[1] - (pieceBezierDer22(x, W, U, t, degree)[0] + ( bzDm[-1] * (t[degree+len(constraints[:,idom-1])]-t[len(constraints[:,idom-1])])  )) ) ])} )
                    # cons.append( {'type': 'eq', 'fun' : lambda x: np.array([ ( pieceBezierDer22(x, W, U, t, degree)[1] - ( bzDm[-2]  ) ) ])} )
                    # cons.append( {'type': 'eq', 'fun' : lambda x: np.array([ ( x[1] - x[0] - ( bzDm[-1] * (t[degree+len(constraints[:,idom-1])]-t[len(constraints[:,idom-1])])  ) ) ])} )
                    
                    
                    # print 'Left delx: ', (t[degree+2]-t[1]), ' and ', (t[degree+1]-t[0])
            if idom < nSubDomains:
                # if bzD is not None:
                # print 'Right derivative: ', bzD[-1]
                cons.append( {'type': 'eq', 'fun' : lambda x: np.array([ ( x[-1] - (constraints[-1,idom-1] + constraints[0,idom])/2 ) ])} )
                if useDerivatives:
                    # cons.append( {'type': 'eq', 'fun' : lambda x: np.array([ ( pieceBezierDer22(x, W, U, t, degree)[-1] - (bzD[-1] + bzDp[0])/2 ) ])} )
                    # cons.append( {'type': 'eq', 'fun' : lambda x: np.array([ ( pieceBezierDer22(x, W, U, t, degree)[-1] - (bzDp[0]) ) ])} )
                    # cons.append( {'type': 'eq', 'fun' : lambda x: np.array([ ( x[-1] - x[-2] - ( constraints[1,idom] - constraints[0,idom] ) ) ])} )
                    cons.append( {'type': 'eq', 'fun' : lambda x: np.array([ ( x[-1] - x[-2] - 0.5*( constraints[-1,idom-1] - constraints[-2,idom-1] + constraints[1,idom] - constraints[0,idom] ) ) ])} )
                    # cons.append( {'type': 'eq', 'fun' : lambda x: np.array([ ( (x[-3] - x[-2])/(t[-1] - t[-degree-2]) - (x[-2] - x[-1])/(t[-2] - t[-degree-3]) + ( (constraints[1,idom] - constraints[0,idom])/(t[-2] - t[-degree-3]) - (constraints[2,idom] - constraints[1,idom])/(t[-1] - t[-degree-2]) ) ) ])} )
                    
                    # cons.append( {'type': 'eq', 'fun' : lambda x: np.array([ ( pieceBezierDDer22(pieceBezierDer22(x, W, U, t, degree), W, U, t, degree)[-1] - (bzDDp[0]) ) ])} )
                    # cons.append( {'type': 'eq', 'fun' : lambda x: np.array([ ( x[-2] - x[-1] - ( bzDp[0] * (t[degree]-t[0])  ) ) ])} )
                    # cons.append( {'type': 'eq', 'fun' : lambda x: np.array([ ( pieceBezierDer22(x, W, U, t, degree)[-2] - (bzDp[1]) ) ])} )
                    
                    
                    # print 'Right delx: ', (t[-2] - t[-degree-3]), ' and ', (t[-1] - t[-degree-2])

            # initSol = constraints[:,idom-1]
            # print initSol
            res = minimize(ComputeL2Error, np.ones_like(W), method='SLSQP', args=(N, W, y, U, t, degree),
                           constraints=cons, #x0=initSol,
                           options={'disp': False, 'ftol': 1e-6})
        else:

            res = minimize(ComputeL2Error, np.ones_like(W), method='SLSQP', args=(N, W, y, U, t, degree),
                           options={'disp': False, 'ftol': 1e-6})

    else:
        if constraints is not None and len(constraints) > 0:
            if idom > 1:
                print(idom, ': Left constraint ', constraints[0][idom-1], constraints[-1][idom-2], end=' ')
                cons.append( {'type': 'ineq', 'fun' : lambda x: np.array([ ( x[0] - (constraints[0][idom-1] + constraints[-1][idom-2])/2 ) ])} )
            if idom < nSubDomains:
                print(idom, ': Right constraint ', constraints[-1][idom-1] + constraints[0][idom])
                cons.append( {'type': 'ineq', 'fun' : lambda x: np.array([ ( x[-1] - (constraints[-1][idom-1] + constraints[0][idom])/2 ) ])} )

        res = minimize(ComputeL2Error, np.ones_like(W), method='COBYLA', args=(N, W, y, U, t, degree),
                       constraints=cons, #x0=constraints,
                       options={'disp': True, 'tol': 1e-6, 'catol': 1e-2})

    print('[', idom, '] : ', res.message)
    return res.x



def compute_mak95_fit_with_constraints(nSubDomains, degree, nControlPoints, x, y, 
                                        interface_constraints, additiveSchwartz=True, useDerivatives=False, 
                                        showplot=False, nosolveplot=False, 
                                        solver='SLSQP'):

    import matplotlib.pyplot as plt
    import numpy as np
    from cycler import cycler

    nPoints = len(x)
    Dmin = min(x)
    Dmax = max(x)
    nControlPointSpans = nControlPoints - 1
    nInternalKnotSpans = nControlPointSpans - degree + 1
    nPointsPerSubD = int(nPoints / nSubDomains)
    if nPoints % nSubDomains > 0:
        print("[WARNING]: The total number of points do not divide equally with subdomains")

    # The variable `additiveSchwartz` controls whether we use
    # Additive vs Multiplicative Schwartz scheme for DD resolution

    if showplot:
        plt.figure()
        if nSubDomains > 5:
            plt.rc('axes', prop_cycle=(cycler('color', ['r', 'g', 'b', 'y', 'c']) + cycler('linestyle', ['-','--',':','-.',''])))
        plt.plot(x, y, 'b-', ms=5, label='Input')

    if len(interface_constraints) > 0:
        POptDomain = interface_constraints[:,:].copy()
    else:
        POptDomain = np.zeros([nControlPoints,nSubDomains])

    for iSubDom in range(1,nSubDomains+1):

        domStart = (iSubDom-1) * 1.0 / nSubDomains
        domEnd   = iSubDom * 1.0 / nSubDomains

        inc = (domEnd - domStart) / nInternalKnotSpans
        t   = np.linspace(domStart + inc, domEnd - inc, nInternalKnotSpans - 1)
        U   = np.linspace(domStart, domEnd, nPointsPerSubD)
        dSpan = [(iSubDom-1) * nPointsPerSubD, iSubDom * nPointsPerSubD]

        # print (U, dSpan)

        Dmini = Dmin + (Dmax - Dmin)*domStart
        Dmaxi = Dmin + (Dmax - Dmin)*domEnd

        ySD = y[dSpan[0]:dSpan[1]]

        spl = LSQUnivariateSpline(U, ySD, t, k=degree)

        # get the control points
        knots    = spl.get_knots()
        knots    = np.concatenate(([knots[0]] * degree, knots, [knots[-1]] * degree))
        P        = spl.get_coeffs()

        # Lets compute the control points and weights with Ma & Kruth method
        W = np.ones_like(P)
        if not nosolveplot:
            N = basis(U[np.newaxis,:],degree,knots[:,np.newaxis]).T
            popt = LSQFit_Constrained(iSubDom, N, W, ySD, U, knots, spl, degree, nSubDomains, interface_constraints, useDerivatives, solver)
        else:
            popt = interface_constraints[:,iSubDom-1]

        if showplot:
            pMK = decode(popt, W, U, knots, degree)
            xlocal = U * (Dmax - Dmin) + Dmin
            plt.plot(xlocal, pMK, 'g--', lw=3, label='Decoded')

            coeffs_x = getControlPoints(knots, degree) * (Dmax - Dmin) + Dmin
            if nSubDomains < 5:
                plt.plot(coeffs_x, popt, marker='o', linestyle='--', color=['r','g','b','y','c'][iSubDom-1], label='Control')
            else:
                plt.plot(coeffs_x, popt, marker='o', label='Control')

        if not nosolveplot:
            errorMAK = L2LinfErrors(popt, W, ySD, U, knots, degree)
            print("Subdomain: ", iSubDom, " -- L2 error: ", errorMAK[0], ", Linf error: ", errorMAK[1])

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

    if showplot:
        plot_derivatives(nSubDomains, degree, nControlPoints, x, y, POptDomain)

    return POptDomain



def plot_derivatives(nSubDomains, degree, nControlPoints, x, y, interface_constraints):

    import matplotlib.pyplot as plt
    import numpy as np
    from cycler import cycler

    nPoints = len(x)
    Dmin = min(x)
    Dmax = max(x)
    nControlPointSpans = nControlPoints - 1
    nInternalKnotSpans = nControlPointSpans - degree + 1
    nPointsPerSubD = int(nPoints / nSubDomains)
    if nPoints % nSubDomains > 0:
        print("[WARNING]: The total number of points do not divide equally with subdomains")

    # The variable `additiveSchwartz` controls whether we use
    # Additive vs Multiplicative Schwartz scheme for DD resolution

    POptDomain = np.zeros([nControlPoints-1,nSubDomains])

    plt.figure()
    if nSubDomains > 5:
        plt.rc('axes', prop_cycle=(cycler('color', ['r', 'g', 'b', 'y', 'c']) + cycler('linestyle', ['-','--',':','-.',''])))

    for iSubDom in range(1,nSubDomains+1):

        domStart = (iSubDom-1) * 1.0 / nSubDomains
        domEnd   = iSubDom * 1.0 / nSubDomains

        inc = (domEnd - domStart) / nInternalKnotSpans
        t   = np.linspace(domStart + inc, domEnd - inc, nInternalKnotSpans - 1)
        U   = np.linspace(domStart, domEnd, nPointsPerSubD)
        dSpan = [(iSubDom-1) * nPointsPerSubD, iSubDom * nPointsPerSubD]

        # print [U, x[dSpan]]

        Dmini = Dmin + (Dmax - Dmin)*domStart
        Dmaxi = Dmin + (Dmax - Dmin)*domEnd

        ySD = y[dSpan[0]:dSpan[1]]

        spl = LSQUnivariateSpline(U, ySD, t, k=degree)

        # get the control points

        splD     = spl.derivative()
        knotsD   = splD.get_knots()
        knotsD   = np.concatenate(([knotsD[0]] * (degree-1), knotsD, [knotsD[-1]] * (degree-1) ))
        P        = splD.get_coeffs()

        knots    = spl.get_knots()
        knots    = np.concatenate(([knots[0]] * degree, knots, [knots[-1]] * degree))
        

        # Lets compute the control points and weights with Ma & Kruth method
        W = np.ones_like(P)

        popt = interface_constraints[:,iSubDom-1]
        bzD = pieceBezierDer22(popt, W, U, knots, degree)

        POptDomain[:,iSubDom-1] = np.array(bzD).copy()

        N = basis(U[np.newaxis,:],degree-1,knotsD[:,np.newaxis]).T
        # popt = LSQFit_Constrained(iSubDom, N, W, ySD, U, knotsD, splD, degree-1, nSubDomains)

        xlocal = U * (Dmax - Dmin) + Dmin
        coeffs_x = getControlPoints(knotsD, degree-1) * (Dmax - Dmin) + Dmin
        # plt.plot(coeffs_x, bzD, marker='o', linestyle='--', color=['r','g','b','y','c'][iSubDom-1], label='Control Derivatives')
        pMK = decode_derivative(bzD, W, U, knotsD, degree-1)
        plt.plot(xlocal, pMK, linestyle='--', color=['r','g','b','y','c'][iSubDom-1], label='Decoded Derivatives')

    # exact_der = np.gradient(np.sinc(x+1))
    exact_der = np.gradient(y)
    exact_der = np.max(POptDomain) * exact_der/np.max(exact_der)
    plt.plot(x, exact_der, linestyle='-', color='c', label='Exact Derivatives')
    # print exact_der
    plt.legend()
    plt.draw()

