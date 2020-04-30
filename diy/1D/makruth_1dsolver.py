import sys, math
import numpy as np
import scipy
import diy
from matplotlib import pyplot as plt

from scipy import linalg, matrix
from scipy.interpolate import LSQUnivariateSpline, UnivariateSpline, interp1d
from scipy.optimize import minimize

from makruth_solver import basis, getControlPoints, decode, Error, L2LinfErrors
from makruth_solver import knotInsert, knotRefine, deCasteljau, pieceBezierDer22

plt.style.use(['seaborn-whitegrid'])
# plt.style.use(['ggplot'])
# plt.style.use(['classic'])
params = {"ytick.color" : "b",
          "xtick.color" : "b",
          "axes.labelcolor" : "b",
          "axes.edgecolor" : "b"}
plt.rcParams.update(params)

# --- set problem input parameters here ---
Dmin           = -4.
Dmax           = 4.
nPoints        = 501
nSubDomains    = 4
degree         = 3
nControlPoints = (3*degree + 1) #minimum number of control points
sincFunc       = True
scale          = 1
# ------------------------------------------
# Solver parameters
solverscheme   = 'SLSQP' # [SLSQP, COBYLA]
useAdditiveSchwartz = True
useDerivativeConstraints = 0

maxAbsErr      = 1e6
maxRelErr      = 1e9
# AdaptiveStrategy = 'extend'
AdaptiveStrategy = 'reset'


if sincFunc:
    x = np.linspace(Dmin, Dmax, nPoints)
    y = scale * np.sinc(x)
    # y = scale * np.sin(math.pi * x/4)
else:
#     y = np.fromfile("data/s3d.raw", dtype=np.float64) #
    y = np.fromfile("data/nek5000.raw", dtype=np.float64) #
    nPoints = y.shape[0]
    x = np.linspace(Dmin, Dmax, nPoints)

# if nPoints % nSubDomains > 0:
#     print ( "[ERROR]: The total number of points do not divide equally with subdomains" )
#     sys.exit(1)

mpl_fig = plt.figure()
plt.plot(x, y, 'r-', ms=2)

# Initialize DIY
w = diy.mpi.MPIComm()           # world

#------------------------------------
EPS    = 1e-14
basis  = lambda u,p,T: ((T[:-1]<=u) * (u<=T[1:])).astype(np.float) if p==0 else ((u - T[:-p]) /(T[p:]  -T[:-p]+EPS))[:-1] * basis(u,p-1,T)[:-1] + ((T[p+1:] - u)/(T[p+1:]-T[1:-p]+EPS))     * basis(u,p-1,T)[1:]


def lsqFit(N, W, y, U, t, degree):
    RN = (N*W)/(np.sum(N*W, axis=1)[:,np.newaxis])
    LHS = np.matmul(RN.T,RN)
    RHS = np.matmul(RN.T, y)
    return linalg.lstsq(LHS, RHS)[0]

def lsqFitWithCons(N, W, ysl, U, t, degree, constraints=[], continuity=0):
    def l2(P, W, ysl, U, t, degree):
        return np.sum(Error(P, W, ysl, U, t, degree)**2)

    res = minimize(l2, np.ones_like(W), method='SLSQP', args=(W, ysl, U, t, degree), 
                   constraints=constraints,
                    options={'disp': True})
    return res.x

def LSQFit_Constrained(idom, N, W, ysl, U, t, degree, nSubDomains, constraintsAll=None, useDerivatives=0, solver='SLSQP'):

    if constraintsAll is not None:
        constraints = np.array(constraintsAll['P'])
        knotsAll = np.array(constraintsAll['T'])
        weightsAll = np.array(constraintsAll['W'])
        # print ('Constraints for Subdom = ', idom, ' is = ', constraints)
    else:
        print ('Constraints are all null. Solving unconstrained.')

    if useDerivatives > 0 and constraints is not None and len(constraints) > 0:

        bzD = pieceBezierDer22(constraints[1], weightsAll[1], U, knotsAll[1], degree)
        # bzDD = pieceBezierDDer22(bzD, W, U, knotsD, degree-1)
        #print ("Subdomain Derivatives (1,2) : ", bzD)
        if idom < nSubDomains:
            bzDp = pieceBezierDer22(constraints[2], weightsAll[2], U, knotsAll[2], degree)
            # bzDDp = pieceBezierDDer22(bzDp, W, U, knotsD, degree-1)
            #print ("Left Derivatives (1,2) : ", bzDp )
            # print ('Right derivative error offset: ', ( bzD[-1] - bzDp[0] ) )
        if idom > 1:
            print('Right derivative shjapes', constraints[0], weightsAll[0], knotsAll[0])
            bzDm = pieceBezierDer22(constraints[0], weightsAll[0], U, knotsAll[0], degree)
            # bzDDm = pieceBezierDDer22(bzDm, W, U, knotsD, degree-1)
            #print ("Right Derivatives (1,2) : ", bzDm )
            # print ('Left derivative error offset: ', ( bzD[0] - bzDm[-1] ) )

    def ComputeL2Error0(P, N, W, ysl, U, t, degree):
        E = np.sum(Error(P, W, ysl, U, t, degree)**2)/len(P)
        return math.sqrt(E)

    def ComputeL2Error(P, N, W, ysl, U, t, degree):
        RN = (N*W)/(np.sum(N*W, axis=1)[:,np.newaxis])
        E = (RN.dot(P) - ysl)
        return math.sqrt(np.sum(E**2)/len(E))

    def print_iterate(P, state):

        print('Iteration %d: max error = %f' % (self.globalIterationNum, state.maxcv))
        self.globalIterationNum += 1
        return False

    cons = []
    if solver is 'SLSQP':
        if constraints is not None and len(constraints) > 0:
            if idom > 1:
                if useDerivatives >= 0:
#                     cons.append( {'type': 'eq', 'fun' : lambda x: np.array([ ( x[0] - (constraints[0][-1]) ) ])} )
                    print('Left delx: ', constraints[1][0], constraints[0][0], (constraints[1][0]-constraints[0][0]))
                    cons.append( {'type': 'eq', 'fun' : lambda x: np.array([ ( x[0] - (constraints[1][0] + constraints[0][0])/2 ) ])} )
                    if useDerivatives > 0 and len(knotsAll[0]):
                        cons.append( {'type': 'eq', 'fun' : lambda x: np.array([ ( pieceBezierDer22(x, W, U, t, degree)[0] - ( bzD[0] + bzDm[-1] )/2 ) ])} )
                        # cons.append( {'type': 'eq', 'fun' : lambda x: np.array([ ( pieceBezierDer22(x, W, U, t, degree)[0] - ( bzDm[-1]  ) ) ])} )
                        # cons.append( {'type': 'eq', 'fun' : lambda x: np.array([ ( (x[1] - x[0])/(knotsAll[idom-1][degree+1]-knotsAll[idom-1][0]) - ( constraints[idom-2][-1] - constraints[idom-2][-2] )/(knotsAll[idom-2][-degree-2] - knotsAll[idom-2][-1]) ) ])} )
                        if useDerivatives > 1:
                            cons.append( {'type': 'eq', 'fun' : lambda x: np.array([ ( (x[2] - x[1])/(knotsAll[1][degree+2]-knotsAll[1][1]) - (x[1] - x[0])/(knotsAll[1][degree+1]-knotsAll[1][0]) - ( (constraints[0][-3] - constraints[0][-2])/(knotsAll[0][-3]-knotsAll[0][-degree-2]) - (constraints[0][-2] - constraints[0][-1])/(knotsAll[0][-2]-knotsAll[0][-degree-3])  ) ) ])} )
                        
                        # cons.append( {'type': 'eq', 'fun' : lambda x: np.array([ ( x[1] - x[0] - 0.5*( constraints[idom-1][1] - constraints[idom-1][0] + constraints[idom-2][-1] - constraints[idom-2][-2] ) ) ])} )

                    # print 'Left delx: ', (knotsAll[idom-1][degree+1]-knotsAll[idom-1][0]), ' and ', (knotsAll[idom-1][degree+2]-knotsAll[idom-1][1])
            if idom < nSubDomains:
                if useDerivatives >= 0:
#                     cons.append( {'type': 'eq', 'fun' : lambda x: np.array([ ( x[-1] - (constraints[2][0]) ) ])} )
                    print('Right delx: ', constraints[1][-1], constraints[2][0], (constraints[1][-1]-constraints[2][0]))
                    cons.append( {'type': 'eq', 'fun' : lambda x: np.array([ ( x[-1] - (constraints[1][-1] + constraints[2][0])/2 ) ])} )
                    if useDerivatives > 0 and len(knotsAll[2]):
                        cons.append( {'type': 'eq', 'fun' : lambda x: np.array([ ( pieceBezierDer22(x, W, U, t, degree)[-1] - (bzD[-1] + bzDp[0])/2 ) ])} )
                        # cons.append( {'type': 'eq', 'fun' : lambda x: np.array([ ( pieceBezierDer22(x, W, U, t, degree)[-1] - (bzDp[0]) ) ])} )
                        # cons.append( {'type': 'eq', 'fun' : lambda x: np.array([ ( (x[-1] - x[-2])/(knotsAll[idom-1][-degree-2] - knotsAll[idom-1][-1]) - ( constraints[idom][1] - constraints[idom][0] )/(knotsAll[idom][degree+1] - knotsAll[idom][0]) ) ])} )
                        if useDerivatives > 1:
                            cons.append( {'type': 'eq', 'fun' : lambda x: np.array([ ( (x[-3] - x[-2])/(knotsAll[1][-2] - knotsAll[1][-degree-3]) - (x[-2] - x[-1])/(knotsAll[1][-1] - knotsAll[1][-degree-2]) + ( (constraints[2][1] - constraints[2][0])/(knotsAll[2][degree+1] - knotsAll[2][0]) - (constraints[2][2] - constraints[2][1])/(knotsAll[2][degree+2] - knotsAll[2][1]) ) ) ])} )
                        # cons.append( {'type': 'eq', 'fun' : lambda x: np.array([ ( x[-1] - x[-2] - 0.5*( constraints[idom-1][-1] - constraints[idom-1][-2] + constraints[idom][1] - constraints[idom][0] ) ) ])} )
                    
                    # print 'Right delx: ', (knotsAll[idom-1][-2] - knotsAll[idom-1][-degree-3]), ' and ', (knotsAll[idom-1][-1] - knotsAll[idom-1][-degree-2])

            initSol = constraints[1][:]
#             initSol = np.ones_like(W)

#             print len(initSol), len(W), len(ysl), len(U), len(t)
#             E = np.sum(Error(initSol, W, ysl, U, t, degree)**2)
#             print "unit error = ", E
            res = minimize(ComputeL2Error, x0=initSol, method='SLSQP', args=(N, W, ysl, U, t, degree),
                           constraints=cons, #callback=print_iterate,
                           options={'disp': True, 'ftol': 1e-10, 'iprint': 1, 'maxiter': 1000})
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



def adaptive(iSubDom, interface_constraints_obj, u, xl, yl, strategy='reset', r=1, MAX_ERR=1e-2, MAX_ITER=5, split_all=True):
    splitIndeces = []
    r = min(r,degree) #multiplicity can not be larger than degree
    nPointsPerSubD = nPoints / nSubDomains

    T = interface_constraints_obj['T'][1]
    if len(interface_constraints_obj['P']):
        P = interface_constraints_obj['P'][1]
        W = interface_constraints_obj['W'][1]

    if len(P) == 0:
        W = np.ones(len(T) - 1 - degree)
        N = basis(u[np.newaxis,:],degree,T[:,np.newaxis]).T
        P = lsqFit(N, W, yl, u, T, degree)

    for iteration in range(MAX_ITER):
        print (" -- Adaptive iteration: ", iteration)
        E = np.abs(Error(P, W, yl, u, T, degree))
        Tnew,splitIndeces = knotRefine(P, W, T, u, degree, yl, E, r, MAX_ERR=MAX_ERR, find_all=split_all)
        if ((len(T)==len(Tnew)) or len(T)-degree-1 > nPointsPerSubD) and not (iteration == 0) :
            break
        
        if strategy == 'extend' and ~split_all:   #only use when coupled with a solver
            k = splitIndeces[0]
            u = Tnew[k+1]
            P,W = deCasteljau(P, W, T, u, k, r)
        elif strategy == 'reset':
            Tnew = np.sort(Tnew)
            W = np.ones(len(Tnew) - 1 - degree)
            N = basis(u[np.newaxis,:],degree,Tnew[:,np.newaxis]).T
            # P = lsqFit(N, W, yl, u, Tnew, degree)
#             P = lsqFitWithCons(N, W, yl, u, Tnew, degree)

#             if len(interface_constraints_obj['P'][iSubDom - 1]) > 0 and len(interface_constraints_obj['P'][iSubDom]) > 0 and len(interface_constraints_obj['P'][iSubDom - 2]) > 0:
            if len(interface_constraints_obj['P'][1]) > 0:
                # Interpolate or project the data to new Knot locations: From (P, T) to (Pnew, Tnew)
                coeffs_x = getControlPoints(T, degree)
                coeffs_xn = getControlPoints(Tnew, degree)
                PnewFn = interp1d(coeffs_x ,P, kind='linear') #, kind='cubic')
                Pnew = PnewFn(coeffs_xn)
                # print 'coeffs_x = ', [coeffs_x, coeffs_xn, P, Pnew]
                interface_constraints_obj['P'][1]=Pnew[:]
                interface_constraints_obj['W'][1]=W[:]
                interface_constraints_obj['T'][1]=Tnew[:]
                # if iSubDom < nSubDomains:
                #     print ('Constraints for left-right Subdom = ', iSubDom, ' is = ', [Pnew, interface_constraints_obj['P'][iSubDom]])
                # else:
                #     print ('Constraints for right-left Subdom = ', iSubDom, ' is = ', [interface_constraints_obj['P'][iSubDom-2], Pnew] )

                print('Solving the boundary-constrained LSQ problem')
                P = LSQFit_Constrained(iSubDom, N, W, yl, u, Tnew, degree, nSubDomains, 
					interface_constraints_obj, 
					useDerivativeConstraints, 'SLSQP')
            else:
                print('Solving the unconstrained LSQ problem')
                P = lsqFit(N, W, yl, u, Tnew, degree)

        else:
            print ("Not Implemented!!")
        
        T = Tnew
        
    return P, W, T, splitIndeces

#--------------------------------------
# Let do the recursive iterations
# The variable `additiveSchwartz` controls whether we use
# Additive vs Multiplicative Schwartz scheme for DD resolution
####

class InputControlBlock:

    def __init__(self, nControlPoints, xb, xl, yl):
        self.nControlPoints = nControlPoints
        self.nControlPointSpans = nControlPoints - 1
        self.nInternalKnotSpans = self.nControlPointSpans - degree + 1
        self.nPointsPerSubD = int(nPoints / nSubDomains)
        self.xbounds = xb
        self.xl = xl
        self.yl = yl
        self.pAdaptive = []
        self.WAdaptive = []
        self.knotsAdaptive = []
        self.xSD = []
        self.ySD = []
        self.Dmini = Dmin
        self.Dmaxi = Dmax
        self.leftconstraint = np.zeros(1)
        self.rightconstraint = np.zeros(1)
        # Allocate for the constraints
        self.interface_constraints_obj = dict()
        self.interface_constraints_obj['P']=[[],[],[]]
        self.interface_constraints_obj['W']=[[],[],[]]
        self.interface_constraints_obj['T']=[[],[],[]]

    def show(self, cp):
        print("Rank: %d, Subdomain %d: Bounds = [%d,%d]" % (w.rank, cp.gid(), self.xbounds.min[0], self.xbounds.max[0]))
        #cp.enqueue(diy.BlockID(1, 0), "abc")

    def plot(self, cp):
#         print(w.rank, cp.gid(), self.core)
        self.pMK = decode(self.pAdaptive, self.WAdaptive, self.xSD, 
                          self.knotsAdaptive * (Dmax - Dmin) + Dmin, degree)
        plt.plot(self.xSD, self.pMK, 'r--', lw=3, label='Decoded')
        coeffs_x = getControlPoints(self.knotsAdaptive, degree) * (Dmax - Dmin) + Dmin
#         print ('Coeffs-x original: ', coeffs_x)
        if nSubDomains < 5:
            plt.plot(coeffs_x, self.pAdaptive, marker='o', linestyle='--', color=['r','g','b','y','c'][cp.gid()], label='Control')
        else:
            plt.plot(coeffs_x, self.pAdaptive, marker='o', label='Control')


    def plot_with_cp(self, cp, cploc, ctrlpts, lgndtitle, indx):
#         print(w.rank, cp.gid(), self.core)
        pMK = decode(ctrlpts, self.WAdaptive, self.xSD, 
                          self.knotsAdaptive * (Dmax - Dmin) + Dmin, degree)
        plt.plot(self.xSD, pMK, linestyle='--', color=['g','b','y','c'][indx], lw=3, label=lgndtitle)

#         if nSubDomains < 5:
#             plt.plot(cploc, ctrlpts, marker='o', linestyle='--', color=['g','b','y','c'][indx], label=lgndtitle+"_Control")
#         else:
#             plt.plot(cploc, ctrlpts, marker='o', label=lgndtitle+"_Control")

            
    def plot_with_cp_and_knots(self, cp, cploc, knots, ctrlpts, weights, lgndtitle, indx):
#         print(w.rank, cp.gid(), self.core)
        print('Plot: shapes = ', ctrlpts.shape[0], cploc.shape[0], knots.shape[0], degree)
        pMK = decode(ctrlpts, weights, self.xSD, 
                          knots * (Dmax - Dmin) + Dmin, degree)
        plt.plot(self.xSD, pMK, linestyle='--', color=['g','b','y','c'][indx], lw=3, label=lgndtitle)

#         if nSubDomains < 5:
#             plt.plot(cploc, ctrlpts, marker='o', linestyle='--', color=['g','b','y','c'][indx], label=lgndtitle+"_Control")
#         else:
#             plt.plot(cploc, ctrlpts, marker='o', label=lgndtitle+"_Control")


    def get_control_points(self):
        return self.pAdaptive

    def get_control_point_locations(self):
        return (getControlPoints(self.knotsAdaptive, degree) * (Dmax - Dmin) + Dmin)

    def get_knot_locations(self):
        return self.knotsAdaptive

    def compute_decoded_errors(self, ctrlpts):
        domStart = 0
        domEnd   = 1.0

        u   = np.linspace(domStart, domEnd, self.nPointsPerSubD)
        err = Error(ctrlpts, self.WAdaptive, self.yl, u, self.knotsAdaptive, degree)

        return err

    def interpolate_spline(self, xnew):
        
        interpOrder = 'linear' # 'linear', 'cubic', 'quintic'
        # Interpolate using coeffs_x and control_points (self.pAdaptive)
        coeffs_x = getControlPoints(self.knotsAdaptive, degree) * (Dmax - Dmin) + Dmin
        #InterpCp = interp1d(coeffs_x, self.pAdaptive, kind=interpOrder)
        InterpCp = Rbf(coeffs_x, self.pAdaptive, function=interpOrder)

        PnewCp = InterpCp(xnew)

        # Interpolate using xSD and pMK
        pMK = decode(self.pAdaptive, self.WAdaptive, self.xSD, 
                      self.knotsAdaptive * (Dmax - Dmin) + Dmin, degree)
        #InterpDec = interp1d(self.xSD, pMK, kind=interpOrder)
        InterpDec = Rbf(self.xSD, pMK, function=interpOrder)
        PnewDec = InterpDec(xnew)
        
        return PnewCp, PnewDec


    def interpolate(self, xnew, tnew):

        r = 1
        Pnew = self.pAdaptive[:]
        knots = self.knotsAdaptive[:]
        W = self.WAdaptive[:]
        print('Original interpolation shapes: ', Pnew.shape[0], knots.shape[0])
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
                if Pnew.shape[0] == knotInd:
                    print('Knot index is same as length of control points')

                Pnew,W = deCasteljau(Pnew[:], W[:], knots, knot, knotInd-1, r)
                knots = np.insert(knots, knotInd, knot)
                # print('New interpolation shapes: ', Pnew.shape[0], knots.shape[0], ' after inserting ', knot, ' at ', knotInd)
        
        cplocCtrPt = getControlPoints(knots, degree) * (Dmax - Dmin) + Dmin

        interpOrder = 'linear' # 'linear', 'cubic', 'quintic'
        # Interpolate using coeffs_x and control_points (self.pAdaptive)
        coeffs_x = getControlPoints(self.knotsAdaptive, degree) * (Dmax - Dmin) + Dmin
        #InterpCp = interp1d(coeffs_x, self.pAdaptive, kind=interpOrder)
        InterpCp = Rbf(coeffs_x, self.pAdaptive, function=interpOrder)

        PnewCp = InterpCp(xnew)

        return Pnew, PnewCp, cplocCtrPt, knots, W

    def send(self, cp):
        link = cp.link()
        for i in range(len(link)):
            target = link.target(i)
            o = [0,0,0]
            if target.gid > cp.gid(): # target is to the right of current subdomain
                if len(self.pAdaptive):
                    o = [self.pAdaptive[-1], self.WAdaptive[-1], self.knotsAdaptive[-1]]
                print("%d sending to %d: %s" % (cp.gid(), target.gid, o))
            else: # target is to the left of current subdomain
                if len(self.pAdaptive):
                    o = [self.pAdaptive[0], self.WAdaptive[0], self.knotsAdaptive[0]]
                print("%d sending to %d: %s" % (cp.gid(), target.gid, o))
            cp.enqueue(target, o) 

    def recv(self, cp):
        link = cp.link()
        for i in range(len(link)):
            tgid = link.target(i).gid
            o = cp.dequeue(tgid)
            if tgid > cp.gid(): # target is to the right of current subdomain; receive constraint for right end point
                self.rightconstraint = o[:] if len(o) else [0,0,0]
            else:
                self.leftconstraint = o[:] if len(o) else [0,0,0]
            print("%d received from %d: %s" % (cp.gid(), tgid, o))

    def solve_adaptive(self, cp):

        ## Subdomain ID: iSubDom = cp.gid()+1
        domStart = (cp.gid()) * 1.0 / nSubDomains
        domEnd   = (cp.gid()+1) * 1.0 / nSubDomains
        U   = np.linspace(domStart, domEnd, self.nPointsPerSubD)

        newSolve = False
        if len(self.pAdaptive) == 0:
            newSolve = True

        if newSolve:

            inc = (domEnd - domStart) / self.nInternalKnotSpans
            t   = np.linspace(domStart + inc, domEnd - inc, self.nInternalKnotSpans - 1)
            knots  = np.concatenate(([domStart] * (degree+1), t, [domEnd] * (degree+1)))
            # spl = LSQUnivariateSpline(U, self.ySD, t, k=degree)
	    # get the control points
            # knots    = spl.get_knots()
            # knots    = np.concatenate(([knots[0]] * degree, knots, [knots[-1]] * degree))
    
            popt = [] # interface_constraints[cp.gid()]
            W = [] # np.ones_like(popt)

        else:

            knots = self.knotsAdaptive[:]
            popt = self.pAdaptive[:]
            W = self.WAdaptive[:]
            nControlPoints = len(popt)

            self.nControlPointSpans = self.nControlPoints - 1
            self.nInternalKnotSpans = self.nControlPointSpans - degree + 1

            inc = (domEnd - domStart) / self.nInternalKnotSpans

        print ("Subdomain -- ", cp.gid())

        dSpan = range( (cp.gid()) * self.nPointsPerSubD, (cp.gid()+1) * self.nPointsPerSubD )

        self.Dmini = Dmin + (Dmax - Dmin)*domStart
        self.Dmaxi = Dmin + (Dmax - Dmin)*domEnd

        self.xSD = U * (Dmax - Dmin) + Dmin
        self.ySD = np.array(y[dSpan])

        # Let do the recursive iterations
        # Use the previous MAK solver solution as initial guess; Could do something clever later
        self.interface_constraints_obj['P'][1] = popt[:]
        self.interface_constraints_obj['T'][1] = knots[:]
        self.interface_constraints_obj['W'][1] = W[:]

        self.interface_constraints_obj['P'][0] = self.leftconstraint[:]
        self.interface_constraints_obj['P'][2] = self.rightconstraint[:]

        # Invoke the adaptive fitting routine for this subdomain
        self.pAdaptive, self.WAdaptive, self.knotsAdaptive,_ = adaptive(cp.gid()+1, self.interface_constraints_obj, U, 
                                                                        self.xSD, self.ySD, 
                                                                        #MAX_ERR=maxAdaptErr,
                                                                        MAX_ERR=maxAbsErr,
                                                                        split_all=True, 
                                                                        strategy=AdaptiveStrategy, 
                                                                        r=1, MAX_ITER=2)

        # NAdaptive = basis(U[np.newaxis,:],degree,knotsAdaptive[:,np.newaxis]).T
        # E = Error(pAdaptive, WAdaptive, ySD, U, knotsAdaptive, degree)
        # print ("Sum of squared error:", np.sum(E**2))
        # print ("Normalized max error:", np.abs(E).max()/yRange)

        errorMAK = L2LinfErrors(self.pAdaptive, self.WAdaptive, self.ySD, U, self.knotsAdaptive, degree)
        print ("Subdomain: ", cp.gid(), " -- L2 error: ", errorMAK[0], ", Linf error: ", errorMAK[1])

#     return PAdaptDomain, WAdaptDomain, KnotAdaptDomains


#########
from cycler import cycler

# Routine to recursively add a block and associated data to it
def add_input_control_block(gid, core, bounds, domain, link):
    print("Subdomain %d: " % gid, core, bounds, domain)
    minb = bounds.min
    maxb = bounds.max
    xlocal = x[minb[0]:maxb[0]+1]
    ylocal = y[minb[0]:maxb[0]+1]
    mc.add(gid, InputControlBlock(nControlPoints,core,xlocal,ylocal), link)

# TODO: If working in parallel with MPI or DIY, do a global reduce here
nPoints = len(x)
Dmin = min(x)
Dmax = max(x)
showplot = True

# print "Initial condition data: ", interface_constraints
errors = np.zeros([10,1]) # Store L2, Linf errors as function of iteration

# Let us initialize DIY and setup the problem
share_face = [True]
wrap = [False]
ghosts = [0]

# Initialize DIY
mc = diy.Master(w)         # master
domain_control = diy.DiscreteBounds([0], [len(x)-1])

d_control = diy.DiscreteDecomposer(1, domain_control, nSubDomains, share_face, wrap, ghosts)
a_control = diy.ContiguousAssigner(w.size, nSubDomains)

d_control.decompose(w.rank, a_control, add_input_control_block)

mc.foreach(InputControlBlock.show)

#########
nmaxiter=5
for iterIdx in range(nmaxiter):

    print ("\n---- Starting Iteration: %d ----" % iterIdx)
    
    # Now let us perform send-receive to get the data on the interface boundaries from 
    # adjacent nearest-neighbor subdomains
    mc.foreach(InputControlBlock.send)
    mc.exchange(False)
    mc.foreach(InputControlBlock.recv)

    mc.foreach(InputControlBlock.solve_adaptive)

    if showplot:
        # Let us plot the initial data
        plt.figure()
        if nSubDomains > 5:
            plt.rc('axes', prop_cycle=(cycler('color', ['r', 'g', 'b', 'y', 'c']) + cycler('linestyle', ['-','--',':','-.',''])))
        plt.plot(x, y, 'b-', ms=5, label='Input')

        mc.foreach(InputControlBlock.plot)

        plt.legend()
        plt.draw()

plt.show()

