#!/usr/bin/env python
import sys, math
import numpy as np
import matplotlib.pyplot as plt
from makruth_solver import compute_mak95_fit_with_constraints

# --- set problem input parameters here ---
Dmin           = -4.
Dmax           = 4.
nPoints        = 480
nSubDomains    = 4
degree         = 3
sincFunc       = True
useAdditiveSchwartz = True
useDerivativeConstraints = False

solverscheme   = 'SLSQP' # [SLSQP, COBYLA]
nControlPoints = (3*degree + 1) #minimum number of control points
nmaxiter       = 6
scale          = 1
maxErr         = 1e-5
# ------------------------------------------

if sincFunc:
    x = np.linspace(Dmin, Dmax, nPoints)
    y = scale * np.sinc(x+1)
    # y = scale * np.sin(math.pi * x/4)
else:
    y = np.fromfile("s3d.raw", dtype=np.float64) #
    nPoints = y.shape[0]
    x = np.linspace(Dmin, Dmax, nPoints)


mpl_fig = plt.figure()
plt.plot(x, y)

# Let do the recursive iterations
# Use the previous MAK solver solution as initial guess; Could do something clever later
interface_constraints=[]

print "Initial condition data: ", interface_constraints
errors = np.zeros([nmaxiter,2]) # Store L2, Linf errors as function of iteration

for iterIdx in range(nmaxiter):

    # Call our MFA functional fitting routine for all the subdomains
    newconstraints = compute_mak95_fit_with_constraints(nSubDomains, degree, nControlPoints, x, y,
                                                        interface_constraints, 
                                                        additiveSchwartz=useAdditiveSchwartz, 
                                                        useDerivatives=useDerivativeConstraints, 
                                                        showplot=False, nosolveplot=False, 
                                                        solver=solverscheme)

    # Let's compute the delta in the iterate
    if len(interface_constraints) > 0:
        iteratedelta = (np.array(newconstraints)-np.array(interface_constraints))
    else:
        iteratedelta = np.array(newconstraints)
    # Essentially performing (unweighted) Richardson's iteration for converging
    # the subdomain problem
    #
    #    $y_{n+1} = y_n - 0.5 * \delta y_n = 0.5 * (y_n + y_{n+1})$
    # 
    # Future work: There are ways to accelerate this convergence with 
    # Aitken or Wynn-Epsilon schemes since Richardson is linearly 
    # convergent and produces approximations to the fixed point solution
    interface_constraints = newconstraints[:,:].copy() # - 0.5 * iteratedelta

    # Compute the error and convergence
    errors[iterIdx, 0] = math.sqrt(np.sum(iteratedelta**2)) if math.sqrt(np.sum(iteratedelta**2)) > 1e-16 else 1e-24
    errors[iterIdx, 1] = np.max(iteratedelta) if np.max(iteratedelta) > 1e-16 else 1e-24
    print "\nIteration: ", iterIdx, " -- L2 error: ", errors[iterIdx, 0], ", Linf error: ", errors[iterIdx, 1]
    if iterIdx == 0:
        print "\tUnconstrained solution: ", interface_constraints
    else:
        print "\tIterate delta: ", iteratedelta
    print "------------------------\n"

    # Check for convergence
    if errors[iterIdx, 0] < maxErr:
        print errors
        break

print "\tConverged solution: ", interface_constraints

# Finally, let's show the plot with the newly computed solution
newconstraints = compute_mak95_fit_with_constraints(nSubDomains, degree, nControlPoints, x, y,
                                                    interface_constraints, 
                                                    additiveSchwartz=useAdditiveSchwartz, 
                                                    useDerivatives=useDerivativeConstraints, 
                                                    showplot=True, nosolveplot=True,
                                                    solver=solverscheme)

if iterIdx > 1:
    plt.figure()
    xx = range(iterIdx+1)
    plt.plot(xx, np.log10(errors[xx,0]), 'b--', ms=5, label='L_2 error')
    plt.plot(xx, np.log10(errors[xx,1]), 'r--', lw=3, label='Linf error')
    plt.legend()
    plt.draw()

plt.show()
