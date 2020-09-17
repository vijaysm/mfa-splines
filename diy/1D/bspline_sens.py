import numpy
import bspline
import splinelab

# Spline setup and evaluation

p = 4              # order of spline (as-is; 3 = cubic)
eps = 1e-12
nknots = 11        # number of knots to generate (here endpoints count only once)
tau = [0+eps, 1-eps]  # collocation sites (i.e. where to evaluate)

knots1 = [-4., -3.42857143, -2.85714286, -2.28571429, -1.71428571, -1.14285714, -0.57142857,  0.]
tau1 = [knots1[0]+eps, knots1[-1]-eps]  # collocation sites (i.e. where to evaluate)
knots2 = [0.,  0.57142857, 1.14285714, 1.71428571, 2.28571429, 2.85714286, 3.42857143, 4.]
tau2 = [knots2[0]+eps, knots2[-1]-eps]  # collocation sites (i.e. where to evaluate)
# knots = numpy.linspace(0, 1, nknots)  # create a knot vector without endpoint repeats
knots = knots1
tau = tau1

k = splinelab.augknt(knots, p)  # add endpoint repeats as appropriate for spline order p
B = bspline.Bspline(k, p)       # create spline basis of order p on knots k

A0 = B.collmat(tau)                 # collocation matrix for function value at sites tau
A1 = B.collmat(tau, deriv_order=1)  # collocation matrix for first derivative at sites tau
A2 = B.collmat(tau, deriv_order=2)  # collocation matrix for second derivative at sites tau

print(B.collmat(numpy.linspace(-2, -1, nknots)))
# print(A0)
print(A1)
print(A2)
