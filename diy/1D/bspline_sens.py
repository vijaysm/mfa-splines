import numpy as np
import bspline
import splinelab
from matplotlib import pyplot as plt
plt.style.use(['seaborn-whitegrid'])
params = {"ytick.color": "b",
          "xtick.color": "b",
          "axes.labelcolor": "b",
          "axes.edgecolor": "b"}
plt.rcParams.update(params)

# mpl_fig = plt.figure()
# plt.plot(x, y, 'r-', ms=2)

# Spline setup and evaluation

p = 4              # order of spline (as-is; 3 = cubic)
eps = 1e-12
nknots = 11        # number of knots to generate (here endpoints count only once)
tau = [0+eps, 1-eps]  # collocation sites (i.e. where to evaluate)

knots1 = np.array([-4., -3.42857143, -2.85714286, -2.28571429, -1.71428571, -1.14285714, -0.57142857,  0.])/(4.0)
tau1 = [knots1[0]+eps, knots1[-1]-eps]  # collocation sites (i.e. where to evaluate)
knots2 = np.array([0.,  0.57142857, 1.14285714, 1.71428571, 2.28571429, 2.85714286, 3.42857143, 4.])/4.0
tau2 = [knots2[0]+eps, knots2[-1]-eps]  # collocation sites (i.e. where to evaluate)
# knots = numpy.linspace(0, 1, nknots)  # create a knot vector without endpoint repeats
knots = knots1
tau = tau1

k = splinelab.augknt(knots, p)  # add endpoint repeats as appropriate for spline order p
B = bspline.Bspline(k, p)       # create spline basis of order p on knots k

A0 = B.collmat(tau)                 # collocation matrix for function value at sites tau
A1 = B.collmat(tau, deriv_order=1)  # collocation matrix for first derivative at sites tau
A2 = B.collmat(tau, deriv_order=2)  # collocation matrix for second derivative at sites tau

#print(B.collmat(numpy.linspace(-2, -1, nknots)))
# print(A0)
# print(A1)
# print(A2)

#######################################################################################################

# Test a 2 subdomain problem: Equal knot spacing
print('Degree 2: Equal knot spacing at interface with 0-ghosted CP')
p = 2
knots1 = np.array([-4., -4., -4., -3.42857143, -2.85714286, -2.28571429, -1.71428571, -1.14285714, -0.57142857,  0.])/(4.0)
# tau1 = [knots1[0]+eps, knots1[-3]-eps]  # collocation sites (i.e. where to evaluate)
tau1 = [knots1[-1]-eps]  # collocation sites (i.e. where to evaluate)
knots2 = np.array([0.,  0.57142857, 1.14285714,
                   1.71428571, 2.28571429, 2.85714286, 3.42857143, 4., 4., 4.])/4.0
# tau2 = [knots2[2]+eps, knots2[-1]-eps]  # collocation sites (i.e. where to evaluate)
tau2 = [knots2[0]+eps]  # collocation sites (i.e. where to evaluate)

B1 = bspline.Bspline(knots1, p)       # create spline basis of order p on knots k
B2 = bspline.Bspline(knots2, p)       # create spline basis of order p on knots k

print(B1.collmat(tau1, deriv_order=0))
print(B2.collmat(tau2, deriv_order=0))

####
# Test a 2 subdomain problem: Equal knot spacing
print('\nDegree 2: Equal knot spacing at interface with 1-ghosted CP')
p = 2
knots1 = np.array([-4., -4., -4., -3.42857143, -2.85714286, -2.28571429, -1.71428571, -1.14285714, -0.57142857,
                   0., 0.57142857])/(4.0)
# tau1 = [knots1[0]+eps, knots1[-3]-eps]  # collocation sites (i.e. where to evaluate)
tau1 = [knots1[-2]-eps]  # collocation sites (i.e. where to evaluate)
knots2 = np.array([-0.57142857,  0.,  0.57142857, 1.14285714,
                   1.71428571, 2.28571429, 2.85714286, 3.42857143, 4., 4., 4.])/4.0
# tau2 = [knots2[2]+eps, knots2[-1]-eps]  # collocation sites (i.e. where to evaluate)
tau2 = [knots2[1]+eps]  # collocation sites (i.e. where to evaluate)

B1 = bspline.Bspline(knots1, p)       # create spline basis of order p on knots k
B2 = bspline.Bspline(knots2, p)       # create spline basis of order p on knots k

print(B1.collmat(tau1, deriv_order=0))
print(B2.collmat(tau2, deriv_order=0))

####
# Test a 2 subdomain problem: Equal knot spacing
print('\nDegree 2: Equal knot spacing at interface with 2-ghosted CP')
p = 2
knots1 = np.array([-4., -4., -4., -3.42857143, -2.85714286, -2.28571429, -1.71428571, -1.14285714, -0.57142857,
                   0., 0.57142857, 1.14285714])/(4.0)
# tau1 = [knots1[0]+eps, knots1[-3]-eps]  # collocation sites (i.e. where to evaluate)
tau1 = [knots1[-3]-eps]  # collocation sites (i.e. where to evaluate)
knots2 = np.array([-1.14285714, -0.57142857,  0.,  0.57142857, 1.14285714,
                   1.71428571, 2.28571429, 2.85714286, 3.42857143, 4., 4., 4.])/4.0
# tau2 = [knots2[2]+eps, knots2[-1]-eps]  # collocation sites (i.e. where to evaluate)
tau2 = [knots2[2]+eps]  # collocation sites (i.e. where to evaluate)

B1 = bspline.Bspline(knots1, p)       # create spline basis of order p on knots k
B2 = bspline.Bspline(knots2, p)       # create spline basis of order p on knots k

print(B1.collmat(tau1, deriv_order=0))
print(B2.collmat(tau2, deriv_order=0))
print(B1.collmat(tau1, deriv_order=1))
print(B2.collmat(tau2, deriv_order=1))
print(B1.collmat(tau1, deriv_order=2))
print(B2.collmat(tau2, deriv_order=2))

####
print('\nDegree 3: Non-Equal knot spacing at interface with 2-ghosted CP')
p = 3
knots1 = np.array([-4., -4., -4., -3.42857143, -2.85714286, -2.28571429, -1.71428571, -1.14285714, -0.57142857,
                   0., 0.57142857, 1.14285714])/(4.0)
tau1 = [knots1[-3]-eps]  # collocation sites (i.e. where to evaluate)
knots2 = np.array([-1.14285714, -0.57142857,  0.,  0.57142857, 1.14285714,
                   1.71428571, 2.28571429, 2.85714286, 3.42857143, 4., 4., 4.])/4.0
tau2 = [knots2[2]+eps]  # collocation sites (i.e. where to evaluate)

B1 = bspline.Bspline(knots1, p)       # create spline basis of order p on knots k
B2 = bspline.Bspline(knots2, p)       # create spline basis of order p on knots k

print(B1.collmat(tau1, deriv_order=0))
print(B2.collmat(tau2, deriv_order=0))

####
print('\nDegree 3: Non-Equal knot spacing at interface with 3-ghosted CP')
p = 3
knots1 = np.array([-4., -4., -4., -3.42857143, -2.85714286, -2.28571429, -1.71428571, -1.14285714, -0.75, -0.57142857,
                   0., 0.57142857, 0.75, 1.14285714])/(4.0)
tau1 = [knots1[-4]-eps]  # collocation sites (i.e. where to evaluate)
knots2 = np.array([-1.14285714, -0.75, -0.57142857,  0.,  0.57142857, 0.75, 1.14285714,
                   1.71428571, 2.28571429, 2.85714286, 3.42857143, 4., 4., 4.])/4.0
tau2 = [knots2[3]+eps]  # collocation sites (i.e. where to evaluate)

B1 = bspline.Bspline(knots1, p)       # create spline basis of order p on knots k
B2 = bspline.Bspline(knots2, p)       # create spline basis of order p on knots k

print(B1.collmat(tau1, deriv_order=0))
print(B2.collmat(tau2, deriv_order=0))
