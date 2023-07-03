# MFA-Splines
Multivariate Functional Approximations with Domain-Decomposed B-spline curves using a tensor product approach using arbitrary degrees.

# Description
Compactly expressing large-scale datasets through Multivariate Functional Approximations (MFA) can be critically important for analysis and visualization to drive scientific discovery. Tackling such problems requires scalable data partitioning approaches to compute MFA representations in amenable wall clock times. We introduce a fully parallel scheme to reduce the total work per task in combination with an overlapping additive Schwarz-based iterative scheme to compute MFA with a tensor expansion of B-spline bases, while preserving full degree continuity across subdomain boundaries. While previous work on MFA has been successfully proven to be effective, the computational complexity of encoding large datasets on a single process can be severely prohibitive. Parallel algorithms for generating reconstructions from the MFA have had to rely on post-processing techniques to blend discontinuities across subdomain boundaries. In contrast, a robust constrained minimization infrastructure to impose higher-order continuity directly on the MFA representation is presented here. We demonstrate the effectiveness of the parallel approach with domain decomposition solvers, to minimize the subdomain error residuals of the decoded MFA, and more specifically to recover continuity across non-matching boundaries at scale. The analysis of the presented scheme for analytical and scientific datasets in 1-, 2- and 3-dimensions are presented. Extensive strong and weak scalability performances are also demonstrated for large-scale datasets to evaluate the parallel speedup of the MPI-based algorithm implementation on leadership computing machines.

# Dependencies
- Python>=3.8
- [DIY](https://github.com/diatomic/diy)
- autograd==1.3
- cycler==0.10.0
- matplotlib==3.5.2
- mpi4py==3.1.4
- numba==0.56.4
- numpy==1.19.3
- packaging==21.3
- pyvista==0.32.1
- scipy==1.8.0
- Splipy==1.5.7
- uvw==0.4.0

# Cite
Mahadevan, V., Lenz, D., Grindeanu, I., Peterka, T. (2023). Accelerating Multivariate Functional Approximation Computation with Domain Decomposition Techniques. In: Mikyška, J., de Mulatier, C., Paszynski, M., Krzhizhanovskaya, V.V., Dongarra, J.J., Sloot, P.M. (eds) Computational Science – ICCS 2023. ICCS 2023. Lecture Notes in Computer Science, vol 14073. Springer, Cham. https://doi.org/10.1007/978-3-031-35995-8_7

# Paper
You can download the paper from [Springer](https://doi.org/10.1007/978-3-031-35995-8_7) or [here](ICCS-2023-Mahadevan.pdf).
