# import autograd.numpy as np
import numpy as np
import splipy as sp
from scipy import linalg
from numba import jit

class ProblemSolver1D:
    def __init__(
        self, icBlock, coreb, xb, xl, degree, augmentSpanSpace=0, useDiagonalBlocks=True, verbose=False
    ):
        icBlock.xbounds = [xb.min[0], xb.max[0]]
        icBlock.corebounds = [[coreb.min[0] - xb.min[0], len(xl)]]
        icBlock.xyzCoordLocal = {"x": np.copy(xl[:])}
        icBlock.Dmini = np.array([min(xl)])
        icBlock.Dmaxi = np.array([max(xl)])
        icBlock.basisFunction = {"x": None}  # Basis function object in x-dir
        icBlock.decodeOpXYZ = {"x": None}
        icBlock.knotsAdaptive = {"x": []}
        icBlock.isClamped = {"left": False, "right": False}

        self.inputCB = icBlock
        self.degree = degree
        self.augmentSpanSpace = augmentSpanSpace
        self.useDiagonalBlocks = useDiagonalBlocks
        self.dimension = 1
        self.verbose = verbose

    def compute_basis(self, constraints=None):
        self.inputCB.basisFunction["x"] = sp.BSplineBasis(
            order=self.degree + 1, knots=self.inputCB.knotsAdaptive["x"]
        )
        print(
            "Number of basis functions = ",
            self.inputCB.basisFunction["x"].num_functions(),
        )
        # print("TU = ", self.inputCB.knotsAdaptive['x'], self.inputCB.UVW['x'][0], self.inputCB.UVW['x'][-1], self.inputCB.basisFunction['x'].greville())
        self.inputCB.NUVW["x"] = np.array(
            self.inputCB.basisFunction["x"].evaluate(self.inputCB.UVW["x"])
        )
        if constraints is not None:
            for entry in constraints:
                self.inputCB.NUVW["x"][entry, :] = 0.0
                self.inputCB.NUVW["x"][entry, entry] = 0.0

    def decode(self, P, RN):
        return np.matmul(RN["x"], P)

    def compute_decode_operators(self, RN):
        for dir in ["x"]:
            RN[dir] = (
                self.inputCB.NUVW[dir]
                / np.sum(self.inputCB.NUVW[dir], axis=1)[:, np.newaxis]
            )

    @jit
    def lsqFit(self):
        self.inputCB.refSolutionLocal = self.inputCB.refSolutionLocal.reshape(
            self.inputCB.refSolutionLocal.shape[0], 1
        )
        return linalg.lstsq(
            self.inputCB.decodeOpXYZ["x"], self.inputCB.refSolutionLocal
        )[0]

    def update_bounds(self):
        return [self.inputCB.corebounds[0][0], self.inputCB.corebounds[0][1]]

    def residual_operator_1D(self, Pin):  # checkpoint3

        # RN = (self.NUVW['x'] * self.weightsData) / \
        #     (np.sum(self.NUVW['x']*self.weightsData,
        #      axis=1)[:, np.newaxis])
        Aoper = np.matmul(self.decodeOpXYZ["x"].T, self.decodeOpXYZ["x"])
        Brhs = self.decodeOpXYZ["x"].T @ self.refSolutionLocal
        Brhs = Brhs.reshape(Pin.shape)
        # print('Input P = ', Pin.shape, Aoper.shape, Brhs.shape)

        oddDegree = self.degree % 2
        oddDegreeImpose = True

        # num_constraints = (self.degree)/2 if self.degree is even
        # num_constraints = (self.degree+1)/2 if self.degree is odd
        nconstraints = self.augmentSpanSpace + (
            int(self.degree / 2.0) if not oddDegree else int((self.degree + 1) / 2.0)
        )
        # if oddDegree and not oddDegreeImpose:
        #     nconstraints -= 1
        # nconstraints = self.degree-1
        # print('nconstraints: ', nconstraints)

        residual_constrained_nrm = 0
        nBndOverlap = 0
        if constraints is not None and len(constraints) > 0:

            if idom > 1:  # left constraint

                loffset = -2 * augmentSpanSpace if oddDegree else -2 * augmentSpanSpace

                if oddDegree and not oddDegreeImpose:
                    # print('left: ', nconstraints, -self.degree+nconstraints+loffset,
                    #       Pin[nconstraints], self.boundaryConstraints['left'][-self.degree+nconstraints+loffset])
                    constraintVal = initSol[nconstraints - 1, 0]
                    print(
                        (constraintVal * Aoper[:, nconstraints]).shape,
                        Aoper[:, nconstraints].shape,
                        Brhs.shape,
                    )
                    Brhs -= (constraintVal * Aoper[:, nconstraints]).reshape(Brhs.shape)

                for ic in range(nconstraints):
                    # Brhs[ic] = 0.5 * \
                    #     (Pin[ic] + self.boundaryConstraints['left']
                    #      [-self.degree+ic+loffset])
                    # print('Left: ', Brhs[ic], Pin[ic])
                    if oddDegree and oddDegreeImpose:
                        Brhs[ic] = initSol[ic, 0]
                    Aoper[ic, :] = 0.0
                    Aoper[ic, ic] = 1.0

            if idom < nSubDomains[0]:  # right constraint

                loffset = (
                    2 * self.augmentSpanSpace
                    if oddDegree
                    else 2 * self.augmentSpanSpace
                )

                if oddDegree and not oddDegreeImpose:
                    # print('right: ', -nconstraints-1, self.degree-1-nconstraints+loffset,
                    #       Pin[-nconstraints-1], self.boundaryConstraints['right']
                    #       [self.degree-1-nconstraints+loffset])
                    constraintVal = initSol[-nconstraints, 0]
                    Brhs -= (constraintVal * Aoper[:, -nconstraints]).reshape(
                        Brhs.shape
                    )

                for ic in range(nconstraints):
                    # Brhs[-ic-1] = 0.5 * \
                    #     (Pin[-ic-1] + self.boundaryConstraints['right']
                    #      [self.degree-1-ic+loffset])
                    # print('Right: ', Brhs[-ic-1], Pin[-ic-1])
                    if oddDegree and oddDegreeImpose:
                        Brhs[-ic - 1] = initSol[-ic - 1]
                    Aoper[-ic - 1, :] = 0.0
                    Aoper[-ic - 1, -ic - 1] = 1.0

        # print(Aoper, Brhs)
        return [Aoper, Brhs]

    def residual(self, Pin, printVerbose=False):

        decoded = self.decode(Pin, self.decodeOpXYZ)
        residual_decoded = (self.refSolutionLocal - decoded) / solutionRange
        residual_decoded = residual_decoded.reshape(-1)
        decoded_residual_norm = np.sqrt(
            np.sum(residual_decoded**2) / len(residual_decoded)
        )
        if type(Pin) is not np.numpy_boxes.ArrayBox and printVerbose:
            print("Residual {0}-d: {1}".format(dimension, decoded_residual_norm))
        return decoded_residual_norm

        # Use previous iterate as initial solution
        # initSol = constraints[1][:] if constraints is not None else np.ones_like(
        #     W)
        # initSol = np.ones_like(W)*0
        oddDegree = self.degree % 2
        nconstraints = augmentSpanSpace + (
            int(self.degree / 2.0) if not oddDegree else int((self.degree + 1) / 2.0)
        )

        lbound = 0
        ubound = len(Pin)
        if not self.isClamped["left"]:
            lbound = nconstraints + 1 if oddDegree else nconstraints
        if not self.isClamped["right"]:
            ubound -= nconstraints + 1 if oddDegree else nconstraints

        if True:
            [Aoper, Brhs] = self.residual_operator_1D(Pin)
            # if type(Pin) is np.numpy_boxes.ArrayBox:
            #     [Aoper, Brhs] = residual_operator_1D(Pin._value[:], False, False)
            # else:
            #     [Aoper, Brhs] = residual_operator_1D(Pin, False, False)

            # lu, piv = scipy.linalg.lu_factor(Aoper)
            # # print(lu, piv)
            # initSol2 = scipy.linalg.lu_solve((lu, piv), Brhs)

            residual_nrm_vec = (Brhs - Aoper @ Pin)[lbound:ubound]
            residual_nrm = np.sqrt(
                np.sum(residual_nrm_vec**2) / len(residual_nrm_vec)
            )
            # solerror = (initSol2-Pin)[lbound:ubound] #Brhs-Aoper@Pin
            # residual_nrm = np.sqrt(np.sum(solerror**2)/len(solerror))

            if type(Pin) is not np.numpy_boxes.ArrayBox and printVerbose:
                print("Residual 1D: ", residual_nrm)

        else:

            # New scheme like 2-D
            # print('Corebounds: ', self.corebounds)
            decoded = self.decode(Pin, self.decodeOpXYZ)
            residual_decoded = (self.refSolutionLocal - decoded) / solutionRange
            residual_decoded2 = residual_decoded[
                self.corebounds[0][0] : self.corebounds[0][1]
            ]
            decoded_residual_norm = np.sqrt(
                np.sum(residual_decoded2**2) / len(residual_decoded2)
            )

            # if type(Pin) is not np.numpy_boxes.ArrayBox and printVerbose:
            #     print('Residual decoded 1D: ', decoded_residual_norm)

            # residual_encoded = np.matmul(self.decodeOpXYZ['x'].T, residual_decoded)
            # bc_penalty = 0
            # decoded_residual_norm = np.sqrt(
            #     np.sum(residual_encoded[lbound:ubound]**2)/len(residual_encoded[lbound:ubound])) + bc_penalty * np.sqrt(np.sum(residual_encoded[:lbound]**2)+np.sum(residual_encoded[ubound:]**2))

            residual_nrm = decoded_residual_norm - initialDecodedError

            if type(Pin) is not np.numpy_boxes.ArrayBox and printVerbose:
                print("Residual 1D: ", decoded_residual_norm, residual_nrm)

        return residual_nrm

    def send_diy(self, inputCB, cp):

        oddDegree = self.degree % 2
        nconstraints = self.augmentSpanSpace + (
            int(self.degree / 2.0) if not oddDegree else int((self.degree + 1) / 2.0)
        )
        loffset = self.degree + 2 * self.augmentSpanSpace
        link = cp.link()
        for i in range(len(link)):
            target = link.target(i)
            if len(inputCB.controlPointData):
                dir = link.direction(i)
                if dir[0] == 0:
                    continue

                # target is coupled in X-direction
                if dir[0] > 0:  # target block is to the right of current subdomain
                    if self.verbose:
                        print(
                            "%d sending to %d" % (cp.gid(), target.gid),
                            "Left: ",
                            inputCB.controlPointData[
                                -1 : -2 - self.degree - self.augmentSpanSpace : -1, :
                            ].shape,
                        )

                    cp.enqueue(target, inputCB.controlPointData[-loffset:])
                    # cp.enqueue(target, inputCB.controlPointData)
                    cp.enqueue(
                        target,
                        inputCB.knotsAdaptive["x"][
                            -1 : -2 - self.degree - self.augmentSpanSpace : -1
                        ],
                    )

                else:  # target block is to the left of current subdomain
                    if self.verbose:
                        print(
                            "%d sending to %d" % (cp.gid(), target.gid),
                            "Right: ",
                            inputCB.controlPointData[
                                self.degree + self.augmentSpanSpace :: -1, :
                            ].shape,
                        )

                    cp.enqueue(target, inputCB.controlPointData[:loffset])
                    # cp.enqueue(target, inputCB.controlPointData)
                    cp.enqueue(
                        target,
                        inputCB.knotsAdaptive["x"][
                            0 : (self.degree + self.augmentSpanSpace + 1)
                        ],
                    )

        return

    def recv_diy(self, inputCB, cp):

        link = cp.link()
        for i in range(len(link)):
            tgid = link.target(i).gid
            dir = link.direction(i)
            # print("%d received from %d: %s from direction %s, with sizes %d+%d" % (cp.gid(), tgid, o, dir, pl, tl))

            # ONLY consider coupling through faces and not through verties
            # This means either dir[0] or dir[1] has to be "0" for subdomain coupling to be active
            # Hence we only consider 4 neighbor cases, instead of 8.
            if dir[0] == 0:
                continue

            if dir[0] < 0:  # target block is to the left of current subdomain
                # print('Right: ', np.array(o[2:pl+2]).reshape(useDerivativeConstraints+1,pll).T)

                inputCB.boundaryConstraints["left"] = cp.dequeue(tgid)
                inputCB.ghostKnots["left"] = cp.dequeue(tgid)
                if self.verbose:
                    print(
                        "Left: %d received from %d: from direction %s"
                        % (cp.gid(), tgid, dir),
                        inputCB.leftconstraint.shape,
                        inputCB.leftconstraintKnots.shape,
                    )

            else:  # target block is to right of current subdomain

                inputCB.boundaryConstraints["right"] = cp.dequeue(tgid)
                inputCB.ghostKnots["right"] = cp.dequeue(tgid)
                if self.verbose:
                    print(
                        "Right: %d received from %d: from direction %s"
                        % (cp.gid(), tgid, dir),
                        inputCB.rightconstraint.shape,
                        inputCB.rightconstraintKnots.shape,
                    )

        return

    def initialize_solution(
        self, inputCB, idom, initSol, degree, augmentSpanSpace, fullyPinned
    ):

        alpha = 0.5 # Between [0, 1.0]
        beta = 0.0 # Between [0, 0.5]
        localBCAssembly = np.zeros(initSol.shape)
        freeBounds = [0, len(localBCAssembly[:, 0])]

        if fullyPinned:
            # First update hte control point vector with constraints for supporting points
            if "left" in inputCB.boundaryConstraints:
                initSol[0] = (
                    alpha * initSol[0]
                    + (1 - alpha) * inputCB.boundaryConstraints["left"][-1]
                )
            if "right" in inputCB.boundaryConstraints:
                initSol[-1] = (
                    alpha * initSol[-1]
                    + (1 - alpha) * inputCB.boundaryConstraints["right"][0]
                )

        else:
            oddDegree = degree % 2
            nconstraints = augmentSpanSpace + (
                int(degree / 2.0) if not oddDegree else int((degree + 1) / 2.0)
            )
            loffset = 2 * augmentSpanSpace
            print("Nconstraints = ", nconstraints, "loffset = ", loffset)

            freeBounds[0] = (
                0
                if inputCB.isClamped["left"]
                else (nconstraints - 1 if oddDegree else nconstraints)
            )
            freeBounds[1] = (
                len(localBCAssembly[:, 0])
                if inputCB.isClamped["right"]
                else len(localBCAssembly[:, 0])
                - (nconstraints - 1 if oddDegree else nconstraints)
            )

            # First update hte control point vector with constraints for supporting points
            if "left" in inputCB.boundaryConstraints:
                if oddDegree:
                    if nconstraints > 1:
                        initSol[: nconstraints - 1] = (
                            beta * initSol[: nconstraints - 1]
                            + (1 - beta)
                            * inputCB.boundaryConstraints["left"][
                                -degree - loffset : -nconstraints
                            ]
                        )
                    initSol[nconstraints - 1] = (
                        alpha * initSol[nconstraints - 1]
                        + (1 - alpha)
                        * inputCB.boundaryConstraints["left"][-nconstraints]
                    )
                else:
                    initSol[:nconstraints] = (
                        beta * initSol[:nconstraints]
                        + (1 - beta)
                        * inputCB.boundaryConstraints["left"][
                            -degree - loffset : -nconstraints
                        ]
                    )

            if "right" in inputCB.boundaryConstraints:
                if oddDegree:
                    if nconstraints > 1:
                        initSol[-nconstraints + 1 :] = (
                            beta * initSol[-nconstraints + 1 :]
                            + (1 - beta)
                            * inputCB.boundaryConstraints["right"][
                                nconstraints : degree + loffset
                            ]
                        )
                    initSol[-nconstraints] = (
                        alpha * initSol[-nconstraints]
                        + (1 - alpha)
                        * inputCB.boundaryConstraints["right"][nconstraints - 1]
                    )
                else:
                    initSol[-nconstraints:] = (
                        beta * initSol[-nconstraints:]
                        + (1 - beta)
                        * inputCB.boundaryConstraints["right"][
                            nconstraints : degree + loffset
                        ]
                    )

        return np.copy(initSol)
