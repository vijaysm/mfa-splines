# import autograd.numpy as np
import numpy as np
import splipy as sp

import scipy.sparse.linalg as la
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spsolve

from numba import jit

class ProblemSolver2D:
    def __init__(
        self,
        icBlock,
        coreb,
        xb,
        xl,
        yl,
        degree,
        augmentSpanSpace=0,
        useDiagonalBlocks=True,
        verbose=False
    ):
        icBlock.xbounds = [xb.min[0], xb.max[0], xb.min[1], xb.max[1]]
        icBlock.corebounds = [
            [coreb.min[0] - xb.min[0], len(xl)],
            [coreb.min[1] - xb.min[1], len(yl)],
        ]
        # int(nPointsX / nSubDomainsX)
        icBlock.xyzCoordLocal = {"x": np.copy(xl[:]), "y": np.copy(yl[:])}
        icBlock.Dmini = np.array([min(xl), min(yl)])
        icBlock.Dmaxi = np.array([max(xl), max(yl)])
        # Basis function object in x-dir and y-dir
        icBlock.basisFunction = {"x": None, "y": None}
        icBlock.decodeOpXYZ = {"x": None, "y": None}
        icBlock.knotsAdaptive = {"x": [], "y": []}
        icBlock.isClamped = {
            "left": False,
            "right": False,
            "top": False,
            "bottom": False,
        }
        icBlock.greville = {
            "leftx": [],
            "lefty": [],
            "rightx": [],
            "righty": [],
            "bottomx": [],
            "bottomy": [],
            "topx": [],
            "topy": [],
            "topleftx": [],
            "toplefty": [],
            "toprightx": [],
            "toprighty": [],
            "bottomleftx": [],
            "bottomlefty": [],
            "bottomrightx": [],
            "bottomrighty": [],
        }

        self.inputCB = icBlock
        self.degree = degree
        self.augmentSpanSpace = augmentSpanSpace
        self.useDiagonalBlocks = useDiagonalBlocks
        self.dimension = 2
        self.verbose = verbose

    def compute_basis(self, constraints=None):
        # self.inputCB.basisFunction['x'].reparam()
        # self.inputCB.basisFunction['y'].reparam()
        # print("TU = ", self.inputCB.knotsAdaptive['x'], self.inputCB.UVW['x'][0], self.inputCB.UVW['x'][-1], self.inputCB.basisFunction['x'].greville())
        # print("TV = ", self.inputCB.knotsAdaptive['y'], self.inputCB.UVW['y'][0], self.inputCB.UVW['y'][-1], self.inputCB.basisFunction['y'].greville())
        for dir in ["x", "y"]:
            self.inputCB.basisFunction[dir] = sp.BSplineBasis(
                order=self.degree + 1, knots=self.inputCB.knotsAdaptive[dir]
            )
            self.inputCB.NUVW[dir] = np.array(
                self.inputCB.basisFunction[dir].evaluate(self.inputCB.UVW[dir])
            )

        print(
            "Number of basis functions = ",
            self.inputCB.basisFunction["x"].num_functions(),
        )
        if constraints is not None:
            for entry in constraints[0]:
                self.inputCB.NUVW["x"][entry, :] = 0.0
                self.inputCB.NUVW["x"][entry, entry] = 1.0
            for entry in constraints[1]:
                self.inputCB.NUVW["y"][entry, :] = 0.0
                self.inputCB.NUVW["y"][entry, entry] = 1.0

    def compute_decode_operators(self, RN):
        for dir in ["x", "y"]:
            RN[dir] = csc_matrix(
                self.inputCB.NUVW[dir]
                / np.sum(self.inputCB.NUVW[dir], axis=1)[:, np.newaxis]
            )

    def decode(self, P, RN):

        # return np.einsum("kj,lj->kl", np.einsum("ki,ij->kj", RN["x"], P, optimize=True), RN["y"], optimize=True)
        XX =((RN["x"] * csc_matrix(P)) * RN["y"].T).toarray()
        return XX

        # return np.einsum("ki,il->kl", RN["x"], np.einsum("lj,ij->il", RN["y"], P))


        # return np.einsum("kj,jl->kl", RN["y"], np.einsum("ij,li->jl", P, RN["x"]))
        DD1 = np.matmul(np.matmul(RN["x"], P), RN["y"].T)

        DD2A = np.einsum("ki,ij->kj", RN["x"], P, optimize=True)
        DD2 = np.einsum("kj,lj->kl", DD2A, RN["y"], optimize=True)

        print(P.shape, RN["x"].shape, RN["y"].shape, DD1.shape)
        # DD2 = np.einsum("lj,ij->il", RN["y"], P)
        # DD2 = np.einsum("ki,il->kl", RN["x"], np.einsum("lj,ij->il", RN["y"], P))
        ### DD2 = np.einsum("kj,lj->kl", RN["y"], np.einsum("li,ij->lj", RN["x"], P))
        # DD3 = np.einsum("il,ik->lk", np.einsum("ij,jl->il", P, RN["y"]), RN["x"])

        print (np.amax(DD1-DD2))

        return DD1

    def lsqFit(self):
        decodeOpXYZ = self.inputCB.decodeOpXYZ
        refSolutionLocal = self.inputCB.refSolutionLocal

        use_cho = False
        if use_cho:
            X = la.cho_solve(
                la.cho_factor(
                    np.matmul(
                        self.inputCB.decodeOpXYZ["x"].T, self.inputCB.decodeOpXYZ["x"]
                    )
                ),
                self.inputCB.decodeOpXYZ["x"].T,
            )
            Y = la.cho_solve(
                la.cho_factor(
                    np.matmul(
                        self.inputCB.decodeOpXYZ["y"].T, self.inputCB.decodeOpXYZ["y"]
                    )
                ),
                self.inputCB.decodeOpXYZ["y"].T,
            )
            zY = np.matmul(self.inputCB.refSolutionLocal, Y.T)
            return np.matmul(X, zY)
        else:

            # NTNxInv = la.inv(
            #     np.matmul(
            #         decodeOpXYZ["x"].T, decodeOpXYZ["x"]
            #     )
            # )
            NTNyInv = la.inv(

                    decodeOpXYZ["y"].T * decodeOpXYZ["y"]

            )
            # XX = np.matmul(
            #         decodeOpXYZ["y"].T, decodeOpXYZ["y"]
            #     )
            # print ('Max', np.amax(referenceLSQ-res))
            NxTQNy = csc_matrix(
                decodeOpXYZ["x"].T *
                refSolutionLocal * decodeOpXYZ["y"],
            )
            # referenceLSQ = np.matmul(NTNxInv, np.matmul(NxTQNy, NTNyInv))

            Aoper = (decodeOpXYZ["x"].T * decodeOpXYZ["x"]).tocsc()
            Brhs = (NxTQNy * NTNyInv).tocsc()
            res = spsolve(Aoper, Brhs)
            # res = la.solve(np.matmul(decodeOpXYZ["x"].T, decodeOpXYZ["x"]),
            #                 la.solve(np.matmul(decodeOpXYZ["y"].T, decodeOpXYZ["y"]), NxTQNy, assume_a='pos', transposed=False), assume_a='pos'
            # )

            # print(res[0])

            # print ('Max', np.amax(referenceLSQ-res))

            return res.todense()

            NTNxInv = np.linalg.inv(
                np.matmul(
                    decodeOpXYZ["x"].T, decodeOpXYZ["x"]
                )
            )
            NTNyInv = np.linalg.inv(
                np.matmul(
                    decodeOpXYZ["y"].T, decodeOpXYZ["y"]
                )
            )
            NxTQNy = np.matmul(
                decodeOpXYZ["x"].T,
                np.matmul(refSolutionLocal, decodeOpXYZ["y"]),
            )
            return np.matmul(NTNxInv, np.matmul(NxTQNy, NTNyInv))

    def update_bounds(self):
        return [
            self.inputCB.corebounds[0][0],
            self.inputCB.corebounds[0][1],
            self.inputCB.corebounds[1][0],
            self.inputCB.corebounds[1][1],
        ]

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
                if dir[0] == 0 and dir[1] == 0 and dir[2] == 0:
                    continue

                # ONLY consider coupling through faces and not through verties
                # This means either dir[0] or dir[1] has to be "0" for subdomain coupling to be active
                # Hence we only consider 4 neighbor cases, instead of 8.
                if dir[0] == 0:  # target is coupled in Y-direction
                    if dir[1] > 0:  # target block is above current subdomain
                        if self.verbose:
                            print(
                                "%d sending to %d" % (cp.gid(), target.gid),
                                " Top: ",
                                inputCB.controlPointData[
                                    :,
                                    -1 : -2 - self.degree - self.augmentSpanSpace : -1,
                                ].shape,
                            )

                        cp.enqueue(target, inputCB.controlPointData[:, -loffset:])
                        # cp.enqueue(target, inputCB.controlPointData)
                        cp.enqueue(
                            target,
                            inputCB.knotsAdaptive["y"][
                                -1 : -2 - self.degree - self.augmentSpanSpace : -1
                            ],
                        )

                    else:  # target block is below current subdomain
                        if self.verbose:
                            print(
                                "%d sending to %d" % (cp.gid(), target.gid),
                                " Bottom: ",
                                inputCB.controlPointData[
                                    :, 0 : 1 + self.degree + self.augmentSpanSpace
                                ].shape,
                            )

                        cp.enqueue(target, inputCB.controlPointData[:, :loffset])
                        # cp.enqueue(target, inputCB.controlPointData)
                        cp.enqueue(
                            target,
                            inputCB.knotsAdaptive["y"][
                                0 : 1 + self.degree + self.augmentSpanSpace
                            ],
                        )

                # target is coupled in X-direction
                elif dir[1] == 0:
                    if dir[0] > 0:  # target block is to the right of current subdomain
                        if self.verbose:
                            print(
                                "%d sending to %d" % (cp.gid(), target.gid),
                                "Left: ",
                                inputCB.controlPointData[
                                    -1 : -2 - self.degree - self.augmentSpanSpace : -1,
                                    :,
                                ].shape,
                            )

                        cp.enqueue(target, inputCB.controlPointData[-loffset:, :])
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

                        cp.enqueue(target, inputCB.controlPointData[:loffset, :])
                        # cp.enqueue(target, inputCB.controlPointData)
                        cp.enqueue(
                            target,
                            inputCB.knotsAdaptive["x"][
                                0 : (self.degree + self.augmentSpanSpace + 1)
                            ],
                        )

                else:

                    if self.useDiagonalBlocks:
                        # target block is diagonally top right to current subdomain
                        if dir[0] > 0 and dir[1] > 0:

                            cp.enqueue(
                                target, inputCB.controlPointData[-loffset:, -loffset:]
                            )
                            # cp.enqueue(target, inputCB.controlPointData[-1-self.degree-self.augmentSpanSpace:, -1-self.degree-self.augmentSpanSpace:])
                            if self.verbose:
                                print(
                                    "%d sending to %d" % (cp.gid(), target.gid),
                                    " Diagonal = right-top: ",
                                    inputCB.controlPointData[
                                        -1 : -2
                                        - self.degree
                                        - self.augmentSpanSpace : -1,
                                        : 1 + self.degree + self.augmentSpanSpace,
                                    ],
                                )
                        # target block is diagonally top left to current subdomain
                        if dir[0] < 0 and dir[1] > 0:
                            cp.enqueue(
                                target, inputCB.controlPointData[:loffset:, -loffset:]
                            )
                            # cp.enqueue(target, inputCB.controlPointData[: 1 + self.degree + self.augmentSpanSpace, -1:-2-self.degree-self.augmentSpanSpace:-1])
                            if self.verbose:
                                print(
                                    "%d sending to %d" % (cp.gid(), target.gid),
                                    " Diagonal = left-top: ",
                                    self.controlPointData[
                                        -1 : -2
                                        - self.degree
                                        - self.augmentSpanSpace : -1,
                                        : 1 + self.degree + self.augmentSpanSpace,
                                    ],
                                )

                        # target block is diagonally left bottom  current subdomain
                        if dir[0] < 0 and dir[1] < 0:
                            cp.enqueue(
                                target, inputCB.controlPointData[:loffset:, :loffset]
                            )
                            # cp.enqueue(target, inputCB.controlPointData[-1-self.degree-self.augmentSpanSpace:, :1+self.degree+self.augmentSpanSpace])

                            if self.verbose:
                                print(
                                    "%d sending to %d" % (cp.gid(), target.gid),
                                    " Diagonal = left-bottom: ",
                                    self.controlPointData[
                                        : 1 + self.degree + self.augmentSpanSpace,
                                        -1 - self.degree - self.augmentSpanSpace :,
                                    ],
                                )
                        # target block is diagonally right bottom of current subdomain
                        if dir[0] > 0 and dir[1] < 0:
                            cp.enqueue(
                                target, inputCB.controlPointData[-loffset:, :loffset]
                            )
                            # cp.enqueue(target, inputCB.controlPointData[:1+self.degree+self.augmentSpanSpace, :1+self.degree+self.augmentSpanSpace])
                            if self.verbose:
                                print(
                                    "%d sending to %d" % (cp.gid(), target.gid),
                                    " Diagonal = right-bottom: ",
                                    inputCB.controlPointData[
                                        : 1 + self.degree + self.augmentSpanSpace,
                                        -1 - self.degree - self.augmentSpanSpace :,
                                    ],
                                )

                if inputCB.basisFunction["x"] is not None:
                    cp.enqueue(target, inputCB.basisFunction["x"].greville())
                    cp.enqueue(target, inputCB.basisFunction["y"].greville())

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
            if dir[0] == 0 and dir[1] == 0:
                continue

            if dir[0] == 0:  # target is coupled in Y-direction
                if dir[1] > 0:  # target block is above current subdomain
                    inputCB.boundaryConstraints["top"] = cp.dequeue(tgid)
                    inputCB.ghostKnots["top"] = cp.dequeue(tgid)
                    if self.verbose:
                        print(
                            "Top: %d received from %d: from direction %s"
                            % (cp.gid(), tgid, dir),
                            inputCB.topconstraint.shape,
                            inputCB.topconstraintKnots.shape,
                        )

                    # if oddDegree:
                    #     inputCB.controlPointData[:,-(degree-nconstraints):] = inputCB.boundaryConstraints['top'][:, :degree-nconstraints]
                    # else:
                    #     inputCB.controlPointData[:,-(degree-nconstraints)+1:] = inputCB.boundaryConstraints['top'][:, :degree-nconstraints]
                    if inputCB.basisFunction["x"] is not None:
                        inputCB.greville["topx"] = cp.dequeue(tgid)
                        inputCB.greville["topy"] = cp.dequeue(tgid)

                else:  # target block is below current subdomain
                    inputCB.boundaryConstraints["bottom"] = cp.dequeue(tgid)
                    inputCB.ghostKnots["bottom"] = cp.dequeue(tgid)
                    if self.verbose:
                        print(
                            "Bottom: %d received from %d: from direction %s"
                            % (cp.gid(), tgid, dir),
                            inputCB.bottomconstraint.shape,
                            inputCB.bottomconstraintKnots.shape,
                        )

                    if inputCB.basisFunction["x"] is not None:
                        inputCB.greville["bottomx"] = cp.dequeue(tgid)
                        inputCB.greville["bottomy"] = cp.dequeue(tgid)

                    # if oddDegree:
                    #     inputCB.controlPointData[:,:(degree-nconstraints)] = inputCB.boundaryConstraints['bottom'][:, -(degree-nconstraints):]
                    # else:
                    #     inputCB.controlPointData[:,:(degree-nconstraints)] = inputCB.boundaryConstraints['bottom'][:, -(degree-nconstraints)+1:]

            # target is coupled in X-direction
            elif dir[1] == 0:
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

                    if inputCB.basisFunction["x"] is not None:
                        inputCB.greville["leftx"] = cp.dequeue(tgid)
                        inputCB.greville["lefty"] = cp.dequeue(tgid)
                    # if oddDegree:
                    #     inputCB.controlPointData[:(degree-nconstraints),:] = inputCB.boundaryConstraints['left'][-(degree-nconstraints):,:]
                    # else:
                    #     inputCB.controlPointData[:(degree-nconstraints),] = inputCB.boundaryConstraints['left'][-(degree-nconstraints)+1:,:]

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

                    if inputCB.basisFunction["x"] is not None:
                        inputCB.greville["rightx"] = cp.dequeue(tgid)
                        inputCB.greville["righty"] = cp.dequeue(tgid)
                    # if oddDegree:
                    #     inputCB.controlPointData[-(degree-nconstraints):,:] = inputCB.boundaryConstraints['right'][:(degree-nconstraints):,:]
                    # else:
                    #     inputCB.controlPointData[-(degree-nconstraints):,:] = inputCB.boundaryConstraints['right'][:(degree-nconstraints):,:]

            else:

                if self.useDiagonalBlocks:
                    # 2-Dimension = 0: left, 1: right, 2: top, 3: bottom, 4: top-left, 5: top-right, 6: bottom-left, 7: bottom-right
                    # sender block is diagonally right top to  current subdomain
                    if dir[0] > 0 and dir[1] > 0:
                        inputCB.boundaryConstraints["top-right"] = cp.dequeue(tgid)
                        if self.verbose:
                            print(
                                "Top-right: %d received from %d: from direction %s"
                                % (cp.gid(), tgid, dir),
                                inputCB.boundaryConstraints["top-right"].shape,
                            )

                        if inputCB.basisFunction["x"] is not None:
                            inputCB.greville["toprightx"] = cp.dequeue(tgid)
                            inputCB.greville["toprighty"] = cp.dequeue(tgid)
                    # sender block is diagonally left top to current subdomain
                    if dir[0] > 0 and dir[1] < 0:
                        inputCB.boundaryConstraints["bottom-right"] = cp.dequeue(tgid)
                        if self.verbose:
                            print(
                                "Bottom-right: %d received from %d: from direction %s"
                                % (cp.gid(), tgid, dir),
                                inputCB.boundaryConstraints["bottom-right"].shape,
                            )

                        if inputCB.basisFunction["x"] is not None:
                            inputCB.greville["bottomrightx"] = cp.dequeue(tgid)
                            inputCB.greville["bottomrighty"] = cp.dequeue(tgid)
                    # sender block is diagonally left bottom  current subdomain
                    if dir[0] < 0 and dir[1] < 0:
                        inputCB.boundaryConstraints["bottom-left"] = cp.dequeue(tgid)
                        if self.verbose:
                            print(
                                "Bottom-left: %d received from %d: from direction %s"
                                % (cp.gid(), tgid, dir),
                                inputCB.boundaryConstraints["bottom-left"].shape,
                            )

                        if inputCB.basisFunction["x"] is not None:
                            inputCB.greville["bottomleftx"] = cp.dequeue(tgid)
                            inputCB.greville["bottomlefty"] = cp.dequeue(tgid)
                    # sender block is diagonally left to current subdomain
                    if dir[0] < 0 and dir[1] > 0:

                        inputCB.boundaryConstraints["top-left"] = cp.dequeue(tgid)
                        if self.verbose:
                            print(
                                "Top-left: %d received from %d: from direction %s"
                                % (cp.gid(), tgid, dir),
                                inputCB.boundaryConstraints["top-left"].shape,
                            )

                        if inputCB.basisFunction["x"] is not None:
                            inputCB.greville["topleftx"] = cp.dequeue(tgid)
                            inputCB.greville["toplefty"] = cp.dequeue(tgid)
        return

    def initialize_solution(
        self, inputCB, initSol, degree, augmentSpanSpace, fullyPinned
    ):

        alpha = 0.5 # Between [0, 1.0]
        beta = 0.0 # Between [0, 0.5]
        localAssemblyWeights = np.zeros(initSol.shape)
        localBCAssembly = np.zeros(initSol.shape)
        freeBounds = [0, len(localBCAssembly[:, 0]), 0, len(localBCAssembly[0, :])]

        if fullyPinned:
            if "left" in inputCB.boundaryConstraints:
                initSol[0, :] += inputCB.boundaryConstraints["left"][-1, :]
                localAssemblyWeights[0, :] += 1.0
            if "right" in inputCB.boundaryConstraints:
                initSol[-1, :] += inputCB.boundaryConstraints["right"][0, :]
                localAssemblyWeights[-1, :] += 1.0
            if "top" in inputCB.boundaryConstraints:
                initSol[:, -1] += inputCB.boundaryConstraints["top"][:, 0]
                localAssemblyWeights[:, -1] += 1.0
            if "bottom" in inputCB.boundaryConstraints:
                initSol[:, 0] += inputCB.boundaryConstraints["bottom"][:, -1]
                localAssemblyWeights[:, 0] += 1.0
            if "top-left" in inputCB.boundaryConstraints:
                initSol[0, -1] += inputCB.boundaryConstraints["top-left"][-1, 0]
                localAssemblyWeights[0, -1] += 1.0
            if "bottom-right" in inputCB.boundaryConstraints:
                initSol[-1, 0] = inputCB.boundaryConstraints["bottom-right"][0, -1]
                localAssemblyWeights[-1, 0] += 1.0
            if "bottom-left" in inputCB.boundaryConstraints:
                initSol[0, 0] = inputCB.boundaryConstraints["bottom-left"][-1, -1]
                localAssemblyWeights[0, 0] += 1.0
            if "top-right" in inputCB.boundaryConstraints:
                initSol[-1, -1] = inputCB.boundaryConstraints["top-right"][0, 0]
                localAssemblyWeights[-1, -1] += 1.0

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
            freeBounds[2] = (
                0
                if inputCB.isClamped["bottom"]
                else (nconstraints - 1 if oddDegree else nconstraints)
            )
            freeBounds[3] = (
                len(localBCAssembly[0, :])
                if inputCB.isClamped["top"]
                else len(localBCAssembly[0, :])
                - (nconstraints - 1 if oddDegree else nconstraints)
            )

            # First update hte control point vector with constraints for supporting points
            if "left" in inputCB.boundaryConstraints:
                if oddDegree:
                    localBCAssembly[
                        nconstraints - 1, freeBounds[2] : freeBounds[3]
                    ] += inputCB.boundaryConstraints["left"][
                        -nconstraints, freeBounds[2] : freeBounds[3]
                    ]
                    localAssemblyWeights[
                        nconstraints - 1, freeBounds[2] : freeBounds[3]
                    ] += 1.0

                    if nconstraints > 1:
                        initSol[
                            : nconstraints - 1, freeBounds[2] : freeBounds[3]
                        ] = beta * initSol[
                            : nconstraints - 1, freeBounds[2] : freeBounds[3] ] + (1 - beta) * inputCB.boundaryConstraints["left"][
                            -degree - loffset : -nconstraints,
                            freeBounds[2] : freeBounds[3] ]

                        localAssemblyWeights[
                            : nconstraints - 1, freeBounds[2] : freeBounds[3]
                        ] += 1.0
                else:
                    localAssemblyWeights[
                        :nconstraints, freeBounds[2] : freeBounds[3]
                    ] += 1.0
                    initSol[
                        :nconstraints, freeBounds[2] : freeBounds[3]
                    ] = beta * initSol[
                        :nconstraints, freeBounds[2] : freeBounds[3]] + (1 - beta) * inputCB.boundaryConstraints["left"][
                        -degree - loffset : -nconstraints, freeBounds[2] : freeBounds[3] ]

            if "right" in inputCB.boundaryConstraints:
                if oddDegree:
                    localBCAssembly[
                        -nconstraints, freeBounds[2] : freeBounds[3]
                    ] += inputCB.boundaryConstraints["right"][
                        nconstraints - 1, freeBounds[2] : freeBounds[3]
                    ]
                    localAssemblyWeights[
                        -nconstraints, freeBounds[2] : freeBounds[3]
                    ] += 1.0

                    if nconstraints > 1:
                        initSol[
                            -nconstraints + 1 :, freeBounds[2] : freeBounds[3]
                        ] = beta * initSol[
                            -nconstraints + 1 :, freeBounds[2] : freeBounds[3] ] + (1-beta) * inputCB.boundaryConstraints["right"][
                            nconstraints : degree + loffset,
                            freeBounds[2] : freeBounds[3]
                        ]
                        localAssemblyWeights[
                            -nconstraints + 1 :, freeBounds[2] : freeBounds[3]
                        ] += 1.0
                else:
                    localAssemblyWeights[
                        -nconstraints:, freeBounds[2] : freeBounds[3]
                    ] += 1.0
                    initSol[
                        -nconstraints:, freeBounds[2] : freeBounds[3]
                    ] = beta * initSol[
                            -nconstraints:, freeBounds[2] : freeBounds[3] ] + (1-beta) * inputCB.boundaryConstraints["right"][
                        nconstraints : degree + loffset, freeBounds[2] : freeBounds[3]
                    ]

            if "top" in inputCB.boundaryConstraints:
                if oddDegree:
                    localBCAssembly[
                        freeBounds[0] : freeBounds[1], -nconstraints
                    ] += inputCB.boundaryConstraints["top"][
                        freeBounds[0] : freeBounds[1], nconstraints - 1
                    ]
                    localAssemblyWeights[
                        freeBounds[0] : freeBounds[1], -nconstraints
                    ] += 1.0

                    if nconstraints > 1:
                        initSol[
                            freeBounds[0] : freeBounds[1], -nconstraints + 1 :
                        ] = beta * initSol[
                            freeBounds[0] : freeBounds[1], -nconstraints + 1 : ] + (1-beta) * inputCB.boundaryConstraints["top"][
                            freeBounds[0] : freeBounds[1],
                            nconstraints : loffset + degree,
                        ]
                        localAssemblyWeights[
                            freeBounds[0] : freeBounds[1], -nconstraints + 1 :
                        ] += 1.0
                else:
                    initSol[
                        freeBounds[0] : freeBounds[1], -nconstraints:
                    ] = beta * initSol[
                            freeBounds[0] : freeBounds[1], -nconstraints: ] + (1-beta) * inputCB.boundaryConstraints["top"][
                        freeBounds[0] : freeBounds[1], nconstraints : loffset + degree
                    ]
                    localAssemblyWeights[
                        freeBounds[0] : freeBounds[1], -nconstraints:
                    ] += 1.0

            if "bottom" in inputCB.boundaryConstraints:
                if oddDegree:
                    localBCAssembly[
                        freeBounds[0] : freeBounds[1], nconstraints - 1
                    ] += inputCB.boundaryConstraints["bottom"][
                        freeBounds[0] : freeBounds[1], -nconstraints
                    ]
                    localAssemblyWeights[
                        freeBounds[0] : freeBounds[1], nconstraints - 1
                    ] += 1.0

                    if nconstraints > 1:
                        initSol[
                            freeBounds[0] : freeBounds[1], : nconstraints - 1
                        ] = beta * initSol[
                            freeBounds[0] : freeBounds[1], : nconstraints - 1 ] + (1-beta) * inputCB.boundaryConstraints["bottom"][
                            freeBounds[0] : freeBounds[1],
                            -degree - loffset : -nconstraints,
                        ]
                        localAssemblyWeights[
                            freeBounds[0] : freeBounds[1], : nconstraints - 1
                        ] += 1.0
                else:
                    initSol[
                        freeBounds[0] : freeBounds[1], :nconstraints
                    ] = beta * initSol[
                            freeBounds[0] : freeBounds[1], :nconstraints ] + (1-beta) * inputCB.boundaryConstraints["bottom"][
                        freeBounds[0] : freeBounds[1], -degree - loffset : -nconstraints
                    ]
                    localAssemblyWeights[
                        freeBounds[0] : freeBounds[1], :nconstraints
                    ] += 1.0

            if "top-left" in inputCB.boundaryConstraints:
                if oddDegree:
                    localBCAssembly[
                        nconstraints - 1, -nconstraints
                    ] += inputCB.boundaryConstraints["top-left"][
                        -nconstraints, nconstraints - 1
                    ]
                    localAssemblyWeights[nconstraints - 1, -nconstraints] += 1.0

                    if nconstraints > 1:
                        assert freeBounds[0] == nconstraints - 1
                        # initSol[: nconstraints -
                        #         1, -nconstraints + 1:] = 0
                        localAssemblyWeights[
                            : nconstraints - 1, -nconstraints + 1 :
                        ] += 1.0
                        initSol[
                            : nconstraints - 1, -nconstraints + 1 :
                        ] = beta * initSol[
                            : nconstraints - 1, -nconstraints + 1 : ] + (1-beta) * inputCB.boundaryConstraints["top-left"][
                            -degree - loffset : -nconstraints,
                            nconstraints : degree + loffset,
                        ]
                else:
                    initSol[
                        :nconstraints, -nconstraints:
                    ] = beta * initSol[
                            :nconstraints, -nconstraints: ] + (1-beta) * inputCB.boundaryConstraints["top-left"][
                        -degree - loffset : -nconstraints,
                        nconstraints : degree + loffset,
                    ]
                    localAssemblyWeights[:nconstraints, -nconstraints:] += 1.0

            if "bottom-right" in inputCB.boundaryConstraints:
                if oddDegree:
                    localBCAssembly[
                        -nconstraints, nconstraints - 1
                    ] += inputCB.boundaryConstraints["bottom-right"][
                        nconstraints - 1, -nconstraints
                    ]
                    localAssemblyWeights[-nconstraints, nconstraints - 1] += 1.0

                    if nconstraints > 1:
                        assert freeBounds[2] == nconstraints - 1
                        initSol[
                            -nconstraints + 1 :, : nconstraints - 1
                        ] = beta * initSol[
                            -nconstraints + 1 :, : nconstraints - 1 ] + (1-beta) * inputCB.boundaryConstraints["bottom-right"][
                            nconstraints : degree + loffset,
                            -degree - loffset : -nconstraints,
                        ]
                        localAssemblyWeights[
                            -nconstraints + 1 :, : nconstraints - 1
                        ] += 1.0
                else:
                    initSol[
                        -nconstraints:, :nconstraints
                    ] = beta * initSol[
                            -nconstraints:, :nconstraints ] + (1-beta) * inputCB.boundaryConstraints["bottom-right"][
                        nconstraints : degree + loffset,
                        -degree - loffset : -nconstraints,
                    ]
                    localAssemblyWeights[-nconstraints:, :nconstraints] += 1.0

            if "bottom-left" in inputCB.boundaryConstraints:
                if oddDegree:
                    localBCAssembly[
                        nconstraints - 1, nconstraints - 1
                    ] += inputCB.boundaryConstraints["bottom-left"][
                        -nconstraints, -nconstraints
                    ]
                    localAssemblyWeights[nconstraints - 1, nconstraints - 1] += 1.0

                    if nconstraints > 1:
                        assert freeBounds[0] == nconstraints - 1
                        initSol[
                            : nconstraints - 1, : nconstraints - 1
                        ] = beta * initSol[
                            : nconstraints - 1, : nconstraints - 1 ] + (1-beta) * inputCB.boundaryConstraints["bottom-left"][
                            -degree - loffset : -nconstraints,
                            -degree - loffset : -nconstraints,
                        ]
                        localAssemblyWeights[
                            : nconstraints - 1, : nconstraints - 1
                        ] += 1.0
                else:
                    initSol[:nconstraints, :nconstraints] = beta * initSol[
                            :nconstraints, :nconstraints ] + (1-beta) * inputCB.boundaryConstraints[
                        "bottom-left"
                    ][
                        -degree - loffset : -nconstraints,
                        -degree - loffset : -nconstraints,
                    ]
                    localAssemblyWeights[:nconstraints, :nconstraints] += 1.0

            if "top-right" in inputCB.boundaryConstraints:
                if oddDegree:
                    localBCAssembly[
                        -nconstraints, -nconstraints
                    ] += inputCB.boundaryConstraints["top-right"][
                        nconstraints - 1, nconstraints - 1
                    ]
                    localAssemblyWeights[-nconstraints, -nconstraints] += 1.0

                    if nconstraints > 1:
                        initSol[
                            -nconstraints + 1 :, -nconstraints + 1 :
                        ] = beta * initSol[
                            -nconstraints + 1 :, -nconstraints + 1 : ] + (1-beta) * inputCB.boundaryConstraints["top-right"][
                            nconstraints : degree + loffset,
                            nconstraints : degree + loffset,
                        ]
                        localAssemblyWeights[
                            -nconstraints + 1 :, -nconstraints + 1 :
                        ] += 1.0
                else:
                    initSol[
                        -nconstraints:, -nconstraints:
                    ] = beta * initSol[
                            -nconstraints:, -nconstraints: ] + (1-beta) * inputCB.boundaryConstraints["top-right"][
                        nconstraints : degree + loffset, nconstraints : degree + loffset
                    ]
                    localAssemblyWeights[-nconstraints:, -nconstraints:] += 1.0

            localAssemblyWeights[
                freeBounds[0] : freeBounds[1], freeBounds[2] : freeBounds[3]
            ] += 1.0
            initSol = np.divide(initSol + localBCAssembly, localAssemblyWeights)

        return np.copy(initSol)
