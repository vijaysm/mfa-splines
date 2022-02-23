import autograd.numpy as np
import splipy as sp


class ProblemSolver3D:
    def __init__(
        self,
        icBlock,
        coreb,
        xb,
        xl,
        yl,
        zl,
        degree,
        augmentSpanSpace=0,
        useDiagonalBlocks=True,
        verbose=False
    ):
        icBlock.xbounds = [
            xb.min[0],
            xb.max[0],
            xb.min[1],
            xb.max[1],
            xb.min[2],
            xb.max[2],
        ]
        icBlock.corebounds = [
            [coreb.min[0] - xb.min[0], len(xl)],
            [coreb.min[1] - xb.min[1], len(yl)],
            [coreb.min[2] - xb.min[2], len(zl)],
        ]
        # int(nPointsX / nSubDomainsX)
        icBlock.xyzCoordLocal = {
            "x": np.copy(xl[:]),
            "y": np.copy(yl[:]),
            "z": np.copy(zl[:]),
        }
        icBlock.Dmini = np.array([min(xl), min(yl), min(zl)])
        icBlock.Dmaxi = np.array([max(xl), max(yl), max(zl)])
        # Basis function object in x-dir and y-dir
        icBlock.basisFunction = {"x": None, "y": None, "z": None}
        icBlock.decodeOpXYZ = {"x": None, "y": None, "z": None}
        icBlock.knotsAdaptive = {"x": [], "y": [], "z": []}
        icBlock.isClamped = {
            "left": False,
            "right": False,
            "top": False,
            "bottom": False,
            "up": False,
            "down": False,
        }

        self.inputCB = icBlock
        self.degree = degree
        self.augmentSpanSpace = augmentSpanSpace
        self.useDiagonalBlocks = useDiagonalBlocks
        self.dimension = 3
        self.verbose = verbose

    def compute_basis(self, constraints=None):
        # self.inputCB.basisFunction['x'].reparam()
        # self.inputCB.basisFunction['y'].reparam()
        # print("TU = ", self.inputCB.knotsAdaptive['x'], self.inputCB.UVW['x'][0], self.inputCB.UVW['x'][-1], self.inputCB.basisFunction['x'].greville())
        # print("TV = ", self.inputCB.knotsAdaptive['y'], self.inputCB.UVW['y'][0], self.inputCB.UVW['y'][-1], self.inputCB.basisFunction['y'].greville())
        for dir in ["x", "y", "z"]:
            self.inputCB.basisFunction[dir] = sp.BSplineBasis(
                order=self.degree + 1, knots=self.inputCB.knotsAdaptive[dir]
            )
            self.inputCB.NUVW[dir] = np.array(
                self.inputCB.basisFunction[dir].evaluate(self.inputCB.UVW[dir])
            )

        print(
            "Number of basis functions = (%d, %d, %d)".format(
                self.inputCB.basisFunction["x"].num_functions(),
                self.inputCB.basisFunction["y"].num_functions(),
                self.inputCB.basisFunction["z"].num_functions(),
            )
        )

        if constraints is not None:
            for entry in constraints[0]:
                self.inputCB.NUVW["x"][entry, :] = 0.0
                self.inputCB.NUVW["x"][entry, entry] = 1.0
            for entry in constraints[1]:
                self.inputCB.NUVW["y"][entry, :] = 0.0
                self.inputCB.NUVW["y"][entry, entry] = 1.0
            for entry in constraints[2]:
                self.inputCB.NUVW["z"][entry, :] = 0.0
                self.inputCB.NUVW["z"][entry, entry] = 1.0

    def compute_decode_operators(self, RN):
        for dir in ["x", "y", "z"]:
            RN[dir] = (
                self.inputCB.NUVW[dir]
                / np.sum(self.inputCB.NUVW[dir], axis=1)[:, np.newaxis]
            )

    def decode(self, P, RN):
        # DD = np.matmul(np.matmul(np.matmul(RN['x'], P), RN['y'].T).T, RN['z'].T)
        DD = np.einsum(
            "ijk,lk->ijl",
            np.einsum("ijk,lj->ilk", np.einsum("ijk,li->ljk", P, RN["x"]), RN["y"]),
            RN["z"],
        )

        # print("Decoded Shapes: ", DD.shape)

        return DD
        # return np.matmul(RN['z'], np.matmul(np.matmul(RN['x'], P), RN['y'].T))

    def lsqFit(self):

        NTNxInv = np.linalg.inv(
            np.matmul(self.inputCB.decodeOpXYZ["x"].T, self.inputCB.decodeOpXYZ["x"])
        )
        NTNyInv = np.linalg.inv(
            np.matmul(self.inputCB.decodeOpXYZ["y"].T, self.inputCB.decodeOpXYZ["y"])
        )
        NTNzInv = np.linalg.inv(
            np.matmul(self.inputCB.decodeOpXYZ["z"].T, self.inputCB.decodeOpXYZ["z"])
        )
        # np.matmul(np.matmul(np.matmul(RN['x'], P), RN['y'].T).T, RN['z'].T)
        # NyTQNz = np.matmul(self.inputCB.decodeOpXYZ['y'].T, np.matmul(self.inputCB.refSolutionLocal, self.inputCB.decodeOpXYZ['z']))
        # NxTQNy = np.matmul(NyTQNz.T, self.inputCB.decodeOpXYZ['x'])
        # NxTQNy = np.matmul(np.matmul(np.matmul(self.inputCB.decodeOpXYZ['z'].T, self.inputCB.refSolutionLocal), self.inputCB.decodeOpXYZ['y']).T, self.inputCB.decodeOpXYZ['x'])
        NxTQNy = np.einsum(
            "ijk,il->ljk",
            np.einsum(
                "ijk,jl->ilk",
                np.einsum(
                    "ijk,kl->ijl",
                    self.inputCB.refSolutionLocal,
                    self.inputCB.decodeOpXYZ["z"],
                ),
                self.inputCB.decodeOpXYZ["y"],
            ),
            self.inputCB.decodeOpXYZ["x"],
        )
        # npo.matmul(np.matmul(self.inputCB.decodeOpXYZ['y'].T, np.matmul(self.inputCB.refSolutionLocal, self.inputCB.decodeOpXYZ['z'])), self.inputCB.decodeOpXYZ['x'], axes=0).shape
        finsol = np.einsum(
            "ijk,il->ljk",
            np.einsum(
                "ijk,jl->ilk", np.einsum("ijk,kl->ijl", NxTQNy, NTNzInv), NTNyInv
            ),
            NTNxInv,
        ).T

        # Rotate axis so that we can do logical indexing
        finsol = np.moveaxis(finsol, 0, -1)
        finsol = np.moveaxis(finsol, 0, 1)

        return finsol
        # sol = np.moveaxis(finsol, 0, -1)
        # return sol
        # return np.matmul(NTNxInv, np.matmul(NTNyInv, np.einsum('ijk,kl->ijl', NxTQNy, NTNzInv)))

        # NTNx = np.matmul(self.inputCB.decodeOpXYZ['x'].T, self.inputCB.decodeOpXYZ['x'])
        # NTNy = np.matmul(self.inputCB.decodeOpXYZ['y'].T, self.inputCB.decodeOpXYZ['y'])
        # NTNz = np.matmul(self.inputCB.decodeOpXYZ['z'].T, self.inputCB.decodeOpXYZ['z'])
        # RHS = np.matmul(np.matmul(self.inputCB.decodeOpXYZ['y'].T, np.matmul(self.inputCB.refSolutionLocal, self.inputCB.decodeOpXYZ['z'])).T, self.inputCB.decodeOpXYZ['x'])
        # Oper = np.einsum('ijk,jk->ijk', np.einsum('ij,jk->ijk', NTNx, NTNy), NTNz)
        # sol = linalg.lstsq(Oper, RHS)[0]

        return sol

    def update_bounds(self):
        return [
            self.inputCB.corebounds[0][0],
            self.inputCB.corebounds[0][1],
            self.inputCB.corebounds[1][0],
            self.inputCB.corebounds[1][1],
            self.inputCB.corebounds[2][0],
            self.inputCB.corebounds[2][1],
        ]

    def residual(self, Pin, printVerbose=False):

        decoded = self.decode(Pin, self.decodeOpXYZ)
        residual_decoded = (self.refSolutionLocal - decoded) / solutionRange
        residual_decoded = residual_decoded.reshape(-1)
        decoded_residual_norm = np.sqrt(
            np.sum(residual_decoded**2) / len(residual_decoded)
        )
        if type(Pin) is not np.numpy_boxes.ArrayBox and printVerbose:
            print("Residual {0}-d: {1}".format(self.dimension, decoded_residual_norm))
        return decoded_residual_norm

    def send_diy(self, inputCB, cp):

        oddDegree = self.degree % 2
        nconstraints = self.augmentSpanSpace + (
            int(self.degree / 2.0) if not oddDegree else int((self.degree + 1) / 2.0)
        )
        loffset = self.degree + 2 * self.augmentSpanSpace

        # if len(inputCB.controlPointData):
        #     inputCB.controlPointData = np.moveaxis(inputCB.controlPointData, 0, -1)
        #     inputCB.controlPointData = np.moveaxis(inputCB.controlPointData, 0, 1)

        link = cp.link()
        for i in range(len(link)):
            target = link.target(i)  # can access gid and proc
            if len(inputCB.controlPointData):
                dir = link.direction(i)
                if dir[0] == 0 and dir[1] == 0 and dir[2] == 0:
                    continue

                direction = (
                    (dir[2] + 1) * self.dimension**2
                    + (dir[1] + 1) * self.dimension
                    + (dir[0] + 1)
                )
                if self.verbose:
                    print(
                        "sending: ",
                        cp.gid(),
                        "to",
                        target.gid,
                        "-- Dir = ",
                        dir,
                        "Direction = ",
                        direction,
                    )

                if dir[2] == 0:  # same z-height layer
                    # ONLY consider coupling through faces and not through verties
                    # This means either dir[0] or dir[1] has to be "0" for subdomain coupling to be active
                    # Hence we only consider 4 neighbor cases, instead of 8.
                    if dir[0] == 0:  # target is coupled in Y-direction
                        if dir[1] > 0:  # target block is above current subdomain
                            cp.enqueue(
                                target, inputCB.controlPointData[:, -loffset:, :]
                            )
                            # cp.enqueue(target, inputCB.controlPointData)
                            cp.enqueue(
                                target,
                                inputCB.knotsAdaptive["y"][
                                    -1 : -2 - self.degree - self.augmentSpanSpace : -1
                                ],
                            )

                        else:  # target block is below current subdomain
                            cp.enqueue(target, inputCB.controlPointData[:, :loffset, :])
                            # cp.enqueue(target, inputCB.controlPointData)
                            cp.enqueue(
                                target,
                                inputCB.knotsAdaptive["y"][
                                    0 : 1 + self.degree + self.augmentSpanSpace
                                ],
                            )

                    # target is coupled in X-direction
                    elif dir[1] == 0:
                        if (
                            dir[0] > 0
                        ):  # target block is to the right of current subdomain
                            cp.enqueue(
                                target, inputCB.controlPointData[-loffset:, :, :]
                            )
                            # cp.enqueue(target, inputCB.controlPointData)
                            cp.enqueue(
                                target,
                                inputCB.knotsAdaptive["x"][
                                    -1 : -2 - self.degree - self.augmentSpanSpace : -1
                                ],
                            )

                        else:  # target block is to the left of current subdomain
                            cp.enqueue(target, inputCB.controlPointData[:loffset, :, :])
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
                                    target,
                                    inputCB.controlPointData[-loffset:, -loffset:, :],
                                )

                            # target block is diagonally top left to current subdomain
                            if dir[0] < 0 and dir[1] > 0:
                                cp.enqueue(
                                    target,
                                    inputCB.controlPointData[:loffset:, -loffset:, :],
                                )

                            # target block is diagonally left bottom  current subdomain
                            if dir[0] < 0 and dir[1] < 0:
                                cp.enqueue(
                                    target,
                                    inputCB.controlPointData[:loffset:, :loffset, :],
                                )

                            # target block is diagonally right bottom of current subdomain
                            if dir[0] > 0 and dir[1] < 0:
                                cp.enqueue(
                                    target,
                                    inputCB.controlPointData[-loffset:, :loffset, :],
                                )

                elif dir[2] > 0:  # communication to layer above in z-direction
                    # ONLY consider coupling through faces and not through verties
                    # This means either dir[0] or dir[1] has to be "0" for subdomain coupling to be active
                    # Hence we only consider 4 neighbor cases, instead of 8.
                    if dir[0] == 0:  # target is coupled in Y-direction
                        if (
                            dir[1] == 0
                        ):  # target block is directly above in z-direction (but same x-y)
                            cp.enqueue(
                                target, inputCB.controlPointData[:, :, -loffset:]
                            )
                            # cp.enqueue(target, inputCB.controlPointData)
                            cp.enqueue(
                                target,
                                inputCB.knotsAdaptive["z"][
                                    -1 : -2 - self.degree - self.augmentSpanSpace : -1
                                ],
                            )
                            # cp.enqueue(
                            #     target, inputCB.knotsAdaptive['y'][-1:-2-degree-augmentSpanSpace:-1])
                        elif (
                            dir[1] > 0
                        ):  # target block is above current subdomain in y-direction
                            cp.enqueue(
                                target,
                                inputCB.controlPointData[:, -loffset:, -loffset:],
                            )
                            # cp.enqueue(target, inputCB.controlPointData)
                            # cp.enqueue(target, inputCB.knotsAdaptive['z'][-1:-2-degree-augmentSpanSpace:-1])
                            # cp.enqueue(
                            #     target, inputCB.knotsAdaptive['y'][-1:-2-degree-augmentSpanSpace:-1])

                        else:  # target block is below current subdomain in y-direction
                            cp.enqueue(
                                target, inputCB.controlPointData[:, :loffset, -loffset:]
                            )
                            # cp.enqueue(target, inputCB.controlPointData)
                            # cp.enqueue(target, inputCB.knotsAdaptive['z'][-1:-2-degree-augmentSpanSpace:-1])
                            # cp.enqueue(
                            #     target, inputCB.knotsAdaptive['y'][0:1+degree+augmentSpanSpace])

                    # target is coupled in X-direction
                    elif dir[1] == 0:
                        if (
                            dir[0] > 0
                        ):  # target block is to the right of current subdomain
                            cp.enqueue(
                                target,
                                inputCB.controlPointData[-loffset:, :, -loffset:],
                            )
                            # cp.enqueue(target, inputCB.controlPointData)
                            # cp.enqueue(
                            #     target, inputCB.knotsAdaptive['z'][0:1+self.degree+self.augmentSpanSpace])
                            # cp.enqueue(
                            #     target, inputCB.knotsAdaptive['x'][-1:-2-self.degree-self.augmentSpanSpace:-1])

                        else:  # target block is to the left of current subdomain
                            cp.enqueue(
                                target, inputCB.controlPointData[:loffset, :, -loffset:]
                            )
                            # cp.enqueue(target, inputCB.controlPointData)
                            # cp.enqueue(target, inputCB.knotsAdaptive['z'][-1:-2-self.degree-self.augmentSpanSpace:-1])
                            # cp.enqueue(target, inputCB.knotsAdaptive['x'][0:(
                            #     self.degree+self.augmentSpanSpace+1)])

                    else:

                        if self.useDiagonalBlocks:
                            # target block is diagonally top right to current subdomain
                            if dir[0] > 0 and dir[1] > 0:
                                cp.enqueue(
                                    target,
                                    inputCB.controlPointData[
                                        -loffset:, -loffset:, -loffset:
                                    ],
                                )

                            # target block is diagonally top left to current subdomain
                            if dir[0] < 0 and dir[1] > 0:
                                cp.enqueue(
                                    target,
                                    inputCB.controlPointData[
                                        :loffset:, -loffset:, -loffset:
                                    ],
                                )

                            # target block is diagonally left bottom  current subdomain
                            if dir[0] < 0 and dir[1] < 0:
                                cp.enqueue(
                                    target,
                                    inputCB.controlPointData[
                                        :loffset:, :loffset, -loffset:
                                    ],
                                )

                            # target block is diagonally right bottom of current subdomain
                            if dir[0] > 0 and dir[1] < 0:
                                cp.enqueue(
                                    target,
                                    inputCB.controlPointData[
                                        -loffset:, :loffset, -loffset:
                                    ],
                                )

                else:  # dir[2] < 0 - sending to layer of blocks below in z-direction
                    # ONLY consider coupling through faces and not through verties
                    # This means either dir[0] or dir[1] has to be "0" for subdomain coupling to be active
                    # Hence we only consider 4 neighbor cases, instead of 8.
                    if dir[0] == 0:  # target is coupled in Y-direction
                        if (
                            dir[1] == 0
                        ):  # target block is directly below in z-direction (but same x-y)
                            cp.enqueue(target, inputCB.controlPointData[:, :, :loffset])
                            # cp.enqueue(target, inputCB.controlPointData)
                            cp.enqueue(
                                target,
                                inputCB.knotsAdaptive["z"][
                                    0 : 1 + self.degree + self.augmentSpanSpace
                                ],
                            )
                            # cp.enqueue(
                            #     target, inputCB.knotsAdaptive['y'][-1:-2-degree-augmentSpanSpace:-1])

                        elif (
                            dir[1] > 0
                        ):  # target block is above current subdomain in y-direction
                            cp.enqueue(
                                target, inputCB.controlPointData[:, -loffset:, :loffset]
                            )
                            # cp.enqueue(target, inputCB.controlPointData)
                            # cp.enqueue(target, inputCB.knotsAdaptive['z'][:loffset])
                            # cp.enqueue(target, inputCB.knotsAdaptive['y'][-1:-2-self.degree-self.augmentSpanSpace:-1])

                        else:  # target block is below current subdomain in y-direction
                            cp.enqueue(
                                target, inputCB.controlPointData[:, :loffset, :loffset]
                            )
                            # cp.enqueue(target, inputCB.controlPointData)
                            # cp.enqueue(target, inputCB.knotsAdaptive['z'][:loffset])
                            # cp.enqueue(target, inputCB.knotsAdaptive['y'][0:1+self.degree+self.augmentSpanSpace])

                    # target is coupled in X-direction
                    elif dir[1] == 0:
                        if (
                            dir[0] > 0
                        ):  # target block is to the right of current subdomain
                            cp.enqueue(
                                target, inputCB.controlPointData[-loffset:, :, :loffset]
                            )
                            # cp.enqueue(target, inputCB.controlPointData)
                            # cp.enqueue(target, inputCB.knotsAdaptive['z'][:loffset])
                            # cp.enqueue(target, inputCB.knotsAdaptive['x'][-1:-2-self.degree-self.augmentSpanSpace:-1])

                        else:  # target block is to the left of current subdomain
                            cp.enqueue(
                                target, inputCB.controlPointData[:loffset, :, :loffset]
                            )
                            # cp.enqueue(target, inputCB.controlPointData)
                            # cp.enqueue(target, inputCB.knotsAdaptive['z'][:loffset])
                            # cp.enqueue(target, inputCB.knotsAdaptive['x'][0:(self.degree+self.augmentSpanSpace+1)])

                    else:

                        if self.useDiagonalBlocks:
                            # target block is diagonally top right to current subdomain
                            if dir[0] > 0 and dir[1] > 0:
                                cp.enqueue(
                                    target,
                                    inputCB.controlPointData[
                                        -loffset:, -loffset:, :loffset
                                    ],
                                )

                            # target block is diagonally top left to current subdomain
                            if dir[0] < 0 and dir[1] > 0:
                                cp.enqueue(
                                    target,
                                    inputCB.controlPointData[
                                        :loffset:, -loffset:, :loffset
                                    ],
                                )

                            # target block is diagonally left bottom  current subdomain
                            if dir[0] < 0 and dir[1] < 0:
                                cp.enqueue(
                                    target,
                                    inputCB.controlPointData[
                                        :loffset:, :loffset, :loffset
                                    ],
                                )

                            # target block is diagonally right bottom of current subdomain
                            if dir[0] > 0 and dir[1] < 0:
                                cp.enqueue(
                                    target,
                                    inputCB.controlPointData[
                                        -loffset:, :loffset, :loffset
                                    ],
                                )

        # if len(inputCB.controlPointData):
        #     inputCB.controlPointData = np.moveaxis(inputCB.controlPointData, -1, 0)
        #     inputCB.controlPointData = np.moveaxis(inputCB.controlPointData, 1, 2)

        return

    def recv_diy(self, inputCB, cp):

        link = cp.link()
        for i in range(len(link)):
            target = link.target(i).gid
            dir = link.direction(i)
            # print("%d received from %d: %s from direction %s, with sizes %d+%d" % (cp.gid(), tgid, o, dir, pl, tl))

            # ONLY consider coupling through faces and not through verties
            # This means either dir[0] or dir[1] has to be "0" for subdomain coupling to be active
            # Hence we only consider 4 neighbor cases, instead of 8.
            if dir[0] == 0 and dir[1] == 0 and dir[2] == 0:
                continue

            direction = (
                (dir[2] + 1) * self.dimension**2
                + (dir[1] + 1) * self.dimension
                + (dir[0] + 1)
            )
            if self.verbose:
                print(
                    "receiving: ",
                    cp.gid(),
                    "from",
                    target,
                    "-- Dir = ",
                    dir,
                    "Direction = ",
                    direction,
                )

            if dir[2] == 0:  # same z-height layer
                # ONLY consider coupling through faces and not through verties
                # This means either dir[0] or dir[1] has to be "0" for subdomain coupling to be active
                # Hence we only consider 4 neighbor cases, instead of 8.
                if dir[0] == 0:  # target is coupled in Y-direction
                    if dir[1] > 0:  # target block is above current subdomain
                        inputCB.boundaryConstraints["top"] = cp.dequeue(target)
                        inputCB.ghostKnots["top"] = cp.dequeue(target)

                    else:  # target block is below current subdomain
                        inputCB.boundaryConstraints["bottom"] = cp.dequeue(target)
                        inputCB.ghostKnots["bottom"] = cp.dequeue(target)

                # target is coupled in X-direction
                elif dir[1] == 0:
                    if dir[0] > 0:  # target block is to the right of current subdomain
                        inputCB.boundaryConstraints["right"] = cp.dequeue(target)
                        inputCB.ghostKnots["right"] = cp.dequeue(target)

                    else:  # target block is to the left of current subdomain
                        inputCB.boundaryConstraints["left"] = cp.dequeue(target)
                        inputCB.ghostKnots["left"] = cp.dequeue(target)

                else:

                    if self.useDiagonalBlocks:

                        # 2-Dimension = 0: left, 1: right, 2: top, 3: bottom, 4: top-left, 5: top-right, 6: bottom-left, 7: bottom-right
                        # sender block is diagonally right top to  current subdomain
                        if dir[0] > 0 and dir[1] > 0:
                            inputCB.boundaryConstraints["top-right"] = cp.dequeue(
                                target
                            )

                        # sender block is diagonally left top to current subdomain
                        if dir[0] > 0 and dir[1] < 0:
                            inputCB.boundaryConstraints["bottom-right"] = cp.dequeue(
                                target
                            )

                        # sender block is diagonally left bottom  current subdomain
                        if dir[0] < 0 and dir[1] < 0:
                            inputCB.boundaryConstraints["bottom-left"] = cp.dequeue(
                                target
                            )

                        # sender block is diagonally left to current subdomain
                        if dir[0] < 0 and dir[1] > 0:
                            inputCB.boundaryConstraints["top-left"] = cp.dequeue(target)

            elif dir[2] > 0:  # communication to layer above in z-direction
                # ONLY consider coupling through faces and not through verties
                # This means either dir[0] or dir[1] has to be "0" for subdomain coupling to be active
                # Hence we only consider 4 neighbor cases, instead of 8.
                if dir[0] == 0:

                    if (
                        dir[1] == 0
                    ):  # target block is directly above in z-direction (but same x-y)
                        inputCB.boundaryConstraints["up"] = cp.dequeue(target)
                        inputCB.ghostKnots["up"] = cp.dequeue(target)

                    elif (
                        dir[1] > 0
                    ):  # target block is above current subdomain in y-direction
                        inputCB.boundaryConstraints["up-top"] = cp.dequeue(target)

                    else:  # target block is below current subdomain in y-direction
                        inputCB.boundaryConstraints["up-bottom"] = cp.dequeue(target)

                # target is coupled in X-direction
                elif dir[1] == 0:
                    if dir[0] > 0:  # target block is to the right of current subdomain
                        inputCB.boundaryConstraints["up-right"] = cp.dequeue(target)

                    else:  # target block is to the left of current subdomain
                        inputCB.boundaryConstraints["up-left"] = cp.dequeue(target)

                else:

                    if self.useDiagonalBlocks:

                        # 2-Dimension = 0: left, 1: right, 2: top, 3: bottom, 4: top-left, 5: top-right, 6: bottom-left, 7: bottom-right
                        # sender block is diagonally right top to  current subdomain
                        if dir[0] > 0 and dir[1] > 0:
                            inputCB.boundaryConstraints["up-top-right"] = cp.dequeue(
                                target
                            )

                        # sender block is diagonally left top to current subdomain
                        if dir[0] > 0 and dir[1] < 0:
                            inputCB.boundaryConstraints["up-bottom-right"] = cp.dequeue(
                                target
                            )

                        # sender block is diagonally left bottom  current subdomain
                        if dir[0] < 0 and dir[1] < 0:
                            inputCB.boundaryConstraints["up-bottom-left"] = cp.dequeue(
                                target
                            )

                        # sender block is diagonally left to current subdomain
                        if dir[0] < 0 and dir[1] > 0:
                            inputCB.boundaryConstraints["up-top-left"] = cp.dequeue(
                                target
                            )

            else:  # dir[2] < 0 - sending to layer of blocks below in z-direction
                # ONLY consider coupling through faces and not through verties
                # This means either dir[0] or dir[1] has to be "0" for subdomain coupling to be active
                # Hence we only consider 4 neighbor cases, instead of 8.
                if dir[0] == 0:
                    if (
                        dir[1] == 0
                    ):  # target block is directly above in z-direction (but same x-y)
                        inputCB.boundaryConstraints["down"] = cp.dequeue(target)
                        inputCB.ghostKnots["down"] = cp.dequeue(target)

                    elif (
                        dir[1] > 0
                    ):  # target block is above current subdomain in y-direction
                        inputCB.boundaryConstraints["down-top"] = cp.dequeue(target)

                    else:  # target block is below current subdomain in y-direction
                        inputCB.boundaryConstraints["down-bottom"] = cp.dequeue(target)

                # target is coupled in X-direction
                elif dir[1] == 0:
                    if dir[0] > 0:  # target block is to the right of current subdomain
                        inputCB.boundaryConstraints["down-right"] = cp.dequeue(target)

                    else:  # target block is to the left of current subdomain
                        inputCB.boundaryConstraints["down-left"] = cp.dequeue(target)

                else:

                    if self.useDiagonalBlocks:

                        # 2-Dimension = 0: left, 1: right, 2: top, 3: bottom, 4: top-left, 5: top-right, 6: bottom-left, 7: bottom-right
                        # sender block is diagonally right top to  current subdomain
                        if dir[0] > 0 and dir[1] > 0:
                            inputCB.boundaryConstraints["down-top-right"] = cp.dequeue(
                                target
                            )

                        # sender block is diagonally left top to current subdomain
                        if dir[0] > 0 and dir[1] < 0:
                            inputCB.boundaryConstraints[
                                "down-bottom-right"
                            ] = cp.dequeue(target)

                        # sender block is diagonally left bottom  current subdomain
                        if dir[0] < 0 and dir[1] < 0:
                            inputCB.boundaryConstraints[
                                "down-bottom-left"
                            ] = cp.dequeue(target)

                        # sender block is diagonally left to current subdomain
                        if dir[0] < 0 and dir[1] > 0:
                            inputCB.boundaryConstraints["down-top-left"] = cp.dequeue(
                                target
                            )

        return

    def initialize_solution(
        self, inputCB, initSol, degree, augmentSpanSpace, fullyPinned
    ):

        alpha = 0.5
        beta = 0.0
        oddDegree = degree % 2
        localAssemblyWeights = np.zeros(initSol.shape)
        localBCAssembly = np.zeros(initSol.shape)
        freeBounds = [0, initSol.shape[0], 0, initSol.shape[1], 0, initSol.shape[2]]

        # Rotate axis so that we can do logical indexing
        # initSol = np.moveaxis(initSol, 0, -1)
        # initSol = np.moveaxis(initSol, 0, 1)

        if fullyPinned:
            # First update hte control point vector with constraints for supporting points
            if "left" in inputCB.boundaryConstraints:
                initSol[:, :, 0] += inputCB.boundaryConstraints["left"][:, :, -1]
                localAssemblyWeights[0, :, :] += 1.0
            if "right" in inputCB.boundaryConstraints:
                initSol[:, :, -1] += inputCB.boundaryConstraints["right"][:, :, 0]
                localAssemblyWeights[-1, :, :] += 1.0
            if "top" in inputCB.boundaryConstraints:
                initSol[:, -1, :] += inputCB.boundaryConstraints["top"][:, 0, :]
                localAssemblyWeights[:, -1, :] += 1.0
            if "bottom" in inputCB.boundaryConstraints:
                initSol[:, 0, :] += inputCB.boundaryConstraints["bottom"][:, -1, :]
                localAssemblyWeights[:, 0, :] += 1.0
            if "up" in inputCB.boundaryConstraints:
                initSol[:, :, -1] += inputCB.boundaryConstraints["up"][:, :, 0]
                localAssemblyWeights[:, :, -1] += 1.0
            if "down" in inputCB.boundaryConstraints:
                initSol[:, :, 0] += inputCB.boundaryConstraints["down"][:, :, -1]
                localAssemblyWeights[:, :, 0] += 1.0

            if "top-left" in inputCB.boundaryConstraints:
                initSol[0, -1, :] += inputCB.boundaryConstraints["top-left"][-1, 0, :]
                localAssemblyWeights[0, -1, :] += 1.0
            if "bottom-right" in inputCB.boundaryConstraints:
                initSol[-1, 0, :] = inputCB.boundaryConstraints["bottom-right"][
                    0, -1, :
                ]
                localAssemblyWeights[-1, 0, :] += 1.0
            if "bottom-left" in inputCB.boundaryConstraints:
                initSol[0, 0, :] = inputCB.boundaryConstraints["bottom-left"][-1, -1, :]
                localAssemblyWeights[0, 0, :] += 1.0
            if "top-right" in inputCB.boundaryConstraints:
                initSol[-1, -1, :] = inputCB.boundaryConstraints["top-right"][0, 0, :]
                localAssemblyWeights[-1, -1, :] += 1.0
        else:
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
                initSol.shape[0]
                if inputCB.isClamped["right"]
                else initSol.shape[0]
                - (nconstraints - 1 if oddDegree else nconstraints)
            )

            freeBounds[2] = (
                0
                if inputCB.isClamped["bottom"]
                else (nconstraints - 1 if oddDegree else nconstraints)
            )
            freeBounds[3] = (
                initSol.shape[1]
                if inputCB.isClamped["top"]
                else initSol.shape[1]
                - (nconstraints - 1 if oddDegree else nconstraints)
            )

            freeBounds[4] = (
                0
                if inputCB.isClamped["down"]
                else (nconstraints - 1 if oddDegree else nconstraints)
            )
            freeBounds[5] = (
                initSol.shape[2]
                if inputCB.isClamped["up"]
                else initSol.shape[2]
                - (nconstraints - 1 if oddDegree else nconstraints)
            )

            if "left" in inputCB.boundaryConstraints:
                if oddDegree:
                    localBCAssembly[
                        nconstraints - 1,
                        freeBounds[2] : freeBounds[3],
                        freeBounds[4] : freeBounds[5],
                    ] += inputCB.boundaryConstraints["left"][
                        -nconstraints,
                        freeBounds[2] : freeBounds[3],
                        freeBounds[4] : freeBounds[5],
                    ]
                    localAssemblyWeights[
                        nconstraints - 1,
                        freeBounds[2] : freeBounds[3],
                        freeBounds[4] : freeBounds[5],
                    ] += 1.0

                    if nconstraints > 1:
                        initSol[
                            : nconstraints - 1,
                            freeBounds[2] : freeBounds[3],
                            freeBounds[4] : freeBounds[5],
                        ] = inputCB.boundaryConstraints["left"][
                            -degree - loffset : -nconstraints,
                            freeBounds[2] : freeBounds[3],
                            freeBounds[4] : freeBounds[5],
                        ]
                        localAssemblyWeights[
                            : nconstraints - 1,
                            freeBounds[2] : freeBounds[3],
                            freeBounds[4] : freeBounds[5],
                        ] = 1.0
                else:
                    localAssemblyWeights[
                        :nconstraints,
                        freeBounds[2] : freeBounds[3],
                        freeBounds[4] : freeBounds[5],
                    ] = 1.0
                    initSol[
                        :nconstraints,
                        freeBounds[2] : freeBounds[3],
                        freeBounds[4] : freeBounds[5],
                    ] = inputCB.boundaryConstraints["left"][
                        -degree - loffset : -nconstraints,
                        freeBounds[2] : freeBounds[3],
                        freeBounds[4] : freeBounds[5],
                    ]

            if "right" in inputCB.boundaryConstraints:
                if oddDegree:
                    localBCAssembly[
                        -nconstraints,
                        freeBounds[2] : freeBounds[3],
                        freeBounds[4] : freeBounds[5],
                    ] += inputCB.boundaryConstraints["right"][
                        nconstraints - 1,
                        freeBounds[2] : freeBounds[3],
                        freeBounds[4] : freeBounds[5],
                    ]
                    localAssemblyWeights[
                        -nconstraints,
                        freeBounds[2] : freeBounds[3],
                        freeBounds[4] : freeBounds[5],
                    ] += 1.0

                    if nconstraints > 1:
                        initSol[
                            -nconstraints + 1 :,
                            freeBounds[2] : freeBounds[3],
                            freeBounds[4] : freeBounds[5],
                        ] = inputCB.boundaryConstraints["right"][
                            nconstraints : degree + loffset,
                            freeBounds[2] : freeBounds[3],
                            freeBounds[4] : freeBounds[5],
                        ]
                        localAssemblyWeights[
                            -nconstraints + 1 :,
                            freeBounds[2] : freeBounds[3],
                            freeBounds[4] : freeBounds[5],
                        ] = 1.0
                else:
                    localAssemblyWeights[
                        -nconstraints:,
                        freeBounds[2] : freeBounds[3],
                        freeBounds[4] : freeBounds[5],
                    ] = 1.0
                    initSol[
                        -nconstraints:,
                        freeBounds[2] : freeBounds[3],
                        freeBounds[4] : freeBounds[5],
                    ] = inputCB.boundaryConstraints["right"][
                        nconstraints : degree + loffset,
                        freeBounds[2] : freeBounds[3],
                        freeBounds[4] : freeBounds[5],
                    ]

            if "top" in inputCB.boundaryConstraints:
                if oddDegree:
                    localBCAssembly[
                        freeBounds[0] : freeBounds[1],
                        -nconstraints,
                        freeBounds[4] : freeBounds[5],
                    ] += inputCB.boundaryConstraints["top"][
                        freeBounds[0] : freeBounds[1],
                        nconstraints - 1,
                        freeBounds[4] : freeBounds[5],
                    ]
                    localAssemblyWeights[
                        freeBounds[0] : freeBounds[1],
                        -nconstraints,
                        freeBounds[4] : freeBounds[5],
                    ] += 1.0

                    if nconstraints > 1:
                        initSol[
                            freeBounds[0] : freeBounds[1],
                            -nconstraints + 1 :,
                            freeBounds[4] : freeBounds[5],
                        ] = inputCB.boundaryConstraints["top"][
                            freeBounds[0] : freeBounds[1],
                            nconstraints : loffset + degree,
                            freeBounds[4] : freeBounds[5],
                        ]
                        localAssemblyWeights[
                            freeBounds[0] : freeBounds[1],
                            -nconstraints + 1 :,
                            freeBounds[4] : freeBounds[5],
                        ] = 1.0
                else:
                    initSol[
                        freeBounds[0] : freeBounds[1],
                        -nconstraints:,
                        freeBounds[4] : freeBounds[5],
                    ] = inputCB.boundaryConstraints["top"][
                        freeBounds[0] : freeBounds[1],
                        nconstraints : loffset + degree,
                        freeBounds[4] : freeBounds[5],
                    ]
                    localAssemblyWeights[
                        freeBounds[0] : freeBounds[1],
                        -nconstraints:,
                        freeBounds[4] : freeBounds[5],
                    ] = 1.0

            if "bottom" in inputCB.boundaryConstraints:
                if oddDegree:
                    localBCAssembly[
                        freeBounds[0] : freeBounds[1],
                        nconstraints - 1,
                        freeBounds[4] : freeBounds[5],
                    ] += inputCB.boundaryConstraints["bottom"][
                        freeBounds[0] : freeBounds[1],
                        -nconstraints,
                        freeBounds[4] : freeBounds[5],
                    ]
                    localAssemblyWeights[
                        freeBounds[0] : freeBounds[1],
                        nconstraints - 1,
                        freeBounds[4] : freeBounds[5],
                    ] += 1.0

                    if nconstraints > 1:
                        initSol[
                            freeBounds[0] : freeBounds[1],
                            : nconstraints - 1,
                            freeBounds[4] : freeBounds[5],
                        ] = inputCB.boundaryConstraints["bottom"][
                            freeBounds[0] : freeBounds[1],
                            -degree - loffset : -nconstraints,
                            freeBounds[4] : freeBounds[5],
                        ]
                        localAssemblyWeights[
                            freeBounds[0] : freeBounds[1],
                            : nconstraints - 1,
                            freeBounds[4] : freeBounds[5],
                        ] = 1.0
                else:
                    initSol[
                        freeBounds[0] : freeBounds[1],
                        :nconstraints,
                        freeBounds[4] : freeBounds[5],
                    ] = inputCB.boundaryConstraints["bottom"][
                        freeBounds[0] : freeBounds[1],
                        -degree - loffset : -nconstraints,
                        freeBounds[4] : freeBounds[5],
                    ]
                    localAssemblyWeights[
                        freeBounds[0] : freeBounds[1],
                        :nconstraints,
                        freeBounds[4] : freeBounds[5],
                    ] = 1.0

            if "down" in inputCB.boundaryConstraints:
                if oddDegree:
                    localBCAssembly[
                        freeBounds[0] : freeBounds[1],
                        freeBounds[2] : freeBounds[3],
                        nconstraints - 1,
                    ] += inputCB.boundaryConstraints["down"][
                        freeBounds[0] : freeBounds[1],
                        freeBounds[2] : freeBounds[3],
                        -nconstraints,
                    ]
                    localAssemblyWeights[
                        freeBounds[0] : freeBounds[1],
                        freeBounds[2] : freeBounds[3],
                        nconstraints - 1,
                    ] += 1.0

                    if nconstraints > 1:
                        initSol[
                            freeBounds[0] : freeBounds[1],
                            freeBounds[2] : freeBounds[3],
                            : nconstraints - 1,
                        ] = inputCB.boundaryConstraints["down"][
                            freeBounds[0] : freeBounds[1],
                            freeBounds[2] : freeBounds[3],
                            -degree - loffset : -nconstraints,
                        ]
                        localAssemblyWeights[
                            freeBounds[0] : freeBounds[1],
                            freeBounds[2] : freeBounds[3],
                            : nconstraints - 1,
                        ] = 1.0
                else:
                    initSol[
                        freeBounds[0] : freeBounds[1],
                        freeBounds[2] : freeBounds[3],
                        :nconstraints,
                    ] = inputCB.boundaryConstraints["down"][
                        freeBounds[0] : freeBounds[1],
                        freeBounds[2] : freeBounds[3],
                        -degree - loffset : -nconstraints,
                    ]
                    localAssemblyWeights[
                        freeBounds[0] : freeBounds[1],
                        freeBounds[2] : freeBounds[3],
                        :nconstraints,
                    ] = 1.0

            if "up" in inputCB.boundaryConstraints:
                if oddDegree:
                    localBCAssembly[
                        freeBounds[0] : freeBounds[1],
                        freeBounds[2] : freeBounds[3],
                        -nconstraints,
                    ] += inputCB.boundaryConstraints["up"][
                        freeBounds[0] : freeBounds[1],
                        freeBounds[2] : freeBounds[3],
                        nconstraints - 1,
                    ]
                    localAssemblyWeights[
                        freeBounds[0] : freeBounds[1],
                        freeBounds[2] : freeBounds[3],
                        -nconstraints,
                    ] += 1.0

                    if nconstraints > 1:
                        initSol[
                            freeBounds[0] : freeBounds[1],
                            freeBounds[2] : freeBounds[3],
                            -nconstraints + 1 :,
                        ] = inputCB.boundaryConstraints["up"][
                            freeBounds[0] : freeBounds[1],
                            freeBounds[2] : freeBounds[3],
                            nconstraints : loffset + degree,
                        ]
                        localAssemblyWeights[
                            freeBounds[0] : freeBounds[1],
                            freeBounds[2] : freeBounds[3],
                            -nconstraints + 1 :,
                        ] = 1.0
                else:
                    initSol[
                        freeBounds[0] : freeBounds[1],
                        freeBounds[2] : freeBounds[3],
                        -nconstraints:,
                    ] = inputCB.boundaryConstraints["up"][
                        freeBounds[0] : freeBounds[1],
                        freeBounds[2] : freeBounds[3],
                        nconstraints : loffset + degree,
                    ]
                    localAssemblyWeights[
                        freeBounds[0] : freeBounds[1],
                        freeBounds[2] : freeBounds[3],
                        -nconstraints:,
                    ] = 1.0

            if "top-left" in inputCB.boundaryConstraints:
                if oddDegree:
                    localBCAssembly[
                        nconstraints - 1, -nconstraints, :
                    ] += inputCB.boundaryConstraints["top-left"][
                        -nconstraints, nconstraints - 1, :
                    ]
                    localAssemblyWeights[nconstraints - 1, -nconstraints, :] += 1.0

                    if nconstraints > 1:
                        assert freeBounds[0] == nconstraints - 1
                        # initSol[: nconstraints -
                        #         1, -nconstraints + 1:, :] = 0
                        localAssemblyWeights[
                            : nconstraints - 1, -nconstraints + 1 :, :
                        ] = 1.0
                        initSol[
                            : nconstraints - 1, -nconstraints + 1 :, :
                        ] = inputCB.boundaryConstraints["top-left"][
                            -degree - loffset : -nconstraints,
                            nconstraints : degree + loffset,
                            :,
                        ]
                else:
                    initSol[
                        :nconstraints, -nconstraints:, :
                    ] = inputCB.boundaryConstraints["top-left"][
                        -degree - loffset : -nconstraints,
                        nconstraints : degree + loffset,
                        :,
                    ]
                    localAssemblyWeights[:nconstraints, -nconstraints:, :] = 1.0

            if "bottom-right" in inputCB.boundaryConstraints:
                if oddDegree:
                    localBCAssembly[
                        -nconstraints, nconstraints - 1, :
                    ] += inputCB.boundaryConstraints["bottom-right"][
                        nconstraints - 1, -nconstraints, :
                    ]
                    localAssemblyWeights[-nconstraints, nconstraints - 1, :] += 1.0

                    if nconstraints > 1:
                        # assert(freeBounds[2] == nconstraints - 1)
                        initSol[
                            -nconstraints + 1 :, : nconstraints - 1, :
                        ] = inputCB.boundaryConstraints["bottom-right"][
                            nconstraints : degree + loffset,
                            -degree - loffset : -nconstraints,
                            :,
                        ]
                        localAssemblyWeights[
                            -nconstraints + 1 :, : nconstraints - 1, :
                        ] = 1.0
                else:
                    initSol[
                        -nconstraints:, :nconstraints, :
                    ] = inputCB.boundaryConstraints["bottom-right"][
                        nconstraints : degree + loffset,
                        -degree - loffset : -nconstraints,
                        :,
                    ]
                    localAssemblyWeights[-nconstraints:, :nconstraints, :] = 1.0

            if "bottom-left" in inputCB.boundaryConstraints:
                if oddDegree:
                    localBCAssembly[
                        nconstraints - 1, nconstraints - 1, :
                    ] += inputCB.boundaryConstraints["bottom-left"][
                        -nconstraints, -nconstraints, :
                    ]
                    localAssemblyWeights[nconstraints - 1, nconstraints - 1, :] += 1.0

                    if nconstraints > 1:
                        # assert(freeBounds[0] == nconstraints - 1)
                        initSol[
                            : nconstraints - 1, : nconstraints - 1, :
                        ] = inputCB.boundaryConstraints["bottom-left"][
                            -degree - loffset : -nconstraints,
                            -degree - loffset : -nconstraints,
                            :,
                        ]
                        localAssemblyWeights[
                            : nconstraints - 1, : nconstraints - 1, :
                        ] = 1.0
                else:
                    initSol[
                        :nconstraints, :nconstraints, :
                    ] = inputCB.boundaryConstraints["bottom-left"][
                        -degree - loffset : -nconstraints,
                        -degree - loffset : -nconstraints,
                        :,
                    ]
                    localAssemblyWeights[:nconstraints, :nconstraints, :] = 1.0

            if "top-right" in inputCB.boundaryConstraints:
                if oddDegree:
                    localBCAssembly[
                        -nconstraints, -nconstraints, :
                    ] += inputCB.boundaryConstraints["top-right"][
                        nconstraints - 1, nconstraints - 1, :
                    ]
                    localAssemblyWeights[-nconstraints, -nconstraints, :] += 1.0

                    if nconstraints > 1:
                        initSol[
                            -nconstraints + 1 :, -nconstraints + 1 :, :
                        ] = inputCB.boundaryConstraints["top-right"][
                            nconstraints : degree + loffset,
                            nconstraints : degree + loffset,
                            :,
                        ]
                        localAssemblyWeights[
                            -nconstraints + 1 :, -nconstraints + 1 :, :
                        ] = 1.0
                else:
                    initSol[
                        -nconstraints:, -nconstraints:, :
                    ] = inputCB.boundaryConstraints["top-right"][
                        nconstraints : degree + loffset,
                        nconstraints : degree + loffset,
                        :,
                    ]
                    localAssemblyWeights[-nconstraints:, -nconstraints:, :] = 1.0

            if "up-top" in inputCB.boundaryConstraints:
                if oddDegree:
                    localBCAssembly[
                        freeBounds[0] : freeBounds[1], -nconstraints, -nconstraints
                    ] += inputCB.boundaryConstraints["up-top"][
                        freeBounds[0] : freeBounds[1],
                        nconstraints - 1,
                        nconstraints - 1,
                    ]
                    localAssemblyWeights[
                        freeBounds[0] : freeBounds[1], -nconstraints, -nconstraints
                    ] += 1.0

                    if nconstraints > 1:
                        initSol[
                            freeBounds[0] : freeBounds[1],
                            -nconstraints + 1 :,
                            -nconstraints + 1 :,
                        ] = inputCB.boundaryConstraints["up-top"][
                            freeBounds[0] : freeBounds[1],
                            nconstraints : loffset + degree,
                            nconstraints : loffset + degree,
                        ]
                        localAssemblyWeights[
                            freeBounds[0] : freeBounds[1],
                            -nconstraints + 1 :,
                            -nconstraints + 1 :,
                        ] = 1.0
                else:
                    initSol[
                        freeBounds[0] : freeBounds[1], -nconstraints:, -nconstraints:
                    ] = inputCB.boundaryConstraints["up-top"][
                        freeBounds[0] : freeBounds[1],
                        nconstraints : loffset + degree,
                        nconstraints : loffset + degree,
                    ]
                    localAssemblyWeights[
                        freeBounds[0] : freeBounds[1], -nconstraints:, -nconstraints:
                    ] = 1.0

            if "up-bottom" in inputCB.boundaryConstraints:
                if oddDegree:
                    localBCAssembly[
                        freeBounds[0] : freeBounds[1], nconstraints - 1, -nconstraints
                    ] += inputCB.boundaryConstraints["up-bottom"][
                        freeBounds[0] : freeBounds[1],
                        nconstraints - 1,
                        nconstraints - 1,
                    ]
                    localAssemblyWeights[
                        freeBounds[0] : freeBounds[1], nconstraints - 1, -nconstraints
                    ] += 1.0

                    if nconstraints > 1:
                        initSol[
                            freeBounds[0] : freeBounds[1],
                            : nconstraints - 1,
                            -nconstraints + 1 :,
                        ] = inputCB.boundaryConstraints["up-bottom"][
                            freeBounds[0] : freeBounds[1],
                            nconstraints : loffset + degree,
                            nconstraints : loffset + degree,
                        ]
                        localAssemblyWeights[
                            freeBounds[0] : freeBounds[1],
                            : nconstraints - 1,
                            -nconstraints + 1 :,
                        ] = 1.0
                else:
                    initSol[
                        freeBounds[0] : freeBounds[1], :nconstraints, -nconstraints:
                    ] = inputCB.boundaryConstraints["up-bottom"][
                        freeBounds[0] : freeBounds[1],
                        nconstraints : loffset + degree,
                        nconstraints : loffset + degree,
                    ]
                    localAssemblyWeights[
                        freeBounds[0] : freeBounds[1], :nconstraints, -nconstraints:
                    ] = 1.0

            if "down-top" in inputCB.boundaryConstraints:
                if oddDegree:
                    localBCAssembly[
                        freeBounds[0] : freeBounds[1], -nconstraints, nconstraints - 1
                    ] += inputCB.boundaryConstraints["down-top"][
                        freeBounds[0] : freeBounds[1], nconstraints - 1, -nconstraints
                    ]
                    localAssemblyWeights[
                        freeBounds[0] : freeBounds[1], -nconstraints, nconstraints - 1
                    ] += 1.0

                    if nconstraints > 1:
                        initSol[
                            freeBounds[0] : freeBounds[1],
                            -nconstraints + 1 :,
                            : nconstraints - 1,
                        ] = inputCB.boundaryConstraints["down-top"][
                            freeBounds[0] : freeBounds[1],
                            nconstraints : loffset + degree,
                            -degree - loffset : -nconstraints,
                        ]
                        localAssemblyWeights[
                            freeBounds[0] : freeBounds[1],
                            -nconstraints + 1 :,
                            : nconstraints - 1,
                        ] = 1.0
                else:
                    initSol[
                        freeBounds[0] : freeBounds[1], -nconstraints:, :nconstraints
                    ] = inputCB.boundaryConstraints["down-top"][
                        freeBounds[0] : freeBounds[1],
                        nconstraints : loffset + degree,
                        -degree - loffset : -nconstraints,
                    ]
                    localAssemblyWeights[
                        freeBounds[0] : freeBounds[1], -nconstraints:, :nconstraints
                    ] = 1.0

            if "down-bottom" in inputCB.boundaryConstraints:
                if oddDegree:
                    localBCAssembly[
                        freeBounds[0] : freeBounds[1],
                        nconstraints - 1,
                        nconstraints - 1,
                    ] += inputCB.boundaryConstraints["down-bottom"][
                        freeBounds[0] : freeBounds[1], nconstraints - 1, -nconstraints
                    ]
                    localAssemblyWeights[
                        freeBounds[0] : freeBounds[1],
                        nconstraints - 1,
                        nconstraints - 1,
                    ] += 1.0

                    if nconstraints > 1:
                        initSol[
                            freeBounds[0] : freeBounds[1],
                            : nconstraints - 1,
                            : nconstraints - 1,
                        ] = inputCB.boundaryConstraints["down-bottom"][
                            freeBounds[0] : freeBounds[1],
                            nconstraints : loffset + degree,
                            -degree - loffset : -nconstraints,
                        ]
                        localAssemblyWeights[
                            freeBounds[0] : freeBounds[1],
                            : nconstraints - 1,
                            : nconstraints - 1,
                        ] = 1.0
                else:
                    initSol[
                        freeBounds[0] : freeBounds[1], :nconstraints, :nconstraints
                    ] = inputCB.boundaryConstraints["down-bottom"][
                        freeBounds[0] : freeBounds[1],
                        nconstraints : loffset + degree,
                        -degree - loffset : -nconstraints,
                    ]
                    localAssemblyWeights[
                        freeBounds[0] : freeBounds[1], :nconstraints, :nconstraints
                    ] = 1.0

            if "up-right" in inputCB.boundaryConstraints:
                if oddDegree:
                    localBCAssembly[
                        -nconstraints, freeBounds[2] : freeBounds[3], -nconstraints
                    ] += inputCB.boundaryConstraints["up-right"][
                        nconstraints - 1,
                        freeBounds[2] : freeBounds[3],
                        nconstraints - 1,
                    ]
                    localAssemblyWeights[
                        -nconstraints, freeBounds[2] : freeBounds[3], -nconstraints
                    ] += 1.0

                    if nconstraints > 1:
                        initSol[
                            -nconstraints + 1 :,
                            freeBounds[2] : freeBounds[3],
                            -nconstraints + 1 :,
                        ] = inputCB.boundaryConstraints["up-right"][
                            nconstraints : loffset + degree,
                            freeBounds[2] : freeBounds[3],
                            nconstraints : loffset + degree,
                        ]
                        localAssemblyWeights[
                            -nconstraints + 1 :,
                            freeBounds[2] : freeBounds[3],
                            -nconstraints + 1 :,
                        ] = 1.0
                else:
                    initSol[
                        -nconstraints:, freeBounds[2] : freeBounds[3], -nconstraints:
                    ] = inputCB.boundaryConstraints["up-right"][
                        nconstraints : loffset + degree,
                        freeBounds[2] : freeBounds[3],
                        nconstraints : loffset + degree,
                    ]
                    localAssemblyWeights[
                        -nconstraints:, freeBounds[2] : freeBounds[3], -nconstraints:
                    ] = 1.0

            if "up-left" in inputCB.boundaryConstraints:
                if oddDegree:
                    localBCAssembly[
                        nconstraints - 1, freeBounds[2] : freeBounds[3], -nconstraints
                    ] += inputCB.boundaryConstraints["up-left"][
                        nconstraints - 1,
                        freeBounds[2] : freeBounds[3],
                        nconstraints - 1,
                    ]
                    localAssemblyWeights[
                        nconstraints - 1, freeBounds[2] : freeBounds[3], -nconstraints
                    ] += 1.0

                    if nconstraints > 1:
                        initSol[
                            : nconstraints - 1,
                            freeBounds[2] : freeBounds[3],
                            -nconstraints + 1 :,
                        ] = inputCB.boundaryConstraints["up-left"][
                            nconstraints : loffset + degree,
                            freeBounds[2] : freeBounds[3],
                            nconstraints : loffset + degree,
                        ]
                        localAssemblyWeights[
                            : nconstraints - 1,
                            freeBounds[2] : freeBounds[3],
                            -nconstraints + 1 :,
                        ] = 1.0
                else:
                    initSol[
                        :nconstraints, freeBounds[2] : freeBounds[3], -nconstraints:
                    ] = inputCB.boundaryConstraints["up-left"][
                        nconstraints : loffset + degree,
                        freeBounds[2] : freeBounds[3],
                        nconstraints : loffset + degree,
                    ]
                    localAssemblyWeights[
                        :nconstraints, freeBounds[2] : freeBounds[3], -nconstraints:
                    ] = 1.0

            if "down-right" in inputCB.boundaryConstraints:
                if oddDegree:
                    localBCAssembly[
                        -nconstraints, freeBounds[2] : freeBounds[3], nconstraints - 1
                    ] += inputCB.boundaryConstraints["down-right"][
                        nconstraints - 1, freeBounds[2] : freeBounds[3], -nconstraints
                    ]
                    localAssemblyWeights[
                        -nconstraints, freeBounds[2] : freeBounds[3], nconstraints - 1
                    ] += 1.0

                    if nconstraints > 1:
                        initSol[
                            -nconstraints + 1 :,
                            freeBounds[2] : freeBounds[3],
                            : nconstraints - 1,
                        ] = inputCB.boundaryConstraints["down-right"][
                            nconstraints : loffset + degree,
                            freeBounds[2] : freeBounds[3],
                            -degree - loffset : -nconstraints,
                        ]
                        localAssemblyWeights[
                            -nconstraints + 1 :,
                            freeBounds[2] : freeBounds[3],
                            : nconstraints - 1,
                        ] = 1.0
                else:
                    initSol[
                        -nconstraints:, freeBounds[2] : freeBounds[3], :nconstraints
                    ] = inputCB.boundaryConstraints["down-right"][
                        nconstraints : loffset + degree,
                        freeBounds[2] : freeBounds[3],
                        -degree - loffset : -nconstraints,
                    ]
                    localAssemblyWeights[
                        -nconstraints:, freeBounds[2] : freeBounds[3], :nconstraints
                    ] = 1.0

            if "down-left" in inputCB.boundaryConstraints:
                if oddDegree:
                    localBCAssembly[
                        nconstraints - 1,
                        freeBounds[2] : freeBounds[3],
                        nconstraints - 1,
                    ] += inputCB.boundaryConstraints["down-left"][
                        nconstraints - 1, freeBounds[2] : freeBounds[3], -nconstraints
                    ]
                    localAssemblyWeights[
                        nconstraints - 1,
                        freeBounds[2] : freeBounds[3],
                        nconstraints - 1,
                    ] += 1.0

                    if nconstraints > 1:
                        initSol[
                            : nconstraints - 1,
                            freeBounds[2] : freeBounds[3],
                            : nconstraints - 1,
                        ] = inputCB.boundaryConstraints["down-left"][
                            nconstraints : loffset + degree,
                            freeBounds[2] : freeBounds[3],
                            -degree - loffset : -nconstraints,
                        ]
                        localAssemblyWeights[
                            : nconstraints - 1,
                            freeBounds[2] : freeBounds[3],
                            : nconstraints - 1,
                        ] = 1.0
                else:
                    initSol[
                        :nconstraints, freeBounds[2] : freeBounds[3], :nconstraints
                    ] = inputCB.boundaryConstraints["down-left"][
                        nconstraints : loffset + degree,
                        freeBounds[2] : freeBounds[3],
                        -degree - loffset : -nconstraints,
                    ]
                    localAssemblyWeights[
                        :nconstraints, freeBounds[2] : freeBounds[3], :nconstraints
                    ] = 1.0

            if "up-top-left" in inputCB.boundaryConstraints:
                if oddDegree:
                    localBCAssembly[
                        nconstraints - 1, -nconstraints, -nconstraints
                    ] += inputCB.boundaryConstraints["up-top-left"][
                        -nconstraints, nconstraints - 1, nconstraints - 1
                    ]
                    localAssemblyWeights[
                        nconstraints - 1, -nconstraints, -nconstraints
                    ] += 1.0

                    if nconstraints > 1:
                        assert freeBounds[0] == nconstraints - 1
                        # initSol[: nconstraints -
                        #         1, -nconstraints + 1:, -nconstraints + 1:] = 0
                        localAssemblyWeights[
                            : nconstraints - 1, -nconstraints + 1 :, -nconstraints + 1 :
                        ] = 1.0
                        initSol[
                            : nconstraints - 1, -nconstraints + 1 :, -nconstraints + 1 :
                        ] = inputCB.boundaryConstraints["up-top-left"][
                            -degree - loffset : -nconstraints,
                            nconstraints : degree + loffset,
                            nconstraints : degree + loffset
                        ]
                else:
                    initSol[
                        :nconstraints, -nconstraints:, -nconstraints:
                    ] = inputCB.boundaryConstraints["up-top-left"][
                        -degree - loffset : -nconstraints,
                        nconstraints : degree + loffset,
                        nconstraints : degree + loffset
                    ]
                    localAssemblyWeights[
                        :nconstraints, -nconstraints:, -nconstraints:
                    ] = 1.0

            if "up-bottom-right" in inputCB.boundaryConstraints:
                if oddDegree:
                    localBCAssembly[
                        -nconstraints, nconstraints - 1, -nconstraints
                    ] += inputCB.boundaryConstraints["up-bottom-right"][
                        nconstraints - 1, -nconstraints, nconstraints - 1
                    ]
                    localAssemblyWeights[
                        -nconstraints, nconstraints - 1, -nconstraints
                    ] += 1.0

                    if nconstraints > 1:
                        # assert(freeBounds[2] == nconstraints - 1)
                        initSol[
                            -nconstraints + 1 :, : nconstraints - 1, -nconstraints + 1 :
                        ] = inputCB.boundaryConstraints["up-bottom-right"][
                            nconstraints : degree + loffset,
                            -degree - loffset : -nconstraints,
                            nconstraints : degree + loffset
                        ]
                        localAssemblyWeights[
                            -nconstraints + 1 :, : nconstraints - 1, -nconstraints + 1 :
                        ] = 1.0
                else:
                    initSol[
                        -nconstraints:, :nconstraints, -nconstraints:
                    ] = inputCB.boundaryConstraints["up-bottom-right"][
                        nconstraints : degree + loffset,
                        -degree - loffset : -nconstraints,
                        nconstraints : degree + loffset
                    ]
                    localAssemblyWeights[
                        -nconstraints:, :nconstraints, -nconstraints:
                    ] = 1.0

            if "up-bottom-left" in inputCB.boundaryConstraints:
                if oddDegree:
                    localBCAssembly[
                        nconstraints - 1, nconstraints - 1, -nconstraints
                    ] += inputCB.boundaryConstraints["up-bottom-left"][
                        -nconstraints, -nconstraints, nconstraints - 1
                    ]
                    localAssemblyWeights[
                        nconstraints - 1, nconstraints - 1, -nconstraints
                    ] += 1.0

                    if nconstraints > 1:
                        # assert(freeBounds[0] == nconstraints - 1)
                        initSol[
                            : nconstraints - 1, : nconstraints - 1, -nconstraints + 1 :
                        ] = inputCB.boundaryConstraints["up-bottom-left"][
                            -degree - loffset : -nconstraints,
                            -degree - loffset : -nconstraints,
                            nconstraints : degree + loffset
                        ]
                        localAssemblyWeights[
                            : nconstraints - 1, : nconstraints - 1, -nconstraints + 1 :
                        ] = 1.0
                else:
                    initSol[
                        :nconstraints, :nconstraints, -nconstraints:
                    ] = inputCB.boundaryConstraints["up-bottom-left"][
                        -degree - loffset : -nconstraints,
                        -degree - loffset : -nconstraints,
                        nconstraints : degree + loffset
                    ]
                    localAssemblyWeights[
                        :nconstraints, :nconstraints, -nconstraints:
                    ] = 1.0

            if "up-top-right" in inputCB.boundaryConstraints:
                if oddDegree:
                    localBCAssembly[
                        -nconstraints, -nconstraints, -nconstraints
                    ] += inputCB.boundaryConstraints["up-top-right"][
                        nconstraints - 1, nconstraints - 1, nconstraints - 1
                    ]
                    localAssemblyWeights[
                        -nconstraints, -nconstraints, -nconstraints
                    ] += 1.0

                    if nconstraints > 1:
                        initSol[
                            -nconstraints + 1 :,
                            -nconstraints + 1 :,
                            -nconstraints + 1 :
                        ] = inputCB.boundaryConstraints["up-top-right"][
                            nconstraints : degree + loffset,
                            nconstraints : degree + loffset,
                            nconstraints : degree + loffset
                        ]
                        localAssemblyWeights[
                            -nconstraints + 1 :,
                            -nconstraints + 1 :,
                            -nconstraints + 1 :
                        ] = 1.0
                else:
                    initSol[
                        -nconstraints:, -nconstraints:, -nconstraints:
                    ] = inputCB.boundaryConstraints["up-top-right"][
                        nconstraints : degree + loffset,
                        nconstraints : degree + loffset,
                        nconstraints : degree + loffset
                    ]
                    localAssemblyWeights[
                        -nconstraints:, -nconstraints:, -nconstraints:
                    ] = 1.0

            if "down-top-left" in inputCB.boundaryConstraints:
                if oddDegree:
                    localBCAssembly[
                        nconstraints - 1, -nconstraints, nconstraints - 1
                    ] += inputCB.boundaryConstraints["down-top-left"][
                        -nconstraints, nconstraints - 1, -nconstraints
                    ]
                    localAssemblyWeights[
                        nconstraints - 1, -nconstraints, nconstraints - 1
                    ] += 1.0

                    if nconstraints > 1:
                        assert freeBounds[0] == nconstraints - 1
                        localAssemblyWeights[
                            : nconstraints - 1, -nconstraints + 1 :, : nconstraints - 1
                        ] = 1.0
                        initSol[
                            : nconstraints - 1, -nconstraints + 1 :, : nconstraints - 1
                        ] = inputCB.boundaryConstraints["down-top-left"][
                            -degree - loffset : -nconstraints,
                            nconstraints : degree + loffset,
                            -degree - loffset : -nconstraints
                        ]
                else:
                    initSol[
                        :nconstraints, -nconstraints:, :nconstraints
                    ] = inputCB.boundaryConstraints["down-top-left"][
                        -degree - loffset : -nconstraints,
                        nconstraints : degree + loffset,
                        -degree - loffset : -nconstraints
                    ]
                    localAssemblyWeights[
                        :nconstraints, -nconstraints:, :nconstraints
                    ] = 1.0

            if "down-bottom-right" in inputCB.boundaryConstraints:
                if oddDegree:
                    localBCAssembly[
                        -nconstraints, nconstraints - 1, nconstraints - 1
                    ] += inputCB.boundaryConstraints["down-bottom-right"][
                        nconstraints - 1, -nconstraints, -nconstraints
                    ]
                    localAssemblyWeights[
                        -nconstraints, nconstraints - 1, nconstraints - 1
                    ] += 1.0

                    if nconstraints > 1:
                        # assert(freeBounds[2] == nconstraints - 1)
                        initSol[
                            -nconstraints + 1 :, : nconstraints - 1, : nconstraints - 1
                        ] = inputCB.boundaryConstraints["down-bottom-right"][
                            nconstraints : degree + loffset,
                            -degree - loffset : -nconstraints,
                            -degree - loffset : -nconstraints
                        ]
                        localAssemblyWeights[
                            -nconstraints + 1 :, : nconstraints - 1, : nconstraints - 1
                        ] = 1.0
                else:
                    initSol[
                        -nconstraints:, :nconstraints, :nconstraints
                    ] = inputCB.boundaryConstraints["down-bottom-right"][
                        nconstraints : degree + loffset,
                        -degree - loffset : -nconstraints,
                        -degree - loffset : -nconstraints
                    ]
                    localAssemblyWeights[
                        -nconstraints:, :nconstraints, :nconstraints
                    ] = 1.0

            if "down-bottom-left" in inputCB.boundaryConstraints:
                if oddDegree:
                    localBCAssembly[
                        nconstraints - 1, nconstraints - 1, nconstraints - 1
                    ] += inputCB.boundaryConstraints["down-bottom-left"][
                        -nconstraints, -nconstraints, -nconstraints
                    ]
                    localAssemblyWeights[
                        nconstraints - 1, nconstraints - 1, nconstraints - 1
                    ] += 1.0

                    if nconstraints > 1:
                        # assert(freeBounds[0] == nconstraints - 1)
                        initSol[
                            : nconstraints - 1, : nconstraints - 1, : nconstraints - 1
                        ] = inputCB.boundaryConstraints["down-bottom-left"][
                            -degree - loffset : -nconstraints,
                            -degree - loffset : -nconstraints,
                            -degree - loffset : -nconstraints,
                        ]
                        localAssemblyWeights[
                            : nconstraints - 1, : nconstraints - 1, : nconstraints - 1
                        ] = 1.0
                else:
                    initSol[
                        :nconstraints, :nconstraints, :nconstraints
                    ] = inputCB.boundaryConstraints["down-bottom-left"][
                        -degree - loffset : -nconstraints,
                        -degree - loffset : -nconstraints,
                        -degree - loffset : -nconstraints
                    ]
                    localAssemblyWeights[
                        :nconstraints, :nconstraints, :nconstraints
                    ] = 1.0

            if "down-top-right" in inputCB.boundaryConstraints:
                if oddDegree:
                    localBCAssembly[
                        -nconstraints, -nconstraints, nconstraints - 1
                    ] += inputCB.boundaryConstraints["down-top-right"][
                        nconstraints - 1, nconstraints - 1, -nconstraints
                    ]
                    localAssemblyWeights[
                        -nconstraints, -nconstraints, nconstraints - 1
                    ] += 1.0

                    if nconstraints > 1:
                        initSol[
                            -nconstraints + 1 :, -nconstraints + 1 :, : nconstraints - 1
                        ] = inputCB.boundaryConstraints["down-top-right"][
                            nconstraints : degree + loffset,
                            nconstraints : degree + loffset,
                            -degree - loffset : -nconstraints
                        ]
                        localAssemblyWeights[
                            -nconstraints + 1 :, -nconstraints + 1 :, : nconstraints - 1
                        ] = 1.0
                else:
                    initSol[
                        -nconstraints:, -nconstraints:, :nconstraints
                    ] = inputCB.boundaryConstraints["down-top-right"][
                        nconstraints : degree + loffset,
                        nconstraints : degree + loffset,
                        -degree - loffset : -nconstraints
                    ]
                    localAssemblyWeights[
                        -nconstraints:, -nconstraints:, :nconstraints
                    ] = 1.0

            localAssemblyWeights[
                freeBounds[0] : freeBounds[1],
                freeBounds[2] : freeBounds[3],
                freeBounds[4] : freeBounds[5],
            ] += 1.0
            initSol = np.divide(initSol + localBCAssembly, localAssemblyWeights)

            # Rotate it back
            # initSol = np.moveaxis(initSol, -1, 0)
            # initSol = np.moveaxis(initSol, 1, 2)

        return np.copy(initSol)
