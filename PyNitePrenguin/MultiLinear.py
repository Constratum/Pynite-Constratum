from __future__ import annotations
import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import List

try:
    from .UniaxialMaterial import UniaxialMaterial
except ImportError:
    from UniaxialMaterial import UniaxialMaterial


class MultiLinear(UniaxialMaterial):
    """
    Uniaxial multilinear material model ported from OpenSees MultiLinear.

    Parameters:
    tag: int
        Material tag
    e: List[float]
        Strain points (must be increasing)
    s: List[float]
        Stress points corresponding to strains
    """

    def __init__(self, tag: int, e: List[float], s: List[float]):
        super().__init__(tag)

        self.numSlope = len(e)
        if self.numSlope != len(s):
            raise ValueError("Number of strain and stress points must match")

        if self.numSlope < 2:
            raise ValueError("At least two points required")

        # Check increasing strains
        for i in range(1, self.numSlope):
            if e[i] <= e[i - 1]:
                raise ValueError("Strain points must be strictly increasing")

        self.e0 = np.array(e)
        self.s0 = np.array(s)

        # Data matrix: rows = slopes, columns = [neg_strain, pos_strain, neg_stress, pos_stress, slope, dist]
        self.data = np.zeros((self.numSlope, 6))

        # First segment (elastic)
        self.data[0, 0] = -e[0]  # neg yield strain
        self.data[0, 1] = e[0]  # pos yield strain
        self.data[0, 2] = -s[0]  # neg yield stress
        self.data[0, 3] = s[0]  # pos yield stress
        self.data[0, 4] = s[0] / e[0]  # slope
        self.data[0, 5] = e[0]  # dist

        # Subsequent segments
        for i in range(1, self.numSlope):
            self.data[i, 0] = -e[i]
            self.data[i, 1] = e[i]
            self.data[i, 2] = -s[i]
            self.data[i, 3] = s[i]
            self.data[i, 4] = (s[i] - s[i - 1]) / (e[i] - e[i - 1])
            self.data[i, 5] = e[i] - e[i - 1]

        # State variables
        self.tStrain = 0.0
        self.tStress = 0.0
        self.tTangent = self.data[0, 4]
        self.tSlope = 0

        self.cStrain = 0.0
        self.cStress = 0.0
        self.cTangent = self.tTangent
        self.cSlope = 0

    def set_trial_strain(self, strain: float, strain_rate: float = 0.0) -> int:
        if abs(self.tStrain - strain) < 1e-14:
            return 0

        self.tStrain = strain
        self.tSlope = 0

        if self.data[0, 0] <= strain <= self.data[0, 1]:  # elastic
            self.tSlope = 0
            self.tStress = (
                self.data[0, 2] + (strain - self.data[0, 0]) * self.data[0, 4]
            )
            self.tTangent = self.data[0, 4]
        elif strain < self.data[0, 0]:  # negative side
            self.tSlope = 1
            while self.tSlope < self.numSlope and strain < self.data[self.tSlope, 0]:
                self.tSlope += 1
            if self.tSlope == self.numSlope:
                self.tSlope = self.numSlope - 1
            self.tStress = (
                self.data[self.tSlope, 2]
                + (strain - self.data[self.tSlope, 0]) * self.data[self.tSlope, 4]
            )
            self.tTangent = self.data[self.tSlope, 4]
        else:  # positive side
            self.tSlope = 1
            while self.tSlope < self.numSlope and strain > self.data[self.tSlope, 1]:
                self.tSlope += 1
            if self.tSlope == self.numSlope:
                self.tSlope = self.numSlope - 1
            self.tStress = (
                self.data[self.tSlope, 3]
                + (strain - self.data[self.tSlope, 1]) * self.data[self.tSlope, 4]
            )
            self.tTangent = self.data[self.tSlope, 4]

        return 0

    def get_strain(self) -> float:
        return self.tStrain

    def get_stress(self) -> float:
        return self.tStress

    def get_tangent(self) -> float:
        return self.tTangent

    def commit_state(self) -> int:
        # If yielded, update the backbone
        if self.tSlope != 0:
            if self.tStrain > self.data[0, 1]:  # positive yield
                # Update elastic bounds
                self.data[0, 1] = self.tStrain
                self.data[0, 3] = self.tStress
                self.data[0, 0] = self.tStrain - 2 * self.data[0, 5]
                self.data[0, 2] = self.tStress - 2 * self.data[0, 5] * self.data[0, 4]

                # Update previous segments
                for i in range(1, self.tSlope):
                    self.data[i, 1] = self.tStrain
                    self.data[i, 3] = self.tStress
                    self.data[i, 0] = self.data[i - 1, 0] - 2 * self.data[i, 5]
                    self.data[i, 2] = (
                        self.data[i - 1, 2] - 2 * self.data[i, 5] * self.data[i, 4]
                    )

                # Update current and subsequent segments
                self.data[self.tSlope, 0] = (
                    self.data[self.tSlope - 1, 0]
                    - 2 * self.data[self.tSlope, 5]
                    + self.data[self.tSlope, 1]
                    - self.data[self.tSlope - 1, 1]
                )
                self.data[self.tSlope, 2] = (
                    self.data[self.tSlope - 1, 2]
                    + (self.data[self.tSlope, 0] - self.data[self.tSlope - 1, 0])
                    * self.data[self.tSlope, 4]
                )

                for i in range(self.tSlope + 1, self.numSlope):
                    self.data[i, 0] = (
                        self.data[i - 1, 0]
                        - 2 * self.data[i, 5]
                        + self.data[i, 1]
                        - self.data[i - 1, 1]
                    )
                    self.data[i, 2] = (
                        self.data[i - 1, 2]
                        + (self.data[i, 0] - self.data[i - 1, 0]) * self.data[i, 4]
                    )

            else:  # negative yield
                # Update elastic bounds
                self.data[0, 0] = self.tStrain
                self.data[0, 2] = self.tStress
                self.data[0, 1] = self.tStrain + 2 * self.data[0, 5]
                self.data[0, 3] = self.tStress + 2 * self.data[0, 5] * self.data[0, 4]

                # Update previous segments
                for i in range(1, self.tSlope):
                    self.data[i, 0] = self.tStrain
                    self.data[i, 2] = self.tStress
                    self.data[i, 1] = self.data[i - 1, 1] + 2 * self.data[i, 5]
                    self.data[i, 3] = (
                        self.data[i - 1, 3] + 2 * self.data[i, 5] * self.data[i, 4]
                    )

                # Update current and subsequent segments
                self.data[self.tSlope, 1] = (
                    self.data[self.tSlope - 1, 1]
                    + 2 * self.data[self.tSlope, 5]
                    + self.data[self.tSlope, 0]
                    - self.data[self.tSlope - 1, 0]
                )
                self.data[self.tSlope, 3] = (
                    self.data[self.tSlope - 1, 3]
                    + (self.data[self.tSlope, 1] - self.data[self.tSlope - 1, 1])
                    * self.data[self.tSlope, 4]
                )

                for i in range(self.tSlope + 1, self.numSlope):
                    self.data[i, 1] = (
                        self.data[i - 1, 1]
                        + 2 * self.data[i, 5]
                        + self.data[i, 0]
                        - self.data[i - 1, 0]
                    )
                    self.data[i, 3] = (
                        self.data[i - 1, 3]
                        + (self.data[i, 1] - self.data[i - 1, 1]) * self.data[i, 4]
                    )

        self.cStrain = self.tStrain
        self.cStress = self.tStress
        self.cTangent = self.tTangent
        self.cSlope = self.tSlope

        return 0

    def revert_to_last_commit(self) -> int:
        self.tStrain = self.cStrain
        self.tStress = self.cStress
        self.tTangent = self.cTangent
        self.tSlope = self.cSlope
        return 0

    def revert_to_start(self) -> int:
        # Reset data to initial
        self.data[0, 0] = -self.e0[0]
        self.data[0, 1] = self.e0[0]
        self.data[0, 2] = -self.s0[0]
        self.data[0, 3] = self.s0[0]
        self.data[0, 4] = self.s0[0] / self.e0[0]
        self.data[0, 5] = self.e0[0]

        for i in range(1, self.numSlope):
            self.data[i, 0] = -self.e0[i]
            self.data[i, 1] = self.e0[i]
            self.data[i, 2] = -self.s0[i]
            self.data[i, 3] = self.s0[i]
            self.data[i, 4] = (self.s0[i] - self.s0[i - 1]) / (
                self.e0[i] - self.e0[i - 1]
            )
            self.data[i, 5] = self.e0[i] - self.e0[i - 1]

        self.tStrain = self.cStrain = 0.0
        self.tStress = self.cStress = 0.0
        self.tTangent = self.cTangent = self.data[0, 4]
        self.tSlope = self.cSlope = 0
        return 0

    def get_copy(self) -> "MultiLinear":
        copy = MultiLinear(self.tag, self.e0.tolist(), self.s0.tolist())
        copy.data = self.data.copy()
        copy.tStrain = self.tStrain
        copy.tStress = self.tStress
        copy.tTangent = self.tTangent
        copy.tSlope = self.tSlope
        copy.cStrain = self.cStrain
        copy.cStress = self.cStress
        copy.cTangent = self.cTangent
        copy.cSlope = self.cSlope
        return copy
