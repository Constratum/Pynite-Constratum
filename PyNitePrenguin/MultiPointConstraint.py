# %%
from __future__ import annotations
from numpy import zeros, array
from typing import TYPE_CHECKING, List, Dict

if TYPE_CHECKING:
    from PyNitePrenguin.Node3D import Node3D
    from PyNitePrenguin.LoadCombo import LoadCombo


# %%
class MultiPointConstraint:
    """A class representing a multi-point constraint between two nodes.

    This class enforces equal displacements/rotations between specified DOFs
    of two nodes, similar to OpenSees EqualDOF functionality.
    """

    def __init__(
        self,
        name: str,
        retained_node: "Node3D",
        constrained_node: "Node3D",
        constrained_dofs: List[int],
        retained_dofs: List[int] = None,
        constraint_matrix: array = None,
    ) -> None:
        """
        Initialize a multi-point constraint.

        :param name: Unique name for the constraint
        :param retained_node: The node that retains its DOF (independent)
        :param constrained_node: The node whose DOF will be constrained (dependent)
        :param constrained_dofs: List of DOF indices for constrained node (0-5)
        :param retained_dofs: List of DOF indices for retained node (0-5). If None, uses constrained_dofs
        :param constraint_matrix: Optional constraint matrix. If None, creates identity constraint
        """
        self.name: str = name
        self.retained_node: "Node3D" = retained_node
        self.constrained_node: "Node3D" = constrained_node
        self.constrained_dofs: List[int] = constrained_dofs
        self.retained_dofs: List[int] = (
            retained_dofs if retained_dofs is not None else constrained_dofs
        )

        # Validate DOF indices
        for dof in self.constrained_dofs + self.retained_dofs:
            if dof < 0 or dof > 5:
                raise ValueError(
                    f"DOF index {dof} is invalid. Must be 0-5 (0,1,2=DX,DY,DZ; 3,4,5=RX,RY,RZ)"
                )

        # Validate matching sizes
        if len(self.constrained_dofs) != len(self.retained_dofs):
            raise ValueError(
                "Number of constrained DOFs must equal number of retained DOFs"
            )

        # Set up constraint matrix
        if constraint_matrix is None:
            # Create identity constraint matrix (equal DOF)
            num_dofs = len(self.constrained_dofs)
            self.constraint_matrix = zeros((num_dofs, num_dofs))
            for i in range(num_dofs):
                self.constraint_matrix[i, i] = 1.0
        else:
            self.constraint_matrix = constraint_matrix

        # Validate constraint matrix dimensions
        if self.constraint_matrix.shape != (
            len(self.constrained_dofs),
            len(self.retained_dofs),
        ):
            raise ValueError("Constraint matrix dimensions must match number of DOFs")

    def get_constraint_info(self) -> Dict:
        """
        Returns information about the constraint.
        """
        return {
            "name": self.name,
            "retained_node": self.retained_node.name,
            "constrained_node": self.constrained_node.name,
            "constrained_dofs": self.constrained_dofs,
            "retained_dofs": self.retained_dofs,
            "constraint_matrix": self.constraint_matrix.tolist(),
        }

    def apply_constraint(self, global_K: array, global_P: array) -> tuple:
        """
        Apply the constraint to the global stiffness matrix and load vector.
        This uses the penalty method approach.

        :param global_K: Global stiffness matrix
        :param global_P: Global load vector
        :return: Modified (K, P) tuple
        """
        # Large penalty parameter
        penalty = 1.0e12

        # Get global DOF indices
        retained_indices = []
        constrained_indices = []

        for dof in self.retained_dofs:
            retained_indices.append(self.retained_node.ID * 6 + dof)

        for dof in self.constrained_dofs:
            constrained_indices.append(self.constrained_node.ID * 6 + dof)

        # Apply constraint using penalty method
        # For each constraint equation: u_constrained = C * u_retained
        for i, (c_idx, r_idx) in enumerate(zip(constrained_indices, retained_indices)):
            for j, (c_idx2, r_idx2) in enumerate(
                zip(constrained_indices, retained_indices)
            ):
                constraint_val = self.constraint_matrix[i, j]
                if abs(constraint_val) > 1e-10:
                    # Add penalty terms to stiffness matrix
                    global_K[c_idx, c_idx] += penalty
                    global_K[c_idx, r_idx2] -= penalty * constraint_val
                    global_K[r_idx2, c_idx] -= penalty * constraint_val
                    global_K[r_idx2, r_idx2] += (
                        penalty * constraint_val * constraint_val
                    )

        return global_K, global_P

    def get_constraint_forces(self, combo_name: str = "Combo 1") -> Dict:
        """
        Calculate the constraint forces for the given load combination.

        :param combo_name: Name of the load combination
        :return: Dictionary with constraint force information
        """
        forces = {
            "retained_node": self.retained_node.name,
            "constrained_node": self.constrained_node.name,
            "constraint_forces": [],
        }

        # Get nodal displacements
        retained_disps = [
            self.retained_node.DX[combo_name],
            self.retained_node.DY[combo_name],
            self.retained_node.DZ[combo_name],
            self.retained_node.RX[combo_name],
            self.retained_node.RY[combo_name],
            self.retained_node.RZ[combo_name],
        ]

        constrained_disps = [
            self.constrained_node.DX[combo_name],
            self.constrained_node.DY[combo_name],
            self.constrained_node.DZ[combo_name],
            self.constrained_node.RX[combo_name],
            self.constrained_node.RY[combo_name],
            self.constrained_node.RZ[combo_name],
        ]

        # Calculate constraint violations
        for i, (c_dof, r_dof) in enumerate(
            zip(self.constrained_dofs, self.retained_dofs)
        ):
            expected_disp = 0.0
            for j, r_dof2 in enumerate(self.retained_dofs):
                expected_disp += self.constraint_matrix[i, j] * retained_disps[r_dof2]

            actual_disp = constrained_disps[c_dof]
            violation = actual_disp - expected_disp

            forces["constraint_forces"].append(
                {
                    "constrained_dof": c_dof,
                    "retained_dof": r_dof,
                    "expected_displacement": expected_disp,
                    "actual_displacement": actual_disp,
                    "violation": violation,
                }
            )

        return forces


# %%
class EqualDOF(MultiPointConstraint):
    """A convenience class for creating equal DOF constraints between nodes."""

    def __init__(
        self,
        name: str,
        retained_node: "Node3D",
        constrained_node: "Node3D",
        dofs: List[int],
    ) -> None:
        """
        Create an equal DOF constraint.

        :param name: Unique name for the constraint
        :param retained_node: The node that retains its DOF (independent)
        :param constrained_node: The node whose DOF will be constrained (dependent)
        :param dofs: List of DOF indices to be made equal (0-5)
        """
        super().__init__(
            name=name,
            retained_node=retained_node,
            constrained_node=constrained_node,
            constrained_dofs=dofs,
            retained_dofs=dofs,
            constraint_matrix=None,  # Will create identity matrix
        )


# %%
class EqualDOFMixed(MultiPointConstraint):
    """A convenience class for creating mixed equal DOF constraints between nodes."""

    def __init__(
        self,
        name: str,
        retained_node: "Node3D",
        constrained_node: "Node3D",
        constrained_dofs: List[int],
        retained_dofs: List[int],
    ) -> None:
        """
        Create a mixed equal DOF constraint.

        :param name: Unique name for the constraint
        :param retained_node: The node that retains its DOF (independent)
        :param constrained_node: The node whose DOF will be constrained (dependent)
        :param constrained_dofs: List of DOF indices for constrained node (0-5)
        :param retained_dofs: List of DOF indices for retained node (0-5)
        """
        super().__init__(
            name=name,
            retained_node=retained_node,
            constrained_node=constrained_node,
            constrained_dofs=constrained_dofs,
            retained_dofs=retained_dofs,
            constraint_matrix=None,  # Will create identity matrix
        )
