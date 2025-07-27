# %%
from __future__ import annotations  # Allows more recent type hints features
from numpy import zeros, array, add, subtract, matmul, insert, cross, divide
from numpy.linalg import inv
from math import isclose
import PyNitePrenguin.FixedEndReactions
from PyNitePrenguin.LoadCombo import LoadCombo
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Dict, Optional, Union, List
    from numpy import float64
    from numpy.typing import NDArray
    from PyNitePrenguin.Node3D import Node3D
    from PyNitePrenguin.UniaxialMaterial import UniaxialMaterial


# %%
class ZeroLength:
    """A class representing a zero-length element in a finite element model.

    This class supports multiple uniaxial materials in different directions,
    similar to OpenSees ZeroLength element. It can handle both linear and
    nonlinear materials.
    """

    # '__plt' is used to store the 'pyplot' from matplotlib once it gets imported
    __plt = None

    # %%
    def __init__(
        self,
        name: str,
        i_node: Node3D,
        j_node: Node3D,
        materials: List["UniaxialMaterial"],
        directions: List[int],
        LoadCombos: Dict[str, LoadCombo] = {
            "Combo 1": LoadCombo("Combo 1", factors={"Case 1": 1.0})
        },
        x_vector: List[float] = [1.0, 0.0, 0.0],
        y_vector: List[float] = [0.0, 1.0, 0.0],
    ) -> None:
        """
        Initializes a new zero-length element.

        :param name: A unique name for the element given by the user
        :type name: str
        :param i_node: The element's i-node
        :type i_node: Node3D
        :param j_node: The element's j-node
        :type j_node: Node3D
        :param materials: List of uniaxial materials
        :type materials: List[UniaxialMaterial]
        :param directions: List of directions for each material (0-5)
        :type directions: List[int]
        :param LoadCombos: The dictionary of load combinations
        :type LoadCombos: Dict[str, LoadCombo]
        :param x_vector: Local x-axis direction vector
        :type x_vector: List[float]
        :param y_vector: Local y-axis direction vector (in x-y plane)
        :type y_vector: List[float]
        """
        self.name: str = name
        self.ID: Optional[int] = None
        self.i_node: Node3D = i_node
        self.j_node: Node3D = j_node
        self.load_combos: Dict[str, LoadCombo] = LoadCombos

        # Validate inputs
        if len(materials) != len(directions):
            raise ValueError("Number of materials must match number of directions")

        self.materials: List["UniaxialMaterial"] = materials
        self.directions: List[int] = directions
        self.num_materials: int = len(materials)

        # Check direction validity (0-5: 0,1,2=trans x,y,z; 3,4,5=rot x,y,z)
        for direction in directions:
            if direction < 0 or direction > 5:
                raise ValueError(f"Direction {direction} is invalid. Must be 0-5.")

        # Store orientation vectors
        self.x_vector: NDArray[float64] = array(x_vector)
        self.y_vector: NDArray[float64] = array(y_vector)

        # Initialize transformation matrix
        self.transformation_matrix: NDArray[float64] = (
            self._compute_transformation_matrix()
        )

        # Initialize transformation matrix for materials
        self.material_transformation: Optional[NDArray[float64]] = None

        # Track active state for tension/compression only analysis
        self.active: Dict[str, bool] = {}

    # %%
    def _compute_transformation_matrix(self) -> NDArray[float64]:
        """
        Computes the transformation matrix for the element orientation.
        """
        # Normalize x-vector
        x_norm = (
            self.x_vector
            / (self.x_vector[0] ** 2 + self.x_vector[1] ** 2 + self.x_vector[2] ** 2)
            ** 0.5
        )

        # Compute z-vector as cross product of x and y
        z_vec = cross(x_norm, self.y_vector)
        z_norm = z_vec / (z_vec[0] ** 2 + z_vec[1] ** 2 + z_vec[2] ** 2) ** 0.5

        # Recompute y-vector as cross product of z and x
        y_norm = cross(z_norm, x_norm)

        # Create transformation matrix
        T = array([x_norm, y_norm, z_norm])

        return T

    # %%
    def _setup_material_transformation(self) -> None:
        """
        Sets up the transformation matrix for materials based on their directions.
        """
        # Initialize transformation matrix (materials x DOF)
        self.material_transformation = zeros((self.num_materials, 12))

        T = self.transformation_matrix

        for i, direction in enumerate(self.directions):
            if direction < 3:  # Translation
                # Apply transformation for translations
                axis = direction  # 0=x, 1=y, 2=z
                # J-node contributions (positive) - distribute across all three global DOFs
                for j in range(3):
                    self.material_transformation[i, 6 + j] = T[axis, j]
                # I-node contributions (negative) - distribute across all three global DOFs
                for j in range(3):
                    self.material_transformation[i, 0 + j] = -T[axis, j]
            else:  # Rotation
                # Apply transformation for rotations
                axis = direction - 3  # 0=rx, 1=ry, 2=rz
                # J-node contributions (positive) - distribute across all three global rotational DOFs
                for j in range(3):
                    self.material_transformation[i, 9 + j] = T[axis, j]
                # I-node contributions (negative) - distribute across all three global rotational DOFs
                for j in range(3):
                    self.material_transformation[i, 3 + j] = -T[axis, j]

    # %%
    def L(self) -> float:
        """
        Returns the length of the element (should be zero for zero-length elements).
        """
        return self.i_node.distance(self.j_node)

    # %%
    def k(self, combo_name: str = "Combo 1") -> NDArray[float64]:
        """
        Returns the local stiffness matrix for the zero-length element.
        """
        if self.material_transformation is None:
            self._setup_material_transformation()

        # Initialize stiffness matrix
        k_matrix = zeros((12, 12))

        # Add contribution from each material
        for i, material in enumerate(self.materials):
            # Get material stiffness
            if hasattr(material, "get_tangent"):
                E = material.get_tangent()
            else:
                E = material.getTangent()

            # Get transformation vector for this material
            t_vec = self.material_transformation[i, :]

            # Add contribution to stiffness matrix
            for j in range(12):
                for k in range(12):
                    k_matrix[j, k] += E * t_vec[j] * t_vec[k]

        return k_matrix

    # %%
    def f(self, combo_name: str = "Combo 1") -> NDArray[float64]:
        """
        Returns the element's local end force vector.
        """
        if self.material_transformation is None:
            self._setup_material_transformation()

        # Get displacement vector
        d_local = self.d(combo_name)

        # Initialize force vector
        f_vector = zeros((12, 1))

        # Add contribution from each material
        for i, material in enumerate(self.materials):
            # Compute strain for this material
            strain = 0.0
            for j in range(12):
                strain += self.material_transformation[i, j] * d_local[j, 0]

            # Set trial strain and get stress
            if hasattr(material, "set_trial_strain"):
                material.set_trial_strain(strain)
                stress = material.get_stress()
            else:
                material.setTrialStrain(strain)
                stress = material.getStress()

            # Add stress contribution to force vector
            for j in range(12):
                f_vector[j, 0] += stress * self.material_transformation[i, j]

        return f_vector

    # %%
    def d(self, combo_name: str = "Combo 1") -> NDArray[float64]:
        """
        Returns the element's local displacement vector.
        """
        return self.D(combo_name)  # For zero-length elements, local = global

    # %%
    def K(self, combo_name: str = "Combo 1") -> NDArray[float64]:
        """
        Returns the element's global stiffness matrix.
        """
        return self.k(combo_name)  # For zero-length elements, local = global

    # %%
    def F(self, combo_name: str = "Combo 1") -> NDArray[float64]:
        """
        Returns the element's global end force vector.
        """
        return self.f(combo_name)  # For zero-length elements, local = global

    # %%
    def D(self, combo_name: str = "Combo 1") -> NDArray[float64]:
        """
        Returns the element's global displacement vector.
        """
        # Initialize displacement vector
        D = zeros((12, 1))

        # Read displacements from nodes
        D[0, 0] = self.i_node.DX[combo_name]
        D[1, 0] = self.i_node.DY[combo_name]
        D[2, 0] = self.i_node.DZ[combo_name]
        D[3, 0] = self.i_node.RX[combo_name]
        D[4, 0] = self.i_node.RY[combo_name]
        D[5, 0] = self.i_node.RZ[combo_name]
        D[6, 0] = self.j_node.DX[combo_name]
        D[7, 0] = self.j_node.DY[combo_name]
        D[8, 0] = self.j_node.DZ[combo_name]
        D[9, 0] = self.j_node.RX[combo_name]
        D[10, 0] = self.j_node.RY[combo_name]
        D[11, 0] = self.j_node.RZ[combo_name]

        return D

    # %%
    def get_material_forces(self, combo_name: str = "Combo 1") -> List[float]:
        """
        Returns the forces/moments in each material.
        """
        if self.material_transformation is None:
            self._setup_material_transformation()

        d_local = self.d(combo_name)
        forces = []

        for i, material in enumerate(self.materials):
            # Compute strain for this material
            strain = 0.0
            for j in range(12):
                strain += self.material_transformation[i, j] * d_local[j, 0]

            # Set trial strain and get stress
            if hasattr(material, "set_trial_strain"):
                material.set_trial_strain(strain)
                stress = material.get_stress()
            else:
                material.setTrialStrain(strain)
                stress = material.getStress()

            forces.append(stress)

        return forces

    # %%
    def get_material_strains(self, combo_name: str = "Combo 1") -> List[float]:
        """
        Returns the strains in each material.
        """
        if self.material_transformation is None:
            self._setup_material_transformation()

        d_local = self.d(combo_name)
        strains = []

        for i in range(self.num_materials):
            # Compute strain for this material
            strain = 0.0
            for j in range(12):
                strain += self.material_transformation[i, j] * d_local[j, 0]

            strains.append(strain)

        return strains

    # %%
    def commit_state(self) -> None:
        """
        Commits the current trial state for all materials.
        """
        for material in self.materials:
            if hasattr(material, "commit_state"):
                material.commit_state()
            else:
                material.commitState()

    # %%
    def revert_to_last_commit(self) -> None:
        """
        Reverts the trial state to the last committed state for all materials.
        """
        for material in self.materials:
            if hasattr(material, "revert_to_last_commit"):
                material.revert_to_last_commit()
            else:
                material.revertToLastCommit()

    # %%
    def revert_to_start(self) -> None:
        """
        Reverts all materials to their initial state.
        """
        for material in self.materials:
            if hasattr(material, "revert_to_start"):
                material.revert_to_start()
            else:
                material.revertToStart()

    # %%
    def get_info(self) -> dict:
        """
        Returns information about the zero-length element.
        """
        return {
            "name": self.name,
            "type": "ZeroLength",
            "i_node": self.i_node.name,
            "j_node": self.j_node.name,
            "num_materials": self.num_materials,
            "directions": self.directions,
            "length": self.L(),
            "materials": [
                mat.tag if hasattr(mat, "tag") else str(mat) for mat in self.materials
            ],
        }

    # %%
    def update_orientation(self, x_vector: List[float], y_vector: List[float]) -> None:
        """
        Updates the element orientation.
        """
        self.x_vector = array(x_vector)
        self.y_vector = array(y_vector)
        self.transformation_matrix = self._compute_transformation_matrix()
        self.material_transformation = None  # Force recalculation
        self._setup_material_transformation()
