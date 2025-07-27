# %%
from __future__ import annotations # Allows more recent type hints features
from numpy import zeros, array, add, subtract, matmul, insert, cross, divide
from numpy.linalg import inv
from math import isclose
import Pynite.FixedEndReactions
from Pynite.LoadCombo import LoadCombo
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Dict, Optional, Union
    from numpy import float64
    from numpy.typing import NDArray
    from Pynite.Node3D import Node3D
    from Pynite.UniaxialMaterial import UniaxialMaterial

# %%
class Spring3D:
    """A class representing a 3D spring element in a finite element model.

    This class supports both linear springs (defined by a spring constant) and
    nonlinear springs (defined by a uniaxial material model).
    """

    # '__plt' is used to store the 'pyplot' from matplotlib once it gets imported. Setting it to 'None' for now allows us to defer importing it until it's actually needed.
    __plt = None

    # %%
    def __init__(
        self,
        name: str,
        i_node: Node3D,
        j_node: Node3D,
        ks: Union[float, "UniaxialMaterial"],
        LoadCombos: Dict[str, LoadCombo] = {
            "Combo 1": LoadCombo("Combo 1", factors={"Case 1": 1.0})
        },
        tension_only: bool = False,
        comp_only: bool = False,
        spring_type: str = "axial",
    ) -> None:
        """
        Initializes a new spring.

        :param name: A unique name for the spring given by the user
        :type name: str
        :param i_node: The spring's i-node
        :type i_node: Node3D
        :param j_node: The spring's j-node
        :type j_node: Node3D
        :param ks: Either a linear spring constant (float) or a nonlinear material (UniaxialMaterial)
        :type ks: Union[float, UniaxialMaterial]
        :param LoadCombos: The dictionary of load combinations in the model this spring belongs to
        :type LoadCombos: Dict[str, LoadCombo]
        :param tension_only: Indicates whether the spring is tension-only
        :type tension_only: bool
        :param comp_only: Indicates whether the spring is compression-only
        :type comp_only: bool
        :param spring_type: Type of spring - 'axial' or 'rotational_z'
        :type spring_type: str
        """
        self.name: str = name  # A unique name for the spring given by the user
        self.ID: Optional[int] = (
            None  # Unique index number for the spring assigned by the program
        )
        self.i_node: Node3D = i_node  # The spring's i-node
        self.j_node: Node3D = j_node  # The spring's j-node
        self.load_combos: Dict[str, LoadCombo] = (
            LoadCombos  # The dictionary of load combinations in the model this spring belongs to
        )
        self.tension_only: bool = (
            tension_only  # Indicates whether the spring is tension-only
        )
        self.comp_only: bool = comp_only
        self.spring_type: str = spring_type  # Type of spring: 'axial' or 'rotational_z'

        # Handle both linear and nonlinear springs
        if isinstance(ks, (int, float)):
            # Linear spring
            self.ks: float = float(ks)
            self.material: Optional["UniaxialMaterial"] = None
            self.is_nonlinear: bool = False
        else:
            # Nonlinear spring with uniaxial material
            self.ks: float = (
                ks.get_initial_tangent()
            )  # Use initial tangent as reference
            self.material: "UniaxialMaterial" = ks
            self.is_nonlinear: bool = True

        # Springs need to track whether they are active or not for any given load combination.
        # They may become inactive for a load combination during a tension/compression-only
        # analysis. This dictionary will be used when the model is solved.
        self.active: Dict[str, bool] = (
            {}
        )  # Key = load combo name, Value = True or False

        # For nonlinear springs, store current deformation state
        self.current_deformation: float = 0.0

    # %%
    def L(self) -> float:
        """
        Returns the length of the spring.
        """

        # Return the distance between the two nodes
        return self.i_node.distance(self.j_node)

    # %%
    def k(self, combo_name: str = "Combo 1") -> NDArray[float64]:
        """
        Returns the local stiffness matrix for the spring.

        For nonlinear springs, this returns the tangent stiffness matrix
        based on the current state of deformation.
        """

        # Get the spring stiffness
        if self.is_nonlinear and self.material is not None:
            # For nonlinear materials, use the current tangent stiffness
            ks = self.material.get_tangent()
        else:
            # For linear springs, use the constant stiffness
            ks = self.ks

        # Create stiffness matrix based on spring type
        if self.spring_type == "rotational_z":
            # Rotational spring about Z-axis (local coordinate system)
            k = array(
                [
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, ks, 0, 0, 0, 0, 0, -ks],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, -ks, 0, 0, 0, 0, 0, ks],
                ]
            )
        else:
            # Default axial spring (existing behavior)
            k = array(
                [
                    [ks, 0, 0, 0, 0, 0, -ks, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [-ks, 0, 0, 0, 0, 0, ks, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                ]
            )

        # Return the local stiffness matrix
        return k

    # %%
    def f(self, combo_name: str = "Combo 1") -> NDArray[float64]:
        """
        Returns the spring's local end force vector for the given load combination.

        For nonlinear springs, this uses the material stress-strain relationship.
        For linear springs, this uses the standard stiffness relationship.

        Parameters
        ----------
        combo_name : string
            The name of the load combination to calculate the local end force vector for (not the load combination itself).
        """

        if self.is_nonlinear and self.material is not None:
            # For nonlinear materials, calculate force from material stress
            # Get the current deformation
            d_local = self.d(combo_name)

            if self.spring_type == "rotational_z":
                # Rotational deformation about Z-axis
                deformation = (
                    d_local[11, 0] - d_local[5, 0]
                )  # Relative rotation between j and i nodes
            else:
                # Axial deformation (existing behavior)
                deformation = (
                    d_local[6, 0] - d_local[0, 0]
                )  # Relative displacement between j and i nodes

            # Set the trial strain in the material and get the stress
            self.material.set_trial_strain(deformation)
            stress = self.material.get_stress()

            # Build the force vector based on spring type
            if self.spring_type == "rotational_z":
                # Rotational spring - moment about Z-axis
                f = array(
                    [
                        [0],
                        [0],
                        [0],
                        [0],
                        [0],
                        [-stress],  # i-node forces/moments
                        [0],
                        [0],
                        [0],
                        [0],
                        [0],
                        [stress],  # j-node forces/moments
                    ]
                )
            else:
                # Axial spring - force along X-axis
                f = array(
                    [
                        [-stress],  # i-node force
                        [0],
                        [0],
                        [0],
                        [0],
                        [0],
                        [stress],  # j-node force
                        [0],
                        [0],
                        [0],
                        [0],
                        [0],
                    ]
                )

            return f
        else:
            # For linear springs, use the standard stiffness relationship
            return matmul(self.k(combo_name), self.d(combo_name))

    # %%
    def d(self, combo_name: str = "Combo 1") -> NDArray[float64]:
        """
        Returns the spring's local displacement vector.

        Parameters
        ----------
        combo_name : string
            The name of the load combination to construct the displacement vector for (not the load combination itself).
        """

        # Calculate and return the local displacement vector
        return matmul(self.T(), self.D(combo_name))

    # %%
    def T(self) -> NDArray[float64]:
        """
        Returns the transformation matrix for the spring.
        """

        x1 = self.i_node.X
        x2 = self.j_node.X
        y1 = self.i_node.Y
        y2 = self.j_node.Y
        z1 = self.i_node.Z
        z2 = self.j_node.Z
        L = self.L()

        # Handle zero-length elements (common for rotational springs and connections)
        if isclose(L, 0.0, abs_tol=1e-10):
            # For zero-length elements, use identity transformation
            # This assumes the local coordinate system aligns with the global coordinate system
            dirCos = array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        else:
            # Calculate the direction cosines for the local x-axis
            x = [(x2 - x1) / L, (y2 - y1) / L, (z2 - z1) / L]

            # Calculate the remaining direction cosines. The local z-axis will be kept parallel to the global XZ plane in all cases
            # Vertical springs
            if isclose(x1, x2) and isclose(z1, z2):

                # For vertical springs, keep the local y-axis in the XY plane to make 2D problems easier to solve in the XY plane
                if y2 > y1:
                    y = [-1, 0, 0]
                    z = [0, 0, 1]
                else:
                    y = [1, 0, 0]
                    z = [0, 0, 1]

            # Horizontal springs
            elif isclose(y1, y2):

                # Find a vector in the direction of the local z-axis by taking the cross-product
                # of the local x-axis and the local y-axis. This vector will be perpendicular to
                # both the local x-axis and the local y-axis.
                y = [0, 1, 0]
                z = cross(x, y)

                # Divide the z-vector by its magnitude to produce a unit vector of direction cosines
                z = divide(z, (z[0] ** 2 + z[1] ** 2 + z[2] ** 2) ** 0.5)

            # Members neither vertical or horizontal
            else:

                # Find the projection of x on the global XZ plane
                proj = [x2 - x1, 0, z2 - z1]

                # Find a vector in the direction of the local z-axis by taking the cross-product
                # of the local x-axis and its projection on a plane parallel to the XZ plane. This
                # produces a vector perpendicular to both the local x-axis and its projection. This
                # vector will always be horizontal since it's parallel to the XZ plane. The order
                # in which the vectors are 'crossed' has been selected to ensure the y-axis always
                # has an upward component (i.e. the top of the beam is always on top).
                if y2 > y1:
                    z = cross(proj, x)
                else:
                    z = cross(x, proj)

                # Divide the z-vector by its magnitude to produce a unit vector of direction cosines
                z = divide(z, (z[0] ** 2 + z[1] ** 2 + z[2] ** 2) ** 0.5)

                # Find the direction cosines for the local y-axis
                y = cross(z, x)
                y = divide(y, (y[0] ** 2 + y[1] ** 2 + y[2] ** 2) ** 0.5)

            # Create the direction cosines matrix
            dirCos = array([x, y, z])

        # Build the transformation matrix
        transMatrix = zeros((12, 12))
        transMatrix[0:3, 0:3] = dirCos
        transMatrix[3:6, 3:6] = dirCos
        transMatrix[6:9, 6:9] = dirCos
        transMatrix[9:12, 9:12] = dirCos

        return transMatrix

    # %%
    def K(self, combo_name: str = "Combo 1") -> NDArray[float64]:
        """
        Spring global stiffness matrix
        """

        # Calculate and return the stiffness matrix in global coordinates
        return matmul(matmul(inv(self.T()), self.k(combo_name)), self.T())

    # %%
    def F(self, combo_name: str = "Combo 1") -> NDArray[float64]:
        """
        Returns the spring's global end force vector for the given load combination.
        """

        # Calculate and return the global force vector
        return matmul(inv(self.T()), self.f(combo_name))

    # %%
    def D(self, combo_name: str = "Combo 1") -> NDArray[float64]:
        """
        Returns the spring's global displacement vector.

        Parameters
        ----------
        combo_name : string
            The name of the load combination to construct the global
            displacement vector for (not the load combination itelf).
        """

        # Initialize the displacement vector
        D = zeros((12, 1))

        # Read in the global displacements from the nodes
        if self.spring_type == "rotational_z":
            # For rotational springs, only apply rotational displacements if spring is active
            if self.active[combo_name] == True:
                D[5, 0] = self.i_node.RZ[combo_name]
                D[11, 0] = self.j_node.RZ[combo_name]

            # Apply the remaining displacements
            D[0, 0] = self.i_node.DX[combo_name]
            D[1, 0] = self.i_node.DY[combo_name]
            D[2, 0] = self.i_node.DZ[combo_name]
            D[3, 0] = self.i_node.RX[combo_name]
            D[4, 0] = self.i_node.RY[combo_name]
            D[6, 0] = self.j_node.DX[combo_name]
            D[7, 0] = self.j_node.DY[combo_name]
            D[8, 0] = self.j_node.DZ[combo_name]
            D[9, 0] = self.j_node.RX[combo_name]
            D[10, 0] = self.j_node.RY[combo_name]
        else:
            # For axial springs, only apply axial displacements if spring is active
            if self.active[combo_name] == True:
                D[0, 0] = self.i_node.DX[combo_name]
                D[6, 0] = self.j_node.DX[combo_name]

            # Apply the remaining displacements
            D[1, 0] = self.i_node.DY[combo_name]
            D[2, 0] = self.i_node.DZ[combo_name]
            D[3, 0] = self.i_node.RX[combo_name]
            D[4, 0] = self.i_node.RY[combo_name]
            D[5, 0] = self.i_node.RZ[combo_name]
            D[7, 0] = self.j_node.DY[combo_name]
            D[8, 0] = self.j_node.DZ[combo_name]
            D[9, 0] = self.j_node.RX[combo_name]
            D[10, 0] = self.j_node.RY[combo_name]
            D[11, 0] = self.j_node.RZ[combo_name]

        # Return the global displacement vector
        return D

    # %%
    def axial(self, combo_name: str = "Combo 1") -> float:
        """
        Returns the axial force in the spring.

        Parameters
        ----------
        combo_name : string
            The name of the load combination to get the results for (not the load combination itself).
        """

        # Calculate the axial force
        return self.f(combo_name)[0, 0]

    # %%
    def rotational_moment(self, combo_name: str = "Combo 1") -> float:
        """
        Returns the rotational moment in the spring (for rotational springs).

        Parameters
        ----------
        combo_name : string
            The name of the load combination to get the results for (not the load combination itself).
        """

        if self.spring_type == "rotational_z":
            # Calculate the rotational moment
            return self.f(combo_name)[5, 0]
        else:
            return 0.0  # No rotational moment for axial springs

    # %%
    def set_trial_deformation(self, combo_name: str = "Combo 1") -> None:
        """
        Sets the trial deformation for nonlinear springs.
        This method should be called during the analysis iteration process.

        Parameters
        ----------
        combo_name : string
            The name of the load combination being analyzed.
        """

        if self.is_nonlinear and self.material is not None:
            # Get the current deformation
            d_local = self.d(combo_name)

            if self.spring_type == "rotational_z":
                # Rotational deformation about Z-axis
                deformation = (
                    d_local[11, 0] - d_local[5, 0]
                )  # Relative rotation between j and i nodes
            else:
                # Axial deformation (existing behavior)
                deformation = (
                    d_local[6, 0] - d_local[0, 0]
                )  # Relative displacement between j and i nodes

            # Set the trial strain in the material
            self.material.set_trial_strain(deformation)
            self.current_deformation = deformation

    # %%
    def commit_state(self) -> None:
        """
        Commits the current trial state for nonlinear springs.
        This method should be called after convergence is achieved.
        """

        if self.is_nonlinear and self.material is not None:
            self.material.commit_state()

    # %%
    def revert_to_last_commit(self) -> None:
        """
        Reverts the trial state to the last committed state for nonlinear springs.
        This method should be called when iterations fail to converge.
        """

        if self.is_nonlinear and self.material is not None:
            self.material.revert_to_last_commit()

    # %%
    def revert_to_start(self) -> None:
        """
        Reverts the spring to its initial state for nonlinear springs.
        """

        if self.is_nonlinear and self.material is not None:
            self.material.revert_to_start()
        self.current_deformation = 0.0

    # %%
    def get_material_info(self) -> dict:
        """
        Returns information about the spring's material properties.

        Returns
        -------
        dict
            Dictionary containing material information
        """

        if self.is_nonlinear and self.material is not None:
            return {
                "type": "nonlinear",
                "material_tag": self.material.tag,
                "current_strain": self.material.get_strain(),
                "current_stress": self.material.get_stress(),
                "current_tangent": self.material.get_tangent(),
            }
        else:
            return {
                "type": "linear",
                "spring_constant": self.ks,
                "current_deformation": self.current_deformation,
            }
