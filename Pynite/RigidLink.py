# -*- coding: utf-8 -*-
"""
Created on Sun, June 25, 2024

@author: Nima
"""

import numpy as np
from numpy.linalg import inv


class RigidLink:
    """
    A class representing a rigid link element in a 3D model.
    """

    def __init__(self, name, i_node, j_node):
        """
        Initializes a new rigid link.
        """
        self.name = name
        self.i_node = i_node
        self.j_node = j_node

    def k(self):
        """
        Returns the global stiffness matrix for the rigid link.
        """

        # Calculate the transformation matrix
        T = self._T()

        # Define a very large stiffness value to simulate rigidity
        k_rigid = 1e12

        # Create a local stiffness matrix for a very stiff member
        k_local = np.zeros((12, 12))
        k_local[0, 0] = k_local[6, 6] = k_rigid
        k_local[0, 6] = k_local[6, 0] = -k_rigid
        k_local[1, 1] = k_local[7, 7] = k_rigid
        k_local[1, 7] = k_local[7, 1] = -k_rigid
        k_local[2, 2] = k_local[8, 8] = k_rigid
        k_local[2, 8] = k_local[8, 2] = -k_rigid
        k_local[3, 3] = k_local[9, 9] = k_rigid
        k_local[3, 9] = k_local[9, 3] = -k_rigid
        k_local[4, 4] = k_local[10, 10] = k_rigid
        k_local[4, 10] = k_local[10, 4] = -k_rigid
        k_local[5, 5] = k_local[11, 11] = k_rigid
        k_local[5, 11] = k_local[11, 5] = -k_rigid

        # Transform the local stiffness matrix to global coordinates
        return T.T @ k_local @ T

    def _T(self):
        """
        Returns the transformation matrix for the rigid link.
        """

        xi, yi, zi = self.i_node.X, self.i_node.Y, self.i_node.Z
        xj, yj, zj = self.j_node.X, self.j_node.Y, self.j_node.Z

        # Calculate the vector from node i to node j
        x = xj - xi
        y = yj - yi
        z = zj - zi

        # Create the transformation matrix to relate local and global DOFs
        T = np.zeros((12, 12))

        # Transformation for translations
        T[0, 0] = T[1, 1] = T[2, 2] = 1
        T[6, 6] = T[7, 7] = T[8, 8] = 1

        # Transformation for rotations (master-slave relationship)
        # Slave translations as a function of master translations and rotations
        # d_slave = d_master + cross(r_master, pos_slave - pos_master)
        # For translations: DXj = DXi - (zj-zi)*RYi + (yj-yi)*RZi
        T[6, 0] = 1
        T[6, 4] = -(z)
        T[6, 5] = y

        # DYj = DYi + (zj-zi)*RXi - (xj-xi)*RZi
        T[7, 1] = 1
        T[7, 3] = z
        T[7, 5] = -(x)

        # DZj = DZi - (yj-yi)*RXi + (xj-xi)*RYi
        T[8, 2] = 1
        T[8, 3] = -(y)
        T[8, 4] = x

        # For rotations: Rj = Ri
        T[9, 3] = 1
        T[10, 4] = 1
        T[11, 5] = 1

        return T
