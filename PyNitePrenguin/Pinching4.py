import numpy as np

# Try relative import first (when used as part of package),
# fall back to absolute import (when used standalone)
try:
    from .UniaxialMaterial import UniaxialMaterial
except ImportError:
    from UniaxialMaterial import UniaxialMaterial


class Pinching4(UniaxialMaterial):
    """
    A Python implementation of the Pinching4 Uniaxial Material model from OpenSees.

    This material model features pinching of load-reversal cycles, as well as degradation
    of strength and stiffness under cyclic loading. The model is defined by a 4-point
    backbone curve in both positive and negative loading directions, parameters to
    control the shape of the hysteretic loops, and parameters to control damage accumulation.

    This implementation is designed to be compatible with the PyNite analysis framework.

    Original C++ Code by: N. Mitrou (nmitra@u.washington.edu), University of Washington
    Python Conversion by: AI Assistant for PyNite integration
    """

    def __init__(
        self,
        tag,
        s1p,
        d1p,
        s2p,
        d2p,
        s3p,
        d3p,
        s4p,
        d4p,
        s1n=None,
        d1n=None,
        s2n=None,
        d2n=None,
        s3n=None,
        d3n=None,
        s4n=None,
        d4n=None,
        r_disp_p=0.0,
        r_force_p=0.0,
        u_force_p=0.0,
        r_disp_n=None,
        r_force_n=None,
        u_force_n=None,
        gamma_k1=0.0,
        gamma_k2=0.0,
        gamma_k3=0.0,
        gamma_k4=0.0,
        gamma_k_limit=0.0,
        gamma_d1=0.0,
        gamma_d2=0.0,
        gamma_d3=0.0,
        gamma_d4=0.0,
        gamma_d_limit=0.0,
        gamma_f1=0.0,
        gamma_f2=0.0,
        gamma_f3=0.0,
        gamma_f4=0.0,
        gamma_f_limit=0.0,
        gamma_e=0.0,
        dmg_type="energy",
    ):
        """
        Initializes the Pinching4 material. Supports both symmetric and non-symmetric definitions.

        :param tag: Unique integer tag for the material.
        :type tag: int

        -- Backbone Curve Parameters (Stress-Strain Pairs) --
        :param s1p, d1p: Stress and strain at the first point on the positive backbone (yield).
        :param s2p, d2p: Stress and strain at the second point on the positive backbone.
        :param s3p, d3p: Stress and strain at the third point on the positive backbone.
        :param s4p, d4p: Stress and strain at the fourth point on the positive backbone (ultimate).
        :param s1n, d1n ... s4n, d4n: Parameters for the negative backbone. If None, symmetric behavior is assumed.

        -- Unloading-Reloading Rule Parameters (Pinching) --
        :param r_disp_p: 'Disp' ratio for pinching on the positive side. (strain_pinch / strain_reversal)
        :param r_force_p: 'Force' ratio for pinching on the positive side. (stress_pinch / stress_reversal)
        :param u_force_p: Stress ratio for reloading on the positive side.
        :param r_disp_n, r_force_n, u_force_n: Pinching parameters for the negative side. If None, symmetric behavior is assumed.

        -- Damage Parameters --
        :param gamma_k1..gamma_k_limit: Parameters for unloading stiffness degradation.
        :param gamma_d1..gamma_d_limit: Parameters for reloading stiffness degradation.
        :param gamma_f1..gamma_f_limit: Parameters for strength degradation.
        :param gamma_e: Exponent for hysteretic energy-based degradation.
        :param dmg_type: Type of damage calculation: 'energy' or 'cycle'.
        :type dmg_type: str
        """
        super().__init__(tag)

        # Store input parameters
        self.s1p, self.d1p = s1p, d1p
        self.s2p, self.d2p = s2p, d2p
        self.s3p, self.d3p = s3p, d3p
        self.s4p, self.d4p = s4p, d4p

        # Handle symmetric vs. non-symmetric properties
        self.s1n = s1n if s1n is not None else -s1p
        self.d1n = d1n if d1n is not None else -d1p
        self.s2n = s2n if s2n is not None else -s2p
        self.d2n = d2n if d2n is not None else -d2p
        self.s3n = s3n if s3n is not None else -s3p
        self.d3n = d3n if d3n is not None else -d3p
        self.s4n = s4n if s4n is not None else -s4p
        self.d4n = d4n if d4n is not None else -d4p

        self.r_disp_p, self.r_force_p, self.u_force_p = r_disp_p, r_force_p, u_force_p
        self.r_disp_n = r_disp_n if r_disp_n is not None else r_disp_p
        self.r_force_n = r_force_n if r_force_n is not None else r_force_p
        self.u_force_n = u_force_n if u_force_n is not None else u_force_p

        # Damage parameters
        (
            self.gamma_k1,
            self.gamma_k2,
            self.gamma_k3,
            self.gamma_k4,
            self.gamma_k_limit,
        ) = (gamma_k1, gamma_k2, gamma_k3, gamma_k4, gamma_k_limit)
        (
            self.gamma_d1,
            self.gamma_d2,
            self.gamma_d3,
            self.gamma_d4,
            self.gamma_d_limit,
        ) = (gamma_d1, gamma_d2, gamma_d3, gamma_d4, gamma_d_limit)
        (
            self.gamma_f1,
            self.gamma_f2,
            self.gamma_f3,
            self.gamma_f4,
            self.gamma_f_limit,
        ) = (gamma_f1, gamma_f2, gamma_f3, gamma_f4, gamma_f_limit)
        self.gamma_e = gamma_e
        self.dmg_cyc = 1 if dmg_type.lower() == "cycle" else 0

        # Envelope vectors (6 points: origin_offset, 4 user points, residual)
        self.envlp_pos_strain = np.zeros(6)
        self.envlp_pos_stress = np.zeros(6)
        self.envlp_neg_strain = np.zeros(6)
        self.envlp_neg_stress = np.zeros(6)

        self.envlp_pos_damgd_stress = np.zeros(6)
        self.envlp_neg_damgd_stress = np.zeros(6)

        # Unload/Reload path definition vectors (4 points)
        self.state3_strain = np.zeros(4)
        self.state3_stress = np.zeros(4)
        self.state4_strain = np.zeros(4)
        self.state4_stress = np.zeros(4)

        # Initialize other variables
        self.energy_capacity = 0.0
        self.k_unload = 0.0
        self.elastic_strain_energy = 0.0

        # Initialize committed and trial state variables
        self.C_strain = 0.0
        self.C_stress = 0.0
        self.C_tangent = 0.0
        self.C_state = 0
        self.C_strain_rate = 0.0

        self.low_C_state_strain = 0.0
        self.low_C_state_stress = 0.0
        self.hgh_C_state_strain = 0.0
        self.hgh_C_state_stress = 0.0
        self.C_min_strain_dmnd = 0.0
        self.C_max_strain_dmnd = 0.0

        self.C_energy = 0.0
        self.C_gamma_k = 0.0
        self.C_gamma_d = 0.0
        self.C_gamma_f = 0.0
        self.C_n_cycle = 0.0
        self.gamma_k_used = 0.0
        self.gamma_f_used = 0.0

        self.k_elastic_pos = 0.0
        self.k_elastic_neg = 0.0
        self.k_elastic_pos_damgd = 0.0
        self.k_elastic_neg_damgd = 0.0
        self.u_max_damgd = 0.0
        self.u_min_damgd = 0.0

        # Initialize the material state
        self._set_envelope()
        self.revert_to_start()

    def _set_envelope(self):
        """Initializes the backbone envelope curves from the user-provided points."""

        k_pos = self.s1p / self.d1p
        k_neg = self.s1n / self.d1n
        k = max(k_pos, k_neg)
        u = 1e-4 * self.d1p if self.d1p > -self.d1n else -1e-4 * self.d1n

        self.envlp_pos_strain[:] = [
            u,
            self.d1p,
            self.d2p,
            self.d3p,
            self.d4p,
            1e6 * self.d4p,
        ]
        self.envlp_pos_stress[:] = [u * k, self.s1p, self.s2p, self.s3p, self.s4p, 0]

        self.envlp_neg_strain[:] = [
            -u,
            self.d1n,
            self.d2n,
            self.d3n,
            self.d4n,
            1e6 * self.d4n,
        ]
        self.envlp_neg_stress[:] = [-u * k, self.s1n, self.s2n, self.s3n, self.s4n, 0]

        # Set residual strength/stiffness
        k1 = (
            (self.s4p - self.s3p) / (self.d4p - self.d3p)
            if (self.d4p - self.d3p) != 0
            else 0
        )
        k2 = (
            (self.s4n - self.s3n) / (self.d4n - self.d3n)
            if (self.d4n - self.d3n) != 0
            else 0
        )
        self.envlp_pos_stress[5] = (
            self.s4p + k1 * (self.envlp_pos_strain[5] - self.d4p)
            if k1 > 0
            else self.s4p * 1.1
        )
        self.envlp_neg_stress[5] = (
            self.s4n + k2 * (self.envlp_neg_strain[5] - self.d4n)
            if k2 > 0
            else self.s4n * 1.1
        )

        self.k_elastic_pos = self.envlp_pos_stress[1] / self.envlp_pos_strain[1]
        self.k_elastic_neg = self.envlp_neg_stress[1] / self.envlp_neg_strain[1]

        # Calculate monotonic energy capacity
        energy_pos = 0.5 * np.dot(
            self.envlp_pos_stress[:-1] + self.envlp_pos_stress[1:],
            np.diff(self.envlp_pos_strain),
        )
        energy_neg = 0.5 * np.dot(
            self.envlp_neg_stress[:-1] + self.envlp_neg_stress[1:],
            np.diff(self.envlp_neg_strain),
        )
        max_energy = max(energy_pos, abs(energy_neg))
        self.energy_capacity = (
            self.gamma_e * max_energy if self.gamma_e > 0 else float("inf")
        )

    def revert_to_start(self):
        """Resets the material to its initial, undeformed state."""
        self.C_state = 0
        self.C_strain = 0.0
        self.C_stress = 0.0
        self.C_strain_rate = 0.0

        self._set_envelope()

        self.low_C_state_strain = self.envlp_neg_strain[0]
        self.low_C_state_stress = self.envlp_neg_stress[0]
        self.hgh_C_state_strain = self.envlp_pos_strain[0]
        self.hgh_C_state_stress = self.envlp_pos_stress[0]
        self.C_min_strain_dmnd = self.envlp_neg_strain[1]
        self.C_max_strain_dmnd = self.envlp_pos_strain[1]

        self.C_energy = 0.0
        self.C_gamma_k = 0.0
        self.C_gamma_d = 0.0
        self.C_gamma_f = 0.0
        self.C_n_cycle = 0.0

        self.gamma_k_used = 0.0
        self.gamma_f_used = 0.0

        self.envlp_pos_damgd_stress = self.envlp_pos_stress.copy()
        self.envlp_neg_damgd_stress = self.envlp_neg_stress.copy()

        self.k_elastic_pos_damgd = self.k_elastic_pos
        self.k_elastic_neg_damgd = self.k_elastic_neg
        self.u_max_damgd = self.C_max_strain_dmnd
        self.u_min_damgd = self.C_min_strain_dmnd

        self.C_tangent = self.k_elastic_pos

        self.revert_to_last_commit()
        return 0

    def revert_to_last_commit(self):
        """Reverts the trial state variables to the last committed state."""
        self.T_strain = self.C_strain
        self.T_stress = self.C_stress
        self.T_tangent = self.C_tangent
        self.T_state = self.C_state

        self.T_strain_rate = self.C_strain_rate
        self.low_T_state_strain = self.low_C_state_strain
        self.low_T_state_stress = self.low_C_state_stress
        self.hgh_T_state_strain = self.hgh_C_state_strain
        self.hgh_T_state_stress = self.hgh_C_state_stress
        self.T_min_strain_dmnd = self.C_min_strain_dmnd
        self.T_max_strain_dmnd = self.C_max_strain_dmnd
        self.T_energy = self.C_energy
        self.T_gamma_k = self.C_gamma_k
        self.T_gamma_d = self.C_gamma_d
        self.T_gamma_f = self.C_gamma_f
        self.T_n_cycle = self.C_n_cycle
        return 0

    def commit_state(self):
        """Commits the current trial state."""
        self.C_state = self.T_state
        self.d_strain = self.T_strain - self.C_strain

        if abs(self.d_strain) > 1e-12:
            self.C_strain_rate = self.d_strain

        self.low_C_state_strain = self.low_T_state_strain
        self.low_C_state_stress = self.low_T_state_stress
        self.hgh_C_state_strain = self.hgh_T_state_strain
        self.hgh_C_state_stress = self.hgh_T_state_stress
        self.C_min_strain_dmnd = self.T_min_strain_dmnd
        self.C_max_strain_dmnd = self.T_max_strain_dmnd
        self.C_energy = self.T_energy

        self.C_stress = self.T_stress
        self.C_strain = self.T_strain
        self.C_tangent = self.T_tangent

        self.C_gamma_k = self.T_gamma_k
        self.C_gamma_d = self.T_gamma_d
        self.C_gamma_f = self.T_gamma_f

        # Finalize damage for the step
        self.gamma_k_used = self.C_gamma_k
        self.gamma_f_used = self.C_gamma_f

        self.k_elastic_pos_damgd = self.k_elastic_pos * (1 - self.gamma_k_used)
        self.k_elastic_neg_damgd = self.k_elastic_neg * (1 - self.gamma_k_used)

        self.u_max_damgd = self.T_max_strain_dmnd * (1 + self.C_gamma_d)
        self.u_min_damgd = self.T_min_strain_dmnd * (1 + self.C_gamma_d)

        self.envlp_pos_damgd_stress = self.envlp_pos_stress * (1 - self.gamma_f_used)
        self.envlp_neg_damgd_stress = self.envlp_neg_stress * (1 - self.gamma_f_used)

        self.C_n_cycle = self.T_n_cycle
        return 0

    def set_trial_strain(self, strain, strain_rate=0.0):
        """
        Sets the trial strain for the material and determines the corresponding
        stress and tangent stiffness.
        """
        self.revert_to_last_commit()

        self.T_strain = strain
        d_strain = self.T_strain - self.C_strain

        if abs(d_strain) < 1e-12:
            d_strain = 0.0

        # Determine new state if there is a change in state
        self._get_state(self.T_strain, d_strain)

        # Calculate stress and tangent based on the current state
        if self.T_state == 0:  # Elastic state
            self.T_tangent = self.k_elastic_pos
            self.T_stress = self.T_tangent * self.T_strain
        elif self.T_state == 1:  # On positive backbone
            self.T_stress = self._interp(
                self.envlp_pos_strain, self.envlp_pos_damgd_stress, strain
            )
            self.T_tangent = self._get_interp_tangent(
                self.envlp_pos_strain, self.envlp_pos_damgd_stress, strain
            )
        elif self.T_state == 2:  # On negative backbone
            self.T_stress = self._interp(
                self.envlp_neg_strain, self.envlp_neg_damgd_stress, strain
            )
            self.T_tangent = self._get_interp_tangent(
                self.envlp_neg_strain, self.envlp_neg_damgd_stress, strain
            )
        elif self.T_state == 3:  # Unloading from positive backbone
            self.k_unload = self.k_elastic_pos_damgd
            self.state3_strain[0] = self.low_T_state_strain
            self.state3_stress[0] = self.low_T_state_stress
            self.state3_strain[3] = self.hgh_T_state_strain
            self.state3_stress[3] = self.hgh_T_state_stress
            self._get_state3(self.state3_strain, self.state3_stress, self.k_unload)
            self.T_stress = self._interp(self.state3_strain, self.state3_stress, strain)
            self.T_tangent = self._get_interp_tangent(
                self.state3_strain, self.state3_stress, strain
            )
        elif self.T_state == 4:  # Unloading from negative backbone
            self.k_unload = self.k_elastic_neg_damgd
            self.state4_strain[0] = self.low_T_state_strain
            self.state4_stress[0] = self.low_T_state_stress
            self.state4_strain[3] = self.hgh_T_state_strain
            self.state4_stress[3] = self.hgh_T_state_stress
            self._get_state4(self.state4_strain, self.state4_stress, self.k_unload)
            self.T_stress = self._interp(self.state4_strain, self.state4_stress, strain)
            self.T_tangent = self._get_interp_tangent(
                self.state4_strain, self.state4_stress, strain
            )

        # Update energy and damage
        d_energy = 0.5 * (self.T_stress + self.C_stress) * d_strain
        k_damgd = (
            self.k_elastic_pos_damgd if self.T_strain > 0 else self.k_elastic_neg_damgd
        )
        if k_damgd != 0:
            self.elastic_strain_energy = 0.5 * self.T_stress**2 / k_damgd
        else:
            self.elastic_strain_energy = 0

        self.T_energy = self.C_energy + d_energy
        self._update_dmg(self.T_strain, d_strain)

        return 0

    def _get_state(self, u, du):
        """
        Determines the current state of the material (loading, unloading, etc.).
        State IDs: 0-Elastic, 1-OnPosEnv, 2-OnNegEnv, 3-UnloadPos, 4-UnloadNeg
        """
        change_in_direction = du * self.C_strain_rate <= 0.0
        out_of_bounds = u < self.low_T_state_strain or u > self.hgh_T_state_strain

        new_state = self.T_state
        state_has_changed = False

        if out_of_bounds or change_in_direction:
            if self.T_state == 0:
                if u > self.hgh_T_state_strain:
                    new_state = 1
                    self.low_T_state_strain = self.envlp_pos_strain[0]
                    self.low_T_state_stress = self.envlp_pos_stress[0]
                    self.hgh_T_state_strain = self.envlp_pos_strain[5]
                    self.hgh_T_state_stress = self.envlp_pos_stress[5]
                    state_has_changed = True
                elif u < self.low_T_state_strain:
                    new_state = 2
                    self.low_T_state_strain = self.envlp_neg_strain[5]
                    self.low_T_state_stress = self.envlp_neg_stress[5]
                    self.hgh_T_state_strain = self.envlp_neg_strain[0]
                    self.hgh_T_state_stress = self.envlp_neg_stress[0]
                    state_has_changed = True

            elif self.T_state == 1 and du < 0.0:  # Unloading from positive envelope
                self.T_max_strain_dmnd = max(self.T_max_strain_dmnd, self.C_strain)
                if u < self.u_min_damgd:
                    new_state = 2  # Go to negative envelope
                else:
                    new_state = 3  # Start unloading path
                    self.low_T_state_strain = self.u_min_damgd
                    self.low_T_state_stress = self._interp(
                        self.envlp_neg_strain,
                        self.envlp_neg_damgd_stress,
                        self.u_min_damgd,
                    )
                    self.hgh_T_state_strain = self.C_strain
                    self.hgh_T_state_stress = self.C_stress
                state_has_changed = True

            elif self.T_state == 2 and du > 0.0:  # Unloading from negative envelope
                self.T_min_strain_dmnd = min(self.T_min_strain_dmnd, self.C_strain)
                if u > self.u_max_damgd:
                    new_state = 1  # Go to positive envelope
                else:
                    new_state = 4  # Start unloading path
                    self.low_T_state_strain = self.C_strain
                    self.low_T_state_stress = self.C_stress
                    self.hgh_T_state_strain = self.u_max_damgd
                    self.hgh_T_state_stress = self._interp(
                        self.envlp_pos_strain,
                        self.envlp_pos_damgd_stress,
                        self.u_max_damgd,
                    )
                state_has_changed = True

            elif self.T_state == 3:  # On positive unloading path
                if u < self.low_T_state_strain:
                    new_state = 2  # Reached negative envelope
                elif u > self.u_max_damgd and du > 0.0:
                    new_state = 1  # Return to positive envelope
                elif du > 0.0:  # Reversed direction again
                    new_state = 4
                    self.low_T_state_strain = self.C_strain
                    self.low_T_state_stress = self.C_stress
                    self.hgh_T_state_strain = self.u_max_damgd
                    self.hgh_T_state_stress = self._interp(
                        self.envlp_pos_strain,
                        self.envlp_pos_damgd_stress,
                        self.u_max_damgd,
                    )
                state_has_changed = True

            elif self.T_state == 4:  # On negative unloading path
                if u > self.hgh_T_state_strain:
                    new_state = 1  # Reached positive envelope
                elif u < self.u_min_damgd and du < 0.0:
                    new_state = 2  # Return to negative envelope
                elif du < 0.0:  # Reversed direction again
                    new_state = 3
                    self.low_T_state_strain = self.u_min_damgd
                    self.low_T_state_stress = self._interp(
                        self.envlp_neg_strain,
                        self.envlp_neg_damgd_stress,
                        self.u_min_damgd,
                    )
                    self.hgh_T_state_strain = self.C_strain
                    self.hgh_T_state_stress = self.C_stress
                state_has_changed = True

        if state_has_changed:
            self.T_state = new_state

    def _get_state3(self, s_strain, s_stress, k_unload):
        """Defines the 4-point unloading/reloading path from positive to negative."""
        s_strain[1] = s_strain[3] * self.r_disp_p
        s_stress[1] = s_stress[3] * self.r_force_p

        # Reloading point
        target_stress = self.u_force_n * self._interp(
            self.envlp_neg_strain, self.envlp_neg_damgd_stress, self.T_min_strain_dmnd
        )
        s_stress[2] = target_stress

        if k_unload != 0:
            s_strain[2] = s_strain[0] + (s_stress[2] - s_stress[0]) / k_unload
        else:
            s_strain[2] = s_strain[0]

        # Check for path validity and correct if necessary
        if (
            s_strain[1] < s_strain[2]
            or (s_stress[2] - s_stress[1]) / (s_strain[2] - s_strain[1]) < 0
            if (s_strain[2] - s_strain[1]) != 0
            else False
        ):
            du = s_strain[3] - s_strain[0]
            df = s_stress[3] - s_stress[0]
            s_strain[1] = s_strain[0] + 0.33 * du
            s_stress[1] = s_stress[0] + 0.33 * df
            s_strain[2] = s_strain[0] + 0.67 * du
            s_stress[2] = s_stress[0] + 0.67 * df

    def _get_state4(self, s_strain, s_stress, k_unload):
        """Defines the 4-point unloading/reloading path from negative to positive."""
        s_strain[1] = (
            s_strain[0] + (s_stress[1] - s_stress[0]) / k_unload
            if k_unload != 0
            else s_strain[0]
        )
        s_stress[1] = self.u_force_p * self._interp(
            self.envlp_pos_strain, self.envlp_pos_damgd_stress, self.T_max_strain_dmnd
        )

        # Pinching point
        s_strain[2] = s_strain[3] * self.r_disp_p
        s_stress[2] = s_stress[3] * self.r_force_p

        # Check for path validity
        if (
            s_strain[1] > s_strain[2]
            or (s_stress[2] - s_stress[1]) / (s_strain[2] - s_strain[1]) < 0
            if (s_strain[2] - s_strain[1]) != 0
            else False
        ):
            du = s_strain[3] - s_strain[0]
            df = s_stress[3] - s_stress[0]
            s_strain[1] = s_strain[0] + 0.33 * du
            s_stress[1] = s_stress[0] + 0.33 * df
            s_strain[2] = s_strain[0] + 0.67 * du
            s_stress[2] = s_stress[0] + 0.67 * df

    def _update_dmg(self, strain, d_strain):
        """Updates the damage indices for the current step."""
        umax_abs = max(self.T_max_strain_dmnd, -self.T_min_strain_dmnd)
        uult_abs = max(self.envlp_pos_strain[4], -self.envlp_neg_strain[4])

        if umax_abs > 0:
            self.T_n_cycle = self.C_n_cycle + abs(d_strain) / (4 * umax_abs)

        if (
            uult_abs > 0
            and umax_abs < uult_abs
            and self.T_energy < self.energy_capacity
        ):
            u_ratio = umax_abs / uult_abs
            self.T_gamma_k = self.gamma_k1 * (u_ratio**self.gamma_k3)
            self.T_gamma_d = self.gamma_d1 * (u_ratio**self.gamma_d3)
            self.T_gamma_f = self.gamma_f1 * (u_ratio**self.gamma_f3)

            if self.dmg_cyc == 0:  # Energy-based damage
                if self.energy_capacity > 0:
                    e_ratio = (
                        self.T_energy - self.elastic_strain_energy
                    ) / self.energy_capacity
                    if e_ratio > 0:
                        self.T_gamma_k += self.gamma_k2 * (e_ratio**self.gamma_k4)
                        self.T_gamma_d += self.gamma_d2 * (e_ratio**self.gamma_d4)
                        self.T_gamma_f += self.gamma_f2 * (e_ratio**self.gamma_f4)
            else:  # Cycle-based damage
                self.T_gamma_k += self.gamma_k2 * (self.T_n_cycle**self.gamma_k4)
                self.T_gamma_d += self.gamma_d2 * (self.T_n_cycle**self.gamma_d4)
                self.T_gamma_f += self.gamma_f2 * (self.T_n_cycle**self.gamma_f4)

            # Apply limits
            self.T_gamma_k = min(self.T_gamma_k, self.gamma_k_limit)
            self.T_gamma_d = min(self.T_gamma_d, self.gamma_d_limit)
            self.T_gamma_f = min(self.T_gamma_f, self.gamma_f_limit)
        elif self.energy_capacity > 0 and self.T_energy >= self.energy_capacity:
            self.T_gamma_k = self.gamma_k_limit
            self.T_gamma_d = self.gamma_d_limit
            self.T_gamma_f = self.gamma_f_limit

    def _interp(self, x_vals, y_vals, x):
        """Linearly interpolates, handling out-of-bounds with constant extrapolation."""
        return np.interp(x, x_vals, y_vals)

    def _get_interp_tangent(self, x_vals, y_vals, x):
        """Gets the tangent (slope) of the segment containing x."""
        if x <= x_vals[0]:
            return (
                (y_vals[1] - y_vals[0]) / (x_vals[1] - x_vals[0])
                if (x_vals[1] - x_vals[0]) != 0
                else 0
            )
        if x >= x_vals[-1]:
            return (
                (y_vals[-1] - y_vals[-2]) / (x_vals[-1] - x_vals[-2])
                if (x_vals[-1] - x_vals[-2]) != 0
                else 0
            )

        # Find the segment
        for i in range(len(x_vals) - 1):
            if x_vals[i] <= x <= x_vals[i + 1]:
                dx = x_vals[i + 1] - x_vals[i]
                dy = y_vals[i + 1] - y_vals[i]
                return dy / dx if dx != 0 else float("inf")
        return 0  # Should not be reached

    def get_copy(self):
        """Returns a copy of the material."""
        # This is a deep copy. A more robust implementation might use copy.deepcopy
        new_mat = Pinching4(
            self.tag,
            self.s1p,
            self.d1p,
            self.s2p,
            self.d2p,
            self.s3p,
            self.d3p,
            self.s4p,
            self.d4p,
            self.s1n,
            self.d1n,
            self.s2n,
            self.d2n,
            self.s3n,
            self.d3n,
            self.s4n,
            self.d4n,
            self.r_disp_p,
            self.r_force_p,
            self.u_force_p,
            self.r_disp_n,
            self.r_force_n,
            self.u_force_n,
            self.gamma_k1,
            self.gamma_k2,
            self.gamma_k3,
            self.gamma_k4,
            self.gamma_k_limit,
            self.gamma_d1,
            self.gamma_d2,
            self.gamma_d3,
            self.gamma_d4,
            self.gamma_d_limit,
            self.gamma_f1,
            self.gamma_f2,
            self.gamma_f3,
            self.gamma_f4,
            self.gamma_f_limit,
            self.gamma_e,
            "cycle" if self.dmg_cyc == 1 else "energy",
        )
        # Copy all state variables (both committed and trial)
        new_mat.C_strain = self.C_strain
        new_mat.C_stress = self.C_stress
        new_mat.C_tangent = self.C_tangent
        new_mat.C_state = self.C_state
        new_mat.T_strain = self.T_strain
        new_mat.T_stress = self.T_stress
        new_mat.T_tangent = self.T_tangent
        new_mat.T_state = self.T_state

        # Copy all other state variables
        new_mat.low_C_state_strain = self.low_C_state_strain
        new_mat.low_C_state_stress = self.low_C_state_stress
        new_mat.hgh_C_state_strain = self.hgh_C_state_strain
        new_mat.hgh_C_state_stress = self.hgh_C_state_stress
        new_mat.C_min_strain_dmnd = self.C_min_strain_dmnd
        new_mat.C_max_strain_dmnd = self.C_max_strain_dmnd
        new_mat.C_energy = self.C_energy
        new_mat.C_gamma_k = self.C_gamma_k
        new_mat.C_gamma_d = self.C_gamma_d
        new_mat.C_gamma_f = self.C_gamma_f
        new_mat.C_n_cycle = self.C_n_cycle

        return new_mat
