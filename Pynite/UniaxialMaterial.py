from __future__ import annotations
from typing import TYPE_CHECKING
from abc import ABC, abstractmethod

if TYPE_CHECKING:
    from typing import Any

class UniaxialMaterial(ABC):
    """
    Abstract base class for uniaxial material models in Pynite.
    
    This class defines the interface that all uniaxial materials must implement
    to be compatible with Pynite's analysis framework.
    """
    
    def __init__(self, tag: int):
        """
        Initialize the uniaxial material.
        
        :param tag: Unique integer identifier for this material
        :type tag: int
        """
        self.tag = tag
        
        # Current trial state
        self.T_strain = 0.0
        self.T_stress = 0.0
        self.T_tangent = 0.0
        
        # Last committed state
        self.C_strain = 0.0
        self.C_stress = 0.0
        self.C_tangent = 0.0
    
    @abstractmethod
    def set_trial_strain(self, strain: float, strain_rate: float = 0.0) -> int:
        """
        Set the trial strain and compute the corresponding stress and tangent.
        
        :param strain: Trial strain value
        :type strain: float
        :param strain_rate: Strain rate (optional, defaults to 0.0)
        :type strain_rate: float
        :return: 0 if successful, non-zero if error
        :rtype: int
        """
        pass
    
    @abstractmethod
    def commit_state(self) -> int:
        """
        Commit the current trial state as the new committed state.
        
        :return: 0 if successful, non-zero if error
        :rtype: int
        """
        pass
    
    @abstractmethod
    def revert_to_last_commit(self) -> int:
        """
        Revert trial state to the last committed state.
        
        :return: 0 if successful, non-zero if error
        :rtype: int
        """
        pass
    
    @abstractmethod
    def revert_to_start(self) -> int:
        """
        Revert material to its initial state.
        
        :return: 0 if successful, non-zero if error
        :rtype: int
        """
        pass
    
    @abstractmethod
    def get_copy(self) -> 'UniaxialMaterial':
        """
        Create and return a copy of this material.
        
        :return: Copy of this material
        :rtype: UniaxialMaterial
        """
        pass
    
    def get_strain(self) -> float:
        """
        Get the current trial strain.
        
        :return: Trial strain
        :rtype: float
        """
        return self.T_strain
    
    def get_stress(self) -> float:
        """
        Get the current trial stress.
        
        :return: Trial stress
        :rtype: float
        """
        return self.T_stress
    
    def get_tangent(self) -> float:
        """
        Get the current trial tangent stiffness.
        
        :return: Trial tangent stiffness
        :rtype: float
        """
        return self.T_tangent
    
    def get_initial_tangent(self) -> float:
        """
        Get the initial tangent stiffness.
        Default implementation returns current tangent.
        
        :return: Initial tangent stiffness
        :rtype: float
        """
        return self.T_tangent 