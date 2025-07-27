"""
PyNitePrenguin - Enhanced 3D Structural Finite Element Library

Based on the original Pynite by D. Craig Brinck, PE, SE
Enhanced with custom modifications for advanced structural analysis.

Original Pynite: https://github.com/JWock82/Pynite
PyNitePrenguin: https://github.com/Constratum/PyNitePrenguin

License: MIT (see LICENSE file)
"""

__version__ = "1.2.1"
__author__ = "D. Craig Brinck (Original), Your Name (Modifications)"
__license__ = "MIT"

# Core imports - Main classes users will need
from PyNitePrenguin.FEModel3D import FEModel3D
from PyNitePrenguin.Node3D import Node3D
from PyNitePrenguin.Member3D import Member3D
from PyNitePrenguin.Spring3D import Spring3D
from PyNitePrenguin.Material import Material
from PyNitePrenguin.Section import Section
from PyNitePrenguin.ShearWall import ShearWall

# Additional imports for advanced users
from PyNitePrenguin.Plate3D import Plate3D
from PyNitePrenguin.Quad3D import Quad3D
from PyNitePrenguin.Tri3D import Tri3D
from PyNitePrenguin.LoadCombo import LoadCombo
from PyNitePrenguin.PhysMember import PhysMember

# Visualization and reporting
from PyNitePrenguin import Visualization
from PyNitePrenguin import Rendering

# Package information functions
def get_version():
    """Get PyNitePrenguin version"""
    return __version__

def get_info():
    """Get package information"""
    return {
        "name": "PyNitePrenguin",
        "version": __version__,
        "author": __author__,
        "license": __license__,
        "based_on": "Pynite by D. Craig Brinck, PE, SE",
        "enhancements": "Custom modifications for advanced analysis"
    }

# Make key classes available at package level
__all__ = [
    'FEModel3D',
    'Node3D', 
    'Member3D',
    'Spring3D',
    'Material',
    'Section',
    'ShearWall',
    'Plate3D',
    'Quad3D', 
    'Tri3D',
    'LoadCombo',
    'PhysMember',
    'Visualization',
    'Rendering',
    'get_version',
    'get_info'
]
