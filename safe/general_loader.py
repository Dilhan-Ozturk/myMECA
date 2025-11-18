import xml.etree.ElementTree as ET
from lxml import etree 
import numpy as np
from typing import Dict, List, Any
from dataclasses import dataclass, field

@dataclass
class GeneralData:
    """General file configuration"""
    # File paths
    general_file: str

    # Display options - These fields will automatically get their default values
    paraview: int = 0
    paraview_wf: int = 0 # Wall flux
    paraview_mf: int = 0 # Membrane flux
    paraview_pf: int = 0 # Plasmodesmata flux
    paraview_wp: int = 0 # Wall potential
    paraview_cp: int = 0 # Cell potential

    # Analysis flags
    sym_contagion: int = 0 # Symplastic contagion
    apo_contagion: int = 0 # Apoplastic contagion
    par_track: int = 0

    # Display parameters
    color_threshold: float = 1.0 # Threshold for color mapping
    thickness_disp: float = 0.0 # Wall thickness display
    thickness_junction_disp: float = 0.0 # Junction thickness display
    radius_plasmodesm_disp: float = 0.0 # Plasmodesmata radius display

    def __post_init__(self):
        self._load_general()

    def _load_general(self):
        """Load general configuration parameters"""
        print('  Loading general parameters...')
        root = etree.parse(self.general_file).getroot()

        self.paraview = int(root.xpath('Paraview')[0].get("value"))
        self.paraview_wf = int(root.xpath('Paraview')[0].get("WallFlux"))
        self.paraview_mf = int(root.xpath('Paraview')[0].get("MembraneFlux"))
        self.paraview_pf = int(root.xpath('Paraview')[0].get("PlasmodesmataFlux"))
        self.paraview_wp = int(root.xpath('Paraview')[0].get("WallPot"))
        self.paraview_cp = int(root.xpath('Paraview')[0].get("CellPot"))

        self.par_track = int(root.xpath('ParTrack')[0].get("value"))
        self.sym_contagion = int(root.xpath('Sym_Contagion')[0].get("value"))
        self.apo_contagion = int(root.xpath('Apo_Contagion')[0].get("value"))

        self.color_threshold = float(root.xpath('color_threshold')[0].get("value"))
        self.thickness_disp = float(root.xpath('thickness_disp')[0].get("value"))
        self.thickness_junction_disp = float(root.xpath('thicknessJunction_disp')[0].get("value"))
        self.radius_plasmodesm_disp = float(root.xpath('radiusPlasmodesm_disp')[0].get("value"))

