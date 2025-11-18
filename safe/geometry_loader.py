import xml.etree.ElementTree as ET
from lxml import etree 
import numpy as np
from typing import Dict, List, Any
from dataclasses import dataclass, field

@dataclass
class GeometryData:
    """General file configuration"""
    # File paths
    geometry_file: str

    # Parsed configuration
    plant_name: str = ""
    im_scale: float = 1.0
   
    # Maturity stages
    maturity_elems: List[Dict[str, int]] = field(default_factory=list) # List of maturity dicts
    maturity_stages: List[Dict[str, int]] = field(default_factory=list) # List of maturity dicts
    n_maturity: int = 0

    # Passage cells and aerenchyma
    passage_cell_ids: List[int] = field(default_factory=list) # not used
    intercellular_ids: List[int] = field(default_factory=list)

    # Intercellular perimeters
    interc_perims: List[float] = field(default_factory=lambda: [0.0]*5)
    k_interc: float = 0.0

    # Cell layers
    cell_per_layer: np.ndarray = field(default_factory=lambda: np.zeros((2, 1)))
    diffusion_length: np.ndarray = field(default_factory=lambda: np.zeros((2, 1))) # not used

    # Geometry parameters
    thickness: float = 0.0
    pd_section: float = 0.0
    xylem_pieces: bool = False

    def __post_init__(self):
        self._load_geometry()

    def _load_geometry(self):
        """Load geometry configuration"""
        print('  Loading geometry parameters...')
        root = etree.parse(self.geometry_file).getroot()
        
        self.plant_name = root.xpath('Plant')[0].get("value")
        self.im_scale = float(root.xpath('im_scale')[0].get("value"))
        
        # Parse maturity stages
        self.maturity_elems = root.xpath('Maturityrange/Maturity')
        for mat in self.maturity_elems:
            self.maturity_stages.append({
                'barrier': int(mat.get("Barrier")),
                'height': int(mat.get("height"))
            })
        
        self.n_maturity = len(self.maturity_stages)
        # Parse passage cells
        passage_elems = root.xpath('passage_cell_range/passage_cell')
        self.passage_cell_ids = [int(pc.get("id")) for pc in passage_elems]
        
        # Parse aerenchyma (intercellular spaces)
        aerenchyma_elems = root.xpath('aerenchyma_range/aerenchyma')
        self.intercellular_ids = [
            int(aer.get("id")) for aer in aerenchyma_elems 
            if int(aer.get("id")) > 0
        ]
        
        # Intercellular perimeters
        for i in range(1, 5):
            self.interc_perims[i-1] = float(
                root.xpath(f'InterC_perim{i}')[0].get("value")
            )
        self.k_interc = float(root.xpath('kInterC')[0].get("value"))
        
        # Cell layers
        cell_layer_elem = root.xpath('cell_per_layer')[0]
        self.cell_per_layer[0][0] = float(cell_layer_elem.get("cortex"))
        self.cell_per_layer[1][0] = float(cell_layer_elem.get("stele"))
        
        diff_length_elem = root.xpath('diffusion_length')[0]
        self.diffusion_length[0][0] = float(diff_length_elem.get("cortex"))
        self.diffusion_length[1][0] = float(diff_length_elem.get("stele"))
        
        self.thickness = float(root.xpath('thickness')[0].get("value")) # in microns
        self.pd_section = float(root.xpath('PD_section')[0].get("value")) # in microns^2
        self.xylem_pieces = float(root.xpath('Xylem_pieces')[0].get("flag")) == 1
    

