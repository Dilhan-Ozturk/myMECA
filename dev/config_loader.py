"""
Configuration loader for MECHA
Handles parsing of all XML input files
"""

import xml.etree.ElementTree as ET
from lxml import etree
import numpy as np
from typing import Dict, List, Any
from dataclasses import dataclass, field

@dataclass
class MECHAConfig:
    """Central configuration object for MECHA simulation"""

    # File paths
    general_file: str
    geometry_file: str
    hydraulics_file: str
    bc_file: str
    hormone_file: str
    cellset_file: str

    # Parsed configuration
    plant_name: str = ""
    im_scale: float = 1.0
    thickness: float = 0.0
    height: float = 0.0

    # Display options
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

    # Maturity stages
    maturity_stages: List[Dict[str, int]] = field(default_factory=list) # List of maturity dicts
    n_maturity: int = 0

    # Passage cells and aerenchyma
    passage_cell_ids: List[int] = field(default_factory=list)
    intercellular_ids: List[int] = field(default_factory=list)

    # Intercellular perimeters
    interc_perims: List[float] = field(default_factory=lambda: [0.0]*5)
    k_interc: float = 0.0

    # Cell layers
    cell_per_layer: np.ndarray = field(default_factory=lambda: np.zeros((2, 1)))
    diffusion_length: np.ndarray = field(default_factory=lambda: np.zeros((2, 1)))

    # Geometry parameters
    pd_section: float = 0.0
    xylem_pieces: bool = False

    # Hydraulic parameters
    hydraulic_configs: List[Dict[str, Any]] = field(default_factory=list)
    n_hydraulics: int = 0

    # Boundary conditions
    scenarios: List[Dict[str, Any]] = field(default_factory=list)
    n_scenarios: int = 0

    # Hormone properties
    hormone_config: Dict[str, Any] = field(default_factory=dict)

    def __init__(self, gen: str, geom: str, hydr: str, bc: str, horm: str, cellset: str):
        # Initialize file paths
        self.general_file = gen
        self.geometry_file = geom
        self.hydraulics_file = hydr
        self.bc_file = bc
        self.hormone_file = horm
        self.cellset_file = cellset

        # Manually initialize all dataclass fields that have default_factory or direct defaults
        # since the custom __init__ overrides the dataclass's default initialization.

        # Parsed configuration defaults
        self.plant_name = ""
        self.im_scale = 1.0
        self.thickness = 0.0
        self.height = 0.0

        # Display options defaults
        self.paraview = 0
        self.paraview_wf = 0
        self.paraview_mf = 0
        self.paraview_pf = 0
        self.paraview_wp = 0
        self.paraview_cp = 0

        # Analysis flags defaults
        self.sym_contagion = 0
        self.apo_contagion = 0
        self.par_track = 0

        # Display parameters defaults
        self.color_threshold = 1.0
        self.thickness_disp = 0.0
        self.thickness_junction_disp = 0.0
        self.radius_plasmodesm_disp = 0.0

        # Initialize lists and dicts using default_factory
        self.maturity_stages: List[Dict[str, int]] = []
        self.passage_cell_ids: List[int] = []
        self.intercellular_ids: List[int] = []
        self.interc_perims: List[float] = [0.0] * 5
        self.k_interc: float = 0.0

        # Initialize numpy arrays
        self.cell_per_layer: np.ndarray = np.zeros((2, 1))
        self.diffusion_length: np.ndarray = np.zeros((2, 1)) # Not used !!!

        # Geometry parameters defaults
        self.pd_section = 0.0
        self.xylem_pieces = False # Initialize as False before loading from XML

        # Hydraulic parameters defaults
        self.hydraulic_configs: List[Dict[str, Any]] = []
        self.n_hydraulics: int = 0

        # Boundary conditions defaults
        self.scenarios: List[Dict[str, Any]] = []
        self.n_scenarios: int = 0

        # Hormone properties defaults
        self.hormone_config: Dict[str, Any] = {}

        # Load all configurations
        self._load_general()
        self._load_geometry()
        self._load_hydraulics()
        self._load_boundary_conditions()
        self._load_hormones()
    
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
    
    def _load_geometry(self):
        """Load geometry configuration"""
        print('  Loading geometry parameters...')
        root = etree.parse(self.geometry_file).getroot()
        
        self.plant_name = root.xpath('Plant')[0].get("value")
        self.im_scale = float(root.xpath('im_scale')[0].get("value"))
        
        # Parse maturity stages
        maturity_elems = root.xpath('Maturityrange/Maturity')
        for mat in maturity_elems:
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
    
    def _load_hormones(self):
        """Load hormone and carrier configuration"""
        if self.sym_contagion or self.apo_contagion:
            print('  Loading hormone parameters...')
            root = etree.parse(self.hormone_file).getroot()

            # Hormone movement parameters
            self.degrad1 = float(root.xpath('Hormone_movement/Degradation_constant_H1')[0].get("value"))
            self.diff_pd1 = float(root.xpath('Hormone_movement/Diffusivity_PD_H1')[0].get("value"))
            self.diff_pw1 = float(root.xpath('Hormone_movement/Diffusivity_PW_H1')[0].get("value"))
            self.d2o1 = int(root.xpath('Hormone_movement/H1_D2O')[0].get("flag")) == 1

            # Parse active transport carriers
            carrier_elems = root.xpath('Hormone_active_transport/carrier_range/carrier')
            # Store carrier data if needed (structure depends on your XML)

            # Parse symplastic contagion
            sym_source_elems = root.xpath('Sym_Contagion/source_range/source')
            self.sym_zombie0 = [int(source.get("id")) for source in sym_source_elems]
            self.sym_cc = [float(source.get("concentration")) for source in sym_source_elems]

            sym_target_elems = root.xpath('Sym_Contagion/target_range/target')
            self.sym_target = [int(target.get("id")) for target in sym_target_elems]

            sym_immune_elems = root.xpath('Sym_Contagion/immune_range/immune')
            self.sym_immune = [int(immune.get("id")) for immune in sym_immune_elems]
            # Parse apoplastic contagion
            apo_source_elems = root.xpath('Apo_Contagion/source_range/source')
            self.apo_zombie0 = [int(source.get("id")) for source in apo_source_elems]
            self.apo_cc = [float(source.get("concentration")) for source in apo_source_elems]

            apo_target_elems = root.xpath('Apo_Contagion/target_range/target')
            self.apo_target = [int(target.get("id")) for target in apo_target_elems]
            
            apo_immune_elems = root.xpath('Apo_Contagion/immune_range/immune')
            self.apo_immune = [int(immune.get("id")) for immune in apo_immune_elems]

            # Parse contact range
            contact_elems = root.xpath('Contactrange/Contact')
            self.contact = [int(contact.get("id")) for contact in contact_elems]

    def _load_hydraulics(self):
        """Load hydraulic parameters"""
        print('  Loading hydraulic parameters...')
        root = etree.parse(self.hydraulics_file).getroot()

        # Parse different hydraulic parameter sets
        self.kw_elems = root.xpath('kwrange/kw')
        self.kw_barrier_elems = root.xpath('kw_barrier_range/kw_barrier')
        self.kaqp_elems = root.xpath('kAQPrange/kAQP')
        self.kpl_elems = root.xpath('Kplrange/Kpl')

        self.n_kw = len(self.kw_elems)
        self.n_kw_barrier = len(self.kw_barrier_elems)
        self.n_kaqp = len(self.kaqp_elems)
        self.n_kpl = len(self.kpl_elems)

        # Extract single-value parameters
        self.kmb = float(root.xpath('km')[0].get("value"))
        self.ratio_cortex = float(root.xpath('ratio_cortex')[0].get("value"))
        self.fplxheight = float(root.xpath('Fplxheight')[0].get("value"))
        self.fplxheight_epi_exo = float(root.xpath('Fplxheight_epi_exo')[0].get("value"))
        self.fplxheight_outer_cortex = float(root.xpath('Fplxheight_outer_cortex')[0].get("value"))
        self.fplxheight_cortex_cortex = float(root.xpath('Fplxheight_cortex_cortex')[0].get("value"))
        self.fplxheight_cortex_endo = float(root.xpath('Fplxheight_cortex_endo')[0].get("value"))
        self.fplxheight_endo_endo = float(root.xpath('Fplxheight_endo_endo')[0].get("value"))
        self.fplxheight_endo_peri = float(root.xpath('Fplxheight_endo_peri')[0].get("value")) # not used
        self.fplxheight_peri_peri = float(root.xpath('Fplxheight_peri_peri')[0].get("value"))
        self.fplxheight_peri_stele = float(root.xpath('Fplxheight_peri_stele')[0].get("value"))
        self.fplxheight_stele_stele = float(root.xpath('Fplxheight_stele_stele')[0].get("value"))
        self.fplxheight_stele_comp = float(root.xpath('Fplxheight_stele_comp')[0].get("value"))
        self.fplxheight_peri_comp = float(root.xpath('Fplxheight_peri_comp')[0].get("value"))
        self.fplxheight_comp_comp = float(root.xpath('Fplxheight_comp_comp')[0].get("value"))
        self.fplxheight_comp_sieve = float(root.xpath('Fplxheight_comp_sieve')[0].get("value"))
        self.fplxheight_peri_sieve = float(root.xpath('Fplxheight_peri_sieve')[0].get("value"))
        self.fplxheight_stele_sieve = float(root.xpath('Fplxheight_stele_sieve')[0].get("value"))
        self.k_sieve = float(root.xpath('K_sieve')[0].get("value"))  # Sieve tube hydraulic conductance
        self.k_xyl = float(root.xpath('K_xyl')[0].get("value"))  # Xylem vessel axial hydraulic conductance
        self.xcontactrange = root.xpath('Xcontactrange/Xcontact')
        self.path_hydraulics = root.xpath('path_hydraulics/Output')

        self.n_hydraulics = len(self.path_hydraulics) if self.path_hydraulics else 1
        self.kw = [float(self.kw_elems[i].get("value")) for i in range(self.n_hydraulics)]
        self.kw_barrier = [float(self.kw_barrier_elems[i].get("value")) for i in range(self.n_hydraulics)]


    def _load_boundary_conditions(self):
        """Load boundary condition scenarios"""
        print('  Loading boundary conditions...')
        root = etree.parse(self.bc_file).getroot()

        # Parse different boundary condition elements
        psi_soil_elems = root.xpath('Psi_soil_range/Psi_soil')
        bc_xyl_elems = root.xpath('BC_xyl_range/BC_xyl')
        bc_sieve_elems = root.xpath('BC_sieve_range/BC_sieve')
        psi_cell_elems = root.xpath('Psi_cell_range/Psi_cell')
        elong_cell_elems = root.xpath('Elong_cell_range/Elong_cell')
        water_fractions = root.xpath('Water_fractions')[0]
        path_scenarios = root.xpath('path_scenarios/Output')[0]


        self.n_scenarios = len(psi_soil_elems)
        # Extract single-value parameters
        self.water_fraction_apo = float(water_fractions.get("Apoplast"))  # Relative volumetric fraction of water in the apoplast
        self.water_fraction_sym = float(water_fractions.get("Symplast"))  # Relative volumetric fraction of water in the symplast

        # Initialize arrays for boundary conditions
        Psi_soil = np.zeros((2, self.n_scenarios))
        Os_soil = np.zeros((6, self.n_scenarios))
        Os_xyl = np.zeros((6, self.n_scenarios))
        self.C_flag = False  # Do we calculate solute stationary fluxes?

        # Load boundary condition scenarios
        self.scenarios = []

        for count in range(self.n_scenarios):
            # Check if we need to calculate solute stationary fluxes
            if not Os_xyl[4][count] == 0 and not Os_soil[4][count] == 0:
                self.C_flag = True
                print('Calculation of analytical solution for radial solute transport in cell walls')

            # Create scenario dictionary
            scenario = {
                'psi_soil_left': float(psi_soil_elems[count].get("pressure_left")),
                'psi_soil_right': float(psi_soil_elems[count].get("pressure_right")),
                'osmotic_left_soil': float(psi_soil_elems[count].get("osmotic_left")),
                'osmotic_right_soil': float(psi_soil_elems[count].get("osmotic_right")),
                'osmotic_symmetry_soil': float(psi_soil_elems[count].get("osmotic_symmetry")),
                'osmotic_shape_soil': float(psi_soil_elems[count].get("osmotic_shape")),
                'osmotic_diffusivity_soil': float(root.xpath('Psi_soil_range/osmotic_diffusivity')[0].get("value")),
                'osmotic_xyl': float(bc_xyl_elems[count].get("osmotic_xyl")),
                'osmotic_endo': float(bc_xyl_elems[count].get("osmotic_endo")),
                'osmotic_symmetry_xyl': float(bc_xyl_elems[count].get("osmotic_symmetry")),
                'osmotic_shape_xyl': float(bc_xyl_elems[count].get("osmotic_shape")),
                'osmotic_diffusivity_xyl': float(root.xpath('BC_xyl_range/osmotic_diffusivity')[0].get("value")),
                'pressure_xyl': float(bc_xyl_elems[count].get("pressure")) if bc_xyl_elems[count].get("pressure") else np.nan,
                'flow_xyl': float(bc_xyl_elems[count].get("flowrate")) if bc_xyl_elems[count].get("flowrate") else np.nan,
                'delta_p_xyl': float(bc_xyl_elems[count].get("deltaP")) if bc_xyl_elems[count].get("deltaP") else np.nan,
                'pressure_sieve': float(bc_sieve_elems[count].get("pressure")) if bc_sieve_elems[count].get("pressure") else np.nan,
                'flow_sieve': float(bc_sieve_elems[count].get("flowrate")) if bc_sieve_elems[count].get("flowrate") else np.nan,
                'delta_p_sieve': float(bc_sieve_elems[count].get("deltaP")) if bc_sieve_elems[count].get("deltaP") else np.nan,
                'osmotic_sieve': float(bc_sieve_elems[count].get("osmotic")) if bc_sieve_elems[count].get("osmotic") else np.nan,
                'psi_s_hetero': int(psi_cell_elems[count].get("s_hetero")),
                'psi_s_factor': float(psi_cell_elems[count].get("s_factor")),
                'psi_os_hetero': int(psi_cell_elems[count].get("Os_hetero")),
                'psi_os_cortex': float(psi_cell_elems[count].get("Os_cortex")),
                'elongation_midpoint_rate': float(elong_cell_elems[count].get("midpoint_rate")),
                'elongation_side_rate_difference': float(elong_cell_elems[count].get("side_rate_difference")),
                # Add other boundary conditions as needed
            }

            self.scenarios.append(scenario)

def parse_cellset_xml(filepath: str, im_scale: float) -> Dict[str, Any]:
    """
    Parse the cell set XML file to extract wall and cell information
        
    Parameters
    ----------
    filepath : str
        Path to the cellset XML file
    im_scale : float
        Image scale factor for coordinate conversion
            
    Returns
    -------
    Dict containing:
        - points: wall point coordinates
        - walls: wall connectivity
        - cells: cell definitions
        - cell_to_wall: mapping of cells to walls
    """
    tree = etree.parse(filepath)
    root = tree.getroot()
        
    return {
        'root': root,
        'points': root.xpath('walls/wall/points'),
        'walls': root.xpath('cells/cell/walls/wall'),
        'cells': root.xpath('cells/cell'),
        'cell_to_wall': root.xpath('cells/cell/walls'),
        'im_scale': im_scale
    }