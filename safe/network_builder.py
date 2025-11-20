"""
Network builder for MECHA
Constructs the hydraulic network graph from cell data
"""

import networkx as nx
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from lxml import etree

from general_loader import GeneralData
from geometry_loader import GeometryData
from hormones_loader import HormonesData
from cellset_parser import parse_cellset

class NetworkBuilder:
    """Builds the hydraulic network graph from XML cell data"""
    def __init__(self):

        self.cellset: Dict[str, Any] = {}

        self.graph: nx.Graph = nx.Graph()

        # Network dimensions
        self.n_walls: int = 0
        self.n_junctions: int = 0
        self.n_cells: int = 0
        self.n_total: int = 0
        self.n_membrane: int = 0
        self.n_membrane_from_epi: int = 0
        
        # Cell and wall properties
        self.cell_areas: Optional[np.ndarray] = None 
        self.cell_perimeters: Optional[np.ndarray] = None 
        self.cell_ranks: Optional[np.ndarray] = None 
        self.cell_groups: Optional[np.ndarray] = None 
        
        # Spatial data
        self.junction_positions: Dict[Any, Any] = {}
        self.wall_lengths: Dict[Any, Any] = {}
        self.distance_wall_cell: Dict[Any, Any] = {}
        self.junction_lengths: Dict[Any, Any] = {}
        self.wall_positions_junctions: Dict[Any, Any] = {}
        
        # Border identification
        self.border_walls: List[int] = []
        self.border_aerenchyma: List[int] = []
        self.border_junctions: List[int] = []
        self.border_link: Optional[np.ndarray] = None 
        
        # Special cells
        self.xylem_cells: List[int] = []
        self.sieve_cells: List[int] = []
        self.proto_sieve_cells: List[int] = []
        self.intercellular_cells: List[int] = []
        self.passage_cells: List[int] = []
        self.xylem_80_percentile_distance: float = 0.0
        self.n_sieve: int = 0
        self.n_protosieve: int = 0
        
        # Connectivity
        self.cell_connections: Optional[np.ndarray] = None 
        self.wall_to_cell: Optional[np.ndarray] = None 
        self.junction_to_wall: Dict[Any, Any] = {}
        self.n_junction_to_wall: Dict[Any, Any] = {} 
        
        # Gravity center and geometry
        self.x_grav: float = 0.0
        self.y_grav: float = 0.0
        self.x_min: float = np.inf
        self.x_max: float = 0.0
        
        # Layer discretization
        self.layer_dist: Optional[np.ndarray] = None 
        self.n_layer: Optional[np.ndarray] = None 
        self.rank_to_row: Optional[np.ndarray] = None 
        self.r_discret: Optional[np.ndarray] = None 

        # Rank 
        self.stele_connec_rank: int = 0
        self.outercortex_connec_rank: int = 0
        
        # Lists for special cells
        self.xylem_distance: List[int] = []
        self.protosieve_list: List[int] = []

        # Distance computation
        self.distance_max_cortex: float = 0.0
        self.distance_min_cortex = np.inf
        self.distance_avg_epi: float = 0.0
        self.distance_to_center: float = 0.0
        self.perimeter: float = 0.0

        # Cell surface computation
        self.len_outer_cortex: float = 0.0
        self.len_cortex_cortex: float = 0.0
        self.len_cortex_endo: float = 0.0
        self.cross_section_outer_cortex: float = 0.0
        self.cross_section_cortex_cortex: float = 0.0
        self.cross_section_cortex_endo: float = 0.0
        self.plasmodesmata_indice: List[int] = []


        # list of contagion parameters
        self.apo_wall_zombies0: List[int] = []
        self.apo_wall_cc: List[int] = []
        self.apo_wall_target: List[int] = []
        self.apo_wall_immune: List[int] = []


    def build_network(self, general, geometry, hormones, cellset_file):
        """Main method to build network from XML data"""
        self.cellset = parse_cellset(cellset_file)
        self.n_walls = len(self.cellset['points'])
        self.n_cells = len(self.cellset['cells'])

        # print('  Creating wall nodes...')
        # self._create_wall_nodes(cellset)   
        
        # print('  Identifying border nodes...')
        # self._identify_border_nodes(cellset)
        
        # print('  Creating junction nodes...')
        # self._create_junction_nodes(cellset)
        
        # print('  Creating cell nodes...')
        # self._create_cell_nodes(cellset, contagion=self.config.apo_contagion)
        
        # print('  Building connectivity...')
        # self._build_membrane_connections(cellset) # have to be initialize first for wall connection
        # self._build_wall_connections()
        # self._build_plasmodesmata_connections(cellset)
        
        # print('  Computing cell properties...')
        # self._compute_cell_properties(cellset)
        # self._compute_gravity_center()
        # self._rank_cells()
        # self._create_layer_discretization()
        
        # self.n_total = self.n_walls + self.n_junctions + self.n_cells
        # self.indice = nx.get_node_attributes(self.graph,'indice')
        # self.position = nx.get_node_attributes(self.graph,'position')
        # self.length = nx.get_node_attributes(self.graph,'length')

        # self._compute_distance(cellset)
        # self._compute_cell_surface()

        # print(f'  Network: {self.n_walls} walls, {self.n_junctions} junctions, '
        #       f'{self.n_cells} cells')
    
    def create_wall_junction_nodes(self, geometry: GeometryData):
        points = self.cellset['points']
        im_scale = geometry.im_scale

        junction_id = 0
        self.junction_to_wall = {}
        self.n_junction_to_wall = {}

        for point_groups in points: #Loop on wall elements

            wall_id = int((point_groups.getparent().get)("id")) # wall_id records the current wall id number

            coords = []
            for point in point_groups:
                x = im_scale * float(point.get("x"))
                y = im_scale * float(point.get("y"))
                coords.append((x, y))
            print(coords)
            # Store junction positions for this wall
            self.wall_positions_junctions[wall_id] = [
                coords[0][0], coords[0][1],  # First junction
                coords[-1][0], coords[-1][1]  # Last junction
            ]

            if len(coords) < 2: # Skip if there are not enough points to define a wall
                continue

            # Calculate wall length
            length = sum(
                np.hypot(coords[i+1][0]-coords[i][0], coords[i+1][1]-coords[i][1])
                for i in range(len(coords)-1)
            )

            # Find midpoint
            mid_x, mid_y = self._find_wall_midpoint(coords, length)

            # Track min/max for later interpolation
            self.x_min = min(self.x_min, mid_x)
            self.x_max = max(self.x_max, mid_x)
            print(mid_x)
            print(mid_y)
            # Add wall node
            self.graph.add_node(
                wall_id,
                indice=wall_id,
                type="apo",
                position=(mid_x, mid_y),
                length=length
            )

            # Add junction node
            for coord in [coords[0], coords[-1]]: # First and last point as junctions
                pos_key = f"x{coord[0]}y{coord[1]}"

                if pos_key not in self.junction_positions:
                    node_id = self.n_walls + junction_id
                    self.graph.add_node(
                        node_id,
                        indice=node_id,
                        type="apo",
                        position=coord,
                        length=0
                    )
                    self.junction_positions[pos_key] = node_id
                    self.junction_to_wall[node_id] = [wall_id]
                    self.n_junction_to_wall[node_id] = 1
                    junction_id += 1 # New junction created 
                else:
                    junc_id = self.junction_positions[pos_key]
                    self.junction_to_wall[junc_id].append(wall_id) # Several cell wall ID numbers can correspond to the same X Y coordinate where they meet
                    self.n_junction_to_wall[junc_id] += 1 # Count how many walls connect to this junction
        
        self.n_junctions = junction_id

    def _find_wall_midpoint(self, coords: List[Tuple], total_length: float) -> Tuple[float, float]:
        """Find the midpoint along a wall defined by coordinates"""
        target_length = total_length / 2.0
        cumulative = 0.0
        
        for i in range(len(coords) - 1):
            segment_length = np.hypot(
                coords[i+1][0] - coords[i][0],
                coords[i+1][1] - coords[i][1]
            )
            
            if cumulative + segment_length >= target_length:
                # Midpoint is in this segment
                remaining = target_length - cumulative
                t = remaining / segment_length if segment_length > 0 else 0
                
                mid_x = coords[i][0] + t * (coords[i+1][0] - coords[i][0])
                mid_y = coords[i][1] + t * (coords[i+1][1] - coords[i][1])
                return mid_x, mid_y
            
            cumulative += segment_length
        
        # Fallback to last point
        return coords[-1]
        
