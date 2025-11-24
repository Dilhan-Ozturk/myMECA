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
        self.n_wall_junction: int = 0
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
        self.wall_lengths: Optional[np.ndarray] = None 
        self.distance_wall_cell: Optional[np.ndarray] = None 
        self.junction_lengths: Dict[Any, Any] = {}
        self.junction_positions: Dict[Any, Any] = {}
        
        # Border identification
        self.border_walls: List[int] = []
        self.border_aerenchyma: List[int] = []
        self.border_junction: List[int] = []
        self.border_link: Optional[np.ndarray] = None 
        
        # Special cells
        self.xylem_cells: List[int] = []
        self.sieve_cells: List[int] = []
        self.xylem_walls: List[int] = []
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
    
    def create_wall_junction_nodes(self, geometry: GeometryData, n_dec_position: int = 6):
        points = self.cellset['points']
        im_scale = geometry.im_scale

        junction_ni = 0
        self.junction_to_wall = {}
        self.n_junction_to_wall = {}
        junction_list = {}
        for point_groups in points: #Loop on wall elements

            wall_id = int((point_groups.getparent().get)("id")) # wall_id records the current wall id number

            coords = []
            for point in point_groups:
                x = round(im_scale * float(point.get("x")), n_dec_position)
                y = round(im_scale * float(point.get("y")), n_dec_position)
                coords.append((x, y))
            # Store junction positions for this wall
            self.junction_positions[wall_id] = [
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

            # Add wall node
            self.graph.add_node(
                wall_id,
                indice=wall_id,
                type="apo",
                position=(round(mid_x, n_dec_position), round(mid_y, n_dec_position)),
                length=length
            )

            # Add junction node
            for coord in [coords[0], coords[-1]]: # First and last point as junctions
                pos_key = f"x{coord[0]}y{coord[1]}"

                if pos_key not in junction_list:
                    node_id = self.n_walls + junction_ni
                    self.graph.add_node(
                        node_id,
                        indice=node_id,
                        type="apo",
                        position=coord,
                        length=0
                    )
                    junction_list[pos_key] = node_id
                    self.junction_to_wall[node_id] = [wall_id]
                    self.n_junction_to_wall[node_id] = 1
                    junction_ni += 1 # New junction created 
                else:
                    junction_id = junction_list[pos_key]
                    self.junction_to_wall[junction_id].append(wall_id) # Several cell wall ID numbers can correspond to the same X Y coordinate where they meet
                    self.n_junction_to_wall[junction_id] += 1 # Count how many walls connect to this junction
        
        self.n_junctions = junction_ni
        self.n_wall_junction = self.n_walls + self.n_junctions

    def identify_border_walls_junctions(self):
        """Identify walls and junctions at the soil-root interface"""
        walls_loop = self.cellset['walls']
        cell_to_wall = self.cellset['cell_to_wall']
        self.wall_lengths = nx.get_node_attributes(self.graph,'length') #Nodes lengths (micrometers)

        # Initialize border tracking
        self.border_link = 2*np.ones((self.n_wall_junction, 1), dtype=int)

        # Count how many cells each wall is connected to
        for wall_elem in walls_loop:
            wall_id = int(wall_elem.get("id"))
            self.border_link[wall_id] -= 1
        
        for cell_group in cell_to_wall:
            cgroup = int(cell_group.getparent().get("group"))
            for wall_ref in cell_group:
                wall_id = int(wall_ref.get("id"))
                if wall_id < self.n_walls:
                    # Wall at soil interface (epidermis and single connection)
                    if self.border_link[wall_id] == 1 and cgroup == 2:
                        if wall_id not in self.border_walls:
                            self.border_walls.append(wall_id)
                    # Wall at aerenchyma surface
                    elif self.border_link[wall_id] == 1 and cgroup != 2:
                        if wall_id not in self.border_aerenchyma:
                            self.border_aerenchyma.append(wall_id)
        
        # Identify border junctions
        junction_id = 0
        for junction, wall in self.junction_to_wall.items():
            count=0
            length=0
            for wall_id in wall:
                if wall_id in self.border_walls:
                    count += 1
                    length += self.wall_lengths[wall_id] / 4.0
            if count == 2:
                self.border_junction.append(junction_id + self.n_walls)
                self.border_link[junction_id + self.n_walls] = 1  # Junction node at the interface with soil
                self.wall_lengths[junction_id + self.n_walls] = length
            else:
                self.border_link[junction_id + self.n_walls] = 0
            junction_id+=1
    
    def create_cell_nodes(self, geometry:GeometryData, contagion: Any = 0):
        """Create nodes for cells"""
        cell_to_wall = self.cellset['cell_to_wall']
        position = nx.get_node_attributes(self.graph,'position') #Nodes XY positions (micrometers)
        # Initialize tracking arrays
        self.intercellular_cells = list(geometry.intercellular_ids)
        self.passage_cells = list(geometry.passage_cell_ids)
        
        for cell_group in cell_to_wall:
            cell_id = int(cell_group.getparent().get("id"))
            cell_type = int(cell_group.getparent().get("group"))
            
            # Calculate cell center from wall position
            wall_positions = []
            for wall_ref in cell_group:
                wall_id = int(wall_ref.get("id"))
                if wall_id in position:
                    wall_positions.append(position[wall_id])
            
            if not wall_positions:
                continue
            
            center_x = np.mean([p[0] for p in wall_positions])
            center_y = np.mean([p[1] for p in wall_positions])
            
            node_id = self.n_walls + self.n_junctions + cell_id
            
            self.graph.add_node(
                node_id,
                indice=node_id,
                type="cell",
                position=(center_x, center_y),
                cgroup=cell_type
            )
            
            # Track special cell types
            if cell_type in [11, 23]:  # Phloem sieve
                self.sieve_cells.append(node_id)
            elif cell_type in [13, 19, 20]:  # Xylem
                self.xylem_cells.append(node_id)
                for cell in cell_group:
                    wall_id = int(cell.get("id"))
                    self.xylem_walls.append(wall_id)

            if contagion:
                for cell in cell_group:
                    wall_id = int(cell.get("id"))
                    cc_id = self.apo_cc[self.apo_zombie0.index(cell_id)]
                    if cell_id in self.apo_zombie0:
                        cc_id = self.apo_cc[self.apo_zombie0.index(cell_id)]
                    if wall_id not in self.apo_wall_zombies0:
                        self.apo_wall_zombies0.append(wall_id)
                        self.apo_wall_cc.append(cc_id)
                    if cell_id in self.apo_target and wall_id not in self.apo_wall_target:
                        self.apo_wall_target.append(wall_id)
                    if cell_id in self.apo_immune and wall_id not in self.apo_wall_immune:
                        self.apo_wall_immune.append(wall_id)
                
    def build_membrane_connections(self):
        """Build membrane connections between cells and walls"""
        cell_to_wall = self.cellset['cell_to_wall']
        self.distance_wall_cell = np.zeros((self.n_walls, 1))
        
        for cell_group in cell_to_wall:
            cell_id = int(cell_group.getparent().get("id"))
            cell_node_id = self.n_walls + self.n_junctions + cell_id
            
            for wall_ref in cell_group:
                wall_id = int(wall_ref.get("id"))
                
                if wall_id >= self.n_walls:
                    continue
                
                # Calculate distance and direction
                pos_c = self.positions[cell_node_id]
                pos_w = self.positions[wall_id]
                
                d_vec = np.array([pos_w[0] - pos_c[0], pos_w[1] - pos_c[1]])
                dist = np.linalg.norm(d_vec)
                
                if dist > 0:
                    d_vec = d_vec / dist
                
                self.distance_wall_cell[wall_id] += dist

                self.graph.add_edge(
                    cell_node_id,
                    wall_id,
                    path='membrane',
                    length=self.wall_lengths[wall_id],
                    dist=dist,
                    d_vec=d_vec
                )
                self.n_membrane += 1 


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
    
    def
        
