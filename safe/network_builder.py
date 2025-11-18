"""
Network builder for MECHA
Constructs the hydraulic network graph from cell data
"""

import networkx as nx
import numpy as np
from typing import Dict, List, Tuple, Any
from lxml import etree

class NetworkBuilder:
    """Builds the hydraulic network graph from XML cell data"""
    
    def __init__(self, cellset: Dict):
        self.config = config
        self.graph = nx.Graph()

        # Network dimensions
        self.n_walls = 0
        self.n_junctions = 0
        self.n_cells = 0
        self.n_total = 0
        self.n_membrane = 0
        self.n_membrane_from_epi = 0
        
        # Cell and wall properties
        self.cell_areas = None
        self.cell_perimeters = None
        self.cell_ranks = None
        self.cell_groups = None
        
        # Spatial data
        self.positions = {}
        self.junction_positions = {}
        self.wall_lengths = {}
        self.distance_wall_cell = {}
        self.junction_lengths = {}
        self.wall_positions_junctions = {}
        
        # Border identification
        self.border_walls = []
        self.border_aerenchyma = []
        self.border_junctions = []
        self.border_link = None 
        
        # Special cells
        self.xylem_cells = []
        self.sieve_cells = []
        self.proto_sieve_cells = []
        self.intercellular_cells = []
        self.passage_cells = []
        self.xylem_80_percentile_distance = 0
        self.n_sieve = 0
        self.n_protosieve = 0
        
        # Connectivity
        self.cell_connections = None
        self.wall_to_cell = None
        self.junction_to_wall = {}
        self.n_junction_to_wall = {} 
        
        # Gravity center and geometry
        self.x_grav = 0.0
        self.y_grav = 0.0
        self.x_min = np.inf
        self.x_max = 0.0
        
        # Layer discretization
        self.layer_dist = None
        self.n_layer = None
        self.rank_to_row = None
        self.r_discret = None

        # Rank 
        self.stele_connec_rank = 0
        self.outercortex_connec_rank=0
        
        # Lists for special cells
        self.xylem_distance = []
        self.protosieve_list = []

        # Distance computation
        self.distance_max_cortex = 0
        self.distance_min_cortex = np.inf
        self.distance_avg_epi = 0
        self.distance_to_center = {}
        self.perimeter = 0

        # Cell surface computation
        self.len_outer_cortex = 0
        self.len_cortex_cortex = 0
        self.len_cortex_endo = 0
        self.cross_section_outer_cortex = 0
        self.cross_section_cortex_cortex = 0
        self.cross_section_cortex_endo = 0
        self.plasmodesmata_indice = []


        # list of contagion parameters
        self.apo_wall_zombies0 = []
        self.apo_wall_cc = []
        self.apo_wall_target = []
        self.apo_wall_immune = []
        
    def build_from_xml(self, cellset: Dict):
        """Main method to build network from XML data"""
        cellset = parse_cellset_xml(self.config.cellset_file, self.config.im_scale)

        print('  Creating wall nodes...')
        self._create_wall_nodes(cellset)   
        
        print('  Identifying border nodes...')
        self._identify_border_nodes(cellset)
        
        print('  Creating junction nodes...')
        self._create_junction_nodes(cellset)
        
        print('  Creating cell nodes...')
        self._create_cell_nodes(cellset, contagion=self.config.apo_contagion)
        
        print('  Building connectivity...')
        self._build_membrane_connections(cellset) # have to be initialize first for wall connection
        self._build_wall_connections()
        self._build_plasmodesmata_connections(cellset)
        
        print(f'  Network: {self.n_walls} walls, {self.n_junctions} junctions, '
              f'{self.n_cells} cells')
    
    def _create_wall_nodes(self, cellset: Dict):
        """Create nodes for cell walls"""
        points = cellset['points']
        self.n_walls = len(points)
        im_scale = cellset['im_scale']
        
        for wall_id, point_group in enumerate(points):
            # Get wall coordinates
            coords = []
            for point in point_group:
                x = im_scale * float(point.get("x"))
                y = im_scale * float(point.get("y"))
                coords.append((x, y))
            
            if len(coords) < 2: # Skip if there are not enough points to define a wall
                continue 
            
            # Store junction positions for this wall
            self.wall_positions_junctions[wall_id] = [
                coords[0][0], coords[0][1],  # First junction
                coords[-1][0], coords[-1][1]  # Last junction
            ]
            
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
                position=(mid_x, mid_y),
                length=length
            )
            
            self.positions[wall_id] = (mid_x, mid_y)
            self.wall_lengths[wall_id] = length
  
    def _identify_border_nodes(self, cellset: Dict):
        """Identify walls and junctions at the soil-root interface"""
        walls_loop = cellset['walls']
        cell_to_wall = cellset['cell_to_wall']

        # Initialize border tracking
        self.border_link = np.ones((self.n_walls, 1), dtype=int)

        # Count how many cells each wall is connected to
        wall_cell_count = {}
        for wall_elem in walls_loop:
            wall_id = int(wall_elem.get("id"))
            if wall_id not in wall_cell_count:
                wall_cell_count[wall_id] = 0
            wall_cell_count[wall_id] += 1
        
        for cell_group in cell_to_wall:
            cgroup = int(cell_group.getparent().get("group"))
            for wall_ref in cell_group:
                wall_id = int(wall_ref.get("id"))
                if wall_id < self.n_walls:
                    self.border_link[wall_id] = wall_cell_count.get(wall_id, 0)
                    
                    # Wall at soil interface (epidermis and single connection)
                    if self.border_link[wall_id] == 1 and cgroup == 2:
                        if wall_id not in self.border_walls:
                            self.border_walls.append(wall_id)
                    # Wall at aerenchyma surface
                    elif self.border_link[wall_id] == 1 and cgroup != 2:
                        if wall_id not in self.border_aerenchyma:
                            self.border_aerenchyma.append(wall_id)
    
    def _create_junction_nodes(self, cellset: Dict):
        """Create nodes at wall junctions"""
        points = cellset['points']
        im_scale = cellset['im_scale']
        # Perform the same logic as walls to find junctions

        junction_id = 0
        self.junction_to_wall = {}
        self.n_junction_to_wall = {}
        for wall_id, point_group in enumerate(points):
            coords = []
            for point in point_group:
                x = im_scale * float(point.get("x"))
                y = im_scale * float(point.get("y"))
                coords.append((x, y))

            if len(coords) < 2: # Skip if there are not enough points to define a wall
                continue

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
                    self.positions[node_id] = coord
                    self.junction_to_wall[node_id] = [wall_id]
                    self.n_junction_to_wall[node_id] = 1
                    junction_id += 1 # New junction created 
                else:
                    junc_id = self.junction_positions[pos_key]
                    self.junction_to_wall[junc_id].append(wall_id) # Several cell wall ID numbers can correspond to the same X Y coordinate where they meet
                    self.n_junction_to_wall[junc_id] += 1 # Count how many walls connect to this junction
        
        self.n_junctions = junction_id # Total number of unique junctions created
        required_size = self.n_walls + self.n_junctions
        
        if self.border_link.shape[0] < required_size:
            new_border_link = np.zeros((required_size, 1), dtype=int)
            new_border_link[:self.border_link.shape[0]] = self.border_link
            self.border_link = new_border_link

        self.border_junctions = []
        for junc_id, wall_ids in self.junction_to_wall.items():
            border_count = sum(1 for wall_id in wall_ids if wall_id in self.border_walls)
            if border_count == 2:
                self.border_junctions.append(junc_id)
                self.border_link[junc_id] = 1 
                # Calculate junction length for border conductance
                total_length = sum(self.wall_lengths.get(wall_id, 0) for wall_id in wall_ids
                                if wall_id in self.border_walls) / 4.0
                self.junction_lengths[junc_id] = total_length
            else:
                self.border_link[junc_id] = 0 
    
    def _create_cell_nodes(self, cellset: Dict, contagion: Any = 0):
        """Create nodes for cells"""
        cell_to_wall = cellset['cell_to_wall']
        self.n_cells = len(cell_to_wall)
        
        # Initialize tracking arrays
        self.intercellular_cells = list(self.config.intercellular_ids)
        self.passage_cells = list(self.config.passage_cell_ids)
        
        for cell_group in cell_to_wall:
            cell_id = int(cell_group.getparent().get("id"))
            cell_type = int(cell_group.getparent().get("group"))
            
            # Normalize cell types
            if cell_type == 19 or cell_type == 20:  # Proto/Meta-xylem
                cell_type = 13
            elif cell_type == 21:  # Xylem pole pericycle
                cell_type = 16
            elif cell_type == 23:  # Phloem
                cell_type = 11
            elif cell_type == 26:  # Companion cell
                cell_type = 12
            
            # Calculate cell center from wall positions
            wall_positions = []
            for wall_ref in cell_group:
                wall_id = int(wall_ref.get("id"))
                if wall_id in self.positions:
                    wall_positions.append(self.positions[wall_id])
            
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
            
            self.positions[node_id] = (center_x, center_y)
            
            # Track special cell types
            if cell_type in [11, 23]:  # Phloem sieve
                self.sieve_cells.append(node_id)
            elif cell_type in [13, 19, 20]:  # Xylem
                self.xylem_cells.append(node_id)

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
                
    def _compute_cell_properties(self, cellset: Dict):
        """Compute cell areas and perimeters using shoelace formula"""
        cell_to_wall = cellset['cell_to_wall']
        self.cell_areas = np.zeros(self.n_cells)
        self.cell_perimeters = np.zeros(self.n_cells)
        
        for cell_group in cell_to_wall:
            cell_id = int(cell_group.getparent().get("id"))
            
            # Collect ordered wall positions for area calculation
            wall_coords = []
            for wall_ref in cell_group:
                wall_id = int(wall_ref.get("id"))
                if wall_id in self.wall_positions_junctions:
                    # Add junction positions for accurate area calculation
                    junc_pos = self.wall_positions_junctions[wall_id]
                    wall_coords.append((junc_pos[0], junc_pos[1]))
                
                # Add to perimeter
                if wall_id in self.wall_lengths:
                    self.cell_perimeters[cell_id] += self.wall_lengths[wall_id]
            
            # Calculate area using shoelace formula
            if len(wall_coords) >= 3:
                area = 0.0
                n = len(wall_coords)
                for i in range(n):
                    j = (i + 1) % n
                    area += wall_coords[i][0] * wall_coords[j][1]
                    area -= wall_coords[j][0] * wall_coords[i][1]
                self.cell_areas[cell_id] = abs(area) / 2.0
    
    def _rank_cells(self):
        """
        Assign ranks to cells based on tissue type and connectivity.
        
        Ranking system:
        - 1: Exodermis
        - 2: Epidermis
        - 3: Endodermis
        - 4: Cortex (updated to 40-49 based on layer)
        - 5: Stele (updated to 50-61 based on layer)
        - 11: Phloem sieve tube
        - 12: Companion cell
        - 13: Xylem
        - 16: Pericycle
        - 40-44: Cortex layers from endodermis outward
        - 45-49: Cortex layers from exodermis inward
        - 50-60: Stele layers from pericycle inward
        - 61: Central stele
        """
        self.cell_ranks = np.zeros(self.n_cells, dtype=int)
        self.layer_dist = np.zeros(62)
        self.n_layer = np.zeros(62, dtype=int)
        
        # First pass: Basic cell type assignment
        for node_id in range(self.n_walls + self.n_junctions, 
                            self.n_walls + self.n_junctions + self.n_cells):
            cell_id = node_id - self.n_walls - self.n_junctions
            cgroup = self.graph.nodes[node_id].get('cgroup', 0)
            
            # Normalize cell groups to standard types
            if cgroup in [19, 20]:  # Proto- and Meta-xylem
                cgroup = 13
            elif cgroup == 21:  # Xylem pole pericycle
                cgroup = 16
            elif cgroup == 23:  # Phloem
                cgroup = 11
            elif cgroup == 26:  # Companion cell
                cgroup = 12
            
            self.cell_ranks[cell_id] = cgroup
            
            # Calculate distance from gravity center
            pos = self.positions[node_id]
            dist = np.hypot(pos[0] - self.x_grav, pos[1] - self.y_grav)
            
            self.layer_dist[cgroup] += dist
            self.n_layer[cgroup] += 1
            
            # Track xylem distances
            if cgroup == 13:
                self.xylem_distance.append(dist)
        
        # Calculate xylem 80th percentile distance
        self.xylem_80_percentile_distance = np.percentile(self.xylem_distance, 80) if self.xylem_distance else 0
        
        # Determine connection ranks based on tissue presence
        if self.n_layer[16] == 0:  # No pericycle
            self.stele_connec_rank = 3  # Endodermis connects to stele
        else:
            self.stele_connec_rank = 16  # Pericycle connects to stele

        if self.n_layer[1] == 0:  # No exodermis
            self.outercortex_connec_rank = 2  # Epidermis connects to cortex
        else:
            self.outercortex_connec_rank = 1  # Exodermis connects to cortex
        
        # Second pass: Initial layer assignment
        for cell_id in range(self.n_cells):
            node_id = self.n_walls + self.n_junctions + cell_id
            celltype = self.cell_ranks[cell_id]
            pos = self.positions[node_id]
            
            # Get connected cells
            connected_ranks = self._get_connected_cell_ranks(cell_id)
            
            if celltype == 4:  # Cortex
                if 3 in connected_ranks:  # Connected to endodermis
                    self.cell_ranks[cell_id] = 40
                    dist = np.hypot(pos[0] - self.x_grav, pos[1] - self.y_grav)
                    self.layer_dist[40] += dist
                    self.n_layer[40] += 1
                    
                    # Check for intercellular spaces
                    if self._get_cell_perimeter(cell_id) < self.config.interc_perims[0]:
                        self.config.intercellular_ids.append(cell_id)
                        
                elif self.outercortex_connec_rank in connected_ranks:  # Connected to outer layer
                    self.cell_ranks[cell_id] = 49
                    dist = np.hypot(pos[0] - self.x_grav, pos[1] - self.y_grav)
                    self.layer_dist[49] += dist
                    self.n_layer[49] += 1
                    
                    if self._get_cell_perimeter(cell_id) < self.config.interc_perims[4]:
                        self.config.intercellular_ids.append(cell_id)
            
            elif celltype in [5, 11, 12, 13]:  # Stele tissues
                if self.stele_connec_rank in connected_ranks:  # Connected to pericycle
                    self.cell_ranks[cell_id] = 50
                    dist = np.hypot(pos[0] - self.x_grav, pos[1] - self.y_grav)
                    self.layer_dist[50] += dist
                    self.n_layer[50] += 1
                    
                    # Track protophloem
                    cgroup = self.graph.nodes[node_id].get('cgroup', 0)
                    if cgroup in [11, 23]:
                        self.protosieve_list.append(node_id)
        
        # Iterative pass: Refine layer rankings
        for iteration in range(12):
            for cell_id in range(self.n_cells):
                node_id = self.n_walls + self.n_junctions + cell_id
                celltype = self.cell_ranks[cell_id]
                pos = self.positions[node_id]
                
                connected_ranks = self._get_connected_cell_ranks(cell_id)
                
                # Cortex layers (up to 4 layers from each side)
                if celltype == 4 and iteration < 4:
                    # Inward from endodermis
                    if (40 + iteration) in connected_ranks:
                        self.cell_ranks[cell_id] = 41 + iteration
                        dist = np.hypot(pos[0] - self.x_grav, pos[1] - self.y_grav)
                        self.layer_dist[41 + iteration] += dist
                        self.n_layer[41 + iteration] += 1
                        
                        # Intercellular space detection
                        perimeter = self._get_cell_perimeter(cell_id)
                        if iteration == 0 and perimeter < self.config.interc_perims[1]:
                            self.config.intercellular_ids.append(cell_id)
                        elif iteration == 1 and perimeter < self.config.interc_perims[2]:
                            self.config.intercellular_ids.append(cell_id)
                        elif iteration == 2 and perimeter < self.config.interc_perims[3]:
                            self.config.intercellular_ids.append(cell_id)
                        elif iteration > 2 and perimeter < self.config.interc_perims[4]:
                            self.config.intercellular_ids.append(cell_id)
                    
                    # Outward from exodermis
                    elif (49 - iteration) in connected_ranks:
                        self.cell_ranks[cell_id] = 48 - iteration
                        dist = np.hypot(pos[0] - self.x_grav, pos[1] - self.y_grav)
                        self.layer_dist[48 - iteration] += dist
                        self.n_layer[48 - iteration] += 1
                        
                        if self._get_cell_perimeter(cell_id) < self.config.interc_perims[4]:
                            self.config.intercellular_ids.append(cell_id)
                
                # Stele layers (up to 10 layers from pericycle)
                elif celltype in [5, 11, 12, 13]:
                    if iteration < 10:
                        if (50 + iteration) in connected_ranks:
                            self.cell_ranks[cell_id] = 51 + iteration
                            dist = np.hypot(pos[0] - self.x_grav, pos[1] - self.y_grav)
                            self.layer_dist[51 + iteration] += dist
                            self.n_layer[51 + iteration] += 1
                    else:
                        # Central stele (beyond 10 layers)
                        self.cell_ranks[cell_id] = 61
                        dist = np.hypot(pos[0] - self.x_grav, pos[1] - self.y_grav)
                        self.layer_dist[61] += dist
                        self.n_layer[61] += 1
        
        # Calculate average layer distances
        for i in range(62):
            if self.n_layer[i] > 0:
                self.layer_dist[i] /= self.n_layer[i]
        
        # Store counts
        self.n_sieve = len(self.config.sieve_ids) if hasattr(self.config, 'sieve_ids') else 0
        self.n_protosieve = len(self.protosieve_list)

    def _get_connected_cell_ranks(self, cell_id):
        """Get ranks of all cells connected to this cell"""
        node_id = self.n_walls + self.n_junctions + cell_id
        connected_ranks = []
        
        for neighbor in self.graph.neighbors(node_id):
            # Check if neighbor is a cell (not wall or junction)
            if neighbor >= self.n_walls + self.n_junctions:
                neighbor_cell_id = neighbor - self.n_walls - self.n_junctions
                if neighbor_cell_id < self.n_cells:
                    connected_ranks.append(self.cell_ranks[neighbor_cell_id])
        
        return connected_ranks

    def _get_cell_perimeter(self, cell_id):
        """Get perimeter of a cell"""
        # This should be computed during cell property calculation
        # For now, return from stored data if available
        if hasattr(self, 'cell_perimeters') and cell_id < len(self.cell_perimeters):
            return self.cell_perimeters[cell_id]
        return 0.0

    def _create_layer_discretization(self):
        """
        Create radial discretization for layer-wise hydraulic analysis.
        
        This maps cell ranks to computational rows and creates layer groups
        for radial hydraulic conductivity calculations.
        
        Layer structure (from center outward):
        - Stele layers (61 → 50)
        - Pericycle (16)
        - Endodermis (3) - 4 rows for inner/outer + passage cells
        - Cortex layers (40 → 49)
        - Exodermis (1) - 2 rows
        - Epidermis (2) - 1 row
        """
        n_layers = 0
        self.rank_to_row = np.full(62, np.nan)
        self.r_discret = [0]  # Will store layer group sizes
        
        # STELE LAYERS (innermost)
        stele_start = n_layers
        for rank in range(61, 49, -1):  # 61 down to 50
            if self.n_layer[rank] > 0:
                self.rank_to_row[rank] = n_layers
                n_layers += 1
        stele_rows = n_layers - stele_start
        
        # PERICYCLE
        if self.n_layer[16] > 0:
            self.rank_to_row[16] = n_layers
            n_layers += 1
            self.r_discret.append(stele_rows + 1)  # Stele + pericycle
        else:
            if stele_rows > 0:
                self.r_discret.append(stele_rows)
        
        # ENDODERMIS (4 computational rows)
        # 2 row for inner and outer
        # 2 rows for passage cells
        endo_start = n_layers
        if self.n_layer[3] > 0:
            self.rank_to_row[3] = n_layers
            n_layers += 4  
            self.r_discret.append(4)
        
        # CORTEX LAYERS
        # Inward layers from endodermis (40-44)
        cortex_start = n_layers
        for rank in range(40, 45):
            if self.n_layer[rank] > 0:
                self.rank_to_row[rank] = n_layers
                n_layers += 1
        
        # Outward layers from exodermis (49-45)
        for rank in range(49, 44, -1):
            if self.n_layer[rank] > 0:
                self.rank_to_row[rank] = n_layers
                n_layers += 1
        
        cortex_rows = n_layers - cortex_start
        if cortex_rows > 0:
            self.r_discret.append(cortex_rows)
        
        # EXODERMIS
        if self.n_layer[1] > 0:
            self.rank_to_row[1] = n_layers
            n_layers += 2  # Inner and outer walls
            self.r_discret.append(2)
        
        # EPIDERMIS (1 row)
        if self.n_layer[2] > 0:
            self.rank_to_row[2] = n_layers
            n_layers += 1
            self.r_discret.append(1)
        
        # Finalize
        self.r_discret[0] = n_layers  # Total number of rows
        self.r_discret = np.array(self.r_discret, dtype=int)
        
        print(f'  Layer discretization: {n_layers} computational rows')
        print(f'  Layer groups: {self.r_discret[1:]}')

    def get_layer_row(self, cell_id):
        """
        Get the computational row index for a cell.
        
        Args:
            cell_id: Cell identifier
            
        Returns:
            Row index in the computational matrix, or None if not found
        """
        if cell_id < len(self.cell_ranks):
            rank = self.cell_ranks[cell_id]
            row = self.rank_to_row[int(rank)]
            return int(row) if not np.isnan(row) else None
        return None

    def get_layer_info(self):
        """
        Get summary of layer discretization.
        
        Returns:
            Dictionary with layer information
        """
        return {
            'total_rows': int(self.r_discret[0]),
            'stele_pericycle': int(self.r_discret[1]) if len(self.r_discret) > 1 else 0,
            'endodermis': int(self.r_discret[2]) if len(self.r_discret) > 2 else 0,
            'cortex': int(self.r_discret[3]) if len(self.r_discret) > 3 else 0,
            'exodermis': int(self.r_discret[4]) if len(self.r_discret) > 4 else 0,
            'epidermis': int(self.r_discret[5]) if len(self.r_discret) > 5 else 0,
            'rank_to_row': self.rank_to_row
        }
 
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
    
    def _build_wall_connections(self):
        """Build connections between walls and junctions"""
        for junction_id, wall_ids in self.junction_to_wall.items():
            lat_dist = 0.0
            for wall_id in wall_ids:
                if wall_id >= self.n_walls:
                    continue
                
                # Calculate direction vector
                pos_j = self.positions[junction_id]
                pos_w = self.positions[wall_id]
                
                d_vec = np.array([pos_w[0] - pos_j[0], pos_w[1] - pos_j[1]])
                dist = np.linalg.norm(d_vec)
                
                if dist > 0:
                    d_vec = d_vec / dist

                self.graph.add_edge(
                    junction_id,
                    wall_id,
                    path='wall',
                    length=self.wall_lengths[wall_id] / 2,
                    lat_dist=self.distance_wall_cell[wall_id].item(),
                    d_vec=d_vec,
                    dist_wall=dist
                )

    def _build_membrane_connections(self, cellset: Dict):
        """Build membrane connections between cells and walls"""
        cell_to_wall = cellset['cell_to_wall']
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
    
    def _build_plasmodesmata_connections(self, cellset: Dict):
        """Build plasmodesmata connections between cells"""
        walls_list = cellset['walls']
        
        # Build wall-to-cells mapping
        wall_to_cells = {}
        for wall_elem in walls_list:
            wall_id = int(wall_elem.get("id"))
            cell_id = int(wall_elem.getparent().getparent().get("id"))
            
            if wall_id not in wall_to_cells:
                wall_to_cells[wall_id] = []
            wall_to_cells[wall_id].append(cell_id)
            
        # Connect cells that share walls
        for wall_id, cell_id in wall_to_cells.items():
            if len(cell_id) == 2:
                cell1_node = self.n_walls + self.n_junctions + cell_id[0]
                cell2_node = self.n_walls + self.n_junctions + cell_id[1]
                
                pos1 = self.positions[cell1_node]
                pos2 = self.positions[cell2_node]
                
                d_vec = np.array([pos2[0] - pos1[0], pos2[1] - pos1[1]])
                dist = np.linalg.norm(d_vec)
                
                if dist > 0:
                    d_vec = d_vec / dist
                
                self.graph.add_edge(
                    cell1_node,
                    cell2_node,
                    path='plasmodesmata',
                    length=self.wall_lengths[wall_id] if wall_id in self.wall_lengths else 0,
                    d_vec=d_vec
                )

    def _compute_gravity_center(self):
        """Compute gravity center of endodermis cells"""
        x_sum = 0.0
        y_sum = 0.0
        count = 0
        
        for node in self.graph.nodes():
            if self.graph.nodes[node].get('type') == 'cell':
                if self.graph.nodes[node].get('cgroup') == 3:  # Endodermis
                    pos = self.positions[node]
                    x_sum += pos[0]
                    y_sum += pos[1]
                    count += 1
        
        if count > 0:
            self.x_grav = x_sum / count
            self.y_grav = y_sum / count
    
    def get_cell_node_id(self, cell_id: int) -> int:
        """Convert cell ID to graph node ID"""
        return self.n_walls + self.n_junctions + cell_id
    
    def get_border_nodes(self) -> Tuple[List[int], List[int]]:
        """Identify nodes at the root-soil interface"""
        # Simplified: walls with only one connected cell
        border_walls = []
        border_junctions = []
        
        for node in range(self.n_walls):
            # Count cell neighbors
            cell_neighbors = sum(
                1 for n in self.graph.neighbors(node)
                if self.graph.nodes[n].get('type') == 'cell'
            )
            
            if cell_neighbors == 1:
                border_walls.append(node)
        
        return border_walls, border_junctions
    
    def _compute_distance(self, cellset: Dict):
        """Compute distance"""
        cell_to_wall = cellset['cell_to_wall']

        self.distance_to_center = np.zeros((self.n_walls,1))
        t = 0
        for cell_group in cell_to_wall:
            cell_id = int(cell_group.getparent().get("id")) #Cell ID number
            node_id = self.n_walls + self.n_junctions + cell_id
            for wall_ref in cell_group:
                wall_id = int(wall_ref.get("id"))
                self.distance_to_center[wall_id] = np.sqrt(
                    (self.position[wall_id][0]-self.x_grav) ** 2 + (self.position[wall_id][1]-self.y_grav) ** 2
                    )
                if self.graph.nodes[node_id]['cgroup'] == 4: #Cortex
                    self.distance_max_cortex = max(
                        self.distance_max_cortex,
                        self.distance_to_center[wall_id]
                        )
                    self.distance_min_cortex = min(
                        self.distance_min_cortex,
                        self.distance_to_center[wall_id]
                        )
                elif self.graph.nodes[node_id]['cgroup']==2: #Epidermis
                    self.distance_avg_epi += self.distance_to_center[wall_id]
                    t += 1.0
        self.distance_avg_epi /=t #Last step of averaging (note that we take both inner and outer membranes into account in the averaging)
        self.perimeter = 2*np.pi*self.distance_avg_epi*1.0E-04 #(cm)

    def _compute_cell_surface(self):
        """Calculate cell surfaces at tissue interfaces"""
        indice = self.indice
        
        # Initialize counters
        self.len_outer_cortex = 0
        self.len_cortex_cortex = 0
        self.len_cortex_endo = 0
        self.cross_section_outer_cortex = 0
        self.cross_section_cortex_cortex = 0
        self.cross_section_cortex_endo = 0
        self.plasmodesmata_indice = []
        
        for node, edges in self.graph.adjacency():
            i = indice[node]
            
            # Skip walls and junctions
            if i < self.n_walls + self.n_junctions:
                continue
                
            node_group = self.graph.nodes[i]['cgroup']
            
            # Handle specific cell groups (16, 21)
            if node_group in [16, 21]:
                for neighboor, eattr in edges.items():
                    if eattr['path'] == "plasmodesmata" and self.graph.nodes[indice[neighboor]]['cgroup'] in [11, 23]:
                        self.plasmodesmata_indice.append(i - (self.n_walls + self.n_junctions))
                continue
            
            # Handle outer cortex, cortex, and endodermis (not intercellular)
            if node_group not in [self.outercortex_connec_rank, 3, 4]:
                continue
            if i - (self.n_walls + self.n_junctions) in self.config.intercellular_ids:
                continue
                
            for neighboor, eattr in edges.items():
                if eattr['path'] != "plasmodesmata":
                    continue
                    
                j = indice[neighboor]
                j_group = self.graph.nodes[j]['cgroup']
                length = eattr['length']
                is_not_intercellular = j - (self.n_walls + self.n_junctions) not in self.config.intercellular_ids
                
                # Outer cortex - cortex
                if {node_group, j_group} == {self.outercortex_connec_rank, 4}:
                    self.len_outer_cortex += length
                    if is_not_intercellular:
                        self.cross_section_outer_cortex += length
                # Cortex - cortex
                elif node_group == j_group == 4:
                    self.len_cortex_cortex += length
                    if is_not_intercellular:
                        self.cross_section_cortex_cortex += length
                # Cortex - endodermis
                elif {node_group, j_group} == {3, 4}:
                    self.len_cortex_endo += length
                    if is_not_intercellular:
                        self.cross_section_cortex_endo += length
