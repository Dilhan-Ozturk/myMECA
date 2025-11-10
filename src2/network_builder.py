"""
Network builder for MECHA
Constructs the hydraulic network graph from cell data
"""

import networkx as nx
import numpy as np
from typing import Dict, List, Tuple, Any
from lxml import etree

from config_loader import MECHAConfig, parse_cellset_xml


class NetworkBuilder:
    """Builds the hydraulic network graph from XML cell data"""
    
    def __init__(self, config: MECHAConfig):
        self.config = config
        self.graph = nx.Graph()
        
        # Network dimensions
        self.n_walls = 0
        self.n_junctions = 0
        self.n_cells = 0
        self.n_total = 0
        
        # Cell and wall properties
        self.cell_areas = None
        self.cell_perimeters = None
        self.cell_ranks = None
        self.cell_groups = None
        
        # Spatial data
        self.positions = {}
        self.junction_positions = {}
        self.wall_lengths = {}
        
        # Border identification
        self.border_walls = []
        self.border_junctions = []
        self.border_aerenchyma = []
        
        # Xylem and phloem cells
        self.xylem_cells = []
        self.sieve_cells = []
        self.proto_sieve_cells = []
        
        # Connectivity
        self.cell_connections = None
        self.wall_to_cell = None
        self.junction_to_wall = {}
        
        # Gravity center (for endodermis)
        self.x_grav = 0.0
        self.y_grav = 0.0
    
    def build_from_xml(self):
        """Main method to build network from XML data"""
        cellset = parse_cellset_xml(self.config.cellset_file, self.config.im_scale)
        
        print('  Creating wall and junction nodes...')
        self._create_wall_nodes(cellset)
        self._create_junction_nodes(cellset)
        
        print('  Creating cell nodes...')
        self._create_cell_nodes(cellset)
        
        print('  Building connectivity...')
        self._build_wall_connections()
        self._build_membrane_connections(cellset)
        self._build_plasmodesmata_connections(cellset)
        
        print('  Computing cell properties...')
        self._compute_cell_properties(cellset)
        self._compute_gravity_center()
        self._rank_cells()
        
        print(f'  Network: {self.n_walls} walls, {self.n_junctions} junctions, '
              f'{self.n_cells} cells')
    
    def _create_wall_nodes(self, cellset: Dict):
        """Create nodes for cell walls"""
        points = cellset['points']
        self.n_walls = len(points)
        im_scale = cellset['im_scale']
        
        for wid, point_group in enumerate(points):
            # Calculate wall geometry
            coords = []
            for point in point_group:
                x = im_scale * float(point.get("x"))
                y = im_scale * float(point.get("y"))
                coords.append((x, y))
            
            if len(coords) < 2:
                continue
            
            # Calculate wall length
            length = sum(
                np.hypot(coords[i+1][0]-coords[i][0], coords[i+1][1]-coords[i][1])
                for i in range(len(coords)-1)
            )
            
            # Find midpoint
            mid_x, mid_y = self._find_wall_midpoint(coords, length)
            
            # Add wall node
            self.graph.add_node(
                wid,
                indice=wid,
                type="apo",
                position=(mid_x, mid_y),
                length=length
            )
            
            self.positions[wid] = (mid_x, mid_y)
            self.wall_lengths[wid] = length
    
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
    
    def _create_junction_nodes(self, cellset: Dict):
        """Create nodes at wall junctions"""
        points = cellset['points']
        im_scale = cellset['im_scale']
        
        junction_id = 0
        
        for wid, point_group in enumerate(points):
            coords = []
            for point in point_group:
                x = im_scale * float(point.get("x"))
                y = im_scale * float(point.get("y"))
                coords.append((x, y))
            
            if len(coords) < 2:
                continue
            
            # First and last points are junctions
            for coord in [coords[0], coords[-1]]:
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
                    
                    if node_id not in self.junction_to_wall:
                        self.junction_to_wall[node_id] = []
                    
                    junction_id += 1
                
                # Map wall to junction
                junc_id = self.junction_positions[pos_key]
                self.junction_to_wall[junc_id].append(wid)
        
        self.n_junctions = junction_id
    
    def _create_cell_nodes(self, cellset: Dict):
        """Create nodes for cells"""
        cell_to_wall = cellset['cell_to_wall']
        self.n_cells = len(cell_to_wall)
        
        for cell_group in cell_to_wall:
            cell_id = int(cell_group.getparent().get("id"))
            cell_type = int(cell_group.getparent().get("group"))
            
            # Calculate cell center from wall positions
            wall_positions = []
            for wall_ref in cell_group:
                wid = int(wall_ref.get("id"))
                if wid in self.positions:
                    wall_positions.append(self.positions[wid])
            
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
    
    def _build_wall_connections(self):
        """Build connections between walls and junctions"""
        for junction_id, wall_ids in self.junction_to_wall.items():
            for wid in wall_ids:
                if wid >= self.n_walls:
                    continue
                
                # Calculate direction vector
                pos_j = self.positions[junction_id]
                pos_w = self.positions[wid]
                
                d_vec = np.array([pos_w[0] - pos_j[0], pos_w[1] - pos_j[1]])
                dist = np.linalg.norm(d_vec)
                
                if dist > 0:
                    d_vec = d_vec / dist
                
                self.graph.add_edge(
                    junction_id,
                    wid,
                    path='wall',
                    length=self.wall_lengths[wid] / 2,
                    d_vec=d_vec,
                    dist_wall=dist
                )
    
    def _build_membrane_connections(self, cellset: Dict):
        """Build membrane connections between cells and walls"""
        cell_to_wall = cellset['cell_to_wall']
        
        for cell_group in cell_to_wall:
            cell_id = int(cell_group.getparent().get("id"))
            cell_node_id = self.n_walls + self.n_junctions + cell_id
            
            for wall_ref in cell_group:
                wid = int(wall_ref.get("id"))
                
                if wid >= self.n_walls:
                    continue
                
                # Calculate distance and direction
                pos_c = self.positions[cell_node_id]
                pos_w = self.positions[wid]
                
                d_vec = np.array([pos_w[0] - pos_c[0], pos_w[1] - pos_c[1]])
                dist = np.linalg.norm(d_vec)
                
                if dist > 0:
                    d_vec = d_vec / dist
                
                self.graph.add_edge(
                    cell_node_id,
                    wid,
                    path='membrane',
                    length=self.wall_lengths[wid],
                    dist=dist,
                    d_vec=d_vec
                )
    
    def _build_plasmodesmata_connections(self, cellset: Dict):
        """Build plasmodesmata connections between cells"""
        walls_list = cellset['walls']
        
        # Build wall-to-cells mapping
        wall_to_cells = {}
        for wall_elem in walls_list:
            wid = int(wall_elem.get("id"))
            cid = int(wall_elem.getparent().getparent().get("id"))
            
            if wid not in wall_to_cells:
                wall_to_cells[wid] = []
            wall_to_cells[wid].append(cid)
        
        # Connect cells that share walls
        for wid, cell_ids in wall_to_cells.items():
            if len(cell_ids) == 2:
                cell1_node = self.n_walls + self.n_junctions + cell_ids[0]
                cell2_node = self.n_walls + self.n_junctions + cell_ids[1]
                
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
                    length=self.wall_lengths[wid] if wid in self.wall_lengths else 0,
                    d_vec=d_vec
                )
    
    def _compute_cell_properties(self, cellset: Dict):
        """Compute cell areas and perimeters"""
        self.cell_areas = np.zeros(self.n_cells)
        self.cell_perimeters = np.zeros(self.n_cells)
        
        # Implementation would calculate areas using wall coordinates
        # Simplified version here
        for node in self.graph.nodes():
            if self.graph.nodes[node].get('type') == 'cell':
                cell_id = node - self.n_walls - self.n_junctions
                
                # Sum lengths of connected walls
                for neighbor in self.graph.neighbors(node):
                    edge_data = self.graph[node][neighbor]
                    if edge_data.get('path') == 'membrane':
                        self.cell_perimeters[cell_id] += edge_data.get('length', 0)
    
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
    
    def _rank_cells(self):
        """Assign ranks to cells based on tissue type and position"""
        self.cell_ranks = np.zeros(self.n_cells, dtype=int)
        self.cell_groups = {}
        
        for node in self.graph.nodes():
            if self.graph.nodes[node].get('type') == 'cell':
                cell_id = node - self.n_walls - self.n_junctions
                cgroup = self.graph.nodes[node].get('cgroup', 0)
                
                self.cell_ranks[cell_id] = cgroup
                
                if cgroup not in self.cell_groups:
                    self.cell_groups[cgroup] = []
                self.cell_groups[cgroup].append(cell_id)
    
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