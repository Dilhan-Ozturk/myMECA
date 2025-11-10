"""
Hydraulic solver for MECHA
Assembles and solves the hydraulic equations
"""

import numpy as np
import scipy.linalg as slin
from typing import Dict, List, Tuple, Any, Optional

from config_loader import MECHAConfig
from network_builder import NetworkBuilder


class HydraulicSolver:
    """Solves the hydraulic equations for the root network"""
    
    def __init__(self, network: NetworkBuilder, config: MECHAConfig, 
                 barrier: int, height: float):
        """
        Initialize hydraulic solver
        
        Parameters
        ----------
        network : NetworkBuilder
            Built network structure
        config : MECHAConfig
            Configuration object
        barrier : int
            Apoplastic barrier type (0-4)
        height : float
            Cell axial height in microns
        """
        self.network = network
        self.config = config
        self.barrier = barrier
        self.height = height
        
        # Get number of nodes
        self.n_nodes = network.graph.number_of_nodes()
        self.n_walls = network.n_walls
        self.n_junctions = network.n_junctions
        self.n_cells = network.n_cells
        
        # Hydraulic parameters (will be set per scenario)
        self.kw = 0.0  # Wall hydraulic conductivity
        self.kw_barrier = 0.0  # Barrier wall conductivity
        self.km = 0.0  # Membrane conductivity
        self.kaqp = 0.0  # Aquaporin contribution
        self.kpl = 0.0  # Plasmodesmata conductance
        
        # Matrices
        self.matrix_W = None  # Water potential matrix
        self.rhs = None  # Right-hand side
        
        # Results storage
        self.solutions = {}
        self.flow_rates = {}
        self.kr_tot = 0.0
        
        # Calculate perimeter based on epidermis cells
        self._calculate_perimeter()
    
    def _calculate_perimeter(self):
        """Calculate root cross-section perimeter from epidermis cells"""
        x_sum = 0.0
        y_sum = 0.0
        n_epi = 0
        
        # Calculate center of gravity from endodermis
        x_grav = 0.0
        y_grav = 0.0
        n_endo = 0
        
        for node_id in range(self.n_walls + self.n_junctions, self.n_nodes):
            cell_data = self.network.graph.nodes[node_id]
            if cell_data.get('cgroup') == 3:  # Endodermis
                pos = cell_data['position']
                x_grav += pos[0]
                y_grav += pos[1]
                n_endo += 1
        
        if n_endo > 0:
            x_grav /= n_endo
            y_grav /= n_endo
        
        # Calculate average distance of epidermis from center
        for node_id in range(self.n_walls + self.n_junctions, self.n_nodes):
            cell_data = self.network.graph.nodes[node_id]
            if cell_data.get('cgroup') == 2:  # Epidermis
                pos = cell_data['position']
                dist = np.hypot(pos[0] - x_grav, pos[1] - y_grav)
                x_sum += dist
                n_epi += 1
        
        if n_epi > 0:
            avg_radius = x_sum / n_epi
            self.perimeter = 2 * np.pi * avg_radius * 1.0E-4  # Convert to cm
        else:
            self.perimeter = 0.01  # Default value in cm
    
    def solve_all_scenarios(self) -> Dict[str, Any]:
        """
        Solve hydraulic equations for all scenarios
        
        Returns
        -------
        Dict containing:
            - kr_tot: total radial conductivity
            - solutions: pressure distributions
            - flows: flow rates
        """
        results = {
            'kr_tot': 0.0,
            'scenarios': []
        }
        
        # Scenario 0: baseline (equilibrium)
        print(f'    Scenario 0 (equilibrium)...')
        sol0 = self._solve_scenario(0)
        results['kr_tot'] = sol0['kr']
        results['scenarios'].append(sol0)
        
        # Additional scenarios
        for scenario_idx in range(1, self.config.n_scenarios):
            print(f'    Scenario {scenario_idx}...')
            sol = self._solve_scenario(scenario_idx)
            results['scenarios'].append(sol)
        
        return results
    
    def _solve_scenario(self, scenario_idx: int) -> Dict[str, Any]:
        """
        Solve a single scenario
        
        Parameters
        ----------
        scenario_idx : int
            Index of the scenario to solve
            
        Returns
        -------
        Dict with solution, flows, and derived quantities
        """
        # Get scenario boundary conditions
        scenario = self.config.scenarios[scenario_idx]
        
        # Initialize matrices
        self._initialize_matrices()
        
        # Set hydraulic parameters
        self._set_hydraulic_parameters(scenario_idx)
        
        # Build conductance matrix
        self._build_conductance_matrix()
        
        # Apply boundary conditions
        self._apply_boundary_conditions(scenario)
        
        # Solve system
        solution = np.linalg.solve(self.matrix_W, self.rhs)
        
        # Calculate flows
        flows = self._calculate_flows(solution, scenario)
        
        # Calculate radial conductivity (scenario 0 only)
        kr = 0.0
        if scenario_idx == 0:
            kr = self._calculate_radial_conductivity(flows, scenario)
        
        return {
            'solution': solution,
            'flows': flows,
            'kr': kr,
            'scenario': scenario
        }
    
    def _initialize_matrices(self):
        """Initialize the conductance matrix and RHS vector"""
        self.matrix_W = np.zeros((self.n_nodes, self.n_nodes))
        self.rhs = np.zeros((self.n_nodes, 1))
    
    def _set_hydraulic_parameters(self, scenario_idx: int):
        """Set hydraulic parameters based on configuration"""
        # Get hydraulic config (use first config for simplicity)
        h_config = self.config.hydraulic_configs[0] if self.config.hydraulic_configs else {}
        
        self.kw = h_config.get('kw', 1.0)
        self.km = h_config.get('km', 1.0)
        self.kaqp = h_config.get('kaqp', 0.001)
        self.kpl = h_config.get('kpl', 0.0001)
        
        # Set barrier conductivities based on barrier type
        if self.barrier == 0:
            self.kw_barrier = self.kw
            self.kw_endo_endo = self.kw
            self.kw_endo_cortex = self.kw
            self.kw_endo_peri = self.kw
            self.kw_exo_exo = self.kw
        elif self.barrier == 1:  # Endodermis radial walls
            self.kw_barrier = self.kw / 10.0
            self.kw_endo_endo = self.kw_barrier
            self.kw_endo_cortex = self.kw
            self.kw_endo_peri = self.kw
            self.kw_exo_exo = self.kw
        elif self.barrier == 2:  # Endodermis with passage cells
            self.kw_barrier = self.kw / 10.0
            self.kw_endo_endo = self.kw_barrier
            self.kw_endo_cortex = self.kw_barrier
            self.kw_endo_peri = self.kw_barrier
            self.kw_exo_exo = self.kw
        elif self.barrier == 3:  # Endodermis full
            self.kw_barrier = self.kw / 10.0
            self.kw_endo_endo = self.kw_barrier
            self.kw_endo_cortex = self.kw_barrier
            self.kw_endo_peri = self.kw_barrier
            self.kw_exo_exo = self.kw
        elif self.barrier == 4:  # Endodermis full and exodermis radial walls
            self.kw_barrier = self.kw / 10.0
            self.kw_endo_endo = self.kw_barrier
            self.kw_endo_cortex = self.kw_barrier
            self.kw_endo_peri = self.kw_barrier
            self.kw_exo_exo = self.kw_barrier
    
    def _build_conductance_matrix(self):
        """Build the conductance matrix from network connections"""
        # Loop through all edges
        for node_i, node_j in self.network.graph.edges():
            edge_data = self.network.graph[node_i][node_j]
            path_type = edge_data.get('path', '')
            
            # Calculate conductance based on path type
            if path_type == 'wall':
                K = self._calculate_wall_conductance(node_i, node_j, edge_data)
            elif path_type == 'membrane':
                K = self._calculate_membrane_conductance(node_i, node_j, edge_data)
            elif path_type == 'plasmodesmata':
                K = self._calculate_plasmodesmata_conductance(node_i, node_j, edge_data)
            else:
                continue
            
            # Add to matrix (symmetric)
            self.matrix_W[node_i, node_i] -= K
            self.matrix_W[node_i, node_j] += K
            self.matrix_W[node_j, node_i] += K
            self.matrix_W[node_j, node_j] -= K
    
    def _calculate_wall_conductance(self, node_i: int, node_j: int, 
                                    edge_data: Dict) -> float:
        """Calculate conductance for wall connection"""
        length = edge_data.get('length', 1.0)
        lat_dist = edge_data.get('lat_dist', 0.0)
        thickness = self.config.thickness
        
        # Section to length ratio (cm)
        section_ratio = 1.0e-4 * ((lat_dist + self.height) * 
                                  thickness - thickness**2) / length
        
        # Determine wall type and apply appropriate conductivity
        # Check cell types around the wall
        count_endo = 0
        count_cortex = 0
        count_exo = 0
        count_stele = 0
        
        if node_i < self.n_walls:
            for neighbor in self.network.graph[node_i]:
                if neighbor >= self.n_walls + self.n_junctions:  # Cell node
                    cgroup = self.network.graph.nodes[neighbor].get('cgroup', 0)
                    if cgroup == 3:
                        count_endo += 1
                    elif cgroup == 4:
                        count_cortex += 1
                    elif cgroup == 1:
                        count_exo += 1
                    elif cgroup > 4:
                        count_stele += 1
        
        # Apply barrier-specific conductivity
        if count_endo >= 2:  # Wall between endodermis cells
            K = self.kw_endo_endo * section_ratio
        elif count_endo > 0 and count_cortex > 0:  # Endodermis-cortex wall
            K = self.kw_endo_cortex * section_ratio
        elif count_endo > 0 and count_stele > 0:  # Endodermis-pericycle wall
            K = self.kw_endo_peri * section_ratio
        elif count_exo >= 2:  # Wall between exodermis cells
            K = self.kw_exo_exo * section_ratio
        else:  # Regular wall
            K = self.kw * section_ratio
        
        return K
    
    def _calculate_membrane_conductance(self, node_i: int, node_j: int,
                                       edge_data: Dict) -> float:
        """Calculate conductance for membrane connection"""
        length = edge_data.get('length', 1.0)
        dist = edge_data.get('dist', 1.0)
        
        # Surface area (cm²)
        area = 1.0e-8 * (self.height + dist) * length
        
        # Get cell-specific AQP activity
        if node_j >= self.n_walls + self.n_junctions:  # Cell node
            cgroup = self.network.graph.nodes[node_j].get('cgroup', 0)
            
            # Get AQP factors from config
            h_config = self.config.hydraulic_configs[0] if self.config.hydraulic_configs else {}
            
            if cgroup == 1:  # Exodermis
                kaqp = self.kaqp * h_config.get('kaqp_exo', 1.0)
            elif cgroup == 2:  # Epidermis
                kaqp = self.kaqp * h_config.get('kaqp_epi', 1.0)
            elif cgroup == 3:  # Endodermis
                kaqp = self.kaqp * h_config.get('kaqp_endo', 1.0)
            elif cgroup == 4:  # Cortex
                kaqp = self.kaqp * h_config.get('kaqp_cortex', 1.0)
            elif cgroup > 4:  # Stele
                kaqp = self.kaqp * h_config.get('kaqp_stele', 1.0)
            else:
                kaqp = self.kaqp
        else:
            kaqp = self.kaqp
        
        # Combined membrane and aquaporin conductance
        K = (self.km + kaqp) * area
        
        return K
    
    def _calculate_plasmodesmata_conductance(self, node_i: int, node_j: int,
                                            edge_data: Dict) -> float:
        """Calculate conductance for plasmodesmata connection"""
        length = edge_data.get('length', 1.0)
        
        # Plasmodesmata frequency (per cm²) - from config
        pd_frequency = self.config.pd_frequency if hasattr(self.config, 'pd_frequency') else 100.0
        
        # Calculate conductance
        K = self.kpl * pd_frequency * length * 1.0e-4
        
        return K
    
    def _apply_boundary_conditions(self, scenario: Dict):
        """Apply boundary conditions to the system"""
        # Soil boundary
        self._apply_soil_bc(scenario)
        
        # Xylem boundary
        if self.barrier > 0:
            self._apply_xylem_bc(scenario)
        
        # Phloem boundary
        if self.barrier == 0:
            self._apply_phloem_bc(scenario)
    
    def _apply_soil_bc(self, scenario: Dict):
        """Apply soil boundary conditions"""
        psi_soil_left = scenario.get('psi_soil_left', 0.0)
        psi_soil_right = scenario.get('psi_soil_right', 0.0)
        
        # Get border nodes
        border_walls, border_junctions = self.network.get_border_nodes()
        
        # Apply to border walls
        for wid in border_walls:
            if wid >= self.n_walls:
                continue
            
            pos = self.network.positions[wid]
            
            # Interpolate soil potential based on x position
            if hasattr(self.network, 'x_min') and hasattr(self.network, 'x_max'):
                x_min = self.network.x_min
                x_max = self.network.x_max
            else:
                all_pos = list(self.network.positions.values())
                x_min = min(p[0] for p in all_pos)
                x_max = max(p[0] for p in all_pos)
            
            x_rel = (pos[0] - x_min) / (x_max - x_min) if x_max > x_min else 0.5
            
            psi_soil = psi_soil_left * (1 - x_rel) + psi_soil_right * x_rel
            
            # Calculate conductance
            length = self.network.wall_lengths.get(wid, 0)
            thickness = self.config.thickness
            K = self.kw * 1.0e-4 * (length / 2 * self.height) / (thickness / 2)
            
            # Apply BC
            self.matrix_W[wid, wid] -= K
            self.rhs[wid] = -K * psi_soil
        
        # Apply to border junctions
        for jid in border_junctions:
            pos = self.network.positions[jid]
            
            x_rel = (pos[0] - x_min) / (x_max - x_min) if x_max > x_min else 0.5
            psi_soil = psi_soil_left * (1 - x_rel) + psi_soil_right * x_rel
            
            # Calculate conductance
            length = self.network.junction_lengths.get(jid, 0)
            thickness = self.config.thickness
            K = self.kw * 1.0e-4 * (length * self.height) / (thickness / 2)
            
            # Apply BC
            self.matrix_W[jid, jid] -= K
            self.rhs[jid] = -K * psi_soil
    
    def _apply_xylem_bc(self, scenario: Dict):
        """Apply xylem boundary conditions"""
        psi_xyl = scenario.get('psi_xyl', np.nan)
        flow_xyl = scenario.get('flow_xyl', np.nan)
        
        if not np.isnan(psi_xyl):
            # Pressure BC
            K_xyl = 1.0e-3  # Axial conductance (cm³/hPa/d)
            
            for cell_node in self.network.xylem_cells:
                self.matrix_W[cell_node, cell_node] -= K_xyl
                self.rhs[cell_node] = -K_xyl * psi_xyl
        
        elif not np.isnan(flow_xyl):
            # Flow BC
            n_xylem = len(self.network.xylem_cells)
            if n_xylem > 0:
                flow_per_vessel = flow_xyl / n_xylem
                
                for cell_node in self.network.xylem_cells:
                    self.rhs[cell_node] = flow_per_vessel
    
    def _apply_phloem_bc(self, scenario: Dict):
        """Apply phloem boundary conditions"""
        psi_sieve = scenario.get('psi_sieve', np.nan)
        flow_sieve = scenario.get('flow_sieve', np.nan)
        
        if not np.isnan(psi_sieve):
            # Pressure BC
            K_sieve = 1.0e-3  # Axial conductance (cm³/hPa/d)
            
            for cell_node in self.network.sieve_cells:
                self.matrix_W[cell_node, cell_node] -= K_sieve
                self.rhs[cell_node] = -K_sieve * psi_sieve
        
        elif not np.isnan(flow_sieve):
            # Flow BC
            n_sieve = len(self.network.sieve_cells)
            if n_sieve > 0:
                flow_per_tube = flow_sieve / n_sieve
                
                for cell_node in self.network.sieve_cells:
                    self.rhs[cell_node] = flow_per_tube
    
    def _calculate_flows(self, solution: np.ndarray, scenario: Dict) -> Dict[str, Any]:
        """Calculate flow rates from solution"""
        flows = {
            'soil': 0.0,
            'xylem': 0.0,
            'phloem': 0.0,
            'total': 0.0
        }
        
        # Get soil boundary conditions
        psi_soil_left = scenario.get('psi_soil_left', 0.0)
        psi_soil_right = scenario.get('psi_soil_right', 0.0)
        
        # Calculate soil inflow
        border_walls, border_junctions = self.network.get_border_nodes()
        
        # Get position bounds
        if hasattr(self.network, 'x_min') and hasattr(self.network, 'x_max'):
            x_min = self.network.x_min
            x_max = self.network.x_max
        else:
            all_pos = list(self.network.positions.values())
            x_min = min(p[0] for p in all_pos)
            x_max = max(p[0] for p in all_pos)
        
        # Calculate flows from border walls
        for wid in border_walls:
            if wid >= self.n_walls:
                continue
            
            pos = self.network.positions[wid]
            x_rel = (pos[0] - x_min) / (x_max - x_min) if x_max > x_min else 0.5
            psi_soil = psi_soil_left * (1 - x_rel) + psi_soil_right * x_rel
            
            # Calculate conductance
            length = self.network.wall_lengths.get(wid, 0)
            thickness = self.config.thickness
            K = self.kw * 1.0e-4 * (length / 2 * self.height) / (thickness / 2)
            
            # Flow = K * (psi_wall - psi_soil)
            flow = K * (solution[wid] - psi_soil)
            flows['soil'] += flow
        
        # Calculate flows from border junctions
        for jid in border_junctions:
            pos = self.network.positions[jid]
            x_rel = (pos[0] - x_min) / (x_max - x_min) if x_max > x_min else 0.5
            psi_soil = psi_soil_left * (1 - x_rel) + psi_soil_right * x_rel
            
            # Calculate conductance
            length = self.network.junction_lengths.get(jid, 0)
            thickness = self.config.thickness
            K = self.kw * 1.0e-4 * (length * self.height) / (thickness / 2)
            
            # Flow = K * (psi_junction - psi_soil)
            flow = K * (solution[jid] - psi_soil)
            flows['soil'] += flow
        
        # Calculate xylem flows
        if self.barrier > 0:
            psi_xyl = scenario.get('psi_xyl', np.nan)
            if not np.isnan(psi_xyl):
                K_xyl = 1.0e-3
                for cell_node in self.network.xylem_cells:
                    flow = K_xyl * (solution[cell_node] - psi_xyl)
                    flows['xylem'] += flow
        
        # Calculate phloem flows
        if self.barrier == 0:
            psi_sieve = scenario.get('psi_sieve', np.nan)
            if not np.isnan(psi_sieve):
                K_sieve = 1.0e-3
                for cell_node in self.network.sieve_cells:
                    flow = K_sieve * (solution[cell_node] - psi_sieve)
                    flows['phloem'] += flow
        
        flows['total'] = flows['soil']
        
        return flows
    
    def _calculate_radial_conductivity(self, flows: Dict, scenario: Dict) -> float:
        """Calculate total radial hydraulic conductivity"""
        # kr = Q / (ΔΨ * perimeter * height)
        
        psi_soil = scenario.get('psi_soil_left', 0.0)
        
        if self.barrier > 0:
            psi_xyl = scenario.get('psi_xyl', 0.0)
        else:
            psi_sieve = scenario.get('psi_sieve', 0.0)
            psi_xyl = psi_sieve
        
        if abs(psi_soil - psi_xyl) < 1e-6:
            return 0.0
        
        height_cm = self.height * 1.0e-4
        
        kr = flows['total'] / ((psi_soil - psi_xyl) * self.perimeter * height_cm)
        
        return abs(kr)
    
    def run_contagion_analysis(self):
        """Run hormone contagion analysis if enabled"""
        if self.config.sym_contagion:
            self._run_symplastic_contagion()
        
        if self.config.apo_contagion:
            self._run_apoplastic_contagion()
    
    def _run_symplastic_contagion(self):
        """Run symplastic hormone contagion analysis"""
        print('    Running symplastic contagion...')
        # This would implement hormone tracking through symplast
        # Following flow directions through plasmodesmata
        pass
    
    def _run_apoplastic_contagion(self):
        """Run apoplastic hormone contagion analysis"""
        print('    Running apoplastic contagion...')
        # This would implement hormone tracking through apoplast
        # Following flow directions through cell walls
        pass