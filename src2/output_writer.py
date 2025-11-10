"""
Output writer for MECHA
Handles writing of results and visualization files
"""

import os
import numpy as np
from typing import Dict, Any, List

from config_loader import MECHAConfig
from network_builder import NetworkBuilder


class OutputWriter:
    """Writes MECHA results to various output formats"""
    
    def __init__(self, output_dir: str, config: MECHAConfig, network: NetworkBuilder):
        """
        Initialize output writer
        
        Parameters
        ----------
        output_dir : str
            Directory for output files
        config : MECHAConfig
            Configuration object
        network : NetworkBuilder
            Network structure
        """
        self.output_dir = output_dir
        self.config = config
        self.network = network
        
        os.makedirs(output_dir, exist_ok=True)
    
    def write_macroscopic_properties(self, maturity_idx: int, barrier: int, 
                                    results: Dict[str, Any]):
        """
        Write macroscopic hydraulic properties to text file
        
        Parameters
        ----------
        maturity_idx : int
            Index of maturity stage
        barrier : int
            Barrier type
        results : Dict
            Simulation results
        """
        filename = f"Macro_prop_{barrier}_{maturity_idx}.txt"
        filepath = os.path.join(self.output_dir, filename)
        
        with open(filepath, 'w') as f:
            f.write(f"Macroscopic root radial hydraulic properties\n")
            f.write(f"Apoplastic barrier: {barrier}, Maturity: {maturity_idx}\n\n")

            f.write(f"Number of scenarios: {len(results['scenarios'])}\n\n")

            # Write scenario details
            for i, scenario_result in enumerate(results['scenarios']):
                f.write(f"\n{'='*60}\n")
                f.write(f"Scenario {i}\n")
                f.write(f"Cross-section radial conductivity: {results['kr_tot'][i]:.6e} cm/hPa/d\n")
                f.write(f"{'='*60}\n\n")

                scenario = scenario_result['scenario']

                f.write(f"Soil pressure: {scenario.get('psi_soil_left', 0):.2f} hPa\n")
                f.write(f"Xylem pressure: {scenario.get('psi_xyl', 'N/A')} hPa\n")
                f.write(f"Phloem pressure: {scenario.get('psi_sieve', 'N/A')} hPa\n\n")

                flows = scenario_result['flows']
                f.write(f"Total flow rate: {flows['total'][i]:.6e} cm³/d\n")
                f.write(f"Soil inflow: {flows['soil'][i]:.6e} cm³/d\n")
                f.write(f"Xylem outflow: {flows['xylem'][i]:.6e} cm³/d\n")
        
        print(f"    Wrote: {filename}")
    
    def write_paraview_files(self, maturity_idx: int, barrier: int, 
                           results: Dict[str, Any]):
        """
        Write ParaView visualization files (.pvtk format)
        
        Parameters
        ----------
        maturity_idx : int
            Index of maturity stage
        barrier : int
            Barrier type
        results : Dict
            Simulation results
        """
        if not self.config.paraview:
            return
        
        # Write different visualization types
        if self.config.paraview_wp:
            self._write_wall_potentials(maturity_idx, barrier, results)
        
        if self.config.paraview_cp:
            self._write_cell_potentials(maturity_idx, barrier, results)
        
        if self.config.paraview_mf:
            self._write_membrane_fluxes(maturity_idx, barrier, results)
    
    def _write_wall_potentials(self, maturity_idx: int, barrier: int, 
                              results: Dict[str, Any]):
        """Write wall potential visualization"""
        for scenario_idx, scenario_result in enumerate(results['scenarios']):
            filename = f"Walls2D_b{barrier}_{maturity_idx}_s{scenario_idx}.pvtk"
            filepath = os.path.join(self.output_dir, filename)
            
            solution = scenario_result['solution']
            
            with open(filepath, 'w') as f:
                f.write("# vtk DataFile Version 4.0\n")
                f.write("Wall geometry 2D\n")
                f.write("ASCII\n\n")
                f.write("DATASET UNSTRUCTURED_GRID\n")
                
                # Write points
                f.write(f"POINTS {self.network.n_walls} float\n")
                for wid in range(self.network.n_walls):
                    pos = self.network.positions[wid]
                    f.write(f"{pos[0]} {pos[1]} 0.0\n")
                
                f.write("\n")
                
                # Write cells (lines between walls)
                n_connections = 0
                connections = []
                for node_i, node_j in self.network.graph.edges():
                    if node_i < self.network.n_walls and node_j < self.network.n_walls:
                        edge_data = self.network.graph[node_i][node_j]
                        if edge_data.get('path') == 'wall':
                            connections.append((node_i, node_j))
                            n_connections += 1
                
                f.write(f"CELLS {n_connections} {n_connections * 3}\n")
                for node_i, node_j in connections:
                    f.write(f"2 {node_i} {node_j}\n")
                
                f.write("\n")
                
                # Write cell types
                f.write(f"CELL_TYPES {n_connections}\n")
                for _ in range(n_connections):
                    f.write("3\n")  # Line cell type
                
                f.write("\n")
                
                # Write point data (potentials)
                f.write(f"POINT_DATA {self.network.n_walls}\n")
                f.write("SCALARS Wall_pressure float\n")
                f.write("LOOKUP_TABLE default\n")
                for wid in range(self.network.n_walls):
                    f.write(f"{solution[wid]}\n")
            
            print(f"    Wrote: {filename}")
    
    def _write_cell_potentials(self, maturity_idx: int, barrier: int, 
                              results: Dict[str, Any]):
        """Write cell potential visualization"""
        for scenario_idx, scenario_result in enumerate(results['scenarios']):
            filename = f"Cells2D_b{barrier}_{maturity_idx}_s{scenario_idx}.pvtk"
            filepath = os.path.join(self.output_dir, filename)
            
            solution = scenario_result['solution']
            
            with open(filepath, 'w') as f:
                f.write("# vtk DataFile Version 4.0\n")
                f.write("Cell pressure distribution 2D\n")
                f.write("ASCII\n\n")
                f.write("DATASET UNSTRUCTURED_GRID\n")
                
                # Write points (all nodes including cells)
                n_all = self.network.n_walls + self.network.n_junctions + self.network.n_cells
                f.write(f"POINTS {n_all} float\n")
                for node_id in range(n_all):
                    pos = self.network.positions.get(node_id, (0, 0))
                    f.write(f"{pos[0]} {pos[1]} 0.0\n")
                
                f.write("\n")
                
                # Write cells (vertices for cell centers)
                f.write(f"CELLS {self.network.n_cells} {self.network.n_cells * 2}\n")
                for cell_id in range(self.network.n_cells):
                    node_id = self.network.n_walls + self.network.n_junctions + cell_id
                    f.write(f"1 {node_id}\n")
                
                f.write("\n")
                
                # Write cell types
                f.write(f"CELL_TYPES {self.network.n_cells}\n")
                for _ in range(self.network.n_cells):
                    f.write("1\n")  # Vertex cell type
                
                f.write("\n")
                
                # Write point data
                f.write(f"POINT_DATA {n_all}\n")
                f.write("SCALARS Cell_pressure float\n")
                f.write("LOOKUP_TABLE default\n")
                for node_id in range(n_all):
                    f.write(f"{solution[node_id]}\n")
            
            print(f"    Wrote: {filename}")
    
    def _write_membrane_fluxes(self, maturity_idx: int, barrier: int, 
                              results: Dict[str, Any]):
        """Write membrane flux visualization"""
        # Implementation would create 3D visualization of membrane fluxes
        # This is simplified for brevity
        print(f"    Membrane flux visualization not yet implemented")
    
    def write_summary(self, all_results: List[Dict[str, Any]]):
        """
        Write a summary file with key results from all maturity stages
        
        Parameters
        ----------
        all_results : List[Dict]
            Results from all maturity stages
        """
        filename = "summary.txt"
        filepath = os.path.join(self.output_dir, filename)
        
        with open(filepath, 'w') as f:
            f.write("MECHA Simulation Summary\n")
            f.write("="*60 + "\n\n")
            
            f.write(f"Plant: {self.config.plant_name}\n")
            f.write(f"Number of maturity stages: {len(all_results)}\n")
            f.write(f"Number of scenarios: {self.config.n_scenarios}\n\n")
            
            f.write("Radial conductivities (cm/hPa/d):\n")
            for i, results in enumerate(all_results):
                f.write(f"  Stage {i}: {results['kr_tot']:.6e}\n")
        
        print(f"  Wrote: {filename}")


def write_network_stats(network: NetworkBuilder, output_dir: str):
    """
    Write network statistics to file
    
    Parameters
    ----------
    network : NetworkBuilder
        Built network
    output_dir : str
        Output directory
    """
    filename = "network_stats.txt"
    filepath = os.path.join(output_dir, filename)
    
    with open(filepath, 'w') as f:
        f.write("Network Statistics\n")
        f.write("="*60 + "\n\n")
        
        f.write(f"Total nodes: {network.graph.number_of_nodes()}\n")
        f.write(f"  - Walls: {network.n_walls}\n")
        f.write(f"  - Junctions: {network.n_junctions}\n")
        f.write(f"  - Cells: {network.n_cells}\n\n")
        
        f.write(f"Total edges: {network.graph.number_of_edges()}\n\n")
        
        # Count edge types
        edge_types = {}
        for _, _, data in network.graph.edges(data=True):
            path_type = data.get('path', 'unknown')
            edge_types[path_type] = edge_types.get(path_type, 0) + 1
        
        f.write("Edge types:\n")
        for path_type, count in edge_types.items():
            f.write(f"  - {path_type}: {count}\n")
        
        f.write("\n")
        
        # Cell type distribution
        if network.cell_groups:
            f.write("Cell type distribution:\n")
            cell_type_names = {
                1: "Exodermis",
                2: "Epidermis",
                3: "Endodermis",
                4: "Cortex",
                5: "Stele",
                11: "Phloem sieve",
                12: "Companion cell",
                13: "Xylem",
                16: "Pericycle"
            }
            
            for cgroup, cell_ids in network.cell_groups.items():
                name = cell_type_names.get(cgroup, f"Type {cgroup}")
                f.write(f"  - {name}: {len(cell_ids)}\n")
    
    print(f"Wrote: {filename}")