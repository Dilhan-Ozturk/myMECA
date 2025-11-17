"""
Output writer module for MECHA hydraulic modeling.
Contains functions for writing results to various file formats.
"""

import numpy as np
from numpy import zeros, isnan, nan


class OutputWriter:
    """Class for writing hydraulic modeling results to files."""
    
    def __init__(self, network, config):
        """
        Initialize the output writer.
        
        Parameters:
        -----------
        network : NetworkData
            Network geometry data
        config : ConfigData
            Configuration parameters
        """
        self.network = network
        self.config = config
        self.SECONDS_PER_DAY = 86400.0
        self.CM_PER_METER = 100.0
    
    def write_macroscopic_properties(self, filepath, Barrier, iMaturity,
                                     height, kr_tot, K_xyl_spec, 
                                     STFlayer_plus, STFlayer_minus,
                                     scenarios_data):
        """
        Write macroscopic hydraulic properties to text file.
        
        Parameters:
        -----------
        filepath : str
            Output file path
        Barrier : int
            Apoplastic barrier type
        iMaturity : int
            Maturity index
        height : float
            Cell axial length (microns)
        kr_tot : float
            Radial conductivity
        K_xyl_spec : float
            Xylem specific axial conductance
        STFlayer_plus : array
            Standard transmembrane uptake fractions
        STFlayer_minus : array
            Standard transmembrane release fractions
        scenarios_data : list
            List of dictionaries containing scenario results
        """
        with open(filepath, "w") as f:
            f.write(f"Macroscopic root radial hydraulic properties, "
                   f"apoplastic barrier {Barrier},{iMaturity}\n\n")
            
            f.write(f"{len(scenarios_data)} scenarios\n\n")
            f.write(f"Cross-section height: {height*1.0E-04} cm\n\n")
            f.write(f"Cross-section perimeter: {self.network.perimeter[0]} cm\n\n")
            f.write(f"Xylem specific axial conductance: {K_xyl_spec} cm^4/hPa/d\n\n")
            f.write(f"Cross-section radial conductivity: {kr_tot} cm/hPa/d\n\n")
            
            # Radial discretization
            f.write("Number of radial discretization boxes:\n")
            f.write(f"{int(self.network.r_discret[0])}\n\n")
            
            f.write("Radial distance from stele centre (microns):\n")
            for dist in self.network.layer_dist:
                f.write(f"{float(dist)}\n")
            f.write("\n")
            
            # Standard transmembrane fractions
            f.write("Standard Transmembrane uptake Fractions (%):\n")
            for j in range(int(self.network.r_discret[0])):
                f.write(f"{STFlayer_plus[j][iMaturity]*100}\n")
            f.write("\n")
            
            f.write("Standard Transmembrane release Fractions (%):\n")
            for j in range(int(self.network.r_discret[0])):
                f.write(f"{STFlayer_minus[j][iMaturity]*100}\n")
            f.write("\n")
            
            # Scenario-specific data
            for i, scenario_data in enumerate(scenarios_data, start=1):
                f.write(f"\nScenario {i}\n\n")
                self._write_scenario_data(f, scenario_data, iMaturity)
    
    def _write_scenario_data(self, f, scenario_data, iMaturity):
        """Write data for a single scenario."""
        f.write(f"h_x: {scenario_data['Psi_xyl']} hPa\n")
        f.write(f"h_s: {scenario_data['psi_soil_left']} to "
               f"{scenario_data['psi_soil_right']} hPa\n")
        f.write(f"h_p: {scenario_data['Psi_sieve']} hPa\n\n")
        
        f.write(f"q_tot: {scenario_data['Q_tot']} cm^2/d\n\n")
        
        # Uptake distributions
        f.write("Uptake distribution cm^3/d:\n")
        for val in scenario_data['UptakeLayer_plus']:
            f.write(f"{val}\n")
        f.write("\n")
        
        f.write("Release distribution cm^3/d:\n")
        for val in scenario_data['UptakeLayer_minus']:
            f.write(f"{val}\n")
        f.write("\n")
    
    def write_paraview_vtk_walls(self, filepath, ThickWallsX, 
                                 WallFlowDensity, list_ghostwalls,
                                 list_ghostjunctions, Wall2NewWallX,
                                 nWall2NewWallX, ThickWallPolygonX,
                                 nGhostJunction2Wall):
        """
        Write wall geometry and flow data for ParaView visualization.
        
        Parameters:
        -----------
        filepath : str
            Output VTK file path
        ThickWallsX : list
            Extended wall node data
        WallFlowDensity : list
            Wall flow density data
        ... (other parameters for geometry)
        """
        with open(filepath, "w") as f:
            f.write("# vtk DataFile Version 4.0\n")
            f.write("Wall geometry 3D including thickness bottom\n")
            f.write("ASCII\n\n")
            f.write("DATASET UNSTRUCTURED_GRID\n")
            
            # Points
            f.write(f"POINTS {len(ThickWallsX)} float\n")
            for node in ThickWallsX:
                f.write(f"{node[1]} {node[2]} 0.0\n")
            f.write("\n")
            
            # Cells (wall polygons)
            n_cells = (self.network.n_walls + self.network.n_junctions + 
                      self.network.n_walls - len(list_ghostwalls)*2 - 
                      len(list_ghostjunctions))
            n_values = (2*self.network.n_walls*5 - len(list_ghostwalls)*10 + 
                       sum(nWall2NewWallX[self.network.n_walls:]) +
                       (self.network.n_walls + self.network.n_junctions) - 
                       self.network.n_walls + 
                       2*len(Wall2NewWallX[self.network.n_walls:]) -
                       nGhostJunction2Wall - len(list_ghostjunctions))
            
            f.write(f"CELLS {int(n_cells)} {int(n_values)}\n")
            
            # Write wall polygons
            i = 0
            for PolygonX in ThickWallPolygonX:
                if np.floor(i/2) not in list_ghostwalls:
                    f.write(f"4 {int(PolygonX[0])} {int(PolygonX[1])} "
                           f"{int(PolygonX[2])} {int(PolygonX[3])}\n")
                i += 1
            
            # Write junction polygons
            j = self.network.n_walls
            for PolygonX in Wall2NewWallX[self.network.n_walls:]:
                if j not in list_ghostjunctions:
                    string = f"{int(nWall2NewWallX[j]+2)}"
                    for id1 in range(int(nWall2NewWallX[j])):
                        string += f" {int(PolygonX[id1])}"
                    string += f" {int(PolygonX[0])} {int(PolygonX[1])}"
                    f.write(string + "\n")
                j += 1
            f.write("\n")
            
            # Cell types
            f.write(f"CELL_TYPES {int(n_cells)}\n")
            i = 0
            for PolygonX in ThickWallPolygonX:
                if np.floor(i/2) not in list_ghostwalls:
                    f.write("7\n")  # Polygon
                i += 1
            
            j = self.network.n_walls
            for PolygonX in Wall2NewWallX[self.network.n_walls:]:
                if j not in list_ghostjunctions:
                    f.write("6\n")  # Triangle strip
                j += 1
            f.write("\n")
            
            # Point data (flow densities)
            f.write(f"POINT_DATA {len(ThickWallsX)}\n")
            f.write("SCALARS Apo_flux_(m/s) float\n")
            f.write("LOOKUP_TABLE default\n")
            
            # Average flow densities at nodes
            NewWallFlowDensity = self._average_wall_flow_density(
                ThickWallsX, WallFlowDensity, ThickWallPolygonX
            )
            
            for i in range(len(ThickWallsX)):
                flow_ms = float(np.mean(NewWallFlowDensity[i])) / \
                         self.SECONDS_PER_DAY / self.CM_PER_METER
                f.write(f"{flow_ms}\n")
    
    def _average_wall_flow_density(self, ThickWallsX, WallFlowDensity, 
                                   ThickWallPolygonX):
        """Average wall flow densities at nodes."""
        NewWallFlowDensity = zeros((len(ThickWallsX), 2))
        
        i = 0
        for PolygonX in ThickWallPolygonX:
            for id1 in range(4):
                if abs(float(WallFlowDensity[i][2])) > \
                   min(NewWallFlowDensity[int(PolygonX[id1])]):
                    NewWallFlowDensity[int(PolygonX[id1])][0] = \
                        max(NewWallFlowDensity[int(PolygonX[id1])])
                    NewWallFlowDensity[int(PolygonX[id1])][1] = \
                        abs(float(WallFlowDensity[i][2]))
            i += 1
        
        return NewWallFlowDensity
    
    def write_paraview_vtk_membranes(self, filepath, ThickWalls, 
                                    MembraneFlowDensity, height,
                                    list_ghostwalls):
        """
        Write membrane geometry and flow data for ParaView.
        
        Parameters:
        -----------
        filepath : str
            Output VTK file path
        ThickWalls : list
            Thick wall node data
        MembraneFlowDensity : list
            Membrane flow density data
        height : float
            Cell axial length
        list_ghostwalls : list
            Ghost wall indices to exclude
        """
        with open(filepath, "w") as f:
            f.write("# vtk DataFile Version 4.0\n")
            f.write("Membranes geometry 3D\n")
            f.write("ASCII\n\n")
            f.write("DATASET UNSTRUCTURED_GRID\n")
            
            # Points (bottom and top)
            f.write(f"POINTS {len(ThickWalls)*2} float\n")
            for node in ThickWalls:
                f.write(f"{node[3]} {node[4]} 0.0\n")
            for node in ThickWalls:
                f.write(f"{node[3]} {node[4]} {height}\n")
            f.write("\n")
            
            # Cells (membrane quads)
            n_cells = len(ThickWalls) - len(list_ghostwalls)*4
            f.write(f"CELLS {n_cells} {n_cells*5}\n")
            
            for node in ThickWalls:
                if node[1] >= self.network.n_walls:  # Junction
                    if ThickWalls[int(node[5])][1] not in list_ghostwalls:
                        f.write(f"4 {int(node[0])} {int(node[5])} "
                               f"{int(node[5])+len(ThickWalls)} "
                               f"{int(node[0])+len(ThickWalls)}\n")
                    if ThickWalls[int(node[6])][1] not in list_ghostwalls:
                        f.write(f"4 {int(node[0])} {int(node[6])} "
                               f"{int(node[6])+len(ThickWalls)} "
                               f"{int(node[0])+len(ThickWalls)}\n")
            f.write("\n")
            
            # Cell types
            f.write(f"CELL_TYPES {n_cells}\n")
            for _ in range(n_cells):
                f.write("9\n")  # Quad
            f.write("\n")
            
            # Point data
            f.write(f"POINT_DATA {len(ThickWalls)*2}\n")
            f.write("SCALARS TM_flux_(m/s) float\n")
            f.write("LOOKUP_TABLE default\n")
            
            for node in ThickWalls:
                if node[0] < len(MembraneFlowDensity):
                    flux = float(MembraneFlowDensity[int(node[0])]) / \
                          self.SECONDS_PER_DAY / self.CM_PER_METER
                else:
                    # Average of neighboring walls
                    flux = float(MembraneFlowDensity[int(node[5])] + 
                                MembraneFlowDensity[int(node[6])]) / 2 / \
                          self.SECONDS_PER_DAY / self.CM_PER_METER
                f.write(f"{flux}\n")
            
            # Repeat for top
            for node in ThickWalls:
                if node[0] < len(MembraneFlowDensity):
                    flux = float(MembraneFlowDensity[int(node[0])]) / \
                          self.SECONDS_PER_DAY / self.CM_PER_METER
                else:
                    flux = float(MembraneFlowDensity[int(node[5])] + 
                                MembraneFlowDensity[int(node[6])]) / 2 / \
                          self.SECONDS_PER_DAY / self.CM_PER_METER
                f.write(f"{flux}\n")
    
    def write_paraview_vtk_cells(self, filepath, G, soln):
        """
        Write cell water potentials for 2D ParaView visualization.
        
        Parameters:
        -----------
        filepath : str
            Output VTK file path
        G : networkx.Graph
            Network graph
        soln : numpy.ndarray
            Solution vector (water potentials)
        """
        with open(filepath, "w") as f:
            f.write("# vtk DataFile Version 4.0\n")
            f.write("Pressure potential distribution in cells 2D\n")
            f.write("ASCII\n\n")
            f.write("DATASET UNSTRUCTURED_GRID\n")
            
            # Points
            f.write(f"POINTS {len(G.nodes())} float\n")
            for node in G:
                pos = self.network.position[node]
                f.write(f"{float(pos[0])} {float(pos[1])} 0.0\n")
            f.write("\n")
            
            # Cells (vertices for cell nodes only)
            f.write(f"CELLS {self.network.n_cells} {self.network.n_cells*2}\n")
            for node, edges in G.adjacency():
                i = self.network.indice[node]
                if i >= (self.network.n_walls + self.network.n_junctions):
                    f.write(f"1 {i}\n")
            f.write("\n")
            
            # Cell types
            f.write(f"CELL_TYPES {self.network.n_cells}\n")
            for _ in range(self.network.n_cells):
                f.write("1\n")  # Vertex
            f.write("\n")
            
            # Point data
            f.write(f"POINT_DATA {len(G.nodes())}\n")
            f.write("SCALARS Cell_pressure float\n")
            f.write("LOOKUP_TABLE default\n")
            
            for node in G:
                f.write(f"{float(soln[node])}\n")
    
    def write_hormone_concentration(self, filepath, ThickWalls, 
                                   Cell2ThickWalls, nCell2ThickWalls,
                                   soln_C, height):
        """
        Write symplastic hormone concentration for visualization.
        
        Parameters:
        -----------
        filepath : str
            Output VTK file path
        ThickWalls : list
            Thick wall node data
        Cell2ThickWalls : array
            Cell to thick wall mapping
        nCell2ThickWalls : array
            Number of thick walls per cell
        soln_C : numpy.ndarray
            Concentration solution
        height : float
            Cell axial length
        """
        with open(filepath, "w") as f:
            f.write("# vtk DataFile Version 4.0\n")
            f.write("Symplastic hormone concentration\n")
            f.write("ASCII\n\n")
            f.write("DATASET UNSTRUCTURED_GRID\n")
            
            # Points
            f.write(f"POINTS {len(ThickWalls)} float\n")
            for node in ThickWalls:
                f.write(f"{node[3]} {node[4]} {height/200}\n")
            f.write("\n")
            
            # Cells (cell polygons)
            n_vals = int(self.network.n_cells + sum(nCell2ThickWalls))
            f.write(f"CELLS {self.network.n_cells} {n_vals}\n")
            
            for cid in range(self.network.n_cells):
                n = int(nCell2ThickWalls[cid])
                Polygon = Cell2ThickWalls[cid][:n]
                
                # Order polygon nodes
                ranking = self._order_polygon_nodes(Polygon, ThickWalls, n)
                
                string = str(n)
                for id1 in ranking:
                    string += f" {int(id1)}"
                f.write(string + "\n")
            f.write("\n")
            
            # Cell types
            f.write(f"CELL_TYPES {self.network.n_cells}\n")
            for _ in range(self.network.n_cells):
                f.write("6\n")  # Triangle strip
            f.write("\n")
            
            # Point data
            f.write(f"POINT_DATA {len(ThickWalls)}\n")
            f.write("SCALARS Hormone_Symplastic_Relative_Concentration_(-) float\n")
            f.write("LOOKUP_TABLE default\n")
            
            for node in ThickWalls:
                cellnumber = node[2] - (self.network.n_walls + 
                                       self.network.n_junctions)
                concentration = float(soln_C[int(cellnumber + 
                                    (self.network.n_walls + 
                                     self.network.n_junctions))])
                f.write(f"{concentration}\n")
    
    def _order_polygon_nodes(self, Polygon, ThickWalls, n):
        """Order polygon nodes for proper rendering."""
        ranking = []
        ranking.append(int(Polygon[0]))
        ranking.append(ThickWalls[int(ranking[0])][5])
        ranking.append(ThickWalls[int(ranking[0])][6])
        
        for id1 in range(1, n):
            wid1 = ThickWalls[int(ranking[id1])][5]
            wid2 = ThickWalls[int(ranking[id1])][6]
            if wid1 not in ranking:
                ranking.append(wid1)
            if wid2 not in ranking:
                ranking.append(wid2)
        
        return ranking