"""
MECHA (Model of Explicit Cross-section Hydraulic Anatomy)
Main entry point for root hydraulic simulations

Refactored for improved readability and maintainability
"""

import os
import numpy as np
from typing import Dict, List, Tuple, Optional

from config_loader import MECHAConfig
from network_builder import NetworkBuilder
from hydraulic_solver import HydraulicSolver
from output_writer import OutputWriter
from utils import update_xml_attributes, set_hydraulic_scenario


def mecha(
    Gen: str = './extdata/Maize_General.xml',
    Geom: str = './extdata/Geometry.xml',
    Hydr: str = './extdata/Hydraulics.xml',
    BC: str = './extdata/Maize_BC_kr.xml',
    Horm: str = './extdata/Maize_Hormones_Carriers.xml',
    cellsetdata: str = './extdata/current_root.xml',
    outdir: str = None
) -> List[float]:
    """
    Main MECHA simulation function.
    
    Parameters
    ----------
    Gen : str
        Path to general configuration XML
    Geom : str
        Path to geometry configuration XML
    Hydr : str
        Path to hydraulics configuration XML
    BC : str
        Path to boundary conditions XML
    Horm : str
        Path to hormone/carrier configuration XML
    cellsetdata : str
        Path to root cross-section cell data XML
    outdir : str, optional
        Output directory path
        
    Returns
    -------
    List[float]
        Radial hydraulic conductivity values for each maturity stage
    """
    
    if outdir is None:
        outdir = os.getcwd()
    
    print('='*60)
    print('MECHA - Model of Explicit Cross-section Hydraulic Anatomy')
    print('='*60)
    
    # 1. Load all configuration files
    print('\n[1/6] Loading configuration...')
    config = MECHAConfig(Gen, Geom, Hydr, BC, Horm, cellsetdata)
    
    # Create output directory
    output_path = os.path.join(outdir, config.plant_name)
    os.makedirs(output_path, exist_ok=True)
    print(f'Output directory: {output_path}')
    
    # 2. Build network structure
    print('\n[2/6] Building network structure...')
    network = NetworkBuilder(config)
    network.build_from_xml()
    
    # Store results for all maturity stages
    kr_results = []
    
    # 3. Loop over maturity stages (barriers)
    print(f'\n[3/6] Processing {len(config.maturity_stages)} maturity stage(s)...')
    
    for maturity_idx, maturity in enumerate(config.maturity_stages):
        barrier = maturity['barrier']
        height = maturity['height']
        
        print(f'\n  Maturity #{maturity_idx}: Barrier={barrier}, Height={height}μm')
        
        # 4. Solve hydraulic equations
        print('  [4/6] Solving hydraulic system...')
        solver = HydraulicSolver(network, config, barrier, height)
        results = solver.solve_all_scenarios()
        
        kr_results.append(results['kr_tot'])
        
        # 5. Perform additional analyses
        if config.sym_contagion or config.apo_contagion:
            print('  [5/6] Running contagion analysis...')
            solver.run_contagion_analysis()
        
        # 6. Write outputs
        print('  [6/6] Writing outputs...')
        writer = OutputWriter(output_path, config, network)
        writer.write_macroscopic_properties(maturity_idx, barrier, results)
        
        if config.paraview:
            writer.write_paraview_files(maturity_idx, barrier, results)
    
    print('\n' + '='*60)
    print('MECHA simulation complete!')
    print(f'Results saved to: {output_path}')
    print('='*60)
    
    return kr_results


if __name__ == "__main__":
    # Example usage
    kr_values = mecha()
    print(f"\nRadial conductivities: {kr_values}")