import xml.etree.ElementTree as ET
from lxml import etree 
import numpy as np
from typing import Dict, List, Any


def parse_cellset(filepath: str) -> Dict[str, Any]:
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
        'cell_to_wall': root.xpath('cells/cell/walls')
    }