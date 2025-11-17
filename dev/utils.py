"""
Utility functions for MECHA
Helper functions for XML manipulation and common operations
"""

import xml.etree.ElementTree as ET
import re
import numpy as np
from typing import Dict, List, Union, Optional


def update_xml_attributes(file_path: str, parent_tag: str, child_tag: str, 
                          updates: Dict[str, Union[str, float, int]], 
                          output_path: Optional[str] = None):
    """
    Update one or more attributes of an XML element.
    Works for parent+child (e.g. <kAQPrange><kAQP .../></kAQPrange>)
    or standalone tags (e.g. <km value="1" />).

    Parameters
    ----------
    file_path : str
        Path to the input XML file
    parent_tag : str
        Name of the parent element (e.g., "kAQPrange")
    child_tag : str
        Name of the child element inside the parent (e.g., "kAQP").
        If the tag has no parent, pass the same value for parent_tag and child_tag
    updates : dict
        Dictionary of attribute updates, e.g. {"value": 0.002, "cortex_factor": 0.9}
    output_path : str or None
        Path to save the modified XML file. If None, overwrites the input file
        
    Examples
    --------
    >>> # Update a nested element
    >>> update_xml_attributes(
    ...     'Hydraulics.xml', 
    ...     'kAQPrange', 
    ...     'kAQP',
    ...     {'value': 0.002, 'cortex_factor': 0.9}
    ... )
    
    >>> # Update a standalone element
    >>> update_xml_attributes(
    ...     'Hydraulics.xml',
    ...     'km',
    ...     'km',
    ...     {'value': 1.5}
    ... )
    """
    tree = ET.parse(file_path)
    root = tree.getroot()

    # Handle parent+child or standalone
    if parent_tag == child_tag:
        elem = root.find(f".//{parent_tag}")
    else:
        elem = root.find(f".//{parent_tag}/{child_tag}")

    if elem is None:
        raise ValueError(
            f"No <{child_tag}> element found (parent={parent_tag})."
        )

    # Apply all updates
    for attr, val in updates.items():
        elem.set(attr, str(val))

    # Overwrite or save new file
    if output_path is None:
        output_path = file_path

    tree.write(output_path, encoding="UTF-8", xml_declaration=True)
    print(f"Updated {output_path}")


def set_hydraulic_scenario(xml_path: str, barriers: Union[int, List[int]]):
    """
    Activate one or multiple hydraulic scenarios (Barrier values)
    inside the <Maturityrange> section of a MECHA XML file.
    Keeps <Maturityrange> tags intact, adding new barriers if missing.

    Parameters
    ----------
    xml_path : str
        Path to the MECHA Geometry XML file
    barriers : int or list of int
        Barrier value(s) to activate (0-4)
        - 0: No apoplastic barrier
        - 1: Endodermis radial walls
        - 2: Endodermis with passage cells
        - 3: Endodermis full
        - 4: Endodermis full and exodermis radial walls
        
    Examples
    --------
    >>> # Activate single barrier
    >>> set_hydraulic_scenario('Geometry.xml', 2)
    
    >>> # Activate multiple barriers
    >>> set_hydraulic_scenario('Geometry.xml', [0, 2, 3])
    """
    if isinstance(barriers, int):
        barriers = [barriers]
    
    barriers = sorted(barriers)

    with open(xml_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Extract the <Maturityrange> section
    range_match = re.search(
        r'(<Maturityrange>)(.*?)(</Maturityrange>)', 
        content, 
        re.DOTALL
    )
    if not range_match:
        raise ValueError("No <Maturityrange> section found in XML.")

    start_tag, inner_text, end_tag = range_match.groups()

    # Match existing <Maturity ... /> lines
    maturity_pattern = re.compile(
        r'(\s*)(?:<!--\s*)?(<Maturity\s+Barrier="(\d+)"[^>]*\/>)(?:\s*-->)?'
    )

    existing_barriers = {}
    
    def replacer(match):
        indent, tag, barrier_str = match.groups()
        barrier = int(barrier_str)
        existing_barriers[barrier] = tag
        
        if barrier in barriers:
            return f"{indent}{tag}"  # Activate
        else:
            return f"{indent}<!-- {tag} -->"  # Deactivate

    # Apply activation/deactivation to existing lines
    new_inner = maturity_pattern.sub(replacer, inner_text)

    # Add missing barriers
    indent_match = re.search(r'(\s*)<Maturity', inner_text)
    indent = indent_match.group(1) if indent_match else '    '
    
    for barrier in barriers:
        if barrier not in existing_barriers:
            new_inner += (
                f"\n{indent}<Maturity Barrier=\"{barrier}\" "
                f"height=\"200\" Nlayers=\"1\"/>"
            )

    # Rebuild the <Maturityrange> section
    new_range_section = f"{start_tag}{new_inner}\n{end_tag}"

    # Replace in the full content
    new_content = (
        content[:range_match.start()] + 
        new_range_section + 
        content[range_match.end():]
    )

    # Write back
    with open(xml_path, "w", encoding="utf-8") as f:
        f.write(new_content)
    
    print(f"Updated {xml_path} with barriers: {barriers}")


def calculate_cell_area(coords: List[tuple]) -> float:
    """
    Calculate cell area using the shoelace formula
    
    Parameters
    ----------
    coords : List[tuple]
        List of (x, y) coordinates defining the cell boundary
        
    Returns
    -------
    float
        Cell area in square microns
    """
    if len(coords) < 3:
        return 0.0
    
    # Shoelace formula
    area = 0.0
    n = len(coords)
    for i in range(n):
        j = (i + 1) % n
        area += coords[i][0] * coords[j][1]
        area -= coords[j][0] * coords[i][1]
    
    return abs(area) / 2.0


def order_points_around_centroid(points: np.ndarray) -> np.ndarray:
    """
    Order points around their centroid (for creating valid polygons)
    
    Parameters
    ----------
    points : np.ndarray
        Array of shape (n, 2) with point coordinates
        
    Returns
    -------
    np.ndarray
        Ordered points
    """
    # Calculate centroid
    cx = np.mean(points[:, 0])
    cy = np.mean(points[:, 1])
    
    # Calculate angles from centroid
    angles = np.arctan2(points[:, 1] - cy, points[:, 0] - cx)
    
    # Sort by angle
    sorted_indices = np.argsort(angles)
    
    return points[sorted_indices]


def interpolate_value(val_left: float, val_right: float, x_position: float,
                     x_min: float, x_max: float) -> float:
    """
    Linearly interpolate a value based on horizontal position
    
    Parameters
    ----------
    val_left : float
        Value at left edge
    val_right : float
        Value at right edge
    x_position : float
        Current x position
    x_min : float
        Minimum x position
    x_max : float
        Maximum x position
        
    Returns
    -------
    float
        Interpolated value
    """
    if x_max == x_min:
        return val_left
    
    x_rel = (x_position - x_min) / (x_max - x_min)
    return val_left * (1 - x_rel) + val_right * x_rel


def format_scientific(value: float, precision: int = 3) -> str:
    """
    Format a number in scientific notation
    
    Parameters
    ----------
    value : float
        Value to format
    precision : int
        Number of decimal places
        
    Returns
    -------
    str
        Formatted string
    """
    return f"{value:.{precision}e}"


def validate_xml_file(filepath: str, required_tags: List[str]) -> bool:
    """
    Validate that an XML file contains required tags
    
    Parameters
    ----------
    filepath : str
        Path to XML file
    required_tags : List[str]
        List of required tag names
        
    Returns
    -------
    bool
        True if all required tags are present
    """
    try:
        tree = ET.parse(filepath)
        root = tree.getroot()
        
        for tag in required_tags:
            if root.find(f".//{tag}") is None:
                print(f"Warning: Required tag '{tag}' not found in {filepath}")
                return False
        
        return True
    
    except Exception as e:
        print(f"Error validating {filepath}: {e}")
        return False


def safe_divide(numerator: float, denominator: float, 
               default: float = 0.0) -> float:
    """
    Safely divide two numbers, returning default if denominator is zero
    
    Parameters
    ----------
    numerator : float
        Numerator
    denominator : float
        Denominator
    default : float
        Value to return if division by zero
        
    Returns
    -------
    float
        Result of division or default value
    """
    if abs(denominator) < 1e-10:
        return default
    return numerator / denominator


# Constants for unit conversion
CM_PER_MICRON = 1.0e-4
SECONDS_PER_DAY = 24.0 * 3600.0
CM_PER_METER = 100.0
HPASCAL_PER_MPA = 10000.0


def micron_to_cm(value_micron: float) -> float:
    """Convert microns to centimeters"""
    return value_micron * CM_PER_MICRON


def cm_to_micron(value_cm: float) -> float:
    """Convert centimeters to microns"""
    return value_cm / CM_PER_MICRON


def flow_rate_to_velocity(flow_rate: float, area: float) -> float:
    """
    Convert flow rate to velocity
    
    Parameters
    ----------
    flow_rate : float
        Flow rate in cm³/d
    area : float
        Cross-sectional area in cm²
        
    Returns
    -------
    float
        Velocity in cm/d
    """
    return safe_divide(flow_rate, area)

def compare_nodes(graph1, graph2):
    diffs = {
        "nodes_in_G1_only": [],
        "nodes_in_G2_only": [],
        "different_node_attributes": {}
    }

    # Compare node sets
    nodes_g1 = set(graph1.nodes())
    nodes_g2 = set(graph2.nodes())

    diffs["nodes_in_G1_only"] = list(nodes_g1 - nodes_g2)
    diffs["nodes_in_G2_only"] = list(nodes_g2 - nodes_g1)

    # Compare attributes for common nodes
    common_nodes = nodes_g1.intersection(nodes_g2)
    for node in common_nodes:
        attrs_g1 = graph1.nodes[node]
        attrs_g2 = graph2.nodes[node]
        if attrs_g1 != attrs_g2:
            diffs["different_node_attributes"][node] = {
                "G1_attrs": attrs_g1,
                "G2_attrs": attrs_g2
            }
    return diffs
  
