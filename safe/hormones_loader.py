import xml.etree.ElementTree as ET
from lxml import etree 
import numpy as np
from typing import Dict, List, Any
from dataclasses import dataclass, field

@dataclass
class HormonesData:
    """Hormones file configuration"""
    # File paths
    hormone_file: str

    
    # Hormone movement parameters
    degrad1: float = 0.0
    diff_pd1: float = 0.0
    diff_pw1: float = 0.0
    d2o1: bool = False

    # Active transport carriers - Use field(default_factory=list) for mutable defaults
    carrier_elems: List[Any] = field(default_factory=list)

    # Symplastic contagion
    sym_zombie0: List[int] = field(default_factory=list)
    sym_cc: List[float] = field(default_factory=list)
    sym_target: List[int] = field(default_factory=list)
    sym_immune: List[int] = field(default_factory=list)

    # Apoplastic contagion
    apo_zombie0: List[int] = field(default_factory=list)
    apo_cc: List[float] = field(default_factory=list)
    apo_target: List[int] = field(default_factory=list)
    apo_immune: List[int] = field(default_factory=list)

    # Contact range
    contact: List[int] = field(default_factory=list)

    def __post_init__(self):
        self._load_hormones()

    def _load_hormones(self):
        """Load hormone and carrier configuration"""
        print('  Loading hormone parameters...')
        root = etree.parse(self.hormone_file).getroot()

        # Hormone movement parameters
        self.degrad1 = float(root.xpath('Hormone_movement/Degradation_constant_H1')[0].get("value"))
        self.diff_pd1 = float(root.xpath('Hormone_movement/Diffusivity_PD_H1')[0].get("value"))
        self.diff_pw1 = float(root.xpath('Hormone_movement/Diffusivity_PW_H1')[0].get("value"))
        self.d2o1 = int(root.xpath('Hormone_movement/H1_D2O')[0].get("flag")) == 1

        # Parse active transport carriers
        self.carrier_elems = root.xpath('Hormone_active_transport/carrier_range/carrier')

        # Parse symplastic contagion
        sym_source_elems = root.xpath('Sym_Contagion/source_range/source')
        self.sym_zombie0 = [int(source.get("id")) for source in sym_source_elems]
        self.sym_cc = [float(source.get("concentration")) for source in sym_source_elems]

        sym_target_elems = root.xpath('Sym_Contagion/target_range/target')
        self.sym_target = [int(target.get("id")) for target in sym_target_elems]

        sym_immune_elems = root.xpath('Sym_Contagion/immune_range/immune')
        self.sym_immune = [int(immune.get("id")) for immune in sym_immune_elems]
        # Parse apoplastic contagion
        apo_source_elems = root.xpath('Apo_Contagion/source_range/source')
        self.apo_zombie0 = [int(source.get("id")) for source in apo_source_elems]
        self.apo_cc = [float(source.get("concentration")) for source in apo_source_elems]

        apo_target_elems = root.xpath('Apo_Contagion/target_range/target')
        self.apo_target = [int(target.get("id")) for target in apo_target_elems]
        
        apo_immune_elems = root.xpath('Apo_Contagion/immune_range/immune')
        self.apo_immune = [int(immune.get("id")) for immune in apo_immune_elems]

        # Parse contact range
        contact_elems = root.xpath('Contactrange/Contact')
        self.contact = [int(contact.get("id")) for contact in contact_elems]