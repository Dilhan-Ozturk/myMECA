import numpy as np 
from numpy.random import *  # for random sampling
import scipy.linalg as slin #Linear algebra functions
from numpy import genfromtxt #Load data from a text file, with missing values handled as specified.
import pylab #Found in the package pyqt
from pylab import *  # for plotting
import networkx as nx
import sys, os
import re
from lxml import etree
import xml.etree.ElementTree as ET
from pylab import *  # for plotting
from numpy import genfromtxt #Load data from a text file, with missing values handled as specified.
from numpy.random import *  # for random sampling
import argparse
from function import *
from config_loader import MECHAConfig, parse_cellset_xml
from network_builder import NetworkBuilder
from utils import *


def mecha(Gen='./extdata/Maize_General.xml',#'Arabido1_General.xml' #'MilletLR3_General.xml' #
          Geom='./extdata/Geometry.xml',#'Arabido4_Geometry_BBSRC.xml' #'Maize2_Geometry.xml' #''MilletLR3_Geometry.xml'    #'Wheat1_Nodal_Geometry_aerenchyma.xml' #'Maize1_Geometry.xml' #
          Hydr='./extdata/Hydraulics.xml', #'Arabido1_Hydraulics_ERC.xml' #'MilletLR3_Hydraulics.xml' #'Test_Hydraulics.xml' #
          BC='./extdata/Maize_BC_kr.xml', #'Arabido4_BC_BBSRC2.xml' #'Arabido1_BC_Emily.xml' #'Arabido3_BC_BBSRC.xml' #'Maize_BC_SoluteAna_krOsmo.xml'#'Maize_BC_OSxyl_hetero.xml' #'Arabido1_BC_Emily.xml' #'BC_Test.xml' #'Maize_BC_Plant_phys.xml'
          Horm='./extdata/Maize_Hormones_Carriers.xml',
          cellsetdata='./extdata/current_root.xml',#present in Geometry.xml
          outdir=os.getcwd()): 
    
    path=outdir
     # 1. Import config files
    print('\n[1/6] Importing config files...')
    config = MECHAConfig(Gen,
                        Geom, 
                        Hydr, 
                        BC, 
                        Horm, 
                        cellsetdata)

    # Create output directory
    output_path = os.path.join(outdir, config.plant_name)
    os.makedirs(output_path, exist_ok=True)

    # 2. Build network structure
    print(config.apo_contagion)
    print('\n[2/6] Building network structure...')
    network = NetworkBuilder(config)
    network.build_from_xml()
    G = network.graph
    
    newpath=outdir + '/' + config.plant_name + '/'
    if not os.path.exists(newpath):
        os.makedirs(newpath)

    #Start the loop of hydraulic properties
    for h in range(config.n_hydraulics):
        #print('   ')
        #print('Hydraulic network #'+str(h))
        newpath=outdir+'/'+config.plant_name+'/'
        print("hydraulics")
        if not os.path.exists(newpath):
            os.makedirs(newpath)
        
        #System solving
        Psi_xyl=empty((config.n_maturity,config.n_scenarios))
        Psi_xyl[:]=np.nan
        dPsi_xyl=empty((config.n_maturity,config.n_scenarios))
        dPsi_xyl[:]=np.nan
        iEquil_xyl=np.nan #index of the equilibrium root xylem pressure scenario
        Flow_xyl=empty((len(network.xylem_cells)+1,config.n_scenarios))
        Flow_xyl[:]=np.nan
        Psi_sieve=empty((config.n_maturity,config.n_scenarios))
        Psi_sieve[:]=np.nan
        dPsi_sieve=empty((config.n_maturity,config.n_scenarios))
        dPsi_sieve[:]=np.nan
        iEquil_sieve=np.nan #index of the equilibrium root phloem pressure scenario
        Flow_sieve=empty((len(network.sieve_cells)+1,config.n_scenarios))
        Flow_sieve[:]=np.nan
        Os_sieve=zeros((1,config.n_scenarios))
        Os_cortex=zeros((1,config.n_scenarios))
        Os_hetero=zeros((1,config.n_scenarios))
        s_factor=zeros((1,config.n_scenarios))
        s_hetero=zeros((1,config.n_scenarios))
        Elong_cell=zeros((1,config.n_scenarios))
        Elong_cell_side_diff=zeros((1,config.n_scenarios))
        UptakeLayer_plus=zeros((int(network.r_discret[0].item()),config.n_maturity,config.n_scenarios))
        UptakeLayer_minus=zeros((int(network.r_discret[0].item()),config.n_maturity,config.n_scenarios))
        Q_xyl_layer=zeros((int(network.r_discret[0].item()),config.n_maturity,config.n_scenarios))
        Q_sieve_layer=zeros((int(network.r_discret[0].item()),config.n_maturity,config.n_scenarios))
        Q_elong_layer=zeros((int(network.r_discret[0].item()),config.n_maturity,config.n_scenarios))
        STFmb=zeros((network.n_membrane,config.n_maturity))
        STFcell_plus=zeros((network.n_cells,config.n_maturity))
        STFcell_minus=zeros((network.n_cells,config.n_maturity))
        STFlayer_plus=zeros((int(network.r_discret[0].item()),config.n_maturity))
        STFlayer_minus=zeros((int(network.r_discret[0].item()),config.n_maturity))
        PsiCellLayer=zeros((int(network.r_discret[0].item()),config.n_maturity,config.n_scenarios))
        PsiWallLayer=zeros((int(network.r_discret[0].item()),config.n_maturity,config.n_scenarios))
        OsCellLayer=zeros((int(network.r_discret[0].item()),config.n_maturity,config.n_scenarios))
        nOsCellLayer=zeros((int(network.r_discret[0].item()),config.n_maturity,config.n_scenarios))
        OsWallLayer=zeros((int(network.r_discret[0].item()),config.n_maturity,config.n_scenarios))
        nOsWallLayer=zeros((int(network.r_discret[0].item()),config.n_maturity,config.n_scenarios)) #Used for averaging OsWallLayer
        NWallLayer=zeros((int(network.r_discret[0].item()),config.n_maturity,config.n_scenarios))
        #UptakeDistri_plus=zeros((40,3,8))#the size will be adjusted, but won't be more than 40. Dimension 1: radial position, 2: compartment, 3: scenario
        #UptakeDistri_minus=zeros((40,3,8))
        Q_tot=zeros((config.n_maturity,config.n_scenarios)) #(cm^3/d) Total flow rate at root surface
        kr_tot=zeros((config.n_maturity,1))
        Hydropatterning=empty((config.n_maturity,config.n_scenarios))
        Hydropatterning[:]=np.nan
        Hydrotropism=empty((config.n_maturity,config.n_scenarios))
        Hydrotropism[:]=np.nan
        
        iMaturity=-1 #Iteration index in the Barriers loop
        for stage in range(len(config.maturity_stages)):
            Barrier=config.maturity_stages[stage]['barrier'] #Apoplastic barriers (0: No apoplastic barrier, 1:Endodermis radial walls, 2:Endodermis with passage cells, 3: Endodermis full, 4: Endodermis full and exodermis radial walls)
            height=config.maturity_stages[stage]['height'] #Cell length in the axial direction (microns)
            
            #Index for barriers loop
            iMaturity+=1
            print('Maturity #'+str(iMaturity)+' with apoplastic barrier type #'+str(Barrier))
            
            #Scenarios concern boundary conditions only
            count=0
            #print('Scenario #'+str(count))
            
            #Soil, xylem, and phloem pressure potentials
            if not isnan(config.scenarios[0].get("flow_sieve")):
                if isnan(config.scenarios[0].get("pressure_xyl")) and isnan(config.scenarios[0].get("delta_p_xyl")):
                    tot_flow=config.scenarios[0].get("flow_sieve")
                    sum_area=0
                    i=1
                    for cid in network.xylem_cells:
                        area=network.cell_areas[cid-(network.n_walls + network.n_junctions)]
                        Flow_xyl[i][0]=tot_flow*area
                        sum_area+=area
                        i+=1
                    i=1
                    for cid in network.xylem_cells:
                        Flow_xyl[i][0]/=sum_area #Total xylem flow rate partitioned proportionnally to xylem cross-section area
                        i+=1
                    if config.scenarios[0].get("flow_sieve")==0.0:
                        iEquil_xyl=0
                else:
                    print('Error: Cannot have both pressure and flow BC at xylem boundary')
            elif not isnan(config.scenarios[0].get("delta_p_xyl")):
                if isnan(config.scenarios[0].get("pressure_xyl")):
                    if not isnan(iEquil_xyl):
                        config.scenarios[0]["pressure_xyl"]=Psi_xyl[iMaturity][iEquil_xyl]+config.scenarios[0].get("delta_p_xyl")
                    else:
                        print('Error: Cannot have xylem pressure change relative to equilibrium without having a prior scenario with equilibrium xylem boundary condition')
                else:
                    print('Error: Cannot have both pressure and pressure change relative to equilibrium as xylem boundary condition')
            
            if not isnan(config.scenarios[0].get("flow_sieve")):
                if isnan(config.scenarios[0].get("pressure_sieve")) and isnan(config.scenarios[0].get("delta_p_sieve")):
                    tot_flow=config.scenarios[0].get("flow_sieve")
                    sum_area=0
                    i=1
                    for cid in listprotosieve:
                        area=network.cell_areas[cid - (network.n_walls + network.n_junctions)]
                        Flow_sieve[i][0]=tot_flow*area
                        sum_area+=area
                        i+=1
                    i=1
                    for cid in listprotosieve:
                        Flow_sieve[i][0]/=sum_area #Total phloem flow rate partitioned proportionnally to phloem cross-section area
                        i+=1
                    if config.scenarios[0].get("flow_sieve")==0.0:
                        iEquil_sieve=0
                else:
                    print('Error: Cannot have both pressure and flow BC at phloem boundary')
            elif not isnan(config.scenarios[0].get("delta_p_sieve")):
                if isnan(config.scenarios[0].get("pressure_sieve")):
                    if not isnan(iEquil_sieve):
                        config.scenarios[0]["pressure_sieve"]=Psi_sieve[iMaturity][iEquil_sieve]+config.scenarios[0].get("delta_p_sieve")
                    else:
                        print('Error: Cannot have phloem pressure change relative to equilibrium without having a prior scenario with equilibrium phloem boundary condition')
                else:
                    print('Error: Cannot have both pressure and pressure change relative to equilibrium as phloem boundary condition')
            
            #Soil - root contact limit
            if len(config.xcontactrange) == config.n_hydraulics:
                Xcontact=float(config.xcontactrange[h].get("value")) #(micrometers) X threshold coordinate of contact between soil and root (lower X not in contact with soil)
            elif len(config.xcontactrange) == 1:
                Xcontact=float(config.xcontactrange[0].get("value"))
            else:
                Xcontact=float(config.xcontactrange[int(h/(config.n_kaqp*config.n_kpl*config.n_kw*config.n_kw_barrier))].get("value")) #OK
            
            #Cell wall hydraulic conductivity
            if config.n_kw == config.n_hydraulics:
                kw = float(config.kw_elems[h].get("value"))
            elif config.n_kw == 1:
                kw = float(config.kw_elems[0].get("value"))
            else:
                kw = float(config.kw_elems[int(h/(config.n_kaqp*config.n_kpl))%config.n_kw].get("value"))
            if config.n_kw_barrier == config.n_hydraulics:
                kw_barrier = float(config.kw_barrier_elems[h].get("value"))
            elif config.n_kw_barrier == 1:
                kw_barrier = float(config.kw_barrier_elems[0].get("value"))
            else:
                kw_barrier = float(config.kw_barrier_elems[int(h/(config.n_kaqp*config.n_kpl*config.n_kw))%config.n_kw_barrier].get("value"))
            #kw_barrier = kw/10.0
            
            if Barrier==0: #No Casparian strip ###Yet to come: Punctured Casparian strip as in Steudle et al. (1993)
                kw_endo_endo=kw
                kw_puncture=kw
                kw_exo_exo=kw #(cm^2/hPa/d) hydraulic conductivity of the suberised walls between exodermis cells
                kw_cortex_cortex=kw
                kw_endo_peri=kw #(cm^2/hPa/d) hydraulic conductivity of the walls between endodermis and pericycle cells
                kw_endo_cortex=kw #(cm^2/hPa/d) hydraulic conductivity of the walls between endodermis and pericycle cells
                kw_passage=kw #(cm^2/hPa/d) hydraulic conductivity of passage cells tangential walls
            elif Barrier==1: #Endodermis radial walls
                kw_endo_endo=kw_barrier
                kw_exo_exo=kw #(cm^2/hPa/d) hydraulic conductivity of the suberised walls between exodermis cells
                kw_cortex_cortex=kw
                kw_endo_peri=kw #(cm^2/hPa/d) hydraulic conductivity of the walls between endodermis and pericycle cells
                kw_endo_cortex=kw #(cm^2/hPa/d) hydraulic conductivity of the walls between endodermis and pericycle cells
                kw_passage=kw #(cm^2/hPa/d) hydraulic conductivity of passage cells tangential walls
            elif Barrier==2: #Endodermis with passage cells
                kw_endo_endo=kw_barrier
                kw_exo_exo=kw #(cm^2/hPa/d) hydraulic conductivity of the suberised walls between exodermis cells
                kw_cortex_cortex=kw
                kw_endo_peri=kw_barrier #(cm^2/hPa/d) hydraulic conductivity of the walls between endodermis and pericycle cells
                kw_endo_cortex=kw_barrier #(cm^2/hPa/d) hydraulic conductivity of the walls between endodermis and pericycle cells
                kw_passage=kw #(cm^2/hPa/d) hydraulic conductivity of passage cells tangential walls
            elif Barrier==3: #Endodermis full
                kw_endo_endo=kw_barrier
                kw_exo_exo=kw #(cm^2/hPa/d) hydraulic conductivity of the suberised walls between exodermis cells
                kw_cortex_cortex=kw
                kw_endo_peri=kw_barrier #(cm^2/hPa/d) hydraulic conductivity of the walls between endodermis and pericycle cells
                kw_endo_cortex=kw_barrier #(cm^2/hPa/d) hydraulic conductivity of the walls between endodermis and pericycle cells
                kw_passage=kw_barrier #(cm^2/hPa/d) hydraulic conductivity of passage cells tangential walls
            elif Barrier==4: #Endodermis full and exodermis radial walls
                kw_endo_endo=kw_barrier
                kw_exo_exo=kw_barrier #(cm^2/hPa/d) hydraulic conductivity of the suberised walls between exodermis cells
                kw_cortex_cortex=kw
                kw_endo_peri=kw_barrier #(cm^2/hPa/d) hydraulic conductivity of the walls between endodermis and pericycle cells
                kw_endo_cortex=kw_barrier #(cm^2/hPa/d) hydraulic conductivity of the walls between endodermis and pericycle cells
                kw_passage=kw_barrier #(cm^2/hPa/d) hydraulic conductivity of passage cells tangential walls

            #Plasmodesmatal hydraulic conductance
            if config.n_kpl == config.n_hydraulics:
                iPD=h
            elif config.n_kpl == 1:
                iPD=0
            else:
                iPD=int(h/config.n_kaqp)%config.n_kpl
            Kpl = float(config.kpl_elems[iPD].get("value"))
            
            #Contribution of aquaporins to membrane hydraulic conductivity
            if config.n_kaqp == config.n_hydraulics:
                iAQP=h
            elif config.n_kaqp == 1:
                iAQP=0
            else:
                iAQP=h%config.n_kaqp
            kaqp = float(config.kaqp_elems[iAQP].get("value"))
            kaqp_stele= kaqp*float(config.kaqp_elems[iAQP].get("stele_factor"))
            kaqp_endo= kaqp*float(config.kaqp_elems[iAQP].get("endo_factor"))
            kaqp_exo= kaqp*float(config.kaqp_elems[iAQP].get("exo_factor"))
            kaqp_epi= kaqp*float(config.kaqp_elems[iAQP].get("epi_factor"))
            kaqp_cortex= kaqp*float(config.kaqp_elems[iAQP].get("cortex_factor"))

            #Calculate parameter a
            if config.ratio_cortex==1: #Uniform AQP activity in all cortex membranes
                a_cortex=0.0  #(1/hPa/d)
                b_cortex=kaqp_cortex #(cm/hPa/d)
            else:
                tot_surf_cortex=0.0 #Total membrane exchange surface in cortical cells (square centimeters)
                temp=0.0 #Term for summation (cm3)
                for w in Cell2Wall_loop: #Loop on cells. Cell2Wall_loop contains cell wall groups info (one group by cell)
                    cellnumber1 = int(w.getparent().get("id")) #Cell ID number
                    for r in w: #Loop for wall elements around the cell
                        wid= int(r.get("id")) #Cell wall ID
                        if G.nodes[(network.n_walls + network.n_junctions) + cellnumber1]['cgroup']==4: #Cortex
                            dist_cell=sqrt(square(network.position[wid][0]-network.position[(network.n_walls + network.n_junctions)+cellnumber1][0])+square(network.position[wid][1]-network.position[(network.n_walls + network.n_junctions)+cellnumber1][1])) #distance between wall node and cell node (micrometers)
                            surf=(height+dist_cell)*network.length[wid]*1.0E-08 #(square centimeters)
                            temp+=surf*1.0E-04*(network.distance_to_center[wid]+(config.ratio_cortex*dmax_cortex-dmin_cortex)/(1-config.ratio_cortex))
                            tot_surf_cortex+=surf
                a_cortex=kaqp_cortex*tot_surf_cortex/temp  #(1/hPa/d)
                b_cortex=a_cortex*1.0E-04*(config.ratio_cortex*dmax_cortex-dmin_cortex)/(1-config.ratio_cortex) #(cm/hPa/d)
            
            ######################
            ##Filling the matrix##
            ######################
            
            matrix_W = np.zeros(((len(G)),len(G))) #Initializes the Doussan matrix
            if config.apo_contagion==2 and config.sym_contagion==2:
                matrix_C = np.zeros(((len(G)),len(G))) #Initializes the matrix of convection diffusion
                rhs_C = np.zeros((len(G),1)) #Initializing the right-hand side matrix of solute apoplastic concentrations
                for i in range(network.n_walls):
                    if i in Apo_w_Zombies0:
                        matrix_C[i][i]=1.0
                        rhs_C[i][0]=Apo_w_cc[Apo_w_Zombies0.index(i)] #1.0 #Concentration in source wall i defined in Geom
                    else: #Decomposition rate (mol decomp/mol-day * cm^3)
                        matrix_C[i][i]-=config.degrad1*1.0E-12*(lat_dists[i][0]*config.thickness*network.length[i]+height*config.thickness*network.length[i]/2-square(config.thickness)*network.length[i])
                for j in range(network.n_walls,(network.n_walls + network.n_junctions)):
                    if j in Apo_j_Zombies0:
                        matrix_C[j][j]=1.0
                        rhs_C[j][0]=Apo_j_cc[Apo_j_Zombies0.index(j)] #1.0 #Concentration in source junction j defined in Geom
                    else: #Decomposition rate (mol decomp/mol-day * cm^3)
                        matrix_C[j][j]-=config.degrad1*1.0E-12*height*config.thickness*network.length[j]/2
                for cellnumber1 in range(network.n_cells):
                    if cellnumber1 in config.sym_zombie0:
                        matrix_C[(network.n_walls + network.n_junctions)+cellnumber1][(network.n_walls + network.n_junctions)+cellnumber1]=1.0
                        rhs_C[(network.n_walls + network.n_junctions)+cellnumber1][0]=config.sym_cc[config.sym_zombie0.index(cellnumber1)] #1.0 #Concentration in source protoplasts defined in Geom
                    else: #Decomposition rate (mol decomp/mol-day * cm^3)
                        matrix_C[(network.n_walls + network.n_junctions)+cellnumber1][(network.n_walls + network.n_junctions)+cellnumber1]-=config.degrad1*1.0E-12*network.cell_areas[cellnumber1]*height
            elif config.apo_contagion==2:
                matrix_ApoC = np.zeros((((network.n_walls + network.n_junctions)),(network.n_walls + network.n_junctions))) #Initializes the matrix of convection
                rhs_ApoC = np.zeros(((network.n_walls + network.n_junctions),1)) #Initializing the right-hand side matrix of solute apoplastic concentrations
                for i in range(network.n_walls):
                    if i in Apo_w_Zombies0:
                        matrix_ApoC[i][i]=1.0
                        rhs_ApoC[i][0]=Apo_w_cc[Apo_w_Zombies0.index(i)] #1 #Concentration in source wall i equals 1 by default
                    else: #Decomposition rate (mol decomp/mol-day * cm^3)
                        matrix_ApoC[i][i]-=config.degrad1*1.0E-12*(lat_dists[i][0]*config.thickness*network.length[i]+height*config.thickness*network.length[i]/2-square(config.thickness)*network.length[i])
                for j in range(network.n_walls,(network.n_walls + network.n_junctions)):
                    if j in Apo_j_Zombies0:
                        matrix_ApoC[j][j]=1.0
                        rhs_ApoC[j][0]=Apo_j_cc[Apo_j_Zombies0.index(j)] #1 #Concentration in source junction j equals 1 by default
                    else: #Decomposition rate (mol decomp/mol-day * cm^3)
                        matrix_ApoC[j][j]-=config.degrad1*1.0E-12*height*config.thickness*network.length[j]/2
            elif config.sym_contagion==2:
                matrix_SymC = np.zeros(((network.n_cells),network.n_cells)) #Initializes the matrix of convection
                rhs_SymC = np.zeros((network.n_cells,1)) #Initializing the right-hand side matrix of solute symplastic concentrations
                for cellnumber1 in range(network.n_cells):
                    if cellnumber1 in config.sym_zombie0:
                        matrix_SymC[cellnumber1][cellnumber1]=1.0
                        rhs_SymC[cellnumber1][0]=config.sym_cc[config.sym_zombie0.index(cellnumber1)] #1 #Concentration in source protoplasts equals 1 by default
                    else: #Decomposition rate (mol decomp/mol-day * cm^3)
                        matrix_SymC[cellnumber1][cellnumber1]-=config.degrad1*1.0E-12*network.cell_areas[cellnumber1]*height
            
            Kmb=zeros((network.n_membrane,1)) #Stores membranes conductances for the second K loop
            jmb=0 #Index of membrane in Kmb
            K_axial=zeros(((network.n_walls + network.n_junctions) + network.n_cells,1)) #Vector of apoplastic and plasmodesmatal axial conductances
            if Barrier>0: #K_xyl_spec calculated from Poiseuille law (cm^3/hPa/d)
                for cid in network.xylem_cells:
                    K_axial[cid]=network.cell_areas[cid-(network.n_walls + network.n_junctions)]**2/(8*3.141592*height*1.0E-05/3600/24)*1.0E-12 #(micron^4/micron)->(cm^3) & (1.0E-3 Pa.s)->(1.0E-05/3600/24 hPa.d) 
                K_xyl_spec=sum(K_axial)*height/1.0E04
                for cid in network.sieve_cells:
                    K_axial[cid]=network.cell_areas[cid-(network.n_walls + network.n_junctions)]**2/(8*3.141592*height*1.0E-05/3600/24)*1.0E-12 #(micron^4/micron)->(cm^3) & (1.0E-3 Pa.s)->(1.0E-05/3600/24 hPa.d) 
            else:
                K_xyl_spec=0.0
            list_ghostwalls=[] #"Fake walls" not to be displayed
            list_ghostjunctions=[] #"Fake junctions" not to be displayed
            nGhostJunction2Wall=0
            #Adding matrix components at cell-cell, cell-wall, and wall-junction connections
            for node, edges in G.adjacency() : #adjacency_iter returns an iterator of (node, adjacency dict) tuples for all nodes. This is the fastest way to look at every edge. For directed graphs, only outgoing adjacencies are included.
                i=network.indice[node] #Node ID number
                #Here we count surrounding cell types in order to position apoplastic barriers
                count_endo=0 #total number of endodermis cells around the wall
                count_xyl=0 #total number of xylem cells around the wall
                count_stele_overall=0 #total number of stelar cells around the wall
                count_exo=0 #total number of exodermis cells around the wall
                count_epi=0 #total number of epidermis cells around the wall
                count_cortex=0 #total number of cortical cells around the wall
                count_passage=0 #total number of passage cells around the wall
                count_interC=0 #total number of intercellular spaces around the wall
                if i<network.n_walls: #wall ID
                    for neighboor, eattr in edges.items(): #Loop on connections (edges)
                        if eattr['path'] == 'membrane': #Wall connection
                            if any(config.passage_cell_ids==array((network.indice[neighboor])-(network.n_walls + network.n_junctions))):
                                count_passage+=1
                            if any(config.intercellular_ids==array((network.indice[neighboor])-(network.n_walls + network.n_junctions))):
                                count_interC+=1
                                if count_interC==2 and i not in list_ghostwalls:
                                    list_ghostwalls.append(i)
                            if G.nodes[neighboor]['cgroup']==3:#Endodermis
                                count_endo+=1
                            elif G.nodes[neighboor]['cgroup']==13 or G.nodes[neighboor]['cgroup']==19 or G.nodes[neighboor]['cgroup']==20:#Xylem cell or vessel
                                count_xyl+=1
                                if (count_xyl==2 and config.xylem_pieces) and i not in list_ghostwalls:
                                    list_ghostwalls.append(i)
                            elif G.nodes[neighboor]['cgroup']>4:#Pericycle or stele but not xylem
                                count_stele_overall+=1
                            elif G.nodes[neighboor]['cgroup']==4:#Cortex
                                count_cortex+=1
                            elif G.nodes[neighboor]['cgroup']==1:#Exodermis
                                count_exo+=1
                            elif G.nodes[neighboor]['cgroup']==2:#Epidermis
                                count_epi+=1
                
                for neighboor, eattr in edges.items(): #Loop on connections (edges)
                    j = (network.indice[neighboor]) #neighbouring node number
                    if j > i: #Only treating the information one way to save time
                        path = eattr['path'] #eattr is the edge attribute (i.e. connection type)
                        if path == 'wall': #Wall connection
                            #K = eattr['kw']*1.0E-04*((eattr['lat_dist']+height)*eattr['thickness']-square(eattr['thickness']))/eattr['length'] #Junction-Wall conductance (cm^3/hPa/d)
                            temp=1.0E-04*((eattr['lat_dist']+height)*config.thickness-square(config.thickness))/eattr['length'] #Wall section to length ratio (cm)
                            if (count_interC>=2 and Barrier>0) or (count_xyl==2 and config.xylem_pieces): #"Fake wall" splitting an intercellular space or a xylem cell in two
                                K = 1.0E-16 #Non conductive
                                if j not in list_ghostjunctions:
                                    fakeJ=True
                                    for ind in range(int(nJunction2Wall[j-network.n_walls])):
                                        if Junction2Wall[j-network.n_walls][ind] not in list_ghostwalls:
                                            fakeJ=False #If any of the surrounding walls is real, the junction is real
                                    if fakeJ:
                                        list_ghostjunctions.append(j)
                                        nGhostJunction2Wall+=int(nJunction2Wall[j-network.n_walls])+2 #The first and second thick junction nodes each appear twice in the text file for Paraview
                            elif count_cortex>=2: #wall between two cortical cells
                                K = kw_cortex_cortex*temp #Junction-Wall conductance (cm^3/hPa/d)
                            elif count_endo>=2: #wall between two endodermis cells
                                K = kw_endo_endo*temp #Junction-Wall conductance (cm^3/hPa/d)  #(height*eattr['thickness'])/eattr['length']#
                            elif count_stele_overall>0 and count_endo>0: #wall between endodermis and pericycle
                                if count_passage>0:
                                    K = kw_passage*temp #(height*eattr['thickness'])/eattr['length']#
                                else:
                                    K = kw_endo_peri*temp #Junction-Wall conductance (cm^3/hPa/d) #(height*eattr['thickness'])/eattr['length']#
                            elif count_stele_overall==0 and count_endo==1: #wall between endodermis and cortex
                                if count_passage>0:
                                    K = kw_passage*temp  #(height*eattr['thickness'])/eattr['length']#
                                else:
                                    K = kw_endo_cortex*temp #Junction-Wall conductance (cm^3/hPa/d)  #(height*eattr['thickness'])/eattr['length']#
                            elif count_exo>=2: #wall between two exodermis cells
                                K = kw_exo_exo*temp #Junction-Wall conductance (cm^3/hPa/d)  #(height*eattr['thickness'])/eattr['length']#
                            else: #other walls
                                K = kw*temp #Junction-Wall conductance (cm^3/hPa/d)  #(height*eattr['thickness'])/eattr['length']#
                            ########Solute fluxes (diffusion across walls and junctions)
                            if config.apo_contagion==2:
                                temp_factor=1.0 #Factor for reduced diffusion across impermeable walls
                                if (count_interC>=2 and Barrier>0) or (count_xyl==2 and config.xylem_pieces): #"fake wall" splitting an intercellular space or a xylem cell in two
                                    temp_factor=1.0E-16 #Correction
                                elif count_endo>=2:
                                    temp_factor=kw_endo_endo/kw
                                elif count_stele_overall>0 and count_endo>0: #wall between endodermis and pericycle
                                    if count_passage>0:
                                        temp_factor=kw_passage/kw #(height*eattr['thickness'])/eattr['length']#
                                    else:
                                        temp_factor=kw_endo_peri/kw #Junction-Wall conductance (cm^3/hPa/d) #(height*eattr['thickness'])/eattr['length']#
                                elif count_stele_overall==0 and count_endo==1: #wall between endodermis and cortex
                                    if count_passage>0:
                                        temp_factor=kw_passage/kw  #(height*eattr['thickness'])/eattr['length']#
                                    else:
                                        temp_factor=kw_endo_cortex/kw #Junction-Wall conductance (cm^3/hPa/d)  #(height*eattr['thickness'])/eattr['length']#
                                elif count_exo>=2: #wall between two exodermis cells
                                    temp_factor=kw_exo_exo/kw #Junction-Wall conductance (cm^3/hPa/d)  #(height*eattr['thickness'])/eattr['length']#
                                DF=temp*temp_factor*config.diff_pw1 #"Diffusive flux" (cm^3/d) temp is the section to length ratio of the wall to junction path
                                if config.sym_contagion==2: #Sym & Apo contagion
                                    if i not in Apo_w_Zombies0:
                                        matrix_C[i][i] -= DF
                                        matrix_C[i][j] += DF #Convection will be dealt with further down
                                    if j not in Apo_j_Zombies0:
                                        matrix_C[j][j] -= DF #temp_factor is the factor for reduced diffusion across impermeable walls
                                        matrix_C[j][i] += DF
                                else: #Only Apo contagion
                                    if i not in Apo_w_Zombies0:
                                        matrix_ApoC[i][i] -= DF
                                        matrix_ApoC[i][j] += DF
                                    if j not in Apo_j_Zombies0:
                                        matrix_ApoC[j][j] -= DF #Convection will be dealt with further down
                                        matrix_ApoC[j][i] += DF
                        elif path == "membrane": #Membrane connection
                            #K = (eattr['kmb']+eattr['kaqp'])*1.0E-08*(height+eattr['dist'])*eattr['length']
                            if config.apo_contagion==2 and config.sym_contagion==2:
                                for carrier in config.carrier_elems:
                                    if int(carrier.get("tissue"))==G.nodes[j]['cgroup']:
                                        #Condition is that the protoplast (j) is an actual protoplast with membranes
                                        if j-(network.n_walls + network.n_junctions) not in config.intercellular_ids and not (Barrier>0 and (G.nodes[j]['cgroup']==13 or G.nodes[j]['cgroup']==19 or G.nodes[j]['cgroup']==20)):
                                            temp=float(carrier.get("constant"))*(height+eattr['dist'])*eattr['length'] #Linear transport constant (Vmax/KM) [liter/day^-1/micron^-2] * membrane surface [micron²]
                                            if int(carrier.get("direction"))==1: #Influx transporter
                                                if j-(network.n_walls + network.n_junctions) not in config.sym_zombie0: #Concentration not affected if set as boundary condition
                                                    matrix_C[j][i] += temp #Increase of concentration in protoplast (j) depends on concentration in cell wall (i)
                                                if i not in Apo_w_Zombies0: #Concentration not affected if set as boundary condition
                                                    matrix_C[i][i] -= temp #Decrease of concentration in apoplast (i) depends on concentration in apoplast (i)
                                            elif int(carrier.get("direction"))==int(-1): #Efflux transporter
                                                if j-(network.n_walls + network.n_junctions) not in config.sym_zombie0: #Concentration not affected if set as boundary condition
                                                    matrix_C[j][j] -= temp #Increase of concentration in protoplast (j) depends on concentration in protoplast (j)
                                                if i not in Apo_w_Zombies0: #Concentration not affected if set as boundary condition
                                                    matrix_C[i][j] += temp #Decrease of concentration in apoplast (i) depends on concentration in protoplast (j)
                                            else:
                                                error('Error, carrier direction is either 1 (influx) or -1 (efflux), please correct in *_Hormones_Carriers_*.xml')
                            if G.nodes[j]['cgroup']==1: #Exodermis
                                kaqp=kaqp_exo
                            elif G.nodes[j]['cgroup']==2: #Epidermis
                                kaqp=kaqp_epi
                            elif G.nodes[j]['cgroup']==3: #Endodermis
                                kaqp=kaqp_endo
                            elif G.nodes[j]['cgroup']==13 or G.nodes[j]['cgroup']==19 or G.nodes[j]['cgroup']==20: #xylem cell or vessel
                                if Barrier>0: #Xylem vessel
                                    kaqp=kaqp_stele*10000 #No membrane resistance because no membrane
                                    if config.apo_contagion==2 and config.sym_contagion==2:
                                        #Diffusion between mature xylem vessels and their walls
                                        temp=1.0E-04*(network.length[i]*height)/config.thickness #Section to length ratio (cm) for the xylem wall
                                        if i not in Apo_w_Zombies0:
                                            matrix_C[i][i] -= temp*config.diff_pw1
                                            matrix_C[i][j] += temp*config.diff_pw1
                                        if j-(network.n_walls + network.n_junctions) not in config.sym_zombie0: #Mature xylem vessels are referred to as cells, so they are on the Sym side even though they are part of the apoplast
                                            matrix_C[j][j] -= temp*config.diff_pw1
                                            matrix_C[j][i] += temp*config.diff_pw1
                                else:
                                    kaqp=kaqp_stele
                            elif G.nodes[j]['cgroup']>4: #Stele and pericycle but not xylem
                                kaqp=kaqp_stele
                            elif (j-(network.n_walls + network.n_junctions) in config.intercellular_ids) and Barrier>0: #the neighbour is an intercellular space "cell". Between j and i connected by a membrane, only j can be cell because j>i
                                kaqp=config.k_interc
                                #No carrier
                            elif G.nodes[j]['cgroup']==4: #Cortex
                                kaqp=float(a_cortex*network.distance_to_center[i].item()*1.0E-04+b_cortex) #AQP activity (cm/hPa/d)
                                if kaqp < 0:
                                    error('Error, negative kaqp in cortical cell, adjust Paqp_cortex')
                            #Calculating each conductance
                            if count_endo>=2: #wall between two endodermis cells, in this case the suberized wall can limit the transfer of water between cell and wall
                                if kw_endo_endo==0.00:
                                    K=0.00
                                else:
                                    K = 1/(1/(kw_endo_endo/(config.thickness/2*1.0E-04))+1/(config.kmb+kaqp))*1.0E-08*(height+eattr['dist'])*eattr['length']
                            elif count_exo>=2: #wall between two exodermis cells, in this case the suberized wall can limit the transfer of water between cell and wall
                                if kw_exo_exo==0.00:
                                    K=0.00
                                else:
                                    K = 1/(1/(kw_exo_exo/(config.thickness/2*1.0E-04))+1/(config.kmb+kaqp))*1.0E-08*(height+eattr['dist'])*eattr['length']
                            elif count_stele_overall>0 and count_endo>0: #wall between endodermis and pericycle, in this case the suberized wall can limit the transfer of water between cell and wall
                                if count_passage>0:
                                    K = 1/(1/(kw_passage/(config.thickness/2*1.0E-04))+1/(config.kmb+kaqp))*1.0E-08*(height+eattr['dist'])*eattr['length']
                                else:
                                    if kw_endo_peri==0.00:
                                        K=0.00
                                    else:
                                        K = 1/(1/(kw_endo_peri/(config.thickness/2*1.0E-04))+1/(config.kmb+kaqp))*1.0E-08*(height+eattr['dist'])*eattr['length']
                            elif count_stele_overall==0 and count_endo==1: #wall between cortex and endodermis, in this case the suberized wall can limit the transfer of water between cell and wall
                                if kaqp==0.0:
                                    K=1.00E-16
                                else:
                                    if count_passage>0:
                                        K = 1/(1/(kw_passage/(config.thickness/2*1.0E-04))+1/(config.kmb+kaqp))*1.0E-08*(height+eattr['dist'])*eattr['length']
                                    else:
                                        if kw_endo_cortex==0.00:
                                            K=0.00
                                        else:
                                            K = 1/(1/(kw_endo_cortex/(config.thickness/2*1.0E-04))+1/(config.kmb+kaqp))*1.0E-08*(height+eattr['dist'])*eattr['length']
                            else:
                                if kaqp==0.0:
                                    K=1.00E-16
                                else:
                                    K = 1/(1/(kw/(config.thickness/2*1.0E-04))+1/(config.kmb+kaqp))*1.0E-08*(height+eattr['dist'])*eattr['length']
                            Kmb[jmb]=K
                            #if jmb<=10:
                            #    print(jmb,'K init',K,'wid',i,'cid',j-(network.n_walls + network.n_junctions))
                            jmb+=1
                        elif path == "plasmodesmata": #Plasmodesmata connection
                            cgroupi=G.nodes[i]['cgroup']
                            cgroupj=G.nodes[j]['cgroup']
                            if cgroupi==19 or cgroupi==20:  #Xylem in new Cellset version
                                cgroupi=13
                            elif cgroupi==21: #Xylem Pole Pericyle in new Cellset version
                                cgroupi=16
                            elif cgroupi==23: #Phloem in new Cellset version
                                cgroupi==11
                            elif cgroupi==26: #Companion Cell in new Cellset version
                                cgroupi==12
                            if cgroupj==19 or cgroupj==20:  #Xylem in new Cellset version
                                cgroupj=13
                            elif cgroupj==21: #Xylem Pole Pericyle in new Cellset version
                                cgroupj=16
                            elif cgroupj==23: #Phloem in new Cellset version
                                cgroupj==11
                            elif cgroupj==26: #Companion Cell in new Cellset version
                                cgroupj==12
                            temp_factor=1.0 #Quantity of plasmodesmata (adjusted by relative aperture)
                            if ((j-(network.n_walls + network.n_junctions) in config.intercellular_ids) or (i-(network.n_walls + network.n_junctions) in config.intercellular_ids)) and Barrier>0: #one of the connected cells is an intercellular space "cell".
                                temp_factor=0.0
                            elif cgroupj==13 and cgroupi==13: #Fake wall splitting a xylem cell or vessel, high conductance in order to ensure homogeneous pressure within the splitted cell
                                temp_factor=10000*config.fplxheight*1.0E-04*eattr['length'] #Quantity of PD
                            elif Barrier>0 and (cgroupj==13 or cgroupi==13): #Mature xylem vessels, so no plasmodesmata with surrounding cells
                                temp_factor=0.0 #If Barrier==0, this case is treated like xylem is a stelar parenchyma cell
                            elif (cgroupi==2 and cgroupj==1) or (cgroupj==2 and cgroupi==1):#Epidermis to exodermis cell or vice versa
                                temp_factor=config.fplxheight_epi_exo*1.0E-04*eattr['length'] #Will not be used in case there is no exodermal layer
                            elif (cgroupi==network.outercortex_connec_rank and cgroupj==4) or (cgroupj==network.outercortex_connec_rank and cgroupi==4):#Exodermis to cortex cell or vice versa
                                temp=float(config.kpl_elems[iPD].get("cortex_factor")) #Correction for specific cell-type PD aperture
                                if Barrier>0:
                                    temp_factor=2*temp/(temp+1)*config.fplxheight_outer_cortex*1.0E-04*eattr['length']*network.len_outer_cortex/network.cross_section_outer_cortex
                                else: #No aerenchyma
                                    temp_factor=2*temp/(temp+1)*config.fplxheight_outer_cortex*1.0E-04*eattr['length']
                            elif (cgroupi==4 and cgroupj==4):#Cortex to cortex cell
                                temp=float(config.kpl_elems[iPD].get("cortex_factor")) #Correction for specific cell-type PD aperture
                                if Barrier>0:
                                    temp_factor=temp*config.fplxheight_cortex_cortex*1.0E-04*eattr['length']*network.len_cortex_cortex/network.cross_section_cortex_cortex
                                else: #No aerenchyma
                                    temp_factor=temp*config.fplxheight_cortex_cortex*1.0E-04*eattr['length']
                            elif (cgroupi==3 and cgroupj==4) or (cgroupj==3 and cgroupi==4):#Cortex to endodermis cell or vice versa
                                temp=float(config.kpl_elems[iPD].get("cortex_factor")) #Correction for specific cell-type PD aperture
                                if Barrier>0:
                                    temp_factor=2*temp/(temp+1)*config.fplxheight_cortex_endo*1.0E-04*eattr['length']*network.len_cortex_endo/network.cross_section_cortex_endo
                                else: #No aerenchyma
                                    temp_factor=2*temp/(temp+1)*config.fplxheight_cortex_endo*1.0E-04*eattr['length']
                            elif (cgroupi==3 and cgroupj==3):#Endodermis to endodermis cell
                                temp_factor=config.fplxheight_endo_endo*1.0E-04*eattr['length']
                            elif (cgroupi==3 and cgroupj==16) or (cgroupj==3 and cgroupi==16):#Pericycle to endodermis cell or vice versa
                                if (i-(network.n_walls + network.n_junctions) in network.plasmodesmata_indice) or (j-(network.n_walls + network.n_junctions) in network.plasmodesmata_indice):
                                    temp=float(config.kpl_elems[iPD].get("PPP_factor")) #Correction for specific cell-type PD aperture
                                else:
                                    temp=1
                                temp_factor=2*temp/(temp+1)*config.fplxheight_endo_peri*1.0E-04*eattr['length']
                            elif (cgroupi==16 and (cgroupj==5 or cgroupj==13)) or (cgroupj==16 and (cgroupi==5 or cgroupi==13)):#Pericycle to stele cell or vice versa
                                if (i-(network.n_walls + network.n_junctions) in network.plasmodesmata_indice) or (j-(network.n_walls + network.n_junctions) in network.plasmodesmata_indice):
                                    temp=float(config.kpl_elems[iPD].get("PPP_factor")) #Correction for specific cell-type PD aperture
                                else:
                                    temp=1
                                temp_factor=2*temp/(temp+1)*config.fplxheight_peri_stele*1.0E-04*eattr['length']
                            elif ((cgroupi==5 or cgroupi==13) and cgroupj==12) or (cgroupi==12 and (cgroupj==5 or cgroupj==13)):#Stele to companion cell
                                temp=float(config.kpl_elems[iPD].get("PCC_factor")) #Correction for specific cell-type PD aperture
                                temp_factor=2*temp/(temp+1)*config.fplxheight_stele_comp*1.0E-04*eattr['length']
                            elif (cgroupi==16 and cgroupj==12) or (cgroupi==12 and cgroupj==16):#Pericycle to companion cell
                                temp1=float(config.kpl_elems[iPD].get("PCC_factor"))
                                if (i-(network.n_walls + network.n_junctions) in network.plasmodesmata_indice) or (j-(network.n_walls + network.n_junctions) in network.plasmodesmata_indice):
                                    temp2=float(config.kpl_elems[iPD].get("PPP_factor")) #Correction for specific cell-type PD aperture
                                else:
                                    temp2=1
                                temp_factor=2*temp1*temp2/(temp1+temp2)*config.fplxheight_peri_comp*1.0E-04*eattr['length']
                            elif (cgroupi==12 and cgroupj==12):#Companion to companion cell
                                temp=float(config.kpl_elems[iPD].get("PCC_factor"))
                                temp_factor=temp*config.fplxheight_peri_comp*1.0E-04*eattr['length']
                            elif (cgroupi==12 and cgroupj==11) or (cgroupi==11 and cgroupj==12):#Companion to phloem sieve tube cell
                                temp=float(config.kpl_elems[iPD].get("PCC_factor"))
                                temp_factor=2*temp/(temp+1)*config.fplxheight_comp_sieve*1.0E-04*eattr['length']
                            elif (cgroupi==16 and cgroupj==11) or (cgroupi==11 and cgroupj==16):#Pericycle to phloem sieve tube cell
                                if (i-(network.n_walls + network.n_junctions) in network.plasmodesmata_indice) or (j-(network.n_walls + network.n_junctions) in network.plasmodesmata_indice):
                                    temp=float(config.kpl_elems[iPD].get("PPP_factor")) #Correction for specific cell-type PD aperture
                                else:
                                    temp=1
                                temp_factor=2*temp/(temp+1)*config.fplxheight_peri_sieve*1.0E-04*eattr['length']
                            elif ((cgroupi==5 or cgroupi==13) and cgroupj==11) or (cgroupi==11 and (cgroupj==5 or cgroupj==13)):#Stele to phloem sieve tube cell
                                temp_factor=config.fplxheight_stele_sieve*1.0E-04*eattr['length']
                            #elif cgroupi==13 and cgroupj==13: #Fake wall splitting a xylem cell or vessel, high conductance in order to ensure homogeneous pressure within the splitted cell
                            #    temp_factor=10000*config.fplxheight*1.0E-04*eattr['length']
                            elif ((cgroupi==5 or cgroupi==13) and (cgroupj==5 or cgroupj==13)):#Stele to stele cell
                                temp_factor=config.fplxheight_stele_stele*1.0E-04*eattr['length']
                            else: #Default plasmodesmatal frequency
                                temp_factor=config.fplxheight*1.0E-04*eattr['length'] #eattr['kpl']
                            K = Kpl*temp_factor
                            ########Solute fluxes (diffusion across plasmodesmata)
                            if config.sym_contagion==2:
                                DF=config.pd_section*temp_factor/config.thickness*1.0E-04*config.diff_pd1 #"Diffusive flux": Total PD cross-section area (micron^2) per unit PD length (micron) (tunred into cm) multiplied by solute diffusivity (cm^2/d) (yields cm^3/d)
                                if config.apo_contagion==2: #Sym & Apo contagion
                                    if i-(network.n_walls + network.n_junctions) not in config.sym_zombie0:
                                        matrix_C[i][i] -= DF
                                        matrix_C[i][j] += DF #Convection will be dealt with further down
                                    if j-(network.n_walls + network.n_junctions) not in config.sym_zombie0:
                                        matrix_C[j][j] -= DF
                                        matrix_C[j][i] += DF
                                else: #Only Sym contagion
                                    if i-(network.n_walls + network.n_junctions) not in config.sym_zombie0:
                                        matrix_SymC[i-(network.n_walls + network.n_junctions)][i-(network.n_walls + network.n_junctions)] -= DF
                                        matrix_SymC[i-(network.n_walls + network.n_junctions)][j-(network.n_walls + network.n_junctions)] += DF
                                    if j-(network.n_walls + network.n_junctions) not in config.sym_zombie0:
                                        matrix_SymC[j-(network.n_walls + network.n_junctions)][j-(network.n_walls + network.n_junctions)] -= DF #Convection will be dealt with further down
                                        matrix_SymC[j-(network.n_walls + network.n_junctions)][i-(network.n_walls + network.n_junctions)] += DF
                        matrix_W[i][i] -= K #Filling the Doussan matrix (symmetric)
                        matrix_W[i][j] += K
                        matrix_W[j][i] += K
                        matrix_W[j][j] -= K
            
            #Adding matrix components at soil-wall and wall-xylem connections & rhs terms
            rhs = np.zeros((len(G),1))
            rhs_s = np.zeros((len(G),1)) #Initializing the right-hand side matrix of soil pressure potentials
            rhs_x = np.zeros((len(G),1)) #Initializing the right-hand side matrix of xylem pressure potentials
            rhs_p = np.zeros((len(G),1)) #Initializing the right-hand side matrix of hydrostatic potentials for phloem BC
            
            #Adding matrix components at soil-wall connections
            for wid in network.border_walls:
                if (network.position[wid][0]>=Xcontact) or (Wall2Cell[wid][0]-(network.n_walls + network.n_junctions) in config.contact): #Wall (not including junctions) connected to soil
                    temp=1.0E-04*(network.length[wid]/2*height)/(config.thickness/2)
                    K=kw*temp #Half the wall length is used here as the other half is attributed to the junction (Only for connection to soil)
                    matrix_W[wid][wid] -= K #Doussan matrix
                    rhs_s[wid][0] = -K    #Right-hand side vector, could become Psi_soil[idwall], which could be a function of the horizontal position
                    #if config.C_flag:
                    #    #Diffusion
                    #    matrix_C[wid][wid] -= temp*Diff1
                    #    rhs_C[wid][0] -= temp*Diff1*config.scenarios[0].get("osmotic_left_soil")
                        
            #Adding matrix components at soil-junction connections
            for jid in network.border_junctions:
                if (network.position[jid][0]>=Xcontact) or (Junction2Wall2Cell[jid-network.n_walls][0]-(network.n_walls + network.n_junctions) in config.contact) or (Junction2Wall2Cell[jid-network.n_walls][1]-(network.n_walls + network.n_junctions) in config.contact) or (Junction2Wall2Cell[jid-network.n_walls][2]-(network.n_walls + network.n_junctions) in config.contact): #Junction connected to soil
                    temp=1.0E-04*(network.length[jid]*height)/(config.thickness/2)
                    K=kw*temp
                    matrix_W[jid][jid] -= K #Doussan matrix
                    rhs_s[jid][0] = -K    #Right-hand side vector, could become Psi_soil[idwall], which could be a function of the horizontal position
                    #if config.C_flag:
                    #    matrix_C[jid][jid] -= temp*Diff1 #Diffusion BC at soil junction
                    #    rhs_C[jid][0] -= temp*Diff1*config.scenarios[0].get("osmotic_left_soil")
            
            #Creating connections to xylem & phloem BC elements for kr calculation (either xylem or phloem flow occurs depending on whether the segment is in the differentiation or elongation zone)
            if Barrier>0:
                if not isnan(config.scenarios[0].get("pressure_xyl")): #Pressure xylem BC
                    for cid in network.xylem_cells:
                        rhs_x[cid][0] = -config.k_xyl  #Axial conductance of xylem vessels
                        matrix_W[cid][cid] -= config.k_xyl
                        #if config.C_flag:
                        #    temp=10E-04*((cellperimeter[cid-(network.n_walls + network.n_junctions)]/2)**2)/pi/height #Cell approximative cross-section area (cm^2) per length (cm)
                        #    matrix_C[cid][cid] -= temp*Diff1*100 #Diffusion BC in xylem open vessels assumed 100 times easier than in walls
                        #    rhs_C[cid][0] -= temp*Diff1*100
                    rhs = rhs_s*config.scenarios[0].get("psi_soil_left") + rhs_x*config.scenarios[0].get("pressure_xyl") #multiplication of rhs components delayed till this point so that rhs_s & rhs_x can be re-used to calculate Q
                elif not isnan(config.scenarios[0].get("flow_sieve")):
                    i=1
                    for cid in network.xylem_cells:
                        rhs_x[cid][0] = Flow_xyl[i][0]
                        i+=1
                    #    if config.C_flag:
                    #        temp=10E-04*((cellperimeter[cid-(network.n_walls + network.n_junctions)]/2)**2)/pi/height #Cell approximative cross-section area (cm^2) per length (cm)
                    #        matrix_C[cid][cid] -= temp*Diff1*100 #Diffusion BC in xylem open vessels assumed 100 times easier than in walls
                    #        rhs_C[cid][0] -= temp*Diff1*100
                    rhs = rhs_s*config.scenarios[0].get("psi_soil_left") + rhs_x #multiplication of rhs components delayed till this point so that rhs_s & rhs_x can be re-used to calculate Q
                else:
                    rhs = rhs_s*config.scenarios[0].get("psi_soil_left")
            elif Barrier==0:
                if not isnan(config.scenarios[0].get("pressure_sieve")):
                    for cid in listprotosieve:
                        rhs_p[cid][0] = -config.k_sieve  #Axial conductance of phloem sieve tube
                        matrix_W[cid][cid] -= config.k_sieve
                    rhs = rhs_s*config.scenarios[0].get("psi_soil_left") + rhs_p*config.scenarios[0].get("pressure_sieve") #multiplication of rhs components delayed till this point so that rhs_s & rhs_x can be re-used to calculate Q
                elif not isnan(config.scenarios[0].get("flow_sieve")):
                    i=1
                    for cid in listprotosieve:
                        rhs_p[cid][0] = Flow_sieve[i][0]
                        i+=1
                    rhs = rhs_s*config.scenarios[0].get("psi_soil_left") + rhs_p #multiplication of rhs components delayed till this point so that rhs_s & rhs_x can be re-used to calculate Q
                else:
                    rhs = rhs_s*config.scenarios[0].get("psi_soil_left")
            
            
            ##################################################
            ##Solve Doussan equation, results in soln matrix##
            ##################################################
            
            soln = np.linalg.solve(matrix_W,rhs) #Solving the equation to get potentials inside the network
            
            #Verification that computation was correct
            verif1=np.allclose(np.dot(matrix_W,soln),rhs)
            
            #print("Correct computation on PSI ?", verif1)
            
            #Removing xylem and phloem BC terms
            if Barrier>0:
                if not isnan(config.scenarios[0].get("pressure_xyl")): #Pressure xylem BC
                    for cid in network.xylem_cells:
                        matrix_W[cid][cid] += config.k_xyl
            elif Barrier==0:
                if not isnan(config.scenarios[0].get("pressure_sieve")):
                    for cid in listprotosieve:
                        matrix_W[cid][cid] += config.k_sieve
            
            #Flow rates at interfaces
            Q_soil=[]
            for ind in network.border_walls:
                Q_soil.append(rhs_s[ind]*(soln[ind]-config.scenarios[0].get("psi_soil_left"))) #(cm^3/d) Positive for water flowing into the root
            for ind in network.border_junctions:
                Q_soil.append(rhs_s[ind]*(soln[ind]-config.scenarios[0].get("psi_soil_left"))) #(cm^3/d) Positive for water flowing into the root
            Q_xyl=[]
            Q_sieve=[]
            if Barrier>0:
                if not isnan(config.scenarios[0].get("pressure_xyl")):
                    for cid in network.xylem_cells:
                        Q=rhs_x[cid]*(soln[cid]-config.scenarios[0].get("pressure_xyl"))
                        Q_xyl.append(Q) #(cm^3/d) Negative for water flowing into xylem tubes
                        rank=int(network.cell_ranks[cid-(network.n_walls + network.n_junctions)].item())
                        row=int(network.rank_to_row[rank].item())
                        Q_xyl_layer[row][iMaturity][0] += Q.item()
                elif not isnan(config.scenarios[0].get("flow_sieve")):
                    for cid in network.xylem_cells:
                        Q=-rhs_x[cid]
                        Q_xyl.append(Q) #(cm^3/d) Negative for water flowing into xylem tubes
                        rank=int(network.cell_ranks[cid-(network.n_walls + network.n_junctions)])
                        row=int(network.rank_to_row[rank])
                        Q_xyl_layer[row][iMaturity][0] += Q.item()
            elif Barrier==0:
                if not isnan(config.scenarios[0].get("pressure_sieve")):
                    for cid in listprotosieve:
                        Q=rhs_p[cid]*(soln[cid]-config.scenarios[0].get("pressure_sieve"))
                        Q_sieve.append(Q) #(cm^3/d) Negative for water flowing into phloem tubes
                        rank=int(network.cell_ranks[cid-(network.n_walls + network.n_junctions)])
                        row=int(network.rank_to_row[rank])
                        Q_sieve_layer[row][iMaturity][0] += Q.item()
                elif not isnan(config.scenarios[0].get("flow_sieve")):
                    for cid in listprotosieve:
                        Q=-rhs_p[cid]
                        Q_sieve.append(Q) #(cm^3/d) Negative for water flowing into xylem tubes
                        rank=int(network.cell_ranks[cid-(network.n_walls + network.n_junctions)])
                        row=int(network.rank_to_row[rank])
                        Q_sieve_layer[row][iMaturity][0] += Q.item()
                
            Q_tot[iMaturity][0]=sum(Q_soil) #Total flow rate at root surface
            if Barrier>0:
                if not isnan(config.scenarios[0].get("pressure_xyl")):
                    kr_tot[iMaturity][0]=Q_tot[iMaturity][0].item()/(config.scenarios[0].get("psi_soil_left")-config.scenarios[0].get("pressure_xyl"))/network.perimeter.item()/height/1.0E-04
                else:
                    print('Error: Scenario 0 should have xylem pressure boundary conditions, except for the elongation zone')
            elif Barrier==0:
                if not isnan(config.scenarios[0].get("pressure_sieve")):
                    kr_tot[iMaturity][0]=Q_tot[iMaturity][0].item()/(config.scenarios[0].get("psi_soil_left")-config.scenarios[0].get("pressure_sieve").item())/network.perimeter.item()/height/1.0E-04
                else:
                    print('Error: Scenario 0 should have phloem pressure boundary conditions in the elongation zone')
            #print("Flow rates per unit root length: soil ",(Q_tot[iMaturity][0]/height/1.0E-04),"cm^2/d, xylem ",(sum(Q_xyl)/height/1.0E-04),"cm^2/d, phloem ",(sum(Q_sieve)/height/1.0E-04),"cm^2/d")
            #print("Mass balance error:",(Q_tot[iMaturity][0]+sum(Q_xyl)+sum(Q_sieve))/height/1.0E-04,"cm^2/d")


            
            print("Radial conductivity:",kr_tot[iMaturity][0],"cm/hPa/d")#, Barrier:",Barrier,", height: ",height," microns")
            
            if Barrier>0 and isnan(config.scenarios[0].get("pressure_xyl")):
                config.scenarios[0]["pressure_xyl"]=0.0
                for cid in network.xylem_cells:
                    config.scenarios[0]["pressure_xyl"]+=soln[cid]/len(network.xylem_cells) #Average of xylem water pressures
            elif Barrier==0 and isnan(config.scenarios[0].get("pressure_sieve")):
                config.scenarios[0]["pressure_sieve"]=0.0
                for cid in listprotosieve:
                    config.scenarios[0]["pressure_sieve"]+=soln[cid]/Nprotosieve #Average of protophloem water pressures
            
            #Calculation of standard transmembrane fractions
            jmb=0 #Index for membrane conductance vector
            for node, edges in G.adjacency() : #adjacency_iter returns an iterator of (node, adjacency dict) tuples for all nodes. This is the fastest way to look at every edge. For directed graphs, only outgoing adjacencies are included.
                i = network.indice[node] #Node ID number
                if i<network.n_walls: #wall ID
                    psi = soln[i]    #Node water potential
                    #print('i',i,'psi',psi)
                    psi_o_cell = inf #Opposite cell water potential
                    #Here we count surrounding cell types in order to identify in which row of the endodermis or exodermis we are.
                    count_endo=0 #total number of endodermis cells around the wall
                    count_stele_overall=0 #total number of stelar cells around the wall
                    count_exo=0 #total number of exodermis cells around the wall
                    count_epi=0 #total number of epidermis cells around the wall
                    #count_stele=0 #total number of epidermis cells around the wall
                    count_cortex=0 #total number of epidermis cells around the wall
                    count_passage=0 #total number of passage cells around the wall
                    for neighboor, eattr in edges.items(): #Loop on connections (edges)
                        if eattr['path'] == 'membrane': #Wall connection
                            if any(config.passage_cell_ids==array((network.indice[neighboor])-(network.n_walls + network.n_junctions))):
                                count_passage+=1
                            if G.nodes[neighboor]['cgroup']==3:#Endodermis
                                count_endo+=1
                            elif G.nodes[neighboor]['cgroup']>4:#Pericycle or stele
                                count_stele_overall+=1
                            elif G.nodes[neighboor]['cgroup']==1:#Exodermis
                                count_exo+=1
                            elif G.nodes[neighboor]['cgroup']==2:#Epidermis
                                count_epi+=1
                            elif G.nodes[neighboor]['cgroup']==4:#Cortex
                                count_cortex+=1
                        # if G.nodes[neighboor]['cgroup']==5:#Stele
                        #     count_stele+=1
                    for neighboor, eattr in edges.items(): #Loop on connections (edges)
                        j = network.indice[neighboor] #Neighbouring node ID number
                        path = eattr['path'] #eattr is the edge attribute (i.e. connection type)
                        if path == "membrane": #Membrane connection
                            psin = soln[j] #Neighbouring node water potential
                            #print('j',j,'psin',psin)
                            K=Kmb[jmb]
                            #if jmb<=10:
                            #    print(jmb,'K STF',K,'wid',i,'cid',j-(network.n_walls + network.n_junctions))
                            jmb+=1
                            #Flow densities calculation
                            #Macroscopic distributed parameter for transmembrane flow
                            #Discretization based on cell layers and apoplasmic barriers
                            rank = int(network.cell_ranks[j-(network.n_walls + network.n_junctions)].item())
                            row = int(network.rank_to_row[rank].item())
                            if rank == 1 and count_epi > 0: #Outer exodermis
                                row += 1
                            if rank == 3 and count_cortex > 0: #Outer endodermis
                                if any(config.passage_cell_ids==array(j-(network.n_walls + network.n_junctions))) and Barrier==2:
                                    row += 2
                                else:
                                    row += 3
                            elif rank == 3 and count_stele_overall > 0: #Inner endodermis
                                if any(config.passage_cell_ids==array(j-(network.n_walls + network.n_junctions))) and Barrier==2:
                                    row += 1
                            Flow = K * (psi - psin) #Note that this is only valid because we are in the scenario 0 with no osmotic potentials
                            #print('Flow',Flow,'dP',soln[node] - soln[neighboor],'Pi',soln[node],'Pj',soln[neighboor])
                            if ((j-(network.n_walls + network.n_junctions) not in config.intercellular_ids) and (j not in network.xylem_cells)) or Barrier==0: #Not part of STF if crosses an intercellular space "membrane" or mature xylem "membrane" (that is no membrane though still labelled like one)
                                if Flow > 0 :
                                    UptakeLayer_plus[row][iMaturity][0] += Flow.item() #grouping membrane flow rates in cell layers
                                else:
                                    UptakeLayer_minus[row][iMaturity][0] += Flow.item()
                                if Flow/Q_tot[iMaturity][0] > 0 :
                                    STFlayer_plus[row][iMaturity] += Flow.item()/Q_tot[iMaturity][0] #Cell standard transmembrane fraction (positive)
                                    STFcell_plus[j-(network.n_walls + network.n_junctions)][iMaturity] += Flow.item()/Q_tot[iMaturity][0].item() #Cell standard transmembrane fraction (positive)
                                    #STFmb[jmb-1][iMaturity] = Flow/Q_tot[iMaturity][0]
                                else:
                                    STFlayer_minus[row][iMaturity] += Flow.item()/Q_tot[iMaturity][0] #Cell standard transmembrane fraction (negative)
                                    STFcell_minus[j-(network.n_walls + network.n_junctions)][iMaturity] += Flow.item()/Q_tot[iMaturity][0]#Cell standard transmembrane fraction (negative)
                                    #STFmb[jmb-1][iMaturity] = Flow/Q_tot[iMaturity][0]
                                STFmb[jmb-1][iMaturity] = Flow.item()/Q_tot[iMaturity][0]
            
            for count in range(1,config.n_scenarios):
                
                #Initializing the connectivity matrix including boundary conditions
                rhs = np.zeros((len(G),1))
                rhs_x = np.zeros((len(G),1)) #Initializing the right-hand side matrix of xylem pressure potentials
                rhs_p = np.zeros((len(G),1)) #Initializing the right-hand side matrix of hydrostatic potentials for phloem BC
                rhs_e = np.zeros((len(G),1)) #Initializing the right-hand side matrix of cell elongation
                rhs_o = np.zeros((len(G),1)) #Initializing the right-hand side matrix of osmotic potentials
                Os_cells = np.zeros((network.n_cells,1)) #Initializing the cell osmotic potential vector
                Os_walls = np.zeros((network.n_walls,1)) #Initializing the wall osmotic potential vector
                s_membranes = np.zeros((network.n_membrane,1)) #Initializing the membrane reflection coefficient vector
                Os_membranes = np.zeros((network.n_membrane,2)) #Initializing the osmotic potential storage side by side of membranes (0 for the wall, 1 for the protoplast)
                #rhs_s invariable between diferent scenarios but can vary for different hydraulic properties
                
                #Apoplastic & symplastic convective direction matrices initialization
                Cell_connec_flow=zeros((network.n_cells,14),dtype=int) #Flow direction across plasmodesmata, positive when entering the cell, negative otherwise
                Apo_connec_flow=zeros(((network.n_walls + network.n_junctions),5),dtype=int) #Flow direction across cell walls, rows correspond to apoplastic nodes, and the listed nodes in each row receive convective flow from the row node
                nApo_connec_flow=zeros(((network.n_walls + network.n_junctions),1),dtype=int)
                
                print('Scenario #'+str(count))
                
                
                #Reflection coefficients of membranes (undimensional)
                if config.scenarios[count].get("psi_s_hetero")==0:
                    s_epi=config.scenarios[count].get("psi_s_factor")*1.0
                    s_exo_epi=config.scenarios[count].get("psi_s_factor")*1.0
                    s_exo_cortex=config.scenarios[count].get("psi_s_factor")*1.0
                    s_cortex=config.scenarios[count].get("psi_s_factor")*1.0
                    s_endo_cortex=config.scenarios[count].get("psi_s_factor")*1.0
                    s_endo_peri=config.scenarios[count].get("psi_s_factor")*1.0
                    s_peri=config.scenarios[count].get("psi_s_factor")*1.0
                    s_stele=config.scenarios[count].get("psi_s_factor")*1.0
                    s_comp=config.scenarios[count].get("psi_s_factor")*1.0
                    s_sieve=config.scenarios[count].get("psi_s_factor")*1.0
                elif config.scenarios[count].get("psi_s_hetero")==1:
                    s_epi=config.scenarios[count].get("psi_s_factor")*1.0
                    s_exo_epi=config.scenarios[count].get("psi_s_factor")*1.0
                    s_exo_cortex=config.scenarios[count].get("psi_s_factor")*1.0
                    s_cortex=config.scenarios[count].get("psi_s_factor")*1.0
                    s_endo_cortex=config.scenarios[count].get("psi_s_factor")*1.0
                    s_endo_peri=config.scenarios[count].get("psi_s_factor")*0.5
                    s_peri=config.scenarios[count].get("psi_s_factor")*0.5
                    s_stele=config.scenarios[count].get("psi_s_factor")*0.5
                    s_comp=config.scenarios[count].get("psi_s_factor")*0.5
                    s_sieve=config.scenarios[count].get("psi_s_factor")*0.5
                elif config.scenarios[count].get("psi_s_hetero")==2:
                    s_epi=config.scenarios[count].get("psi_s_factor")*0.5
                    s_exo_epi=config.scenarios[count].get("psi_s_factor")*0.5
                    s_exo_cortex=config.scenarios[count].get("psi_s_factor")*0.5
                    s_cortex=config.scenarios[count].get("psi_s_factor")*0.5
                    s_endo_cortex=config.scenarios[count].get("psi_s_factor")*0.5
                    s_endo_peri=config.scenarios[count].get("psi_s_factor")*1.0
                    s_peri=config.scenarios[count].get("psi_s_factor")*1.0
                    s_stele=config.scenarios[count].get("psi_s_factor")*1.0
                    s_comp=config.scenarios[count].get("psi_s_factor")*1.0
                    s_sieve=config.scenarios[count].get("psi_s_factor")*1.0
                
                #Osmotic potentials (hPa)
                if config.scenarios[count].get("psi_os_hetero")==0:
                    #Os_apo=-3000 #-0.3 MPa (Enns et al., 2000) applied stress
                    #-0.80 MPa (Enns et al., 2000) concentration of cortical cells, no KNO3
                    Os_epi=config.scenarios[count].get("psi_os_cortex")
                    Os_exo=config.scenarios[count].get("psi_os_cortex")
                    Os_c1=config.scenarios[count].get("psi_os_cortex")
                    Os_c2=config.scenarios[count].get("psi_os_cortex")
                    Os_c3=config.scenarios[count].get("psi_os_cortex")
                    Os_c4=config.scenarios[count].get("psi_os_cortex")
                    Os_c5=config.scenarios[count].get("psi_os_cortex")
                    Os_c6=config.scenarios[count].get("psi_os_cortex")
                    Os_c7=config.scenarios[count].get("psi_os_cortex")
                    Os_c8=config.scenarios[count].get("psi_os_cortex")
                    Os_endo=config.scenarios[count].get("psi_os_cortex")
                    Os_peri=config.scenarios[count].get("psi_os_cortex")
                    Os_stele=config.scenarios[count].get("psi_os_cortex")
                    Os_comp=(config.scenarios[count].get("osmotic_sieve")+config.scenarios[count].get("psi_os_cortex"))/2 #Average phloem and parenchyma
                    #Os_sieve=config.scenarios[count].get("psi_os_cortex")
                elif config.scenarios[count].get("psi_os_hetero")==1:
                    Os_epi=-5000 #(Rygol et al. 1993) #config.scenarios[count].get("psi_os_cortex") #-0.80 MPa (Enns et al., 2000) concentration of cortical cells, no KNO3
                    Os_exo=-5700 #(Rygol et al. 1993) #config.scenarios[count].get("psi_os_cortex") #-0.80 MPa (Enns et al., 2000) concentration of cortical cells, no KNO3
                    Os_c1=-6400 #(Rygol et al. 1993)
                    Os_c2=-7100 #(Rygol et al. 1993)
                    Os_c3=-7800 #(Rygol et al. 1993)
                    Os_c4=-8500 #(Rygol et al. 1993)
                    Os_c5=-9000 #(Rygol et al. 1993)
                    Os_c6=-9300 #(Rygol et al. 1993)
                    Os_c7=-9000 #(Rygol et al. 1993)
                    Os_c8=-8500 #(Rygol et al. 1993)
                    Os_endo=-6200 #-0.62 MPa (Enns et al., 2000) concentration of endodermis cells, no KNO3
                    Os_peri=-5000 #-0.50 MPa (Enns et al., 2000) concentration of pericycle cells, no KNO3
                    Os_stele=-7400 #-0.74 MPa (Enns et al., 2000) concentration of xylem parenchyma cells, no KNO3
                    Os_comp=(config.scenarios[count].get("osmotic_sieve")-7400)/2 #Average phloem and parenchyma
                    #Os_sieve=-14200 #-1.42 MPa (Pritchard, 1996) in barley phloem
                elif config.scenarios[count].get("psi_os_hetero")==2:
                    Os_epi=-11200 #(Rygol et al. 1993) #config.scenarios[count].get("psi_os_cortex") #-1.26 MPa (Enns et al., 2000) concentration of cortical cells, with KNO3
                    Os_exo=-11500 #(Rygol et al. 1993) #config.scenarios[count].get("psi_os_cortex") #-1.26 MPa (Enns et al., 2000) concentration of cortical cells, with KNO3
                    Os_c1=-11800 #(Rygol et al. 1993)
                    Os_c2=-12100 #(Rygol et al. 1993)
                    Os_c3=-12400 #(Rygol et al. 1993)
                    Os_c4=-12700 #(Rygol et al. 1993)
                    Os_c5=-12850 #(Rygol et al. 1993)
                    Os_c6=-12950 #(Rygol et al. 1993)
                    Os_c7=-12850 #(Rygol et al. 1993)
                    Os_c8=-12700 #(Rygol et al. 1993)
                    Os_endo=-10500 #-1.05 MPa (Enns et al., 2000) concentration of endodermis cells, with KNO3
                    Os_peri=-9200 #-0.92 MPa (Enns et al., 2000) concentration of pericycle cells, with KNO3
                    Os_stele=-12100 #-1.21 MPa (Enns et al., 2000) concentration of xylem parenchyma cells, with KNO3
                    Os_comp=(config.scenarios[count].get("osmotic_sieve")-12100)/2 #Average of phloem and parenchyma
                    #Os_sieve=-14200 #-1.42 MPa (Pritchard, 1996) in barley phloem
                elif config.scenarios[count].get("psi_os_hetero")==3:
                    Os_epi=config.scenarios[count].get("psi_os_cortex")
                    Os_exo=config.scenarios[count].get("psi_os_cortex")
                    Os_c1=config.scenarios[count].get("psi_os_cortex")
                    Os_c2=config.scenarios[count].get("psi_os_cortex")
                    Os_c3=config.scenarios[count].get("psi_os_cortex")
                    Os_c4=config.scenarios[count].get("psi_os_cortex")
                    Os_c5=config.scenarios[count].get("psi_os_cortex")
                    Os_c6=config.scenarios[count].get("psi_os_cortex")
                    Os_c7=config.scenarios[count].get("psi_os_cortex")
                    Os_c8=config.scenarios[count].get("psi_os_cortex")
                    Os_endo=float((config.scenarios[count].get("psi_os_cortex")-5000.0)/2.0)
                    Os_peri=-5000.0 #Simple case with no stele pushing water out
                    Os_stele=-5000.0
                    Os_comp=(config.scenarios[count].get("osmotic_sieve")-5000.0)/2 #Average phloem and parenchyma
                    #Os_sieve=-5000.0
                
                if config.C_flag:
                    jmb=0 #Index for membrane conductance vector
                    for node, edges in G.adjacency() : #adjacency_iter returns an iterator of (node, adjacency dict) tuples for all nodes. This is the fastest way to look at every edge. For directed graphs, only outgoing adjacencies are included.
                        i=network.indice[node] #Node ID number
                        #Here we count surrounding cell types in order to identify on which side of the endodermis or exodermis we are.
                        count_endo=0 #total number of endodermis cells around the wall
                        count_stele_overall=0 #total number of stelar cells around the wall
                        count_exo=0 #total number of exodermis cells around the wall
                        count_epi=0 #total number of epidermis cells around the wall
                        count_cortex=0 #total number of cortical cells around the wall
                        count_passage=0 #total number of passage cells around the wall
                        if i<network.n_walls: #wall ID
                            for neighboor, eattr in edges.items(): #Loop on connections (edges)
                                if eattr['path'] == 'membrane': #Wall connection
                                    if any(config.passage_cell_ids==array((network.indice[neighboor])-(network.n_walls + network.n_junctions))):
                                        count_passage+=1
                                    if G.nodes[neighboor]['cgroup']==3:#Endodermis
                                        count_endo+=1
                                    elif G.nodes[neighboor]['cgroup']>4:#Pericycle or stele
                                        count_stele_overall+=1
                                    elif G.nodes[neighboor]['cgroup']==4:#Cortex
                                        count_cortex+=1
                                    elif G.nodes[neighboor]['cgroup']==1:#Exodermis
                                        count_exo+=1
                                    elif G.nodes[neighboor]['cgroup']==2:#Epidermis
                                        count_epi+=1
                        for neighboor, eattr in edges.items(): #Loop on connections (edges)
                            j = (network.indice[neighboor]) #neighbouring node number
                            if j > i: #Only treating the information one way to save time
                                path = eattr['path'] #eattr is the edge attribute (i.e. connection type)
                                if path == "membrane": #Membrane connection
                                    #Cell and wall osmotic potentials (cell types: 1=Exodermis;2=epidermis;3=endodermis;4=cortex;5=stele;16=pericycle)
                                    rank=int(network.cell_ranks[int(j-(network.n_walls + network.n_junctions))])
                                    row=int(network.rank_to_row[rank])
                                    if rank==1:#Exodermis
                                        Os_membranes[jmb][1]=Os_exo
                                        if count_epi==1: #wall between exodermis and epidermis
                                            s_membranes[jmb]=s_exo_epi
                                        elif count_epi==0: #wall between exodermis and cortex or between two exodermal cells
                                            s_membranes[jmb]=s_exo_cortex
                                    elif rank==2:#Epidermis
                                        Os_membranes[jmb][1]=Os_epi
                                        s_membranes[jmb]=s_epi
                                    elif rank==3:#Endodermis
                                        Os_membranes[jmb][1]=Os_endo
                                        if count_stele_overall==0: #wall between endodermis and cortex or between two endodermal cells
                                            s_membranes[jmb]=s_endo_cortex
                                        elif count_stele_overall>0 and count_endo>0: #wall between endodermis and pericycle
                                            s_membranes[jmb]=s_endo_peri
                                    elif rank>=40 and rank<50:#Cortex
                                        if j-(network.n_walls + network.n_junctions) in config.intercellular_ids:
                                            Os_membranes[jmb][1]=0
                                            s_membranes[jmb]=0
                                        else:
                                            if row==row_outercortex-7:
                                                Os_membranes[jmb][1]=Os_c8
                                            elif row==row_outercortex-6:
                                                Os_membranes[jmb][1]=Os_c7
                                            elif row==row_outercortex-5:
                                                Os_membranes[jmb][1]=Os_c6
                                            elif row==row_outercortex-4:
                                                Os_membranes[jmb][1]=Os_c5
                                            elif row==row_outercortex-3:
                                                Os_membranes[jmb][1]=Os_c4
                                            elif row==row_outercortex-2:
                                                Os_membranes[jmb][1]=Os_c3
                                            elif row==row_outercortex-1:
                                                Os_membranes[jmb][1]=Os_c2
                                            elif row==row_outercortex:
                                                Os_membranes[jmb][1]=Os_c1
                                            s_membranes[jmb]=s_cortex
                                    elif G.nodes[j]['cgroup']==5:#Stelar parenchyma
                                        Os_membranes[jmb][1]=Os_stele
                                        s_membranes[jmb]=s_stele
                                    elif rank==16:#Pericycle
                                        Os_membranes[jmb][1]=Os_peri
                                        s_membranes[jmb]=s_peri
                                    elif G.nodes[j]['cgroup']==11 or G.nodes[j]['cgroup']==23:#Phloem sieve tube cell
                                        if not isnan(config.scenarios[count].get("osmotic_sieve")):
                                            if Barrier>0 or j in listprotosieve:
                                                Os_membranes[jmb][1]=config.scenarios[count].get("osmotic_sieve")
                                            else:
                                                Os_membranes[jmb][1]=Os_stele
                                        else:
                                            Os_membranes[jmb][1]=Os_stele
                                        s_membranes[jmb]=s_sieve
                                    elif G.nodes[j]['cgroup']==12 or G.nodes[j]['cgroup']==26:#Companion cell
                                        if not isnan(config.scenarios[count].get("osmotic_sieve")):
                                            Os_membranes[jmb][1]=Os_comp
                                        else:
                                            Os_membranes[jmb][1]=Os_stele
                                        s_membranes[jmb]=s_comp
                                    elif G.nodes[j]['cgroup']==13 or G.nodes[j]['cgroup']==19 or G.nodes[j]['cgroup']==20:#Xylem cell or vessel
                                        if Barrier==0:
                                            Os_membranes[jmb][1]=Os_stele
                                            s_membranes[jmb]=s_stele
                                        else:
                                            Os_membranes[jmb][1]=0.0
                                            s_membranes[jmb]=0.0
                                    jmb+=1
                
                #Soil and xylem water potentials
                #config.scenarios[count].get("psi_soil_left")=float(Psi_soil_range[count].get("pressure_left")) #Soil pressure potential (hPa)
                if not isnan(config.scenarios[count].get("flow_xyl")):
                    if isnan(config.scenarios[count].get("pressure_xyl")) and isnan(config.scenarios[count].get("delta_p_xyl")):
                        tot_flow=config.scenarios[count].get("flow_xyl")
                        sum_area=0.0
                        i=1
                        for cid in network.xylem_cells:
                            area=network.cell_areas[cid-(network.n_walls + network.n_junctions)]
                            Flow_xyl[i][count]=tot_flow*area
                            sum_area+=area
                            i+=1
                        i=1
                        for cid in network.xylem_cells:
                            Flow_xyl[i][count]/=sum_area #Total xylem flow rate partitioned proportionnally to xylem cross-section area
                            i+=1
                        if config.scenarios[count].get("flow_xyl")==0.0:
                            iEquil_xyl=count
                        if config.C_flag:
                            #Estimate the radial distribution of solutes later on from "u"
                            #First estimate water radial velocity in the apoplast
                            u=zeros((2,1))
                            u[0][0]=tot_flow/(height*1.0E-04)/(config.thickness*1.0E-04)/config.cell_per_layer[0][0] #Cortex (cm/d)
                            u[1][0]=tot_flow/(height*1.0E-04)/(config.thickness*1.0E-04)/config.cell_per_layer[1][0] #Stele (cm/d)
                    else:
                        print('Error: Cannot have both pressure and flow BC at xylem boundary')
                elif not isnan(config.scenarios[count].get("delta_p_xyl")):
                    if isnan(config.scenarios[count].get("pressure_xyl")):
                        config.scenarios[count]["pressure_xyl"]=Psi_xyl[iMaturity][iEquil_xyl]+config.scenarios[count].get("delta_p_xyl")
                    else:
                        print('Error: Cannot have both pressure and pressure change relative to equilibrium as xylem boundary condition')
                if not isnan(config.scenarios[count].get("pressure_xyl")):
                    if config.C_flag:
                        #Estimate the radial distribution of solutes
                        #First estimate total flow rate (cm^3/d) from BC & kr
                        tot_flow1=0.0
                        u=zeros((2,1))
                        iter=0
                        tot_flow2=kr_tot[iMaturity][0]*network.perimeter*height*1.0E-04*(config.scenarios[count].get("psi_soil_left")+config.scenarios[count].get("osmotic_left_soil")-config.scenarios[count].get("pressure_xyl")-config.scenarios[count].get("osmotic_xyl")) 
                        print('flow_rate =',tot_flow2,' iter =',iter)
                        #Convergence loop of water radial velocity and solute apoplastic convection-diffusion
                        while abs(tot_flow1-tot_flow2)/abs(tot_flow2)>0.001 and iter<30:
                            iter+=1
                            if iter==1:
                                tot_flow1=tot_flow2
                            elif iter>1 and sign(tot_flow1/tot_flow2)==1:
                                tot_flow1=(tot_flow1+tot_flow2)/2
                            else:
                                tot_flow1=tot_flow1/2
                            #Then estimate water radial velocity in the apoplast
                            u[0][0]=tot_flow1/(height*1.0E-04)/(config.thickness*1.0E-04)/config.cell_per_layer[0][0] #Cortex apoplastic water velocity (cm/d) positive inwards
                            u[1][0]=tot_flow1/(height*1.0E-04)/(config.thickness*1.0E-04)/config.cell_per_layer[1][0] #Stele apoplastic water velocity (cm/d) positive inwards
                            #Then estimate the radial solute distribution from an analytical solution (C(x)=C0+C0*(exp(u*x/D)-1)/(u/D*exp(u*x/D)-exp(u*L/D)+1)
                            Os_apo_cortex_eq=0.0
                            Os_apo_stele_eq=0.0
                            Os_sym_cortex_eq=0.0
                            Os_sym_stele_eq=0.0
                            #temp1=0.0
                            #temp2=0.0
                            jmb=0 #Index for membrane vector
                            for node, edges in G.adjacency() : #adjacency_iter returns an iterator of (node, adjacency dict) tuples for all nodes. This is the fastest way to look at every edge. For directed graphs, only outgoing adjacencies are included.
                                i = network.indice[node] #Node ID number
                                if i<network.n_walls: #wall ID
                                    for neighboor, eattr in edges.items(): #Loop on connections (edges)
                                        if eattr['path'] == 'membrane': #Wall connection
                                            if r_rel[i]>=0: #cortical side
                                                Os_apo=config.scenarios[count].get("osmotic_left_soil")*exp(u[0][0]*abs(r_rel[i])*L_diff[0]/config.scenarios[count].get("osmotic_diffusivity_soil"))
                                                Os_apo_cortex_eq+=STFmb[jmb][iMaturity]*(Os_apo*s_membranes[jmb])
                                                Os_sym_cortex_eq+=STFmb[jmb][iMaturity]*(Os_membranes[jmb][1]*s_membranes[jmb])
                                                #temp1+=STFmb[jmb][iMaturity]
                                            else: #Stelar side
                                                Os_apo=config.scenarios[count].get("osmotic_xyl")*exp(-u[1][0]*abs(r_rel[i])*L_diff[1]/config.scenarios[count].get("osmotic_diffusivity_xyl"))
                                                Os_apo_stele_eq-=STFmb[jmb][iMaturity]*(Os_apo*s_membranes[jmb])
                                                Os_sym_stele_eq-=STFmb[jmb][iMaturity]*(Os_membranes[jmb][1]*s_membranes[jmb])
                                                #temp2+=STFmb[jmb][iMaturity]
                                            Os_membranes[jmb][0]=Os_apo
                                            jmb+=1
                            tot_flow2=kr_tot[iMaturity][0]*network.perimeter*height*1.0E-04*(config.scenarios[count].get("psi_soil_left")+Os_apo_cortex_eq-Os_sym_cortex_eq-config.scenarios[count].get("pressure_xyl")-Os_apo_stele_eq+Os_sym_stele_eq)
                            print('flow_rate =',tot_flow2,' iter =',iter)
                        u[0][0]=tot_flow2/(height*1.0E-04)/(config.thickness*1.0E-04)/config.cell_per_layer[0][0] #Cortex (cm/d)
                        u[1][0]=tot_flow2/(height*1.0E-04)/(config.thickness*1.0E-04)/config.cell_per_layer[1][0] #Stele (cm/d)
                        ##Then estimate osmotic potentials in radial walls later on: C(x)=C0+C0*(exp(u*x/D)-1)/(u/D*exp(u*x/D)-exp(u*L/D)+1)
                
                #Elongation BC
                if Barrier==0: #No elongation from the Casparian strip on
                    for wid in range(network.n_walls):
                        rhs_e[wid][0]=network.length[wid]*config.thickness/2*1.0E-08*(config.scenarios[count].get("elongation_midpoint_rate")+(x_rel[wid]-0.5)*config.scenarios[count].get("elongation_side_rate_difference"))*config.water_fraction_apo #cm^3/d Cell wall horizontal surface assumed to be rectangular (junctions are pointwise elements)
                    for cid in range(network.n_cells):
                        if network.cell_areas[cid]>cellperimeter[cid]*config.thickness/2:
                            rhs_e[(network.n_walls + network.n_junctions)+cid][0]=(network.cell_areas[cid]-cellperimeter[cid]*config.thickness/2)*1.0E-8*(config.scenarios[count].get("elongation_midpoint_rate")+(x_rel[(network.n_walls + network.n_junctions)+cid]-0.5)*config.scenarios[count].get("elongation_side_rate_difference"))*config.water_fraction_sym #cm^3/d Wall thickness removed from cell horizontal area to obtain protoplast horizontal area
                        else:
                            rhs_e[(network.n_walls + network.n_junctions)+cid][0]=0 #The cell elongation virtually does not imply water influx, though its walls do (typically intercellular spaces
                            #print('Cell too small to have '+str(thickness/2)+' micron thick walls')
                            #print('Cell ID & rank: '+str(cid)+' '+str(network.cell_ranks[cid]),'Total horizontal area: '+str(network.cell_areas[cid])+' microns^2','Wall horizontal area: '+str(cellperimeter[cid]*thickness/2)+' microns^2')
                
                if not isnan(config.scenarios[count].get("flow_sieve")):
                    if isnan(config.scenarios[count].get("pressure_sieve")) and isnan(config.scenarios[count].get("delta_p_sieve")):
                        if Barrier==0:
                            if config.scenarios[count].get("flow_sieve")==0:
                                tot_flow=-float(sum(rhs_e)) #"Equilibrium condition" with phloem water fully used for elongation
                            else:
                                tot_flow=config.scenarios[count].get("flow_sieve")
                            sum_area=0
                            i=1
                            for cid in listprotosieve:
                                area=network.cell_areas[cid-(network.n_walls + network.n_junctions)]
                                Flow_sieve[i][count]=tot_flow*area
                                sum_area+=area
                                i+=1
                            i=1
                            for cid in listprotosieve:
                                Flow_sieve[i][count]/=sum_area #Total phloem flow rate partitioned proportionnally to phloem cross-section area
                                i+=1
                        elif Barrier>0:
                            tot_flow=config.scenarios[count].get("flow_sieve")
                            sum_area=0
                            i=1
                            for cid in network.sieve_cells:
                                area=network.cell_areas[cid-(network.n_walls + network.n_junctions)]
                                Flow_sieve[i][count]=tot_flow*area
                                sum_area+=area
                                i+=1
                            i=1
                            for cid in network.sieve_cells:
                                Flow_sieve[i][count]/=sum_area #Total phloem flow rate partitioned proportionnally to phloem cross-section area
                                i+=1
                        if config.scenarios[count].get("flow_sieve")==0.0:
                            iEquil_sieve=count
                    else:
                        print('Error: Cannot have both pressure and flow BC at phloem boundary')
                elif not isnan(config.scenarios[count].get("delta_p_sieve")):
                    if isnan(config.scenarios[count].get("pressure_sieve")):
                        if not isnan(iEquil_sieve):
                            config.scenarios[count]["pressure_sieve"]=Psi_sieve[iMaturity][iEquil_sieve]+config.scenarios[count].get("delta_p_sieve")
                        else:
                            print('Error: Cannot have phloem pressure change relative to equilibrium without having a prior scenario with equilibrium phloem boundary condition')
                    else:
                        print('Error: Cannot have both pressure and pressure change relative to equilibrium as phloem boundary condition')
                
                jmb=0 #Index for membrane conductance vector
                for node, edges in G.adjacency() : #adjacency_iter returns an iterator of (node, adjacency dict) tuples for all nodes. This is the fastest way to look at every edge. For directed graphs, only outgoing adjacencies are included.
                    i=network.indice[node] #Node ID number
                    #Here we count surrounding cell types in order to identify on which side of the endodermis or exodermis we are.
                    count_endo=0 #total number of endodermis cells around the wall
                    count_stele_overall=0 #total number of stelar cells around the wall
                    count_exo=0 #total number of exodermis cells around the wall
                    count_epi=0 #total number of epidermis cells around the wall
                    count_cortex=0 #total number of cortical cells around the wall
                    count_passage=0 #total number of passage cells around the wall
                    if i<network.n_walls: #wall ID
                        if config.scenarios[count].get("osmotic_symmetry_soil") == 2: #Central symmetrical gradient for apoplastic osmotic potential
                            if config.scenarios[count].get("osmotic_diffusivity_soil") == 0: #Not the analytical solution
                                Os_soil_local=float(config.scenarios[count].get("osmotic_left_soil")+(config.scenarios[count].get("osmotic_right_soil")-config.scenarios[count].get("osmotic_left_soil"))*abs(r_rel[i])**config.scenarios[count].get("osmotic_shape_soil"))
                            else:
                                if r_rel[i]>=0: #cortical side
                                    Os_soil_local=config.scenarios[count].get("osmotic_left_soil")*exp(u[0][0]*abs(r_rel[i])*L_diff[0]/config.scenarios[count].get("osmotic_diffusivity_soil"))
                        elif config.scenarios[count].get("osmotic_symmetry_soil") == 1: #Left-right gradient for apoplastic osmotic potential
                            Os_soil_local=float(config.scenarios[count].get("osmotic_left_soil")*(1-x_rel[i])+config.scenarios[count].get("osmotic_right_soil")*x_rel[i])
                        if config.scenarios[count].get("osmotic_symmetry_xyl") == 2:
                            if config.scenarios[count].get("osmotic_diffusivity_xyl") == 0: #Not the analytical solution
                                Os_xyl_local=float(config.scenarios[count].get("osmotic_endo")+(config.scenarios[count].get("osmotic_xyl")-config.scenarios[count].get("osmotic_endo"))*(1-abs(r_rel[i]))**config.scenarios[count].get("osmotic_shape_xyl"))
                            else:
                                if r_rel[i]<0: #cortical side
                                    Os_xyl_local=config.scenarios[count].get("osmotic_xyl")*exp(-u[1][0]*abs(r_rel[i])*L_diff[1]/config.scenarios[count].get("osmotic_diffusivity_xyl"))
                        elif config.scenarios[count].get("osmotic_symmetry_xyl") == 1:
                            Os_xyl_local=float((config.scenarios[count].get("osmotic_xyl")+config.scenarios[count].get("osmotic_endo"))/2)
                        for neighboor, eattr in edges.items(): #Loop on connections (edges)
                            if eattr['path'] == 'membrane': #Wall connection
                                if any(config.passage_cell_ids==array((network.indice[neighboor])-(network.n_walls + network.n_junctions))):
                                    count_passage+=1
                                if G.nodes[neighboor]['cgroup']==3:#Endodermis
                                    count_endo+=1
                                elif G.nodes[neighboor]['cgroup']>4:#Pericycle or stele
                                    count_stele_overall+=1
                                elif G.nodes[neighboor]['cgroup']==4:#Cortex
                                    count_cortex+=1
                                elif G.nodes[neighboor]['cgroup']==1:#Exodermis
                                    count_exo+=1
                                elif G.nodes[neighboor]['cgroup']==2:#Epidermis
                                    count_epi+=1
                    for neighboor, eattr in edges.items(): #Loop on connections (edges)
                        j = (network.indice[neighboor]) #neighbouring node number
                        if j > i: #Only treating the information one way to save time
                            path = eattr['path'] #eattr is the edge attribute (i.e. connection type)
                            if path == "membrane": #Membrane connection
                                #Cell and wall osmotic potentials (cell types: 1=Exodermis;2=epidermis;3=endodermis;4=cortex;5=stele;16=pericycle)
                                rank=int(network.cell_ranks[int(j-(network.n_walls + network.n_junctions))])
                                row=int(network.rank_to_row[rank])
                                if rank==1:#Exodermis
                                    Os_cells[j-(network.n_walls + network.n_junctions)]=Os_exo
                                    Os_membranes[jmb][1]=Os_exo
                                    OsCellLayer[row][iMaturity][count]+=Os_exo
                                    nOsCellLayer[row][iMaturity][count]+=1
                                    OsCellLayer[row+1][iMaturity][count]+=Os_exo
                                    nOsCellLayer[row+1][iMaturity][count]+=1
                                    Os_walls[i]=Os_soil_local
                                    if count_epi==1: #wall between exodermis and epidermis
                                        s_membranes[jmb]=s_exo_epi
                                        OsWallLayer[row+1][iMaturity][count]+=Os_soil_local
                                        nOsWallLayer[row+1][iMaturity][count]+=1
                                    elif count_epi==0: #wall between exodermis and cortex or between two exodermal cells
                                        s_membranes[jmb]=s_exo_cortex
                                        OsWallLayer[row][iMaturity][count]+=Os_soil_local
                                        nOsWallLayer[row][iMaturity][count]+=1
                                elif rank==2:#Epidermis
                                    Os_cells[j-(network.n_walls + network.n_junctions)]=Os_epi
                                    Os_membranes[jmb][1]=Os_epi
                                    OsCellLayer[row][iMaturity][count]+=Os_epi
                                    nOsCellLayer[row][iMaturity][count]+=1
                                    Os_walls[i]=Os_soil_local
                                    s_membranes[jmb]=s_epi
                                    OsWallLayer[row][iMaturity][count]+=Os_soil_local
                                    nOsWallLayer[row][iMaturity][count]+=1
                                elif rank==3:#Endodermis
                                    Os_cells[j-(network.n_walls + network.n_junctions)]=Os_endo
                                    Os_membranes[jmb][1]=Os_endo
                                    OsCellLayer[row][iMaturity][count]+=Os_endo
                                    nOsCellLayer[row][iMaturity][count]+=1
                                    OsCellLayer[row+3][iMaturity][count]+=Os_endo
                                    nOsCellLayer[row+3][iMaturity][count]+=1
                                    if count_stele_overall==0 and count_cortex>0: #wall between endodermis and cortex or between two endodermal cells
                                        Os_walls[i]=Os_soil_local
                                        s_membranes[jmb]=s_endo_cortex
                                        #Not including the osmotic potential of walls that are located at the same place as the casparian strip
                                        OsWallLayer[row+3][iMaturity][count]+=Os_soil_local
                                        nOsWallLayer[row+3][iMaturity][count]+=1
                                    elif count_stele_overall>0 and count_endo>0: #wall between endodermis and pericycle
                                        if Barrier==0: #No apoplastic barrier
                                            Os_walls[i]=Os_soil_local
                                            OsWallLayer[row][iMaturity][count]+=Os_soil_local
                                            nOsWallLayer[row][iMaturity][count]+=1
                                        else:
                                            Os_walls[i]=Os_xyl_local #float(config.scenarios[count].get("osmotic_xyl"))
                                            OsWallLayer[row][iMaturity][count]+=Os_xyl_local #float(config.scenarios[count].get("osmotic_xyl"))
                                            nOsWallLayer[row][iMaturity][count]+=1
                                        s_membranes[jmb]=s_endo_peri
                                    else: #Wall between endodermal cells
                                        if Barrier==0: #No apoplastic barrier
                                            Os_walls[i]=Os_soil_local
                                            OsWallLayer[row][iMaturity][count]+=Os_soil_local
                                            nOsWallLayer[row][iMaturity][count]+=1
                                        else:
                                            Os_walls[i]=Os_xyl_local #float(config.scenarios[count].get("osmotic_xyl"))
                                            OsWallLayer[row][iMaturity][count]+=Os_xyl_local #float(config.scenarios[count].get("osmotic_xyl"))
                                            nOsWallLayer[row][iMaturity][count]+=1
                                        s_membranes[jmb]=s_endo_peri
                                elif rank>=40 and rank<50:#Cortex
                                    if j-(network.n_walls + network.n_junctions) in config.intercellular_ids: 
                                        Os_cells[j-(network.n_walls + network.n_junctions)]=Os_soil_local
                                        Os_walls[i]=Os_soil_local
                                        Os_membranes[jmb][1]=Os_soil_local
                                        Os_membranes[jmb][0]=Os_soil_local
                                        s_membranes[jmb]=0
                                        OsWallLayer[row][iMaturity][count]+=Os_soil_local
                                        nOsWallLayer[row][iMaturity][count]+=1
                                    else:
                                        if row==row_outercortex-7:
                                            Os_cells[j-(network.n_walls + network.n_junctions)]=Os_c8
                                            Os_membranes[jmb][1]=Os_c8
                                            OsCellLayer[row][iMaturity][count]+=Os_c8
                                            nOsCellLayer[row][iMaturity][count]+=1
                                        elif row==row_outercortex-6:
                                            Os_cells[j-(network.n_walls + network.n_junctions)]=Os_c7
                                            Os_membranes[jmb][1]=Os_c7
                                            OsCellLayer[row][iMaturity][count]+=Os_c7
                                            nOsCellLayer[row][iMaturity][count]+=1
                                        elif row==row_outercortex-5:
                                            Os_cells[j-(network.n_walls + network.n_junctions)]=Os_c6
                                            Os_membranes[jmb][1]=Os_c6
                                            OsCellLayer[row][iMaturity][count]+=Os_c6
                                            nOsCellLayer[row][iMaturity][count]+=1
                                        elif row==row_outercortex-4:
                                            Os_cells[j-(network.n_walls + network.n_junctions)]=Os_c5
                                            Os_membranes[jmb][1]=Os_c5
                                            OsCellLayer[row][iMaturity][count]+=Os_c5
                                            nOsCellLayer[row][iMaturity][count]+=1
                                        elif row==row_outercortex-3:
                                            Os_cells[j-(network.n_walls + network.n_junctions)]=Os_c4
                                            Os_membranes[jmb][1]=Os_c4
                                            OsCellLayer[row][iMaturity][count]+=Os_c4
                                            nOsCellLayer[row][iMaturity][count]+=1
                                        elif row==row_outercortex-2:
                                            Os_cells[j-(network.n_walls + network.n_junctions)]=Os_c3
                                            Os_membranes[jmb][1]=Os_c3
                                            OsCellLayer[row][iMaturity][count]+=Os_c3
                                            nOsCellLayer[row][iMaturity][count]+=1
                                        elif row==row_outercortex-1:
                                            Os_cells[j-(network.n_walls + network.n_junctions)]=Os_c2
                                            Os_membranes[jmb][1]=Os_c2
                                            OsCellLayer[row][iMaturity][count]+=Os_c2
                                            nOsCellLayer[row][iMaturity][count]+=1
                                        elif row==row_outercortex:
                                            Os_cells[j-(network.n_walls + network.n_junctions)]=Os_c1
                                            Os_membranes[jmb][1]=Os_c1
                                            OsCellLayer[row][iMaturity][count]+=Os_c1
                                            nOsCellLayer[row][iMaturity][count]+=1
                                        Os_walls[i]=Os_soil_local
                                        s_membranes[jmb]=s_cortex
                                        OsWallLayer[row][iMaturity][count]+=Os_soil_local
                                        nOsWallLayer[row][iMaturity][count]+=1
                                elif G.nodes[j]['cgroup']==5:#Stelar parenchyma
                                    Os_cells[j-(network.n_walls + network.n_junctions)]=Os_stele
                                    Os_membranes[jmb][1]=Os_stele
                                    OsCellLayer[row][iMaturity][count]+=Os_stele
                                    nOsCellLayer[row][iMaturity][count]+=1
                                    if Barrier==0: #No apoplastic barrier
                                        Os_walls[i]=Os_soil_local
                                        OsWallLayer[row][iMaturity][count]+=Os_soil_local
                                        nOsWallLayer[row][iMaturity][count]+=1
                                    else:
                                        Os_walls[i]=Os_xyl_local #float(config.scenarios[count].get("osmotic_xyl"))
                                        OsWallLayer[row][iMaturity][count]+=Os_xyl_local #float(config.scenarios[count].get("osmotic_xyl"))
                                        nOsWallLayer[row][iMaturity][count]+=1
                                    s_membranes[jmb]=s_stele
                                elif rank==16:#Pericycle
                                    Os_cells[j-(network.n_walls + network.n_junctions)]=Os_peri
                                    Os_membranes[jmb][1]=Os_peri
                                    OsCellLayer[row][iMaturity][count]+=Os_peri
                                    nOsCellLayer[row][iMaturity][count]+=1
                                    if Barrier==0: #No apoplastic barrier
                                        Os_walls[i]=Os_soil_local
                                        OsWallLayer[row][iMaturity][count]+=Os_soil_local
                                        nOsWallLayer[row][iMaturity][count]+=1
                                    else:
                                        Os_walls[i]=Os_xyl_local #float(config.scenarios[count].get("osmotic_xyl"))
                                        OsWallLayer[row][iMaturity][count]+=Os_xyl_local #float(config.scenarios[count].get("osmotic_xyl"))
                                        nOsWallLayer[row][iMaturity][count]+=1
                                    s_membranes[jmb]=s_peri
                                elif G.nodes[j]['cgroup']==11 or G.nodes[j]['cgroup']==23:#Phloem sieve tube cell
                                    if not isnan(config.scenarios[count].get("osmotic_sieve")):
                                        if Barrier>0 or j in listprotosieve:
                                            Os_cells[j-(network.n_walls + network.n_junctions)]=config.scenarios[count].get("osmotic_sieve")
                                            Os_membranes[jmb][1]=config.scenarios[count].get("osmotic_sieve")
                                            OsCellLayer[row][iMaturity][count]+=config.scenarios[count].get("osmotic_sieve")
                                            nOsCellLayer[row][iMaturity][count]+=1
                                        else:
                                            Os_cells[j-(network.n_walls + network.n_junctions)]=Os_stele
                                            Os_membranes[jmb][1]=Os_stele
                                            OsCellLayer[row][iMaturity][count]+=Os_stele
                                            nOsCellLayer[row][iMaturity][count]+=1
                                    else:
                                        Os_cells[j-(network.n_walls + network.n_junctions)]=Os_stele
                                        Os_membranes[jmb][1]=Os_stele
                                        OsCellLayer[row][iMaturity][count]+=Os_stele
                                        nOsCellLayer[row][iMaturity][count]+=1
                                    if Barrier==0: #No apoplastic barrier
                                        Os_walls[i]=Os_soil_local
                                        OsWallLayer[row][iMaturity][count]+=Os_soil_local
                                        nOsWallLayer[row][iMaturity][count]+=1
                                    else:
                                        Os_walls[i]=Os_xyl_local #float(config.scenarios[count].get("osmotic_xyl"))
                                        OsWallLayer[row][iMaturity][count]+=Os_xyl_local #float(config.scenarios[count].get("osmotic_xyl"))
                                        nOsWallLayer[row][iMaturity][count]+=1
                                    s_membranes[jmb]=s_sieve
                                elif G.nodes[j]['cgroup']==12 or G.nodes[j]['cgroup']==26:#Companion cell
                                    if not isnan(config.scenarios[count].get("osmotic_sieve")):
                                        Os_cells[j-(network.n_walls + network.n_junctions)]=Os_comp
                                        Os_membranes[jmb][1]=Os_comp
                                        OsCellLayer[row][iMaturity][count]+=Os_comp
                                        nOsCellLayer[row][iMaturity][count]+=1
                                    else:
                                        Os_cells[j-(network.n_walls + network.n_junctions)]=Os_stele
                                        Os_membranes[jmb][1]=Os_stele
                                        OsCellLayer[row][iMaturity][count]+=Os_stele
                                        nOsCellLayer[row][iMaturity][count]+=1
                                    if Barrier==0: #No apoplastic barrier
                                        Os_walls[i]=Os_soil_local
                                        OsWallLayer[row][iMaturity][count]+=Os_soil_local
                                        nOsWallLayer[row][iMaturity][count]+=1
                                    else:
                                        Os_walls[i]=Os_xyl_local
                                        OsWallLayer[row][iMaturity][count]+=Os_xyl_local
                                        nOsWallLayer[row][iMaturity][count]+=1
                                    s_membranes[jmb]=s_comp
                                elif G.nodes[j]['cgroup']==13 or G.nodes[j]['cgroup']==19 or G.nodes[j]['cgroup']==20:#Xylem cell or vessel
                                    if Barrier==0:
                                        Os_cells[j-(network.n_walls + network.n_junctions)]=Os_stele
                                        Os_membranes[jmb][1]=Os_stele
                                        OsCellLayer[row][iMaturity][count]+=Os_stele
                                        nOsCellLayer[row][iMaturity][count]+=1
                                        Os_walls[i]=Os_soil_local
                                        s_membranes[jmb]=s_stele
                                        OsWallLayer[row][iMaturity][count]+=Os_soil_local
                                        nOsWallLayer[row][iMaturity][count]+=1
                                    else:
                                        Os_cells[j-(network.n_walls + network.n_junctions)]=Os_xyl_local
                                        Os_membranes[jmb][0]=Os_xyl_local
                                        Os_membranes[jmb][1]=Os_xyl_local
                                        Os_membranes[jmb][1]=Os_xyl_local
                                        Os_walls[i]=Os_xyl_local
                                        s_membranes[jmb]=0.0
                                        OsWallLayer[row][iMaturity][count]+=Os_xyl_local #float(config.scenarios[count].get("osmotic_xyl"))
                                        nOsWallLayer[row][iMaturity][count]+=1
                                K=Kmb[jmb]
                                rhs_o[i]+= K*s_membranes[jmb]*(Os_walls[i] - Os_cells[j-(network.n_walls + network.n_junctions)]) #Wall node
                                rhs_o[j]+= K*s_membranes[jmb]*(Os_cells[j-(network.n_walls + network.n_junctions)] - Os_walls[i]) #Cell node 
                                jmb+=1
                for row in range(int(network.r_discret[0])):
                    if nOsWallLayer[row][iMaturity][count]>0:
                        OsWallLayer[row][iMaturity][count]=OsWallLayer[row][iMaturity][count]/nOsWallLayer[row][iMaturity][count]
                    if nOsCellLayer[row][iMaturity][count]>0:
                        OsCellLayer[row][iMaturity][count]=OsCellLayer[row][iMaturity][count]/nOsCellLayer[row][iMaturity][count]
                
                #Xylem BC
                if Barrier>0: #No mature xylem before the Casparian strip stage
                    if not isnan(config.scenarios[count].get("pressure_xyl")): #Pressure xylem BC
                        for cid in network.xylem_cells:
                            rhs_x[cid][0] = -config.k_xyl  #Axial conductance of xylem vessels
                            matrix_W[cid][cid] -= config.k_xyl
                    elif not isnan(config.scenarios[count].get("flow_xyl")): #Flow xylem BC
                        i=1
                        for cid in network.xylem_cells:
                            rhs_x[cid][0] = Flow_xyl[i][count] #(cm^3/d)
                            i+=1
                
                #Phloem BC
                if Barrier==0: #Protophloem only
                    if not isnan(config.scenarios[count].get("pressure_sieve")):
                        for cid in listprotosieve:
                            rhs_p[cid][0] = -config.k_sieve  #Axial conductance of phloem sieve tube
                            matrix_W[cid][cid] -= config.k_sieve
                    elif not isnan(config.scenarios[count].get("flow_sieve")):
                        i=1
                        for cid in listprotosieve:
                            rhs_p[cid][0] = Flow_sieve[i][count] #(cm^3/d)
                            i+=1
                elif Barrier>0: #Includes mature phloem
                    if not isnan(config.scenarios[count].get("pressure_sieve")): #Then there is a phloem BC in scenarios (assuming that we did not pick scenarios with and others without)
                        for cid in network.sieve_cells: #both proto and metaphloem
                            rhs_p[cid][0] = -config.k_sieve  #Axial conductance of xylem vessels
                            matrix_W[cid][cid] -= config.k_sieve
                    elif not isnan(config.scenarios[count].get("flow_sieve")):
                        i=1
                        for cid in network.sieve_cells:
                            rhs_p[cid][0] = Flow_sieve[i][count] #(cm^3/d)
                            i+=1
                
                
                #Adding up all BC
                #Elongation BC
                rhs += rhs_e
                
                #Osmotic BC
                rhs += rhs_o
                
                #Soil BC
                rhs += np.multiply(rhs_s,config.scenarios[count].get("psi_soil_left")*(1-x_rel)+config.scenarios[count].get("psi_soil_right")*x_rel)
                
                #Xylem BC
                if not isnan(config.scenarios[count].get("pressure_xyl")): #Pressure xylem BC
                    rhs += rhs_x*config.scenarios[count].get("pressure_xyl")  #multiplication of rhs components delayed till this point so that rhs_s & rhs_x can be re-used
                elif not isnan(config.scenarios[count].get("flow_xyl")): #Flow xylem BC
                    rhs += rhs_x
                
                #Phloem BC
                if not isnan(config.scenarios[count].get("flow_sieve")):
                    rhs += rhs_p
                elif not isnan(config.scenarios[count].get("pressure_sieve")):
                    rhs += rhs_p*config.scenarios[count].get("pressure_sieve")
                
                ##################################################
                ##Solve Doussan equation, results in soln matrix##
                ##################################################
                
                soln = np.linalg.solve(matrix_W,rhs) #Solving the equation to get potentials inside the network
                
                #Verification that computation was correct
                verif1=np.allclose(np.dot(matrix_W,soln),rhs)
                
                #print("Correct computation on PSI ?", verif1)
                
                #Removing Xylem and phloem BC terms in "matrix" in case they would change in the next scenario
                if Barrier>0:
                    if not isnan(config.scenarios[count].get("pressure_xyl")): #Pressure xylem BC
                        for cid in network.xylem_cells:
                            matrix_W[cid][cid] += config.k_xyl
                if Barrier==0: #Protophloem only
                    if not isnan(config.scenarios[count].get("pressure_sieve")):
                        for cid in listprotosieve:
                            matrix_W[cid][cid] += config.k_sieve
                elif Barrier>0: #Includes mature phloem
                    if not isnan(config.scenarios[count].get("pressure_sieve")): #Then there is a phloem BC in scenarios (assuming that we did not pick scenarios with and others without)
                        for cid in network.sieve_cells: #both proto and metaphloem
                            matrix_W[cid][cid] += config.k_sieve
                
                #Flow rates at interfaces
                Q_soil=[]
                for ind in network.border_walls:
                    Q=rhs_s[ind]*(soln[ind]-(config.scenarios[count].get("psi_soil_left")*(1-x_rel[ind])+config.scenarios[count].get("psi_soil_right")*x_rel[ind]))
                    Q_soil.append(Q) #(cm^3/d) Positive for water flowing into the root, rhs_s is minus the conductance at the soil root interface
                    if config.apo_contagion==2:
                        if config.sym_contagion==2:
                            if ind not in Apo_w_Zombies0:
                                if Q<0.0:
                                    matrix_C[ind][ind]+=Q
                        else:
                            if ind not in Apo_w_Zombies0:
                                if Q<0.0:
                                    matrix_ApoC[ind][ind]+=Q
                    #if config.C_flag and Os_soil[5][count]==1:
                    #if config.apo_contagion==2:
                    #    #if not Q==0:
                    #    #    list_walls_apo_conv.append(ind)
                    #    if Q>0:
                    #        rhs_ApoC[ind][0] -= Q*config.scenarios[count].get("osmotic_left_soil") #Flow rate to be multiplied by concentration BC
                    #    else:
                    #        matrix_ApoC[ind][ind] += Q
                            
                for ind in network.border_junctions:
                    Q=rhs_s[ind]*(soln[ind]-(config.scenarios[count].get("psi_soil_left")*(1-x_rel[ind])+config.scenarios[count].get("psi_soil_right")*x_rel[ind]))
                    Q_soil.append(Q) #(cm^3/d) Positive for water flowing into the root
                    if config.apo_contagion==2:
                        if config.sym_contagion==2:
                            if ind not in Apo_j_Zombies0:
                                if Q<0.0:
                                    matrix_C[ind][ind]+=Q
                        else:
                            if ind not in Apo_j_Zombies0:
                                if Q<0.0:
                                    matrix_ApoC[ind][ind]+=Q
                    #if config.C_flag and Os_soil[5][count]==1:
                    #    if not Q==0:
                    #        list_walls_apo_conv.append(ind)
                    #    if Q>0:
                    #        rhs_C[ind][0] -= Q*config.scenarios[count].get("osmotic_left_soil") #Flow rate times concentration BC
                    #    else:
                    #        matrix_C[ind][ind] += Q
                
                Q_xyl=[]
                if Barrier>0:
                    if not isnan(config.scenarios[count].get("pressure_xyl")): #Xylem pressure BC
                        for cid in network.xylem_cells:
                            Q=rhs_x[cid]*(soln[cid]-config.scenarios[count].get("pressure_xyl"))
                            Q_xyl.append(Q) #(cm^3/d) Negative for water flowing into xylem tubes
                            rank=int(network.cell_ranks[cid-(network.n_walls + network.n_junctions)])
                            row=int(network.rank_to_row[rank])
                            Q_xyl_layer[row][iMaturity][count] += Q
                            #if config.C_flag:
                            #    if Q>0: #Water leaving the cross-section
                            #        matrix_C[cid][cid] -= Q
                            #    else: #Water entering the cross-section through xylem
                            #        rhs_C[cid][0] += Q #Flow rate times concentration BC
                            #    rhs_C[cid][0] *= config.scenarios[count].get("osmotic_endo")
                    elif not isnan(config.scenarios[count].get("flow_xyl")): #Xylem flow BC
                        for cid in network.xylem_cells:
                            Q=-rhs_x[cid]
                            Q_xyl.append(Q) #(cm^3/d) Negative for water flowing into xylem tubes
                            rank=int(network.cell_ranks[cid-(network.n_walls + network.n_junctions)])
                            row=int(network.rank_to_row[rank])
                            Q_xyl_layer[row][iMaturity][count] += Q
                            #if config.C_flag:
                            #    if Q>0: #Water leaving the cross-section
                            #        matrix_C[cid][cid] -= Q
                            #    else: #Water entering the cross-section through xylem
                            #        rhs_C[cid][0] += Q #Flow rate times concentration BC
                            #    rhs_C[cid][0] *= config.scenarios[count].get("osmotic_endo")
                
                Q_sieve=[]
                if Barrier==0:
                    if not isnan(config.scenarios[count].get("pressure_sieve")): #Phloem pressure BC
                        for cid in listprotosieve: #Q will be 0 for metaphloem if Barrier==0 because rhs_p=0 for these cells
                            Q=rhs_p[cid]*(soln[cid]-config.scenarios[count].get("pressure_sieve"))
                            Q_sieve.append(Q) #(cm^3/d) Positive for water flowing from sieve tubes
                            rank=int(network.cell_ranks[cid-(network.n_walls + network.n_junctions)])
                            row=int(network.rank_to_row[rank])
                            Q_sieve_layer[row][iMaturity][count] += Q
                    elif not isnan(config.scenarios[count].get("flow_sieve")): #Phloem flow BC
                        for cid in listprotosieve:
                            Q=-rhs_p[cid]
                            Q_sieve.append(Q) #(cm^3/d) Negative for water flowing into xylem tubes
                            rank=int(network.cell_ranks[cid-(network.n_walls + network.n_junctions)])
                            row=int(network.rank_to_row[rank])
                            Q_sieve_layer[row][iMaturity][count] += Q
                elif Barrier>0:
                    if not isnan(config.scenarios[count].get("pressure_sieve")): #Phloem pressure BC
                        for cid in network.sieve_cells: #Q will be 0 for metaphloem if Barrier==0 because rhs_p=0 for these cells
                            Q=rhs_p[cid]*(soln[cid]-config.scenarios[count].get("pressure_sieve"))
                            Q_sieve.append(Q) #(cm^3/d) Positive for water flowing from sieve tubes
                            rank=int(network.cell_ranks[cid-(network.n_walls + network.n_junctions)])
                            row=int(network.rank_to_row[rank])
                            Q_sieve_layer[row][iMaturity][count] += Q
                    elif not isnan(config.scenarios[count].get("flow_sieve")): #Phloem flow BC
                        for cid in network.sieve_cells:
                            Q=-rhs_p[cid]
                            Q_sieve.append(Q) #(cm^3/d) Negative for water flowing into xylem tubes
                            rank=int(network.cell_ranks[cid-(network.n_walls + network.n_junctions)])
                            row=int(network.rank_to_row[rank])
                            Q_sieve_layer[row][iMaturity][count] += Q
                Q_elong=-rhs_e #(cm^3/d) The elongation flux virtually disappears from the cross-section => negative
                for cid in range(network.n_cells):
                    rank=int(network.cell_ranks[cid])
                    row=int(network.rank_to_row[rank])
                    Q_elong_layer[row][iMaturity][count] += Q_elong[(network.n_walls + network.n_junctions)+cid]
                Q_tot[iMaturity][count]=sum(Q_soil) #(cm^3/d) Total flow rate at root surface
                for ind in range((network.n_walls + network.n_junctions),len(G.nodes())): #(network.n_walls + network.n_junctions) is the index of the first cell
                    cellnumber1=ind-(network.n_walls + network.n_junctions)
                    rank = int(network.cell_ranks[cellnumber1])
                    row = int(network.rank_to_row[rank])
                    if rank == 1: #Exodermis
                        PsiCellLayer[row][iMaturity][count] += soln[ind]*(STFcell_plus[cellnumber1][iMaturity]+abs(STFcell_minus[cellnumber1][iMaturity]))/(STFlayer_plus[row][iMaturity]+abs(STFlayer_minus[row][iMaturity])+STFlayer_plus[row+1][iMaturity]+abs(STFlayer_minus[row+1][iMaturity])) #(hPa)
                        PsiCellLayer[row+1][iMaturity][count] += soln[ind]*(STFcell_plus[cellnumber1][iMaturity]+abs(STFcell_minus[cellnumber1][iMaturity]))/(STFlayer_plus[row][iMaturity]+abs(STFlayer_minus[row][iMaturity])+STFlayer_plus[row+1][iMaturity]+abs(STFlayer_minus[row+1][iMaturity])) #(hPa)
                    elif rank == 3: #Endodermis
                        if any(config.passage_cell_ids==array(cellnumber1)) and Barrier==2: #Passage cell
                            PsiCellLayer[row+1][iMaturity][count] += soln[ind]*(STFcell_plus[cellnumber1][iMaturity]+abs(STFcell_minus[cellnumber1][iMaturity]))/(STFlayer_plus[row+1][iMaturity]+abs(STFlayer_minus[row+1][iMaturity])+STFlayer_plus[row+2][iMaturity]+abs(STFlayer_minus[row+2][iMaturity])) #(hPa)
                            PsiCellLayer[row+2][iMaturity][count] += soln[ind]*(STFcell_plus[cellnumber1][iMaturity]+abs(STFcell_minus[cellnumber1][iMaturity]))/(STFlayer_plus[row+1][iMaturity]+abs(STFlayer_minus[row+1][iMaturity])+STFlayer_plus[row+2][iMaturity]+abs(STFlayer_minus[row+2][iMaturity])) #(hPa)
                        else:
                            PsiCellLayer[row][iMaturity][count] += soln[ind]*(STFcell_plus[cellnumber1][iMaturity]+abs(STFcell_minus[cellnumber1][iMaturity]))/(STFlayer_plus[row][iMaturity]+abs(STFlayer_minus[row][iMaturity])+STFlayer_plus[row+3][iMaturity]+abs(STFlayer_minus[row+3][iMaturity])) #(hPa)
                            PsiCellLayer[row+3][iMaturity][count] += soln[ind]*(STFcell_plus[cellnumber1][iMaturity]+abs(STFcell_minus[cellnumber1][iMaturity]))/(STFlayer_plus[row][iMaturity]+abs(STFlayer_minus[row][iMaturity])+STFlayer_plus[row+3][iMaturity]+abs(STFlayer_minus[row+3][iMaturity])) #(hPa)
                            if not Barrier==2:
                                PsiCellLayer[row+1][iMaturity][count] = nan
                                PsiCellLayer[row+2][iMaturity][count] = nan
                    elif (ind not in network.xylem_cells) or Barrier==0: #Not for mature xylem
                        PsiCellLayer[row][iMaturity][count] += soln[ind]*(STFcell_plus[cellnumber1][iMaturity]+abs(STFcell_minus[cellnumber1][iMaturity]))/(STFlayer_plus[row][iMaturity]+abs(STFlayer_minus[row][iMaturity])) #(hPa)
                
                if Barrier>0 and isnan(config.scenarios[count].get("pressure_xyl")):
                    config.scenarios[count]["pressure_xyl"]=0.0
                    for cid in network.xylem_cells:
                        config.scenarios[count]["pressure_xyl"]+=soln[cid]/len(network.xylem_cells)
                if Barrier>0:
                    if isnan(config.scenarios[count].get("pressure_sieve")):
                        config.scenarios[count]["pressure_sieve"]=0.0
                        for cid in network.sieve_cells:
                            config.scenarios[count]["pressure_sieve"]+=soln[cid]/len(network.sieve_cells) #Average of phloem water pressures
                elif Barrier==0:
                    if isnan(config.scenarios[count].get("pressure_sieve")):
                        config.scenarios[count]["pressure_sieve"]=0.0
                        for cid in listprotosieve:
                            config.scenarios[count]["pressure_sieve"]+=soln[cid]/Nprotosieve #Average of protophloem water pressures
                
                print("Uptake rate per unit root length: soil ",(sum(Q_soil)/height/1.0E-04),"cm^2/d, xylem ",(sum(Q_xyl)/height/1.0E-04),"cm^2/d, phloem ",(sum(Q_sieve)/height/1.0E-04),"cm^2/d, elongation ",(sum(Q_elong)/height/1.0E-04),"cm^2/d")
                if not isnan(sum(Q_sieve)):
                    print("Mass balance error:",(sum(Q_soil)+sum(Q_xyl)+sum(Q_sieve)+sum(Q_elong))/height/1.0E-04,"cm^2/d")
                else:
                    print("Mass balance error:",(sum(Q_soil)+sum(Q_xyl)+sum(Q_elong))/height/1.0E-04,"cm^2/d")
                
                #################################################################
                ##Calul of Fluxes between nodes and Creating the edge_flux_list##
                #################################################################
                
                #Creating a list for the fluxes
                #edge_flux_list=[]
                
                #Filling the fluxes list
                MembraneFlowDensity=[]
                WallFlowDensity=[]
                WallFlowDensity_cos=[]
                PlasmodesmFlowDensity=[]
                Fjw_list=[]
                Fcw_list=[]
                Fcc_list=[]
                jmb=0 #Index for membrane conductance vector
                for node, edges in G.adjacency() : #adjacency_iter returns an iterator of (node, adjacency dict) tuples for all nodes. This is the fastest way to look at every edge. For directed graphs, only outgoing adjacencies are included.
                    i = network.indice[node] #Node ID number
                    psi = soln[i] #Node water potential
                    psi_o_cell = inf #Opposite cell water potential
                    ind_o_cell = inf #Opposite cell index
                    #Here we count surrounding cell types in order to know if the wall is part of an apoplastic barrier, as well as to know on which side of the exodermis or endodermis the membrane is located
                    count_endo=0 #total number of endodermis cells around the wall
                    count_peri=0 #total number of pericycle cells around the wall
                    count_PPP=0 #total number of network.plasmodesmata_indice cells arount the wall
                    count_exo=0 #total number of exodermis cells around the wall
                    count_epi=0 #total number of epidermis cells around the wall
                    count_stele=0 #total number of stelar parenchyma cells around the wall
                    count_stele_overall=0 #total number of stele cells (of any type) around the wall
                    count_comp=0 #total number of companion cells around the wall
                    count_sieve=0 #total number of stelar parenchyma cells around the wall
                    count_xyl=0 #total number of xylem cells around the wall
                    count_cortex=0 #total number of phloem sieve cells around the wall
                    count_passage=0 #total number of passage cells around the wall
                    count_interC=0 #total number of intercellular spaces around the wall
                    noPD=False #Initializes the flag for wall connected to an intercellular space -> does not have plasmodesmata
                    if i<network.n_walls: #wall ID
                        for neighboor, eattr in edges.items(): #Loop on connections (edges)
                            if eattr['path'] == 'membrane': #Wall connection
                                if any(config.passage_cell_ids==array((network.indice[neighboor])-(network.n_walls + network.n_junctions))):
                                    count_passage+=1
                                if any(config.intercellular_ids==array((network.indice[neighboor])-(network.n_walls + network.n_junctions))):
                                    count_interC+=1
                                if G.nodes[neighboor]['cgroup']==3:#Endodermis
                                    count_endo+=1
                                elif G.nodes[neighboor]['cgroup']==13 or G.nodes[neighboor]['cgroup']==19 or G.nodes[neighboor]['cgroup']==20:#Xylem cell or vessel
                                    count_xyl+=1
                                elif G.nodes[neighboor]['cgroup']==16 or G.nodes[neighboor]['cgroup']==21:#Pericycle or stele
                                    count_peri+=1
                                    if neighboor in network.plasmodesmata_indice:
                                        count_PPP+=1
                                elif G.nodes[neighboor]['cgroup']==1:#Exodermis
                                    count_exo+=1
                                elif G.nodes[neighboor]['cgroup']==2:#Epidermis
                                    count_epi+=1
                                elif G.nodes[neighboor]['cgroup']==4:#Cortex
                                    count_cortex+=1
                                elif G.nodes[neighboor]['cgroup']==5:#Stelar parenchyma
                                    count_stele+=1
                                elif G.nodes[neighboor]['cgroup']==11 or G.nodes[neighboor]['cgroup']==23:#Phloem sieve tube
                                    count_sieve+=1
                                elif G.nodes[neighboor]['cgroup']==12 or G.nodes[neighboor]['cgroup']==26:#Companion cell
                                    count_comp+=1
                                if G.nodes[neighboor]['cgroup']>4:#Stele overall
                                    count_stele_overall+=1
                    ijunction=0
                    for neighboor, eattr in edges.items(): #Loop on connections (edges)
                        j = network.indice[neighboor] #Neighbouring node ID number
                        #if j > i: #Only treating the information one way to save time
                        psin = soln[j] #Neighbouring node water potential
                        path = eattr['path'] #eattr is the edge attribute (i.e. connection type)
                        if i<network.n_walls:
                            if config.paraview==1 or config.par_track==1 or config.apo_contagion>0 or config.sym_contagion>0:
                                if path == "wall":
                                    #K = eattr['kw']*1.0E-04*((eattr['lat_dist']+height)*eattr['thickness']-square(eattr['thickness']))/eattr['length'] #Junction-Wall conductance (cm^3/hPa/d)
                                    if (count_interC>=2 and Barrier>0) or (count_xyl==2 and config.xylem_pieces): #"Fake wall" splitting an intercellular space or a xylem cell in two
                                        K = 1.0E-16 #Non conductive
                                    elif count_cortex>=2: #wall between two cortical cells
                                        K = kw_cortex_cortex*1.0E-04*((eattr['lat_dist']+height)*config.thickness-square(config.thickness))/eattr['length'] #Junction-Wall conductance (cm^3/hPa/d)
                                    elif count_endo>=2: #wall between two endodermis cells
                                        K = kw_endo_endo*1.0E-04*((eattr['lat_dist']+height)*config.thickness-square(config.thickness))/eattr['length'] #Junction-Wall conductance (cm^3/hPa/d)
                                    elif count_stele_overall>0 and count_endo>0: #wall between endodermis and pericycle
                                        if count_passage>0:
                                            K = kw_passage*1.0E-04*((eattr['lat_dist']+height)*config.thickness-square(config.thickness))/eattr['length']
                                        else:
                                            K = kw_endo_peri*1.0E-04*((eattr['lat_dist']+height)*config.thickness-square(config.thickness))/eattr['length'] #Junction-Wall conductance (cm^3/hPa/d)
                                    elif count_stele_overall==0 and count_endo==1: #wall between endodermis and cortex
                                        if count_passage>0:
                                            K = kw_passage*1.0E-04*((eattr['lat_dist']+height)*config.thickness-square(config.thickness))/eattr['length']
                                        else:
                                            K = kw_endo_cortex*1.0E-04*((eattr['lat_dist']+height)*config.thickness-square(config.thickness))/eattr['length'] #Junction-Wall conductance (cm^3/hPa/d)
                                    elif count_exo>=2: #wall between two exodermis cells
                                        K = kw_exo_exo*1.0E-04*((eattr['lat_dist']+height)*config.thickness-square(config.thickness))/eattr['length'] #Junction-Wall conductance (cm^3/hPa/d)
                                    else: #other walls
                                        K = kw*1.0E-04*((eattr['lat_dist']+height)*config.thickness-square(config.thickness))/eattr['length'] #Junction-Wall conductance (cm^3/hPa/d)
                                    Fjw = K * (psin - psi) * sign(j-i) #(cm^3/d) Water flow rate positive from junction to wall
                                    Fjw_list.append((i,j,Fjw))
                                    #The ordering in WallFlowDensity will correspond to the one of ThickWallsX, saved for display only
                                    WallFlowDensity.append((i,j, Fjw / (((eattr['lat_dist']+height)*config.thickness-square(config.thickness))*1.0E-08))) # (cm/d) Positive towards lower node ID 
                                    cos_angle=(network.position[i][0]-network.position[j][0])/(hypot(network.position[j][0]-network.position[i][0],network.position[j][1]-network.position[i][1])) #Vectors junction1-wall
                                    WallFlowDensity_cos.append((i,j, cos_angle * Fjw / (((eattr['lat_dist']+height)*config.thickness-square(config.thickness))*1.0E-08))) # (cm/d) Positive towards lower node ID 
                                    #if config.C_flag and Os_soil[5][count]*Os_xyl[5][count]==1:
                                    if config.apo_contagion==2:
                                        if config.sym_contagion==2: # Apo & Sym contagion
                                            if Fjw>0: #Flow from junction to wall
                                                if i not in Apo_w_Zombies0:
                                                    matrix_C[i][j] += Fjw
                                                if j not in Apo_j_Zombies0:
                                                    matrix_C[j][j] -= Fjw
                                            else: #Flow from wall to junction
                                                if i not in Apo_w_Zombies0:
                                                    matrix_C[i][i] += Fjw
                                                if j not in Apo_j_Zombies0:
                                                    matrix_C[j][i] -= Fjw
                                        else: #Only Apo contagion
                                            if Fjw>0: #Flow from junction to wall
                                                if i not in Apo_w_Zombies0:
                                                    matrix_ApoC[i][j] += Fjw
                                                if j not in Apo_j_Zombies0:
                                                    matrix_ApoC[j][j] -= Fjw
                                            else: #Flow from wall to junction
                                                if i not in Apo_w_Zombies0:
                                                    matrix_ApoC[i][i] += Fjw
                                                if j not in Apo_j_Zombies0:
                                                    matrix_ApoC[j][i] -= Fjw
                                    
                                    if config.apo_contagion==1:
                                        if Fjw>0:
                                            Apo_connec_flow[j][nApo_connec_flow[j]]=i
                                            nApo_connec_flow[j]+=1
                                        elif Fjw<0:
                                            Apo_connec_flow[i][nApo_connec_flow[i]]=j
                                            nApo_connec_flow[i]+=1
                                elif path == "membrane": #Membrane connection
                                    #K = (eattr['kmb']+eattr['kaqp'])*1.0E-08*(height+eattr['dist'])*eattr['length']
                                    if G.nodes[j]['cgroup']==1: #Exodermis
                                        kaqp=kaqp_exo
                                    elif G.nodes[j]['cgroup']==2: #Epidermis
                                        kaqp=kaqp_epi
                                    elif G.nodes[j]['cgroup']==3: #Endodermis
                                        kaqp=kaqp_endo
                                    elif G.nodes[j]['cgroup']==13 or G.nodes[j]['cgroup']==19 or G.nodes[j]['cgroup']==20: #xylem cell or vessel
                                        if Barrier>0: #Xylem vessel
                                            kaqp=kaqp_stele*10000 #No membrane resistance because no membrane
                                            noPD=True
                                        elif Barrier==0: #Xylem cell
                                            kaqp=kaqp_stele
                                            if (count_xyl==2 and config.xylem_pieces):
                                                noPD=True
                                    elif G.nodes[j]['cgroup']>4: #Stele and pericycle
                                        kaqp=kaqp_stele
                                    elif (j-(network.n_walls + network.n_junctions) in config.intercellular_ids) and Barrier>0: #the neighbour is an intercellular space "cell"
                                        kaqp=config.k_interc
                                        noPD=True
                                    elif G.nodes[j]['cgroup']==4: #Cortex
                                        kaqp=float(a_cortex*network.distance_to_center[wid]*1.0E-04+b_cortex) #AQP activity (cm/hPa/d)
                                        if kaqp < 0:
                                            error('Error, negative kaqp in cortical cell, adjust Paqp_cortex')
                                    #Calculating conductances
                                    if count_endo>=2: #wall between two endodermis cells, in this case the suberized wall can limit the transfer of water between cell and wall
                                        if kw_endo_endo==0.00:
                                            K=0.00
                                        else:
                                            K = 1/(1/(kw_endo_endo/(config.thickness/2*1.0E-04))+1/(config.kmb+kaqp))*1.0E-08*(height+eattr['dist'])*eattr['length'] #(cm^3/hPa/d)
                                    elif count_exo>=2: #wall between two exodermis cells, in this case the suberized wall can limit the transfer of water between cell and wall
                                        if kw_exo_exo==0.00:
                                            K=0.00
                                        else:
                                            K = 1/(1/(kw_exo_exo/(config.thickness/2*1.0E-04))+1/(config.kmb+kaqp))*1.0E-08*(height+eattr['dist'])*eattr['length'] #(cm^3/hPa/d)
                                    elif count_stele_overall>0 and count_endo>0: #wall between endodermis and pericycle, in this case the suberized wall can limit the transfer of water between cell and wall
                                        if count_passage>0:
                                            K = 1/(1/(kw_passage/(config.thickness/2*1.0E-04))+1/(config.kmb+kaqp))*1.0E-08*(height+eattr['dist'])*eattr['length']
                                        else:
                                            if kw_endo_peri==0.00:
                                                K=0.00
                                            else:
                                                K = 1/(1/(kw_endo_peri/(config.thickness/2*1.0E-04))+1/(config.kmb+kaqp))*1.0E-08*(height+eattr['dist'])*eattr['length']
                                    elif count_stele_overall==0 and count_endo==1: #wall between endodermis and cortex, in this case the suberized wall can limit the transfer of water between cell and wall
                                        if kaqp==0.0:
                                            K=1.00E-16
                                        else:
                                            if count_passage>0:
                                                K = 1/(1/(kw_passage/(config.thickness/2*1.0E-04))+1/(config.kmb+kaqp))*1.0E-08*(height+eattr['dist'])*eattr['length']
                                            else:
                                                if kw_endo_cortex==0.00:
                                                    K=0.00
                                                else:
                                                    K = 1/(1/(kw_endo_cortex/(config.thickness/2*1.0E-04))+1/(config.kmb+kaqp))*1.0E-08*(height+eattr['dist'])*eattr['length']
                                    else:
                                        if kaqp==0.0:
                                            K=1.00E-16
                                        else:
                                            K = 1/(1/(kw/(config.thickness/2*1.0E-04))+1/(config.kmb+kaqp))*1.0E-08*(height+eattr['dist'])*eattr['length'] #(cm^3/hPa/d)
                                    Fcw = K * (psi - psin + s_membranes[jmb]*(Os_walls[i] - Os_cells[j-(network.n_walls + network.n_junctions)])) #(cm^3/d) Water flow rate positive from wall to protoplast
                                    Fcw_list.append((i,j,-Fcw,s_membranes[jmb])) #Water flow rate positive from protoplast to wall
                                    #Flow densities calculation
                                    #The ordering in MembraneFlowDensity will correspond to the one of ThickWalls, saved for display only 
                                    MembraneFlowDensity.append(Fcw / (1.0E-08*(height+eattr['dist'])*eattr['length']))
                                    ####Solute convection across membranes####
                                    if config.apo_contagion==2 and config.sym_contagion==2:
                                        if Fcw>0: #Flow from wall to protoplast
                                            if i not in Apo_w_Zombies0:
                                                if config.d2o1==1:#Solute that moves across membranes like water 
                                                    matrix_C[i][i] -= Fcw
                                                else: #Solute that moves across membranes independently of water (the membrane is possibly not one) 
                                                    matrix_C[i][i] -= Fcw*(1-s_membranes[jmb])
                                            if j-(network.n_walls + network.n_junctions) not in config.sym_zombie0:
                                                if config.d2o1==1:#Solute that moves across membranes like water 
                                                    matrix_C[j][i] += Fcw
                                                else: #Solute that moves across membranes independently of water (the membrane is possibly not one) 
                                                    matrix_C[j][i] += Fcw*(1-s_membranes[jmb])
                                        else: #Flow from protoplast to wall
                                            if j-(network.n_walls + network.n_junctions) not in config.sym_zombie0:
                                                if config.d2o1==1:#Solute that moves across membranes like water 
                                                    matrix_C[j][j] += Fcw
                                                else: #Solute that moves across membranes independently of water (the membrane is possibly not one) 
                                                    matrix_C[j][j] += Fcw*(1-s_membranes[jmb])
                                            if i not in Apo_w_Zombies0:
                                                if config.d2o1==1:#Solute that moves across membranes like water 
                                                    matrix_C[i][j] -= Fcw
                                                else: #Solute that moves across membranes independently of water (the membrane is possibly not one) 
                                                    matrix_C[i][j] -= Fcw*(1-s_membranes[jmb])
                                    
                                    #Macroscopic distributed parameter for transmembrane flow
                                    #Discretization based on cell layers and apoplasmic barriers
                                    rank = int(network.cell_ranks[j-(network.n_walls + network.n_junctions)])
                                    row = int(network.rank_to_row[rank])
                                    if rank == 1 and count_epi > 0: #Outer exodermis
                                        row += 1
                                    if rank == 3 and count_cortex > 0: #Outer endodermis
                                        if any(config.passage_cell_ids==array(j-(network.n_walls + network.n_junctions))) and Barrier==2:
                                            row += 2
                                        else:
                                            row += 3
                                    elif rank == 3 and count_stele_overall > 0: #Inner endodermis
                                        if any(config.passage_cell_ids==array(j-(network.n_walls + network.n_junctions))) and Barrier==2:
                                            row += 1
                                            #print('PsiWallPassage:',psi)
                                    Flow = K * (psi - psin + s_membranes[jmb]*(Os_walls[i] - Os_cells[j-(network.n_walls + network.n_junctions)]))
                                    jmb+=1
                                    if ((j-(network.n_walls + network.n_junctions) not in config.intercellular_ids) and (j not in network.xylem_cells)) or Barrier==0: #No aerenchyma in the elongation zone
                                        if Flow > 0 :
                                            UptakeLayer_plus[row][iMaturity][count] += Flow #grouping membrane flow rates in cell layers
                                        else:
                                            UptakeLayer_minus[row][iMaturity][count] += Flow
                                    
                                    if K>1.0e-18: #Not an impermeable wall
                                        PsiWallLayer[row][iMaturity][count] += psi
                                        NWallLayer[row][iMaturity][count] += 1
                                    
                                    if psi_o_cell == inf:
                                        psi_o_cell=psin
                                        ind_o_cell=j
                                    else:
                                        if noPD: #No plasmodesmata because the wall i is connected to an intercellular space or xylem vessel
                                            temp=0 #The ordering in PlasmodesmFlowDensity will correspond to the one of ThickWalls except for boderline walls, saved for display only                        
                                        elif count_epi==1 and count_exo==1: #wall between epidermis and exodermis
                                            temp=Kpl*config.fplxheight_epi_exo * (psin - psi_o_cell)
                                        elif (count_exo==1 or count_epi==1) and count_cortex==1: #wall between exodermis and cortex
                                            temp1=float(config.kpl_elems[iPD].get("cortex_factor"))
                                            temp=Kpl*2*temp1/(temp1+1)*config.fplxheight_outer_cortex * network.len_outer_cortex/ network.cross_section_outer_cortex * (psin - psi_o_cell)
                                        elif count_cortex==2: #wall between cortical cells
                                            temp1=float(config.kpl_elems[iPD].get("cortex_factor"))
                                            temp=Kpl*temp1*config.fplxheight_cortex_cortex * network.len_cortex_cortex/ network.cross_section_cortex_cortex * (psin - psi_o_cell)
                                        elif count_cortex==1 and count_endo==1: #wall between cortex and endodermis
                                            temp1=float(config.kpl_elems[iPD].get("cortex_factor"))
                                            temp=Kpl*2*temp1/(temp1+1)*config.fplxheight_cortex_endo * network.len_cortex_endo/ network.cross_section_cortex_endo * (psin - psi_o_cell)
                                        elif count_endo==2: #wall between endodermal cells
                                            temp=Kpl*config.fplxheight_endo_endo * (psin - psi_o_cell)
                                        elif count_stele_overall>0 and count_endo>0: #wall between endodermis and pericycle
                                            if count_PPP>0:
                                                temp1=float(config.kpl_elems[iPD].get("PPP_factor"))
                                            else:
                                                temp1=1
                                            temp=Kpl*2*temp1/(temp1+1)*config.fplxheight_endo_peri * (psin - psi_o_cell)
                                        elif count_stele==2: #wall between stelar parenchyma cells
                                            temp=Kpl*config.fplxheight_stele_stele * (psin - psi_o_cell)
                                        elif count_peri>0 and count_stele==1: #wall between stele and pericycle
                                            if count_PPP>0:
                                                temp1=float(config.kpl_elems[iPD].get("PPP_factor"))
                                            else:
                                                temp1=1
                                            temp=Kpl*2*temp1/(temp1+1)*config.fplxheight_peri_stele * (psin - psi_o_cell)
                                        elif count_comp==1 and count_stele==1: #wall between stele and companion cell
                                            temp1=float(config.kpl_elems[iPD].get("PCC_factor"))
                                            temp=Kpl*2*temp1/(temp1+1)*config.fplxheight_stele_comp * (psin - psi_o_cell)
                                        elif count_peri==1 and count_comp==1: #wall between pericycle and companion cell
                                            temp1=float(config.kpl_elems[iPD].get("PCC_factor"))
                                            if count_PPP>0:
                                                temp2=float(config.kpl_elems[iPD].get("PPP_factor"))
                                            else:
                                                temp2=1
                                            temp=Kpl*2*temp1*temp2/(temp1+temp2)*config.fplxheight_peri_comp * (psin - psi_o_cell)
                                        elif count_comp==2: #wall between companion cells 
                                            temp1=float(config.kpl_elems[iPD].get("PCC_factor"))
                                            temp=Kpl*temp1*config.fplxheight_peri_comp * (psin - psi_o_cell)
                                        elif count_comp==1 and count_sieve==1: #wall between companion cell and sieve tube
                                            temp1=float(config.kpl_elems[iPD].get("PCC_factor"))
                                            temp=Kpl*2*temp1/(temp1+1)*config.fplxheight_comp_sieve * (psin - psi_o_cell)
                                        elif count_peri==1 and count_sieve==1: #wall between stele and sieve tube
                                            temp=Kpl*config.fplxheight_peri_sieve * (psin - psi_o_cell)
                                        elif count_stele==1 and count_sieve==1: #wall between stele and pericycle
                                            if count_PPP>0:
                                                temp1=float(config.kpl_elems[iPD].get("PPP_factor"))
                                            else:
                                                temp1=1
                                            temp=Kpl*2*temp1/(temp1+1)*config.fplxheight_stele_sieve * (psin - psi_o_cell)
                                        else: #Default plasmodesmatal frequency
                                            temp=Kpl*config.fplxheight * (psin - psi_o_cell)  #The ordering in PlasmodesmFlowDensity will correspond to the one of ThickWalls except for boderline walls, saved for display only 
                                        PlasmodesmFlowDensity.append(temp/(1.0E-04*height))
                                        PlasmodesmFlowDensity.append(-temp/(1.0E-04*height))
                                        Fcc=temp*1.0E-04*eattr['length']*sign(j-ind_o_cell)
                                        if ind_o_cell<j:
                                            Fcc_list.append((ind_o_cell,j,Fcc)) #(cm^3/d) Water flow rate positive from high index to low index cell
                                        else:
                                            Fcc_list.append((j,ind_o_cell,Fcc))
                                        #if config.C_flag:
                                        if config.sym_contagion==2: #Convection across plasmodesmata
                                            if config.apo_contagion==2: #Apo & Sym Contagion
                                                if Fcc>0: #Flow from high index to low index cell
                                                    if ind_o_cell<j: #From j to ind_o_cell
                                                        if j-(network.n_walls + network.n_junctions) not in config.sym_zombie0:
                                                            matrix_C[j][j] -= Fcc
                                                        if ind_o_cell-(network.n_walls + network.n_junctions) not in config.sym_zombie0:
                                                            matrix_C[ind_o_cell][j] += Fcc
                                                    else: #From ind_o_cell to j
                                                        if ind_o_cell-(network.n_walls + network.n_junctions) not in config.sym_zombie0:
                                                            matrix_C[ind_o_cell][ind_o_cell] -= Fcc
                                                        if j-(network.n_walls + network.n_junctions) not in config.sym_zombie0:
                                                            matrix_C[j][ind_o_cell] += Fcc
                                                else: #Flow from low index to high index cell
                                                    if ind_o_cell<j: #From ind_o_cell to j
                                                        if ind_o_cell-(network.n_walls + network.n_junctions) not in config.sym_zombie0:
                                                            matrix_C[ind_o_cell][ind_o_cell] += Fcc
                                                        if j-(network.n_walls + network.n_junctions) not in config.sym_zombie0:
                                                            matrix_C[j][ind_o_cell] -= Fcc
                                                    else: #From j to ind_o_cell
                                                        if j-(network.n_walls + network.n_junctions) not in config.sym_zombie0:
                                                            matrix_C[j][j] += Fcc
                                                        if ind_o_cell-(network.n_walls + network.n_junctions) not in config.sym_zombie0:
                                                            matrix_C[ind_o_cell][j] -= Fcc
                                            else: #Only Sym contagion
                                                if Fcc>0: #Flow from high index to low index cell
                                                    if ind_o_cell<j: #From j to ind_o_cell
                                                        if j-(network.n_walls + network.n_junctions) not in config.sym_zombie0:
                                                            matrix_SymC[j-(network.n_walls + network.n_junctions)][j-(network.n_walls + network.n_junctions)] -= Fcc
                                                        if ind_o_cell-(network.n_walls + network.n_junctions) not in config.sym_zombie0:
                                                            matrix_SymC[ind_o_cell-(network.n_walls + network.n_junctions)][j-(network.n_walls + network.n_junctions)] += Fcc
                                                    else: #From ind_o_cell to j
                                                        if ind_o_cell-(network.n_walls + network.n_junctions) not in config.sym_zombie0:
                                                            matrix_SymC[ind_o_cell-(network.n_walls + network.n_junctions)][ind_o_cell-(network.n_walls + network.n_junctions)] -= Fcc
                                                        if j-(network.n_walls + network.n_junctions) not in config.sym_zombie0:
                                                            matrix_SymC[j-(network.n_walls + network.n_junctions)][ind_o_cell-(network.n_walls + network.n_junctions)] += Fcc
                                                else: #Flow from low index to high index cell
                                                    if ind_o_cell<j: #From ind_o_cell to j
                                                        if ind_o_cell-(network.n_walls + network.n_junctions) not in config.sym_zombie0:
                                                            matrix_SymC[ind_o_cell-(network.n_walls + network.n_junctions)][ind_o_cell-(network.n_walls + network.n_junctions)] += Fcc
                                                        if j-(network.n_walls + network.n_junctions) not in config.sym_zombie0:
                                                            matrix_SymC[j-(network.n_walls + network.n_junctions)][ind_o_cell-(network.n_walls + network.n_junctions)] -= Fcc
                                                    else: #From j to ind_o_cell
                                                        if j-(network.n_walls + network.n_junctions) not in config.sym_zombie0:
                                                            matrix_SymC[j-(network.n_walls + network.n_junctions)][j-(network.n_walls + network.n_junctions)] += Fcc
                                                        if ind_o_cell-(network.n_walls + network.n_junctions) not in config.sym_zombie0:
                                                            matrix_SymC[ind_o_cell-(network.n_walls + network.n_junctions)][j-(network.n_walls + network.n_junctions)] -= Fcc
                                        
                                        if config.sym_contagion==1:
                                            itemp=0
                                            while not Cell_connec[ind_o_cell-(network.n_walls + network.n_junctions)][itemp] == j-(network.n_walls + network.n_junctions):
                                                itemp+=1
                                            Cell_connec_flow[ind_o_cell-(network.n_walls + network.n_junctions)][itemp]=sign(temp)
                                            itemp=0
                                            while not Cell_connec[j-(network.n_walls + network.n_junctions)][itemp] == ind_o_cell-(network.n_walls + network.n_junctions):
                                                itemp+=1
                                            Cell_connec_flow[j-(network.n_walls + network.n_junctions)][itemp]=-sign(temp)
                            elif config.paraview==0 and config.par_track==0:
                                if path == "membrane": #Membrane connection
                                    K=Kmb[jmb]
                                    #Flow densities calculation
                                    #Macroscopic distributed parameter for transmembrane flow
                                    #Discretization based on cell layers and apoplasmic barriers
                                    rank = int(network.cell_ranks[j-(network.n_walls + network.n_junctions)])
                                    row = int(network.rank_to_row[rank])
                                    if rank == 1 and count_epi > 0: #Outer exodermis
                                        row += 1
                                    if rank == 3 and count_cortex > 0: #Outer endodermis
                                        if any(config.passage_cell_ids==array(j-(network.n_walls + network.n_junctions))) and Barrier==2:
                                            row += 2
                                        else:
                                            row += 3
                                    elif rank == 3 and count_stele_overall > 0: #Inner endodermis
                                        if any(config.passage_cell_ids==array(j-(network.n_walls + network.n_junctions))) and Barrier==2:
                                            row += 1
                                    Flow = K * (psi - psin + s_membranes[jmb]*(Os_walls[i] - Os_cells[j-(network.n_walls + network.n_junctions)]))
                                    jmb+=1
                                    if ((j-(network.n_walls + network.n_junctions) not in config.intercellular_ids) and (j not in network.xylem_cells)) or Barrier==0:
                                        if Flow > 0 :
                                            UptakeLayer_plus[row][iMaturity][count] += Flow #grouping membrane flow rates in cell layers
                                        else:
                                            UptakeLayer_minus[row][iMaturity][count] += Flow
                                    
                                    if K>1.0e-12: #Not an impermeable wall
                                        PsiWallLayer[row][iMaturity][count] += psi
                                        NWallLayer[row][iMaturity][count] += 1
                
                #if config.C_flag: #Calculates stationary solute concentration
                if config.apo_contagion==2 or config.sym_contagion==2: #Sym & Apo contagion
                    if config.apo_contagion==2 and config.sym_contagion==2: #Sym & Apo contagion
                        #Solving apoplastic & symplastic concentrations
                        soln_C = np.linalg.solve(matrix_C,rhs_C) #Solving the equation to get apoplastic relative concentrations
                    elif config.apo_contagion==2:
                        #Solving apoplastic concentrations
                        soln_ApoC = np.linalg.solve(matrix_ApoC,rhs_ApoC) #Solving the equation to get apoplastic & symplastic relative concentrations
                    else: # Only Symplastic contagion
                        #Solving apoplastic concentrations
                        soln_SymC = np.linalg.solve(matrix_SymC,rhs_SymC) #Solving the equation to get symplastic relative concentrations
                    
                    ##Including BC diffusion terms
                    #for wid in listxylwalls:
                    #    temp=1.0E-04*(network.length[wid]*height)/thickness #Section to length ratio (cm) for the xylem wall
                    #    if not temp==0:
                    #        list_walls_apo_diff.append(wid)
                    #    matrix_C[wid][wid] -= temp*Diff1 #Adding BC diffusion term
                    #    rhs_C[wid][0] -= temp*Diff1*config.scenarios[count].get("osmotic_xyl") #new #Adding BC diffusion term
                    #for wid in network.border_walls:
                    #    if (position[wid][0]>=Xcontact) or (Wall2Cell[wid][0]-(network.n_walls + network.n_junctions) in Contact): #Wall (not including junctions) connected to soil
                    #        temp=1.0E-04*(network.length[wid]/2*height)/(thickness/2)
                    #        if not temp==0:
                    #            list_walls_apo_diff.append(wid)
                    #        matrix_C[wid][wid] -= temp*Diff1 #Adding diffusion BC at soil junction
                    #        rhs_C[wid][0] -= temp*Diff1*config.scenarios[count].get("osmotic_left_soil") #Adding BC diffusion term
                    #for jid in network.border_junctions:
                    #    if (position[jid][0]>=Xcontact) or (Junction2Wall2Cell[jid-network.n_walls][0]-(network.n_walls + network.n_junctions) in Contact) or (Junction2Wall2Cell[jid-network.n_walls][1]-(network.n_walls + network.n_junctions) in Contact) or (Junction2Wall2Cell[jid-network.n_walls][2]-(network.n_walls + network.n_junctions) in Contact): #Junction connected to soil
                    #        temp=1.0E-04*(network.length[jid]*height)/(thickness/2)
                    #        if not temp==0:
                    #            list_walls_apo_diff.append(jid)
                    #        matrix_C[jid][jid] -= temp*Diff1 #Adding diffusion BC at soil junction
                    #        rhs_C[jid][0] -= temp*Diff1*config.scenarios[count].get("osmotic_left_soil") #Adding BC diffusion term
                    #
                    #Nwalls_apo_diff=np.zeros(((network.n_walls + network.n_junctions),2))
                    #Nwalls_apo_conv=np.zeros(((network.n_walls + network.n_junctions),2))
                    #Nwalls_TM_conv=np.zeros(((network.n_walls + network.n_junctions),2))
                    #for wid in list_walls_apo_diff:
                    #    Nwalls_apo_diff[wid][1]+=1
                    #    Nwalls_apo_diff[wid][0]=wid
                    #for wid in list_walls_apo_conv:
                    #    Nwalls_apo_conv[wid][1]+=1
                    #    Nwalls_apo_conv[wid][0]=wid
                    #for wid in list_walls_TM_conv:
                    #    Nwalls_TM_conv[wid][1]+=1
                    #    Nwalls_TM_conv[wid][0]=wid
                    #
                    
                    #Solving apoplastic concentrations
                    #soln_C = np.linalg.solve(matrix_C,rhs_C) #Solving the equation to get potentials inside the network
                    
                #Resets matrix_C and rhs_C to geometrical factor values
                if config.apo_contagion==2:
                    if config.sym_contagion==2: # Apo & Sym contagion
                        for i,j,Fjw in Fjw_list:
                            if Fjw>0: #Flow from junction to wall
                                if i not in Apo_w_Zombies0:
                                    matrix_C[i][j] -= Fjw #Removing convective term
                                if j not in Apo_j_Zombies0:
                                    matrix_C[j][j] += Fjw #Removing convective term
                            else: #Flow from wall to junction
                                if i not in Apo_w_Zombies0:
                                    matrix_C[i][i] -= Fjw #Removing convective term
                                if j not in Apo_j_Zombies0:
                                    matrix_C[j][i] += Fjw #Removing convective term
                    else: #Only Apo contagion
                        for i,j,Fjw in Fjw_list:
                            if Fjw>0: #Flow from junction to wall
                                if i not in Apo_w_Zombies0:
                                    matrix_ApoC[i][j] -= Fjw #Removing convective term
                                if j not in Apo_j_Zombies0:
                                    matrix_ApoC[j][j] += Fjw #Removing convective term
                            else: #Flow from wall to junction
                                if i not in Apo_w_Zombies0:
                                    matrix_ApoC[i][i] -= Fjw #Removing convective term
                                if j not in Apo_j_Zombies0:
                                    matrix_ApoC[j][i] += Fjw #Removing convective term
                
                if config.sym_contagion==2: #Convection across plasmodesmata
                    if config.apo_contagion==2: #Apo & Sym Contagion
                        for i,j,Fcc in Fcc_list:
                            if Fcc>0: #Flow from j to i
                                if j-(network.n_walls + network.n_junctions) not in config.sym_zombie0:
                                    matrix_C[j][j] += Fcc #Removing convective term
                                if i-(network.n_walls + network.n_junctions) not in config.sym_zombie0:
                                    matrix_C[i][j] -= Fcc #Removing convective term
                            else: #Flow from i to j
                                if i-(network.n_walls + network.n_junctions) not in config.sym_zombie0:
                                    matrix_C[i][i] -= Fcc #Removing convective term
                                if j-(network.n_walls + network.n_junctions) not in config.sym_zombie0:
                                    matrix_C[j][i] += Fcc #Removing convective term
                    else: #Only Sym contagion
                        for i,j,Fcc in Fcc_list:
                            if Fcc>0: #Flow from j to i
                                if j-(network.n_walls + network.n_junctions) not in config.sym_zombie0:
                                    matrix_SymC[j-(network.n_walls + network.n_junctions)][j-(network.n_walls + network.n_junctions)] += Fcc #Removing convective term
                                if ind_o_cell-(network.n_walls + network.n_junctions) not in config.sym_zombie0:
                                    matrix_SymC[i-(network.n_walls + network.n_junctions)][j-(network.n_walls + network.n_junctions)] -= Fcc #Removing convective term
                            else: #Flow from i to j
                                if i-(network.n_walls + network.n_junctions) not in config.sym_zombie0:
                                    matrix_SymC[i-(network.n_walls + network.n_junctions)][i-(network.n_walls + network.n_junctions)] -= Fcc #Removing convective term
                                if j-(network.n_walls + network.n_junctions) not in config.sym_zombie0:
                                    matrix_SymC[j-(network.n_walls + network.n_junctions)][i-(network.n_walls + network.n_junctions)] += Fcc #Removing convective term
                
                if config.apo_contagion==2 and config.sym_contagion==2:
                    for i,j,Fcw,s in Fcw_list:
                        Fcw=-Fcw #Attention, -Fcw was saved
                        if Fcw>0: #Flow from wall to protoplast
                            if i not in Apo_w_Zombies0:
                                if config.d2o1==1:#Solute that moves across membranes like water 
                                    matrix_C[i][i] += Fcw #Removing convective term
                                else: #Solute that moves across membranes independently of water (the membrane is possibly not one) 
                                    matrix_C[i][i] += Fcw*(1-s) #Removing convective term
                            if j-(network.n_walls + network.n_junctions) not in config.sym_zombie0:
                                if config.d2o1==1:#Solute that moves across membranes like water 
                                    matrix_C[j][i] -= Fcw #Removing convective term
                                else: #Solute that moves across membranes independently of water (the membrane is possibly not one) 
                                    matrix_C[j][i] -= Fcw*(1-s) #Removing convective term
                        else: #Flow from protoplast to wall
                            if j-(network.n_walls + network.n_junctions) not in config.sym_zombie0:
                                if config.d2o1==1:#Solute that moves across membranes like water 
                                    matrix_C[j][j] -= Fcw #Removing convective term
                                else: #Solute that moves across membranes independently of water (the membrane is possibly not one) 
                                    matrix_C[j][j] -= Fcw*(1-s) #Removing convective term
                            if i not in Apo_w_Zombies0:
                                if config.d2o1==1:#Solute that moves across membranes like water 
                                    matrix_C[i][j] += Fcw #Removing convective term
                                else: #Solute that moves across membranes independently of water (the membrane is possibly not one) 
                                    matrix_C[i][j] += Fcw*(1-s) #Removing convective term
                
                if config.apo_contagion==2:
                    if config.sym_contagion==2: # Apo & Sym contagion
                        i=0
                        for ind in network.border_walls:
                            if ind not in Apo_w_Zombies0:
                                Q=Q_soil[i] #(cm^3/d) Positive for water flowing into the root, rhs_s is minus the conductance at the soil root interface
                                if Q<0.0:
                                    matrix_C[ind][ind]-=Q #Removing convective term
                            i+=1
                        for ind in network.border_junctions:
                            if ind not in Apo_j_Zombies0:
                                Q=Q_soil[i] #(cm^3/d) Positive for water flowing into the root, rhs_s is minus the conductance at the soil root interface
                                if Q<0.0:
                                    matrix_C[ind][ind]-=Q #Removing convective term
                            i+=1
                    else:
                        i=0
                        for ind in network.border_walls:
                            if ind not in Apo_w_Zombies0:
                                Q=Q_soil[i] #(cm^3/d) Positive for water flowing into the root, rhs_s is minus the conductance at the soil root interface
                                if Q<0.0:
                                    matrix_ApoC[ind][ind]-=Q #Removing convective term
                            i+=1
                        for ind in network.border_junctions:
                            if ind not in Apo_j_Zombies0:
                                Q=Q_soil[i] #(cm^3/d) Positive for water flowing into the root, rhs_s is minus the conductance at the soil root interface
                                if Q<0.0:
                                    matrix_ApoC[ind][ind]-=Q #Removing convective term
                            i+=1
                
                    ##Removing diffusion terms linked to BC
                    #for jid in network.border_junctions:
                    #    if (position[jid][0]>=Xcontact) or (Junction2Wall2Cell[jid-network.n_walls][0]-(network.n_walls + network.n_junctions) in Contact) or (Junction2Wall2Cell[jid-network.n_walls][1]-(network.n_walls + network.n_junctions) in Contact) or (Junction2Wall2Cell[jid-network.n_walls][2]-(network.n_walls + network.n_junctions) in Contact): #Junction connected to soil
                    #        temp=1.0E-04*(network.length[jid]*height)/(thickness/2)
                    #        matrix_C[jid][jid] += temp*Diff1 #Removing diffusion BC at soil junction
                    #        rhs_C[jid][0] += temp*Diff1*config.scenarios[count].get("osmotic_left_soil") #Removing BC diffusion term
                    #for wid in network.border_walls:
                    #    if (position[wid][0]>=Xcontact) or (Wall2Cell[wid][0]-(network.n_walls + network.n_junctions) in Contact): #Wall (not including junctions) connected to soil
                    #        temp=1.0E-04*(network.length[wid]/2*height)/(thickness/2)
                    #        matrix_C[wid][wid] += temp*Diff1 #Removing diffusion BC at soil junction
                    #        rhs_C[wid][0] += temp*Diff1*config.scenarios[count].get("osmotic_left_soil") #Removing BC diffusion term
                    #for wid in listxylwalls:
                    #    temp=1.0E-04*(network.length[wid]*height)/thickness #Section to length ratio (cm) for the xylem wall
                    #    matrix_C[wid][wid] += temp*Diff1 #Removing BC diffusion term
                    #    rhs_C[wid][0] += temp*Diff1*config.scenarios[count].get("osmotic_xyl") #new #Removing BC diffusion term
                    
                
                ####################################
                ## Creates .vtk file for Paraview ##
                ####################################
                
                if config.sym_contagion==1:
                    Sym_Zombies=[]
                    for source in Sym_source_range:
                        Sym_Zombies.append(int(source.get("id")))
                    iZombie=0
                    while not iZombie == size(Sym_Zombies):
                        itemp=0
                        for cid in Cell_connec[int(Sym_Zombies[iZombie])][0:int(nCell_connec[int(Sym_Zombies[iZombie])])]:
                            if Cell_connec_flow[int(Sym_Zombies[iZombie])][itemp] == -1 and (cid not in Sym_Zombies): #Infection
                                if cid in config.sym_immune:
                                    print(cid,': "You shall not pass!"')
                                else:
                                    Sym_Zombies.append(cid)
                                    print(cid,': "Aaargh!"      Zombie count:', size(Sym_Zombies)+1)
                            itemp+=1
                        iZombie+=1
                    print('End of the propagation. Survivor count:', network.n_cells-size(Sym_Zombies)-1)
                    for cid in config.sym_target:
                        if cid in Sym_Zombies:
                            print('Target '+ str(cid) +' down. XXX')
                        else:
                            print('Target '+ str(cid) +' missed!')
                    if config.sym_target[0] in Sym_Zombies:
                        if config.sym_target[1] in Sym_Zombies:
                            Hydropatterning[iMaturity][count]=0 #Both targets reached
                        else:
                            Hydropatterning[iMaturity][count]=1 #Target1 reached only
                    elif config.sym_target[1] in Sym_Zombies:
                        Hydropatterning[iMaturity][count]=2 #Target2 reached only
                    else:
                        Hydropatterning[iMaturity][count]=-1 #Not target reached
                    
                    
                    text_file = open(newpath+"Sym_Contagion_bottomb"+str(Barrier)+","+str(iMaturity)+"s"+str(count)+".pvtk", "w")
                    with open(newpath+"Sym_Contagion_bottomb"+str(Barrier)+","+str(iMaturity)+"s"+str(count)+".pvtk", "a") as myfile:
                        myfile.write("# vtk DataFile Version 4.0 \n")
                        myfile.write("Contaminated symplastic space geometry \n")
                        myfile.write("ASCII \n")
                        myfile.write(" \n")
                        myfile.write("DATASET UNSTRUCTURED_GRID \n")
                        myfile.write("POINTS "+str(len(ThickWalls))+" float \n")
                        for ThickWallNode in ThickWalls:
                            myfile.write(str(ThickWallNode[3]) + " " + str(ThickWallNode[4]) + " " + str(height/200) + " \n")
                        myfile.write(" \n")
                        myfile.write("CELLS " + str(len(Sym_Zombies)) + " " + str(int(len(Sym_Zombies)+sum(nCell2ThickWalls[Sym_Zombies]))) + " \n") #The number of cells corresponds to the number of intercellular spaces
                        Sym_Contagion_order=zeros((network.n_cells,1))
                        temp=0
                        for cid in Sym_Zombies:
                            n=int(nCell2ThickWalls[cid]) #Total number of thick wall nodes around the protoplast
                            Polygon=Cell2ThickWalls[cid][:n]
                            ranking=list()
                            ranking.append(int(Polygon[0]))
                            ranking.append(ThickWalls[int(ranking[0])][5])
                            ranking.append(ThickWalls[int(ranking[0])][6])
                            for id1 in range(1,n):
                                wid1=ThickWalls[int(ranking[id1])][5]
                                wid2=ThickWalls[int(ranking[id1])][6]
                                if wid1 not in ranking:
                                    ranking.append(wid1)
                                if wid2 not in ranking:
                                    ranking.append(wid2)
                            string=str(n)
                            for id1 in ranking:
                                string=string+" "+str(int(id1))
                            myfile.write(string + " \n")
                            Sym_Contagion_order[cid]=temp
                            temp+=1
                        myfile.write(" \n")
                        myfile.write("CELL_TYPES " + str(len(Sym_Zombies)) + " \n")
                        for i in range(len(Sym_Zombies)):
                            myfile.write("6 \n") #Triangle-strip cell type
                        myfile.write(" \n")
                        myfile.write("POINT_DATA " + str(len(ThickWalls)) + " \n")
                        myfile.write("SCALARS Sym_Contagion_order_(#) float \n")
                        myfile.write("LOOKUP_TABLE default \n")
                        for ThickWallNode in ThickWalls:
                            cellnumber1=ThickWallNode[2]-(network.n_walls + network.n_junctions)
                            myfile.write(str(int(Sym_Contagion_order[int(cellnumber1)])) + " \n") #Flow rate from wall (non junction) to cell    min(sath1,max(satl1,  ))
                    myfile.close()
                    text_file.close()
                    
                elif config.sym_contagion==2:
                    text_file = open(newpath+"Sym_Contagion_bottomb"+str(Barrier)+","+str(iMaturity)+"s"+str(count)+".pvtk", "w")
                    with open(newpath+"Sym_Contagion_bottomb"+str(Barrier)+","+str(iMaturity)+"s"+str(count)+".pvtk", "a") as myfile:
                        myfile.write("# vtk DataFile Version 4.0 \n")
                        myfile.write("Symplastic hormone concentration \n")
                        myfile.write("ASCII \n")
                        myfile.write(" \n")
                        myfile.write("DATASET UNSTRUCTURED_GRID \n")
                        myfile.write("POINTS "+str(len(ThickWalls))+" float \n")
                        for ThickWallNode in ThickWalls:
                            myfile.write(str(ThickWallNode[3]) + " " + str(ThickWallNode[4]) + " " + str(height/200) + " \n")
                        myfile.write(" \n")
                        myfile.write("CELLS " + str(network.n_cells) + " " + str(int(network.n_cells+sum(nCell2ThickWalls))) + " \n") #The number of cells corresponds to the number of intercellular spaces
                        for cid in range(network.n_cells):
                            n=int(nCell2ThickWalls[cid]) #Total number of thick wall nodes around the protoplast
                            Polygon=Cell2ThickWalls[cid][:n]
                            ranking=list()
                            ranking.append(int(Polygon[0]))
                            ranking.append(ThickWalls[int(ranking[0])][5])
                            ranking.append(ThickWalls[int(ranking[0])][6])
                            for id1 in range(1,n):
                                wid1=ThickWalls[int(ranking[id1])][5]
                                wid2=ThickWalls[int(ranking[id1])][6]
                                if wid1 not in ranking:
                                    ranking.append(wid1)
                                if wid2 not in ranking:
                                    ranking.append(wid2)
                            string=str(n)
                            for id1 in ranking:
                                string=string+" "+str(int(id1))
                            myfile.write(string + " \n")
                        myfile.write(" \n")
                        myfile.write("CELL_TYPES " + str(network.n_cells) + " \n")
                        for i in range(network.n_cells):
                            myfile.write("6 \n") #Triangle-strip cell type
                        myfile.write(" \n")
                        myfile.write("POINT_DATA " + str(len(ThickWalls)) + " \n")
                        myfile.write("SCALARS Hormone_Symplastic_Relative_Concentration_(-) float \n")
                        myfile.write("LOOKUP_TABLE default \n")
                        if config.apo_contagion==2:
                            for ThickWallNode in ThickWalls:
                                cellnumber1=ThickWallNode[2]-(network.n_walls + network.n_junctions)
                                #print(cellnumber1, soln_C[int(cellnumber1)+(network.n_walls + network.n_junctions)])
                                myfile.write(str(float(soln_C[int(cellnumber1+(network.n_walls + network.n_junctions))])) + " \n")
                        else:
                            for ThickWallNode in ThickWalls:
                                cellnumber1=ThickWallNode[2]-(network.n_walls + network.n_junctions)
                                #print(cellnumber1, soln_SymC[int(cellnumber1)])
                                myfile.write(str(float(soln_SymC[int(cellnumber1)])) + " \n") #
                    myfile.close()
                    text_file.close()
                    
                    #text_file = open(newpath+"Contagion"+str(Barrier)+","+str(iMaturity)+"s"+str(count)+".pvtk", "w")
                    ##sath01=max(soln[(network.n_walls + network.n_junctions):(network.n_walls + network.n_junctions)+network.n_cells-1])
                    ##satl01=min(soln[(network.n_walls + network.n_junctions):(network.n_walls + network.n_junctions)+network.n_cells-1])
                    #with open(newpath+"Contagion"+str(Barrier)+","+str(iMaturity)+"s"+str(count)+".pvtk", "a") as myfile:
                    #    myfile.write("# vtk DataFile Version 4.0 \n")     #("Purchase Amount: %s" % TotalAmount)
                    #    myfile.write("Symplastic hormonal spread by convection \n")
                    #    myfile.write("ASCII \n")
                    #    myfile.write(" \n")
                    #    myfile.write("DATASET UNSTRUCTURED_GRID \n")
                    #    myfile.write("POINTS "+str(len(G.nodes()))+" float \n")
                    #    for node in G:
                    #        myfile.write(str(float(position[node][0])) + " " + str(float(position[node][1])) + " " + str(0.0) + " \n")
                    #    myfile.write(" \n")
                    #    myfile.write("CELLS " + str(network.n_cells) + " " + str(network.n_cells*2) + " \n") #
                    #    for node, edges in G.adjacency():
                    #        i=network.indice[node]
                    #        if i>=(network.n_walls + network.n_junctions): #Cell node
                    #            myfile.write("1 " + str(i) + " \n")
                    #    myfile.write(" \n")
                    #    myfile.write("CELL_TYPES " + str(network.n_cells) + " \n") #
                    #    for node, edges in G.adjacency():
                    #        i=network.indice[node]
                    #        if i>=(network.n_walls + network.n_junctions): #Cell node
                    #            myfile.write("1 \n")
                    #    myfile.write(" \n")
                    #    myfile.write("POINT_DATA " + str(len(G.nodes)) + " \n")
                    #    myfile.write("SCALARS Cell_pressure float \n")
                    #    myfile.write("LOOKUP_TABLE default \n")
                    #    for node in G:
                    #        if node-(network.n_walls + network.n_junctions) in [Zombie0]: #Source cell
                    #            myfile.write(str(float(0.0)) + " \n")
                    #        elif node-(network.n_walls + network.n_junctions) in Zombies:
                    #            myfile.write(str(float(1.0)) + " \n")
                    #        else:
                    #            myfile.write(str(float(-1.0)) + " \n")
                    #myfile.close()
                    #text_file.close()
                
                if config.apo_contagion==1:
                    Apo_w_Zombies=Apo_w_Zombies0
                    iZombie=0
                    while not iZombie == size(Apo_w_Zombies):
                        id1=Apo_w_Zombies[iZombie]
                        for id2 in Apo_connec_flow[id1][0:nApo_connec_flow[id1]]:
                            if id2 not in Apo_w_Zombies: #Infection
                                if id2 in Apo_w_Immune:
                                    print(id2,': "You shall not pass!"')
                                else:
                                    Apo_w_Zombies.append(id2)
                                    print(id2,': "Aaargh!"      Zombie count:', size(Apo_w_Zombies))
                        iZombie+=1
                    print('End of the propagation. Survivor count:', (network.n_walls + network.n_junctions)-size(Apo_w_Zombies))
                    temp=0
                    for wid in Apo_w_Target:
                        if wid in Apo_w_Zombies:
                            temp+=1
                            print('Target '+ str(wid) +' down. XXX')
                        else:
                            print('Target '+ str(wid) +' missed!')
                    Hydrotropism[iMaturity][count]=float(temp)/size(Apo_w_Target) #0: No apoplastic target reached; 1: All apoplastic targets reached
                    
                    
                    text_file = open(newpath+"Apo_Contagion_bottomb"+str(Barrier)+","+str(iMaturity)+"s"+str(count)+".pvtk", "w")
                    with open(newpath+"Apo_Contagion_bottomb"+str(Barrier)+","+str(iMaturity)+"s"+str(count)+".pvtk", "a") as myfile:
                        myfile.write("# vtk DataFile Version 4.0 \n")
                        myfile.write("Contaminated Apoplastic space geometry \n")
                        myfile.write("ASCII \n")
                        myfile.write(" \n")
                        myfile.write("DATASET UNSTRUCTURED_GRID \n")
                        myfile.write("POINTS "+str(len(ThickWallsX))+" float \n")
                        for ThickWallNodeX in ThickWallsX:
                            myfile.write(str(ThickWallNodeX[1]) + " " + str(ThickWallNodeX[2]) + " 0.0 \n")
                        myfile.write(" \n")
                        myfile.write("CELLS " + str(int((network.n_walls + network.n_junctions)+network.n_walls-len(list_ghostwalls)*2-len(list_ghostjunctions))) + " " + str(int(2*network.n_walls*5-len(list_ghostwalls)*10+sum(nWall2NewWallX[network.n_walls:])+(network.n_walls + network.n_junctions)-network.n_walls+2*len(Wall2NewWallX[network.n_walls:])-nGhostJunction2Wall-len(list_ghostjunctions))) + " \n") #The number of cells corresponds to the number of lines in ThickWalls (if no ghost wall & junction)
                        i=0
                        for PolygonX in ThickWallPolygonX:
                            if floor(i/2) not in list_ghostwalls:
                                myfile.write("4 " + str(int(PolygonX[0])) + " " + str(int(PolygonX[1])) + " " + str(int(PolygonX[2])) + " " + str(int(PolygonX[3])) + " \n")
                            i+=1
                        j=network.n_walls
                        for PolygonX in Wall2NewWallX[network.n_walls:]: #"junction" polygons
                            #Would need to order them based on x or y position to make sure display fully covers the surface (but here we try a simpler not so good solution instead)
                            if j not in list_ghostjunctions:
                                string=str(int(nWall2NewWallX[j]+2)) #Added +2 so that the first and second nodes could be added again at the end (trying to fill the polygon better)
                                for id1 in range(int(nWall2NewWallX[j])):
                                    string=string+" "+str(int(PolygonX[id1]))
                                string=string+" "+str(int(PolygonX[0]))+" "+str(int(PolygonX[1])) #Adding the 1st and 2nd nodes again to the end
                                myfile.write(string + " \n")
                            j+=1
                        myfile.write(" \n")
                        myfile.write("CELL_TYPES " + str((network.n_walls + network.n_junctions)+network.n_walls-len(list_ghostwalls)*2-len(list_ghostjunctions)) + " \n")
                        i=0
                        for PolygonX in ThickWallPolygonX:
                            if floor(i/2) not in list_ghostwalls:
                                myfile.write("7 \n") #Polygon cell type (wall)
                            i+=1
                        j=network.n_walls
                        for PolygonX in Wall2NewWallX[network.n_walls:]:
                            if j not in list_ghostjunctions:
                                myfile.write("6 \n") #Triangle-strip cell type (wall junction)
                            j+=1
                        myfile.write(" \n")
                        myfile.write("POINT_DATA " + str(len(ThickWallsX)) + " \n")
                        myfile.write("SCALARS Apo_Contagion_order_(#) float \n")
                        myfile.write("LOOKUP_TABLE default \n")
                        Apo_Contagion_order=zeros(((network.n_walls + network.n_junctions),1))+int(len(Apo_w_Zombies)*1.6)
                        temp=0
                        for wid in Apo_w_Zombies:
                            Apo_Contagion_order[wid]=temp
                            temp+=1
                        NewApo_Contagion_order=zeros((len(ThickWallsX),1))
                        j=0
                        for PolygonX in Wall2NewWallX:
                            for id1 in range(int(nWall2NewWallX[j])):
                                NewApo_Contagion_order[int(PolygonX[id1])]=Apo_Contagion_order[j]
                            j+=1
                        for i in range(len(ThickWallsX)):
                            myfile.write(str(float(NewApo_Contagion_order[i])) + " \n")
                    myfile.close()
                    text_file.close()
                    
                elif config.apo_contagion==2:
                    text_file = open(newpath+"Apo_Contagion_bottomb"+str(Barrier)+","+str(iMaturity)+"s"+str(count)+".pvtk", "w")
                    with open(newpath+"Apo_Contagion_bottomb"+str(Barrier)+","+str(iMaturity)+"s"+str(count)+".pvtk", "a") as myfile:
                        myfile.write("# vtk DataFile Version 4.0 \n")
                        myfile.write("Apoplastic hormone concentration \n")
                        myfile.write("ASCII \n")
                        myfile.write(" \n")
                        myfile.write("DATASET UNSTRUCTURED_GRID \n")
                        myfile.write("POINTS "+str(len(ThickWallsX))+" float \n")
                        for ThickWallNodeX in ThickWallsX:
                            myfile.write(str(ThickWallNodeX[1]) + " " + str(ThickWallNodeX[2]) + " 0.0 \n")
                        myfile.write(" \n")
                        myfile.write("CELLS " + str(int((network.n_walls + network.n_junctions)+network.n_walls-len(list_ghostwalls)*2-len(list_ghostjunctions))) + " " + str(int(2*network.n_walls*5-len(list_ghostwalls)*10+sum(nWall2NewWallX[network.n_walls:])+(network.n_walls + network.n_junctions)-network.n_walls+2*len(Wall2NewWallX[network.n_walls:])-nGhostJunction2Wall-len(list_ghostjunctions))) + " \n") #The number of cells corresponds to the number of lines in ThickWalls (if no ghost wall & junction)
                        i=0
                        for PolygonX in ThickWallPolygonX:
                            if floor(i/2) not in list_ghostwalls:
                                myfile.write("4 " + str(int(PolygonX[0])) + " " + str(int(PolygonX[1])) + " " + str(int(PolygonX[2])) + " " + str(int(PolygonX[3])) + " \n")
                            i+=1
                        j=network.n_walls
                        for PolygonX in Wall2NewWallX[network.n_walls:]: #"junction" polygons
                            #Would need to order them based on x or y position to make sure display fully covers the surface (but here we try a simpler not so good solution instead)
                            if j not in list_ghostjunctions:
                                string=str(int(nWall2NewWallX[j]+2)) #Added +2 so that the first and second nodes could be added again at the end (trying to fill the polygon better)
                                for id1 in range(int(nWall2NewWallX[j])):
                                    string=string+" "+str(int(PolygonX[id1]))
                                string=string+" "+str(int(PolygonX[0]))+" "+str(int(PolygonX[1])) #Adding the 1st and 2nd nodes again to the end
                                myfile.write(string + " \n")
                            j+=1
                        myfile.write(" \n")
                        myfile.write("CELL_TYPES " + str((network.n_walls + network.n_junctions)+network.n_walls-len(list_ghostwalls)*2-len(list_ghostjunctions)) + " \n")
                        i=0
                        for PolygonX in ThickWallPolygonX:
                            if floor(i/2) not in list_ghostwalls:
                                myfile.write("7 \n") #Polygon cell type (wall)
                            i+=1
                        j=network.n_walls
                        for PolygonX in Wall2NewWallX[network.n_walls:]:
                            if j not in list_ghostjunctions:
                                myfile.write("6 \n") #Triangle-strip cell type (wall junction)
                            j+=1
                        myfile.write(" \n")
                        myfile.write("POINT_DATA " + str(len(ThickWallsX)) + " \n")
                        myfile.write("SCALARS Hormone_Symplastic_Relative_Concentration_(-) float \n")
                        myfile.write("LOOKUP_TABLE default \n")
                        if config.sym_contagion==2:
                            Newsoln_C=zeros((len(ThickWallsX),1))
                            j=0
                            for PolygonX in Wall2NewWallX:
                                for id1 in range(int(nWall2NewWallX[j])):
                                    Newsoln_C[int(PolygonX[id1])]=soln_C[j]
                                j+=1
                            for i in range(len(ThickWallsX)):
                                myfile.write(str(float(Newsoln_C[i])) + " \n")
                        else:
                            Newsoln_ApoC=zeros((len(ThickWallsX),1))
                            j=0
                            for PolygonX in Wall2NewWallX:
                                for id1 in range(int(nWall2NewWallX[j])):
                                    Newsoln_ApoC[int(PolygonX[id1])]=soln_ApoC[j]
                                j+=1
                            for i in range(len(ThickWallsX)):
                                myfile.write(str(float(Newsoln_ApoC[i])) + " \n")
                    myfile.close()
                    text_file.close()
                
                
                if config.paraview==1:
                    if config.paraview_wp==1: #2D visualization of walls pressure potentials
                        text_file = open(newpath+"Walls2Db"+str(Barrier)+","+str(iMaturity)+"s"+str(count)+".pvtk", "w")
                        #sath0=max(soln[0:(network.n_walls + network.n_junctions)-1])
                        #satl0=min(soln[0:(network.n_walls + network.n_junctions)-1])
                        with open(newpath+"Walls2Db"+str(Barrier)+","+str(iMaturity)+"s"+str(count)+".pvtk", "a") as myfile:
                            myfile.write("# vtk DataFile Version 4.0 \n")     #("Purchase Amount: %s" % TotalAmount)
                            myfile.write("Wall geometry 2D \n")
                            myfile.write("ASCII \n")
                            myfile.write(" \n")
                            myfile.write("DATASET UNSTRUCTURED_GRID \n")
                            myfile.write("POINTS "+str(len(G.nodes()))+" float \n")
                            for node in G:
                                myfile.write(str(float(network.position[node][0])) + " " + str(float(network.position[node][1])) + " " + str(0.0) + " \n")
                            myfile.write(" \n")
                            myfile.write("CELLS " + str(network.n_walls*2-len(list_ghostwalls)*2) + " " + str(network.n_walls*6-len(list_ghostwalls)*6) + " \n") #len(G.nodes())
                            for node, edges in G.adjacency():
                                i=network.indice[node]
                                if i not in list_ghostwalls:
                                    for neighboor, eattr in edges.items(): #Loop on connections (edges)
                                        j=network.indice[neighboor]
                                        if j>i and eattr['path']=='wall':
                                            #print(nx.get_node_attributes(edges,'path'))
                                            myfile.write(str(2) + " " + str(i) + " " + str(j) + " \n")
                            myfile.write(" \n")
                            myfile.write("CELL_TYPES " + str(network.n_walls*2-len(list_ghostwalls)*2) + " \n") #The number of nodes corresponds to the number of wall to wall connections.... to be checked, might not be generality
                            for node, edges in G.adjacency():
                                i=network.indice[node]
                                if i not in list_ghostwalls:
                                    for neighboor, eattr in edges.items(): #Loop on connections (edges)
                                        j=network.indice[neighboor]
                                        if j>i and eattr['path']=='wall':
                                            #print(nx.get_node_attributes(edges,'path'))
                                            myfile.write(str(3) + " \n") #Line cell type
                            myfile.write(" \n")
                            myfile.write("POINT_DATA " + str(len(G.nodes())) + " \n")
                            myfile.write("SCALARS Wall_pressure float \n")
                            myfile.write("LOOKUP_TABLE default \n")
                            for node in G:
                                myfile.write(str(float(soln[node])) + " \n") #Line cell type      min(sath0,max(satl0,   ))
                        myfile.close()
                        text_file.close()
                    
                    if config.paraview_wp==1 and config.paraview_cp: #2D visualization of walls & cells osmotic potentials
                        text_file = open(newpath+"WallsOsAndCellsOs2Db"+str(Barrier)+","+str(iMaturity)+"s"+str(count)+".pvtk", "w")
                        with open(newpath+"WallsOsAndCellsOs2Db"+str(Barrier)+","+str(iMaturity)+"s"+str(count)+".pvtk", "a") as myfile:
                            myfile.write("# vtk DataFile Version 4.0 \n")     #("Purchase Amount: %s" % TotalAmount)
                            myfile.write("Wall geometry 2D \n")
                            myfile.write("ASCII \n")
                            myfile.write(" \n")
                            myfile.write("DATASET UNSTRUCTURED_GRID \n")
                            myfile.write("POINTS "+str(len(G.nodes()))+" float \n")
                            for node in G:
                                myfile.write(str(float(network.position[node][0])) + " " + str(float(network.position[node][1])) + " " + str(0.0) + " \n")
                            myfile.write(" \n")                                     
                            myfile.write("CELLS " + str(network.n_walls*2-len(list_ghostwalls)*2+network.n_cells) + " " + str(network.n_walls*6-len(list_ghostwalls)*6+network.n_cells*2) + " \n") #
                            for node, edges in G.adjacency():
                                i=network.indice[node]
                                if i not in list_ghostwalls:
                                    for neighboor, eattr in edges.items(): #Loop on connections (edges)
                                        j=network.indice[neighboor]
                                        if j>i and eattr['path']=='wall':
                                            #print(nx.get_node_attributes(edges,'path'))
                                            myfile.write(str(2) + " " + str(i) + " " + str(j) + " \n")
                                if i>=(network.n_walls + network.n_junctions): #Cell node
                                    myfile.write("1 " + str(i) + " \n")
                            myfile.write(" \n")
                            myfile.write("CELL_TYPES " + str(network.n_walls*2-len(list_ghostwalls)*2+network.n_cells) + " \n") #
                            for node, edges in G.adjacency():
                                i=network.indice[node]
                                if i not in list_ghostwalls:
                                    for neighboor, eattr in edges.items(): #Loop on connections (edges)
                                        j=network.indice[neighboor]
                                        if j>i and eattr['path']=='wall':
                                            #print(nx.get_node_attributes(edges,'path'))
                                            myfile.write(str(3) + " \n") #Line cell type
                                if i>=(network.n_walls + network.n_junctions): #Cell node
                                    myfile.write("1 \n")
                            myfile.write(" \n")
                            myfile.write("POINT_DATA " + str(len(G.nodes())) + " \n")
                            myfile.write("SCALARS Wall_and_Cell_osmotic_pot float \n")
                            myfile.write("LOOKUP_TABLE default \n")
                            for node, edges in G.adjacency():
                                i=network.indice[node] #Node ID number
                                if i<network.n_walls: #Wall node
                                    myfile.write(str(float(Os_walls[i])) + " \n")
                                elif i<(network.n_walls + network.n_junctions): #Junction node
                                    myfile.write(str(float(0.0)) + " \n")
                                else: #Cell node
                                    myfile.write(str(float(Os_cells[i-(network.n_walls + network.n_junctions)])) + " \n")
                        myfile.close()
                        text_file.close()
                        
    
                    
                    if config.paraview_wp==1 and config.paraview_cp==1: #2D visualization of walls & cells water potentials
                        text_file = open(newpath+"WallsAndCells2Db"+str(Barrier)+","+str(iMaturity)+"s"+str(count)+".pvtk", "w")
                        with open(newpath+"WallsAndCells2Db"+str(Barrier)+","+str(iMaturity)+"s"+str(count)+".pvtk", "a") as myfile:
                            myfile.write("# vtk DataFile Version 4.0 \n")     #("Purchase Amount: %s" % TotalAmount)
                            myfile.write("Water potential distribution in cells and walls 2D \n")
                            myfile.write("ASCII \n")
                            myfile.write(" \n")
                            myfile.write("DATASET UNSTRUCTURED_GRID \n")
                            myfile.write("POINTS "+str(len(G.nodes()))+" float \n")
                            for node in G:
                                myfile.write(str(float(network.position[node][0])) + " " + str(float(network.position[node][1])) + " " + str(0.0) + " \n")
                            myfile.write(" \n")
                            myfile.write("CELLS " + str(network.n_walls*2-len(list_ghostwalls)*2+network.n_cells) + " " + str(network.n_walls*6-len(list_ghostwalls)*6+network.n_cells*2) + " \n") #
                            for node, edges in G.adjacency():
                                i=network.indice[node]
                                if i not in list_ghostwalls:
                                    for neighboor, eattr in edges.items(): #Loop on connections (edges)
                                        j=network.indice[neighboor]
                                        if j>i and eattr['path']=='wall':
                                            #print(nx.get_node_attributes(edges,'path'))
                                            myfile.write(str(2) + " " + str(i) + " " + str(j) + " \n")
                                if i>=(network.n_walls + network.n_junctions): #Cell node
                                    myfile.write("1 " + str(i) + " \n")
                            myfile.write(" \n")
                            myfile.write("CELL_TYPES " + str(network.n_walls*2-len(list_ghostwalls)*2+network.n_cells) + " \n") #
                            for node, edges in G.adjacency():
                                i=network.indice[node]
                                if i not in list_ghostwalls:
                                    for neighboor, eattr in edges.items(): #Loop on connections (edges)
                                        j=network.indice[neighboor]
                                        if j>i and eattr['path']=='wall':
                                            #print(nx.get_node_attributes(edges,'path'))
                                            myfile.write(str(3) + " \n") #Line cell type
                                if i>=(network.n_walls + network.n_junctions): #Cell node
                                    myfile.write("1 \n")
                            myfile.write(" \n")
                            myfile.write("POINT_DATA " + str(len(G.nodes())) + " \n")
                            myfile.write("SCALARS pressure float \n")
                            myfile.write("LOOKUP_TABLE default \n")
                            for node in G:
                                myfile.write(str(float(soln[node])) + " \n") #Line cell type
                        myfile.close()
                        text_file.close()
                    
                    if config.paraview_cp==1: #2D visualization of cells water potentials
                        text_file = open(newpath+"Cells2Db"+str(Barrier)+","+str(iMaturity)+"s"+str(count)+".pvtk", "w")
                        #sath01=max(soln[(network.n_walls + network.n_junctions):(network.n_walls + network.n_junctions)+network.n_cells-1])
                        #satl01=min(soln[(network.n_walls + network.n_junctions):(network.n_walls + network.n_junctions)+network.n_cells-1])
                        with open(newpath+"Cells2Db"+str(Barrier)+","+str(iMaturity)+"s"+str(count)+".pvtk", "a") as myfile:
                            myfile.write("# vtk DataFile Version 4.0 \n")     #("Purchase Amount: %s" % TotalAmount)
                            myfile.write("Pressure potential distribution in cells 2D \n")
                            myfile.write("ASCII \n")
                            myfile.write(" \n")
                            myfile.write("DATASET UNSTRUCTURED_GRID \n")
                            myfile.write("POINTS "+str(len(G.nodes()))+" float \n")
                            for node in G:
                                myfile.write(str(float(network.position[node][0])) + " " + str(float(network.position[node][1])) + " " + str(0.0) + " \n")
                            myfile.write(" \n")
                            myfile.write("CELLS " + str(network.n_cells) + " " + str(network.n_cells*2) + " \n") #
                            for node, edges in G.adjacency():
                                i=network.indice[node]
                                if i>=(network.n_walls + network.n_junctions): #Cell node
                                    myfile.write("1 " + str(i) + " \n")
                            myfile.write(" \n")
                            myfile.write("CELL_TYPES " + str(network.n_cells) + " \n") #
                            for node, edges in G.adjacency():
                                i=network.indice[node]
                                if i>=(network.n_walls + network.n_junctions): #Cell node
                                    myfile.write("1 \n")
                            myfile.write(" \n")
                            myfile.write("POINT_DATA " + str(len(G.nodes())) + " \n")
                            myfile.write("SCALARS Cell_pressure float \n")
                            myfile.write("LOOKUP_TABLE default \n")
                            for node in G:
                                myfile.write(str(float(soln[node])) + " \n") #Line cell type      min(sath01,max(satl01,   ))
                        myfile.close()
                        text_file.close()
                        
                    
                    if config.paraview_mf==1: #3D visualization of membrane fluxes
                        text_file = open(newpath+"Membranes3Db"+str(Barrier)+","+str(iMaturity)+"s"+str(count)+".pvtk", "w")
                        #sath1=max(MembraneFlowDensity)*config.color_threshold
                        #satl1=min(MembraneFlowDensity)*config.color_threshold
                        #if satl1<-sath1: #min(MembraneFlowDensity)<0:
                        #    sath1=-satl1
                        #else:
                        #    satl1=-sath1
                        with open(newpath+"Membranes3Db"+str(Barrier)+","+str(iMaturity)+"s"+str(count)+".pvtk", "a") as myfile:
                            myfile.write("# vtk DataFile Version 4.0 \n")
                            myfile.write("Membranes geometry 3D \n")
                            myfile.write("ASCII \n")
                            myfile.write(" \n")
                            myfile.write("DATASET UNSTRUCTURED_GRID \n")
                            myfile.write("POINTS "+str(len(ThickWalls)*2)+" float \n")
                            for ThickWallNode in ThickWalls:
                                myfile.write(str(ThickWallNode[3]) + " " + str(ThickWallNode[4]) + " 0.0 \n")
                            for ThickWallNode in ThickWalls:
                                myfile.write(str(ThickWallNode[3]) + " " + str(ThickWallNode[4]) + " " + str(height) + " \n")
                            myfile.write(" \n")
                            myfile.write("CELLS " + str(len(ThickWalls)-len(list_ghostwalls)*4) + " " + str(len(ThickWalls)*5-len(list_ghostwalls)*20) + " \n") #The number of cells corresponds to the number of lines in ThickWalls
                            for ThickWallNode in ThickWalls:
                                if ThickWallNode[1]>=network.n_walls: #wall that is a junction
                                    if ThickWalls[int(ThickWallNode[5])][1] not in list_ghostwalls:
                                        myfile.write("4 " + str(int(ThickWallNode[0])) + " " + str(int(ThickWallNode[5])) + " " + str(int(ThickWallNode[5])+len(ThickWalls)) + " " + str(int(ThickWallNode[0])+len(ThickWalls)) + " \n") #All points were repeated twice (once at z=0 and once at z=height), so adding len(ThickWalls) is the same point at z=height
                                    if ThickWalls[int(ThickWallNode[6])][1] not in list_ghostwalls:
                                        myfile.write("4 " + str(int(ThickWallNode[0])) + " " + str(int(ThickWallNode[6])) + " " + str(int(ThickWallNode[6])+len(ThickWalls)) + " " + str(int(ThickWallNode[0])+len(ThickWalls)) + " \n")
                            myfile.write(" \n")
                            myfile.write("CELL_TYPES " + str(len(ThickWalls)-len(list_ghostwalls)*4) + " \n")
                            for ThickWallNode in ThickWalls:
                                if ThickWallNode[1]>=network.n_walls: #wall that is a junction
                                    if ThickWalls[int(ThickWallNode[5])][1] not in list_ghostwalls:
                                        myfile.write("9 \n") #Quad cell type
                                    if ThickWalls[int(ThickWallNode[6])][1] not in list_ghostwalls:
                                        myfile.write("9 \n") #Quad cell type
                            myfile.write(" \n")
                            myfile.write("POINT_DATA " + str(len(ThickWalls)*2) + " \n")
                            myfile.write("SCALARS TM_flux_(m/s) float \n")
                            myfile.write("LOOKUP_TABLE default \n")
                            for ThickWallNode in ThickWalls:
                                if ThickWallNode[0]<len(MembraneFlowDensity):
                                    myfile.write(str(float(MembraneFlowDensity[int(ThickWallNode[0])])/SECONDS_PER_DAY/CM_PER_METER) + " \n") #Flow rate from wall (non junction) to cell   min(sath1,max(satl1,  ))
                                else:
                                    myfile.write(str(float((MembraneFlowDensity[int(ThickWallNode[5])]+MembraneFlowDensity[int(ThickWallNode[6])])/2)/SECONDS_PER_DAY/CM_PER_METER) + " \n") #Flow rate from junction wall to cell is the average of the 2 neighbouring wall flow rates   min(sath1,max(satl1,  ))
                            for ThickWallNode in ThickWalls:
                                if ThickWallNode[0]<len(MembraneFlowDensity):
                                    myfile.write(str(float(MembraneFlowDensity[int(ThickWallNode[0])])/SECONDS_PER_DAY/CM_PER_METER) + " \n") #Flow rate from wall (non junction) to cell   min(sath1,max(satl1,  ))
                                else:
                                    myfile.write(str(float((MembraneFlowDensity[int(ThickWallNode[5])]+MembraneFlowDensity[int(ThickWallNode[6])])/2)/SECONDS_PER_DAY/CM_PER_METER) + " \n") #Flow rate from junction wall to cell is the average of the 2 neighbouring wall flow rates   min(sath1,max(satl1,  ))
                        myfile.close()
                        text_file.close()
                    
                    if config.paraview_wf==1: #Wall flow density data
                        maxWallFlowDensity=0.0
                        for ir in range(int(len(WallFlowDensity))):
                            maxWallFlowDensity=max(maxWallFlowDensity,abs(WallFlowDensity[ir][2]))
                        sath2=maxWallFlowDensity*config.color_threshold #(1-(1-config.color_threshold)/2)
                        #satl2=0.0
                        text_file = open(newpath+"WallsThick3D_bottomb"+str(Barrier)+","+str(iMaturity)+"s"+str(count)+".pvtk", "w")
                        with open(newpath+"WallsThick3D_bottomb"+str(Barrier)+","+str(iMaturity)+"s"+str(count)+".pvtk", "a") as myfile:
                            myfile.write("# vtk DataFile Version 4.0 \n")
                            myfile.write("Wall geometry 3D including thickness bottom \n")
                            myfile.write("ASCII \n")
                            myfile.write(" \n")
                            myfile.write("DATASET UNSTRUCTURED_GRID \n")
                            myfile.write("POINTS "+str(len(ThickWallsX))+" float \n")
                            for ThickWallNodeX in ThickWallsX:
                                myfile.write(str(ThickWallNodeX[1]) + " " + str(ThickWallNodeX[2]) + " 0.0 \n")
                            myfile.write(" \n")
                            myfile.write("CELLS " + str(int((network.n_walls + network.n_junctions)+network.n_walls-len(list_ghostwalls)*2-len(list_ghostjunctions))) + " " + str(int(2*network.n_walls*5-len(list_ghostwalls)*10+sum(nWall2NewWallX[network.n_walls:])+(network.n_walls + network.n_junctions)-network.n_walls+2*len(Wall2NewWallX[network.n_walls:])-nGhostJunction2Wall-len(list_ghostjunctions))) + " \n") #The number of cells corresponds to the number of lines in ThickWalls (if no ghost wall & junction)
                            i=0
                            for PolygonX in ThickWallPolygonX:
                                if floor(i/2) not in list_ghostwalls:
                                    myfile.write("4 " + str(int(PolygonX[0])) + " " + str(int(PolygonX[1])) + " " + str(int(PolygonX[2])) + " " + str(int(PolygonX[3])) + " \n")
                                i+=1
                            j=network.n_walls
                            for PolygonX in Wall2NewWallX[network.n_walls:]: #"junction" polygons
                                #Would need to order them based on x or y position to make sure display fully covers the surface (but here we try a simpler not so good solution instead)
                                if j not in list_ghostjunctions:
                                    string=str(int(nWall2NewWallX[j]+2)) #Added +2 so that the first and second nodes could be added again at the end (trying to fill the polygon better)
                                    for id1 in range(int(nWall2NewWallX[j])):
                                        string=string+" "+str(int(PolygonX[id1]))
                                    string=string+" "+str(int(PolygonX[0]))+" "+str(int(PolygonX[1])) #Adding the 1st and 2nd nodes again to the end
                                    myfile.write(string + " \n")
                                j+=1
                            myfile.write(" \n")
                            myfile.write("CELL_TYPES " + str((network.n_walls + network.n_junctions)+network.n_walls-len(list_ghostwalls)*2-len(list_ghostjunctions)) + " \n")
                            i=0
                            for PolygonX in ThickWallPolygonX:
                                if floor(i/2) not in list_ghostwalls:
                                    myfile.write("7 \n") #Polygon cell type (wall)
                                i+=1
                            j=network.n_walls
                            for PolygonX in Wall2NewWallX[network.n_walls:]:
                                if j not in list_ghostjunctions:
                                    myfile.write("6 \n") #Triangle-strip cell type (wall junction)
                                j+=1
                            myfile.write(" \n")
                            myfile.write("POINT_DATA " + str(len(ThickWallsX)) + " \n")
                            myfile.write("SCALARS Apo_flux_(m/s) float \n")
                            myfile.write("LOOKUP_TABLE default \n")
                            NewWallFlowDensity=zeros((len(ThickWallsX),2))
                            i=0
                            for PolygonX in ThickWallPolygonX:
                                for id1 in range(4):
                                    if abs(float(WallFlowDensity[i][2]))>min(NewWallFlowDensity[int(PolygonX[id1])]):
                                        NewWallFlowDensity[int(PolygonX[id1])][0]=max(NewWallFlowDensity[int(PolygonX[id1])])
                                        NewWallFlowDensity[int(PolygonX[id1])][1]=abs(float(WallFlowDensity[i][2]))
                                i+=1
                            for i in range(len(ThickWallsX)):
                                myfile.write(str(float(mean(NewWallFlowDensity[i]))/SECONDS_PER_DAY/CM_PER_METER) + " \n")  # min(sath2,  )
                        myfile.close()
                        text_file.close()
                        
                        text_file = open(newpath+"WallsThick3Dcos_bottomb"+str(Barrier)+","+str(iMaturity)+"s"+str(count)+".pvtk", "w")
                        with open(newpath+"WallsThick3Dcos_bottomb"+str(Barrier)+","+str(iMaturity)+"s"+str(count)+".pvtk", "a") as myfile:
                            myfile.write("# vtk DataFile Version 4.0 \n")
                            myfile.write("Wall geometry 3D including thickness bottom \n")
                            myfile.write("ASCII \n")
                            myfile.write(" \n")
                            myfile.write("DATASET UNSTRUCTURED_GRID \n")
                            myfile.write("POINTS "+str(len(ThickWallsX))+" float \n")
                            for ThickWallNodeX in ThickWallsX:
                                myfile.write(str(ThickWallNodeX[1]) + " " + str(ThickWallNodeX[2]) + " 0.0 \n")
                            myfile.write(" \n")
                            myfile.write("CELLS " + str(int((network.n_walls + network.n_junctions)+network.n_walls-len(list_ghostwalls)*2-len(list_ghostjunctions))) + " " + str(int(2*network.n_walls*5-len(list_ghostwalls)*10+sum(nWall2NewWallX[network.n_walls:])+(network.n_walls + network.n_junctions)-network.n_walls+2*len(Wall2NewWallX[network.n_walls:])-nGhostJunction2Wall-len(list_ghostjunctions))) + " \n") #The number of cells corresponds to the number of lines in ThickWalls (if no ghost wall & junction)
                            i=0
                            for PolygonX in ThickWallPolygonX:
                                if floor(i/2) not in list_ghostwalls:
                                    myfile.write("4 " + str(int(PolygonX[0])) + " " + str(int(PolygonX[1])) + " " + str(int(PolygonX[2])) + " " + str(int(PolygonX[3])) + " \n")
                                i+=1
                            j=network.n_walls
                            for PolygonX in Wall2NewWallX[network.n_walls:]: #"junction" polygons
                                #Would need to order them based on x or y position to make sure display fully covers the surface (but here we try a simpler not so good solution instead)
                                if j not in list_ghostjunctions:
                                    string=str(int(nWall2NewWallX[j]+2)) #Added +2 so that the first and second nodes could be added again at the end (trying to fill the polygon better)
                                    for id1 in range(int(nWall2NewWallX[j])):
                                        string=string+" "+str(int(PolygonX[id1]))
                                    string=string+" "+str(int(PolygonX[0]))+" "+str(int(PolygonX[1])) #Adding the 1st and 2nd nodes again to the end
                                    myfile.write(string + " \n")
                                j+=1
                            myfile.write(" \n")
                            myfile.write("CELL_TYPES " + str((network.n_walls + network.n_junctions)+network.n_walls-len(list_ghostwalls)*2-len(list_ghostjunctions)) + " \n")
                            i=0
                            for PolygonX in ThickWallPolygonX:
                                if floor(i/2) not in list_ghostwalls:
                                    myfile.write("7 \n") #Polygon cell type (wall)
                                i+=1
                            j=network.n_walls
                            for PolygonX in Wall2NewWallX[network.n_walls:]:
                                if j not in list_ghostjunctions:
                                    myfile.write("6 \n") #Triangle-strip cell type (wall junction)
                                j+=1
                            myfile.write(" \n")
                            myfile.write("POINT_DATA " + str(len(ThickWallsX)) + " \n")
                            myfile.write("SCALARS Apo_flux_cosine_(m/s) float \n")
                            myfile.write("LOOKUP_TABLE default \n")
                            NewWallFlowDensity_cos=zeros((len(ThickWallsX),2))
                            i=0
                            for PolygonX in ThickWallPolygonX:
                                for id1 in range(4):
                                    if abs(float(WallFlowDensity_cos[i][2]))>min(abs(NewWallFlowDensity_cos[int(PolygonX[id1])])):
                                        #Horizontal component of the flux
                                        if abs(NewWallFlowDensity_cos[int(PolygonX[id1])][1])>abs(NewWallFlowDensity_cos[int(PolygonX[id1])][0]): #Keeping the most extreme value
                                            NewWallFlowDensity_cos[int(PolygonX[id1])][0]=NewWallFlowDensity_cos[int(PolygonX[id1])][1]
                                        NewWallFlowDensity_cos[int(PolygonX[id1])][1]=float(WallFlowDensity_cos[i][2])
                                i+=1
                            for i in range(len(ThickWallsX)):
                                myfile.write(str(float(mean(NewWallFlowDensity_cos[i]))/SECONDS_PER_DAY/CM_PER_METER) + " \n")  # min(sath2,  )
                        myfile.close()
                        text_file.close()
                    
                        if Barrier>0:
                            text_file = open(newpath+"InterC3D_bottomb"+str(Barrier)+","+str(iMaturity)+"s"+str(count)+".pvtk", "w")
                            with open(newpath+"InterC3D_bottomb"+str(Barrier)+","+str(iMaturity)+"s"+str(count)+".pvtk", "a") as myfile:
                                myfile.write("# vtk DataFile Version 4.0 \n")
                                myfile.write("Intercellular space geometry 3D \n")
                                myfile.write("ASCII \n")
                                myfile.write(" \n")
                                myfile.write("DATASET UNSTRUCTURED_GRID \n")
                                myfile.write("POINTS "+str(len(ThickWalls))+" float \n")
                                for ThickWallNode in ThickWalls:
                                    myfile.write(str(ThickWallNode[3]) + " " + str(ThickWallNode[4]) + " " + str(height/200) + " \n")
                                myfile.write(" \n")
                                myfile.write("CELLS " + str(len(config.intercellular_ids)) + " " + str(int(len(config.intercellular_ids)+sum(nCell2ThickWalls[config.intercellular_ids]))) + " \n") #The number of cells corresponds to the number of intercellular spaces
                                InterCFlowDensity=zeros((network.n_cells,1))
                                for cid in config.intercellular_ids:
                                    n=int(nCell2ThickWalls[cid]) #Total number of thick wall nodes around the protoplast
                                    Polygon=Cell2ThickWalls[cid][:n]
                                    ranking=list()
                                    ranking.append(int(Polygon[0]))
                                    ranking.append(ThickWalls[int(ranking[0])][5])
                                    ranking.append(ThickWalls[int(ranking[0])][6])
                                    for id1 in range(1,n):
                                        wid1=ThickWalls[int(ranking[id1])][5]
                                        wid2=ThickWalls[int(ranking[id1])][6]
                                        if wid1 not in ranking:
                                            ranking.append(wid1)
                                        if wid2 not in ranking:
                                            ranking.append(wid2)
                                    string=str(n)
                                    for id1 in ranking:
                                        string=string+" "+str(int(id1))
                                    myfile.write(string + " \n")
                                    for twpid in Polygon[:int(n/2)]: #The first half of nodes are wall nodes actually connected to cells
                                        InterCFlowDensity[cid]+=abs(MembraneFlowDensity[int(twpid)])/n #Mean absolute flow density calculation
                                myfile.write(" \n")
                                myfile.write("CELL_TYPES " + str(len(config.intercellular_ids)) + " \n")
                                for i in range(len(config.intercellular_ids)):
                                    myfile.write("6 \n") #Triangle-strip cell type
                                myfile.write(" \n")
                                myfile.write("POINT_DATA " + str(len(ThickWalls)) + " \n")
                                myfile.write("SCALARS Apo_flux_(m/s) float \n")
                                myfile.write("LOOKUP_TABLE default \n")
                                for ThickWallNode in ThickWalls:
                                    cellnumber1=ThickWallNode[2]-(network.n_walls + network.n_junctions)
                                    myfile.write(str(float(InterCFlowDensity[int(cellnumber1)])/SECONDS_PER_DAY/CM_PER_METER) + " \n") #Flow rate from wall (non junction) to cell    min(sath1,max(satl1,  ))
                            myfile.close()
                            text_file.close()
                    
                    
                    
                    if config.paraview_pf==1: #Plasmodesmata flow density data disks
                        text_file = open(newpath+"Plasmodesm3Db"+str(Barrier)+","+str(iMaturity)+"s"+str(count)+".pvtk", "w")
                        #sath3=max(PlasmodesmFlowDensity)*config.color_threshold
                        #satl3=min(PlasmodesmFlowDensity)*config.color_threshold
                        #if satl3<-sath3: #min(PlasmodesmFlowDensity)<0:
                        #    sath3=-satl3
                        #else:
                        #    satl3=-sath3
                        with open(newpath+"Plasmodesm3Db"+str(Barrier)+","+str(iMaturity)+"s"+str(count)+".pvtk", "a") as myfile:
                            myfile.write("# vtk DataFile Version 4.0 \n")
                            myfile.write("PD flux disks 3D \n")
                            myfile.write("ASCII \n")
                            myfile.write(" \n")
                            myfile.write("DATASET UNSTRUCTURED_GRID \n")
                            myfile.write("POINTS "+str(len(PlasmodesmFlowDensity)*12)+" float \n")
                            for ThickWallNode in ThickWalls:
                                if ThickWallNode[1]<network.n_walls: #selection of new walls (not new junctions)
                                    if ThickWallNode[7]==0: #new walls that are not at the interface with soil or xylem, where there is no plasmodesmata   #if G.nodes[int(ThickWallNode[1])]['borderlink']==0
                                        #calculate the XY slope between the two neighbouring new junctions
                                        twpid1=int(ThickWallNode[5])
                                        twpid2=int(ThickWallNode[6])
                                        if not ThickWalls[twpid1][3]==ThickWalls[twpid2][3]: #Otherwise we'll get a division by 0 error
                                            slopeNJ=(ThickWalls[twpid1][4]-ThickWalls[twpid2][4])/(ThickWalls[twpid1][3]-ThickWalls[twpid2][3]) #slope of the line connecting the new junction nodes neighbouring the new wall
                                        else:
                                            slopeNJ=inf
                                        x0=ThickWallNode[3]
                                        y0=ThickWallNode[4]
                                        z0=config.radius_plasmodesm_disp*3
                                        #Calculate the horizontal distance between XY0 and the cell center, compare it with the distance between the mean position of the new junctions. If the latter is closer to the cell center, it becomes the new XY0 to make sur the disk is visible
                                        xC=network.position[int(ThickWallNode[2])][0]
                                        yC=network.position[int(ThickWallNode[2])][1]
                                        xNJ=(ThickWalls[twpid1][3]+ThickWalls[twpid2][3])/2.0
                                        yNJ=(ThickWalls[twpid1][4]+ThickWalls[twpid2][4])/2.0
                                        if sqrt(square(x0-xC)+square(y0-yC)) > sqrt(square(xNJ-xC)+square(yNJ-yC)):
                                            x0=xNJ
                                            y0=yNJ
                                        for i in range(12):
                                            x=x0+cos(arctan(slopeNJ))*config.radius_plasmodesm_disp*cos(int(i)*pi/6.0)
                                            y=y0+sin(arctan(slopeNJ))*config.radius_plasmodesm_disp*cos(int(i)*pi/6.0)
                                            z=z0+config.radius_plasmodesm_disp*sin(int(i)*pi/6.0)
                                            myfile.write(str(x) + " " + str(y) + " " + str(z) + " \n")
                                else:
                                    break #interrupts the for loop in case we reached the new junction nodes
                            myfile.write(" \n")
                            myfile.write("CELLS " + str(len(PlasmodesmFlowDensity)) + " " + str(len(PlasmodesmFlowDensity)*13) + " \n") #The number of cells corresponds to the number of lines in ThickWalls
                            for i in range(len(PlasmodesmFlowDensity)):
                                if PlasmodesmFlowDensity[i]==0:
                                    myfile.write("12 " + str(i*12+0) + " " + str(i*12+0) + " " + str(i*12+0) + " " + str(i*12+0) + " " + str(i*12+0) + " " + str(i*12+0) + " " + str(i*12+0) + " " + str(i*12+0) + " " + str(i*12+0) + " " + str(i*12+0) + " " + str(i*12+0) + " " + str(i*12+0) + " \n")
                                else:
                                    myfile.write("12 " + str(i*12+0) + " " + str(i*12+1) + " " + str(i*12+2) + " " + str(i*12+3) + " " + str(i*12+4) + " " + str(i*12+5) + " " + str(i*12+6) + " " + str(i*12+7) + " " + str(i*12+8) + " " + str(i*12+9) + " " + str(i*12+10) + " " + str(i*12+11) + " \n")
                            myfile.write(" \n")
                            myfile.write("CELL_TYPES " + str(len(PlasmodesmFlowDensity)) + " \n")
                            for i in range(len(PlasmodesmFlowDensity)):
                                myfile.write("7 \n") #Polygon cell type 
                            myfile.write(" \n")
                            myfile.write("POINT_DATA " + str(len(PlasmodesmFlowDensity)*12) + " \n")
                            myfile.write("SCALARS PD_Flux_(m/s) float \n")
                            myfile.write("LOOKUP_TABLE default \n")
                            for i in range(len(PlasmodesmFlowDensity)):
                                for j in range(12):
                                    myfile.write(str(float(PlasmodesmFlowDensity[i])/SECONDS_PER_DAY/CM_PER_METER) + " \n") #min(sath3,max(satl3, ))
                        myfile.close()
                        text_file.close()
                    
                    
                    if config.paraview_mf==1 and config.paraview_pf==1: #Membranes and plasmodesms in the same file
                        text_file = open(newpath+"Membranes_n_plasmodesm3Db"+str(Barrier)+","+str(iMaturity)+"s"+str(count)+".pvtk", "w")
                        with open(newpath+"Membranes_n_plasmodesm3Db"+str(Barrier)+","+str(iMaturity)+"s"+str(count)+".pvtk", "a") as myfile:
                            myfile.write("# vtk DataFile Version 4.0 \n")
                            myfile.write("Membranes geometry and plasmodesm disks 3D \n")
                            myfile.write("ASCII \n")
                            myfile.write(" \n")
                            myfile.write("DATASET UNSTRUCTURED_GRID \n")
                            myfile.write("POINTS "+str(len(ThickWalls)*2+len(PlasmodesmFlowDensity)*12)+" float \n")
                            for ThickWallNode in ThickWalls:
                                myfile.write(str(ThickWallNode[3]) + " " + str(ThickWallNode[4]) + " 0.0 \n")
                            for ThickWallNode in ThickWalls:
                                myfile.write(str(ThickWallNode[3]) + " " + str(ThickWallNode[4]) + " " + str(height) + " \n")
                            for ThickWallNode in ThickWalls:
                                if ThickWallNode[1]<network.n_walls: #selection of new walls (not new junctions)
                                    if ThickWallNode[7]==0: #new walls that are not at the interface with soil or xylem, where there is no plasmodesmata   #if G.nodes[int(ThickWallNode[1])]['borderlink']==0
                                        #calculate the XY slope between the two neighbouring new junctions
                                        twpid1=int(ThickWallNode[5])
                                        twpid2=int(ThickWallNode[6])
                                        if not ThickWalls[twpid1][3]==ThickWalls[twpid2][3]: #Otherwise we'll get a division by 0 error
                                            slopeNJ=(ThickWalls[twpid1][4]-ThickWalls[twpid2][4])/(ThickWalls[twpid1][3]-ThickWalls[twpid2][3]) #slope of the line connecting the new junction nodes neighbouring the new wall
                                        else:
                                            slopeNJ=inf
                                        x0=ThickWallNode[3]
                                        y0=ThickWallNode[4]
                                        z0=config.radius_plasmodesm_disp*3
                                        #Calculate the horizontal distance between XY0 and the cell center, compare it with the distance between the mean network.position of the new junctions. If the latter is closer to the cell center, it becomes the new XY0 to make sur the disk is visible
                                        xC=network.position[int(ThickWallNode[2])][0]
                                        yC=network.position[int(ThickWallNode[2])][1]
                                        xNJ=(ThickWalls[twpid1][3]+ThickWalls[twpid2][3])/2.0
                                        yNJ=(ThickWalls[twpid1][4]+ThickWalls[twpid2][4])/2.0
                                        if sqrt(square(x0-xC)+square(y0-yC)) > sqrt(square(xNJ-xC)+square(yNJ-yC)):
                                            x0=xNJ
                                            y0=yNJ
                                        for i in range(12):
                                            x=x0+cos(arctan(slopeNJ))*config.radius_plasmodesm_disp*cos(int(i)*pi/6.0)
                                            y=y0+sin(arctan(slopeNJ))*config.radius_plasmodesm_disp*cos(int(i)*pi/6.0)
                                            z=z0+config.radius_plasmodesm_disp*sin(int(i)*pi/6.0)
                                            myfile.write(str(x) + " " + str(y) + " " + str(z) + " \n")
                                else:
                                    break #interrupts the for loop in case we reached the new junction nodes
                            myfile.write(" \n")
                            myfile.write("CELLS " + str(len(ThickWalls)-len(list_ghostwalls)*4+len(PlasmodesmFlowDensity)) + " " + str(len(ThickWalls)*5-len(list_ghostwalls)*20+len(PlasmodesmFlowDensity)*13) + " \n") #The number of cells corresponds to the number of lines in ThickWalls
                            for ThickWallNode in ThickWalls:
                                if ThickWallNode[1]>=network.n_walls: #wall that is a junction
                                    if ThickWalls[int(ThickWallNode[5])][1] not in list_ghostwalls:
                                        myfile.write("4 " + str(int(ThickWallNode[0])) + " " + str(int(ThickWallNode[5])) + " " + str(int(ThickWallNode[5])+len(ThickWalls)) + " " + str(int(ThickWallNode[0])+len(ThickWalls)) + " \n")
                                    if ThickWalls[int(ThickWallNode[6])][1] not in list_ghostwalls:
                                        myfile.write("4 " + str(int(ThickWallNode[0])) + " " + str(int(ThickWallNode[6])) + " " + str(int(ThickWallNode[6])+len(ThickWalls)) + " " + str(int(ThickWallNode[0])+len(ThickWalls)) + " \n")
                            for i in range(len(PlasmodesmFlowDensity)):
                                if PlasmodesmFlowDensity[i]==0:
                                    myfile.write("12 " + str(i*12+0+len(ThickWalls)*2) + " " + str(i*12+0+len(ThickWalls)*2) + " " + str(i*12+0+len(ThickWalls)*2) + " " + str(i*12+0+len(ThickWalls)*2) + " " + str(i*12+0+len(ThickWalls)*2) + " " + str(i*12+0+len(ThickWalls)*2) + " " + str(i*12+0+len(ThickWalls)*2) + " " + str(i*12+0+len(ThickWalls)*2) + " " + str(i*12+0+len(ThickWalls)*2) + " " + str(i*12+0+len(ThickWalls)*2) + " " + str(i*12+0+len(ThickWalls)*2) + " " + str(i*12+0+len(ThickWalls)*2) + " \n")
                                else:
                                    myfile.write("12 " + str(i*12+0+len(ThickWalls)*2) + " " + str(i*12+1+len(ThickWalls)*2) + " " + str(i*12+2+len(ThickWalls)*2) + " " + str(i*12+3+len(ThickWalls)*2) + " " + str(i*12+4+len(ThickWalls)*2) + " " + str(i*12+5+len(ThickWalls)*2) + " " + str(i*12+6+len(ThickWalls)*2) + " " + str(i*12+7+len(ThickWalls)*2) + " " + str(i*12+8+len(ThickWalls)*2) + " " + str(i*12+9+len(ThickWalls)*2) + " " + str(i*12+10+len(ThickWalls)*2) + " " + str(i*12+11+len(ThickWalls)*2) + " \n")
                            myfile.write(" \n")
                            myfile.write("CELL_TYPES " + str(len(ThickWalls)-len(list_ghostwalls)*4+len(PlasmodesmFlowDensity)) + " \n")
                            for ThickWallNode in ThickWalls:
                                if ThickWallNode[1]>=network.n_walls: #wall that is a junction
                                    if ThickWalls[int(ThickWallNode[5])][1] not in list_ghostwalls:
                                        myfile.write("9 \n") #Quad cell type
                                    if ThickWalls[int(ThickWallNode[6])][1] not in list_ghostwalls:
                                        myfile.write("9 \n") #Quad cell type
                            for i in range(len(PlasmodesmFlowDensity)):
                                myfile.write("7 \n") #Polygon cell type 
                            myfile.write(" \n")
                            myfile.write("POINT_DATA " + str(len(ThickWalls)*2+len(PlasmodesmFlowDensity)*12) + " \n")
                            myfile.write("SCALARS TM_n_PD_flux_(m/s) float \n")
                            myfile.write("LOOKUP_TABLE default \n")
                            for ThickWallNode in ThickWalls:
                                if ThickWallNode[0]<len(MembraneFlowDensity):
                                    myfile.write(str(float(MembraneFlowDensity[int(ThickWallNode[0])])/SECONDS_PER_DAY/CM_PER_METER) + " \n") #Flow rate from wall (non junction) to cell
                                else:
                                    myfile.write(str(float((MembraneFlowDensity[int(ThickWallNode[5])]+MembraneFlowDensity[int(ThickWallNode[6])])/2)/SECONDS_PER_DAY/CM_PER_METER) + " \n") #Flow rate from junction wall to cell is the average of the 2 neighbouring wall flow rates
                            for ThickWallNode in ThickWalls:
                                if ThickWallNode[0]<len(MembraneFlowDensity):
                                    myfile.write(str(float(MembraneFlowDensity[int(ThickWallNode[0])])/SECONDS_PER_DAY/CM_PER_METER) + " \n") #Flow rate from wall (non junction) to cell
                                else:
                                    myfile.write(str(float((MembraneFlowDensity[int(ThickWallNode[5])]+MembraneFlowDensity[int(ThickWallNode[6])])/2)/SECONDS_PER_DAY/CM_PER_METER) + " \n") #Flow rate from junction wall to cell is the average of the 2 neighbouring wall flow rates
                            for i in range(len(PlasmodesmFlowDensity)):
                                for j in range(12):
                                    myfile.write(str(float(PlasmodesmFlowDensity[i])/SECONDS_PER_DAY/CM_PER_METER) + " \n")
                        myfile.close()
                        text_file.close()
            
        
        #write down kr_tot and Uptake distributions in matrices
        iMaturity=-1
        kr_tot_saved = []
        for stage in range(len(config.maturity_stages)):
            Barrier=config.maturity_stages[stage]['barrier'] #Apoplastic barriers (0: No apoplastic barrier, 1:Endodermis radial walls, 2:Endodermis with passage cells, 3: Endodermis full, 4: Endodermis full and exodermis radial walls)
            height=config.maturity_stages[stage]['height'] #Cell length in the axial direction (microns)
            
            iMaturity+=1
            kr_tot_saved.append(kr_tot[iMaturity][0])
            text_file = open(newpath+"Macro_prop_"+str(Barrier)+","+str(iMaturity)+".txt", "w")
            with open(newpath+"Macro_prop_"+str(Barrier)+","+str(iMaturity)+".txt", "a") as myfile:
                myfile.write("Macroscopic root radial hydraulic properties, apoplastic barrier "+str(Barrier)+","+str(iMaturity)+" \n")
                myfile.write("\n")
                myfile.write(str(config.n_scenarios-1)+" scenarios \n")
                myfile.write("\n")
                myfile.write("Cross-section height: "+str(height*1.0E-04)+" cm \n")
                myfile.write("\n")
                myfile.write("Cross-section network.perimeter: "+str(network.perimeter[0])+" cm \n")
                myfile.write("\n")
                myfile.write("Xylem specific axial conductance: "+str(K_xyl_spec)+" cm^4/hPa/d \n")
                myfile.write("\n")
                myfile.write("Cross-section radial conductivity: "+str(kr_tot[iMaturity][0])+" cm/hPa/d \n")
                myfile.write("\n")
                myfile.write("Number of radial discretization boxes: \n")
                r_discret_txt=' '.join(map(str, network.r_discret.T)) 
                myfile.write(r_discret_txt[1:21]+" \n")
                myfile.write("\n")
                myfile.write("Radial distance from stele centre (microns): \n")
                for j in network.layer_dist:
                    myfile.write(str(float(j.item()))+" \n")
                myfile.write("\n")
                myfile.write("Standard Transmembrane uptake Fractions (%): \n")
                for j in range(int(network.r_discret[0].item())):
                    myfile.write(str(STFlayer_plus[j][iMaturity]*100)+" \n")
                myfile.write("\n")
                myfile.write("Standard Transmembrane release Fractions (%): \n")
                for j in range(int(network.r_discret[0].item())):
                    myfile.write(str(STFlayer_minus[j][iMaturity]*100)+" \n")
                for i in range(1,config.n_scenarios):
                    myfile.write("\n")
                    myfile.write("\n")
                    myfile.write("Scenario "+str(i)+" \n")
                    myfile.write("\n")
                    myfile.write("h_x: "+str(Psi_xyl[iMaturity][i])+" hPa \n")
                    myfile.write("\n")
                    myfile.write("h_s: "+str(config.scenarios[i].get("psi_soil_left"))+" to "+str(config.scenarios[i].get("psi_soil_right"))+" hPa \n")
                    myfile.write("\n")
                    myfile.write("h_p: "+str(Psi_sieve[iMaturity][i])+" hPa \n")
                    myfile.write("\n")
                    myfile.write("O_x: "+str(config.scenarios[i].get("osmotic_xyl"))+" to "+str(config.scenarios[i].get("osmotic_endo"))+" hPa \n")
                    myfile.write("\n")
                    myfile.write("O_s: "+str(config.scenarios[i].get("osmotic_left_soil"))+" to "+str(config.scenarios[i].get("osmotic_right_soil"))+" hPa \n")
                    myfile.write("\n")
                    myfile.write("O_p: "+str(Os_sieve[0][i])+" hPa \n")
                    myfile.write("\n")
                    myfile.write("Xcontact: "+str(Xcontact)+" microns \n")
                    myfile.write("\n")
                    if Barrier==0:
                        myfile.write("Elong_cell: "+str(Elong_cell[0][i])+" cm/d \n")
                        myfile.write("\n")
                        myfile.write("Elong_cell_side_diff: "+str(Elong_cell_side_diff[0][i])+" cm/d \n")
                        myfile.write("\n")
                    else:
                        myfile.write("Elong_cell: "+str(0.0)+" cm/d \n")
                        myfile.write("\n")
                        myfile.write("Elong_cell_side_diff: "+str(0.0)+" cm/d \n")
                        myfile.write("\n")
                    myfile.write("kw: "+str(kw)+" cm^2/hPa/d \n")
                    myfile.write("\n")
                    myfile.write("Kpl: "+str(Kpl)+" cm^3/hPa/d \n")
                    myfile.write("\n")
                    myfile.write("kAQP: "+str(kaqp_cortex)+" cm/hPa/d \n")
                    myfile.write("\n")
                    myfile.write("s_hetero: "+str(config.scenarios[count].get("psi_s_hetero"))+" \n")
                    myfile.write("\n")
                    myfile.write("s_factor: "+str(config.scenarios[count].get("psi_s_factor"))+" \n")
                    myfile.write("\n")
                    myfile.write("Os_hetero: "+str(config.scenarios[count].get("psi_os_hetero"))+" \n")
                    myfile.write("\n")
                    myfile.write("Os_cortex: "+str(config.scenarios[count].get("psi_os_cortex"))+" hPa \n")
                    myfile.write("\n")
                    myfile.write("q_tot: "+str(Q_tot[iMaturity][i]/height/1.0E-04)+" cm^2/d \n")
                    myfile.write("\n")
                    myfile.write("Stele, cortex, and epidermis uptake distribution cm^3/d: \n")
                    for j in range(int(network.r_discret[0])):
                        myfile.write(str(UptakeLayer_plus[j][iMaturity][i])+" \n")
                    myfile.write("\n")
                    myfile.write("Stele, cortex, and epidermis release distribution cm^3/d: \n")
                    for j in range(int(network.r_discret[0])):
                        myfile.write(str(UptakeLayer_minus[j][iMaturity][i])+" \n")
                    myfile.write("\n")
                    myfile.write("Xylem uptake distribution cm^3/d: \n")
                    for j in range(int(network.r_discret[0])):
                        myfile.write(str(Q_xyl_layer[j][iMaturity][i])+" \n")
                    myfile.write("\n")
                    myfile.write("Phloem uptake distribution cm^3/d: \n")
                    for j in range(int(network.r_discret[0])):
                        myfile.write(str(Q_sieve_layer[j][iMaturity][i])+" \n")
                    myfile.write("\n")
                    myfile.write("Elongation flow convergence distribution cm^3/d: \n")
                    for j in range(int(network.r_discret[0])):
                        myfile.write(str(Q_elong_layer[j][iMaturity][i])+" \n")
                    myfile.write("\n")
                    myfile.write("Cell layers pressure potentials: \n")
                    for j in range(int(network.r_discret[0])):
                        myfile.write(str(PsiCellLayer[j][iMaturity][i])+" \n")
                    myfile.write("\n")
                    myfile.write("Cell layers osmotic potentials: \n")
                    for j in range(int(network.r_discret[0])):
                        myfile.write(str(OsCellLayer[j][iMaturity][i])+" \n")
                    myfile.write("\n")
                    myfile.write("Wall layers pressure potentials: \n")
                    for j in range(int(network.r_discret[0])):
                        if NWallLayer[j][iMaturity][i]>0:
                            myfile.write(str(PsiWallLayer[j][iMaturity][i]/NWallLayer[j][iMaturity][i])+" \n")
                        else:
                            myfile.write("nan \n")
                    myfile.write("\n")
                    myfile.write("Wall layers osmotic potentials: \n")
                    for j in range(int(network.r_discret[0])):
                        myfile.write(str(OsWallLayer[j][iMaturity][i])+" \n")
            myfile.close()
            text_file.close()
        
        return kr_tot_saved
        
        if config.sym_contagion == 1: #write down results of the hydropatterning study
            iMaturity=-1
            for Maturity in Maturityrange:
                Barrier=int(Maturity.get("Barrier"))
                height=int(Maturity.get("height")) #(microns)
                iMaturity+=1
                text_file = open(newpath+"Hydropatterning_"+str(Barrier)+","+str(iMaturity)+".txt", "w")
                with open(newpath+"Hydropatterning_"+str(Barrier)+","+str(iMaturity)+".txt", "a") as myfile:
                    myfile.write("Is there symplastic mass flow from source to target cells? Apoplastic barrier "+str(Barrier)+","+str(iMaturity)+" \n")
                    myfile.write("\n")
                    myfile.write(str(config.n_scenarios-1)+" scenarios \n")
                    myfile.write("\n")
                    myfile.write("Template: "+path+" \n")
                    myfile.write("\n")
                    myfile.write("Source cell: "+str(config.sym_zombie0)+" \n")
                    myfile.write("\n")
                    myfile.write("Target cells: "+str(config.sym_target)+" \n")
                    myfile.write("\n")
                    myfile.write("Immune cells: "+str(config.sym_immune)+" \n")
                    myfile.write("\n")
                    myfile.write("Cross-section height: "+str(height*1.0E-04)+" cm \n")
                    myfile.write("\n")
                    myfile.write("Cross-section network.perimeter: "+str(network.perimeter[0])+" cm \n")
                    myfile.write("\n")
                    myfile.write("Xcontact: "+str(Xcontact)+" microns \n")
                    myfile.write("\n")
                    myfile.write("kw: "+str(kw)+" cm^2/hPa/d \n")
                    myfile.write("\n")
                    myfile.write("Kpl: "+str(Kpl)+" cm^3/hPa/d \n")
                    myfile.write("\n")
                    myfile.write("kAQP: "+str(kaqp_cortex)+" cm/hPa/d \n")
                    myfile.write("\n")
                    if Barrier==0:
                        myfile.write("Cell elongation rate: "+str(Elong_cell)+" cm/d \n")
                    else: #No elongation after formation of the Casparian strip
                        myfile.write("Cell elongation rate: "+str(0.0)+" cm/d \n")
                    myfile.write("\n")
                    for i in range(1,config.n_scenarios):
                        myfile.write("\n")
                        myfile.write("\n")
                        myfile.write("Scenario "+str(i)+" \n")
                        myfile.write("\n")
                        myfile.write("Expected hydropatterining response (1: Wet-side XPP; -1 to 0: Unclear; 2: Dry-side XPP) \n")
                        myfile.write("Hydropat.: "+str(int(Hydropatterning[iMaturity][i]))+" \n")
                        myfile.write("\n")
                        myfile.write("h_x: "+str(Psi_xyl[iMaturity][i])+" hPa, h_s: "+str(config.scenarios[i].get("psi_soil_left"))+" to "+str(config.scenarios[i].get("psi_soil_right"))+" hPa, h_p: "+str(Psi_sieve[iMaturity][i])+" hPa \n")
                        myfile.write("\n")
                        myfile.write("O_x: "+str(config.scenarios[i].get("osmotic_xyl"))+" to "+str(config.scenarios[i].get("osmotic_xyl"))+" hPa, O_s: "+str(config.scenarios[i].get("osmotic_left_soil"))+" to "+str(config.scenarios[i].get("osmotic_right_soil"))+" hPa, O_p: "+str(Os_sieve[0][i])+" hPa \n")
                        myfile.write("\n")
                        myfile.write("Os_cortex: "+str(config.scenarios[count].get("psi_os_cortex"))+" hPa, Os_hetero: "+str(config.scenarios[count].get("psi_os_hetero"))+", s_hetero: "+str(config.scenarios[count].get("psi_s_hetero"))+", s_factor: "+str(config.scenarios[count].get("psi_s_factor"))+" \n")
                        myfile.write("\n")
                        myfile.write("q_tot: "+str(Q_tot[iMaturity][i]/height/1.0E-04)+" cm^2/d \n")
                        myfile.write("\n")
                myfile.close()
                text_file.close()
        
        if config.apo_contagion == 1: #write down results of the hydrotropism study
            iMaturity=-1
            for Maturity in Maturityrange:
                Barrier=int(Maturity.get("Barrier"))
                height=int(Maturity.get("height")) #(microns)
                iMaturity+=1
                text_file = open(newpath+"Hydrotropism_"+str(Barrier)+","+str(iMaturity)+".txt", "w")
                with open(newpath+"Hydrotropism_"+str(Barrier)+","+str(iMaturity)+".txt", "a") as myfile:
                    myfile.write("Is there apoplastic mass flow from source to target cells? Apoplastic barrier "+str(Barrier)+","+str(iMaturity)+" \n")
                    myfile.write("\n")
                    myfile.write(str(config.n_scenarios-1)+" scenarios \n")
                    myfile.write("\n")
                    myfile.write("Template: "+path+" \n")
                    myfile.write("\n")
                    myfile.write("Source cell: "+str(config.apo_zombie0)+" \n")
                    myfile.write("\n")
                    myfile.write("Target cells: "+str(config.apo_target)+" \n")
                    myfile.write("\n")
                    myfile.write("Immune cells: "+str(config.apo_immune)+" \n")
                    myfile.write("\n")
                    myfile.write("Cross-section height: "+str(height*1.0E-04)+" cm \n")
                    myfile.write("\n")
                    myfile.write("Cross-section network.perimeter: "+str(network.perimeter[0])+" cm \n")
                    myfile.write("\n")
                    myfile.write("Xcontact: "+str(Xcontact)+" microns \n")
                    myfile.write("\n")
                    myfile.write("kw: "+str(kw)+" cm^2/hPa/d \n")
                    myfile.write("\n")
                    myfile.write("Kpl: "+str(Kpl)+" cm^3/hPa/d \n")
                    myfile.write("\n")
                    myfile.write("kAQP: "+str(kaqp_cortex)+" cm/hPa/d \n")
                    myfile.write("\n")
                    if Barrier==0:
                        myfile.write("Cell elongation rate: "+str(Elong_cell)+" cm/d \n")
                    else: #No elongation after formation of the Casparian strip
                        myfile.write("Cell elongation rate: "+str(0.0)+" cm/d \n")
                    myfile.write("\n")
                    for i in range(1,config.n_scenarios):
                        myfile.write("\n")
                        myfile.write("\n")
                        myfile.write("Scenario "+str(i)+" \n")
                        myfile.write("\n")
                        myfile.write("Expected hydrotropism response (1: All cell walls reached by ABA; 0: No target walls reached by ABA) \n")
                        myfile.write("Hydropat.: "+str(int(Hydrotropism[iMaturity][i]))+" \n")
                        myfile.write("\n")
                        myfile.write("h_x: "+str(Psi_xyl[iMaturity][i])+" hPa, h_s: "+str(config.scenarios[i].get("psi_soil_left"))+" to "+str(config.scenarios[i].get("psi_soil_right"))+" hPa, h_p: "+str(Psi_sieve[iMaturity][i])+" hPa \n")
                        myfile.write("\n")
                        myfile.write("O_x: "+str(config.scenarios[i].get("osmotic_xyl"))+" to "+str(config.scenarios[i].get("osmotic_xyl"))+" hPa, O_s: "+str(config.scenarios[i].get("osmotic_left_soil"))+" to "+str(config.scenarios[i].get("osmotic_right_soil"))+" hPa, O_p: "+str(Os_sieve[0][i])+" hPa \n")
                        myfile.write("\n")
                        myfile.write("Os_cortex: "+str(config.scenarios[count].get("psi_os_cortex"))+" hPa, Os_hetero: "+str(config.scenarios[count].get("psi_os_hetero"))+", s_hetero: "+str(config.scenarios[count].get("psi_s_hetero"))+", s_factor: "+str(config.scenarios[count].get("psi_s_factor"))+" \n")
                        myfile.write("\n")
                        myfile.write("q_tot: "+str(Q_tot[iMaturity][i]/height/1.0E-04)+" cm^2/d \n")
                        myfile.write("\n")
                myfile.close()
                text_file.close()
    
    print("End of mecha")



def update_xml_attributes(file_path, parent_tag, child_tag, updates, output_path=None):
    """
    Update one or more attributes of an XML element.
    Works for parent+child (e.g. <config.kaqp_elems><kAQP .../></config.kaqp_elems>)
    or standalone tags (e.g. <km value="1" />).

    Parameters
    ----------
    file_path : str
        Path to the input XML file.
    parent_tag : str
        Name of the parent element (e.g., "config.kaqp_elems").
    child_tag : str
        Name of the child element inside the parent (e.g., "kAQP").
        If the tag has no parent, pass the same value for parent_tag and child_tag.
    updates : dict
        Dictionary of attribute updates, e.g. {"value": 0.002, "cortex_factor": 0.9}.
    output_path : str or None
        Path to save the modified XML file. If None, overwrites the input file.
    """
    tree = ET.parse(file_path)
    root = tree.getroot()

    # handle parent+child or standalone
    if parent_tag == child_tag:
        elem = root.find(f".//{parent_tag}")
    else:
        elem = root.find(f".//{parent_tag}/{child_tag}")

    if elem == None:
        raise ValueError(f"No <{child_tag}> element found (parent={parent_tag}).")

    # apply all updates
    for attr, val in updates.items():
        elem.set(attr, str(val))

    # overwrite or save new file
    if output_path == None:
        output_path = file_path  # overwrite original

    tree.write(output_path, encoding="UTF-8", xml_declaration=True)


def set_hydraulic_scenario(xml_path, barriers):
    """
    Activate one or multiple hydraulic scenarios (Barrier values)
    inside the <Maturityrange> section of a MECHA XML file.
    Keeps <Maturityrange> tags intact, adding new barriers if missing.

    Parameters
    ----------
    xml_path : str
        Path to the MECHA XML file
    barriers : int | list[int]
        Barrier value(s) to activate
    """
    if isinstance(barriers, int):
        barriers = [barriers].sort()

    with open(xml_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Extract the <Maturityrange> section
    range_match = re.search(r'(<Maturityrange>)(.*?)(</Maturityrange>)', content, re.DOTALL)
    if not range_match:
        raise ValueError("No <Maturityrange> section found in XML.")

    start_tag, inner_text, end_tag = range_match.groups()

    # Match existing <Maturity ... /> lines
    maturity_pattern = re.compile(r'(\s*)(?:<!--\s*)?(<Maturity\s+Barrier="(\d+)"[^>]*\/>)(?:\s*-->)?')

    existing_barriers = {}
    def replacer(match):
        indent, tag, barrier_str = match.groups()
        barrier = int(barrier_str)
        existing_barriers[barrier] = tag  # remember existing tags
        if barrier in barriers:
            return f"{indent}{tag}"  # activate
        else:
            return f"{indent}<!-- {tag} -->"  # deactivate

    # Apply activation/deactivation to existing lines
    new_inner = maturity_pattern.sub(replacer, inner_text)

    # Add missing barriers
    indent_match = re.search(r'(\s*)<Maturity', inner_text)
    indent = indent_match.group(1) if indent_match else '    '  # default 4 spaces
    for barrier in barriers:
        if barrier not in existing_barriers:
            new_inner += f"\n{indent}<Maturity Barrier=\"{barrier}\" height=\"200\" Nlayers=\"1\"/>"

    # Rebuild the <Maturityrange> section
    new_range_section = f"{start_tag}{new_inner}\n{end_tag}"

    # Replace in the full content
    new_content = content[:range_match.start()] + new_range_section + content[range_match.end():]

    # Write back
    with open(xml_path, "w", encoding="utf-8") as f:
        f.write(new_content)

mecha()