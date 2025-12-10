#%%
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import LineString,Point,Polygon
from flopy.discretization import VertexGrid
from flopy.utils.triangle import Triangle as Triangle
from flopy.utils.voronoi import VoronoiGrid
import flopy
import pandas as pd
import os
from sklearn.metrics import r2_score
import ttim as ttm


def circle_poly(radius:int) -> list:
    """Create polygon of approximate circle

    Parameters
    ----------
    radius : int

    Returns
    -------
    list
        x,y coordinates in list
    """    
    theta = np.arange(0.0, 2 * np.pi, 0.2)
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    circle_poly = list(zip(x, y))
    circle_poly.append(circle_poly[0])
    return circle_poly

def MakeGrid(ws:str, radius:float, max_area:list, areas:list, dists:list):
    """Create circular voronoi grid with varying cell size

    Parameters
    ----------
    ws : str
        folder with modflow model
    radius : float
        Radius of circular grid
    max_area : float
        maximum area of largest cells at outer boundaryin triangular grid. voronoi grid cells become larger. 
    areas : list
        maximum area of cells per zone in triangular grid. voronoi grid cells become larger. 
    dists : list
        radial distances of zones

    Returns
    -------
    _type_
        modelgrid, intersect boject, disv_gridprops dict and tri
    """    
    wstri = os.path.join(ws,'tri')
    if not os.path.isdir(wstri):
        os.makedirs(wstri)
    tri = Triangle(angle=0, model_ws=wstri, exe_name = os.path.join('..', 'exe','triangle'), nodes = np.array([[0,0]]))
    areas = np.array(areas)
    #create full triangular grid
    tri.add_polygon(circle_poly(radius))
    tri.add_region((0,radius-1),0, maximum_area = max_area)
    #Refine regions
    for no,x in enumerate(dists):
        tri.add_polygon(circle_poly(x))
        tri.add_region((0,x),no+1, maximum_area = areas[no])
    tri.add_region((0,0),no+2, maximum_area = areas[no+1])
    tri.build(verbose=False)
    #Convert to Voronoi
    vor = VoronoiGrid(tri)
    gridprops = vor.get_gridprops_vertexgrid()
    disv_gridprops = vor.get_disv_gridprops()
    VG = VertexGrid(**gridprops)
    GI = flopy.utils.GridIntersect(VG)
    return VG, GI,disv_gridprops

def PlotGrid(VG, r = 200, ObsR = []):
    fig,ax = plt.subplots(2)
    fig.set_size_inches(10,20)
    ax[0].set_aspect("equal")
    VG.plot(ax=ax[0], linewidth = 0.2)
    ax[1].set_aspect("equal")
    VG.plot(ax=ax[1], linewidth = 0.2)
    ax[1].set_xlim(-r,r)
    ax[1].set_ylim(-r,r)
    for o in ObsR:
        ax[1].scatter(0,o)
    ax[1].scatter(0,0)


def get_zonearray(zonedist : int,VG):
    """Create binary array which is true for all cells within a radius

    Parameters
    ----------
    zonedist : int
        radial distance for zone boundary
    VG : VoronoiGrid
        _description_

    Returns
    -------
    np.array
        binary array with zones
    """    
    circle = Polygon(circle_poly(zonedist))
    zonearray = np.ones(VG.ncpl, dtype=bool)
    for node in range(VG.ncpl):
        x, y = VG.xcellcenters[node], VG.ycellcenters[node]
        if not circle.contains(Point(x, y)):
                        zonearray[node] = False
    return zonearray


def Init_Modflow(ws, name,gridprops,radius, GI, k,D,c, Q,TBC, ObsCoords, ss=0.001, nlay =1, Tlen = 1, Tsteps = 240,Tmult = 1.1):
    sim = flopy.mf6.MFSimulation(sim_name=name, version="mf6", exe_name=os.path.join('..', 'exe','mf6'), sim_ws=ws)
    gwf = flopy.mf6.ModflowGwf(sim, modelname=name, save_flows=True)
    tdis = flopy.mf6.ModflowTdis(sim, time_units="DAYS",nper = 1, perioddata=[Tlen, Tsteps, Tmult])
    ims = flopy.mf6.ModflowIms(sim, print_option="SUMMARY", complexity="complex", outer_dvclose=1.0e-8,inner_dvclose=1.0e-8)
    disv = flopy.mf6.ModflowGwfdisv(gwf, nlay=nlay, **gridprops, top=0, botm=[-D, -2*D, -3*D])
    npf = flopy.mf6.ModflowGwfnpf(gwf,k=[k[0]/24,(D/c)/24,k[1]/24],save_specific_discharge=False)
    npf.set_all_data_external(True)
    ic = flopy.mf6.ModflowGwfic(gwf,strt = 0)
    oc = flopy.mf6.ModflowGwfoc(gwf,budget_filerecord=f"{name}.bud",head_filerecord=f"{name}.hds",
        saverecord=None,  printrecord=None)
    ssa = np.full((nlay,disv.ncpl.data), ss)
    ssa[1,:] = 1 #make sure sto become external files per layer
    wellcell = GI.intersect(Point([0,0]))[0][0]
    # ssa[-1,wellcell] = 1
    sto = flopy.mf6.ModflowGwfsto(gwf,pname="sto",ss=ssa,steady_state=False,transient=True)
    sto.set_all_data_external(True)
    # chd = Set_CHD(gwf, radius,GI,nlay)
    wel = flopy.mf6.ModflowGwfwel(gwf, stress_period_data=[[(nlay-1,wellcell), -Q]])

    if TBC == 'open':
        tbc = Set_GHB(gwf)
    obs = Set_Obs(gwf, nlay,GI, ObsCoords)
    # sim.set_all_data_external(True)
    sim.write_simulation()

    # fix_ss(ws, f"{name}.sto_ss_layer{2}.txt",ss) #
    success, buff = sim.run_simulation(report=True, silent=False)
    return gwf,sim


def Set_GHB(gwf,c= 400):
    grid = gwf.modelgrid 
    areas = grid.geo_dataframe.area
    ghb_cells = []
    for cellid in range(grid.ncpl):
        area = areas[cellid]
        cond = area/(c*24)
        ghb_cells.append([(0,cellid),0,cond])
    ghb = flopy.mf6.ModflowGwfghb(gwf, stress_period_data= ghb_cells)
    return ghb


def Set_Obs(gwf, nlay,GI, ObsCoords):
    cont = {}
    for id,row in ObsCoords.iterrows():
        obscell = GI.intersect(Point([row.x,row.y]))[0][0]
        for n in [0,2]:
            cont[f'head_{n}_{int(row.x)}_{int(row.y)}'] = ['head', 'head', (n,obscell)]
    obs = flopy.mf6.ModflowUtlobs(gwf, continuous=cont, print_input = True, filename = 'Obs', digits = 16)
    return obs   

def Set_CHD(gwf, radius,GI,nlay):
    bounds = GI.intersect(LineString(circle_poly(radius-5)))["cellids"]
    bounds = np.array(list(bounds))
    chdlist = []
    for n in range(nlay):
        for icpl in bounds:
            chdlist.append([(n, icpl), 0])
    chd = flopy.mf6.ModflowGwfchd(gwf, stress_period_data=chdlist)
    return chd

def fix_ss(ws, f, ss):
    p = os.path.join(ws, f)
    with open(p, 'r') as file:
        lines = file.readlines()
    
    with open(p, 'w') as file:
        for line in lines:
            values = line.split()
            for value in values:
                file.write(f'{ss:.16f}\n')

def PlotHeads(gwf,t, r =200):
    fig,ax = plt.subplots(2)
    fig.set_size_inches(10,20)
    hds = gwf.output.head().get_data(kstpkper = (t,0))

    pmv0 = flopy.plot.PlotMapView(gwf,ax = ax[0])
    ax[0].set_aspect('equal')
    pmv0.plot_array(hds, cmap="jet", alpha=1)
    pmv1 = flopy.plot.PlotMapView(gwf,ax = ax[1])
    pmv1.plot_array(hds, cmap="jet", alpha=1)
    ax[1].set_aspect('equal')
    ax[1].set_xlim(-r,r)
    ax[1].set_ylim(-r,r)

@np.vectorize
def Theis(r,t,Q0,kD,SsC = 0.001):
    # Short time and close to the extraction well
    # Theis (1935), https://www.waterloohydrogeologic.com/help/aquifertest/_theis_recovery_test_confined.htm
    Q = -Q0
    u2 = (SsC*r**2)/(4*kD*t)
    sdrawdown = Q/(4*np.pi*kD)*ExpInt(1,u2)
    return sdrawdown


def ExpInt(n,u):
    # n wordt niet gebruikt
    # Fast approximation for Wu according to equation 7a and 7b from Srivastava(1998)
    gamma = np.euler_gamma #0.5772 # Euler-Macheroni constant
    # Wu for u<1
    u0  = np.where(u<1.0,u,1) # u=1 is just a dummy to make ln() work on all values
    Wu0 = np.log(np.exp(-gamma)/u0) + 0.9653*u - 0.1690*u0**2
    #Wu for u>=1
    u1 = np.where(u>=1.0,u,1) # u=1 is just a dummy to make ln() work on all values
    Wu1 = 1/(u1*np.exp(u1))*(u1+0.3575)/(u1+1.280)
    # combine Wu0 and Wu1
    Wu = np.where(u<1.0,Wu0,Wu1)
    return Wu

def PlotTS(TS,Theis_TS):
    fig, ax = plt.subplots()

    TS.plot(ax = ax)
    TS['Theis'] = Theis_TS
    TS.Theis.plot(ax = ax, ls = '--')
    # ax.legend(title = f'$R^2$:{round(r2_score(TS.HEAD.values, TS.Theis.values),4)}')

def GetObsCells(GI,ObsR):
    obscell = []    
    for o in ObsR:
        obscell.append(GI.intersect(Point([0,o]))[0][0])
    return obscell


def run_TTIM(gwf,k,c,TBC,rw):
    t = gwf.output.head().get_times()
    ss = gwf.sto.ss.data[0][0]
    q = gwf.wel.stress_period_data.data[0][0][-1]
    # c = 10/gwf.npf.k.data[1][0]
    if TBC == 'closed':
        ml = ttm.ModelMaq(kaq=k/24, z=(0,-10,-20, -30),c=c*24, Saq=ss,Sll = ss, tmin=min(t), tmax=max(t))
    else:
        ml = ttm.ModelMaq(kaq=k/24, z=(0.1,0,-10,-20, -30),c=[c*24,c*24], Saq=ss,Sll = ss, tmin=min(t), tmax=max(t), topboundary= 'semi')
    w = ttm.Well(model=ml, xw=0, yw=0, rw=rw, tsandQ=[(0, -q)], layers=1)
    ml.solve()
    return ml

def select_coordinates(radius, distances):
    # Define the 8 directions (horizontal, vertical, and diagonals)
    directions = [(1, 0), (0, 1), (-1, 0), (0, -1), (1, 1), (-1, 1), (-1, -1), (1, -1)]
    normalized_directions = [(dx / np.sqrt(dx**2 + dy**2), dy / np.sqrt(dx**2 + dy**2)) for dx, dy in directions]
    # Initialize the list of coordinates
    xls= [0]
    yls = [0]
    # Loop through each direction
    for direction in normalized_directions:
        # Loop through each distance
        for distance in distances:
            # Calculate the coordinates in the given direction
                x = distance * direction[0]
                y = distance * direction[1]
                # Add the coordinates to the list if they are within the circle
                if np.sqrt(x**2 + y**2) <= radius:
                    xls.append(x)
                    yls.append(y)
        
    ObsCoords = pd.DataFrame({'x' : xls, 'y' : yls})
    return ObsCoords
# %%
