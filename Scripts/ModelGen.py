#%% Import stuff
import os
from functions.grid_funcs import *
import shutil
import matplotlib.pyplot as plt
import pandas as pd

name = 'pomp'

#grid settings
radius = 2000.0 #radius circle
max_area = 50000 # max area at boundary
zonearea = 50
zonedist = 475

areas = [50000,10000,2500,1200, zonearea]
dists = [1500,1000,600,zonedist]

#Parameter settings
k = np.array([2.5,5]) #K in m/d
D = 10 #in m 
c = 400 #in d
ss = 0.00001 
Q = 1000 # in m³/h
nlay = 3

#Observation settings
SL = np.sqrt(k[1]*D*c) #Spreidingslengte in m
ObsR = [ 0.5 * SL, SL, 2*SL,3*SL]
ObsCoords = select_coordinates(zonedist, ObsR)

#temporal settings
Tlen = 8 #Time (h)
Tsteps = 50 #Timesteps 
Tmult = 1.05 #Time multiplier

VG,GI,gridprops = MakeGrid(os.path.join('..', 'grid'), radius, max_area, areas, dists)
# PlotGrid(VG, zonedist, ObsR)

ws = os.path.join('..', f'ws')
if not os.path.isdir(ws):
    os.makedirs(ws)
else:
    shutil.rmtree(ws, ignore_errors=True)

gwf,sim = Init_Modflow(ws, name,gridprops, radius,GI, k,D,c, Q, 'open', ObsCoords, ss, nlay, Tlen, Tsteps,Tmult )

zonearray = get_zonearray(zonedist-5,VG,ws)

df = pd.DataFrame({'x' : gwf.modelgrid.xcellcenters, 
                   'y' : gwf.modelgrid.ycellcenters,
                   'zone' : zonearray})

df.to_csv(os.path.join('..', 'inter', 'ZoneCoords.csv'), index_label = 'index')
ObsCoords.to_csv(os.path.join('..', 'inter', 'ObsCoords.csv'), index_label = 'index')





# %%
