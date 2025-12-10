"""Setup generic pest files with pilot points based on MF6 model in /ws"""
#%%
import pyemu
import pypestutils.helpers as helpers
import os
import pandas as pd
import numpy as np
from functions.pest_funcs import *
import flopy
#Load paths
name = 'pomp'
PestDir = os.path.join('..','pest_files')
mfexe = os.path.join('..','exe','mf6')

OrgDir = os.path.join('..', f'ws')
master_dir = os.path.join('..',f'Master');

#Load zones and MF
zonearray =  pd.read_csv(os.path.join('..', 'inter', 'ZoneCoords.csv'), index_col = 'index')

#Spatial reference from modelgrid
sr = Get_SR(OrgDir, name)

#Initiate PEST
pf = pyemu.utils.PstFrom(OrgDir,PestDir, remove_existing=True,spatial_reference=sr)
#Fix inputfiles
for n in [1,2,3]:
    fix_k(PestDir, f'pomp.npf_k_layer{n}.txt')

#Pilot point locations and settings
SL = int(np.sqrt(20000))
ppdf = generate_points_within_circle(475, [ 0.5 * SL, SL, 1.5*SL,2*SL,2.5*SL, 3*SL], no_dirs = 16)
ppdf.to_csv(os.path.join('..', 'inter', 'ppdf.csv'), index_label = 'index')



pp_opt= {'try_use_ppu' : True, 'pp_space' : ppdf, 'search_radius' : 2.5*SL, 
         'maxpts_interp' : 20,
         'num_threads'  : 1
         }
v = pyemu.geostats.SphVario(contribution=1.0, a=2*SL, anisotropy=1, bearing=0)
gs = pyemu.geostats.GeoStruct(variograms=v,nugget=0.0,transform = 'log')

#Add pilotpionts as parameters
pf.add_parameters('pomp.npf_k_layer2.txt', 'pilotpoints', transform= 'log', geostruct=gs,upper_bound = 1000, lower_bound = 0.001,
                pp_options = pp_opt, zone_array=np.array([zonearray.zone.values.astype(int)]))
pf.add_parameters('pomp.npf_k_layer1.txt', 'constant', transform= 'log', geostruct=gs,upper_bound = 1000, lower_bound = 0.001)
pf.add_parameters('pomp.npf_k_layer3.txt', 'constant', transform= 'log', geostruct=gs,upper_bound = 1000, lower_bound = 0.001)

for lay in [1,2,3]:
    pf.add_parameters(f'pomp.sto_ss_layer{lay}.txt', 'constant', transform= 'log', geostruct=gs,upper_bound = 1000, lower_bound = 0.001)

#Add all observations
for subdir, dirs, files in os.walk(OrgDir):
    for f in files:
        if f.startswith('head_0') or f.startswith('head_2'):
            pf.add_observations(f,f'{f}.ins',index_cols = 'time',ofile_sep= ',', use_cols='HEAD', obsgp = f)

#forward model run command
pf.mod_sys_cmds.append(f'{mfexe} {name}.nam')

#build pst file
pst = pf.build_pst(os.path.join(PestDir,'eg.pst'))

#Save observation data for synthetic 'real' data
pst.observation_data.to_csv(os.path.join('..','inter', 'Observation_data.csv'), index_label = 'index')
# os.chmod(os.path.join(PestDir,'forward_run.py'), 0o755)



#%%
# pst.control_data.noptmax = -1
# pst.control_data.facparmax = 1000
# pst.write(os.path.join(PestDir,'eg.pst'))
# print(f'Expected time for PESTPP-GLM: {0.885*len(ppdf)/8/60} h')
# pyemu.os_utils.start_workers(os.path.join(PestDir), # the folder which contains the "template" PEST dataset
#                             os.path.join(glm), #the PEST software version we want to run
#                             'eg.pst', # the control file to use with PEST
#                             num_workers=8, #how many agents to deploy
#                             worker_root='..', #where to deploy the agent directories; relative to where python is running
#                             master_dir= master_dir,
#                             verbose = True
#                             )


    # %%
