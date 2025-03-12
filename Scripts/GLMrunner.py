#%%
import pyemu
import os
import pandas as pd
import numpy as np
from pest_funcs import rotate_point
import shutil
PestDir = os.path.join('..','pest_files')
pst = pyemu.Pst(os.path.join(PestDir, 'eg.pst'))

obs_real_data = pd.read_csv(os.path.join('..', 'inter', 'Observed_heads.csv'), index_col = 'index')
realdf = pd.read_csv(os.path.join('..', 'inter', 'Realizations.csv'), index_col = 'index')
realnames = realdf.columns[3:]
glm = os.path.join('..','exe','pestpp-glm')

SL = np.sqrt(20000)
runstring = '2obs'
if not os.path.isdir(os.path.join('..', runstring)): 
    os.mkdir(os.path.join('..', runstring))
for sim in realnames:
        name = f'{runstring}'
        useobs = [(0,int(SL)),(0,-int(SL)), (0,0)]

        #assign 'real' observations
        useobs_str = [f'_{x}_{y}' for x, y in useobs] # make string from observation locs
        #Change weights of observations to 0 except useobs
        pst.observation_data['weight'] = pst.observation_data['obgnme'].apply(lambda x: 1 if any(x.endswith(s) for s in useobs_str) else 0)
        pst.observation_data.loc[pst.observation_data['obgnme'].str.endswith('2_0_0'), 'weight'] = 0


        pst.observation_data['obsval'] = obs_real_data[sim].values

        master_dir = os.path.join('..', f'Master_{sim}_{name}')


        pst.control_data.noptmax = 7
        pst.pestpp_options['n_iter_super'] = 10
        pst.pestpp_options['n_iter_base'] = -1
        pst.pestpp_options['uncertainty'] = 0
        pst.svd_data.maxsing = pst.npar_adj
        os.chmod(os.path.join(PestDir,'forward_run.py'), 0o755)
        pst.write(os.path.join(PestDir,'eg.pst'))
        pyemu.os_utils.start_workers(os.path.join(PestDir), # the folder which contains the "template" PEST dataset
                                    os.path.join(glm), #the PEST software version we want to run
                                    'eg.pst', # the control file to use with PEST
                                    num_workers=8, #how many agents to deploy
                                    worker_root='..', #where to deploy the agent directories; relative to where python is running
                                    master_dir= master_dir,
                                    verbose = True
                                    )

        pst.parrep(parfile=os.path.join(master_dir, 'eg.par'))
        pst.write_input_files(pst_path=master_dir)
        pyemu.os_utils.run('python forward_run.py', cwd=master_dir)
        shutil.move(master_dir, os.path.join('..',runstring,os.path.basename(master_dir)))

# %%
