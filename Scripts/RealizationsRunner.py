#%%
import pandas as pd 
import flopy 
import os
import shutil
from tqdm import tqdm
import numpy as np

#copy folder 
rundir =  os.path.join('..', 'runner_ws')
shutil.copytree(os.path.join('..', 'ws'), rundir, dirs_exist_ok= True)

#load MF from new folder
sim = flopy.mf6.MFSimulation.load(sim_ws=rundir, lazy_io=True,verbosity_level=0, exe_name=os.path.join('..', 'exe','mf6'))
gwf = sim.get_model('pomp')

#get realizations
realdf = pd.read_csv(os.path.join('..', 'inter', 'Realizations.csv'), index_col = 'index')
obs_data = pd.DataFrame()
obsnames = gwf.obs.continuous.data.keys()

for col in tqdm(realdf.columns[3:]):
    realdf[col].to_csv(os.path.join(rundir, 'pomp.npf_k_layer2.txt'), header = False, index = False)
    success, buff = sim.run_simulation(report=True)
    assert success
    templs = np.empty(50 * len(obsnames))
    row = 50
    for obs in obsnames:
        drawdown = pd.read_csv(os.path.join(rundir, obs)).HEAD
        templs[(row-50):row] = drawdown
        row += 50
    obs_data[col] = templs

obs_data.to_csv(os.path.join('..', 'inter', 'Observed_heads.csv'), index_label = 'index')

# %%
