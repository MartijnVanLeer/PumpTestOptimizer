#%%
import gstools as gs
from gstools.random import MasterRNG
import pandas as pd
import flopy
import os
import numpy as np
from tqdm import tqdm
import shutil

#read xy_coordinates
zonedf = pd.read_csv(os.path.join('..', 'inter', 'ZoneCoords.csv'), index_col = 'index')
sim = flopy.mf6.MFSimulation.load(sim_ws=os.path.join('..', 'ws'), lazy_io=True, verbosity_level=0)
gwf = sim.get_model('pomp')
aq_reals = zonedf.copy()
## Generate random conductivity field(s) within zone
n = 10
len_scales = [70,141,282]
mean = 10/400/24
logvar = 1
normalizer = gs.normalizer.LogNormal()

#aquitard realizations
for ls in len_scales:
    model = gs.Gaussian(dim=2, var=logvar, len_scale=ls)
    srf = gs.SRF(model, seed=4524519, mean = normalizer.normalize(mean)-logvar/2, normalizer = normalizer)
    seed = MasterRNG(20170519)
    for i in range(n):
        field = srf((zonedf.x.values, zonedf.y.values), seed = seed())
        zonedf[f'sim_{ls}_{i}'] = np.where(zonedf.zone == True, field, mean)
zonedf.to_csv(os.path.join('..', 'inter', 'Realizations.csv'), index_label = 'index')

#aquifer realizations
ls_aq = 141
mean = 5/24
logvar = 0.5
model = gs.Gaussian(dim=2, var=logvar, len_scale=ls_aq)
srf = gs.SRF(model, seed=4524519, mean = normalizer.normalize(mean)-logvar/2, normalizer = normalizer)
seed = MasterRNG(20170519)
for ls in len_scales:
    for i in range(n):
        for j in [1,3]:
            field = srf((zonedf.x.values, zonedf.y.values), seed = seed())
            aq_reals[f'sim_{ls}_{i}_{j}'] = np.where(zonedf.zone == True, field, mean)


#get obs
rundir =  os.path.join('..', 'runner_ws')
shutil.copytree(os.path.join('..', 'ws'), rundir, dirs_exist_ok= True)

#load MF from new folder
sim = flopy.mf6.MFSimulation.load(sim_ws=rundir, lazy_io=True,verbosity_level=0, exe_name=os.path.join('..', 'exe','mf6'))
gwf = sim.get_model('pomp')

#Names of obs in MF6 model
obsnames = gwf.obs.continuous.data.keys()

obs_data = pd.DataFrame()

#Run model with realization and assign to df
for col in tqdm(zonedf.columns[3:]):
    zonedf[col].to_csv(os.path.join(rundir, 'pomp.npf_k_layer2.txt'), header = False, index = False)
    for j in [1,3]:
        aq_reals[f'{col}_{j}'].to_csv(os.path.join(rundir, f'pomp.npf_k_layer{j}.txt'), header = False, index = False)
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
