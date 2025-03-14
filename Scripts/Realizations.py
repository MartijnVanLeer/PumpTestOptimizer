#%%
import gstools as gs
from gstools.random import MasterRNG
import pandas as pd
import flopy
import os
import numpy as np

#read xy_coordinates
zonedf = pd.read_csv(os.path.join('..', 'inter', 'ZoneCoords.csv'), index_col = 'index')
sim = flopy.mf6.MFSimulation.load(sim_ws=os.path.join('..', 'ws'), lazy_io=True, verbosity_level=0)
gwf = sim.get_model('pomp')

## Generate random conductivity field(s) within zone
n = 10
len_scales = [70,141,282]
mean = 10/400/24
logvar = 1
normalizer = gs.normalizer.LogNormal()
for ls in len_scales:
    model = gs.Gaussian(dim=2, var=logvar, len_scale=ls)
    srf = gs.SRF(model, seed=4524519, mean = normalizer.normalize(10/400/24)-logvar/2, normalizer = normalizer)
    seed = MasterRNG(20170519)
    for i in range(n):
        field = srf((zonedf.x.values, zonedf.y.values), seed = seed())
        zonedf[f'sim_{ls}_{i}'] = np.where(zonedf.zone == True, field, 10/400/24)
zonedf.to_csv(os.path.join('..', 'inter', 'Realizations.csv'), index_label = 'index')

