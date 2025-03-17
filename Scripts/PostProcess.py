#%%
import pyemu
import os
import flopy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
import io
import tempfile
#load bunch of stuff
ds = xr.open_dataset(os.path.join('..', 'Results', 'Results.nc'))
realdf = pd.read_csv(os.path.join('..', 'inter', 'Realizations.csv'), index_col = 'index')
ppdf =pd.read_csv(os.path.join('..', 'inter', 'ppdf.csv'), index_col = 'index')
mfsim = flopy.mf6.MFSimulation.load(sim_ws=os.path.join('..', 'ws'), verbosity_level=0, lazy_io=True)
gwf = mfsim.get_model()

#%%
####################
corlen = 141
simno = 0
obsno = 2
angle = 0
welldist = 0.5
###################

sel = ds.where(ds.corlen == corlen,drop = True)
sel = sel.where(ds.simno == simno,drop = True)
sel = sel.where(ds.obsno == obsno,drop = True)
sel = sel.where(ds.angle == angle,drop = True)

if obsno == 1:
    sel = sel.where(ds.welldist == welldist, drop = True)

#write pst file to temporary file to read with pyemu to create Pst object
with tempfile.NamedTemporaryFile(delete=False, mode="w", suffix=".txt") as temp_file:
    temp_file.write(io.StringIO(sel.pst.values[0]).getvalue())
    pst = pyemu.Pst(temp_file.name)

obs_data = pst.observation_data


obs_data[['x','y']] = obs_data['obgnme'].str.split('_').str[-2:].apply(pd.Series)
obs_data['x'] = obs_data['x'].astype(int)
obs_data['y'] = obs_data['y'].astype(int)
plot_obs = obs_data[obs_data.weight == 1]
fig, ax = plt.subplots(2,3, dpi = 600)
fig.set_size_inches(10,7)
diff = np.log10(sel.RealK) - np.log10(sel.CalibratedK)
vmin = min(min(min(np.log10(sel.RealK))),min(min(np.log10(sel.CalibratedK))))
vmax = max(max(max(np.log10(sel.RealK))),max(max(np.log10(sel.CalibratedK))))
for no, r in enumerate([500, 250]):

    cal = flopy.plot.PlotMapView(gwf, extent = [-r, r, -r, r], ax = ax[no,0])
    cal.plot_array(np.log10(sel.CalibratedK),vmin = vmin, vmax = vmax)
    ax[no,0].set_title('Calibrated')
    pmv = flopy.plot.PlotMapView(gwf, extent = [-r, r, -r, r], ax = ax[no,1])
    pmv.plot_array(np.log10(sel.RealK),vmin = vmin, vmax = vmax)
    ax[no,1].set_title('Real')
    dif = flopy.plot.PlotMapView(gwf, extent = [-r, r, -r, r], ax = ax[no,2])

    dif.plot_array(abs(diff))
    ax[no,2].set_title('Difference')

for a in ax.flatten():
    a.set_aspect('equal')
    plot_obs.plot.scatter(x = 'x', y = 'y',ax =a, s = 1, color= 'red', zorder = 3)
    ppdf.plot.scatter(x = 'x', y = 'y',ax =a, s = 1, color= 'white',zorder = 2)
fig.tight_layout()


# %%
