#%%
import pyemu
import os
import flopy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

realdf = pd.read_csv(os.path.join('..', 'inter', 'Realizations.csv'), index_col = 'index')
obs_data = pd.read_csv(os.path.join('..', 'inter', 'Observation_data.csv'), index_col = 'index')
ppdf =pd.read_csv(os.path.join('..', 'inter', 'ppdf.csv'), index_col = 'index')
realnames = realdf.columns[3:]

sim= 0
master_dir = os.path.join('..', f'Master_{realnames[sim]}_0.5SL_1obs')

pst = pyemu.Pst(os.path.join(master_dir,'eg.pst'))
df_obj = pd.read_csv(os.path.join(master_dir, "eg.iobj"),index_col=0)
obs_data = pst.observation_data

mfsim = flopy.mf6.MFSimulation.load(sim_ws=master_dir, verbosity_level=0, lazy_io=True)
# load flow model
gwf = mfsim.get_model()

#%%
obs_data[['x','y']] = obs_data['obgnme'].str.split('_').str[-2:].apply(pd.Series)
obs_data['x'] = obs_data['x'].astype(int)
obs_data['y'] = obs_data['y'].astype(int)
plot_obs = obs_data[obs_data.weight == 1]
fig, ax = plt.subplots(2,3, dpi = 600)
fig.set_size_inches(10,7)
diff = np.log10(realdf[realnames[sim]]) - np.log10(gwf.np.k.data[1])
vmin = min(min(np.log10(realdf[realnames[sim]])),min(np.log10(gwf.np.k.data[1])))
vmax = max(max(np.log10(realdf[realnames[sim]])),max(np.log10(gwf.np.k.data[1])))
for no, r in enumerate([500, 250]):

    cal = flopy.plot.PlotMapView(gwf, extent = [-r, r, -r, r], ax = ax[no,0])
    cal.plot_array(np.log10(gwf.np.k.data[1]),vmin = vmin, vmax = vmax)
    ax[no,0].set_title('Calibrated')
    pmv = flopy.plot.PlotMapView(gwf, extent = [-r, r, -r, r], ax = ax[no,1])
    pmv.plot_array(np.log10(realdf[realnames[sim]]),vmin = vmin, vmax = vmax)
    ax[no,1].set_title('Real')
    dif = flopy.plot.PlotMapView(gwf, extent = [-r, r, -r, r], ax = ax[no,2])

    dif.plot_array(abs(diff))
    ax[no,2].set_title('Difference')

for a in ax.flatten():
    a.set_aspect('equal')
    plot_obs.plot.scatter(x = 'x', y = 'y',ax =a, s = 1, color= 'red', zorder = 3)
    ppdf.plot.scatter(x = 'x', y = 'y',ax =a, s = 1, color= 'white',zorder = 2)
fig.tight_layout()

#%%
end = df_obj[df_obj.index == 20].T
end = end[4:]
end.reset_index(inplace =True)
end[['layer','x','y']] = end['index'].str.split('_').str[-3:].apply(pd.Series)
end['x'] = end['x'].astype(int)
end['y'] = end['y'].astype(int)
end['layer'] = end['layer'].astype(int)
end0 = end[end.layer == 0]
end1 = end[end.layer == 2]
plt.scatter(x = end0.x, y = end0.y, c = end0[20])

