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


#%% Plot calibrated fields

####################
corlen = 141
simno = 3
obsno = 5
angle = 0
welldist = 0.5
###################
fig, axs = plt.subplots(4,4, dpi = 600,constrained_layout =True, sharex = True, sharey = True)
fig.set_size_inches(7.5,7.5)
axs = axs.ravel()
crop = ds.where(ds.corlen == corlen,drop = True)
crop = crop.where(crop.simno == simno,drop = True)
crop = crop.where(crop.angle == angle,drop = True)
for obsno in range(2,17):
    ax = axs[obsno-2]
    sel = crop.where(crop.obsno == obsno,drop = True)

    if obsno == 1:
        sel = sel.where(ds.welldist == welldist, drop = True)
    vmin = min(min(np.log10(sel.RealK)))#,min(min(np.log10(sel.CalibratedK))))
    vmax = max(max(np.log10(sel.RealK)))#,max(max(np.log10(sel.CalibratedK))))
    #write pst file to temporary file to read with pyemu to create Pst object
    with tempfile.NamedTemporaryFile(delete=False, mode="w", suffix=".txt") as temp_file:
        temp_file.write(io.StringIO(sel.pst.values[0]).getvalue())
        pst = pyemu.Pst(temp_file.name)

    obs_data = pst.observation_data
    obs_data[['x','y']] = obs_data['obgnme'].str.split('_').str[-2:].apply(pd.Series)
    obs_data['x'] = obs_data['x'].astype(int)
    obs_data['y'] = obs_data['y'].astype(int)
    plot_obs = obs_data[obs_data.weight == 1]
    r = 500

    cal = flopy.plot.PlotMapView(gwf, extent = [-r, r, -r, r], ax = ax)
    cal.plot_array(np.log10(sel.CalibratedK),vmin = vmin, vmax = vmax)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])

    plot_obs.plot.scatter(x = 'x', y = 'y',ax =ax, s = 1, color= 'red', zorder = 3)
    # ppdf.plot.scatter(x = 'x', y = 'y',ax =ax, s = 1, color= 'white',zorder = 2)
    # fig.tight_layout()
pmv = flopy.plot.PlotMapView(gwf, extent = [-r, r, -r, r], ax = axs[-1])
pmv.plot_array(np.log10(sel.RealK),vmin = vmin, vmax = vmax)
axs[-1].set_aspect('equal')
axs[-1].set_xlabel('x')
fig.subplots_adjust(hspace = 0.1, wspace = 0.1)

# %% Plot Results
import seaborn as sns
fig, ax = plt.subplots(7,4, dpi = 600,constrained_layout = True, sharex = True, sharey = 'row')
fig.set_size_inches(8,12)

df = ds[['RMSE','MAE','R²','obsno','VarRatio', 'fitcorlen','CorLenError','sill','simno','corlen','angle']].to_dataframe().reset_index()
df = df[df.obsno != 1]
# df = df[df.simno.isin([0,1])]
sargs = {'s' : 3}
for i,metric in enumerate(['RMSE','MAE','R²','fitcorlen', 'CorLenError','sill','VarRatio']):
    for j,corlen in enumerate([70,141,282]):
        sns.scatterplot(df[df.corlen == corlen], x = 'obsno',y = metric,hue = 'simno',
                      s = 10,ax = ax[i,j], alpha = 0.3, edgecolor = 'black')
        # sns.regplot(df[df.corlen == corlen], x = 'obsno',y = metric,lowess=True, 
        #             ax = ax[i,j], scatter = False, truncate = True, line_kws={'linewidth' : 1, 'color' : 'red'})
        sns.lineplot(data = df[df.corlen == corlen], x = 'obsno',y = metric,errorbar=None, 
                     estimator=np.median, ax = ax[i,j], hue = 'simno')
        sns.lineplot(data = df[df.corlen == corlen], x = 'obsno',y = metric,errorbar=None, 
                estimator=np.mean, ax = ax[i,j], color = 'red')
        ax[i,j].legend().remove()
        if i == 0:
            ax[i,j].set_title(f'L = {corlen}')
        if metric == 'fitcorlen':
            ax[i,j].hlines(corlen, 1,df.obsno.max(), linestyle = '--',color = 'black')
        if metric == 'VarRatio':
            ax[i,j].set_yscale('log')
        if metric == 'CorLenError':
            ax[i,j].set_yscale('log')
    sns.scatterplot(df, x = 'obsno',y = metric,hue = 'corlen',
                ax = ax[i,3], s = 10,alpha = 0.3)
    sns.lineplot(data = df, x = 'obsno',y = metric,estimator=np.median, ax = ax[i,3])
    # sns.regplot(df, x = 'obsno',y = metric,lowess=True, scatter = False, 
    #             ax = ax[i,3], truncate = True, line_kws={'linewidth' : 1, 'color' : 'red'})
    ax[i,3].legend().remove()
ax[0,3].set_title('All')


# %%
df = ds[['RMSE','obsno','corlen','welldist']].to_dataframe().reset_index()
df = df[df.obsno == 1]
sns.boxplot(df, x = 'welldist', y = 'RMSE')
# %%
