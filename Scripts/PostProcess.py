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
simno = 2
obsno = 5
angle = 0
welldist = 0.5
###################

#%% Plot increasing number of obs
fig, axs = plt.subplots(4,4, dpi = 600, sharex = True, sharey = True)
fig.set_size_inches(7.5,8.5)
axs = axs.ravel()
crop = ds.where(ds.corlen == corlen,drop = True)
crop = crop.where(crop.simno == simno,drop = True)
crop = crop.where(crop.angle == angle,drop = True)
for obsno in range(2,17):
    ax = axs[obsno-2]
    sel = crop.where(crop.obsno == obsno,drop = True)


    if obsno == 1:
        sel = sel.where(ds.welldist == welldist, drop = True)
    vmin = np.log10(sel.RealK).min().values#,min(min(np.log10(sel.CalibratedK))))
    vmax = np.log10(sel.RealK).max().values#,max(max(np.log10(sel.CalibratedK))))
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
    cb = cal.plot_array(np.log10(sel.CalibratedK),vmin = vmin, vmax = vmax)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])

    plot_obs.plot.scatter(x = 'x', y = 'y',ax =ax, s = 1, color= 'red', zorder = 3, label = 'Observation well')
    ax.text(.015, .985, obsno, ha='left', va='top', transform=ax.transAxes)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend().set_visible(False)
    
    # ppdf.plot.scatter(x = 'x', y = 'y',ax =ax, s = 1, color= 'white',zorder = 2)
    # fig.tight_layout()

pmv = flopy.plot.PlotMapView(gwf, extent = [-r, r, -r, r], ax = axs[-1])
pmv.plot_array(np.log10(sel.RealK),vmin = vmin, vmax = vmax)
axs[-1].set_aspect('equal')
axs[-1].set_xlabel('x')
axs[-1].text(.015, .985, 'Ref', ha='left', va='top', transform=axs[-1].transAxes)

#Legend and colorbar
fig.legend(handles,labels, bbox_to_anchor = [0.05, 0.04, 0.2, 0.02], frameon = True, framealpha = 1 )
cbar_ax = fig.add_axes([0.27, 0.03, 0.58, 0.02]) 
fig.subplots_adjust(left = 0, bottom = 0.1, hspace = 0.1, wspace = 0.1)
fig.colorbar(cb, cax = cbar_ax, orientation = 'horizontal', label = '$\log_{10} K$')
fig.savefig(os.path.join('..','Images','3. Kfields.pdf'), bbox_inches = 'tight')

#%% Plot rotation angles 
fig, axs = plt.subplots(4,4, dpi = 600, sharex = True, sharey = True)
fig.set_size_inches(7.5,8.5)
axs = axs.ravel()
crop = ds.where(ds.corlen == corlen,drop = True)
crop = crop.where(crop.simno == simno,drop = True)

n = 0
for obsno in [8, 16]:
    for angle in range(8):
    
        ax = axs[n]
        obscrop = crop.where(crop.obsno == obsno,drop = True)
        sel = obscrop.where(crop.angle == angle,drop = True)


        if obsno == 1:
            sel = sel.where(ds.welldist == welldist, drop = True)
        vmin = np.log10(sel.RealK).min().values#,min(min(np.log10(sel.CalibratedK))))
        vmax = np.log10(sel.RealK).max().values#,max(max(np.log10(sel.CalibratedK))))
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
        cb = cal.plot_array(np.log10(sel.CalibratedK),vmin = vmin, vmax = vmax)
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])
        
        
        plot_obs.plot.scatter(x = 'x', y = 'y',ax =ax, s = 1, color= 'red', zorder = 3, label = 'Observation well')
        handles, labels = ax.get_legend_handles_labels()
        ax.legend().set_visible(False)
        n += 1
fig.subplots_adjust(left = 0, bottom = 0.1, hspace = 0.1, wspace = 0.1)
axs[0].text(.015, .985, 8, ha='left', va='top', transform=axs[0].transAxes)
axs[8].text(.015, .985, 16, ha='left', va='top', transform=axs[8].transAxes)
fig.legend(handles,labels, bbox_to_anchor = [0.05, 0.04, 0.2, 0.02], frameon = True, framealpha = 1)
cbar_ax = fig.add_axes([0.27, 0.03, 0.58, 0.02]) 
# fig.subplots_adjust(left = 0, bottom = 0.2, hspace = 0.1, wspace = 0.1)
fig.colorbar(cb, cax = cbar_ax, orientation = 'horizontal', label = '$\log_{10} K$')

fig.savefig(os.path.join('..','Images','4. Rotations.pdf'), bbox_inches = 'tight')











#%% Plot correlation lengths
fig, axs = plt.subplots(3,4, dpi = 600, sharex = True, sharey = True)
fig.set_size_inches(7.5,6.5)
axs = axs.ravel()
crop = ds.where(ds.angle == 0,drop = True)
crop = crop.where(crop.simno == 2,drop = True)
vmin = np.log10(crop.RealK).min().values#,min(min(np.log10(sel.CalibratedK))))
vmax = np.log10(crop.RealK).max().values#,max(max(np.log10(sel.CalibratedK))))
n = 0
for corlen in [70,141,282]:
    for obsno in [4, 8,16]:
    
        ax = axs[n]
        obscrop = crop.where(crop.obsno == obsno,drop = True)
        sel = obscrop.where(obscrop.corlen == corlen,drop = True)


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
        r = 500

        cal = flopy.plot.PlotMapView(gwf, extent = [-r, r, -r, r], ax = ax)
        cb = cal.plot_array(np.log10(sel.CalibratedK),vmin = vmin, vmax = vmax)
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])

        plot_obs.plot.scatter(x = 'x', y = 'y',ax =ax, s = 1, color= 'red', zorder = 3, label = 'Observation well')
        handles, labels = ax.get_legend_handles_labels()
        ax.legend().set_visible(False)
        n += 1
    ax = axs[n]
    cal = flopy.plot.PlotMapView(gwf, extent = [-r, r, -r, r], ax = ax)
    cal.plot_array(np.log10(sel.RealK),vmin = vmin, vmax = vmax)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    n+=1

fig.subplots_adjust(left = 0, bottom = 0.1, hspace = 0.1, wspace = 0.1)
axs[0].text(-.30, .5, 'CL = 0.5λ', ha='left', va='center', rotation = 'vertical', transform=axs[0].transAxes)
axs[4].text(-.30, .5, 'CL = 1λ', ha='left', va='center', rotation = 'vertical',transform=axs[4].transAxes)
axs[8].text(-.30, .5,'CL = 2λ', ha='left', va='center', rotation = 'vertical',transform=axs[8].transAxes)
axs[-1].set_xlabel('x')
axs[0].set_title('4 wells')
axs[1].set_title('8 wells')
axs[2].set_title('16 wells')
axs[3].set_title('Ref')

fig.legend(handles,labels, bbox_to_anchor = [0.05, 0.04, 0.2, 0.02], frameon = True, framealpha = 1 )
cbar_ax = fig.add_axes([0.27, 0.03, 0.58, 0.02]) 
# fig.subplots_adjust(left = 0, bottom = 0.2, hspace = 0.1, wspace = 0.1)
fig.colorbar(cb, cax = cbar_ax, orientation = 'horizontal', label = '$\log_{10} K$')

fig.savefig(os.path.join('..','Images','5. Kfields.pdf'), bbox_inches = 'tight')






# %% Plot Results
import seaborn as sns
fig, ax = plt.subplots(6,4, dpi = 600, sharex = True, sharey = 'row')
fig.set_size_inches(7.5,9)

df = ds[['RMSE','MAE','R²','obsno','fixed','VarRatio', 'fitcorlen','CorLenError','sill','simno','corlen','angle']].to_dataframe().reset_index()
df = df[df.obsno != 1]
# df = df[df.simno.isin([0,1])]
sargs = {'s' : 3}
for i,metric in enumerate(['RMSE','MAE', 'R²','VarRatio','fitcorlen', 'sill']):
    for j,corlen in enumerate([70,141,282]):
        sns.scatterplot(df[df.corlen == corlen], x = 'obsno',y = metric,hue = 'fixed',
                      s = 6,ax = ax[i,j], alpha = 0.3)
        # sns.regplot(df[df.corlen == corlen], x = 'obsno',y = metric,lowess=True, 
        #             ax = ax[i,j], scatter = False, truncate = True, line_kws={'linewidth' : 1, 'color' : 'red'})
        sns.lineplot(data = df[df.corlen == corlen], x = 'obsno',y = metric,errorbar=None, 
                     estimator=np.mean, ax = ax[i,j], hue = 'fixed')
        # sns.lineplot(data = df[df.corlen == corlen], x = 'obsno',y = metric,errorbar=None, 
        #         estimator=np.mean, ax = ax[i,j], color = 'red')
        ax[i,j].legend().remove()
        if i == 0:
            ax[i,j].set_title(f'L = {corlen}')
        if metric == 'RMSE':
            ax[i,j].set_title(f'CL = {round(corlen/141,1)}λ')
        if metric == 'fitcorlen':
            ax[i,j].hlines(corlen, 1,df.obsno.max(), linestyle = '--',color = 'black')
            ax[i,0].set_ylabel('CL')
        if metric == 'VarRatio':
            ax[i,j].set_yscale('log')
            ax[i,0].set_ylabel('VR')
        if metric == 'sill':
            # ax[i,j].set_yscale('log')
            # ax[i,j].set_ylim(0.01, 1)
            ax[i,0].set_ylabel('Sill')
        ax[i,j].set_xlabel('# of observations')
    handles, labels = ax[i,0].get_legend_handles_labels()
    sns.scatterplot(df, x = 'obsno',y = metric,hue = 'fixed',
                ax = ax[i,3], s = 4,alpha = 0.3)
    sns.lineplot(data = df, x = 'obsno',y = metric,estimator=np.mean, ax = ax[i,3], hue = 'fixed')
    # sns.regplot(df, x = 'obsno',y = metric,lowess=True, scatter = False, 
    #             ax = ax[i,3], truncate = True, line_kws={'linewidth' : 1, 'color' : 'red'})
    ax[-1,-1].set_xlabel('# observations')
    ax[i,3].legend().remove()
ax[0,3].set_title('All')
labels = ['Realizations - No conditioning',
          'Realizations - Conditioned on boreholes',
          'Mean - No conditioning',
          'Mean - Conditioned on boreholes']
fig.subplots_adjust(left = 0, bottom = 0.12, hspace = 0.1, wspace = 0.1)
fig.legend(handles,labels, loc = 'lower center', 
           #bbox_to_anchor = [0.05, 0.04, 0.2, 0.02],
            frameon = True, framealpha = 1, ncol = 2)
fig.savefig(os.path.join('..','Images','6. Metrics.pdf'), bbox_inches = 'tight')

#%% Check improvement from calibration
mean_rmse_fixed = df[df.fixed == True].groupby('obsno')['RMSE'].mean()
mean_rmse_nofix = df[df.fixed == False].groupby('obsno')['RMSE'].mean()

diff = mean_rmse_nofix - mean_rmse_fixed
diff.diff().plot()


# %%
df = ds[['RMSE','obsno','corlen','welldist','fixed']].to_dataframe().reset_index()
df = df[df.obsno == 1]
sns.boxplot(df, x = 'welldist', y = 'RMSE')
# %%
# %% Plot Results
import seaborn as sns
fig, ax = plt.subplots(6,3, dpi = 600, sharex = True, sharey = 'row')
fig.set_size_inches(7.5,9)

df = ds[['RMSE','MAE','R²','obsno','fixed','VarRatio', 'fitcorlen','CorLenError','sill','simno','corlen','angle','simref']].to_dataframe().reset_index()
df = df[df.obsno != 1]
df = df[df.fixed == True]
df = df[df.simno == 0]
# df = df[df.simno.isin([0,1])]
sargs = {'s' : 3}
for i,metric in enumerate(['RMSE','MAE', 'R²','VarRatio','fitcorlen', 'sill']):
    for j,corlen in enumerate([70,141,282]):
        sns.lineplot(df[df.corlen == corlen], x = 'obsno',y = metric,hue = 'angle',
                      ax = ax[i,j], alpha =0.5)

        ax[i,j].legend().remove()
        if i == 0:
            ax[i,j].set_title(f'L = {corlen}λ')
        if metric == 'RMSE':
            ax[i,j].set_title(f'CL = {round(corlen/141,1)}λ')
        if metric == 'fitcorlen':
            ax[i,j].hlines(corlen, 1,df.obsno.max(), linestyle = '--',color = 'black')
            ax[i,0].set_ylabel('CL')
        if metric == 'VarRatio':
            ax[i,j].set_yscale('log')
            ax[i,0].set_ylabel('VR')
        if metric == 'sill':
            # ax[i,j].set_yscale('log')
            # ax[i,j].set_ylim(0.01, 1)
            ax[i,0].set_ylabel('Sill')
        ax[i,j].set_xlabel('# of observations')
    handles, labels = ax[i,0].get_legend_handles_labels()
    ax[i,j].legend().remove()
fig.savefig(os.path.join('..','Images','7. Metrics.pdf'), bbox_inches = 'tight')
# labels = ['Realizations - No conditioning',
#           'Realizations - Conditioned on boreholes',
#           'Mean - No conditioning',
#           'Mean - Conditioned on boreholes']
# fig.subplots_adjust(left = 0, bottom = 0.12, hspace = 0.1, wspace = 0.1)
# fig.legend(handles,labels, loc = 'lower center', 
#            #bbox_to_anchor = [0.05, 0.04, 0.2, 0.02],
#             frameon = True, framealpha = 1, ncol = 2)