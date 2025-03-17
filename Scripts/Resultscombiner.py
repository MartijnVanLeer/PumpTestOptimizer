#%%
import pandas as pd
import os
import numpy as np
import pyemu
import dill as pickle
realdf = pd.read_csv(os.path.join('..', 'inter', 'Realizations.csv'), index_col = 'index')
resultsdir = os.path.join('..', 'Results')
runs = os.listdir(resultsdir)

simcorlen = []
simno = []
welldist = []
dirname = []
simref = []
angle = []
obsno = []
RMSE = []
kfields = []
RealK = []
pestfiles = []
for folder in runs:
        for subdir, dirs, files in os.walk(os.path.join(resultsdir, folder)):
            for dir in dirs:
                if dir.startswith('Master'):
                    simcorlen.append(dir.split('_')[2])
                    simno.append(dir.split('_')[3])
                    simref.append(f'sim_{simcorlen[-1]}_{simno[-1]}')
                    obsno.append(folder[0])
                    wd =dir.split('_')[4]
                    wd = None if wd == '' else wd
                    welldist.append(wd)
                    angle.append(dir.split('_')[5])
                    dirname.append(dir)
                    getk = pd.read_csv(os.path.join(resultsdir, folder,dir,'pomp.npf_k_layer2.txt'), sep = '   ', header= None).T
                    kfields.append(getk.values)
                    pestfile = pyemu.Pst(os.path.join(resultsdir, folder,dir,'eg.pst'))
                    pestfiles.append(pestfile)
                    RealK.append(realdf[simref[-1]].values)
                    RMSE.append(np.sqrt(np.mean((np.log10(realdf[simref[-1]].values) - np.log10(getk.values))**2)))

fitdf = pd.DataFrame()
fitdf['corlen'] = np.array(simcorlen, dtype = 'int')
fitdf['simno'] = np.array(simno, dtype = 'int')
fitdf['simref'] = np.array(simref, dtype = 'str')
fitdf['angle'] = np.array(angle,dtype = 'int')
fitdf['dirname'] = np.array(dirname, dtype = 'str')
fitdf['welldist'] = np.array(welldist,dtype= float)
fitdf['pst'] = [pickle.dumps(pf) for pf in pestfiles]
fitdf['RMSE'] = RMSE
fitdf.rename_axis('index', inplace = True)
ds = fitdf.to_xarray()
ds = ds.assign_coords(cellid = range(len(getk.values)))
kfieldsfix = np.array(kfields)[:,:,0]
realkfix = np.array(RealK)
ds['CalibratedK'] = (['index', 'cellid'], kfieldsfix)
ds['RealK'] = (['index', 'cellid'], realkfix)
ds.attrs['Runs'] = runs
ds.to_netcdf(os.path.join('..','Results', 'Results.nc'))

#%%
