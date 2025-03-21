#%%
import pandas as pd
import os
import numpy as np
from functions import ErrorMetrics
from tqdm import tqdm
#Get 'real' realizations
realdf = pd.read_csv(os.path.join('..', 'inter', 'Realizations.csv'), index_col = 'index')
maskeddf = realdf[realdf.zone]
#readfolder
resultsdir = os.path.join('..', 'Results')
runs = os.listdir(resultsdir)

#empty lists
simcorlen = []
simno = []
welldist = []
dirname = []
simref = []
angle = []
obsno = []
RMSE = []
NRMSE = []
MAE = []
r2 = []
VarRatio = []
Fitted_corlen = []
sills = []
kfields = []
RealK = []
pestfiles = []
CorLenError = []

#read pest run folders and append to lists
for folder in tqdm(runs, 'Reading files..'):
        for subdir, dirs, files in os.walk(os.path.join(resultsdir, folder)):
            for dir in dirs:
                if dir.startswith('Master'):
                    simcorlen.append(dir.split('_')[2])
                    simno.append(dir.split('_')[3])
                    simref.append(f'sim_{simcorlen[-1]}_{simno[-1]}')
                    obsno.append(folder.split('o')[0])
                    wd =dir.split('_')[4]
                    wd = None if wd == '' else wd #only 1obs has this value
                    welldist.append(wd)
                    angle.append(dir.split('_')[5])
                    dirname.append(dir)
                    #read calibrated K
                    getk = pd.read_csv(os.path.join(resultsdir, folder,dir,'pomp.npf_k_layer2.txt'), sep = '   ', header= None, engine = 'python').T
                    getk = getk[0].values
                    kfields.append(getk)
                    #read pest files for all runinfo
                    pestfile = (os.path.join(resultsdir, folder,dir,'eg.pst'))
                    with open(pestfile, 'r') as file:
                        pestfiles.append(file.read())
                    #assign only modelled RealK to list
                    RealK.append(realdf[simref[-1]].values)
                    #calc RMSE of Kreal vs Kcal
                    RMSE.append(np.sqrt(np.mean((np.log10(realdf[simref[-1]].values) - np.log10(getk))**2)))
                    NRMSE.append(RMSE[-1]/np.std(np.log10(getk)))
                    MAE.append(np.mean(abs(np.log10(realdf[simref[-1]].values) - np.log10(getk))))
                    r2.append(ErrorMetrics.calculate_r2(np.log10(realdf[simref[-1]].values), np.log10(getk)))
                    VarRatio.append(np.var(np.log10(realdf[simref[-1]].values))/np.var(np.log10(getk)))
                    #fit variogram
                    sill,fitcorlen = ErrorMetrics.fit_gaussian_variogram(maskeddf.x.values, maskeddf.y.values, np.squeeze(np.log10(getk[realdf.zone.values])),num_bins = 30)
                    sills.append(sill)
                    Fitted_corlen.append(fitcorlen)
                    CorLenError.append(abs(int(simcorlen[-1])-fitcorlen)/int(simcorlen[-1]))

print('constructing netcdf..')
#assign lists to df
fitdf = pd.DataFrame()
fitdf['corlen'] = np.array(simcorlen, dtype = 'int')
fitdf['simno'] = np.array(simno, dtype = 'int')
fitdf['simref'] = np.array(simref, dtype = 'str')
fitdf['angle'] = np.array(angle,dtype = 'int')
fitdf['obsno'] = np.array(obsno, dtype = 'int')
fitdf['dirname'] = np.array(dirname, dtype = 'str')
fitdf['welldist'] = np.array(welldist,dtype= float)
fitdf['pst'] = np.array(pestfiles, dtype = 'str') #save Pst objects as pickled
fitdf['RMSE'] = RMSE
fitdf['NRMSE'] = NRMSE
fitdf['MAE'] = MAE
fitdf['R²'] = r2
fitdf['VarRatio'] =VarRatio
fitdf['fitcorlen'] = Fitted_corlen
fitdf['CorLenError'] = CorLenError
fitdf['sill'] = sills
fitdf.rename_axis('index', inplace = True)
#construct xarray
ds = fitdf.to_xarray()
ds = ds.assign_coords(cellid = range(len(getk))) #cellid for realizations
kfieldsfix = np.array(kfields)
realkfix = np.array(RealK)
ds['CalibratedK'] = (['index', 'cellid'], kfieldsfix)
ds['RealK'] = (['index', 'cellid'], realkfix)
ds.attrs['Runs'] = runs
ds.to_netcdf(os.path.join('..','Results', 'Results.nc'))

#%%
