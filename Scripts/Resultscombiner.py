#%%
import pandas as pd
import os
import numpy as np
realdf = pd.read_csv(os.path.join('..', 'inter', 'Realizations.csv'), index_col = 'index')
folder = os.path.join('..', '1obs_for_optimum')
folders = os.listdir(folder)
simcorlen = []
simno = []
welldist = []
dirname = []
simref = []
angle = []
fitdf = pd.DataFrame()
RMSE = []
for subdir, dirs, files in os.walk(folder):
    for dir in dirs:
        if dir.startswith('Master'):
            print(dir)
            simcorlen.append(dir.split('_')[2])
            simno.append(dir.split('_')[3])
            simref.append(f'sim_{simcorlen[-1]}_{simno[-1]}')
            welldist.append(dir.split('_')[4])
            angle.append(dir.split('_')[5])
            dirname.append(dir)
            getk = pd.read_csv(os.path.join(folder,dir,'pomp.npf_k_layer2.txt'), sep = '   ', header= None).T
            RMSE.append(np.sqrt(np.mean((np.log10(realdf[simref[-1]].values) - np.log10(getk.values))**2)))

fitdf['corlen'] = simcorlen
fitdf['simno'] = simno
fitdf['simref'] = simref
fitdf['welldist'] = welldist
fitdf['angle'] = angle
fitdf['dirname'] = dirname
fitdf['RMSE'] = RMSE
fitdf.rename_axis('index', inplace = True)

BaselineRMSE = np.sqrt(np.mean((np.log10(realdf.loc[:,simref]) - np.log10(0.001042))**2))
# fitdf.to_csv(os.path.join(folder,'RMSE.csv'))
import matplotlib.pyplot as plt
import seaborn as sns
fig,ax = plt.subplots()
# sns.boxplot(fitdf,x = 'welldist', y = 'RMSE', ax = ax, zorder = 3, showmeans = True)
sns.scatterplot(fitdf,x = 'angle', y = 'RMSE',hue = 'welldist', ax = ax, zorder = 4)
ax.hlines(BaselineRMSE, 0,7)
# ax.set_xlabel('Distance observation well (r/λ)')
# ax.legend('').remove()

# %%
