#%%
import matplotlib.pyplot as plt
from ConceptProfile import *
import os
plt.rcParams.update({'font.size': 8})
fig,ax = plt.subplots(2)
fig.set_size_inches(8.25/2.54, 4)
ax[0] = profile(ax[0], True)
ax[0] = add_text(ax[0], True)
ax[1] = profile(ax[1], False)
ax[1] = add_text(ax[1], False)
for a in ax:
    a.set_xticks([0,200], [0,1])
    a.set_xlabel('$x/λ$ ')

fig.tight_layout()
fig.savefig(os.path.join('..','..', 'Images', '1. Profiles.pdf'))