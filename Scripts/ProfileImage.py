#%%
import matplotlib.pyplot as plt
from functions.ConceptProfile import *

import os
plt.rcParams.update({'font.size': 8})
fig,ax = plt.subplots(dpi = 600)
fig.set_size_inches(8.25/2.54, 2.5)

ax = profile(ax, False)
ax = add_text(ax, False)
ax.set_xticks([0,200], [0,1])
ax.set_xlabel('$x/λ$ [-]')

fig.tight_layout()
fig.savefig(os.path.join('..', 'Images', '1. Profile.pdf'))