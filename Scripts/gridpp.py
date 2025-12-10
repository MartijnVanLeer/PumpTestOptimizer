
import flopy 
import pandas as pd
import os
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

sim = flopy.mf6.MFSimulation.load(sim_ws=os.path.join('..','ws' ))

gwf = sim.get_model('pomp')
ppdf = pd.read_csv(os.path.join('..', 'pest_files','p_inst0pp.dat'),sep=" ", names = ['name', 'x', 'y', 'z','value'])
obsdf = pd.read_csv(os.path.join('..', 'inter', 'ObsCoords.csv'), index_col = 'index')

fig, ax = plt.subplots(constrained_layout=True, dpi = 600)
fig.set_size_inches(8.25*0.393701, 8.25*0.393701)

pmva = flopy.plot.PlotMapView(gwf, ax = ax)
axins = inset_axes(ax, width="30%", height="30%",bbox_to_anchor=(0, 0, 1, 1),
                      bbox_transform=ax.transAxes,
                      loc="lower left")
pmvb = flopy.plot.PlotMapView(gwf, ax = axins)

pmva.plot_grid(linewidth=  0.2)
pmvb.plot_grid(linewidth=  0.2)
pp = ppdf.plot.scatter(x = 'x', y = 'y', ax = ax,s = 0.5, zorder = 5, label = 'Pilot point', color = 'black')
ax.scatter(0, 0, color = 'red', s = 12, label = 'Pumping well', zorder = 4)
obsdf.plot.scatter(x = 'x', y = 'y',ax = ax, zorder = 4, label = 'Observation well',color = 'lime', s = 8,ec = 'black', linewidth = 0.2)
# ax.scatter(0, 200, color = 'lime', s = 2, label = 'Observation well', zorder = 4)
r = 700
ax.set_xlim(-r, r)
ax.set_ylim(-r,r)
axins.add_patch(plt.Rectangle((-r,-r), 2*r, 2*r,fc = 'none', ec = 'black', linewidth = 1, zorder = 10))
axins.set_xlim(-2000, 2000)
axins.set_ylim(-2000,2000)

ax.set_aspect('equal')
axins.set_aspect('equal')
locs = [-564,-423, -282,-141,0,141,282,423,564]
labels = [int(item)/141 for item in locs]
ax.set_xticks(locs, labels)
ax.set_xlabel('$x/λ$ [-]')

ax.set_yticks(locs, labels)
ax.set_ylabel('$y/λ$ [-]')
ax.set_xlim(-r, r)
ax.set_ylim(-r,r)

axins.set_xticks([])
axins.set_yticks([])
# mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")
ax.legend(framealpha = 1, fontsize = 7, edgecolor = 'black', loc = 'lower right', fancybox = False)
fig.savefig(os.path.join('..','Images','2. Grid_pilot_points.pdf'), bbox_inches = 'tight')

