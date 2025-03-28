#%%
import pyemu
import os
import pandas as pd
import numpy as np
from functions.pest_funcs import get_use_obs
import shutil
import flopy
PestDir = os.path.join('..','pest_files')

#Read files
obs_real_data = pd.read_csv(os.path.join('..', 'inter', 'Observed_heads.csv'), index_col = 'index')
realdf = pd.read_csv(os.path.join('..', 'inter', 'Realizations.csv'), index_col = 'index')
realnames = realdf.columns[3:]
glm = os.path.join('..','exe','pestpp-glm')
mfsim = flopy.mf6.MFSimulation.load(sim_ws=os.path.join('..', 'ws'), verbosity_level=0, lazy_io=True)
gwf = mfsim.get_model()
SL = np.sqrt(20000)
fixed = True

#Loop over number of iterations
for obsno in range(2,17):
    runstring = f'{obsno}obs'
    #make results folder
    if not os.path.isdir(os.path.join('..', runstring)): 
        os.mkdir(os.path.join('..', runstring))
    testonce = True
    #loop over realizations
    for sim in ['sim_70_0','sim_70_1','sim_70_2''sim_70_3','sim_141_0','sim_141_1','sim_141_2''sim_141_3','sim_282_0','sim_282_1','sim_282_2','sim_282_3']:
        # for fac in [0.5,1,2,3]:
            #Loop over angles
            for angle in range(8):
                pst = pyemu.Pst(os.path.join(PestDir, 'eg.pst'))

                #reset parameter values
                pst.parameter_data.loc[:,'partrans'] = 'log'
                pst.parameter_data.loc[:,'parval1'] = 1
                pst.parameter_data.loc[:,'parlbnd'] = 0.001
                pst.parameter_data.loc[:,'parubnd'] = 1000
                
                
                name = f'_{angle}'
            
                #assign 'real' observations
                angle -= 4 
                useobs, useobs_str = get_use_obs(obsno, angle,SL)
                 # make string from observation locs
                #Change weights of observations to 0 except useobs
                pst.observation_data['weight'] = pst.observation_data['obgnme'].apply(lambda x: 1 if any(x.endswith(s) for s in useobs_str) else 0)
                #Check if all obs are correctly changed
                assert pst.observation_data['weight'].value_counts()[1] == len(useobs)*100
                #Turn off pumping well obs
                pst.observation_data.loc[pst.observation_data['obgnme'].str.endswith('2_0_0'), 'weight'] = 0
                #Set 'real' observation data to wells
                pst.observation_data['obsval'] = obs_real_data[sim].values


                master_dir = os.path.join('..', f'Master_{sim}_{name}_fixed')
                #Test pst file with noptmax = 0
                if testonce:
                    pst.control_data.noptmax = 0
                    pst.write(os.path.join(PestDir,'eg.pst'))
                    pyemu.os_utils.run(f"{glm} eg.pst", os.path.join('..',"pest_files"))
                    testonce = False



                if fixed:
                     for x,y in useobs:
                        #find actual pilot point coordinates
                        x = pst.parameter_data['x'].loc[(pst.parameter_data['x'].astype(float) - x).abs().idxmin()]
                        y = pst.parameter_data['y'].loc[(pst.parameter_data['y'].astype(float) - y).abs().idxmin()]
                        ppname = f'pname:p_inst:0_ptype:pp_pstyle:m_x:{x}_y:{y}_zone:1'
                        cellid = gwf.modelgrid.intersect(float(x),float(y))

                        realval = realdf.loc[cellid,sim]/realdf.loc[0,sim] # value as factor
                        pst.parameter_data.loc[ppname, 'parval1'] = realval
                        pst.parameter_data.loc[ppname, 'parubnd'] = realval*1.25 # 25% bounds
                        pst.parameter_data.loc[ppname, 'parlbnd'] = realval*0.75


                pst.control_data.noptmax = 7 # no of iterations
                pst.pestpp_options['n_iter_super'] = 10
                pst.pestpp_options['n_iter_base'] = -1 #Only super iter
                pst.pestpp_options['uncertainty'] = 0 #No uncertainty every iteration
                pst.svd_data.maxsing = pst.npar_adj
                os.chmod(os.path.join(PestDir,'forward_run.py'), 0o755)
                pst.write(os.path.join(PestDir,'eg.pst'))
                #Run PEST++ Parallel
                pyemu.os_utils.start_workers(os.path.join(PestDir), # the folder which contains the "template" PEST dataset
                                            os.path.join(glm), #the PEST software version we want to run
                                            'eg.pst', # the control file to use with PEST
                                            num_workers=12, #how many agents to deploy
                                            worker_root='..', #where to deploy the agent directories; relative to where python is running
                                            master_dir= master_dir,
                                            verbose = True
                                            )
                

                #Get calibrated parameters and run forward model again
                pst.parrep(parfile=os.path.join(master_dir, 'eg.par'))
                pst.write_input_files(pst_path=master_dir)
                pyemu.os_utils.run('python forward_run.py', cwd=master_dir)
                #Move master_dir to results folder
                shutil.move(master_dir, os.path.join('..','Results',runstring,os.path.basename(master_dir)))

# %%



# %%
