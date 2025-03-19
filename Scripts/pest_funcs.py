import os
import pypestutils.helpers as helpers
import pandas as pd
import numpy as np
import pyemu
import xarray as xr
def Get_SR(OrgDir, name):
    grb_fname = os.path.join(OrgDir,f"{name}.disv.grb")
    grid_info = helpers.get_2d_grid_info_from_mf6_grb(grb_fname)
    df = pd.DataFrame({'x' : grid_info['x'], 'y':grid_info['y']})
    sr = df.apply(tuple,axis=1).to_dict()
    return sr

def fix_k(PestDir, f):
    p = os.path.join(PestDir, f)
    with open(p, 'r') as file:
        lines = file.readlines()
    
    with open(os.path.join(PestDir,f), 'w') as file:
        for line in lines:
            values = line.split()
            for value in values:
                file.write(value + '\n')

def generate_points_within_circle(radius, distances, no_dirs = 8):
    if no_dirs == 16:
        directions = [
        (1, 0), (0.7071, 0.7071), (0, 1), (-0.7071, 0.7071), 
        (-1, 0), (-0.7071, -0.7071), (0, -1), (0.7071, -0.7071),
        (0.9239, 0.3827), (0.3827, 0.9239), (-0.3827, 0.9239), (-0.9239, 0.3827),
        (-0.9239, -0.3827), (-0.3827, -0.9239), (0.3827, -0.9239), (0.9239, -0.3827)]
    elif no_dirs == 8:
        dirs = [(1, 0), (0, 1), (-1, 0), (0, -1), (1, 1), (-1, 1), (-1, -1), (1, -1)]
        directions = [(dx / np.sqrt(dx**2 + dy**2), dy / np.sqrt(dx**2 + dy**2)) for dx, dy in dirs]
    xls= [0]
    yls = [0]
    # Loop through each direction
    for direction in directions:
        # Loop through each distance
        for distance in distances:
            # Calculate the coordinates in the given direction
                x = distance * direction[0]
                y = distance * direction[1]
                # Add the coordinates to the list if they are within the circle
                if np.sqrt(x**2 + y**2) <= radius:
                    xls.append(int(x))
                    yls.append(int(y))
    df = pd.DataFrame({'x' : xls, 'y' : yls, 'name':list(range(len(xls))), 'zone' : 1, 'value' : 1})
    df['name'] = 'pp' + df.name.astype(str)
    return df


def Get_Sens_fields(k1,top, PlotFor, ds,SL = [200], layers = [0,2]):
    dir = f'Master_{k1}_10_{top}'

    #Read pilot point file
    ppdf = pd.read_csv(os.path.join('..', 'pest_files', 'p_inst0pp.dat.tpl'), sep= ' ',skiprows=1, names = ['name','x', 'y'], usecols=[0,1,2])
    ppdf['index'] = ppdf.name 
    ppdf.set_index('index', inplace= True)

    #read jacobian and get values for ss or k
    jco = pyemu.Jco.from_binary(os.path.join('..', dir, 'eg.jcb'))
    jcodf = abs(jco.to_dataframe().T)
    jco = jcodf[:int(len(jcodf)/2)] if PlotFor == 'k' else jcodf[int(len(jcodf)/2):]

    #get times from jacobian file
    times = np.unique([i.split(':')[-1] for i in jcodf.columns.unique()]).astype(float)
    factors = os.path.join('..', 'pest_files', 'p_inst0pp.fac')
    for tno, t in enumerate(times):
        for no,l in enumerate(layers): 
            for o in SL: 
                tdf = ppdf.copy()
                obsname = f'oname:head_{l}_{o}_otype:lst_usecol:head_time:{t}'            
                tdf[obsname] = jco[obsname].values
                ppdfpath = os.path.join('..', 'pest_files','ppdf.csv')
                tdf.to_csv(ppdfpath, sep = ' ', header = False)
                c = pyemu.geostats.fac2real(ppdfpath, factors_file=factors,out_file=None, fill_value = 0)
                cdf  = c[0]*4*np.pi*10/24*10/1000
                if not isinstance(ds, xr.Dataset):
                    ds = xr.Dataset(coords={'cellid': np.arange(len(cdf)),
                        'k1': [2.5,10,40],
                        'time': times*10/(0.00001*SL[0]**2),
                        'top': ['open', 'closed'],
                        'layer': layers})
                    emp = np.full((len(cdf), len([2.5,10,40]), len(times), len(['open', 'closed']), len(layers)), np.nan)
                    ds['k'] = (['cellid', 'k1', 'time', 'top', 'layer'],emp)
                    ds['ss'] = (['cellid', 'k1', 'time', 'top', 'layer'],emp)
                if PlotFor == 'k':
                    ds['k'].loc[dict(k1=k1, time=t*10/(0.00001*SL[0]**2), top=top, layer=l)] = cdf
                elif PlotFor == 'ss':
                    ds['ss'].loc[dict(k1=k1, time=t*10/(0.00001*SL[0]**2), top=top, layer=l)] = cdf
    return ds

def rotate_point(x, y, angle_degrees):
    # Convert angle to radians manually
    angle_radians = np.pi * angle_degrees / 180
    
    # Apply rotation formula manually
    x_new = x * np.cos(angle_radians) - y * np.sin(angle_radians)
    y_new = x * np.sin(angle_radians) + y * np.cos(angle_radians)

    return int(fix_val(x_new)), int(fix_val(y_new))

def fix_val(val):
    if int(val) in [49,99, 199,299]:
        return int(val)+1
    elif int(val) in [-49,-99,-199,-299]:
        return int(val)-1
    else:
        return int(val)
    
def get_use_obs(obsno, angle,SL):
    useobs =  [(0,0),rotate_point(0,0.5*SL,angle*45)]
    if obsno >= 2:
        useobs.append(rotate_point(0,SL,angle*45+90))
    if obsno == 3:
        useobs.append(rotate_point(0,2*SL,angle*45-135))
    if obsno >= 4:
        useobs.append(rotate_point(0,2*SL,angle*45+180))
        useobs.append(rotate_point(0,3*SL,angle*45-90))
    if obsno >=5:
        useobs.append(rotate_point(0,0.5*SL,angle*45-135))
    if obsno >=6:
        useobs.append(rotate_point(0,SL,angle*45-45))
    if obsno >= 7:
        useobs.append(rotate_point(0,2*SL,angle*45+45))
    if obsno >= 8:
        useobs.append(rotate_point(0,3*SL,angle*45+135))
    if obsno >=9:
        useobs.append(rotate_point(0,3*SL,angle*45))
    if obsno >= 10:
        useobs.append(rotate_point(0,2*SL,angle*45+90))
    if obsno >= 11:
        useobs.append(rotate_point(0,SL,angle*45+180))
    if obsno >= 12:
        useobs.append(rotate_point(0,2*SL,angle*45-90))
    if obsno >=13:
        useobs.append(rotate_point(0,3*SL,angle*45-135))
    if obsno >=14:
        useobs.append(rotate_point(0,2*SL,angle*45-45))
    if obsno >= 15:
        useobs.append(rotate_point(0,0.5*SL,angle*45+45))
    if obsno >= 16:
        useobs.append(rotate_point(0,2*SL,angle*45+135))
    useobs_str = [f'_{x}_{y}' for x, y in useobs]
    return useobs, useobs_str

def plot_use_obs(useobs):
    import matplotlib.pyplot as plt
    x,y = zip(*useobs)
    plt.scatter(x,y,alpha = 0.5)
