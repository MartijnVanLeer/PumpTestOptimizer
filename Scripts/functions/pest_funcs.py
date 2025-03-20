import os
import pypestutils.helpers as helpers
import pandas as pd
import numpy as np
import pyemu
import xarray as xr
def Get_SR(OrgDir : str, name: str):
    """Get spatial reference from grb file for pilot point interpolation

    Parameters
    ----------
    OrgDir : str
        MF6 model folder
    name : str
        model name

    Returns
    -------
    dict
        spatial reference
    """    
    grb_fname = os.path.join(OrgDir,f"{name}.disv.grb")
    grid_info = helpers.get_2d_grid_info_from_mf6_grb(grb_fname)
    df = pd.DataFrame({'x' : grid_info['x'], 'y':grid_info['y']})
    sr = df.apply(tuple,axis=1).to_dict()
    return sr

def fix_k(PestDir : str, f : str):
    """fix auto written modflow K file to have values on new lines

    Parameters
    ----------
    PestDir : str
        folder with pest files from PstFrom
    f : str
        filename
    """    
    p = os.path.join(PestDir, f)
    with open(p, 'r') as file:
        lines = file.readlines()
    
    with open(os.path.join(PestDir,f), 'w') as file:
        for line in lines:
            values = line.split()
            for value in values:
                file.write(value + '\n')

def generate_points_within_circle(radius : float, distances : list, no_dirs = 8) -> pd.DataFrame:
    """
    Generate points in a circle at a range of angles and radial distances to create pilot point input for PESTPP

    Parameters
    ----------
    radius : float
        Radius of circle
    distances : list
        radial distances from center
    no_dirs : int, optional
        8 or 16, number of angles, by default 8

    Returns
    -------
    pd.DataFrame
        df with points, ready for PEST pilot point input
    """  
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


def rotate_point(x :float, y :float, angle_degrees : float):
    """Rotate points around pumping well for a certain distance

    Parameters
    ----------
    x : float
        x-coordinate
    y : float
        y-coordinate
    angle_degrees : float
        angle of rotation

    Returns
    -------
    int, int
        x, y coordinates of point rotated around pumping well 
    """    
    # Convert angle to radians manually
    angle_radians = np.pi * angle_degrees / 180
    
    # Apply rotation formula manually
    x_new = x * np.cos(angle_radians) - y * np.sin(angle_radians)
    y_new = x * np.sin(angle_radians) + y * np.cos(angle_radians)

    return int(fix_val(x_new)), int(fix_val(y_new))

def fix_val(val:int)-> int:
    """
    Hacky way to make sure the observation location exists
    Parameters
    ----------
    val : int
        coordinate value

    Returns
    -------
    int
        coordinate at observation coordinate
    """    
    if int(val) in [49,99, 199,299]:
        return int(val)+1
    elif int(val) in [-49,-99,-199,-299]:
        return int(val)-1
    else:
        return int(val)
    
def get_use_obs(obsno : int, angle : int,SL : int) -> list:
    """_summary_

    Parameters
    ----------
    obsno : int
        Number of obs in use
    angle : int
        number of 45 degrees angle rotations (0 to 7)
    SL : int
        Leakage factor (sqrt(KDc))

    Returns
    -------
    list
        list with tuples of observation locations
    list
        list with strings of observation locations
    """

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
