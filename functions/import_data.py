import numpy as np
import pandas as pd
import xarray as xr
import gsw # for conversion functions

def load_SAM_data(file_path='SAM.txt'):
    """
    Load the Southern Annular Mode (SAM) data from a text file, convert it to an xarray Dataset, 
    and return the processed SAM data between 2004 and 2025.

    Parameters
    ----------
    file_path : str, optional
        The path to the 'SAM.txt' file. Default is 'SAM.txt'.

    Returns
    -------
    xarray.DataArray
        Processed SAM data for the time range from January 2004 to January 2025.

    Raises
    ------
    FileNotFoundError
        If the specified file does not exist.
    ValueError
        If there are issues with data conversion or incorrect formats.
    
    Example
    -------
    >>> sam_data = load_SAM_data('path/to/SAM.txt')
    >>> print(sam_data)
    """
    
    try:
        # Read the data from the file
        data = pd.read_csv(file_path, sep='\s+', header=0, index_col=0)
    except FileNotFoundError:
        raise FileNotFoundError(f"The file '{file_path}' was not found. Please check the file path.")
    except Exception as e:
        raise ValueError(f"An error occurred while reading the file: {e}")
    
    try:
        # Convert the DataFrame to an xarray Dataset
        ds = xr.Dataset.from_dataframe(data)
        
        # Rename 'index' to 'time' (if index is named differently, adapt this)
        ds = ds.rename({'index': 'time'})
        
        # Extract the variable names
        vars = list(ds.data_vars)
        
        # Initialize a NaN-filled array with the size based on time and variables
        arr = np.full(ds.time.size * 12, np.nan)
        
        # Fill the array with data from each variable for each time step
        for i in range(ds.time.size):
            for n, v in enumerate(vars):
                arr[(i * 12) + n] = ds[v][i].data
        
        # Create a time range from the minimum to maximum 'time'
        time = pd.date_range(
            str(ds.time.min().data) + '-01-31',
            str(ds.time.max().data + 1) + '-01-01',
            freq='1M'
        )
        
        # Create a new xarray Dataset with SAM data and correct time coordinates
        ds = xr.Dataset(
            data_vars={'SAM': (['time'], arr)},
            coords={'time': time}
        )
        
        # Select the SAM data for the range 2004-01-01 to 2025-01-01
        sam = ds.SAM.sel(time=slice(np.datetime64('2004-01-01'), np.datetime64('2025-01-01')))
    
    except Exception as e:
        raise ValueError(f"An error occurred during data processing: {e}")
    
    return sam

def add_bathym_to_ds(ds,):
    """
    Add bathymetry data to the provided xarray dataset.

    This function loads bathymetry data from a predefined source file, processes it to match
    the spatial resolution of the dataset, and then adds the bathymetry data as a new variable
    'bth' within the dataset.

    Parameters:
    -----------
    ds : xarray.Dataset
        The input dataset to which the bathymetry data will be added.

    Returns:
    --------
    xarray.Dataset
        The input dataset with an additional 'bth' variable containing the bathymetry data.

    Notes:
    ------
    The bathymetry data is loaded from a set of NetCDF files located in the directory
    '/home/theospira/notebooks/data/GEBCO/'. The data is then downsampled to a specified grid cell size.
    """

    # load bathymetry data
    bth = xr.open_mfdataset('/home/theospira/notebooks/data/GEBCO/gebco**.nc')
    
    # add bathym data to dataset
    gs  = 1 # gridcell size
    bth = bth.sel(lat=slice(-80,-40)
            ).coarsen(lat=int(14400/60*gs),).mean(
            ).coarsen(lon=int(86400/360*gs),).mean().elevation.load()

    ds['bth'] = xr.DataArray(bth.sel(lat=slice(ds.lat.min().data,ds.lat.max().data)).data,dims=('lat','lon'))
    return ds

def add_adt_to_ds(ds):
    """
    Add Absolute Dynamic Topography (ADT) data to the provided xarray dataset.

    This function loads sea surface height (SSH) data, which represents the Absolute Dynamic
    Topography (ADT), from a predefined NetCDF file. The data is then re-gridded to a 1°x1°
    spatial resolution and added to the dataset as a new variable 'adt'.

    Parameters:
    -----------
    ds : xarray.Dataset
        The input dataset to which the ADT data will be added.

    Returns:
    --------
    xarray.Dataset
        The input dataset with an additional 'adt' variable containing the re-gridded ADT data.

    Notes:
    ------
    The SSH data is loaded from a NetCDF file located at 
    '/home/theospira/notebooks/data/Copernicus/ssh_monthly_climatology_2004-2021.nc'.
    The data is averaged across months, re-gridded to a 1°x1° grid, and then restricted to 
    latitudes between -80° and -40°.
    """

    # load ssh (adt) data
    ssh = xr.open_dataset('/home/theospira/notebooks/data/Copernicus/ssh_monthly_climatology_2004-2021.nc').mean('month',skipna=True)
    
    # re-grid onto 1°x1°
    gs = 1 # grid cell size
    ssh = (ssh.coarsen(lat=int(240*gs/60),).mean(
            ).coarsen(lon=int(1440*gs/360),).mean()).sel(lat=slice(-80,-40))
    
    # add adt to ds
    # Create a new DataArray with the calculated mean values and add it to the dataset
    ds['adt'] = xr.DataArray(ssh.adt.transpose('lon', 'lat'
                                 ).sel(lat=slice(ds.lat.min().data,ds.lat.max().data)).data,
                             dims=('lon','lat'))
    return ds

def cut_shallow_data(ds,dpt=2000):
    """
    Remove shallow water or above sea level data from the dataset based on bathymetry.

    This function filters out data from the provided xarray dataset where the bathymetry
    indicates shallow water (depth less than the specified threshold) or land (above sea level).
    If bathymetry data is not already present in the dataset, it calls `add_bathym_to_ds` to
    add it.

    Parameters:
    -----------
    ds : xarray.Dataset
        The input dataset from which shallow water or land data will be removed.
    
    dpt : int, optional
        The depth threshold (in meters) below which data will be retained. 
        Default is 2000 meters.

    Returns:
    --------
    xarray.Dataset
        The filtered dataset with shallow water or land data removed.

    Notes:
    ------
    The function assumes that the dataset contains a variable 'bth' representing
    bathymetry. If not, it will call `add_bathym_to_ds` to add this variable.
    """
    
    if 'bth' not in ds.data_vars.keys():
        ds = add_bathym_to_ds(ds)
    # remove any data that is "on shelf" or above sea level
    ds = ds.where(ds.bth < -dpt, drop=True)
    del(ds['bth'])
    return ds

def mask_data(ds, vars=None, t_ref=0, b_ref=2000):
    """
    Mask data, removing on-shelf data and any data further north than the 0 °C WW core temperature isotherm.
    Remove data from 2004.
    
    Parameters:
    - ds (xarray.Dataset): The dataset containing the data to be masked.
    - vars (list, optional): List of variables to retain in the dataset after masking. If None, all variables are retained.
    - t_ref (float, optional): Reference temperature value for masking. Default is 0 °C.
    - b_ref (int, optional): Reference value for data filtering. Default is 2000 m.

    Returns:
    - ds (xarray.Dataset): Masked dataset.

    Steps:
    1. Cut shallow data from the dataset based on a reference value.
    2. If specific variables are specified, retain only those variables in the dataset.
    3. Extract the circumpolar mean warm water core temperature data.
    4. Find the nearest point to the reference temperature.
    5. Select the nearest point to the reference temperature that is south of the Polar Front.
    6. Compute the rolling mean along the longitude axis with a window size of 5.
    7. Return the masked dataset.
    """
    
    # If specific variables are specified, retain only those variables in the dataset.
    if vars is not None:
        ds = ds[vars]
    
    # Cut shallow data from the dataset based on a reference value.
    ds = cut_shallow_data(ds, b_ref)

    # remove 2004 data
    if 'time' in ds.dims.keys():
        ds = ds.sel(time=slice('2005-01-01','2022-01-01'))
    
    # Extract the circumpolar mean warm water core temperature data.
    if 'ww_ct' not in ds.data_vars.keys():
        print('adding ww_ct')
        tmp = xr.open_dataset('/home/theospira/notebooks/projects/03-WW-timeseries/data/hydrographic_profiles/gridded-timeseries-ww-recalc1.nc')['ww_ct'].mean('time')
    else:
        print('contains ww_ct')
        tmp = ds.ww_ct.mean('time')
    
    # add adt to dataset
    if 'adt' not in ds.data_vars.keys():
        ds = add_adt_to_ds(ds)
    
    # Find the nearest point to the reference temperature.
    tmp2 = (tmp - t_ref).__abs__()
    # Select the nearest point to the reference temperature that is south of the Polar Front.
    tmp2 = tmp2.where(tmp2.lat<=(ds.adt - -0.58).__abs__().idxmin(dim='lat')).idxmin('lat').rolling({'lon':5},min_periods=1).mean()

    # mask all data north of the ref temp
    ds = ds.where(ds.lat<=tmp2)
    
    return ds

import sys
sys.path.append('/home/theospira/notebooks/projects/03-WW-timeseries/funcs')
from computations import calc_grid_cell_area


def load_sea_ice(ds=None):
    # load sea ice and create ds
    arr = np.load('/home/theospira/notebooks/projects/03-WW-timeseries/data/sea-ice/gridded-sea-ice-1deg.npy')
    
    x    = 1 
    time = pd.date_range('2004-01-01', '2024-01-01', freq='1M')
    lat  = np.arange(-79,-40+x,x)
    lon  = np.arange(0,360,x)
    
    ds1 = xr.Dataset(data_vars = dict(sic = (['time','lon','lat'], arr),),
                     coords   = dict(time = (['time'], time),
                                     lon  = (['lon'], lon),
                                     lat  = (['lat'], lat),),)
    
    # reorganise lon coords
    ds1['lon'] = np.concatenate((np.arange(0,180,1),np.arange(-180,0,1)))
    ds1 = ds1.sortby('lon')

    # calculate sea ice area
    ds1 = calc_grid_cell_area(ds1)
    ds1['sia'] = ds1.sic * ds1.gca
    ds1.sia.attrs['description']  = 'Sea ice area (km²) from grid cell area (km²) × SIC [-].'
    
    if ds != None:
        ds['sic'] = ds1['sic']
        ds['sia'] = ds1['sia']
        return ds
    else:
        return ds1

from warnings import filterwarnings as fw
fw('ignore')

def load_wind_data(f_path='/home/theospira/notebooks/data/ERA5/winds/wind_stress/tau_monthly_2004-2021.nc', mask_ds=1,
                   d_path='/home/theospira/notebooks/projects/03-WW-timeseries/data/hydrographic_profiles/'):
    """
    load wind data. adjust wind stress data from [N m-2 s] to [N m-2] by dividing by accumulated daily time (that is, seconds in a day).
    The default filepath is an nc file that contains the eastward and northward turbulent surface stresses from the following ERA5 product: https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-single-levels-monthly-means?tab=overview
    """
    
    # load wind data
    tau = xr.open_dataset(f_path)
    tau = tau.rename({'latitude':'lat','longitude':'lon'})
    
    # adjust from [N m-2 s] to [N m-2] by dividing by accumulated daily time (that is, seconds in a day)
    tau['ewss'] = tau['ewss']/(60*60*24)
    tau['nsss'] = tau['nsss']/(60*60*24)
    
    tau = tau.sel(lat=slice(-40,-79)).sortby('lat')

    if mask_ds:
        tau['ww_ct'] = xr.DataArray(xr.open_dataset(d_path+'gridded-timeseries-ww-recalc1.nc')['ww_ct'].data,dims=['time','lon','lat'])
        tau = mask_data(tau,vars=['ewss','nsss'])
    
    # get magnitude of tau
    tau['tau'] = np.sqrt(tau.ewss**2 + tau.nsss**2) 
    return tau