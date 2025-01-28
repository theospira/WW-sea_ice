import pandas as pd
import numpy as np
import xarray as xr

def calc_mean_var(ds, v):
    """
    Calculates the climatological mean of the variable 'v' over each month and adds it to the dataset as a new variable.

    Parameters:
    - ds (xarray.Dataset): Input dataset containing the variable 'v'.
    - v (str): Name of the variable for which to calculate the climatological mean.

    Returns:
    - ds (xarray.Dataset): Modified dataset with the climatological mean variable added.

    This function calculates the climatological mean of the variable 'v' over each month and adds it to the input dataset 
    as a new variable with the suffix '_clim'. The climatological mean is calculated by averaging the variable 'v' over 
    the time dimension for each month.
    """
    
    # Calculate the mean value of the variable 'v' over each month
    if v is None:
        n2 = ds.groupby('time.month').mean(('time')).data
        
        # Create an empty array with the same shape as the variable 'v'
        arr = np.ndarray(ds.shape) * np.nan
        
        # Fill the array with the monthly mean values
        for i in range(int(arr.shape[0] / 12)):
            arr[int(i * 12):int((i + 1) * 12)] = n2
        
        # Create a new DataArray with the calculated mean values and add it to the dataset
        dsvar = xr.DataArray(arr, dims=ds.dims)
        
        return dsvar
        
    else:
        n2 = ds[v].groupby('time.month').mean(('time')).data
        
        # Create an empty array with the same shape as the variable 'v'
        arr = np.ndarray(ds[v].shape) * np.nan
        
        # Fill the array with the monthly mean values
        for i in range(int(arr.shape[0] / 12)):
            arr[int(i * 12):int((i + 1) * 12)] = n2
        
        # Create a new DataArray with the calculated mean values and add it to the dataset
        ds[v + '_mn'] = xr.DataArray(arr, dims=ds[v].dims)
        
        return ds

def polyfit_xr(dsvar, dim='time', deg=1):
    """
    Polynomial fitting for xarray DataArray along a specified dimension.

    Parameters:
    - dsvar (xarray.DataArray): The dataset variable.
    - dim (str): The dimension along which to perform the polynomial fitting.
    - deg (int): The degree of the polynomial.

    Returns:
    - slope (float): The slope coefficient of the polynomial fit.
    - intercept (float): The intercept coefficient of the polynomial fit.
    """
    # Perform polynomial fitting using xarray's polyfit method
    pf = dsvar.polyfit(dim, deg=deg)
    
    # Extract slope and intercept coefficients from the polynomial fit
    slope = pf.polyfit_coefficients[0]
    intercept = pf.polyfit_coefficients[1]
    
    return slope, intercept

def calculate_whiskers(dsvar):
    """
    Calculates the lower and upper whiskers of a dataset excluding NaN values.

    Parameters:
    - dsvar (xarray.DataArray): The dataset variable to calculate whiskers for.

    Returns:
    - lower_whisker (float): The lower whisker value.
    - upper_whisker (float): The upper whisker value.
    """
    # Flatten the dataset and remove NaN values
    data = dsvar.data.flatten()
    data = data[~np.isnan(data)]
    
    # Calculate quartiles and interquartile range (IQR)
    quartiles = np.percentile(data, [25, 75])
    iqr = quartiles[1] - quartiles[0]
    
    # Calculate lower and upper whiskers
    lower_whisker = np.max([np.min(data), quartiles[0] - 1.5 * iqr])
    upper_whisker = np.min([np.max(data), quartiles[1] + 1.5 * iqr])
    
    return lower_whisker, upper_whisker


import gsw # for conversion functions
def calc_grid_cell_area(ds):
    """
    Calculate the area of each grid cell in a given dataset.

    Args:
        ds (xarray.Dataset): Input dataset containing latitude and longitude coordinates.

    Returns:
        xarray.Dataset: Dataset with a new variable 'gca' representing grid cell areas.
    """

    # Extract latitude and longitude coordinates from the dataset
    lon = ds.lon.data
    lat = ds.lat.data

    # Determine the number of longitude and latitude points
    num_lon = len(lon)
    num_lat = len(lat)

    # Initialize an array to store grid cell areas
    grid_cell_areas = np.zeros((lat.size, lon.size))

    # Iterate over latitude and longitude coordinates
    for i in range(lat.size):
        for j in range(lon.size):
            # Calculate coordinates of the current grid cell
            lon1 = lon[j]
            lat1 = lat[i]
            lon2 = lon[j] + (lon[1] - lon[0])  # Assuming regular grid spacing
            lat2 = lat[i] + (lat[1] - lat[0])  # Assuming regular grid spacing
            
            # Calculate distance between two points using geospatial library (e.g., gsw)
            distance = gsw.distance([lon1, lon2], [lat1, lat2])[0] 

            # Calculate area of the grid cell
            grid_cell_areas[i, j] = distance * (lon[1] - lon[0]) * (lat[1] - lat[0])

    # Add grid cell area as a new variable to the dataset
    ds['gca'] = xr.DataArray(grid_cell_areas, dims=('lat', 'lon'))

    return ds

def grid_year_month(dsvar):
    """
    Computes a grid of mean values for each month across different years from a given dataset.
    
    Args:
    - dsvar (xarray.Dataset or xarray.DataArray): Dataset or DataArray containing the variable of interest with a 'time' dimension.

    Returns:
    - arr (numpy.ndarray): 2D array where each row represents a year and each column represents a month. 
                           Contains the mean values of the variable across latitudes and longitudes for each month and year.
                           NaN values are filled for missing data.
    """
    # Create an empty array to store mean values for each month of each year
    arr = np.ndarray((np.unique(dsvar.time.dt.year).size, np.unique(dsvar.time.dt.month).size)) * np.nan
    
    # Group the dataset by year
    yr = dsvar.groupby('time.year')
    
    # Iterate over each group (year) and compute the mean for each month
    for i, y in enumerate(list(yr.groups.keys())):
        arr[i, :] = yr[y].groupby('time.month').mean(('lat', 'lon')).data
    
    return arr

def calc_mlp(ds, den_lim=0.03):
    """
    Calculate the mixed layer pressure (mlp) for an oceanographic dataset.

    Parameters:
    ds (xarray.Dataset): The dataset containing the oceanographic profiles with density ('rho') and pressure ('pres') variables.
    den_lim (float): The density threshold to determine the mixed layer depth, default is 0.03 kg/m^3 as per de Boyer Montegut et al.

    Returns:
    xarray.Dataset: The input dataset with an additional 'mlp' variable representing the mixed layer pressure for each profile.

    Procedure:
    1. Determine the index of the first non-null density data point for each profile.
    2. Ensure that the reference depth is at least 10 dbar (corresponding to index 5, since pressure is likely binned at 2 dbar intervals).
    3. Filter out profiles that do not have data within the top 30 dbar (15 indices if binned at 2 dbar intervals).
    4. Recalculate the first non-null density data point index for the filtered profiles.
    5. Calculate the mixed layer pressure by finding the first pressure level where the absolute difference in density from the reference depth exceeds the density threshold (den_lim).
       - The result is adjusted by adding 5 to the index to account for starting from the 5th bin onward.
       - The index is multiplied by 2 to convert from index to dbar (assuming 2 dbar per bin).

    Notes:
    - This function assumes the dataset has been appropriately preprocessed and contains 'rho' (density) and 'pres' (pressure) variables.
    - Profiles without valid data within the top 30 dbar are dropped from the dataset.
    - The calculation follows the methodology described by de Boyer Montegut et al. for determining mixed layer depth based on density criteria.

    Example:
    >>> ds = xr.open_dataset('path_to_dataset.nc')
    >>> ds_with_mlp = calc_mlp(ds)
    """

    if 'sig' in ds.data_vars:
        dens = ds.sig
    else:
        dens = ds.rho
    # Determine the first non-null density data point index for each profile
    ref_dpt_idx = dens.isnull().argmin('pres')

    # Ensure the reference depth is at least 10 dbar (index 5)
    ref_dpt_idx = ref_dpt_idx.where(ref_dpt_idx > 5, 5)

    # Remove profiles without data in the top 30 dbar (index 15)
    dens = dens.where(ref_dpt_idx <= 15, drop=True)
    ref_dpt_idx = dens.isnull().argmin('pres')

    # Calculate mixed layer pressure
    ds['mlp'] = ((np.abs(dens.isel(pres=ref_dpt_idx) - dens.isel(pres=slice(5, 2000))) > den_lim
                  ).argmax(dim='pres', skipna=True) + 5) * 2

    return ds
