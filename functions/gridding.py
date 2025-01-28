import pandas as pd
import numpy as np
import xarray as xr
from tqdm.notebook import tqdm_notebook as tqdm

def count_ww_profs(dsvar,condn=None):
    
    t_bins = pd.date_range('2003-12-31', '2022-01-01', freq='1M')
    # create empty array
    arr = np.ndarray([t_bins.size-1,360,40])*np.nan
    
    # define lon min and max resp
    gs = 1
    lon_min = -180
    lon_max = 180
    lon = np.arange(lon_min,lon_max+gs,gs)
    lon_labels = range(0,lon_max-lon_min,gs)
    
    lat_min = -80
    lat_max = -40
    lat = np.arange(lat_min,lat_max+gs,gs)
    lat_labels = range(0,lat_max-lat_min,gs)
    
    # group by seasons
    var = dsvar.groupby_bins(dsvar.time, bins=t_bins, labels=np.arange(len(t_bins)-1))
    
    # group into lon bins
    for t,gr_t in tqdm(var):
        var1 = gr_t.groupby_bins('lon',lon,labels=lon_labels,restore_coord_dims=True)
        
        # now group into lat bins for each lon group:
        for ln,gr_ln in var1:
            var2 = gr_ln.groupby_bins('lat',lat,labels=lat_labels,restore_coord_dims=True)
            
            # now take the mode!
            for lt,gr_lt in var2:
                if condn!=None:
                    arr[t,ln,lt] = np.where(gr_lt==condn)[0].size
                else:
                    arr[t,ln,lt] = np.where(gr_lt>0)[0].size
                    
    return arr

from scipy.stats import median_abs_deviation as mad

def mad_gridding(dsvar):
    """
    annual median absolute deviation on 1°x1° grid
    """
    
    t_bins = pd.date_range('2003-12-31', '2022-01-01', freq='1M')
    
    if 'pres' in dsvar.dims:
        arr = np.ndarray([360,40,ds.pres.size])*np.nan
    else:
        arr = np.ndarray([360,40])*np.nan
    
    # define lon min and max resp
    gs = 1
    lon_min = -180
    lon_max = 180
    lon = np.arange(lon_min,lon_max+gs,gs)
    lon_labels = range(0,lon_max-lon_min,gs)
    
    lat_min = -80
    lat_max = -40
    lat = np.arange(lat_min,lat_max+gs,gs)
    lat_labels = range(0,lat_max-lat_min,gs)
    
    # group into lon bins
    var1 = dsvar.groupby_bins('lon',lon,labels=lon_labels,restore_coord_dims=True)
    
    # now group into lat bins for each lon group:
    for ln,gr_ln in var1:
        var2 = gr_ln.groupby_bins('lat',lat,labels=lat_labels,restore_coord_dims=True)
        
        for lt,gr_lt in var2:
            arr[t,ln,lt] = mad(gr_lt,nan_policy='omit')
        
    return arr

def count_gridding_ts(dsvar,condn=None):
    
    t_bins = pd.date_range('2003-12-31', '2022-01-01', freq='1M')
    # create empty array
    arr = np.ndarray([t_bins.size-1,360,40])*np.nan
    
    # define lon min and max resp
    gs = 1
    lon_min = -180
    lon_max = 180
    lon = np.arange(lon_min,lon_max+gs,gs)
    lon_labels = range(0,lon_max-lon_min,gs)
    
    lat_min = -80
    lat_max = -40
    lat = np.arange(lat_min,lat_max+gs,gs)
    lat_labels = range(0,lat_max-lat_min,gs)
    
    # group by seasons
    var = dsvar.groupby_bins(dsvar.time, bins=t_bins, labels=np.arange(len(t_bins)-1))
    
    # group into lon bins
    for t,gr_t in tqdm(var):
        var1 = gr_t.groupby_bins('lon',lon,labels=lon_labels,restore_coord_dims=True)
        
        # now group into lat bins for each lon group:
        for ln,gr_ln in var1:
            var2 = gr_ln.groupby_bins('lat',lat,labels=lat_labels,restore_coord_dims=True)
            
            # now take the mode!
            for lt,gr_lt in var2:
                if condn!=None:
                    arr[t,ln,lt] = np.where(gr_lt==condn)[0].size
                else:
                    arr[t,ln,lt] = np.where(gr_lt>0)[0].size
                #arr[t,ln,lt] = gr_lt.notnull().sum()
            
    return arr

from scipy.stats import mode

def mode_gridding_ts(dsvar):
    
    t_bins = pd.date_range('2003-12-31', '2022-01-01', freq='1M')
    
    if 'pres' in dsvar.dims:
        arr = np.ndarray([t_bins.size-1,360,40,ds.pres.size])*np.nan
    else:
        arr = np.ndarray([t_bins.size-1,360,40])*np.nan
    
    # define lon min and max resp
    gs = 1
    lon_min = -180
    lon_max = 180
    lon = np.arange(lon_min,lon_max+gs,gs)
    lon_labels = range(0,lon_max-lon_min,gs)
    
    lat_min = -80
    lat_max = -40
    lat = np.arange(lat_min,lat_max+gs,gs)
    lat_labels = range(0,lat_max-lat_min,gs)
    
    # group by seasons
    var = dsvar.groupby_bins(dsvar.time, bins=t_bins, labels=np.arange(len(t_bins)-1))
    
    # group into lon bins
    for t,gr_t in tqdm(var):
        var1 = gr_t.groupby_bins('lon',lon,labels=lon_labels,restore_coord_dims=True)
        
        # now group into lat bins for each lon group:
        for ln,gr_ln in var1:
            var2 = gr_ln.groupby_bins('lat',lat,labels=lat_labels,restore_coord_dims=True)
            
            # now take the mode!
            for lt,gr_lt in var2:
                arr[t,ln,lt] = mode(gr_lt,nan_policy='omit')[0]
            
    return arr

def median_gridding_ts(dsvar, gs=1, lat_max=-40):
    """
    Calculate the median of a variable over specific time, longitude, and latitude bins, 
    creating a gridded time series array.

    Parameters
    ----------
    dsvar : xarray.DataArray
        The input data array containing the variable to be gridded. It must have at least 
        the dimensions 'time', 'lat', and 'lon'. If a 'pres' dimension is present, the output 
        array will include it as well.
        
    gs : int, optional
        The grid size in degrees for the spatial resolution. This value determines the 
        spacing between longitude and latitude bins. Default is 1 degree.
        
    lat_max : float, optional
        The maximum latitude value for the grid. The latitude grid will range from -80° 
        to `lat_max`. Default is -40 degrees.

    Returns
    -------
    arr : numpy.ndarray
        A 3D or 4D array containing the median values of the variable across the specified 
        time, longitude, and latitude bins. The dimensions are:
        
        - Time (number of time bins - 1)
        - Longitude (number of longitude bins - 1)
        - Latitude (number of latitude bins - 1)
        - Pressure levels (if 'pres' dimension is present in `dsvar`)
    
    Notes
    -----
    - The function groups the input data by monthly time bins from December 2003 to January 2022.
    - Longitude bins range from -180° to 180°, and latitude bins range from -80° to `lat_max`, 
      both using the grid size (`gs`) as the bin width.
    - The median is calculated for each bin, ignoring NaN values.
    """
    
    t_bins = pd.date_range('2003-12-31', '2022-01-01', freq='1M')
    
    # define lon min and max resp
    lon_min = int(-180)
    lon_max = int(180)
    lon = np.arange(lon_min,lon_max+gs,gs) #+gs/2
    lon_labels = np.arange(0,lon.size-1,1).astype(int)
    
    lat_min = -80.
    lat_max = lat_max
    lat = np.arange(lat_min,lat_max+gs,gs) #- gs/2
    lat_labels = np.arange(0,lat.size-1,1).astype(int)
    
    if 'pres' in dsvar.dims:
        arr = np.ndarray([t_bins.size-1,lon_labels.size,lat_labels.size,dsvar.pres.size])*np.nan
    else:
        arr = np.ndarray([t_bins.size-1,lon_labels.size,lat_labels.size])*np.nan
    
    # group by seasons
    var = dsvar.groupby_bins(dsvar.time, bins=t_bins, labels=np.arange(len(t_bins)-1))
    
    # group into lon bins
    for t,gr_t in tqdm(var):
        var1 = gr_t.groupby_bins('lon',lon,labels=lon_labels,restore_coord_dims=True)
        
        # now group into lat bins for each lon group:
        for ln,gr_ln in var1:
            var2 = gr_ln.groupby_bins('lat',lat,labels=lat_labels,restore_coord_dims=True)
            
            # now take the mode!
            for lt,gr_lt in var2:
                arr[t,ln,lt] = gr_lt.median('n_prof',skipna=True)
            
    return arr


def vincents_median_grid_var(ds, var=None):
    """
    Apply Vincent Deroit's median gridding method to a specified variable in an xarray Dataset.

    This function converts the specified variable in the input dataset to a pandas DataFrame,
    grids the data into latitude, longitude, and time bins, and calculates the median value 
    within each bin. The result is converted back to an xarray Dataset with gridded data.

    Parameters
    ----------
    ds : xarray.Dataset
        The input dataset containing the variable to be gridded.
        
    var : str
        The name of the variable in the dataset that you want to grid using the median.

    Returns
    -------
    ds_cut : xarray.Dataset
        The resulting dataset with the median-gridded variable. The dataset contains the same 
        variable, now gridded over the defined latitude, longitude, and time bins. The dimensions 
        are 'time', 'lat', and 'lon'.
    
    Notes
    -----
    - Latitude and longitude bins are defined using 1-degree intervals, from the minimum to 
      the maximum values found in the dataset.
    - Time bins are defined monthly, spanning from the month before the first time entry to 
      the last month in the dataset.
    - The function calculates the median for each bin, ignoring NaN values.
    - Any zero values in the resulting dataset are replaced with NaNs.

    Example
    -------
    >>> ds = xr.Dataset(...)
    >>> var = 'temperature'
    >>> ds_gridded = vincents_median_grid_var(ds, var)
    >>> print(ds_gridded)
    """

    if var is not None:
        df = ds[var].to_dataframe()
    else:
        df = ds.to_dataframe()
    #df = df.rename(columns={"LONGITUDE":"longitude","LATITUDE":"latitude"})
    
    lat_bins = np.arange(np.floor(df.lat.min()),np.ceil(df.lat.max())+1,1)
    lon_bins = np.arange(np.floor(df.lon.min()),np.ceil(df.lon.max())+1,1)
    ### Cut containing pandas intervals lon/lat/time
    cut_lat_label = pd.cut(df["lat"],lat_bins)
    cut_lon_label = pd.cut(df["lon"],lon_bins)
    bins_dt = pd.date_range(start=df["time"].min()+pd.DateOffset(months=-1), end=df["time"].max(), freq='M')
    bins_str = bins_dt.astype(str).values
    cut_time_label = pd.cut(df['time'], bins=bins_dt)
    cut_time_label.dropna()
    df_cut_label = df.drop(["lat","lon","time"],axis=1)
    df_cut_label = df_cut_label.groupby([cut_time_label,cut_lon_label,cut_lat_label]).median()
    
    lat_mid = (pd.IntervalIndex(df_cut_label.index.get_level_values('lat')).mid).unique()
    lon_mid = (pd.IntervalIndex(df_cut_label.index.get_level_values('lon')).mid).unique()
    time_mid = (pd.IntervalIndex(df_cut_label.index.get_level_values('time')).mid).unique()
    
    df_cut_label.index = df_cut_label.index.set_levels(time_mid.values,level=0)
    df_cut_label.index = df_cut_label.index.set_levels(lon_mid,level=1)
    df_cut_label.index = df_cut_label.index.set_levels(lat_mid.values,level=2)
    
    df_cut = df_cut_label.copy()
    df_cut.replace(0,np.nan,inplace=True)
    ds_cut = df_cut.to_xarray()
    ds_cut["lat"] = sorted(lat_mid)
    ds_cut["lon"] = sorted(lon_mid)
    ds_cut["time"] = time_mid
    #df = ds_cut.to_dataframe()
    #df["strat"] = df[var]
    #df["ist"] = df[var]
    return ds_cut