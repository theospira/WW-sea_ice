import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append('/home/theospira/notebooks/projects/03-WW-timeseries/funcs')
from computations import polyfit_xr

def plot_sea_ice_climate_stripes(si, ax, fig, cax=None, fs=18, title_loc='right', **cb_kwargs):
    """
    Plot sea ice climate stripes as a 1D heatmap.

    This function visualizes sea ice area (SIA) anomalies over time using climate stripe-style 
    heatmaps, with a colorbar indicating the anomaly magnitude.

    Parameters
    ----------
    si : xarray.Dataset
        The dataset containing sea ice area (`sia`) and its climatological mean (`sia_mn`). Both 
        variables must have dimensions including `lon` and `lat`.

    ax : list of matplotlib.axes.Axes
        The axes on which the heatmap and colorbar will be plotted. The first axis in the list 
        (`ax[0]`) is used for the heatmap.

    fig : matplotlib.figure.Figure
        The figure object to which the colorbar is added.

    fs : int, optional
        Font size for the title of the heatmap. Default is 18.

    Returns
    -------
    None
        The function directly plots the heatmap and colorbar on the provided axes.
    """
    
    hmp = plot_1D_heatmap(((si.sia.sum(('lon','lat')) - si.sia_mn.sum(('lon','lat')))*1e-6),ax=ax)
    ax.set_title('SIA Anomaly 'r'($\times 10^6$ km$^2$)',loc=title_loc,fontsize=fs)
    if cax is None:
        cb = fig.colorbar(hmp,ax=ax,extend='both',**cb_kwargs)
    else:
        cb = fig.colorbar(hmp,cax=cax,extend='both')
    cb.set_ticks([-25,25])

from matplotlib.dates import AutoDateLocator
def si_timeseries_plot_formatting(ax=None, si_date='2015-08-15', lw=2.5):
    """
    Format a sea ice timeseries plot with specific labels, limits, and a reference line.

    This function applies custom formatting to a timeseries plot related to sea ice concentration, 
    including setting a vertical reference line on a specified date and defining x-axis limits and 
    minor tick intervals.

    Parameters
    ----------
    ax : matplotlib.axes.Axes, optional
        The axes on which to apply formatting. If None, formatting is applied to the current axes.

    si_date : str, optional
        The date for the vertical reference line in 'YYYY-MM-DD' format. Default is '2015-08-15'.
        
    lw : float, optional
        The linewidth of the reference line. Default is 2.5.

    Returns
    -------
    None
    """

    if ax is None:
        a = plt.gca()
    else:
        a = ax
        a.axvline(np.datetime64(si_date),c='k',ls='--',lw=lw)
        a.set_xlabel('')
        a.set_xlim(np.datetime64('2004-12-31'),np.datetime64('2022-01-01'))
        a.xaxis.set_minor_locator(AutoDateLocator(maxticks=20),)

from warnings import filterwarnings as fw
fw('ignore')
from matplotlib.colors import LinearSegmentedColormap
def get_cmap(colors = ['#67001f', '#b2182b', '#d6604d', '#f4a582', '#fddbc7', '#f7f7f7', '#d1e5f0', '#92c5de', '#4393c3', '#2166ac', '#053061'][::-1]):
    """
    Create and return a custom colormap for visualizations.

    This function generates a linear segmented colormap that transitions through a specified 
    set of colors, ordered from dark red to dark blue. The colormap is defined using 
    hexadecimal color codes and is intended for use in visualizations where a smooth 
    transition between these colors is desired.

    Returns
    -------
    cmap : matplotlib.colors.LinearSegmentedColormap
        A custom colormap object that can be used in plotting functions to map data values 
        to colors.

    Notes
    -----
    - The colormap transitions through the following colors (in reverse order for the final colormap):
      '#053061', '#2166ac', '#4393c3', '#92c5de', '#d1e5f0', '#f7f7f7', 
      '#fddbc7', '#f4a582', '#d6604d', '#b2182b', '#67001f'.
    - The resulting colormap can be applied to any data visualization that supports colormaps, 
      such as those created with matplotlib.
    
    Example
    -------
    >>> cmap = get_cmap()
    >>> plt.imshow(data, cmap=cmap)
    >>> plt.colorbar()
    """
    
    # Define the colors in the order you've specified
    # Create a colormap that transitions from the first color to the last
    cmap_name = 'custom_cmap'
    cmap = LinearSegmentedColormap.from_list('custom_cmap', colors)
    return cmap

def plot_1D_heatmap(dsvar, ax=None, cmap=get_cmap().reversed(), add_cbar=False, **kwargs):
    """
    Plot a 1D heatmap of anomalies over time using a custom colormap.

    This function generates a 1D heatmap plot similar to the IPCC or Greta Thunberg temperature anomaly
    plot, visualizing a dataset variable (`dsvar`) over time with a color-coded anomaly representation. 
    If overlaying this heatmap with a line plot, ensure the variable for both plots is the same.

    Parameters
    ----------
    dsvar : xarray.DataArray
        The dataset variable to be plotted, representing anomalies or similar data over time.
        
    ax : matplotlib.axes.Axes, optional
        The matplotlib axes on which the heatmap will be plotted. If not provided, the function 
        will create a new figure and axes.
        
    cmap : matplotlib.colors.Colormap, optional
        The colormap used for plotting. By default, a reversed custom colormap (blue-red divergent; `cmap_br.reversed()`) is used.
        
    add_cbar : bool, optional
        Whether to add a colorbar to the plot. Default is False.

    Returns
    -------
    hmp : matplotlib.collections.QuadMesh
        The heatmap (pcolormesh) object is returned if `add_cbar` is False, allowing for additional modifications
        to the plot outside this function.
        If `add_cbar` is True, the heatmap is plotted directly without returning the object.

    Notes
    -----
    - The y-axis labels and ticks are hidden since this is a 1D plot without spatial dimensions.
    - The heatmap grid is created by generating a meshgrid from `dsvar`, filling the plot with the data.

    Example
    -------
    >>> fig, ax = plt.subplots(figsize=(15, 1))
    >>> plot_1D_heatmap(dsvar, ax, add_cbar=True)
    """
    
    # Create a meshgrid with time and anomaly data
    anom, arr = np.meshgrid(dsvar, dsvar.time)
    
    # Create a new figure and axis if none is provided
    if ax is None:
        plt.figure(figsize=(15, 5))
        ax = plt.gca()
    
    # Plot the heatmap using the specified colormap
    hmp = ax.pcolormesh(dsvar.time, arr, anom, cmap=cmap, **kwargs)
    
    # Set the x-axis label to 'Year'
    ax.set_xlabel('Year')
    
    # Remove y-axis labels and ticks
    ax.set_ylabel('')
    ax.set_yticklabels('')
    ax.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)

    # Add colorbar if specified
    if add_cbar:
        plt.colorbar(hmp, label='SIA Anomaly')
    else:
        return hmp

def plot_lin_reg_single_dim(dsvar, ax, color='#984ea3',**kwargs):
    """
    Plot a linear regression trend for a dataset variable without accounting for spatial dimensions.

    This function calculates and plots a simple linear regression for the input dataset variable 
    (`dsvar`) on the provided matplotlib axis (`ax`), assuming no latitude or longitude 
    dimensions are present.

    Parameters
    ----------
    dsvar : xarray.DataArray
        The dataset variable to be used for linear regression.
        
    ax : matplotlib.axes.Axes
        The matplotlib axes on which the regression line will be plotted.
        
    color : str, optional
        The color of the regression line. Default is '#984ea3'.

    Returns
    -------
    None
        The function plots the linear regression line on the provided axis but does not return any values.

    Example
    -------
    >>> fig, ax = plt.subplots()
    >>> plot_lin_reg_single_dim(dsvar, ax, color='blue')
    """
   
    # Calculate the linear regression using a helper function `polyfit_xr`
    s1, i1 = polyfit_xr(dsvar)
    
    # Convert slope and intercept to numpy data
    s1 = s1.data
    i1 = i1.data
    
    # Apply the linear equation y = slope * time + intercept
    #time = pd.date_range(str(dsvar.time.min().data)+'-01-31',
     #                    str(dsvar.time.max().data+1)+'-01-01',freq='1M')
    time = dsvar.time.data
    y = pd.to_numeric(time) * s1 + i1
    
    # Plot the linear regression line on the given axis
    ax.plot(time, y, c=color, **kwargs)

def plot_lin_reg(dsvar, ax, date=None, compute_slope_rate=True, return_coeffs=False, return_slope_rate=False):
    """
    Plot linear regression of a dataset before and after a significant event, such as the 2016 sea ice melt.

    This function fits two linear regression models on the dataset `dsvar`, split by a key date (2016). It plots the
    regression lines before and after the event on the provided axes. Optionally, it can compute and print the slope
    rates for each segment and return the regression coefficients.

    Parameters
    ----------
    dsvar : xarray.DataArray
        The dataset variable to be analyzed, typically representing data over time with longitude and latitude dimensions.
        
    ax : matplotlib.axes.Axes
        The matplotlib axes on which to plot the linear regressions.
    
    date : numpy.datetime64, optional
        The date used to split the dataset for regression. Default is the last date in 2015. 

    compute_slope_rate : bool, optional
        If True, prints the slope rates for the linear regressions before and after the specified event.
        Default is True.

    return_coeffs : bool, optional
        If True, returns the coefficients (slope and intercept) of the linear regressions.
        Default is False.

    Returns
    -------
    If return_coeffs is True:
    - s1 (float): Slope of the linear regression before the specified event (default is 2016).
    - i1 (float): Intercept of the linear regression before the event.
    - s2 (float): Slope of the linear regression after the event.
    - i2 (float): Intercept of the linear regression after the event.

    Example
    -------
    >>> fig, ax = plt.subplots()
    >>> plot_lin_reg(dsvar, ax, compute_slope_rate=True, return_coeffs=True)
    """
    # Get the date of 2015
    if date is None:
        date = dsvar.time.sel(time='2015')[-1].data
        # Get the index corresponding to 2015
        idx = int(np.where(dsvar.time.dt.floor('D') == date)[0])
        # Get time values before and after 2016
        time = dsvar.time.data[:idx]
    else:
        # Get the index corresponding to 2015
        idx = int(np.where(dsvar.time.dt.floor('D') == date)[0])
        # Get time values before and after 2016
        time = dsvar.time.data[:idx]
    
    # Calculate linear regression before 2016
    s1, i1 = polyfit_xr(dsvar.mean(('lon', 'lat'))[:idx])
    s1 = s1.data
    i1 = i1.data
    y1 = pd.to_numeric(time) * s1.data + i1.data
    ax.plot(time, y1, c='k', lw=3, ls='--')  # Plot before 2016 in purple (#984ea3)
    
    # Get time values after 2016
    time = dsvar.time.data[idx:]
    
    # Calculate linear regression after 2016
    s2, i2 = polyfit_xr(dsvar.mean(('lon', 'lat'))[idx:])
    s2 = s2.data
    i2 = i2.data
    y2 = pd.to_numeric(time) * s2 + i2
    ax.plot(time, y2, c='k', lw=3, ls='--')  # Plot after 2016 in red (#e41a1c)

    if compute_slope_rate:
        slope1 = (y1[-1]-y1[0]) / idx
        print('slope 1: ',slope1, ' per unit time')
        slope2 = (y2[-1]-y2[0]) / (dsvar.time.size-idx)
        print('slope 2: ',slope2, ' per unit time')
        if return_slope_rate:
            return slope1, slope2

    if return_coeffs:
        return s1, i1, s2, i2

def fig_labels(a_x, a_y, ax, j=0, lower_case=True, label_format="({})", fs=15, add_bbox=True):
    """
    Add alphabetical labels to figures or subplots.

    This function adds sequential alphabetical labels to each subplot in a figure, using either lowercase 
    or uppercase letters of the English alphabet. Labels are positioned at specified coordinates within 
    the axes. Optional formatting is available for the label style, font size, and the addition of bounding boxes.

    Parameters
    ----------
    a_x : float
        The x-coordinate for the label's position within the axes, in axes fraction units (0 to 1 range).

    a_y : float
        The y-coordinate for the label's position within the axes, in axes fraction units (0 to 1 range).

    ax : matplotlib.axes._axes.Axes or list of matplotlib.axes._axes.Axes
        The axes or list of axes where labels should be added. If a single `Axes` object is provided, 
        only that subplot will be labeled. If a list of axes is provided, each subplot will be labeled 
        sequentially.

    j : int, optional
        The starting index for labeling, where 0 corresponds to 'a' or 'A'. This allows for customized 
        label sequences starting from any letter. Default is 0 (i.e., 'a' or 'A').

    lower_case : bool, optional
        If True, labels are generated using lowercase letters ('a' to 'z'). If False, uppercase 
        letters ('A' to 'Z') are used. Default is True.

    label_format : str, optional
        A format string for how the label appears. The placeholder {} will be replaced by the 
        corresponding letter. Default is "({})", which gives labels like "(a)" or "(A)". Customizations 
        like "{}." will produce labels like "a." or "A.".

    fs : int, optional
        Font size for the labels. Default is 15.

    add_bbox : bool, optional
        If True, a bounding box will be added around the label for better visibility. The bounding box 
        will have a light background and border. Default is True.

    Returns
    -------
    None
        The function adds labels directly to the provided axes or list of axes, without returning any value.

    Example
    -------
    >>> fig, ax = plt.subplots(2, 2)
    >>> fig_labels(0.1, 0.9, ax=ax, lower_case=False, label_format="{}.", fs=12)
    """

    # add bbox arguments
    bbox_kw = None
    if add_bbox:
        bbox_kw = dict(facecolor='#f7f7f7', edgecolor='#f7f7f7', boxstyle='round,pad=0.1', alpha=0.75)
    
    # Generate the alphabet list
    alphabet = [chr(i) for i in range(ord('a'), ord('z') + 1)] if lower_case else [chr(i) for i in range(ord('A'), ord('Z') + 1)]
    
    # Ensure ax is iterable (if a single Axes is passed, make it a list)
    if not isinstance(ax, (list, np.ndarray)):
        ax = [ax]

    # Check if the number of labels exceeds the alphabet length
    if j + len(ax) > len(alphabet):
        raise ValueError("The number of subplots exceeds the available alphabet letters.")

    # Add labels to the axes
    for i, a in enumerate(ax):
        label = label_format.format(alphabet[j + i])
        a.annotate(label, xy=(a_x, a_y), xycoords='axes fraction', fontsize=fs, fontweight='bold', ha='center',va='center',
                   bbox=bbox_kw)

def plot_with_season_background(ds,ax=None,alpha=0.075, **kwargs):
    """
    Plot the mean temperature over time with background fill for austral winter (JJA) and summer (DJF) seasons.

    Parameters:
        ds (xarray.Dataset): Dataset containing time, lon, lat, and temperature data.
        ax (matplotlib.axes.Axes): Axes object to plot on.
    """
    # Define months for winter and summer
    sum = np.asarray([12, 1, 2])  # December, January, February
    win = np.asarray([6, 7, 8])  # June, July, August

    # Determine whether to use plt or ax
    if ax is None:
        ax = plt.gca()
    
    for t in ds.time:
        if t.dt.month.data in np.append(sum, win):
            t2 = t.dt.month.data + 1
            y2 = t.dt.year.data
            if t2 == 13:
                t2 = 1
                y2 = y2 + 1
            dates = pd.date_range(
                str(t.dt.year.data)+'-'+str(t.dt.month.data),
                str(y2)+'-'+str(t2),
                freq='1MS'
            )

            if t.dt.month.data in win:
                c = 'blue'
            elif t.dt.month.data in sum:
                c = 'red'

            ax.axvspan(dates[0], dates[1], color=c, alpha=alpha, zorder=0, **kwargs)

import xarray as xr
import matplotlib.pyplot as plt
import cmocean.cm as cmo

def plot_meridional_anomaly_hovmoller(ds, var, vm, si, mask_si=1, ax=None, **cbar_kwargs):
    """
    Plot a meridional anomaly Hovmöller diagram.
    
    This function calculates and plots the anomaly of a given variable, averaged meridionally (along the longitude).
    It also allows for masking based on sea ice concentration.

    Parameters:
    - ds (xarray.Dataset): The input dataset containing the variable to be plotted.
    - var (str): The variable name in the dataset to be plotted.
    - vm (float): The maximum absolute value for color scaling.
    - ax (matplotlib.axes.Axes, optional): The axes object to plot on. Default is None.
    - si (xarray.Dataset): Sea ice dataset used for masking.
    - mask_si (int, optional): Flag to mask the variable based on sea ice concentration. Default is 1 (masking enabled).
    - **cbar_kwargs: Additional keyword arguments for the color bar.

    Returns:
    - None: The function plots directly on the provided axes object.
    """
    
    # Calculate the anomaly by subtracting the mean of the variable
    tmp = (ds[var] - ds[var + '_mn'])
    
    # If masking based on sea ice concentration is enabled
    if mask_si:
        # Mask and calculate the meridional (longitude) mean where sea ice concentration is less than 0.15
        tmp = tmp.where(si.sic < 0.15)

    # compute statistic
    tmp = tmp.mean(dim='lon')
    
    # Limit the anomaly values to the specified range (±vm) for plotting
    tmp = tmp.where(~((tmp > vm) | (tmp < -vm)), other=vm * 0.875)
    
    # Plot the anomaly using the 'cmo.balance' colormap
    if ax==None:
        plt.figure(figsize=(15,5))
        ax = plt.gca()
    tmp.plot(x='time', cmap=get_cmap(), vmin=-vm, vmax=vm, ax=ax,
             cbar_kwargs=dict(extend='both', **cbar_kwargs))