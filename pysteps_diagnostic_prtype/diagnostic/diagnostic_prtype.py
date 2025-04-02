# -*- coding: utf-8 -*-
"""
pysteps.postprocessing.diagnostics.diagnostic_prtype
======================
Precipitation Type calculator.

This plugin allows a user to calculate the precipitation types of the hydro-meteors detected in a pysteps blended
nowcast through the use of both the nowcast data and snow level, temperature, and ground temperature data taken from
another weather model, such as INCA or Cosmo.
"""

import os
import numpy as np
import datetime

import matplotlib.pyplot as plt
import matplotlib.pylab as plb
from matplotlib import colors

from pysteps.utils.reprojection import reproject_grids
from pysteps.io import import_netcdf_pysteps
from pysteps.visualization import get_geogrid, get_basemap_axis


def diagnostic_prtype(filename,
                      startdate,
                      snowLevelData,
                      temperatureData,
                      groundTemperatureData,
                      modelMetadataDictionary,
                      topoFilename,
                      nwc_projectionString,
                      members=None,
                      timeBase=None,
                      timeStep=None,
                      topo_interpolation=None,
                      desired_output=None,
                      **kwargs):
    """
    Calculate the precipitation types for ensemble data at particular time steps from a combination of a pysteps nowcast
    and external model data (such as from INCA or COSMO).

    Parameters
    ----------

    filename : str
      Path and name of the NetCDF file to import.
      The NetCDF files and the projection files must be for the same date and time.

    startdate : datetime
      The time and date of the model files in the format "%Y%m%d%H%M" e.g. 202305010000 for midnight on the 1st of
      May 2023.

    snowLevelData: 3D Array
      Data should be in the form of a 3D matrix. [time step, X-coord, Y-coord]

    temperatureData: 3D Array
      Data should be in the form of a 3D matrix. [time step, X-coord, Y-coord]

    groundTemperatureData: 3D Array
      Data should be in the form of a 3D matrix. [time step, X-coord, Y-coord]

    modelMetadataDictionary: dict
      A dictionary containing the metadata for the snow level, temperature, and ground temperature data.

    topoFilename : str
      The path to the model topography file. The topography file is required to be in a text readable format such as
      .asc

    nwc_projectionString : str
      The nowcast projection string.

    members : int The number of ensemble members which you would like to receive the results of. This number can be
    between 1 and 26.

    timeBase : int
      The base for the time period. (min)

    timeStep : int
      The step for the time period. (min)

    topo_interpolation : boolean
      Whether the topography data requires interpolation. Default no, but can be adjusted
      to yes. The topography data will require interpolation if it does not already have the same shape as that of the
      weather model data.

    desired_output : str
      The desired output is an indicator used by the user to indicate their desired output parameter.
      Currently, the function only features the optional outputs of full arrays of the number of members.
      This option is chosen by the string 'members'.
      Or the mean precipitation type across all members.
      This option is chosen by the string 'mean',
      This input field could be extended in the future to allow users to produce other desired outputs.

    {extra_kwargs_doc}

    Returns
    -------
    output:
        A 3 or 4D array containing the desired output of the user.
        Output arrays take the form [member, timeStamp, X-coord, Y-coord].
        Or [timeStep, X-coord, Y-coord] in the case of mean calculations.
    """

    # Run checks to ensure correct input parameters
    if not isinstance(filename, str):
        raise TypeError("Filename must be a string containing the path to the netCDF file")
    if filename.rsplit('.', 1)[1] != 'nc':
        raise ValueError("Filename must be a netCDF file ending in '.nc'")

    if not isinstance(modelMetadataDictionary, dict):
        raise TypeError(
            "modelMetadataDictionary must be a dictionary containing the metadata about the snow level, temperature, "
            "and ground temperature files")

    if not isinstance(topoFilename, str):
        raise TypeError("topoFilename must be a string containing the path to the topography file")

    if nwc_projectionString is not None and not isinstance(nwc_projectionString, str):
        raise TypeError("nwc_projectionString must be a string containing the NWC projection string")

    if members is not None and not isinstance(members, int):
        raise TypeError("members must be an integer")
    if not members > 0 or not members < 27:
        raise ValueError("members must be a positive integer between 1 and 26 inclusive")

    if timeBase is not None and not isinstance(timeBase, int):
        raise TypeError("timeBase must be a positive integer")
    if not timeBase > 0:
        raise ValueError("timeBase must be a positive integer")

    if timeStep is not None and not isinstance(timeStep, int):
        raise TypeError("timeStep must be a positive integer")
    if not timeStep > 0:
        raise ValueError("timeStep must be a positive integer")

    if topo_interpolation is not None and not isinstance(topo_interpolation, bool):
        raise TypeError("topo_interpolation must be a boolean")

    if desired_output is not None and not isinstance(desired_output, str):
        raise TypeError("desired_output must be one of the strings in the list [\"mean\", \"members\"]")
    if desired_output not in ["mean", "members"]:
        raise ValueError("desired_output must be one of the strings in the list [\"mean\", \"members\"]")

    ####################################################################################

    # Define default parameter values
    if members is None:
        members = 1
    if timeBase is None:
        timeBase = 60
    if timeStep is None:
        timeStep = 5
    if topo_interpolation is None:
        topo_interpolation = False
    if desired_output is None:
        desired_output = 'members'

    # --------------------------------------------------------------------------

    # Load Data
    R_ZS = snowLevelData
    R_TT = temperatureData
    R_TG = groundTemperatureData
    print('Data load done')

    # --------------------------------------------------------------------------

    # Load Topography
    topo_grid = np.loadtxt(topoFilename)
    topo_grid = topo_grid[::-1, :]  # Reorientation
    print('Topography load done')

    # ---------------------------------------------------------------------------

    # Load PYSTEPS data

    # import netCDF file
    r_nwc, metadata_nwc = import_netcdf_pysteps(filename)

    # Set Metadata info
    metadata_nwc['projection'] = nwc_projectionString
    metadata_nwc['cartesian_unit'] = metadata_nwc['projection'][
                                     nwc_projectionString.find('units=') + 6:
                                     nwc_projectionString.find(
                                         ' +no_defs', nwc_projectionString.find(
                                             'units=') + 6)]
    print('netCDF4 load done')

    # --------------------------------------------------------------------------

    # Reproject
    #     projection over pySTEPS grid
    R_ZS, _ = reproject_grids(R_ZS, r_nwc[0, 0, :, :], modelMetadataDictionary, metadata_nwc)
    R_TT, _ = reproject_grids(R_TT, r_nwc[0, 0, :, :], modelMetadataDictionary, metadata_nwc)
    R_TG, _ = reproject_grids(R_TG, r_nwc[0, 0, :, :], modelMetadataDictionary, metadata_nwc)
    # The topography file is not required to be interpolated if it is already of the shape matching the grib files,
    # i.e (591, 601)
    if topo_interpolation is True:
        topo_grid, _ = reproject_grids(np.array([topo_grid]), r_nwc[0, 0, :, :], modelMetadataDictionary, metadata_nwc)
    print('Re-projection done')

    # --------------------------------------------------------------------------

    # Calculate interpolation matrices

    # Calculate interpolations values for matching timestamps between model and pySTEPS
    interpolations_ZS, timestamps_idxs = generate_interpolations(R_ZS, metadata_nwc['timestamps'],
                                                                 startdate, timeStep, timeBase)
    interpolations_TT, _ = generate_interpolations(R_TT, metadata_nwc['timestamps'], startdate, timeStep,
                                                   timeBase)
    interpolations_TG, _ = generate_interpolations(R_TG, metadata_nwc['timestamps'], startdate, timeStep,
                                                   timeBase)
    print("Interpolation done!")

    # Clean (After interpolation, we don't need the reprojected data anymore)
    del R_ZS, R_TT,

    # --------------------------------------------------------------------------

    # Diagnose precipitation type per member over time, using mean mask

    # WARNING (1): The grids have been sub-scripted to the model size. This requires the model metadata to
    # be used for plotting. If the original PYSTEPS grid size is used (700x700) for plotting, the pysteps metadata_nwc
    # should be used instead.
    #
    # WARNING (2): Topography does not need to be re-projected if it matches the grid size of the model.

    print("Calculate precipitation type per member over time...")

    # Find subscript indexes for model grid
    x1, x2, y1, y2 = get_reprojected_indexes(interpolations_ZS[0])

    # Result list
    ptype_list = np.zeros((r_nwc.shape[0] + 1, r_nwc.shape[1], x2 - x1, y2 - y1))

    # loop over timestamps
    for ts in range(len(timestamps_idxs)):
        print("Calculating precipitation types at: ", str(timestamps_idxs[ts]))

        # Members Mean matrix
        r_nwc_mean = calculate_members_mean(r_nwc[:, ts, x1:x2, y1:y2])

        # calculate precipitation type result with members mean
        ptype_mean = calculate_precip_type(Znow=interpolations_ZS[ts, x1:x2, y1:y2],
                                           Temp=interpolations_TT[ts, x1:x2, y1:y2],
                                           GroundTemp=interpolations_TG[ts, x1:x2, y1:y2],
                                           precipGrid=r_nwc_mean,
                                           topographyGrid=topo_grid)

        # Intersect precipitation type by member using ptype_mean
        for member in range(0, members):
            res = np.copy(ptype_mean)
            res[r_nwc[member, ts, x1:x2, y1:y2] == 0] = 0
            ptype_list[member, ts, :, :] = res

        # Add mean result at the end
        ptype_list[-1, ts, :, :] = ptype_mean

    if desired_output == 'members':
        output = ptype_list[:members]
    else:
        output = ptype_list[-1]

    print("--Script finished--")
    return output


def plot_precipType_field(
        precipType,
        ax=None,
        geodata=None,
        bbox=None,
        colorscale="pysteps",
        title=None,
        colorbar=True,
        cBarLabel="",
        categoryNr=4,
        axis="on",
        cax=None,
        map_kwargs=None,
):
    """
    Function to plot a precipitation types field with a colorbar.

    .. _Axes: https://matplotlib.org/api/axes_api.html#matplotlib.axes.Axes

    .. _SubplotSpec: https://matplotlib.org/api/_as_gen/matplotlib.gridspec.SubplotSpec.html

    Parameters
    ----------
    precipType: array-like
        Two-dimensional array containing the input precipitation types.
    ax: fig Axes_
        Axes for the basemap.
    geodata: dictionary or None, optional
        Optional dictionary containing geographical information about
        the field. Required is map is not None.

        If geodata is not None, it must contain the following key-value pairs:

        .. tabularcolumns:: |p{1.5cm}|L|

        +-----------------+---------------------------------------------------+
        |        Key      |                  Value                            |
        +=================+===================================================+
        |    projection   | PROJ.4-compatible projection definition           |
        +-----------------+---------------------------------------------------+
        |    x1           | x-coordinate of the lower-left corner of the data |
        |                 | raster                                            |
        +-----------------+---------------------------------------------------+
        |    y1           | y-coordinate of the lower-left corner of the data |
        |                 | raster                                            |
        +-----------------+---------------------------------------------------+
        |    x2           | x-coordinate of the upper-right corner of the     |
        |                 | data raster                                       |
        +-----------------+---------------------------------------------------+
        |    y2           | y-coordinate of the upper-right corner of the     |
        |                 | data raster                                       |
        +-----------------+---------------------------------------------------+
        |    yorigin      | a string specifying the location of the first     |
        |                 | element in the data raster w.r.t. y-axis:         |
        |                 | 'upper' = upper border, 'lower' = lower border    |
        +-----------------+---------------------------------------------------+
    bbox : tuple, optional
        Four-element tuple specifying the coordinates of the bounding box. Use
        this for plotting a subdomain inside the input grid. The coordinates are
        of the form (lower left x, lower left y ,upper right x, upper right y).
        If 'geodata' is not None, the bbox is in map coordinates, otherwise
        it represents image pixels.
    colorscale : {'pysteps', 'STEPS-BE', 'STEPS-NL', 'BOM-RF3'}, optional
        Which colorscale to use. TO BE DEFINED
    title : str, optional
        If not None, print the title on top of the plot.
    colorbar : bool, optional
        If set to True, add a colorbar on the right side of the plot.
    cBarLabel :
        Set color bar label.
    categoryNr :
        Number of categories to be plotted (2 to 6)
    axis : {'off','on'}, optional
        Whether to turn off or on the x and y axis.
    cax : Axes_ object, optional
        Axes into which the colorbar will be drawn. If no axes is provided
        the colorbar axes are created next to the plot.

    Other parameters
    ----------------
    map_kwargs: dict
        Optional parameters that need to be passed to
        :py:func:`pysteps.visualization.basemaps.plot_geography`.

    Returns
    -------
    ax : fig Axes_
        Figure axes. Needed if one wants to add e.g. text inside the plot.
    """

    if map_kwargs is None:
        map_kwargs = {}

    if len(precipType.shape) != 2:
        raise ValueError("The input is not two-dimensional array")

    # Assumes the input dimensions are lat/lon
    nlat, nlon = precipType.shape

    x_grid, y_grid, extent, regular_grid, origin = get_geogrid(
        nlat, nlon, geodata=geodata
    )

    ax = get_basemap_axis(extent, ax=ax, geodata=geodata, map_kwargs=map_kwargs)

    precipType = np.ma.masked_invalid(precipType)
    # plot rainfield
    if regular_grid:
        im = _plot_field(precipType, ax, colorscale, categoryNr, extent, origin=origin)
    else:
        im = _plot_field(
            precipType, ax, colorscale, categoryNr, extent, x_grid=x_grid, y_grid=y_grid
        )

    plb.title(title, loc='center', fontsize=25)

    # add colorbar
    cbar = None
    if colorbar:
        # get colormap and color levels
        _, _, clevs, clevs_str = get_colormap(colorscale, categoryNr)
        cbar = plb.colorbar(
            im, ticks=clevs, spacing="uniform", extend="neither", shrink=0.8, cax=cax, drawedges=False
        )
        if clevs_str is not None:
            cbar.ax.set_yticklabels('')
            cbar.ax.tick_params(size=0)
            cbar.ax.set_yticks([i + .5 for i in clevs][:-1], minor=True)
            cbar.ax.set_yticklabels(clevs_str[:-1], minor=True, fontsize=15)
    cbar.set_label(cBarLabel)

    if geodata is None or axis == "off":
        ax.xaxis.set_ticks([])
        ax.xaxis.set_ticklabels([])
        ax.yaxis.set_ticks([])
        ax.yaxis.set_ticklabels([])

    if bbox is not None:
        ax.set_xlim(bbox[0], bbox[2])
        ax.set_ylim(bbox[1], bbox[3])

    return ax


def _plot_field(precipType, ax, colorscale, categoryNr, extent, origin=None, x_grid=None, y_grid=None):
    precipType = precipType.copy()

    # Get colormap and color levels
    cmap, norm, _, _ = get_colormap(colorscale, categoryNr)

    if (x_grid is None) or (y_grid is None):
        im = ax.imshow(
            precipType,
            cmap=cmap,
            norm=norm,
            extent=extent,
            interpolation="nearest",
            origin=origin,
            zorder=10,
        )
    else:
        im = ax.pcolormesh(
            x_grid,
            y_grid,
            precipType,
            cmap=cmap,
            norm=norm,
            zorder=10,
        )

    return im


def get_colormap(colorscale="pysteps", categoryNr=4):
    """
    Function to generate a colormap (cmap) and norm.

    Parameters
    ----------

    colorscale : {'pysteps', 'STEPS-BE', 'STEPS-NL', 'BOM-RF3'}, optional
      Which colorscale to use. Applicable if units is 'mm/h', 'mm' or 'dBZ'.

    Returns
    -------
    cmap : Colormap instance
      colormap
    norm : colors.Normalize object
      Colors norm
    clevs: list(float)
      List of precipitation values defining the color limits.
    clevs_str: list(str)
      List of precipitation values defining the color limits (with correct
      number of decimals).
      :param categoryNr:
    """
    # Get list of colors
    color_list, clevs, clevs_str = _get_colorlist(colorscale, categoryNr)
    cmap = colors.LinearSegmentedColormap.from_list(
        "cmap", color_list, len(clevs) - 1
    )
    cmap.set_over("darkred", 1)
    cmap.set_bad("gray", alpha=0.5)
    cmap.set_under("none")
    norm = colors.BoundaryNorm(clevs, cmap.N)

    return cmap, norm, clevs, clevs_str


def _get_colorlist(colorscale="pysteps", categoryNr=4):
    """
    Function to get a list of colors to generate the colormap.

    Parameters
    ----------
    colorscale : str
        Which colorscale to use (BOM-RF3, pysteps, STEPS-BE, STEPS-NL)
    categoryNr  :
        How many categories should be plotted

    Returns
    -------
    color_list : list(str)
        List of color strings.

    clevs : list(float)
        List of precipitation values defining the color limits.

    clevs_str : list(str)
        List of precipitation type names
    """

    if categoryNr < 1 or categoryNr > 6:
        raise ValueError("Invalid category index [1 to 6] " + str(categoryNr))

    if colorscale == "pysteps":
        color_list = ["#ffe38f", "#ceda86", "#009489", "#3897ed", "#b0a0dc", "#ec623b"]
    # elif colorscale == 'other color scale': ... [6 colors]
    else:
        print("Invalid colorscale", colorscale)
        raise ValueError("Invalid colorscale " + colorscale)

    # Ticks and labels
    clevs = [1, 2, 3, 4, 5, 6, 7]
    clevs_str = ['Rain', 'Wet Snow', 'Snow', 'Freezing Rain', 'Hail', 'Severe Hail']

    # filter by category number
    color_list = color_list[0:categoryNr]
    clevs = clevs[0:(categoryNr + 1)]
    clevs_str = clevs_str[0:categoryNr] + ['']

    return color_list, clevs, clevs_str


def plot_ptype(ptype_grid, metadata, i, date_time, dir_gif, categoryNr=4):
    title = 'Precipitation type ' + date_time.strftime("%Y-%m-%d %H:%M")
    fig = plt.figure(figsize=(15, 15))
    # fig.add_subplot(1, 1, 1)
    plot_precipType_field(ptype_grid, geodata=metadata, title=title, colorscale="pysteps", categoryNr=categoryNr)
    # plt.suptitle('Precipitation Type', fontsize=30)
    plt.tight_layout()
    filename = f'{i}.png'
    #  filenames.append(filename)
    plt.savefig(os.path.join(dir_gif, filename), dpi=72)
    plt.close()
    return filename


def calculate_precip_type(Znow, Temp, GroundTemp, precipGrid, topographyGrid, DZML=100., TT0=2., TG0=0.,
                          RRMIN=0):
    """Precipitation type algorithm, returns a 2D matrix with categorical values:
    # PT=0  no precip
    # PT=1  rain
    # PT=2  rain/snow mix
    # PT=3  snow
    # PT=4  freezing rain

    Znow:
        snow level 2D grid
    Temp:
        temperature 2D grid
    GroundTemp:
        ground temperature 2D grid
    precipGrid:
        Precipitation (netCDF PYSTEPS) 2D grid
    topographyGrid:
        Topography grid 2D
    DZML:
        thickness of the melting layer (default 100m)
    TT0:
        temperature threshold below which rain will freeze when it hits a freezing ground (default 2C)
    TG0:
        temperature threshold below which we have freezing rain or ground (default 0C)
    RRMIN:
        minimum precipitation threshold (default 0mm/h)

    returns:
        2D matrix with categorical data for each type
    """

    # Result grid
    result = np.zeros((precipGrid.shape[0], precipGrid.shape[1]))
    topoZSDiffGrid = (Znow - topographyGrid)  # dzs
    precipMask = (precipGrid > RRMIN)

    # SNOW ((dzs<-1.5*DZML) || ( (ZH[i][j] <= 1.5*DZML) && (dzs<=0)))
    snowMask = (topoZSDiffGrid < (-1.5 * DZML)) | ((topographyGrid <= (1.5 * DZML)) & (topoZSDiffGrid <= 0))
    result[snowMask & precipMask] = 3

    # RAIN+SNOW DIAGNOSIS (dzs < 0.5 * DZML) = 2
    rainSnowMask = ~snowMask & (topoZSDiffGrid < (0.5 * DZML))
    result[rainSnowMask & precipMask] = 2

    # RAIN
    rainMask = ~snowMask & ~rainSnowMask
    result[rainMask & precipMask] = 1

    # FREEZING RAIN DIAGNOSIS 4
    # if ((PT[i][j]==1) && ( (tg_<TG0 && TT[i][j]<TT0) || TT[i][j]<TG0))
    freezingMask = (result == 1) & (((GroundTemp < TG0) & (Temp < TT0)) | (Temp < TG0))
    result[freezingMask] = 4

    return result


def calculate_members_mean(membersData):
    """Function to calculate the members average over time

    membersData:
        3D matrix composed by [members, grid dimension 1, grid dimension 2]
    """

    if len(membersData.shape) != 3:
        raise ValueError("Invalid members data shape (expected [:,:,:]) " + str(membersData.shape))

    meanMatrix = np.zeros((membersData.shape[1], membersData.shape[2]))
    for member_idx in range(membersData.shape[0]):
        meanMatrix = meanMatrix + membersData[member_idx, :, :]
    meanMatrix = meanMatrix / membersData.shape[0]
    # print('Mean member matrix done!')

    return meanMatrix


def get_reprojected_indexes(reprojectedGrid):
    """Reprojected model grids contains a frame of NAN values, this function returns the start and end indexes
    of the model grid over the reprojected grid

    reprojectedGrid:
        model reprojected Grid

    ---
    Returns:
        x y indexes of model reprojected grid over pysteps dimensions
    """

    x_start = np.where(~np.isnan(reprojectedGrid))[0][0]
    x_end = np.where(~np.isnan(reprojectedGrid))[0][-1] + 1
    y_start = np.where(~np.isnan(reprojectedGrid))[-1][0]
    y_end = np.where(~np.isnan(reprojectedGrid))[-1][-1] + 1

    return x_start, x_end, y_start, y_end


def grid_interpolation(numpyGridStart, numpyGridEnd, timeStep=5, timeBase=60):
    """ Time interpolation between 2 2D grids

    numpyGridStart:
        Numpy 2-D grid of start values
    numpyGridEnd:
        Numpy 2-D grid of end values
    timeStep:
        Size of the time step for interpolation (every 5, 10, 15. min)
    timeBase:
        Time period considered in minutes (e.g. over one hour = 60, 2 hours = 120)
    applyOver:
        Array with sub-indexes to calculate interpolation (inner grid)
    ----

    Return:
        Returns a list of 3D numpy interpolation matrix
    """
    if numpyGridStart.shape != numpyGridEnd.shape:
        raise ValueError("ERROR: Grids have different dimensions")

    interPoints = np.arange(0, (timeBase + timeStep), timeStep)
    interpolationGrid = np.zeros((len(interPoints), numpyGridStart.shape[0], numpyGridStart.shape[1]))
    interpolationGrid[:, :, :] = np.nan

    # print('Calculating linear interpolation..', end=' ')
    for i in range(len(interPoints)):
        interpolationGrid[i, :, :] = numpyGridStart + ((numpyGridEnd - numpyGridStart) / interPoints[-1]) * interPoints[
            i]
    # print('Done')

    return interpolationGrid


def create_timestamp_indexing(nrOfModelMessages, startDateTime, timeStep=5, timeBase=60):
    """create a timestamp array for model indexing

    nrOfModelMessages:
        Number of model available messages

    startDateTime:
        Start date and time

    timeStep:
        Defines the size of the time step for interpolation

    timeBase:
        Time between messages in minutes

    ___
    Return:
          Array of timestamps similar to pysteps timestamps
    """

    if nrOfModelMessages < 2:
        raise ValueError("Not enough interpolation messages, should be at least 2")

    result = []
    timestamp = startDateTime
    interPoints = np.arange(0, (timeBase + timeStep), timeStep)

    for i in range(nrOfModelMessages - 1):
        for j in interPoints[:-1]:
            result.append(timestamp)
            timestamp = timestamp + datetime.timedelta(minutes=timeStep)

    result.append(timestamp)
    return np.array(result)


def generate_interpolations(model_reprojected_data, nwc_timestamps, startdate, timeStep=5, timeBase=60,
                            dateFormat='%Y%m%d%H%M'):
    """Generate a sub-selection of the interpolation matrix for all messages available from model data

    model_reprojected_data:
        model reprojected data.

    model_timestamps:
        Array of timestamps every timeSteps period.

    nwc_timestamps:
        Array of timestamps available from PYSTEPS metadata ['timestamps']

    ----
    Return:
        3D matrix with depth equal to the common matching timestamps between the model and PYSTEPS.

    """

    # Create a timestamp index array for model interpolation matrix
    model_timestamps = create_timestamp_indexing(model_reprojected_data.shape[0], startdate, timeStep=timeStep,
                                                 timeBase=timeBase)
    # Convert metadata_nwc['timestamps'] to datetime
    nwc_ts = [datetime.datetime.strptime(ts.strftime(dateFormat), dateFormat) for ts in nwc_timestamps]

    model_start = np.where(model_timestamps == nwc_ts[0])[0][0]
    model_end = np.where(model_timestamps == nwc_ts[-1])[0][0] + 1
    timestamp_selection = model_timestamps[model_start:model_end]  # to be returned

    # interpolation indexes
    resultMatrix = np.zeros(
        (model_start + len(timestamp_selection), model_reprojected_data.shape[1], model_reprojected_data.shape[2]))
    result_idx = 0

    # loop over the messages
    for m in range(1, model_reprojected_data.shape[0]):
        if result_idx < resultMatrix.shape[0]:
            # calculate interpolations
            interpolationMatrix = grid_interpolation(model_reprojected_data[m - 1], model_reprojected_data[m],
                                                     timeStep=timeStep, timeBase=timeBase)
            interp_idx = 0
            # Add the interpolation values to the result matrix (this assignment can be done without looping...)
            while interp_idx < interpolationMatrix.shape[0] and (result_idx < resultMatrix.shape[0]):
                resultMatrix[result_idx, :, :] = interpolationMatrix[interp_idx, :, :]
                result_idx = result_idx + 1
                interp_idx = interp_idx + 1
            result_idx = result_idx - 1  # overwrite the last value

            return resultMatrix[model_start:], timestamp_selection