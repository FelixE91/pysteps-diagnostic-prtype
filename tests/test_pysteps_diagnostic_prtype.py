#!/usr/bin/env python

"""Tests for `pysteps_diagnostic_prtype` package."""

import pandas as pd

def test_plugins_discovery():
    """It is recommended to at least test that the plugin modules provided by the plugin are
    correctly detected by pysteps. For this, the tests should be ran on the installed
    version of the plugin (and not against the plugin sources).
    """
    # plugin exists as interface method
    from pysteps.postprocessing import interface as pp_interface
    assert 'diagnostic_prtype' in pp_interface._diagnostics_methods
    # plugin exists as module
    import importlib
    available_module_methods = [
            attr
            for attr in dir(importlib.import_module('pysteps.postprocessing.diagnostics'))
        ]
    assert 'diagnostic_prtype' in available_module_methods

def test_prtype_function():
    """Additionally, you can test that your plugin correctly reads the corresponding
    some example data.
    """
    import numpy as np
    from pysteps.postprocessing.diagnostics import diagnostic_prtype
    # load function with 8 required arguments:
    #    'precip_field'
    #    'precipMetadataDictionary'
    #    'startdate'
    #    'snowLevelData'
    #    'temperatureData'
    #    'groundTemperatureData'
    #    'modelMetadataDictionary'
    #    'topographyData'
    #    'topoMetadataDictionary'

    ### load the test data (artificial)
    startdate = "204002291545" 
    # use projection and dimension as in pysteps output with RADQPE (RMI) input
    # precipitation data
    precip_field = np.random.random((12,700,700))*10
    projection = "+proj=lcc +lat_1=49.83333333333334 +lat_2=51.16666666666666 +lat_0=50.797815 +lon_0=4.359215833333333 +x_0=649328 +y_0=665262 +ellps=GRS80 +towgs84=0,0,0,0,0,0,0 +units=m +no_defs "
    precipMetadataDictionary = {
        'projection':projection,    
        'xpixelsize': 1000.0,
        'ypixelsize': 1000.0,
        'cartesian_unit':'m',
        'x1': 300000.0,
        'y1': 300000.0,
        'x2': 1000000.0,
        'y2': 1000000.0,
        'yorigin': 'upper',
        'accutime': 5.0,
        'unit': 'mm/h',
        'transform': None,
        'zerovalue': 0.0,
        'threshold': 0.10000015050172806,
        'timestamps':pd.date_range(start=pd.to_datetime(startdate,format='%Y%m%d%H%M'),periods=12,freq='5min')
        }
    
    # mimick the INCA basic fields transformed to a 3D array with
    # dimension (timestep,x,y)
    # timestep: 13 (analysis + 12h forecast, hourly)
    # x: 600, y: 590

    # create artifical snow level as array [m]
    snowLevelData = np.ones((13,600,590))*300

    # create artifical temperature field as array [K]
    temperatureData = np.ones((13,600,590))*280
    
    # create artifical surface temperature field as array [K]
    groundTemperatureData = np.ones((13,600,590))*270
    
    # Model metadata is defined in pysteps.io.importers
    ## EPSG: 3812 (Belgian Lambert 2008) projection string
    projection='+proj=lcc +lat_1=49.83333333333334 +lat_2=51.16666666666666 +lat_0=50.797815 +lon_0=4.359215833333333 +x_0=649328 +y_0=665262 +ellps=GRS80 +towgs84=0,0,0,0,0,0,0 +units=m +no_defs '
    modelMetadataDictionary = {
        'projection':projection,
        'x1':360000,
        'y1':350000,
        'x2':960000,
        'y2':940000,
        'xpixelsize':1000,
        'ypixelsize':1000,
        'cartesian_unit':'m',
        'yorigin':'upper',
        }
    
    # create artifical topo data (or read Belgium data)
    ### need to change function input: from tope filename to tope array (to match other input)
    topographyData = np.zeros((600,590))
    topoMetadataDictionary = modelMetadataDictionary.copy()
    
    # test the prtype function
    prtype_list = diagnostic_prtype(precip_field,
                          precipMetadataDictionary,
                          startdate,
                          snowLevelData,
                          temperatureData,
                          groundTemperatureData,
                          modelMetadataDictionary,
                          topographyData,
                          topoMetadataDictionary)
    print(prtype_list.shape)
    print(precip_field.shape)
    print(temperatureData.shape)
    assert prtype_list.shape == (precip_field.shape[0],)

