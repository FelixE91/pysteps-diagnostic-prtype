#!/usr/bin/env python

"""Tests for `pysteps_diagnostic_prtype` package."""


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
    #    'filename'
    #    'startdate'
    #    'snowLevelData'
    #    'temperatureData'
    #    'groundTemperatureData'
    #    'modelMetadataDictionary'
    #    'topoFilename' -> should become 'topographyData' and 'topoMetadataDictionary'
    #    'nwc_projectionString'

    ### load the test data (artificial)
    # mimick the INCA basic fields transformed to a 3D array with
    # dimension (timestep,x,y)
    # timestep: 13 (analysis + 12h forecast, hourly)
    # x: 601, y: 591

    # create artifical snow level as array [m]
    snowLevelData = np.ones((13,601,591))*300

    # create artifical temperature field as array [K]
    temperatureData = np.ones((13,601,591))*280
    
    # create artifical surface temperature field as array [K]
    groundTemperatureData = np.ones((13,601,591))*270
    
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
    topo_proj = ''
    topographyData = np.zeros((601,591))
    topoMetadataDictionary = modelMetadataDictionary.copy()
    
    # test the prtype function

