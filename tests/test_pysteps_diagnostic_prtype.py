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
    from pysteps.postprocessing.diagnostics import diagnostic_prtype
    # load function with 8 arguments:
    #    'filename'
    #    'startdate'
    #    'snowLevelData'
    #    'temperatureData'
    #    'groundTemperatureData'
    #    'modelMetadataDictionary'
    #    'topoFilename'
    #    'nwc_projectionString'

    ### load the test data (artificial) -> see example 

    # create artifical snow level as array

    # create artifical temperature field as array
    
    # create artifical topo data (or read Belgium data)
    ### need to change function input: from tope filename to tope array (to match other input)

    # Write the test here.
    pass
