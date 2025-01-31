#!/usr/bin/env python

"""Tests for `pysteps_diagnostic_prtype` package."""


def test_plugins_discovery():
    """It is recommended to at least test that the plugin modules provided by the plugin are
    correctly detected by pysteps. For this, the tests should be ran on the installed
    version of the plugin (and not against the plugin sources).
    """

    from pysteps.io import interface as io_interface
    from pysteps.postprocessing import interface as pp_interface

    plugin_type = "diagnostic"
    if plugin_type == "importer":
        new_importers = ["diagnostic_prtype"]
        for importer in new_importers:
            assert importer.replace("import_", "") in io_interface._importer_methods

    elif plugin_type == "diagnostic":
        new_dagnostics = ["diagnostic_prtype"]
        for diagnostic in new_diagnostics:
            assert diagnostic in pp_interface._diagnostics_methods
            
    elif plugin_type == "ensemblestat":
        new_ensemblestats = ["diagnostic_prtype"]
        for ensemblestat in new_ensemblestats:
            assert ensemblestat in pp_interface._ensemblestats_methods

def test_importers_with_files():
    """Additionally, you can test that your plugin correctly reads the corresponding
    some example data.
    """

    # Write the test here.
    pass
