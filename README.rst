=========================
pysteps-diagnostic-prtype
=========================

Pysteps plugin for calculating the precipitation type of hydrometeors.

Note: The installed package is called "pysteps-diagnostic-prtype" (with dashes), and so is the git repository.
To handle your installed python package, use the package name (e.g., ``pip uninstall pysteps-diagnostic-prtype``)

The python module is called "pysteps_diagnostic_prtype" (with underscores, Python import friendly). To use your installed package, you'd run ``import pysteps_diagnostic_prtype``.


License
=======
* BSD license



Documentation
=============

This is a plugin designed for implementation alongside the pySTEPS package. This plugin contains functions which will allow for the calculation of the precipitation type of the hydrometeors present in a pySTEPS nowcast. In order to use this functionality, the user must provide a pySTEPS nowcast as well as arrays featuring the snowfall level, air temperature, and surface temperature data of the region covered by the nowcast. A digital elevation model of the region and the metadata of the data will also be required. The plugin is weather model independent and, as such, the user will have to utilize their own data importer to extract the required information from their weather model. An example data importer is provided in the docs folder which can be used to extract the required data from INCA grib files.

Installation instructions
=========================

This plugin can be installed using:

.. code-block:: console

   git clone git@github.com:FelixE91/pysteps-diagnostic-prtype.git

and then install with

.. code-block:: console

   pip install pysteps-diagnostic-prtype/

Credits
=======

- This package was created with Cookiecutter_ and the `pysteps/cookiecutter-pysteps-plugin`_ project template.

- It is a copy of `original_prtype`_ precipitation type plugin that used an older version of the `cookiecutter-pypackage`_ and was created by @JoeyCasey87

.. Since this plugin template is based in the cookiecutter-pypackage template,
it is encouraged to leave the following credits to acknowledge Audrey Greenfeld's work.

- The `pysteps/cookiecutter-pysteps-plugin`_ template was adapted from the cookiecutter-pypackage_
template.

.. _cookiecutter-pypackage: https://github.com/audreyfeldroy/cookiecutter-pypackage
.. _original_prtype: https://github.com/joeycasey87/pysteps_postprocessor_diagnostics_prtype
.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`pysteps/cookiecutter-pysteps-plugin`: https://github.com/pysteps/cookiecutter-pysteps-plugin
