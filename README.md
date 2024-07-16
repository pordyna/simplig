# SIMPLIG
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Formatter: docformatter](https://img.shields.io/badge/%20formatter-docformatter-fedcba.svg)](https://github.com/PyCQA/docformatter)
[![Style: sphinx](https://img.shields.io/badge/%20style-sphinx-0a507a.svg)](https://www.sphinx-doc.org/en/master/usage/index.html)

**Sim**ulation **Pl**asma F**ig**ures 

A **simple** way of plotting simulation mesh data from openPMD outputs.

* Load openPMD mesh data and visualize them.
* Perform basic operations on loaded data such as slicing, rotation, or axes transposition while correctly handling metadata.

Currently, using `pint` for units and own data structure. 
In the feature it will probably switch to `scipp` instead.
