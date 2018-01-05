# pyMETRIC

pyMETRIC is a set of Python based tools for estimating and mapping evapotranspiration (ET).  It currently computes ET estimates using the [METRIC](http://www.uidaho.edu/cals/kimberly-research-and-extension-center/research/water-resources) surface energy balance model, developed at the University of Idaho.

## Example

A detailed walk-through on the setup and operation of pyMETRIC has been assembled in a series of documentation.  These examples are setup to process a portion of the Harney Basin, located in eastern Oregon.  The documentation is contained in the following links:
1. [Data Preparation](docs/EXAMPLE_DATA.md)
2. [Project Setup](docs/EXAMPLE_SETUP.md)
3. [Running METRIC](docs/EXAMPLE_METRIC.md)

## Install

Details on installing pyMETRIC, Python, and necessary modules can be found in the [installation instructions.](docs/INSTALL.md).

## References
* [Satellite-Based Energy Balance for Mapping Evapotranspiration with Internalized Calibration (METRIC)—Model](https://ascelibrary.org/doi/abs/10.1061/(ASCE)0733-9437(2007)133:4(380))
* [Satellite-Based Energy Balance for Mapping Evapotranspiration with Internalized Calibration (METRIC)—Applications](https://ascelibrary.org/doi/abs/10.1061/(ASCE)0733-9437(2007)133:4(395))
* [Assessing calibration uncertainty and automation for estimating evapotranspiration from agricultural areas using METRIC](https://www.dri.edu/images/stories/divisions/dhs/dhsfaculty/Justin-Huntington/Morton_et_al._2013.pdf)

## Limitations
METRIC requires an assemblage of several datasets in order to produce accurate estimates of evapotranpsiration.  The pyMETRIC framework serve to download and process the required data.  Please note that this code is written for the data as it is currently provided, however the data and it’s formatting is controlled by the data providers and by third-party hosts.  The maintainers of pyMETRIC will attempt to keep the package functional, however changes in the data and data availability may impact the functionality of pyMETRIC.
