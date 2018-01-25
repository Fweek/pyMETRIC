## Installation

pyMETRIC is most easily installed by cloning the [GitHub repository](https://github.com/cgmorton/pyMETRIC).

Most of the documentation and examples are written assuming you are running pyMETRIC on a Windows PC and that the pyMETRIC repository was cloned directly to the D: drive.  If you are using a different operating system or cloned the repository to a different location, you will need adjust commands, drive letters, and paths accordingly.

## Python

pyMETRIC has only been tested using Python 2.7 and 3.6, but may work with other versions.

## Dependencies

The following external Python modules must be present to run pyMETRIC:
* [future](https://pypi.python.org/pypi/future) (adds features from Python 3 to Python 2 installations)
* [requests](http://docs.python-requests.org/en/master/) (adds enhanced http functionality)
* [scipy](https://www.scipy.org/) (provides numerous packages required for the processing of data)
* [pandas](http://pandas.pydata.org) (used to perform data processing) 
* [matplotlib](https://matplotlib.org/) (necessary for creating plots of ET related data)
* [gdal](http://www.gdal.org/) (version >2.0) (the Geospatial Data Abstraction Library is used to interact with raster and vector geospatial data)
* [netcdf4](https://www.unidata.ucar.edu/software/netcdf/) (for interacting with multi-dimensional scientific datasets, such as GRIDMET/DAYMET)
* [Landsat578](https://github.com/dgketchum/Landsat578) (for downloading Landsat images)

Please see the [requirements](../requirements.txt) file for details on the versioning requirements.  The module version numbers listed in the file were tested and are known to work.  Other combinations of versions may work but have not been tested.

### Python 2
The following external Python modules must be present to run pyMETRIC on Python 2
* [configparser]()(Python 2 implementation of the Python 3 configparser module)

## Anaconda

The easiest way of obtaining Python and all of the necessary external modules, is to install [Anaconda](https://www.anaconda.com/download/) or [Miniconda](https://conda.io/miniconda.html).

After installing Anaconda, make sure to add the [conda-forge](https://conda-forge.github.io/) channel by entering the following in the command prompt or terminal:
```
> conda config --add channels conda-forge
```

## Conda Environment

The user is strongly encouraged to setup a dedicated conda environment for pyMETRIC:
```
conda create -n pyMETRIC python=3.6
```

The environment must be "activated" before use:
```
activate pyMETRIC
```

The external modules can then be installed by calling:
```
> conda install numpy scipy pandas matplotlib gdal netcdf4 future requests
```

The Landsat download module (Landsat578) must be installed separately with pip:
```
> pip install Landsat578
```

## Environment Variables

#### PYTHONPATH

Many of the pyMETRIC scripts reference the "common" functions in the [pyMETRIC/code/support](code/support) folder.  To be able to access these functions, you will need to add/append this path to the PYTHONPATH environment variable.

The environment variable can be set at the command line.  First check if PYTHONPATH is already set by typing:
```
echo %PYTHONPATH%
```
If PYTHONPATH is not set, type the following in the command prompt:
```
> setx PYTHONPATH "D:\pyMETRIC\code\support"
```
To append to an existing PYTHONPATH, type:
```
setx PYTHONPATH "D:\pyMETRIC\code\support;%PYTHONPATH%"
```

#### GDAL_DATA

In order to execute pyMETRIC code, the GDAL_DATA environmental variable may need to be set (*example*: GDAL_DATA = C:\Anaconda3\Library\share\gdal). **Depending on your specific installation of Python on pyMETRIC, you file path for GDAL_DATA may be different**


On a Windows PC, the user environment variables can be set through the Control Panel (System -> Advanced system settings -> Environment Variables).  Assuming that pyMETRIC was cloned/installed directly to the D: drive and Python 3 is used, the GDAL_DATA environmental variable may be set as:
```
C:\Anaconda3\Library\share\gdal
```

This environment variable can also be set at the command line.  First check if GDAL_DATA is already set by typing:
```
echo %GDAL_DATA%
```
If GDAL_DATA is not set, type the following in the command prompt:
```
> setx GDAL_DATA "C:\Anaconda3\Library\share\gdal" 
```
