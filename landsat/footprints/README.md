The Landsat WRS2 Descending footprint shapefile can be downloaded directly from the [Landsat website](https://landsat.usgs.gov/pathrow-shapefiles) or using the [download script](code/download/landsat_footprints.py) with the following command:

```
D:\pyMETRIC>python code\download\download_footprints.py
```

The file "wrs2_tile_utm_zones.json" was generated from the bulk metadata XML files.  Path/rows that did not have any images in the metadata XML files were removed.  Please refer to the [cloud-free-scene-counts repository](https://github.com/Open-ET/cloud-free-scene-counts) for additional details on acquiring the bulk metadata XML files.
