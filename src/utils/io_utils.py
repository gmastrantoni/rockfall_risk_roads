"""
I/O Utilities

This module provides functions for reading and writing geospatial data
for the rockfall risk assessment workflow.
"""

import os
import geopandas as gpd
import rioxarray as rxr
import numpy as np
from pathlib import Path
from typing import Union, Optional, Dict, Any


def read_vector(
    file_path: str,
    crs: Optional[str] = "EPSG:32633"
) -> gpd.GeoDataFrame:
    """
    Read vector data from file and ensure correct CRS.

    Parameters
    ----------
    file_path : str
        Path to the vector file
    crs : str, optional
        Coordinate reference system to use, by default "EPSG:32633" (UTM Zone 33N)

    Returns
    -------
    gpd.GeoDataFrame
        GeoDataFrame containing the vector data

    Raises
    ------
    FileNotFoundError
        If the file does not exist
    ValueError
        If the file format is not supported
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    # Check file extension to determine format
    ext = os.path.splitext(file_path)[1].lower()
    supported_formats = {
        '.shp': 'ESRI Shapefile',
        '.geojson': 'GeoJSON',
        '.gpkg': 'GPKG',
        '.gml': 'GML',
        '.kml': 'KML'
    }
    
    if ext not in supported_formats:
        raise ValueError(f"Unsupported file format: {ext}. Supported formats: {list(supported_formats.keys())}")
    
    # Read vector data
    gdf = gpd.read_file(file_path)
    
    # Reproject if needed
    if crs and gdf.crs != crs:
        gdf = gdf.to_crs(crs)
    
    return gdf


def read_raster(
    file_path: str,
    band: Optional[int] = 1,
    crs: Optional[str] = "EPSG:32633"
) -> rxr.raster_array.RasterArray:
    """
    Read raster data from file and ensure correct CRS.

    Parameters
    ----------
    file_path : str
        Path to the raster file
    band : int, optional
        Band to read, by default 1
    crs : str, optional
        Coordinate reference system to use, by default "EPSG:32633" (UTM Zone 33N)

    Returns
    -------
    rxr.raster_array.RasterArray
        RasterArray containing the raster data

    Raises
    ------
    FileNotFoundError
        If the file does not exist
    ValueError
        If the file format is not supported
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    # Check file extension to determine format
    ext = os.path.splitext(file_path)[1].lower()
    supported_formats = ['.tif', '.tiff', '.asc', '.nc']
    
    if ext not in supported_formats:
        raise ValueError(f"Unsupported file format: {ext}. Supported formats: {supported_formats}")
    
    # Read raster data
    raster = rxr.open_rasterio(file_path, masked=True)
    
    # Select band
    raster = raster[band-1]
    
    # Reproject if needed
    if crs and raster.rio.crs != crs:
        raster = raster.rio.reproject(crs)
        print(f"Reprojected raster to {crs}")
    
    return raster


def write_vector(
    gdf: gpd.GeoDataFrame,
    output_path: str,
    driver: Optional[str] = None
) -> None:
    """
    Write GeoDataFrame to file.

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        GeoDataFrame to write
    output_path : str
        Path to write the file
    driver : str, optional
        Driver to use for writing, by default None (auto-detected from extension)

    Raises
    ------
    ValueError
        If the file format is not supported
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Check file extension to determine format
    ext = os.path.splitext(output_path)[1].lower()
    supported_formats = {
        '.shp': 'ESRI Shapefile',
        '.geojson': 'GeoJSON',
        '.gpkg': 'GPKG',
        '.gml': 'GML'
    }
    
    if ext not in supported_formats and driver is None:
        raise ValueError(f"Unsupported file format: {ext}. Supported formats: {list(supported_formats.keys())}")
    
    # Use driver from extension if not specified
    if driver is None:
        driver = supported_formats[ext]
    
    # Write vector data
    gdf.to_file(output_path, driver=driver)


def write_results_to_csv(
    results: Dict[str, Any],
    output_path: str
) -> None:
    """
    Write analysis results to CSV file.

    Parameters
    ----------
    results : Dict[str, Any]
        Dictionary containing analysis results
    output_path : str
        Path to write the CSV file
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Convert dictionary to DataFrame
    import pandas as pd
    df = pd.DataFrame.from_dict(results)
    
    # Write to CSV
    df.to_csv(output_path, index=False)


def ensure_directory(directory_path: str) -> None:
    """
    Ensure a directory exists, creating it if necessary.

    Parameters
    ----------
    directory_path : str
        Path to the directory
    """
    os.makedirs(directory_path, exist_ok=True)
