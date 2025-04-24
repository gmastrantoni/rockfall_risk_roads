"""
Geospatial Utilities

This module provides functions for geospatial operations used in the
rockfall risk assessment workflow.
"""

import geopandas as gpd
import numpy as np
import rioxarray as rxr
from shapely.geometry import Point, LineString, Polygon, MultiPolygon
from typing import Union, List, Tuple, Optional, Dict, Any


def buffer_road_segments(
    road_segments: gpd.GeoDataFrame,
    buffer_distance: float = 15.0
) -> gpd.GeoDataFrame:
    """
    Create buffer zones around road segments.

    Parameters
    ----------
    road_segments : gpd.GeoDataFrame
        GeoDataFrame containing road segments
    buffer_distance : float, optional
        Buffer distance in meters, by default 15.0

    Returns
    -------
    gpd.GeoDataFrame
        GeoDataFrame containing buffered geometries
    """
    # Create a copy to avoid modifying the original
    buffered = road_segments.copy()
    
    # Create buffer
    buffered['geometry'] = road_segments.geometry.buffer(buffer_distance)
    
    return buffered


def raster_to_polygon(
    raster: rxr.raster_array.RasterArray,
    value: float = 1,
    all_touched: bool = False
) -> gpd.GeoDataFrame:
    """
    Convert a raster to polygons based on cell values.

    Parameters
    ----------
    raster : rxr.raster_array.RasterArray
        Input raster
    value : float, optional
        Value to extract polygons for, by default 1
    all_touched : bool, optional
        Whether to include cells that are touched, by default False

    Returns
    -------
    gpd.GeoDataFrame
        GeoDataFrame containing polygon geometries
    """
    import rasterio
    from rasterio import features

    # Ensure raster is 2D
    raster_values = raster.values
    if len(raster_values.shape) > 2:
        if raster_values.shape[0] == 1:
            # Single band with extra dimension
            raster_values = raster_values[0]
        else:
            # Multi-band raster - use first band or calculate sum/mean
            raster_values = raster_values[0]
    
    # Create a mask where raster == value
    mask = raster_values == value
    
    # Handle no-data values if present
    if hasattr(raster, 'rio') and hasattr(raster.rio, 'nodata'):
        if raster.rio.nodata is not None:
            mask = np.logical_and(mask, raster_values != raster.rio.nodata)
    
    # Extract shapes from the mask
    shapes = features.shapes(
        mask.astype('uint8'),
        mask=mask,
        transform=raster.rio.transform(),
        connectivity=8
    )
    
    # Convert shapes to geometries
    geometries = [Polygon(shape[0]["coordinates"][0]) for shape in shapes]
    
    # Create GeoDataFrame
    gdf = gpd.GeoDataFrame(
        {'geometry': geometries},
        crs=raster.rio.crs
    )
    
    return gdf


def extract_zonal_statistics(
    vector: gpd.GeoDataFrame,
    raster: rxr.raster_array.RasterArray,
    stats: List[str] = ['min', 'mean', 'max', 'std']
) -> gpd.GeoDataFrame:
    """
    Extract zonal statistics from a raster using vector geometries.

    Parameters
    ----------
    vector : gpd.GeoDataFrame
        GeoDataFrame containing geometries
    raster : rxr.raster_array.RasterArray
        Raster to extract statistics from
    stats : List[str], optional
        Statistics to calculate, by default ['min', 'mean', 'max', 'std']

    Returns
    -------
    gpd.GeoDataFrame
        GeoDataFrame with additional columns for statistics
    """
    import rasterstats
    
    # Make a copy to avoid modifying the original
    result = vector.copy()
    
    try:
        # Extract raster values and transform
        if hasattr(raster, 'rio'):
            # Get the name to use for the output columns
            raster_name = getattr(raster, 'name', 'raster')
            
            # Extract the values as a 2D numpy array
            raster_values = raster.values
            if len(raster_values.shape) > 2:
                if raster_values.shape[0] == 1:
                    # Single band with extra dimension
                    raster_values = raster_values[0]
                else:
                    # Multi-band raster - use first band
                    raster_values = raster_values[0]
                    print(f"Warning: Using only the first band of a multi-band raster for {raster_name}")
            
            transform = raster.rio.transform()
            nodata = raster.rio.nodata if hasattr(raster.rio, 'nodata') else None
        else:
            raster_name = 'raster'
            raster_values = raster
            transform = None  # This will cause an error, need proper transform
            nodata = None
            
        # Calculate zonal statistics
        zonal_stats = rasterstats.zonal_stats(
            vector,
            raster_values,
            affine=transform,
            stats=stats,
            nodata=nodata,
            all_touched=True
        )
        
        # Add statistics to the result
        for i, stat_dict in enumerate(zonal_stats):
            for stat_name, value in stat_dict.items():
                result.loc[i, f"{raster_name}_{stat_name}"] = value
    
    except Exception as e:
        import logging
        logging.getLogger(__name__).error(f"Error extracting zonal statistics: {e}")
        # Add default values to prevent downstream errors
        for stat in stats:
            result[f"{getattr(raster, 'name', 'raster')}_{stat}"] = np.nan
    
    return result


def reproject_geometry(
    geometry,
    source_crs: str,
    target_crs: str
) -> Union[Point, LineString, Polygon, MultiPolygon]:
    """
    Reproject a geometry from source CRS to target CRS.

    Parameters
    ----------
    geometry : Union[Point, LineString, Polygon, MultiPolygon]
        Geometry to reproject
    source_crs : str
        Source coordinate reference system
    target_crs : str
        Target coordinate reference system

    Returns
    -------
    Union[Point, LineString, Polygon, MultiPolygon]
        Reprojected geometry
    """
    # Create a temporary GeoDataFrame
    gdf = gpd.GeoDataFrame(geometry=[geometry], crs=source_crs)
    
    # Reproject
    gdf = gdf.to_crs(target_crs)
    
    # Return the geometry
    return gdf.geometry.iloc[0]


def spatial_join(
    left_gdf: gpd.GeoDataFrame,
    right_gdf: gpd.GeoDataFrame,
    how: str = 'inner',
    predicate: str = 'intersects'
) -> gpd.GeoDataFrame:
    """
    Perform a spatial join between two GeoDataFrames.

    Parameters
    ----------
    left_gdf : gpd.GeoDataFrame
        Left GeoDataFrame
    right_gdf : gpd.GeoDataFrame
        Right GeoDataFrame
    how : str, optional
        Type of join, by default 'inner'
    predicate : str, optional
        Spatial predicate, by default 'intersects'

    Returns
    -------
    gpd.GeoDataFrame
        Joined GeoDataFrame
    """
    # Ensure both GeoDataFrames have the same CRS
    if left_gdf.crs != right_gdf.crs:
        right_gdf = right_gdf.to_crs(left_gdf.crs)
    
    # Perform spatial join
    joined = gpd.sjoin(left_gdf, right_gdf, how=how, predicate=predicate)
    
    return joined
