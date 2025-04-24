"""
Road Classification Module

This module provides functions for classifying road segments based on
their spatial relationship with rockfall hazard areas.
"""

import geopandas as gpd
import pandas as pd
import numpy as np
import logging
import rioxarray as rxr
from shapely.geometry import LineString, Polygon, MultiPolygon
from typing import Union, List, Tuple, Dict, Optional, Any

# Import utility functions
from ..utils.geo_utils import raster_to_polygon, spatial_join
from ..hazard.runout_analysis import RunoutAnalysis

# Set up logger
logger = logging.getLogger(__name__)


def classify_road_segments_by_runout(
    road_segments: gpd.GeoDataFrame,
    runout_raster: rxr.raster_array.RasterArray,
    runout_value: float = 1,
    slope_units: Optional[gpd.GeoDataFrame] = None,
    source_areas: Optional[gpd.GeoDataFrame] = None,
    clumps: Optional[gpd.GeoDataFrame] = None
) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """
    Classify road segments based on their spatial relationship with rockfall runout zones.

    Parameters
    ----------
    road_segments : gpd.GeoDataFrame
        GeoDataFrame containing road segments
    runout_raster : rxr.raster_array.RasterArray
        Raster representing rockfall runout zone
    runout_value : float, optional
        Value in runout raster representing runout zone, by default 1
    slope_units : Optional[gpd.GeoDataFrame], optional
        GeoDataFrame containing slope units, by default None
    source_areas : Optional[gpd.GeoDataFrame], optional
        GeoDataFrame containing rockfall source areas, by default None
    clumps : Optional[gpd.GeoDataFrame], optional
        GeoDataFrame containing rockfall clumps, by default None

    Returns
    -------
    Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame]
        Tuple containing:
        - runout_segments: Segments intersecting runout zone
        - attention_segments: "Area of Attention" segments
        - safe_segments: "Not at Risk" segments
    """
    # Validate inputs
    if not isinstance(road_segments, gpd.GeoDataFrame):
        raise ValueError("road_segments must be a GeoDataFrame")
    
    # Check for segment_id column
    id_column = None
    for col in ['segment_id', 'id', 'ID']:
        if col in road_segments.columns:
            id_column = col
            break
            
    if id_column is None:
        logger.warning("No ID column found in road_segments, using index as ID")
        road_segments = road_segments.copy()
        road_segments['segment_id'] = road_segments.index.astype(str)
        id_column = 'segment_id'
    
    # Convert runout raster to vector for spatial analysis
    try:
        logger.info("Converting runout raster to polygons")
        runout_polygons = raster_to_polygon(runout_raster, value=runout_value)
        logger.info(f"Created {len(runout_polygons)} runout polygons")
    except Exception as e:
        logger.error(f"Error converting runout raster to polygons: {e}")
        runout_polygons = gpd.GeoDataFrame(geometry=[], crs=road_segments.crs)
    
    # Ensure both datasets have the same CRS
    if road_segments.crs != runout_polygons.crs and not runout_polygons.empty:
        if runout_polygons.crs is not None:
            logger.info(f"Reprojecting runout polygons from {runout_polygons.crs} to {road_segments.crs}")
            runout_polygons = runout_polygons.to_crs(road_segments.crs)
        else:
            logger.warning("Runout polygons have no CRS, assuming same as road segments")
            runout_polygons.crs = road_segments.crs
    
    # 1. Identify segments intersecting the runout zone
    try:
        if not runout_polygons.empty:
            # Make sure there are no problematic column names
            if 'index_right' in road_segments.columns:
                road_segments = road_segments.rename(columns={'index_right': 'index_right_orig'})
            if 'index_left' in road_segments.columns:
                road_segments = road_segments.rename(columns={'index_left': 'index_left_orig'})
            
            if 'index_right' in runout_polygons.columns:
                runout_polygons = runout_polygons.rename(columns={'index_right': 'index_right_orig'})
            if 'index_left' in runout_polygons.columns:
                runout_polygons = runout_polygons.rename(columns={'index_left': 'index_left_orig'})
            
            runout_analysis = RunoutAnalysis(runout_raster=runout_raster, runout_value=runout_value)
            runout_segments = runout_analysis.identify_intersecting_segments(road_segments=road_segments)
            # runout_segments = spatial_join(
            #     road_segments,
            #     runout_polygons,
            #     how='inner',
            #     predicate='intersects'
            # )
            
            # Check for duplicates
            if id_column in runout_segments.columns:
                duplicates = runout_segments.duplicated(id_column)
                if duplicates.any():
                    logger.warning(f"Found {duplicates.sum()} duplicate segments after runout zone intersection")
                    runout_segments = runout_segments.drop_duplicates(id_column)
            
            # Keep only the original columns plus a classification column
            keep_columns = list(road_segments.columns)
            for col in runout_segments.columns:
                if col in keep_columns or col == 'geometry':
                    continue
                runout_segments = runout_segments.drop(col, axis=1)
            
            runout_segments['risk_class'] = 'Runout Zone'
            runout_segments['hazard_score'] = 0.0  # Will be calculated in detail later
            
            logger.info(f"Identified {len(runout_segments)} segments in runout zone")
        else:
            logger.warning("No runout polygons, creating empty runout_segments")
            runout_segments = gpd.GeoDataFrame(
                columns=list(road_segments.columns) + ['risk_class', 'hazard_score'],
                geometry='geometry',
                crs=road_segments.crs
            )
    except Exception as e:
        logger.error(f"Error identifying segments in runout zone: {e}")
        runout_segments = gpd.GeoDataFrame(
            columns=list(road_segments.columns) + ['risk_class', 'hazard_score'],
            geometry='geometry',
            crs=road_segments.crs
        )
    
    # Get the IDs of segments already classified
    classified_ids = set(runout_segments[id_column]) if id_column in runout_segments.columns else set()
    
    # 2. Identify "Area of Attention" segments
    logger.info("Identifying Area of Attention segments")
    
    try:
        # Prepare attention_segments with the same structure as road_segments
        attention_segments = gpd.GeoDataFrame(
            columns=list(road_segments.columns) + ['risk_class', 'hazard_score'],
            geometry='geometry',
            crs=road_segments.crs
        )
        
        # Only proceed if we have slope units and source areas or clumps
        if (slope_units is not None and not slope_units.empty and 
            ((source_areas is not None and not source_areas.empty) or 
             (clumps is not None and not clumps.empty))):
            
            # Fix potential column name conflicts
            if 'index_right' in slope_units.columns:
                slope_units = slope_units.rename(columns={'index_right': 'index_right_orig'})
            if 'index_left' in slope_units.columns:
                slope_units = slope_units.rename(columns={'index_left': 'index_left_orig'})
            
            # Identify slope units containing source areas
            hazard_slope_units = None
            if source_areas is not None and not source_areas.empty:
                # Fix potential column name conflicts
                if 'index_right' in source_areas.columns:
                    source_areas = source_areas.rename(columns={'index_right': 'index_right_orig'})
                if 'index_left' in source_areas.columns:
                    source_areas = source_areas.rename(columns={'index_left': 'index_left_orig'})
                
                # Ensure same CRS
                if slope_units.crs != source_areas.crs:
                    source_areas = source_areas.to_crs(slope_units.crs)
                
                source_slope_units = spatial_join(
                    slope_units,
                    source_areas,
                    how='inner',
                    predicate='intersects'
                )
                
                if not source_slope_units.empty:
                    # Keep only necessary columns
                    geometry_col = source_slope_units.geometry.name
                    source_slope_units = gpd.GeoDataFrame(
                        geometry=source_slope_units.geometry,
                        crs=source_slope_units.crs
                    )
                    source_slope_units['has_source'] = True
                    
                    hazard_slope_units = source_slope_units
                    logger.info(f"Identified {len(source_slope_units)} slope units containing source areas")
            
            # Identify slope units containing clumps
            if clumps is not None and not clumps.empty:
                # Fix potential column name conflicts
                if 'index_right' in clumps.columns:
                    clumps = clumps.rename(columns={'index_right': 'index_right_orig'})
                if 'index_left' in clumps.columns:
                    clumps = clumps.rename(columns={'index_left': 'index_left_orig'})
                
                # Ensure same CRS
                if slope_units.crs != clumps.crs:
                    clumps = clumps.to_crs(slope_units.crs)
                
                clump_slope_units = spatial_join(
                    slope_units,
                    clumps,
                    how='inner',
                    predicate='intersects'
                )
                
                if not clump_slope_units.empty:
                    logger.info(f"Identified {len(clump_slope_units)} slope units containing clumps")
                    
                    # Keep only necessary columns
                    geometry_col = clump_slope_units.geometry.name
                    clump_slope_units = gpd.GeoDataFrame(
                        geometry=clump_slope_units.geometry,
                        crs=clump_slope_units.crs
                    )
                    clump_slope_units['has_clump'] = True
                    
                    if hazard_slope_units is not None:
                        # Combine with existing hazard slope units
                        hazard_slope_units = pd.concat([hazard_slope_units, clump_slope_units])
                        # Remove duplicates if any based on geometry
                        hazard_slope_units = hazard_slope_units.drop_duplicates(subset='geometry')
                    else:
                        hazard_slope_units = clump_slope_units
            
            # If we have identified hazard slope units, find road segments intersecting them
            if hazard_slope_units is not None and not hazard_slope_units.empty:
                # Ensure same CRS
                if road_segments.crs != hazard_slope_units.crs:
                    hazard_slope_units = hazard_slope_units.to_crs(road_segments.crs)
                
                # Only consider segments not already classified as runout zone
                remaining_segments = road_segments[~road_segments[id_column].isin(classified_ids)]
                if not remaining_segments.empty:
                    # Fix potential column name conflicts
                    if 'index_right' in remaining_segments.columns:
                        remaining_segments = remaining_segments.rename(columns={'index_right': 'index_right_orig'})
                    if 'index_left' in remaining_segments.columns:
                        remaining_segments = remaining_segments.rename(columns={'index_left': 'index_left_orig'})
                    
                    if 'index_right' in hazard_slope_units.columns:
                        hazard_slope_units = hazard_slope_units.rename(columns={'index_right': 'index_right_orig'})
                    if 'index_left' in hazard_slope_units.columns:
                        hazard_slope_units = hazard_slope_units.rename(columns={'index_left': 'index_left_orig'})
                    
                    attention_candidates = spatial_join(
                        remaining_segments,
                        hazard_slope_units,
                        how='inner',
                        predicate='intersects'
                    )
                    
                    # Keep only the original columns plus a classification column
                    if not attention_candidates.empty:
                        keep_columns = list(road_segments.columns)
                        for col in attention_candidates.columns:
                            if col in keep_columns or col == 'geometry':
                                continue
                            attention_candidates = attention_candidates.drop(col, axis=1)
                        
                        attention_candidates['risk_class'] = 'Area of Attention'
                        attention_candidates['hazard_score'] = 0.0  # As specified
                        
                        # Check for duplicates
                        if id_column in attention_candidates.columns:
                            duplicates = attention_candidates.duplicated(id_column)
                            if duplicates.any():
                                logger.warning(f"Found {duplicates.sum()} duplicate segments in Area of Attention")
                                attention_candidates = attention_candidates.drop_duplicates(id_column)
                        
                        attention_segments = attention_candidates
                        logger.info(f"Identified {len(attention_segments)} segments in Area of Attention")
                        
                        # Update classified IDs
                        classified_ids.update(attention_segments[id_column])
    except Exception as e:
        logger.error(f"Error identifying Area of Attention segments: {e}")
    
    # 3. Identify "Not at Risk" segments
    try:
        # Get segments not classified yet
        if not classified_ids:
            # If no segments have been classified yet, all are safe
            safe_segments = road_segments.copy()
        else:
            safe_segments = road_segments[~road_segments[id_column].isin(classified_ids)].copy()
        
        safe_segments['risk_class'] = 'Not at Risk'
        safe_segments['hazard_score'] = -1.0  # As specified
        
        logger.info(f"Identified {len(safe_segments)} segments Not at Risk")
    except Exception as e:
        logger.error(f"Error identifying Not at Risk segments: {e}")
        safe_segments = gpd.GeoDataFrame(
            columns=list(road_segments.columns) + ['risk_class', 'hazard_score'],
            geometry='geometry', 
            crs=road_segments.crs
        )
    
    return runout_segments, attention_segments, safe_segments


def merge_classified_segments(
    runout_segments: gpd.GeoDataFrame,
    attention_segments: gpd.GeoDataFrame,
    safe_segments: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
    """
    Merge the classified road segments into a single GeoDataFrame.

    Parameters
    ----------
    runout_segments : gpd.GeoDataFrame
        Segments intersecting runout zone
    attention_segments : gpd.GeoDataFrame
        "Area of Attention" segments
    safe_segments : gpd.GeoDataFrame
        "Not at Risk" segments

    Returns
    -------
    gpd.GeoDataFrame
        Combined GeoDataFrame with all segments
    """
    # Validate inputs
    if not all(isinstance(gdf, gpd.GeoDataFrame) for gdf in [runout_segments, attention_segments, safe_segments]):
        raise ValueError("All inputs must be GeoDataFrames")
    
    # Check if all GeoDataFrames are empty
    if all(gdf.empty for gdf in [runout_segments, attention_segments, safe_segments]):
        logger.warning("All input GeoDataFrames are empty")
        return gpd.GeoDataFrame(geometry=[], crs=runout_segments.crs if hasattr(runout_segments, 'crs') else None)
    
    # Make sure all have the same columns
    all_columns = set()
    for gdf in [runout_segments, attention_segments, safe_segments]:
        if not gdf.empty:
            all_columns.update(gdf.columns)
    
    # Fill missing columns with None
    for gdf in [runout_segments, attention_segments, safe_segments]:
        if not gdf.empty:
            for col in all_columns:
                if col not in gdf.columns:
                    gdf[col] = None
    
    # Concatenate the three GeoDataFrames
    try:
        merged = pd.concat([runout_segments, attention_segments, safe_segments])
        
        # Check for duplicates
        id_column = None
        for col in ['segment_id', 'id', 'ID']:
            if col in merged.columns:
                id_column = col
                break
                
        if id_column is not None:
            duplicates = merged.duplicated(id_column)
            if duplicates.any():
                logger.warning(f"Found {duplicates.sum()} duplicate segments after merging")
                merged = merged.drop_duplicates(id_column)
        
        # Reset index
        merged = merged.reset_index(drop=True)
        
        # Ensure result is a GeoDataFrame
        if not isinstance(merged, gpd.GeoDataFrame):
            merged = gpd.GeoDataFrame(merged, geometry='geometry')
        
        return merged
    except Exception as e:
        logger.error(f"Error merging classified segments: {e}")
        return gpd.GeoDataFrame(geometry=[], crs=runout_segments.crs if hasattr(runout_segments, 'crs') else None)
