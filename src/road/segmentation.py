"""
Road Segmentation Module

This module provides functions for dividing road networks into equal-length
segments for uniform analysis in rockfall risk assessment.
"""

import geopandas as gpd
import pandas as pd
import numpy as np
import logging
from shapely.geometry import Point, LineString, MultiLineString
from shapely.ops import split, snap
from typing import Union, List, Tuple, Optional, Dict, Any
import uuid

logger = logging.getLogger(__name__)

def segment_road(
    line: LineString,
    segment_length: float = 200.0
) -> List[LineString]:
    """
    Split a line into segments of equal length.

    Parameters
    ----------
    line : LineString
        Line to split
    segment_length : float, optional
        Length of each segment in meters, by default 200.0

    Returns
    -------
    List[LineString]
        List of LineString segments
    """
    # Get line length
    line_length = line.length
    
    # If line is shorter than segment_length, return the original line
    if line_length <= segment_length:
        return [line]
    
    # Calculate number of segments
    num_segments = int(np.ceil(line_length / segment_length))
    
    # Calculate actual segment length to ensure equal segments
    actual_segment_length = line_length / num_segments
    
    # Create cut points at segment boundaries
    cut_points = []
    for i in range(1, num_segments):
        distance = i * actual_segment_length
        point = line.interpolate(distance)
        # Add a small buffer to ensure the point is on the line
        buffered_point = point.buffer(1e-8)
        cut_points.append(buffered_point)
    
    # Try to split the line using shapely's split operation
    try:
        # Create MultiPoint geometry for splitting
        from shapely.geometry import MultiPolygon
        if cut_points:
            multi_polygon = MultiPolygon(cut_points)
            split_line = split(line, multi_polygon)
            segments = [segment for segment in split_line.geoms if segment.length >= 1.0]
            
            # Verify result - segments should be approximately equal length
            lengths = [segment.length for segment in segments]
            if max(lengths) > 1.5 * min(lengths):
                logger.warning(f"Segments have uneven lengths: min={min(lengths):.2f}, max={max(lengths):.2f}")
            
            return segments
        else:
            return [line]
    except Exception as e:
        logger.warning(f"Error splitting line using shapely: {e}")
        
        # Fallback: manually create segments by interpolating points
        segments = []
        for i in range(num_segments):
            start_dist = i * actual_segment_length
            end_dist = min((i + 1) * actual_segment_length, line_length)
            
            # Get points along the line
            points = []
            # Add intermediate points for better accuracy
            num_points = max(10, int(segment_length / 10))  # At least 10 points per segment
            for j in range(num_points + 1):
                dist = start_dist + (end_dist - start_dist) * j / num_points
                point = line.interpolate(dist)
                points.append((point.x, point.y))
            
            # Create segment
            segment = LineString(points)
            segments.append(segment)
        
        return segments


def segment_road_network(
    road_gdf: gpd.GeoDataFrame,
    segment_length: float = 200.0,
    id_column: Optional[str] = None,
    preserve_attributes: bool = True
) -> gpd.GeoDataFrame:
    """
    Segment a road network into equal-length segments.

    Parameters
    ----------
    road_gdf : gpd.GeoDataFrame
        GeoDataFrame containing road network
    segment_length : float, optional
        Length of each segment in meters, by default 200.0
    id_column : str, optional
        Column containing unique identifiers, by default None
    preserve_attributes : bool, optional
        Whether to preserve original attributes, by default True

    Returns
    -------
    gpd.GeoDataFrame
        GeoDataFrame containing segmented road network
    """
    # Check that the input is a GeoDataFrame with LineString geometries
    if not isinstance(road_gdf, gpd.GeoDataFrame):
        raise ValueError("Input must be a GeoDataFrame")
    
    # Get the column names to be preserved
    columns = list(road_gdf.columns)
    if 'geometry' in columns:
        columns.remove('geometry')
    
    # Create an ID column if it doesn't exist
    if id_column is None or id_column not in road_gdf.columns:
        road_gdf = road_gdf.copy()
        road_gdf['road_id'] = [str(i+1) for i in range(len(road_gdf))]
        id_column = 'road_id'
    
    # Create a list to store the segments
    segments = []
    
    # Iterate through each road
    for idx, road in road_gdf.iterrows():
        # Get the road geometry
        road_geom = road.geometry
        
        # Skip invalid geometries
        if road_geom is None or road_geom.is_empty:
            logger.warning(f"Skipping road with empty geometry at index {idx}")
            continue
        
        # Handle MultiLineString geometries by processing each part separately
        if isinstance(road_geom, MultiLineString):
            line_parts = list(road_geom.geoms)
            logger.info(f"Processing MultiLineString with {len(line_parts)} parts at index {idx}")
        elif isinstance(road_geom, LineString):
            line_parts = [road_geom]
        else:
            logger.warning(f"Unsupported geometry type {type(road_geom)} at index {idx}, skipping")
            continue
        
        part_idx = 0
        for line_part in line_parts:
            # Segment the road part
            try:
                road_segments = segment_road(line_part, segment_length)
                logger.debug(f"Split road {road[id_column]} part {part_idx} into {len(road_segments)} segments")
            except Exception as e:
                logger.error(f"Error segmenting road {road[id_column]} part {part_idx}: {e}")
                continue
            
            # Create a new row for each segment
            for i, segment in enumerate(road_segments):
                try:
                    # Create a new row with original attributes if requested
                    if preserve_attributes:
                        new_row = {col: road[col] for col in columns if col in road}
                    else:
                        new_row = {}
                        new_row[id_column] = road[id_column]
                    
                    # Create a unique segment ID
                    segment_id = f"{road[id_column]}_{part_idx}_{i}"
                    
                    # Add segment-specific attributes
                    new_row['segment_id'] = segment_id
                    new_row['parent_id'] = road[id_column]
                    new_row['part_idx'] = part_idx
                    new_row['segment_idx'] = i
                    new_row['segment_length'] = segment.length
                    new_row['geometry'] = segment
                    
                    segments.append(new_row)
                except Exception as e:
                    logger.error(f"Error creating segment {i} for road {road[id_column]} part {part_idx}: {e}")
            
            part_idx += 1
    
    # Create a new GeoDataFrame from the segments
    if segments:
        # Get all column names from segments
        all_cols = set()
        for segment in segments:
            all_cols.update(segment.keys())
        
        # Ensure all segments have all columns
        for segment in segments:
            for col in all_cols:
                if col not in segment:
                    segment[col] = None
        
        segments_gdf = gpd.GeoDataFrame(segments, crs=road_gdf.crs)
        logger.info(f"Created {len(segments_gdf)} segments from {len(road_gdf)} roads")
        return segments_gdf
    else:
        logger.warning("No valid segments created")
        return gpd.GeoDataFrame(geometry=[], crs=road_gdf.crs)


def dissolve_segments(
    segments_gdf: gpd.GeoDataFrame,
    dissolve_column: str = 'parent_id'
) -> gpd.GeoDataFrame:
    """
    Dissolve segmented roads back into original roads.

    Parameters
    ----------
    segments_gdf : gpd.GeoDataFrame
        GeoDataFrame containing segmented road network
    dissolve_column : str, optional
        Column to dissolve by, by default 'parent_id'

    Returns
    -------
    gpd.GeoDataFrame
        GeoDataFrame containing dissolved road network
    """
    # Check input
    if dissolve_column not in segments_gdf.columns:
        raise ValueError(f"Dissolve column '{dissolve_column}' not found in the GeoDataFrame")
    
    # Dissolve segments
    dissolved = segments_gdf.dissolve(by=dissolve_column, aggfunc='first')
    
    # Reset index
    dissolved = dissolved.reset_index()
    
    return dissolved
