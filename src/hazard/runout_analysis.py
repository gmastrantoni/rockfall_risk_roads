"""
Runout Analysis Module

This module provides functions for analyzing rockfall runout models and
extracting relevant information for hazard assessment.
"""

import geopandas as gpd
import rioxarray as rxr
import numpy as np
import logging
from shapely.geometry import Polygon
from typing import Union, List, Tuple, Dict, Optional, Any

# Import utility functions
from ..utils.geo_utils import buffer_road_segments, raster_to_polygon, extract_zonal_statistics, spatial_join

class RunoutAnalysis:
    """
    Class for analyzing rockfall runout models and extracting parameters.
    
    This class handles the conversion of runout rasters to vector polygons
    and the spatial analysis of runout zone intersection with road segments.
    """
    
    def __init__(self, runout_raster: rxr.raster_array.RasterArray, runout_value: float = 1.0):
        """
        Initialize the RunoutAnalysis.

        Parameters
        ----------
        runout_raster : rxr.raster_array.RasterArray
            Raster representing rockfall runout zone
        runout_value : float, optional
            Value in runout raster representing runout zone, by default 1.0
        """
        self.runout_raster = runout_raster
        self.runout_value = runout_value
        self.runout_polygons = None
        self.logger = logging.getLogger(__name__)
        self.buffer_distance = 15.0  # Default buffer distance in meters
        
        # Convert runout raster to vector polygons
        self._raster_to_polygons()
    
    def _raster_to_polygons(self):
        """
        Convert runout raster to vector polygons.
        """
        try:
            self.runout_polygons = raster_to_polygon(
                self.runout_raster,
                value=self.runout_value
            )
            self.logger.info(f"Converted runout raster to {len(self.runout_polygons)} polygons")
        except Exception as e:
            self.logger.error(f"Error converting runout raster to polygons: {e}")
            # Create an empty GeoDataFrame as fallback
            self.runout_polygons = gpd.GeoDataFrame(geometry=[])
            if hasattr(self.runout_raster, 'rio') and hasattr(self.runout_raster.rio, 'crs'):
                self.runout_polygons.crs = self.runout_raster.rio.crs
    
    def identify_intersecting_segments(self, road_segments: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Identify road segments intersecting the runout zone.

        Parameters
        ----------
        road_segments : gpd.GeoDataFrame
            GeoDataFrame containing road segments

        Returns
        -------
        gpd.GeoDataFrame
            GeoDataFrame containing segments intersecting runout zone
        """
        if self.runout_polygons is None or self.runout_polygons.empty:
            self.logger.warning("No runout polygons available for intersection analysis")
            return gpd.GeoDataFrame(columns=road_segments.columns)
        
        # Ensure both datasets have the same CRS
        if road_segments.crs != self.runout_polygons.crs:
            if self.runout_polygons.crs is not None:
                self.logger.info(f"Reprojecting runout polygons from {self.runout_polygons.crs} to {road_segments.crs}")
                runout_polygons = self.runout_polygons.to_crs(road_segments.crs)
            else:
                self.logger.warning("Runout polygons have no CRS, assuming same as road segments")
                runout_polygons = self.runout_polygons
                runout_polygons.crs = road_segments.crs
        else:
            runout_polygons = self.runout_polygons
        
        # Perform spatial join to identify segments intersecting runout zone
        try:
            road_segments_buffered = buffer_road_segments(
            road_segments,
            buffer_distance=self.buffer_distance
            )
            # Perform spatial join
            intersecting_segments = spatial_join(
                road_segments_buffered,
                runout_polygons,
                how='inner',
                predicate='intersects'
            )
            
            # Check if we've received duplicate rows (can happen with multiple overlapping polygons)
            if len(intersecting_segments) > len(road_segments):
                self.logger.warning(f"Spatial join produced {len(intersecting_segments)} rows from {len(road_segments)} segments")
                
                # Drop duplicates by segment_id if available
                if 'segment_id' in intersecting_segments.columns:
                    before_dedupe = len(intersecting_segments)
                    intersecting_segments = intersecting_segments.drop_duplicates('segment_id')
                    self.logger.info(f"Removed {before_dedupe - len(intersecting_segments)} duplicate segments")
            
            # Keep only the original columns
            keep_columns = [col for col in road_segments.columns if col in intersecting_segments.columns]
            intersecting_segments = intersecting_segments[keep_columns]
            
            self.logger.info(f"Identified {len(intersecting_segments)} segments intersecting runout zone")
            
            return intersecting_segments
            
        except Exception as e:
            self.logger.error(f"Error identifying intersecting segments: {e}")
            return gpd.GeoDataFrame(columns=road_segments.columns)
