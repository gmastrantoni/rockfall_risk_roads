"""
Parameter Extraction Module

This module provides functions for extracting hazard parameters from
rockfall runout model results for risk assessment.
"""

import geopandas as gpd
import rioxarray as rxr
import numpy as np
import logging
from typing import Union, List, Tuple, Dict, Optional, Any

# Import utility functions
from ..utils.geo_utils import buffer_road_segments, extract_zonal_statistics

class HazardParameterExtraction:
    """
    Class for extracting hazard parameters from rockfall runout models.
    
    This class handles the extraction of various hazard parameters such as
    susceptibility, velocity, and energy from raster data for road segments.
    """
    
    def __init__(
        self,
        buffer_distance: float = 15.0,
        params: Optional[Dict[str, rxr.raster_array.RasterArray]] = None
    ):
        """
        Initialize the HazardParameterExtraction.

        Parameters
        ----------
        buffer_distance : float, optional
            Buffer distance in meters, by default 15.0
        params : Optional[Dict[str, rxr.raster_array.RasterArray]], optional
            Dictionary of parameter names to rasters, by default None
        """
        self.buffer_distance = buffer_distance
        self.params = params or {}
        self.logger = logging.getLogger(__name__)
    
    def add_parameter(self, name: str, raster: rxr.raster_array.RasterArray):
        """
        Add a parameter raster to the extraction.

        Parameters
        ----------
        name : str
            Name of the parameter
        raster : rxr.raster_array.RasterArray
            Raster containing parameter values
        """
        # Set the name attribute on the raster for easier identification in logs
        setattr(raster, 'name', name)
        self.params[name] = raster
        self.logger.info(f"Added parameter raster: {name}")
    
    def extract_parameters(
        self,
        road_segments: gpd.GeoDataFrame,
        statistics: List[str] = ['min', 'mean', 'max', 'std']
    ) -> gpd.GeoDataFrame:
        """
        Extract hazard parameters for road segments.

        Parameters
        ----------
        road_segments : gpd.GeoDataFrame
            GeoDataFrame containing road segments
        statistics : List[str], optional
            List of statistics to extract, by default ['min', 'mean', 'max', 'std']

        Returns
        -------
        gpd.GeoDataFrame
            GeoDataFrame with added hazard parameter columns
        """
        # Create buffer zones around road segments
        self.logger.info(f"Creating {self.buffer_distance}m buffer zones around {len(road_segments)} road segments")
        buffered_segments = buffer_road_segments(
            road_segments,
            buffer_distance=self.buffer_distance
        )
        
        # Create a copy to avoid modifying the original
        result = road_segments.copy()
        
        if not self.params:
            self.logger.warning("No parameter rasters available for extraction")
            return result
        
        # Extract statistics for each parameter
        for param_name, raster in self.params.items():
            self.logger.info(f"Extracting {statistics} statistics for parameter: {param_name}")
            
            try:
                # Log raster information for debugging
                if hasattr(raster, 'values'):
                    shape = raster.values.shape
                    self.logger.info(f"Raster shape: {shape}")
                    if len(shape) > 2:
                        self.logger.info(f"Multi-dimensional raster detected for {param_name}: {shape}")
                
                # Extract statistics
                stats_result = extract_zonal_statistics(
                    buffered_segments,
                    raster,
                    stats=statistics
                )
                
                # Add statistics to the result
                for stat in statistics:
                    col_name = f"{param_name}_{stat}"
                    if col_name in stats_result.columns:
                        result[col_name] = stats_result[col_name]
                        self.logger.info(f"Successfully extracted {stat} for {param_name}")
                    else:
                        result[col_name] = np.nan
                        self.logger.warning(f"Failed to extract {stat} for {param_name}")
            
            except Exception as e:
                self.logger.error(f"Error extracting statistics for {param_name}: {str(e)}")
                # Add placeholder columns with NaN values
                for stat in statistics:
                    result[f"{param_name}_{stat}"] = np.nan
        
        return result
    
    def simulate_parameters(
        self,
        road_segments: gpd.GeoDataFrame,
        params: Dict[str, Tuple[float, float]] = None
    ) -> gpd.GeoDataFrame:
        """
        Simulate hazard parameters for testing purposes.

        Parameters
        ----------
        road_segments : gpd.GeoDataFrame
            GeoDataFrame containing road segments
        params : Dict[str, Tuple[float, float]], optional
            Dictionary of parameter names to (min, max) ranges, by default None

        Returns
        -------
        gpd.GeoDataFrame
            GeoDataFrame with simulated hazard parameter columns
        """
        # Default parameter ranges if not provided
        if params is None:
            params = {
                'susceptibility': (0.1, 1.0),
                'velocity': (5.0, 20.0),
                'energy': (50.0, 500.0)
            }
        
        self.logger.info(f"Simulating parameters for {len(road_segments)} road segments")
        
        # Create a copy to avoid modifying the original
        result = road_segments.copy()
        
        # Simulate values for each parameter
        for param_name, (min_val, max_val) in params.items():
            self.logger.info(f"Simulating {param_name} values in range [{min_val}, {max_val}]")
            
            # Generate random values within the specified range
            result[f"{param_name}_max"] = np.random.uniform(
                min_val,
                max_val,
                len(road_segments)
            )
            
            # Also add mean values (slightly lower than max)
            result[f"{param_name}_mean"] = result[f"{param_name}_max"] * np.random.uniform(0.7, 0.9, len(road_segments))
            
            # And min values (even lower)
            result[f"{param_name}_min"] = result[f"{param_name}_mean"] * np.random.uniform(0.5, 0.8, len(road_segments))
            
            # Calculate standard deviation
            result[f"{param_name}_std"] = (result[f"{param_name}_max"] - result[f"{param_name}_min"]) / 4
        
        self.logger.info("Parameter simulation completed")
        return result
