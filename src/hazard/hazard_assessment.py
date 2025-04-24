"""
Hazard Assessment Module

This module provides a comprehensive framework for assessing rockfall hazard
by integrating runout analysis, parameter extraction, and hazard classification.
"""

import geopandas as gpd
import pandas as pd
import rioxarray as rxr
import numpy as np
import logging
import configparser
from typing import Union, List, Tuple, Dict, Optional, Any

# Import hazard components
from .runout_analysis import RunoutAnalysis
from .parameter_extraction import HazardParameterExtraction
from .hazard_classification import HazardClassification

# Import road classification
from ..road.classification import classify_road_segments_by_runout, merge_classified_segments

class HazardAssessment:
    """
    Class for comprehensive rockfall hazard assessment.
    
    This class integrates runout analysis, parameter extraction, and hazard
    classification to provide a comprehensive hazard assessment for road segments.
    """
    
    def __init__(
        self,
        runout_raster: Optional[rxr.raster_array.RasterArray] = None,
        runout_value: float = 1.0,
        buffer_distance: float = 15.0,
        parameter_weights: Optional[Dict[str, float]] = None,
        class_thresholds: Optional[Dict[str, Dict[str, Tuple[float, float]]]] = None
    ):
        """
        Initialize the HazardAssessment.

        Parameters
        ----------
        runout_raster : Optional[rxr.raster_array.RasterArray], optional
            Raster representing rockfall runout zone, by default None
        runout_value : float, optional
            Value in runout raster representing runout zone, by default 1.0
        buffer_distance : float, optional
            Buffer distance in meters, by default 15.0
        parameter_weights : Optional[Dict[str, float]], optional
            Dictionary of parameter names to weights, by default None
        class_thresholds : Optional[Dict[str, Dict[str, Tuple[float, float]]]], optional
            Dictionary of parameter names to class thresholds, by default None
        """
        # Initialize components
        self.runout_analysis = None
        if runout_raster is not None:
            self.runout_analysis = RunoutAnalysis(runout_raster, runout_value)
        
        self.parameter_extraction = HazardParameterExtraction(buffer_distance)
        self.hazard_classification = HazardClassification(parameter_weights, class_thresholds)
        
        # Store configuration
        self.runout_value = runout_value
        self.buffer_distance = buffer_distance
        
        # Initialize data containers
        self.runout_raster = runout_raster
        self.slope_units = None
        self.source_areas = None
        self.rockfall_clumps = None
        
        # Initialize result containers
        self.runout_segments = None
        self.attention_segments = None
        self.safe_segments = None
        self.hazard_parameters = None
        self.hazard_classification_result = None
        
        # Set up logger
        self.logger = logging.getLogger(__name__)
        self.logger.info("Hazard Assessment initialized")
    
    def set_runout_raster(self, runout_raster: rxr.raster_array.RasterArray, runout_value: Optional[float] = None):
        """
        Set or update the runout raster.

        Parameters
        ----------
        runout_raster : rxr.raster_array.RasterArray
            Raster representing rockfall runout zone
        runout_value : Optional[float], optional
            Value in runout raster representing runout zone, by default None (uses current value)
        """
        if runout_value is not None:
            self.runout_value = runout_value
        
        self.runout_raster = runout_raster
        self.runout_analysis = RunoutAnalysis(runout_raster, self.runout_value)
        
        self.logger.info("Runout raster updated")
    
    def set_slope_units(self, slope_units: gpd.GeoDataFrame):
        """
        Set or update the slope units.

        Parameters
        ----------
        slope_units : gpd.GeoDataFrame
            GeoDataFrame containing slope units
        """
        self.slope_units = slope_units
        self.logger.info(f"Slope units updated: {len(slope_units)} units")
    
    def set_source_areas(self, source_areas: gpd.GeoDataFrame):
        """
        Set or update the rockfall source areas.

        Parameters
        ----------
        source_areas : gpd.GeoDataFrame
            GeoDataFrame containing rockfall source areas
        """
        self.source_areas = source_areas
        self.logger.info(f"Source areas updated: {len(source_areas)} areas")
    
    def set_rockfall_clumps(self, rockfall_clumps: gpd.GeoDataFrame):
        """
        Set or update the rockfall clumps.

        Parameters
        ----------
        rockfall_clumps : gpd.GeoDataFrame
            GeoDataFrame containing rockfall clumps
        """
        self.rockfall_clumps = rockfall_clumps
        self.logger.info(f"Rockfall clumps updated: {len(rockfall_clumps)} clumps")
    
    def add_parameter_raster(self, name: str, raster: rxr.raster_array.RasterArray):
        """
        Add a parameter raster for hazard assessment.

        Parameters
        ----------
        name : str
            Name of the parameter
        raster : rxr.raster_array.RasterArray
            Raster containing parameter values
        """
        self.parameter_extraction.add_parameter(name, raster)
        
        self.logger.info(f"Parameter raster '{name}' added")
    
    
    def extract_hazard_parameters(
        self,
        road_segments: Optional[gpd.GeoDataFrame] = None,
        statistics: List[str] = ['min', 'mean', 'max', 'std'],
        simulate: bool = False
    ) -> gpd.GeoDataFrame:
        """
        Extract hazard parameters for road segments.

        Parameters
        ----------
        road_segments : Optional[gpd.GeoDataFrame], optional
            GeoDataFrame containing road segments, by default None (uses runout_segments)
        statistics : List[str], optional
            List of statistics to extract, by default ['min', 'mean', 'max', 'std']
        simulate : bool, optional
            Whether to simulate parameter values for testing, by default False

        Returns
        -------
        gpd.GeoDataFrame
            GeoDataFrame with added hazard parameter columns
        """
        # Use provided segments or fall back to runout segments
        segments = road_segments if road_segments is not None else self.runout_segments
        
        if segments is None or segments.empty:
            self.logger.warning("No segments to analyze")
            return gpd.GeoDataFrame()
        
        # Check for segment_id column
        id_column = None
        for col in ['segment_id', 'id', 'ID']:
            if col in segments.columns:
                id_column = col
                break
        
        if id_column is None:
            self.logger.warning("No ID column found in segments for parameter extraction")
        
        if simulate or not self.parameter_extraction.params:
            # Simulate parameter values for testing
            self.logger.info("Simulating hazard parameters")
            self.hazard_parameters = self.parameter_extraction.simulate_parameters(segments)
        else:
            # Extract actual parameter values from rasters
            self.logger.info("Extracting hazard parameters from rasters")
            self.hazard_parameters = self.parameter_extraction.extract_parameters(segments, statistics)
        
        # Ensure segment_id is preserved
        if id_column is not None and id_column not in self.hazard_parameters.columns:
            self.hazard_parameters[id_column] = segments[id_column].values
        
        # Check for duplicates
        if id_column is not None:
            duplicates = self.hazard_parameters.duplicated(id_column)
            if duplicates.any():
                self.logger.warning(f"Found {duplicates.sum()} duplicates after parameter extraction")
                # Remove duplicates
                self.hazard_parameters = self.hazard_parameters.drop_duplicates(id_column)
        
        return self.hazard_parameters
    
    def classify_hazard(
        self,
        segments_with_parameters: Optional[gpd.GeoDataFrame] = None,
        suffix: str = '_max'
    ) -> gpd.GeoDataFrame:
        """
        Classify hazard based on extracted parameters.

        Parameters
        ----------
        segments_with_parameters : Optional[gpd.GeoDataFrame], optional
            GeoDataFrame with hazard parameters, by default None (uses hazard_parameters)
        suffix : str, optional
            Suffix for parameter columns, by default '_max'

        Returns
        -------
        gpd.GeoDataFrame
            GeoDataFrame with added hazard classifications
        """
        # Use provided segments or fall back to hazard parameters
        segments = segments_with_parameters if segments_with_parameters is not None else self.hazard_parameters
        
        if segments is None or segments.empty:
            self.logger.warning("No segments with parameters to classify")
            return gpd.GeoDataFrame()
        
        self.logger.info("Classifying hazard parameters")
        
        # Check for segment_id column
        id_column = None
        for col in ['segment_id', 'id', 'ID']:
            if col in segments.columns:
                id_column = col
                break
        
        # Store segment IDs and geometries before classification
        if id_column is not None:
            segment_ids = segments[id_column].copy()
        else:
            segment_ids = None
            
        geometries = segments['geometry'].copy()
        
        # Classify hazard parameters
        self.hazard_classification_result = self.hazard_classification.classify_dataframe(segments, suffix)
        
        # Restore segment_id and geometry if they were lost
        if segment_ids is not None and id_column not in self.hazard_classification_result.columns:
            self.hazard_classification_result[id_column] = segment_ids.values
            
        if 'geometry' not in self.hazard_classification_result.columns:
            self.hazard_classification_result['geometry'] = geometries.values
        
        # Ensure the result is a GeoDataFrame
        if not isinstance(self.hazard_classification_result, gpd.GeoDataFrame):
            self.hazard_classification_result = gpd.GeoDataFrame(
                self.hazard_classification_result, 
                geometry='geometry',
                crs=segments.crs if hasattr(segments, 'crs') else None
            )
        
        # Check for duplicates
        if id_column is not None:
            duplicates = self.hazard_classification_result.duplicated(id_column)
            if duplicates.any():
                self.logger.warning(f"Found {duplicates.sum()} duplicates after hazard classification")
                # Remove duplicates
                self.hazard_classification_result = self.hazard_classification_result.drop_duplicates(id_column)
        
        return self.hazard_classification_result
    
    def assess_hazard(
        self,
        road_segments: gpd.GeoDataFrame,
        simulate_parameters: bool = False
    ) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
        """
        Perform complete hazard assessment for road segments.

        Parameters
        ----------
        road_segments : gpd.GeoDataFrame
            GeoDataFrame containing road segments
        simulate_parameters : bool, optional
            Whether to simulate parameter values for testing, by default False

        Returns
        -------
        Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]
            Tuple containing:
            - segments_with_hazard: GeoDataFrame with hazard assessment for segments in runout zone
            - segments_not_in_runout: GeoDataFrame with segments not in runout zone
            (Note: segments_not_in_runout includes both "Area of Attention" and "Not at Risk" segments)
        """
        if self.runout_raster is None:
            self.logger.error("Runout raster not set")
            return gpd.GeoDataFrame(), gpd.GeoDataFrame()
        
        # Check for segment_id column
        id_column = None
        for col in ['segment_id', 'id', 'ID']:
            if col in road_segments.columns:
                id_column = col
                break
                
        if id_column is None:
            self.logger.warning("No ID column found in road_segments, using index as ID")
            road_segments = road_segments.copy()
            road_segments['segment_id'] = road_segments.index.astype(str)
            id_column = 'segment_id'
        
        # Use the comprehensive classification function
        self.logger.info("Classifying road segments using runout zone, slope units, and rockfall source areas")
        runout_segments, attention_segments, safe_segments = classify_road_segments_by_runout(
            road_segments,
            self.runout_raster,
            runout_value=self.runout_value,
            slope_units=self.slope_units,
            source_areas=self.source_areas,
            clumps=self.rockfall_clumps
        )
        
        # Store the classification results
        self.runout_segments = runout_segments
        self.attention_segments = attention_segments
        self.safe_segments = safe_segments
        
        # Log results
        self.logger.info(f"Classified segments:")
        self.logger.info(f"  Runout Zone: {len(runout_segments)} segments")
        self.logger.info(f"  Area of Attention: {len(attention_segments)} segments")
        self.logger.info(f"  Not at Risk: {len(safe_segments)} segments")
        
        # Process runout segments with detailed hazard assessment
        if not runout_segments.empty:
            # Extract hazard parameters
            self.logger.info("Extracting hazard parameters for runout segments")
            segments_with_parameters = self.extract_hazard_parameters(
                runout_segments,
                simulate=simulate_parameters
            )
            
            # Classify hazard
            self.logger.info("Classifying hazard for runout segments")
            runout_segments_with_hazard = self.classify_hazard(segments_with_parameters)
            
            # Check for missing segments
            if id_column in runout_segments.columns and id_column in runout_segments_with_hazard.columns:
                if len(runout_segments_with_hazard) < len(runout_segments):
                    self.logger.warning(f"Some segments were lost during processing: {len(runout_segments)} -> {len(runout_segments_with_hazard)}")
                    
                    # Get the missing segment IDs
                    missing_ids = set(runout_segments[id_column]) - set(runout_segments_with_hazard[id_column])
                    self.logger.warning(f"Missing segment IDs: {missing_ids}")
                    
                    # Attempt to recover missing segments
                    missing_segments = runout_segments[runout_segments[id_column].isin(missing_ids)]
                    if not missing_segments.empty:
                        self.logger.info(f"Recovering {len(missing_segments)} missing segments with default hazard values")
                        
                        # Add default hazard values
                        missing_segments['hazard_score'] = 3.0  # Moderate hazard
                        missing_segments['hazard_class'] = 'Moderate'
                        
                        # Combine with existing results
                        runout_segments_with_hazard = pd.concat([runout_segments_with_hazard, missing_segments])
                
                # Ensure the result is a GeoDataFrame
                if not isinstance(runout_segments_with_hazard, gpd.GeoDataFrame):
                    runout_segments_with_hazard = gpd.GeoDataFrame(
                        runout_segments_with_hazard,
                        geometry='geometry',
                        crs=runout_segments.crs if hasattr(runout_segments, 'crs') else None
                    )
                
                # Ensure risk_class column exists
                if 'risk_class' not in runout_segments_with_hazard.columns:
                    runout_segments_with_hazard['risk_class'] = 'Runout Zone'
            
            else:
                # If ID column is lost, fall back to the original runout segments
                self.logger.warning("ID column lost during processing, falling back to original runout segments")
                runout_segments_with_hazard = runout_segments
        else:
            runout_segments_with_hazard = runout_segments
        
        # Combine attention and safe segments
        non_runout_segments = pd.concat([attention_segments, safe_segments])
        
        # Consolidate non-runout segments hazard information
        if not non_runout_segments.empty:
            # Ensure hazard scores are set correctly for non-runout segments
            attention_mask = non_runout_segments['risk_class'] == 'Area of Attention'
            safe_mask = non_runout_segments['risk_class'] == 'Not at Risk'
            
            # Set hazard scores according to specifications
            non_runout_segments.loc[attention_mask, 'hazard_score'] = 0.0
            non_runout_segments.loc[safe_mask, 'hazard_score'] = -1.0
            
            # Set hazard classes
            non_runout_segments.loc[attention_mask, 'hazard_class'] = 'Area of Attention'
            non_runout_segments.loc[safe_mask, 'hazard_class'] = 'Not at Risk'
        
        # Ensure both results are GeoDataFrames
        if not isinstance(runout_segments_with_hazard, gpd.GeoDataFrame) and not runout_segments_with_hazard.empty:
            runout_segments_with_hazard = gpd.GeoDataFrame(
                runout_segments_with_hazard,
                geometry='geometry',
                crs=road_segments.crs
            )
        
        if not isinstance(non_runout_segments, gpd.GeoDataFrame) and not non_runout_segments.empty:
            non_runout_segments = gpd.GeoDataFrame(
                non_runout_segments,
                geometry='geometry',
                crs=road_segments.crs
            )
        
        self.logger.info("Hazard assessment completed")
        
        return runout_segments_with_hazard, non_runout_segments
    
    def get_all_segments(self) -> gpd.GeoDataFrame:
        """
        Get all segments with their hazard classification.

        Returns
        -------
        gpd.GeoDataFrame
            GeoDataFrame containing all segments with hazard classification
        """
        # Merge the three segment types
        merged = merge_classified_segments(
            self.runout_segments if self.runout_segments is not None else gpd.GeoDataFrame(),
            self.attention_segments if self.attention_segments is not None else gpd.GeoDataFrame(),
            self.safe_segments if self.safe_segments is not None else gpd.GeoDataFrame()
        )
        
        return merged
    
    def get_hazard_classification(
        self,
        parameters: Dict[str, float]
    ) -> Tuple[float, str]:
        """
        Get hazard classification for a set of parameter values.

        Parameters
        ----------
        parameters : Dict[str, float]
            Dictionary of parameter names to values

        Returns
        -------
        Tuple[float, str]
            Tuple containing hazard score and hazard class
        """
        return self.hazard_classification.calculate_hazard_score(parameters)
    
    @classmethod
    def from_config(cls, config_file: str, io_utils_module=None):
        """
        Create a HazardAssessment from a configuration file.

        Parameters
        ----------
        config_file : str
            Path to the configuration file
        io_utils_module : module, optional
            Module containing the read_raster function, by default None

        Returns
        -------
        HazardAssessment
            Configured HazardAssessment instance with loaded rasters
        """
        # Load configuration
        config = configparser.ConfigParser()
        config.read(config_file)
        
        # Extract hazard parameters from config
        hazard_config = config['HAZARD']
        
        parameter_weights = {
            'susceptibility': float(hazard_config.get('susceptibility_weight', 0.4)),
            'velocity': float(hazard_config.get('velocity_weight', 0.3)),
            'energy': float(hazard_config.get('energy_weight', 0.3))
        }
        
        # Extract class thresholds from config
        class_thresholds = {
            'susceptibility': {
                'Very Low': (0.0, float(hazard_config.get('susceptibility_very_low_max', 0.2))),
                'Low': (float(hazard_config.get('susceptibility_very_low_max', 0.2)), 
                        float(hazard_config.get('susceptibility_low_max', 0.4))),
                'Moderate': (float(hazard_config.get('susceptibility_low_max', 0.4)), 
                             float(hazard_config.get('susceptibility_moderate_max', 0.6))),
                'High': (float(hazard_config.get('susceptibility_moderate_max', 0.6)), 
                         float(hazard_config.get('susceptibility_high_max', 0.8))),
                'Very High': (float(hazard_config.get('susceptibility_high_max', 0.8)), float('inf'))
            },
            'velocity': {
                'Very Low': (0.0, float(hazard_config.get('velocity_very_low_max', 4.0))),
                'Low': (float(hazard_config.get('velocity_very_low_max', 4.0)), 
                        float(hazard_config.get('velocity_low_max', 8.0))),
                'Moderate': (float(hazard_config.get('velocity_low_max', 8.0)), 
                             float(hazard_config.get('velocity_moderate_max', 12.0))),
                'High': (float(hazard_config.get('velocity_moderate_max', 12.0)), 
                         float(hazard_config.get('velocity_high_max', 16.0))),
                'Very High': (float(hazard_config.get('velocity_high_max', 16.0)), float('inf'))
            },
            'energy': {
                'Very Low': (0.0, float(hazard_config.get('energy_very_low_max', 100.0))),
                'Low': (float(hazard_config.get('energy_very_low_max', 100.0)), 
                        float(hazard_config.get('energy_low_max', 200.0))),
                'Moderate': (float(hazard_config.get('energy_low_max', 200.0)), 
                             float(hazard_config.get('energy_moderate_max', 300.0))),
                'High': (float(hazard_config.get('energy_moderate_max', 300.0)), 
                         float(hazard_config.get('energy_high_max', 400.0))),
                'Very High': (float(hazard_config.get('energy_high_max', 400.0)), float('inf'))
            }
        }
        
        # Create HazardAssessment instance
        runout_value = float(config['PARAMETERS'].get('runout_value', 1.0))
        buffer_distance = float(config['PARAMETERS'].get('buffer_distance', 15.0))
        
        hazard_assessment = cls(
            runout_value=runout_value,
            buffer_distance=buffer_distance,
            parameter_weights=parameter_weights,
            class_thresholds=class_thresholds
        )
        
        # Load rasters if io_utils_module is provided
        if io_utils_module is not None:
            read_raster = io_utils_module.read_raster
            read_vector = io_utils_module.read_vector
            logger = logging.getLogger(__name__)
            
            # Load runout extent raster
            if 'runout_extent_raster' in config['INPUT_DATA']:
                runout_raster_file = config['INPUT_DATA']['runout_extent_raster']
                try:
                    logger.info(f"Loading runout extent raster from {runout_raster_file}")
                    runout_raster = read_raster(runout_raster_file)
                    hazard_assessment.set_runout_raster(runout_raster, runout_value)
                    logger.info(f"Successfully loaded runout extent raster")
                except Exception as e:
                    logger.error(f"Failed to load runout extent raster: {e}")
            
            # Load susceptibility raster
            if 'susceptibility_raster' in config['INPUT_DATA']:
                susceptibility_raster_file = config['INPUT_DATA']['susceptibility_raster']
                try:
                    logger.info(f"Loading susceptibility raster from {susceptibility_raster_file}")
                    susceptibility_raster = read_raster(susceptibility_raster_file)
                    hazard_assessment.add_parameter_raster('susceptibility', susceptibility_raster)
                    logger.info(f"Successfully loaded susceptibility raster")
                except Exception as e:
                    logger.error(f"Failed to load susceptibility raster: {e}")
            
            # Load velocity raster
            if 'velocity_raster' in config['INPUT_DATA']:
                velocity_raster_file = config['INPUT_DATA']['velocity_raster']
                try:
                    logger.info(f"Loading velocity raster from {velocity_raster_file}")
                    velocity_raster = read_raster(velocity_raster_file)
                    hazard_assessment.add_parameter_raster('velocity', velocity_raster)
                    logger.info(f"Successfully loaded velocity raster")
                except Exception as e:
                    logger.error(f"Failed to load velocity raster: {e}")
            
            # Load energy raster
            if 'energy_raster' in config['INPUT_DATA']:
                energy_raster_file = config['INPUT_DATA']['energy_raster']
                try:
                    logger.info(f"Loading energy raster from {energy_raster_file}")
                    energy_raster = read_raster(energy_raster_file)
                    hazard_assessment.add_parameter_raster('energy', energy_raster)
                    logger.info(f"Successfully loaded energy raster")
                except Exception as e:
                    logger.error(f"Failed to load energy raster: {e}")
            
            # Load slope units
            if 'slope_units_file' in config['INPUT_DATA']:
                slope_units_file = config['INPUT_DATA']['slope_units_file']
                try:
                    logger.info(f"Loading slope units from {slope_units_file}")
                    slope_units = read_vector(slope_units_file)
                    hazard_assessment.set_slope_units(slope_units)
                    logger.info(f"Successfully loaded slope units: {len(slope_units)} units")
                except Exception as e:
                    logger.error(f"Failed to load slope units: {e}")
            
            # Load source areas
            if 'source_areas_file' in config['INPUT_DATA']:
                source_areas_file = config['INPUT_DATA']['source_areas_file']
                try:
                    logger.info(f"Loading source areas from {source_areas_file}")
                    source_areas = read_vector(source_areas_file)
                    hazard_assessment.set_source_areas(source_areas)
                    logger.info(f"Successfully loaded source areas: {len(source_areas)} areas")
                except Exception as e:
                    logger.error(f"Failed to load source areas: {e}")
            
            # Load rockfall clumps
            if 'rockfall_clumps_file' in config['INPUT_DATA']:
                rockfall_clumps_file = config['INPUT_DATA']['rockfall_clumps_file']
                try:
                    logger.info(f"Loading rockfall clumps from {rockfall_clumps_file}")
                    rockfall_clumps = read_vector(rockfall_clumps_file)
                    hazard_assessment.set_rockfall_clumps(rockfall_clumps)
                    logger.info(f"Successfully loaded rockfall clumps: {len(rockfall_clumps)} clumps")
                except Exception as e:
                    logger.error(f"Failed to load rockfall clumps: {e}")
        
        return hazard_assessment
