#!/usr/bin/env python
"""
Hazard Assessment Example

This script demonstrates how to use the Hazard Assessment module
to analyze rockfall hazard for road segments using the configuration file.
"""

import os
import sys
import logging
import configparser
from pathlib import Path

# Add the parent directory to the Python path to import the project modules
sys.path.append(str(Path(__file__).parent.parent))

# Import required modules
from src.utils.io_utils import read_vector, read_raster, write_vector
from src.hazard import HazardAssessment
from src.utils import io_utils
from src.road.segmentation import segment_road_network

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def run_hazard_assessment(config_file="config.ini", simulate_parameters=True):
    """
    Run hazard assessment using the specified configuration file.

    Parameters
    ----------
    config_file : str, optional
        Path to the configuration file, by default "config.ini"
    simulate_parameters : bool, optional
        Whether to simulate parameter values instead of using rasters, by default True
    """
    # Check if config file exists
    if not os.path.exists(config_file):
        logging.error(f"Configuration file not found: {config_file}")
        return None, None
    
    logging.info(f"Using configuration file: {config_file}")
    
    # Load configuration
    config = configparser.ConfigParser()
    config.read(config_file)
    
    # Load road network data
    road_network_file = config['INPUT_DATA'].get('road_network_file')
    if not road_network_file or not os.path.exists(road_network_file):
        logging.error(f"Road network file not found: {road_network_file}")
        return None, None
    
    try:
        road_network = read_vector(road_network_file)
        logging.info(f"Loaded road network with {len(road_network)} features")
        
        # Segment the roads into 200m segments
        segment_length = float(config['PARAMETERS'].get('segment_length', 200.0))
        logging.info(f"Segmenting road network into {segment_length}m segments")
        
        # Check if road_network has ID column
        if 'ID' not in road_network.columns and 'id' not in road_network.columns:
            road_network['id'] = range(1, len(road_network) + 1)
            id_column = 'id'
        else:
            id_column = 'ID' if 'ID' in road_network.columns else 'id'
        
        road_segments = segment_road_network(
            road_network,
            segment_length=segment_length,
            id_column=id_column,
            preserve_attributes=True
        )
        logging.info(f"Created {len(road_segments)} road segments")
        
    except Exception as e:
        logging.error(f"Error loading or segmenting road network: {e}")
        return None, None
    
    # Create HazardAssessment from config file
    try:
        hazard_assessment = HazardAssessment.from_config(config_file, io_utils_module=io_utils)
    except Exception as e:
        logging.error(f"Error creating HazardAssessment from config: {e}")
        return None, None
    
    # Check if we have runout raster
    if hazard_assessment.runout_raster is None:
        runout_raster_file = config['INPUT_DATA'].get('runout_extent_raster')
        if not runout_raster_file or not os.path.exists(runout_raster_file):
            logging.error(f"Runout raster file not found: {runout_raster_file}")
            return None, None
        
        try:
            runout_raster = read_raster(runout_raster_file)
            runout_value = float(config['PARAMETERS'].get('runout_value', 1.0))
            hazard_assessment.set_runout_raster(runout_raster, runout_value)
        except Exception as e:
            logging.error(f"Error loading runout raster: {e}")
            return None, None
    
    # Perform hazard assessment
    try:
        logging.info(f"Performing hazard assessment with simulate_parameters={simulate_parameters}")
        segments_with_hazard, segments_not_in_runout = hazard_assessment.assess_hazard(
            road_segments,
            simulate_parameters=simulate_parameters
        )
        
        # Get information about segment classification
        runout_count = len(segments_with_hazard) if not segments_with_hazard.empty else 0
        
        # Count different types of segments
        attention_count = 0
        safe_count = 0
        if not segments_not_in_runout.empty and 'risk_class' in segments_not_in_runout.columns:
            attention_count = len(segments_not_in_runout[segments_not_in_runout['risk_class'] == 'Area of Attention'])
            safe_count = len(segments_not_in_runout[segments_not_in_runout['risk_class'] == 'Not at Risk'])
        
        logging.info(f"Hazard assessment completed:")
        logging.info(f"  Segments in runout zone: {runout_count}")
        logging.info(f"  Area of Attention segments: {attention_count}")
        logging.info(f"  Not at Risk segments: {safe_count}")
        
        # Example: Print hazard classification summary for runout segments
        if not segments_with_hazard.empty and 'hazard_class' in segments_with_hazard.columns:
            hazard_summary = segments_with_hazard['hazard_class'].value_counts()
            logging.info("Hazard classification summary for runout segments:")
            for hazard_class, count in hazard_summary.items():
                logging.info(f"  {hazard_class}: {count} segments")
    except Exception as e:
        logging.error(f"Error during hazard assessment: {e}")
        return None, None
    
    # Save results
    output_dir = config['OUTPUT'].get('output_dir', './data/output')
    os.makedirs(output_dir, exist_ok=True)
    
    # Save segments in runout zone
    if not segments_with_hazard.empty:
        hazard_output_file = os.path.join(output_dir, 'hazard_assessment_runout.gpkg')
        try:
            write_vector(segments_with_hazard, hazard_output_file)
            logging.info(f"Hazard assessment for runout segments saved to: {hazard_output_file}")
        except Exception as e:
            logging.error(f"Error saving hazard assessment for runout segments: {e}")
    
    # Split non-runout segments into "Area of Attention" and "Not at Risk" for separate files
    if not segments_not_in_runout.empty and 'risk_class' in segments_not_in_runout.columns:
        # Save "Area of Attention" segments
        attention_segments = segments_not_in_runout[segments_not_in_runout['risk_class'] == 'Area of Attention']
        if not attention_segments.empty:
            attention_output_file = os.path.join(output_dir, 'hazard_assessment_attention.gpkg')
            try:
                write_vector(attention_segments, attention_output_file)
                logging.info(f"Area of Attention segments saved to: {attention_output_file}")
            except Exception as e:
                logging.error(f"Error saving Area of Attention segments: {e}")
        
        # Save "Not at Risk" segments
        safe_segments = segments_not_in_runout[segments_not_in_runout['risk_class'] == 'Not at Risk']
        if not safe_segments.empty:
            safe_output_file = os.path.join(output_dir, 'hazard_assessment_safe.gpkg')
            try:
                write_vector(safe_segments, safe_output_file)
                logging.info(f"Not at Risk segments saved to: {safe_output_file}")
            except Exception as e:
                logging.error(f"Error saving Not at Risk segments: {e}")
    
    # Save all segments with hazard information for visualization
    try:
        # Combine all segments
        import pandas as pd
        all_segments = pd.concat([segments_with_hazard, segments_not_in_runout])
        
        # Save combined results
        all_segments_file = os.path.join(output_dir, 'all_segments_with_hazard.gpkg')
        write_vector(all_segments, all_segments_file)
        logging.info(f"All segments with hazard information saved to: {all_segments_file}")
    except Exception as e:
        logging.error(f"Error saving combined segments: {e}")
    
    return segments_with_hazard, segments_not_in_runout

if __name__ == "__main__":
    # Check if a config file was provided as an argument
    if len(sys.argv) > 1:
        config_file = sys.argv[1]
    else:
        config_file = "config.ini"
    
    # Run hazard assessment with simulation mode enabled by default
    # This ensures the example works even if parameter rasters are not available
    segments_with_hazard, segments_not_in_runout = run_hazard_assessment(
        config_file, 
        simulate_parameters=False
    )
