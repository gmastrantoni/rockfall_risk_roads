#!/usr/bin/env python
"""
Rockfall Risk Assessment Workflow

This script implements the complete rockfall risk assessment workflow,
integrating all components:
1. Road segmentation
2. Initial classification based on runout zones
3. Hazard assessment
4. Vulnerability assessment
5. Exposure assessment (including network relevance)
6. Final risk calculation

Usage:
    python risk_assessment_workflow.py [config_file]
"""

import os
import sys
import logging
import configparser
from pathlib import Path
import geopandas as gpd
import pandas as pd
import numpy as np
import time
from datetime import datetime

# Add the parent directory to the Python path to import the project modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))


# Import core modules
from src.road.segmentation import segment_road_network
from src.hazard.run_hazard_assessment import run_hazard_assessment
from src.vulnerability.vulnerability_assessment import assess_binary_vulnerability
from src.exposure.exposure_assessment import perform_exposure_assessment
from src.risk.risk_calculation import calculate_risk_for_segments, generate_risk_summary, identify_priority_segments
from src.utils.io_utils import read_vector, read_raster, write_vector, ensure_directory, write_results_to_csv
from src.utils.visualization import plot_classified_segments, create_risk_component_map, plot_risk_matrix

# Configure logging
def setup_logging(output_dir):
    """Set up logging configuration."""
    log_dir = os.path.join(output_dir, 'logs')
    ensure_directory(log_dir)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'risk_assessment_{timestamp}.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return log_file

def run_risk_assessment(config_file="config.ini"):
    """
    Run the complete risk assessment workflow.

    Parameters
    ----------
    config_file : str, optional
        Path to the configuration file, by default "config.ini"

    Returns
    -------
    dict
        Dictionary containing the assessment results
    """
    # Check if config file exists
    if not os.path.exists(config_file):
        print(f"Configuration file not found: {config_file}")
        return None
    
    # Load configuration
    config = configparser.ConfigParser()
    config.read(config_file)
    
    # Get output directory
    output_dir = config['OUTPUT'].get('output_dir', './data/output')
    ensure_directory(output_dir)
    
    # Set up logging
    log_file = setup_logging(output_dir)
    logger = logging.getLogger(__name__)
    
    logger.info(f"Starting rockfall risk assessment using configuration file: {config_file}")
    start_time = time.time()
    
    # Create results dictionary
    results = {
        'config_file': config_file,
        'output_dir': output_dir,
        'log_file': log_file,
        'timestamp': datetime.now().isoformat()
    }
    
    try:
        # 1. Load and segment road network
        logger.info("STEP 1: Loading and segmenting road network")
        
        # road_network_file = config['INPUT_DATA'].get('road_network_file')
        # if not road_network_file or not os.path.exists(road_network_file):
        #     logger.error(f"Road network file not found: {road_network_file}")
        #     return None
        
        # try:
        #     road_network = read_vector(road_network_file)
        #     logger.info(f"Loaded road network with {len(road_network)} features")
            
        #     # Segment the roads
        #     segment_length = float(config['PARAMETERS'].get('segment_length', 200.0))
        #     logger.info(f"Segmenting road network into {segment_length}m segments")
        #     # Check if road_network has ID column
        #     if 'ID' not in road_network.columns and 'id' not in road_network.columns:
        #         road_network['id'] = range(1, len(road_network) + 1)
        #         id_column = 'id'
        #     else:
        #         id_column = 'ID' if 'ID' in road_network.columns else 'id'
            
        #     if id_column is None:
        #         logger.warning("No ID column found in road network, creating one")
        #         road_network['id'] = range(1, len(road_network) + 1)
        #         id_column = 'id'
            
        #     road_segments = segment_road_network(
        #         road_network,
        #         segment_length=segment_length,
        #         id_column=id_column,
        #         preserve_attributes=True
        #     )
        #     logger.info(f"Created {len(road_segments)} road segments")
            
        #     # Save segmented roads
        #     segments_output = os.path.join(output_dir, 'road_segments.gpkg')
        #     write_vector(road_segments, segments_output)
        #     logger.info(f"Saved road segments to: {segments_output}")
            
        #     results['road_segments'] = {
        #         'count': len(road_segments),
        #         'file': segments_output
        #     }
            
        # except Exception as e:
        #     logger.error(f"Error in road segmentation: {str(e)}")
        #     return None
        
        # 2. Hazard Assessment
        logger.info("STEP 2: Hazard Assessment")
        
        try:
            # Run hazard assessment
            _, _, all_segments_with_hazard = run_hazard_assessment(config_file,simulate_parameters=False)
            
            # results['hazard_assessment'] = {
            #     'runout_count': runout_count,
            #     'attention_count': attention_count,
            #     'safe_count': safe_count,
            #     'file': hazard_output
            # }
            
        except Exception as e:
            logger.error(f"Error in hazard assessment: {str(e)}")
            # Continue with other assessments
            all_segments_with_hazard = road_segments.copy()
            logger.warning("Using original road segments for subsequent steps")
        
        # 3. Vulnerability Assessment
        logger.info("STEP 3: Vulnerability Assessment")
        try:
            # Use binary vulnerability assessment
            segments_with_vulnerability = assess_binary_vulnerability(
                all_segments_with_hazard,
                roadbed_column= config['VULNERABILITY'].get('roadbed_column', 'tr_str_sed'),
                level_column=config['VULNERABILITY'].get('level_column', 'tr_str_liv'),
                output_column='vulnerability'
            )
            vulnerability_output = os.path.join(output_dir, 'vulnerability_assessment.gpkg')
            write_vector(segments_with_vulnerability, vulnerability_output)
            logger.info(f"Saved vulnerability assessment to: {vulnerability_output}")
            results['vulnerability_assessment'] = {
                'file': vulnerability_output
            }
        except Exception as e:
            logger.error(f"Error in vulnerability assessment: {str(e)}")
            segments_with_vulnerability = all_segments_with_hazard.copy()
            logger.warning("Using segments from hazard assessment for subsequent steps")
        
        # 4. Exposure Assessment (integrated)
        logger.info("STEP 4: Exposure Assessment (integrated)")
        try:
            intrinsic_weight = float(config['EXPOSURE'].get('intrinsic_weight', 0.4))
            network_weight = float(config['EXPOSURE'].get('network_weight', 0.6))
            segments_with_exposure = perform_exposure_assessment(
                segments_with_vulnerability,
                segment_id_col='segment_id',
                relevance_weight=network_weight,
                intrinsic_weight=intrinsic_weight
            )
            exposure_output = os.path.join(output_dir, 'exposure_assessment.gpkg')
            write_vector(segments_with_exposure, exposure_output)
            logger.info(f"Saved exposure assessment to: {exposure_output}")
            results['exposure_assessment'] = {
                'file': exposure_output
            }
        except Exception as e:
            logger.error(f"Error in exposure assessment: {str(e)}")
            segments_with_exposure = segments_with_vulnerability.copy()
            if 'exposure_score' not in segments_with_exposure.columns:
                segments_with_exposure['exposure_score'] = 3.0
            if 'exposure_class' not in segments_with_exposure.columns:
                segments_with_exposure['exposure_class'] = 'Moderate'
            logger.warning("Using default exposure values for subsequent steps")
        
        # 5. Risk Calculation
        logger.info("STEP 5: Risk Calculation")
        try:
            segments_with_risk = calculate_risk_for_segments(
                segments_with_exposure,
                hazard_column='hazard_score',
                vulnerability_column='vulnerability',
                exposure_column='exposure_factor',
                risk_class_column='risk_class',
                output_raw_column='risk_score',
                output_normalized_column='risk_score_normalized',
                output_class_column='risk_class_final'
            )
            risk_output = os.path.join(output_dir, 'risk_assessment.gpkg')
            write_vector(segments_with_risk, risk_output)
            logger.info(f"Saved final risk assessment to: {risk_output}")
            risk_summary = generate_risk_summary(segments_with_risk)
            summary_output = os.path.join(output_dir, 'risk_summary.csv')
            risk_summary.to_csv(summary_output)
            logger.info(f"Saved risk summary to: {summary_output}")
            priority_segments = identify_priority_segments(segments_with_risk)
            priority_output = os.path.join(output_dir, 'priority_segments.gpkg')
            write_vector(priority_segments, priority_output)
            logger.info(f"Saved priority segments to: {priority_output}")
            results['risk_assessment'] = {
                'file': risk_output,
                'summary_file': summary_output,
                'priority_file': priority_output
            }
        except Exception as e:
            logger.error(f"Error in risk calculation: {str(e)}")
            results['risk_assessment'] = {
                'error': str(e)
            }
        
        # 7. Create Visualizations
        # logger.info("STEP 7: Create Visualizations")
        
        # try:
        #     # Check if visualizations are enabled
        #     create_visualizations = config['OUTPUT'].getboolean('create_visualizations', True)
            
        #     if create_visualizations:
        #         viz_dir = os.path.join(output_dir, 'visualizations')
        #         ensure_directory(viz_dir)
                
        #         # Create risk class map
        #         logger.info("Creating risk classification map")
        #         risk_map_file = os.path.join(viz_dir, 'risk_classification.png')
        #         fig = plot_classified_segments(
        #             segments_with_risk,
        #             column='risk_class_final',
        #             title='Rockfall Risk Classification',
        #             figsize=(12, 10),
        #             save_path=risk_map_file
        #         )
                
        #         # Create component maps
        #         logger.info("Creating risk component maps")
        #         components_file = os.path.join(viz_dir, 'risk_components.png')
        #         fig = create_risk_component_map(
        #             segments_with_risk,
        #             components=['hazard_class', 'vulnerability', 'exposure_class', 'risk_class_final'],
        #             titles=['Hazard', 'Vulnerability', 'Exposure', 'Risk'],
        #             figsize=(20, 15),
        #             save_path=components_file
        #         )
                
        #         # Create risk vs network relevance matrix visualization
        #         logger.info("Creating risk vs network relevance matrix")
        #         matrix_file = os.path.join(viz_dir, 'risk_network_matrix.png')
        #         fig = plot_risk_matrix(
        #             segments_with_risk,
        #             x_column='network_relevance_score',
        #             y_column='risk_score_normalized',
        #             title='Risk vs Network Relevance',
        #             save_path=matrix_file
        #         )
                
        #         logger.info(f"Visualizations saved to: {viz_dir}")
                
        #         results['visualizations'] = {
        #             'directory': viz_dir,
        #             'files': [risk_map_file, components_file, matrix_file]
        #         }
        #     else:
        #         logger.info("Visualizations disabled in configuration")
                
        # except Exception as e:
        #     logger.error(f"Error creating visualizations: {str(e)}")
        #     results['visualizations'] = {
        #         'error': str(e)
        #     }
        
        # 8. Completion
        end_time = time.time()
        elapsed_time = end_time - start_time
        logger.info(f"Risk assessment completed in {elapsed_time:.2f} seconds")
        
        results['elapsed_time'] = elapsed_time
        
        # Save results metadata
        # metadata_output = os.path.join(output_dir, 'assessment_metadata.json')
        # write_results_to_csv(results, metadata_output)
        # logger.info(f"Saved assessment metadata to: {metadata_output}")
        
        return results
        
    except Exception as e:
        logger.error(f"Error in risk assessment workflow: {str(e)}")
        return None


def main():
    """Main function."""
    # Get config file from command line argument
    if len(sys.argv) > 1:
        config_file = sys.argv[1]
    else:
        config_file = "config.ini"
    
    # Run risk assessment
    results = run_risk_assessment(config_file)
    
    print(f"Results saved to: {results['output_dir']}")
    sys.exit(0)


if __name__ == "__main__":
    main()