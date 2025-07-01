#!/usr/bin/env python
"""
Optimized Rockfall Risk Assessment Workflow

This script implements the complete rockfall risk assessment workflow with optimizations
for handling large networks efficiently. It includes progress tracking, sampling strategies,
and parallel processing capabilities.

Usage:
    python optimized_risk_assessment.py [config_file]
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
from tqdm import tqdm

# Add the parent directory to the Python path to import the project modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

# Import core modules
from src.road.segmentation import segment_road_network
from src.hazard.run_hazard_assessment import run_hazard_assessment
from src.vulnerability.vulnerability_assessment import assess_binary_vulnerability
from src.exposure.exposure_assessment_optimized import (
    perform_optimized_exposure_assessment, 
    ExposureAssessmentMonitor
)
from src.risk.risk_calculation import (
    calculate_risk_for_segments, 
    generate_risk_summary, 
    identify_priority_segments
)
from src.utils.io_utils import read_vector, write_vector, ensure_directory
from src.utils.visualization import plot_classified_segments, create_risk_component_map

# Configure logging
def setup_logging(output_dir):
    """Set up logging configuration with progress tracking."""
    log_dir = os.path.join(output_dir, 'logs')
    ensure_directory(log_dir)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'optimized_risk_assessment_{timestamp}.log')
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    simple_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    
    # File handler with detailed format (DEBUG level for our modules only)
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(detailed_formatter)
    file_handler.setLevel(logging.DEBUG)
    
    # Console handler with simple format (INFO level)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(simple_formatter)
    console_handler.setLevel(logging.INFO)
    
    # Configure root logger with INFO level
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    # Set specific loggers to appropriate levels
    # Our project modules - keep at DEBUG for detailed logging
    logging.getLogger('src').setLevel(logging.DEBUG)
    logging.getLogger('__main__').setLevel(logging.INFO)
    
    # # Suppress verbose third-party library logging
    # logging.getLogger('fiona').setLevel(logging.WARNING)
    # logging.getLogger('fiona.ogrext').setLevel(logging.WARNING)
    # logging.getLogger('fiona._env').setLevel(logging.WARNING)
    # logging.getLogger('rasterio').setLevel(logging.WARNING)
    # logging.getLogger('urllib3').setLevel(logging.WARNING)
    # logging.getLogger('matplotlib').setLevel(logging.WARNING)
    # logging.getLogger('PIL').setLevel(logging.WARNING)
    
    return log_file


class OptimizedRiskAssessmentRunner:
    """
    Main class for running optimized risk assessment workflow.
    """
    
    def __init__(self, config_file: str):
        """
        Initialize the runner with configuration.
        
        Parameters
        ----------
        config_file : str
            Path to configuration file
        """
        self.config_file = config_file
        self.config = configparser.ConfigParser()
        self.config.read(config_file)
        
        # Set up output directory and logging
        self.output_dir = self.config['OUTPUT'].get('output_dir', './data/output')
        ensure_directory(self.output_dir)
        self.log_file = setup_logging(self.output_dir)
        
        self.logger = logging.getLogger(__name__)
        self.monitor = ExposureAssessmentMonitor(log_interval=5)
        
        # Store results
        self.results = {
            'config_file': config_file,
            'output_dir': self.output_dir,
            'log_file': self.log_file,
            'timestamp': datetime.now().isoformat()
        }
        
    def progress_callback(self, description: str, current: int, total: int):
        """Progress callback for monitoring operations."""
        self.monitor.update_progress(description, current, total)
        
    def run_assessment(self) -> dict:
        """
        Run the complete optimized risk assessment workflow.
        
        Returns
        -------
        dict
            Dictionary containing assessment results and file paths
        """
        self.logger.info("="*60)
        self.logger.info("STARTING OPTIMIZED ROCKFALL RISK ASSESSMENT")
        self.logger.info("="*60)
        self.logger.info(f"Configuration file: {self.config_file}")
        self.logger.info(f"Output directory: {self.output_dir}")
        
        start_time = time.time()
        self.monitor.start_monitoring(5)  # 5 main steps
        
        try:
            # Step 1: Hazard Assessment
            self.logger.info("\n" + "="*50)
            self.logger.info("STEP 1: HAZARD ASSESSMENT")
            self.logger.info("="*50)
            
            self.progress_callback("Running hazard assessment", 1, 5)
            
            try:
                # Try to open the hazard assessment file checking if it exists
                if os.path.exists(os.path.join(self.output_dir, 'all_segments_with_hazard.gpkg')):
                    self.logger.info("Hazard assessment file already exists, skipping hazard assessment step.")
                    all_segments_with_hazard = read_vector(os.path.join(self.output_dir, 'all_segments_with_hazard.gpkg'))
                else:
                    # Run hazard assessment
                    _, _, all_segments_with_hazard = run_hazard_assessment(
                        self.config_file, 
                        simulate_parameters=False
                    )
                
                self.logger.info(f"Hazard assessment completed: {len(all_segments_with_hazard)} segments processed")
                self.results['hazard_assessment'] = {
                    'segments_processed': len(all_segments_with_hazard),
                    'status': 'completed'
                }
                
            except Exception as e:
                self.logger.error(f"Error in hazard assessment: {str(e)}")
                self.logger.warning("Creating fallback road segments")
                
                # Fallback: create basic road segments
                road_network_file = self.config['INPUT_DATA'].get('road_network_file')
                if road_network_file and os.path.exists(road_network_file):
                    road_network = read_vector(road_network_file)
                    segment_length = float(self.config['PARAMETERS'].get('segment_length', 200.0))
                    
                    id_column = 'ID' if 'ID' in road_network.columns else 'id'
                    if id_column not in road_network.columns:
                        road_network[id_column] = range(1, len(road_network) + 1)
                    
                    all_segments_with_hazard = segment_road_network(
                        road_network,
                        segment_length=segment_length,
                        id_column=id_column,
                        preserve_attributes=True
                    )
                    
                    # Add default hazard values
                    all_segments_with_hazard['hazard_score'] = 3.0
                    all_segments_with_hazard['hazard_class'] = 'Moderate'
                    all_segments_with_hazard['risk_class'] = 'Moderate Risk'
                    
                    self.logger.info(f"Created fallback segments: {len(all_segments_with_hazard)}")
                else:
                    raise Exception("Cannot proceed without road network data")
                
                self.results['hazard_assessment'] = {
                    'segments_processed': len(all_segments_with_hazard),
                    'status': 'fallback',
                    'error': str(e)
                }
            
            # Step 2: Vulnerability Assessment
            self.logger.info("\n" + "="*50)
            self.logger.info("STEP 2: VULNERABILITY ASSESSMENT")
            self.logger.info("="*50)
            
            self.progress_callback("Assessing vulnerability", 2, 5)
            
            try:
                # Try to open the hazard assessment file checking if it exists
                if os.path.exists(os.path.join(self.output_dir, 'vulnerability_assessment.gpkg')):
                    self.logger.info("Vulnerability assessment file already exists, skipping vulnerability assessment step.")
                    segments_with_vulnerability = read_vector(os.path.join(self.output_dir, 'vulnerability_assessment.gpkg'))
                else:
                    segments_with_vulnerability = assess_binary_vulnerability(
                        all_segments_with_hazard,
                        roadbed_column=self.config['VULNERABILITY'].get('roadbed_column', 'tr_str_sed'),
                        level_column=self.config['VULNERABILITY'].get('level_column', 'tr_str_liv'),
                        output_column='vulnerability'
                    )
                
                vulnerability_dist = segments_with_vulnerability['vulnerability'].value_counts()
                self.logger.info("Vulnerability distribution:")
                for vuln, count in vulnerability_dist.items():
                    percentage = (count / len(segments_with_vulnerability)) * 100
                    vuln_label = "Vulnerable" if vuln == 1 else "Protected"
                    self.logger.info(f"  {vuln_label}: {count} segments ({percentage:.1f}%)")
                
                vulnerability_output = os.path.join(self.output_dir, 'vulnerability_assessment.gpkg')
                write_vector(segments_with_vulnerability, vulnerability_output)
                self.logger.info(f"Vulnerability assessment saved to: {vulnerability_output}")
                
                self.results['vulnerability_assessment'] = {
                    'file': vulnerability_output,
                    'distribution': vulnerability_dist.to_dict(),
                    'status': 'completed'
                }
                
            except Exception as e:
                self.logger.error(f"Error in vulnerability assessment: {str(e)}")
                segments_with_vulnerability = all_segments_with_hazard.copy()
                segments_with_vulnerability['vulnerability'] = 1  # Default to vulnerable
                
                self.results['vulnerability_assessment'] = {
                    'status': 'fallback',
                    'error': str(e)
                }
            
            # Step 3: Optimized Exposure Assessment
            self.logger.info("\n" + "="*50)
            self.logger.info("STEP 3: OPTIMIZED EXPOSURE ASSESSMENT")
            self.logger.info("="*50)
            
            self.progress_callback("Running exposure assessment", 3, 5)
            
            try:
                # Get optimization parameters from config
                max_network_size = int(self.config['PARAMETERS'].get('max_network_size', 5000))
                sample_fraction = float(self.config['PARAMETERS'].get('network_sample_fraction', 0.1))
                enable_parallel = self.config['PARAMETERS'].getboolean('enable_parallel', True)
                intrinsic_weight = float(self.config['EXPOSURE'].get('intrinsic_weight', 0.4))
                network_weight = float(self.config['EXPOSURE'].get('network_weight', 0.6))
                
                self.logger.info(f"Exposure assessment parameters:")
                self.logger.info(f"  Max network size for full analysis: {max_network_size}")
                self.logger.info(f"  Sample fraction for large networks: {sample_fraction*100:.1f}%")
                self.logger.info(f"  Parallel processing: {enable_parallel}")
                self.logger.info(f"  Intrinsic weight: {intrinsic_weight}")
                self.logger.info(f"  Network weight: {network_weight}")
                
                segments_with_exposure = perform_optimized_exposure_assessment(
                    segments_with_vulnerability,
                    segment_id_col='segment_id',
                    relevance_weight=network_weight,
                    intrinsic_weight=intrinsic_weight,
                    max_network_size=max_network_size,
                    sample_fraction=sample_fraction,
                    enable_parallel=enable_parallel,
                    progress_callback=self.progress_callback
                )
                
                exposure_output = os.path.join(self.output_dir, 'exposure_assessment.gpkg')
                write_vector(segments_with_exposure, exposure_output)
                self.logger.info(f"Exposure assessment saved to: {exposure_output}")
                
                # Log exposure statistics
                if 'exposure_class' in segments_with_exposure.columns:
                    exposure_dist = segments_with_exposure['exposure_class'].value_counts()
                    self.logger.info("Exposure class distribution:")
                    for cls, count in exposure_dist.items():
                        percentage = (count / len(segments_with_exposure)) * 100
                        self.logger.info(f"  {cls}: {count} segments ({percentage:.1f}%)")
                
                self.results['exposure_assessment'] = {
                    'file': exposure_output,
                    'segments_processed': len(segments_with_exposure),
                    'distribution': exposure_dist.to_dict() if 'exposure_class' in segments_with_exposure.columns else {},
                    'status': 'completed'
                }
                
            except Exception as e:
                self.logger.error(f"Error in exposure assessment: {str(e)}")
                segments_with_exposure = segments_with_vulnerability.copy()
                
                # Add default exposure values
                segments_with_exposure['exposure_factor'] = 0.5
                segments_with_exposure['exposure_class'] = 'Medium'
                segments_with_exposure['relevance_score'] = 0.5
                segments_with_exposure['relevance_class'] = 'Medium'
                
                self.logger.warning("Using default exposure values")
                
                self.results['exposure_assessment'] = {
                    'status': 'fallback',
                    'error': str(e)
                }
            
            # Step 4: Risk Calculation
            self.logger.info("\n" + "="*50)
            self.logger.info("STEP 4: RISK CALCULATION")
            self.logger.info("="*50)
            
            self.progress_callback("Calculating final risk", 4, 5)
            
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
                
                # Save final risk assessment
                risk_output = os.path.join(self.output_dir, 'final_risk_assessment.gpkg')
                write_vector(segments_with_risk, risk_output)
                self.logger.info(f"Final risk assessment saved to: {risk_output}")
                
                # Generate and save risk summary
                risk_summary = generate_risk_summary(segments_with_risk)
                summary_output = os.path.join(self.output_dir, 'risk_summary.csv')
                risk_summary.to_csv(summary_output)
                self.logger.info(f"Risk summary saved to: {summary_output}")
                
                # Identify and save priority segments
                priority_segments = identify_priority_segments(segments_with_risk)
                priority_count = priority_segments['priority_segment'].sum()
                
                if priority_count > 0:
                    priority_output = os.path.join(self.output_dir, 'priority_segments.gpkg')
                    priority_data = priority_segments[priority_segments['priority_segment'] == True]
                    write_vector(priority_data, priority_output)
                    self.logger.info(f"Priority segments saved to: {priority_output}")
                else:
                    self.logger.info("No priority segments identified")
                    priority_output = None
                
                self.results['risk_assessment'] = {
                    'file': risk_output,
                    'summary_file': summary_output,
                    'priority_file': priority_output,
                    'priority_count': int(priority_count),
                    'total_segments': len(segments_with_risk),
                    'status': 'completed'
                }
                
            except Exception as e:
                self.logger.error(f"Error in risk calculation: {str(e)}")
                self.results['risk_assessment'] = {
                    'status': 'error',
                    'error': str(e)
                }
                segments_with_risk = segments_with_exposure  # Use previous data
            
            # Step 5: Generate Visualizations (optional)
            self.logger.info("\n" + "="*50)
            self.logger.info("STEP 5: VISUALIZATION GENERATION")
            self.logger.info("="*50)
            
            self.progress_callback("Creating visualizations", 5, 5)
            
            try:
                create_visualizations = self.config['OUTPUT'].getboolean('create_visualizations', True)
                
                if create_visualizations and 'segments_with_risk' in locals():
                    viz_dir = os.path.join(self.output_dir, 'visualizations')
                    ensure_directory(viz_dir)
                    
                    self.logger.info("Creating risk classification map")
                    risk_map_file = os.path.join(viz_dir, 'risk_classification.png')
                    try:
                        fig = plot_classified_segments(
                            segments_with_risk,
                            column='risk_class_final',
                            title='Rockfall Risk Classification',
                            figsize=(12, 10),
                            save_path=risk_map_file
                        )
                        self.logger.info(f"Risk classification map saved to: {risk_map_file}")
                    except Exception as e:
                        self.logger.error(f"Error creating risk classification map: {e}")
                    
                    self.logger.info("Creating risk component maps")
                    components_file = os.path.join(viz_dir, 'risk_components.png')
                    try:
                        required_columns = ['hazard_class', 'vulnerability', 'exposure_class', 'risk_class_final']
                        available_columns = [col for col in required_columns if col in segments_with_risk.columns]
                        
                        if len(available_columns) >= 2:
                            fig = create_risk_component_map(
                                segments_with_risk,
                                components=available_columns,
                                figsize=(20, 15),
                                save_path=components_file
                            )
                            self.logger.info(f"Risk component maps saved to: {components_file}")
                        else:
                            self.logger.warning("Insufficient columns for component maps")
                            
                    except Exception as e:
                        self.logger.error(f"Error creating risk component maps: {e}")
                    
                    self.results['visualizations'] = {
                        'directory': viz_dir,
                        'status': 'completed'
                    }
                    
                else:
                    self.logger.info("Visualization generation disabled or no data available")
                    self.results['visualizations'] = {
                        'status': 'skipped'
                    }
                    
            except Exception as e:
                self.logger.error(f"Error in visualization generation: {str(e)}")
                self.results['visualizations'] = {
                    'status': 'error',
                    'error': str(e)
                }
            
            # Completion summary
            self.logger.info("\n" + "="*60)
            self.logger.info("ASSESSMENT COMPLETED SUCCESSFULLY")
            self.logger.info("="*60)
            
            elapsed_time = time.time() - start_time
            self.logger.info(f"Total processing time: {elapsed_time:.2f} seconds")
            
            # Log final statistics
            if 'segments_with_risk' in locals():
                self.logger.info(f"Total segments processed: {len(segments_with_risk)}")
                
                if 'risk_class_final' in segments_with_risk.columns:
                    risk_dist = segments_with_risk['risk_class_final'].value_counts()
                    self.logger.info("Final risk distribution:")
                    for risk_class, count in risk_dist.items():
                        percentage = (count / len(segments_with_risk)) * 100
                        self.logger.info(f"  {risk_class}: {count} segments ({percentage:.1f}%)")
            
            self.results['elapsed_time'] = elapsed_time
            self.results['status'] = 'completed'
            
            # Save results metadata
            results_file = os.path.join(self.output_dir, 'assessment_results.json')
            import json
            with open(results_file, 'w') as f:
                # Convert any numpy types to native Python types for JSON serialization
                json_results = {}
                for key, value in self.results.items():
                    if isinstance(value, dict):
                        json_results[key] = {k: int(v) if isinstance(v, (np.integer, np.int64)) else v 
                                           for k, v in value.items()}
                    else:
                        json_results[key] = int(value) if isinstance(value, (np.integer, np.int64)) else value
                
                json.dump(json_results, f, indent=2)
            
            self.logger.info(f"Assessment results metadata saved to: {results_file}")
            
            self.monitor.finish_monitoring()
            return self.results
            
        except Exception as e:
            self.logger.error(f"Critical error in risk assessment workflow: {str(e)}")
            self.results['status'] = 'failed'
            self.results['error'] = str(e)
            self.results['elapsed_time'] = time.time() - start_time
            
            self.monitor.finish_monitoring()
            return self.results


def main():
    """Main function to run the optimized risk assessment."""
    print("Optimized Rockfall Risk Assessment")
    print("=" * 50)
    
    # Get config file from command line argument
    if len(sys.argv) > 1:
        config_file = sys.argv[1]
    else:
        config_file = "config.ini"
    
    # Check if config file exists
    if not os.path.exists(config_file):
        print(f"Error: Configuration file not found: {config_file}")
        print("Usage: python optimized_risk_assessment.py [config_file]")
        sys.exit(1)
    
    try:
        # Initialize and run assessment
        runner = OptimizedRiskAssessmentRunner(config_file)
        results = runner.run_assessment()
        
        # Print summary
        print("\n" + "="*50)
        print("ASSESSMENT SUMMARY")
        print("="*50)
        print(f"Status: {results.get('status', 'unknown')}")
        print(f"Output directory: {results.get('output_dir', 'unknown')}")
        print(f"Processing time: {results.get('elapsed_time', 0):.2f} seconds")
        
        if results.get('status') == 'completed':
            print("\nFiles created:")
            for component, info in results.items():
                if isinstance(info, dict) and 'file' in info:
                    print(f"  {component}: {info['file']}")
            
            if 'risk_assessment' in results and 'total_segments' in results['risk_assessment']:
                total_segments = results['risk_assessment']['total_segments']
                priority_count = results['risk_assessment'].get('priority_count', 0)
                print(f"\nSegments processed: {total_segments}")
                print(f"Priority segments identified: {priority_count}")
        
        sys.exit(0 if results.get('status') == 'completed' else 1)
        
    except KeyboardInterrupt:
        print("\nAssessment interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nCritical error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
