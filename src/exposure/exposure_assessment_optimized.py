"""
exposure_assessment_optimized.py

Optimized Exposure Assessment Module
This module integrates intrinsic value with optimized network relevance analysis
for efficient exposure assessment of road segments, especially for large networks.
"""

import geopandas as gpd
import pandas as pd
import numpy as np
import logging
import time
from typing import Optional, Callable
from tqdm import tqdm

from ..road.enhanced_network_analysis import EnhancedRoadNetworkGraph
from .network_relevance_optimized import OptimizedNetworkRelevanceAnalyzer, calculate_network_exposure_optimized
from .intrinsic_value import IntrinsicValueCalculator
from ..utils.geo_utils import multilinestring_to_linestring

logger = logging.getLogger(__name__)


def perform_optimized_exposure_assessment(
    segmented_roads: gpd.GeoDataFrame,
    segment_id_col: str = 'segment_id',
    relevance_weight: float = 0.6,
    intrinsic_weight: float = 0.4,
    max_network_size: int = 5000,
    sample_fraction: float = 0.1,
    enable_parallel: bool = True,
    max_workers: Optional[int] = None,
    progress_callback: Optional[Callable] = None
) -> gpd.GeoDataFrame:
    """
    Perform optimized integrated exposure assessment using intrinsic value and network relevance.

    Parameters
    ----------
    segmented_roads : gpd.GeoDataFrame
        Road segments GeoDataFrame
    segment_id_col : str, optional
        Column name for segment ID, by default 'segment_id'
    relevance_weight : float, optional
        Weight for network relevance, by default 0.6
    intrinsic_weight : float, optional
        Weight for intrinsic value, by default 0.4
    max_network_size : int, optional
        Maximum network size before using sampling strategies, by default 5000
    sample_fraction : float, optional
        Fraction of edges to sample for large networks, by default 0.1
    enable_parallel : bool, optional
        Whether to use parallel processing, by default True
    max_workers : Optional[int], optional
        Maximum number of parallel workers, by default None
    progress_callback : Optional[Callable], optional
        Callback function for progress updates, by default None

    Returns
    -------
    gpd.GeoDataFrame
        GeoDataFrame with integrated exposure assessment
    """
    logger.info("Starting optimized exposure assessment")
    start_time = time.time()
    
    if progress_callback:
        progress_callback("Starting exposure assessment", 0, 5)
    
    # Load road segments
    roads = segmented_roads.copy()
    logger.info(f"Processing {len(roads)} road segments")

    # Step 1: Transform MultiLineString to LineString if needed
    if progress_callback:
        progress_callback("Processing geometries", 1, 5)
    
    if (roads.geometry.geom_type == 'MultiLineString').any():
        logger.info("Converting MultiLineString geometries to LineString")
        road_segments = multilinestring_to_linestring(roads)
    else:
        road_segments = roads

    # Step 2: Calculate intrinsic values
    if progress_callback:
        progress_callback("Calculating intrinsic values", 2, 5)
    
    logger.info("Calculating intrinsic road values")
    intrinsic_calc = IntrinsicValueCalculator()
    road_segments = intrinsic_calc.calculate_for_dataframe(road_segments)

    # Step 3: Create network graph
    if progress_callback:
        progress_callback("Creating network graph", 3, 5)
    
    logger.info("Creating road network graph")
    road_network = EnhancedRoadNetworkGraph(
        enable_progress_tracking=True,
        use_spatial_index=True,
        logger=logger
    )
    
    try:
        graph = road_network.create_graph_from_segments(
            road_segments, 
            segment_id_col=segment_id_col
        )
        logger.info(f"Created network graph with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges")
    except Exception as e:
        logger.error(f"Error creating network graph: {e}")
        # Fallback: use simplified exposure calculation
        logger.warning("Using simplified exposure calculation without network analysis")
        return _calculate_simplified_exposure(road_segments, intrinsic_weight, progress_callback)

    # Step 4: Analyze network relevance (optimized)
    if progress_callback:
        progress_callback("Analyzing network relevance", 4, 5)
    
    logger.info("Analyzing network relevance with optimization")
    
    try:
        # Initialize optimized analyzer
        relevance_analyzer = OptimizedNetworkRelevanceAnalyzer(
            graph=graph,
            road_network=road_network,
            max_network_size=max_network_size,
            sample_fraction=sample_fraction,
            enable_parallel=enable_parallel,
            max_workers=max_workers
        )
        
        # Create progress callback for network analysis
        def network_progress_callback(completed, total):
            if progress_callback:
                # Convert to overall progress (step 4 is 80% of the way through)
                overall_progress = 3.8 + (completed / total) * 0.2
                progress_callback(f"Analyzing network segments: {completed}/{total}", overall_progress, 5)
        
        # Calculate baseline metrics
        logger.info("Calculating baseline network metrics")
        relevance_analyzer.calculate_baseline_metrics()
        
        # Analyze all segments
        logger.info("Analyzing segment impacts")
        df_impacts = relevance_analyzer.analyze_all_segments(progress_callback=network_progress_callback)
        
        if df_impacts.empty:
            logger.warning("No network analysis results, using simplified approach")
            return _calculate_simplified_exposure(road_segments, intrinsic_weight, progress_callback)
        
        # Classify segments by relevance
        logger.info("Classifying segments by network relevance")
        df_classified = relevance_analyzer.classify_segments_by_relevance(df_impacts)
        
        # Export results back to GeoDataFrame
        logger.info("Merging network relevance results with road segments")
        gdf_with_relevance = relevance_analyzer.export_relevance_to_geodataframe(
            road_segments, df_classified, segment_id_col=segment_id_col
        )
        
    except Exception as e:
        logger.error(f"Error in network relevance analysis: {e}")
        logger.warning("Using simplified exposure calculation")
        return _calculate_simplified_exposure(road_segments, intrinsic_weight, progress_callback)

    # Step 5: Calculate integrated exposure
    if progress_callback:
        progress_callback("Calculating integrated exposure", 5, 5)
    
    logger.info("Calculating integrated exposure factors")
    
    try:
        gdf_final = calculate_network_exposure_optimized(
            gdf_with_relevance,
            relevance_weight=relevance_weight,
            intrinsic_weight=intrinsic_weight,
            progress_callback=progress_callback
        )
    except Exception as e:
        logger.error(f"Error calculating integrated exposure: {e}")
        # Fallback to basic exposure calculation
        gdf_final = gdf_with_relevance.copy()
        gdf_final['exposure_factor'] = gdf_final.get('intrinsic_value_score', 3.0) / 5.0
        gdf_final['exposure_class'] = 'Medium'
    
    # Log completion statistics
    elapsed_time = time.time() - start_time
    logger.info(f"Optimized exposure assessment completed in {elapsed_time:.2f} seconds")
    
    # Log exposure distribution
    if 'exposure_class' in gdf_final.columns:
        exposure_dist = gdf_final['exposure_class'].value_counts()
        logger.info("Exposure class distribution:")
        for cls, count in exposure_dist.items():
            percentage = (count / len(gdf_final)) * 100
            logger.info(f"  {cls}: {count} segments ({percentage:.1f}%)")
    
    if progress_callback:
        progress_callback("Exposure assessment completed", 5, 5)
    
    return gdf_final


def _calculate_simplified_exposure(
    road_segments: gpd.GeoDataFrame,
    intrinsic_weight: float = 1.0,
    progress_callback: Optional[Callable] = None
) -> gpd.GeoDataFrame:
    """
    Calculate simplified exposure based only on intrinsic values (fallback method).
    
    Parameters
    ----------
    road_segments : gpd.GeoDataFrame
        Road segments with intrinsic values
    intrinsic_weight : float, optional
        Weight for intrinsic value, by default 1.0
    progress_callback : Optional[Callable], optional
        Progress callback function, by default None
    
    Returns
    -------
    gpd.GeoDataFrame
        Road segments with simplified exposure assessment
    """
    logger.info("Calculating simplified exposure (intrinsic values only)")
    
    gdf_result = road_segments.copy()
    
    # Use intrinsic value as the primary exposure factor
    if 'intrinsic_value_score' in gdf_result.columns:
        # Normalize intrinsic values to 0-1 scale
        intrinsic_values = gdf_result['intrinsic_value_score'].fillna(3.0)
        gdf_result['exposure_factor'] = (intrinsic_values - 1) / 4 * intrinsic_weight
    else:
        # Fallback: use moderate exposure for all segments
        logger.warning("No intrinsic value scores found, using default moderate exposure")
        gdf_result['exposure_factor'] = 0.5
    
    # Ensure exposure factor is in valid range
    gdf_result['exposure_factor'] = gdf_result['exposure_factor'].clip(0, 1)
    
    # Classify exposure
    gdf_result['exposure_class'] = pd.cut(
        gdf_result['exposure_factor'],
        bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
        labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'],
        include_lowest=True
    )
    
    # Handle any NaN values
    gdf_result['exposure_class'] = gdf_result['exposure_class'].fillna('Medium')
    
    # Add default network relevance columns for consistency
    gdf_result['relevance_score'] = 0.5  # Default medium relevance
    gdf_result['relevance_class'] = 'Medium'
    gdf_result['betweenness_centrality'] = 0.0
    gdf_result['disconnects_graph'] = False
    gdf_result['components_increase'] = 0
    gdf_result['largest_component_reduction'] = 0.0
    gdf_result['reachability_impact'] = 0.0
    
    if progress_callback:
        progress_callback("Simplified exposure calculation completed", 5, 5)
    
    logger.info("Simplified exposure calculation completed")
    return gdf_result


def batch_exposure_assessment(
    road_segments_list: list,
    output_dir: str,
    **kwargs
) -> list:
    """
    Perform exposure assessment on multiple road segment datasets in batch.
    
    Parameters
    ----------
    road_segments_list : list
        List of road segment GeoDataFrames or file paths
    output_dir : str
        Directory to save results
    **kwargs
        Additional arguments passed to perform_optimized_exposure_assessment
    
    Returns
    -------
    list
        List of result file paths
    """
    import os
    from ..utils.io_utils import write_vector, ensure_directory
    
    ensure_directory(output_dir)
    result_files = []
    
    logger.info(f"Starting batch exposure assessment for {len(road_segments_list)} datasets")
    
    for i, road_data in enumerate(road_segments_list):
        logger.info(f"Processing dataset {i+1}/{len(road_segments_list)}")
        
        # Load data if it's a file path
        if isinstance(road_data, str):
            from ..utils.io_utils import read_vector
            road_segments = read_vector(road_data)
        else:
            road_segments = road_data
        
        # Perform exposure assessment
        try:
            result = perform_optimized_exposure_assessment(road_segments, **kwargs)
            
            # Save results
            output_file = os.path.join(output_dir, f'exposure_assessment_{i+1}.gpkg')
            write_vector(result, output_file)
            result_files.append(output_file)
            
            logger.info(f"Saved results to: {output_file}")
            
        except Exception as e:
            logger.error(f"Error processing dataset {i+1}: {e}")
            continue
    
    logger.info(f"Batch exposure assessment completed. {len(result_files)} datasets processed successfully.")
    return result_files


class ExposureAssessmentMonitor:
    """
    Monitor and log progress of exposure assessment operations.
    """
    
    def __init__(self, log_interval: int = 10):
        """
        Initialize the monitor.
        
        Parameters
        ----------
        log_interval : int, optional
            Interval in seconds for logging progress, by default 10
        """
        self.log_interval = log_interval
        self.start_time = None
        self.last_log_time = None
        self.total_operations = 0
        self.completed_operations = 0
        
    def start_monitoring(self, total_operations: int):
        """Start monitoring progress."""
        self.start_time = time.time()
        self.last_log_time = self.start_time
        self.total_operations = total_operations
        self.completed_operations = 0
        logger.info(f"Starting monitoring for {total_operations} operations")
        
    def update_progress(self, description: str, completed: int, total: int):
        """
        Update progress and log if necessary.
        
        Parameters
        ----------
        description : str
            Description of current operation
        completed : int
            Number of completed operations
        total : int
            Total number of operations
        """
        current_time = time.time()
        
        # Update counters
        self.completed_operations = completed
        
        # Log progress at intervals
        if (current_time - self.last_log_time) >= self.log_interval:
            elapsed_time = current_time - self.start_time
            rate = completed / elapsed_time if elapsed_time > 0 else 0
            eta = (total - completed) / rate if rate > 0 else float('inf')
            
            progress_percent = (completed / total) * 100 if total > 0 else 0
            
            logger.info(
                f"Progress: {completed}/{total} ({progress_percent:.1f}%) - "
                f"Rate: {rate:.2f} items/s - ETA: {eta:.1f}s - {description}"
            )
            
            self.last_log_time = current_time
    
    def finish_monitoring(self):
        """Finish monitoring and log final statistics."""
        if self.start_time:
            total_time = time.time() - self.start_time
            avg_rate = self.completed_operations / total_time if total_time > 0 else 0
            
            logger.info(
                f"Monitoring completed: {self.completed_operations}/{self.total_operations} operations "
                f"in {total_time:.2f}s (avg rate: {avg_rate:.2f} items/s)"
            )
